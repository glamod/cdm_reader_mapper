from __future__ import annotations

import pytest  # noqa

import logging

import pandas as pd
import xarray as xr  # noqa

from pandas.testing import assert_frame_equal

from cdm_reader_mapper.mdf_reader.utils.parser import (
    _get_index,
    _get_ignore,
    _convert_dtype_to_default,
    _parse_fixed_width,
    _parse_delimited,
    _parse_line,
    parse_pandas,
    parse_netcdf,  # noqa
)


def test_get_index_single_length():
    assert _get_index("AT", "_SECTION_", 1) == "AT"


def test_get_index_multiple_length():
    assert _get_index("AT", "core", 2) == ("core", "AT")


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
    ],
)
def test_get_ignore_string_and_bool_values(value, expected):
    assert _get_ignore({"ignore": value}) is expected


def test_get_ignore_missing_key():
    assert _get_ignore({}) is False


def test_convert_dtype_none():
    assert _convert_dtype_to_default(None) is None


def test_convert_dtype_float():
    assert _convert_dtype_to_default("float") == "float"


def test_convert_dtype_int():
    assert _convert_dtype_to_default("int") == "Int64"


def test_convert_deprecated_float(caplog):
    with caplog.at_level(logging.WARNING):
        result = _convert_dtype_to_default("Float64")
    assert result == "float"
    assert "deprecated" in caplog.text


def test_convert_deprecated_int(caplog):
    with caplog.at_level(logging.WARNING):
        result = _convert_dtype_to_default("Int32")
    assert result == "Int64"
    assert "deprecated" in caplog.text


def test_convert_unknown_dtype():
    assert _convert_dtype_to_default("string") == "string"


@pytest.mark.parametrize(
    "line, header, elements, exp_end, exp_out",
    [
        (
            "2010 7 1    ",
            {},
            {
                "YR": {"index": ("core", "YR"), "field_length": 4},
                "MO": {"index": ("core", "MO"), "field_length": 2},
                "DY": {"index": ("core", "DY"), "field_length": 2},
                "HR": {"index": ("core", "HR"), "field_length": 4},
            },
            12,
            {
                ("core", "YR"): "2010",
                ("core", "MO"): " 7",
                ("core", "DY"): " 1",
                ("core", "HR"): True,
            },
        ),
        (
            " 165 ",
            {"sentinel": " 165"},
            {
                "ATTI": {"index": ("c1", "ATTI"), "field_length": 2},
                "ATTL": {"index": ("c1", "ATTL"), "field_length": 2},
                "BSI": {"index": ("c1", "BSI"), "field_length": 1},
            },
            5,
            {
                ("c1", "ATTI"): " 1",
                ("c1", "ATTL"): "65",
                ("c1", "BSI"): True,
            },
        ),
        (
            "9815IS7NQU",
            {"sentinel": " 594"},
            {
                "ATTI": {"index": ("c5", "ATTI"), "field_length": 2},
                "ATTL": {"index": ("c5", "ATTL"), "field_length": 2},
                "OS": {"index": ("c5", "OS"), "field_length": 1},
                "OP": {"index": ("c5", "OP"), "field_length": 1},
            },
            0,
            {
                ("c5", "ATTI"): False,
                ("c5", "ATTL"): False,
                ("c5", "OS"): False,
                ("c5", "OP"): False,
            },
        ),
        (
            "9815IS7NQU",
            {"sentinel": "9815"},
            {
                "ATTI": {"index": ("c98", "ATTI"), "field_length": 2},
                "ATTL": {"index": ("c98", "ATTL"), "field_length": 2, "ignore": True},
                "UID": {"index": ("c98", "UID"), "field_length": 6},
            },
            10,
            {
                ("c98", "ATTI"): "98",
                ("c98", "UID"): "IS7NQU",
            },
        ),
    ],
)
def test_parse_fixed_width(line, header, elements, exp_end, exp_out):
    out = {}
    end = _parse_fixed_width(
        line=line,
        i=0,
        header=header,
        elements=elements,
        sections=None,
        excludes=set(),
        out=out,
    )

    assert end == exp_end
    assert out == exp_out


@pytest.mark.parametrize(
    "sections, excludes, exp_out",
    [
        (
            ["core"],
            set(),
            {
                ("core", "YR"): "2010",
                ("core", "MO"): " 7",
                ("core", "DY"): " 1",
                ("core", "HR"): True,
            },
        ),
        (["c1"], set(), {}),
        (None, ["core"], {}),
        (
            None,
            ["c1"],
            {
                ("core", "YR"): "2010",
                ("core", "MO"): " 7",
                ("core", "DY"): " 1",
                ("core", "HR"): True,
            },
        ),
    ],
)
def test_parse_fixed_width_kwargs(sections, excludes, exp_out):
    out = {}
    elements = {
        "YR": {"index": ("core", "YR"), "field_length": 4},
        "MO": {"index": ("core", "MO"), "field_length": 2},
        "DY": {"index": ("core", "DY"), "field_length": 2},
        "HR": {"index": ("core", "HR"), "field_length": 4},
    }
    end = _parse_fixed_width(
        line="2010 7 1    ",
        i=0,
        header={},
        elements=elements,
        sections=sections,
        excludes=excludes,
        out=out,
    )

    assert end == 12
    assert out == exp_out


def test_parse_delimited():
    line = "13615}Peder Aneus"
    header = {"delimiter": "}"}
    elements = {
        "control_No": {"index": ("c99_data", "control_No")},
        "name": {"index": ("c99_data", "name")},
    }
    out = {}
    end = _parse_delimited(
        line=line,
        i=0,
        header=header,
        elements=elements,
        sections=None,
        excludes=set(),
        out=out,
    )

    assert end == len(line)
    assert out == {
        ("c99_data", "control_No"): "13615",
        ("c99_data", "name"): "Peder Aneus",
    }


@pytest.fixture
def order_specs():
    return {
        "core": {
            "header": {},
            "elements": {
                "YR": {"index": ("core", "YR"), "field_length": 4},
                "MO": {"index": ("core", "MO"), "field_length": 2},
                "DY": {"index": ("core", "DY"), "field_length": 2},
                "HR": {"index": ("core", "HR"), "field_length": 4},
            },
            "is_delimited": False,
        },
        "c1": {
            "header": {"sentinel": " 165"},
            "elements": {
                "ATTI": {"index": ("c1", "ATTI"), "field_length": 2},
                "ATTL": {"index": ("c1", "ATTL"), "field_length": 2},
                "BSI": {"index": ("c1", "BSI"), "field_length": 1},
            },
            "is_delimited": False,
        },
        "c5": {
            "header": {"sentinel": " 594"},
            "elements": {
                "ATTI": {"index": ("c5", "ATTI"), "field_length": 2},
                "ATTL": {"index": ("c5", "ATTL"), "field_length": 2},
                "OS": {"index": ("c5", "OS"), "field_length": 1},
                "OP": {"index": ("c5", "OP"), "field_length": 1},
            },
            "is_delimited": False,
        },
        "c98": {
            "header": {"sentinel": "9815"},
            "elements": {
                "ATTI": {"index": ("c98", "ATTI"), "field_length": 2},
                "ATTL": {"index": ("c98", "ATTL"), "field_length": 2, "ignore": True},
                "UID": {"index": ("c98", "UID"), "field_length": 6},
            },
            "is_delimited": False,
        },
        "c99_data": {
            "header": {"delimiter": "}"},
            "elements": {
                "control_No": {"index": ("c99_data", "control_No")},
                "name": {"index": ("c99_data", "name")},
            },
            "is_delimited": True,
        },
    }


def test_parse_line(order_specs):
    line = "2010 7 1     165 9815IS7NQU13615}Peder Aneus"
    out = _parse_line(
        line=line,
        order_specs=order_specs,
        sections=None,
        excludes=set(),
    )

    assert out == {
        ("core", "YR"): "2010",
        ("core", "MO"): " 7",
        ("core", "DY"): " 1",
        ("core", "HR"): True,
        ("c1", "ATTI"): " 1",
        ("c1", "ATTL"): "65",
        ("c1", "BSI"): True,
        ("c5", "ATTI"): False,
        ("c5", "ATTL"): False,
        ("c5", "OS"): False,
        ("c5", "OP"): False,
        ("c98", "ATTI"): "98",
        ("c98", "UID"): "IS7NQU",
        ("c99_data", "control_No"): "13615",
        ("c99_data", "name"): "Peder Aneus",
    }


def test_parse_pandas(order_specs):
    df = pd.DataFrame(
        [
            "2010 7 1     165 9815IS7NQU13615}Peder Aneus",
            "2010 7 20100 165 9815IS7NQU13615}Peder Aneus",
            "2010 7 30200 165 9815IS7NQU13615}Peder Aneus",
        ]
    )
    out = parse_pandas(
        df=df,
        order_specs=order_specs,
    )

    data = {
        ("core", "YR"): ["2010", "2010", "2010"],
        ("core", "MO"): [" 7", " 7", " 7"],
        ("core", "DY"): [" 1", " 2", " 3"],
        ("core", "HR"): [True, "0100", "0200"],
        ("c1", "ATTI"): [" 1", " 1", " 1"],
        ("c1", "ATTL"): ["65", "65", "65"],
        ("c1", "BSI"): [True, True, True],
        ("c5", "ATTI"): [False, False, False],
        ("c5", "ATTL"): [False, False, False],
        ("c5", "OS"): [False, False, False],
        ("c5", "OP"): [False, False, False],
        ("c98", "ATTI"): ["98", "98", "98"],
        ("c98", "UID"): ["IS7NQU", "IS7NQU", "IS7NQU"],
        ("c99_data", "control_No"): ["13615", "13615", "13615"],
        ("c99_data", "name"): ["Peder Aneus", "Peder Aneus", "Peder Aneus"],
    }

    exp = pd.DataFrame(data, columns=list(data.keys()))

    assert_frame_equal(out, exp)
