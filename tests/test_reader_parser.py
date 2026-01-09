from __future__ import annotations

import pytest  # noqa

import logging

import pandas as pd
import xarray as xr  # noqa

from pandas.testing import assert_frame_equal

from types import MethodType

from cdm_reader_mapper.mdf_reader.utils.parser import (
    _get_index,
    _get_ignore,
    _convert_dtype_to_default,
    _parse_fixed_width,
    _parse_delimited,
    _parse_line,
    parse_pandas,
    parse_netcdf,  # noqa
    update_pd_config,
    update_xr_config,
    ParserConfig,
    build_parser_config,
)


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


@pytest.fixture
def base_config_pd():
    return ParserConfig(
        order_specs={},
        disable_reads=[],
        dtypes={},
        parse_dates=[],
        convert_decode={},
        validation={},
        encoding="utf-8",
        columns=None,
    )


@pytest.fixture
def base_config_xr():
    return ParserConfig(
        order_specs={
            "core": {
                "elements": {
                    "TEMP": {
                        "index": ("core", "TEMP"),
                        "ignore": False,
                    },
                    "PRES": {
                        "index": ("core", "PRES"),
                        "ignore": False,
                    },
                }
            }
        },
        disable_reads=[],
        dtypes={},
        parse_dates=[],
        convert_decode={},
        validation={
            ("core", "TEMP"): {"units": "__from_file__"},
            ("core", "PRES"): {"units": "__from_file__"},
        },
        encoding="utf-8",
        columns=None,
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


def test_parse_netcdf(order_specs):
    ds = xr.Dataset(
        {
            "YR": ("time", [2010, 2010, 2010]),
            "MO": ("time", [7, 7, 7]),
            "DY": ("time", [1, 2, 3]),
            "HR": ("time", [10, 20, 30]),
        },
        coords={"time": [0, 1, 2]},
        attrs={"source": "fake"},
    )
    out = parse_netcdf(
        ds=ds,
        order_specs=order_specs,
    )

    data = {
        ("core", "YR"): [2010, 2010, 2010],
        ("core", "MO"): [7, 7, 7],
        ("core", "DY"): [1, 2, 3],
        ("core", "HR"): [10, 20, 30],
        ("c1", "ATTI"): [False, False, False],
        ("c1", "ATTL"): [False, False, False],
        ("c1", "BSI"): [False, False, False],
        ("c5", "ATTI"): [False, False, False],
        ("c5", "ATTL"): [False, False, False],
        ("c5", "OS"): [False, False, False],
        ("c5", "OP"): [False, False, False],
        ("c98", "ATTI"): [False, False, False],
        ("c98", "UID"): [False, False, False],
        ("c99_data", "control_No"): [False, False, False],
        ("c99_data", "name"): [False, False, False],
    }

    exp = pd.DataFrame(data, columns=list(data.keys()))

    assert_frame_equal(out, exp)


def test_update_pd_config_updates_encoding(base_config_pd):
    pd_kwargs = {"encoding": "latin-1"}

    new_config = update_pd_config(pd_kwargs, base_config_pd)

    assert new_config.encoding == "latin-1"
    assert base_config_pd.encoding == "utf-8"
    assert new_config is not base_config_pd


def test_update_pd_config_no_encoding_key(base_config_pd):
    pd_kwargs = {"sep": ","}

    new_config = update_pd_config(pd_kwargs, base_config_pd)

    assert new_config is base_config_pd


def test_update_pd_config_empty_encoding(base_config_pd):
    pd_kwargs = {"encoding": ""}

    new_config = update_pd_config(pd_kwargs, base_config_pd)

    assert new_config is base_config_pd


def test_update_pd_config_none_encoding(base_config_pd):
    pd_kwargs = {"encoding": None}

    new_config = update_pd_config(pd_kwargs, base_config_pd)

    assert new_config is base_config_pd


def test_update_xr_config_ignores_missing_elements(base_config_xr):
    ds = xr.Dataset(
        data_vars={
            "TEMP": xr.DataArray([1, 2, 3], attrs={"units": "K"}),
        }
    )

    new_config = update_xr_config(ds, base_config_xr)

    elements = new_config.order_specs["core"]["elements"]
    assert elements["PRES"]["ignore"] is True
    assert elements["TEMP"]["ignore"] is False


def test_update_xr_config_populates_validation_from_attrs(base_config_xr):
    ds = xr.Dataset(
        data_vars={
            "TEMP": xr.DataArray([1, 2, 3], attrs={"units": "K"}),
            "PRES": xr.DataArray([1010, 1011, 1012], attrs={"units": "hPa"}),
        }
    )

    new_config = update_xr_config(ds, base_config_xr)

    assert new_config.validation[("core", "TEMP")]["units"] == "K"
    assert new_config.validation[("core", "PRES")]["units"] == "hPa"


def test_update_xr_config_removes_missing_validation_attrs(base_config_xr):
    ds = xr.Dataset(
        data_vars={
            "TEMP": xr.DataArray([1, 2, 3], attrs={}),
            "PRES": xr.DataArray([1010, 1011, 1012], attrs={"units": "hPa"}),
        }
    )

    new_config = update_xr_config(ds, base_config_xr)

    assert "units" not in new_config.validation[("core", "TEMP")]
    assert new_config.validation[("core", "PRES")]["units"] == "hPa"


def test_update_xr_config_does_not_mutate_original(base_config_xr):
    ds = xr.Dataset(
        data_vars={
            "TEMP": xr.DataArray([1, 2, 3], attrs={"units": "K"}),
        }
    )

    _ = update_xr_config(ds, base_config_xr)

    assert base_config_xr.order_specs["core"]["elements"]["PRES"]["ignore"] is False
    assert base_config_xr.validation[("core", "TEMP")]["units"] == "__from_file__"


def test_build_parser_config_imodel():
    config = build_parser_config("icoads")

    assert isinstance(config, ParserConfig)

    assert hasattr(config, "order_specs")
    assert isinstance(config.order_specs, dict)
    assert "core" in config.order_specs
    spec = config.order_specs["core"]
    assert isinstance(spec, dict)
    assert "header" in spec
    assert isinstance(spec["header"], dict)
    assert "elements" in spec
    assert isinstance(spec["elements"], dict)
    assert "is_delimited" in spec
    assert isinstance(spec["is_delimited"], bool)

    assert hasattr(config, "disable_reads")
    assert isinstance(config.disable_reads, list)
    assert all(isinstance(x, str) for x in config.disable_reads)

    assert hasattr(config, "dtypes")
    assert isinstance(config.dtypes, dict)
    assert all(isinstance(x, tuple) for x in config.dtypes.keys())
    assert all(isinstance(x, str) for x in config.dtypes.values())

    assert hasattr(config, "parse_dates")
    assert isinstance(config.parse_dates, list)
    assert config.parse_dates == []

    assert hasattr(config, "convert_decode")
    assert isinstance(config.convert_decode, dict)

    assert "converter_dict" in config.convert_decode
    converter_dict = config.convert_decode["converter_dict"]
    assert isinstance(converter_dict, dict)
    assert all(isinstance(x, tuple) for x in converter_dict.keys())
    assert all(isinstance(x, MethodType) for x in converter_dict.values())

    assert "converter_kwargs" in config.convert_decode
    converter_kwargs = config.convert_decode["converter_kwargs"]
    assert isinstance(converter_kwargs, dict)
    assert all(isinstance(x, tuple) for x in converter_kwargs.keys())
    assert all(isinstance(x, dict) for x in converter_kwargs.values())

    assert "decoder_dict" in config.convert_decode
    decoder_dict = config.convert_decode["converter_dict"]
    assert isinstance(decoder_dict, dict)
    assert all(isinstance(x, tuple) for x in decoder_dict.keys())
    assert all(isinstance(x, MethodType) for x in decoder_dict.values())

    assert hasattr(config, "validation")
    assert isinstance(config.validation, dict)
    assert all(isinstance(x, tuple) for x in config.validation.keys())
    assert all(isinstance(x, dict) for x in config.validation.values())

    assert hasattr(config, "encoding")
    assert isinstance(config.encoding, str)

    assert hasattr(config, "columns")
    assert config.columns is None
