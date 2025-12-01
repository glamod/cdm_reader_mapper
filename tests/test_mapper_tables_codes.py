from __future__ import annotations

import pytest

import datetime

from cdm_reader_mapper.cdm_mapper import properties
from cdm_reader_mapper.cdm_mapper.tables.tables import get_cdm_atts, get_imodel_maps
from cdm_reader_mapper.cdm_mapper.codes.codes import (
    _eval,
    _to_int,
    _expand_integer_range_key,
    get_code_table,
)


def _assert_dict_keys(d: dict, expected_keys: str | list | set):
    """Assert that the dictionary `d` is a dictionary and has exactly the keys in `expected_keys`."""
    assert isinstance(d, dict)

    actual_keys = set(d.keys())

    if isinstance(expected_keys, str):
        expected_keys_set = {expected_keys}
    else:
        expected_keys_set = set(expected_keys)

    assert actual_keys == expected_keys_set, (
        f"Unexpected keys: {actual_keys - expected_keys_set}, "
        f"Missing keys: {expected_keys_set - actual_keys}"
    )


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("123", 123),
        ("3.14", 3.14),
        ("'hello'", "hello"),
        ('"world"', "world"),
        ("[1, 2, 3]", [1, 2, 3]),
        ("{'a': 1}", {"a": 1}),
        ("True", True),
        ("False", False),
        ("None", None),
        ("not_a_literal", "not_a_literal"),
        ("[1, 2,", "[1, 2,"),
    ],
)
def test_eval_literals(input_str, expected):
    result = _eval(input_str)
    assert result == expected


@pytest.mark.parametrize(
    "input_value, expected",
    [
        (123, 123),
        ("456", 456),
        (0, 0),
        ("0", 0),
        (-10, -10),
        ("-20", -20),
        ("abc", None),
        ("12.34", None),
        (12.34, 12),
        (None, None),
        ("", None),
    ],
)
def test_to_int(input_value, expected):
    assert _to_int(input_value) == expected


def test_expand_single_range():
    d = {"[2000, 2002]": "val"}
    expected = {"2000": "val", "2001": "val", "2002": "val"}
    assert _expand_integer_range_key(d) == expected


def test_expand_range_with_step():
    d = {"[2000, 2006, 2]": "val"}
    expected = {"2000": "val", "2002": "val", "2004": "val", "2006": "val"}
    assert _expand_integer_range_key(d) == expected


def test_expand_yyyy_upper_bound():
    current_year = datetime.date.today().year
    d = {"[2020, 'yyyy']": "val"}
    result = _expand_integer_range_key(d)
    for y in range(2020, current_year + 1):
        assert str(y) in result
        assert result[str(y)] == "val"


def test_expand_nested_dict():
    d = {"[2000, 2001]": {"[1, 2]": "nested"}}
    expected = {
        "2000": {"1": "nested", "2": "nested"},
        "2001": {"1": "nested", "2": "nested"},
    }
    assert _expand_integer_range_key(d) == expected


def test_expand_non_dict_input():
    assert _expand_integer_range_key(42) == 42
    assert _expand_integer_range_key("hello") == "hello"


def test_expand_invalid_keys_skipped():
    d = {"not_a_range": "val", "[2000]": "val2", "[2000, 'x']": "val3"}
    result = _expand_integer_range_key(d)
    assert "not_a_range" in result
    assert "[2000]" not in result
    assert "[2000, 'x']" not in result


@pytest.mark.parametrize(
    "cdm_tables",
    [
        None,
        [],
        "header",
        ["header", "observations-at"],
    ],
)
def test_get_cdm_atts(cdm_tables):
    expected_tables = properties.cdm_tables if cdm_tables is None else cdm_tables

    cdm_atts = get_cdm_atts(cdm_tables)
    _assert_dict_keys(cdm_atts, expected_tables)


@pytest.mark.parametrize(
    "dataset,cdm_tables",
    [
        ("icoads", ["header", "observations"]),
        ("icoads_r302", ["header"]),
        ("icoads_r302_d992", ["observations"]),
        ("icoads", []),
        ("icoads", None),
        ("icoads_r302", ["observations-at"]),
    ],
)
def test_get_imodel_maps(dataset, cdm_tables):
    expected_tables = properties.cdm_tables if cdm_tables is None else cdm_tables

    imaps = get_imodel_maps(*dataset.split("_"), cdm_tables=cdm_tables)
    _assert_dict_keys(imaps, expected_tables)

    if "observations-at" in imaps:
        for v in imaps["observations-at"].values():
            elements = v.get("elements", [])
            assert isinstance(elements, list)


@pytest.mark.parametrize(
    "dataset,code_table,expected",
    [
        ("icoads", None, {}),
        ("icoads", "location_quality", {"1": 2}),
        ("icoads_r302", "baro_units", {}),
        (
            "icoads_r300_d702",
            "baro_units",
            {"1": 1001, "2": 1002, "4": None, "9": None},
        ),
        ("icoads", "platform_sub_type", {"7": 69}),
        (
            "icoads_r300_d704",
            "platform_sub_type",
            {
                "7": 69,
                "01": 26,
                "02": 105,
                "03": 106,
                "04": 107,
                "05": 108,
                "06": 109,
                "99": 26,
            },
        ),
    ],
)
def test_get_code_table(dataset, code_table, expected):

    cdict = get_code_table(*dataset.split("_"), code_table=code_table)
    assert isinstance(cdict, dict)
    assert cdict == expected
