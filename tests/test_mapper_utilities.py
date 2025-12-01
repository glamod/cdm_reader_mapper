from __future__ import annotations

import pytest


from cdm_reader_mapper.cdm_mapper.properties import cdm_tables

from cdm_reader_mapper.cdm_mapper.utils.utilities import (
    dict_to_tuple_list,
    get_cdm_subset,
    get_usecols,
    adjust_filename,
)


@pytest.mark.parametrize(
    "dic, expected",
    [
        (
            {"A": 1, "B": 2},
            [("A", 1), ("B", 2)],
        ),
        (
            {"A": [1, 2], "B": [3]},
            [("A", 1), ("A", 2), ("B", 3)],
        ),
        (
            {"A": [1, 2], "B": 3},
            [("A", 1), ("A", 2), ("B", 3)],
        ),
        (
            {"A": []},
            [],
        ),
        (
            {},
            [],
        ),
        (
            {"A": "abc", "B": (1, 2), "C": {5}},
            [("A", "abc"), ("B", (1, 2)), ("C", {5})],
        ),
        (
            {"A": [[1, 2], [3]]},
            [("A", [1, 2]), ("A", [3])],
        ),
    ],
)
def test_dict_to_tuple_list(dic, expected):
    assert dict_to_tuple_list(dic) == expected


@pytest.mark.parametrize(
    "input_value, expected",
    [
        (None, cdm_tables),
        ("header", ["header"]),
        (["header"], ["header"]),
        (
            ["observations-at", "observations-sst"],
            ["observations-at", "observations-sst"],
        ),
    ],
)
def test_get_cdm_subset_valid(input_value, expected):
    assert get_cdm_subset(input_value) == expected


@pytest.mark.parametrize(
    "invalid_value",
    [
        "invalid-table",
        ["header", "does-not-exist"],
        ["wrong"],
        ["observations-at", "xxx"],
        (
            "header",
            "observations-at",
        ),
        123,
        [1, 2, 3],
    ],
)
def test_get_cdm_subset_invalid(invalid_value):
    with pytest.raises(ValueError):
        get_cdm_subset(invalid_value)


@pytest.mark.parametrize(
    "tb, col_subset, expected",
    [
        ("table1", "colA", ["colA"]),
        ("table1", ["a", "b"], ["a", "b"]),
        ("table1", ("x", "y"), ["x", "y"]),
        ("table1", {"table1": ["c", "d"]}, ["c", "d"]),
        ("missing", {"table1": ["c", "d"]}, None),
        ("table1", None, None),
    ],
)
def test_get_usecols(tb, col_subset, expected):
    assert get_usecols(tb, col_subset) == expected


@pytest.mark.parametrize(
    "filename, table, extension, expected",
    [
        ("data", "header", "psv", "header-data.psv"),
        ("header-data", "header", "psv", "header-data.psv"),
        ("header-data.txt", "header", "psv", "header-data.txt"),
        ("data", "", "psv", "data.psv"),
        ("data", "", "csv", "data.csv"),
        ("info", "observations-ws", "txt", "observations-ws-info.txt"),
        ("info.log", "observations-ws", "txt", "observations-ws-info.log"),
    ],
)
def test_adjust_filename(filename, table, extension, expected):
    assert adjust_filename(filename, table, extension) == expected
