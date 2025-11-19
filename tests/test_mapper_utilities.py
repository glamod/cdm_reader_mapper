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
            # Test: strings, tuples, sets must NOT be expanded
            {"A": "abc", "B": (1, 2), "C": {5}},
            [("A", "abc"), ("B", (1, 2)), ("C", {5})],
        ),
        (
            # Nested lists: only expand the first level
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
        (None, cdm_tables),  # None ? full list
        ("header", ["header"]),  # single valid string
        (["header"], ["header"]),  # list with valid entry
        (
            ["observations-at", "observations-sst"],
            ["observations-at", "observations-sst"],
        ),  # multiple valid entries
    ],
)
def test_get_cdm_subset_valid(input_value, expected):
    assert get_cdm_subset(input_value) == expected


@pytest.mark.parametrize(
    "invalid_value",
    [
        "invalid-table",  # invalid string
        ["header", "does-not-exist"],  # partially invalid list
        ["wrong"],
        ["observations-at", "xxx"],  # one invalid element
        (
            "header",
            "observations-at",
        ),  # tuple ? becomes ["('header', 'observations-at')"] ? invalid
        123,  # non-string input
        [1, 2, 3],  # list but non-strings
    ],
)
def test_get_cdm_subset_invalid(invalid_value):
    with pytest.raises(ValueError):
        get_cdm_subset(invalid_value)


@pytest.mark.parametrize(
    "tb, col_subset, expected",
    [
        # Single string
        ("table1", "colA", ["colA"]),
        # List of columns
        ("table1", ["a", "b"], ["a", "b"]),
        # Any iterable of strings (tuple)
        ("table1", ("x", "y"), ["x", "y"]),
        # Dictionary with table mapping
        ("table1", {"table1": ["c", "d"]}, ["c", "d"]),
        # Dictionary missing table ? returns None
        ("missing", {"table1": ["c", "d"]}, None),
        # None ? returns None
        ("table1", None, None),
    ],
)
def test_get_usecols(tb, col_subset, expected):
    assert get_usecols(tb, col_subset) == expected


@pytest.mark.parametrize(
    "filename, table, extension, expected",
    [
        # Prepend table and add default extension
        ("data", "header", "psv", "header-data.psv"),
        # Table already in filename, add default extension
        ("header-data", "header", "psv", "header-data.psv"),
        # Table already in filename, filename already has extension
        ("header-data.txt", "header", "psv", "header-data.txt"),
        # No table provided, add default extension
        ("data", "", "psv", "data.psv"),
        # No table provided, custom extension
        ("data", "", "csv", "data.csv"),
        # Table prepended, custom extension
        ("info", "observations-ws", "txt", "observations-ws-info.txt"),
        # Table prepended, filename already has extension
        ("info.log", "observations-ws", "txt", "observations-ws-info.log"),
    ],
)
def test_adjust_filename(filename, table, extension, expected):
    assert adjust_filename(filename, table, extension) == expected
