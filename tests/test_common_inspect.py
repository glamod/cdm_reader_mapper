from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

from io import StringIO

from cdm_reader_mapper.common.inspect import _count_by_cat, get_length, count_by_cat


def make_parser(text, **kwargs):
    """Helper: create a TextFileReader similar to user code."""
    buffer = StringIO(text)
    return pd.read_csv(buffer, chunksize=2, **kwargs)


@pytest.mark.parametrize(
    "data, expected",
    [
        (["a", "b", "a"], {"a": 2, "b": 1}),
        ([np.nan, "x", np.nan], {"nan": 2, "x": 1}),
        ([], {}),
        ([1, 2, 1, 3], {1: 2, 2: 1, 3: 1}),
        ([None, "a", None], {"nan": 2, "a": 1}),
    ],
)
def test_count_by_cat_i(data, expected):
    series = pd.DataFrame(data, columns=["test"])
    assert _count_by_cat(series, ["test"])["test"] == expected


@pytest.mark.parametrize(
    "data, columns, expected",
    [
        (pd.DataFrame({"A": ["x", "y", "x"]}), "A", {"A": {"x": 2, "y": 1}}),
        (
            pd.DataFrame({"A": ["x", "y", "x"], "B": [1, 2, np.nan]}),
            ["A", "B"],
            {"A": {"x": 2, "y": 1}, "B": {1: 1, 2: 1, "nan": 1}},
        ),
        (
            pd.DataFrame({"C": ["a", "a", "b"]}),
            "C",
            {"C": {"a": 2, "b": 1}},
        ),
        (pd.DataFrame(columns=["D"]), ["D"], {"D": {}}),
        (pd.DataFrame(columns=["D"]), None, {"D": {}}),
    ],
)
def test_count_by_cat_dataframe(data, columns, expected):
    result = count_by_cat(data, columns)
    assert result == expected


def test_count_by_cat_single_column_string():
    df = pd.DataFrame({"A": [1, 2, 2, np.nan]})
    result = count_by_cat(df, "A")
    assert result == {"A": {1: 1, 2: 2, "nan": 1}}


def test_count_by_cat_textfilereader():
    text = "A,B\n1,x\n2,y\n2,x\nnan,z"
    parser = make_parser(text)

    result = count_by_cat(parser, ["A", "B"])
    expected = {
        "A": {1: 1, 2: 2, "nan": 1},
        "B": {"x": 2, "y": 1, "z": 1},
    }
    assert result == expected


@pytest.mark.parametrize(
    "data, expected_len",
    [
        (pd.DataFrame({"A": [1, 2, 3]}), 3),
        (make_parser("A,B\n1,x\n2,y\n3,z"), 3),
    ],
)
def test_get_length_inspect(data, expected_len):
    assert get_length(data) == expected_len
