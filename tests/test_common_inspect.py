from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdm_reader_mapper.common.inspect import _count_by_cat, count_by_cat, get_length
from cdm_reader_mapper.common.iterators import ParquetStreamReader


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
def test_count_by_cat_df(data, columns, expected):
    result = count_by_cat(data, columns)
    assert result == expected


def test_count_by_cat_single_column_string():
    df = pd.DataFrame({"A": [1, 2, 2, np.nan]})
    result = count_by_cat(df, "A")
    assert result == {"A": {1: 1, 2: 2, "nan": 1}}


def test_count_by_cat_psr():
    df1 = pd.DataFrame({"A": 1, "B": "x"}, index=[0])
    df2 = pd.DataFrame({"A": 2, "B": "y"}, index=[1])
    df3 = pd.DataFrame({"A": 2, "B": "x"}, index=[2])
    df4 = pd.DataFrame({"A": "nan", "B": "z"}, index=[3])
    psr = ParquetStreamReader([df1, df2, df3, df4])

    result = count_by_cat(psr, ["A", "B"])
    expected = {
        "A": {1: 1, 2: 2, "nan": 1},
        "B": {"x": 2, "y": 1, "z": 1},
    }
    assert result == expected


def test_get_length_df():
    df = pd.DataFrame({"A": [1, 2, 3]})
    assert get_length(df) == 3


def test_get_length_psr():
    df1 = pd.DataFrame({"A": [1, 2]})
    df2 = pd.DataFrame({"A": [3]}, index=[2])
    psr = ParquetStreamReader([df1, df2])
    assert get_length(psr) == 3
