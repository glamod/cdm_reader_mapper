from __future__ import annotations

import pandas as pd
import pytest

from cdm_reader_mapper.common.iterators import ParquetStreamReader
from cdm_reader_mapper.common.replace import _replace_columns, replace_columns


def test_replace_columns_raises():
    with pytest.raises(TypeError, match="Input left and right data must be pandas DataFrames."):
        _replace_columns([1, 2, 3], [4, 2, 6])


def test_basic_replacement_df():
    df_l = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
    df_r = pd.DataFrame({"id": [1, 2], "x": [100, 200]})

    out = replace_columns(df_l, df_r, pivot_c="id", rep_c="x")
    assert out["x"].tolist() == [100, 200]


def test_basic_replacement_psr():
    df1 = pd.DataFrame({"id": [1], "x": [10]}, index=[0])
    df2 = pd.DataFrame({"id": [2], "x": [20]}, index=[1])
    df3 = pd.DataFrame({"id": [1], "x": [100]}, index=[0])
    df4 = pd.DataFrame({"id": [2], "x": [200]}, index=[1])

    left = ParquetStreamReader([df1, df2])
    right = ParquetStreamReader([df3, df4])

    out = replace_columns(left, right, pivot_c="id", rep_c="x")
    out = out.read()
    assert out["x"].tolist() == [100, 200]


def test_rep_map_different_names():
    df_l = pd.DataFrame({"id": [1, 2], "a": [1, 2]})
    df_r = pd.DataFrame({"id": [1, 2], "b": [10, 20]})

    out = replace_columns(df_l, df_r, pivot_c="id", rep_map={"a": "b"})
    assert out["a"].tolist() == [10, 20]


def test_missing_pivot_raises():
    df_l = pd.DataFrame({"id": [1]})
    df_r = pd.DataFrame({"id": [1]})

    with pytest.raises(ValueError):
        replace_columns(df_l, df_r, rep_c="x")


def test_missing_replacement_raises():
    df_l = pd.DataFrame({"id": [1]})
    df_r = pd.DataFrame({"id": [1]})

    with pytest.raises(ValueError):
        replace_columns(df_l, df_r, pivot_c="id")


def test_missing_source_col_raises():
    df_l = pd.DataFrame({"id": [1], "a": [10]})
    df_r = pd.DataFrame({"id": [1]})

    with pytest.raises(ValueError):
        replace_columns(df_l, df_r, pivot_c="id", rep_map={"a": "missing"})


def test_index_reset():
    df_l = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
    df_r = pd.DataFrame({"id": [1, 2], "x": [100, 200]})

    out = replace_columns(df_l, df_r, pivot_c="id", rep_c="x")
    assert list(out.index) == [0, 1]
