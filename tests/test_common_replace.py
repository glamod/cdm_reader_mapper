from __future__ import annotations

import pytest

import pandas as pd

from io import StringIO

from cdm_reader_mapper.common.replace import replace_columns


def make_parser(text, **kwargs):
    """Helper: create a TextFileReader similar to user code."""
    buffer = StringIO(text)
    return pd.read_csv(buffer, chunksize=2, **kwargs)


def test_basic_replacement_df():
    df_l = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
    df_r = pd.DataFrame({"id": [1, 2], "x": [100, 200]})

    out = replace_columns(df_l, df_r, pivot_c="id", rep_c="x")
    assert out["x"].tolist() == [100, 200]


def test_basic_replacement_textfilereader():
    parser_l = make_parser("id,x\n1,10\n2,20")
    parser_r = make_parser("id,x\n1,100\n2,200")

    out = replace_columns(parser_l, parser_r, pivot_c="id", rep_c="x")
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
