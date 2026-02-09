from __future__ import annotations

import pandas as pd
import pytest

from io import StringIO
from pandas.io.parsers import TextFileReader
from pathlib import Path

from cdm_reader_mapper.mdf_reader.utils.utilities import (
    as_list,
    as_path,
    join,
    update_dtypes,
    update_column_names,
    update_column_labels,
    read_csv,
    convert_dtypes,
    validate_arg,
    _adjust_dtype,
    convert_str_boolean,
    _remove_boolean_values,
    remove_boolean_values,
    process_textfilereader,
)


def make_parser(text: str, chunksize: int = 1) -> pd.io.parsers.TextFileReader:
    """Helper: create a TextFileReader similar to user code."""
    buffer = StringIO(text)
    return pd.read_csv(buffer, chunksize=chunksize)


@pytest.fixture
def sample_reader() -> pd.io.parsers.TextFileReader:
    buffer = StringIO("A,B\n1,2\n3,4\n")
    return pd.read_csv(buffer, chunksize=1)


@pytest.fixture
def tmp_csv_file(tmp_path):
    data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    file_path = tmp_path / "test.csv"
    data.to_csv(file_path, index=False)
    return file_path, data


def sample_func(df):
    df_new = df * 2
    extra = {"note": "first_chunk_only"}
    return df_new, extra


def sample_func_only_df(df):
    return df * 2


@pytest.mark.parametrize(
    "input_value, expected",
    [
        (None, None),
        ("hello", ["hello"]),
        ([1, 2, 3], [1, 2, 3]),
        ((4, 5), [4, 5]),
    ],
)
def test_as_list(input_value, expected):
    result = as_list(input_value)
    assert result == expected


def test_as_list_with_set_order_warning():
    s = {"a", "b"}  # sets are unordered
    result = as_list(s)
    assert set(result) == s


def test_as_path_with_string(tmp_path):
    p = tmp_path / "file.txt"
    result = as_path(str(p), "test_param")
    assert isinstance(result, Path)
    assert result == p


def test_as_path_with_pathlike(tmp_path):
    p = tmp_path / "file.txt"
    result = as_path(p, "test_param")
    assert isinstance(result, Path)
    assert result == p


def test_as_path_with_invalid_type():
    with pytest.raises(TypeError):
        as_path(123, "number_param")


@pytest.mark.parametrize(
    "input_col, expected",
    [
        ("single", "single"),
        (["a", "b"], "a:b"),
        (("x", "y", "z"), "x:y:z"),
        ([1, 2], "1:2"),
        (42, "42"),
    ],
)
def test_join(input_col, expected):
    assert join(input_col) == expected


def test_update_dtypes():
    dtypes = {"A": int, "B": float, "C": str}
    columns = ["A", "C"]
    expected = {"A": int, "C": str}
    assert update_dtypes(dtypes, columns) == expected


def test_update_dtypes_with_empty_columns():
    dtypes = {"A": int, "B": float}
    assert update_dtypes(dtypes, []) == {}


def test_update_column_names_dict():
    dtypes = {"A": int, "B": float}
    updated = update_column_names(dtypes.copy(), "A", "X")
    assert updated == {"X": int, "B": float}


def test_update_column_names_no_change():
    dtypes = {"A": int}
    updated = update_column_names(dtypes.copy(), "B", "Y")
    assert updated == {"A": int}


def test_update_column_names_string_input():
    value = "some string"
    assert update_column_names(value, "A", "X") == "some string"


def test_update_column_labels_simple_strings():
    cols = ["A", "B", "C"]
    result = update_column_labels(cols)
    assert isinstance(result, pd.Index)
    assert list(result) == ["A", "B", "C"]


def test_update_column_labels_colon_strings():
    cols = ["A:B", "C:D"]
    result = update_column_labels(cols)
    assert isinstance(result, pd.MultiIndex)
    assert result.tolist() == [("A", "B"), ("C", "D")]


def test_update_column_labels_tuple_strings():
    cols = ["('A','B')", "('C','D')"]
    result = update_column_labels(cols)
    assert isinstance(result, pd.MultiIndex)
    assert result.tolist() == [("A", "B"), ("C", "D")]


def test_update_column_labels_mixed():
    cols = ["A", "('B','C')", "D:E"]
    result = update_column_labels(cols)
    assert isinstance(result, pd.Index)  # Not all tuples
    assert result.tolist() == ["A", ("B", "C"), ("D", "E")]


def test_read_csv_file_exists(tmp_csv_file):
    file_path, data = tmp_csv_file
    df, info = read_csv(file_path)
    pd.testing.assert_frame_equal(df, data)

    assert "columns" in info
    pd.testing.assert_index_equal(info["columns"], df.columns)
    assert "dtypes" in info
    pd.testing.assert_series_equal(info["dtypes"], df.dtypes)


def test_read_csv_file_missing(tmp_path):
    missing_file = tmp_path / "missing.csv"
    df, info = read_csv(missing_file)
    assert df.empty
    assert info == {}


def test_read_csv_with_col_subset(tmp_csv_file):
    file_path, _ = tmp_csv_file
    df, info = read_csv(file_path, col_subset=["B"])
    assert list(df.columns) == ["B"]

    assert "columns" in info
    pd.testing.assert_index_equal(info["columns"], df.columns)
    assert "dtypes" in info
    pd.testing.assert_series_equal(info["dtypes"], df.dtypes)


def test_convert_dtypes_basic():
    dtypes = {"A": "int", "B": "datetime", "C": "float"}
    updated, dates = convert_dtypes(dtypes)
    assert updated["B"] == "object"
    assert dates == ["B"]


def test_validate_arg_correct_type():
    assert validate_arg("x", 5, int)


def test_validate_arg_none():
    assert validate_arg("x", None, int)


def test_validate_arg_wrong_type():
    with pytest.raises(ValueError):
        validate_arg("x", "hello", int)


def test_convert_str_boolean():
    assert convert_str_boolean("True") is True
    assert convert_str_boolean("False") is False
    assert convert_str_boolean("hello") == "hello"
    assert convert_str_boolean(1) == 1


def test_remove_boolean_values_helper():
    assert _remove_boolean_values("True") is None
    assert _remove_boolean_values("False") is None
    assert _remove_boolean_values(True) is None
    assert _remove_boolean_values(False) is None
    assert _remove_boolean_values("abc") == "abc"


def test_adjust_dtype():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    dtype = {"A": "int", "B": "float", "C": "str"}
    adjusted = _adjust_dtype(dtype, df)
    assert adjusted == {"A": "int", "B": "float"}
    assert _adjust_dtype("str", df) == "str"


def test_remove_boolean_values():
    df = pd.DataFrame({"A": ["True", "False", "hello"], "B": [1, 2, 3]})
    dtypes = {"A": "object", "B": "int"}
    result = remove_boolean_values(df, dtypes)
    assert result.loc[0, "A"] is None
    assert result.loc[1, "A"] is None
    assert result.loc[2, "A"] == "hello"
    assert result["B"].dtype.name == "int64"


def test_process_textfilereader(sample_reader):
    reader_out, extra_out = process_textfilereader(
        sample_reader, sample_func, read_kwargs={"chunksize": 1}
    )
    assert isinstance(reader_out, TextFileReader)
    df_out = reader_out.read()
    assert df_out.shape == (2, 2)
    assert df_out["A"].iloc[0] == 2
    assert df_out["B"].iloc[1] == 8
    assert extra_out == {"note": "first_chunk_only"}


def test_process_textfilereader_only_df(sample_reader):
    (reader_out,) = process_textfilereader(
        sample_reader, sample_func_only_df, read_kwargs={"chunksize": 1}
    )
    assert isinstance(reader_out, TextFileReader)
    df_out = reader_out.read()
    assert df_out.shape == (2, 2)
    assert df_out["A"].iloc[0] == 2
    assert df_out["B"].iloc[1] == 8


def test_process_textfilereader_makecopy_flag(sample_reader):
    reader_out, extra_out = process_textfilereader(
        sample_reader, sample_func, makecopy=True, read_kwargs={"chunksize": 1}
    )
    assert isinstance(reader_out, TextFileReader)
    df_out = reader_out.read()
    assert df_out.shape == (2, 2)
    assert df_out["A"].iloc[0] == 2
    assert df_out["B"].iloc[1] == 8
    assert extra_out == {"note": "first_chunk_only"}
