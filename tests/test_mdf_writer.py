from __future__ import annotations
import json
import pathlib

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cdm_reader_mapper.common.iterators import ParquetStreamReader
from cdm_reader_mapper.mdf_reader.writer import (
    _normalize_data_chunks,
    write_data,
)


@pytest.fixture
def example_data():
    return pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["1", "2", "3"],
        }
    )


@pytest.fixture
def example_mask():
    return pd.DataFrame(
        {
            "A": [True, True, False],
            "B": [False, True, True],
        }
    )


def test_normalize_data_chunks_empty():
    result = _normalize_data_chunks(None)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], pd.DataFrame)
    assert result[0].empty


def test_normalize_data_chunks_df():
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = _normalize_data_chunks(df)

    assert isinstance(result, list)
    pd.testing.assert_frame_equal(df, result[0])


def test_noramlize_data_chunks_iter():
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = _normalize_data_chunks(iter([df]))

    assert isinstance(result, ParquetStreamReader)

    pd.testing.assert_frame_equal(df, result.read())


def test_noramlize_data_chunks_psr():
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = _normalize_data_chunks(ParquetStreamReader([df]))

    assert isinstance(result, ParquetStreamReader)

    pd.testing.assert_frame_equal(df, result.read())


def test_noramlize_data_chunks_list():
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = _normalize_data_chunks([df])

    assert isinstance(result, ParquetStreamReader)

    pd.testing.assert_frame_equal(df, result.read())


def test_noramlize_data_chunks_tuple():
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = _normalize_data_chunks((df,))

    assert isinstance(result, ParquetStreamReader)

    pd.testing.assert_frame_equal(df, result.read())


def test_noramlize_data_chunks_raises():
    with pytest.raises(TypeError, match="Unsupported data type found"):
        _normalize_data_chunks(42)


def test_write_data_csv(tmp_path, example_data, example_mask):
    info = {
        "dtypes": {"A": "int", "B": "str"},
        "parse_dates": [],
        "encoding": "utf-8",
    }

    write_data(
        example_data,
        mask=example_mask,
        out_dir=tmp_path,
        prefix="test_write",
        suffix="basic",
        data_format="csv",
        **info,
    )

    data_file = tmp_path / "test_write_data_basic.csv"
    mask_file = tmp_path / "test_write_mask_basic.csv"
    info_file = tmp_path / "test_write_info_basic.json"

    assert data_file.is_file()
    assert mask_file.is_file()
    assert info_file.is_file()

    with pathlib.Path(info_file).open() as read_file:
        info_res = json.load(read_file)

    assert info_res == info

    data_res = pd.read_csv(data_file, dtype=info["dtypes"])
    assert_frame_equal(example_data, data_res)

    mask_res = pd.read_csv(mask_file, dtype="bool")
    assert_frame_equal(example_mask, mask_res)


def test_write_data_col_subset(tmp_path, example_data, example_mask):
    info = {
        "dtypes": {"A": "int"},
        "parse_dates": [],
        "encoding": "utf-8",
    }
    subset = "A"

    write_data(
        example_data,
        mask=example_mask,
        out_dir=tmp_path,
        prefix="test_write",
        suffix="subset",
        data_format="csv",
        col_subset=subset,
        **info,
    )

    data_file = tmp_path / "test_write_data_subset.csv"
    mask_file = tmp_path / "test_write_mask_subset.csv"
    info_file = tmp_path / "test_write_info_subset.json"

    assert data_file.is_file()
    assert mask_file.is_file()
    assert info_file.is_file()

    with pathlib.Path(info_file).open() as read_file:
        info_res = json.load(read_file)

    assert info_res == info

    data_res = pd.read_csv(data_file, dtype=info["dtypes"])
    assert_frame_equal(example_data[[subset]], data_res)

    mask_res = pd.read_csv(mask_file, dtype="bool")
    assert_frame_equal(example_mask[[subset]], mask_res)


def test_write_data_parquet(tmp_path, example_data, example_mask):
    info = {
        "dtypes": {"A": "int", "B": "str"},
        "parse_dates": [],
        "encoding": "utf-8",
    }

    write_data(
        example_data,
        mask=example_mask,
        out_dir=tmp_path,
        prefix="test_write",
        suffix="basic",
        data_format="parquet",
        **info,
    )

    data_file = tmp_path / "test_write_data_basic.parquet"
    mask_file = tmp_path / "test_write_mask_basic.parquet"

    assert data_file.is_file()
    assert mask_file.is_file()

    data_res = pd.read_parquet(data_file)
    assert_frame_equal(example_data, data_res)

    mask_res = pd.read_parquet(mask_file)
    assert_frame_equal(example_mask, mask_res)


def test_write_data_feather(tmp_path, example_data, example_mask):
    info = {
        "dtypes": {"A": "int", "B": "str"},
        "parse_dates": [],
        "encoding": "utf-8",
    }

    write_data(
        example_data,
        mask=example_mask,
        out_dir=tmp_path,
        prefix="test_write",
        suffix="basic",
        data_format="feather",
        **info,
    )

    data_file = tmp_path / "test_write_data_basic.feather"
    mask_file = tmp_path / "test_write_mask_basic.feather"

    assert data_file.is_file()
    assert mask_file.is_file()

    data_res = pd.read_feather(data_file)
    assert_frame_equal(example_data, data_res)

    mask_res = pd.read_feather(mask_file)
    assert_frame_equal(example_mask, mask_res)


def test_write_data_invalid_data_format(example_data):
    with pytest.raises(ValueError, match="data_format must be one of"):
        write_data(example_data, data_format="invalid")


def test_write_data_invalid_mask_type(example_data):
    with pytest.raises(ValueError, match="type of 'data' and type of 'mask' do not match."):
        write_data(example_data, mask=[True, False, True])
