from __future__ import annotations

import json

import pandas as pd
import pytest  # noqa

from pandas.testing import assert_frame_equal

from cdm_reader_mapper.mdf_reader.writer import (
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
        **info,
    )

    data_file = tmp_path / "test_write-data-basic.csv"
    mask_file = tmp_path / "test_write-mask-basic.csv"
    info_file = tmp_path / "test_write-info-basic.json"

    assert data_file.is_file()
    assert mask_file.is_file()
    assert info_file.is_file()

    with open(info_file) as read_file:
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
        col_subset=subset,
        **info,
    )

    data_file = tmp_path / "test_write-data-subset.csv"
    mask_file = tmp_path / "test_write-mask-subset.csv"
    info_file = tmp_path / "test_write-info-subset.json"

    assert data_file.is_file()
    assert mask_file.is_file()
    assert info_file.is_file()

    with open(info_file) as read_file:
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

    data_file = tmp_path / "test_write-data-basic.parquet"
    mask_file = tmp_path / "test_write-mask-basic.parquet"
    info_file = tmp_path / "test_write-info-basic.json"

    assert data_file.is_file()
    assert mask_file.is_file()
    assert info_file.is_file()

    with open(info_file) as read_file:
        info_res = json.load(read_file)

    assert info_res == info

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

    data_file = tmp_path / "test_write-data-basic.feather"
    mask_file = tmp_path / "test_write-mask-basic.feather"
    info_file = tmp_path / "test_write-info-basic.json"

    assert data_file.is_file()
    assert mask_file.is_file()
    assert info_file.is_file()

    with open(info_file) as read_file:
        info_res = json.load(read_file)

    assert info_res == info

    data_res = pd.read_feather(data_file)
    assert_frame_equal(example_data, data_res)

    mask_res = pd.read_feather(mask_file)
    assert_frame_equal(example_mask, mask_res)


def test_write_data_invalid(example_data):
    with pytest.raises(ValueError):
        write_data(example_data, data_format="invalid")
