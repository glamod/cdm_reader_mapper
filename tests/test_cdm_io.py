from __future__ import annotations

import pandas as pd
import pytest

from cdm_reader_mapper.cdm_mapper.reader import read_tables
from cdm_reader_mapper.cdm_mapper.writer import write_tables


@pytest.fixture
def example_data():
    arrays = [
        ["header", "header", "observations-sst", "observations-sst"],
        ["report_id", "A", "report_id", "B"],
    ]
    multi_cols = pd.MultiIndex.from_arrays(arrays)
    return pd.DataFrame(
        [
            [1, 0.5, 1, "a"],
            [2, 5.0, 2, "b"],
            [3, 1.3, 3, "c"],
        ],
        columns=multi_cols,
    )


@pytest.fixture
def csv_path(tmp_path, example_data):
    header_file = tmp_path / "header-test.csv"
    obssst_file = tmp_path / "observations-sst-test.csv"

    example_data["header"].to_csv(header_file, index=False)
    example_data["observations-sst"].to_csv(obssst_file, index=False)

    return tmp_path


@pytest.fixture
def parquet_path(tmp_path, example_data):
    header_file = tmp_path / "header-test.parquet"
    obssst_file = tmp_path / "observations-sst-test.parquet"

    example_data["header"].to_parquet(header_file, index=False)
    example_data["observations-sst"].to_parquet(obssst_file, index=False)

    return tmp_path


@pytest.fixture
def feather_path(tmp_path, example_data):
    header_file = tmp_path / "header-test.feather"
    obssst_file = tmp_path / "observations-sst-test.feather"

    example_data["header"].to_feather(
        header_file,
    )
    example_data["observations-sst"].to_feather(obssst_file)

    return tmp_path


def test_read_data_csv(csv_path, example_data):
    bundle = read_tables(csv_path, delimiter=",")
    pd.testing.assert_frame_equal(bundle.data, example_data.astype(str))


def test_read_data_parquet(parquet_path, example_data):
    bundle = read_tables(parquet_path, data_format="parquet")
    pd.testing.assert_frame_equal(bundle.data, example_data)


def test_read_data_feather(feather_path, example_data):
    bundle = read_tables(feather_path, data_format="feather")
    pd.testing.assert_frame_equal(bundle.data, example_data)


def test_write_data_csv(tmp_path, example_data):
    write_tables(example_data, out_dir=tmp_path, suffix="test")
    data_header = pd.read_csv(tmp_path / "header-test.csv", delimiter="|")
    data_obssst = pd.read_csv(tmp_path / "observations-sst-test.csv", delimiter="|")
    pd.testing.assert_frame_equal(example_data["header"], data_header)
    pd.testing.assert_frame_equal(example_data["observations-sst"], data_obssst)


def test_write_data_parquet(tmp_path, example_data):
    write_tables(example_data, out_dir=tmp_path, suffix="test", data_format="parquet")
    data_header = pd.read_parquet(tmp_path / "header-test.parquet")
    data_obssst = pd.read_parquet(tmp_path / "observations-sst-test.parquet")
    pd.testing.assert_frame_equal(example_data["header"], data_header)
    pd.testing.assert_frame_equal(example_data["observations-sst"], data_obssst)


def test_write_data_feather(tmp_path, example_data):
    write_tables(example_data, out_dir=tmp_path, suffix="test", data_format="feather")
    data_header = pd.read_feather(tmp_path / "header-test.feather")
    data_obssst = pd.read_feather(tmp_path / "observations-sst-test.feather")
    pd.testing.assert_frame_equal(example_data["header"], data_header)
    pd.testing.assert_frame_equal(example_data["observations-sst"], data_obssst)
