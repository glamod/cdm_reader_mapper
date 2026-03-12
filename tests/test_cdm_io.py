from __future__ import annotations

import pandas as pd
import pytest

from cdm_reader_mapper import DataBundle
from cdm_reader_mapper.common import logging_hdlr

from cdm_reader_mapper.cdm_mapper.reader import (
    _read_single_file,
    _read_multiple_files,
    read_tables,
)
from cdm_reader_mapper.cdm_mapper.writer import _table_to_file, write_tables


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
def empty_data():
    return pd.DataFrame({"report_id": []})


@pytest.fixture
def csv_path(tmp_path, example_data, empty_data):
    header_file = tmp_path / "header.csv"
    obssst_file = tmp_path / "observations-sst.csv"
    empty_file = tmp_path / "observations-at.csv"

    example_data["header"].to_csv(header_file, index=False)
    example_data["observations-sst"].to_csv(obssst_file, index=False)
    empty_data.to_csv(empty_file, index=False)

    return tmp_path


@pytest.fixture
def csv_path_prefix_suffix(tmp_path, example_data):
    header_file = tmp_path / "prefix-header-suffix.csv"
    obssst_file = tmp_path / "prefix-observations-sst-suffix.csv"

    example_data["header"].to_csv(header_file, index=False)
    example_data["observations-sst"].to_csv(obssst_file, index=False)

    return tmp_path


@pytest.fixture
def parquet_path(tmp_path, example_data):
    header_file = tmp_path / "header.parquet"
    obssst_file = tmp_path / "observations-sst.parquet"

    example_data["header"].to_parquet(header_file, index=False)
    example_data["observations-sst"].to_parquet(obssst_file, index=False)

    return tmp_path


@pytest.fixture
def feather_path(tmp_path, example_data):
    header_file = tmp_path / "header.feather"
    obssst_file = tmp_path / "observations-sst.feather"

    example_data["header"].to_feather(
        header_file,
    )
    example_data["observations-sst"].to_feather(obssst_file)

    return tmp_path


def test_read_single_file_subset(csv_path, example_data):
    df = _read_single_file(csv_path / "header.csv", "csv", "header", None)

    assert isinstance(df, pd.DataFrame)

    exp = example_data["header"].set_index("report_id", drop=False)
    pd.testing.assert_frame_equal(df, exp)


def test_read_single_file_null(csv_path, example_data):
    df = _read_single_file(
        csv_path / "header.csv", "csv", ["header"], None, null_label=3
    )

    assert isinstance(df, pd.DataFrame)

    exp = example_data["header"].set_index("report_id", drop=False).drop(3)
    pd.testing.assert_frame_equal(df, exp)


def test_read_multiple_files_subset(csv_path, example_data):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    df_list = _read_multiple_files(
        csv_path, "csv", extension="csv", cdm_subset="header", logger=logger
    )

    assert isinstance(df_list, list)
    assert len(df_list) == 1

    df = df_list[0]

    assert isinstance(df, pd.DataFrame)

    exp = example_data["header"].set_index("report_id", drop=False)
    pd.testing.assert_frame_equal(df["header"], exp)


def test_read_multiple_files_prefix_suffix(csv_path_prefix_suffix, example_data):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    args = (csv_path_prefix_suffix, "csv")
    kwargs = {
        "cdm_subset": ["header", "observations-sst"],
        "extension": "csv",
        "logger": logger,
    }
    df_list_1 = _read_multiple_files(
        *args,
        **kwargs,
        prefix="*",
        suffix="*",
    )

    assert isinstance(df_list_1, list)
    assert len(df_list_1) == 2

    header_1 = df_list_1[0]
    obs_1 = df_list_1[1]

    assert isinstance(header_1, pd.DataFrame)
    assert isinstance(obs_1, pd.DataFrame)

    df_list_2 = _read_multiple_files(
        *args,
        **kwargs,
        prefix="prefix",
        suffix="suffix",
    )

    assert isinstance(df_list_2, list)
    assert len(df_list_2) == 2

    header_2 = df_list_2[0]
    obs_2 = df_list_2[1]

    assert isinstance(header_2, pd.DataFrame)
    assert isinstance(obs_2, pd.DataFrame)

    pd.testing.assert_frame_equal(header_2, header_2)
    pd.testing.assert_frame_equal(obs_1, obs_2)


def test_read_multiple_files_false_table(csv_path, example_data):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    df_list = _read_multiple_files(
        csv_path,
        "csv",
        extension="csv",
        cdm_subset=["header", "false_table"],
        logger=logger,
    )

    assert isinstance(df_list, list)
    assert len(df_list) == 1

    df = df_list[0]

    assert isinstance(df, pd.DataFrame)

    exp = example_data["header"].set_index("report_id", drop=False)
    pd.testing.assert_frame_equal(df["header"], exp)


def test_read_multiple_files_raises(csv_path, example_data):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    with pytest.raises(FileNotFoundError, match="No files found matching pattern"):
        _read_multiple_files(
            csv_path,
            "csv",
            cdm_subset="header",
            extension="invalid_extension",
            logger=logger,
        )


def test_read_data_csv(csv_path, example_data):
    bundle = read_tables(csv_path, delimiter=",")

    assert isinstance(bundle, DataBundle)
    assert hasattr(bundle, "data")

    data = bundle.data
    assert isinstance(data, pd.DataFrame)

    pd.testing.assert_frame_equal(bundle.data, example_data.astype(str))


def test_read_data_single_csv(csv_path, example_data):
    bundle = read_tables(csv_path / "observations-sst.csv", delimiter=",")

    assert isinstance(bundle, DataBundle)
    assert hasattr(bundle, "data")

    data = bundle.data
    assert isinstance(data, pd.DataFrame)

    pd.testing.assert_frame_equal(
        bundle.data, example_data["observations-sst"].astype(str)
    )


def test_read_data_raises_data_format(csv_path):
    with pytest.raises(ValueError, match="data_format must be one of"):
        read_tables(csv_path, delimiter=",", data_format="invalid_format")


def test_read_data_raises_filenotfound(csv_path):
    with pytest.raises(
        FileNotFoundError,
        match="Source is neither a valid file name nor a valid directory path",
    ):
        read_tables(csv_path / "header_invalid.csv", delimiter=",")


def test_read_data_raises_empty(csv_path):
    with pytest.raises(ValueError, match="All tables empty in file system"):
        read_tables(csv_path / "observations-at.csv", delimiter=",")


def test_read_data_parquet(parquet_path, example_data):
    bundle = read_tables(parquet_path, data_format="parquet")

    assert isinstance(bundle, DataBundle)
    assert hasattr(bundle, "data")

    data = bundle.data
    assert isinstance(data, pd.DataFrame)

    pd.testing.assert_frame_equal(bundle.data, example_data)


def test_read_data_feather(feather_path, example_data):
    bundle = read_tables(feather_path, data_format="feather")

    assert isinstance(bundle, DataBundle)
    assert hasattr(bundle, "data")

    data = bundle.data
    assert isinstance(data, pd.DataFrame)

    pd.testing.assert_frame_equal(bundle.data, example_data)


def test_table_to_file_raises(csv_path, example_data):
    with pytest.raises(ValueError, match="data_format must be one of"):
        _table_to_file(example_data, "invalid.csv", data_format="invalid_format")


def test_write_data_csv(tmp_path, example_data):
    write_tables(example_data, out_dir=tmp_path)
    data_header = pd.read_csv(tmp_path / "header.csv", delimiter="|")
    data_obssst = pd.read_csv(tmp_path / "observations-sst.csv", delimiter="|")
    pd.testing.assert_frame_equal(example_data["header"], data_header)
    pd.testing.assert_frame_equal(example_data["observations-sst"], data_obssst)


def test_write_data_raises(tmp_path, example_data):
    with pytest.raises(ValueError):
        write_tables(example_data, out_dir=tmp_path, data_format="invalid_format")


def test_write_data_empty(tmp_path, empty_data):
    write_tables(empty_data, outdir=tmp_path)


def test_write_data_parquet(tmp_path, example_data):
    write_tables(example_data, out_dir=tmp_path, data_format="parquet")
    data_header = pd.read_parquet(tmp_path / "header.parquet")
    data_obssst = pd.read_parquet(tmp_path / "observations-sst.parquet")
    pd.testing.assert_frame_equal(example_data["header"], data_header)
    pd.testing.assert_frame_equal(example_data["observations-sst"], data_obssst)


def test_write_data_feather(tmp_path, example_data):
    write_tables(example_data, out_dir=tmp_path, data_format="feather")
    data_header = pd.read_feather(tmp_path / "header.feather")
    data_obssst = pd.read_feather(tmp_path / "observations-sst.feather")
    pd.testing.assert_frame_equal(example_data["header"], data_header)
    pd.testing.assert_frame_equal(example_data["observations-sst"], data_obssst)
