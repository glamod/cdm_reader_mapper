from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from cdm_reader_mapper import test_data, DataBundle
from cdm_reader_mapper.mdf_reader.reader import (
    read_mdf,
    read_data,
    validate_read_mdf_args,
)
from cdm_reader_mapper.mdf_reader.utils.filereader import _apply_multiindex
from cdm_reader_mapper.mdf_reader.utils.utilities import ParquetStreamReader


def _get_columns(columns, select):
    if isinstance(columns, pd.MultiIndex):
        return columns.get_level_values(0).isin(select)
    mask = [(type(c) is tuple and c[0] in select) or (c in select) for c in columns]
    return np.array(mask)


def _drop_rows(df, drops):
    if drops == "all":
        return df.drop(df.index)
    elif drops:
        return df.drop(drops).reset_index(drop=True)
    return df


def _read_mdf_test_data(data_model, select=None, drop=None, drop_idx=None, **kwargs):
    source = test_data[f"test_{data_model}"]["source"]
    result = read_mdf(source, imodel=data_model, **kwargs)

    data = test_data[f"test_{data_model}"]["mdf_data"]
    mask = test_data[f"test_{data_model}"]["mdf_mask"]
    info = test_data[f"test_{data_model}"]["mdf_info"]

    expected = read_data(data, mask=mask, info=info)

    if not isinstance(result.data, pd.DataFrame):
        result.data = result.data.read()

    if not isinstance(result.mask, pd.DataFrame):
        result.mask = result.mask.read()

    if select:
        selected = _get_columns(expected.data.columns, select)
        expected.data = expected.data.loc[:, selected]
        expected.mask = expected.mask.loc[:, selected]

    if drop:
        unselected = _get_columns(expected.data.columns, drop)
        expected.data = expected.data.loc[:, ~unselected]
        expected.mask = expected.mask.loc[:, ~unselected]

    if drop_idx:
        expected.data = _drop_rows(expected.data, drop_idx)
        expected.mask = _drop_rows(expected.mask, drop_idx)

    expected.data = _apply_multiindex(expected.data)
    expected.mask = _apply_multiindex(expected.mask)

    pd.testing.assert_frame_equal(result.data, expected.data)
    pd.testing.assert_frame_equal(result.mask, expected.mask)


@pytest.mark.parametrize(
    "data_model",
    [
        "icoads_r300_d714",
        "icoads_r300_d701",
        "icoads_r300_d706",
        "icoads_r300_d705",
        "icoads_r300_d702",
        "icoads_r300_d707",
        "icoads_r302_d794",
        "icoads_r300_d704",
        "icoads_r300_d721",
        "icoads_r300_d730",
        "icoads_r300_d781",
        "icoads_r300_d703",
        "icoads_r300_d201",
        "icoads_r300_d892",
        "icoads_r300_d700",
        "icoads_r302_d792",
        "icoads_r302_d992",
        "craid",
        "gdac",
    ],
)
def test_read_mdf_test_data_basic(data_model):
    _read_mdf_test_data(data_model)


@pytest.mark.parametrize(
    "data_model, kwargs",
    [
        ("icoads_r300_d714", {"chunksize": 3}),
        ("icoads_r300_d721", {"chunksize": 3}),
        (
            "icoads_r300_d703",
            {
                "ext_schema_path": os.path.join(
                    ".", "cdm_reader_mapper", "mdf_reader", "schemas", "icoads"
                )
            },
        ),
        (
            "icoads_r300_d703",
            {
                "ext_table_path": os.path.join(
                    ".", "cdm_reader_mapper", "mdf_reader", "codes", "icoads"
                )
            },
        ),
        (
            "icoads_r300_d703",
            {
                "ext_table_path": os.path.join(
                    ".", "cdm_reader_mapper", "mdf_reader", "codes", "icoads"
                ),
                "ext_schema_path": os.path.join(
                    ".", "cdm_reader_mapper", "mdf_reader", "schemas", "icoads"
                ),
            },
        ),
        (
            "icoads_r300_d703",
            {
                "ext_schema_file": os.path.join(
                    ".",
                    "cdm_reader_mapper",
                    "mdf_reader",
                    "schemas",
                    "icoads",
                    "icoads.json",
                )
            },
        ),
    ],
)
def test_read_mdf_test_data_kwargs(data_model, kwargs):
    _read_mdf_test_data(data_model, **kwargs)


@pytest.mark.parametrize(
    "data_model, kwargs, select",
    [
        ("icoads_r300_d714", {"sections": ["c99"], "chunksize": 3}, ["c99"]),
        ("icoads_r300_d714", {"sections": ["c99"]}, ["c99"]),
        ("icoads_r300_d714", {"sections": "c99"}, ["c99"]),
        (
            "icoads_r300_d714",
            {"sections": ["core", "c99"]},
            ["core", "c99"],
        ),
        ("craid", {"sections": ["drifter_measurements"]}, ["drifter_measurements"]),
    ],
)
def test_read_mdf_test_data_select(data_model, kwargs, select):
    _read_mdf_test_data(data_model, select=select, **kwargs)


@pytest.mark.parametrize(
    "data_model, kwargs, drop",
    [
        ("icoads_r300_d714", {"excludes": ["c98"]}, ["c98"]),
        ("icoads_r300_d714", {"excludes": "c98"}, ["c98"]),
        ("icoads_r300_d714", {"excludes": ["c5", "c98"]}, ["c5", "c98"]),
        ("icoads_r300_mixed", {"excludes": ["c99"], "encoding": "cp1252"}, ["c99"]),
        ("icoads_r300_mixed", {"excludes": "c99", "encoding": "cp1252"}, ["c99"]),
        (
            "craid",
            {"excludes": ["drifter_measurements", "drifter_history"]},
            ["drifter_measurements", "drifter_history"],
        ),
        ("gdac", {"excludes": "AAAA"}, ["AAAA"]),
    ],
)
def test_read_mdf_test_data_exclude(data_model, kwargs, drop):
    _read_mdf_test_data(data_model, drop=drop, **kwargs)


@pytest.mark.parametrize(
    "data_model, kwargs, drop_idx",
    [
        ("icoads_r300_d702", {"year_init": 1874}, [0, 1, 2, 3, 4]),
        ("icoads_r300_d702", {"year_end": 1874}, [5, 6, 7, 8, 9]),
        (
            "icoads_r300_d702",
            {"year_init": 1874, "year_end": 1874},
            "all",
        ),
        ("gdac", {"year_init": 2002}, [0, 1, 2, 3, 4]),
        ("craid", {"year_end": 2003}, "all"),
    ],
)
def test_read_mdf_test_data_drop_idx(data_model, kwargs, drop_idx):
    _read_mdf_test_data(data_model, drop_idx=drop_idx, **kwargs)


def test_read_data_basic():
    data_model = "icoads_r300_d721"
    data = test_data[f"test_{data_model}"]["mdf_data"]
    mask = test_data[f"test_{data_model}"]["mdf_mask"]
    info = test_data[f"test_{data_model}"]["mdf_info"]
    db = read_data(data, mask, info)

    assert isinstance(db, DataBundle)

    for attr in [
        "data",
        "mask",
        "columns",
        "dtypes",
        "parse_dates",
        "encoding",
        "imodel",
        "mode",
    ]:
        assert hasattr(db, attr)

    assert isinstance(db.data, pd.DataFrame)
    assert isinstance(db.mask, pd.DataFrame)
    assert isinstance(db.columns, pd.MultiIndex)
    assert isinstance(db.dtypes, dict)
    assert isinstance(db.parse_dates, list)
    assert isinstance(db.encoding, str)
    assert db.encoding == "cp1252"
    assert db.imodel is None
    assert isinstance(db.mode, str)
    assert db.mode == "data"
    assert len(db) == 5
    assert db.shape == (5, 341)
    assert db.size == 1705


def test_read_data_no_mask():
    data_model = "icoads_r300_d721"
    data = test_data[f"test_{data_model}"]["mdf_data"]
    info = test_data[f"test_{data_model}"]["mdf_info"]
    db = read_data(data, info=info)

    assert isinstance(db, DataBundle)

    for attr in [
        "data",
        "mask",
        "columns",
        "dtypes",
        "parse_dates",
        "encoding",
        "imodel",
        "mode",
    ]:
        assert hasattr(db, attr)

    assert isinstance(db.data, pd.DataFrame)
    assert isinstance(db.mask, pd.DataFrame)
    assert isinstance(db.columns, pd.MultiIndex)
    assert isinstance(db.dtypes, dict)
    assert isinstance(db.parse_dates, list)
    assert isinstance(db.encoding, str)
    assert db.encoding == "cp1252"
    assert db.imodel is None
    assert isinstance(db.mode, str)
    assert db.mode == "data"
    assert len(db) == 5
    assert db.shape == (5, 341)
    assert db.size == 1705


def test_read_data_no_info():
    data_model = "icoads_r300_d721"
    data = test_data[f"test_{data_model}"]["mdf_data"]

    db = read_data(data)

    assert isinstance(db, DataBundle)

    for attr in [
        "data",
        "mask",
        "columns",
        "dtypes",
        "parse_dates",
        "encoding",
        "imodel",
        "mode",
    ]:
        assert hasattr(db, attr)

    assert isinstance(db.data, pd.DataFrame)
    assert isinstance(db.mask, pd.DataFrame)
    assert isinstance(db.columns, pd.MultiIndex)
    assert isinstance(db.dtypes, dict)
    assert db.parse_dates is False
    assert db.encoding is None
    assert db.imodel is None
    assert isinstance(db.mode, str)
    assert db.mode == "data"
    assert len(db) == 5
    assert db.shape == (5, 341)
    assert db.size == 1705


def test_read_data_col_subset():
    data_model = "icoads_r300_d721"
    data = test_data[f"test_{data_model}"]["mdf_data"]
    info = test_data[f"test_{data_model}"]["mdf_info"]
    db = read_data(data, info=info, col_subset="core")

    assert isinstance(db, DataBundle)

    for attr in [
        "data",
        "mask",
        "columns",
        "dtypes",
        "parse_dates",
        "encoding",
        "imodel",
        "mode",
    ]:
        assert hasattr(db, attr)

    assert isinstance(db.data, pd.DataFrame)
    assert isinstance(db.mask, pd.DataFrame)
    assert isinstance(db.columns, pd.Index)
    assert isinstance(db.dtypes, dict)
    assert isinstance(db.parse_dates, list)
    assert isinstance(db.encoding, str)
    assert db.encoding == "cp1252"
    assert db.imodel is None
    assert isinstance(db.mode, str)
    assert db.mode == "data"
    assert len(db) == 5
    assert db.shape == (5, 48)
    assert db.size == 240


def test_read_data_encoding():
    data_model = "icoads_r300_d721"
    data = test_data[f"test_{data_model}"]["mdf_data"]
    db = read_data(data, encoding="cp1252")

    assert isinstance(db, DataBundle)

    for attr in [
        "data",
        "mask",
        "columns",
        "dtypes",
        "parse_dates",
        "encoding",
        "imodel",
        "mode",
    ]:
        assert hasattr(db, attr)

    assert isinstance(db.data, pd.DataFrame)
    assert isinstance(db.mask, pd.DataFrame)
    assert isinstance(db.columns, pd.Index)
    assert isinstance(db.dtypes, dict)
    assert db.parse_dates is False
    assert isinstance(db.encoding, str)
    assert db.encoding == "cp1252"
    assert db.imodel is None
    assert isinstance(db.mode, str)
    assert db.mode == "data"
    assert len(db) == 5
    assert db.shape == (5, 341)
    assert db.size == 1705


def test_read_data_textfilereader():
    data_model = "icoads_r300_d721"
    data = test_data[f"test_{data_model}"]["mdf_data"]
    mask = test_data[f"test_{data_model}"]["mdf_mask"]
    info = test_data[f"test_{data_model}"]["mdf_info"]
    db = read_data(data, mask=mask, info=info, chunksize=3)

    assert isinstance(db, DataBundle)

    for attr in [
        "data",
        "mask",
        "columns",
        "dtypes",
        "parse_dates",
        "encoding",
        "imodel",
        "mode",
    ]:
        assert hasattr(db, attr)

    assert isinstance(db.data, ParquetStreamReader)
    assert isinstance(db.mask, ParquetStreamReader)
    assert isinstance(db.columns, pd.MultiIndex)
    assert isinstance(db.dtypes, dict)
    assert db.parse_dates == []
    assert isinstance(db.encoding, str)
    assert db.encoding == "cp1252"
    assert db.imodel is None
    assert isinstance(db.mode, str)
    assert db.mode == "data"
    assert len(db) == 5
    assert db.shape == (5, 341)
    assert db.size == 1705


def test_validate_read_mdf_args_pass(tmp_path):
    source = tmp_path / "file.mdf"
    source.touch()

    validate_read_mdf_args(
        source=source,
        imodel=object(),
        ext_schema_path=None,
        ext_schema_file=None,
        year_init=2000,
        year_end=2020,
        chunksize=100,
        skiprows=0,
    )


def test_validate_read_mdf_args_invalid_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_read_mdf_args(
            source=tmp_path / "missing.mdf",
            imodel=object(),
            ext_schema_path=None,
            ext_schema_file=None,
            year_init=None,
            year_end=None,
            chunksize=None,
            skiprows=0,
        )


def test_validate_read_mdf_args_missing_all_sources(tmp_path):
    source = tmp_path / "file.mdf"
    source.touch()

    with pytest.raises(
        ValueError,
        match="One of imodel or ext_schema_path/ext_schema_file must be provided",
    ):
        validate_read_mdf_args(
            source=source,
            imodel=None,
            ext_schema_path=None,
            ext_schema_file=None,
            year_init=None,
            year_end=None,
            chunksize=None,
            skiprows=0,
        )


def test_validate_read_mdf_args_invalid_chunksize(tmp_path):
    source = tmp_path / "file.mdf"
    source.touch()

    with pytest.raises(ValueError, match="chunksize must be a positive integer"):
        validate_read_mdf_args(
            source=source,
            imodel=object(),
            ext_schema_path=None,
            ext_schema_file=None,
            year_init=None,
            year_end=None,
            chunksize=0,
            skiprows=0,
        )


def test_validate_read_mdf_args_invalid_skiprows(tmp_path):
    source = tmp_path / "file.mdf"
    source.touch()

    with pytest.raises(ValueError, match="skiprows must be >= 0"):
        validate_read_mdf_args(
            source=source,
            imodel=object(),
            ext_schema_path=None,
            ext_schema_file=None,
            year_init=None,
            year_end=None,
            chunksize=None,
            skiprows=-1,
        )


def test_validate_read_mdf_args_invalid_years(tmp_path):
    source = tmp_path / "file.mdf"
    source.touch()

    with pytest.raises(ValueError, match="year_init must be <= year_end"):
        validate_read_mdf_args(
            source=source,
            imodel=object(),
            ext_schema_path=None,
            ext_schema_file=None,
            year_init=2021,
            year_end=2020,
            chunksize=None,
            skiprows=0,
        )
