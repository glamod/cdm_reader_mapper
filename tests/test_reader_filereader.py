from __future__ import annotations

import pytest

import pandas as pd
import xarray as xr

from io import StringIO

from pandas.io.parsers import TextFileReader
from pandas.testing import assert_frame_equal, assert_index_equal

from cdm_reader_mapper import DataBundle

from cdm_reader_mapper.mdf_reader.utils.parser import OrderSpec, ParserConfig

from cdm_reader_mapper.mdf_reader.utils.filereader import (
    _apply_or_chunk,
    _merge_kwargs,
    _apply_multiindex,
    _select_years,
    FileReader,
)


def f(x, y):
    return x + y


def test_merge_kwargs_success():
    out = _merge_kwargs({"a": 1}, {"b": 2})
    assert out == {"a": 1, "b": 2}


def test_merge_kwargs_duplicate_key():
    with pytest.raises(ValueError):
        _merge_kwargs({"a": 1}, {"a": 2})


def test_apply_multiindex_no_tuples():
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = _apply_multiindex(df)
    assert out.columns.equals(df.columns)


def test_apply_multiindex_with_tuples():
    df = pd.DataFrame({("core", "YR"): [2010], ("core", "MO"): [7]})
    out = _apply_multiindex(df)
    assert isinstance(out.columns, pd.MultiIndex)
    assert out.columns.tolist() == [("core", "YR"), ("core", "MO")]


def test_select_years_no_selection():
    df = pd.DataFrame({"YR": [2000, 2001]})
    out = _select_years(df, (None, None), "YR")
    pd.testing.assert_frame_equal(out, df)


def test_select_years_range():
    df = pd.DataFrame({"YR": [1999, 2000, 2001, 2002]})
    out = _select_years(df, (2000, 2001), "YR")
    assert out["YR"].tolist() == [2000, 2001]


def test_select_years_handles_non_numeric():
    df = pd.DataFrame({"YR": ["2000", "bad", "2001"]})
    out = _select_years(df, (2000, 2001), "YR")
    assert out["YR"].tolist() == ["2000", "2001"]


def test_apply_or_chunk_dataframe():
    df = pd.DataFrame({"test": [1, 2, 3, 4]})
    out = _apply_or_chunk(df, f, func_args=[2])
    assert isinstance(out, pd.DataFrame)
    assert_frame_equal(out, pd.DataFrame({"test": [3, 4, 5, 6]}))


def test_apply_or_chunk_textfilereader():
    buffer = StringIO("test\n1\n2\n3\n4")
    read_kwargs = {"chunksize": 2}
    reader = pd.read_csv(buffer, **read_kwargs)
    (out,) = _apply_or_chunk(reader, f, func_args=[2], read_kwargs=read_kwargs)
    assert isinstance(out, TextFileReader)
    assert_frame_equal(out.read(), pd.DataFrame({"test": [3, 4, 5, 6]}))


@pytest.fixture
def dtypes():
    return {
        ("core", "YR"): "Int64",
        ("core", "MO"): "Int64",
        ("core", "DY"): "Int64",
        ("core", "HR"): "Int64",
    }


@pytest.fixture
def fake_pandas_df():
    data = {
        "0": [
            "2010 7 1  100",
            "2010 7 2  200",
            "2010 7 3  300",
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def fake_pandas_df_file(fake_pandas_df, tmp_path):
    file_path = tmp_path / "fake_dataframe.csv"
    fake_pandas_df.to_csv(file_path, header=False, index=False)
    return file_path


@pytest.fixture
def fake_xr_dataset():
    return xr.Dataset(
        {
            "YR": ("time", [2010, 2010, 2010]),
            "MO": ("time", [7, 7, 7]),
            "DY": ("time", [1, 2, 3]),
            "HR": ("time", [10, 20, 30]),
        },
        coords={"time": [0, 1, 2]},
        attrs={"source": "fake"},
    )


@pytest.fixture
def fake_xr_dataset_file(fake_xr_dataset, tmp_path):
    file_path = tmp_path / "fake_dataset.nc"
    fake_xr_dataset.to_netcdf(file_path)
    return file_path


@pytest.fixture
def fake_out_dataset(dtypes):
    data = {
        ("core", "YR"): [2010, 2010, 2010],
        ("core", "MO"): [7, 7, 7],
        ("core", "DY"): [1, 2, 3],
        ("core", "HR"): [10, 20, 30],
    }
    df = pd.DataFrame(data)

    for col, dtype in dtypes.items():
        df[col] = df[col].astype(dtype)

    return df


@pytest.fixture
def fake_config(dtypes):
    order_specs = {
        "core": OrderSpec(
            header={"length": 12, "field_layout": "fixed_width"},
            elements={
                "YR": {"index": ("core", "YR"), "ignore": False, "field_length": 4},
                "MO": {"index": ("core", "MO"), "ignore": False, "field_length": 2},
                "DY": {"index": ("core", "DY"), "ignore": False, "field_length": 2},
                "HR": {"index": ("core", "HR"), "ignore": False, "field_length": 4},
            },
            is_delimited=False,
        )
    }
    return ParserConfig(
        order_specs=order_specs,
        disable_reads=[],
        dtypes=dtypes,
        parse_dates=[],
        convert_decode={
            "converter_dict": {},
            "converter_kwargs": {},
            "decoder_dict": {},
        },
        validation={},
        encoding="utf-8",
    )


@pytest.fixture
def reader_pd(fake_config):
    r = FileReader("icoads")
    # override config for test
    r.config = fake_config
    return r


@pytest.fixture
def reader_xr(fake_config):
    r = FileReader("craid")
    # override config for test
    r.config = fake_config
    return r


def test_process_data_pandas(reader_pd, fake_pandas_df, fake_out_dataset):
    data, mask, config = reader_pd._process_data(
        fake_pandas_df,
        convert_flag=False,
        decode_flag=False,
        converter_dict=None,
        converter_kwargs=None,
        decoder_dict=None,
        validate_flag=False,
        ext_table_path=None,
        sections=None,
        excludes=None,
        config=reader_pd.config,
        parse_mode="pandas",
    )

    assert isinstance(data, pd.DataFrame)
    assert isinstance(mask, pd.DataFrame)
    assert_index_equal(data.columns, mask.columns)
    assert len(data) == len(mask)

    assert config.columns is not None

    assert_frame_equal(data, fake_out_dataset)
    assert_index_equal(data.columns, config.columns)

    assert mask.all().all()


def test_process_data_netcdf(reader_xr, fake_xr_dataset, fake_out_dataset):
    data, mask, config = reader_xr._process_data(
        fake_xr_dataset,
        convert_flag=False,
        decode_flag=False,
        converter_dict=None,
        converter_kwargs=None,
        decoder_dict=None,
        validate_flag=False,
        ext_table_path=None,
        sections=None,
        excludes=None,
        config=reader_xr.config,
        parse_mode="netcdf",
    )

    assert isinstance(data, pd.DataFrame)
    assert isinstance(mask, pd.DataFrame)
    assert_index_equal(data.columns, mask.columns)
    assert len(data) == len(mask)

    assert config.columns is not None

    assert_frame_equal(data, fake_out_dataset)
    assert_index_equal(data.columns, config.columns)

    assert mask.all().all()


def test_open_data_pandas(reader_pd, fake_pandas_df_file, fake_out_dataset):
    data, mask, config = reader_pd.open_data(
        fake_pandas_df_file,
        open_with="pandas",
    )
    assert isinstance(data, pd.DataFrame)
    assert isinstance(mask, pd.DataFrame)
    assert_index_equal(data.columns, mask.columns)
    assert len(data) == len(mask)

    assert config.columns is not None

    assert_frame_equal(data, fake_out_dataset)
    assert_index_equal(data.columns, config.columns)

    assert mask.all().all()


def test_open_data_netcdf(reader_xr, fake_xr_dataset_file, fake_out_dataset):
    data, mask, config = reader_xr.open_data(
        fake_xr_dataset_file,
        open_with="netcdf",
    )
    assert isinstance(data, pd.DataFrame)
    assert isinstance(mask, pd.DataFrame)
    assert_index_equal(data.columns, mask.columns)
    assert len(data) == len(mask)

    assert config.columns is not None

    assert_frame_equal(data, fake_out_dataset)
    assert_index_equal(data.columns, config.columns)

    assert mask.all().all()


def test_read_pandas(reader_pd, fake_pandas_df_file, dtypes, fake_out_dataset):
    databundle = reader_pd.read(
        fake_pandas_df_file,
    )
    assert isinstance(databundle, DataBundle)
    assert hasattr(databundle, "data")
    assert hasattr(databundle, "mask")
    assert hasattr(databundle, "columns")
    assert hasattr(databundle, "dtypes")
    assert hasattr(databundle, "parse_dates")
    assert hasattr(databundle, "encoding")
    assert hasattr(databundle, "imodel")

    data = databundle.data
    mask = databundle.mask

    assert isinstance(data, pd.DataFrame)
    assert isinstance(mask, pd.DataFrame)
    assert_index_equal(data.columns, mask.columns)
    assert len(data) == len(mask)
    assert_frame_equal(data, fake_out_dataset)

    assert_index_equal(data.columns, databundle.columns)

    assert mask.all().all()

    assert databundle.dtypes == dtypes
    assert databundle.parse_dates == []
    assert databundle.encoding == "utf-8"
    assert databundle.imodel == reader_pd.imodel


def test_read_netcdf(reader_xr, fake_xr_dataset_file, dtypes, fake_out_dataset):
    databundle = reader_xr.read(
        fake_xr_dataset_file,
    )
    assert isinstance(databundle, DataBundle)
    assert hasattr(databundle, "data")
    assert hasattr(databundle, "mask")
    assert hasattr(databundle, "columns")
    assert hasattr(databundle, "dtypes")
    assert hasattr(databundle, "parse_dates")
    assert hasattr(databundle, "encoding")
    assert hasattr(databundle, "imodel")

    data = databundle.data
    mask = databundle.mask

    assert isinstance(data, pd.DataFrame)
    assert isinstance(mask, pd.DataFrame)
    assert_index_equal(data.columns, mask.columns)
    assert len(data) == len(mask)
    assert_frame_equal(data, fake_out_dataset)

    assert_index_equal(data.columns, databundle.columns)

    assert mask.all().all()

    assert databundle.dtypes == dtypes
    assert databundle.parse_dates == []
    assert databundle.encoding == "utf-8"
    assert databundle.imodel == reader_xr.imodel
