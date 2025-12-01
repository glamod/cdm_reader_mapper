from __future__ import annotations

import os

import pandas as pd
import pytest

from cdm_reader_mapper import test_data
from cdm_reader_mapper.mdf_reader.reader import (
    read_mdf,
    read_data,
)


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
        expected.data = expected.data[select]
        expected.mask = expected.mask[select]

    if drop:
        result.data = result.data.drop(columns=drop)
        result.mask = result.mask.drop(columns=drop)
        expected.data = expected.data.drop(columns=drop)
        expected.mask = expected.mask.drop(columns=drop)

    if drop_idx:
        expected.data = _drop_rows(expected.data, drop_idx)
        expected.mask = _drop_rows(expected.mask, drop_idx)

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
def test_read_mdf_test_data(data_model):
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
        (
            "icoads_r300_d714",
            {"sections": ["core", "c99"], "chunksize": 3},
            ["core", "c99"],
        ),
    ],
)
def test_read_mdf_test_data_select(data_model, kwargs, select):
    _read_mdf_test_data(data_model, select=select, **kwargs)


def test_read_mdf_test_data_drop():
    _read_mdf_test_data("icoads_r300_mixed", drop=["c99"], encoding="cp1252")


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
