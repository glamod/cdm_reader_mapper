from __future__ import annotations

import pandas as pd
import pytest

from cdm_reader_mapper import read_mdf, test_data
from cdm_reader_mapper.common import replace_columns
from cdm_reader_mapper.common.pandas_TextParser_hdlr import make_copy

from ._results import cdm_header, correction_df

data_dict = dict(test_data.test_icoads_r300_d721)


def _read_data(**kwargs):
    return read_mdf(**kwargs)


def _get_data(TextParser, **kwargs):
    if TextParser is True:
        kwargs["chunksize"] = 10000
    return _read_data(**data_dict, imodel="icoads_r300_d721", **kwargs)


@pytest.mark.parametrize("TextParser", [True, False])
def test_select_true(TextParser):
    read_ = _get_data(TextParser, sections=["c99_data"])
    result = read_.select_true(overwrite=False, out_rejected=True)
    selected = result[0]
    deselected = result[1]
    data = read_.data

    if TextParser is True:
        data = make_copy(data).read()
        selected = make_copy(selected).read()
        deselected = make_copy(deselected).read()

    exp1 = data[:5].reset_index(drop=True)
    exp2 = data[5:].reset_index(drop=True)
    pd.testing.assert_frame_equal(exp1, selected)
    pd.testing.assert_frame_equal(exp2, deselected)


@pytest.mark.parametrize("TextParser", [False, True])
def test_select_from_index(TextParser):
    read_ = _get_data(TextParser)
    result = read_.select_from_index([0, 2, 4], overwrite=False)
    data = read_.data
    selected = result[0]

    if TextParser is True:
        data = make_copy(data).read()
        selected = make_copy(selected).read()

    idx = data.index.isin([0, 2, 4])
    exp = data[idx].reset_index(drop=True)
    pd.testing.assert_frame_equal(exp, selected)


@pytest.mark.parametrize("TextParser", [True, False])
def test_select_from_list(TextParser):
    read_ = _get_data(TextParser)
    selection = {("c1", "B1"): [26, 41]}
    result = read_.select_from_list(
        selection, overwrite=False, out_rejected=True, in_index=True
    )
    selected = result[0]
    deselected = result[1]
    data = read_.data

    if TextParser is True:
        data = make_copy(data).read()
        selected = make_copy(selected).read()
        deselected = make_copy(deselected).read()

    idx1 = data.index.isin([1, 3])
    idx2 = data.index.isin([0, 2, 4])
    exp1 = data[idx1].reset_index(drop=True)
    exp2 = data[idx2].reset_index(drop=True)
    pd.testing.assert_frame_equal(exp1, selected)
    pd.testing.assert_frame_equal(exp2, deselected)


@pytest.mark.parametrize("TextParser", [True, False])
def test_inspect_count_by_cat(TextParser):
    read_ = _get_data(TextParser)
    result = read_.unique(columns=("c1", "B1"))
    assert result == {("c1", "B1"): {19: 1, 26: 1, 27: 1, 41: 1, 91: 1}}


def test_replace():
    tables = cdm_header.data.copy()
    table_df = tables["header"]
    result = replace_columns(
        table_df,
        correction_df,
        pivot_c="report_id",
        rep_c=["primary_station_id", "primary_station_id.isChange"],
    )
    table_df["primary_station_id"] = [
        "MASKSTID2",
        "MASKSTID2",
        "MASKSTID",
        "MASKSTID2",
        "MASKSTID2",
    ]
    pd.testing.assert_frame_equal(table_df, result)
