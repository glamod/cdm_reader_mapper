from __future__ import annotations

import pandas as pd
import pytest

from cdm_reader_mapper import read_mdf, test_data
from cdm_reader_mapper.common.pandas_TextParser_hdlr import make_copy
from cdm_reader_mapper.operations import replace

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
    read_.select_true(overwrite=False, out_rejected=True)
    data = read_.data
    selected = read_.selected
    deselected = read_.deselected

    if TextParser is True:
        data = make_copy(data).read()
        selected = make_copy(selected).read()
        deselected = make_copy(deselected).read()

    exp1 = data[:4].reset_index(drop=True)
    exp2 = data[4:].reset_index(drop=True)
    pd.testing.assert_frame_equal(exp1, selected)
    pd.testing.assert_frame_equal(exp2, deselected)


@pytest.mark.parametrize("TextParser", [False, True])
def test_select_from_index(TextParser):
    read_ = _get_data(TextParser)
    read_.select_from_index([0, 2, 4], overwrite=False)
    data = read_.data
    result = read_.selected

    if TextParser is True:
        data = make_copy(data).read()
        result = make_copy(result).read()

    idx = data.index.isin([0, 2, 4])
    exp = data[idx].reset_index(drop=True)
    pd.testing.assert_frame_equal(exp, result)


@pytest.mark.parametrize("TextParser", [True, False])
def test_select_from_list(TextParser):
    read_ = _get_data(TextParser)
    selection = {("c1", "B1"): [26, 41]}
    read_.select_from_list(selection, overwrite=False, out_rejected=True, in_index=True)
    data = read_.data
    selected = read_.selected
    deselected = read_.deselected

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
    table_df = cdm_header.tables["header"]
    result = replace.replace_columns(
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
