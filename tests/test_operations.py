from __future__ import annotations

import pandas as pd
import pytest

from cdm_reader_mapper import mdf_reader, test_data
from cdm_reader_mapper.operations import inspect, replace, select

from ._results import correction_df, table_df

data_dict = dict(test_data.test_icoads_r300_d721)


def _read_data(**kwargs):
    return mdf_reader.read(**kwargs)


def _get_data(TextParser, **kwargs):
    return _read_data(**data_dict, imodel="icoads_r300_d721", **kwargs)


def test_select_true():
    read_ = _get_data(sections=["c99_data"])
    data = read_.data
    mask = read_.mask
    result = select.select_true(data, mask, out_rejected=True)

    exp1 = data[:4].reset_index(drop=True)
    exp2 = data[4:].reset_index(drop=True)
    pd.testing.assert_frame_equal(exp1, result[0])
    pd.testing.assert_frame_equal(exp2, result[1])


def test_select_from_index():
    read_ = _get_data()
    data = read_.data
    result = select.select_from_index(data, [0, 2, 4])

    idx = data.index.isin([0, 2, 4])
    exp = data[idx].reset_index(drop=True)
    pd.testing.assert_frame_equal(exp, result)


def test_select_from_list():
    read_ = _get_data()
    data = read_.data
    selection = {("c1", "B1"): [26, 41]}
    result = select.select_from_list(data, selection, out_rejected=True, in_index=True)

    idx1 = data.index.isin([1, 3])
    idx2 = data.index.isin([0, 2, 4])
    exp1 = data[idx1].reset_index(drop=True)
    exp2 = data[idx2].reset_index(drop=True)
    pd.testing.assert_frame_equal(exp1, result[0])
    pd.testing.assert_frame_equal(exp2, result[1])


def test_inspect_get_length():
    read_ = _get_data()
    result = inspect.get_length(read_.data)
    assert result == 5


def test_inspect_count_by_cat():
    read_ = _get_data()
    result = inspect.count_by_cat(read_.data, ("c1", "B1"))
    assert result == {19: 1, 26: 1, 27: 1, 41: 1, 91: 1}


def test_replace():
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
