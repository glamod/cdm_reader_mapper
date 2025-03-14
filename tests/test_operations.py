from __future__ import annotations

import pandas as pd
import pytest  # noqa

from cdm_reader_mapper import read, test_data
from cdm_reader_mapper.common import replace_columns

from ._results import cdm_header, correction_df

data_dict = dict(test_data.test_icoads_r300_d721)


def _read_data(**kwargs):
    return read(**kwargs)


def _get_data(**kwargs):
    return _read_data(**data_dict, imodel="icoads_r300_d721", **kwargs)


def test_select_true():
    data = _get_data(sections=["c99_data"])
    result = data.select_true(out_rejected=True)
    expected = data.data
    selected = result.data

    expected = expected[:5].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, selected)


def test_select_from_index():
    data = _get_data()
    result = data.select_from_index([0, 2, 4])
    expected = data.data
    selected = result.data

    idx = expected.index.isin([0, 2, 4])
    expected = expected[idx].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, selected)


def test_select_from_list():
    data = _get_data()
    selection = {("c1", "B1"): [26, 41]}
    result = data.select_from_list(selection, out_rejected=True, in_index=True)
    expected = data.data
    selected = result.data

    idx = expected.index.isin([1, 3])
    expected = expected[idx].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, selected)


def test_inspect_count_by_cat():
    data = _get_data()
    result = data.unique(columns=("c1", "B1"))
    assert result == {("c1", "B1"): {19: 1, 26: 1, 27: 1, 41: 1, 91: 1}}


def test_replace():
    result = cdm_header.replace_columns(
        correction_df,
        subset="header",
        pivot_c="report_id",
        rep_c=["primary_station_id", "primary_station_id.isChange"],
    )
    cdm_header[("header", "primary_station_id")] = [
        "MASKSTID2",
        "MASKSTID2",
        "MASKSTID",
        "MASKSTID2",
        "MASKSTID2",
    ]
    pd.testing.assert_frame_equal(cdm_header.data, result.data)
