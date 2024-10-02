from __future__ import annotations

import pandas as pd
import pytest  # noqa

from cdm_reader_mapper.operations import inspect, replace, select

from ._data import data_df, data_pa, mask_df, mask_pa
from ._results import correction_df, table_df


def test_select_true_pandas():
    index = "c99_data"
    data_df_ = data_df[index]
    mask_df_ = mask_df[index]
    exp1 = data_df_[:4].reset_index(drop=True)
    exp2 = data_df_[4:].reset_index(drop=True)
    result = select.select_true(data_df_, mask_df_, out_rejected=True)
    pd.testing.assert_frame_equal(exp1, result[0])
    pd.testing.assert_frame_equal(exp2, result[1])


def test_select_from_list_pandas():
    selection = {("c1", "B1"): [26, 41]}
    idx1 = data_df.index.isin([1, 3])
    idx2 = data_df.index.isin([0, 2, 4])
    exp1 = data_df[idx1].reset_index(drop=True)
    exp2 = data_df[idx2].reset_index(drop=True)
    result = select.select_from_list(
        data_df, selection, out_rejected=True, in_index=True
    )
    pd.testing.assert_frame_equal(exp1, result[0])
    pd.testing.assert_frame_equal(exp2, result[1])


def test_inspect_get_length_pandas():
    result = inspect.get_length(data_df)
    assert result == 5


def test_inspect_count_by_ca_pandas():
    result = inspect.count_by_cat(data_df, ("c1", "B1"))
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


def test_select_true_parser():
    select.select_true(data_pa, mask_pa, out_rejected=True)


def test_select_from_list_parser():
    selection = {("c1", "PT"): ["7", "6"]}
    select.select_from_list(data_pa, selection, out_rejected=True, in_index=True)


def test_select_from_index_parser():
    select.select_from_index(data_pa, [0, 1, 2, 3, 4, 5])


def test_inspect_get_length_parser():
    inspect.get_length(data_pa)


def test_inspect_count_by_cat_parser():
    inspect.count_by_cat(data_pa, ("c1", "PT"))
