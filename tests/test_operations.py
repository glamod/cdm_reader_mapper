from __future__ import annotations

import pandas as pd
import pytest  # noqa

from cdm_reader_mapper.operations import corrections, inspect, replace, select

from ._data import attrs_df, attrs_pa, data_df, data_pa, mask_df, mask_pa
from ._results import correction_df, table_df


def test_select_true_pandas():
    mask_df["global_mask"] = mask_df.all(axis=1)
    select.select_true(data_df, mask_df, "global_mask", out_rejected=True)


def test_select_true_parser():
    select.select_true(data_pa, mask_pa, "c99", out_rejected=True)


def test_select_from_list_pandas():
    selection = {("c1", "PT"): ["7", "6"]}
    select.select_from_list(data_df, selection, out_rejected=True, in_index=True)


def test_select_from_list_parser():
    selection = {("c1", "PT"): ["7", "6"]}
    select.select_from_list(data_pa, selection, out_rejected=True, in_index=True)


def test_select_from_index_pandas():
    select.select_from_index(data_df, [0, 1, 2, 3, 4, 5])


def test_select_from_index_parser():
    select.select_from_index(data_pa, [0, 1, 2, 3, 4, 5])


def test_inspect_get_length_pandas():
    inspect.get_length(data_df)


def test_inspect_get_length_parser():
    inspect.get_length(data_pa)


def test_inspect_count_by_ca_pandas():
    inspect.count_by_cat(data_df, ("c1", "PT"))


def test_inspect_count_by_cat_parser():
    inspect.count_by_cat(data_pa, ("c1", "PT"))


def test_corrections_pandas():
    corrections.corrections(
        data_df, dataset="test_data", correction_path=".", yr="2010", mo="07"
    )


def test_corrections_parser():
    corrections.corrections(
        data_pa, dataset="test_data", correction_path=".", yr="2010", mo="07"
    )


def test_replace():
    replace.replace_columns(
        table_df,
        correction_df,
        pivot_c="report_id",
        rep_c=["primary_station_id", "primary_station_id.isChange"],
    )
