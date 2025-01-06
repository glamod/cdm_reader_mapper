from __future__ import annotations

import pandas as pd
import pytest

from cdm_reader_mapper import DataBundle, read_mdf, read_tables, test_data

from ._results import result_data
from ._utilities import read_result_data


def get_result_data(imodel):
    test_ = read_mdf(**dict(getattr(test_data, f"test_{imodel}")), imodel=imodel)
    columns = test_.data.columns
    results_ = getattr(result_data, f"expected_{imodel}")
    data_ = read_result_data(results_["data"], columns)
    mask_ = read_result_data(results_["mask"], columns)
    db = read_tables(results_["cdm_table"], suffix=f"{imodel}*")
    return db.add({"data": data_, "mask": mask_})


def update_columns(list_of):
    if not isinstance(list_of, list):
        list_of = [list_of]
    i = 0
    updated = []
    while i < len(list_of):
        columns = [(f"test{i}_{c[0]}", c[1]) for c in list_of[i].tables.columns]
        updated_tables = list_of[i].copy()
        updated_tables.tables.columns = pd.MultiIndex.from_tuples(columns)
        updated.append(DataBundle(tables=updated_tables.tables))
        i += 1
    return updated


data = {
    "data_700": get_result_data("icoads_r300_d700"),
    "data_703": get_result_data("icoads_r300_d703"),
    "data_201": get_result_data("icoads_r300_d201"),
}


@pytest.mark.parametrize(
    "test_data", [data["data_700"], data["data_703"], data["data_201"]]
)
def test_len(test_data):
    assert len(test_data) == 5


@pytest.mark.parametrize(
    "test_data",
    [[data["data_703"]], [data["data_703"], data["data_201"]]],
)
def test_stack_v(test_data):
    orig_data = data["data_700"].copy()
    test_data = [data.copy() for data in test_data]
    orig_data.stack_v(test_data)


@pytest.mark.parametrize(
    "test_data",
    [data["data_703"], [data["data_703"], data["data_201"]]],
)
def test_stack_h(test_data):
    orig_data = data["data_700"].copy()
    test_data = update_columns(test_data)
    orig_data.stack_h(test_data, datasets=["tables"])
