from __future__ import annotations

import pandas as pd
import pytest

from cdm_reader_mapper import DataBundle, read

from ._results import result_data


def get_result_data(imodel):
    results_ = getattr(result_data, f"expected_{imodel}")
    db_ = read(
        results_["data"],
        mask=results_["mask"],
        info=results_["info"],
        mode="data",
    )
    data_ = db_.data.copy()
    mask_ = db_.mask.copy()
    db = read(results_["cdm_table"], suffix=f"{imodel}*", mode="tables")
    return db.add({"data": data_, "mask": mask_})


def update_columns(list_of):
    if not isinstance(list_of, list):
        list_of = [list_of]
    i = 0
    updated = []
    while i < len(list_of):
        columns = [(f"test{i}_{c[0]}", c[1]) for c in list_of[i].data.columns]
        updated_tables = list_of[i].copy()
        updated_tables.data.columns = pd.MultiIndex.from_tuples(columns)
        updated.append(DataBundle(data=updated_tables.data))
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
    orig_data.stack_h(test_data, datasets=["data"])


def test_print():
    print(data["data_703"])
