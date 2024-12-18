from __future__ import annotations

import pytest

from cdm_reader_mapper import CDM, read_mdf, read_tables, test_data

from ._results import result_data
from ._utilities import read_result_data


def get_result_data(imodel):
    test_ = read_mdf(**dict(getattr(test_data, f"test_{imodel}")), imodel=imodel)
    columns = test_.data.columns
    results_ = getattr(result_data, f"expected_{imodel}")
    data_ = read_result_data(results_["data"], columns)
    mask_ = read_result_data(results_["mask"], columns)
    cdm_ = read_tables(results_["cdm_table"], suffix=f"{imodel}*")
    return CDM(data=data_, mask=mask_, cdm_tables=cdm_)


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
    [data["data_700"], data["data_703"], [data["data_703"], data["data_201"]]],
)
def test_append(test_data):
    data["data_700"].append(test_data)


@pytest.mark.parametrize(
    "test_data",
    [data["data_700"], data["data_703"], [data["data_703"], data["data_201"]]],
)
def test_merge(test_data):
    test_data.cdm.columns = [("test", c) for c in test_data.cdm.columns]
    data["data_700"].append(test_data, datasets=["cdm"])
