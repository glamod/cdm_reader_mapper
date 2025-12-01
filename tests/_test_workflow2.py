from __future__ import annotations

import os

import pytest  # noqa

from cdm_reader_mapper import test_data

from ._testing_workflow_suite import _testing_suite


@pytest.mark.parametrize(
    "imodel, cdm_subset, codes_subset, mapping, drops, mdf_kwargs",
    [
        ("icoads_r300_d701", None, None, True, None, {}),  # p
    ],
)
def test_read_data(
    imodel,
    cdm_subset,
    codes_subset,
    mapping,
    drops,
    mdf_kwargs,
):
    _testing_suite(
        # **dict(getattr(test_data, f"test_{imodel}")),
        test_data[f"test_{imodel}"]["source"],
        imodel=imodel,
        cdm_subset=cdm_subset,
        codes_subset=codes_subset,
        mapping=mapping,
        drops=drops,
        **mdf_kwargs,
    )
