from __future__ import annotations

import numpy as np
import pandas as pd
import pytest  # noqa

from cdm_reader_mapper import read_data

from ._results import result_data


def get_result_data(imodel):
    results_ = getattr(result_data, f"expected_{imodel}")
    return read_data(
        results_["data"],
        mask=results_["mask"],
        info=results_["info"],
    )


test_data = get_result_data("icoads_r300_d700")


def test_index():
    index_db = test_data.index
    index_pd = test_data.data.index
    np.testing.assert_equal(list(index_db), list(index_pd))


def test_size():
    size_db = test_data.size
    size_pd = test_data.data.size
    np.testing.assert_equal(size_db, size_pd)


def test_dropna():
    dropna_db = test_data.dropna(how="any")
    dropna_pd = test_data.data.dropna(how="any")
    pd.testing.assert_frame_equal(dropna_db, dropna_pd)


def test_rename():
    _renames = {("core", "MO"): ("core", "MONTH")}
    rename_db = test_data.rename(columns=_renames)
    rename_pd = test_data.data.rename(columns=_renames)
    pd.testing.assert_frame_equal(rename_db, rename_pd)


def test_inplace():
    _renames = {("core", "MO"): ("core", "MONTH")}
    db1 = test_data.copy()
    db1.rename(columns=_renames, inplace=True)
    db2 = test_data.copy()
    db2.data.rename(columns=_renames, inplace=True)
    pd.testing.assert_frame_equal(db1.data, db2.data)
