from __future__ import annotations

import json

import pandas as pd
import pytest  # noqa

from pandas.testing import assert_frame_equal

from cdm_reader_mapper.mdf_reader.writer import (
    write_data,
)


def test_write_data_basic(tmp_path):
    data = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["1", "2", "3"],
        }
    )
    mask = pd.DataFrame(
        {
            "A": [True, True, False],
            "B": [False, True, True],
        }
    )
    info = {
        "dtypes": {"A": "int", "B": "str"},
        "parse_dates": [],
        "encoding": "utf-8",
    }

    write_data(
        data,
        mask=mask,
        out_dir=tmp_path,
        prefix="test_write",
        suffix="basic",
        **info,
    )

    data_file = tmp_path / "test_write-data-basic.csv"
    mask_file = tmp_path / "test_write-mask-basic.csv"
    info_file = tmp_path / "test_write-info-basic.json"

    assert data_file.is_file()
    assert mask_file.is_file()
    assert info_file.is_file()

    with open(info_file) as read_file:
        info_res = json.load(read_file)

    assert info_res == info

    data_res = pd.read_csv(data_file, dtype=info["dtypes"])
    assert_frame_equal(data, data_res)

    mask_res = pd.read_csv(mask_file, dtype="bool")
    assert_frame_equal(mask, mask_res)


def test_write_data_col_subset(tmp_path):
    data = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["1", "2", "3"],
        }
    )
    mask = pd.DataFrame(
        {
            "A": [True, True, False],
            "B": [False, True, True],
        }
    )
    info = {
        "dtypes": {"A": "int"},
        "parse_dates": [],
        "encoding": "utf-8",
    }
    subset = "A"

    write_data(
        data,
        mask=mask,
        out_dir=tmp_path,
        prefix="test_write",
        suffix="subset",
        col_subset=subset,
        **info,
    )

    data_file = tmp_path / "test_write-data-subset.csv"
    mask_file = tmp_path / "test_write-mask-subset.csv"
    info_file = tmp_path / "test_write-info-subset.json"

    assert data_file.is_file()
    assert mask_file.is_file()
    assert info_file.is_file()

    with open(info_file) as read_file:
        info_res = json.load(read_file)

    assert info_res == info

    data_res = pd.read_csv(data_file, dtype=info["dtypes"])
    assert_frame_equal(data[[subset]], data_res)

    mask_res = pd.read_csv(mask_file, dtype="bool")
    assert_frame_equal(mask[[subset]], mask_res)
