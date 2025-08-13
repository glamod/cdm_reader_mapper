from __future__ import annotations

import os

import pandas as pd

from cdm_reader_mapper import read

from ._results import result_data
from ._utilities import (
    drop_rows,
    get_col_subset,
    read_validation,
    remove_datetime_columns,
)


def _testing_suite(
    source=None,
    imodel=None,
    cdm_subset=None,
    codes_subset=None,
    mapping=True,
    drops=None,
    **kwargs,
):
    out_dir = ".pytest_cache"
    os.makedirs(out_dir, exist_ok=True)

    exp = f"expected_{imodel}"

    db_mdf = read(
        source,
        imodel=imodel,
        mode="mdf",
        **kwargs,
    )

    db_mdf.correct_datetime(inplace=True)
    db_mdf.correct_pt(inplace=True)

    val_dt = db_mdf.validate_datetime()

    val_id = db_mdf.validate_id()

    db_mdf.write(suffix=imodel, out_dir=out_dir)

    db_res = read(
        os.path.join(out_dir, f"data-{imodel}.csv"),
        mask=os.path.join(out_dir, f"mask-{imodel}.csv"),
        info=os.path.join(out_dir, f"info-{imodel}.json"),
        mode="data",
        **kwargs,
    )

    data_res = db_res.data.copy()
    mask_res = db_res.mask.copy()

    expected_data = getattr(result_data, exp)
    result_data_file = expected_data["data"]

    if not os.path.isfile(result_data_file):
        return

    db_exp = read(
        result_data_file,
        mask=expected_data["mask"],
        info=expected_data["info"],
        col_subset=data_res.columns,
        mode="data",
        **kwargs,
    )
    data_exp = db_exp.data.copy()
    mask_exp = db_exp.mask.copy()

    data_exp = drop_rows(data_exp, drops)
    mask_exp = drop_rows(mask_exp, drops)

    if data_res.empty and data_exp.empty:
        return

    pd.testing.assert_frame_equal(data_res, data_exp)
    pd.testing.assert_frame_equal(mask_res, mask_exp, check_dtype=False)

    if val_dt is not None and os.path.isfile(expected_data["vadt"]):
        val_dt_ = read_validation(
            expected_data["vadt"],
            name=None,
        )
        val_dt_ = drop_rows(val_dt_, drops)
        pd.testing.assert_series_equal(val_dt, val_dt_, check_dtype=False)

    if val_id is not None and os.path.isfile(expected_data["vaid"]):
        val_id_ = read_validation(
            expected_data["vaid"],
            name=val_id.name,
        )
        val_id_ = drop_rows(val_id_, drops)
        pd.testing.assert_series_equal(val_id, val_id_, check_dtype=False)

    if mapping is False:
        return

    db_mdf.map_model(
        cdm_subset=cdm_subset,
        codes_subset=codes_subset,
        log_level="DEBUG",
        inplace=True,
    )

    col_subset = get_col_subset(db_mdf.data, codes_subset)

    db_mdf.write(suffix=imodel, out_dir=out_dir)
    output = read(out_dir, suffix=imodel, cdm_subset=cdm_subset, mode="tables")

    output_exp = read(
        expected_data["cdm_table"],
        suffix=f"{imodel}*",
        cdm_subset=cdm_subset,
        mode="tables",
    )

    output = output.data
    output_exp = output_exp.data
    output, output_exp = remove_datetime_columns(output, output_exp, col_subset)
    output_exp = drop_rows(output_exp, drops)
    pd.testing.assert_frame_equal(output, output_exp)
