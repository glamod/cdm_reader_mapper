from __future__ import annotations

import os

import pandas as pd

from cdm_reader_mapper import read_mdf, read_tables
from cdm_reader_mapper.common.pandas_TextParser_hdlr import make_copy

from ._results import result_data
from ._utilities import (
    drop_rows,
    get_col_subset,
    pandas_read_csv,
    read_result_data,
    remove_datetime_columns,
)


def _testing_suite(
    source=None,
    imodel=None,
    cdm_subset=None,
    codes_subset=None,
    mapping=True,
    out_path=None,
    drops=None,
    **kwargs,
):
    exp = f"expected_{imodel}"

    read_ = read_mdf(
        source=source,
        imodel=imodel,
        out_path=out_path,
        **kwargs,
    )

    read_.correct_datetime()
    read_.correct_pt()

    val_dt = read_.validate_datetime()

    val_id = read_.validate_id()

    expected_data = getattr(result_data, exp)
    result_data_file = expected_data["data"]
    if not os.path.isfile(result_data_file):
        return

    data_exp = read_result_data(
        result_data_file,
        read_.columns,
        dtype=read_.dtypes,
        parse_dates=read_._parse_dates,
    )
    mask_exp = read_result_data(expected_data["mask"], read_.columns)

    data_exp = drop_rows(data_exp, drops)
    mask_exp = drop_rows(mask_exp, drops)

    if isinstance(read_.data, pd.io.parsers.TextFileReader):
        data = make_copy(read_.data).read()
        mask = make_copy(read_.mask).read()
    else:
        data = read_.data.copy()
        mask = read_.mask.copy()

    pd.testing.assert_frame_equal(data, data_exp)
    pd.testing.assert_frame_equal(mask, mask_exp, check_dtype=False)

    if len(read_) == 0:
        return

    if val_dt is not None:
        val_dt_ = pandas_read_csv(
            expected_data["vadt"],
            header=None,
            squeeze=True,
            name=None,
        )
        val_dt_ = drop_rows(val_dt_, drops)
        pd.testing.assert_series_equal(val_dt, val_dt_, check_dtype=False)

    if val_id is not None:
        val_id_ = pandas_read_csv(
            expected_data["vaid"],
            header=None,
            squeeze=True,
            name=val_id.name,
        )
        val_id_ = drop_rows(val_id_, drops)
        pd.testing.assert_series_equal(val_id, val_id_, check_dtype=False)

    if mapping is False:
        return

    read_.map_model(
        cdm_subset=cdm_subset,
        codes_subset=codes_subset,
        log_level="DEBUG",
    )

    col_subset = get_col_subset(read_.tables, codes_subset)

    read_.write_tables(suffix=imodel)
    output = read_tables(".", suffix=imodel, cdm_subset=cdm_subset)

    output_exp = read_tables(
        expected_data["cdm_table"], suffix=f"{imodel}*", cdm_subset=cdm_subset
    )

    output = output.tables
    output_exp = output_exp.tables
    output, output_exp = remove_datetime_columns(output, output_exp, col_subset)
    output_exp = drop_rows(output_exp, drops)
    pd.testing.assert_frame_equal(output, output_exp)
