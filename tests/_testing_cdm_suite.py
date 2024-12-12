from __future__ import annotations

import ast
import os

import pandas as pd

from cdm_reader_mapper import mdf_reader
from cdm_reader_mapper.cdm_mapper import read_tables

from ._results import result_data


def _pandas_read_csv(
    *args,
    delimiter=mdf_reader.properties.internal_delimiter,
    squeeze=False,
    name=False,
    **kwargs,
):
    df = pd.read_csv(
        *args,
        **kwargs,
        quotechar="\0",
        escapechar="\0",
        delimiter=delimiter,
    )
    if squeeze is True:
        df = df.squeeze()

    if name is not False:
        df.name = name

    return df


def _evaluate_columns(columns):
    columns_ = []
    for col in columns:
        try:
            columns_.append(ast.literal_eval(col))
        except ValueError:
            columns_.append(col)
    return columns_


def _read_result_data(data_file, columns, **kwargs):
    columns_ = _pandas_read_csv(data_file, nrows=0).columns
    columns_ = _evaluate_columns(columns_)
    data_ = _pandas_read_csv(
        data_file,
        names=columns_,
        skiprows=1,
        **kwargs,
    )
    return data_[columns]


def drop_rows(df, drops):
    if drops == "all":
        return df.drop(df.index)
    elif drops:
        return df.drop(drops).reset_index(drop=True)
    return df


def get_col_subset(output, codes_subset):
    col_subset = []
    if codes_subset is not None:
        for key in output.keys():
            for att in output[key]["atts"].keys():
                if att in codes_subset:
                    col_subset.append((key, att))
    return col_subset


def remove_datetime_columns(output, output_, col_subset):
    del output[("header", "record_timestamp")]
    del output[("header", "history")]
    del output_[("header", "record_timestamp")]
    del output_[("header", "history")]

    if len(col_subset) > 0:
        output = output[col_subset]
        output_ = output_[col_subset]
    return output, output_


def _testing_suite(
    source=None,
    imodel=None,
    cdm_subset=None,
    codes_subset=None,
    suffix="exp",
    mapping=True,
    out_path=None,
    drops=None,
    **kwargs,
):
    exp = f"expected_{imodel}"

    read_ = mdf_reader.read(
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

    data_exp = _read_result_data(
        result_data_file,
        read_.columns,
        dtype=read_.dtypes,
        parse_dates=read_.parse_dates,
    )
    mask_exp = _read_result_data(expected_data["mask"], read_.columns)

    data_exp = drop_rows(data_exp, drops)
    mask_exp = drop_rows(mask_exp, drops)

    pd.testing.assert_frame_equal(read_.data, data_exp)
    pd.testing.assert_frame_equal(read_.mask, mask_exp, check_dtype=False)

    if len(read_) == 0:
        return

    if val_dt is not None:
        val_dt_ = _pandas_read_csv(
            expected_data["vadt"],
            header=None,
            squeeze=True,
            name=None,
        )
        val_dt_ = drop_rows(val_dt_, drops)
        pd.testing.assert_series_equal(val_dt, val_dt_, check_dtype=False)

    if val_id is not None:
        val_id_ = _pandas_read_csv(
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

    col_subset = get_col_subset(read_.cdm, codes_subset)

    read_.write_tables(suffix=imodel)
    output = read_tables(".", tb_id=imodel, cdm_subset=cdm_subset).cdm

    output_exp = read_tables(
        expected_data["cdm_table"], tb_id=f"{imodel}*", cdm_subset=cdm_subset
    ).cdm

    output, output_exp = remove_datetime_columns(output, output_exp, col_subset)

    output_exp = drop_rows(output_exp, drops)
    pd.testing.assert_frame_equal(output, output_exp)
