from __future__ import annotations

import ast
import os

import pandas as pd

from cdm_reader_mapper import cdm_mapper, mdf_reader
from cdm_reader_mapper.cdm_mapper import read_tables
from cdm_reader_mapper.common.pandas_TextParser_hdlr import make_copy
from cdm_reader_mapper.metmetpy import (
    correct_datetime,
    correct_pt,
    validate_datetime,
    validate_id,
)
from cdm_reader_mapper.operations.inspect import get_length

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

    data = read_.data
    mask = read_.mask
    dtypes = read_.dtypes
    parse_dates = read_.parse_dates
    columns = read_.columns

    data = correct_datetime.correct(
        data=data,
        imodel=imodel,
    )

    data = correct_pt.correct(
        data=data,
        imodel=imodel,
    )

    if not isinstance(data, pd.DataFrame):
        data_pd = make_copy(data).read()
    else:
        data_pd = data.copy()
    if not isinstance(mask, pd.DataFrame):
        mask_pd = make_copy(mask).read()
    else:
        mask_pd = mask.copy()

    val_dt = validate_datetime.validate(
        data=data_pd,
        imodel=imodel,
    )

    val_id = validate_id.validate(
        data=data_pd,
        imodel=imodel,
    )

    expected_data = getattr(result_data, exp)
    result_data_file = expected_data["data"]
    if not os.path.isfile(result_data_file):
        return

    data_ = _read_result_data(
        result_data_file,
        columns,
        dtype=dtypes,
        parse_dates=parse_dates,
    )
    mask_ = _read_result_data(expected_data["mask"], columns)

    data_ = drop_rows(data_, drops)
    mask_ = drop_rows(mask_, drops)

    pd.testing.assert_frame_equal(data_pd, data_)
    pd.testing.assert_frame_equal(mask_pd, mask_, check_dtype=False)

    if isinstance(data, pd.DataFrame):
        if data.empty:
            return
    else:
        if get_length(data) == 0:
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

    output = cdm_mapper.map_model(
        data=data,
        imodel=imodel,
        cdm_subset=cdm_subset,
        codes_subset=codes_subset,
        log_level="DEBUG",
    )

    col_subset = get_col_subset(output, codes_subset)

    cdm_mapper.cdm_to_ascii(output, suffix=imodel)
    output = read_tables(".", tb_id=imodel, cdm_subset=cdm_subset)

    output_ = read_tables(
        expected_data["cdm_table"], tb_id=f"{imodel}*", cdm_subset=cdm_subset
    )

    output, output_ = remove_datetime_columns(output, output_, col_subset)

    output_ = drop_rows(output_, drops)
    pd.testing.assert_frame_equal(output, output_)
