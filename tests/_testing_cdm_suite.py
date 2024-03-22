from __future__ import annotations

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


def _read_result_data(data_file, columns, **kwargs):
    columns_ = _pandas_read_csv(data_file, nrows=0).columns
    columns_ = [eval(col) for col in columns_]
    data_ = _pandas_read_csv(
        data_file,
        names=columns_,
        skiprows=1,
        **kwargs,
    )
    return data_[columns]


def _testing_suite(
    source=None,
    data_model=None,
    dm=None,
    ds=None,
    deck=None,
    cdm_name=None,
    cdm_subset=None,
    codes_subset=None,
    suffix="exp",
    mapping=True,
    review=True,
    out_path=None,
    **kwargs,
):
    exp = "expected_" + suffix
    splitted = suffix.split("_")
    tb_id = splitted[0] + "-" + "_".join(splitted[1:])

    read_ = mdf_reader.read(
        source=source,
        data_model=data_model,
        out_path=out_path,
        **kwargs,
    )
    data = read_.data
    attrs = read_.attrs
    mask = read_.mask
    dtypes = read_.dtypes
    parse_dates = read_.parse_dates
    columns = read_.columns

    data = correct_datetime.correct(
        data=data,
        data_model=dm,
        deck=deck,
    )

    data = correct_pt.correct(
        data,
        dataset=ds,
        data_model=dm,
        deck=deck,
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
        data_model=dm,
        dck=deck,
    )

    val_id = validate_id.validate(
        data=data_pd,
        dataset=ds,
        data_model=dm,
        dck=deck,
    )

    if review is True:
        expected_data = result_data[exp]
        result_data_file = expected_data["data"]
        if not os.path.isfile(result_data_file):
            return

        data_ = _read_result_data(
            result_data_file, columns, dtype=dtypes, parse_dates=parse_dates
        )
        mask_ = _read_result_data(expected_data["mask"], columns)

        pd.testing.assert_frame_equal(data_pd, data_)
        pd.testing.assert_frame_equal(mask_pd, mask_, check_dtype=False)

        if val_dt is not None:
            val_dt_ = _pandas_read_csv(
                expected_data["vadt"],
                header=None,
                squeeze=True,
                name=None,
            )
            pd.testing.assert_series_equal(val_dt, val_dt_, check_dtype=False)

        if val_id is not None:
            val_id_ = _pandas_read_csv(
                expected_data["vaid"],
                header=None,
                squeeze=True,
                name=val_id.name,
            )
            pd.testing.assert_series_equal(val_id, val_id_, check_dtype=False)

    if mapping is False:
        return

    output = cdm_mapper.map_model(
        cdm_name,
        data,
        attrs,
        cdm_subset=cdm_subset,
        codes_subset=codes_subset,
        log_level="DEBUG",
    )

    col_subset = []
    if codes_subset is not None:
        for key in output.keys():
            for att in output[key]["atts"].keys():
                if att in codes_subset:
                    col_subset.append((key, att))

    cdm_mapper.cdm_to_ascii(output, suffix=tb_id)
    output = read_tables(".", tb_id=tb_id, cdm_subset=cdm_subset)

    if review is True:
        output_ = read_tables(
            expected_data["cdm_table"], tb_id=tb_id + "*", cdm_subset=cdm_subset
        )

        del output[("header", "record_timestamp")]
        del output[("header", "history")]
        del output_[("header", "record_timestamp")]
        del output_[("header", "history")]

        if len(col_subset) > 0:
            output = output[col_subset]
            output_ = output_[col_subset]

        pd.testing.assert_frame_equal(output, output_)
