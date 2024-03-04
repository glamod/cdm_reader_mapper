from __future__ import annotations

import os

import pandas as pd

from cdm_reader_mapper import cdm_mapper, mdf_reader
from cdm_reader_mapper.cdm_mapper import read_tables

from ._results import result_data


def _pandas_read_csv(
    *args, delimiter=mdf_reader.properties.internal_delimiter, **kwargs
):
    return pd.read_csv(
        *args,
        **kwargs,
        quotechar="\0",
        escapechar="\0",
        delimiter=delimiter,
    )


def _testing_suite(
    source=None,
    data_model=None,
    sections=None,
    cdm_name=None,
    cdm_subset=None,
    codes_subset=None,
    suffix="exp",
    mapping=True,
    out_path=None,
    **kwargs,
):
    name_ = source.split("/")[-1].split(".")[0]
    exp = "expected_" + suffix
    if sections:
        if isinstance(sections, str):
            sections = [sections]
        name_ = name_ + "_" + "_".join(sections)
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

    if not isinstance(data, pd.DataFrame):
        data = data.read()
    if not isinstance(mask, pd.DataFrame):
        mask = mask.read()

    result_data_file = result_data[exp]["data"]
    if not os.path.isfile(result_data_file):
        return

    data_ = _pandas_read_csv(
        result_data_file,
        names=data.columns,
        dtype=dtypes,
        parse_dates=parse_dates,
    )

    mask_ = _pandas_read_csv(
        result_data[exp]["mask"],
        names=data.columns,
    )

    pd.testing.assert_frame_equal(data, data_, check_dtype=False)
    pd.testing.assert_frame_equal(mask, mask_, check_dtype=False)

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

    cdm_mapper.cdm_to_ascii(output, suffix=suffix)
    output = read_tables(".", tb_id=suffix, cdm_subset=cdm_subset)
    output_ = read_tables(result_data[exp]["cdm_table"], cdm_subset=cdm_subset)

    del output[("header", "record_timestamp")]
    del output[("header", "history")]
    del output_[("header", "record_timestamp")]
    del output_[("header", "history")]

    if len(col_subset) > 0:
        output = output[col_subset]
        output_ = output_[col_subset]

    pd.testing.assert_frame_equal(output, output_)
