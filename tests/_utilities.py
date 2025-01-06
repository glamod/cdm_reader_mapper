from __future__ import annotations

import ast

import pandas as pd

from cdm_reader_mapper import mdf_reader


def drop_rows(df, drops):
    if drops == "all":
        return df.drop(df.index)
    elif drops:
        return df.drop(drops).reset_index(drop=True)
    return df


def get_col_subset(output, codes_subset):
    col_subset = []
    if codes_subset is None:
        return col_subset
    for subset in codes_subset:
        for column in output.columns:
            if subset in column:
                col_subset.append(column)
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


def pandas_read_csv(
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


def evaluate_columns(columns):
    columns_ = []
    for col in columns:
        try:
            columns_.append(ast.literal_eval(col))
        except ValueError:
            columns_.append(col)
    return columns_


def read_result_data(data_file, columns, **kwargs):
    columns_ = pandas_read_csv(data_file, nrows=0).columns
    columns_ = evaluate_columns(columns_)
    data_ = pandas_read_csv(
        data_file,
        names=columns_,
        skiprows=1,
        **kwargs,
    )
    return data_[columns]
