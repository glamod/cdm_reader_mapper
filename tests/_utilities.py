from __future__ import annotations

import pandas as pd


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


def read_validation(
    *args,
    name=False,
    **kwargs,
):
    df = pd.read_csv(
        *args,
        header=None,
        **kwargs,
    )

    df = df.squeeze()

    if name is not False:
        df.name = name

    return df
