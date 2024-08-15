"""Common Data Model (CDM) pandas duplicate check."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import recordlinkage as rl


def set_comparer(compare_dict):
    """DOCUMENTATION."""
    comparer = rl.Compare()
    for column, c_dict in compare_dict.items():
        method = c_dict["method"]
        kwargs = c_dict["kwargs"]
        getattr(comparer, method)(column, column, label=f"-{column}-", **kwargs)
    return comparer


def dataframe_apply_check(df, method, method_kwargs, compare_kwargs):
    """DOCUMENTATION."""
    indexer = getattr(rl.index, method)(**method_kwargs)
    pairs = indexer.index(df)
    comparer = set_comparer(compare_kwargs)
    compared = comparer.compute(pairs, df)
    return compared


def duplicate_check(
    data,
    method="SortedNeighbourhood",
    method_kwargs={},
    compare_kwargs={},
):
    """DOCUMENTATION."""

    def dataframe(df, method, method_kwargs, compare_kwargs):
        return dataframe_apply_check(
            df,
            method,
        )

    def parser(data_parser, method, method_kwargs, compare_kwargs):
        read_params = [
            "chunksize",
            "names",
            "dtype",
            "parse_dates",
            "date_parser",
            "infer_datetime_format",
        ]
        read_dict = {x: data_parser.orig_options.get(x) for x in read_params}
        in_buffer = StringIO()

        for df in data_parser:
            o = dataframe(df, method, method_kwargs, compare_kwargs)
            o.to_csv(in_buffer, header=False, index=False, mode="a")

        in_buffer.seek(0)
        output = [pd.read_csv(in_buffer, **read_dict)]
        return output

    if not isinstance(data, pd.io.parsers.TextFileReader):
        output = dataframe(data, method, method_kwargs, compare_kwargs)
    else:
        output = parser(data, method, method_kwargs, compare_kwargs)

    if len(output) > 1:
        return output
    else:
        return output[0]
