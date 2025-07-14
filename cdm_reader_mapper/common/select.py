# noqa: D100
"""
Common Data Model (CDM) pandas selection operators.

Created on Wed Jul  3 09:48:18 2019

@author: iregon
"""
from __future__ import annotations

from io import StringIO

import pandas as pd


def _dataframe_apply_index(
    df,
    index_list,
    reset_index=False,
    inverse=False,
) -> pd.DataFrame | pd.Series:
    """Apply index to pandas DataFrame."""
    if isinstance(index_list, str):
        index_list = [index_list]
    if isinstance(index_list, list):
        index_list = pd.Index(index_list)
    index = df.index.isin(index_list)
    if inverse is True:
        in_df = df[~index]
    else:
        in_df = df[index]

    if reset_index is True:
        in_df = in_df.reset_index(drop=True)

    in_df.__dict__["_prev_index"] = index_list
    return in_df


def _split_dataframe_by_index(
    df,
    index,
    reset_index=False,
    inverse=False,
    return_rejected=False,
) -> tuple[pd.DataFrame]:
    """Common pandas DataFrame selection function."""
    out1 = _dataframe_apply_index(
        df,
        index,
        reset_index=reset_index,
        inverse=inverse,
    )
    if return_rejected is True:
        index2 = [idx for idx in df.index if idx not in index]
        out2 = _dataframe_apply_index(
            df,
            index2,
            reset_index=reset_index,
            inverse=inverse,
        )
    else:
        out2 = pd.DataFrame(columns=out1.columns)
        out2.__dict__["_prev_index"] = pd.Index([])

    return out1, out2


def split_dataframe_by_boolean(df, mask, boolean, **kwargs) -> tuple[pd.DataFrame]:
    """DOCUMENTATION."""
    # get the index values and pass to the general function
    # If a mask is empty, assume True (...)
    if boolean is True:
        global_mask = mask.all(axis=1)
    else:
        global_mask = ~(mask.any(axis=1))
    index = global_mask[global_mask.fillna(boolean)].index
    return _split_dataframe_by_index(
        df,
        index,
        **kwargs,
    )


def split_dataframe_by_column_entries(df, col, values, **kwargs) -> tuple[pd.DataFrame]:
    """DOCUMENTATION."""
    # get the index values and pass to the general function
    in_df = df.loc[df[col].isin(values)]
    index = list(in_df.index)
    return _split_dataframe_by_index(
        df,
        index,
        **kwargs,
    )


def split_dataframe_by_index(df, index, **kwargs) -> tuple[pd.DataFrame]:
    """DOCUMENTATION."""
    return _split_dataframe_by_index(
        df,
        index,
        **kwargs,
    )


def split_parser(
    data, *args, func=None, reset_index=False, inverse=False, return_rejected=False
) -> tuple[pd.io.parsers.TextFileReader]:
    """Common pandas TextFileReader selection function."""
    read_params = [
        "chunksize",
        "names",
        "dtype",
        "parse_dates",
        "date_parser",
        "infer_datetime_format",
    ]
    write_dict = {"header": None, "mode": "a", "index": not reset_index}
    read_dict = {x: data[0].orig_options.get(x) for x in read_params}
    buffer1 = StringIO()
    buffer2 = StringIO()
    _prev_index1 = None
    _prev_index2 = None
    for zipped in zip(*data):
        if not isinstance(zipped, tuple):
            zipped = tuple(zipped)
        out1, out2 = func(
            *zipped,
            *args,
            reset_index=reset_index,
            inverse=inverse,
            return_rejected=return_rejected,
        )
        if _prev_index1 is None:
            _prev_index1 = out1.__dict__["_prev_index"]
        else:
            _prev_index1 = _prev_index1.union(out1.__dict__["_prev_index"])
        if _prev_index2 is None:
            _prev_index2 = out2.__dict__["_prev_index"]
        else:
            _prev_index2 = _prev_index2.union(out2.__dict__["_prev_index"])
        out1.to_csv(buffer1, **write_dict)
        if return_rejected is True:
            out2.to_csv(buffer2, **write_dict)
    dtypes = {}
    for k, v in out1.dtypes.items():
        if v == "object":
            v = "str"
        dtypes[k] = v
    read_dict["dtype"] = dtypes
    buffer1.seek(0)
    buffer2.seek(0)
    TextParser1 = pd.read_csv(buffer1, **read_dict)
    TextParser1.__dict__["_prev_index"] = _prev_index1
    TextParser2 = pd.read_csv(buffer2, **read_dict)
    TextParser2.__dict__["_prev_index"] = _prev_index2
    return TextParser1, TextParser2


def split(
    data, func, *args, **kwargs
) -> tuple[pd.DataFrame | pd.io.parsers.TextfileReader]:
    """DOCUMENTATION."""
    if not isinstance(data, list):
        data = [data]
    if isinstance(data[0], pd.io.parsers.TextFileReader):
        return split_parser(
            data,
            *args,
            func=func,
            **kwargs,
        )
    return func(*data, *args, **kwargs)


def split_by_boolean(
    data, mask, boolean, reset_index=False, inverse=False, return_rejected=False
) -> tuple[pd.DataFrame | pd.io.parsers.TextfileReader]:
    """DOCUMENTATION."""
    func = split_dataframe_by_boolean
    return split(
        [data, mask],
        func,
        boolean,
        reset_index=reset_index,
        inverse=inverse,
        return_rejected=return_rejected,
    )


def split_by_boolean_true(
    data, mask, reset_index=False, inverse=False, return_rejected=False
) -> tuple[pd.DataFrame | pd.io.parsers.TextfileReader]:
    """DOCUMENTATION."""
    return split_by_boolean(
        data,
        mask,
        True,
        reset_index=reset_index,
        inverse=inverse,
        return_rejected=return_rejected,
    )


def split_by_boolean_false(
    data, mask, reset_index=False, inverse=False, return_rejected=False
) -> tuple[pd.DataFrame | pd.io.parsers.TextfileReader]:
    """DOCUMENTATION."""
    return split_by_boolean(
        data,
        mask,
        False,
        reset_index=reset_index,
        inverse=inverse,
        return_rejected=return_rejected,
    )


def split_by_column_entries(
    data, selection, reset_index=False, inverse=False, return_rejected=False
) -> tuple[pd.DataFrame | pd.io.parsers.TextfileReader]:
    """DOCUMENTATION."""
    func = split_dataframe_by_column_entries
    col = list(selection.keys())[0]
    values = list(selection.values())[0]
    return split(
        data,
        func,
        col,
        values,
        reset_index=reset_index,
        inverse=inverse,
        return_rejected=return_rejected,
    )


def split_by_index(
    data, index, reset_index=False, inverse=False, return_rejected=False
) -> tuple[pd.DataFrame | pd.io.parsers.TextfileReader]:
    """DOCUMENTATION."""
    func = split_dataframe_by_index
    return split(
        data,
        func,
        index,
        reset_index=reset_index,
        inverse=inverse,
        return_rejected=return_rejected,
    )
