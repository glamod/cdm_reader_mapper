# noqa: D100
"""
Common Data Model (CDM) pandas selection operators.

Created on Wed Jul  3 09:48:18 2019

@author: iregon
"""
from __future__ import annotations

from io import StringIO
from typing import Sequence, Iterable, Callable

import pandas as pd


def _select_rows_by_index(
    df: pd.DataFrame,
    index_list: str | int | Sequence,
    **kwargs,
) -> pd.DataFrame:
    """Select rows from a DataFrame based on index values."""
    reset_index = kwargs.get("reset_index", False)
    inverse = kwargs.get("inverse", False)

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


def _split_by_index(
    df: pd.DataFrame,
    indexes: str | Sequence,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into two parts based on index values."""
    return_rejected = kwargs.get("return_rejected", False)

    out1 = _select_rows_by_index(
        df,
        indexes,
        **kwargs,
    )
    if return_rejected is True:
        index2 = [idx for idx in df.index if idx not in indexes]
        out2 = _select_rows_by_index(df, index2, **kwargs)
    else:
        out2 = pd.DataFrame(columns=out1.columns)
        out2.__dict__["_prev_index"] = pd.Index([])

    return out1, out2


def _split_by_boolean_mask(
    df: pd.DataFrame, mask: pd.DataFrame, boolean: bool, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame based on a boolean mask using `_split_by_index`."""
    if boolean is True:
        global_mask = mask.all(axis=1)
    else:
        global_mask = ~(mask.any(axis=1))
    indexes = global_mask[global_mask.fillna(boolean)].index

    return _split_by_index(df, indexes, **kwargs)


def _split_by_column_values(
    df: pd.DataFrame, col: str, values: Iterable, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame based on entries in a specific column using `_split_by_index`."""
    in_df = df.loc[df[col].isin(values)]
    index = list(in_df.index)
    return _split_by_index(
        df,
        index,
        **kwargs,
    )


def _split_by_index_values(
    df: pd.DataFrame, index, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame based on index values using `_split_by_index`."""
    return _split_by_index(df, index, **kwargs)


def _split_parser(
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


def _split(
    data: pd.DataFrame | Iterable[pd.DataFrame],
    func: Callable[..., tuple[pd.DataFrame, pd.DataFrame]],
    *args,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply a split function to one or more DataFrames.

    - If `data` is a single DataFrame, pass it to `func`.
    - If `data` is a list of DataFrames, only the first is used here
      (TextFileReader logic is handled separately).
    """
    if not isinstance(data, list):
        data = [data]
    if isinstance(data[0], pd.io.parsers.TextFileReader):
        return _split_parser(data, *args, func=func, **kwargs)
    return func(*data, *args, **kwargs)


def split_by_boolean(
    data: pd.DataFrame,
    mask: pd.DataFrame,
    boolean: bool,
    reset_index: bool = False,
    inverse: bool = False,
    return_rejected: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame using a boolean mask via ``split_dataframe_by_boolean``.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be split.
    mask : pandas.DataFrame
        Boolean mask with the same length as ``data``.
    boolean : bool
        Determines mask interpretation:

        - ``True``  ? select rows where **all** mask columns are True.
        - ``False`` ? select rows where **any** mask column is False.
    reset_index : bool, optional
        If ``True``, reset the index of returned DataFrames.
    inverse : bool, optional
        If ``True``, invert the selection performed by the underlying function.
    return_rejected : bool, optional
        If ``True``, return rejected rows as the second output.
        If ``False``, the rejected output is empty but dtype-preserving.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Tuple ``(selected, rejected)`` returned by the underlying
        ``split_dataframe_by_boolean`` implementation.
    """
    func = _split_by_boolean_mask
    return _split(
        [data, mask],
        func,
        boolean,
        reset_index=reset_index,
        inverse=inverse,
        return_rejected=return_rejected,
    )


def split_by_boolean_true(
    data: pd.DataFrame,
    mask: pd.DataFrame,
    reset_index: bool = False,
    inverse: bool = False,
    return_rejected: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split rows where all mask columns are ``True``.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be split.
    mask : pandas.DataFrame
        Boolean mask with the same length as ``data``.
    reset_index : bool, optional
        If ``True``, reset indices in returned DataFrames.
    inverse : bool, optional
        If ``True``, invert the selection.
    return_rejected : bool, optional
        If ``True``, also return rejected rows.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Selected rows (all mask columns True) and rejected rows.
    """
    return split_by_boolean(
        data,
        mask,
        True,
        reset_index=reset_index,
        inverse=inverse,
        return_rejected=return_rejected,
    )


def split_by_boolean_false(
    data: pd.DataFrame,
    mask: pd.DataFrame,
    reset_index: bool = False,
    inverse: bool = False,
    return_rejected: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split rows where at least one mask column is ``False``.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be split.
    mask : pandas.DataFrame
        Boolean mask with the same length as ``data``.
    reset_index : bool, optional
        If ``True``, reset indices in returned DataFrames.
    inverse : bool, optional
        If ``True``, invert the selection.
    return_rejected : bool, optional
        If ``True``, return rejected rows as well.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Selected rows (any mask column False) and rejected rows.
    """
    return split_by_boolean(
        data,
        mask,
        False,
        reset_index=reset_index,
        inverse=inverse,
        return_rejected=return_rejected,
    )


def split_by_column_entries(
    data: pd.DataFrame,
    selection: dict[str, Iterable],
    reset_index: bool = False,
    inverse: bool = False,
    return_rejected: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame based on matching values in a given column.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be split.
    selection : dict
        Mapping of a column name to an iterable of allowed values.
        Example: ``{"city": ["London", "Berlin"]}``.
    reset_index : bool, optional
        Whether to reset index in returned DataFrames.
    inverse : bool, optional
        If ``True``, invert the selection.
    return_rejected : bool, optional
        If ``True``, return rejected rows as the second DataFrame.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Selected rows (column value in provided list) and rejected rows.
    """
    func = _split_by_column_values
    col = next(iter(selection.keys()))
    values = next(iter(selection.values()))
    return _split(
        data,
        func,
        col,
        values,
        reset_index=reset_index,
        inverse=inverse,
        return_rejected=return_rejected,
    )


def split_by_index(
    data: pd.DataFrame,
    index,
    reset_index: bool = False,
    inverse: bool = False,
    return_rejected: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame by selecting specific index labels.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be split.
    index : label or sequence of labels
        Index values to select.
    reset_index : bool, optional
        If ``True``, reset index in returned DataFrames.
    inverse : bool, optional
        If ``True``, select rows **not** in ``index``.
    return_rejected : bool, optional
        If ``True``, return rejected rows as well.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Selected rows (index in given list) and rejected rows.
    """
    func = _split_by_index_values
    return _split(
        data,
        func,
        index,
        reset_index=reset_index,
        inverse=inverse,
        return_rejected=return_rejected,
    )
