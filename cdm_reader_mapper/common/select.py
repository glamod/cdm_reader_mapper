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
    df: pd.DataFrame,
    index_list: Union[str, Sequence],
    reset_index: bool = False,
    inverse: bool = False,
) -> Union[pd.DataFrame, pd.Series]:
    """
    Select rows from a DataFrame based on index values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    index_list : str or sequence of index values
        Index value(s) to filter by. If a single string is provided, it is converted to a list.
    reset_index : bool, default False
        If True, reset the resulting DataFrame index.
    inverse : bool, default False
        If True, select rows NOT in `index_list`.

    Returns
    -------
    pd.DataFrame or pd.Series
        Subset of the original DataFrame (or Series if the input is a Series) with an
        additional attribute `_prev_index` storing the applied index_list.
    
    Notes
    -----
    - `_prev_index` attribute is attached to the resulting DataFrame/Series for tracking.
    """
    if isinstance(index_list, str):
        index_list = [index_list]
    if isinstance(index_list, list):
        index_list = pd.Index(index_list)
    
    mask = df.index.isin(index_list)
    if inverse is True:
        in_df = df[~mask]
    else:
        in_df = df[mask]

    if reset_index is True:
        in_df = in_df.reset_index(drop=True)

    in_df.__dict__["_prev_index"] = index_list
    return in_df


def _split_dataframe_by_index(
    df: pd.DataFrame,
    index: Union[str, Sequence],
    reset_index: bool = False,
    inverse: bool = False,
    return_rejected: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into two parts based on index values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    index : str or sequence
        Index value(s) to select. Can be a string, list, or tuple.
    reset_index : bool, default False
        If True, reset the index of the resulting DataFrames.
    inverse : bool, default False
        If True, invert the selection.
    return_rejected : bool, default False
        If True, also return the rejected (non-selected) rows as the second DataFrame.
        If False, the second DataFrame will be empty but maintain the column structure.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - First DataFrame: selected rows according to `index` and `inverse`.
        - Second DataFrame: rejected rows (if `return_rejected=True`) or empty DataFrame.
        
        Both DataFrames have an attribute `_prev_index` storing the applied index list.
    """
    out1 = _dataframe_apply_index(
        df,
        index,
        reset_index=reset_index,
        inverse=inverse,
    )
    if return_rejected is True:
        if isinstance(index, str):
            index = [index]
        index2 = [idx for idx in df.index if idx not in index]
        out2 = _dataframe_apply_index(
            df,
            index2,
            reset_index=reset_index,
            inverse=inverse,
        )
    else:
        out2 = pd.DataFrame({col: df[col].iloc[0:0] for col in df.columns})
        out2.__dict__["_prev_index"] = pd.Index([])

    return out1, out2


def split_dataframe_by_boolean(
    df: pd.DataFrame,
    mask: pd.DataFrame,
    boolean: bool,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame based on a boolean mask using `_split_dataframe_by_index`.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    mask : pd.DataFrame
        Boolean DataFrame of the same length as `df`. Determines which rows
        satisfy the condition.
    boolean : bool
        Determines the selection logic:
        - True: select rows where all mask values are True.
        - False: select rows where all mask values are False (inverse selection).
    **kwargs : dict
        Additional arguments passed to `_split_dataframe_by_index` such as:
        - reset_index : bool
        - inverse : bool
        - return_rejected : bool

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - First DataFrame: rows matching the boolean mask.
        - Second DataFrame: remaining rows, or empty if `return_rejected=False`.
        
        Both DataFrames maintain a `_prev_index` attribute storing the applied indices.
    """
    # get the index values and pass to the general function
    # If a mask is empty, assume True (...)
    if mask.empty:
        index = df.index
    else:
      if boolean is True:
        global_mask = mask.eq(True).all(axis=1)
      else:
        global_mask = mask.eq(False).any(axis=1)
      index = global_mask[global_mask].index 

    selected, rejected =  _split_dataframe_by_index(
        df,
        index,
        **kwargs,
    )

    for df_part in [selected, rejected]:
        if df_part.empty:
            df_part[:] = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in df.dtypes.items()})
            df_part.__dict__["_prev_index"] = pd.Index([])

    return selected, rejected    


def split_dataframe_by_column_entries(
    df: pd.DataFrame,
    col: str,
    values: list,
    **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame based on entries in a specific column using `_split_dataframe_by_index`.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Column name to check for specific values.
    values : list
        List of values to select from the column.
    **kwargs : dict
        Additional arguments passed to `_split_dataframe_by_index` such as:
        - reset_index : bool
        - inverse : bool
        - return_rejected : bool

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - First DataFrame: rows where `col` contains a value in `values`.
        - Second DataFrame: remaining rows, or empty if `return_rejected=False`.
        
        Both DataFrames maintain a `_prev_index` attribute storing the applied indices.
    """
    # select rows where column contains one of the values
    in_df = df.loc[df[col].isin(values)]
    index = list(in_df.index)

    selected, rejected = _split_dataframe_by_index(df, index, **kwargs)

    # preserve dtypes for empty selections
    for df_part in [selected, rejected]:
        if df_part.empty:
            df_part[:] = pd.DataFrame({c: pd.Series(dtype=dt) for c, dt in df.dtypes.items()})
            df_part.__dict__["_prev_index"] = pd.Index([])

    return selected, rejected

def split_dataframe_by_index(df: pd.DataFrame, index, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame based on index values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    index : str or sequence
        Index value(s) to select.
    **kwargs : dict
        Additional arguments passed to `_split_dataframe_by_index`, e.g.:
        - reset_index : bool
        - inverse : bool
        - return_rejected : bool

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - First DataFrame: selected rows based on `index`.
        - Second DataFrame: remaining rows (empty if `return_rejected=False`).
        
        Both DataFrames maintain a `_prev_index` attribute storing the applied indices.
    """
    selected, rejected = _split_dataframe_by_index(df, index, **kwargs)

    # Ensure empty selections have proper dtypes
    for df_part in [selected, rejected]:
        if df_part.empty:
            df_part[:] = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in df.dtypes.items()})
            df_part.__dict__["_prev_index"] = pd.Index([])

    return selected, rejected



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
    data: Union[pd.DataFrame, pd.io.parsers.TextFileReader, Iterable],
    func: Callable[..., Tuple[pd.DataFrame, pd.DataFrame]],
    *args,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply a split function to one or more DataFrames.

    This function normalizes `data` into a list, then dispatches to either:
    - `split_parser` if the elements are `TextFileReader` objects.
    - `func` otherwise (user-provided split function such as 
      `split_dataframe_by_index`, `split_dataframe_by_column_entries`, etc.).

    Parameters
    ----------
    data : DataFrame, TextFileReader, or iterable of either
        Input to be processed. If a single object is provided, it is wrapped
        into a list. If the first element is a `TextFileReader`, the call is
        delegated to `split_parser`.
    func : callable
        Function to apply when working with DataFrames. Must accept the 
        unpacked elements of `data` followed by `*args` and `**kwargs`, and 
        return a tuple of two DataFrames.
    *args : tuple
        Additional positional arguments forwarded to `func` or `split_parser`.
    **kwargs : dict
        Additional keyword arguments forwarded to `func` or `split_parser`.

    Returns
    -------
    tuple of DataFrame
        Tuple containing:
        - The selected subset DataFrame
        - The rejected subset DataFrame

    Notes
    -----
    - When `data` contains `TextFileReader` objects, the execution path routes
      to `split_parser` and the behavior depends on that function.
    """
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
    data: pd.DataFrame,
    mask: pd.DataFrame,
    boolean: bool,
    reset_index: bool = False,
    inverse: bool = False,
    return_rejected: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame using a boolean mask via `split_dataframe_by_boolean`.

    This is a convenience wrapper around :func:`split` that applies
    `split_dataframe_by_boolean` to a DataFrame and mask pair.

    Parameters
    ----------
    data : DataFrame
        The DataFrame to be split.
    mask : DataFrame
        A boolean DataFrame of the same length as `data`.
    boolean : bool
        Determines mask interpretation:
        - True  ? select rows where *all* mask columns are True
        - False ? select rows where *any* mask column is False
    reset_index : bool, default False
        Whether to reset the index in the two returned DataFrames.
    inverse : bool, default False
        If True, invert the selection when passing to the underlying function.
    return_rejected : bool, default False
        If True, return both selected and rejected rows.
        If False, rejected part is empty but retains column structure.

    Returns
    -------
    (DataFrame, DataFrame)
        The selected and rejected DataFrames returned by
        `split_dataframe_by_boolean`.

    Notes
    -----
    - The `_prev_index` attribute is preserved as implemented in the
      underlying split functions.
    """
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
    data: pd.DataFrame,
    mask: pd.DataFrame,
    reset_index: bool = False,
    inverse: bool = False,
    return_rejected: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame using a boolean mask, selecting rows where all mask columns are True.

    This is a convenience wrapper around `split_by_boolean` that automatically sets
    `boolean=True`.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be split.
    mask : pd.DataFrame
        A boolean DataFrame of the same length as `data`.
    reset_index : bool, default False
        Whether to reset the index in the two returned DataFrames.
    inverse : bool, default False
        If True, invert the selection when passing to the underlying function.
    return_rejected : bool, default False
        If True, return both selected and rejected rows.
        If False, the rejected part is empty but retains column structure.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - The first DataFrame contains rows where all mask columns are True.
        - The second DataFrame contains the remaining rows (empty if `return_rejected=False`).
        
    Notes
    -----
    - The `_prev_index` attribute is preserved as implemented in the underlying split functions.
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame using a boolean mask, selecting rows where any mask column is False.

    This is a convenience wrapper around `split_by_boolean` that automatically sets
    `boolean=False`.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be split.
    mask : pd.DataFrame
        A boolean DataFrame of the same length as `data`.
    reset_index : bool, default False
        Whether to reset the index in the two returned DataFrames.
    inverse : bool, default False
        If True, invert the selection when passing to the underlying function.
    return_rejected : bool, default False
        If True, return both selected and rejected rows.
        If False, the rejected part is empty but retains column structure.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - The first DataFrame contains rows where any mask column is False.
        - The second DataFrame contains the remaining rows (empty if `return_rejected=False`).

    Notes
    -----
    - The `_prev_index` attribute is preserved as implemented in the underlying split functions.
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
    selection: Dict[str, Iterable],
    reset_index: bool = False,
    inverse: bool = False,
    return_rejected: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame based on whether values in a specified column match
    any value in a given list.

    This is a convenience wrapper around `split` that uses
    `split_dataframe_by_column_entries` as the splitting function.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be split.
    selection : dict
        A dictionary mapping a column name to an iterable of values to select.
        Example: {"city": ["London", "Berlin"]}.
    reset_index : bool, default False
        Whether to reset the index in both returned DataFrames.
    inverse : bool, default False
        If True, invert the selection when performing the split.
    return_rejected : bool, default False
        If True, return rejected rows as the second DataFrame.
        If False, return an empty (dtype-preserving) DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple (selected, rejected) where:
        - selected contains rows with column values in `selection[col]`
        - rejected contains the remaining rows (or empty if `return_rejected=False`)

    Notes
    -----
    - The `_prev_index` attribute is preserved by the underlying split functions.
    """
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
    data: pd.DataFrame,
    index,
    reset_index: bool = False,
    inverse: bool = False,
    return_rejected: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame based on index values.

    This is a convenience wrapper around `split` using
    `split_dataframe_by_index` as the underlying function.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be split.
    index : label or list-like
        Index values to select from the DataFrame.
    reset_index : bool, default False
        Whether to reset the index of returned DataFrames.
    inverse : bool, default False
        If True, select all rows *except* those in `index`.
    return_rejected : bool, default False
        If True, return both selected and rejected rows.
        If False, rejected is empty but dtype-preserving.

    Returns
    -------
    (DataFrame, DataFrame)
        Selected and rejected DataFrames.
    """
    func = split_dataframe_by_index
    return split(
        data,
        func,
        index,
        reset_index=reset_index,
        inverse=inverse,
        return_rejected=return_rejected,
    )
