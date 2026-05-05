# noqa: D100
"""
Common Data Model (CDM) pandas selection operators.

Created on Wed Jul  3 09:48:18 2019

@author: iregon
"""

from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

from .iterators import ParquetStreamReader, ProcessFunction, process_function


def _concat_indexes(idx_dict: dict[int, Any]) -> tuple[pd.Index, pd.Index]:
    """
    Concatenate and deduplicate index collections.

    Parameters
    ----------
    idx_dict : dict
        Dictionary containing index-like objects under keys `0` (selected)
        and `1` (rejected).

    Returns
    -------
    tuple of pd.Index and pd.Index
        Tuple of unique selected and rejected indices.
    """
    selected_idx = pd.Index([]).append(idx_dict[0])
    rejected_idx = pd.Index([]).append(idx_dict[1])
    selected_idx = selected_idx.drop_duplicates()
    rejected_idx = rejected_idx.drop_duplicates()
    return selected_idx, rejected_idx


def _reset_index(data: pd.DataFrame | pd.Series, reset_index: bool = False) -> pd.DataFrame | pd.Series:
    """
    Optionally reset the index of a DataFrame or Series.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Input data.
    reset_index : bool, default: False
        If True, reset the index and drop the old index.

    Returns
    -------
    pd.DataFrame or pd.Series
        Data with reset index if requested, otherwise unchanged.
    """
    if reset_index is False:
        return data
    return data.reset_index(drop=True)


def _split_df(
    df: pd.DataFrame,
    mask: pd.DataFrame | pd.Series,
    inverse: bool = False,
    return_rejected: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into selected and rejected subsets using a boolean mask.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to split.
    mask : pd.DataFrame or pandas.Series
        Boolean mask used for selection.
    inverse : bool, default: False
        If True invert the selection logic.
    return_rejected : bool, default: False
        If True return the rejected subset; otherwise returns an empty DataFrame.

    Returns
    -------
    tuple of pd.DataFrame and pd.DataFrame and pd.Index and pd.Index]
        Selected DataFrame, rejected DataFrame, selected indices, rejected indices.
    """
    if inverse:
        selected = df[~mask]
        rejected = df[mask] if return_rejected else df.iloc[0:0]
    else:
        selected = df[mask]
        rejected = df[~mask] if return_rejected else df.iloc[0:0]

    selected_idx = mask.index[mask]
    rejected_idx = mask.index[~mask]
    return selected, rejected, selected_idx, rejected_idx


def _split_by_boolean_df(
    df: pd.DataFrame, mask: pd.DataFrame, boolean: bool, **kwargs: Any
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r"""
    Split a DataFrame based on a boolean DataFrame mask.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to split.
    mask : pd.DataFrame
        Boolean DataFrame used to compute row-wise selection.
    boolean : bool
        Determines selection logic:
        - True: select rows where all mask values are True
        - False: select rows where no mask values are True
    \**kwargs : Any
        Additional keyword arguments passed to `_split_df`.

    Returns
    -------
    tuple of pd.DataFrame and pd.DataFrame and pd.Index and pd.Index]
        Selected DataFrame, rejected DataFrame, selected indices, rejected indices.
    """
    if mask.empty:
        mask_sel = pd.Series(boolean, index=df.index)
    else:
        mask_sel = mask.all(axis=1) if boolean else ~mask.any(axis=1)
        mask_sel = mask_sel.fillna(boolean)
    return _split_df(df=df, mask=mask_sel, **kwargs)


def _split_by_column_df(
    df: pd.DataFrame,
    col: str,
    values: Iterable[Any],
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r"""
    Split a DataFrame based on column membership.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to split.
    col : str
        Column name used for filtering.
    values : Iterable of Any
        Values to match against the column.
    \**kwargs : Any
        Additional keyword arguments passed to `_split_df`.

    Returns
    -------
    tuple of pd.DataFrame and pd.DataFrame and pd.Index and pd.Index
        Selected DataFrame, rejected DataFrame, selected indices, rejected indices.
    """
    mask_sel = df[col].isin(values)
    mask_sel.name = col

    return _split_df(df=df, mask=mask_sel, **kwargs)


def _split_by_index_df(
    df: pd.DataFrame,
    index: int | Iterable[int],
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r"""
    Split a DataFrame based on index values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to split.
    index : int or Iterable of int
        Index value(s) used for selection.
    \**kwargs : Any
        Additional keyword arguments passed to `_split_df`.

    Returns
    -------
    tuple of pd.DataFrame and pd.DataFrame and pd.Index and pd.Index
        Selected DataFrame, rejected DataFrame, selected indices, rejected indices.
    """
    index = pd.Index(index if isinstance(index, Iterable) else [index])
    mask_sel = pd.Series(df.index.isin(index), index=df.index)
    return _split_df(df=df, mask=mask_sel, **kwargs)


PSR_KWARGS = {
    "makecopy": False,
    "non_data_output": "acc",
    "non_data_proc": _concat_indexes,
}


def split_by_boolean(
    data: pd.DataFrame | Iterable[pd.DataFrame],
    mask: pd.DataFrame | Iterable[pd.DataFrame],
    boolean: bool,
    reset_index: bool = False,
    inverse: bool = False,
    return_rejected: bool = False,
) -> tuple[
    pd.DataFrame | ParquetStreamReader,
    pd.DataFrame | ParquetStreamReader,
    pd.Index | pd.MultiIndex,
    pd.Index | pd.MultiIndex,
]:
    """
    Split both a DataFrame and an Iterable of DataFrames using a boolean mask via ``split_dataframe_by_boolean``.

    Parameters
    ----------
    data : pd.DataFrame or Iterable of pd.DataFrame
        DataFrame to be split.
    mask : pd.DataFrame or Iterable of pd.DataFrame
        Boolean mask with the same length as `data`.
    boolean : bool
        Determines mask interpretation:

        - If True select rows where **all** mask columns are True.
        - If False select rows where **any** mask column is False.
    reset_index : bool, optional
        If True reset the index of returned DataFrames.
    inverse : bool, optional
        If True invert the selection performed by the underlying function.
    return_rejected : bool, optional
        If True return rejected rows as the second output.
        If False the rejected output is empty but dtype-preserving.

    Returns
    -------
    tuple of pd.DataFrame or ParquetStreamReader and pd.DataFrame or ParquetStreamReader and pd.Index or pd.MultiIndex and pd.Index or pd.MultiIndex
        Selected rows (all mask columns True), rejected rows, original indexes of selection and
        original indexes of rejection.
    """

    @process_function(postprocessing={"func": _reset_index, "kwargs": "reset_index"})
    def _split_by_boolean(reset_index: bool = reset_index) -> ProcessFunction:
        """
        Split both a DataFrame or an Iterable of DataFrames using a boolean mask.

        Parameters
        ----------
        reset_index : bool
            If True reset the index of returned DataFrames.

        Returns
        -------
        ProcessFunction
            Containing selected DataFrame, rejected DataFrame, selected indices, rejected indices.
        """
        return ProcessFunction(
            data=data,
            func=_split_by_boolean_df,
            func_args=(mask, boolean),
            func_kwargs={"inverse": inverse, "return_rejected": return_rejected},
            **PSR_KWARGS,
        )

    result = _split_by_boolean()
    return tuple(result)


def split_by_boolean_true(
    data: pd.DataFrame | Iterable[pd.DataFrame],
    mask: pd.DataFrame | Iterable[pd.DataFrame],
    reset_index: bool = False,
    inverse: bool = False,
    return_rejected: bool = False,
) -> tuple[
    pd.DataFrame | ParquetStreamReader,
    pd.DataFrame | ParquetStreamReader,
    pd.Index | pd.MultiIndex,
    pd.Index | pd.MultiIndex,
]:
    """
    Split both a DataFrame or an Iterable of DataFrames where boolean mask is True.

    Parameters
    ----------
    data : pd.DataFrame or Iterable of pd.DataFrame
        DataFrame to be split.
    mask : pd.DataFrame or Iterable of pd.DataFrame
        Boolean mask with the same length as `data`.
    reset_index : bool, optional
        If True reset indices in returned DataFrames.
    inverse : bool, optional
        If True invert the selection.
    return_rejected : bool, optional
        If True return rejected rows as the second output.
        If False the rejected output is empty but dtype-preserving.

    Returns
    -------
    tuple of pd.DataFrame or ParquetStreamReader and pd.DataFrame or ParquetStreamReader and pd.Index or pd.MultiIndex and pd.Index or pd.MultiIndex
        Selected rows (all mask columns True), rejected rows, original indexes of selection and
        original indexes of rejection.
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
    data: pd.DataFrame | Iterable[pd.DataFrame],
    mask: pd.DataFrame | Iterable[pd.DataFrame],
    reset_index: bool = False,
    inverse: bool = False,
    return_rejected: bool = False,
) -> tuple[
    pd.DataFrame | ParquetStreamReader,
    pd.DataFrame | ParquetStreamReader,
    pd.Index | pd.MultiIndex,
    pd.Index | pd.MultiIndex,
]:
    """
    Split both a DataFrame or an Iterable of DataFrames where boolean mask is False.

    Parameters
    ----------
    data : pd.DataFrame or Iterable[pd.DataFrame]
        DataFrame to be split.
    mask : pd.DataFrame or Iterable[pd.DataFrame]
        Boolean mask with the same length as `data`.
    reset_index : bool, optional
        If True reset indices in returned DataFrames.
    inverse : bool, optional
        If True invert the selection.
    return_rejected : bool, optional
        If True return rejected rows as the second output.
        If False the rejected output is empty but dtype-preserving.

    Returns
    -------
    tuple of pd.DataFrame or ParquetStreamReader and pd.DataFrame or ParquetStreamReader and pd.Index or pd.MultiIndex and pd.Index or pd.MultiIndex
        Selected rows (all mask columns True), rejected rows, original indexes of selection and
        original indexes of rejection.
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
    data: pd.DataFrame | Iterable[pd.DataFrame],
    selection: dict[str | tuple[str, str], Sequence[Any]],
    reset_index: bool = False,
    inverse: bool = False,
    return_rejected: bool = False,
) -> tuple[
    pd.DataFrame | ParquetStreamReader,
    pd.DataFrame | ParquetStreamReader,
    pd.Index | pd.MultiIndex,
    pd.Index | pd.MultiIndex,
]:
    """
    Split both a DataFrame or an Iterable of DataFrames based on matching values in a given column.

    Parameters
    ----------
    data : pd.DataFrame or Iterable of pd.DataFrame
        DataFrame to be split.
    selection : dict
        Mapping of a column name to an iterable of allowed values.
        Example: {"city": ["London", "Berlin"]}.
    reset_index : bool, optional
        Whether to reset index in returned DataFrames.
    inverse : bool, optional
        If True invert the selection.
    return_rejected : bool, optional
        If True return rejected rows as the second output.
        If False the rejected output is empty but dtype-preserving.

    Returns
    -------
    tuple of pd.DataFrame or ParquetStreamReader and pd.DataFrame or ParquetStreamReader and pd.Index or pd.MultiIndex and pd.Index or pd.MultiIndex
        Selected rows (all mask columns True), rejected rows, original indexes of selection and
        original indexes of rejection.
    """

    @process_function(postprocessing={"func": _reset_index, "kwargs": "reset_index"})
    def _split_by_column_entries(reset_index: bool = reset_index) -> ProcessFunction:
        """
        Split both a DataFrame or an Iterable of DataFrames based on matching values in a given column.

        Parameters
        ----------
        reset_index : bool
            If True reset the index of returned DataFrames.

        Returns
        -------
        ProcessFunction
            Containing selected DataFrame, rejected DataFrame, selected indices, rejected indices.
        """
        return ProcessFunction(
            data=data,
            func=_split_by_column_df,
            func_args=(col, values),
            func_kwargs={"inverse": inverse, "return_rejected": return_rejected},
            **PSR_KWARGS,
        )

    col, values = next(iter(selection.items()))
    result = _split_by_column_entries()
    return tuple(result)


def split_by_index(
    data: pd.DataFrame | Iterable[pd.DataFrame],
    index: int | Iterable[int],
    reset_index: bool = False,
    inverse: bool = False,
    return_rejected: bool = False,
) -> tuple[
    pd.DataFrame | ParquetStreamReader,
    pd.DataFrame | ParquetStreamReader,
    pd.Index | pd.MultiIndex,
    pd.Index | pd.MultiIndex,
]:
    """
    Split both a DataFrame or an Iterable of DataFrames by selecting specific index labels.

    Parameters
    ----------
    data : pd.DataFrame or Iterable of DataFrame
        DataFrame to be split.
    index : label or sequence of labels
        Index values to select.
    reset_index : bool, optional
        If True reset index in returned DataFrames.
    inverse : bool, optional
        If True select rows **not** in `index`.
    return_rejected : bool, optional
        If True return rejected rows as the second output.
        If False the rejected output is empty but dtype-preserving.

    Returns
    -------
    tuple of pd.DataFrame or ParquetStreamReader and pd.DataFrame or ParquetStreamReader and pd.Index or pd.MultiIndex and pd.Index or pd.MultiIndex
        Selected rows (all mask columns True), rejected rows, original indexes of selection and
        original indexes of rejection.
    """

    @process_function(postprocessing={"func": _reset_index, "kwargs": "reset_index"})
    def _split_by_index(reset_index: bool = reset_index) -> ProcessFunction:
        """
        Split both a DataFrame or an Iterable of DataFrames by selecting specific index labels.

        Parameters
        ----------
        reset_index : bool
            If True reset the index of returned DataFrames.

        Returns
        -------
        ProcessFunction
            Containing selected DataFrame, rejected DataFrame, selected indices, rejected indices.
        """
        return ProcessFunction(
            data=data,
            func=_split_by_index_df,
            func_args=(index,),
            func_kwargs={"inverse": inverse, "return_rejected": return_rejected},
            **PSR_KWARGS,
        )

    result = _split_by_index()
    return tuple(result)
