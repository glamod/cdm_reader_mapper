# noqa: D100
"""
Common Data Model (CDM) pandas selection operators.

Created on Wed Jul  3 09:48:18 2019

@author: iregon
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .iterators import ParquetStreamReader, ProcessFunction, process_function


def _concat_indexes(idx_dict):
    selected_idx = pd.Index([]).append(idx_dict[0])
    rejected_idx = pd.Index([]).append(idx_dict[1])
    selected_idx = selected_idx.drop_duplicates()
    rejected_idx = rejected_idx.drop_duplicates()
    return selected_idx, rejected_idx


def _reset_index(data, reset_index=False):
    if reset_index is False:
        return data
    return data.reset_index(drop=True)


def _split_df(
    df: pd.DataFrame,
    mask: pd.DataFrame,
    inverse: bool = False,
    return_rejected: bool = False,
):
    if inverse:
        selected = df[~mask]
        rejected = df[mask] if return_rejected else df.iloc[0:0]
    else:
        selected = df[mask]
        rejected = df[~mask] if return_rejected else df.iloc[0:0]

    selected_idx = mask.index[mask]
    rejected_idx = mask.index[~mask]
    return selected, rejected, selected_idx, rejected_idx


def _split_by_boolean_df(df: pd.DataFrame, mask: pd.DataFrame, boolean: bool, **kwargs):
    if mask.empty:
        mask_sel = pd.Series(boolean, index=df.index)
    else:
        mask_sel = mask.all(axis=1) if boolean else ~mask.any(axis=1)
        mask_sel = mask_sel.fillna(boolean)
    return _split_df(df=df, mask=mask_sel, **kwargs)


def _split_by_column_df(
    df: pd.DataFrame,
    col: str,
    values: Iterable,
    **kwargs,
):
    mask_sel = df[col].isin(values)
    mask_sel.name = col

    return _split_df(df=df, mask=mask_sel, **kwargs)


def _split_by_index_df(
    df: pd.DataFrame,
    index,
    **kwargs,
):
    index = pd.Index(index if isinstance(index, Iterable) else [index])
    mask_sel = pd.Series(df.index.isin(index), index=df.index)
    return _split_df(df=df, mask=mask_sel, **kwargs)


PSR_KWARGS = {
    "makecopy": False,
    "non_data_output": "acc",
    "non_data_proc": _concat_indexes,
}


@process_function(postprocessing={"func": _reset_index, "kwargs": "reset_index"})
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
    Split a DataFrame using a boolean mask via ``split_dataframe_by_boolean``.

    Parameters
    ----------
    data : pandas.DataFrame or Iterable[pd.DataFrame]
        DataFrame to be split.
    mask : pandas.DataFrame or Iterable[pd.DataFrame]
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
    (pandas.DataFrame or ParquetStreamReader, pandas.DataFrame or ParquetStreamReader), pd.Index or pd.MultiIndex, pd.Index or pd.MultiIndex)
        Selected rows (all mask columns True), rejected rows, original indexes of selection and
        original indexes of rejection.
    """
    return ProcessFunction(
        data=data,
        func=_split_by_boolean_df,
        func_args=(mask, boolean),
        func_kwargs={"inverse": inverse, "return_rejected": return_rejected},
        **PSR_KWARGS,
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
        If ``True``, return rejected rows as the second output.
        If ``False``, the rejected output is empty but dtype-preserving.

    Returns
    -------
    (pandas.DataFrame or ParquetStreamReader, pandas.DataFrame or ParquetStreamReader), pd.Index or pd.MultiIndex, pd.Index or pd.MultiIndex)
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
        If ``True``, return rejected rows as the second output.
        If ``False``, the rejected output is empty but dtype-preserving.

    Returns
    -------
    (pandas.DataFrame or ParquetStreamReader, pandas.DataFrame or ParquetStreamReader), pd.Index or pd.MultiIndex, pd.Index or pd.MultiIndex)
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


@process_function(postprocessing={"func": _reset_index, "kwargs": "reset_index"})
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
        If ``True``, return rejected rows as the second output.
        If ``False``, the rejected output is empty but dtype-preserving.

    Returns
    -------
    (pandas.DataFrame or ParquetStreamReader, pandas.DataFrame or ParquetStreamReader), pd.Index or pd.MultiIndex, pd.Index or pd.MultiIndex)
        Selected rows (all mask columns True), rejected rows, original indexes of selection and
        original indexes of rejection.
    """
    col, values = next(iter(selection.items()))
    return ProcessFunction(
        data=data,
        func=_split_by_column_df,
        func_args=(col, values),
        func_kwargs={"inverse": inverse, "return_rejected": return_rejected},
        **PSR_KWARGS,
    )


@process_function(postprocessing={"func": _reset_index, "kwargs": "reset_index"})
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
        If ``True``, return rejected rows as the second output.
        If ``False``, the rejected output is empty but dtype-preserving.

    Returns
    -------
    (pandas.DataFrame or ParquetStreamReader, pandas.DataFrame or ParquetStreamReader), pd.Index or pd.MultiIndex, pd.Index or pd.MultiIndex)
        Selected rows (all mask columns True), rejected rows, original indexes of selection and
        original indexes of rejection.
    """
    return ProcessFunction(
        data=data,
        func=_split_by_index_df,
        func_args=(index,),
        func_kwargs={"inverse": inverse, "return_rejected": return_rejected},
        **PSR_KWARGS,
    )
