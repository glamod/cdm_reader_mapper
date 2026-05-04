"""
Common Data Model (CDM) pandas inspection operators.

Created on Wed Jul  3 09:48:18 2019

@author: iregon
"""

from __future__ import annotations
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from .iterators import ParquetStreamReader, ProcessFunction, process_function


def merge_sum_dicts(dicts: list[Mapping[str, Any]]) -> dict[str, Any]:
    """
    Recursively merge dictionaries, summing numeric values at the leaves.

    Parameters
    ----------
    dicts : list of Mapping
        A list of dictionaries for recursiv merging.

    Returns
    -------
    dict
        Recursively merged dictionary.
    """
    result = {}

    for d in dicts:
        for key, value in d.items():
            if key not in result:
                result[key] = value
            else:
                if isinstance(value, Mapping) and isinstance(result[key], Mapping):
                    result[key] = merge_sum_dicts([result[key], value])
                else:
                    result[key] += value

    return result


# def _is_counts_dict(value: Any) -> TypeGuard[dict[str | tuple[str, str], int]]:
#    """Return True if *value* is a dict with the expected key/value types."""
#    return isinstance(value, dict)


def _count_by_cat(df: pd.DataFrame, columns: list[Any]) -> dict[Any, int]:
    """
    Count unique values in a pandas DataFrame, including NaNs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to count unique values.
    columns : list of Any
        Column names for counting unique values.

    Returns
    -------
    dict
        Dictionary containing name and amount of unique values.
    """
    count_dict: dict[Any, int] = {}
    for column in columns:
        counts = df[column].value_counts(dropna=False)
        counts.index = counts.index.where(~counts.index.isna(), "nan")
        count_dict[column] = counts.to_dict()
    return count_dict


@process_function()
def count_by_cat(
    data: pd.DataFrame | Iterable[pd.DataFrame],
    columns: str | tuple[str, str] | list[str | tuple[str, str]] | None = None,
) -> dict[str | tuple[str, str], dict[Any, int]]:
    """
    Count unique values per column in a DataFrame or a Iterable of DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame or Iterable[pd.DataFrame]
        Input dataset.
    columns : str, list or tuple, optional
        Name(s) of the data column(s) to be selected. If None, all columns are used.

    Returns
    -------
    Dict[str | tuple[str, str], int]
        Dictionary where each key is a column name, and each value is a dictionary
        mapping unique values (including NaN as 'nan') to their counts.

    Notes
    -----
    - Works with large files via ParquetStreamReader by iterating through chunks.
    """
    if columns is None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be a pandas DataFrame, not {type(data)}.")
        columns = list(data.columns)

    if isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(columns)

    result = ProcessFunction(
        data=data,
        func=_count_by_cat,
        func_kwargs={"columns": columns},
        non_data_output="acc",
        makecopy=False,
        non_data_proc=merge_sum_dicts,
    )

    if isinstance(result, dict):
        return result

    raise TypeError(f"result is not a dictionary, {type(result)}.")


def _get_length(data: pd.DataFrame) -> int:
    """
    Get length pd.DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to get length.

    Returns
    -------
    int
        Length of `data`.
    """
    return len(data)


@process_function()
def get_length(data: pd.DataFrame | Iterable[pd.DataFrame] | ParquetStreamReader) -> int:
    """
    Get the total number of rows in a pandas object.

    Parameters
    ----------
    data : pandas.DataFrame or Iterable[pd.DataFrame]
        Input dataset.

    Returns
    -------
    int
        Total number of rows.

    Notes
    -----
    - Works with large files via ParquetStreamReader by using a specialized handler
      to count rows without loading the entire file into memory.
    """
    if hasattr(data, "_row_count"):
        return int(data._row_count)

    result = ProcessFunction(
        data=data,
        func=_get_length,
        non_data_output="acc",
        makecopy=True,
        non_data_proc=sum,
    )

    if isinstance(result, int):
        return result

    raise TypeError(f"result is not a integer, {type(result)}.")
