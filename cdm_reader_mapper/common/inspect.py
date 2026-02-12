"""
Common Data Model (CDM) pandas inspection operators.

Created on Wed Jul  3 09:48:18 2019

@author: iregon
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd

from .iterators import process_disk_backed, is_valid_iterator


def _count_by_cat(df, columns) -> dict:
    """Count unique values in a pandas Series, including NaNs."""
    count_dict = {}
    for column in columns:
        counts = df[column].value_counts(dropna=False)
        counts.index = counts.index.where(~counts.index.isna(), "nan")
        count_dict[column] = counts.to_dict()
    return count_dict


def count_by_cat(
    data: pd.DataFrame | pd.io.parsers.TextFileReader,
    columns: str | list[str] | tuple | None = None,
) -> dict[str, dict[Any, int]]:
    """
    Count unique values per column in a DataFrame or a TextFileReader.

    Parameters
    ----------
    data : pandas.DataFrame or pd.io.parsers.TextFileReader
        Input dataset.
    columns : str, list or tuple, optional
        Name(s) of the data column(s) to be selected. If None, all columns are used.

    Returns
    -------
    Dict[str, Dict[Any, int]]
        Dictionary where each key is a column name, and each value is a dictionary
        mapping unique values (including NaN as 'nan') to their counts.

    Notes
    -----
    - Works with large files via TextFileReader by iterating through chunks.
    """

    def merge_sum_dicts(*dicts):
        """Recursively merge dictionaries, summing numeric values at the leaves."""
        result = {}

        for d in dicts:
            for key, value in d.items():
                if key not in result:
                    result[key] = value
                else:
                    if isinstance(value, Mapping) and isinstance(result[key], Mapping):
                        result[key] = merge_sum_dicts(result[key], value)
                    else:
                        result[key] += value

        return result

    if columns is None:
        columns = data.columns
    if not isinstance(columns, list):
        columns = [columns]

    if isinstance(data, pd.DataFrame):
        return _count_by_cat(data, columns)

    if is_valid_iterator(data):
        dicts = process_disk_backed(
            data,
            _count_by_cat,
            func_kwargs={"columns": columns},
            non_data_output="acc",
            makecopy=False,
        )
        return merge_sum_dicts(*dicts[0])

    raise TypeError(f"Unsupported data type: {type(data)}")


def _get_length(data: pd.DataFrame):
    """Get length pd.DataFrame."""
    return len(data)


def get_length(data: pd.DataFrame | Iterable[pd.DataFrame]) -> int:
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
    - Works with large files via TextFileReader by using a specialized handler
      to count rows without loading the entire file into memory.
    """
    if isinstance(data, pd.DataFrame):
        return _get_length(data)

    if hasattr(data, "_row_count"):
        return data._row_count

    if is_valid_iterator(data):
        result = process_disk_backed(
            data,
            _get_length,
            non_data_output="acc",
            makecopy=False,
        )
        print(result)
        return sum(result[0])

    raise TypeError(f"Unsupported data type: {type(data)}")
