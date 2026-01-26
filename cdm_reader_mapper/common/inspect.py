"""
Common Data Model (CDM) pandas inspection operators.

Created on Wed Jul  3 09:48:18 2019

@author: iregon
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .pandas_TextParser_hdlr import make_copy
from .pandas_TextParser_hdlr import get_length as get_length_hdlr


def _count_by_cat(series) -> dict:
    """Count unique values in a pandas Series, including NaNs."""
    counts = series.value_counts(dropna=False)
    counts.index = counts.index.where(~counts.index.isna(), "nan")
    return counts.to_dict()


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
    if columns is None:
        columns = data.columns
    if not isinstance(columns, list):
        columns = [columns]

    counts = {col: {} for col in columns}

    if isinstance(data, pd.DataFrame):
        for column in columns:
            counts[column] = _count_by_cat(data[column])
        return counts

    data_cp = make_copy(data)
    if data_cp is None:
        return counts

    for chunk in data_cp:
        for column in columns:
            chunk_counts = _count_by_cat(chunk[column])
            for k, v in chunk_counts.items():
                counts[column][k] = counts[column].get(k, 0) + v

    data_cp.close()
    return counts


def get_length(data: pd.DataFrame | pd.io.parsers.TextFileReader) -> int:
    """
    Get the total number of rows in a pandas object.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.io.parsers.TextFileReader
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
    if not isinstance(data, pd.io.parsers.TextFileReader):
        return len(data)
    return get_length_hdlr(data)
