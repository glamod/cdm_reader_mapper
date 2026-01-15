"""
Common Data Model (CDM) pandas inspection operators.

Created on Wed Jul  3 09:48:18 2019

@author: iregon
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from . import pandas_TextParser_hdlr


def _count_by_cat(series) -> dict:
    """Count unique values in a pandas Series, including NaNs."""
    counts = series.value_counts(dropna=False)
    counts.index = counts.index.map(lambda x: "nan" if pd.isna(x) else x)
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

    counts = {}

    if isinstance(data, pd.DataFrame):
        for column in columns:
            counts[column] = _count_by_cat(data[column])
        return counts

    for column in columns:
        data_cp = pandas_TextParser_hdlr.make_copy(data)
        count_dicts = []

        for df in data_cp:
            count_dicts.append(_count_by_cat(df[column]))

        data_cp.close()

        merged_counts = {}
        for d in count_dicts:
            for k, v in d.items():
                merged_counts[k] = merged_counts.get(k, 0) + v

        counts[column] = merged_counts

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
    return pandas_TextParser_hdlr.get_length(data)
