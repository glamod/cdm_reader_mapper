"""
Common Data Model (CDM) pandas inspection operators.

Created on Wed Jul  3 09:48:18 2019

@author: iregon
"""

from __future__ import annotations

import numpy as np


def count_by_cat_i(series):
    """Count unique values."""
    counts = series.value_counts(dropna=False)
    counts.index.fillna(str(np.nan))
    return counts.to_dict()


def get_length(data):
    """Get length of pandas object.

    Parameters
    ----------
    data: pandas.DataFrame, pd.io.parsers.TextFileReader
        Input dataset

    Returns
    -------
    int
        Total row count
    """
    return len(data)


def count_by_cat(data, columns=None):
    """Count unique values.

    Parameters
    ----------
    data: pandas.DataFrame, pd.io.parsers.TextFileReader
        Input dataset.
    col: str, list or tuple
        Name of the data column to be selected.

    Returns
    -------
    dict
        Dictionary containing number of unique values.
    """
    if columns is None:
        columns = data.columns
    if not isinstance(columns, list):
        columns = [columns]
    counts = {}
    for column in columns:
        counts[column] = count_by_cat_i(data[column])
    return counts
