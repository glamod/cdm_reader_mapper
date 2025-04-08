"""
Common Data Model (CDM) pandas inspection operators.

Created on Wed Jul  3 09:48:18 2019

@author: iregon
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cdm_reader_mapper.common import pandas_TextParser_hdlr


def count_by_cat_i(series) -> dict:
    """Count unique values."""
    counts = series.value_counts(dropna=False)
    counts.index.fillna(str(np.nan))
    return counts.to_dict()


def get_length(data) -> int:
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
    if not isinstance(data, pd.io.parsers.TextFileReader):
        return len(data)
    else:
        return pandas_TextParser_hdlr.get_length(data)


def count_by_cat(data, columns=None) -> dict:
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
    if not isinstance(data, pd.io.parsers.TextFileReader):
        for column in columns:
            counts[column] = count_by_cat_i(data[column])
        return counts
    for column in columns:
        data_cp = pandas_TextParser_hdlr.make_copy(data)
        count_dicts = []
        for df in data_cp:
            count_dicts.append(count_by_cat_i(df[column]))

        data_cp.close()
        cats = [list(x.keys()) for x in count_dicts]
        cats = list({x for y in cats for x in y})
        cats.sort
        count_dict = {}
        for cat in cats:
            count_dict[cat] = sum([x.get(cat) for x in count_dicts if x.get(cat)])
        counts[column] = count_dict
    return counts
