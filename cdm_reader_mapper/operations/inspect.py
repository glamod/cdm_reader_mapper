"""
Common Data Model (CDM) pandas inspection operators.

Created on Wed Jul  3 09:48:18 2019

@author: iregon
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cdm_reader_mapper.common import pandas_TextParser_hdlr


def count_by_cat_i(series):
    """Count unique values."""
    counts = series.value_counts(dropna=False)
    counts.index.fillna(str(np.nan))
    return counts.to_dict()


def get_length(data):
    """Get length of pandas object.

    Parameters
    ----------
    data: pd.DataFrame, pd.io.parsers.TextFileReader
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


def count_by_cat(data, col):
    """Count unique values.

    Parameters
    ----------
    data: pd.DataFrame, pd.io.parsers.TextFileReader
        Input dataset.
    col: str, tuple
        Name of the data column to be selected.

    Returns
    -------
    dict
        Dictionary containing number of unique values.
    """
    if not isinstance(data, pd.io.parsers.TextFileReader):
        return count_by_cat_i(data[col])
    else:
        data_cp = pandas_TextParser_hdlr.make_copy(data)
        count_dicts = []
        for df in data_cp:
            count_dicts.append(count_by_cat_i(df[col]))

        data_cp.close()
        cats = [list(x.keys()) for x in count_dicts]
        cats = list({x for y in cats for x in y})
        cats.sort
        count_dict = {}
        for cat in cats:
            count_dict[cat] = sum([x.get(cat) for x in count_dicts if x.get(cat)])
        return count_dict
