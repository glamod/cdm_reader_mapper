"""
Common Data Model (CDM) pandas inspection operators.

Created on Wed Jul  3 09:48:18 2019

@author: iregon
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def count_by_cat_i(serie):
    """Count unique values."""
    counts = serie.value_counts(dropna=False)
    counts.index.fillna(str(np.nan))
    return counts.to_dict()


def get_length(data):
    """Get length of pandas object."""
    return len(data)


def count_by_cat(data, col):
    """Count unique values."""
    return count_by_cat_i(data[col])
