"""
metmetpy platfrom_type icoads_r3000_NRT library.

Created on Tue Jun 25 09:07:05 2019

@author: iregon
"""

from __future__ import annotations

import re

from .. import properties


def is_num(X):
    """DOCUMENTATION."""
    try:
        a = X.isnumeric()
    except Exception:
        a = False
    return a


def overwrite_data(data, loc, pt_col, value):
    """Overwrite data."""
    if pt_col not in data.columns:
        return data
    if not any(loc):
        return data
    data.loc[data[pt_col][loc], pt_col] = value
    return data


def deck_792_imma1(data):
    """DOCUMENTATION."""
    sid = "103"
    pt = "5"
    buoys = "6"
    regex = re.compile("^[0-9]+$")  # is numeric
    id_col = properties.metadata_datamodels.get("id").get("imma1")
    sid_col = properties.metadata_datamodels.get("source").get("imma1")
    pt_col = properties.metadata_datamodels.get("platform").get("imma1")

    loc = (
        (data[id_col].str.match(regex))
        & (data[id_col].apply(len) != 7)
        & (data[id_col].apply(len) != 5)
        & (data[id_col].str.startswith("7") is False)
        & (data[sid_col] == sid)
        & (data[pt_col] == pt)
    )

    if not any(loc):
      return data

    try:
        data.loc[data[pt_col][loc], pt_col] = buoys
    except KeyError:
        print("Data stays untouched.")
    return data


def deck_992_imma1(data):
    """DOCUMENTATION."""
    sid = "114"
    pt = "5"
    lv = "4"  # light vessels
    buoys = "6"
    regex = re.compile("^6202+$")
    id_col = properties.metadata_datamodels.get("id").get("imma1")
    sid_col = properties.metadata_datamodels.get("source").get("imma1")
    pt_col = properties.metadata_datamodels.get("platform").get("imma1")

    loc = (
        (data[id_col].str.match(regex))
        & (data[id_col].str.len() == 7)
        & (data[sid_col] == sid)
        & (data[pt_col] == pt)
    )

    data = overwrite_data(data, loc, pt_col, lv)

    # light vessels in dck 992
    regex = re.compile("^[0-9]+$")  # is numeric

    loc = (
        (data[id_col].str.match(regex))
        & (data[id_col].str.len() != 7)
        & (data[id_col].str.len() != 5)
        & (data[id_col].str.startswith("7") is False)
        & (data[sid_col] == sid)
        & (data[pt_col] == pt)
    )
    return overwrite_data(data, loc, pt_col, buoys)
