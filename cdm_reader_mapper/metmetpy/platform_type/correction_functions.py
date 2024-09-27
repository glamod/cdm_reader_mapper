"""
metmetpy correction functions.

Created on Tue Jun 25 09:07:05 2019

@author: iregon
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

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


def fill_value(
    fill_serie,
    fill_value,
    self_condition_value=None,
    fillna=False,
    out_condition=pd.DataFrame(),
    out_condition_values=None,
    self_out_conditions="intersect",
):
    """DOCUMENTATION."""
    # Modes:
    #   - if not self_condition_value and not out_condition: force fillna = True
    #   - if self_condition: fillna as requested
    #   - if self_condition and out_coundition:
    #       fillna: as requested
    #       self_out_conditions: intersect(and) or join(or) as requested
    #           - out_condition need always to intersect between them
    #   - fillna always joins with self and out condition combination

    if not self_condition_value and not out_condition_values:
        return fill_serie.fillna(fill_value)

    msk_na = (
        fill_serie.isna() if fillna else pd.Series(index=fill_serie.index, data=False)
    )
    msk_self = (
        (fill_serie == self_condition_value)
        if self_condition_value
        else pd.Series(index=fill_serie.index, data=True)
    )

    if len(out_condition) > 0:
        condition_dataframe = out_condition
        condition_values = out_condition_values
        if isinstance(condition_dataframe, pd.Series):
            msk_out = condition_dataframe == list(condition_values.values())[0]
        else:
            msk_out = pd.concat(
                (condition_dataframe[k] == v for k, v in condition_values.items()),
                axis=1,
            ).all(axis=1)
    else:
        msk_out = pd.Series(index=fill_serie.index, data=True)
        self_out_conditions == "intersect"

    if self_out_conditions == "join":
        msk = pd.concat([msk_self, msk_out], axis=1).any(axis=1)
    else:
        msk = pd.concat([msk_self, msk_out], axis=1).all(axis=1)

    msk = pd.concat([msk, msk_na], axis=1).any(axis=1)

    return fill_serie.mask(msk, other=fill_value)


def deck_717_immt(data):
    """DOCUMENTATION."""
    # idt=="NNNNN" & dck==700 & sid == 147 & pt == 5
    drifters = "7"
    # sid = "005"
    # pt = "5"
    buoys = "9"
    # regex = re.compile(r"^\d{5,5}$")
    # id_col = properties.metadata_datamodels.get("id").get("immt")
    # sid_col = properties.metadata_datamodels.get("source").get("immt")
    pt_col = properties.metadata_datamodels.get("platform").get("immt")

    data[pt_col].iloc[np.where(data[pt_col].isna())] = drifters
    loc = np.where((np.isnan(data["N"])) & (data[pt_col] == 0))

    data[pt_col].iloc[loc] = buoys

    return data


def deck_700_imma1(data):
    """DOCUMENTATION."""
    # idt=="NNNNN" & dck==700 & sid == 147 & pt == 5
    drifters = "7"
    sid = "147"
    pt = "5"
    buoys = "6"
    regex = re.compile(r"^\d{5,5}$")
    id_col = properties.metadata_datamodels.get("id").get("imma1")
    sid_col = properties.metadata_datamodels.get("source").get("imma1")
    pt_col = properties.metadata_datamodels.get("platform").get("imma1")

    data[pt_col] = data[pt_col].fillna(drifters)
    loc = (
        (data[id_col].str.match(regex)) & (data[sid_col] == sid) & (data[pt_col] == pt)
    )

    data[pt_col] = data[pt_col].where(~loc, buoys)
    return data


def deck_892_imma1(data):
    """DOCUMENTATION."""
    # idt=="NNNNN" & dck==892 & sid == 29 & pt == 5
    sid = "29"
    pt = "5"
    buoys = "6"
    regex = re.compile(r"^\d{5,5}$")
    id_col = properties.metadata_datamodels.get("id").get("imma1")
    sid_col = properties.metadata_datamodels.get("source").get("imma1")
    pt_col = properties.metadata_datamodels.get("platform").get("imma1")

    loc = (
        (data[id_col].str.match(regex)) & (data[sid_col] == sid) & (data[pt_col] == pt)
    )
    data[pt_col] = data[pt_col].where(~loc, buoys)
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
    return overwrite_data(data, loc, pt_col, buoys)


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
