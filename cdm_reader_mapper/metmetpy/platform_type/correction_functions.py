"""
metmetpy correction functions.

Created on Tue Jun 25 09:07:05 2019

@author: iregon
"""

from __future__ import annotations

import re

import pandas as pd

from typing import Any

from .. import properties


def is_num(x: Any) -> bool:
    """
    Check whether a value represents a numeric string.

    Parameters
    ----------
    x : Any
        The value to test.

    Returns
    -------
    bool
        True if x is a string containing only numeric characters (0-9),
        False otherwise. Non-string values will return False.
    """
    if isinstance(x, str):
        return x.isnumeric()
    return False


def overwrite_data(
    data: pd.DataFrame,
    loc: pd.Series | pd.Index | list | pd.Series,
    pt_col: str,
    value,
) -> pd.DataFrame:
    """
    Overwrite values in a DataFrame column based on a boolean location mask.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    loc : array-like of bool
        Boolean mask indicating which rows to overwrite.
    pt_col : str
        Name of the column to overwrite.
    value : any
        Value to assign to the specified rows and column.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated values in the specified column.
    """
    if pt_col not in data.columns:
        return data

    if not any(loc):
        return data

    data.loc[loc, pt_col] = value
    return data


def fill_value(
    fill_serie: pd.Series,
    fill_value,
    self_condition_value=None,
    fillna: bool = False,
    out_condition: pd.DataFrame | None = None,
    out_condition_values: dict | None = None,
    self_out_conditions: str = "intersect",
) -> pd.Series:
    """
    Fill values in a Series conditionally, with optional self and external conditions.

    Modes:
    1. If no `self_condition_value` and no `out_condition_values`, fills all NA values.
    2. If `self_condition_value` is given, fill where Series equals this value.
    3. If `out_condition` and `out_condition_values` are given, fill only where
    the external conditions match (can combine with self_condition_value).
    4. `self_out_conditions` controls whether self and external conditions are combined with
    "intersect" (AND) or "join" (OR).
    5. `fillna` always allows filling of NA values.

    Parameters
    ----------
    fill_serie : pd.Series
        Series to fill.
    fill_value : any
        Value used to fill.
    self_condition_value : optional
        Value in `fill_serie` that triggers filling.
    fillna : bool, default False
        Whether to fill NA values in addition to conditions.
    out_condition : pd.DataFrame, optional
        External DataFrame with conditions for filling.
    out_condition_values : dict, optional
        Mapping of columns in `out_condition` to values that trigger filling.
    self_out_conditions : {"intersect", "join"}, default "intersect"
        How to combine self_condition and out_condition:
        - "intersect" = AND
        - "join" = OR

    Returns
    -------
    pd.Series
        Series with values conditionally filled.
    """
    if out_condition is None:
        out_condition = pd.DataFrame(index=fill_serie.index)
    if out_condition_values is None:
        out_condition_values = {}

    if self_condition_value is None and not out_condition_values:
        return fill_serie.fillna(fill_value)

    msk_na = fill_serie.isna() if fillna else pd.Series(False, index=fill_serie.index)

    msk_self = (
        fill_serie == self_condition_value
        if self_condition_value is not None
        else pd.Series(True, index=fill_serie.index)
    )

    if len(out_condition) > 0 and out_condition_values:
        if isinstance(out_condition, pd.Series):
            _, value = list(out_condition_values.items())[0]
            msk_out = out_condition == value
        else:
            msk_out = pd.concat(
                (out_condition[k] == v for k, v in out_condition_values.items()), axis=1
            ).all(axis=1)
    else:
        msk_out = pd.Series(True, index=fill_serie.index)
        self_out_conditions = "intersect"

    if self_out_conditions == "join":
        msk = pd.concat([msk_self, msk_out], axis=1).any(axis=1)
    else:
        msk = pd.concat([msk_self, msk_out], axis=1).all(axis=1)

    msk = pd.concat([msk, msk_na], axis=1).any(axis=1)

    return fill_serie.mask(msk, other=fill_value)


def deck_717_gdac(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill and adjust platform type for GDAC deck 717 dataset.

    Rules:
    - If platform column is NA, assign '7' (drifters)
    - If column 'N' is NaN and platform == 0, assign '9' (buoys)

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing platform column and 'N' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted platform column.
    """
    drifters = "7"
    buoys = "9"
    pt_col = properties.metadata_datamodels["platform"]["gdac"]

    data[pt_col] = data[pt_col].astype(object)
    data.loc[data[pt_col].isna(), pt_col] = drifters

    condition = data["N"].isna() & (data[pt_col] == 0)
    data.loc[condition, pt_col] = buoys

    return data


def deck_700_icoads(data) -> pd.DataFrame:
    """Adjust ICOADS platform codes for dataset 700.

    - Fill missing platform values with '7' (drifters)
    - Change platform from '5' to '6' (buoys) for records with:
        - ID matching 5-digit regex
        - Source ID == '147'

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing ICOADS columns for ID, source, and platform.

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted platform codes.
    """
    drifters = "7"
    sid = "147"
    pt = "5"
    buoys = "6"
    regex = re.compile(r"^\d{5,5}$")
    id_col = properties.metadata_datamodels.get("id").get("icoads")
    sid_col = properties.metadata_datamodels.get("source").get("icoads")
    pt_col = properties.metadata_datamodels.get("platform").get("icoads")

    data[pt_col] = data[pt_col].fillna(drifters)
    loc = (
        (data[id_col].str.match(regex)) & (data[sid_col] == sid) & (data[pt_col] == pt)
    )

    data[pt_col] = data[pt_col].where(~loc, buoys)
    return data


def deck_892_icoads(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust ICOADS platform codes for dataset 892.

    For records matching all of the following conditions:
      - ID column matches exactly 5 digits
      - Source ID equals '29'
      - Platform code equals '5'

    The platform code is changed to '6' (buoy).

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing ICOADS columns for ID, source, and platform.
        Column names should correspond to `properties.metadata_datamodels` entries
        (can be tuples for multi-index or strings).

    Returns
    -------
    pd.DataFrame
        DataFrame with updated platform codes where conditions are met.
        All other rows remain unchanged.

    Notes
    -----
    - Only affects rows meeting all three conditions simultaneously.
    - This function preserves the original DataFrame structure, including column
      order and type (multi-index or single-level).
    """
    sid = "29"
    pt = "5"
    buoys = "6"
    regex = re.compile(r"^\d{5,5}$")
    id_col = properties.metadata_datamodels.get("id").get("icoads")
    sid_col = properties.metadata_datamodels.get("source").get("icoads")
    pt_col = properties.metadata_datamodels.get("platform").get("icoads")

    loc = (
        (data[id_col].str.match(regex)) & (data[sid_col] == sid) & (data[pt_col] == pt)
    )
    data[pt_col] = data[pt_col].where(~loc, buoys)
    return data


def deck_792_icoads(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust ICOADS platform codes for dataset 792.

    For rows meeting all of the following conditions:
      - ID consists of digits only
      - ID length is not 5 or 7
      - ID does not start with '7'
      - Source ID equals '103'
      - Platform code equals '5'

    The platform code is updated to '6' (buoy).

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing ICOADS columns for ID, source, and platform.
        Column names should correspond to `properties.metadata_datamodels`
        (tuples for multi-index or strings).

    Returns
    -------
    pd.DataFrame
        DataFrame with updated platform codes where conditions are met.
        All other rows remain unchanged.

    Notes
    -----
    - Uses `overwrite_data` internally to modify only the specified rows.
    - Preserves the original DataFrame structure and column types.
    """
    sid = "103"
    pt = "5"
    buoys = "6"
    regex = re.compile("^[0-9]+$")
    id_col = properties.metadata_datamodels.get("id").get("icoads")
    sid_col = properties.metadata_datamodels.get("source").get("icoads")
    pt_col = properties.metadata_datamodels.get("platform").get("icoads")

    loc = (
        (data[id_col].str.match(regex))
        & (data[id_col].apply(len) != 7)
        & (data[id_col].apply(len) != 5)
        & (~data[id_col].str.startswith("7"))
        & (data[sid_col] == sid)
        & (data[pt_col] == pt)
    )
    return overwrite_data(data, loc, pt_col, buoys)


def deck_992_icoads(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust ICOADS platform codes for dataset 992:

    Rules:
    1. If ID matches regex '^6202+$', has length 7, source='114', and platform='5',
    change platform to '4' (light vessels).
    2. If ID is numeric, length not 7 or 5, does not start with '7', source='114', platform='5',
    change platform to '6' (buoys).

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with ICOADS columns for ID, source, and platform.

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted platform codes.
    """
    sid = "114"
    pt = "5"
    lv = "4"
    buoys = "6"
    regex = re.compile("^6202+$")

    id_col = properties.metadata_datamodels.get("id").get("icoads")
    sid_col = properties.metadata_datamodels.get("source").get("icoads")
    pt_col = properties.metadata_datamodels.get("platform").get("icoads")

    loc = (
        data[id_col].str.match(regex)
        & (data[id_col].str.len() == 7)
        & (data[sid_col] == sid)
        & (data[pt_col] == pt)
    )
    data = overwrite_data(data, loc, pt_col, lv)

    regex_numeric = re.compile("^[0-9]+$")
    loc = (
        data[id_col].str.match(regex_numeric)
        & (data[id_col].str.len() != 7)
        & (data[id_col].str.len() != 5)
        & (~data[id_col].str.startswith("7"))
        & (data[sid_col] == sid)
        & (data[pt_col] == pt)
    )
    return overwrite_data(data, loc, pt_col, buoys)
