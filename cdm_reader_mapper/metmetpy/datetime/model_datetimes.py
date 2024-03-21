"""
metmetpy modelo datetime package.

Created on Wed Jul 10 09:18:41 2019

Defines the datetime field extraction or generation for data models.

Reference names of different metadata fields used in the metmetpy modules
and its location column|(section,column) in a data model are
registered in ../properties.py in metadata_datamodels.

@author: iregon
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import properties


# ---------------- General purpose functions ----------------------------------
def datetime_decimalhour_to_HM(ds):
    """DOCUMENTATiON."""
    hours = int(math.floor(ds))
    minutes = int(math.floor(60.0 * math.fmod(ds, 1)))
    return hours, minutes


# ---------------- Data model conversions -------------------------------------
def imma1(data, conversion):
    """DOCUMENTATiON."""

    def to_datetime(data):
        dt_data = data[datetime_cols]
        not_na = dt_data.notna().all(axis=1)
        date_format = "%Y-%m-%d-%H-%M"
        empty = True if len(not_na.loc[not_na]) == 0 else False
        if not empty:
            hours, minutes = np.vectorize(datetime_decimalhour_to_HM)(
                dt_data.iloc[np.where(not_na)[0], -1].values
            )
        dt_data.drop(dt_data.columns[len(dt_data.columns) - 1], axis=1, inplace=True)
        if not empty:
            dt_data.loc[not_na, "H"] = hours
            dt_data.loc[not_na, "M"] = minutes
        dt_series = pd.Series(data=pd.NaT, index=data.index)
        # The following astype chain to make sure when promotion of column from int
        # to float in the event of missing data (NaN) is reverted before conversion
        # to string   !!!!!
        if not empty:
            dt_series.loc[not_na] = pd.to_datetime(
                dt_data.loc[not_na, :]
                .astype(int)
                .astype(str)
                .apply("-".join, axis=1)
                .values,
                format=date_format,
                errors="coerce",
            )
        return dt_series

    def from_datetime(ds):
        imma1 = pd.DataFrame(index=ds.index, columns=datetime_cols)
        locs = ds.notna()
        # Note however that if there is missing data, the corresponding column
        # will be float despite the 'int' conversion
        imma1[yr_col] = ds.dt.year[locs].astype("int")
        imma1[mo_col] = ds.dt.month[locs].astype("int")
        imma1[dd_col] = ds.dt.day[locs].astype("int")
        imma1[hr_col] = ds.dt.hour[locs] + ds.dt.minute[locs] / 60
        return imma1

    yr_col = properties.metadata_datamodels.get("year").get("imma1")
    mo_col = properties.metadata_datamodels.get("month").get("imma1")
    dd_col = properties.metadata_datamodels.get("day").get("imma1")
    hr_col = properties.metadata_datamodels.get("hour").get("imma1")
    datetime_cols = [yr_col, mo_col, dd_col, hr_col]
    datetime_cols = [dt_ for dt_ in datetime_cols if dt_ in data.columns]

    if not datetime_cols:
        return pd.Series()
    elif len(datetime_cols) == 1:
        datetime_cols = datetime_cols[0]
    if conversion == "to_datetime":
        return to_datetime(data)
    elif conversion == "from_datetime":
        return from_datetime(data)
    else:
        return


# ---------------- Send input to appropriate function -------------------------
def to_datetime(data, model):
    """DOCUMENTATiON."""
    if model == "imma1":
        return imma1(data, "to_datetime")
    else:
        return


def from_datetime(data, model):
    """DOCUMENTATiON."""
    if model == "imma1":
        return imma1(data, "from_datetime")
    else:
        return
