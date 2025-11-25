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


def datetime_decimalhour_to_hm(decimal_hours: float) -> tuple[int, int]:
    """
    Convert a decimal-hour value (e.g., 12.5) to (hours, minutes).

    Parameters
    ----------
    decimal_hours : float
        Decimal hour value.

    Returns
    -------
    tuple[int, int]
        Integer hours and minutes.
    """
    decimal_hours = float(decimal_hours)
    hours = int(math.floor(decimal_hours))
    minutes = int(math.floor(60.0 * math.fmod(decimal_hours, 1)))
    return hours, minutes


def icoads(data: pd.DataFrame | pd.Series, conversion: str) -> pd.DataFrame | pd.Series:
    """
    Convert ICOADS date/time fields between DataFrame representation
    and pandas datetime Series.

    Parameters
    ----------
    data : DataFrame or Series
        For conversion="to_datetime": a DataFrame containing the ICOADS
        date fields YR, MO, DY, HR (can be strings or tuple column names).
        For conversion="from_datetime": a Series of datetime64 values.

    conversion : {"to_datetime", "from_datetime"}
        Direction of conversion.

    Returns
    -------
    Series or DataFrame
        Converted date values.
    """
    yr_col = properties.metadata_datamodels["year"]["icoads"]
    mo_col = properties.metadata_datamodels["month"]["icoads"]
    dd_col = properties.metadata_datamodels["day"]["icoads"]
    hr_col = properties.metadata_datamodels["hour"]["icoads"]

    datetime_cols = [yr_col, mo_col, dd_col, hr_col]
    if isinstance(data, pd.DataFrame):
        datetime_cols = [c for c in datetime_cols if c in data.columns]

    def to_datetime(df: pd.DataFrame) -> pd.Series:
        if not datetime_cols:
            return pd.Series(dtype="datetime64[ns]")

        dt_data = df[datetime_cols].copy()
        valid = dt_data.notna().all(axis=1)

        out = pd.Series(pd.NaT, index=df.index)
        if not valid.any():
            return out

        hour_col = datetime_cols[-1]
        hours, minutes = np.vectorize(datetime_decimalhour_to_hm)(
            dt_data.loc[valid, hour_col].values
        )

        dt_data = dt_data.drop(columns=[hour_col])
        dt_data.loc[valid, "H"] = hours
        dt_data.loc[valid, "M"] = minutes

        strings = dt_data.loc[valid].astype(int).astype(str).apply("-".join, axis=1)

        out.loc[valid] = pd.to_datetime(
            strings, format="%Y-%m-%d-%H-%M", errors="coerce"
        )
        return out

    def from_datetime(ds: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(index=ds.index, columns=datetime_cols)
        df.columns = pd.MultiIndex.from_tuples(datetime_cols)
        valid = ds.notna()

        df.loc[valid, yr_col] = ds.dt.year[valid].astype(int)
        df.loc[valid, mo_col] = ds.dt.month[valid].astype(int)
        df.loc[valid, dd_col] = ds.dt.day[valid].astype(int)
        df.loc[valid, hr_col] = ds.dt.hour[valid] + ds.dt.minute[valid] / 60

        return df

    if conversion == "to_datetime":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("to_datetime requires a DataFrame")
        return to_datetime(data)

    if conversion == "from_datetime":
        if not isinstance(data, pd.Series):
            raise TypeError("from_datetime requires a Series")
        return from_datetime(data)

    raise ValueError("conversion must be one of {'to_datetime','from_datetime'}")


def to_datetime(data: pd.DataFrame, model: str = "icoads") -> pd.Series:
    """Dispatch conversion to datetime according to model."""
    if model == "icoads":
        return icoads(data, "to_datetime")
    raise ValueError(f"Unknown model: {model}")


def from_datetime(data: pd.Series, model: str = "icoads") -> pd.DataFrame:
    """Dispatch conversion from datetime according to model."""
    if model == "icoads":
        return icoads(data, "from_datetime")
    raise ValueError(f"Unknown model: {model}")
