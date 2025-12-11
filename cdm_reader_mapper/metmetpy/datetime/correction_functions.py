"""
metmetpy correction functions.

Created on Tue Jun 25 09:07:05 2019

@author: iregon
"""

from __future__ import annotations

import pandas as pd

from .. import properties
from . import model_datetimes


def dck_201_icoads(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust ICOADS date/time fields for dataset DCK 201:
    - If year <= 1899 and hour == 0, shift the datetime back by one day.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing ICOADS date columns (YR, MO, DY, HR),
        using multi-index or string column names as defined in properties.

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted ICOADS date/time fields.
    """
    yr_col = properties.metadata_datamodels["year"]["icoads"]
    mo_col = properties.metadata_datamodels["month"]["icoads"]
    dd_col = properties.metadata_datamodels["day"]["icoads"]
    hr_col = properties.metadata_datamodels["hour"]["icoads"]

    datetime_cols = [yr_col, mo_col, dd_col, hr_col]

    loc = (data[yr_col] <= 1899) & (data[hr_col] == 0)

    if loc.any():
        datetime_ = model_datetimes.to_datetime(data, "icoads")
        datetime_.loc[loc] = datetime_.loc[loc] - pd.Timedelta(days=1)
        data[datetime_cols] = model_datetimes.from_datetime(datetime_, "icoads")

    return data
