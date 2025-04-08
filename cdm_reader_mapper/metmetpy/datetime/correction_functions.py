"""
metmetpy correction functions.

Created on Tue Jun 25 09:07:05 2019

@author: iregon
"""

from __future__ import annotations

import pandas as pd

from .. import properties
from . import model_datetimes


def dck_201_icoads(data) -> pd.DataFrame:
    """DOCUMENTATION."""
    yr_col = properties.metadata_datamodels.get("year").get("icoads")
    mo_col = properties.metadata_datamodels.get("month").get("icoads")
    dd_col = properties.metadata_datamodels.get("day").get("icoads")
    hr_col = properties.metadata_datamodels.get("hour").get("icoads")

    datetime_cols = [yr_col, mo_col, dd_col, hr_col]
    loc = (data[yr_col] <= 1899) & (data[hr_col] == 0)

    if len(data.loc[loc]) > 0:
        datetime_ = model_datetimes.to_datetime(data, "icoads")
        datetime_[loc] = datetime_[loc] + pd.DateOffset(days=-1)
        data[datetime_cols] = model_datetimes.from_datetime(datetime_, "icoads")
    return data
