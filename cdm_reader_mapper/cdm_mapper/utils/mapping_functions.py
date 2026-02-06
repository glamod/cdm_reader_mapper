"""
Common Data Model (CDM) mappings.

Created on Wed Apr  3 10:31:18 2019

imodel: imma1

Functions to map imodel elements to CDM elements

Main functions are those invoqued in the mappings files (table_name.json)

Main functions need to be part of class mapping_functions()

Main functions get:
    - 1 positional argument (pd.Series or pd.DataFrame with imodel data or imodel element name)
    - Optionally, keyword arguments

Main function return: pd.Series, np.array or scalars

Auxiliary functions can be used and defined in or outside class mapping_functions

@author: iregon
"""

from __future__ import annotations

import datetime
import math
import uuid
from typing import Any

import numpy as np
import pandas as pd
import swifter  # noqa
from timezonefinder import TimezoneFinder

icoads_lineage = ". Initial conversion from ICOADS R3.0.0T"
imodel_lineages = {
    "icoads": icoads_lineage,
    "icoads_r300_d714": icoads_lineage + " with supplemental data recovery",
    "icoads_r302": ". Initial conversion from ICOADS R3.0.2T NRT",
    "craid": ". Initial conversion from C-RAID",
    "marob": ". Initial conversion from DWD MAROB data base",
}

c2k_methods = {
    "gdac": "method_b",
}

k_elements = {
    "gdac": 1,
}

tf = TimezoneFinder()


def find_entry(imodel: str | None, d: dict) -> str | None:
    """
    Find entry in a dictionary, handling imodel suffix stripping.

    Parameters
    ----------
    imodel : str or None
        Imodel element name.
    d : dict
        Dictionary to search.

    Returns
    -------
    str or None
        Corresponding value if found, otherwise None.
    """
    if not imodel:
        return
    if imodel in d.keys():
        return d[imodel]
    imodel = "_".join(imodel.split("_")[:-1])
    return find_entry(imodel, d)


def coord_360_to_180i(lon: float) -> float:
    """
    Convert longitude from 0-360 to -180 to 180 degrees.

    Parameters
    ----------
    lon : float
        Longitude in degrees (0-360).

    Returns
    -------
    float
        Longitude in decimal degrees (-180 to 180).
    """
    return (lon + 180.0) % 360.0 - 180.0


def coord_dmh_to_90i(deg: float, min: float, hemis: str) -> float:
    """
    Convert latitude from degrees, minutes, hemisphere to decimal degrees.

    Parameters
    ----------
    deg : float
        Degrees.
    min : float
        Minutes (0 <= min < 60).
    hemis : str
        Hemisphere, "N" or "S".

    Returns
    -------
    float
        Latitude in decimal degrees (-90 to 90).
    """
    if hemis not in ("N", "S"):
        raise ValueError(f"Hemisphere must be 'N' or 'S' not {hemis}.")
    if not (0 <= min < 60):
        raise ValueError(f"Minutes must be between 0 and 60, not {min}.")

    abs_deg = abs(deg)
    min_df = min / 60

    decimal = abs_deg + min_df

    if hemis == "S":
        decimal *= -1
    return np.round(decimal, 2)


def convert_to_utc_i(date: pd.Series, zone: str) -> pd.DatetimeIndex:
    """
    Convert a pandas datetime series from local timezone to UTC.

    Parameters
    ----------
    date : pd.Series
        Datetime series.
    zone : str
        Timezone string.

    Returns
    -------
    pd.DatetimeIndex
        Datetime series converted to UTC.
    """
    datetime_index_aware = date.tz_localize(tz=zone)
    return datetime_index_aware.tz_convert("UTC")


def time_zone_i(lat: float, lon: float) -> str | None:
    """
    Get timezone for latitude and longitude.

    Parameters
    ----------
    lat : float
        Latitude (-90 to 90).
    lon : float
        Longitude (-180 to 180).

    Returns
    -------
    str or None
        Timezone name if available, otherwise None.
    """
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return
    return tf.timezone_at(lng=lon, lat=lat)


def longitude_360to180_i(lon: float) -> float:
    """
    Convert longitude from 0-360 to -180 to 180 degrees.

    Parameters
    ----------
    lon : float
        Longitude in degrees.

    Returns
    -------
    float
        Longitude in decimal degrees (-180 to 180).
    """
    if lon > 180:
        return -180 + math.fmod(lon, 180)
    return lon


def location_accuracy_i(li: int | float, lat: float) -> float:
    """
    Compute approximate location accuracy in km based on ICOADS code.

    Parameters
    ----------
    li : int or float
        Location index code.
    lat : float
        Latitude.

    Returns
    -------
    float
        Location accuracy in km.
    """
    degrees = {0: 0.1, 1: 1, 4: 1 / 60, 5: 1 / 3600}
    deg_km = 111
    try:
        accuracy = degrees.get(int(li), np.nan) * math.sqrt(
            (deg_km**2) * (1 + math.cos(math.radians(lat)) ** 2)
        )
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(accuracy):
        return np.nan
    return max(1, int(round(accuracy)))


def convert_to_str(a: Any) -> str | None:
    """
    Convert a value to string.

    Parameters
    ----------
    a : any
        Input value.

    Returns
    -------
    str or None
        Converted string or None if input is None or empty.
    """
    if a:
        a = str(a)
    return a


def string_add_i(a: Any, b: Any, c: Any, sep: str) -> str | None:
    """
    Concatenate strings a, b, c with separator, ignoring None values.

    Parameters
    ----------
    a, b, c : any
        Input values.
    sep : str
        Separator string.

    Returns
    -------
    str or None
        Concatenated string.
    """
    a = convert_to_str(a)
    b = convert_to_str(b)
    c = convert_to_str(c)
    if b:
        return sep.join(filter(None, [a, b, c]))


def to_int(value: Any) -> int | pd.NA:
    """
    Convert a value to integer, return pd.NA for invalid input.

    Parameters
    ----------
    value : any
        Input value.

    Returns
    -------
    int or pd.NA
        Converted integer or NA if invalid.
    """
    try:
        if pd.isna(value):
            return pd.NA
    except ValueError:
        return pd.NA
    try:
        return int(value)
    except (TypeError, ValueError):
        return pd.NA


class mapping_functions:
    """Class for mapping Common Data Model (CDM) elements from IMMA1, GDAC, ICOADS, C-RAID, and IMMT datasets."""

    def __init__(self, imodel):
        self.imodel = imodel
        self.utc = datetime.timezone.utc

    def datetime_decimalhour_to_hm(self, row: pd.Series) -> pd.Series:
        """
        Convert a decimal hour to hours and minutes.

        Parameters
        ----------
        row : pd.Series
            A Series containing a decimal hour value at index 4.

        Returns
        -------
        pd.Series
            A Series with 'HR' (hour) and 'M' (minute).
        """
        try:
            hr = row.values[4]
        except IndexError:
            return pd.Series({"HR": None, "M": None})

        if hr is None or pd.isna(hr) or not np.issubdtype(type(hr), np.number):
            return pd.Series({"HR": None, "M": None})

        total_seconds = float(hr) * 3600
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)

        return pd.Series({"HR": hours, "M": minutes})

    def datetime_imma1(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """
        Convert IMMA1 dataset to pandas datetime object.

        Parameters
        ----------
        df : pd.DataFrame
            IMMA1 dataset with columns for year, month, day, and decimal hour.

        Returns
        -------
        pd.DatetimeIndex
            DatetimeIndex of converted timestamps.
        """
        if df.empty:
            return pd.DatetimeIndex([])

        df = df.iloc[:, 0:4]
        date_format = "%Y-%m-%d-%H-%M"
        hr_ = df.columns[-1]
        df = df.assign(HR=df.iloc[:, -1])
        df["M"] = df["HR"].copy()
        df = df.drop(columns=hr_, axis=1)

        hr_min = df.apply(lambda x: self.datetime_decimalhour_to_hm(x), axis=1)
        df["HR"] = hr_min["HR"]
        df["M"] = hr_min["M"]
        df = df.apply(lambda col: col.map(to_int))

        strings = df.astype(str).apply("-".join, axis=1).values
        result = pd.to_datetime(
            strings,
            format=date_format,
            errors="coerce",
        )
        result.index = df.index
        return result

    def datetime_imma1_to_utc(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """
        Convert to pandas datetime object for IMMA1 deck 701 format.
        Set missing hour to 12 and use latitude and longitude information
        to convert local midday to UTC time.

        Parameters
        ----------
        df : pd.DataFrame
            IMMA1 deck 701 dataset containing year, month, day, latitude, and longitude.

        Returns
        -------
        pd.DatetimeIndex
            DatetimeIndex with timestamps converted to UTC.
        """
        if df.empty:
            return pd.DatetimeIndex([])

        date_format = "%Y-%m-%d-%H-%M"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[1] for col in df.columns]

        df_dates = df.iloc[:, 0:3].astype(str)
        df_dates["HR"] = "12"
        df_dates["M"] = "0"

        df_coords = df.iloc[:, 4:6].astype(float)
        lon_ = df_coords.columns[0]
        lat_ = df_coords.columns[1]
        df_coords["lon_converted"] = coord_360_to_180i(df_coords[lon_])

        time_zone = df_coords.swifter.apply(
            lambda x: time_zone_i(x[lat_], x["lon_converted"]),
            axis=1,
        )

        strings = df_dates.swifter.apply("-".join, axis=1).values
        data = pd.to_datetime(strings, format=date_format, errors="coerce")
        df_time = pd.DataFrame(data={"Dates": data, "Time_zone": time_zone.values})

        results = df_time.swifter.apply(
            lambda x: convert_to_utc_i(x["Dates"], x["Time_zone"]), axis=1
        )
        results.index = df.index
        return pd.DatetimeIndex(results.dt.tz_convert(None))

    def datetime_imma1_701(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """
        Convert IMMA1 deck 701 dataset to pandas datetime object with UTC fallback.

        Parameters
        ----------
        df : pd.DataFrame
            IMMA1 deck 701 dataset with columns for date and time.

        Returns
        -------
        pd.DatetimeIndex
            DatetimeIndex with converted timestamps.
        """
        if df.empty:
            return pd.DatetimeIndex([])

        hr = df.iloc[:, 3]
        valid_mask = hr.notna()

        results = pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns]")

        if valid_mask.any():
            results[valid_mask] = self.datetime_imma1(df[valid_mask])

        if (~valid_mask).any():
            results[~valid_mask] = self.datetime_imma1_to_utc(df[~valid_mask])

        return pd.DatetimeIndex(results)

    def datetime_immt(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """
        Convert IMMT dataset to pandas datetime object.

        Parameters
        ----------
        df : pd.DataFrame
            IMMT dataset containing year, month, day, hour.

        Returns
        -------
        pd.DatetimeIndex
            DatetimeIndex of converted timestamps.
        """
        if df.empty:
            return pd.DatetimeIndex([])

        date_format = "%Y-%m-%d-%H-%M"
        df = df.copy()
        df["M"] = 0

        strings = df.astype(str).apply("-".join, axis=1).values
        result = pd.to_datetime(
            strings,
            format=date_format,
            errors="coerce",
        )
        return pd.DatetimeIndex(result)

    def datetime_utcnow(self, df) -> datetime.datetime:
        """
        Return the current UTC datetime.

        Parameters
        ----------
        df : pd.DataFrame
            Ignored. Present for API consistency.

        Returns
        -------
        datetime.datetime
            Current UTC datetime.
        """
        return datetime.datetime.now(self.utc)

    def datetime_craid(
        self, series: pd.Series, format: str = "%Y-%m-%d %H:%M:%S.%f"
    ) -> pd.DatetimeIndex:
        """
        Convert C-RAID date strings to pandas datetime.

        Parameters
        ----------
        series : pd.Series
            Series of date strings.
        format : str, optional
            Datetime format string (default: "%Y-%m-%d %H:%M:%S.%f").

        Returns
        -------
        pd.DatetimeIndex
            DatetimeIndex of converted dates.
        """
        if series.empty:
            return pd.DatetimeIndex([])
        data_1d = series.values.ravel()
        return pd.to_datetime(data_1d, format=format, errors="coerce")

    def df_col_join(self, df: pd.DataFrame, sep: str) -> pd.Series:
        """
        Join all columns of a pandas DataFrame into a single Series of strings.

        Parameters
        ----------
        df : pd.DataFrame
          Input DataFrame.
        sep : str
          Separator to use between column values.

        Returns
        -------
        pd.Series
          Series with joined string values from each row.
        """
        if df.empty:
            return pd.Series([], dtype=str)

        return df.astype(str).agg(sep.join, axis=1)

    def float_opposite(self, series: pd.Series) -> pd.Series:
        """
        Return the opposite (negation) of a numeric Series.

        Parameters
        ----------
        series : pd.Series
          Input numeric Series.

        Returns
        -------
        pd.Series
          Series with negated values.
        """
        return -series

    def select_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Select the last column with non-null values, prioritizing the rightmost column.

        Parameters
        ----------
        df : pd.DataFrame
          Input DataFrame.

        Returns
        -------
        pd.Series
          Series with selected column values.
        """
        if df.empty or df.shape[1] == 0:
            return pd.Series(dtype=float)

        c = df.columns.to_list()
        c.reverse()
        s = df[c[0]].copy()
        if len(c) > 1:
            for ci in c[1:]:
                s.update(df[ci])
        return s

    def float_scale(self, series: pd.Series, factor: float = 1) -> pd.Series:
        """
        Multiply a numeric Series by a scale factor.

        Parameters
        ----------
        series : pd.Series
          Numeric Series to scale.
        factor : float, default=1
          Scale factor to multiply by.

        Returns
        -------
        pd.Series
          Scaled Series, or empty float Series if input is non-numeric.
        """
        if pd.api.types.is_numeric_dtype(series):
            return series * factor
        return pd.Series(dtype=float, name=series.name)

    def integer_to_float(self, s: pd.Series) -> pd.Series:
        """
        Convert a numeric or integer Series to float. Non-numeric Series returns empty float Series.

        Parameters
        ----------
        s : pd.Series
            Input Series.

        Returns
        -------
        pd.Series
          Float Series.

        Raises
        ------
        TypeError
          If input is not a pandas Series.
        """
        if not isinstance(s, pd.Series):
            raise TypeError("integer_to_float only supports Series")

        return s.astype(float)

    def icoads_wd_conversion(self, series: pd.Series) -> pd.Series:
        """
        Convert ICOADS wind direction codes.

        Codes 361 -> 0, 362 -> NaN.

        Parameters
        ----------
        series : pd.Series
          Input ICOADS wind direction Series.

        Returns
        -------
        pd.Series
          Converted wind direction Series.
        """
        series = series.mask(series == 361, 0)
        series = series.mask(series == 362, np.nan)
        return series

    def icoads_wd_integer_to_float(self, series: pd.Series) -> pd.Series:
        """
        Convert ICOADS wind direction integer Series to float, applying conversion rules.

        Parameters
        ----------
        series : pd.Series
          ICOADS wind direction integer Series.

        Returns
        -------
        pd.Series
          Float wind direction Series.
        """
        s = series.copy()
        notna = s.notna()
        s.loc[notna] = self.icoads_wd_conversion(s.loc[notna])
        return self.integer_to_float(s)

    def lineage(self, df: pd.DataFrame) -> str:
        """
        Get the lineage string for a dataset, combining timestamp and model lineage.

        Parameters
        ----------
        df : pd.DataFrame
          Input dataset (used for context, not data manipulation).

        Returns
        -------
        str
          Lineage string including timestamp and imodel entry.
        """
        strf = datetime.datetime.now(self.utc).strftime("%Y-%m-%d %H:%M:%S")
        imodel_lineage = find_entry(self.imodel, imodel_lineages)
        if imodel_lineage:
            strf = strf + imodel_lineage
        return strf

    def longitude_360to180(self, series: pd.Series) -> pd.Series:
        """
        Convert longitudes from 0-360 to -180 to 180 range.

        Parameters
        ----------
        series : pd.Series
          Input longitude Series.

        Returns
        -------
        pd.Series
          Converted longitude Series.
        """
        result = np.vectorize(longitude_360to180_i, otypes="f")(series)
        return pd.Series(
            result, name=series.name, index=series.index, dtype=series.dtypes
        )

    def location_accuracy(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute location accuracy based on two columns (li_array, lat_array).

        Parameters
        ----------
        df : pd.DataFrame
          Input DataFrame with at least two columns.

        Returns
        -------
        pd.Series
          Series of location accuracy values.
        """
        if df.empty:
            return pd.Series([], dtype=float)

        li_array = df.iloc[:, 0]
        lat_array = df.iloc[:, 1]

        result = np.vectorize(location_accuracy_i, otypes="f")(
            li_array, lat_array
        )  # last minute tweak so that is does no fail on nans!
        return pd.Series(result, dtype=float, index=df.index)

    def observing_programme(self, series: pd.Series) -> pd.Series:
        """
        Map observing programme codes to lists.

        Parameters
        ----------
        series : pd.Series
          Series of programme codes (string or int).

        Returns
        -------
        pd.Series
          Series of mapped observing programme lists.
        """
        op = {str(i): [5, 7, 56] for i in range(0, 6)}
        op.update({"7": [5, 7, 9]})
        return series.map(op, na_action="ignore")

    def string_add(
        self,
        series: pd.Series,
        prepend: str = "",
        append: str = "",
        separator: str = "",
        zfill_col: list = None,
        zfill: list = None,
    ) -> pd.Series:
        """
        Add strings to Series elements with optional zero-fill.

        Parameters
        ----------
        series : pd.Series
          Series to modify.
        prepend : str, default=""
          String to prepend.
        append : str, default=""
          String to append.
        separator : str, default=""
          Separator between series values.
        zfill_col : list, optional
          Columns to zero-fill.
        zfill : list, optional
          Widths for zero-fill.

        Returns
        -------
        pd.Series
          Series with modified string values.
        """
        if zfill_col and zfill:
            for col, width in zip(zfill_col, zfill):
                series.iloc[:, col] = series.iloc[:, col].astype(str).str.zfill(width)

        result = np.vectorize(string_add_i, otypes="O")(
            prepend, series, append, separator
        )

        return pd.Series(result, index=series.index, dtype="object")

    def string_join_add(
        self,
        df: pd.DataFrame,
        prepend=None,
        append=None,
        separator: str = "",
        zfill_col: list = None,
        zfill: list = None,
    ) -> pd.Series:
        """
        Join DataFrame columns into a single string and optionally prepend/append strings.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with string or numeric columns.
        prepend : str or None, optional
            String to prepend to each joined value, by default None.
        append : str or None, optional
            String to append to each joined value, by default None.
        separator : str, default=""
            Separator to use when joining columns.
        zfill_col : list, optional
            List of column indices to apply zero-fill.
        zfill : list, optional
            List of widths for zero-fill, corresponding to zfill_col.

        Returns
        -------
        pd.Series
            Series of joined and modified strings.
        """
        df = df.copy()
        if zfill_col and zfill:
            for col, width in zip(zfill_col, zfill):
                column_name = df.columns[col]
                df[column_name] = df[column_name].astype("object")
                df[column_name] = df[column_name].astype(str).str.zfill(width)

        joint = self.df_col_join(df, separator)
        result = np.vectorize(string_add_i, otypes="O")(
            prepend, joint, append, sep=separator
        )
        return pd.Series(result, index=df.index, dtype="object")

    def temperature_celsius_to_kelvin(self, df: pd.DataFrame) -> pd.Series:
        """
        Convert temperatures from Celsius to Kelvin using the model-specific method.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with temperature data.

        Returns
        -------
        pd.Series
            Series of temperatures in Kelvin.
        """
        method = find_entry(self.imodel, c2k_methods)
        if not method:
            method = "method_a"
        if method == "method_a":
            result = df + 273.15
        if method == "method_b":
            df.iloc[:, 0] = np.where((df.iloc[:, 0] == 0) | (df.iloc[:, 0] == 5), 1, -1)
            result = df.iloc[:, 0] * df.iloc[:, 1] + 273.15

        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
        return pd.Series(result, dtype=float)

    def time_accuracy(self, series: pd.Series) -> pd.Series:
        """
        Map time accuracy codes to seconds.

        Parameters
        ----------
        series : pd.Series
            Series of time accuracy codes as strings.

        Returns
        -------
        pd.Series
            Series with time accuracy in seconds.
        """
        # Shouldn't we use the code_table mapping for this? see CDM!
        secs = {
            "0": 3600,
            "1": int(round(3600 / 10)),
            "2": int(round(3600 / 60)),
            "3": int(round(3600 / 100)),
        }
        return series.map(secs, na_action="ignore")

    def feet_to_m(self, series: pd.Series) -> pd.Series:
        """
        Convert values from feet to meters.

        Parameters
        ----------
        series : pd.Series
            Series of values in feet.

        Returns
        -------
        pd.Series
            Series of values in meters, rounded to 2 decimals.
        """
        series = series.astype(float)
        return np.round(series / 3.2808, 2)

    def gdac_uid(
        self, df: pd.DataFrame, prepend: str = "", append: str = ""
    ) -> pd.Series:
        """
        Generate a unique UID based on timestamp and ship's callsign (ID).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with columns 'AAAA', 'MM', 'YY', 'GG'.
        prepend : str, default=""
            String to prepend to UID.
        append : str, default=""
            String to append to UID.

        Returns
        -------
        pd.Series
            Series of generated unique IDs.
        """
        if df.empty:
            return pd.Series([], dtype="object")

        df = df.copy()
        df["AAAA"] = df["AAAA"].apply(lambda x: f"{x:04d}")
        df["MM"] = df["MM"].apply(lambda x: f"{x:02d}")
        df["YY"] = df["YY"].apply(lambda x: f"{x:02d}")
        df["GG"] = df["GG"].astype("int64").apply(lambda x: f"{x:02d}")
        name = df.apply(lambda x: "".join(x), axis=1)
        uid = np.empty(np.shape(df["AAAA"]), dtype="U126")
        for i, n in enumerate(name):
            uid[i] = (
                str(prepend) + uuid.uuid5(uuid.NAMESPACE_OID, str(n)).hex + str(append)
            )
        df["UUID"] = uid
        return df["UUID"]

    def gdac_latitude(self, df: pd.DataFrame) -> pd.Series:
        """
        Adjust latitude sign based on quadrant.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with columns 'Qc' and 'LaLaLa'.

        Returns
        -------
        pd.Series
            Series of latitude values with adjusted sign.

        Raises
        ------
        KeyError
            If required columns are missing.
        """
        if "Qc" not in df.columns or "LaLaLa" not in df.columns:
            raise KeyError("DataFrame must contain 'Qc' and 'LaLaLa' columns")
        lat = df["LaLaLa"].copy()
        lat[df["Qc"].isin([3, 5])] *= -1
        return lat

    def gdac_longitude(self, df: pd.DataFrame) -> pd.Series:
        """
        Adjust longitude sign based on quadrant.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with columns 'Qc' and 'LoLoLoLo'.

        Returns
        -------
        pd.Series
            Series of longitude values with adjusted sign.

        Raises
        ------
        KeyError
            If required columns are missing.
        """
        if "Qc" not in df.columns or "LoLoLoLo" not in df.columns:
            raise KeyError("DataFrame must contain 'Qc' and 'LoLoLoLo' columns")
        lon = df["LoLoLoLo"].copy()
        lon[df["Qc"].isin([5, 7])] *= -1
        return lon
