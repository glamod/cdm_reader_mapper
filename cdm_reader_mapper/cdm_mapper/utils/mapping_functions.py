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
}

c2k_methods = {
    "gdac": "method_b",
}

k_elements = {
    "gdac": 1,
}

tf = TimezoneFinder()


def find_entry(imodel, d):
    """Find entry in dict."""
    if not imodel:
        return
    if imodel in d.keys():
        return d[imodel]
    imodel = "_".join(imodel.split("_")[:-1])
    return find_entry(imodel, d)


def coord_360_to_180i(long3) -> float:
    """
    Convert longitudes within -180 and 180 degrees.

    Converts longitudes from degrees express in 0 to 360
    to decimal degrees between -180 to 180.
    According to
    https://confluence.ecmwf.int/pages/viewpage.action?pageId=149337515

    Parameters
    ----------
    long3: longitude or latitude in degrees

    Returns
    -------
    long1: longitude in decimal degrees
    """
    return (long3 + 180.0) % 360.0 - 180.0


def coord_dmh_to_90i(deg, min, hemis) -> float:
    """
    Convert latitudes within -90 and 90 degrees.

    Converts latitudes from degrees, minutes and hemisphere
    to decimal degrees between -90 to 90.

    Parameters
    ----------
    deg: longitude or latitude in degrees
    min: longitude or latitude in minutes
    hemis: Hemisphere N or S

    Returns
    -------
    var: latitude in decimal degrees
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


def convert_to_utc_i(date, zone) -> pd.DateTimeIndex:
    """
    Convert local time zone to utc.

    Parameters
    ----------
    date: datetime.series object
    zone: timezone as a string

    Returns
    -------
    date.time_index.obj in utc
    """
    datetime_index_aware = date.tz_localize(tz=zone)
    return datetime_index_aware.tz_convert("UTC")


def time_zone_i(lat, lon) -> str | None:
    """Return time zone."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return
    return tf.timezone_at(lng=lon, lat=lat)


def longitude_360to180_i(lon) -> int | float:
    """Convert latitudes within 1-80 and 180 degrees."""
    if lon > 180:
        return -180 + math.fmod(lon, 180)
    return lon


def location_accuracy_i(li, lat) -> int | float:
    """Calculate location accuracy."""
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


def convert_to_str(a) -> str:
    """Convert to string."""
    if a:
        a = str(a)
    return a


def string_add_i(a, b, c, sep) -> str | None:
    """Add string."""
    a = convert_to_str(a)
    b = convert_to_str(b)
    c = convert_to_str(c)
    if b:
        return sep.join(filter(None, [a, b, c]))


def to_int(value):
    """Convert value to integer if possible, return pd.NA for invalid input."""
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
    """Class for mapping Common Data Model (CDM)."""

    def __init__(self, imodel):
        self.imodel = imodel
        self.utc = datetime.timezone.utc

    def datetime_decimalhour_to_hm(self, row: pd.Series) -> pd.Series:
        """Convert datetime object to hours and minutes."""
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

    def datetime_imma1(self, df) -> pd.DateTimeIndex:  # TZ awareness?
        """Convert to pandas datetime object for IMMA1 format."""
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
        df = df.applymap(np.vectorize(to_int))

        strings = df.astype(str).apply("-".join, axis=1).values
        result = pd.to_datetime(
            strings,
            format=date_format,
            errors="coerce",
        )
        result.index = df.index
        return result

    def datetime_imma1_to_utc(self, df) -> pd.DatatimeIndex:
        """
        Convert to pandas datetime object for IMMA1 deck 701 format.
        Set missing hour to 12 and use latitude and longitude information
        to convert local midday to UTC time.
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

    def datetime_imma1_701(self, df) -> pd.DateTimeIndex:
        """Convert to pandas datetime object for IMMA1 deck 701 format."""
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

    def datetime_immt(self, df) -> pd.DatetimeIndex:
        """Convert to pandas datetime object for IMMT format."""
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
        """Get actual UTC time."""
        return datetime.datetime.now(self.utc)

    def datetime_craid(self, series, format="%Y-%m-%d %H:%M:%S.%f") -> pd.DateTimeIndex:
        """Convert string to datetime object."""
        if series.empty:
            return pd.DatetimeIndex([])
        data_1d = series.values.ravel()
        return pd.to_datetime(data_1d, format=format, errors="coerce")

    def df_col_join(self, df, sep: str) -> pd.Series:
        """Join pandas Dataframe."""
        if df.empty:
            return pd.Series([], dtype=str)

        return df.astype(str).agg(sep.join, axis=1)

    def float_opposite(self, series) -> float | pd.Series:
        """Return float opposite."""
        return -series

    def select_column(self, df) -> pd.Series:
        """Select columns."""
        if df.empty or df.shape[1] == 0:
            return pd.Series(dtype=float)

        c = df.columns.to_list()
        c.reverse()
        s = df[c[0]].copy()
        if len(c) > 1:
            for ci in c[1:]:
                s.update(df[ci])
        return s

    def float_scale(self, series, factor=1) -> pd.Series:
        """Multiply with scale factor."""
        if pd.api.types.is_numeric_dtype(series):
            return series * factor
        return pd.Series(dtype=float, name=series.name)

    def integer_to_float(self, s: pd.Series) -> pd.Series:
        """Convert integer or numeric Series to float. Non-numeric ? empty float Series."""
        if not isinstance(s, pd.Series):
            raise TypeError("integer_to_float only supports Series")

        if pd.api.types.is_numeric_dtype(s):
            return s.astype(float)

        return pd.Series(dtype=float, name=s.name)

    def icoads_wd_conversion(self, series) -> pd.Series:
        """Convert ICOADS WD."""
        series = series.mask(series == 361, 0)
        series = series.mask(series == 362, np.nan)
        return series

    def icoads_wd_integer_to_float(self, series) -> pd.Series:
        """Convert ICOADS WD integer to float."""
        s = series.copy()
        notna = s.notna()
        s.loc[notna] = self.icoads_wd_conversion(s.loc[notna])
        return self.integer_to_float(s)

    def lineage(self, df) -> str:
        """Get lineage."""
        strf = datetime.datetime.now(self.utc).strftime("%Y-%m-%d %H:%M:%S")
        imodel_lineage = find_entry(self.imodel, imodel_lineages)
        if imodel_lineage:
            strf = strf + imodel_lineage
        return strf

    def longitude_360to180(self, series) -> pd.Series:
        """Convert longitudes within -180 and 180 degrees."""
        result = np.vectorize(longitude_360to180_i, otypes="f")(series)
        return pd.Series(
            result, name=series.name, index=series.index, dtype=series.dtypes
        )

    def location_accuracy(self, df) -> pd.Series:
        """Calculate location accuracy."""
        if df.empty:
            return pd.Series([], dtype=float)

        li_array = df.iloc[:, 0]
        lat_array = df.iloc[:, 1]

        result = np.vectorize(location_accuracy_i, otypes="f")(
            li_array, lat_array
        )  # last minute tweak so that is does no fail on nans!
        return pd.Series(result, dtype=float, index=df.index)

    def observing_programme(self, series) -> pd.Series:
        """Map observing programme."""
        op = {str(i): [5, 7, 56] for i in range(0, 6)}
        op.update({"7": [5, 7, 9]})
        return series.map(op, na_action="ignore")

    def string_add(
        self, series, prepend="", append="", separator="", zfill_col=None, zfill=None
    ) -> pd.Series:
        """Add string."""
        if zfill_col and zfill:
            for col, width in zip(zfill_col, zfill):
                series.iloc[:, col] = series.iloc[:, col].astype(str).str.zfill(width)

        result = np.vectorize(string_add_i, otypes="O")(
            prepend, series, append, separator
        )

        return pd.Series(result, index=series.index, dtype="object")

    def string_join_add(
        self, df, prepend=None, append=None, separator="", zfill_col=None, zfill=None
    ) -> pd.Series:
        """Join string."""
        if zfill_col and zfill:
            for col, width in zip(zfill_col, zfill):
                df.iloc[:, col] = df.iloc[:, col].astype(str).str.zfill(width)
        joint = self.df_col_join(df, separator)
        result = np.vectorize(string_add_i, otypes="O")(
            prepend, joint, append, sep=separator
        )
        return pd.Series(result, index=df.index, dtype="object")

    def temperature_celsius_to_kelvin(self, df) -> pd.Series:
        """Convert temperature from Celsius to Kelvin."""
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

    def time_accuracy(self, series) -> pd.Series:  # ti_core
        """Calculate time accuracy."""
        # Shouldn't we use the code_table mapping for this? see CDM!
        secs = {
            "0": 3600,
            "1": int(round(3600 / 10)),
            "2": int(round(3600 / 60)),
            "3": int(round(3600 / 100)),
        }
        return series.map(secs, na_action="ignore")

    def feet_to_m(self, df) -> float:
        """Convert feet into meter."""
        df.astype(float)
        return np.round(df / 3.2808, 2)

    def gdac_uid(self, df, prepend="", append="") -> pd.DataFrame | pd.Series:
        """Generate unique UID from timestamp + ship's callsign (ID)"""
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

    def gdac_latitude(self, df) -> pd.DataFrame | pd.Series:
        """Add sign to latitude based on quadrant"""
        if "Qc" not in df.columns or "LaLaLa" not in df.columns:
            raise KeyError("DataFrame must contain 'Qc' and 'LaLaLa' columns")
        lat = df["LaLaLa"].copy()
        lat[df["Qc"].isin([3, 5])] *= -1
        return lat

    def gdac_longitude(self, df) -> pd.DataFrame | pd.Series:
        """Add sign to longitude based on quadrant"""
        if "Qc" not in df.columns or "LoLoLoLo" not in df.columns:
            raise KeyError("DataFrame must contain 'Qc' and 'LoLoLoLo' columns")
        lon = df["LoLoLoLo"].copy()
        lon[df["Qc"].isin([5, 7])] *= -1
        return lon
