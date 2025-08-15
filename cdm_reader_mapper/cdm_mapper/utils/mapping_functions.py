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
    hemisphere = 1
    min_df = min / 60
    if hemis == "S":
        hemisphere = -1
    return np.round((deg + min_df), 2) * hemisphere


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
    return tf.timezone_at(lng=lon, lat=lat)


def longitude_360to180_i(lon) -> int | float:
    """Convert latitudes within 1-80 and 180 degrees."""
    if lon > 180:
        return -180 + math.fmod(lon, 180)
    else:
        return lon


def location_accuracy_i(li, lat) -> int | np.nan:
    """Calculate location accuracy."""
    degrees = {0: 0.1, 1: 1, 4: 1 / 60, 5: 1 / 3600}
    deg_km = 111
    try:
        accuracy = degrees.get(int(li), np.nan) * math.sqrt(
            (deg_km**2) * (1 + math.cos(math.radians(lat)) ** 2)
        )
    except ValueError:
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
    if pd.isna(value):
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

    def datetime_decimalhour_to_hm(self, series) -> pd.Series:
        """Convert datetime object to hours and minutes."""
        hr = series.values[4]
        if not isinstance(hr, (int, float)):
            return series.apply(lambda x: None)
        timedelta = datetime.timedelta(hours=hr)
        seconds = timedelta.total_seconds()
        series["HR"] = int(seconds / 3600)
        series["M"] = int(seconds / 60) % 60
        return series

    def datetime_imma1(self, df) -> pd.DateTimeIndex:  # TZ awareness?
        """Convert to pandas datetime object for IMMA1 format."""
        if df.empty:
            return
        df = df.iloc[:, 0:4]
        date_format = "%Y-%m-%d-%H-%M"
        hr_ = df.columns[-1]
        df = df.assign(HR=df.iloc[:, -1])
        df["M"] = df["HR"].copy()
        df = df.drop(columns=hr_, axis=1)
        df = df.apply(lambda x: self.datetime_decimalhour_to_hm(x), axis=1)
        df = df.applymap(np.vectorize(to_int))
        strings = df.astype(str).apply("-".join, axis=1).values
        result = pd.to_datetime(
            strings,
            format=date_format,
            errors="coerce",
        )
        result.index = df.index
        return result

    def datetime_imma1_to_utc(self, df) -> pd.DataTimeIndex:
        """
        Convert to pandas datetime object for IMMA1 deck 701 format.
        Set missing hour to 12 and use latitude and longitude information
        to convert local midday to UTC time.
        """
        if df.empty:
            return

        date_format = "%Y-%m-%d-%H-%M"
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
        data = pd.to_datetime(
            df_dates.swifter.apply("-".join, axis=1).values,
            format=date_format,
            errors="coerce",
        )
        d = {"Dates": data, "Time_zone": time_zone.values}
        df_time = pd.DataFrame(data=d)

        results = df_time.swifter.apply(
            lambda x: convert_to_utc_i(x["Dates"], x["Time_zone"]), axis=1
        )
        results.index = df.index
        return results.dt.tz_convert(None)

    def datetime_imma1_701(self, df) -> pd.DateTimeIndex:
        """Convert to pandas datetime object for IMMA1 deck 701 format."""
        if df.empty:
            return
        hr = df.iloc[:, 3]
        valid_mask = hr.notna()

        results = pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns]")
        results[valid_mask] = self.datetime_imma1(df[valid_mask])
        results[~valid_mask] = self.datetime_imma1_to_utc(df[~valid_mask])
        return results

    def datetime_immt(self, df) -> pd.DatetimeIndex:
        """Convert to pandas datetime object for IMMT format."""
        date_format = "%Y-%m-%d-%H-%M"
        df["M"] = 0
        strings = df.astype(str).apply("-".join, axis=1).values
        return pd.to_datetime(
            strings,
            format=date_format,
            errors="coerce",
        )

    def datetime_utcnow(self, df) -> datetime.datetime:
        """Get actual UTC time."""
        return datetime.datetime.now(self.utc)

    def datetime_craid(self, df, format="%Y-%m-%d %H:%M:%S.%f") -> pd.DateTimeIndex:
        """Convert string to datetime object."""
        return pd.to_datetime(df.values, format=format, errors="coerce")

    def df_col_join(self, df, sep) -> str:
        """Join pandas Dataframe."""
        joint = df.iloc[:, 0].astype(str)
        for i in range(1, len(df.columns)):
            joint = joint + sep + df.iloc[:, i].astype(str)
        return joint

    def float_opposite(self, df) -> float:
        """Return float opposite."""
        return -df

    def select_column(self, df) -> pd.Series:
        """Select columns."""
        c = df.columns.to_list()
        c.reverse()
        s = df[c[0]].copy()
        if len(c) > 1:
            for ci in c[1:]:
                s.update(df[ci])
        return s

    def float_scale(self, df, factor=1) -> pd.DataFrame | pd.Series:
        """Multiply with scale factor."""
        return df * factor

    def integer_to_float(self, df) -> pd.DataFrame | pd.Series:
        """Convert integer to float."""
        return df.astype(float)

    def icoads_wd_conversion(self, df) -> pd.DataFrame | pd.Series:
        """Convert ICOADS WD."""
        df = df.mask(df == 361, 0)
        df = df.mask(df == 362, np.nan)
        return df

    def icoads_wd_integer_to_float(self, df) -> pd.DataFrame | pd.Series:
        """Convert ICOADS WD integer to float."""
        notna = df.notna()
        df[notna] = self.icoads_wd_conversion(df[notna])
        return self.integer_to_float(df)

    def lineage(self, df) -> str:
        """Get lineage."""
        strf = datetime.datetime.now(self.utc).strftime("%Y-%m-%d %H:%M:%S")
        imodel_lineage = find_entry(self.imodel, imodel_lineages)
        if imodel_lineage:
            strf = strf + imodel_lineage
        return strf

    def longitude_360to180(self, df) -> pd.DataFrame | pd.Series:
        """Convert longitudes within -180 and 180 degrees."""
        return np.vectorize(longitude_360to180_i)(df)

    def location_accuracy(self, df) -> pd.DataFrame | pd.Series:
        """Calculate location accuracy."""
        return np.vectorize(location_accuracy_i, otypes="f")(
            df.iloc[:, 0], df.iloc[:, 1]
        )  # last minute tweak so that is does no fail on nans!

    def observing_programme(self, df) -> pd.DataFrame | pd.Series:
        """Map observing programme."""
        op = {str(i): [5, 7, 56] for i in range(0, 6)}
        op.update({"7": [5, 7, 9]})
        return df.map(op, na_action="ignore")

    def string_add(
        self, df, prepend="", append="", separator="", zfill_col=None, zfill=None
    ) -> pd.DataFrame | pd.Series:
        """Add string."""
        if zfill_col and zfill:
            for col, width in zip(zfill_col, zfill):
                df.iloc[:, col] = df.iloc[:, col].astype(str).str.zfill(width)
        return np.vectorize(string_add_i)(prepend, df, append, separator)

    def string_join_add(
        self, df, prepend=None, append=None, separator="", zfill_col=None, zfill=None
    ) -> pd.DataFrame | pd.Series:
        """Join string."""
        if zfill_col and zfill:
            for col, width in zip(zfill_col, zfill):
                df.iloc[:, col] = df.iloc[:, col].astype(str).str.zfill(width)
        joint = self.df_col_join(df, separator)
        return np.vectorize(string_add_i)(prepend, joint, append, sep=separator)

    def temperature_celsius_to_kelvin(self, df) -> pd.DataFrame | pd.Series:
        """Convert temperature from Celsius to Kelvin."""
        method = find_entry(self.imodel, c2k_methods)
        if not method:
            method = "method_a"
        if method == "method_a":
            return df + 273.15
        if method == "method_b":
            df.iloc[:, 0] = np.where((df.iloc[:, 0] == 0) | (df.iloc[:, 0] == 5), 1, -1)
            return df.iloc[:, 0] * df.iloc[:, 1] + 273.15

    def time_accuracy(self, df) -> pd.DataFrame | pd.Series:  # ti_core
        """Calculate time accuracy."""
        # Shouldn't we use the code_table mapping for this? see CDM!
        secs = {
            "0": 3600,
            "1": int(round(3600 / 10)),
            "2": int(round(3600 / 60)),
            "3": int(round(3600 / 100)),
        }
        return df.map(secs, na_action="ignore")

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
