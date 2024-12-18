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
    "gcc": "method_b",
}

k_elements = {
    "gcc": 1,
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


def coord_360_to_180i(long3):
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


def coord_dmh_to_90i(deg, min, hemis):
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


def convert_to_utc_i(date, zone):
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


def time_zone_i(lat, lon):
    """Return time zone."""
    return tf.timezone_at(lng=lon, lat=lat)


def longitude_360to180_i(lon):
    """Convert latitudes within 1-80 and 180 degrees."""
    if lon > 180:
        return -180 + math.fmod(lon, 180)
    else:
        return lon


def location_accuracy_i(li, lat):
    """Calculate location accuracy."""
    degrees = {0: 0.1, 1: 1, 4: 1 / 60, 5: 1 / 3600}
    deg_km = 111
    accuracy = degrees.get(int(li), np.nan) * math.sqrt(
        (deg_km**2) * (1 + math.cos(math.radians(lat)) ** 2)
    )
    return np.nan if np.isnan(accuracy) else max(1, int(round(accuracy)))


def convert_to_str(a):
    """Convert to string."""
    if a:
        a = str(a)
    return a


def string_add_i(a, b, c, sep):
    """Add string."""
    a = convert_to_str(a)
    b = convert_to_str(b)
    c = convert_to_str(c)
    if b:
        return sep.join(filter(None, [a, b, c]))


class mapping_functions:
    """Class for mapping Common Data Model (CDM)."""

    def __init__(self, imodel):
        self.imodel = imodel
        self.utc = datetime.timezone.utc

    def datetime_decimalhour_to_hm(self, df):
        """Convert datetime object to hours and minutes."""
        hr = df.values[4]
        if not isinstance(hr, (int, float)):
            df = df.apply(lambda x: None)
            return df
        timedelta = datetime.timedelta(hours=hr)
        seconds = timedelta.total_seconds()
        df["HR"] = int(seconds / 3600)
        df["M"] = int(seconds / 60) % 60
        return df

    def datetime_imma1(self, df):  # TZ awareness?
        """Convert to pandas datetime object."""
        date_format = "%Y-%m-%d-%H-%M"
        hr_ = df.columns[-1]
        df = df.assign(HR=df.iloc[:, -1])
        df["M"] = df["HR"].copy()
        df = df.drop(columns=hr_, axis=1)
        df = df.apply(lambda x: self.datetime_decimalhour_to_hm(x), axis=1)
        df = df.astype(int, errors="ignore")
        return pd.to_datetime(
            df.astype(str).apply("-".join, axis=1).values,
            format=date_format,
            errors="coerce",
        )

    def datetime_utcnow(self, df):
        """Get actual UTC time."""
        return datetime.datetime.now(self.utc)

    def datetime_craid(self, df, format="%Y-%m-%d %H:%M:%S.%f"):
        """Convert string to datetime object."""
        return pd.to_datetime(df.values, format=format, errors="coerce")

    def datetime_to_cdm_time(self, df):
        """
        Convert time object to datetime object.

        Converts year, month, day and time indicator to
        a datetime obj with a 24hrs format '%Y-%m-%d-%H'

        Parameters
        ----------
        dates: list of elements from a date array

        Returns
        -------
        date: datetime obj
        """
        df = df["core"].dropna(how="any")
        date_format = "%Y-%m-%d-%H-%M"

        df_dates = df.iloc[:, 0:3].astype(str)
        df_dates["H"] = "12"
        df_dates["M"] = "0"
        df_coords = df.iloc[:, 3:5].astype(float)

        # Convert long to -180 to 180 for time zone finding
        df_coords["lon_converted"] = coord_360_to_180i(df_coords["LON"])
        time_zone = df_coords.swifter.apply(
            lambda x: time_zone_i(x["LAT"], x["lon_converted"]),
            axis=1,
        )

        data = pd.to_datetime(
            df_dates.swifter.apply("-".join, axis=1).values,
            format=date_format,
            errors="coerce",
        )

        d = {"Dates": data, "Time_zone": time_zone.values}
        df_time = pd.DataFrame(data=d)

        return df_time.swifter.apply(
            lambda x: convert_to_utc_i(x["Dates"], x["Time_zone"]), axis=1
        )

    def df_col_join(self, df, sep):
        """Join pandas Dataframe."""
        joint = df.iloc[:, 0].astype(str)
        for i in range(1, len(df.columns)):
            joint = joint + sep + df.iloc[:, i].astype(str)
        return joint

    def float_opposite(self, df):
        """Return float opposite."""
        return -df

    def select_column(self, df):
        """Select columns."""
        c = df.columns.to_list()
        c.reverse()
        s = df[c[0]].copy()
        if len(c) > 1:
            for ci in c[1:]:
                s.update(df[ci])
        return s

    def float_scale(self, df, factor=1):
        """Multiply with scale factor."""
        return df * factor

    def integer_to_float(self, df):
        """Convert integer to float."""
        return df.astype(float)

    def icoads_wd_conversion(self, df):
        """Convert ICOADS WD."""
        df = df.mask(df == 361, 0)
        df = df.mask(df == 362, np.nan)
        return df

    def icoads_wd_integer_to_float(self, df):
        """Convert ICOADS WD integer to float."""
        notna = df.notna()
        df[notna] = self.icoads_wd_conversion(df[notna])
        return self.integer_to_float(df)

    def lineage(self, df):
        """Get lineage."""
        strf = datetime.datetime.now(self.utc).strftime("%Y-%m-%d %H:%M:%S")
        imodel_lineage = find_entry(self.imodel, imodel_lineages)
        if imodel_lineage:
            strf = strf + imodel_lineage
        return strf

    def longitude_360to180(self, df):
        """Convert longitudes within -180 and 180 degrees."""
        return np.vectorize(longitude_360to180_i)(df)

    def location_accuracy(self, df):  # (li_core,lat_core) math.radians(lat_core)
        """Calculate location accuracy."""
        return np.vectorize(location_accuracy_i, otypes="f")(
            df.iloc[:, 0], df.iloc[:, 1]
        )  # last minute tweak so that is does no fail on nans!

    def observing_programme(self, df):
        """Map observing programme."""
        op = {str(i): [5, 7, 56] for i in range(0, 6)}
        op.update({"7": [5, 7, 9]})
        return df.map(op, na_action="ignore")

    def string_add(
        self, df, prepend="", append="", separator="", zfill_col=None, zfill=None
    ):
        """Add string."""
        if zfill_col and zfill:
            for col, width in zip(zfill_col, zfill):
                df.iloc[:, col] = df.iloc[:, col].astype(str).str.zfill(width)
        return np.vectorize(string_add_i)(prepend, df, append, separator)

    def string_join_add(
        self, df, prepend=None, append=None, separator="", zfill_col=None, zfill=None
    ):
        """Join string."""
        if zfill_col and zfill:
            for col, width in zip(zfill_col, zfill):
                df.iloc[:, col] = df.iloc[:, col].astype(str).str.zfill(width)
        joint = self.df_col_join(df, separator)
        return np.vectorize(string_add_i)(prepend, joint, append, sep=separator)

    def temperature_celsius_to_kelvin(self, df):
        """Convert temperature from degrre Ceslius to Kelvin."""
        method = find_entry(self.imodel, c2k_methods)
        if not method:
            method = "method_a"
        if method == "method_a":
            return df + 273.15
        if method == "method_b":
            df.iloc[:, 0] = np.where((df.iloc[:, 0] == 0) | (df.iloc[:, 0] == 5), 1, -1)
            return df.iloc[:, 0] * df.iloc[:, 1] + 273.15

    def time_accuracy(self, df):  # ti_core
        """Calculate time accuracy."""
        # Shouldn't we use the code_table mapping for this? see CDM!
        secs = {
            "0": 3600,
            "1": int(round(3600 / 10)),
            "2": int(round(3600 / 60)),
            "3": int(round(3600 / 100)),
        }
        return df.map(secs, na_action="ignore")

    def feet_to_m(self, df):
        """Convert feet into meter."""
        df.astype(float)
        return np.round(df / 3.2808, 2)

    def guid(self, df, prepend="", append=""):
        """DOCUMENTATION."""
        df = df.copy()
        df["YR"] = df["YR"].apply(lambda x: f"{x:04d}")
        df["MO"] = df["MO"].apply(lambda x: f"{x:02d}")
        df["DY"] = df["DY"].apply(lambda x: f"{x:02d}")
        df["GG"] = df["GG"].astype("int64").apply(lambda x: f"{x:02d}")
        name = df.apply(lambda x: "".join(x), axis=1)
        uid = np.empty(np.shape(df["YR"]), dtype="U126")
        for i, n in enumerate(name):
            uid[i] = (
                str(prepend) + uuid.uuid5(uuid.NAMESPACE_OID, str(n)).hex + str(append)
            )
        df["UUID"] = uid
        return df["UUID"]
