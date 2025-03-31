"""Convert dtypes."""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd


def convert_integer(data, null_label) -> pd.Series:
    """
    Convert all elements that have 'int' as type attribute.

    Parameters
    ----------
    data: data tables to convert
    null_label: specified how nan are represented

    Returns
    -------
    Series
        Data as int type.
    """

    def _return_str(x, null_label):
        if pd.isna(x):
            return null_label
        try:
            return str(int(float(x)))
        except ValueError:
            return null_label

    return data.apply(lambda x: _return_str(x, null_label))


def convert_float(data, null_label, decimal_places) -> pd.Series:
    """
    Convert all elements that have 'float' as type attribute.

    Parameters
    ----------
    data: data tables to convert
    null_label: specified how nan are represented
    decimal_places: number of decimal places

    Returns
    -------
    Series
        Data as float type.
    """

    def _return_str(x, null_label, format_float):
        if pd.isna(x):
            return null_label
        try:
            return format_float.format(float(x))
        except ValueError:
            return null_label

    format_float = "{:." + str(decimal_places) + "f}"
    return data.apply(lambda x: _return_str(x, null_label, format_float))


def convert_datetime(data, null_label) -> pd.Series:
    """
    Convert datetime objects in the format: "%Y-%m-%d %H:%M:%S".

    Parameters
    ----------
    data: date time elements
    null_label: specified how nan are represented

    Returns
    -------
    Series
        Data as datetime objects.
    """

    def _return_str(x, null_label):
        if pd.isna(x):
            return null_label
        if isinstance(x, str):
            return x
        return x.strftime("%Y-%m-%d %H:%M:%S")

    return data.apply(lambda x: _return_str(x, null_label))


def convert_str(data, null_label) -> pd.Series:
    """
    Convert string elements.

    Parameters
    ----------
    data: data tables to convert
    null_label: specified how nan are represented

    Returns
    -------
    Series
        Data as string objects.
    """

    def _return_str(x, null_label):
        if isinstance(x, list):
            return str(x)
        if pd.isna(x):
            return null_label
        return str(x)

    return data.apply(lambda x: _return_str(x, null_label))


def convert_integer_array(data, null_label) -> pd.Series:
    """
    Convert a series of integer objects as array.

    Parameters
    ----------
    data: data tables to convert
    null_label: specified how nan are represented

    Returns
    -------
    Series
       Data as array of int objects.
    """
    return data.apply(convert_integer_array_i, null_label=null_label)


def convert_str_array(data, null_label) -> pd.Series:
    """
    Convert a series of string objects as array.

    Parameters
    ----------
    data: data tables to convert
    null_label: specified how nan are represented

    Returns
    -------
    Series
        Data as array of str objects.
    """
    return data.apply(convert_str_array_i)


def convert_integer_array_i(row, null_label=None) -> str | None:
    """
    Convert a series of integer objects.

    Parameters
    ----------
    row
    null_label

    Returns
    -------
    str or null_label
        List of integers as string array or null_label if list of integers is empty.
    """

    def _return_str(x):
        if x is None:
            return x
        if np.isfinite(x):
            return str(int(x))

    row = row if not isinstance(row, str) else ast.literal_eval(row)
    row = row if isinstance(row, list) else [row]

    str_row = [_return_str(x) for x in row]
    string = ",".join(filter(bool, str_row))
    if len(string) > 0:
        return "{" + string + "}"
    return null_label


def convert_str_array_i(row, null_label=None) -> str | None:
    """
    Convert a series of string objects.

    Parameters
    ----------
    row
    null_label

    Returns
    -------
    str
        List of strings as string array or null_label if list of strings is empty.
    """

    def _return_str(x):
        if x is None:
            return x
        if np.isfinite(x):
            return str(x)

    row = row if not isinstance(row, str) else ast.literal_eval(row)
    row = row if isinstance(row, list) else [row]
    str_row = [_return_str(x) for x in row]
    string = ",".join(filter(bool, str_row))
    if len(string) > 0:
        return "{" + string + "}"
    return null_label


converters = {
    "int": convert_integer,
    "numeric": convert_float,
    "varchar": convert_str,
    "timestamp with timezone": convert_datetime,
    "int[]": convert_integer_array,
    "varchar[]": convert_str_array,
}

iconverters_kwargs = {
    "numeric": ["decimal_places"],
    "numeric[]": ["decimal_places"],
}
