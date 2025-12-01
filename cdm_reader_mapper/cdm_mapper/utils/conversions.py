"""Convert dtypes."""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd


def _convert_array_general(
    data: pd.Series, null_label: str, convert_to_int: bool = False
) -> pd.Series:
    """
    Convert a series of values (single or list) into string arrays in "{...}" format.

    Parameters
    ----------
    data : pd.Series
        Series containing values or lists of values.
    null_label : str
        Label to use for null/empty values.
    convert_to_int : bool, default=False
        If True, numeric values are converted to integer strings only.

    Returns
    -------
    pd.Series
        Series of string arrays in "{...}" format or null_label if empty.
    """

    def _convert_value(x):
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except (SyntaxError, ValueError):
                x = [x]

        x_list = x if isinstance(x, list) else [x]

        str_list = []
        for v in x_list:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if convert_to_int:
                if isinstance(v, (int, float, np.integer, np.floating)):
                    str_list.append(str(int(v)))
            else:
                str_list.append(str(v))

        return "{" + ",".join(str_list) + "}" if str_list else null_label

    return data.apply(_convert_value).astype(object)


def convert_integer(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert numeric elements to integer strings.

    Parameters
    ----------
    data : pd.Series
        Series of numeric values to convert.
    null_label : str
        Label to use for NaN or invalid values.

    Returns
    -------
    pd.Series
        Series with values as integer strings, NaNs replaced by null_label.
    """

    def _return_str(x, null_label):
        if pd.isna(x):
            return null_label
        try:
            return str(int(float(x)))
        except ValueError:
            return null_label

    return data.apply(lambda x: _return_str(x, null_label)).astype(object)


def convert_float(data: pd.Series, null_label: str, decimal_places: int) -> pd.Series:
    """
    Convert numeric elements to float strings with specified decimals.

    Parameters
    ----------
    data : pd.Series
        Series of numeric values to convert.
    null_label : str
        Label to use for NaN or invalid values.
    decimal_places : int
        Number of decimal places for formatting.

    Returns
    -------
    pd.Series
        Series with values as formatted float strings, NaNs replaced by null_label.
    """

    def _return_str(x, null_label, format_float):
        if pd.isna(x):
            return null_label
        try:
            return format_float.format(float(x))
        except ValueError:
            return null_label

    format_float = "{:." + str(decimal_places) + "f}"
    return data.apply(lambda x: _return_str(x, null_label, format_float)).astype(object)


def convert_datetime(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert datetime elements to string format "%Y-%m-%d %H:%M:%S".

    Parameters
    ----------
    data : pd.Series
        Series of datetime objects or strings.
    null_label : str
        Label to use for NaN or invalid values.

    Returns
    -------
    pd.Series
        Series with datetime strings, NaNs replaced by null_label.
    """

    def _return_str(x, null_label):
        if pd.isna(x):
            return null_label
        if isinstance(x, str):
            return x
        return x.strftime("%Y-%m-%d %H:%M:%S")

    return data.apply(lambda x: _return_str(x, null_label)).astype(object)


def convert_str(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert elements to string representation.

    Parameters
    ----------
    data : pd.Series
        Series containing elements to convert.
    null_label : str
        Label to use for NaN or invalid values.

    Returns
    -------
    pd.Series
        Series with string representations of elements.
    """

    def _return_str(x, null_label):
        if isinstance(x, list):
            return str(x)
        if pd.isna(x):
            return null_label
        return str(x)

    return data.apply(lambda x: _return_str(x, null_label))


def convert_integer_array(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert a series of integer values or lists to string array format.

    Parameters
    ----------
    data : pd.Series
        Series containing integer values or lists of integers.
    null_label : str
        Label to use for NaN, empty, or invalid values.

    Returns
    -------
    pd.Series
        Series with integer arrays in "{...}" format.
    """
    return _convert_array_general(data, null_label, convert_to_int=True)


def convert_str_array(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert a series of values or lists to string array format.

    Parameters
    ----------
    data : pd.Series
        Series containing elements or lists of elements.
    null_label : str
        Label to use for NaN, empty, or invalid values.

    Returns
    -------
    pd.Series
        Series with string arrays in "{...}" format.
    """
    return _convert_array_general(data, null_label, convert_to_int=False)


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
