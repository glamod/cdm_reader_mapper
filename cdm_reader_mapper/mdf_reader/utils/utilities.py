"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import logging
import os

import polars as pl


def convert_dtypes(dtypes):
    """DOCUMENTATION."""
    parse_dates = []
    for i, element in enumerate(list(dtypes)):
        if dtypes[element] == "datetime":
            parse_dates.append(element)
            dtypes[element] = "object"
    return dtypes, parse_dates


def validate_arg(arg_name, arg_value, arg_type):
    """Validate input argument is as expected type.

    Parameters
    ----------
    arg_name : str
        Name of the argument
    arg_value : arg_type
        Value of the argument
    arg_type : type
        Type of the argument

    Returns
    -------
    boolean:
        Returns True if type of `arg_value` equals `arg_type`
    """
    if arg_value and not isinstance(arg_value, arg_type):
        logging.error(
            f"Argument {arg_name} must be {arg_type}, input type is {type(arg_value)}"
        )
        return False
    return True


def validate_path(arg_name, arg_value):
    """Validate input argument is an existing directory.

    Parameters
    ----------
    arg_name : str
        Name of the argument
    arg_value : str
        Value of the argument

    Returns
    -------
    boolean
        Returns True if `arg_name` is an existing directory.
    """
    if arg_value and not os.path.isdir(arg_value):
        logging.error(f"{arg_name} could not find path {arg_value}")
        return False
    return True


def convert_entries(series, converter_func, **kwargs):
    """DOCUMENTATION."""
    return converter_func(series, **kwargs)


def decode_entries(series, decoder_func):
    """DOCUMENTATION."""
    return decoder_func(series)


def set_missing_values(df: pl.DataFrame) -> pl.DataFrame:
    """DOCUMENTATION."""
    # QUESTION: Do I need to re-order the columns here?
    return (
        df.explode(columns="missing_values")
        .with_columns(pl.lit(True).alias("values"))
        .pivot("missing_values", index="index", values="values")
        .fill_null(False)
    )


def adjust_dtype(dtype, df):
    """DOCUMENTATION."""
    if not isinstance(dtype, dict):
        return dtype
    return {k: v for k, v in dtype.items() if k in df.columns}


def convert_str_boolean(x):
    """DOCUMENTATION."""
    if x == "True":
        x = True
    if x == "False":
        x = False
    return x
