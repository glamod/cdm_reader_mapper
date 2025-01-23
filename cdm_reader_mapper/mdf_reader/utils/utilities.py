"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import logging
import os

import pandas as pd


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


def convert_value(value, index, converter_dict, converter_kwargs):
    """DOCUMENTATION."""
    if index not in converter_dict.keys():
        return value
    kwargs = converter_kwargs.get(index, {})
    return converter_dict[index](value, **kwargs)


def decode_value(value, index, decoder_dict):
    """DOCUMENTATION."""
    if index not in decoder_dict.keys():
        return value
    return decoder_dict[index](value)


def validate_value(value, isna, missing):
    """DOCUMENTATION."""
    mask = mask_value(value, isna, missing)
    return mask


def set_missing_values(df, ref):
    """DOCUMENTATION."""
    explode_ = df.explode("missing_values")
    explode_["index"] = explode_.index
    explode_["values"] = True
    pivots_ = explode_.pivot_table(
        columns="missing_values",
        index="index",
        values="values",
    )
    missing_values = pd.DataFrame(data=pivots_, columns=ref.columns, index=ref.index)
    return missing_values.notna()


def mask_value(value, isna, missing):
    """DOCUMENTATION."""
    if missing:
        return False
    if isna is None:
        isna = not value
    valid = bool(value)
    return isna | valid


def adjust_dtype(dtype, df):
    """DOCUMENTATION."""
    if not isinstance(dtype, dict):
        return dtype
    return {k: v for k, v in dtype.items() if k in df.columns}
