"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import csv
import os

from io import StringIO

import pandas as pd

from .. import properties

from cdm_reader_mapper.common.pandas_TextParser_hdlr import make_copy


def convert_dtypes(dtypes) -> tuple[str]:
    """Convert datetime to object."""
    parse_dates = []
    for key, value in dtypes.items():
        if value == "datetime":
            parse_dates.append(key)
            dtypes[key] = "object"
    return dtypes, parse_dates


def validate_arg(arg_name, arg_value, arg_type) -> bool:
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
        raise ValueError(
            f"Argument {arg_name} must be {arg_type} or None, not {type(arg_value)}"
        )

    return True


def validate_path(arg_name, arg_value) -> bool:
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
    if not os.path.isdir(arg_value):
        raise FileNotFoundError(f"{arg_name}: could not find path {arg_value}")
    return True


def validate_file(arg_name, arg_value) -> bool:
    """Validate input argument is an existing file.

    Parameters
    ----------
    arg_name : str
        Name of the argument
    arg_value : str
        Value of the argument

    Returns
    -------
    boolean
        Returns True if `arg_name` is an existing file.
    """
    if not os.path.isfile(arg_value):
        raise FileNotFoundError(f"{arg_name}: could not find file {arg_value}")
    return True


def adjust_dtype(dtype, df) -> dict:
    """Adjust dtypes to DataFrame."""
    if not isinstance(dtype, dict):
        return dtype
    return {k: v for k, v in dtype.items() if k in df.columns}


def convert_str_boolean(x) -> str | bool:
    """Convert str boolean value to boolean value."""
    if x == "True":
        x = True
    if x == "False":
        x = False
    return x


def _remove_boolean_values(x) -> str | None:
    """Remove boolean values."""
    x = convert_str_boolean(x)
    if x is True:
        return
    if x is False:
        return
    return x


def remove_boolean_values(data, dtypes) -> pd.DataFrame:
    """DOCUMENTATION"""
    data = data.map(_remove_boolean_values)
    dtype = adjust_dtype(dtypes, data)
    return data.astype(dtype)


def process_textfilereader(
    reader,
    func,
    func_args=[],
    func_kwargs={},
    read_kwargs={},
    write_kwargs={},
    makecopy=True,
):
    data_buffer = StringIO()
    if makecopy is True:
        reader = make_copy(reader)
    for df in reader:
        df = func(df, *func_args, **func_kwargs)
        df.to_csv(
            data_buffer,
            header=False,
            mode="a",
            index=False,
            quoting=csv.QUOTE_NONE,
            sep=properties.internal_delimiter,
            quotechar="\0",
            escapechar="\0",
            **write_kwargs,
        )
    data_buffer.seek(0)
    data = pd.read_csv(
        data_buffer,
        names=df.columns,
        delimiter=properties.internal_delimiter,
        quotechar="\0",
        escapechar="\0",
        **read_kwargs,
    )
    return data
