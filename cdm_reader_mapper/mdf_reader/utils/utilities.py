"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import ast
import csv
import logging
import os

from io import StringIO
from pathlib import Path

import pandas as pd

from .. import properties

from cdm_reader_mapper.common.pandas_TextParser_hdlr import make_copy


def as_list(x):
    """Ensure the input is a list; keep None as None."""
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return list(x)


def as_path(value, name: str) -> Path:
    """Ensure the input is a Path-like object."""
    if isinstance(value, (str, os.PathLike)):
        return Path(value)
    raise TypeError(f"{name} must be str or Path-like")


def join(col) -> str:
    """Join multi-level columns as colon-separated string."""
    if isinstance(col, (list, tuple)):
        return ":".join(str(c) for c in col)
    return str(col)


def update_dtypes(dtypes: dict, columns) -> dict:
    """Filter dtypes dict to only include columns in 'columns'."""
    if isinstance(dtypes, dict):
        dtypes = {k: v for k, v in dtypes.items() if k in columns}
    return dtypes


def update_column_names(dtypes: dict | str, col_o, col_n) -> dict | str:
    """Rename column in dtypes dict if present."""
    if isinstance(dtypes, str):
        return dtypes
    if col_o in dtypes.keys():
        dtypes[col_n] = dtypes[col_o]
        del dtypes[col_o]
    return dtypes


def update_column_labels(columns):
    """Convert string column labels to tuples if needed."""
    new_cols = []
    all_tuples = True

    for col in columns:
        try:
            col_ = ast.literal_eval(col)
        except Exception:
            if isinstance(col, str) and ":" in col:
                col_ = tuple(col.split(":"))
            else:
                col_ = col
        all_tuples &= isinstance(col_, tuple)
        new_cols.append(col_)

    if all_tuples:
        return pd.MultiIndex.from_tuples(new_cols)
    return pd.Index(new_cols)


def read_csv(filepath, col_subset=None, **kwargs) -> pd.DataFrame:
    """Safe CSV reader that handles missing files and column subsets."""
    if filepath is None or not Path(filepath).is_file():
        logging.warning(f"File not found: {filepath}")
        return pd.DataFrame()

    df = pd.read_csv(filepath, delimiter=",", **kwargs)
    df.columns = update_column_labels(df.columns)
    if col_subset is not None:
        df = df[col_subset]

    return df


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


def _adjust_dtype(dtype, df) -> dict:
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
    data = data.map(_remove_boolean_values)
    dtype = _adjust_dtype(dtypes, data)
    return data.astype(dtype)


def process_textfilereader(
    reader,
    func,
    func_args=(),
    func_kwargs=None,
    read_kwargs={},
    write_kwargs={},
    makecopy=True,
):
    if func_kwargs is None:
        func_kwargs = {}

    buffers = []
    columns = []

    if makecopy is True:
        reader = make_copy(reader)

    for df in reader:
        outputs = func(df, *func_args, **func_kwargs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        output_dfs = []
        output_add = []
        for out in outputs:
            if isinstance(out, pd.DataFrame):
                output_dfs.append(out)
            else:
                output_add.append(out)

        if not buffers:
            buffers = [StringIO() for _ in output_dfs]
            columns = [out.columns for out in output_dfs]

        for buffer, out_df in zip(buffers, output_dfs):
            if not isinstance(out_df, pd.DataFrame):
                continue
            out_df.to_csv(
                buffer,
                header=False,
                mode="a",
                index=False,
                quoting=csv.QUOTE_NONE,
                sep=properties.internal_delimiter,
                quotechar="\0",
                escapechar="\0",
                **write_kwargs,
            )

    if isinstance(read_kwargs, dict):
        read_kwargs = tuple(read_kwargs for _ in range(buffers))

    result_dfs = []
    for buffer, cols, rk in zip(buffers, columns, read_kwargs):
        buffer.seek(0)
        result_dfs.append(
            pd.read_csv(
                buffer,
                names=cols,
                delimiter=properties.internal_delimiter,
                quotechar="\0",
                escapechar="\0",
                **rk,
            )
        )
    return tuple(result_dfs + output_add)
