"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import ast
import logging
import os
import pandas as pd
from pathlib import Path
from typing import Any, Iterable

from cdm_reader_mapper.common.iterators import process_disk_backed


def as_list(x: str | Iterable[Any] | None) -> list[Any] | None:
    """
    Ensure the input is a list; keep None as None.

    Parameters
    ----------
    x : str, iterable, or None
        Input value to convert. Strings become single-element lists.
        Other iterables are converted to a list preserving iteration order.
        If None is passed, None is returned.

    Returns
    -------
    list or None
        Converted list or None if input was None.

    Notes
    -----
    Sets are inherently unordered; the resulting list may not have a predictable order.
    """
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return list(x)


def as_path(value: str | os.PathLike, name: str) -> Path:
    """
    Ensure the input is a Path-like object.

    Parameters
    ----------
    value : str or os.PathLike
        The value to convert to a Path.
    name : str
        Name of the parameter, used in error messages.

    Returns
    -------
    pathlib.Path
        Path object representing `value`.

    Raises
    ------
    TypeError
        If `value` is not a string or Path-like object.
    """
    if isinstance(value, (str, os.PathLike)):
        return Path(value)
    raise TypeError(f"{name} must be str or Path-like")


def join(col: Any | Iterable[Any]) -> str:
    """
    Join multi-level columns as a colon-separated string.

    Parameters
    ----------
    col : any or iterable of any
        A column name, which may be a single value or a list/tuple of values.

    Returns
    -------
    str
        Colon-separated string if input is iterable, or string of the single value.
    """
    if isinstance(col, (list, tuple)):
        return ":".join(str(c) for c in col)
    return str(col)


def update_dtypes(dtypes: dict[str, Any], columns: Iterable[str]) -> dict[str, Any]:
    """
    Filter dtypes dictionary to only include columns present in 'columns'.

    Parameters
    ----------
    dtypes : dict
        Dictionary mapping column names to their data types.
    columns : iterable of str
        List of columns to keep.

    Returns
    -------
    dict
        Filtered dictionary containing only keys present in 'columns'.
    """
    if isinstance(dtypes, dict):
        dtypes = {k: v for k, v in dtypes.items() if k in columns}
    return dtypes


def update_column_names(
    dtypes: dict[str, Any] | str, col_o: str, col_n: str
) -> dict[str, Any] | str:
    """
    Rename a column in a dtypes dictionary if it exists.

    Parameters
    ----------
    dtypes : dict or str
        Dictionary mapping column names to data types, or a string.
    col_o : str
        Original column name to rename.
    col_n : str
        New column name.

    Returns
    -------
    dict or str
        Updated dictionary with column renamed, or string unchanged.
    """
    if isinstance(dtypes, str):
        return dtypes
    if col_o != col_n and col_o in dtypes.keys():
        dtypes[col_n] = dtypes[col_o]
        del dtypes[col_o]
    return dtypes


def update_column_labels(columns: Iterable[str | tuple]) -> pd.Index | pd.MultiIndex:
    """
    Convert string column labels to tuples if needed, producing a pandas Index or MultiIndex.

    This function attempts to parse each column label:
    - If the label is a string representation of a tuple (e.g., "('A','B')"), it will be converted to a tuple.
    - If the label is a string containing a colon (e.g., "A:B"), it will be split into a tuple ("A", "B").
    - Otherwise, the label is left unchanged.

    If all resulting labels are tuples, a pandas MultiIndex is returned.
    Otherwise, a regular pandas Index is returned.

    Parameters
    ----------
    columns : iterable of str or tuple
        Column labels to convert.

    Returns
    -------
    pd.Index or pd.MultiIndex
        Converted column labels as a pandas Index or MultiIndex.
    """
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


def update_and_select(
    df: pd.DataFrame,
    subset: str | list | None = None,
    columns: pd.Index | pd.MultiIndex | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Update string column labels and select subset from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be updated
    subset : str or list, optional
        Column names to be selected
    columns:
        Column labels for re-indexing.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - The CSV as a DataFrame. Empty if file does not exist.
        - dictionary containing data column labels and data types
    """
    df.columns = update_column_labels(df.columns)
    if subset is not None:
        df = df[subset]
    if columns is not None and not df.empty:
        df = df.reindex(columns=columns)
    return df, {"columns": df.columns, "dtypes": df.dtypes}


def read_csv(
    filepath: Path,
    col_subset: str | list | None = None,
    columns: pd.Index | pd.MultiIndex | None = None,
    **kwargs,
) -> tuple[pd.DataFrame | Iterable[pd.DataFrame], dict[str, Any]]:
    """
    Safe CSV reader that handles missing files and column subsets.

    Parameters
    ----------
    filepath : str or Path or None
        Path to the CSV file.
    col_subset : list of str, optional
        Subset of columns to read from the CSV.
    columns:
        Column labels for re-indexing.
    kwargs : any
        Additional keyword arguments passed to pandas.read_csv.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - The CSV as a DataFrame. Empty if file does not exist.
        - dictionary containing data column labels and data types
    """
    if filepath is None or not Path(filepath).is_file():
        logging.warning(f"File not found: {filepath}")
        return pd.DataFrame(), {}

    data = pd.read_csv(filepath, delimiter=",", **kwargs)

    if isinstance(data, pd.DataFrame):
        data, info = update_and_select(data, subset=col_subset, columns=columns)
        return data, info

    data, info = process_disk_backed(
        data,
        func=update_and_select,
        func_kwargs={"subset": col_subset, "columns": columns},
        makecopy=False,
    )
    return data, info


def convert_dtypes(dtypes) -> tuple[str]:
    """
    Convert datetime columns to object dtype and return columns to parse as dates.

    Parameters
    ----------
    dtypes : dict[str, str]
        Dictionary mapping column names to pandas dtypes.

    Returns
    -------
    tuple
        - Updated dtypes dictionary (datetime converted to object).
        - List of columns originally marked as datetime.
    """
    parse_dates = []
    for key, value in dtypes.items():
        if value == "datetime":
            parse_dates.append(key)
            dtypes[key] = "object"
    return dtypes, parse_dates


def validate_arg(arg_name, arg_value, arg_type) -> bool:
    """
    Validate that the input argument is of the expected type.

    Parameters
    ----------
    arg_name : str
        Name of the argument.
    arg_value : Any
        Value of the argument.
    arg_type : type
        Expected type of the argument.

    Returns
    -------
    bool
        True if `arg_value` is of type `arg_type` or None.

    Raises
    ------
    ValueError
        If `arg_value` is not of type `arg_type` and not None.
    """
    if arg_value and not isinstance(arg_value, arg_type):
        raise ValueError(
            f"Argument {arg_name} must be {arg_type} or None, not {type(arg_value)}"
        )

    return True


def _adjust_dtype(dtype, df) -> dict:
    """Filter dtype dictionary to only include columns present in the DataFrame."""
    if not isinstance(dtype, dict):
        return dtype
    return {k: v for k, v in dtype.items() if k in df.columns}


def convert_str_boolean(x) -> str | bool:
    """
    Convert string boolean values 'True'/'False' to Python booleans.

    Parameters
    ----------
    x : Any
        Input value.

    Returns
    -------
    bool or original value
        True if 'True', False if 'False', else original value.
    """
    if x == "True":
        x = True
    if x == "False":
        x = False
    return x


def _remove_boolean_values(x) -> str | None:
    """Remove boolean values or string representations of boolean."""
    x = convert_str_boolean(x)
    if x is True or x is False:
        return None
    return x


def remove_boolean_values(data, dtypes) -> pd.DataFrame:
    """
    Remove boolean values from a DataFrame and adjust dtypes.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    dtypes : dict
        Dictionary mapping column names to desired dtypes.

    Returns
    -------
    pd.DataFrame
        DataFrame with booleans removed and dtype adjusted.
    """
    data = data.map(_remove_boolean_values)
    dtype = _adjust_dtype(dtypes, data)
    return data.astype(dtype)
