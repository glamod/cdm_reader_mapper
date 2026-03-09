"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import ast
import csv
import os

from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Callable

import pandas as pd

from .. import properties

from cdm_reader_mapper.common.pandas_TextParser_hdlr import make_copy


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
    column_names: pd.Index | pd.MultiIndex | None = None,
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
    if column_names is not None and not df.empty:
        df = df.reindex(columns=column_names)
    return df, {"columns": df.columns, "dtypes": df.dtypes}


def _read_data_from_file(
    filepath: Path,
    reader: Callable[..., Any],
    col_subset: str | list | None = None,
    column_names: pd.Index | pd.MultiIndex | None = None,
    reader_kwargs: dict | None = None,
    iterator: bool = False,
) -> tuple[pd.DataFrame | Iterable[pd.DataFrame], dict[str, Any]]:
    """Helper file reader."""
    if filepath is None or not Path(filepath).is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    reader_kwargs = reader_kwargs or {}

    data = reader(filepath, **reader_kwargs)

    if isinstance(data, pd.DataFrame):
        return update_and_select(data, subset=col_subset, column_names=column_names)

    if iterator is True:
        writer_kwargs = {}
        if "encoding" in reader_kwargs:
            writer_kwargs["encoding"] = reader_kwargs["encoding"]

        return process_textfilereader(
            data,
            func=update_and_select,
            func_kwargs={
                "subset": col_subset,
                "column_names": column_names,
            },
            read_kwargs=reader_kwargs,
            write_kwargs=writer_kwargs,
            makecopy=False,
        )

    raise ValueError(f"Unsupported reader return type: {type(data)}")


def read_csv(
    filepath: Path,
    col_subset: str | list | None = None,
    column_names: pd.Index | pd.MultiIndex | None = None,
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
    column_names:
        Column labels for re-indexing.
    kwargs : any
        Additional keyword arguments passed to pandas.read_csv.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - The CSV as a DataFrame. Empty if file does not exist.
        - dictionary containing data column labels and data types
    """
    return _read_data_from_file(
        filepath,
        reader=pd.read_csv,
        col_subset=col_subset,
        column_names=column_names,
        reader_kwargs=kwargs,
        iterator=True,
    )


def read_parquet(
    filepath: Path,
    col_subset: str | list | None = None,
    column_names: pd.Index | pd.MultiIndex | None = None,
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
    column_names:
        Column labels for re-indexing.
    kwargs : any
        Additional keyword arguments passed to pandas.read_csv.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - The CSV as a DataFrame. Empty if file does not exist.
        - dictionary containing data column labels and data types
    """
    return _read_data_from_file(
        filepath,
        reader=pd.read_parquet,
        col_subset=col_subset,
        column_names=column_names,
        reader_kwargs=kwargs,
    )


def read_feather(
    filepath: Path,
    col_subset: str | list | None = None,
    column_names: pd.Index | pd.MultiIndex | None = None,
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
    column_names:
        Column labels for re-indexing.
    kwargs : any
        Additional keyword arguments passed to pandas.read_csv.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - The CSV as a DataFrame. Empty if file does not exist.
        - dictionary containing data column labels and data types
    """
    return _read_data_from_file(
        filepath,
        reader=pd.read_feather,
        col_subset=col_subset,
        column_names=column_names,
        reader_kwargs=kwargs,
    )


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


def process_textfilereader(
    reader: Iterable[pd.DataFrame],
    func: Callable,
    func_args: tuple = (),
    func_kwargs: dict[str, Any] | None = None,
    read_kwargs: dict[str, Any] | tuple[dict[str, Any], ...] | None = None,
    write_kwargs: dict[str, Any] | None = None,
    makecopy: bool = True,
) -> tuple[Iterable[pd.DataFrame], ...]:
    """
    Process a stream of DataFrames using a function and return processed results.

    Each DataFrame from `reader` is passed to `func`, which can return one or more
    DataFrames or other outputs. DataFrame outputs are concatenated in memory and
    returned as a tuple along with any additional non-DataFrame outputs.

    Parameters
    ----------
    reader : Iterable[pd.DataFrame]
        An iterable of DataFrames (e.g., a CSV reader returning chunks).
    func : Callable
        Function to apply to each DataFrame.
    func_args : tuple, optional
        Positional arguments passed to `func`.
    func_kwargs : dict, optional
        Keyword arguments passed to `func`.
    read_kwargs : dict or tuple of dict, optional
        Arguments to pass to `pd.read_csv` when reconstructing output DataFrames.
    write_kwargs : dict, optional
        Arguments to pass to `DataFrame.to_csv` when buffering output.
    makecopy : bool, default True
        If True, makes a copy of each input DataFrame before processing.

    Returns
    -------
    tuple
        A tuple containing:
            - One or more processed DataFrames (in the same order as returned by `func`)
            - Any additional outputs from `func` that are not DataFrames
    """
    func_kwargs = func_kwargs or {}
    read_kwargs = read_kwargs or {}
    write_kwargs = write_kwargs or {}

    buffers = []
    columns = []

    if makecopy is True:
        reader = make_copy(reader)

    output_add = []

    for df in reader:
        outputs = func(df, *func_args, **func_kwargs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        output_dfs = []
        first_chunk = not buffers

        for out in outputs:
            if isinstance(out, pd.DataFrame):
                output_dfs.append(out)
            elif first_chunk:
                output_add.append(out)

        if not buffers:
            buffers = [StringIO() for _ in output_dfs]
            columns = [out.columns for out in output_dfs]

        for buffer, out_df in zip(buffers, output_dfs):
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
        read_kwargs = tuple(read_kwargs for _ in range(len(buffers)))

    result_dfs = []
    for buffer, cols, rk in zip(buffers, columns, read_kwargs):
        buffer.seek(0)
        rk = {k: v for k, v in rk.items() if k != "delimiter"}
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
