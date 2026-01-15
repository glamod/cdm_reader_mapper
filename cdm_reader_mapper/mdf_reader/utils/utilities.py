"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import ast
import logging
import os
import pandas as pd
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence


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


def read_csv(filepath, col_subset=None, **kwargs) -> pd.DataFrame:
    """
    Safe CSV reader that handles missing files and column subsets.

    Parameters
    ----------
    filepath : str or Path or None
        Path to the CSV file.
    col_subset : list of str, optional
        Subset of columns to read from the CSV.
    kwargs : any
        Additional keyword arguments passed to pandas.read_csv.

    Returns
    -------
    pd.DataFrame
        The CSV as a DataFrame. Empty if file does not exist.
    """
    if filepath is None or not Path(filepath).is_file():
        logging.warning(f"File not found: {filepath}")
        return pd.DataFrame()

    df = pd.read_csv(filepath, delimiter=",", **kwargs)
    df.columns = update_column_labels(df.columns)
    if col_subset is not None:
        df = df[col_subset]

    return df


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


class ParquetStreamReader:
    """A wrapper that mimics pandas.io.parsers.TextFileReader."""

    def __init__(self, generator: Iterator[pd.DataFrame]):
        self._generator = generator
        self._closed = False
        self._buffer = []

    def __iter__(self):
        """Allows: for df in reader: ..."""
        return self

    def __next__(self):
        """Allows: next(reader)"""
        return next(self._generator)

    def prepend(self, chunk: pd.DataFrame):
        """
        Push a chunk back onto the front of the stream.
        Useful for peeking at the first chunk without losing it.
        """
        # Insert at 0 ensures FIFO order (peeking logic)
        self._buffer.insert(0, chunk)

    def get_chunk(self):
        """
        Safe for Large Files.
        Returns the next single chunk from disk.
        (Note: 'size' is ignored here as chunks are pre-determined by the write step)
        """
        if self._closed:
            raise ValueError("I/O operation on closed file.")

        try:
            return next(self._generator)
        except StopIteration:
            raise ValueError("No more data to read (End of stream).")

    def read(self):
        """
        WARNING: unsafe for Files > RAM.
        Reads ALL remaining data into memory at once.
        """
        if self._closed:
            raise ValueError("I/O operation on closed file.")

        # Consume the entire rest of the stream
        chunks = list(self._generator)

        if not chunks:
            return pd.DataFrame()

        return pd.concat(chunks, ignore_index=True)

    def close(self):
        """Close the stream and release resources."""
        if not self._closed:
            self._generator.close()
            self._closed = True

    def __enter__(self):
        """Allows: with ParquetStreamReader(...) as reader: ..."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Allows: with ParquetStreamReader(...) as reader: ..."""
        self.close()


def _sort_chunk_outputs(
    outputs: tuple, accumulators_initialized: bool
) -> tuple[list[pd.DataFrame], list[Any]]:
    """Separates DataFrames from metadata in the function output."""
    current_dfs = []
    new_metadata = []

    for out in outputs:
        if isinstance(out, pd.DataFrame):
            current_dfs.append(out)
        elif isinstance(out, list) and out and isinstance(out[0], pd.DataFrame):
            current_dfs.extend(out)
        elif not accumulators_initialized:
            # Only capture metadata from the first chunk
            new_metadata.append(out)

    return current_dfs, new_metadata


def _write_chunks_to_disk(current_dfs: list, temp_dirs: list, chunk_counter: int):
    """Writes the current batch of DataFrames to their respective temp directories."""
    for i, df_out in enumerate(current_dfs):
        if i < len(temp_dirs):
            file_path = Path(temp_dirs[i].name) / f"part_{chunk_counter:05d}.parquet"
            # Fix: Ensure index=False to prevent index "ghosting"
            df_out.to_parquet(
                file_path, engine="pyarrow", compression="snappy", index=False
            )


def process_disk_backed(
    reader: Iterable[pd.DataFrame],
    func: Callable,
    func_args: Sequence[Any] | None = None,
    func_kwargs: dict[str, Any] | None = None,
    makecopy: bool = True,
) -> tuple[Any, ...]:
    """
    Consumes a stream of DataFrames, processes them, and returns a tuple of
    results. DataFrames are cached to disk (Parquet) and returned as generators.
    """
    if func_args is None:
        func_args = ()
    if func_kwargs is None:
        func_kwargs = {}

    temp_dirs: list[tempfile.TemporaryDirectory] = []
    output_non_df = []
    directories_to_cleanup = []

    try:
        accumulators_initialized = False
        chunk_counter = 0

        for df in reader:
            if makecopy:
                df = df.copy()

            outputs = func(df, *func_args, **func_kwargs)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            # Sort outputs
            current_dfs, new_meta = _sort_chunk_outputs(
                outputs, accumulators_initialized
            )
            if new_meta:
                output_non_df.extend(new_meta)

            # Initialize temp dirs on the first valid chunk
            if not accumulators_initialized:
                if current_dfs:
                    for _ in range(len(current_dfs)):
                        t = tempfile.TemporaryDirectory()
                        temp_dirs.append(t)
                        directories_to_cleanup.append(t)
                    accumulators_initialized = True

            if accumulators_initialized:
                _write_chunks_to_disk(current_dfs, temp_dirs, chunk_counter)

            chunk_counter += 1

        if not accumulators_initialized:
            return tuple(output_non_df)

        # Create generators that own the temp directories
        def create_generator(temp_dir_obj):
            try:
                files = sorted(Path(temp_dir_obj.name).glob("*.parquet"))
                for f in files:
                    yield pd.read_parquet(f)
            finally:
                temp_dir_obj.cleanup()

        final_iterators = [ParquetStreamReader(create_generator(d)) for d in temp_dirs]

        # Explicitly clear this list. This transfers ownership of the TempDirectory
        # objects to the closures created above.
        directories_to_cleanup.clear()

        return tuple(final_iterators + output_non_df)

    finally:
        # Safety net: cleans up only if we crashed or failed to hand off ownership
        for d in directories_to_cleanup:
            d.cleanup()
