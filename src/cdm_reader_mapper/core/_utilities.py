"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations
from collections.abc import Iterable, Iterator
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from cdm_reader_mapper.common import (
    ParquetStreamReader,
    is_valid_iterator,
    parquet_stream_from_iterable,
    process_disk_backed,
)


properties = {
    "data",
    "columns",
    "dtypes",
    "mask",
    "imodel",
    "mode",
    "parse_dates",
    "encoding",
}


def _copy(value: Any) -> Any:
    """
    Make copy of value.

    Parameters
    ----------
    value : Any
        Value to make a copy of.

    Returns
    -------
    Any
        Copy of `value`.
    """
    if isinstance(value, dict):
        return deepcopy(value)
    elif isinstance(value, (pd.DataFrame, pd.Series)):
        return value.copy()
    elif isinstance(value, ParquetStreamReader):
        return value.copy()
    elif hasattr(value, "copy"):
        return value.copy()
    return value


def method(attr_func: Any, *args: Any, **kwargs: Any) -> Any:
    r"""
    Handle both method calls and subscriptable attributes.

    Parameters
    ----------
    attr_func : Any
        A callable object (e.g., function or method) or a subscriptable object
        (e.g., list, tuple, dict, or array-like).
    \*args : Any
        Positional arguments passed to `attr_func`, or used as the index/key
        when `attr_func` is subscriptable.
    \**kwargs : Any
        Keyword arguments passed to `attr_func`. Ignored if `attr_func` is not callable.

    Returns
    -------
    Any
        The result of calling `attr_func(*args, **kwargs)` if it is callable,
        or the result of `attr_func[args]` if it is subscriptable.

    Raises
    ------
    ValueError
        If `attr_func` is neither callable nor subscriptable, or if indexing
        fails due to an invalid key or index.
    """
    if callable(attr_func):
        return attr_func(*args, **kwargs)

    try:
        return attr_func[args]
    except (ValueError, TypeError, AttributeError) as err:
        raise ValueError("Attribute is neither callable nor subscriptable.") from err


def reader_method(
    data: pd.DataFrame | ParquetStreamReader,
    attr: str,
    *args: Any,
    process_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ParquetStreamReader | None:
    r"""
    Handle operations on chunked data (ParquetStreamReader).

    Uses process_disk_backed to stream processing without loading into RAM.

    Parameters
    ----------
    data : pd.DataFrame or ParquetStreamReader
        Input data to operate on.
    attr : str
        Name of attribute or method of to apply.
    \*args : Any
        Positional arguments passed to the attribute or method.
    process_kwargs : dict, optional
        Additional keyword arguments passed to the streaming processor.
    \**kwargs : Any
        Keyword arguments passed to the attribute or method. Supports
        `inplace` to update `db` instead of returning a result.

    Returns
    -------
    ParquetStreamReader or None
        A new stream with the applied operation.
    """

    # Define the transformation function to apply per chunk
    def apply_operation(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply operation to `df`.

        Parameters
        ----------
        df : pd.DataFrame
            Data wo apply operation on.

        Returns
        -------
        pd.DataFrame
            Manipulated data.
        """
        # Fetch the attribute (method or property) from the chunk
        attr_obj = getattr(df, attr)

        # Use the 'method' helper to execute it (call or subscript)
        return method(attr_obj, *args, **kwargs)

    if process_kwargs is None:
        process_kwargs = {}

    # Process stream using Disk-Backed Parquet Engine
    result_tuple = process_disk_backed(
        data,
        apply_operation,
        makecopy=False,
        **process_kwargs,
    )

    if result_tuple is None:
        return None

    # The result is a tuple: (ParquetStreamReader, [extra_outputs])
    new_reader: ParquetStreamReader = result_tuple[0]

    return new_reader


def combine_attribute_values(first_value: Any, iterator: Iterator[Any] | ParquetStreamReader, attr: str) -> Any:
    """
    Collect values of an attribute across all chunks and combine them.

    Parameters
    ----------
    first_value : Any
        The value from the first chunk (already read).
    iterator : Iterator/ParquetStreamReader
        The stream positioned at the second chunk.
    attr : str
        The attribute name to fetch from remaining chunks.

    Returns
    -------
    Any
        Combined attribute values of `iterator`.
    """
    combined_values = [first_value]
    # Iterate through the rest of the stream
    combined_values.extend(getattr(chunk, attr) for chunk in iterator)

    # Logic to merge results based on type
    if isinstance(first_value, pd.Index):
        combined_index = first_value
        for idx in combined_values[1:]:
            combined_index = combined_index.union(idx)
        return combined_index

    if isinstance(first_value, (int, float)):
        return sum(combined_values)

    if isinstance(first_value, tuple) and len(first_value) == 2:
        # Tuple usually implies shape (rows, cols)
        # Sum rows (0), keep cols (1) constant
        first_ = sum(value[0] for value in combined_values)
        second_ = first_value[1]
        return (first_, second_)

    if isinstance(first_value, (list, np.ndarray)):
        return np.concatenate(combined_values)

    if isinstance(first_value, (pd.DataFrame, pd.Series)):
        return pd.concat(combined_values)

    return combined_values


class SubscriptableMethod:
    """
    Allows both method calls and subscript access.

    Parameters
    ----------
    func : Any
        Underlying callable or subscriptable object.
    """

    def __init__(self, func: Any) -> None:
        """
        Initialization of a SubscriptableMethod instance.

        Parameters
        ----------
        func : Any
            Underlying callable or subscriptable object.
        """
        self.func = func

    def __getitem__(self, item: Any) -> Any:
        """
        Ensure subscript access is handled properly.

        Parameters
        ----------
        item : Any
            Index or key used for subscripting.

        Returns
        -------
        Any
            Result of `self.func[item]`.

        Raises
        ------
        NotImplementedError
            If the wrapped object does not support subscripting.
        """
        try:
            return self.func[item]
        except TypeError as err:
            raise NotImplementedError("Calling subscriptable methods have not been implemented for chunked data yet.") from err

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        r"""
        Ensure function calls work properly.

        Parameters
        ----------
        \*args : Any
            Positional arguments passed to the callable.
        \**kwargs : Any
            Keyword arguments passed to the callable.

        Returns
        -------
        Any
            Result of `self.func(*args, **kwargs)`.
        """
        return self.func(*args, **kwargs)


def _validate_mode(mode: str) -> None:
    """
    Validate the data bundle mode.

    Parameters
    ----------
    mode : str
        Mode string to validate.

    Raises
    ------
    ValueError
        If the mode is not one of the supported values.
    """
    if mode not in {"data", "tables"}:
        raise ValueError(f"'mode' {mode} is not valid, use one of ['data', 'tables'].")


def _normalize_data_input(
    data: pd.DataFrame | Iterable[pd.DataFrame] | None,
    columns: pd.Index | pd.MultiIndex | list[Any] | None,
    dtypes: pd.Series | dict[str | tuple[str, str], Any] | None,
) -> pd.DataFrame | ParquetStreamReader:
    """
    Normalize and validate the input data.

    Parameters
    ----------
    data : pd.DataFrame, Iterable of pd.DataFrame, or None
        Input data.
    columns : pd.Index, pd.MultiIndex, or list, optional
        Column labels used when initializing empty data.
    dtypes : pd.Series or dict, optional
        Data types for columns.

    Returns
    -------
    pd.DataFrame or ParquetStreamReader
        Normalized data representation.

    Raises
    ------
    TypeError
        If the data type is unsupported.
    """
    if data is None:
        data = pd.DataFrame(columns=columns, dtype=dtypes)

    if isinstance(data, (list, tuple)):
        data = iter(data)

    if is_valid_iterator(data) and not isinstance(data, ParquetStreamReader):
        data = parquet_stream_from_iterable(data)

    if not isinstance(data, (pd.DataFrame, ParquetStreamReader)):
        raise TypeError(f"'data' has unsupported type {type(data)}.")

    return data


def _normalize_mask_input(
    mask: pd.DataFrame | Iterable[pd.DataFrame] | None,
    data: pd.DataFrame | ParquetStreamReader,
) -> pd.DataFrame | ParquetStreamReader:
    """
    Normalize and validate the mask aligned with the input data.

    Parameters
    ----------
    mask : pd.DataFrame, Iterable of pd.DataFrame, or None
        Input mask.
    data : pd.DataFrame or ParquetStreamReader
        Normalized data used to infer mask structure when mask is None.

    Returns
    -------
    pd.DataFrame or ParquetStreamReader
        Normalized mask.

    Raises
    ------
    TypeError
        If the mask type is unsupported.
    """
    if mask is None:
        if isinstance(data, pd.DataFrame):
            mask = pd.DataFrame(columns=data.columns, index=data.index, dtype=bool)
        else:
            data_cp = data.copy()
            mask = [pd.DataFrame(columns=df.columns, index=df.index, dtype=bool) for df in data_cp]

    if isinstance(mask, (list, tuple)):
        mask = iter(mask)

    if is_valid_iterator(mask) and not isinstance(mask, ParquetStreamReader):
        mask = parquet_stream_from_iterable(mask)

    if not isinstance(mask, (pd.DataFrame, ParquetStreamReader)):
        raise TypeError(f"mask has unsupported type {type(mask)}")

    return mask
