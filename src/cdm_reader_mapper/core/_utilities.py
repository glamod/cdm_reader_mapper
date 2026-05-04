"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations
from collections.abc import Iterable, Iterator
from copy import deepcopy
from typing import Any, Literal

import numpy as np
import pandas as pd

from cdm_reader_mapper.common import (
    get_length,
)
from cdm_reader_mapper.common.iterators import (
    ParquetStreamReader,
    is_valid_iterator,
    parquet_stream_from_iterable,
    process_disk_backed,
)
from cdm_reader_mapper.duplicates.duplicates import DupDetect


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
    except (ValueError, AttributeError) as err:
        raise ValueError("Attribute is neither callable nor subscriptable.") from err


def reader_method(
    db: _DataBundle, data: pd.DataFrame | ParquetStreamReader, attr: str, *args: Any, process_kwargs: dict[str, Any] | None = None, **kwargs: Any
) -> ParquetStreamReader | None:
    r"""
    Handle operations on chunked data (ParquetStreamReader).

    Uses process_disk_backed to stream processing without loading into RAM.

    Parameters
    ----------
    db : _DataBundle
        DataBundle used for optional in-place updates.
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
        A new stream with the applied operation, or `None` if`inplace=True`.

    Raises
    ------
    AttributeError
        If `attr` is not found on `db.data`.
    """
    inplace = kwargs.pop("inplace", False)

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

    # The result is a tuple: (ParquetStreamReader, [extra_outputs])
    new_reader: ParquetStreamReader = result_tuple[0]

    # Handle inplace logic
    if inplace:
        db._data = new_reader
        return None

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


class _DataBundle:
    """
    Container for tabular data and associated metadata.

    This class wraps either an in-memory `pd.DataFrame` or a
    `ParquetStreamReader` for chunked, disk-backed processing. It provides
    a unified interface for accessing DataFrame-like attributes and methods,
    transparently handling streaming data where required.

    Parameters
    ----------
    data : pandas.DataFrame or Iterable[pandas.DataFrame] or ParquetStreamReader, optional
        Input data. If an iterable is provided, it is converted into a
        `ParquetStreamReader` for streaming.
    columns : pandas.Index or pandas.MultiIndex or list, optional
        Column labels used when initializing empty data.
    dtypes : pandas.Series or dict, optional
        Data types for columns.
    parse_dates : list or bool, optional
        Instructions for parsing dates.
    encoding : str, optional
        Encoding associated with the data.
    mask : pandas.DataFrame or Iterable[pandas.DataFrame] or ParquetStreamReader, optional
        Boolean mask aligned with `data`. If not provided, an empty mask is created.
    imodel : str, optional
        Name of the input data model.
    mode : {"data", "tables"}, default "data"
        Data representation mode.

    Raises
    ------
    ValueError
        If `mode` is invalid.
    TypeError
        If `data` and/or `mask` has an unsupported type.
    """

    def __init__(
        self,
        data: pd.DataFrame | Iterable[pd.DataFrame] | None = None,
        columns: pd.Index | pd.MultiIndex | list[Any] | None = None,
        dtypes: pd.Series | dict[str | tuple[str, str], Any] | None = None,
        parse_dates: list[Any] | bool | None = None,
        encoding: str | None = None,
        mask: pd.DataFrame | Iterable[pd.DataFrame] | None = None,
        imodel: str | None = None,
        mode: Literal["data", "tables"] = "data",
    ) -> None:
        """
        Initialization of a _DataBundle instance.

        Parameters
        ----------
        data : pandas.DataFrame or Iterable[pandas.DataFrame] or ParquetStreamReader, optional
            Input data. If an iterable is provided, it is converted into a
            `ParquetStreamReader` for streaming.
        columns : pandas.Index or pandas.MultiIndex or list, optional
            Column labels used when initializing empty data.
        dtypes : pandas.Series or dict, optional
            Data types for columns.
        parse_dates : list or bool, optional
            Instructions for parsing dates.
        encoding : str, optional
            Encoding associated with the data.
        mask : pandas.DataFrame or Iterable[pandas.DataFrame] or ParquetStreamReader, optional
            Boolean mask aligned with `data`. If not provided, an empty mask is created.
        imodel : str, optional
            Name of the input data model.
        mode : {"data", "tables"}, default "data"
            Data representation mode.
        """
        if mode not in ["data", "tables"]:
            raise ValueError(f"'mode' {mode} is not valid, use one of ['data', 'tables'].")

        if data is None:
            data = pd.DataFrame(columns=columns, dtype=dtypes)

        if isinstance(data, (list, tuple)):
            data = iter(data)

        if data is None:
            raise AssertionError("data should never be None here")

        if (is_valid_iterator(data) and not isinstance(data, ParquetStreamReader)) or isinstance(data, (list, tuple)):
            data = parquet_stream_from_iterable(data)

        if not isinstance(data, (pd.DataFrame, ParquetStreamReader)):
            raise TypeError(f"data has unsupported type {type(data)}")

        if mask is None:
            if isinstance(data, pd.DataFrame):
                mask = pd.DataFrame(columns=data.columns, index=data.index, dtype=bool)
            elif isinstance(data, ParquetStreamReader):
                data_cp = data.copy()
                mask = [pd.DataFrame(columns=df.columns, index=df.index, dtype=bool) for df in data_cp]

        if isinstance(mask, (list, tuple)):
            mask = iter(mask)

        if mask is None:
            raise AssertionError("mask should never be None here")

        if (is_valid_iterator(mask) and not isinstance(mask, ParquetStreamReader)) or isinstance(mask, (list, tuple)):
            mask = parquet_stream_from_iterable(mask)

        if not isinstance(mask, (pd.DataFrame, ParquetStreamReader)):
            raise TypeError(f"mask has unsupported type {type(data)}")

        self._data: pd.DataFrame | ParquetStreamReader = data
        self._columns = columns
        self._dtypes = dtypes
        self._parse_dates = parse_dates
        self._encoding = encoding
        self._mask: pd.DataFrame | ParquetStreamReader = mask
        self._imodel = imodel
        self._mode = mode
        self.DupDetect: DupDetect | None = None

    def __len__(self) -> int:
        """
        Length of :py:attr:`data`.

        Returns
        -------
        int
            Number of rows in the underlying data.

        Raises
        ------
        TypeError
             If the computed length is not an integer.
        """
        length = get_length(self._data)
        if isinstance(length, int):
            return length
        raise TypeError(f"Length is not an integer: {length}, {type(length)}")

    def __getattr__(self, attr: str) -> Any:
        """
        Apply attribute to :py:attr:`data` if attribute is not defined for :py:class:`~DataBundle` .

        Parameters
        ----------
        attr : str
            Name of the attribute.

        Returns
        -------
        Any
            Attribute value, callable wrapper, or computed result.

        Raises
        ------
        AttributeError
            If the attribute does not exist.
        ValueError
            If the data stream is empty.
        TypeError
            If the underlying data type is unsupported.
        """
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(f"DataBundle object has no attribute {attr}.")

        data = self._data

        if isinstance(data, pd.DataFrame):
            attr_func = getattr(data, attr)
            if not callable(attr_func):
                return attr_func
            return SubscriptableMethod(attr_func)

        if isinstance(data, ParquetStreamReader):
            # This allows db.read(), db.close(), db.get_chunk() to work
            if hasattr(data, attr):
                return getattr(data, attr)

            data = data.copy()

            try:
                first_chunk = data.get_chunk()
            except (StopIteration, ValueError) as err:
                raise ValueError("Cannot access attribute on empty data stream.") from err

            if not hasattr(first_chunk, attr):
                # Restore state before raising error
                data.prepend(first_chunk)
                raise AttributeError(f"DataFrame chunk has no attribute '{attr}'.")

            attr_value = getattr(first_chunk, attr)

            if callable(attr_value):
                # METHOD CALL (e.g., .dropna(), .fillna())
                # Put the chunk BACK so the reader_method sees the full stream.
                data.prepend(first_chunk)

                def wrapped_reader_method(*args: Any, **kwargs: Any) -> ParquetStreamReader | None:
                    return reader_method(self, data, attr, *args, **kwargs)

                return SubscriptableMethod(wrapped_reader_method)
            else:
                # PROPERTY ACCESS (e.g., .shape, .dtypes)
                # DO NOT put the chunk back yet. Pass the 'first_value'
                # and the 'data' iterator (which is now at chunk 2) to the combiner.
                # The combiner will consume the rest.
                return combine_attribute_values(attr_value, data, attr)

        raise TypeError(f"'data' is {type(data)}, expected DataFrame or ParquetStreamReader.")

    def __repr__(self) -> str:
        """
        Return a string representation for :py:attr:`data`.

        Returns
        -------
        str
            String representation for the underlying data.
        """
        return self._data.__repr__()

    def __setitem__(self, item: Any, value: Any) -> None:
        """
        Make class support item assignment for :py:attr:`data`.

        Parameters
        ----------
        item : Any
            Column name or property key.
        value : Any
            Value to assign.
        """
        if isinstance(item, str) and item in properties:
            setattr(self, item, value)
        else:
            self._data[item] = value

    def __getitem__(self, item: Any) -> Any:
        """
        Make class subscriptable.

        Parameters
        ----------
        item : Any
            Key or column name.

        Returns
        -------
        Any
            Item `item` of underlying data.
        """
        if isinstance(item, str):
            if hasattr(self, item):
                return getattr(self, item)
        return self._data.__getitem__(item)

    def _return_property(self, property: str) -> Any:
        """
        Return an internal property if it exists.

        Parameters
        ----------
        property : str
            Name of the attribute.

        Returns
        -------
        Any
            Internal property `property`.
        """
        if hasattr(self, property):
            return getattr(self, property)

    @property
    def data(self) -> pd.DataFrame | ParquetStreamReader:
        """
        Underlying MDF data.

        Returns
        -------
        pd.DataFrame or ParquetStreamReader
            Underlying MDf data.
        """
        return self._return_property("_data")

    @data.setter
    def data(self, value: pd.DataFrame | ParquetStreamReader) -> None:
        """
        Set the underlying MDF data.

        Parameters
        ----------
        value : pandas.DataFrame or ParquetStreamReader
            Value to be set.
        """
        self._data = value

    @property
    def columns(self) -> pd.Index | pd.MultiIndex:
        """
        Column labels of :py:attr:`data`.

        Returns
        -------
        pd.Index or pd.MultiIndex
            Column labels of the underlying MDf data.
        """
        return self._data.columns

    @columns.setter
    def columns(self, value: pd.Index | pd.MultiIndex | list[Any]) -> None:
        """
        Set column labels of the underlying MDF data.

        Parameters
        ----------
        value : pandas.Index or pandas.MultiIndex or list
            Value to be set.
        """
        self._columns = value

    @property
    def dtypes(self) -> pd.Series | dict[str, Any] | None:
        """
        Dictionary of data types on :py:attr:`data`.

        Returns
        -------
        pd.Series or dict or None
            Data types of underlying MDF data.
        """
        return self._return_property("_dtypes")

    @property
    def parse_dates(self) -> list[Any] | bool | None:
        """
        Information of how to parse dates in :py:attr:`data`.

        Returns
        -------
        list or bool or None
            Information of how to parse dates in underlying MDF data.

        See Also
        --------
        :py:func:`pd.read_csv` : Read CSV file using pandas.
        """
        parse_dates_ = self._return_property("_parse_dates")
        if parse_dates_ is None:
            return None
        if isinstance(parse_dates_, (list, bool)):
            return parse_dates_
        raise TypeError(f"parse_dates has type {type(parse_dates_)}; expected list[Any], bool, or None.")

    @property
    def encoding(self) -> str | None:
        """
        A string representing the encoding to use in the :py:attr:`data`.

        Returns
        -------
        str or None
            String representing the encoding to use in the underlying MDF data.

        See Also
        --------
        :py:func:`pd.to_csv` : Write data with encoding to CSV file.
        """
        encoding_ = self._return_property("_encoding")
        if encoding_ is None:
            return None
        if isinstance(encoding_, str):
            return encoding_
        raise TypeError(f"encoding has type {type(encoding_)}; expected str or None.")

    @property
    def mask(self) -> pd.DataFrame | ParquetStreamReader:
        """
        MDF validation mask.

        Returns
        -------
        pd.DataFrame or ParquetStreamReader
            Validation mask of the underlying MDF data.
        """
        return self._return_property("_mask")

    @mask.setter
    def mask(self, value: pd.DataFrame | ParquetStreamReader) -> None:
        """
        Set the validation mask of underlying MDF data.

        Parameters
        ----------
        value : pd.DataFrame or ParquetStreamReader
            Value to be set.
        """
        self._mask = value

    @property
    def imodel(self) -> str | None:
        """
        Name of the MDF/CDM input model.

        Returns
        -------
        str or None
            Name of the MDF/CDM input model if available.
        """
        imodel_ = self._return_property("_imodel")
        if imodel_ is None:
            return None
        if isinstance(imodel_, str):
            return imodel_
        raise TypeError(f"imodel has type {type(imodel_)}; expected str or None.")

    @imodel.setter
    def imodel(self, value: str) -> None:
        """
        Set the data model name of underlying MDF data.

        Parameters
        ----------
        value : str
            Value to be set.
        """
        self._imodel = value

    @property
    def mode(self) -> str:
        """
        Data mode.

        Returns
        -------
        str
            Current data mode.

        Raises
        ------
        TypeError
            If mode of the underlying data is not a string.
        """
        mode_ = self._return_property("_mode")
        if isinstance(mode_, str):
            return mode_
        raise TypeError(f"mode_ has type {type(mode_)}; expected str.")

    @mode.setter
    def mode(self, value: Literal["data", "tables"]) -> None:
        """
        Set the data mode name of underlying data.

        Parameters
        ----------
        value : {'data', 'tables'}
            Value to be set

        Raises
        ------
        ValueError
            If `value` is not one of `data` or `tables`.
        """
        if value not in ("data", "tables"):
            raise ValueError("value must be one of 'data' or 'tables'.")
        self._mode = value

    def copy(self) -> _DataBundle:
        """
        Make deep copy of a :py:class:`~_DataBundle`.

        Returns
        -------
        :py:class:`~_DataBundle`
              Copy of a _DataBundle.

        Examples
        --------
        >>> db2 = db.copy()
        """
        db = _DataBundle()
        for key, value in self.__dict__.items():
            value = _copy(value)
            setattr(db, key, value)
        return db
