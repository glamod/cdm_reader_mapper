"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from cdm_reader_mapper.common import (
    get_length,
)

from cdm_reader_mapper.common.iterators import process_disk_backed

from cdm_reader_mapper.common.pandas_TextParser_hdlr import make_copy


def _copy(value):
    """Make copy of value"""
    if isinstance(value, dict):
        return deepcopy(value)
    elif isinstance(value, pd.DataFrame):
        return value.copy()
    elif isinstance(value, pd.io.parsers.TextFileReader):
        return make_copy(value)
    elif hasattr(value, "copy"):
        return value.copy()
    return value


def method(attr_func, *args, **kwargs):
    """Handles both method calls and subscriptable attributes."""
    try:
        return attr_func(*args, **kwargs)
    except TypeError:
        return attr_func[args]
    except Exception:
        raise ValueError("Attribute is neither callable nor subscriptable.")


def reader_method(DataBundle, data, attr, *args, **kwargs):
    """
    Handles operations on chunked data (ParquetStreamReader).
    Uses process_disk_backed to stream processing without loading into RAM.
    """
    inplace = kwargs.pop("inplace", False)

    # Define the transformation function to apply per chunk
    def apply_operation(df):
        # Fetch the attribute (method or property) from the chunk
        attr_obj = getattr(df, attr)

        # Use the 'method' helper to execute it (call or subscript)
        result = method(attr_obj, *args, **kwargs)

        # If the operation was inplace on the DataFrame (returns None), yield the modified DataFrame itself.
        if result is None:
            return df
        return result

    # Process stream using Disk-Backed Parquet Engine
    result_tuple = process_disk_backed(
        data,
        apply_operation,
        makecopy=True,
    )

    # The result is a tuple: (ParquetStreamReader, [extra_outputs])
    new_reader = result_tuple[0]

    # Handle inplace logic
    if inplace:
        DataBundle._data = new_reader
        return None

    return new_reader


def combine_attribute_values(first_value, iterator, attr):
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
    """
    combined_values = [first_value]

    # Iterate through the rest of the stream
    for chunk in iterator:
        combined_values.append(getattr(chunk, attr))

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

    return combined_values


class SubscriptableMethod:
    """Allows both method calls and subscript access."""

    def __init__(self, func):
        self.func = func

    def __getitem__(self, item):
        """Ensure subscript access is handled properly."""
        try:
            return self.func[item]
        except TypeError:
            raise NotImplementedError(
                "Calling subscriptable methods have not been implemented for chunked data yet."
            )

    def __call__(self, *args, **kwargs):
        """Ensure function calls work properly."""
        return self.func(*args, **kwargs)


class _DataBundle:

    def __init__(
        self,
        data=pd.DataFrame(),
        columns=None,
        dtypes=None,
        parse_dates=None,
        encoding=None,
        mask=pd.DataFrame(),
        imodel=None,
        mode="data",
    ):
        self._data = data
        self._columns = columns
        self._dtypes = dtypes
        self._parse_dates = parse_dates
        self._encoding = encoding
        self._mask = mask
        self._imodel = imodel
        self._mode = mode

    def __len__(self) -> int:
        """Length of :py:attr:`data`."""
        return get_length(self._data)

    def __getattr__(self, attr):
        """Apply attribute to :py:attr:`data` if attribute is not defined for :py:class:`~DataBundle` ."""
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(f"DataBundle object has no attribute {attr}.")

        data = self._data

        if isinstance(data, pd.DataFrame):
            attr_func = getattr(data, attr)
            if not callable(attr_func):
                return attr_func
            return SubscriptableMethod(attr_func)
        elif hasattr(data, "get_chunk") and hasattr(data, "prepend"):
            # This allows db.read(), db.close(), db.get_chunk() to work
            if hasattr(data, attr):
                return getattr(data, attr)

            try:
                first_chunk = data.get_chunk()
            except ValueError:
                raise ValueError("Cannot access attribute on empty data stream.")

            if not hasattr(first_chunk, attr):
                # Restore state before raising error
                data.prepend(first_chunk)
                raise AttributeError(f"DataFrame chunk has no attribute '{attr}'.")

            attr_value = getattr(first_chunk, attr)

            if callable(attr_value):
                # METHOD CALL (e.g., .dropna(), .fillna())
                # Put the chunk BACK so the reader_method sees the full stream.
                data.prepend(first_chunk)

                def wrapped_reader_method(*args, **kwargs):
                    return reader_method(self, data, attr, *args, **kwargs)

                return SubscriptableMethod(wrapped_reader_method)
            else:
                # PROPERTY ACCESS (e.g., .shape, .dtypes)
                # DO NOT put the chunk back yet. Pass the 'first_value'
                # and the 'data' iterator (which is now at chunk 2) to the combiner.
                # The combiner will consume the rest.
                return combine_attribute_values(attr_value, data, attr)

        else:
            raise TypeError(
                f"'data' is {type(data)}, expected DataFrame or ParquetStreamReader."
            )

    def __repr__(self) -> str:
        """Return a string representation for :py:attr:`data`."""
        return self._data.__repr__()

    def __setitem__(self, item, value):
        """Make class support item assignment for :py:attr:`data`."""
        if isinstance(item, str):
            if hasattr(self, item):
                setattr(self, item, value)
        else:
            self._data.__setitem__(item, value)

    def __getitem__(self, item):
        """Make class subscriptable."""
        if isinstance(item, str):
            if hasattr(self, item):
                return getattr(self, item)
        return self._data.__getitem__(item)

    def _return_property(self, property):
        if hasattr(self, property):
            return getattr(self, property)

    @property
    def data(self):
        """MDF pandas.DataFrame data."""
        return self._return_property("_data")

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def columns(self):
        """Column labels of :py:attr:`data`."""
        return self._data.columns

    @columns.setter
    def columns(self, value):
        self._columns = value

    @property
    def dtypes(self):
        """Dictionary of data types on :py:attr:`data`."""
        return self._return_property("_dtypes")

    @property
    def parse_dates(self):
        """Information of how to parse dates in :py:attr:`data`.

        See Also
        --------
        :py:func:pandas.`read_csv`
        """
        return self._return_property("_parse_dates")

    @property
    def encoding(self):
        """A string representing the encoding to use in the :py:attr:`data`.

        See Also
        --------
        :py:func:pandas.`to_csv`
        """
        return self._return_property("_encoding")

    @property
    def mask(self):
        """MDF pandas.DataFrame validation mask."""
        return self._return_property("_mask")

    @mask.setter
    def mask(self, value):
        self._mask = value

    @property
    def imodel(self):
        """Name of the MDF/CDM input model."""
        return self._return_property("_imodel")

    @imodel.setter
    def imodel(self, value):
        self._imodel = value

    @property
    def mode(self):
        """Data mode."""
        return self._return_property("_mode")

    @mode.setter
    def mode(self, value):
        self._mode = value

    def _get_db(self, inplace):
        if inplace is True:
            return self
        return self.copy()

    def _return_db(self, db, inplace):
        if inplace is True:
            return
        return db

    def _stack(self, other, datasets, inplace, **kwargs):
        db_ = self._get_db(inplace)
        if not isinstance(other, list):
            other = [other]
        if not isinstance(datasets, list):
            datasets = [datasets]
        for data in datasets:
            _data = f"_{data}"
            _df = getattr(db_, _data) if hasattr(db_, _data) else pd.DataFrame()

            if isinstance(_df, pd.io.parsers.TextFileReader):
                raise ValueError("Data must be a DataFrame not a TextFileReader.")

            to_concat = [
                getattr(concat, _data) for concat in other if hasattr(concat, _data)
            ]
            if not to_concat:
                continue
            if not _df.empty:
                to_concat = [_df] + to_concat
            _df = pd.concat(to_concat, **kwargs)
            _df = _df.reset_index(drop=True)
            setattr(self, f"_{data}", _df)

        return self._return_db(db_, inplace)
