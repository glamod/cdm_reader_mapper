"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from cdm_reader_mapper.common.pandas_TextParser_hdlr import make_copy

from cdm_reader_mapper.common import (
    get_length,
)

from io import StringIO as StringIO


def _copy(value):
    """Make copy of value"""
    if isinstance(value, dict):
        return deepcopy(value)
    elif isinstance(value, pd.DataFrame):
        return value.copy()
    elif isinstance(value, pd.io.parsers.TextFileReader):
        return make_copy(value)
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
    """Handles operations on chunked DataFrame (TextFileReader)."""
    data_buffer = StringIO()
    TextParser = make_copy(data)
    read_params = [
        "chunksize",
        "parse_dates",
        "date_parser",
        "infer_datetime_format",
    ]
    write_dict = {"header": None, "mode": "a", "index": True}
    read_dict = {x: TextParser.orig_options.get(x) for x in read_params}
    inplace = kwargs.get("inplace", False)
    for df_ in TextParser:
        attr_func = getattr(df_, attr)
        result_df = method(attr_func, *args, **kwargs)
        if result_df is None:
            result_df = df_
        result_df.to_csv(data_buffer, **write_dict)
    dtypes = {}
    for k, v in result_df.dtypes.items():
        if v == "object":
            v = "str"
        dtypes[k] = v
    read_dict["dtype"] = dtypes
    read_dict["names"] = result_df.columns
    data_buffer.seek(0)
    TextParser = pd.read_csv(data_buffer, **read_dict)
    if inplace:
        DataBundle._data = TextParser
        return
    return TextParser


def combine_attribute_values(attr_func, TextParser, attr):
    """Collect values of the attribute across all chunks and combine them."""
    combined_values = [attr_func]
    for chunk in TextParser:
        combined_values.append(getattr(chunk, attr))

    if isinstance(attr_func, pd.Index):
        combined_index = combined_values[0]
        for idx in combined_values[1:]:
            combined_index = combined_index.union(idx)
        return combined_index
    if isinstance(attr_func, (int, float)):
        return sum(combined_values)
    if isinstance(attr_func, tuple) and len(attr_func) == 2:
        first_ = sum(value[0] for value in combined_values)
        second_ = attr_func[1]
        return (first_, second_)
    if isinstance(attr_func, (list, np.ndarray)):
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
        elif isinstance(data, pd.io.parsers.TextFileReader):

            def wrapped_reader_method(*args, **kwargs):
                return reader_method(self, data, attr, *args, **kwargs)

            TextParser = make_copy(data)
            first_chunk = next(TextParser)
            attr_func = getattr(first_chunk, attr)
            if callable(attr_func):
                return SubscriptableMethod(wrapped_reader_method)
            return combine_attribute_values(attr_func, TextParser, attr)

        raise TypeError("'data' is neither a DataFrame nor a TextFileReader object.")

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
