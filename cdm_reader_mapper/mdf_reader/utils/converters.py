"""pandas converting operators."""

from __future__ import annotations

import pandas as pd

from .. import properties
from .utilities import convert_str_boolean


class df_converters:
    """Class for converting pandas DataFrame."""

    def __init__(self, dtype):
        self.dtype = dtype
        self.numeric_scale = 1.0 if self.dtype == "float" else 1
        self.numeric_offset = 0.0 if self.dtype == "float" else 0

    def decode(self, data):
        """Decode object type elements of a pandas series to UTF-8."""
        def _decode(x):
            if not isinstance(x, str):
                return x
            encoded = x.endode("latin1")
            return encoded.decode("utf-8")
            
        return data.apply(lambda x: _decode(x))

    def to_numeric(self, data, offset, scale):
        """Convert object type elements of a pandas series to numeric type."""

        def _to_numeric(x):
            x = convert_str_boolean(x)
            if isinstance(x, bool):
                return x
            if isinstance(x, str):
                x = x.strip()
                x.replace(" ", "0")
            try:
                return offset + float(x) * scale
            except ValueError:
                return False

        return data.apply(lambda x: _to_numeric(x))

    def object_to_numeric(self, data, scale=None, offset=None):
        """
        Convert the object type elements of a pandas series to numeric type.

        Right spaces are trated as ceros. Scale and offset can optionally be applied.
        The final data type according to the class dtype.

        Parameters
        ----------
        self : dtype, numeric_scale and numeric_offset
            Pandas dataframe with a column per report sections.
            The sections in the columns as a block strings.
        data : pandas.Series
            Series with data to convert. Data must be object type

        Keyword Arguments
        -----------------
        scale : numeric, optional
            Scale to apply after conversion to numeric
        offset : numeric, optional
            Offset to apply after conversion to numeric

        Returns
        -------
        data : pandas.Series
            Data series of type self.dtype

        """
        scale = scale if scale else self.numeric_scale
        offset = offset if offset else self.numeric_offset
        if data.dtype == "object":
            data = self.to_numeric(data, offset, scale)

        return data

    def object_to_object(self, data, disable_white_strip=False):
        """DOCUMENTATION."""
        # With strip() an empty element after stripping, is just an empty element, no NaN...
        if data.dtype != "object":
            return data
        data = self.decode(data)
        if not disable_white_strip:
            data = data.str.strip()
        else:
            if disable_white_strip == "l":
                data = data.str.rstrip()
            elif disable_white_strip == "r":
                data = data.str.lstrip()
        return data.apply(
            lambda x: None if isinstance(x, str) and (x.isspace() or not x) else x
        )

    def object_to_datetime(self, data, datetime_format="%Y%m%d"):
        """DOCUMENTATION."""
        if data.dtype != "object":
            return data
        return pd.to_datetime(data, format=datetime_format, errors="coerce")


converters = dict()
for dtype in properties.numeric_types:
    converters[dtype] = df_converters(dtype).object_to_numeric
converters["datetime"] = df_converters("datetime").object_to_datetime
converters["str"] = df_converters("str").object_to_object
converters["object"] = df_converters("object").object_to_object
converters["key"] = df_converters("key").object_to_object
