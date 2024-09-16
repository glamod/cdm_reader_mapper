"""pandas converting operators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import properties


class df_converters:
    """Class for converting pandas DataFrame."""

    def __init__(self, dtype):
        self.dtype = dtype
        self.numeric_scale = 1.0 if self.dtype == "float" else 1
        self.numeric_offset = 0.0 if self.dtype == "float" else 0

    def decode(self, data):
        """Decode object type elements of a pandas series to UTF-8."""
        decoded = data.str.decode("utf-8")
        if decoded.dtype != "object":
            return data
        return decoded

    def to_numeric(self, data):
        """Convert object type elements of a pandas series to numeric type."""
        data = data.apply(
            lambda x: np.nan if isinstance(x, str) and (x.isspace() or not x) else x
        )

        # str method fails if all nan, pd.Series.replace method is not the same
        # as pd.Series.str.replace!
        if data.count() > 0:
            data = self.decode(data)
            data = data.str.strip()
            data = data.str.replace(" ", "0")

        #  Convert to numeric, then scale (?!) and give it's actual int type
        return pd.to_numeric(
            data, errors="coerce"
        )  # astype fails on strings, to_numeric manages errors....!

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
            data = self.to_numeric(data)

        data = offset + data * scale
        return pd.Series(data, dtype=self.dtype)

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
            lambda x: np.nan if isinstance(x, str) and (x.isspace() or not x) else x
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
