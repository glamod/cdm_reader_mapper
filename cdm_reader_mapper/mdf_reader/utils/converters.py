"""pandas converting operators."""

from __future__ import annotations

from datetime import datetime

from .. import properties


class df_converters:
    """Class for converting pandas DataFrame."""

    def __init__(self, dtype):
        self.dtype = dtype
        self.numeric_scale = 1.0 if self.dtype == "float" else 1
        self.numeric_offset = 0.0 if self.dtype == "float" else 0

    def to_numeric(self, data):
        """Convert object type elements of a pandas series to numeric type."""
        data = None if isinstance(data, str) and (data.isspace() or not data) else data
        if not data:
            return
        data = data.strip()
        data = data.replace(" ", "0")
        try:
            return float(data)
        except ValueError:
            return

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
        if not isinstance(data, str):
            return data
        scale = scale if scale else self.numeric_scale
        offset = offset if offset else self.numeric_offset
        data = self.to_numeric(data)
        if data is None:
            return
        return offset + data * scale

    def object_to_object(self, data, disable_white_strip=False):
        """DOCUMENTATION."""
        # With strip() an empty element after stripping, is just an empty element, no NaN...
        if not isinstance(data, str):
            return data
        if not disable_white_strip:
            data = data.strip()
        else:
            if disable_white_strip == "l":
                data = data.rstrip()
            elif disable_white_strip == "r":
                data = data.lstrip()
        return None if isinstance(data, str) and (data.isspace() or not data) else data

    def object_to_datetime(self, data, datetime_format="%Y%m%d"):
        """DOCUMENTATION."""
        if not isinstance(data, str):
            return data
        return datetime.strptime(data, datetime_format)


converters = dict()
for dtype in properties.numeric_types:
    converters[dtype] = df_converters(dtype).object_to_numeric
converters["datetime"] = df_converters("datetime").object_to_datetime
converters["str"] = df_converters("str").object_to_object
converters["object"] = df_converters("object").object_to_object
converters["key"] = df_converters("key").object_to_object
