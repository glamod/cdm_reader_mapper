"""pandas converting operators."""

from __future__ import annotations

import logging

import polars as pl

from .. import properties


class df_converters:
    """Class for converting pandas DataFrame."""

    def __init__(self, dtype):
        self.dtype = dtype
        self.numeric_scale = 1.0 if self.dtype == "float" else 1
        self.numeric_offset = 0.0 if self.dtype == "float" else 0

    def _check_conversion(self, data: pl.Series, converted: pl.Series, threshold: int):
        if (
            bad_converts := data.filter(converted.is_null() & data.is_not_null())
        ).len() > 0:
            msg = f"Have {bad_converts.len()} values that failed to be converted to {self.dtype}"
            if bad_converts.len() <= threshold:
                msg += f": values = {', '.join(bad_converts)}"
            logging.warning(msg)
        return None

    def _drop_whitespace_vals(self, data: pl.Series):
        data_name = data.name
        return (
            data.to_frame()
            .select(
                pl.when(data.str.contains(r"^\s*$"))
                .then(pl.lit(None))
                .otherwise(data)
                .alias(data_name)
            )
            .get_column(data_name)
        )

    # May not be needed?
    def decode(self, data):
        """Decode object type elements of a pandas series to UTF-8."""

        def _decode(x):
            if not isinstance(x, str):
                return x

            try:
                encoded = x.encode("latin1")
                return encoded.decode("utf-8")
            except (UnicodeDecodeError, UnicodeEncodeError):
                return x

        return data.apply(lambda x: _decode(x))

    def to_numeric(self, data: pl.Series):
        """Convert object type elements of a pandas series to numeric type."""
        data = self._drop_whitespace_vals(data)
        # str method fails if all nan, pd.Series.replace method is not the same
        # as pd.Series.str.replace!
        data = data.str.strip_chars().str.replace_all(" ", "0")

        converted = data.cast(self.dtype, strict=False)
        self._check_conversion(data, converted, 20)

        #  Convert to numeric, then scale (?!) and give it's actual int type
        return converted

    def object_to_numeric(self, data: pl.Series, scale=None, offset=None):
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
        if not data.dtype.is_numeric():
            data = self.to_numeric(data)

        data = offset + data * scale
        return data.cast(self.dtype)

    def object_to_object(self, data: pl.Series, disable_white_strip=None):
        """DOCUMENTATION."""
        if data.dtype != "object":
            return data
        data = self.decode(data)
        if disable_white_strip is None:
            data = data.str.strip_chars(" ")
        else:
            if disable_white_strip == "l":
                data = data.str.strip_chars_end(" ")
            elif disable_white_strip == "r":
                data = data.str.strip_chars_start(" ")
        return self._drop_whitespace_vals(data)

    def object_to_datetime(self, data: pl.Series, datetime_format="%Y%m%d"):
        """DOCUMENTATION."""
        if data.dtype != "object":
            return data
        converted = data.str.to_datetime(format=datetime_format, strict=False)
        self._check_conversion(data, converted, 20)
        return converted


converters = dict()
for dtype in properties.numeric_types:
    converters[dtype] = df_converters(dtype).object_to_numeric
converters[pl.Datetime] = df_converters(pl.Datetime).object_to_datetime
converters[pl.String] = df_converters(pl.String).object_to_object
converters[pl.Utf8] = df_converters(pl.Utf8).object_to_object
converters[pl.Object] = df_converters(pl.Object).object_to_object
converters[pl.Categorical] = df_converters(pl.Categorical).object_to_object
