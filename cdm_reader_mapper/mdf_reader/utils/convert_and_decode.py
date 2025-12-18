"""pandas converting operators."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

import pandas as pd

from .. import properties
from .utilities import convert_str_boolean


def max_decimal_places(*decimals):
    """Get maximum number of decimal places for each Decimal number."""
    decimal_places = [
        -d.as_tuple().exponent if d.as_tuple().exponent < 0 else 0 for d in decimals
    ]
    return max(decimal_places)


def to_numeric(x, scale, offset):
    x = convert_str_boolean(x)
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        x = x.strip()
        x.replace(" ", "0")
    try:
        x = Decimal(str(x))
        decimal_places = max_decimal_places(offset, scale, x)
        result = offset + x * scale
        return result.quantize(Decimal("1." + "0" * decimal_places))
    except (InvalidOperation, ValueError):
        return False


class Decoders:

    def __init__(self, dtype, encoding="base36"):
        self.dtype = dtype
        self.encoding = encoding

        self._registry = {"key": self.base36}
        for dtype in properties.numeric_types:
            self._registry[dtype] = self.base36

    def decoder(self):
        if self.encoding != "base36":
            return

        try:
            return self._registry[self.dtype]
        except KeyError:
            raise KeyError(f"No converter registered for '{self.dtype}'")

    def base36(self, data) -> pd.Series:
        """DOCUMENTATION."""

        def _base36(x):
            x = convert_str_boolean(x)
            if isinstance(x, bool):
                return x
            return str(int(str(x), 36))

        return data.apply(lambda x: _base36(x))


class Converters:
    """Class for converting pandas DataFrame."""

    def __init__(self, dtype):
        self.dtype = dtype
        self.numeric_scale = 1.0 if self.dtype == "float" else 1
        self.numeric_offset = 0.0 if self.dtype == "float" else 0

        self.preprocessing_functions = {
            "PPPP": lambda x: (
                str(10000 + int(x)) if isinstance(x, str) and x.startswith("0") else x
            )
        }

        self._registry = {
            "datetime": self.object_to_datetime,
            "str": self.object_to_object,
            "object": self.object_to_object,
            "key": self.object_to_object,
        }

        for dtype in properties.numeric_types:
            self._registry[dtype] = self.object_to_numeric

    def converter(self):
        try:
            return self._registry[self.dtype]
        except KeyError:
            raise KeyError(f"No converter registered for '{self.dtype}'")

    def object_to_numeric(self, data, scale=None, offset=None) -> pd.Series:
        """
        Convert the object type elements of a pandas series to numeric type.

        Right spaces are treated as zeros. Scale and offset can optionally be applied.
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
        column_name : str, optional
            Name of the column being processed

        Returns
        -------
        data : pandas.Series
            Data series of type self.dtype

        """
        if data.dtype != "object":
            return data

        scale = scale if scale else self.numeric_scale
        offset = offset if offset else self.numeric_offset

        scale = Decimal(str(scale))
        offset = Decimal(str(offset))

        column_name = data.name
        if column_name in self.preprocessing_functions:
            data = data.apply(self.preprocessing_functions[column_name])

        return data.apply(lambda x: to_numeric(x, scale, offset))

    def object_to_object(self, data, disable_white_strip=False) -> pd.Series:
        """DOCUMENTATION."""
        if data.dtype != "object":
            return data

        if not disable_white_strip:
            data = data.str.strip()
        elif disable_white_strip == "l":
            data = data.str.rstrip()
        elif disable_white_strip == "r":
            data = data.str.lstrip()

        return data.apply(
            lambda x: None if isinstance(x, str) and (x.isspace() or not x) else x
        )

    def object_to_datetime(self, data, datetime_format="%Y%m%d") -> pd.DateTimeIndex:
        """DOCUMENTATION."""
        if data.dtype != "object":
            return data
        return pd.to_datetime(data, format=datetime_format, errors="coerce")


def convert_and_decode(
    data,
    convert_flag=True,
    decode_flag=True,
    converter_dict=None,
    converter_kwargs=None,
    decoder_dict=None,
) -> pd.DataFrame:
    """Convert and decode data entries by using a pre-defined data model.

    Overwrite attribute `data` with converted and/or decoded data.

    Parameters
    ----------
    data: pd.DataFrame
      Data to convert and decode.
    convert: bool, default: True
      If True convert entries by using a pre-defined data model.
    decode: bool, default: True
      If True decode entries by using a pre-defined data model.
    converter_dict: dict of {Hashable: func}, optional
      Functions for converting values in specific columns.
      If None use information from a pre-defined data model.
    converter_kwargs: dict of {Hashable: kwargs}, optional
      Key-word arguments for converting values in specific columns.
      If None use information from a pre-defined data model.
    decoder_dict: dict, optional
      Functions for decoding values in specific columns.
      If None use information from a pre-defined data model.
    """
    if converter_dict is None:
        converter_dict = {}
    if converter_kwargs is None:
        converter_kwargs = {}
    if decoder_dict is None:
        decoder_dict = {}

    if not (convert_flag and decode_flag):
        return data

    if convert_flag is not True:
        converter_dict = {}
        converter_kwargs = {}
    if decode_flag is not True:
        decoder_dict = {}

    for section, conv_func in converter_dict.items():
        if section not in data.columns:
            continue

        if section in decoder_dict.keys():
            decoded = decoder_dict[section](data[section])
            decoded.index = data[section].index
            data[section] = decoded

        converted = conv_func(data[section], **converter_kwargs[section])
        converted.index = data[section].index
        data[section] = converted

    return data
