"""pandas converting operators."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Callable, Any, get_args

import pandas as pd

from .. import properties
from .utilities import convert_str_boolean

numeric_types = get_args(properties.NumericTypes)


def max_decimal_places(*decimals: Decimal) -> int:
    """
    Return the maximum number of decimal places among Decimal values.

    Parameters
    ----------
    decimals : Decimal
        One or more Decimal values.

    Returns
    -------
    int
        Maximum number of decimal places.
    """
    return max(
        (-d.as_tuple().exponent if d.as_tuple().exponent < 0 else 0) for d in decimals
    )


def to_numeric(x: Any, scale: Decimal, offset: Decimal) -> Decimal | bool:
    """
    Convert a value to a scaled Decimal with offset applied.

    Rules
    -----
    - Boolean values are returned unchanged
    - Empty or invalid values return False
    - Strings are stripped and spaces replaced with zeros
    - Result is quantized to the maximum decimal precision
      of input, scale, or offset

    Parameters
    ----------
    x : Any
        Input value to convert.
    scale : Decimal
        Scale factor.
    offset : Decimal
        Offset value.

    Returns
    -------
    Decimal | bool
        Converted Decimal value, boolean, or False if invalid.
    """
    x = convert_str_boolean(x)

    if isinstance(x, bool):
        return x

    if isinstance(x, str):
        x = x.strip()
        x = x.replace(" ", "0")

    try:
        x_dec = Decimal(str(x))
        decimal_places = max_decimal_places(offset, scale, x_dec)
        result = offset + x_dec * scale

        if decimal_places == 0:
            return result

        return result.quantize(Decimal("1." + "0" * decimal_places))

    except (InvalidOperation, TypeError, ValueError):
        return False


class Decoders:
    """
    Registry-based decoder dispatcher for column-wise decoding.

    Currently supports Base36 decoding for numeric-like fields.
    """

    def __init__(self, dtype: str, encoding: str = "base36") -> None:
        """
        Initialization.

        Parameters
        ----------
        dtype : str
            Target data type name (e.g. numeric field type)
        encoding : str, default "base36"
            Encoding scheme to use
        """
        self.dtype = dtype
        self.encoding = encoding

        self._registry = {"key": self.base36}

        for numeric_type in numeric_types:
            self._registry[numeric_type] = self.base36

    def decoder(self) -> Callable[[pd.Series], pd.Series] | None:
        """
        Return the decoder function for the configured dtype and encoding.

        Returns
        -------
        callable or None
            Decoder function accepting a pandas Series, or None if encoding
            is unsupported.

        Raises
        ------
        KeyError
            If no decoder is registered for the given dtype.
        """
        if self.encoding != "base36":
            return None

        try:
            return self._registry[self.dtype]
        except KeyError as exc:
            raise KeyError(f"No converter registered for '{self.dtype}'") from exc

    def base36(self, data: pd.Series) -> pd.Series:
        """
        Decode a pandas Series from Base36 to stringified base-10 integers.

        Boolean values are preserved.
        Invalid values raise ValueError via `int(..., 36)`.

        Parameters
        ----------
        data : pd.Series
            Input Series containing base36-encoded values

        Returns
        -------
        pd.Series
            Decoded Series with stringified integers or booleans
        """

        def _base36(x):
            x = convert_str_boolean(x)
            if isinstance(x, bool):
                return x
            return str(int(str(x), 36))

        return data.apply(_base36)


class Converters:
    """
    Registry-based converter for pandas Series.

    Converts object-typed Series into numeric, datetime, or cleaned object
    representations based on the configured dtype.
    """

    def __init__(self, dtype: str) -> None:
        """
        Initialization.

        Parameters
        ----------
        dtype : str
            Target output dtype identifier
        """
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

        for numeric_type in numeric_types:
            self._registry[numeric_type] = self.object_to_numeric

    def converter(self) -> Callable[..., pd.Series]:
        """
        Return the converter function registered for the configured dtype.

        Returns
        -------
        callable
            Converter function

        Raises
        ------
        KeyError
            If no converter is registered for the dtype
        """
        try:
            return self._registry[self.dtype]
        except KeyError as exc:
            raise KeyError(f"No converter registered for '{self.dtype}'") from exc

    def object_to_numeric(
        self,
        data: pd.Series,
        scale: float | int | None = None,
        offset: float | int | None = None,
    ) -> pd.Series:
        """
        Convert object Series to numeric using Decimal arithmetic.

        - Right spaces are treated as zeros
        - Optional scale and offset may be applied
        - Boolean values are preserved
        - Invalid conversions return False

        Parameters
        ----------
        data : pd.Series
            Object-typed Series
        scale : numeric, optional
            Scale factor
        offset : numeric, optional
            Offset value

        Returns
        -------
        pd.Series
            Converted Series
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

    def object_to_object(
        self,
        data: pd.Series,
        disable_white_strip: bool | str = False,
    ) -> pd.Series:
        """
        Clean object Series by stripping whitespace and nullifying empty strings.

        Parameters
        ----------
        data : pd.Series
            Object-typed Series
        disable_white_strip : bool or {"l", "r"}, default False
            Control whitespace stripping behavior

        Returns
        -------
        pd.Series
            Cleaned Series
        """
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

    def object_to_datetime(
        self,
        data: pd.Series,
        datetime_format: str = "%Y%m%d",
    ) -> pd.Series:
        """
        Convert object Series to pandas datetime.

        Invalid values are coerced to NaT.

        Parameters
        ----------
        data : pd.Series
            Object-typed Series
        datetime_format : str, default "%Y%m%d"
            Datetime parsing format

        Returns
        -------
        pd.Series
            Datetime Series
        """
        if data.dtype != "object":
            return data

        return pd.to_datetime(data, format=datetime_format, errors="coerce")


def convert_and_decode(
    data: pd.DataFrame,
    convert_flag: bool = True,
    decode_flag: bool = True,
    converter_dict: dict[str, Callable[[pd.Series], pd.Series]] | None = None,
    converter_kwargs: dict[str, dict] | None = None,
    decoder_dict: dict[str, Callable[[pd.Series], pd.Series]] | None = None,
) -> pd.DataFrame:
    """Convert and decode data entries by using a pre-defined data model.

    Overwrite attribute `data` with converted and/or decoded data.

    Parameters
    ----------
    data : pd.DataFrame
        Data to convert and decode.
    convert_flag : bool, default True
        If True, apply converters to the columns defined in `converter_dict`.
    decode_flag : bool, default True
        If True, apply decoders to the columns defined in `decoder_dict`.
    converter_dict : dict[str, callable], optional
        Column-specific converter functions. If None, defaults to empty dict.
    converter_kwargs : dict[str, dict], optional
        Keyword arguments for each converter function.
    decoder_dict : dict[str, callable], optional
        Column-specific decoder functions. If None, defaults to empty dict.

    Returns
    -------
    pd.DataFrame
        DataFrame with converted and decoded columns.
    """
    converter_dict = converter_dict or {}
    converter_kwargs = converter_kwargs or {}
    decoder_dict = decoder_dict or {}

    if decode_flag:
        for column, dec_func in decoder_dict.items():
            if column in data.columns:
                decoded = dec_func(data[column])
                decoded.index = data[column].index
                data[column] = decoded

    if convert_flag:
        for column, conv_func in converter_dict.items():
            if column in data.columns:
                kwargs = converter_kwargs.get(column, {})
                converted = conv_func(data[column], **kwargs)
                converted.index = data[column].index
                data[column] = converted

    return data
