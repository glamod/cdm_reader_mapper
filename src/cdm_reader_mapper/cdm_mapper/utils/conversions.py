"""Convert Common Datamodel (CDM) mapping table elements from/to string types."""

from __future__ import annotations
import ast
from collections.abc import Callable
from typing import Any, get_args

import numpy as np
import pandas as pd

from .. import properties
from ..tables.tables import get_cdm_atts, get_imodel_maps


class BaseConverter:
    """
    Base class for managing type conversion functions.

    Parameters
    ----------
    converters : dict
        Mapping of type names to conversion functions.
    args : dict
        Optional mapping of type names to argument names for converters.
    """

    def __init__(self, converters: dict[str, Any], args: dict[str, Any] | None = None):
        """
        Initialize the BaseConverter with converters and optional arguments.

        Parameters
        ----------
        converters : dict
            Mapping of type names to conversion functions.
        args : dict, optional
            Mapping of type names to argument names for converters.
        """
        self._converters = converters or {}
        self._args = args or {}

    def __getitem__(self, key: str) -> Callable[..., Any] | None:
        """
        Retrieve a converter function for a given type.

        Parameters
        ----------
        key : str
            The type name to look up.

        Returns
        -------
        callable or None
            The conversion function if found, else None.
        """
        return self._converters.get(key)

    def get_args(self, key: str) -> str | None:
        """
        Retrieve the argument name associated with a converter type.

        Parameters
        ----------
        key : str
            The type name to look up.

        Returns
        -------
        str or None
            The argument name if found, else None.
        """
        return self._args.get(key)

    def __contains__(self, key: str) -> bool:
        """
        Check if a converter exists for a given type.

        Parameters
        ----------
        key : str
            The type name to check.

        Returns
        -------
        bool
            True if the converter exists, False otherwise.
        """
        return key in self._converters


class ConvertFromStr(BaseConverter):
    """
    Converter class for converting string representations into Python types.

    Provides default converters for integers, floats, timestamps, and strings,
    including array variants.
    """

    def __init__(self) -> None:
        """Initialize ConvertFromStr with default string-to-type converters."""
        super().__init__(
            converters={
                "int": _convert_integer_from_str,
                "int[]": _convert_integer_array_from_str,
                "numeric": _convert_float_from_str,
                "numeric[]": _convert_float_array_from_str,
                "timestamp with timezone": _convert_datetime_from_str,
                "varchar": _convert_str_from_str,
                "varchar[]": _convert_str_array_from_str,
            }
        )


class ConvertToStr(BaseConverter):
    """
    Converter class for converting Python types to string representations.

    Provides default converters for integers, floats, timestamps, and strings,
    including array variants. Supports optional arguments for certain types
    (e.g., decimal_places for numeric types).
    """

    def __init__(self) -> None:
        """Initialize ConvertToStr with default type-to-string converters and optional arguments."""
        super().__init__(
            converters={
                "int": _convert_integer_to_str,
                "int[]": _convert_integer_array_to_str,
                "numeric": _convert_float_to_str,
                "numeric[]": _convert_float_array_to_str,
                "timestamp with timezone": _convert_datetime_to_str,
                "varchar": _convert_str_to_str,
                "varchar[]": _convert_str_array_to_str,
            },
            args={
                "numeric": "decimal_places",
                "numeric[]": "decimal_places",
            },
        )


def _convert_array_general_from_str(data: pd.Series, null_label: str, dtype: type | str) -> pd.Series:
    """
    Convert a series of string values (single or list) into an array.

    Parameters
    ----------
    data : pd.Series
        Series containing string values or lists of values.
    null_label : str
        Label to replace missing value with.
    dtype : type or str
        Result data type.

    Returns
    -------
    pd.Series
        Series of arrays.
    """

    def _convert_value(x: Any) -> Any:
        """
        Convert value to `dtype`.

        Parameters
        ----------
        x : Any
            Value to be converted.

        Returns
        -------
        Any
            Converted value.
        """
        if isinstance(x, list):
            x_list = x
        elif pd.isna(x):
            return pd.NA
        elif x == null_label:
            return pd.NA
        else:
            x_list = str(x).strip("{}").split(",")

        v_list = []
        for x_ in x_list:
            if pd.isna(x_):
                v_list.append(pd.NA)
            elif x_ == null_label:
                v_list.append(pd.NA)
            elif not x_:
                v_list.append(pd.NA)
            else:
                v_list.append(x_)

        if len(v_list) == 0:
            return pd.NA

        if len(v_list) == 1 and pd.isna(v_list[0]):
            return pd.NA

        return list(pd.array(v_list, dtype=dtype))

    return data.apply(_convert_value)


def _convert_array_general_to_str(data: pd.Series, null_label: str, dtype: type | str) -> pd.Series:
    """
    Convert a series of values (single or list) into an string array.

    Parameters
    ----------
    data : pd.Series
        Series containing values or lists of values.
    null_label : str
        Label to replace missing value with.
    dtype : type or str
        Result data type.

    Returns
    -------
    pd.Series
        Series of string arrays.
    """

    def _convert_value(x: Any) -> str:
        """
        Convert value to str.

        Parameters
        ----------
        x : Any
            Value to be converted.

        Returns
        -------
        str
            Converted value.
        """
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except (SyntaxError, ValueError):
                x = [x]

        if x is None:
            return null_label

        if isinstance(x, np.ndarray):
            x_list = x.tolist()
        elif isinstance(x, list):
            x_list = x
        else:
            x_list = [x]

        str_list = []
        for x_ in x_list:
            if pd.isna(x_):
                str_list.append(null_label)
            elif not x_ and pd.to_numeric(x_, errors="coerce"):
                str_list.append(null_label)
            else:
                t = pd.array([x_], dtype=dtype)[0]
                str_list.append(str(t))

        if len(str_list) == 0:
            return null_label

        if len(str_list) == 1 and str_list[0] == null_label:
            return null_label

        return "{" + ",".join(str_list) + "}"

    return data.apply(_convert_value).astype(object)


def _convert_str_to_str(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert elements to string representation.

    Parameters
    ----------
    data : pd.Series
        Series containing elements to convert.
    null_label : str
        Label to use for NaN or invalid values.

    Returns
    -------
    pd.Series
        Series with string representations of elements.
    """

    def _return_str(x: Any, null_label: str) -> str:
        """
        Convert value to string.

        Parameters
        ----------
        x : Any
            Value to be converted.
        null_label : str
            Label to replace missing value with.

        Returns
        -------
        str
            Converted value.
        """
        if isinstance(x, list):
            return str(x)
        if pd.isna(x):
            return null_label
        return str(x)

    return data.apply(lambda x: _return_str(x, null_label))


def _convert_str_from_str(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert elements from string representation.

    Parameters
    ----------
    data : pd.Series
        Series containing string elements to convert.
    null_label : str
        Label to use for NaN or invalid values.

    Returns
    -------
    pd.Series
        Series with data type representtions of elements.
    """

    def _return_str(x: Any, null_label: str) -> str | pd.NA:
        """
        Convert value to object.

        Parameters
        ----------
        x : Any
            Value to be converted.
        null_label : str
            Label to replace missing value with.

        Returns
        -------
        str | pd.NA
            Converted value.
        """
        if pd.isna(x):
            return pd.NA
        if x == null_label:
            return pd.NA
        return str(x)

    return data.apply(lambda x: _return_str(x, null_label))


def _convert_str_array_to_str(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert a series of string values or lists to string array format.

    Parameters
    ----------
    data : pd.Series
        Series containing integer values or lists of integers.
    null_label : str
        Label to use for NaN, empty, or invalid values.

    Returns
    -------
    pd.Series
        Series with integer arrays in "{...}" format.
    """
    return _convert_array_general_to_str(data, null_label, dtype=str)


def _convert_str_array_from_str(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert a series of string arrays in "{...}" format to Python lists of strings.

    Parameters
    ----------
    data : pd.Series
        Series containing string arrays in "{...}" format.
    null_label : str
        Label to use for NaN, empty, or invalid values.

    Returns
    -------
    pd.Series
        Series with Python lists of strings.
    """
    return _convert_array_general_from_str(data, null_label, object)


def _convert_integer_to_str(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert numeric elements to integer strings.

    Parameters
    ----------
    data : pd.Series
        Series of numeric values to convert.
    null_label : str
        Label to use for NaN or invalid values.

    Returns
    -------
    pd.Series
        Series with values as integer strings, NaNs replaced by null_label.
    """

    def _return_str(x: Any, null_label: str) -> str:
        """
        Convert value to str.

        Parameters
        ----------
        x : Any
            Value to be converted.
        null_label : str
            Label to replace missing value with.

        Returns
        -------
        str
            Converted value.
        """
        if pd.isna(x):
            return null_label
        try:
            return str(int(float(x)))
        except ValueError:
            return null_label

    return data.apply(lambda x: _return_str(x, null_label)).astype(object)


def _convert_integer_from_str(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert a series of string values to nullable integer type.

    Parameters
    ----------
    data : pd.Series
        Series containing string representations of numeric values.
    null_label : str
        Label to use for NaN, empty, or invalid values.

    Returns
    -------
    pd.Series
        Series with values converted to pandas nullable integer dtype ("Int64").
        Invalid or non-convertible values are set to pd.NA.
    """
    return pd.to_numeric(data, errors="coerce").astype("Int64")


def _convert_integer_array_to_str(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert a series of integer values or lists to string array format.

    Parameters
    ----------
    data : pd.Series
        Series containing integer values or lists of integers.
    null_label : str
        Label to use for NaN, empty, or invalid values.

    Returns
    -------
    pd.Series
        Series with integer arrays in "{...}" format.
    """
    return _convert_array_general_to_str(data, null_label, dtype="Int64")


def _convert_integer_array_from_str(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert a series of string arrays in "{...}" format to lists of integers.

    Parameters
    ----------
    data : pd.Series
        Series containing string arrays in "{...}" format.
    null_label : str
        Label to use for NaN, empty, or invalid values.

    Returns
    -------
    pd.Series
        Series with values converted to lists of integers using pandas nullable
        integer dtype ("Int64"). Invalid or non-convertible elements are set to NaN.
    """
    return _convert_array_general_from_str(data, null_label, "Int64")


def _convert_float_to_str(data: pd.Series, null_label: str, decimal_places: int) -> pd.Series:
    """
    Convert numeric elements to float strings with specified decimals.

    Parameters
    ----------
    data : pd.Series
        Series of numeric values to convert.
    null_label : str
        Label to use for NaN or invalid values.
    decimal_places : int
        Number of decimal places for formatting.

    Returns
    -------
    pd.Series
        Series with values as formatted float strings, NaNs replaced by null_label.
    """

    def _return_str(x: Any, null_label: str, format_float: str) -> str:
        """
        Convert value to str.

        Parameters
        ----------
        x : Any
            Value to be converted.
        null_label : str
            Label to replace missing value with.
        format_float : str
            Format specifier for floating-point values (e.g., ``'.2f'``).

        Returns
        -------
        str
            Converted value.
        """
        if pd.isna(x):
            return null_label
        try:
            return format_float.format(float(x))
        except ValueError:
            return null_label

    format_float = "{:." + str(decimal_places) + "f}"
    return data.apply(lambda x: _return_str(x, null_label, format_float)).astype(object)


def _convert_float_from_str(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert a series of string values to nullable float type.

    Parameters
    ----------
    data : pd.Series
        Series containing string representations of numeric values.
    null_label : str
        Label to use for NaN, empty, or invalid values.

    Returns
    -------
    pd.Series
        Series with values converted to pandas nullable float dtype ("Float64").
        Invalid or non-convertible values are set to NaN.
    """
    return pd.to_numeric(data, errors="coerce").astype("Float64")


def _convert_float_array_to_str(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert a series of float values or lists to string array format.

    Parameters
    ----------
    data : pd.Series
        Series containing float values or lists of integers.
    null_label : str
        Label to use for NaN, empty, or invalid values.

    Returns
    -------
    pd.Series
        Series with float arrays in "{...}" format.
    """
    return _convert_array_general_to_str(data, null_label, dtype=float)


def _convert_float_array_from_str(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert a series of string arrays in "{...}" format to lists of floats.

    Parameters
    ----------
    data : pd.Series
        Series containing string arrays in "{...}" format.
    null_label : str
        Label to use for NaN, empty, or invalid values.

    Returns
    -------
    pd.Series
        Series with values converted to lists of floats using pandas nullable
        float dtype ("Float64"). Invalid or non-convertible elements are set to NaN.
    """
    return _convert_array_general_from_str(data, null_label, "Float64")


def _convert_datetime_to_str(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert datetime elements to string format "%Y-%m-%d %H:%M:%S".

    Parameters
    ----------
    data : pd.Series
        Series of datetime objects or strings.
    null_label : str
        Label to use for NaN or invalid values.

    Returns
    -------
    pd.Series
        Series with datetime strings, NaNs replaced by null_label.
    """

    def _return_str(x: Any, null_label: str) -> str:
        """
        Convert value to str.

        Parameters
        ----------
        x : Any
            Value to be converted.
        null_label : str
            Label to replace missing value with.

        Returns
        -------
        str
            Converted value.
        """
        if pd.isna(x):
            return null_label
        if isinstance(x, str):
            return x
        return str(x.strftime("%Y-%m-%d %H:%M:%S"))

    return data.apply(lambda x: _return_str(x, null_label)).astype(object)


def _convert_datetime_from_str(data: pd.Series, null_label: str) -> pd.Series:
    """
    Convert a series of string values to datetime objects.

    Parameters
    ----------
    data : pd.Series
        Series containing string representations of datetime values.
    null_label : str
        Label to use for NaN, empty, or invalid values.

    Returns
    -------
    pd.Series
        Series with values converted to pandas datetime dtype (datetime64[ns]).
        Invalid or non-convertible values are set to NaT.
    """
    return pd.to_datetime(data, errors="coerce")


def _convert_column(
    series: pd.Series,
    column_atts: dict[str, Any],
    null_label: str,
    converters: ConvertFromStr | ConvertToStr,
) -> pd.Series:
    """
    Apply a type-specific converter to a pandas Series based on column attributes.

    Parameters
    ----------
    series : pd.Series
        Input series to be converted.
    column_atts : dict
        Dictionary containing metadata about the column. Must include the key
        "data_type" to determine which converter to apply. May also include
        additional parameters such as "decimal_places".
    null_label : str or None
        Label used to represent null values when converting to string format.
        Ignored when ``mode="from_str"``.
    converters : ConvertFromStr or ConvertToStr
        Converter registry that maps data types to conversion functions and
        optional argument requirements.

    Returns
    -------
    pd.Series
        Converted series if a matching converter is found; otherwise, the original
        series is returned unchanged.

    Raises
    ------
    TypeError
        If ``series`` is not a pandas Series.

    Notes
    -----
    - If ``column_atts`` does not define a "data_type", no conversion is applied.
    - If no converter exists for the given data type, the input series is returned unchanged.
    - For converters requiring additional arguments (e.g., "decimal_places" for numeric types),
      the value is taken from ``column_atts`` or falls back to a default.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pd.Series.")

    data_type = column_atts.get("data_type")
    if data_type is None:
        return series

    converter = converters[data_type]

    if converter is None:
        return series

    converter_args = converters.get_args(data_type)

    kwargs = {}
    if converter_args == "decimal_places":
        kwargs["decimal_places"] = column_atts.get(
            "decimal_places",
            properties.default_decimal_places,
        )

    return converter(series, null_label, **kwargs)


def _convert_columns(
    data: pd.DataFrame,
    imodel: str,
    cdm_subset: str | list[str] | None,
    null_label: str,
    mode: str,
) -> pd.DataFrame:
    """
    Apply type-based conversions to columns in a DataFrame based on a data model.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing columns to be converted.
    imodel : str
        Input data model identifier. Must follow the expected naming convention
        and correspond to a supported data model.
    cdm_subset : str or list, optional
        Subset of CDM (Common Data Model) tables to process. If None or empty,
        all available tables are considered.
    null_label : str
        Label used to represent null values when converting to string format.
        Ignored when ``mode="from_str"``.
    mode : {"from_str", "to_str"}
        Conversion mode:
        - "from_str": Convert string representations to native pandas dtypes.
        - "to_str": Convert values to string representations.

    Returns
    -------
    pd.DataFrame
        DataFrame with selected columns converted according to the data model
        specifications.

    Raises
    ------
    TypeError
        If ``data`` is not a pandas DataFrame or ``imodel`` is not a string.
    ValueError
        If ``imodel`` is not defined or not supported, or if ``mode`` is invalid.

    Notes
    -----
    - Column conversion is driven by CDM metadata obtained via ``get_cdm_atts``.
    - If ``mode="to_str"``, additional mappings from ``get_imodel_maps`` may
      override or extend column attributes.
    - Columns can be referenced either by name or by (table, column) tuples.
    - Columns without matching metadata or converters are left unchanged.
    - For "from_str" mode, occurrences of ``null_label`` are replaced with ``pd.NA``.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pd.DataFrame.")

    if imodel is None:
        raise ValueError("Input data model 'imodel' is not defined.")

    if not isinstance(imodel, str):
        raise TypeError(f"Input data model type is not supported: {type(imodel)}")

    data_model = imodel.split("_")
    if data_model[0] not in get_args(properties.SupportedDataModels):
        raise ValueError(f"Input data model {data_model[0]} not supported")

    if mode not in ("from_str", "to_str"):
        raise ValueError("mode must be one of {'from_str', 'to_str'}")

    converters: ConvertFromStr | ConvertToStr
    if mode == "from_str":
        data = data.replace(null_label, pd.NA)
        data = data.fillna(pd.NA)
        converters = ConvertFromStr()
        imodel_maps = {}
    else:
        converters = ConvertToStr()
        imodel_maps = get_imodel_maps(*data_model, cdm_tables=cdm_subset)

    if not cdm_subset:
        cdm_subset = properties.cdm_tables

    cdm_atts = get_cdm_atts(cdm_subset)

    data_column: str | tuple[str, str]
    for table, table_atts in cdm_atts.items():
        table_maps = imodel_maps.get(table, {})
        for column, column_atts in table_atts.items():
            if column in data.columns:
                data_column = column
            elif (table, column) in data.columns:
                data_column = (table, column)
            else:
                continue

            column_maps = table_maps.get(column, {})
            column_atts = {**column_atts, **column_maps}

            data[data_column] = _convert_column(
                data[data_column],
                column_atts,
                null_label,
                converters,
            )
    return data


def convert_from_str_df(
    data: pd.DataFrame,
    imodel: str | None,
    cdm_subset: str | list[str] | None = None,
    null_label: str = "null",
) -> pd.DataFrame:
    """
    Convert string-encoded values in a DataFrame to native pandas dtypes.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing string representations of values.
    imodel : str
        Input data model identifier used to determine column types.
    cdm_subset : str or list, optional
        Subset of CDM tables to process. If None, all tables are considered.
    null_label : str or None, default "null"
        Label representing null values in the input data.

    Returns
    -------
    pd.DataFrame
        DataFrame with values converted from string representations to
        appropriate pandas dtypes.
    """
    if imodel is None:
        raise ValueError("imodel must be a string, not None.")
    return _convert_columns(
        data,
        imodel,
        cdm_subset,
        null_label,
        "from_str",
    )


def convert_from_str_series(
    series: pd.Series,
    column_atts: dict[Any, Any],
    null_label: str = "null",
) -> pd.Series:
    """
    Convert a Series of string values to a native pandas dtype.

    Parameters
    ----------
    series : pd.Series
        Input Series containing string representations of values.
    column_atts : dict
        Dictionary defining column metadata, including the "data_type"
        used to select the appropriate converter.
    null_label : str or None, default "null"
        Label representing null values in the input data.

    Returns
    -------
    pd.Series
        Series with values converted to the appropriate pandas dtype.
    """
    series = series.fillna(pd.NA)
    return _convert_column(
        series,
        column_atts,
        null_label,
        ConvertFromStr(),
    )


def convert_to_str_df(
    data: pd.DataFrame,
    imodel: str | None,
    cdm_subset: str | list[str] | None = None,
    null_label: str = "null",
) -> pd.DataFrame:
    """
    Convert DataFrame values to string representations based on a data model.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing values to convert.
    imodel : str
        Input data model identifier used to determine column types.
    cdm_subset : str or list, optional
        Subset of CDM tables to process. If None, all tables are considered.
    null_label : str or None, default "null"
        Label to use for null or missing values in the output.

    Returns
    -------
    pd.DataFrame
        DataFrame with values converted to string representations.
    """
    if imodel is None:
        raise ValueError("imodel must be a string, not None.")
    return _convert_columns(
        data,
        imodel,
        cdm_subset,
        null_label,
        "to_str",
    )


def convert_to_str_series(
    series: pd.Series,
    column_atts: dict[Any, Any],
    null_label: str = "null",
) -> pd.Series:
    """
    Convert a Series to string representations based on column metadata.

    Parameters
    ----------
    series : pd.Series
        Input Series containing values to convert.
    column_atts : dict
        Dictionary defining column metadata, including the "data_type"
        used to select the appropriate converter.
    null_label : str or None, default "null"
        Label representing null values in the input data.

    Returns
    -------
    pd.Series
        Series with values converted to string representations.
    """
    return _convert_column(
        series,
        column_atts,
        null_label,
        ConvertToStr(),
    )
