"""Convert Common Datamodel (CDM) mapping table elements from/to string types"""

from __future__ import annotations

import ast

from typing import get_args

import numpy as np
import pandas as pd

from . import properties

from .tables.tables import get_imodel_maps, get_cdm_atts


class BaseConverter:
    def __init__(self, converters: dict, args: dict | None = None):
        self._converters = converters or {}
        self._args = args or {}

    def __getitem__(self, key: str):
        return self._converters.get(key)

    def get_args(self, key: str):
        return self._args.get(key)

    def __contains__(self, key: str):
        return key in self._converters


class ConvertFromStr(BaseConverter):

    def __init__(self):
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

    def __init__(self):
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


def _convert_array_general_from_str(data: pd.Series, dtype: type) -> pd.Series:
    """
    Convert a series of values (single or list) into an array.

    Parameters
    ----------
    data : pd.Series
        Series containing values or lists of values.

    Returns
    -------
    pd.Series
        Series of arrays.
    """

    def _convert_value(x):
        if isinstance(x, list):
            x_list = x
        elif pd.isna(x):
            x_list = []
        else:
            x_list = str(x).strip("{}").split(",")

        value_list = list(pd.array(x_list, dtype=dtype))
        if not value_list:
            value_list = pd.NA

        return value_list

    return data.apply(_convert_value)


def _convert_array_general_to_str(
    data: pd.Series, null_label: str, dtype: type
) -> pd.Series:
    """
    Convert a series of values (single or list) into an array.

    Parameters
    ----------
    data : pd.Series
        Series containing values or lists of values.

    Returns
    -------
    pd.Series
        Series of arrays.
    """

    def _convert_value(x):
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

        str_list = [str(dtype(x_)) for x_ in x_list]

        return "{" + ",".join(str_list) + "}" if x_list else null_label

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

    def _return_str(x, null_label):
        if isinstance(x, list):
            return str(x)
        if pd.isna(x):
            return null_label
        return str(x)

    return data.apply(lambda x: _return_str(x, null_label))


def _convert_str_from_str(data: pd.Series) -> pd.Series:
    """
    Convert elements from string representation.

    Parameters
    ----------
    data : pd.Series
        Series containing string elements to convert.

    Returns
    -------
    pd.Series
        Series with data type representtions of elements.
    """
    return data.astype(object)


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


def _convert_str_array_from_str(data: pd.Series) -> pd.Series:
    return _convert_array_general_from_str(data, object)


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

    def _return_str(x, null_label):
        if pd.isna(x):
            return null_label
        try:
            return str(int(float(x)))
        except ValueError:
            return null_label

    return data.apply(lambda x: _return_str(x, null_label)).astype(object)


def _convert_integer_from_str(data: pd.Series) -> pd.Series:
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
    return _convert_array_general_to_str(data, null_label, dtype=int)


def _convert_integer_array_from_str(data: pd.Series) -> pd.Series:
    return _convert_array_general_from_str(data, "Int64")


def _convert_float_to_str(
    data: pd.Series, null_label: str, decimal_places: int
) -> pd.Series:
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

    def _return_str(x, null_label, format_float):
        if pd.isna(x):
            return null_label
        try:
            return format_float.format(float(x))
        except ValueError:
            return null_label

    format_float = "{:." + str(decimal_places) + "f}"
    return data.apply(lambda x: _return_str(x, null_label, format_float)).astype(object)


def _convert_float_from_str(data: pd.Series) -> pd.Series:
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


def _convert_float_array_from_str(data: pd.Series) -> pd.Series:
    return _convert_array_general_from_str(data, "Float64")


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

    def _return_str(x, null_label):
        if pd.isna(x):
            return null_label
        if isinstance(x, str):
            return x
        return x.strftime("%Y-%m-%d %H:%M:%S")

    return data.apply(lambda x: _return_str(x, null_label)).astype(object)


def _convert_datetime_from_str(data: pd.Series) -> pd.Series:
    return pd.to_datetime(data)


def _convert_column(
    series: pd.Series,
    column_atts: dict,
    converters: ConvertFromStr | ConvertToStr,
    **kwargs,
) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pd.Series.")

    data_type = column_atts.get("data_type")
    if data_type is None:
        return series

    converter = converters[data_type]

    if converter is None:
        return series

    converter_args = converters.get_args(data_type)

    column_kwargs = {**kwargs}
    if converter_args == "decimal_places":
        column_kwargs["decimal_places"] = column_atts.get(
            "decimal_places",
            properties.default_decimal_places,
        )

    return converter(series, **column_kwargs)


def _convert_columns(
    data: pd.DataFrame,
    imodel: str,
    cdm_subset: str | list | None,
    null_label: str | None,
    mode: str,
) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pd.DataFrame.")

    if imodel is None:
        raise ValueError("Input data model 'imodel' is not defined.")

    if not isinstance(imodel, str):
        raise TypeError(f"Input data model type is not supported: {type(imodel)}")

    data_model = imodel.split("_")
    if data_model[0] not in get_args(properties.SupportedDataModels):
        raise ValueError("Input data model " f"{data_model[0]}" " not supported")

    if mode not in ("from_str", "to_str"):
        raise ValueError("mode must be one of {'from_str', 'to_str'}")

    if mode == "from_str":
        data = data.replace(null_label, pd.NA)
        data = data.fillna(pd.NA)
        converters = ConvertFromStr()
        kwargs = {}
        imodel_maps = {}
    else:
        converters = ConvertToStr()
        kwargs = {"null_label": null_label}
        imodel_maps = get_imodel_maps(*data_model, cdm_tables=cdm_subset)

    if not cdm_subset:
        cdm_subset = properties.cdm_tables

    cdm_atts = get_cdm_atts(cdm_subset)

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
                converters,
                **kwargs,
            )
    return data


def convert_from_str_df(
    data: pd.DataFrame,
    imodel: str,
    cdm_subset: str | list | None = None,
    null_label: str | None = "null",
) -> pd.DataFrame:
    return _convert_columns(
        data,
        imodel,
        cdm_subset,
        null_label,
        "from_str",
    )


def convert_from_str_series(
    series: pd.Series,
    column_atts: dict,
) -> pd.Series:
    series = series.fillna(pd.NA)
    return _convert_column(
        series,
        column_atts,
        ConvertFromStr(),
    )


def convert_to_str_df(
    data: pd.DataFrame,
    imodel: str,
    cdm_subset: str | list | None = None,
    null_label: str | None = "null",
):
    return _convert_columns(
        data,
        imodel,
        cdm_subset,
        null_label,
        "to_str",
    )


def convert_to_str_series(
    series: pd.Series,
    column_atts: dict,
) -> pd.Series:
    return _soncert_column(
        series,
        column_atts,
        ConvertToStr(),
    )
