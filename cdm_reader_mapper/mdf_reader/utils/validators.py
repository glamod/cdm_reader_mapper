"""Data validation module."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from typing import Any, Iterable

from .. import properties
from ..codes import codes
from .utilities import convert_str_boolean


def _is_false(x: Any) -> bool:
    """Check if a value is exactly False."""
    return x is False


def _is_true(x: Any) -> bool:
    """Check if a value is exactly False."""
    return x is True


def validate_datetime(series: pd.Series) -> pd.Series:
    """
    Validate that entries in a pandas Series can be converted to datetime.

    Missing values are treated as valid.

    Parameters
    ----------
    series : pd.Series
        Series of object values to validate

    Returns
    -------
    pd.Series
        Boolean Series indicating valid entries
    """
    dates = pd.to_datetime(series, errors="coerce")
    return dates.notna() | series.isna()


def validate_numeric(
    series: pd.Series, valid_min: float, valid_max: float
) -> pd.Series:
    """
    Validate that entries in a pandas Series are numeric and within a range.

    - Converts boolean-like strings to bools.
    - Invalid or missing values are marked as False unless missing (NaN).

    Parameters
    ----------
    series : pd.Series
        Series of object values to validate
    valid_min : float
        Minimum valid value
    valid_max : float
        Maximum valid value

    Returns
    -------
    pd.Series
        Boolean Series indicating valid entries
    """
    converted = series.apply(convert_str_boolean)
    numeric = pd.to_numeric(converted, errors="coerce")
    valid_range = numeric.between(valid_min, valid_max)
    return valid_range | series.isna()


def validate_str(series: pd.Series) -> pd.Series:
    """
    Validate that entries in a pandas Series are strings.

    Currently all values are treated as valid.

    Parameters
    ----------
    series : pd.Series
        Series of object values to validate

    Returns
    -------
    pd.Series
        Boolean Series with all True
    """
    return pd.Series(True, index=series.index, dtype="boolean")


def validate_codes(
    series: pd.Series, code_table: Iterable[Any], column_type: str
) -> pd.Series:
    """
    Validate that entries in a pandas Series exist in a provided code table.

    Missing values are treated as valid.

    Parameters
    ----------
    series : pd.Series
        Series of object values to validate
    code_table : Iterable
        Allowed codes for validation
    column_type : str
        Column type for dtype lookup (via properties.pandas_dtypes)

    Returns
    -------
    pd.Series
        Boolean Series indicating valid entries
    """
    if not code_table:
        logging.error(f"Code table not found for element {series.name}")
        return pd.Series(False, index=series.index)

    keys = set(code_table)
    dtype = properties.pandas_dtypes.get(column_type, object)
    converted = series.astype(dtype)
    as_str = converted.astype(str)
    return converted.isna() | as_str.isin(keys)


def validate(
    data: pd.DataFrame,
    imodel: str,
    ext_table_path: str,
    attributes: dict[str, dict[str, Any]],
    disables: list[str] | None = None,
) -> pd.DataFrame:
    """
    Validate a pandas DataFrame according to a data model and code tables.

    Each column is validated based on its `column_type` attribute. Supports:
      - Numeric types: checked against valid_min and valid_max
      - Keys: checked against a code table
      - Datetime and string: validated using simple validators
      - Explicit boolean literals ("True"/"False") override column validation

    Parameters
    ----------
    data : pd.DataFrame
        Input data to validate.
    imodel : str
        Name of the internal data model, e.g., 'icoads_r300_d704'.
    ext_table_path : str
        Path to external code tables for validation.
    attributes : dict[str, dict]
        Dictionary of column attributes (e.g., type, valid ranges, codetable).
    disables : list[str], optional
        Columns to skip during validation.

    Returns
    -------
    pd.DataFrame
        Boolean mask of the same shape as `data`. True indicates a valid entry.
    """
    if not isinstance(data, pd.DataFrame):
        logging.error("input data must be a pandas DataFrame.")
        return None

    mask = pd.DataFrame(pd.NA, index=data.index, columns=data.columns, dtype="boolean")
    if data.empty:
        return mask

    disables = disables or []
    elements = [col for col in data.columns if col not in disables]
    element_atts = {
        element: attributes[element] for element in elements if element in attributes
    }

    validated_columns = []
    validated_dtypes = set(properties.numeric_types) | {"datetime", "key"}

    basic_functions = {
        "datetime": validate_datetime,
        "str": validate_str,
    }

    for column in data.columns:
        if column in disables or column not in attributes:
            continue

        series = data[column]
        column_atts = element_atts.get(column, {})
        column_type = column_atts.get("column_type")

        if column_type in properties.numeric_types:
            valid_min = column_atts.get("valid_min", -np.inf)
            valid_max = column_atts.get("valid_max", np.inf)
            column_mask = validate_numeric(series, valid_min, valid_max)
        elif column_type == "key":
            code_table_name = column_atts.get("codetable")
            code_table = codes.read_table(
                code_table_name, imodel=imodel, ext_table_path=ext_table_path
            )
            column_mask = validate_codes(series, code_table, column_type)
        elif column_type in basic_functions:
            column_mask = basic_functions[column_type](series)
        else:
            logging.warning(
                f"Unknown column_type '{column_type}' for column '{column}'"
            )
            continue

        mask[column] = column_mask
        if column_type in validated_dtypes:
            validated_columns.append(column)

    # Explicit boolean literals ("True"/"False") override validation results
    if validated_columns:
        validated_columns = list(dict.fromkeys(validated_columns))
        to_bool = data[validated_columns].applymap(convert_str_boolean)
        false_mask = to_bool.applymap(_is_false)
        true_mask = to_bool.applymap(_is_true)
        mask[validated_columns] = mask[validated_columns].mask(false_mask, False)
        mask[validated_columns] = mask[validated_columns].mask(true_mask, True)

    return mask.astype("boolean")
