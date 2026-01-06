"""Data validation module."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from .. import properties
from ..codes import codes
from .utilities import convert_str_boolean


def _is_false(x):
    return x is False


def _is_true(x):
    return x is True


def validate_datetime(series):
    dates = pd.to_datetime(series, errors="coerce")
    return dates.notna() | series.isna()


def validate_numeric(series, valid_min, valid_max):
    converted = series.apply(convert_str_boolean)
    numeric = pd.to_numeric(converted, errors="coerce")
    valid_range = (numeric >= valid_min) & (numeric <= valid_max)
    return valid_range | numeric.isna()


def validate_str(series):
    return pd.Series(True, index=series.index, dtype="boolean")


def validate_codes(series, code_table, column_type):
    if not code_table:
        logging.error(f"Code table not found for element {series.name}")
        return pd.Series(False, index=series.index)

    keys = set(code_table)
    dtype = properties.pandas_dtypes.get(column_type, object)
    converted = series.astype(dtype)
    as_str = converted.astype(str)
    return converted.isna() | as_str.isin(keys)


def validate(
    data,
    imodel,
    ext_table_path,
    attributes,
    disables=None,
) -> pd.DataFrame:
    """Validate data.

    Parameters
    ----------
    data: pd.DataFrame
      DataFrame for validation.

    imodel: str
      Name of internally available input data model.
      e.g. icoads_r300_d704

    ext_table_path: str
      Path to the code tables for an external data model

    attributes: dict
      Data model attributes.

    disables: list, optional
      List of column names to be ignored.

    Returns
    -------
    pd.DataFrame
      Validated boolean mask.
    """
    if not isinstance(data, pd.DataFrame):
        logging.error("input data must be a pandas DataFrame.")
        return None

    mask = pd.DataFrame(pd.NA, index=data.index, columns=data.columns, dtype="boolean")
    if data.empty:
        return mask

    disables = disables or []
    elements = [c for c in data.columns if c not in disables]
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
        if column in disables:
            continue

        if column not in attributes:
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
            column_mask = validate_codes(
                series,
                code_table,
                column_type,
            )
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
