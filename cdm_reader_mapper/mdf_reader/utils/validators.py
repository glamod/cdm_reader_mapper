"""Data validation module."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from .. import properties
from ..codes import codes
from .utilities import convert_str_boolean


def _mask_boolean(x, boolean) -> bool:
    x = convert_str_boolean(x)
    return x is boolean


def validate_datetime(series):
    dates = pd.to_datetime(series, errors="coerce")
    return dates.notna() | series.isna()


def validate_numeric(series, atts):
    converted = series.map(convert_str_boolean)
    numeric = pd.to_numeric(converted, errors="coerce")
    lower = atts.get("valid_min", -np.inf)
    upper = atts.get("valid_max", np.inf)
    return ((numeric >= lower) & numeric <= upper) | numeric.isna()


def validate_str(series):
    return pd.Series(True, index=series.index)


def validate_codes(series, atts, imodel, ext_table_path):
    code_table_name = atts.get("codetable")
    if not code_table_name:
        logging.error(f"Code table not defined for element {series.name}")
        return pd.Series(False, index=series.index)

    table = codes.read_table(
        code_table_name, imodel=imodel, ext_table_path=ext_table_path
    )
    if not table:
        logging.error(
            f"Code table not found for element {series.name} in {ext_table_path}"
        )
        return pd.Series(False, index=series.index)

    keys = set(table)
    dtype = properties.pandas_dtypes.get(atts.get("column_type"), object)
    converted = series.astype(dtype)
    return converted.isna() | converted.astype(str).isin(keys)


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
    logging.basicConfig(
        format="%(levelname)s\t[%(asctime)s](%(filename)s)\t%(message)s",
        level=logging.INFO,
        datefmt="%Y%m%d %H:%M:%S",
        filename=None,
    )

    if not isinstance(data, pd.DataFrame):
        logging.error("input data must be a pandas DataFrame.")
        return

    mask = pd.DataFrame(index=data.index, columns=data.columns, dtype="boolean")
    if data.empty:
        return mask

    disables = disables or []
    elements = [c for c in data.columns if c not in disables]
    element_atts = {element: attributes[element] for element in elements}

    validated_columns = []
    for column in data.columns:
        series = data[column]
        column_atts = attributes[column]
        column_type = column_atts.get("column_type")

        if column_type in properties.numeric_types:
            column_mask = validate_numeric(series, element_atts.get(column, {}))
        elif column_type == "datetime":
            column_mask = validate_datetime(series)
        elif column_type == "str":
            column_mask = validate_str(series)
        elif column_type == "key":
            column_mask = validate_codes(
                series, element_atts.get(column), imodel, ext_table_path
            )
        else:
            continue

        mask[column] = column_mask
        if column_type != "str":
            validated_columns.append(column)

    false_mask = data[validated_columns].map(_mask_boolean, boolean=False)
    true_mask = data[validated_columns].map(_mask_boolean, boolean=True)
    mask[validated_columns] = mask[validated_columns].mask(false_mask, False)
    mask[validated_columns] = mask[validated_columns].mask(true_mask, True)
    mask.loc[:, mask.columns.intersection(disables)] = pd.NA
    return mask.astype("boolean")
