"""Validate entries."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from .. import properties
from ..codes import codes
from ..schemas import schemas
from .utilities import convert_str_boolean


def validate_datetime(mask, elements, data: pl.DataFrame):
    """DOCUMENTATION."""
    for element in elements:
        col = data.get_column(element)
        if not col.dtype.is_temporal():
            mask = mask.with_columns(pl.lit(False).alias(element))
            continue

        mask = mask.with_columns((pl.col(element) & col.is_not_null()).alias(element))

    return mask


def validate_numeric(mask: pl.DataFrame, elements, data: pl.DataFrame, schema):
    """DOCUMENTATION."""
    # Handle cases where value is explicitly None in the dictionary
    lower = {x: schema.get(x).get("valid_min") or -np.inf for x in elements}
    upper = {x: schema.get(x).get("valid_max") or np.inf for x in elements}

    set_elements = [
        x for x in lower.keys() if lower.get(x) != -np.inf and upper.get(x) != np.inf
    ]

    if len([x for x in elements if x not in set_elements]) > 0:
        logging.warning(
            "Data numeric elements with missing upper or lower threshold: {}".format(
                ",".join([str(x) for x in elements if x not in set_elements])
            )
        )
        logging.warning(
            "Corresponding upper and/or lower bounds set to +/-inf for validation"
        )

    mask = mask.with_columns(
        [
            (
                pl.col(element)
                | (
                    data[element].is_not_null()
                    & data[element].is_not_nan()
                    & data[element].is_between(
                        lower.get(element), upper.get(element), closed="both"
                    )
                )
            ).alias(element)
            for element in elements
        ]
    )
    return mask


def validate_str(mask, elements, data):
    """DOCUMENTATION."""
    return mask


def validate_codes(
    mask: pl.DataFrame,
    elements: list[str],
    data: pl.DataFrame,
    schema,
    imodel,
    ext_table_path,
):
    """DOCUMENTATION."""
    for element in elements:
        code_table_name = schema.get(element).get("codetable")
        if not code_table_name:
            logging.error(f"Code table not defined for element {element}")
            logging.warning("Element mask set to False")
            mask = mask.with_columns(pl.lit(False).alias(element))
            continue

        table = codes.read_table(
            code_table_name,
            imodel=imodel,
            ext_table_path=ext_table_path,
        )
        if not table:
            continue

        dtype = properties.polars_dtypes.get(schema.get(element).get("column_type"))

        table_keys = list(table.keys())
        col = data.get_column(element)
        col = col.cast(dtype).cast(pl.String)
        valid = col.is_not_null() & col.is_not_nan() & col.is_in(table_keys)
        mask = mask.with_columns((pl.col(element) | valid).alias(element))

    return mask


def _get_elements(elements, element_atts, key) -> list[str]:
    def _condition(x):
        column_types = element_atts.get(x).get("column_type")
        if key == "numeric_types":
            return column_types in properties.numeric_types
        return column_types == key

    return [x for x in elements if _condition(x)]


def _element_tuples(numeric_elements, datetime_elements, coded_elements):
    ele_tpl = [
        isinstance(x, tuple)
        for x in numeric_elements + datetime_elements + coded_elements
    ]
    return any(ele_tpl)


def _mask_boolean(x, boolean):
    x = convert_str_boolean(x)
    if x is boolean:
        return True
    return False


def validate(
    data: pl.DataFrame,
    mask: pl.DataFrame,
    imodel,
    ext_table_path,
    schema,
    disables=None,
):
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
    schema: dict
        Data model schema.
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
    # Check input
    if not isinstance(data, pl.DataFrame) or not isinstance(mask, pl.DataFrame):
        # logging.error("Input data and mask must be a pandas data frame object")
        logging.error("input data must be a pandas DataFrame.")
        return

    if data.is_empty():
        return mask

    disables = disables or []

    # Get the data elements from the input data: might be just a subset of
    # data model and flatten the schema to get a simple and sequential list
    # of elements included in the input data
    elements = [x for x in data.columns if x not in disables]
    element_atts = schemas.df_schema(elements, schema)

    # 1. Numeric elements
    numeric_elements = _get_elements(elements, element_atts, "numeric_types")
    mask = validate_numeric(mask, numeric_elements, data, element_atts)

    # 2. Table coded elements
    # See following: in multiple keys code tables, the non parameter element,
    # won't have a code_table attribute in the element_atts:
    # So we need to check the code_table.keys files in addition to the element_atts
    # Additionally, a YEAR key can fail in one table, but be compliant with anbother, then, how would we mask this?
    #               also, a YEAR defined as an integer, will undergo its own check.....
    # So I think we need to check nested keys as a whole, and mask only the actual parameterized element:
    # Get the full list of keys combinations (tuples, triplets...) and check the column combination against that: if it fails, mark the element!
    # Need to see how to grab the YEAR part of a datetime when YEAR comes from a datetime element
    # pd.DatetimeIndex(df['_datetime']).year
    coded_elements = _get_elements(elements, element_atts, "key")
    if len(coded_elements) > 0:
        mask = validate_codes(
            mask,
            coded_elements,
            data,
            element_atts,
            imodel,
            ext_table_path,
        )

    # 3. Datetime elements
    datetime_elements = _get_elements(elements, element_atts, "datetime")
    mask = validate_datetime(mask, datetime_elements, data)

    # 4. str elements
    str_elements = _get_elements(elements, element_atts, "str")
    mask = validate_str(mask, str_elements, data)

    mask = mask.with_columns([pl.lit(None).alias(disable) for disable in disables])
    return mask
