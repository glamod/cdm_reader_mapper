"""Validate entries."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cdm_reader_mapper.common.json_dict import get_table_keys

from .. import properties
from ..codes import codes
from ..schemas import schemas


def validate_datetime(elements, data):
    """DOCUMENTATION."""

    def is_date_object(object):
        if hasattr(object, "year"):
            return True

    mask = pd.DataFrame(index=data.index, data=False, columns=elements)
    mask[elements] = (
        data[elements].apply(np.vectorize(is_date_object)) | data[elements].isna()
    )
    return mask


def validate_numeric(element, data, schema):
    """DOCUMENTATION."""
    # Find thresholds in schema. Flag if not available -> warn
    lower = schema.get(element).get("valid_min", -np.inf)
    upper = schema.get(element).get("valid_max", np.inf)
    return (data >= lower) & (data <= upper) | (data == np.nan)


def validate_codes(element, data, schema, imodel, ext_table_path):
    """DOCUMENTATION."""
    code_table_name = schema.get(element).get("codetable")
    if not code_table_name:
        return False

    table = codes.read_table(
        code_table_name,
        imodel=imodel,
        ext_table_path=ext_table_path,
    )
    if not table:
        return False

    table_keys = get_table_keys(table)
    table_keys_str = ["~".join(x) if isinstance(x, list) else x for x in table_keys]
    for table_key_str in table_keys_str:
        if "~" in table_key_str:
            logging.warning(f"Cross checks for {element} not implemented now!")
            return True

    if isinstance(data, (list, tuple)):
        data = "~".join(data)

    if data in table_keys_str:
        return True
    return False


def _get_elements(element, element_atts):
    def _condition(element, etype):
        column_types = element_atts.get(element).get("column_type")
        if etype == "numeric_types":
            return column_types in properties.numeric_types
        return column_types == etype

    for etype in ["numeric_types", "datetime", "key", "str"]:
        if _condition(element, etype):
            return {"element": element, "etype": etype}


def isnan(data):
    """Returns bool value if data is valid value."""
    if data is None:
        return True
    if isinstance(data, str):
        return False
    if np.isnan(data):
        return True
    return False


def validate(
    data,
    mask0,
    imodel,
    index,
    ext_table_path,
    schema,
):
    """Validate data.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame for validation.
    mask0: pd.DataFrame
        Boolean mask.
    imodel: str
        Name of internally available input data model.
        e.g. icoads_r300_d704
    ext_table_path: str
        Path to the code tables for an external data model
    schema: dict
        Data model schema.

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

    element_atts = schemas.df_schema([index], schema)

    # See what elements we need to validate
    element_dict = _get_elements(index, element_atts)
    if not element_dict:
        return None

    element = element_dict["element"]
    etype = element_dict["etype"]

    if isnan(data):
        mask = True
    elif etype == "numeric_types":
        mask = validate_numeric(element, data, element_atts)
    elif etype == "key":
        mask = validate_codes(element, data, element_atts, imodel, ext_table_path)
    elif etype == "datetime":
        mask = validate_datetime(element, data)
    elif etype == "str":
        mask = True
    else:
        logging.error(f"{etype} is not a valid data type")
        return

    if etype in ["numeric_types", "key", "datetime"]:
        if mask0 is False:
            mask = False

    return mask
