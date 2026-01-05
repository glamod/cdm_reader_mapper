"""Validate entries."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .. import properties
from ..codes import codes
from .utilities import convert_str_boolean

def _get_elements(elements, element_atts, key) -> list[str]:
    """Select elements by schema types."""
    if key == "numeric_types":
        return [
            e for e in elements
            if element_atts.get(e, {}).get("column_type")
            in properties.numeric_types
        ]

    return [
        e for e in elements
        if element_atts.get(e, {}).get("column_type") == key
    ]


def _element_tuples(*groups) -> bool:
    """Check whether any element name is a tuple (MultiIndex)."""
    return any(isinstance(x, tuple) for group in groups for x in group)


def _mask_boolean(x, boolean) -> bool:
    x = convert_str_boolean(x)
    return x is boolean

def validate_datetime(elements, data) -> pd.DataFrame:
    def is_date_object(object):
        return hasattr(object, "year")

    df = data[elements]
    return df.applymap(is_date_object) | df.isna()


def validate_numeric(elements, data, schema) -> pd.DataFrame:
    def _to_numeric(x):
        if x is None:
            return np.nan
        x = convert_str_boolean(x)
        if isinstance(x, bool):
            return x
        try:
            return float(x)
        except (ValueError, TypeError):
            return False
            
    df = data[elements].applymap(_to_numeric)
    
    lower = pd.Series(
      {x: schema.get(x).get("valid_min", -np.inf) for x in elements}
    )
    upper = pd.Series(
      {x: schema.get(x).get("valid_max", np.inf) for x in elements}
    )
    
    missing_bounds = lower.eq(-np.inf) | upper.eq(np.inf)

    if missing_bounds.any():
        logging.warning(
            "Data numeric elements with missing upper or lower threshold: %s",
                ",".join(map(str, lower.index[missing_bounds])),
        )
        logging.warning(
            "Corresponding upper and/or lower bounds set to +/-inf for validation"
        )
        
    return ((df >= lower) & (df <= upper)) | df.isna()    


def validate_str(elements, data) -> pd.DataFrame:
    return pd.DataFrame(index=data.index, data=True, columns=elements)


def validate_codes(elements, data, schema, imodel, ext_table_path) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=data.index, columns=elements)
    
    for element in elements:
        code_table_name = schema.get(element).get("codetable")
        if not code_table_name:
            logging.error(f"Code table not defined for element {element}")
            continue  
            
        table = codes.read_table(
            code_table_name,
            imodel=imodel,
            ext_table_path=ext_table_path,        
        )  
        if not table:
          continue   
          
        dtype = properties.pandas_dtypes.get(schema.get(element).get("column_type"), object)
        
        series = data[element]
        valid = series.notna()
        value = series.astype(dtype).astype(str)
        
        mask[element] = value.isin(set(table)).where(valid, True)
    
    return mask 


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

    numeric_elements  = _get_elements(elements, element_atts, "numeric_types")
    datetime_elements = _get_elements(elements, element_atts, "datetime")
    coded_elements    = _get_elements(elements, element_atts, "key")
    str_elements      = _get_elements(elements, element_atts, "str")
    
    validated_columns = list(
        set(numeric_elements + coded_elements + datetime_elements)
    )

    if _element_tuples(numeric_elements, datetime_elements, coded_elements):
        validated_columns = pd.MultiIndex.from_tuples(validated_columns)

    if numeric_elements:
        mask[numeric_elements] = validate_numeric(numeric_elements, data, element_atts)

    if coded_elements:
        mask[coded_elements] = validate_codes(
            coded_elements,
            data,
            element_atts,
            imodel,
            ext_table_path,
        )
        
    if datetime_elements:
      mask[datetime_elements] = validate_datetime(datetime_elements, data)

    if str_elements:
      mask[str_elements] = validate_str(str_elements, data)
      
    false_mask = data[validated_columns].map(_mask_boolean, boolean=False)
    true_mask  = data[validated_columns].map(_mask_boolean, boolean=True)
    
    mask[validated_columns] = mask[validated_columns].mask(false_mask, False)
    mask[validated_columns] = mask[validated_columns].mask(true_mask, True)
    
    mask.loc[:, mask.columns.intersection(disables)] = pd.NA
    
    return mask.astype("boolean")
