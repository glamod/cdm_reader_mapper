"""Validate entries."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .. import properties
from ..codes import codes
from ..schemas import schemas
from .utilities import convert_str_boolean


def validate_datetime(elements, data) -> pd.DataFrame:
    """DOCUMENTATION."""

    def is_date_object(object):
        if hasattr(object, "year"):
            return True

    mask = pd.DataFrame(index=data.index, data=False, columns=elements)
    mask[elements] = (
        data[elements].apply(np.vectorize(is_date_object)) | data[elements].isna()
    )
    return mask


def validate_numeric(elements, data, schema) -> pd.DataFrame:
    """DOCUMENTATION."""

    # Find thresholds in schema. Flag if not available -> warn
    def _to_numeric(x):
        if x is None:
            return np.nan
        x = convert_str_boolean(x)
        if isinstance(x, bool):
            return x
        try:
            return float(x)
        except ValueError:
            return False

    data[elements] = data[elements].map(_to_numeric)
    mask = pd.DataFrame(index=data.index, data=False, columns=elements)
    lower = {x: schema.get(x).get("valid_min", -np.inf) for x in elements}
    upper = {x: schema.get(x).get("valid_max", np.inf) for x in elements}

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
    mask[elements] = (
        (data[elements] >= [lower.get(x) for x in elements])
        & (data[elements] <= [upper.get(x) for x in elements])
    ) | data[elements].isna()
    return mask


def validate_str(elements, data) -> pd.DataFrame:
    """DOCUMENTATION."""
    return pd.DataFrame(index=data.index, data=True, columns=elements)


def validate_codes(elements, data, schema, imodel, ext_table_path) -> pd.DataFrame:
    """DOCUMENTATION."""
    mask = pd.DataFrame(index=data.index, data=False, columns=elements)
    for element in elements:
        code_table_name = schema.get(element).get("codetable")
        if not code_table_name:
            logging.error(f"Code table not defined for element {element}")
            logging.warning("Element mask set to False")
            continue

        table = codes.read_table(
            code_table_name,
            imodel=imodel,
            ext_table_path=ext_table_path,
        )
        if not table:
            continue

        dtype = properties.pandas_dtypes.get(schema.get(element).get("column_type"))

        table_keys = list(table.keys())
        validation_df = data[element]
        value = validation_df.astype(dtype).astype("str")
        valid = validation_df.notna()
        mask_ = value.isin(table_keys)
        mask[element] = mask_.where(valid, True) | validation_df.isna()

    return mask


def _get_elements(elements, element_atts, key) -> list[str]:
    def _condition(x):
        column_types = element_atts.get(x).get("column_type")
        if key == "numeric_types":
            return column_types in properties.numeric_types
        return column_types == key

    return [x for x in elements if _condition(x)]


def _element_tuples(numeric_elements, datetime_elements, coded_elements) -> bool:
    ele_tpl = [
        isinstance(x, tuple)
        for x in numeric_elements + datetime_elements + coded_elements
    ]
    return any(ele_tpl)


def _mask_boolean(x, boolean) -> bool:
    x = convert_str_boolean(x)
    if x is boolean:
        return True
    return False


def validate(
    data,
    imodel,
    ext_table_path,
    schema,
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
    if not isinstance(data, pd.DataFrame):  # or not isinstance(mask0, pd.DataFrame):
        # logging.error("Input data and mask must be a pandas data frame object")
        logging.error("input data must be a pandas DataFrame.")
        return

    mask = pd.DataFrame(index=data.index, columns=data.columns, dtype=object)
    if data.empty:
        return mask

    # Get the data elements from the input data: might be just a subset of
    # data model and flatten the schema to get a simple and sequential list
    # of elements included in the input data
    elements = [x for x in data if x not in disables]
    element_atts = schemas.df_schema(elements, schema)

    # See what elements we need to validate
    numeric_elements = _get_elements(elements, element_atts, "numeric_types")
    datetime_elements = _get_elements(elements, element_atts, "datetime")
    coded_elements = _get_elements(elements, element_atts, "key")
    str_elements = _get_elements(elements, element_atts, "str")

    if _element_tuples(numeric_elements, datetime_elements, coded_elements):
        validated_columns = pd.MultiIndex.from_tuples(
            list(set(numeric_elements + coded_elements + datetime_elements))
        )
    else:
        validated_columns = list(
            set(numeric_elements + coded_elements + datetime_elements)
        )

    mask[numeric_elements] = validate_numeric(numeric_elements, data, element_atts)

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
    if len(coded_elements) > 0:
        mask[coded_elements] = validate_codes(
            coded_elements,
            data,
            element_atts,
            imodel,
            ext_table_path,
        )

    # 3. Datetime elements
    mask[datetime_elements] = validate_datetime(datetime_elements, data)

    # 4. str elements
    mask[str_elements] = validate_str(str_elements, data)

    # 5. Set False values
    mask[validated_columns] = mask[validated_columns].mask(
        data[validated_columns].map(_mask_boolean, boolean=False),
        False,
    )

    mask[validated_columns] = mask[validated_columns].mask(
        data[validated_columns].map(_mask_boolean, boolean=True),
        True,
    )

    mask[disables] = np.nan
    return mask
