"""Validate entries."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cdm_reader_mapper.common.json_dict import get_table_keys

from . import properties
from .codes import codes
from .schemas import schemas


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


def validate_numeric(elements, data, schema):
    """DOCUMENTATION."""
    # Find thresholds in schema. Flag if not available -> warn
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


def validate_str(elements, data):
    """DOCUMENTATION."""
    return pd.DataFrame(index=data.index, data=True, columns=elements)


def validate_codes(elements, data, schema, imodel, ext_table_path, supp=False):
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
        if supp:
            key_elements = (
                [element[1]]
                if not table.get("_keys")
                else list(table["_keys"].get(element[1]))
            )
        else:
            key_elements = (
                [element]
                if not table.get("_keys")
                else list(table["_keys"].get(element))
            )
        dtypes = {
            x: properties.pandas_dtypes.get(schema.get(x).get("column_type"))
            for x in key_elements
        }

        table_keys = get_table_keys(table)
        table_keys_str = ["~".join(x) if isinstance(x, list) else x for x in table_keys]
        validation_df = data[key_elements]
        imask = pd.Series(index=data.index, data=True)
        val = validation_df.notna()
        val = val.all(axis=1)
        masked = np.where(val)
        masked = masked[0]
        value = validation_df.iloc[masked, :]
        value = value.astype(dtypes).astype("str")
        value = value.apply("~".join, axis=1)
        value = value.isin(table_keys_str)
        if masked.size != 0:
            imask.iloc[masked] = value
        mask[element] = imask

    return mask


def validate(
    data,
    mask0,
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
    mask0: pd.DataFrame
        Boolean mask.
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
    if not isinstance(data, pd.DataFrame) or not isinstance(mask0, pd.DataFrame):
        logging.error("Input data and mask must be a pandas data frame object")
        return

    # Get the data elements from the input data: might be just a subset of
    # data model and flatten the schema to get a simple and sequential list
    # of elements included in the input data
    elements = [x for x in data if x not in disables]
    element_atts = schemas.df_schema(elements, schema)

    # See what elements we need to validate
    numeric_elements = [
        x
        for x in elements
        if element_atts.get(x).get("column_type") in properties.numeric_types
    ]
    datetime_elements = [
        x for x in elements if element_atts.get(x).get("column_type") == "datetime"
    ]
    coded_elements = [
        x for x in elements if element_atts.get(x).get("column_type") == "key"
    ]
    str_elements = [
        x for x in elements if element_atts.get(x).get("column_type") == "str"
    ]

    if any(
        [
            isinstance(x, tuple)
            for x in numeric_elements + datetime_elements + coded_elements
        ]
    ):
        validated_columns = pd.MultiIndex.from_tuples(
            list(set(numeric_elements + coded_elements + datetime_elements))
        )
    else:
        validated_columns = list(
            set(numeric_elements + coded_elements + datetime_elements)
        )

    mask = pd.DataFrame(index=data.index, columns=data.columns, dtype=object)

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
    mask[datetime_elements] = validate_datetime(
        datetime_elements, data
    )  # data[datetime_elements].notna()

    # 4. str elements
    mask[str_elements] = validate_str(str_elements, data)

    # 5. Set False values
    mask0_n = mask0[validated_columns].fillna(False)
    mask[validated_columns] = mask[validated_columns].mask(
        ~mask0_n,
        False,
    )
    for column in disables:
        mask[column] = np.nan
    return mask
