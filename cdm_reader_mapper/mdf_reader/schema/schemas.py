"""
Manage data model schema files.

Functions to manage data model
schema files and objects according to the
requirements of the data reader tool

"""

from __future__ import annotations

import logging
import os

from cdm_reader_mapper.common.json_dict import collect_json_files, combine_dicts

from .. import properties


def convert_dtype_to_default(dtype, section, element):
    """Convert data type to defaults (int, float)."""
    if dtype is None:
        return
    elif dtype == "float":
        return dtype
    elif dtype == "int":
        return properties.pandas_int
    elif "float" in dtype.lower():
        logging.warning(
            f"Set column type of ({section}, {element}) from deprecated {dtype} to float."
        )
        return "float"
    elif "int" in dtype.lower():
        logging.warning(
            f"Set column type of ({section}, {element}) from deprecated {dtype} to int."
        )
        return properties.pandas_int
    return dtype


def _read_schema(schema):
    """DOCUMENTATION."""
    if not schema["header"]:
        if not schema["sections"]:
            logging.error(
                f"'sections' block needs to be defined in a schema with no header. Error in data model schema file {schema['name']}"
            )
            return
        schema["header"] = dict()

    if schema["header"].get("multiple_reports_per_line"):
        logging.error("Multiple reports per line data model: not yet supported")
        return

    # 3.2. Make no section formats be internally treated as 1 section format
    if not schema.get("sections"):
        if not schema.get("elements"):
            logging.error(
                f"Data elements not defined in data model schema file {schema['name']} under key 'elements' "
            )
            return
        schema["sections"] = {
            properties.dummy_level: {
                "header": {},
                "elements": schema.get("elements"),
            }
        }
        schema["header"]["parsing_order"] = [{"s": [properties.dummy_level]}]
        schema.pop("elements", None)
        schema["sections"][properties.dummy_level]["header"]["delimiter"] = schema[
            "header"
        ].get("delimiter")
        schema["header"].pop("delimiter", None)
        schema["sections"][properties.dummy_level]["header"]["field_layout"] = schema[
            "header"
        ].get("field_layout")
        schema["header"].pop("field_layout", None)
        schema["sections"][properties.dummy_level]["header"]["format"] = schema[
            "header"
        ].get("format")
        schema["header"].pop("format", None)

    # 3.3. Make parsing order explicit
    if not schema["header"].get("parsing_order"):  # assume sequential
        schema["header"]["parsing_order"] = [{"s": list(schema["sections"].keys())}]

    # 3.4. Make disable_read and field_layout explicit: this is ruled by delimiter being set,
    # unless explicitly set
    for section in schema["sections"].keys():
        if schema["sections"][section]["header"].get("disable_read"):
            continue
        else:
            schema["sections"][section]["header"]["disable_read"] = False
        if not schema["sections"][section]["header"].get("field_layout"):
            delimiter = schema["sections"][section]["header"].get("delimiter")
            schema["sections"][section]["header"]["field_layout"] = (
                "delimited" if delimiter else "fixed_width"
            )
        for element in schema["sections"][section]["elements"].keys():
            column_type = schema["sections"][section]["elements"][element].get(
                "column_type"
            )
            schema["sections"][section]["elements"][element]["column_type"] = (
                convert_dtype_to_default(
                    column_type,
                    section,
                    element,
                )
            )
    return schema


def read_schema(
    data_model=None, release=None, deck=None, ext_schema_path=None, ext_schema_file=None
):
    """
    Read a data model schema file.

    Read a data model schema file to a dictionary and
    completes it by adding explicitly information the
    reader tool needs

    Keyword Arguments
    -----------------
    data_model: str, optional
        The name of the data model to read. This is for
        data models included in the tool
    release: str, optional
        The name of the data model release. If chosen, overwrite data model schema.
        `data_model` is needed.
    deck: str, optional
        The name of the data model deck. If chosen, overwrite data model release schema.
        `data_model` is needed.
    ext_schema_path: str, optional
        The path to the external data model schema file
    ext_schema_file: str, optional
        The external data model schema file


    Either schema_name or one of ext_schema_path or ext_schema_file must be provided.


    Returns
    -------
    dict
        Data model schema

    """
    # 1. Validate input
    if data_model:
        schema_files = collect_json_files(data_model, release, deck)
    else:
        if ext_schema_file is None:
            schema_path = os.path.abspath(ext_schema_path)
            schema_name = os.path.basename(schema_path)
            schema_files = os.path.join(schema_path, schema_name + ".json")
        else:
            schema_files = ext_schema_file

        if not os.path.isfile(schema_files):
            logging.error(f"Can't find input schema file {schema_files}")
            return

    if isinstance(schema_files, str):
        schema_files = [schema_files]

    # 2. Get schema
    schema = combine_dicts(schema_files)
    schema["name"] = schema_files

    # 3. Expand schema
    # Fill in the initial schema to "full complexity": to homogenize schema,
    # explicitly add info that is implicit to given situations/data models

    # One report per record: make sure later changes are reflected in MULTIPLE
    # REPORTS PER RECORD case below if we ever use it!
    # Currently only supported case: one report per record (line)
    # 3.1. First check for no header case: sequential sections
    return _read_schema(schema)


def df_schema(df_columns, schema):
    """
    Create simple data model schema dictionary.

    Create a simple attribute dictionary for the elements
    in a dataframe from its data model schema

    Parameters
    ----------
    df_columns : list
        The columns in the data frame (data elements from
        the data model)
    schema : dict
        The data model schema


    Returns
    -------
    dict
        Data elements attributes

    """

    def clean_schema(columns, schema):
        # Could optionally add cleaning of element descriptors that only apply
        # to the initial reading of the data model: field_length, etc....
        for element in list(schema):
            if element not in columns:
                schema.pop(element)

    def get_index(idx, lst, section):
        if len(lst) == 1:
            return idx
        return (section, idx)

    flat_schema = dict()
    for section in schema.get("sections"):
        if schema["sections"].get(section).get("header").get("disable_read"):
            flat_schema.update({section: {"column_type": "object"}})
        else:
            flat_schema.update(
                {
                    get_index(x, list(schema.get("sections")), section): schema[
                        "sections"
                    ]
                    .get(section)
                    .get("elements")
                    .get(x)
                    for x in schema["sections"].get(section).get("elements")
                }
            )

    clean_schema(df_columns, flat_schema)
    return flat_schema
