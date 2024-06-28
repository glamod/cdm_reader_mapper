"""
Manage data model schema files.

Functions to manage data model
schema files and objects according to the
requirements of the data reader tool

"""

from __future__ import annotations

import json
import logging
import os

from cdm_reader_mapper.common.getting_files import get_files

from .. import properties


def _read_schema(schema, schema_file=""):
    """DOCUMENTATION."""
    if not schema["header"]:
        if not schema["sections"]:
            logging.error(
                f"'sections' block needs to be defined in a schema with no header. Error in data model schema file {schema_file}"
            )
            return
        schema["header"] = dict()

    if not schema["header"].get("multiple_reports_per_line"):
        # 3.2. Make no section formats be internally treated as 1 section format
        if not schema.get("sections"):
            if not schema.get("elements"):
                logging.error(
                    f"Data elements not defined in data model schema file {schema_file} under key 'elements' "
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
            schema["sections"][properties.dummy_level]["header"][
                "field_layout"
            ] = schema["header"].get("field_layout")
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
                if (
                    schema["sections"][section]["elements"][element].get("column_type") == "int"
                ):
                    schema["sections"][section]["elements"][element].update(
                        {"column_type": "Int64"}
                    )
        return schema
    else:
        logging.error("Multiple reports per line data model: not yet supported")
        return


def read_schema(schema_name=None, ext_schema_path=None, ext_schema_file=None):
    """
    Read a data model schema file.

    Read a data model schema file to a dictionary and
    completes it by adding explicitly information the
    reader tool needs

    Keyword Arguments
    -----------------
    schema_name : str, optional
        The name of data model to read. This is for
        data models included in the tool
    ext_schema_path : str, optional
        The path to the external data model schema file
    ext_schema_file : str, optional
        The external data model schema file


    Either schema_name or one of ext_schema_path or ext_schema_file must be provided.


    Returns
    -------
    dict
        Data model schema

    """
    # 1. Validate input
    if schema_name:
        if schema_name not in properties.supported_data_models:
            print(f"ERROR: \n\tInput data model {schema_name} not supported.")
            print(
                "See mdf_reader.properties.supported_data_models for supported data models"
            )
            return
        else:
            schema_path = f"{properties._base}.schema"
            schema_data = None
            try:
                schema_data = get_files(schema_path)
            except ModuleNotFoundError:
                logging.error(f"Can't find input schema files in {schema_data}")
                return
            schema_file = list(schema_data.glob(f"{schema_name}.json"))[0]
    else:
        if ext_schema_file is None:
            schema_path = os.path.abspath(ext_schema_path)
            schema_name = os.path.basename(schema_path)
            schema_file = os.path.join(schema_path, schema_name + ".json")
        else:
            schema_file = ext_schema_file

        if not os.path.isfile(schema_file):
            logging.error(f"Can't find input schema file {schema_file}")
            return

    # 2. Get schema
    with open(schema_file) as fileObj:
        schema = json.load(fileObj)

    # 3. Expand schema
    # Fill in the initial schema to "full complexity": to homogenize schema,
    # explicitly add info that is implicit to given situations/data models

    # One report per record: make sure later changes are reflected in MULTIPLE
    # REPORTS PER RECORD case below if we ever use it!
    # Currently only supported case: one report per record (line)
    # 3.1. First check for no header case: sequential sections
    return _read_schema(schema, schema_file=schema_file)


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
