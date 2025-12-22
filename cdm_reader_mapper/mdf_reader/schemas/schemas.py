"""
Manage data model schema files.

Functions to manage data model
schema files and objects according to the
requirements of the data reader tool

"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from cdm_reader_mapper.common.json_dict import collect_json_files, combine_dicts

from .. import properties


def make_dummy_sections(schema):
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


def make_parsing_order(schema):
    if not schema["header"].get("parsing_order"):  # assume sequential
        schema["header"]["parsing_order"] = [{"s": list(schema["sections"].keys())}]


def read_schema(imodel=None, ext_schema_path=None, ext_schema_file=None) -> dict:
    """
    Read a data model schema file.

    Read a data model schema file to a dictionary and
    completes it by adding explicitly information the
    reader tool needs

    Parameters
    ----------
    imodel: str, optional
        Name of internally available input data model.
        e.g. icoads_r300_d704
    ext_schema_path: str, optional
        The path to the external input data model schema file.
        The schema file must have the same name as the directory.
        One of ``imodel`` and ``ext_schema_path`` or ``ext_schema_file`` must be set.
    ext_schema_file: str, optional
        The external input data model schema file.
        One of ``imodel`` and ``ext_schema_path`` or ``ext_schema_file`` must be set.

    Returns
    -------
    dict
        Data model schema
    """
    # 1. Validate input
    if ext_schema_file:
        if not os.path.isfile(ext_schema_file):
            logging.error(f"Can't find input schema file {ext_schema_file}")
            return
        schema_files = Path(ext_schema_file)
    elif ext_schema_path:
        schema_path = os.path.abspath(ext_schema_path)
        schema_name = os.path.basename(schema_path)
        schema_files = os.path.join(schema_path, schema_name + ".json")
        if not os.path.isfile(schema_files):
            logging.error(f"Can't find input schema file {schema_files}")
            return
        schema_files = Path(schema_files)
    elif imodel:
        imodel = imodel.split("_")
        if imodel[0] not in properties.supported_data_models:
            logging.error("Input data model " f"{imodel[0]}" " not supported")
            return
        schema_files = collect_json_files(*imodel, base=f"{properties._base}.schemas")
    else:
        raise ValueError(
            "One of ['imodel', 'ext_schema_path', 'ext_schema_file'] must be set."
        )

    if isinstance(schema_files, Path):
        schema_files = [schema_files]

    # 2. Get schema
    schema = combine_dicts(schema_files, base=f"{properties._base}.schemas")
    schema["name"] = schema_files

    if not schema["header"]:
        if not schema["sections"]:
            raise KeyError(
                f"'sections' block needs to be defined in a schema with no header. Error in data model schema file {schema['name']}"
            )
        schema["header"] = dict()

    if schema["header"].get("multiple_reports_per_line"):
        raise NotImplementedError(
            "Multiple reports per line data model: not yet supported"
        )

    make_dummy_sections(schema)
    make_parsing_order(schema)

    return schema
