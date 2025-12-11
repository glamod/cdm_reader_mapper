"""
Climate Model Data (CDM) mapping table routinies.

Created on Thu Apr 11 13:45:38 2019

Module to handle C3S Climate Data Store Common Data Model (CMD) tables within
the cdm tool.

@author: iregon
"""

from __future__ import annotations

from copy import deepcopy

from typing import Any

from cdm_reader_mapper.common.json_dict import (
    collect_json_files,
    combine_dicts,
    open_json_file,
)

from .. import properties


def get_cdm_atts(
    cdm_tables: str | list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Get CDM attribute tables.

    Parameters
    ----------
    cdm_tables : str or list of str, optional
        List of CDM tables to retrieve.
        - If `None`, includes all tables defined in `properties.cdm_tables`.
        - If a string, treated as a single table name.
        - If an empty list, returns an empty mapping.

    Returns
    -------
    dict
        Dictionary mapping table names to their attribute dictionaries.
        Keys are table names like `header` or `observations-*`.
        Values are dictionaries loaded from JSON files.
    """
    header_file = collect_json_files(
        "common", base=f"{properties._base}.tables", name="header"
    )[0]
    header_dict = open_json_file(header_file)

    observations_file = collect_json_files(
        "common", base=f"{properties._base}.tables", name="observations"
    )[0]
    observations_dict = open_json_file(observations_file)

    if cdm_tables is None:
        cdm_table_list = properties.cdm_tables
    elif isinstance(cdm_tables, str):
        cdm_table_list = [cdm_tables]
    else:
        cdm_table_list = cdm_tables

    cdm_atts = {}
    for cdm_table in cdm_table_list:
        if cdm_table == "header":
            cdm_atts[cdm_table] = deepcopy(header_dict)
        else:
            cdm_atts[cdm_table] = deepcopy(observations_dict)

    return cdm_atts


def get_imodel_maps(
    data_model: str,
    *sub_models: str,
    cdm_tables: str | list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Retrieve CDM attribute maps for a data model and optional submodels.

    Parameters
    ----------
    data_model : str
        The main data model name, e.g., `icoads`.
    sub_models : str
        Optional submodel names, e.g. `r300`, `d721`.
    cdm_tables : str or list of str, optional
        List of CDM tables to retrieve.
        - If `None`, includes all tables defined in `properties.cdm_tables`.
        - If a string, treated as a single table name.
        - If an empty list, returns an empty mapping.

    Returns
    -------
    dict
        Mapping of table names to their attribute dictionaries.
        Each table dictionary may have its `elements` normalized to lists,
        and tuples of (section, element) if sections exist.
    """
    if cdm_tables is None:
        cdm_table_list = properties.cdm_tables
    elif isinstance(cdm_tables, str):
        cdm_table_list = [cdm_tables]
    else:
        cdm_table_list = cdm_tables

    imodel_maps = {}
    observations_files = []

    for cdm_table in cdm_table_list:
        cdm_files = collect_json_files(
            data_model, *sub_models, base=f"{properties._base}.tables", name=cdm_table
        )

        if not observations_files:
            observations_files = collect_json_files(
                data_model,
                *sub_models,
                base=f"{properties._base}.tables",
                name="observations",
            )

        if "observations" in cdm_table:
            cdm_files = observations_files + cdm_files

        table_dict = combine_dicts(cdm_files)

        for k, v in table_dict.items():
            elements = v.get("elements")
            if elements and not isinstance(elements, list):
                v["elements"] = [elements]
            section = v.get("sections")
            if section:
                if not isinstance(section, list):
                    section = [section] * len(v.get("elements"))
                v["elements"] = [(s, e) for s, e in zip(section, v["elements"])]
                v.pop("sections", None)

        imodel_maps[cdm_table] = table_dict

    return imodel_maps
