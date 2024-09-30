"""
Climate Model Data (CDM) mapping table routinies.

Created on Thu Apr 11 13:45:38 2019

Module to handle C3S Climate Data Store Common Data Model (CMD) tables within
the cdm tool.

@author: iregon
"""

from __future__ import annotations

from copy import deepcopy

from cdm_reader_mapper.common.json_dict import (
    collect_json_files,
    combine_dicts,
    open_json_file,
)

from .. import properties


def get_cdm_atts(cdm_tables=None):
    """Get CDM attribute tables.

    Parameters
    ----------
    cdm_tables: str, list, optional
        List of cdm_tables
        If None select all available tables.

    Returns
    -------
    dict
        CDM attribute dictionary.
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
        cdm_tables = properties.cdm_tables
    if isinstance(cdm_tables, str):
        cdm_tables = [cdm_tables]

    cdm_atts = {}
    for cdm_table in cdm_tables:
        if cdm_table == "header":
            cdm_atts[cdm_table] = deepcopy(header_dict)
        else:
            cdm_atts[cdm_table] = deepcopy(observations_dict)
    return cdm_atts


def get_imodel_maps(data_model, *sub_models, cdm_tables=[]):
    """DOCUMENTATION."""
    imodel_maps = {}
    observations = []
    obs_files = []
    for cdm_table in cdm_tables:
        cdm_files = collect_json_files(
            data_model, *sub_models, base=f"{properties._base}.tables", name=cdm_table
        )
        if not observations:
            obs_files = collect_json_files(
                data_model,
                *sub_models,
                base=f"{properties._base}.tables",
                name="observations",
            )
        if "observations" in cdm_table:
            cdm_files = obs_files + cdm_files
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
