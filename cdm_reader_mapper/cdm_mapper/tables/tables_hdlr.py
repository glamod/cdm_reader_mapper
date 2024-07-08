"""
Climate Model Data (CDM) mapping table routinies.

Created on Thu Apr 11 13:45:38 2019

Module to handle C3S Climate Data Store Common Data Model (CMD) tables within
the cdm tool.

@author: iregon
"""

from __future__ import annotations

import csv
import json
import os

import requests

from cdm_reader_mapper.common import logging_hdlr
from cdm_reader_mapper.common.getting_files import get_files

from ..properties import _base, cdm_tables


class tables_hdlr:
    """Class for loading mapping tables."""

    def __init__(self, log_level="DEBUG"):
        self.logger = logging_hdlr.init_logger(__name__, level=log_level)
        _tables = f"{_base}.tables"
        self.table_path = get_files(_tables)

    def load_tables_maps(self, imodel, cdm_subset=None):
        """Load CDM mapping tables."""
        _imodel = f"{_base}.tables.{imodel}"
        try:
            _imodel_data = get_files(_imodel)
        except ModuleNotFoundError:
            self.logger.error(f"No mapping functions module for model {imodel}")
            return
        map_paths = _imodel_data.glob("*.json")
        map_paths = {os.path.basename(x).split(".")[0]: x for x in list(map_paths)}
        if isinstance(cdm_subset, str):
            cdm_subset = [cdm_subset]
        if isinstance(cdm_subset, list):
            not_in_cdm = [x for x in cdm_subset if x not in map_paths.keys()]
            if any(not_in_cdm):
                self.logger.error(
                    "A wrong cdm table was requested for in model {}: {}".format(
                        imodel, ",".join(not_in_cdm)
                    )
                )
                self.logger.info(
                    "cdm tables registered for model are: {}".format(
                        ",".join(list(map_paths.keys()))
                    )
                )
                return
            remove_tables = [x for x in map_paths.keys() if x not in cdm_subset]
            for x in remove_tables:
                map_paths.pop(x, None)
        maps = dict()
        try:
            for key in map_paths.keys():
                with open(map_paths.get(key)) as json_file:
                    maps[key] = json.load(json_file)
                for k, v in maps[key].items():
                    elements = v.get("elements")
                    if elements and not isinstance(elements, list):
                        v["elements"] = [elements]
                    section = v.get("sections")
                    if section:
                        if not isinstance(section, list):
                            section = [section] * len(v.get("elements"))
                        v["elements"] = [(s, e) for s, e in zip(section, v["elements"])]
                        v.pop("sections", None)
        except Exception as e:
            self.logger.error(f"Could not load mapping file {map_paths.get(key)}: {e}")
            return
        return maps

    def load_tables(self, log_level="DEBUG"):
        """Load mapping tables."""
        table_paths = self.table_path.glob("*.json")
        table_paths = {os.path.basename(x).split(".")[0]: x for x in list(table_paths)}

        observation_tables = [x for x in cdm_tables if x.startswith("observations-")]
        # Make a copy from the generic observations table for each to the observations
        # table defined in properties
        observation_path = table_paths.get("observations")
        table_paths.pop("observations", None)
        table_paths.update(
            {
                observation_table: observation_path
                for observation_table in observation_tables
            }
        )

        tables = dict()
        try:
            for key in table_paths.keys():
                with open(table_paths.get(key)) as json_file:
                    tables[key] = json.load(json_file)
        except Exception:
            self.logger.error(f"Could not load table {key}", exc_info=True)
            return
        return tables


# cdm elements dtypes
# Mail sent may 7th to Dave. Are the types there real SQL types, or just approximations?
# Numeric type in table definition not useful here to define floats with a specific precision
# We should be able to use those definitions. Keep in mind that arrays are object type in pandas!
# Remember any int and float (int, numeric) need to be tied for the parser!!!!
# Also datetimes!
# Until CDM table definition gets clarified:
# We map from cdm table definition types to those in properties.pandas_dtypes.get('from_sql'), else: 'object'
# We update to df column dtype if is of float type


def from_glamod(
    table_filename,
    gitlinkroot=None,
    element_col=1,
    type_col=2,
    field_separator="\t",
    skip_lines=3,
):
    """Load mapping tables from GLAMOD."""
    # Get tables from GLAMOD Git repo and format to nested dictionary with:
    # { cdm_name: {'data_type':value}}
    #
    # table_filename: table filename in repo directory
    # gitlinkroot: url to directory where tables are stored
    # element_col: column with element names (first is 1)
    # type_col: column with element data typs (first is 1)
    #
    # About data type definitions in this source (table_definitions in GitHub):
    # it is not controlled vocab. and might change in the future!!!!
    # Get data types and clean primary key, optional and whitespaces: '(pk)', '*'

    logger = logging_hdlr.init_logger(__name__, level="INFO")
    if not gitlinkroot:
        gitlinkroot = (
            "https://github.com/glamod/common_data_model/blob/master/table_definitions/"
        )
        logger.info(f"Setting gitlink root to default: {gitlinkroot}")

    gitlinkroot = gitlinkroot.replace("blob/", "")
    gitlinkroot = gitlinkroot.replace("https://", "https://raw.")
    response = requests.get(os.path.join(gitlinkroot, table_filename), timeout=100)
    field_separator = "\t"
    lines = list(
        csv.reader(
            response.content.decode("utf-8").splitlines(), delimiter=field_separator
        )
    )
    for i in range(0, skip_lines):
        lines.pop(0)
    return {
        x[element_col - 1]: {
            "data_type": x[type_col - 1].strip("(pk)").strip("*").strip()
        }
        for x in lines
    }
