"""
Created on Thu Apr 11 13:45:38 2019.

Module to handle data models mappings to C3S Climate Data Store
Common Data Model (CMD) tables within the cdm tool.

@author: iregon
"""

from __future__ import annotations

from cdm_reader_mapper.common.json_dict import (
    collect_json_files,
    combine_dicts,
    open_code_table,
)

from .. import properties


def get_code_table(data_model, *sub_models, code_table=None):
    """Load code tables into dictionary.

    Parameters
    ----------
    data_model: str
        The name of the data model to read. This is for
        data models included in the tool.
    sub_models*: optionally
        Sub-directories of ``data_model``.
        E.g. r300 d701 type2
    code_table: str
        Name of the code table to find.
    """
    common_files = collect_json_files(
        "common", base=f"{properties._base}.codes", name=code_table
    )
    table_files = collect_json_files(
        data_model, *sub_models, base=f"{properties._base}.codes", name=code_table
    )
    table_files = common_files + table_files
    tables = [open_code_table(ifile) for ifile in table_files]
    return combine_dicts(tables)
