"""
Manage data model code table files.

Functions to manage data model
code table files and objects according to the
requirements of the data reader tool

"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from cdm_reader_mapper.common.json_dict import (
    collect_json_files,
    combine_dicts,
    open_code_table,
)

from .. import properties


def read_table(
    code_table_name,
    imodel=None,
    ext_table_path=None,
):
    """
    Read a data model code table file to a dictionary.

    It completes the code table to the full complexity
    the data reader expects, by appending information
    on secondary keys and expanding range keys.

    Parameter
    ---------
    code_table_name: str
        The external code table file.
    imodel: str, optional
        Name of internally available input data model.
        e.g. icoads_r300_d704
    ext_table_path: str, optional
        The path to the external code table file.
        One of ``imodel`` and ``ext_table_path`` must be set.

    Returns
    -------
    dict
        Code table
    """
    # 1. Validate input
    if ext_table_path:
        table_path = os.path.abspath(ext_table_path)
        table_files = os.path.join(table_path, code_table_name + ".json")
        if not os.path.isfile(table_files):
            logging.error(f"Can't find input code table file {table_files}")
            return
        table_files = Path(table_files)
    else:
        imodel = imodel.split("_")
        table_files = collect_json_files(
            *imodel,
            base=f"{properties._base}.codes",
            name=code_table_name,
        )

    if isinstance(table_files, Path):
        table_files = [table_files]
    # 2. Get tables
    tables = [open_code_table(ifile) for ifile in table_files]

    # 3. Combine tables
    return combine_dicts(tables)
