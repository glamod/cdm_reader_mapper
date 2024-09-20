"""
Manage data model code table files.

Functions to manage data model
code table files and objects according to the
requirements of the data reader tool

"""

from __future__ import annotations

import ast
import logging
import os

from cdm_reader_mapper.common.json_dict import (
    collect_json_files,
    combine_dicts,
    open_code_table,
)

from .. import properties


def eval_dict_items(item):
    """DOCUMENTATION."""
    try:
        return ast.literal_eval(item)
    except Exception:
        return item


def read_table(
    code_table_name,
    data_model=None,
    release=None,
    deck=None,
    ext_table_path=None,
    ext_table_file=None,
):
    """
    Read a data model code table file to a dictionary.

    It completes the code table to the full complexity
    the data reader expects, by appending information
    on secondary keys and expanding range keys.

    Arguments
    ---------
    code_table_name: str
        The external code table file
    data_model: str, optional
        The name of the data model to read. This is for
        data models included in the tool.
    release: str, optional
        The name of the data model release. If chosen, overwrite data model schema.
        `data_model` is needed.
    deck: str, optional
        The name of the data model deck. If chosen, overwrite data model release schema.
        `data_model` is needed.
    ext_table_path: str, optional
        The path to the external code table file

    Returns
    -------
    dict
        Code table
    """
    # 1. Validate input
    if data_model:
        table_files = collect_json_files(
            data_model,
            release,
            deck,
            base=f"{properties._base}.code_tables",
            name=code_table_name,
        )
    else:
        if ext_table_file:
            table_path = os.path.abspath(ext_table_path)
            table_files = os.path.join(table_path, code_table_name + ".json")
        else:
            table_files = code_table_name

        if not os.path.isfile(table_files):
            logging.error(f"Can't find input code table file {table_files}")
            return

    if isinstance(table_files, str):
        table_files = [table_files]
    # 2. Get tables
    tables = [open_code_table(ifile) for ifile in table_files]

    # 3. Combine tables
    return combine_dicts(tables)
