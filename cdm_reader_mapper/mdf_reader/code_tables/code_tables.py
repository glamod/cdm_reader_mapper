"""
Manage data model code table files.

Functions to manage data model
code table files and objects according to the
requirements of the data reader tool

"""

from __future__ import annotations

import datetime
import logging
import os
from copy import deepcopy

try:
    from pandas.io.json._normalize import nested_to_record
except Exception:
    from pandas.io.json.normalize import nested_to_record

import ast

from cdm_reader_mapper.common.json_dict import (
    collect_json_files,
    combine_dicts,
    open_json_file,
)

from .. import properties


def table_keys(table):
    """DOCUMENTATION."""
    separator = "∿"  # something hopefully not in keys...
    if table.get("_keys"):
        _table = deepcopy(table)
        _table.pop("_keys")
        keys = list(nested_to_record(_table, sep=separator).keys())

        return [x.split(separator) for x in keys]
    else:
        return list(table.keys())


def eval_dict_items(item):
    """DOCUMENTATION."""
    try:
        return ast.literal_eval(item)
    except Exception:
        return item


def expand_integer_range_key(d):
    """DOCUMENTATION."""
    # Looping based on print_nested above
    if isinstance(d, dict):
        for k, v in list(d.items()):
            if "range_key" in k[0:9]:
                range_params = k[10:-1].split(",")
                try:
                    lower = int(range_params[0])
                except Exception as e:
                    logging.error(f"Lower bound parsing error in range key: {k}")
                    logging.error("Error is:")
                    logging.error(e)
                    return
                try:
                    upper = int(range_params[1])
                except Exception as e:
                    if range_params[1] == "yyyy":
                        upper = datetime.date.today().year
                    else:
                        logging.error(f"Upper bound parsing error in range key: {k}")
                        logging.error("Error is:")
                        logging.error(e)
                        return
                if len(range_params) > 2:
                    try:
                        step = int(range_params[2])
                    except Exception as e:
                        logging.error(f"Range step parsing error in range key: {k}")
                        logging.error("Error is:")
                        logging.error(e)
                        return
                else:
                    step = 1
                for i_range in range(lower, upper + 1, step):
                    deep_copy_value = deepcopy(
                        d[k]
                    )  # Otherwiserepetitions are linked and act as one!
                    d.update({str(i_range): deep_copy_value})
                d.pop(k, None)
            else:
                for k, v in d.items():
                    expand_integer_range_key(v)


def _read_table(table_path):
    """DOCUMENTATION."""
    table = open_json_file(table_path)
    # Expand range keys
    expand_integer_range_key(table)

    return table


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
    tables = [_read_table(ifile) for ifile in table_files]

    # 3. Combine tables
    return combine_dicts(tables)
