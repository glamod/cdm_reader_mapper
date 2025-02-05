"""
Created on Thu Apr 11 13:45:38 2019.

Module to handle data models mappings to C3S Climate Data Store
Common Data Model (CMD) tables within the cdm tool.

@author: iregon
"""

from __future__ import annotations

import ast
import datetime
import logging

from cdm_reader_mapper.common.json_dict import (
    collect_json_files,
    combine_dicts,
    open_json_file,
)

from .. import properties


def _eval(s):
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return s


def expand_integer_range_key(d):
    """DOCUMENTATION."""
    # Looping based on print_nested above
    if not isinstance(d, dict):
        return d
    d_ = {}
    for k, v in d.items():
        k_ = _eval(k)
        if not isinstance(k_, list):
            d_[k] = expand_integer_range_key(v)
            continue
        try:
            lower = int(k_[0])
        except Exception as e:
            logging.error(f"Lower bound parsing error in range key: {k}")
            logging.error(e)
            return
        upper = k_[1]
        if upper == "yyyy":
            upper = datetime.date.today().year
        else:
            try:
                upper = int(upper)
            except Exception as e:
                logging.error(f"Lower bound parsing error in range key: {k}")
                logging.error(e)
                return
        if len(k_) > 2:
            try:
                step = int(k_[2])
            except Exception as e:
                logging.error(f"Range step parsing error in range key: {k}")
                logging.error(e)
                return
        else:
            step = 1

        for yr in range(lower, upper + 1, step):
            d_[str(yr)] = v
    return d_


def _open_code_table(ifile):
    json_dict = open_json_file(ifile)
    return expand_integer_range_key(json_dict)


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
    tables = [_open_code_table(ifile) for ifile in table_files]
    return combine_dicts(tables)
