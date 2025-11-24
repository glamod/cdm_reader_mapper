"""
Created on Thu Apr 11 13:45:38 2019.

Module to handle data models mappings to C3S Climate Data Store
Common Data Model (CMD) tables within the cdm tool.

@author: iregon
"""

from __future__ import annotations

import ast
import datetime

from typing import Any

from cdm_reader_mapper.common.json_dict import (
    collect_json_files,
    combine_dicts,
    open_json_file,
)

from .. import properties


def _eval(s: str) -> Any:
    """Safely evaluate a string as a Python literal."""
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return s


def _to_int(x: Any) -> int | None:
    """Convert input to an integer if possible."""
    try:
        return int(x)
    except (TypeError, ValueError):
        None


def _expand_integer_range_key(d: Any) -> Any:
    """Expand dictionary keys that are integer ranges into individual year keys."""
    if not isinstance(d, dict):
        return d

    expanded = {}

    for k, v in d.items():
        v = _expand_integer_range_key(v)

        k_eval = _eval(k)

        if isinstance(k_eval, list) and len(k_eval) >= 2:
            lower = _to_int(k_eval[0])
            upper = _to_int(
                k_eval[1] if k_eval[1] != "yyyy" else datetime.date.today().year
            )
            step = _to_int(k_eval[2] if len(k_eval) > 2 else 1)

            if None in (lower, upper, step):
                continue

            for i in range(lower, upper + 1, step):
                expanded[str(i)] = v
        elif not isinstance(k_eval, list):
            expanded[k] = v

    return expanded


def open_code_table(ifile) -> dict:
    """Open code table from json file on disk."""
    json_dict = open_json_file(ifile)
    return _expand_integer_range_key(json_dict)


def get_code_table(
    data_model: str, *sub_models: str, code_table: str | None = None
) -> dict[str, dict[str, Any]]:
    """Load code tables into dictionary.

    Combine JSON code table files from a specified data model,
    optional submodels, and common code tables.

    Parameters
    ----------
    data_model : str
        The main data model name, e.g., `icoads`.
    sub_models : str
        Optional submodel names, e.g. `r300`, `d721`.
    code_table: str
        Name of the code table to load. If None, return empty dictionary.

    Returns
    -------
    dict
        Combined dictionary of code tables. Nested tables are merged recursively.
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
