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
from typing import Optional, Dict

from cdm_reader_mapper.common.json_dict import (
    collect_json_files,
    combine_dicts,
    open_json_file,
)

from .. import properties


def read_table(
    code_table_name: str,
    imodel: Optional[str] = None,
    ext_table_path: Optional[str] = None,
) -> Dict:
    """
    Load a data model code table into a Python dictionary.

    The code table may define secondary keys, range expansions, or other
    structures required by the data reader. This function resolves the
    file location either from an external path or an internal data model.

    Parameters
    ----------
    code_table_name : str
        The name of the code table (without file extension).  
        e.g., `"ICOADS.C0.IM"`
    imodel : str, optional
        Internal data model name, e.g., `"icoads_r300_d704"`. Required if
        `ext_table_path` is not provided.
    ext_table_path : str, optional
        External path containing the code table file. If set, this path
        takes precedence over `imodel`.

    Returns
    -------
    Dict
        The fully combined code table dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified table file cannot be found.
    ValueError
        If neither `imodel` nor `ext_table_path` is provided.
    """
    if ext_table_path:
        table_path = Path(ext_table_path).resolve()
        table_file = table_path / f"{code_table_name}.json"
        if not table_file.is_file():
            raise FileNotFoundError(f"Can't find input code table file {table_file}")
        table_files = [table_file]
    elif imodel:
        parts = imodel.split("_")
        table_files = collect_json_files(
            *parts,
            base=f"{properties._base}.codes",
            name=code_table_name,
        )

        if isinstance(table_files, Path):
          table_files = [table_files]
    else:
        raise ValueError("One of 'imodel' or 'ext_table_path' must be set")

    tables = [open_json_file(ifile) for ifile in table_files]

    return combine_dicts(tables)
