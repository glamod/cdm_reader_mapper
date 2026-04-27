"""Common Data Model (CDM) reader and mapper package."""

from __future__ import annotations

from .cdm_mapper.mapper import map_model
from .cdm_mapper.properties import cdm_tables
from .cdm_mapper.reader import read_tables
from .cdm_mapper.writer import write_tables
from .common import count_by_cat as unique
from .common import (
    replace_columns,
    split_by_boolean,
    split_by_boolean_false,
    split_by_boolean_true,
    split_by_column_entries,
    split_by_index,
)
from .core.databundle import DataBundle
from .core.reader import read
from .core.writer import write
from .data import test_data
from .duplicates.duplicates import (
    DupDetect,
    duplicate_check,
)
from .mdf_reader.reader import read_data, read_mdf
from .mdf_reader.writer import write_data
from .metmetpy import (
    correct_datetime,
    correct_pt,
    validate_datetime,
    validate_id,
)


__all__ = [
    "DataBundle",
    "DupDetect",
    "cdm_tables",
    "correct_datetime",
    "correct_pt",
    "duplicate_check",
    "map_model",
    "read",
    "read_data",
    "read_mdf",
    "read_tables",
    "replace_columns",
    "split_by_boolean",
    "split_by_boolean_false",
    "split_by_boolean_true",
    "split_by_column_entries",
    "split_by_index",
    "test_data",
    "unique",
    "validate_datetime",
    "validate_id",
    "write",
    "write_data",
    "write_tables",
]

__author__ = """Ludwig Lierhammer"""
__email__ = "ludwiglierhammer@dwd.de"
__version__ = "2.4.1"
