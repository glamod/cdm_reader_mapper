"""Common Data Model (CDM) reader and mapper package."""

from __future__ import annotations

from .cdm_mapper.mapper import map_model  # noqa
from .cdm_mapper.properties import cdm_tables  # noqa
from .cdm_mapper.reader import read_tables  # noqa
from .cdm_mapper.writer import write_tables  # noqa
from .common import count_by_cat as unique  # noqa
from .common import (  # noqa
    replace_columns,
    select_from_index,
    select_from_list,
    select_true,
)
from .core import DataBundle, read, write  # noqa
from .data import test_data  # noqa
from .duplicates.duplicates import DupDetect  # noqa
from .duplicates.duplicates import duplicate_check  # noqa
from .mdf_reader.reader import read_data, read_mdf  # noqa
from .mdf_reader.writer import write_data  # noqa
from .metmetpy import (  # noqa
    correct_datetime,
    correct_pt,
    validate_datetime,
    validate_id,
)

__author__ = """Ludwig Lierhammer"""
__email__ = "ludwiglierhammer@dwd.de"
__version__ = "2.0.0"
