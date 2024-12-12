"""Common Data Model (CDM) reader and mapper package."""

from __future__ import annotations

from .cdm_mapper.mapper import map_model  # noqa
from .cdm_mapper.properties import cdm_tables  # noqa
from .cdm_mapper.table_reader import read_tables  # noqa
from .cdm_mapper.table_writer import cdm_to_ascii  # noqa
from .data import test_data  # noqa
from .duplicates.duplicates import duplicate_check  # noqa
from .mdf_reader.read import read as read_mdf  # noqa

__author__ = """Ludwig Lierhammer"""
__email__ = "ludwiglierhammer@dwd.de"
__version__ = "1.0.2"
