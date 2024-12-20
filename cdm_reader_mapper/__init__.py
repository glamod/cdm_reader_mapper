"""Common Data Model (CDM) reader and mapper package."""

from __future__ import annotations

from .cdm_mapper.mapper import map_model  # noqa
from .cdm_mapper.properties import cdm_tables  # noqa
from .cdm_mapper.table_reader import read_tables  # noqa
from .cdm_mapper.table_writer import write_tables  # noqa
from .core import DataBundle  # noqa
from .data import test_data  # noqa
from .duplicates.duplicates import DupDetect  # noqa
from .duplicates.duplicates import duplicate_check  # noqa
from .mdf_reader.read import read as read_mdf  # noqa
from .metmetpy.datetime.correct import correct as correct_datetime  # noqa
from .metmetpy.datetime.validate import validate as validate_datetime  # noqa
from .metmetpy.platform_type.correct import correct as correct_pt  # noqa
from .metmetpy.station_id.validate import validate as validate_id  # noqa
from .operations.inspect import count_by_cat as unique  # noqa
from .operations.replace import replace_columns  # noqa
from .operations.select import select_from_index, select_from_list, select_true  # noqa

__author__ = """Ludwig Lierhammer"""
__email__ = "ludwiglierhammer@dwd.de"
__version__ = "1.0.2"
