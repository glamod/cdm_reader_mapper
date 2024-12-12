"""Common Data Model (CDM) reader and mapper package."""

from __future__ import annotations

from . import cdm_mapper  # noqa
from .cdm_mapper import read_tables  # noqa
from .cdm_mapper.properties import cdm_tables  # noqa
from . import common  # noqa
from . import mdf_reader  # noqa
from .mdf_reader import read as read_mdf  # noqa
from . import metmetpy  # noqa
from . import operations  # noqa
from .data import test_data  # noqa

__author__ = """Ludwig Lierhammer"""
__email__ = "ludwiglierhammer@dwd.de"
__version__ = "1.0.2"
