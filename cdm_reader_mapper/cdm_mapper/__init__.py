"""Climate Data Model (CDM) mapper package."""

from __future__ import annotations

from .duplicates import duplicate_check  # noqa
from .mapper import map_model  # noqa
from .table_reader import read_tables  # noqa
from .table_writer import write_tables  # noqa
from .tables.tables import get_cdm_atts  # noqa
