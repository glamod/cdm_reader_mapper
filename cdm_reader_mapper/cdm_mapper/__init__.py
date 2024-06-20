"""Cliamte Data Model (CDM) mapper package."""
from __future__ import annotations

from .mapper import map_model  # noqa
from .table_reader import read_tables  # noqa
from .table_writer import cdm_to_ascii  # noqa
from .table_writer import table_to_ascii  # noqa
from .tables.tables_hdlr import tables_hdlr

load_tables = tables_hdlr().load_tables  # noqa
