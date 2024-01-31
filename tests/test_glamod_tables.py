from __future__ import annotations

import pytest  # noqa

from cdm_reader_mapper.cdm_mapper.tables.tables_hdlr import from_glamod


def test_get_glamod_tables():
    assert from_glamod("adjustment.csv")
