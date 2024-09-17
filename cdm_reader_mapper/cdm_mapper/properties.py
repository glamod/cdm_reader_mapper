"""Common Data Model (CDM) mapper properties."""

from __future__ import annotations

from ..properties import numeric_types, object_types  # noqa

_base = "cdm_reader_mapper.cdm_mapper"

supported_models = [
    "craid",
    "gcc",
    "icoads_r3000",
    "icoads_r3000_d701_type1",
    "icoads_r3000_d701_type2",
    "icoads_r3000_d702",
    "icoads_r3000_d704",
    "icoads_r3000_d705-707",
    "icoads_r3000_d714",
    "icoads_r3000_d721",
    "icoads_r3000_d730",
    "icoads_r3000_d781",
    "icoads_r3000_NRT",
]

cdm_tables = [
    "header",
    "observations-at",
    "observations-sst",
    "observations-dpt",
    "observations-wbt",
    "observations-wd",
    "observations-ws",
    "observations-slp",
]

# ...from CDM table definitions psuedo-sql(...) --------------------------------
pandas_dtypes = {}
pandas_dtypes["from_sql"] = {}
pandas_dtypes["from_sql"]["timestamp with timezone"] = "object"
pandas_dtypes["from_sql"]["numeric"] = "float"
pandas_dtypes["from_sql"]["int"] = "int"

# Some defaults ---------------------------------------------------------------
default_decimal_places = 5
