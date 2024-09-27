"""Common Data Model (CDM) mapper properties."""

from __future__ import annotations

from ..properties import numeric_types, object_types, supported_data_models  # noqa

_base = "cdm_reader_mapper.cdm_mapper"

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
