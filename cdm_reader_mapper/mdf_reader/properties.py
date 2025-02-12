"""Common Data Model (CDM) reader properties."""

from __future__ import annotations

import polars as pl

from ..properties import numeric_types, object_types, supported_data_models  # noqa

_base = "cdm_reader_mapper.mdf_reader"

open_file = {
    "craid": "netcdf",
}

year_column = {
    "gcc": "YR",
    "icoads": ("core", "YR"),
    "craid": ("drifter_measurements", "JULD"),
}

polars_dtypes = {}
for dtype in object_types:
    polars_dtypes[dtype] = pl.String
polars_dtypes.update({x: x for x in numeric_types})
polars_dtypes[pl.Datetime] = pl.Datetime

polars_int = pl.Int64

# ....and how they are managed
data_type_conversion_args = {}
for dtype in numeric_types:
    data_type_conversion_args[dtype] = ["scale", "offset"]
data_type_conversion_args[pl.Utf8] = ["disable_white_strip"]
data_type_conversion_args[pl.String] = ["disable_white_strip"]
data_type_conversion_args[pl.Categorical] = ["disable_white_strip"]
data_type_conversion_args[pl.Object] = ["disable_white_strip"]
data_type_conversion_args[pl.Datetime] = ["datetime_format"]

# Misc ------------------------------------------------------------------------
dummy_level = "_SECTION_"
# Length of reports in initial read
MAX_FULL_REPORT_WIDTH = 100000
# This is a delimiter internally used when writing to buffers
# It is the Unicode Character 'END OF TEXT'
# It is supposed to be safe because we don;t expect it in a string
# It's UTF-8 encoding length is not > 1, so it is supported by pandas 'c'
# engine, which is faster than the python engine.
internal_delimiter = "\u0003"
