"""Common Data Model (CDM) reader properties."""

from __future__ import annotations

from ..properties import numeric_types, object_types, supported_data_models  # noqa

_base = "cdm_reader_mapper.mdf_reader"

open_file = {
    "c_raid": "netcdf",
}

year_column = {
    "gcc_immt": "YR",
    "imma1": ("core", "YR"),
    "imma1_d701": ("core", "YR"),
    "imma1_d702": ("core", "YR"),
    "imma1_d704": ("core", "YR"),
    "imma1_d705-707": ("core", "YR"),
    "imma1_d714": ("core", "YR"),
    "imma1_d721": ("core", "YR"),
    "imma1_d730": ("core", "YR"),
    "imma1_d781": ("core", "YR"),
    "imma1_nodt": ("core", "YR"),
    "td11": ("core1", "YEAR"),
    "td11_d110": ("core1", "YEAR"),
    "c_raid": ("drifter_measurements", "JULD"),
}

pandas_dtypes = {}
for dtype in object_types:
    pandas_dtypes[dtype] = "object"
pandas_dtypes.update({x: x for x in numeric_types})
pandas_dtypes["datetime"] = "datetime"

pandas_int = "Int64"

# ....and how they are managed
data_type_conversion_args = {}
for dtype in numeric_types:
    data_type_conversion_args[dtype] = ["scale", "offset"]
data_type_conversion_args["str"] = ["disable_white_strip"]
data_type_conversion_args["object"] = ["disable_white_strip"]
data_type_conversion_args["key"] = ["disable_white_strip"]
data_type_conversion_args["datetime"] = ["datetime_format"]

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
