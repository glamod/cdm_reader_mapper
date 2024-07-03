"""Common Data Model (CDM) reader properties."""

from __future__ import annotations

from ..properties import numeric_types, object_types  # noqa

_base = "cdm_reader_mapper.mdf_reader"

supported_data_models = [
    "gcc_immt",
    "imma1",
    "imma1_d701",
    "imma1_d702",
    "imma1_d704",
    "imma1_d705-707",
    "imma1_d714",
    "imma1_d721",
    "imma1_d730",
    "imma1_d781",
    "imma1_nodt",
    "td11",
    "td11_d110",
    "c_raid",
]

open_file = {
    "gcc_immt": "pandas",
    "imma1": "pandas",
    "imma1_d701": "pandas",
    "imma1_d702": "pandas",
    "imma1_d704": "pandas",
    "imma1_d705-707": "pandas",
    "imma1_d714": "pandas",
    "imma1_d721": "pandas",
    "imma1_d730": "pandas",
    "imma1_d781": "pandas",
    "imma1_nodt": "pandas",
    "td11": "pandas",
    "td11_d110": "pandas",
    "c_raid": "netcdf",
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
