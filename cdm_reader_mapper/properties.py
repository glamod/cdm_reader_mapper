"""Common Data Model (CDM) reader and mapper common properties."""

from __future__ import annotations
import polars as pl

# numeric_types = ["Int64", "int", "float"]
numeric_types = [
    pl.Float64,
    pl.Float32,
    pl.Int64,
    pl.Int32,
    pl.Int16,
    pl.Int8,
    pl.UInt64,
    pl.UInt32,
    pl.UInt16,
    pl.UInt8,
]

# object_types = ["str", "object", "key", "datetime"]
object_types = [pl.String, pl.Utf8, pl.Datetime, pl.Object, pl.Categorical]

supported_data_models = ["craid", "gcc", "icoads", "pub47"]
