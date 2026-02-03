"""Common Data Model (CDM) reader and mapper common properties."""

from __future__ import annotations

from typing import Literal

NumericTypes = Literal["Int64", "int", "float"]

ObjectTypes = Literal["str", "object", "key", "datetime"]

SupportedDataModels = Literal["craid", "gdac", "icoads", "pub47"]

SupportedFileTypes = Literal["csv", "parquet", "feather"]
