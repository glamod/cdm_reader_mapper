"""Common Data Model (CDM) reader and mapper common properties."""

from __future__ import annotations

numeric_types = ["int", "float"]

object_types = ["str", "object", "key", "datetime"]

data_types = object_types.copy()
data_types.extend(numeric_types)
