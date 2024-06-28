"""Common Data Model (CDM) reader and mapper common properties."""

from __future__ import annotations

numpy_integers = "int"
numpy_floats = "float"

numeric_types = [numpy_integers, numpy_floats]

object_types = ["str", "object", "key", "datetime"]

data_types = object_types.copy()
data_types.extend(numeric_types)
