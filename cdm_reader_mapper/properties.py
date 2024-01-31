"""Common Data Model (CDM) reader and mapper common properties."""

numpy_integers = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]
numpy_floats = ["float16", "float32", "float64"]

pandas_nan_integers = {
    "int8": "Int8",
    "int16": "Int16",
    "int32": "Int32",
    "int64": "Int64",
    "uint8": "UInt8",
    "uint16": "UInt16",
    "uint32": "UInt32",
    "uint64": "UInt64",
}

numeric_types = numpy_integers.copy()
numeric_types.extend(numpy_floats)
numeric_types.extend(pandas_nan_integers.values())

object_types = ["str", "object", "key", "datetime"]

data_types = object_types.copy()
data_types.extend(numpy_integers)
data_types.extend(numpy_floats)
