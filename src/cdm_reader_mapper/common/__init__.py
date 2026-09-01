"""Common Data Model (CDM) reader and mapper common pandas operators."""

from __future__ import annotations

from .dataframe_helpers import restore_columns, standardize_object_columns
from .getting_files import load_file
from .inspect import count_by_cat, get_length
from .io_files import get_filename
from .iterators import ParquetStreamReader, ProcessFunction, is_valid_iterator, parquet_stream_from_iterable, process_disk_backed, process_function
from .json_dict import collect_json_files, combine_dicts, open_json_file
from .replace import replace_columns
from .select import (
    split_by_boolean,
    split_by_boolean_false,
    split_by_boolean_true,
    split_by_column_entries,
    split_by_index,
)


__all__ = [
    "ParquetStreamReader",
    "ProcessFunction",
    "collect_json_files",
    "combine_dicts",
    "count_by_cat",
    "get_filename",
    "get_length",
    "is_valid_iterator",
    "load_file",
    "open_json_file",
    "parquet_stream_from_iterable",
    "process_disk_backed",
    "process_function",
    "replace_columns",
    "split_by_boolean",
    "split_by_boolean_false",
    "split_by_boolean_true",
    "split_by_column_entries",
    "split_by_index",
    "standardize_object_columns",
]
