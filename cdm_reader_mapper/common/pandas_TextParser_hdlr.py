"""Utilities for handling pandas TextParser objects safely."""

from __future__ import annotations

import pandas as pd
from pandas.io.parsers import TextFileReader
from io import StringIO
import logging

logger = logging.getLogger(__name__)

READ_CSV_KWARGS = [
    "chunksize",
    "names",
    "dtype",
    "parse_dates",
    "date_parser",
    "infer_datetime_format",
    "delimiter",
    "quotechar",
    "escapechar",
    "skip_blank_lines",
]


def _get_raw_buffer(parser: TextFileReader) -> str | None:
    if hasattr(parser, "_raw_buffer"):
        return parser._raw_buffer

    f = getattr(parser.handles, "handle", None)
    if f is None:
        raise ValueError("TextFileReader has no accessible handle for copying.")

    try:
        f = parser.handles.handle
        raw = f.getvalue()
        parser._raw_buffer = raw
        return raw
    except Exception as e:
        raise RuntimeError("Failed to read raw buffer") from e


def _new_reader_from_buffer(parser: TextFileReader) -> TextFileReader | None:
    raw = _get_raw_buffer(parser)
    if raw is None:
        return None

    read_dict = read_dict = {
        k: parser.orig_options.get(k)
        for k in READ_CSV_KWARGS
        if k in parser.orig_options
    }
    return pd.read_csv(StringIO(raw), **read_dict)


def make_copy(parser: TextFileReader) -> TextFileReader | None:
    """
    Create a duplicate of a pandas TextFileReader object.

    Parameters
    ----------
    Parser : pandas.io.parsers.TextFileReader
        The TextFileReader whose state will be copied.

    Returns
    -------
    pandas.io.parsers.TextFileReader or None
        A new TextFileReader with identical content and read options,
        or None if copying fails.

    Notes
    -----
    - The source handle must support `.getvalue()`, meaning this works
      only for in-memory file-like objects such as `StringIO`.
    """
    try:
        return _new_reader_from_buffer(parser)
    except Exception as e:
        raise RuntimeError(f"Failed to copy TextParser: {e}") from e


def restore(parser: TextFileReader) -> TextFileReader | None:
    """
    Restore a TextFileReader to its initial read position and state.

    Parameters
    ----------
    Parser : pandas.io.parsers.TextFileReader
        The TextFileReader to restore.

    Returns
    -------
    pandas.io.parsers.TextFileReader or None
        Restored TextFileReader, or None if restoration fails.
    """
    return make_copy(parser)


def is_not_empty(parser: TextFileReader) -> bool | None:
    """
    Determine whether a TextFileReader contains at least one row.

    Parameters
    ----------
    Parser : pandas.io.parsers.TextFileReader
        The parser to inspect.

    Returns
    -------
    bool or None
        True if not empty.
        False if empty.
        None if an error occurs.
    """
    if hasattr(parser, "_is_not_empty"):
        return parser._is_not_empty

    reader = make_copy(parser)
    if reader is None:
        return None

    try:
        chunk = next(reader)
        result = not chunk.empty
        parser._is_not_empty = result
        return result
    except StopIteration:
        parser._is_not_empty = False
        return False


def get_length(parser: TextFileReader) -> int | None:
    """
    Count total rows in a TextFileReader (consuming a copied stream).

    Parameters
    ----------
    Parser : pandas.io.parsers.TextFileReader
        The parser to measure.

    Returns
    -------
    int or None
        Total number of rows, or None if processing fails.
    """
    if hasattr(parser, "_row_count"):
        return parser._row_count

    reader = make_copy(parser)
    if reader is None:
        return None

    total = 0
    try:
        for chunk in reader:
            total += len(chunk)
        parser._row_count = total
        return total
    except Exception as e:
        raise RuntimeError("Failed while counting rows") from e
