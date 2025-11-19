"""Utilities for handling pandas TextParser objects safely."""

from __future__ import annotations
import pandas as pd
from pandas.io.parsers import TextFileReader
from io import StringIO
import logging

logger = logging.getLogger(__name__)

read_params = [
    "chunksize",
    "names",
    "dtype",
    "parse_dates",
    "date_parser",
    "infer_datetime_format",
    "delimiter",
    "quotechar",
    "escapechar",
]


def make_copy(Parser: TextFileReader) -> TextFileReader | None:
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
        f = Parser.handles.handle
        new_ref = StringIO(f.getvalue())
        read_dict = {k: Parser.orig_options.get(k) for k in read_params}
        return pd.read_csv(new_ref, **read_dict)
    except Exception:
        logger.error("Failed to copy TextParser", exc_info=True)
        return None


def restore(Parser: TextFileReader) -> TextFileReader | None:
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
    try:
        f = Parser.handles.handle
        f.seek(0)
        read_dict = {k: Parser.orig_options.get(k) for k in read_params}
        return pd.read_csv(f, **read_dict)
    except Exception:
        logger.error("Failed to restore TextParser", exc_info=True)
        return None


def is_not_empty(Parser: TextFileReader) -> bool | None:
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
    try:
        parser_copy = make_copy(Parser)
        if parser_copy is None:
            return None
    except Exception:
        logger.error(
            f"Failed to process input. Input type is {type(Parser)}",
            exc_info=True,
        )
        return None

    try:
        chunk = parser_copy.get_chunk()
        parser_copy.close()
        return len(chunk) > 0
    except Exception:
        logger.debug("Error while checking emptiness", exc_info=True)
        return False


def get_length(Parser: TextFileReader) -> int | None:
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
    try:
        parser_copy = make_copy(Parser)
        if parser_copy is None:
            return None
    except Exception:
        logger.error(
            f"Failed to process input. Input type is {type(Parser)}",
            exc_info=True,
        )
        return None

    total = 0
    try:
        for df in parser_copy:
            total += len(df)
        return total
    except Exception:
        logger.error("Failed while counting rows", exc_info=True)
        return None
