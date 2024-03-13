"""
Functions for pandas TextParser objects.

Created on Tue Apr  2 10:34:56 2019

Assumes we are never writing a header!

@author: iregon
"""

from __future__ import annotations

from io import StringIO

import pandas as pd

from . import logging_hdlr

logger = logging_hdlr.init_logger(__name__, level="DEBUG")

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


def make_copy(OParser):
    """Make a copy of a pandas TextParser object."""
    try:
        f = OParser.handles.handle
        NewRef = StringIO(f.getvalue())
        read_dict = {x: OParser.orig_options.get(x) for x in read_params}
        NParser = pd.read_csv(NewRef, **read_dict)
        return NParser
    except Exception:
        logger.error("Failed to copy TextParser", exc_info=True)
        return


def restore(Parser):
    """Restore pandas TextParser object."""
    try:
        f = Parser.handles.handle
        f.seek(0)
        read_dict = {x: Parser.orig_options.get(x) for x in read_params}
        Parser = pd.read_csv(f, **read_dict)
        return Parser
    except Exception:
        logger.error("Failed to restore TextParser", exc_info=True)
        return Parser


def is_not_empty(Parser):
    """Return boolean whether pandas TextParser object is empty."""
    try:
        Parser_copy = make_copy(Parser)
    except Exception:
        logger.error(
            f"Failed to process input. Input type is {type(Parser)}", exc_info=True
        )
        return
    try:
        first_chunk = Parser_copy.get_chunk()
        Parser_copy.close()
        if len(first_chunk) > 0:
            logger.debug("Is not empty")
            return True
        else:
            return False
    except Exception:
        logger.debug("Something went wrong", exc_info=True)
        return False


def get_length(Parser):
    """Get length of pandas TextParser object."""
    try:
        Parser_copy = make_copy(Parser)
    except Exception:
        logger.error(
            f"Failed to process input. Input type is {type(Parser)}", exc_info=True
        )
        return
    no_records = 0
    for df in Parser_copy:
        no_records += len(df)
    return no_records
