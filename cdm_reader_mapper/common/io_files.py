"""Utility function for reading and writing files."""

from __future__ import annotations

import os


def get_filename(pattern, path=".", extension="psv") -> str:
    """Get file name."""
    if extension[0] != ".":
        extension = f".{extension}"
    filename = "-".join(filter(bool, pattern)) + extension
    return os.path.join(path, filename)
