"""Utility function for reading and writing files."""

from __future__ import annotations

import os

from typing import Sequence


def get_filename(
    pattern: Sequence[str],
    path: str = ".",
    extension: str = "psv",
    separator: str = "-",
) -> str:
    """
    Construct a filename from a sequence of string components.

    Parameters
    ----------
    pattern : Sequence[str]
        Iterable of string components to be joined with hyphens
        (e.g., ["sales", "2024", "Q1"] ? "sales-2024-Q1").
        Empty or falsy items are ignored.
    path : str, optional
        Directory in which the file should be placed.
        Default is current directory `"."`.
    extension : str, optional
        File extension, with or without a leading dot
        (e.g., `"psv"` or `".psv"`). Default is `"psv"`.
    separator : str, optional
        Separator to join the pattern components (default "-").

    Returns
    -------
    str
        Full file path including directory, filename, and extension.
        Returns an empty string if pattern is empty or contains no truthy elements.

    Examples
    --------
    >>> get_filename(["data", "2025"])
    './data-2025.psv'

    >>> get_filename(["report", ""], path="/tmp", extension=".txt")
    '/tmp/report.txt'

    >>> get_filename(["part1", "part2"], separator="_")
    './part1_part2.psv'

    Notes
    -----
    - Any empty or falsy parts of `pattern` will be removed.
    - The extension is normalized to always begin with a leading dot.
    """
    if not pattern or not any(pattern):
        return ""

    if not extension:
        extension = ""
    elif not extension.startswith("."):
        extension = f".{extension}"

    name = separator.join(filter(bool, pattern))

    filename = f"{name}{extension}"
    return os.path.join(path, filename)
