"""Utility function for reading and writing files."""

from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path


def get_filename(
    pattern: Sequence[str | None],
    path: str | Path = ".",
    extension: str | None = "pq",
    separator: str | None = "_",
) -> str:
    """
    Construct a filename from a sequence of string components.

    Parameters
    ----------
    pattern : Sequence[str]
        Iterable of string components to be joined with hyphens
        (e.g., ["sales", "2024", "Q1"]).
        Empty or falsy items are ignored.
    path : str or Path-like, optional
        Directory in which the file should be placed.
        Default is current directory `"."`.
    extension : str, optional
        File extension, with or without a leading dot
        (e.g., `"pq"` or `".pq"`). Default is `"pq"`.
    separator : str, optional
        Separator to join the pattern components (default "_").

    Returns
    -------
    str
        Full file path including directory, filename, and extension.
        Returns an empty string if pattern is empty or contains no truthy elements.

    Notes
    -----
    - Any empty or falsy parts of `pattern` will be removed.
    - The extension is normalized to always begin with a leading dot.

    Examples
    --------
    >>> get_filename(["data", "2025"])
    './data-2025.psv'

    >>> get_filename(["report", ""], path="/tmp", extension=".txt")
    '/tmp/report.txt'

    >>> get_filename(["part1", "part2"], separator="_")
    './part1_part2.psv'
    """
    if not pattern or not any(pattern):
        return ""

    if not extension:
        extension = ""
    elif not extension.startswith("."):
        extension = f".{extension}"

    if not pattern:
        raise ValueError("pattern is empty.")

    if len(pattern) == 1:
        name = pattern[0]
    else:
        if not separator:
            raise ValueError("Length of pattern is greater than 1. A separator must be set.")
        name = separator.join(filter(None, pattern))

    filename = f"{name}{extension}"

    p = Path(path)
    return str(p / filename)
