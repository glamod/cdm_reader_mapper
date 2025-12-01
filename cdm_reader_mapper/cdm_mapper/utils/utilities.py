"""Utility function for reading and writing CDM tables."""

from __future__ import annotations

from .. import properties

from typing import Iterable, Any


def dict_to_tuple_list(dic: dict[Any, Any]) -> list[tuple[Any, Any]]:
    """
    Convert a dictionary with scalar or list values into a list of (key, value) tuples.

    If a value is a list, each item in the list will produce its own tuple.
    If a value is a scalar, a single tuple is produced.

    Parameters
    ----------
    dic : dict
        Dictionary containing keys and values. Values may be scalars or lists.

    Returns
    -------
    list of tuple
        List of (key, value) tuples. If a dictionary value is a list,
        each list item becomes a separate tuple.

    Examples
    --------
    >>> dict_to_tuple_list({"A": [1, 2], "B": 3})
    [('A', 1), ('A', 2), ('B', 3)]
    """
    tuple_list = []
    for key, value in dic.items():
        if isinstance(value, list):
            tuple_list.extend((key, item) for item in value)
        else:
            tuple_list.append((key, value))

    return tuple_list


def get_cdm_subset(cdm_subset: Iterable[str] | None) -> list[str]:
    """
    Normalize and validate a CDM subset specification.

    This function ensures that the returned value is always a list of valid
    CDM table names (as defined in ``properties.cdm_tables``). It accepts:

    - ``None`` ? returns the full list of CDM tables.
    - A single string ? validated and returned as a one-element list.
    - An iterable of strings ? each entry is validated and returned unchanged.

    Any value not present in ``properties.cdm_tables`` will raise ``ValueError``.

    Parameters
    ----------
    cdm_subset : iterable of str or str or None
        CDM subset input to normalize. May be:
        - ``None`` ? full list of CDM tables is returned.
        - ``str`` ? returned as a list containing that string.
        - Any iterable (e.g., list) of strings ? returned unchanged after validation.

    Returns
    -------
    list of str
        A list of CDM table names that are guaranteed to exist in
        ``properties.cdm_tables``.

    Raises
    ------
    ValueError
        If any provided table name is not in ``properties.cdm_tables``.
    """
    if cdm_subset is None:
        return properties.cdm_tables

    if not isinstance(cdm_subset, list):
        cdm_subset = [cdm_subset]

    for item in cdm_subset:
        if item not in properties.cdm_tables:
            raise ValueError(
                f"Invalid CDM subset '{item}'. must be one of {properties.cdm_tables}."
            )

    return cdm_subset


def get_usecols(
    tb: str, col_subset: str | Iterable[str] | dict | None
) -> list[str] | None:
    """
    Normalize a column subset specification for use with pandas.read_csv.

    This function converts various forms of column subset input into a
    standardized list of column names suitable for the `usecols` argument
    in `pandas.read_csv`.

    Rules for conversion:
    ---------------------
    1. If `col_subset` is a string, it is returned as a single-element list.
    2. If `col_subset` is an iterable of strings (e.g., list, tuple, set),
       it is converted to a list.
    3. If `col_subset` is a dictionary, it is interpreted as a mapping
       {table_name: list_of_columns} and returns the entry corresponding
       to the given table `tb` (or None if missing).
    4. If `col_subset` is None, the function returns None, meaning all columns
       should be read.

    Parameters
    ----------
    tb : str
        Table name. Only used if `col_subset` is a dictionary.
    col_subset : str, iterable of str, dict, or None
        Column subset specification. Acceptable formats:
        - A single column name as a string.
        - An iterable of column names (list, tuple, set, etc.).
        - A dictionary mapping table names to column lists.
        - None (read all columns).

    Returns
    -------
    list of str or None
        Normalized list of column names suitable for pandas `usecols`,
        or None if no restriction is applied.

    Raises
    ------
    TypeError
        If `col_subset` is not a string, iterable, dict, or None.
    """
    if isinstance(col_subset, str):
        return [col_subset]

    if isinstance(col_subset, dict):
        return col_subset.get(tb)

    if col_subset is None:
        return None

    # Any other iterable ? convert to list
    try:
        return list(col_subset)
    except TypeError:
        raise TypeError(
            f"col_subset must be str, iterable of str, dict, or None, got {type(col_subset)}"
        )


def adjust_filename(filename: str, table: str = "", extension: str = "psv") -> str:
    """
    Adjust a filename by optionally prepending a table name and appending an extension.

    Rules:
    ------
    1. If `table` is not already part of the filename, it will be prepended with a dash.
    2. If the filename does not contain an extension (no '.'), the specified `extension` is appended.
       Default extension is 'psv'.

    Parameters
    ----------
    filename : str
        Original filename.
    table : str, optional
        Table name to prepend if not already present in the filename (default is "").
    extension : str, optional
        File extension to append if not already present (default is "psv").

    Returns
    -------
    str
        Adjusted filename with optional table prefix and file extension.

    Examples
    --------
    >>> adjust_filename("data", table="header")
    'header-data.psv'

    >>> adjust_filename("header-data.psv", table="header")
    'header-data.psv'

    >>> adjust_filename("data.txt", table="header")
    'header-data.txt'
    """
    if table not in filename:
        filename = f"{table}-{filename}"
    if "." not in filename:
        filename = f"{filename}.{extension}"
    return filename
