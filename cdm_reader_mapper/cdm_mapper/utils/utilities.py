"""Utility function for reading and writing CDM tables."""

from __future__ import annotations

from .. import properties


def dict_to_tuple_list(dic) -> list[tuple]:
    """Convert dictionary with list values to list of tuples."""
    tuple_list = []
    for k, v in dic.items():
        if isinstance(v, list):
            for v_ in v:
                tuple_list += [(k, v_)]
        else:
            tuple_list += [(k, v)]
    return tuple_list


def get_cdm_subset(cdm_subset) -> list[str]:
    """Return cdm_subset."""
    if cdm_subset is None:
        return properties.cdm_tables
    elif not isinstance(cdm_subset, list):
        return [cdm_subset]
    return cdm_subset


def get_usecols(tb, col_subset=None) -> list[str]:
    """Return usecols for pandas.read_csv function."""
    if isinstance(col_subset, str):
        return [col_subset]
    elif isinstance(col_subset, list):
        return col_subset
    elif isinstance(col_subset, dict):
        return col_subset.get(tb)


def adjust_filename(filename, table="", extension="psv") -> str:
    """Adjust filename."""
    if table not in filename:
        filename = f"{table}-{filename}"
    if "." not in filename:
        filename = f"{filename}.{extension}"
    return filename
