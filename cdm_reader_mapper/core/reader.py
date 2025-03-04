"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations

from cdm_reader_mapper.cdm_mapper.reader import read_tables
from cdm_reader_mapper.mdf_reader.reader import read_mdf, read_data


def read(
    source,
    mode="mdf",
    **kwargs,
):
    """Read either original marine-meteorological data or MDF data or CDM tables from disk.

    Parameters
    ----------
    source: str
        Source of the input data.
    mode: str, {mdf, data, tables}
        Data mode.
        Default: mdf

    See Also
    --------
    read_mdf : Read original marine-meteorological data from disk.
    read_data : Read MDF data and validation mask from disk.
    read_tables : Read CDM tables from disk.
    write: Write either MDF data or CDM tables on disk.
    write_data : Write MDF data and validation mask to disk.
    write_tables: Write CDM tables to disk.

    Note
    ----
    If `mode` is "mdf" use :py:func:`read_mdf`.
    If `mode` is "data" use :py:func:`read_data`.
    If `mode` is "tables" use :py:func:`read_tables`.

    Note
    ----
    `kwargs` are the keyword arguments for the specific `mode` reader.

    """
    match mode.lower():
        case "mdf":
            return read_mdf(source, **kwargs)
        case "data":
            return read_data(source, **kwargs)
        case "tables":
            return read_tables(source, **kwargs)
        case _:
            raise ValueError(
                f"No valid mode: {mode}. Choose one of ['mdf', 'data', 'tables']"
            )
