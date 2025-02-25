"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations

from cdm_reader_mapper.cdm_mapper.writer import write_tables
from cdm_reader_mapper.mdf_reader.writer import write_data


def write(
    data,
    mode="data",
    **kwargs,
):
    """Write either MDF data or CDM tables on disk.

    Parameters
    ----------
    data: pandas.DataFrame
        pandas.DataFrame to export.
    mode: str, ["data", "tables"]
        Data mode.
        Default: "data"

    See Also
    --------
    write_data : Write MDF data and validation mask to disk.
    write_tables: Write CDM tables to disk.
    read: Read either original marine-meteorological data or MDF data or CDM tables from disk.
    read_mdf : Read original marine-meteorological data from disk.
    read_data : Read MDF data and validation mask from disk.
    read_tables : Read CDM tables from disk.

    Note
    ----
    If `mode` is "data" write data use :py:func:`write_data`.
    If `mode` is "tables" write data use :py:func:`write_tables`.
    """
    if mode == "data":
        write_data(data, **kwargs)
    elif mode == "tables":
        write_tables(data, **kwargs)
    else:
        raise ValueError(f"No valid mode: {mode}. Choose one of ['data', 'tables']")
