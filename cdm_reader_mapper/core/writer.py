"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations

from cdm_reader_mapper.cdm_mapper.writer import write_tables
from cdm_reader_mapper.mdf_reader.writer import write_data


def write(
    data,
    mode="data",
    **kwargs,
) -> None:
    """Write either MDF data or CDM tables on disk.

    Parameters
    ----------
    data: pandas.DataFrame
        pandas.DataFrame to export.
    mode: str, {data, tables}
        Write data mode:

          * "data" to write MDF data to disk
          * "tables" to write CDM tables to disk. Map MDF data to CDM tables with :py:func:`DataBundle.map_model`.


        Default: data

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
    `kwargs` are the keyword arguments for the specific `mode` reader.
    """
    match mode.lower():
        case "data":
            write_data(data, **kwargs)
        case "tables":
            write_tables(data, **kwargs)
        case _:
            raise ValueError(f"No valid mode: {mode}. Choose one of ['data', 'tables']")
