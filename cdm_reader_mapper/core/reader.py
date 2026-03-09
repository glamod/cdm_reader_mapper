"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations

from typing import get_args

from cdm_reader_mapper.cdm_mapper.reader import read_tables
from cdm_reader_mapper.mdf_reader.reader import read_mdf, read_data

from .databundle import DataBundle

from ..properties import SupportedReadModes

supported_read_modes = get_args(SupportedReadModes)

READERS = {
    "mdf": read_mdf,
    "data": read_data,
    "tables": read_tables,
}


def read(
    source: str,
    mode: SupportedReadModes = "mdf",
    **kwargs,
) -> DataBundle:
    """Read either original marine-meteorological data or MDF data or CDM tables from disk.

    Parameters
    ----------
    source: str
        Source of the input data.
    mode: str, {mdf, data, tables}, default: mdf
        Read data mode:

          * "mdf" to read original marine-meteorological data from disk and convert them to MDF data
          * "data" to read MDF data from disk
          * "tables" to read CDM tables from disk. Map MDF data to CDM tables with :py:func:`DataBundle.map_model`.

    Returns
    -------
    DataBundle

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
    `kwargs` are the keyword arguments for the specific `mode` reader.

    """
    if mode not in supported_read_modes:
        raise ValueError(
            f"No valid mode: {mode}. Choose one of {supported_read_modes}."
        )

    return READERS[mode](source, **kwargs)
