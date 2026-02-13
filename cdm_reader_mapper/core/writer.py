"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations

from typing import Iterable, get_args

import pandas as pd

from cdm_reader_mapper.cdm_mapper.writer import write_tables
from cdm_reader_mapper.mdf_reader.writer import write_data

from ..properties import SupportedWriteModes

supported_write_modes = get_args(SupportedWriteModes)

WRITERS = {
    "data": write_data,
    "tables": write_tables,
}


def write(
    data: pd.DataFrame | Iterable[pd.DataFrame],
    mode: SupportedWriteModes = "data",
    **kwargs,
) -> None:
    """Write either MDF data or CDM tables on disk.

    Parameters
    ----------
    data: pandas.DataFrame or Iterable[pd.DataFrame]
        Data to export.
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
    if mode not in supported_write_modes:
        raise ValueError(
            f"No valid mode: {mode}. Choose one of {supported_write_modes}."
        )

    return WRITERS[mode](data, **kwargs)
