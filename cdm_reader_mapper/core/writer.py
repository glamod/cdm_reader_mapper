"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations

from cdm_reader_mapper.cdm_mapper.writer import write_tables
from cdm_reader_mapper.mdf_reader.writer import write_data


def write(
    data,
    mask=None,
    dtypes="object",
    parse_dates=[],
    encoding="utf-8",
    mode="data",
    **kwargs,
):
    """Write either MDF data or CDM tables on disk.

    Parameters
    ----------
    data: pandas.DataFrame
        pandas.DataFrame to export.
    mask: pandas.DataFrame, optional
        validation mask to export.
        Use only if ``mode`` is "data".
    dtypes: dict
        Dictionary of data types on ``data``.
        Dump ``dtypes`` and ``parse_dates`` to json information file.
        Use only if ``mode`` is "data".
        Default: "object"
    parse_dates:
        Information of how to parse dates in :py:attr:`data`.
        Dump ``dtypes`` and ``parse_dates`` to json information file.
        For more information see :py:func:`pandas.read_csv`.
        Use only if ``mode`` is "data".
        Default: []
    encoding: str
        A string representing the encoding to use in the output file.
        Default: utf-8.
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
        write_data(
            data,
            mask=mask,
            dtypes=dtypes,
            parse_dates=parse_dates,
            encoding=encoding,
            **kwargs,
        )
    elif mode == "tables":
        write_tables(data, encoding=encoding, **kwargs)
    else:
        raise ValueError(f"No valid mode: {mode}. Choose one of ['data', 'tables']")
