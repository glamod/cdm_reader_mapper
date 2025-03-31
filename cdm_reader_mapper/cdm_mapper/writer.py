"""
Write Common Data Model (CDM) mapping tables.

Created on Thu Apr 11 13:45:38 2019

Exports tables written in the C3S Climate Data Store Common Data Model (CDM) format to ascii files,
The tables format is contained in a python dictionary, stored as an attribute in a pandas.DataFrame
(or pd.io.parsers.TextFileReader).

This module uses a set of printer functions to "print" element values to a
string object before exporting them to a final ascii file.

Each of the CDM table element's has a data type (pseudo-sql as defined in the CDM documentation) which defines
which printer function needs to be used.

Numeric data types are printed with an specific number of decimal places, defined in the data element attributes. This
can vary according to each CDM, element, imodel and mapping .json file. If this is not defined in the input attributes
of the imodel, the number of decimal places used comes from a default tool defined in properties.py

@author: iregon
"""

from __future__ import annotations

import pandas as pd

from cdm_reader_mapper.common import get_filename, logging_hdlr

from .tables.tables import get_cdm_atts
from .utils.utilities import adjust_filename, dict_to_tuple_list, get_cdm_subset


def _table_to_ascii(
    data,
    delimiter="|",
    encoding="utf-8",
    col_subset=None,
    filename=None,
) -> None:
    data = data.dropna(how="all")

    header = True
    wmode = "w"
    data.to_csv(
        filename,
        index=False,
        sep=delimiter,
        header=header,
        mode=wmode,
        encoding=encoding,
    )


def write_tables(
    data,
    out_dir=".",
    prefix=None,
    suffix=None,
    extension="psv",
    filename=None,
    cdm_subset=None,
    col_subset=None,
    delimiter="|",
    encoding="utf-8",
    **kwargs,
) -> None:
    """Write pandas.DataFrame to CDM-table file on file system.

    Parameters
    ----------
    data: pandas.DataFrame
        pandas.DataFrame to export.
    out_dir: str
        Path to the output directory.
        Default: current directory
    prefix: str, optional
        Prefix of file name structure: ``<prefix>-<table>-*<suffix>.<extension>``.
    suffix: str, optional
        Suffix of file name structure: ``<prefix>-<table>-*<suffix>.<extension>``.
    extension: str
        Extension of file name structure: ``<prefix>-<table>-*<suffix>.<extension>``.
        Default: psv
    filename: str or dict, optional
        Name of the output file name(s).
        List one filename for each table name in ``data`` ({<table>:<filename>}).
        Default: Automatically create file name from table name, ``prefix`` and ``suffix``.
    cdm_subset: str or list, optional
        Specifies a subset of tables or a single table.

        - For multiple subsets of tables:
          This function returns a pandas.DataFrame that is multi-index at
          the columns, with (table-name, field) as column names. Tables are merged via the report_id field.

        - For a single table:
          This function returns a pandas.DataFrame with a simple indexing for the columns.
    col_subset: str, list or dict, optional
        Specify the section or sections of the file to write.

        - For multiple sections of the tables:
          e.g col_subset = {table0:[columns0],...tableN:[columnsN]}

        - For a single section:
          e.g. list type object col_subset = [columns]
          This variable assumes that the column names are all conform to the cdm field names.
    delimiter: str
        Character or regex pattern to treat as the delimiter while reading with df.to_csv.
        Default: '|'
    encoding: str
        A string representing the encoding to use in the output file, defaults to utf-8.

    See Also
    --------
    write: Write either MDF data or CDM tables to disk.
    write_data : Write MDF data and validation mask to disk.
    read: Read either original marine-meteorological data or MDF data or CDM tables from disk.
    read_tables : Read CDM tables from disk.
    read_data : Read MDF data and validation mask from disk.
    read_mdf : Read original marine-meteorological data from disk.

    Note
    ----
    Use this function after reading CDM tables.
    """
    logger = logging_hdlr.init_logger(__name__, level="INFO")

    cdm_subset = get_cdm_subset(cdm_subset)

    if col_subset:
        if isinstance(col_subset, dict):
            col_subset = dict_to_tuple_list(col_subset)
        data = data[col_subset]

    if data.empty:
        logger.warning("All CDM tables are empty")
        return

    if isinstance(filename, str):
        filename = {table_name: filename for table_name in cdm_subset}
    elif filename is None:
        filename = {}

    for table in cdm_subset:
        if table not in data:
            cdm_atts = get_cdm_atts(table)
            cdm_table = pd.DataFrame(columns=cdm_atts.keys())
        else:
            cdm_table = data[table]

        filename_ = filename.get(table)
        if not filename_:
            filename_ = get_filename(
                [prefix, table, suffix], path=out_dir, extension=extension
            )
        filename_ = adjust_filename(filename_, table=table, extension=extension)
        logger.info(f"Writing table {table}: {filename_}")
        _table_to_ascii(
            cdm_table,
            delimiter=delimiter,
            encoding=encoding,
            filename=filename_,
        )
