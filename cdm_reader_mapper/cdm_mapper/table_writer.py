"""
Write Common Data Model (CDM) mapping tables.

Created on Thu Apr 11 13:45:38 2019

Exports tables written in the C3S Climate Data Store Common Data Model (CDM) format to ascii files,
The tables format is contained in a python dictionary, stored as an attribute in a pandas.DataFrame.

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

import ast

import numpy as np
import pandas as pd

from cdm_reader_mapper.common import logging_hdlr

from ._utilities import dict_to_tuple_list, get_cdm_subset, get_filename


def print_integer(data, null_label):
    """
    Print all elements that have 'int' as type attribute.

    Parameters
    ----------
    data: data tables to print
    null_label: specified how nan are represented

    Returns
    -------
    data: data as int type
    """

    def _return_str(x, null_label):
        if pd.isna(x):
            return null_label
        return str(
            int(float(x))
        )  # ValueError: invalid literal for int() with base 10: '5.0'

    return data.apply(lambda x: _return_str(x, null_label))


def print_float(data, null_label, decimal_places):
    """
    Print all elements that have 'float' as type attribute.

    Parameters
    ----------
    data: data tables to print
    null_label: specified how nan are represented
    decimal_places: number of decimal places

    Returns
    -------
    data: data as float type
    """

    def _return_str(x, null_label, format_float):
        if pd.isna(x):
            return null_label
        return format_float.format(float(x))

    format_float = "{:." + str(decimal_places) + "f}"
    return data.apply(lambda x: _return_str(x, null_label, format_float))


def print_datetime(data, null_label):
    """
    Print datetime objects in the format: "%Y-%m-%d %H:%M:%S".

    Parameters
    ----------
    data: date time elements
    null_label: specified how nan are represented

    Returns
    -------
    data: data as datetime objects
    """

    def _return_str(x, null_label):
        if pd.isna(x):
            return null_label
        if isinstance(x, str):
            return x
        return x.strftime("%Y-%m-%d %H:%M:%S")

    return data.apply(lambda x: _return_str(x, null_label))


def print_varchar(data, null_label):
    """
    Print string elements.

    Parameters
    ----------
    data: data tables to print
    null_label: specified how nan are represented

    Returns
    -------
    data: data as string objects
    """

    def _return_str(x, null_label):
        if isinstance(x, list):
            if len(x) == 0:
                return null_label
            return str(x)
        if pd.isna(x):
            return null_label
        return str(x)

    return data.apply(lambda x: _return_str(x, null_label))


def print_integer_array(data, null_label):
    """
    Print a series of integer objects as array.

    Parameters
    ----------
    data: data tables to print
    null_label: specified how nan are represented

    Returns
    -------
    data: array of int objects
    """
    return data.apply(print_integer_array_i, null_label=null_label)


def print_varchar_array(data, null_label):
    """
    Print a series of string objects as array.

    Parameters
    ----------
    data: data tables to print
    null_label: specified how nan are represented

    Returns
    -------
    data: array of varchar objects
    """
    return data.apply(print_varchar_array_i)


def print_integer_array_i(row, null_label=None):
    """
    NEED DOCUMENTATION.

    Parameters
    ----------
    row
    null_label

    Returns
    -------
    data: int
    """
    row = row if not isinstance(row, str) else ast.literal_eval(row)
    row = row if isinstance(row, list) else [row]
    str_row = [str(int(x)) for x in row if np.isfinite(x)]
    string = ",".join(filter(bool, str_row))
    if len(string) > 0:
        return "{" + string + "}"
    return null_label


def print_varchar_array_i(row, null_label=None):
    """
    NEED DOCUMENTATION.

    Parameters
    ----------
    row
    null_label

    Returns
    -------
    data: varchar
    """
    row = row if not isinstance(row, str) else ast.literal_eval(row)
    row = row if isinstance(row, list) else [row]
    str_row = [str(x) for x in row if np.isfinite(x)]
    string = ",".join(filter(bool, str_row))
    if len(string) > 0:
        return "{" + string + "}"
    return null_label


def print_values(table, table_atts, null_label=None, logger=None):
    """
    NEED DOCUMENTATION.

    Parameters
    ----------
    table
    table_atts
    null_label

    Returns
    -------
    pd.DataFrame
    """
    ascii_table = pd.DataFrame(
        index=table.index, columns=table_atts.keys(), dtype="object"
    )
    for iele in table_atts.keys():
        if iele in table:
            itype = table_atts.get(iele).get("data_type")
            if printers.get(itype):
                iprinter_kwargs = iprinters_kwargs.get(itype)
                if iprinter_kwargs:
                    kwargs = {x: table_atts.get(iele).get(x) for x in iprinter_kwargs}
                else:
                    kwargs = {}
                ascii_table[iele] = printers.get(itype)(
                    table[iele], null_label, **kwargs
                )
            else:
                logger.error(f"No printer defined for element {iele}")
    return ascii_table


def table_to_ascii(
    data,
    atts={},
    delimiter="|",
    null_label="null",
    col_subset=None,
    cdm_complete=True,
    filename=None,
):
    """
    Export a cdm table to an ascii file.

    Exports tables written in the C3S Climate Data Store Common Data Model (CDM) format to ascii files.
    The tables format is contained in a python dictionary, stored as an attribute in a ``pandas.DataFrame``.

    Parameters
    ----------
    table:
        pandas.Dataframe to export
    table_atts: attributes of the pandas.Dataframe stored as a python dictionary.
            This contains all element names, characteristics and types encoding,
            as well as other characteristics e.g. decimal places, etc.
    delimiter:
        default '|'
    null_label:
        specified how nan are represented
    cdm_complete: if we export the entire set of tables.
        default is ``True``
    filename:
        the name of the file to stored the data

    Returns
    -------
    Saves cdm tables as ascii files
    """
    logger = logging_hdlr.init_logger(__name__, level="INFO")

    if "observation_value" in data:
        data = data.dropna(subset=["observation_value"])
    elif "observation_value" in atts.keys():
        data = pd.DataFrame()

    csv_kwargs = {
        "index": False,
        "sep": delimiter,
        "header": True,
        "mode": "w",
    }

    if data.empty:
        logger.warning("No observation values in table")
        ascii_table = pd.DataFrame(columns=atts.keys(), dtype="object")
        ascii_table.to_csv(filename, **csv_kwargs)
        return

    if col_subset:
        if isinstance(col_subset, dict):
            col_subset = dict_to_tuple_list(col_subset)
        cdm_complete = False
        data = data[col_subset]

    if atts:
        data = print_values(data, atts, null_label=null_label, logger=logger)
        columns_to_ascii = (
            [x for x in atts.keys() if x in data.columns]
            if not cdm_complete
            else atts.keys()
        )
    else:
        data = data.fillna(null_label)
        columns_to_ascii = data.columns

    data.to_csv(
        filename,
        columns=columns_to_ascii,
        na_rep=null_label,
        **csv_kwargs,
    )


def write_tables(
    cdm_table,
    out_dir=".",
    table_name=None,
    prefix=None,
    suffix=None,
    extension="psv",
    filename=None,
    cdm_subset=None,
    col_subset=None,
    cdm_complete=True,
    delimiter="|",
    null_label="null",
):
    """Write pandas.DataFrame to CDM-table file on file system.

    Parameters
    ----------
    cdm_table: pandas.DataFrame
        Dataframe to export.
    out_dir: str
        Path of the output directory.
        Default: current directory
    table_name: str, optional
        Name of the CDM table in `cdm_table`.
        Note: This is necessary if ``cdm_table`` contains only one single table with single-index columns.
    prefix: str, optional
        Prefix of file name structure: ``<prefix>-<table>-*<suffix>.<extension>``.
    suffix: str, optional
        Suffix of file name structure: ``<prefix>-<table>-*<suffix>.<extension>``.
    extension: str
        Extension of file name structure: ``<prefix>-<table>-*<suffix>.<extension>``.
        Default: psv
    filename: str or dict, optional
        Name of the output file name(s).
        List one filename for each table name in ``cdm_table`` ({<table>:<filename>}).
        Default: Automatically create file name from table name, ``prefix`` and ``suffix``.
    null_label: str
        Specifies how NaN values are represented in the DataFrame.
        Default: null
    cdm_subset: str or list, optional
        Specifies a subset of tables or a single table.

        - For multiple subsets of tables:
          This function returns a pandas.DataFrame that is multi-index at
          the columns, with (table-name, field) as column names. Tables are merged via the report_id field.

        - For a single table:
          This function returns a pandas.DataFrame with a simple indexing for the columns.
    col_subset: str, list or dict, optional
        Specify the section or sections of the file to read.

        - For multiple sections of the tables:
          e.g ``col_subset = {table0:[columns0],...tableN:[columnsN]}``

        - For a single section:
          e.g. ``list type object col_subset = [columns]``
          This variable assumes that the column names are all conform to the cdm field names.
    cdm_complete: bool
        If True extract the all available CDM columns.
        Default: True
    delimiter: str
        Character or regex pattern to treat as the delimiter while reading with pandas.read_csv.
        Default: '|'

    Note
    ----
    Use this function after reading CDM tables.
    """
    logger = logging_hdlr.init_logger(__name__, level="INFO")

    cdm_subset = get_cdm_subset(cdm_subset)

    if isinstance(cdm_table, pd.DataFrame):
        if table_name:
            cdm_table = {table_name: {"data": cdm_table}}
        else:
            data = cdm_table.copy()
            cdm_table = {}
            for table in cdm_subset:
                if table in data.columns:
                    cdm_table[table] = {"data": data[table]}

    if not cdm_table:
        logger.warning("All CDM tables are empty")
        return

    if isinstance(filename, str):
        filename = {table_name: filename}
    elif filename is None:
        filename = {}

    for table in cdm_subset:
        if table not in cdm_table.keys():
            logger.warning(f"No file for table {table} found.")
            continue
        logger.info(f"Printing table {table}")

        filename_ = filename.get(table)
        if not filename_:
            filename_ = get_filename(
                [prefix, table, suffix], path=out_dir, extension=extension
            )

        table_to_ascii(
            **cdm_table[table],
            delimiter=delimiter,
            null_label=null_label,
            filename=filename_,
            col_subset=col_subset,
            cdm_complete=cdm_complete,
        )


printers = {
    "int": print_integer,
    "numeric": print_float,
    "varchar": print_varchar,
    "timestamp with timezone": print_datetime,
    "int[]": print_integer_array,
    "varchar[]": print_varchar_array,
}

iprinters_kwargs = {
    "numeric": ["decimal_places"],
    "numeric[]": ["decimal_places"],
}
