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

import ast
import os

import numpy as np
import pandas as pd

from cdm_reader_mapper.common import logging_hdlr


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
        return format_float.format(x)

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


# TODO: tell this to dave and delete them... put error messages in functions above
def print_float_array(data, null_label, decimal_places=None):
    """Print a series of float objects as array."""
    return "float array not defined in printers"


def print_datetime_array(data, null_label):
    """Print a series of datetime objects as array."""
    return "datetime tz array not defined in printers"


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


def table_to_ascii(
    table,
    table_atts,
    delimiter="|",
    null_label="null",
    cdm_complete=True,
    filename=None,
    log_level="INFO",
):
    """
    Export a cdm table to an ascii file.

    Exports tables written in the C3S Climate Data Store Common Data Model (CDM) format to ascii files.
    The tables format is contained in a python dictionary, stored as an attribute in a ``pandas.DataFrame``
    (or ``pd.io.parsers.TextFileReader``).

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
    log_level:
        level of logging information to be saved

    Returns
    -------
    Saves cdm tables as ascii files
    """
    logger = logging_hdlr.init_logger(__name__, level=log_level)
    empty_table = False
    if "observation_value" in table:
        table = table.dropna(subset=["observation_value"])
        empty_table = True if len(table) == 0 else False
    elif "observation_value" in table_atts.keys():
        empty_table = True
    else:
        empty_table = True if len(table) == 0 else False
    if empty_table:
        logger.warning("No observation values in table")
        ascii_table = pd.DataFrame(columns=table_atts.keys(), dtype="object")
        ascii_table.to_csv(filename, index=False, sep=delimiter, header=True, mode="w")
        return

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
        else:
            ascii_table[iele] = null_label

    header = True
    wmode = "w"
    columns_to_ascii = (
        [x for x in table_atts.keys() if x in table.columns]
        if not cdm_complete
        else table_atts.keys()
    )
    ascii_table.to_csv(
        filename,
        index=False,
        sep=delimiter,
        columns=columns_to_ascii,
        header=header,
        mode=wmode,
    )


def cdm_to_ascii(
    cdm_table,
    delimiter="|",
    null_label="null",
    cdm_complete=True,
    extension="psv",
    out_dir=None,
    suffix=None,
    prefix=None,
    log_level="INFO",
):
    """
    Export a complete cdm file with multiple tables to an ascii file.

    Exports a complete cdm file with multiple tables written in the C3S Climate Data Store Common Data Model (CDM)
    format to ascii files.
    The tables format is contained in a python dictionary, stored as an attribute in a ``pandas.DataFrame``
    (or ``pd.io.parsers.TextFileReader``).

    Parameters
    ----------
    cdm_table:
        common data model tables to export
    delimiter:
        default '|'
    null_label:
        specified how nan are represented
    cdm_complete:
        extract the entire cdm file
    extension:
        default 'psv'
    out_dir:
        where to stored the ascii file
    suffix:
        file suffix
    prefix:
        file prefix
    log_level:
        level of logging information

    Returns
    -------
    Saves the cdm tables as ascii files in the given directory with a psv extension.
    """
    logger = logging_hdlr.init_logger(__name__, level=log_level)
    # Because how the printers are written, they modify the original data frame!,
    # also removing rows with empty observation_value in observation_tables
    extension = "." + extension
    if not cdm_table:
        logger.warning("All CDM tables are empty")
        return
    for table in cdm_table.keys():
        logger.info(f"Printing table {table}")
        filename = "-".join(filter(bool, [prefix, table, suffix])) + extension
        filepath = filename if not out_dir else os.path.join(out_dir, filename)
        table_to_ascii(
            cdm_table[table]["data"],
            cdm_table[table]["atts"],
            delimiter=delimiter,
            null_label=null_label,
            cdm_complete=cdm_complete,
            filename=filepath,
            log_level=log_level,
        )


printers = {
    "int": print_integer,
    "numeric": print_float,
    "varchar": print_varchar,
    "timestamp with timezone": print_datetime,
    "int[]": print_integer_array,
    "numeric[]": print_float_array,
    "varchar[]": print_varchar_array,
    "timestamp with timezone[]": print_datetime_array,
}

iprinters_kwargs = {
    "numeric": ["decimal_places"],
    "numeric[]": ["decimal_places"],
}
