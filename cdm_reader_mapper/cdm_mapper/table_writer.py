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

import os

import pandas as pd

from cdm_reader_mapper.common import logging_hdlr

from . import properties
from .tables.tables import get_cdm_atts


def table_to_ascii(
    table,
    delimiter="|",
    filename=None,
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
    filename:
        the name of the file to stored the data

    Returns
    -------
    Saves cdm tables as ascii files
    """
    table = table.dropna(how="all")

    header = True
    wmode = "w"
    table.to_csv(
        filename,
        index=False,
        sep=delimiter,
        header=header,
        mode=wmode,
    )


def cdm_to_ascii(
    cdm_tables,
    delimiter="|",
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
    cdm_tables:
        common data model tables to export
    delimiter:
        default '|'
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
    if cdm_tables.empty:
        logger.warning("All CDM tables are empty")
        return
    for table in properties.cdm_tables:
        if table not in cdm_tables:
            cdm_atts = get_cdm_atts(table)
            cdm_table = pd.DataFrame(columns=cdm_atts.keys())
        else:
            cdm_table = cdm_tables[table]
        filename = "-".join(filter(bool, [prefix, table, suffix])) + extension
        filepath = filename if not out_dir else os.path.join(out_dir, filename)
        logger.info(f"Writing table {table}: {filepath}")
        table_to_ascii(
            cdm_table,
            delimiter=delimiter,
            filename=filepath,
        )
