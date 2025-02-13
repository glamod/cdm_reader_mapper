"""
Read Common Data Model (CDM) mapping tables.

Created on Thu Apr 11 13:45:38 2019

Reads files with the CDM table format from a file system to a pandas.Dataframe.

All CDM fields are read as objects. Null values are read with the specified null value
in the table files, or as NaN if the na_values argument is set to the a specific null
value in the file.

Reads the full set of files (default), a subset or a single table, as controlled
by cdm_subset:

    - When reading multiple tables, the resulting dataframe is multi-indexed in
        the columns, with (table-name, field) as column names. Merging of tables
        occurs on the report_id field.
    - When reading a single table, the resulting dataframe has simple indexing
        in the columns.

Reads the full set of fields (default) or a subset of it, as controlled by
param col_subset:

    - When reading multiple tables (default or subset), the col_subset is a
        dictionary like: col_subset = {table0:[columns],...tablen:[columns]}
        If a table is not specified in col_subset, all its fields are read.
    - When reading a single table, the col_subset is a list like:
        col_subset = [columns]
    - It is assumed that the column names are all conform to the cdm field names

The full table set (header, observations-"*") is assumed to be in the same directory.

Filenames for tables are assumed to be:
    tableName-<tb_id>.<extension>
with:
    valid tableName: as declared in properties.cdm_tables
    tb_id: any identifier including wildcards if required
    extension: defaulting to 'psv'

When specifying a subset of tables, valid names are those in properties.cdm_tables

@author: iregon
"""

from __future__ import annotations

import glob
import os

import pandas as pd

from cdm_reader_mapper.common import get_filename, logging_hdlr
from cdm_reader_mapper.core.databundle import DataBundle

from . import properties
from .utils.utilities import get_cdm_subset, get_usecols


def read_tables(
    inp_dir,
    prefix=None,
    suffix=None,
    extension="psv",
    cdm_subset=None,
    col_subset=None,
    delimiter="|",
    na_values=None,
):
    """
    Read CDM-table-like files from file system to a pandas.DataFrame.

    Parameters
    ----------
    inp_dir: str
        Path to the input file(s).
    prefix: str, optional
        Prefix of file name structure: ``<prefix>-<table>-*<suffix>.<extension>``.
    suffix: str, optional
        Suffix of file name structure: ``<prefix>-<table>-*<suffix>.<extension>``.
    extension: str
        Extension of file name structure: ``<prefix>-<table>-*<suffix>.<extension>``.
        Default: psv
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
          e.g col_subset = {table0:[columns0],...tableN:[columnsN]}

        - For a single section:
          e.g. list type object col_subset = [columns]
          This variable assumes that the column names are all conform to the cdm field names.
    delimiter: str
        Character or regex pattern to treat as the delimiter while reading with pandas.read_csv.
        Default: '|'
    na_values: Hashable, Iterable of Hashable or dict of {Hashable: Iterable}, optional
        Additional strings to recognize as Na/NaN while reading input file with pandas.read_csv.
        For more details see: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

    Returns
    -------
    cdm_reader_mapper.DataBundle

    See Also
    --------
    read_data : Read MDF data and validation mask from disk.
    read_mdf : Read original marine-meteorological data from disk.
    write_tables : Write CDM tables to disk.
    write_data : Write MDF data and validation mask to disk.
    """
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    # Because how the printers are written, they modify the original data frame!,
    # also removing rows with empty observation_value in observation_tables
    if not os.path.isdir(inp_dir):
        logger.error(f"Data path not found {inp_dir}: ")
        return DataBundle(tables=pd.DataFrame())

    if suffix is None:
        suffix = ""

    # See if there's anything at all:
    pattern = get_filename([prefix, f"*{suffix}"], path=inp_dir, extension=extension)
    files = glob.glob(pattern)

    if len(files) == 0:
        logger.error(f"No files found matching pattern {pattern}")
        return DataBundle(tables=pd.DataFrame())

    # See if subset, if any of the tables is not as specs
    cdm_subset = get_cdm_subset(cdm_subset)

    df_list = []
    for table in cdm_subset:
        if table not in properties.cdm_tables:
            logger.error(f"Requested table {table} not defined in CDM")
            return DataBundle(tables=pd.DataFrame())
        logger.info(f"Getting file path for pattern {table}")
        pattern_ = get_filename(
            [prefix, table, f"*{suffix}"], path=inp_dir, extension=extension
        )
        paths_ = glob.glob(pattern_)
        if len(paths_) != 1:
            logger.warning(
                f"Pattern {pattern_} resulted in multiple files for table {table}: {paths_} "
                "Cannot securely retrieve cdm table(s)"
            )
            continue

        usecols = get_usecols(table, col_subset)
        dfi = pd.read_csv(
            paths_[0],
            delimiter=delimiter,
            usecols=usecols,
            dtype="object",
            na_values=na_values,
            keep_default_na=False,
        )
        if len(dfi) == 0:
            logger.warning(
                f"Table {table} empty in file system, not added to the final DF"
            )
            continue

        dfi = dfi.set_index("report_id", drop=False)
        dfi.columns = pd.MultiIndex.from_product([[table], dfi.columns])
        df_list.append(dfi)

    if len(df_list) == 0:
        logger.error("All tables empty in file system")
        return DataBundle(tables=pd.DataFrame())

    merged = pd.concat(df_list, axis=1, join="outer")
    merged = merged.reset_index(drop=True)
    return DataBundle(tables=merged)
