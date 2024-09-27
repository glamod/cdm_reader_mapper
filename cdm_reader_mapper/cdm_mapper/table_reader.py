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

from cdm_reader_mapper.common import logging_hdlr

from . import properties


def get_cdm_subset(cdm_subset):
    """Return cdm_subset."""
    if cdm_subset is None:
        return properties.cdm_tables
    elif not isinstance(cdm_subset, list):
        return [cdm_subset]
    return cdm_subset


def get_usecols(tb, col_subset=None):
    """Return usecols for pandas.read_csv function."""
    if isinstance(col_subset, str):
        return [col_subset]
    elif isinstance(col_subset, list):
        return col_subset
    elif isinstance(col_subset, dict):
        return col_subset.get(tb)


def read_tables(
    tb_path,
    tb_id="*",
    cdm_subset=None,
    delimiter="|",
    extension="psv",
    col_subset=None,
    log_level="INFO",
    na_values=[],
):
    """
    Read CDM table like files from file system to a pandas data frame.

    Parameters
    ----------
    tb_path:
        path to the file
    tb_id:
        any identifier including wildcards if required extension, defaulting to 'psv'
    cdm_subset:
        specifies a subset of tables or a single table.

        - For multiple subsets of tables:
          This function returns a pandas.DataFrame that is multi-index at
          the columns, with (table-name, field) as column names. Tables are merged via the report_id field.

        - For a single table:
          This function returns a pandas.DataFrame with a simple indexing for the columns.

    delimiter:
        default is '|'
    extension:
        default is psv
    col_subset:
        a python dictionary specifying the section or sections of the file to read

        - For multiple sections of the tables:
          e.g ``col_subset = {table0:[columns],...tablen:[columns]}``

        - For a single section:
          e.g. ``list type object col_subset = [columns]``
          This variable assumes that the column names are all conform to the cdm field names.

    log_level: Level of logging messages to save
    na_values: specifies the format of NaN values

    Returns
    -------
    pandas.DataFrame: either the entire file or a subset of it.

    Note
    ----
    Logs specific messages if there is any error.
    """
    logger = logging_hdlr.init_logger(__name__, level=log_level)
    # Because how the printers are written, they modify the original data frame!,
    # also removing rows with empty observation_value in observation_tables
    if not os.path.isdir(tb_path):
        logger.error(f"Data path not found {tb_path}: ")
        return pd.DataFrame()

    # See if there's anything at all:
    files = glob.glob(os.path.join(tb_path, f"*{tb_id}*.{extension}"))
    if len(files) == 0:
        logger.error(f"No files found matching pattern {tb_id}")
        return pd.DataFrame()

    # See if subset, if any of the tables is not as specs
    cdm_subset = get_cdm_subset(cdm_subset)

    for tb in cdm_subset:
        if tb not in properties.cdm_tables:
            logger.error(f"Requested table {tb} not defined in CDM")
            return pd.DataFrame()

    file_paths = {}
    for tb in cdm_subset:
        logger.info(f"Getting file path for pattern {tb}")
        patterns_ = os.path.join(tb_path, f"{tb}-{tb_id}.{extension}")
        paths_ = glob.glob(patterns_)
        if len(paths_) == 1:
            file_paths[tb] = paths_[0]
            continue
        logger.error(
            f"Pattern {tb_id} resulted in multiple files for table {tb}. "
            "Cannot securely retrieve cdm table(s)"
        )
        return pd.DataFrame()

    if len(file_paths) == 0:
        logger.error(f"No cdm table files found for search patterns: {files}")
        return pd.DataFrame()

    logger.info(
        "Reading into dataframe data files {}: ".format(
            ",".join(list(file_paths.values()))
        )
    )

    if len(cdm_subset) == 1:
        indexing = False
    else:
        indexing = True

    df_list = []
    for tb, tb_file in file_paths.items():
        usecols = get_usecols(tb, col_subset)

        dfi = pd.read_csv(
            tb_file,
            delimiter=delimiter,
            usecols=usecols,
            dtype="object",
            na_values=na_values,
            keep_default_na=False,
        )
        if len(dfi) == 0:
            logger.warning(
                f"Table {tb} empty in file system, not added to the final DF"
            )
            continue

        if indexing is False:
            return dfi

        dfi = dfi.set_index("report_id", drop=False)
        dfi.columns = pd.MultiIndex.from_product([[tb], dfi.columns])
        df_list.append(dfi)

    if len(df_list) == 0:
        logger.error("All tables empty in file system")
        return pd.DataFrame()

    merged = pd.concat(df_list, axis=1, join="outer")
    merged = merged.reset_index(drop=True)
    return merged
