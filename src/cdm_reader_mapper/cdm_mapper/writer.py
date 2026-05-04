"""
Write Common Data Model (CDM) mapping tables.

Created on Thu Apr 11 13:45:38 2019

Exports tables written in the C3S Climate Data Store Common Data Model (CDM) format to ascii files,
The tables format is contained in a python dictionary, stored as an attribute in a pandas.DataFrame
(or Iterable[pd.DataFrame]).

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
from pathlib import Path
from typing import Any, get_args

import pandas as pd

from cdm_reader_mapper.common import get_filename, logging_hdlr

from ..properties import SupportedFileTypes
from .tables.tables import get_cdm_atts
from .utils.conversions import convert_from_str_df, convert_to_str_df
from .utils.utilities import adjust_filename, dict_to_tuple_list, get_cdm_subset


def _table_to_file(
    data: pd.DataFrame,
    filename: str | Path,
    data_format: SupportedFileTypes = "parquet",
    delimiter: str = "|",
    encoding: str = "utf-8",
    **kwargs: Any,
) -> None:
    r"""
    Write a pandas DataFrame to disk in a selected file format.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to be written to disk.
    filename : str or Path-like
        Destination file path.
    data_format : {"parquet", "csv", "feather"}, default "parquet"
        Output file format.
    delimiter : str, default "|"
        Field delimiter used when writing CSV files.
    encoding : str, default "utf-8"
        Text encoding used when writing CSV files.
    \**kwargs : Any
        Additional keyword arguments forwarded to the underlying pandas
        serialization function.

    Returns
    -------
    None
        This function performs a write operation and returns no value.

    Raises
    ------
    ValueError
        If `data_format` is not one of the supported formats defined by
        ``SupportedFileTypes``.
    """
    data = data.dropna(how="all")
    if data_format == "csv":
        header = True
        wmode = "w"
        data.to_csv(
            filename,
            index=False,
            header=header,
            mode=wmode,
            sep=delimiter,
            encoding=encoding,
            **kwargs,
        )
    elif data_format == "parquet":
        data.to_parquet(filename, engine="pyarrow", compression="snappy", **kwargs)
    elif data_format == "feather":
        data.to_feather(filename, **kwargs)
    else:
        raise ValueError(f"data_format must be one of {get_args(SupportedFileTypes)} not {data_format}.")


def write_tables(
    data: pd.DataFrame,
    data_format: SupportedFileTypes = "parquet",
    out_dir: str | Path | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    extension: str | None = None,
    filename: str | Path | dict[str, str | Path] | None = None,
    separator: str | None = "-",
    cdm_subset: str | list[str] | None = None,
    col_subset: str | list[str] | dict[str, str] | None = None,
    delimiter: str = "|",
    encoding: str = "utf-8",
    from_str: bool | None = None,
    to_str: bool | None = None,
    imodel: str | None = None,
) -> None:
    """
    Write pandas.DataFrame to CDM-table file on file system.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to export.
    data_format : {"csv", "parquet", "feather"}, default: "parqeut"
        Format of input data file(s).
    out_dir : str, optional
        Path to the output directory.
        Defaults to current directory.
    prefix : str, optional
        Prefix of file name structure: ``<prefix>-<table>-*<suffix>.<extension>``.
    suffix : str, optional
        Suffix of file name structure: ``<prefix>-<table>-*<suffix>.<extension>``.
    extension : str, optional
        Extension of file name structure: ``<prefix>-<table>-*<suffix>.<extension>``.
    filename : str, Path-like or dict, optional
        Name of the output file name(s).
        List one filename for each table name in ``data`` ({<table>:<filename>}).
        If None, automatically create file name from table name, ``prefix`` and ``suffix``.
    separator : str, optional
        Separator to join the file name pattern components (default "-").
    cdm_subset : str or list of str, optional
        Specifies a subset of tables or a single table.

        - For multiple subsets of tables:
          This function returns a pandas.DataFrame that is multi-index at
          the columns, with (table-name, field) as column names. Tables are merged via the report_id field.

        - For a single table:
          This function returns a pandas.DataFrame with a simple indexing for the columns.
    col_subset : str, list or dict, optional
        Specify the section or sections of the file to write.

        - For multiple sections of the tables:
          e.g col_subset = {table0:[columns0],...tableN:[columnsN]}

        - For a single section:
          e.g. list type object col_subset = [columns]
          This variable assumes that the column names are all conform to the cdm field names.
    delimiter : str, default: "|"
        Character or regex pattern to treat as the delimiter while reading with df.to_csv.
        This is only relevant if `data_format` is "csv".
    encoding : str
        A string representing the encoding to use in the output file, defaults to utf-8.
        This is only relevant if `data_format` is "csv".
    from_str : bool, optional
        If True convert original string data to `imodel`-specific data types.
    to_str : bool, optional
        If True convert original `imodel`-specific data types to strings.
    imodel : str , optional
        Name of data model, e.g. icoads.
        Must be set if either `from_str` or `to_str` is set.

    See Also
    --------
    write: Write either MDF data or CDM tables to disk.
    write_data : Write MDF data and validation mask to disk.
    read: Read either original marine-meteorological data or MDF data or CDM tables from disk.
    read_tables : Read CDM tables from disk.
    read_data : Read MDF data and validation mask from disk.
    read_mdf : Read original marine-meteorological data from disk.

    Notes
    -----
    Use this function after reading CDM tables.
    """
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    supported_file_types = get_args(SupportedFileTypes)
    if data_format not in supported_file_types:
        raise ValueError(f"data_format must be one of {supported_file_types}, not {data_format}.")

    cdm_subset = get_cdm_subset(cdm_subset)

    if col_subset:
        to_select: str | list[str] | list[tuple[str, str]]
        if isinstance(col_subset, dict):
            to_select = dict_to_tuple_list(col_subset)
        else:
            to_select = cdm_subset
        data = data[to_select]

    if data.empty:
        logger.warning("All CDM tables are empty")
        return

    if isinstance(filename, dict):
        cdm_subset = list(filename.keys())
    elif isinstance(filename, (str, Path)):
        filename = {table_name: filename for table_name in cdm_subset}
    elif filename is None:
        filename = {}

    out_dir = out_dir or "."
    out_dir = Path(out_dir)

    extension = extension or data_format

    if to_str is True:
        data = convert_to_str_df(data.copy(), imodel=imodel, cdm_subset=cdm_subset)

    if from_str is True:
        data = convert_from_str_df(data.copy(), imodel=imodel, cdm_subset=cdm_subset)

    for table in cdm_subset:
        cdm_atts = get_cdm_atts(table)[table]
        table_columns = pd.Index(cdm_atts.keys())
        if table in data:
            cdm_table = data[table]
        elif data.columns.equals(table_columns):
            cdm_table = data
        else:
            cdm_table = pd.DataFrame(columns=table_columns)

        filename_ = filename.get(table)
        if not filename_:
            filename_ = get_filename(
                [prefix, table, suffix],
                path=out_dir,
                extension=extension,
                separator=separator,
            )
        filename_ = adjust_filename(str(filename_), table=table, extension=extension)
        if len(Path(filename_).parts) == 1:
            filename_ = out_dir / filename_

        logger.info("Writing table %s: %s", table, filename_)
        _table_to_file(
            cdm_table,
            delimiter=delimiter,
            encoding=encoding,
            filename=filename_,
            data_format=data_format,
        )
