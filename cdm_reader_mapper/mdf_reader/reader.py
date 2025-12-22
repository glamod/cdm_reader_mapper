"""Common Data Model (CDM) MDF reader."""

from __future__ import annotations

import ast
import logging
import os
from io import StringIO as StringIO

import pandas as pd

from ..common.json_dict import open_json_file
from ..core.databundle import DataBundle

from .utils.filereader import FileReader
from .utils.utilities import validate_arg, validate_path


def read_mdf(
    source,
    imodel: str | None = None,
    ext_schema_path: str | None = None,
    ext_schema_file: str | None = None,
    ext_table_path: str | None = None,
    year_init: int | None = None,
    year_end: int | None = None,
    encoding: str | None = None,
    chunksize: int | None = None,
    skiprows: int = 0,
    convert_flag: bool = True,
    converter_dict: dict | None = None,
    converter_kwargs: dict | None = None,
    decode_flag: bool = True,
    decoder_dict: dict | None = None,
    validate_flag: bool = True,
    sections: list | None = None,
    pd_kwargs: dict | None = None,
    xr_kwargs: dict | None = None,
) -> DataBundle:
    """Read data files compliant with a user specific data model.

    Reads a data file to a pandas DataFrame using a pre-defined data model.
    Read data is validates against its data model producing a boolean mask
    on output.

    The data model needs to be input to the module as a named model
    (included in the module) or as the path to a valid data model.

    Parameters
    ----------
    source: str
        The file (including path) to be read.
    imodel: str, optional
        Name of internally available input data model.
        e.g. icoads_r300_d704
    ext_schema_path: str, optional
        The path to the external input data model schema file.
        The schema file must have the same name as the directory.
        One of ``imodel`` and ``ext_schema_path`` or ``ext_schema_file`` must be set.
    ext_schema_file: str, optional
        The external input data model schema file.
        One of ``imodel`` and ``ext_schema_path`` or ``ext_schema_file`` must be set.
    ext_table_path: str, optional
        The path to the external input data model code tables.
    year_init: str or int, optional
        Left border of time axis.
    year_end: str or int, optional
        Right border of time axis.
    encoding : str, optional
        The encoding of the input file. Overrides the value in the imodel schema file.
    chunksize : int, optional
          Number of reports per chunk.
    skiprows : int
          Number of initial rows to skip from file, default: 0
    convert_flag: bool, default: True
          If True convert entries by using a pre-defined data model.
    converter_dict: dict of {Hashable: func}, optional
          Functions for converting values in specific columns.
          If None use information from a pre-defined data model.
    converter_kwargs: dict of {Hashable: kwargs}, optional
          Key-word arguments for converting values in specific columns.
          If None use information from a pre-defined data model.
    decode_flag: bool, default: True
          If True decode entries by using a pre-defined data model.
    decoder_dict: dict of {Hashable: func}, optional
          Functions for decoding values in specific columns.
          If None use information from a pre-defined data model.
    validate_flag: bool, default: True
          Validate data entries by using a pre-defined data model.
    sections : list, optional
          List with subset of data model sections to output, optional
          If None read pre-defined data model sections.
    pd_kwargs: dict, optional
          Additional pandas arguments
    xr_kwargs: dict, optional
          Additional xarray arguments

    Returns
    -------
    cdm_reader_mapper.DataBundle

    See Also
    --------
    read: Read either original marine-meteorological or MDF data or CDM tables from disk.
    read_data : Read MDF data and validation mask from disk.
    read_tables : Read CDM tables from disk.
    write: Write either MDF data or CDM tables to disk.
    write_data : Write MDF data and validation mask to disk.
    write_tables : Write CDM tables to disk.
    """
    logging.basicConfig(
        format="%(levelname)s\t[%(asctime)s](%(filename)s)\t%(message)s",
        level=logging.INFO,
        datefmt="%Y%m%d %H:%M:%S",
        filename=None,
    )

    if ext_schema_file:
        validate_path("ext_schema_path", ext_schema_path)

    validate_arg("sections", sections, list)
    validate_arg("chunksize", chunksize, int)
    validate_arg("skiprows", skiprows, int)

    if pd_kwargs is None:
        pd_kwargs = {}

    pd_kwargs["encoding"] = encoding
    pd_kwargs["chunksize"] = chunksize
    pd_kwargs["skiprows"] = skiprows

    convert_kwargs = {
        "convert_flag": convert_flag,
        "converter_dict": converter_dict,
        "converter_kwargs": converter_kwargs,
    }

    decode_kwargs = {
        "decode_flag": decode_flag,
        "decoder_dict": decoder_dict,
    }

    validate_kwargs = {
        "validate_flag": validate_flag,
        "ext_table_path": ext_table_path,
    }

    select_kwargs = {
        "sections": sections,
        "year_init": year_init,
        "year_end": year_end,
    }

    return FileReader(
        imodel=imodel,
        ext_schema_path=ext_schema_path,
        ext_schema_file=ext_schema_file,
    ).read(
        source=source,
        pd_kwargs=pd_kwargs,
        convert_kwargs=convert_kwargs,
        decode_kwargs=decode_kwargs,
        validate_kwargs=validate_kwargs,
        select_kwargs=select_kwargs,
    )


def read_data(
    source,
    mask=None,
    info=None,
    imodel=None,
    col_subset=None,
    encoding: str | None = None,
    **kwargs,
) -> DataBundle:
    """Read MDF data which is already on a pre-defined data model.

    Parameters
    ----------
    source: str
        The data file (including path) to be read.
    mask: str, optional
        The validation file (including path) to be read.
    info: str, optional
        The information file (including path) to be read.
    imodel: str, optional
        Name of internally available input data model.
        e.g. icoads_r300_d704
    col_subset: str, tuple or list, optional
        Specify the section or sections of the file to write.

        - For multiple sections of the tables:
          e.g col_subset = [columns0,...,columnsN]

        - For a single section:
          e.g. list type object col_subset = [columns]

        Column labels could be both string or tuple.
    encoding : str, optional
        The encoding of the input file. Overrides the value in the imodel schema file.

    Returns
    -------
    cdm_reader_mapper.DataBundle

    See Also
    --------
    read: Read original marine-meteorological data as well as MDF data or CDM tables from disk.
    read_mdf : Read original marine-meteorological data from disk.
    read_tables : Read CDM tables from disk.
    write: Write both MDF data or CDM tables to disk.
    write_data : Write MDF data and validation mask to disk.
    write_tables : Write CDM tables to disk.
    """

    def _update_column_labels(columns):
        new_cols = []
        for col in columns:
            try:
                col_ = ast.literal_eval(col)
            except SyntaxError:
                col_ = tuple(col.split(":"))
            except ValueError:
                col_ = col
            new_cols.append(col_)

        if all(isinstance(c, tuple) for c in new_cols):
            return pd.MultiIndex.from_tuples(new_cols)

        return pd.Index(new_cols)

    def _read_csv(ifile, col_subset=None, **kwargs):
        if ifile is None or not os.path.isfile(ifile):
            return pd.DataFrame()

        df = pd.read_csv(ifile, delimiter=",", **kwargs)
        df.columns = _update_column_labels(df.columns)
        if col_subset is not None:
            df = df[col_subset]

        return df

    if info is None:
        info_dict = {}
    else:
        info_dict = open_json_file(info)

    dtype = info_dict.get("dtypes", "object")
    parse_dates = info_dict.get("parse_dates", False)
    if encoding is None:
        encoding = info_dict.get("encoding", None)

    data = _read_csv(
        source,
        col_subset=col_subset,
        dtype=dtype,
        parse_dates=parse_dates,
        encoding=encoding,
    )
    mask = _read_csv(mask, col_subset=col_subset, dtype="boolean")
    return DataBundle(
        data=data,
        columns=data.columns,
        dtypes=dtype,
        parse_dates=parse_dates,
        mask=mask,
        imodel=imodel,
        encoding=encoding,
    )
