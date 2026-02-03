"""Common Data Model (CDM) MDF reader."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Any, get_args

import pandas as pd

from cdm_reader_mapper import DataBundle

from ..common.json_dict import open_json_file

from .utils.filereader import FileReader
from .utils.utilities import validate_arg

from .utils.utilities import as_list, as_path, read_csv, read_parquet, read_feather

from ..properties import SupportedFileTypes

READERS = {
    "csv": read_csv,
    "parquet": read_parquet,
    "feather": read_feather,
}


def validate_read_mdf_args(
    *,
    source: str | Path,
    imodel: str | None = None,
    ext_schema_path: str | Path | None = None,
    ext_schema_file: str | Path | None = None,
    year_init: int | None = None,
    year_end: int | None = None,
    chunksize: int | None = None,
    skiprows: int | None = None,
):
    """
    Validate arguments for reading an MDF file.

    This function performs validation on file paths and numeric arguments
    required for reading an MDF dataset.

    Raises
    ------
    FileNotFoundError
        If the source file does not exist.
    ValueError
        If required arguments are missing or numeric constraints are violated.
    """
    source = as_path(source, "source")

    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    if not imodel and not (ext_schema_path or ext_schema_file):
        raise ValueError(
            "One of imodel or ext_schema_path/ext_schema_file must be provided"
        )

    validate_arg("chunksize", chunksize, int)
    if chunksize is not None and chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")

    validate_arg("skiprows", skiprows, int)
    if skiprows is not None and skiprows < 0:
        raise ValueError("skiprows must be >= 0")

    if year_init is not None and year_end is not None:
        if year_init > year_end:
            raise ValueError("year_init must be <= year_end")


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
    skiprows: int = None,
    convert_flag: bool = True,
    converter_dict: dict | None = None,
    converter_kwargs: dict | None = None,
    decode_flag: bool = True,
    decoder_dict: dict | None = None,
    validate_flag: bool = True,
    sections: str | list | None = None,
    excludes: str | list | None = None,
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
    year_init: str or int, optional
        Left border of time axis.
    year_end: str or int, optional
        Right border of time axis.
    encoding : str, optional
        The encoding of the input file. Overrides the value in the imodel schema file.
    chunksize : int, optional
          Number of reports per chunk.
    skiprows : int, optional
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
    skiprows = skiprows or 0

    validate_read_mdf_args(
        source=source,
        imodel=imodel,
        ext_schema_path=ext_schema_path,
        ext_schema_file=ext_schema_file,
        year_init=year_init,
        year_end=year_end,
        chunksize=chunksize,
        skiprows=skiprows,
    )

    pd_kwargs = pd_kwargs or {}
    pd_kwargs.setdefault("encoding", encoding)
    pd_kwargs.setdefault("chunksize", chunksize)
    pd_kwargs.setdefault("skiprows", skiprows)

    xr_kwargs = xr_kwargs or {}

    convert_kwargs = dict(
        convert_flag=convert_flag,
        converter_dict=converter_dict,
        converter_kwargs=converter_kwargs,
    )

    decode_kwargs = dict(
        decode_flag=decode_flag,
        decoder_dict=decoder_dict,
    )

    validate_kwargs = dict(
        validate_flag=validate_flag,
        ext_table_path=ext_table_path,
    )

    sections = as_list(sections)
    excludes = as_list(excludes)

    validate_arg("sections", sections, list)
    validate_arg("excludes", excludes, list)

    select_kwargs = dict(
        sections=sections,
        excludes=excludes,
        year_init=year_init,
        year_end=year_end,
    )

    filereader = FileReader(
        imodel=imodel,
        ext_schema_path=ext_schema_path,
        ext_schema_file=ext_schema_file,
    )

    return filereader.read(
        source=source,
        pd_kwargs=pd_kwargs,
        xr_kwargs=xr_kwargs,
        convert_kwargs=convert_kwargs,
        decode_kwargs=decode_kwargs,
        validate_kwargs=validate_kwargs,
        select_kwargs=select_kwargs,
    )


def _read_data(
    data_file: str,
    mask_file: str,
    reader: Callable[..., Any],
    col_subset: str | list | tuple | None,
    data_kwargs: dict,
    mask_kwargs: dict,
):
    """Helper function for reading data files from disk."""
    data, info = reader(
        data_file,
        col_subset=col_subset,
        **data_kwargs,
    )

    if mask_file is None:
        mask = pd.DataFrame()
    else:
        mask, _ = reader(
            mask_file,
            col_subset=col_subset,
            column_names=info["columns"],
            **mask_kwargs,
        )

    return data, mask, info


def read_data(
    data_file: str,
    mask_file: str | None = None,
    info_file: str | None = None,
    data_format: SupportedFileTypes = "csv",
    imodel: str | None = None,
    col_subset: str | list | tuple | None = None,
    encoding: str | None = None,
    **kwargs,
) -> DataBundle:
    """Read MDF data which is already on a pre-defined data model.

    Parameters
    ----------
    data_file: str
        The data file (including path) to be read.
    mask_file: str, optional
        The validation file (including path) to be read.
    info_file: str, optional
        The information file (including path) to be read.
    data_format: {"csv", "parquet", "feather"}, default: "csv"
        Format of input data file(s).
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
    supported_file_types = get_args(SupportedFileTypes)
    if data_format not in supported_file_types:
        raise ValueError(
            f"data_format must be one of {supported_file_types}, not {data_format}."
        )

    data_kwargs = kwargs.copy()
    mask_kwargs = kwargs.copy()
    parse_dates = False
    if data_format == "csv":
        info_dict = open_json_file(info_file) if info_file else {}
        dtype = info_dict.get("dtypes", "object")
        parse_dates = info_dict.get("parse_dates", False)
        encoding = encoding or info_dict.get("encoding", None)

        data_kwargs.setdefault("dtype", dtype)
        data_kwargs.setdefault("parse_dates", parse_dates)
        data_kwargs.setdefault("encoding", encoding)

        mask_kwargs.setdefault("dtype", "boolean")

    data, mask, info = _read_data(
        data_file=data_file,
        mask_file=mask_file,
        reader=READERS[data_format],
        col_subset=col_subset,
        data_kwargs=data_kwargs,
        mask_kwargs=mask_kwargs,
    )

    return DataBundle(
        data=data,
        columns=info["columns"],
        dtypes=info["dtypes"].to_dict(),
        parse_dates=parse_dates,
        mask=mask,
        imodel=imodel,
        encoding=encoding,
    )
