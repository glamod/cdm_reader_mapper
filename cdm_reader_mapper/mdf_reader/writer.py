"""Common Data Model (CDM) MDF writer."""

from __future__ import annotations

import json
import logging
from io import StringIO as StringIO
from pathlib import Path
from typing import Iterable, get_args

import pandas as pd

from .utils.utilities import join, update_column_names, update_dtypes

from ..common import get_filename
from ..common.iterators import (
    ParquetStreamReader,
    is_valid_iterator,
    parquet_stream_from_iterable,
)

from ..properties import SupportedFileTypes

WRITERS = {
    "csv": "to_csv",
    "parquet": "to_parquet",
    "feather": "to_feather",
}


def _normalize_data_chunks(
    data: pd.DataFrame | Iterable[pd.DataFrame] | None,
) -> list | ParquetStreamReader:
    """Helper function to normalize data chunks."""
    if data is None:
        data = pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return [data]
    if is_valid_iterator(data):
        if not isinstance(data, ParquetStreamReader):
            data = parquet_stream_from_iterable(data)
        return data.copy()
    if isinstance(data, (list, tuple)):
        return parquet_stream_from_iterable(data)
    raise TypeError(f"Unsupported data type found: {type(data)}.")


def write_data(
    data: pd.DataFrame | Iterable[pd.DataFrame],
    mask: pd.DataFrame | Iterable[pd.DataFrame] | None = None,
    data_format: SupportedFileTypes = "csv",
    dtypes: pd.Series | dict | None = None,
    parse_dates: list | bool = False,
    encoding: str = "utf-8",
    out_dir: str = ".",
    prefix: str | None = None,
    suffix: str | None = None,
    extension: str = None,
    filename: str | dict | None = None,
    col_subset: str | list[str] | tuple[str] | None = None,
    delimiter: str = ",",
    **kwargs,
) -> None:
    """Write pandas.DataFrame to MDF file on file system.

    Parameters
    ----------
    data: pandas.DataFrame or Iterable[pd.DataFrame]
        Data to export.
    mask: pandas.DataFrame or Iterable[pd.DataFrame], optional
        Validation mask to export.
    data_format: {"csv", "parquet", "feather"}, default: "csv"
        Format of output data file(s).
    dtypes: dict, optional
        Dictionary of data types on ``data``.
        Dump ``dtypes`` and ``parse_dates`` to json information file.
    parse_dates: list | bool, default: False
        Information of how to parse dates in :py:attr:`data`.
        Dump ``dtypes`` and ``parse_dates`` to json information file.
        For more information see :py:func:`pandas.read_csv`.
    encoding: str, default: "utf-8"
        A string representing the encoding to use in the output file, defaults to utf-8.
    out_dir: str, default: "."
        Path to the output directory.
    prefix: str, optional
        Prefix of file name structure: ``<prefix>-data-*<suffix>.<extension>``.
    suffix: str, optional
        Suffix of file name structure: ``<prefix>-data-*<suffix>.<extension>``.
    extension: str, optional
        Extension of file name structure: ``<prefix>-data-*<suffix>.<extension>``.
        By default, extension depends on `data_format`.
    filename: str or dict, optional
        Name of the output file name(s).
        List one filename for both ``data`` and ``mask`` ({"data":<filenameD>, "mask":<filenameM>}).
        By default, automatically create file name from table name, ``prefix`` and ``suffix``.
    col_subset: str, tuple or list, optional
        Specify the section or sections of the file to write.

        - For multiple sections of the tables:
          e.g col_subset = [columns0,...,columnsN]

        - For a single section:
          e.g. list type object col_subset = [columns]

        Column labels could be both string or tuple.
    delimiter: str, default: ","
        Character or regex pattern to treat as the delimiter while reading with df.to_csv.

    See Also
    --------
    write: Write either MDF data or CDM tables to disk.
    write_tables : Write CDM tables to disk.
    read: Read either original marine-meteorological data or MDF data or CDM tables from disk.
    read_data : Read MDF data and validation mask from disk.
    read_mdf : Read original marine-meteorological data from disk.
    read_tables : Read CDM tables from disk.

    Note
    ----
    Use this function after reading MDF data.
    """
    supported_file_types = get_args(SupportedFileTypes)
    if data_format not in supported_file_types:
        raise ValueError(
            f"data_format must be one of {supported_file_types}, not {data_format}."
        )

    if mask is not None and not isinstance(data, type(mask)):
        raise ValueError("type of 'data' and type of 'mask' do not match.")

    extension = extension or data_format

    if not isinstance(dtypes, (dict, pd.Series)):
        dtypes = {}

    if isinstance(parse_dates, bool):
        parse_dates = []

    data_list = _normalize_data_chunks(data)
    mask_list = _normalize_data_chunks(mask)

    info = {
        "dtypes": {k: str(v) for k, v in dtypes.items()},
        "parse_dates": [join(p) for p in parse_dates],
    }

    logging.info(f"WRITING DATA TO FILES IN: {out_dir}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename_data = get_filename(
        [prefix, "data", suffix], path=out_dir, extension=extension
    )
    filename_mask = get_filename(
        [prefix, "mask", suffix], path=out_dir, extension=extension
    )
    filename_info = get_filename(
        [prefix, "info", suffix], path=out_dir, extension="json"
    )

    for i, (data_df, mask_df) in enumerate(zip(data_list, mask_list)):
        if col_subset is not None:
            data_df = data_df[col_subset]
            mask_df = mask_df[col_subset]

        if isinstance(data_df, pd.Series):
            data_df = data_df.to_frame()
        if isinstance(mask_df, pd.Series):
            mask_df = mask_df.to_frame()

        mode = "w" if i == 0 else "a"
        header = [join(c) for c in data_df.columns] if i == 0 else False

        if i == 0:
            info["dtypes"] = update_dtypes(info["dtypes"], data_df.columns)
            for col in data_df.columns:
                info["dtypes"] = update_column_names(info["dtypes"], col, join(col))

            info["parse_dates"] = [p for p in info["parse_dates"] if p in header]
            info["encoding"] = encoding

        write_kwargs = {}
        if data_format == "csv":
            write_kwargs = dict(
                header=header,
                mode=mode,
                index=False,
                sep=delimiter,
                encoding=encoding,
                **kwargs,
            )

        writer = WRITERS[data_format]
        getattr(data_df, writer)(filename_data, **write_kwargs)
        if not mask_df.empty:
            getattr(mask_df, writer)(filename_mask, **write_kwargs)

    with open(filename_info, "w") as fileObj:
        json.dump(info, fileObj, indent=4)
