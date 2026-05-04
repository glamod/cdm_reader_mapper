"""Common Data Model (CDM) MDF writer."""

from __future__ import annotations
import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, get_args

import pandas as pd

from ..common import get_filename
from ..common.iterators import (
    ParquetStreamReader,
    is_valid_iterator,
    parquet_stream_from_iterable,
)
from ..properties import SupportedFileTypes
from .utils.utilities import join, update_column_names, update_dtypes


WRITERS = {
    "csv": "to_csv",
    "parquet": "to_parquet",
    "feather": "to_feather",
}


def _validate_write_inputs(
    data: pd.DataFrame | Iterable[pd.DataFrame],
    mask: pd.DataFrame | Iterable[pd.DataFrame],
    data_format: str,
    supported: Sequence[str],
) -> None:
    """
    Validate writing inputs.

    Parameters
    ----------
    data : pd.DataFrame or Iterable of pd.DataFrame
        Data to export.
    mask : pd.DataFrame or Iterable of pd.DataFrame
        Validation mask to export.
    data_format : str
        Format of output data file(s).
    supported : Sequence of str
        Names of supported data models.

    Raises
    ------
    ValueError
        If `data_format` is not in `supported`.
        If type of `data` does not match type of `mask`.
    """
    if data_format not in supported:
        raise ValueError(f"data_format must be one of {supported}, not {data_format}.")

    if mask is not None and not isinstance(mask, type(data)):
        raise ValueError("Type of 'data' and type of 'mask' do not match.")


def _build_info(dtypes: dict[Any, Any], parse_dates: list[Any]) -> dict[str, Any]:
    """
    Build information dictionary.

    Parameters
    ----------
    dtypes : dict
        Dictionary of data types on `data`.
    parse_dates : list of Any
        Information of how to parse dates.

    Returns
    -------
    dict
        Dictionary including information about both data types and parse dates.
    """
    return {
        "dtypes": {k: str(v) for k, v in dtypes.items()},
        "parse_dates": [join(p) for p in parse_dates],
    }


def _get_write_kwargs(data_format: str, header: Any, mode: str, encoding: str, delimiter: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Build keyword arguments for writing data in different formats.

    Parameters
    ----------
    data_format : str
        Output format, e.g. 'csv' or 'parquet'.
    header : Any
        Header configuration used for CSV output.
    mode : str
        File write mode (e.g. 'w', 'a').
    encoding : str
        Encoding used for text-based formats.
    delimiter : str
        Column separator used for CSV output.
    kwargs : dict[str, Any]
        Additional format-specific keyword arguments.

    Returns
    -------
    dict[str, Any]
        Keyword arguments passed to the underlying writer.

    Notes
    -----
    - For 'csv', returns full pandas-compatible kwargs.
    - For 'parquet', returns fixed engine and compression settings.
    """
    if data_format == "csv":
        return dict(
            header=header,
            mode=mode,
            index=False,
            sep=delimiter,
            encoding=encoding,
            **kwargs,
        )
    if data_format == "parquet":
        return dict(engine="pyarrow", compression="snappy")
    return {}


def _normalize_data_chunks(
    data: pd.DataFrame | Iterable[pd.DataFrame] | None,
) -> list[pd.DataFrame] | ParquetStreamReader:
    """
    Helper function to normalize data chunks.

    Parameters
    ----------
    data : pd.DataFrame of Iterable of pd.DataFrame or None
        Data to be normalized.

    Returns
    -------
    list of pd.DataFrame or ParquetStreamReader
        Normalized data.

    Raises
    ------
    TypeError
        If `data` has an unsupported data type.
    """
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
    data_format: SupportedFileTypes = "parquet",
    dtypes: pd.Series | dict[Any, Any] | None = None,
    parse_dates: list[Any] | bool = False,
    encoding: str = "utf-8",
    out_dir: str = ".",
    prefix: str | None = None,
    suffix: str | None = None,
    extension: str | None = None,
    filename: str | dict[str, str] | None = None,
    separator: str | None = "_",
    col_subset: str | list[str] | tuple[str] | None = None,
    delimiter: str = ",",
    **kwargs: Any,
) -> None:
    r"""
    Write pandas.DataFrame to MDF file on file system.

    Parameters
    ----------
    data : pandas.DataFrame or Iterable[pd.DataFrame]
        Data to export.
    mask : pandas.DataFrame or Iterable[pd.DataFrame], optional
        Validation mask to export.
    data_format : {"csv", "parquet", "feather"}, default: "parquet"
        Format of output data file(s).
    dtypes : dict, optional
        Dictionary of data types on `data`.
        Dump `dtypes` and `parse_dates` to json information file.
    parse_dates : list | bool, default: False
        Information of how to parse dates in :py:attr:`data`.
        Dump `dtypes` and `parse_dates` to json information file.
        For more information see :py:func:`pandas.read_csv`.
    encoding : str, default: "utf-8"
        A string representing the encoding to use in the output file, defaults to utf-8.
    out_dir : str, default: "."
        Path to the output directory.
    prefix : str, optional
        Prefix of file name structure: `<prefix>-data-*<suffix>.<extension>`.
    suffix : str, optional
        Suffix of file name structure: `<prefix>-data-*<suffix>.<extension>`.
    extension : str, optional
        Extension of file name structure: `<prefix>-data-*<suffix>.<extension>`.
        By default, extension depends on `data_format`.
    filename : str or dict, optional
        Name of the output file name(s).
        List one filename for both `data` and `mask` ({"data":<filenameD>, "mask":<filenameM>}).
        By default, automatically create file name from table name, `prefix` and `suffix`.
    separator : str, optional
        Separator to join the file name pattern components (default "_").
    col_subset : str, tuple or list, optional
        Specify the section or sections of the file to write.

        - For multiple sections of the tables:
          e.g col_subset = [columns0,...,columnsN]

        - For a single section:
          e.g. list type object col_subset = [columns]

        Column labels could be both string or tuple.
    delimiter : str, default: ","
        Character or regex pattern to treat as the delimiter while reading with df.to_csv.
    \**kwargs : Any
        Additional keyword-arguments passed to `to_csv` when `data_format` is 'csv'.

    Raises
    ------
    ValueError
        If `data_foramt` is not one of 'csv', 'parquet' or 'feather'.
        If type of `data` and type of `mask` do not match.

    See Also
    --------
    write : Write either MDF data or CDM tables to disk.
    write_tables : Write CDM tables to disk.
    read : Read either original marine-meteorological data or MDF data or CDM tables from disk.
    read_data : Read MDF data and validation mask from disk.
    read_mdf : Read original marine-meteorological data from disk.
    read_tables : Read CDM tables from disk.

    Notes
    -----
    Use this function after reading MDF data.
    """
    supported_file_types = get_args(SupportedFileTypes)
    _validate_write_inputs(data, mask, data_format, supported_file_types)

    extension = extension or data_format

    dtypes = dtypes if isinstance(dtypes, (dict, pd.Series)) else {}
    parse_dates = [] if isinstance(parse_dates, bool) else parse_dates

    data_list = _normalize_data_chunks(data)
    mask_list = _normalize_data_chunks(mask)

    info = _build_info(dtypes, parse_dates)

    logging.info("WRITING DATA TO FILES IN: %s", out_dir)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    filename_data = get_filename([prefix, "data", suffix], path=out_dir_path, extension=extension, separator=separator)
    filename_mask = get_filename([prefix, "mask", suffix], path=out_dir_path, extension=extension, separator=separator)
    filename_info = get_filename([prefix, "info", suffix], path=out_dir_path, extension="json", separator=separator)

    writer = WRITERS[data_format]

    for i, (data_df, mask_df) in enumerate(zip(data_list, mask_list, strict=True)):
        if col_subset is not None:
            data_df = data_df[col_subset]
            mask_df = mask_df[col_subset]

        data_df = data_df.to_frame() if isinstance(data_df, pd.Series) else data_df
        mask_df = mask_df.to_frame() if isinstance(mask_df, pd.Series) else mask_df

        mode = "w" if i == 0 else "a"
        header = [join(c) for c in data_df.columns] if i == 0 else False

        if i == 0:
            info["dtypes"] = update_dtypes(info["dtypes"], data_df.columns)
            if isinstance(info["dtypes"], dict):
                for col in data_df.columns:
                    info["dtypes"] = update_column_names(info["dtypes"], col, join(col))

            info["parse_dates"] = [p for p in info["parse_dates"] if isinstance(header, list) and p in header]
            info["encoding"] = encoding

        write_kwargs = _get_write_kwargs(data_format, header, mode, encoding, delimiter, kwargs)

        getattr(data_df, writer)(filename_data, **write_kwargs)
        if not mask_df.empty:
            getattr(mask_df, writer)(filename_mask, **write_kwargs)

    if data_format == "csv":
        with Path(filename_info).open("w") as f:
            json.dump(info, f, indent=4)
