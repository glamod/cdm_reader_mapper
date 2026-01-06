"""Common Data Model (CDM) MDF writer."""

from __future__ import annotations

import json
import logging
from io import StringIO as StringIO
from pathlib import Path

import pandas as pd

from .utils.utilities import join, update_column_names, update_dtypes

from ..common import get_filename
from ..common.pandas_TextParser_hdlr import make_copy


def write_data(
    data,
    mask=None,
    dtypes: dict | None = None,
    parse_dates: list | bool = False,
    encoding="utf-8",
    out_dir=".",
    prefix=None,
    suffix=None,
    extension="csv",
    filename=None,
    col_subset=None,
    delimiter=",",
    **kwargs,
) -> None:
    """Write pandas.DataFrame to MDF file on file system.

    Parameters
    ----------
    data: pandas.DataFrame
        pandas.DataFrame to export.
    mask: pandas.DataFrame
        validation mask to export.
    dtypes: dict
        Dictionary of data types on ``data``.
        Dump ``dtypes`` and ``parse_dates`` to json information file.
    parse_dates:
        Information of how to parse dates in :py:attr:`data`.
        Dump ``dtypes`` and ``parse_dates`` to json information file.
        For more information see :py:func:`pandas.read_csv`.
    encoding: str
        A string representing the encoding to use in the output file, defaults to utf-8.
    out_dir: str
        Path to the output directory.
        Default: current directory
    prefix: str, optional
        Prefix of file name structure: ``<prefix>-data-*<suffix>.<extension>``.
    suffix: str, optional
        Suffix of file name structure: ``<prefix>-data-*<suffix>.<extension>``.
    extension: str
        Extension of file name structure: ``<prefix>-data-*<suffix>.<extension>``.
        Default: psv
    filename: str or dict, optional
        Name of the output file name(s).
        List one filename for both ``data`` and ``mask`` ({"data":<filenameD>, "mask":<filenameM>}).
        Default: Automatically create file name from table name, ``prefix`` and ``suffix``.
    col_subset: str, tuple or list, optional
        Specify the section or sections of the file to write.

        - For multiple sections of the tables:
          e.g col_subset = [columns0,...,columnsN]

        - For a single section:
          e.g. list type object col_subset = [columns]

        Column labels could be both string or tuple.
    delimiter: str
        Character or regex pattern to treat as the delimiter while reading with df.to_csv.
        Default: ","

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
    dtypes = dtypes or {}
    if isinstance(parse_dates, bool):
        parse_dates = []

    if not isinstance(data, pd.io.parsers.TextFileReader):
        data_list = [data]
    else:
        data_list = make_copy(data)

    if mask is None:
        mask = pd.DataFrame()

    if not isinstance(mask, pd.io.parsers.TextFileReader):
        mask_list = [mask]
    else:
        mask_list = make_copy(mask)

    info = {"dtypes": dtypes.copy(), "parse_dates": [join(p) for p in parse_dates]}

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

        mode = "w" if i == 0 else "a"
        header = [join(c) for c in data_df.columns] if i == 0 else False

        if i == 0:
            info["dtypes"] = update_dtypes(info["dtypes"], data_df.columns)
            for col in data_df.columns:
                info["dtypes"] = update_column_names(info["dtypes"], col, join(col))
            info["parse_dates"] = [p for p in info["parse_dates"] if p in header]
            info["encoding"] = encoding

        csv_kwargs = dict(
            header=header,
            mode=mode,
            index=False,
            sep=delimiter,
            encoding=encoding,
            **kwargs,
        )

        data_df.to_csv(filename_data, **csv_kwargs)
        if not mask_df.empty:
            mask_df.to_csv(filename_mask, **csv_kwargs)

    with open(filename_info, "w") as fileObj:
        json.dump(info, fileObj, indent=4)
