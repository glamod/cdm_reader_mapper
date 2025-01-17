"""Common Data Model (CDM) MDF writer."""

from __future__ import annotations

import json
import logging
import os
from io import StringIO as StringIO

import pandas as pd

from cdm_reader_mapper.common import get_filename


def write_data(
    data,
    mask=None,
    dtypes={},
    parse_dates=False,
    out_dir=".",
    prefix=None,
    suffix=None,
    extension="csv",
    filename=None,
    col_subset=None,
    delimiter=",",
):
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
    write_tables : Write CDM tables to disk.
    read_data : Read MDF data and validation mask from disk.
    read_mdf : Read original marine-meteorological data from disk.
    read_tables : Read CDM tables from disk.

    Note
    ----
    Use this function after reading MDF data.
    """

    def _join(col):
        if isinstance(col, (list, tuple)):
            return ":".join(col)
        return col

    if mask is None:
        mask = pd.DataFrame()

    info = {}
    info["dtypes"] = dtypes
    info["parse_dates"] = [_join(parse_date) for parse_date in parse_dates]

    logging.info(f"WRITING DATA TO FILES IN: {out_dir}")
    filename_data = get_filename(
        [prefix, "data", suffix], path=out_dir, extension=extension
    )
    filename_mask = get_filename(
        [prefix, "mask", suffix], path=out_dir, extension=extension
    )
    filename_info = get_filename(
        [prefix, "info", suffix], path=out_dir, extension="json"
    )

    if col_subset is not None:
        data = data[col_subset]
        mask = mask[col_subset]

    header = []
    info["dtypes"] = {k: v for k, v in info["dtypes"].items() if k in data.columns}
    for col in data.columns:
        col_ = _join(col)
        header.append(col_)

    if col in info["dtypes"]:
        info["dtypes"][col_] = info["dtypes"][col]
        del info["dtypes"][col]
    info["parse_dates"] = [
        parse_date for parse_date in info["parse_dates"] if parse_date in header
    ]

    kwargs = {
        "header": header,
        "mode": "w",
        "encoding": "utf-8",
        "index": False,
        "sep": delimiter,
    }
    data.to_csv(os.path.join(out_dir, filename_data), **kwargs)
    mask.to_csv(os.path.join(out_dir, filename_mask), **kwargs)

    if info:
        with open(os.path.join(out_dir, filename_info), "w") as fileObj:
            json.dump(info, fileObj, indent=4)
