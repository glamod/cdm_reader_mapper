"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import logging

import pandas as pd
import xarray as xr

from dataclasses import replace
from pandas.io.parsers import TextFileReader

from .. import properties
from .utilities import (
    process_textfilereader,
    remove_boolean_values,
)

from .convert_and_decode import convert_and_decode
from .validators import validate
from .parser import Parser

from cdm_reader_mapper.core.databundle import DataBundle


def _apply_or_chunk(data, func, func_args=None, func_kwargs=None, **kwargs):
    func_args = func_args or []
    func_kwargs = func_kwargs or {}
    if not isinstance(data, TextFileReader):
        return func(data, *func_args, **func_kwargs)
    return process_textfilereader(
        data,
        func,
        func_args,
        func_kwargs,
        **kwargs,
    )


def _merge_kwargs(*dicts):
    merged = {}
    for d in dicts:
        for k in d:
            if k in merged:
                raise ValueError(f"Duplicate kwarg '{k}' in open_data()")
            merged[k] = d[k]
    return merged


def _apply_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.map(lambda x: isinstance(x, tuple)).all():
        return df

    df.columns = pd.MultiIndex.from_tuples(
        [col if isinstance(col, tuple) else (None, col) for col in df.columns],
    )
    return df


def _select_years(df, selection, year_col) -> pd.DataFrame:
    year_init, year_end = selection
    if year_init is None and year_end is None:
        return df

    years = pd.to_numeric(df[year_col], errors="coerce")

    mask = pd.Series(True, index=df.index)

    if year_init is not None:
        mask &= years >= year_init

    if year_end is not None:
        mask &= years <= year_end

    mask &= years.notna()

    return df.loc[mask].reset_index(drop=True)


class FileReader:
    """Class to read marine-meteorological data."""

    def __init__(self, *args, **kwargs):
        self.parser = Parser(*args, **kwargs)
        self.config = self.parser.config

    def _process_data(
        self,
        data,
        convert_flag,
        decode_flag,
        converter_dict,
        converter_kwargs,
        decoder_dict,
        validate_flag,
        ext_table_path,
        sections,
        excludes,
        year_init,
        year_end,
        config,
        parse_mode="pandas",
    ) -> pd.DataFrame | TextFileReader:
        if parse_mode == "pandas":
            data = self.parser.parse_pandas(data, sections, excludes)
        elif parse_mode == "netcdf":
            data = self.parser.parse_netcdf(data, sections, excludes)
        else:
            raise ValueError("open_with has to be one of ['pandas', 'netcdf']")

        data = _apply_multiindex(data)
        imodel = self.config.imodel

        data_model = imodel.split("_")[0]
        year_col = properties.year_column[data_model]

        data = _select_years(data, [year_init, year_end], year_col)

        if converter_dict is None:
            converter_dict = config.convert_decode["converter_dict"]
        if converter_kwargs is None:
            converter_kwargs = config.convert_decode["converter_kwargs"]
        if decoder_dict is None:
            decoder_dict = config.convert_decode["decoder_dict"]

        data = convert_and_decode(
            data,
            convert_flag=convert_flag,
            decode_flag=decode_flag,
            converter_dict=converter_dict,
            converter_kwargs=converter_kwargs,
            decoder_dict=decoder_dict,
        )

        if validate_flag:
            mask = validate(
                data,
                imodel=imodel,
                ext_table_path=ext_table_path,
                attributes=config.validation,
                disables=config.disable_reads,
            )
        else:
            mask = pd.DataFrame(True, index=data.index, columns=data.columns)

        data = remove_boolean_values(data, config.dtypes)
        config = replace(config, columns=data.columns)
        return data, mask, config

    def open_data(
        self,
        source,
        open_with="pandas",
        pd_kwargs=None,
        xr_kwargs=None,
        convert_kwargs=None,
        decode_kwargs=None,
        validate_kwargs=None,
        select_kwargs=None,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[TextFileReader, TextFileReader]:
        pd_kwargs = dict(pd_kwargs or {})
        xr_kwargs = dict(xr_kwargs or {})
        convert_kwargs = convert_kwargs or {}
        decode_kwargs = decode_kwargs or {}
        validate_kwargs = validate_kwargs or {}
        select_kwargs = select_kwargs or {}

        func_kwargs = _merge_kwargs(
            convert_kwargs,
            decode_kwargs,
            validate_kwargs,
            select_kwargs,
        )
        func_kwargs["parse_mode"] = open_with

        if open_with == "netcdf":
            to_parse = xr.open_mfdataset(source, **xr_kwargs).squeeze()
            config = self.parser.update_xr_config(to_parse)
            write_kwargs, read_kwargs = {}, {}
        elif open_with == "pandas":
            config = self.parser.update_pd_config(pd_kwargs)
            pd_kwargs["encoding"] = config.encoding

            pd_kwargs.setdefault("widths", [properties.MAX_FULL_REPORT_WIDTH])
            pd_kwargs.setdefault("header", None)
            pd_kwargs.setdefault("quotechar", "\0")
            pd_kwargs.setdefault("escapechar", "\0")
            pd_kwargs.setdefault("dtype", object)
            pd_kwargs.setdefault("skip_blank_lines", False)

            write_kwargs = {"encoding": pd_kwargs["encoding"]}
            chunksize = pd_kwargs.get("chunksize")
            read_kwargs = (
                {"chunksize": chunksize, "dtype": config.dtypes},
                {"chunksize": chunksize, "dtype": "boolean"},
            )
            to_parse = pd.read_fwf(source, **pd_kwargs)
        else:
            raise ValueError("open_with has to be one of ['pandas', 'netcdf']")

        func_kwargs["config"] = config

        return _apply_or_chunk(
            to_parse,
            self._process_data,
            func_kwargs=func_kwargs,
            makecopy=False,
            write_kwargs=write_kwargs,
            read_kwargs=read_kwargs,
        )

    def read(
        self,
        source: str,
        pd_kwargs: dict | None = None,
        xr_kwargs: dict | None = None,
        convert_kwargs: dict | None = None,
        decode_kwargs: dict | None = None,
        validate_kwargs: dict | None = None,
        select_kwargs: dict | None = None,
    ) -> DataBundle:
        pd_kwargs = pd_kwargs or {}
        xr_kwargs = xr_kwargs or {}
        convert_kwargs = convert_kwargs or {}
        decode_kwargs = decode_kwargs or {}
        validate_kwargs = validate_kwargs or {}
        select_kwargs = select_kwargs or {}

        imodel = self.config.imodel

        logging.info(f"EXTRACTING DATA FROM MODEL: {imodel}")

        logging.info("Reading and parsing source data...")
        result = self.open_data(
            source,
            # INFO: Set default as "pandas" to account for custom schema
            open_with=properties.open_file.get(imodel, "pandas"),
            pd_kwargs=pd_kwargs,
            xr_kwargs=xr_kwargs,
            convert_kwargs=convert_kwargs,
            decode_kwargs=decode_kwargs,
            validate_kwargs=validate_kwargs,
            select_kwargs=select_kwargs,
        )

        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError("open_data() must return (data, mask, config)")

        data, mask, config = result

        return DataBundle(
            data=data,
            columns=config.columns,
            dtypes=config.dtypes,
            parse_dates=config.parse_dates,
            encoding=config.encoding,
            mask=mask,
            imodel=config.imodel,
        )
