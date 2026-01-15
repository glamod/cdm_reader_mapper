"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import logging

from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import xarray as xr

from dataclasses import replace
from pandas.io.parsers import TextFileReader

from .. import properties
from .utilities import (
    process_disk_backed,
    remove_boolean_values,
    ParquetStreamReader,
)

from .convert_and_decode import convert_and_decode
from .validators import validate
from .parser import (
    update_xr_config,
    update_pd_config,
    parse_pandas,
    parse_netcdf,
    build_parser_config,
    ParserConfig,
)

from cdm_reader_mapper.core.databundle import DataBundle


def _apply_or_chunk(
    data: pd.DataFrame | TextFileReader,
    func: Callable[..., Any],
    func_args: Sequence[Any] | None = None,
    func_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Mapping[str, Any],
):
    """Apply a function directly or chunk-wise.  If data is an iterator, it uses disk-backed streaming."""
    func_args = func_args or []
    func_kwargs = func_kwargs or {}
    if not isinstance(data, (TextFileReader, ParquetStreamReader)):
        result = func(data, *func_args, **func_kwargs)
    else:
        result = process_disk_backed(
            data,
            func,
            func_args,
            func_kwargs,
            **kwargs,
        )

    return result


def _merge_kwargs(*dicts: Mapping[str, Any]) -> dict[str, Any]:
    """Merge multiple keyword-argument dictionaries."""
    merged = {}
    for d in dicts:
        for k in d:
            if k in merged:
                raise ValueError(f"Duplicate kwarg '{k}' in open_data()")
            merged[k] = d[k]
    return merged


def _apply_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Convert tuple-based columns to a pandas MultiIndex."""
    if not df.columns.map(lambda x: isinstance(x, tuple)).all():
        return df

    df.columns = pd.MultiIndex.from_tuples(
        [col if isinstance(col, tuple) else (None, col) for col in df.columns],
    )
    return df


def _select_years(
    df: pd.DataFrame,
    selection: tuple[int | None, int | None],
    year_col,
) -> pd.DataFrame:
    """Filter rows of a DataFrame by a year range."""
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
    """
    Class to read marine-meteorological data.

    Provides a high-level interface to read, parse, filter, convert,
    decode, and validate data from multiple sources (FWF, CSV, NetCDF).
    """

    def __init__(
        self,
        imodel: str | None = None,
        ext_schema_path: str | None = None,
        ext_schema_file: str | None = None,
    ):
        """
        Initialize FileReader with a data model and parser configuration.

        Parameters
        ----------
        imodel : str
            Name of the data model (e.g., 'ICOADS').
        args, kwargs
            Arguments passed to ``build_parser_config``.
        """
        self.imodel: str = imodel
        self.config: ParserConfig = build_parser_config(
            imodel=imodel,
            ext_schema_path=ext_schema_path,
            ext_schema_file=ext_schema_file,
        )

    def _process_data(
        self,
        data: pd.DataFrame | TextFileReader,
        convert_flag: bool = False,
        decode_flag: bool = False,
        converter_dict: dict | None = None,
        converter_kwargs: dict | None = None,
        decoder_dict: dict | None = None,
        validate_flag: bool = False,
        ext_table_path: str | None = None,
        sections: Sequence[str] | None = None,
        excludes: Sequence[str] | None = None,
        year_init: int | None = None,
        year_end: int | None = None,
        config: ParserConfig | None = None,
        parse_mode: str = "pandas",
    ) -> tuple[pd.DataFrame, pd.DataFrame, ParserConfig]:
        """
        Core processing of raw data: parse, filter, convert, decode, validate.

        Parameters
        ----------
        data : pandas.DataFrame or TextFileReader
            Input data.
        convert_flag : bool
            Whether to apply converters.
        decode_flag : bool
            Whether to apply decoders.
        converter_dict : dict, optional
            Mapping of columns to converter functions.
        converter_kwargs : dict, optional
            Keyword arguments for converters.
        decoder_dict : dict, optional
            Mapping of columns to decoder functions.
        validate_flag : bool
            Whether to apply validation.
        ext_table_path : str, optional
            Path to external validation tables.
        sections : sequence of str, optional
            Sections to include.
        excludes : sequence of str, optional
            Sections to exclude.
        year_init : int, optional
            Initial year for filtering.
        year_end : int, optional
            End year for filtering.
        config : ParserConfig, optional
            Parser configuration.
        parse_mode : str
            Parsing backend ('pandas' or 'netcdf').

        Returns
        -------
        tuple of (data, mask, config)
            - data : pandas.DataFrame with parsed, filtered, converted data
            - mask : pandas.DataFrame with boolean mask for validation
            - config : ParserConfig updated with final columns
        """
        config = config or self.config

        if parse_mode == "pandas":
            data = parse_pandas(data, config.order_specs, sections, excludes)
        elif parse_mode == "netcdf":
            data = parse_netcdf(data, config.order_specs, sections, excludes)
        else:
            raise ValueError("parse_mode must be 'pandas' or 'netcdf'")

        data = _apply_multiindex(data)

        data_model = self.imodel.split("_")[0]
        year_col = properties.year_column[data_model]

        data = _select_years(data, (year_init, year_end), year_col)

        converter_dict = converter_dict or config.convert_decode["converter_dict"]
        converter_kwargs = converter_kwargs or config.convert_decode["converter_kwargs"]
        decoder_dict = decoder_dict or config.convert_decode["decoder_dict"]

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
                imodel=self.imodel,
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
        source: str,
        open_with: str = "pandas",
        pd_kwargs: dict | None = None,
        xr_kwargs: dict | None = None,
        convert_kwargs: dict | None = None,
        decode_kwargs: dict | None = None,
        validate_kwargs: dict | None = None,
        select_kwargs: dict | None = None,
    ) -> (
        tuple[pd.DataFrame, pd.DataFrame, ParserConfig]
        | tuple[TextFileReader, TextFileReader, ParserConfig]
    ):
        """
        Open and parse source data according to parser configuration.

        Parameters
        ----------
        source : str
            Path or pattern for input file(s).
        open_with : str
            Parser backend: 'pandas' or 'netcdf'.
        pd_kwargs: dict, optional
            Additional key-word arguments for parsing pandas-readable data.
        xr_kwargs: dict, optional
            Additional key-word arguments for parsing xarray-readable data.
        convert_kwargs: dict, optional
            Additional key-word arguments for data conversion.
        decode_kwargs: dict, optional
            Additional key-word arguments for data decoding.
        validate_kwargs: dict, optional
            Additional key-word arguments for data validation.
        select_kwargs : dict, optional
            Additional key-word arguments for selecting/filtering data.

        Returns
        -------
        tuple
            (data, mask, config) or chunked equivalents if using TextFileReader.
        """
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
            config = update_xr_config(to_parse, self.config)
        elif open_with == "pandas":
            config = update_pd_config(pd_kwargs, self.config)
            pd_kwargs["encoding"] = config.encoding
            pd_kwargs.setdefault("widths", [properties.MAX_FULL_REPORT_WIDTH])
            pd_kwargs.setdefault("header", None)
            pd_kwargs.setdefault("quotechar", "\0")
            pd_kwargs.setdefault("escapechar", "\0")
            pd_kwargs.setdefault("dtype", object)
            pd_kwargs.setdefault("skip_blank_lines", False)
            to_parse = pd.read_fwf(source, **pd_kwargs)
        else:
            raise ValueError("open_with must be 'pandas' or 'netcdf'")

        func_kwargs["config"] = config

        return _apply_or_chunk(
            to_parse,
            self._process_data,
            func_kwargs=func_kwargs,
            makecopy=False,
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
        """
        Read and process data from the given source.

        Parameters
        ----------
        source : str
            Path to input file(s).
        pd_kwargs: dict, optional
            Additional key-word arguments for parsing pandas-readable data.
        xr_kwargs: dict, optional
            Additional key-word arguments for parsing xarray-readable data.
        convert_kwargs: dict, optional
            Additional key-word arguments for data conversion.
        decode_kwargs: dict, optional
            Additional key-word arguments for data decoding.
        validate_kwargs: dict, optional
            Additional key-word arguments for data validation.
        select_kwargs : dict, optional
            Additional key-word arguments for selecting/filtering data.

        Notes
        -----
        All kwargs are forwarded to ``open_data`` to customize the
        parsing, conversion, decoding, validation, and selection steps.

        Returns
        -------
        DataBundle
            Container with processed data, mask, columns, dtypes, and metadata.
        """
        logging.info(f"EXTRACTING DATA FROM MODEL: {self.imodel}")
        logging.info("Reading and parsing source data...")

        result = self.open_data(
            source,
            open_with=properties.open_file.get(self.imodel, "pandas"),
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
            imodel=self.imodel,
        )
