"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import logging

import pandas as pd
import xarray as xr

from .. import properties
from .utilities import (
    process_textfilereader,
    remove_boolean_values,
)

from .convert_and_decode import convert_and_decode
from .validators import validate
from .parser import Parser

from cdm_reader_mapper.core.databundle import DataBundle


def _apply_or_chunk(data, func, func_args=[], func_kwargs={}, **kwargs):
    if not isinstance(data, pd.io.parsers.TextFileReader):
        return func(data, *func_args, **func_kwargs)
    return process_textfilereader(
        data,
        func,
        func_args,
        func_kwargs,
        **kwargs,
    )


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


class FileReader(Parser):
    """Class to read marine-meteorological data."""

    def __init__(self, *args, **kwargs):
        Parser.__init__(self, *args, **kwargs)

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
        year_init,
        year_end,
        parse_mode="pandas",
    ) -> pd.DataFrame | pd.io.parsers.TextFileReader:
        if parse_mode == "pandas":
            data = self.parse_pandas(data, sections)
        elif parse_mode == "netcdf":
            data = self.parse_netcdf(data, sections)
        else:
            raise ValueError("open_with has to be one of ['pandas', 'netcdf']")

        data = _apply_multiindex(data)

        data_model = self.imodel.split("_")[0]
        year_col = properties.year_column[data_model]

        data = _select_years(data, [year_init, year_end], year_col)

        if converter_dict is None:
            converter_dict = self.convert_decode["converter_dict"]
        if converter_kwargs is None:
            converter_kwargs = self.convert_decode["converter_kwargs"]
        if decoder_dict is None:
            decoder_dict = self.convert_decode["decoder_dict"]

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
                attributes=self.validation,
                disables=self.disable_reads,
            )
        else:
            mask = pd.DataFrame(True, index=data.index, columns=data.columns)

        self.columns = data.columns
        data = remove_boolean_values(data, self.dtypes)
        return data, mask

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
    ) -> pd.DataFrame | pd.io.parsers.TextFileReader:
        """DOCUMENTATION."""
        func_kwargs = {
            **convert_kwargs,
            **decode_kwargs,
            **validate_kwargs,
            **select_kwargs,
            "parse_mode": open_with,
        }
        if open_with == "netcdf":
            to_parse = xr.open_mfdataset(source, xr_kwargs).squeeze()
            self.adjust_schema(to_parse)
            write_kwargs, read_kwargs = {}, {}
        elif open_with == "pandas":
            if pd_kwargs.get("encoding"):
                self.encoding = pd_kwargs["encoding"]
            else:
                pd_kwargs["encoding"] = self.encoding
            if not pd_kwargs.get("widths"):
                pd_kwargs["widths"] = [properties.MAX_FULL_REPORT_WIDTH]
            if not pd_kwargs.get("header"):
                pd_kwargs["header"] = None
            if not pd_kwargs.get("quotechar"):
                pd_kwargs["quotechar"] = "\0"
            if not pd_kwargs.get("escapechar"):
                pd_kwargs["escapechar"] = "\0"
            if not pd_kwargs.get("dtype"):
                pd_kwargs["dtype"] = object
            if not pd_kwargs.get("skip_blank_lines"):
                pd_kwargs["skip_blank_lines"] = False

            write_kwargs = {"encoding": pd_kwargs["encoding"]}
            read_kwargs = (
                {
                    "chunksize": pd_kwargs["chunksize"] or None,
                    "dtype": self.dtypes,
                },
                {
                    "chunksize": pd_kwargs["chunksize"] or None,
                    "dtype": "boolean",
                },
            )

            to_parse = pd.read_fwf(source, **pd_kwargs)
        else:
            raise ValueError("open_with has to be one of ['pandas', 'netcdf']")

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
        if pd_kwargs is None:
            pd_kwargs = {}
        if xr_kwargs is None:
            xr_kwargs = {}
        if convert_kwargs is None:
            convert_kwargs = {}
        if decode_kwargs is None:
            decode_kwargs = {}
        if validate_kwargs is None:
            validate_kwargs = {}
        if select_kwargs is None:
            select_kwargs = {}

        # 2. READ AND VALIDATE DATA
        logging.info(f"EXTRACTING DATA FROM MODEL: {self.imodel}")
        # 2.1. Subset data model sections to requested sections

        # 2.2 Homogenize input data to an iterable with dataframes:
        # a list with a single dataframe or a pd.io.parsers.TextFileReader
        logging.info("Getting data string from source...")
        data, mask = self.open_data(
            # INFO: Set default as "pandas" to account for custom schema
            source,
            open_with=properties.open_file.get(self.imodel, "pandas"),
            pd_kwargs=pd_kwargs,
            xr_kwargs=xr_kwargs,
            convert_kwargs=convert_kwargs,
            decode_kwargs=decode_kwargs,
            validate_kwargs=validate_kwargs,
            select_kwargs=select_kwargs,
        )

        return DataBundle(
            data=data,
            columns=self.columns,
            dtypes=self.dtypes,
            parse_dates=self.parse_dates,
            encoding=self.encoding,
            mask=mask,
            imodel=self.imodel,
        )
