"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import logging
import os

import pandas as pd


from .. import properties
from .utilities import (
    validate_path,
    process_textfilereader,
    validate_arg,
    remove_boolean_values,
)

from .convert_and_decode import convert_and_decode
from .validators import validate
from .parser import parse_line, Parser

from cdm_reader_mapper.core.databundle import DataBundle


def _apply_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.map(lambda x: isinstance(x, tuple)).all():
        return df

    df.columns = pd.MultiIndex.from_tuples(
        [col if isinstance(col, tuple) else (None, col) for col in df.columns],
    )
    return df


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


class FileReader:
    """Class to read marine-meteorological data."""

    def __init__(
        self,
        source,
        imodel=None,
        ext_schema_path=None,
        ext_schema_file=None,
        ext_table_path=None,
        year_init=None,
        year_end=None,
    ):
        if not imodel and not ext_schema_path:
            logging.error(
                "A valid input data model name or path to data model must be provided"
            )
            return
        if not os.path.isfile(source):
            logging.error(f"Can't find input data file {source}")
            return
        if not validate_path("ext_schema_path", ext_schema_path):
            return

        self.source = source
        self.imodel = imodel
        self.year_init = year_init
        self.year_end = year_end
        self.ext_table_path = ext_table_path

        self.pd_kwargs = {}
        self.xr_kwargs = {}

        self.sections = None
        self.encoding = None

        self.parser = Parser(
            imodel=imodel,
            ext_schema_path=ext_schema_path,
            ext_schema_file=ext_schema_file,
        )

    def _read_line(self, line: str) -> dict:
        i = 0
        out = {}

        for order, spec in self.parser.compiled_specs.items():
            header = spec.get("header")
            elements = spec.get("elements")
            is_delimited = header.get("is_delimited")

            if header.get("disable_read"):
                out[order] = line[i : properties.MAX_FULL_REPORT_WIDTH]
                continue

            i = parse_line(
                line,
                i,
                header,
                elements,
                self.sections,
                out,
                is_delimited=is_delimited,
            )

        return out

    def _select_years(self, df) -> pd.DataFrame:
        if self.year_init is None and self.year_end is None:
            return df

        data_model = self.imodel.split("_")[0]
        year_col = properties.year_column[data_model]

        years = pd.to_numeric(df[year_col], errors="coerce")

        mask = pd.Series(True, index=df.index)

        if self.year_init is not None:
            mask &= years >= self.year_init

        if self.year_end is not None:
            mask &= years <= self.year_end

        mask &= years.notna()

        return df.loc[mask].reset_index(drop=True)

    def _open_pandas(self, df) -> pd.DataFrame:
        """Parse text lines into a Pandas DataFrame."""
        col = df.columns[0]
        records = df[col].map(self._read_line)
        df = pd.DataFrame.from_records(records)
        return _apply_multiindex(df)

    def _apply_schema(
        self,
        data,
        convert_flag,
        decode_flag,
        converter_dict,
        converter_kwargs,
        decoder_dict,
        validate_flag,
    ) -> pd.DataFrame | pd.io.parsers.TextFileReader:
        data = self._open_pandas(data)
        if converter_dict is None:
            converter_dict = self.parser.convert_decode["converter_dict"]
        if converter_kwargs is None:
            converter_kwargs = self.parser.convert_decode["converter_kwargs"]
        if decoder_dict is None:
            decoder_dict = self.parser.convert_decode["decoder_dict"]
        data = convert_and_decode(
            data,
            convert_flag=convert_flag,
            decode_flag=decode_flag,
            converter_dict=converter_dict,
            converter_kwargs=converter_kwargs,
            decoder_dict=decoder_dict,
        )
        data = self._select_years(data)
        if validate_flag:
            mask = validate(
                data,
                imodel=self.imodel,
                ext_table_path=self.ext_table_path,
                schema=self.parser.schema,
                disables=self.parser.disable_reads,
            )
        else:
            mask = pd.DataFrame(True, index=data.index, columns=data.columns)
        data = remove_boolean_values(data, self.parser.dtypes)
        return data, mask

    def _open_with_pandas(
        self, **kwargs
    ) -> pd.DataFrame | pd.io.parsers.TextFileReader:
        to_parse = pd.read_fwf(
            self.source,
            header=None,
            quotechar="\0",
            escapechar="\0",
            dtype=object,
            skip_blank_lines=False,
            **self.pd_kwargs,
        )
        return _apply_or_chunk(
            to_parse,
            self._apply_schema,
            func_kwargs=kwargs,
            makecopy=False,
        )

    def open_data(
        self,
        open_with="pandas",
        encoding=None,
        chunksize=None,
        skiprows=0,
        sections=None,
        convert_flag=True,
        decode_flag=True,
        converter_dict=None,
        converter_kwargs=None,
        decoder_dict=None,
        validate_flag=True,
    ) -> pd.DataFrame | pd.io.parsers.TextFileReader:
        """DOCUMENTATION."""
        func_kwargs = {
            "convert_flag": convert_flag,
            "decode_flag": decode_flag,
            "converter_dict": converter_dict,
            "converter_kwargs": converter_kwargs,
            "decoder_dict": decoder_dict,
            "validate_flag": validate_flag,
        }
        if open_with == "netcdf":
            raise NotImplementedError
        elif open_with == "pandas":
            self.sections = sections
            self.encoding = encoding or self.parser.encoding
            self.pd_kwargs = {
                "encoding": self.encoding,
                "chunksize": chunksize,
                "skiprows": skiprows,
                "widths": [properties.MAX_FULL_REPORT_WIDTH],
            }
            return self._open_with_pandas(**func_kwargs)

    def read(
        self,
        chunksize=None,
        sections=None,
        skiprows=0,
        convert_flag=True,
        decode_flag=True,
        converter_dict=None,
        converter_kwargs=None,
        decoder_dict=None,
        validate_flag=True,
        encoding: str | None = None,
        **kwargs,
    ) -> DataBundle:
        """Read data from disk.

        Parameters
        ----------
        chunksize : int, optional
          Number of reports per chunk.
        sections : list, optional
          List with subset of data model sections to output, optional
          If None read pre-defined data model sections.
        skiprows : int
          Number of initial rows to skip from file, default: 0
        convert_flag: bool, default: True
          If True convert entries by using a pre-defined data model.
        decode_flag: bool, default: True
          If True decode entries by using a pre-defined data model.
        converter_dict: dict of {Hashable: func}, optional
          Functions for converting values in specific columns.
          If None use information from a pre-defined data model.
        converter_kwargs: dict of {Hashable: kwargs}, optional
          Key-word arguments for converting values in specific columns.
          If None use information from a pre-defined data model.
        validate_flag: bool, default: True
          Validate data entries by using a pre-defined data model.
        encoding: str, optional
          Encoding of the input file, overrides the value in the imodel schema
        """
        # 0. VALIDATE INPUT
        if not validate_arg("sections", sections, list):
            return
        if not validate_arg("chunksize", chunksize, int):
            return
        if not validate_arg("skiprows", skiprows, int):
            return

        # 2. READ AND VALIDATE DATA
        logging.info(f"EXTRACTING DATA FROM MODEL: {self.imodel}")
        # 2.1. Subset data model sections to requested sections

        # 2.2 Homogenize input data to an iterable with dataframes:
        # a list with a single dataframe or a pd.io.parsers.TextFileReader
        logging.info("Getting data string from source...")
        data, mask = self.open_data(
            # INFO: Set default as "pandas" to account for custom schema
            open_with=properties.open_file.get(self.imodel, "pandas"),
            chunksize=chunksize,
            skiprows=skiprows,
            encoding=encoding,
            sections=sections,
            convert_flag=convert_flag,
            decode_flag=decode_flag,
            converter_dict=converter_dict,
            converter_kwargs=converter_kwargs,
            decoder_dict=decoder_dict,
            validate_flag=validate_flag,
        )

        return DataBundle(
            data=data,
            columns=data.columns,
            dtypes=data.dtypes,
            parse_dates=self.parser.parse_dates,
            encoding=self.encoding,
            mask=mask,
            imodel=self.imodel,
        )
