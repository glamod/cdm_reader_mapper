"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import logging
import os

import pandas as pd


from .. import properties
from .utilities import validate_path, process_textfilereader
from .utilities import remove_boolean_values

from .convert_and_decode import convert_and_decode
from .validators import validate
from .parser import parse_fixed_width, parse_delimited, Parser


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

        for (
            order,
            header,
            elements,
            compiled_elements,
            is_delimited,
        ) in self.parser.compiled_specs:
            if header.get("disable_read"):
                out[order] = line[i : properties.MAX_FULL_REPORT_WIDTH]
                continue

            if is_delimited:
                i = parse_delimited(
                    line, i, order, header, elements, self.parser.olength, out
                )
            else:
                i = parse_fixed_width(
                    line, i, header, compiled_elements, self.sections, out
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
    ) -> pd.DataFrame | pd.io.parsers.TextFileReader:
        data = self._open_pandas(data)
        data = convert_and_decode(
            data,
            converter_dict=self.parser.convert_decode["converter_dict"],
            converter_kwargs=self.parser.convert_decode["converter_kwargs"],
            decoder_dict=self.parser.convert_decode["decoder_dict"],
        )
        data = self._select_years(data)
        mask = validate(
            data,
            imodel=self.imodel,
            ext_table_path=self.ext_table_path,
            schema=self.parser.schema,
            disables=self.parser.disable_reads,
        )
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
            **kwargs,
        )
        return _apply_or_chunk(
            to_parse,
            self._apply_schema,
            makecopy=False,
        )

    def open_data(
        self,
        open_with="pandas",
        encoding=None,
        chunksize=None,
        skiprows=0,
        sections=None,
    ) -> pd.DataFrame | pd.io.parsers.TextFileReader:
        """DOCUMENTATION."""
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
            return self._open_with_pandas(**self.pd_kwargs)
