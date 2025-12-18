"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import ast
import csv
import logging
import os

import pandas as pd

from itertools import zip_longest

from .. import properties
from ..schemas import schemas
from .utilities import validate_path, process_textfilereader
from .utilities import convert_dtypes, remove_boolean_values

from .convert_and_decode import Converters, Decoders, convert_and_decode
from .validators import validate


def _apply_multiindex(df: pd.DataFrame, length) -> pd.DataFrame:
    if length == 1:
        return df

    df.columns = pd.MultiIndex.from_tuples(
        [col if isinstance(col, tuple) else (None, col) for col in df.columns],
    )
    return df


def _validate_sentinel(i: int, line: str, sentinel: str) -> bool:
    return line.startswith(sentinel, i)


def _get_index(section, order, length):
    if length == 1:
        return section
    return (order, section)


def _get_ignore(section_dict) -> bool:
    ignore = section_dict.get("ignore", False)
    if isinstance(ignore, str):
        ignore = ast.literal_eval(ignore)
    return bool(ignore)


def _parse_fixed_width(
    line: str,
    i: int,
    header: dict,
    compiled_elements: list,
    sections: list,
    out: dict,
) -> int:
    section_length = header.get("length", properties.MAX_FULL_REPORT_WIDTH)
    delimiter = header.get("delimiter")
    sentinel = header.get("sentinel")

    bad_sentinel = sentinel is not None and not _validate_sentinel(i, line, sentinel)
    k = i + section_length

    for index, na_value, field_length, ignore in compiled_elements:
        if isinstance(index, tuple):
            in_sections = index[0] in sections
        else:
            in_sections = index in sections

        missing = True

        j = i if bad_sentinel else i + field_length
        if j > k:
            missing = False
            j = k

        if not ignore and in_sections:
            value = line[i:j]
            if not value.strip() or value == na_value:
                value = True
            if i == j and missing:
                value = False
            out[index] = value

        if delimiter and line[j : j + len(delimiter)] == delimiter:
            j += len(delimiter)

        i = j

    return i


def _parse_delimited(
    line: str,
    i: int,
    order: str,
    header: dict,
    elements: dict,
    olength: int,
    out: dict,
) -> int:
    delimiter = header["delimiter"]
    fields = next(csv.reader([line[i:]], delimiter=delimiter))

    for name, value in zip_longest(elements.keys(), fields):
        out[_get_index(name, order, olength)] = (
            value.strip() if value is not None else None
        )
        if value is not None:
            i += len(value)

    return i


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

        logging.info("READING DATA MODEL SCHEMA FILE...")
        if ext_schema_path or ext_schema_file:
            self.schema = schemas.read_schema(
                ext_schema_path=ext_schema_path, ext_schema_file=ext_schema_file
            )
        else:
            self.schema = schemas.read_schema(imodel=imodel)

        parsing_order = self.schema["header"].get("parsing_order")
        sections_ = [x.get(y) for x in parsing_order for y in x]
        self.orders = [y for x in sections_ for y in x]
        self.olength = len(self.orders)

        self._build_compiled_specs_and_convertdecode()

        self.pd_kwargs = {}
        self.xr_kwargs = {}

        self.sections = None
        self.encoding = None

    def _build_compiled_specs_and_convertdecode(self):
        compiled_specs = []
        disable_reads = []
        dtypes = {}
        converter_dict = {}
        converter_kwargs = {}
        decoder_dict = {}

        for order in self.orders:
            section = self.schema["sections"][order]
            header = section["header"]
            elements = section["elements"]

            disable_read = header.get("disable_read", False)
            if disable_reads:
                disable_reads.append(order)

            compiled_elements = []
            for name, meta in elements.items():
                index = _get_index(name, order, self.olength)
                ignore = _get_ignore(meta)

                compiled_elements.append(
                    (
                        index,
                        meta.get("missing_value"),
                        meta.get("field_length", properties.MAX_FULL_REPORT_WIDTH),
                        ignore,
                    )
                )

                if disable_read:
                    continue

                if ignore:
                    continue

                ctype = meta.get("column_type")
                dtype = properties.pandas_dtypes.get(ctype)

                if dtype:
                    dtypes[index] = dtype

                conv_func = Converters(ctype).converter()
                if conv_func:
                    converter_dict[index] = conv_func

                conv_kwargs = {
                    k: meta.get(k)
                    for k in properties.data_type_conversion_args.get(ctype, [])
                }
                if conv_kwargs:
                    converter_kwargs[index] = conv_kwargs

                encoding = meta.get("encoding")
                if encoding:
                    dec_func = Decoders(ctype, encoding).decoder()
                    if dec_func:
                        decoder_dict[index] = dec_func

            compiled_specs.append(
                (
                    order,
                    header,
                    elements,
                    compiled_elements,
                    header.get("format") == "delimited",
                )
            )

        self.dtypes, self.parse_dates = convert_dtypes(dtypes)

        self.disable_reads = disable_reads

        self.convert_decode = {
            "converter_dict": converter_dict,
            "converter_kwargs": converter_kwargs,
            "decoder_dict": decoder_dict,
        }

        self.compiled_specs = compiled_specs

    def _apply_or_chunk(self, data, func, func_args=[], func_kwargs={}, **kwargs):
        if not isinstance(data, pd.io.parsers.TextFileReader):
            return func(data, *func_args, **func_kwargs)
        return process_textfilereader(
            data,
            func,
            func_args,
            func_kwargs,
            **kwargs,
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
        ) in self.compiled_specs:
            if header.get("disable_read"):
                out[order] = line[i : properties.MAX_FULL_REPORT_WIDTH]
                continue

            if is_delimited:
                i = _parse_delimited(
                    line, i, order, header, elements, self.olength, out
                )
            else:
                i = _parse_fixed_width(
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
        return _apply_multiindex(df, self.olength)

    def _apply_schema(
        self,
        data,
    ) -> pd.DataFrame | pd.io.parsers.TextFileReader:
        data = self._open_pandas(data)
        data = convert_and_decode(
            data,
            converter_dict=self.convert_decode["converter_dict"],
            converter_kwargs=self.convert_decode["converter_kwargs"],
            decoder_dict=self.convert_decode["decoder_dict"],
        )
        data = self._select_years(data)
        mask = validate(
            data,
            imodel=self.imodel,
            ext_table_path=self.ext_table_path,
            schema=self.schema,
            disables=self.disable_reads,
        )
        data = remove_boolean_values(data, self.dtypes)
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
        return self._apply_or_chunk(
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
            if encoding is None:
                encoding = self.schema["header"].get("encoding", "utf-8")

            self.sections = sections
            self.encoding = encoding
            self.pd_kwargs = {
                "encoding": encoding,
                "chunksize": chunksize,
                "skiprows": skiprows,
                "widths": [properties.MAX_FULL_REPORT_WIDTH],
            }
            return self._open_with_pandas(**self.pd_kwargs)
