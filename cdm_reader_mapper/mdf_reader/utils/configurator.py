"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import ast
import csv
import logging

import numpy as np
import pandas as pd

from .. import properties
from . import converters, decoders
from .utilities import convert_dtypes


class Configurator:
    """Class for configuring MDF reader information."""

    def __init__(
        self,
        df=pd.DataFrame(),
        schema=None,
        order=None,
        valid=None,
    ):
        self.df = df
        self.orders = order or []
        self.valid = valid or []
        self.schema = schema or {}

    def _validate_sentinel(self, i, line, sentinel) -> bool:
        slen = len(sentinel)
        str_start = line[i : i + slen]
        return str_start == sentinel

    def _get_index(self, section, order) -> dict | tuple[str, dict]:
        if len(self.orders) == 1:
            return section
        else:
            return (order, section)

    def _get_ignore(self, section_dict) -> bool:
        ignore = section_dict.get("ignore")
        if isinstance(ignore, str):
            ignore = ast.literal_eval(ignore)
        return ignore

    def _get_dtype(self) -> str:
        return properties.pandas_dtypes.get(self.sections_dict.get("column_type"))

    def _get_converter(self) -> callable:
        return converters.get(self.sections_dict.get("column_type"))

    def _get_conv_kwargs(self) -> dict:
        column_type = self.sections_dict.get("column_type")
        if column_type is None:
            return
        return {
            converter_arg: self.sections_dict.get(converter_arg)
            for converter_arg in properties.data_type_conversion_args.get(column_type)
        }

    def _get_decoder(self) -> callable | None:
        encoding = self.sections_dict.get("encoding")
        if encoding is None:
            return
        column_type = self.sections_dict.get("column_type")
        if column_type is None:
            return
        return decoders.get(encoding).get(column_type)

    def _update_dtypes(self, dtypes, index) -> dict:
        dtype = self._get_dtype()
        if dtype:
            dtypes[index] = dtype
        return dtypes

    def _update_converters(self, converters, index) -> dict:
        converter = self._get_converter()
        if converter:
            converters[index] = converter
        return converters

    def _update_kwargs(self, kwargs, index) -> dict:
        conv_kwargs = self._get_conv_kwargs()
        if conv_kwargs:
            kwargs[index] = conv_kwargs
        return kwargs

    def _update_decoders(self, decoders, index) -> dict:
        decoder = self._get_decoder()
        if decoder:
            decoders[index] = decoder
        return decoders

    def get_configuration(self) -> dict:
        """Get ICOADS data model specific information."""
        disable_reads = []
        dtypes = {}
        converters = {}
        kwargs = {}
        decoders = {}
        for order in self.orders:
            self.order = order
            header = self.schema["sections"][order]["header"]
            disable_read = header.get("disable_read")
            if disable_read is True:
                disable_reads.append(order)
                continue
            sections = self.schema["sections"][order]["elements"]
            for section in sections.keys():
                self.sections_dict = sections[section]
                index = self._get_index(section, order)
                ignore = (order not in self.valid) or self._get_ignore(
                    self.sections_dict
                )
                if ignore is True:
                    continue
                dtypes = self._update_dtypes(dtypes, index)
                converters = self._update_converters(converters, index)
                kwargs = self._update_kwargs(kwargs, index)
                decoders = self._update_decoders(decoders, index)

        dtypes, parse_dates = convert_dtypes(dtypes)
        return {
            "convert_decode": {
                "converter_dict": converters,
                "converter_kwargs": kwargs,
                "decoder_dict": decoders,
            },
            "self": {
                "dtypes": dtypes,
                "disable_reads": disable_reads,
                "parse_dates": parse_dates,
                "encoding": self.schema["header"].get("encoding", "utf-8"),
            },
        }

    def open_pandas(self) -> pd.DataFrame:
        """Open TextParser to pd.DataSeries."""
        return self.df.apply(lambda x: self._read_line(x[0]), axis=1)

    def _read_line(self, line: str) -> pd.Series:
        i = j = 0
        data_dict = {}
        for order in self.orders:
            header = self.schema["sections"][order]["header"]

            disable_read = header.get("disable_read")
            if disable_read is True:
                data_dict[order] = line[i : properties.MAX_FULL_REPORT_WIDTH]
                continue

            sentinel = header.get("sentinel")
            bad_sentinel = sentinel is not None and not self._validate_sentinel(
                i, line, sentinel
            )

            section_length = header.get("length", properties.MAX_FULL_REPORT_WIDTH)
            sections = self.schema["sections"][order]["elements"]

            field_layout = header.get("field_layout")
            delimiter = header.get("delimiter")
            if delimiter is not None:
                delimiter_format = header.get("format")
                if delimiter_format == "delimited":
                    # Read as CSV
                    field_names = sections.keys()
                    fields = list(csv.reader([line[i:]], delimiter=delimiter))[0]
                    for field_name, field in zip(field_names, fields):
                        index = self._get_index(field_name, order)
                        data_dict[index] = field.strip()
                        i += len(field)
                    j = i
                    continue
                elif field_layout != "fixed_width":
                    logging.error(
                        f"Delimiter for {order} is set to {delimiter}. Please specify either format or field_layout in your header schema {header}."
                    )
                    return

            k = i + section_length
            for section, section_dict in sections.items():
                missing = True
                index = self._get_index(section, order)
                ignore = (order not in self.valid) or self._get_ignore(section_dict)
                na_value = section_dict.get("missing_value")
                field_length = section_dict.get(
                    "field_length", properties.MAX_FULL_REPORT_WIDTH
                )

                j = (i + field_length) if not bad_sentinel else i
                if j > k:
                    missing = False
                    j = k

                if ignore is not True:
                    value = line[i:j]

                    if not value.strip():
                        value = True
                    if value == na_value:
                        value = True

                    if i == j and missing is True:
                        value = False

                    data_dict[index] = value

                if delimiter is not None and line[j : j + len(delimiter)] == delimiter:
                    j += len(delimiter)
                i = j

        return pd.Series(data_dict)

    def open_netcdf(self) -> pd.DataFrame:
        """Open netCDF to pd.Series."""

        def replace_empty_strings(series):
            if series.dtype == "object":
                series = series.str.decode("utf-8")
                series = series.str.strip()
                series = series.map(lambda x: True if x == "" else x)
            return series

        missing_values = []
        attrs = {}
        renames = {}
        disables = []
        for order in self.orders:
            self.order = order
            header = self.schema["sections"][order]["header"]
            disable_read = header.get("disable_read")
            if disable_read is True:
                disables.append(order)
                continue
            sections = self.schema["sections"][order]["elements"]
            for section in sections.keys():
                self.sections_dict = sections[section]
                index = self._get_index(section, order)
                ignore = (order not in self.valid) or self._get_ignore(
                    self.sections_dict
                )
                if ignore is True:
                    continue
                if section in self.df.data_vars:
                    renames[section] = index
                elif section in self.df.dims:
                    renames[section] = index
                elif section in self.df.attrs:
                    attrs[index] = self.df.attrs[section]
                else:
                    missing_values.append(index)

        df = self.df[renames.keys()].to_dataframe().reset_index()
        attrs = {k: v.replace("\n", "; ") for k, v in attrs.items()}
        df = df.rename(columns=renames)
        df = df.assign(**attrs)
        df[disables] = np.nan
        df = df.apply(lambda x: replace_empty_strings(x))
        df[missing_values] = False
        return df
