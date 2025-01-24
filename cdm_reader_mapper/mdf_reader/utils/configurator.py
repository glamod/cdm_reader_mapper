"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import ast
import logging

import pandas as pd

from .. import properties
from . import converters, decoders
from .utilities import convert_dtypes, convert_value, decode_value, validate_value


class Configurator:
    """Class for configuring MDF reader information."""

    def __init__(
        self,
        df=pd.DataFrame(),
        schema={},
        order=[],
        valid=[],
    ):
        self.df = df
        self.orders = order
        self.valid = valid
        self.schema = schema
        self.str_line = ""
        if isinstance(df, pd.Series) or isinstance(df, pd.DataFrame):
            if len(df) > 0:
                self.str_line = df.iloc[0]

    def _add_field_length(self, index):
        if "field_length" in self.sections_dict.keys():
            field_length = self.sections_dict["field_length"]
        else:
            field_length = properties.MAX_FULL_REPORT_WIDTH
        return index + field_length

    def _validate_sentinal(self, i):
        slen = len(self.sentinal)
        str_start = self.str_line[i : i + slen]
        if str_start != self.sentinal:
            self.length = 0
            return i
        else:
            self.sentinal = None
            return self._add_field_length(i)

    def _validate_delimited(self, i, j):
        i = self._skip_delimiter(self.str_line, i)
        if self.delimiter_format == "delimited":
            j = self._next_delimiter(self.str_line, i)
            return i, j
        elif self.field_layout == "fixed_width":
            j = self._add_field_length(i)
            return i, j
        return None, None

    def _skip_delimiter(self, line, index):
        length = len(line)
        while True:
            if index == length:
                break
            if line[index] == self.delimiter:
                index += 1
            break
        return index

    def _next_delimiter(self, line, index):
        while True:
            if index == len(line):
                break
            if line[index] == self.delimiter:
                break
            index += 1
        return index

    def _get_index(self, section):
        if len(self.orders) == 1:
            return section
        else:
            return (self.order, section)

    def _get_ignore(self):
        if self.order in self.valid:
            ignore = self.sections_dict.get("ignore")
        else:
            ignore = True
        if isinstance(ignore, str):
            ignore = ast.literal_eval(ignore)
        return ignore

    def _get_borders(self, i, j):
        if self.sentinal is not None:
            j = self._validate_sentinal(i)
        elif self.delimiter is None:
            j = self._add_field_length(i)
        else:
            i, j = self._validate_delimited(i, j)
            self.missing = False
        return i, j

    def _adjust_right_borders(self, j, k):
        if self.length is None:
            self.length = j - k
        if j - k > self.length:
            self.missing = False
            j = k + self.length
        return j, k

    def _get_dtype(self):
        return properties.pandas_dtypes.get(self.sections_dict.get("column_type"))

    def _get_converter(self):
        return converters.get(self.sections_dict.get("column_type"))

    def _get_conv_kwargs(self):
        column_type = self.sections_dict.get("column_type")
        if column_type is None:
            return
        return {
            converter_arg: self.sections_dict.get(converter_arg)
            for converter_arg in properties.data_type_conversion_args.get(column_type)
        }

    def _get_decoder(self):
        encoding = self.sections_dict.get("encoding")
        if encoding is None:
            return
        column_type = self.sections_dict.get("column_type")
        if column_type is None:
            return
        return decoders.get(encoding).get(column_type)

    def _update_dtypes(self, dtypes, index):
        dtype = self._get_dtype()
        if dtype:
            dtypes[index] = dtype
        return dtypes

    def _update_converters(self, converters, index):
        converter = self._get_converter()
        if converter:
            converters[index] = converter
        return converters

    def _update_kwargs(self, kwargs, index):
        conv_kwargs = self._get_conv_kwargs()
        if conv_kwargs:
            kwargs[index] = conv_kwargs
        return kwargs

    def _update_decoders(self, decoders, index):
        decoder = self._get_decoder()
        if decoder:
            decoders[index] = decoder
        return decoders

    def get_configuration(self):
        """Get ICOADS data model specific information."""
        dtypes = {}
        converters = {}
        kwargs = {}
        decoders = {}
        for order in self.orders:
            self.order = order
            header = self.schema["sections"][order]["header"]
            disable_read = header.get("disable_read")
            if disable_read is True:
                continue
            sections = self.schema["sections"][order]["elements"]
            for section in sections.keys():
                self.sections_dict = sections[section]
                index = self._get_index(section)
                ignore = self._get_ignore()
                if ignore is True:
                    continue
                dtypes = self._update_dtypes(dtypes, index)
                converters = self._update_converters(converters, index)
                kwargs = self._update_kwargs(kwargs, index)
                decoders = self._update_decoders(decoders, index)

        dtypes, parse_dates = convert_dtypes(dtypes)
        return {
            "converter_dict": converters,
            "converter_kwargs": kwargs,
            "decoder_dict": decoders,
            "dtype": dtypes,
            "self": {
                "dtypes": dtypes,
                "parse_dates": parse_dates,
            },
        }

    def open_pandas(self, configurations, imodel, ext_table_path):
        """Open TextParser to pd.DataSeries."""
        self.delimiter = None
        i = 0
        j = 0
        data_dict = {}
        mask_dict = {}
        convert = configurations.get("convert", False)
        converter_dict = configurations.get("converter_dict", {})
        converter_kwargs = configurations.get("converter_kwargs", {})
        decode = configurations.get("decode", False)
        decoder_dict = configurations.get("decoder_dict", {})
        validate = configurations.get("validate", False)

        for order in self.orders:
            self.order = order
            header = self.schema["sections"][order]["header"]
            self.sentinal = header.get("sentinal")
            self.sentinal_length = header.get("sentinal_length")
            self.delimiter = header.get("delimiter")
            self.field_layout = header.get("field_layout")
            self.delimiter_format = header.get("format")
            disable_read = header.get("disable_read")
            if disable_read is True:
                data_dict[order] = self.str_line[i : properties.MAX_FULL_REPORT_WIDTH]
                continue
            sections = self.schema["sections"][order]["elements"]
            k = i
            for section in sections.keys():
                self.length = header.get("length")
                self.missing = True
                self.sections_dict = sections[section]
                index = self._get_index(section)
                ignore = self._get_ignore()
                na_value = sections[section].get("missing_value")

                i, j = self._get_borders(i, j)

                if i is None:
                    logging.error(
                        f"Delimiter is set to {self.delimiter}. Please specify either format or field_layout in your header schema {header}."
                    )
                    return

                j, k = self._adjust_right_borders(j, k)

                if ignore is True:
                    i = j
                    continue

                value = self.str_line[i:j]
                if not value.strip():
                    value = None
                if value == na_value:
                    value = None
                isna = not value
                if decode is True:
                    value = decode_value(value, index, decoder_dict)
                if convert is True:
                    value = convert_value(
                        value, index, converter_dict, converter_kwargs
                    )

                if validate is True:
                    missing = False
                    if i == j and self.missing is True:
                        missing = True
                    mask_dict[index] = validate_value(
                        value,
                        isna,
                        missing,
                        imodel,
                        index,
                        ext_table_path,
                        self.schema,
                        disable_read,
                    )
                data_dict[index] = value

                i = j

        df = pd.Series(data_dict)
        mask = pd.Series(mask_dict)
        return pd.concat([df, mask])

    def open_netcdf(self, configurations):
        """Open netCDF to pd.Series."""

        def replace_empty_strings(series):
            if series.dtype == "object":
                series = series.str.decode("utf-8")
                series = series.str.strip().replace("", None)
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
                index = self._get_index(section)
                ignore = self._get_ignore()
                if ignore is True:
                    continue
                if section in self.df.data_vars:
                    renames[section] = index
                elif section in self.df.dims:
                    renames[section] = index
                elif section in self.df.attrs:
                    attrs[index] = self.df.attrs[index]
                else:
                    missing_values.append(index)

        df = self.df[renames.keys()].to_dataframe().reset_index()
        attrs = {k: v.replace("\n", "; ") for k, v in attrs.items()}
        df = df.rename(columns=renames)
        df = df.assign(**attrs)
        for column in disables:
            df[column] = None
        df = df.apply(lambda x: replace_empty_strings(x))
        df["missing_values"] = [missing_values] * len(df)
        return df
