"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import ast

import pandas as pd
import polars as pl
import polars.selectors as cs
import xarray as xr

from .. import properties
from . import converters, decoders
from .utilities import convert_dtypes


class Configurator:
    """Class for configuring MDF reader information."""

    def __init__(
        self,
        df: pd.DataFrame | pl.DataFrame | xr.Dataset = pl.DataFrame(),
        schema=None,
        order=None,
        valid=None,
    ):
        self.df = df
        self.orders = order or []
        self.valid = valid or []
        self.schema = schema or {}

    def _validate_sentinal(self, i, line, sentinal):
        slen = len(sentinal)
        str_start = line[i : i + slen]
        return str_start == sentinal

    def _get_index(self, section, order):
        if len(self.orders) == 1:
            return section
        else:
            return ":".join([order, section])

    def _get_ignore(self, section_dict):
        ignore = section_dict.get("ignore")
        if isinstance(ignore, str):
            ignore = ast.literal_eval(ignore)
        return ignore

    def _get_dtype(self):
        return properties.polars_dtypes.get(self.sections_dict.get("column_type"))

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

    def open_text(self):
        """Open TextParser to a polars.DataFrame"""
        if isinstance(self.df, pd.DataFrame):
            self.df = pl.from_pandas(self.df)
        if not isinstance(self.df, pl.DataFrame):
            raise TypeError(f"Cannot open with polars for {type(self.df) = }")
        self.df = self.df.with_row_index("index")
        mask_df = pl.DataFrame()
        for section in self.orders:
            header = self.schema["sections"][section]["header"]

            sentinal = header.get("sentinal")

            section_length = header.get("length", properties.MAX_FULL_REPORT_WIDTH)

            # Get data associated with current section
            if sentinal is not None:
                self.df = self.df.with_columns(
                    [
                        (
                            pl.when(pl.col("full_str").str.starts_with(sentinal))
                            .then(pl.col("full_str").str.head(section_length))
                            .otherwise(pl.lit(None))
                            .alias(section)
                        ),
                        (
                            pl.when(pl.col("full_str").str.starts_with(sentinal))
                            .then(pl.col("full_str").str.tail(-section_length))
                            .otherwise(pl.col("full_str"))
                            .alias("full_str")
                        ),
                    ]
                )
            else:
                # Sentinal is None, the section is always present
                self.df = self.df.with_columns(
                    [
                        pl.col("full_str").str.head(section_length).alias(section),
                        pl.col("full_str").str.tail(-section_length).alias("full_str"),
                    ]
                )

            # Used for validation
            section_missing = self.df.get_column(section).is_null()

            # Don't read fields
            disable_read = header.get("disable_read", False)
            if disable_read is True:
                continue

            fields = self.schema["sections"][section]["elements"]
            field_layout = header.get("field_layout")
            delimiter = header.get("delimiter")

            # Handle delimited section
            if delimiter is not None:
                delimiter_format = header.get("format")
                if delimiter_format == "delimited":
                    # Read as CSV
                    field_names = fields.keys()
                    field_names = [self._get_index(section, x) for x in field_names]
                    n_fields = len(field_names)
                    self.df = self.df.with_columns(
                        pl.col(section)
                        .str.splitn(delimiter, n_fields)
                        .struct.rename_fields(field_names)
                        .struct.unnest()
                    )
                    for field in field_names:
                        self.df = self.df.with_columns(
                            pl.col(field).str.strip_chars(" ").name.keep()
                        )
                        mask_df = mask_df.with_columns(
                            (
                                section_missing
                                | self.df.get_column(field).is_not_null()
                            ).alias(field)
                        )

                    continue
                elif field_layout != "fixed_width":
                    raise ValueError(
                        f"Delimiter for {section} is set to {delimiter}. "
                        + f"Please specify either format or field_layout in your header schema {header}."
                    )

            # Loop through fixed-width fields
            for field, field_dict in fields.items():
                index = self._get_index(field, section)
                ignore = (section not in self.valid) or self._get_ignore(field_dict)
                field_length = field_dict.get(
                    "field_length", properties.MAX_FULL_REPORT_WIDTH
                )
                na_value = field_dict.get("missing_value")

                if ignore:
                    # Move to next field
                    self.df = self.df.with_columns(
                        pl.col(section).str.slice(field_length).name.keep(),
                    )
                    if delimiter is not None:
                        self.df = self.df.with_columns(
                            pl.col(section).str.strip_prefix(delimiter).name.keep()
                        )
                    continue

                missing_map = {"": None}
                if na_value is not None:
                    missing_map[na_value] = None

                self.df = self.df.with_columns(
                    [
                        # If section not present in a row, then both these are null
                        (
                            pl.col(section)
                            .str.head(field_length)
                            .str.strip_chars(" ")
                            .replace(missing_map)
                            .alias(index)
                        ),
                        pl.col(section).str.tail(-field_length).name.keep(),
                    ]
                )
                mask_df = mask_df.with_columns(
                    (section_missing | self.df.get_column(field).is_not_null()).alias(
                        field
                    )
                )
                if delimiter is not None:
                    self.df = self.df.with_columns(
                        pl.col(section).str.strip_prefix(delimiter).name.keep()
                    )

            self.df = self.df.drop([section])

        return self.df.drop("full_str"), mask_df.with_row_index("index")

    def open_netcdf(self):
        """Open netCDF to polars.DataFrame."""
        if not isinstance(self.df, xr.Dataset):
            raise TypeError(f"Cannot open with netCDF for {type(self.df) = }")

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
                    # Initialise a constant column
                    attrs[index] = pl.lit(self.df.attrs[section].replace("\n", "; "))
                else:
                    missing_values.append(index)

        df: pl.DataFrame = pl.from_pandas(
            self.df[renames.keys()].to_dataframe().reset_index()
        ).with_row_index("index")
        df = df.rename(mapping=renames)
        df = df.with_columns(**attrs)
        df = df.with_columns(
            [pl.lit(None).alias(missing) for missing in missing_values]
        )
        df = df.with_columns([pl.lit(None).alias(disable) for disable in disables])
        # Replace empty or whitespace string with None
        df = df.with_columns(cs.string().str.strip_chars().replace("", None))

        # Create missing mask
        mask_df = df.select(pl.all().is_not_null())
        mask_df = mask_df.with_columns(
            [pl.lit(True).alias(c) for c in missing_values + disables]
        )
        return df, mask_df.with_row_index("index")
