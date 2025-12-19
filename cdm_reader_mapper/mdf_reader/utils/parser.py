"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import ast
import csv
import logging

from copy import deepcopy
from itertools import zip_longest

import numpy as np
import pandas as pd

from .. import properties
from ..schemas import schemas
from .utilities import convert_dtypes

from .convert_and_decode import Converters, Decoders


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


def _is_in_sections(index, sections):
    if not sections:
        return True
    elif isinstance(index, tuple):
        return index[0] in sections
    return index in sections


def _element_specs(
    order,
    olength,
    elements,
    converter_dict,
    converter_kwargs,
    decoder_dict,
    validation_dict,
    dtypes,
):
    element_specs = {}

    for name, meta in elements.items():
        index = _get_index(name, order, olength)
        ignore = _get_ignore(meta)

        element_specs[name] = {
            "missing_value": meta.get("missing_value"),
            "field_length": meta.get("field_length", properties.MAX_FULL_REPORT_WIDTH),
            "ignore": ignore,
            "index": index,
        }

        if meta.get("disable_read", False) or ignore:
            continue

        validation_dict[index] = {}

        ctype = meta.get("column_type")
        if ctype:
            validation_dict[index]["column_type"] = ctype

        dtype = properties.pandas_dtypes.get(ctype)

        if dtype:
            dtypes[index] = dtype

        vmin = meta.get("valid_min")

        if vmin:
            validation_dict[index]["valid_min"] = vmin

        vmax = meta.get("valid_max")
        if vmax:
            validation_dict[index]["valid_max"] = vmax

        ctable = meta.get("codetable")
        if ctable:
            validation_dict[index]["codetable"] = ctable

        conv_func = Converters(ctype).converter()
        if conv_func:
            converter_dict[index] = conv_func

        conv_kwargs = {
            k: meta.get(k) for k in properties.data_type_conversion_args.get(ctype, [])
        }
        if conv_kwargs:
            converter_kwargs[index] = conv_kwargs

        encoding = meta.get("encoding")
        if encoding:
            dec_func = Decoders(ctype, encoding).decoder()
            if dec_func:
                decoder_dict[index] = dec_func
    return element_specs


def _order_specs(orders, sections, *args):
    order_specs = {}
    disable_reads = []

    olength = len(orders)
    for order in orders:
        section = sections[order]
        header = section["header"]
        elements = section.get("elements", {})

        if header.get("disable_read", False):
            disable_reads.append(order)

        element_specs = _element_specs(
            order,
            olength,
            elements,
            *args,
        )

        order_specs[order] = {
            "header": header,
            "elements": element_specs,
            "is_delimited": header.get("format") == "delimited",
        }

    return order_specs, disable_reads


def _parse_fixed_width(
    line: str,
    i: int,
    header: dict,
    elements: dict,
    sections: list,
    out: dict,
) -> int:
    section_length = header.get("length", properties.MAX_FULL_REPORT_WIDTH)
    delimiter = header.get("delimiter")
    sentinel = header.get("sentinel")

    bad_sentinel = sentinel is not None and not _validate_sentinel(i, line, sentinel)
    k = i + section_length

    for element, spec in elements.items():
        missing_value = spec.get("missing_value")
        field_length = spec.get("field_length")
        ignore = spec.get("ignore")
        index = spec.get("index")

        missing = True

        j = i if bad_sentinel else i + field_length
        if j > k:
            missing = False
            j = k

        if not ignore and _is_in_sections(index, sections):
            value = line[i:j]
            if not value.strip() or value == missing_value:
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
    header: dict,
    elements: dict,
    sections: list,
    out: dict,
) -> int:
    delimiter = header["delimiter"]
    fields = next(csv.reader([line[i:]], delimiter=delimiter))

    for element, value in zip_longest(elements.keys(), fields):
        index = elements[element].get("index")
        if _is_in_sections(index, sections):
            out[index] = value.strip() if value is not None else None
        if value is not None:
            i += len(value)

    return i


def parse_line(*args, is_delimited):
    if is_delimited:
        return _parse_delimited(*args)
    return _parse_fixed_width(*args)


class Parser:

    def __init__(self, imodel, ext_schema_path, ext_schema_file):

        self.imodel = imodel

        logging.info("READING DATA MODEL SCHEMA FILE...")
        if ext_schema_path or ext_schema_file:

            self.schema = schemas.read_schema(
                ext_schema_path=ext_schema_path, ext_schema_file=ext_schema_file
            )
        elif imodel:
            self.schema = schemas.read_schema(imodel=imodel)
        else:
            raise ValueError(
                "One of ['imodel', 'ext_schema_path', 'ext_schema_file'] must be set."
            )

        self.build_parsing_order()
        self.build_compiled_specs_and_convertdecode()

    def build_parsing_order(self):
        parsing_order = self.schema["header"].get("parsing_order")
        sections_ = [x.get(y) for x in parsing_order for y in x]
        self.orders = [y for x in sections_ for y in x]
        self.olength = len(self.orders)

    def build_compiled_specs_and_convertdecode(self):
        dtypes = {}
        converter_dict = {}
        converter_kwargs = {}
        decoder_dict = {}
        validation_dict = {}

        self.order_specs, self.disable_reads = _order_specs(
            self.orders,
            self.schema["sections"],
            converter_dict,
            converter_kwargs,
            decoder_dict,
            validation_dict,
            dtypes,
        )

        self.encoding = self.schema["header"].get("encoding", "utf-8")

        self.dtypes, self.parse_dates = convert_dtypes(dtypes)

        self.convert_decode = {
            "converter_dict": converter_dict,
            "converter_kwargs": converter_kwargs,
            "decoder_dict": decoder_dict,
        }

        self.validation = validation_dict

    def adjust_schema(self, ds) -> dict:
        sections = deepcopy(self.schema["sections"])

        for section_name, section in sections.items():
            elements = section["elements"]
            schema_elements = self.schema["sections"][section_name]["elements"]
            spec_elements = self.order_specs[section_name]["elements"]

            for data_var, attrs in elements.items():

                if (
                    data_var not in ds.data_vars
                    and data_var not in ds.attrs
                    and data_var not in ds.dims
                ):
                    spec_elements[data_var]["ignore"] = True
                    schema_elements.pop(data_var, None)
                    continue

                for attr, value in list(attrs.items()):
                    if value != "__from_file__":
                        continue

                    ds_attrs = ds[data_var].attrs
                    if attr in ds_attrs:
                        schema_elements[data_var][attr] = ds_attrs[attr]
                    else:
                        schema_elements[data_var].pop(attr, None)

        return self.schema

    def _parse_line(self, line: str) -> dict:
        i = 0
        out = {}

        for order, spec in self.order_specs.items():
            header = spec.get("header")
            elements = spec.get("elements")
            is_delimited = spec.get("is_delimited")

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

    def parse_pandas(self, df, sections) -> pd.DataFrame:
        """Parse text lines into a pandas DataFrame."""
        self.sections = sections
        col = df.columns[0]
        records = df[col].map(self._parse_line)
        return pd.DataFrame.from_records(records)

    def parse_netcdf(self, ds, sections) -> pd.DataFrame:
        """Parse netcdf arrays into a pandas DataFrame."""

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

        for order, ospec in self.order_specs.items():
            header = ospec.get("header")
            disable_read = header.get("disable_read")
            if not _is_in_sections(order, sections):
                continue

            if disable_read is True:
                disables.append(order)
                continue

            elements = ospec.get("elements")
            for element, espec in elements.items():
                ignore = espec.get("ignore")
                index = espec.get("index")
                if ignore:
                    continue
                if element in ds.data_vars:
                    renames[element] = index
                elif element in ds.dims:
                    renames[element] = index
                elif element in ds.attrs:
                    attrs[index] = ds.attrs[element]
                else:
                    missing_values.append(index)

        df = ds[renames.keys()].to_dataframe().reset_index()
        df = df[renames.keys()]
        attrs = {k: v.replace("\n", "; ") for k, v in attrs.items()}
        df = df.rename(columns=renames)
        df = df.assign(**attrs)
        df[disables] = np.nan
        df = df.apply(lambda x: replace_empty_strings(x))
        df[missing_values] = False
        return df
