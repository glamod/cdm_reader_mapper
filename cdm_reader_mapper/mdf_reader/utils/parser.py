"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import ast
import csv
import logging

from itertools import zip_longest

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


def parse_fixed_width(
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


def parse_delimited(
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


class Parser:

    def __init__(self, imodel, ext_schema_path, ext_schema_file):
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

        self.encoding = self.schema["header"].get("encoding", "utf-8")

        self.dtypes, self.parse_dates = convert_dtypes(dtypes)

        self.disable_reads = disable_reads

        self.convert_decode = {
            "converter_dict": converter_dict,
            "converter_kwargs": converter_kwargs,
            "decoder_dict": decoder_dict,
        }

        self.compiled_specs = compiled_specs
