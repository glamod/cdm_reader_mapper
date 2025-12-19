"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import ast
import csv
import logging

from copy import deepcopy
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


def _is_in_sections(index, sections):
    if not sections:
        return True
    elif isinstance(index, tuple):
        return index[0] in sections
    return index in sections


def _element_specs(
    order, olength, elements, converter_dict, converter_kwargs, decoder_dict, dtypes
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

        ctype = meta.get("column_type")
        dtype = properties.pandas_dtypes.get(ctype)

        if dtype:
            dtypes[index] = dtype

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

    for index, value in zip_longest(elements.keys(), fields):
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
        logging.info("READING DATA MODEL SCHEMA FILE...")
        if ext_schema_path or ext_schema_file:
            self.schema = schemas.read_schema(
                ext_schema_path=ext_schema_path, ext_schema_file=ext_schema_file
            )
        else:
            self.schema = schemas.read_schema(imodel=imodel)

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

        self.order_specs, self.disable_reads = _order_specs(
            self.orders,
            self.schema["sections"],
            converter_dict,
            converter_kwargs,
            decoder_dict,
            dtypes,
        )

        self.encoding = self.schema["header"].get("encoding", "utf-8")

        self.dtypes, self.parse_dates = convert_dtypes(dtypes)

        self.convert_decode = {
            "converter_dict": converter_dict,
            "converter_kwargs": converter_kwargs,
            "decoder_dict": decoder_dict,
        }

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
