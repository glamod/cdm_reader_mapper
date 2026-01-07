"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import csv
import logging

from dataclasses import dataclass, replace
from copy import deepcopy
from itertools import zip_longest

import numpy as np
import pandas as pd

from .. import properties
from ..schemas import schemas
from .utilities import convert_dtypes

from .convert_and_decode import Converters, Decoders

@dataclass(frozen=True)
class ParserConfig:
    imodel: str
    orders: list[str]
    order_specs: dict
    disable_reads: list[str]
    dtypes: dict
    parse_dates: list[str]
    convert_decode: dict
    validation: dict
    encoding: str
    columns: pd.Index | pd.MultiIndex | None = None

def _validate_sentinel(i: int, line: str, sentinel: str) -> bool:
    return line.startswith(sentinel, i)


def _get_index(section: str, order: str, length: int) -> str | tuple[str, str]:
    if length == 1:
        return section
    return (order, section)


def _get_ignore(section_dict: dict) -> bool:
    ignore = section_dict.get("ignore", False)
    if isinstance(ignore, str):
        ignore = ignore.lower() in {"true", "1", "yes"}
    return bool(ignore)


def _is_in_sections(index: str | tuple, sections: list | None) -> bool:
    if sections is None:
        return True
    key = index[0] if isinstance(index, tuple) else index
    return key in sections


def _convert_dtype_to_default(dtype: str | None) -> str | None:
    if dtype is None:
        return None
    elif dtype == "float":
        return dtype
    elif dtype == "int":
        return properties.pandas_int
    elif "float" in dtype.lower():
        logging.warning(f"Set column type from deprecated {dtype} to float.")
        return "float"
    elif "int" in dtype.lower():
        logging.warning(f"Set column type from deprecated {dtype} to int.")
        return properties.pandas_int
    return dtype

def _build_element_specs(
    order: str,
    olength: int,
    elements: dict,
    dtypes: dict,
    validation_dict: dict,
    converter_dict: dict,
    converter_kwargs: dict,
    decoder_dict: dict,
) -> dict:
    """Build specs for all elements in a section and update related dicts."""
    element_specs = {}

    for name, meta in elements.items():
        index = _get_index(name, order, olength)
        ignore = _get_ignore(meta)
        ctype = _convert_dtype_to_default(meta.get("column_type"))

        element_specs[name] = {
            "index": index,
            "ignore": ignore,
            "column_type": ctype,
            "missing_value": meta.get("missing_value"),
            "field_length": meta.get("field_length", properties.MAX_FULL_REPORT_WIDTH),
        }

        if ignore or meta.get("disable_read", False):
            continue

        # Pandas dtype
        dtype = properties.pandas_dtypes.get(ctype)
        if dtype is not None:
            dtypes[index] = dtype

        # Conversion & decoding
        conv_func = Converters(ctype).converter()
        if conv_func:
            converter_dict[index] = conv_func
        conv_kwargs = {k: meta.get(k) for k in properties.data_type_conversion_args.get(ctype, [])}
        if conv_kwargs:
            converter_kwargs[index] = conv_kwargs
        encoding = meta.get("encoding")
        if encoding:
            dec_func = Decoders(ctype, encoding).decoder()
            if dec_func:
                decoder_dict[index] = dec_func
                
        # Validation
        validation_dict[index] = {}
        if ctype:
            validation_dict[index]["column_type"] = ctype
        for k in ("valid_min", "valid_max", "codetable"):
            if meta.get(k) is not None:
                validation_dict[index][k] = meta[k]                

    return element_specs


def _parse_fixed_width(
    line: str,
    i: int,
    header: dict,
    elements: dict,
    sections: list,
    excludes: list,
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

        if (
            not ignore
            and _is_in_sections(index, sections)
            and not _is_in_sections(index, excludes)
        ):
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
    excludes: list,
    out: dict,
) -> int:
    delimiter = header["delimiter"]
    fields = next(csv.reader([line[i:]], delimiter=delimiter))

    for element, value in zip_longest(elements.keys(), fields):
        index = elements[element].get("index")
        if _is_in_sections(index, sections) and not _is_in_sections(index, excludes):
            out[index] = value.strip() if value is not None else None

    return len(line)


def parse_line(*args, is_delimited: bool) -> int:
    if is_delimited:
        return _parse_delimited(*args)
    return _parse_fixed_width(*args)

def parse_line_with_config(
    line: str,
    config: ParserConfig,
    sections: list | None,
    excludes: list | None,
) -> dict:
    i = 0
    out = {}
    excludes = excludes or []

    for order, spec in config.order_specs.items():
        header = spec["header"]
        elements = spec["elements"]

        if header.get("disable_read"):
            if order in excludes:
                continue
            out[order] = line[i : properties.MAX_FULL_REPORT_WIDTH]
            continue

        i = parse_line(
            line,
            i,
            header,
            elements,
            sections,
            excludes,
            out,
            is_delimited=spec["is_delimited"],
        )

    return out


class Parser:

    def __init__(self, imodel: str | None, ext_schema_path: str | None, ext_schema_file: str | None):

        self.imodel = imodel

        logging.info("READING DATA MODEL SCHEMA FILE...")
        schema = schemas.read_schema(
            imodel=imodel,
            ext_schema_path=ext_schema_path,
            ext_schema_file=ext_schema_file,
        )
        self.schema = schema
        self.config = self._build_config(schema)
        
    def _build_config(self, schema: dict) -> ParserConfig:
      """Build a ParserConfig from a schema."""
      # Parsing order
      parsing_order = schema["header"].get("parsing_order", [])
      sections = [x.get(y) for x in parsing_order for y in x]
      orders = [y for x in sections for y in x]

      # Initialize dicts
      dtypes = {}
      converter_dict = {}
      converter_kwargs = {}
      decoder_dict = {}
      validation_dict = {}
      order_specs = {}
      disable_reads = []

      olength = len(orders)
      for order in orders:
        section = schema["sections"][order]
        header = section["header"]
        elements = section.get("elements", {})

        if header.get("disable_read", False):
            disable_reads.append(order)

        if not header.get("field_layout"):
            header["field_layout"] = "delimited" if header.get("delimiter") else "fixed_width"

        element_specs = _build_element_specs(
            order,
            olength,
            elements,
            dtypes,
            validation_dict,
            converter_dict,
            converter_kwargs,
            decoder_dict,
        )

        order_specs[order] = {
            "header": header,
            "elements": element_specs,
            "is_delimited": header.get("format") == "delimited",
        }

      encoding = schema["header"].get("encoding", "utf-8")
      dtypes, parse_dates = convert_dtypes(dtypes)

      convert_decode = {
        "converter_dict": converter_dict,
        "converter_kwargs": converter_kwargs,
        "decoder_dict": decoder_dict,
      }

      return ParserConfig(
        imodel=schema.get("imodel"),
        orders=orders,
        order_specs=order_specs,
        disable_reads=disable_reads,
        dtypes=dtypes,
        parse_dates=parse_dates,
        convert_decode=convert_decode,
        validation=validation_dict,
        encoding=encoding,
      )        

    def update_xr_config(self, ds: xr.Dataset) -> ParserConfig:
        new_order_specs = deepcopy(self.config.order_specs)
        new_validation = deepcopy(self.config.validation)
        for order, ospecs in list(self.config.order_specs.items()):
            elements = ospecs["elements"]
            
            for element, especs in elements.items():
                if (
                    element not in ds.data_vars
                    and element not in ds.attrs
                    and element not in ds.dims
                ):
                    elements[element]["ignore"] = True
                    continue

                index = especs.get("index")
                if index not in new_validation:
                    continue

                for attr in list(new_validation[index].keys()):
                    if new_validation[index][attr] != "__from_file__":
                        continue

                    ds_attrs = ds[element].attrs
                    if attr in ds_attrs:
                        new_validation[index][attr] = ds_attrs[attr]
                    else:
                        new_validation[index].pop(attr, None)
                        
        return replace(
            self.config,
            order_specs=new_order_specs,
            validation=new_validation,
        )

        
    def update_pd_config(self, pd_kwargs: dict) -> ParserConfig:
        if "encoding" in pd_kwargs and pd_kwargs["encoding"]:
            return replace(self.config, encoding=pd_kwargs["encoding"])
        return self.config

    def parse_pandas(self, df: pd.DataFrame, sections: list | None, excludes: list | None) -> pd.DataFrame:
        """Parse text lines into a pandas DataFrame."""
        col = df.columns[0]
        records = df[col].map(
            lambda line: parse_line_with_config(
                line,
                self.config,
                sections,
                excludes,
            )
        )
        records = records.to_list()
        return pd.DataFrame.from_records(records)

    def parse_netcdf(self, ds: xr.Dataset, sections: list | None, excludes: list | None) -> pd.DataFrame:
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

        excludes = excludes or []

        for order, ospec in self.config.order_specs.items():
            header = ospec.get("header")
            disable_read = header.get("disable_read")
            if not _is_in_sections(order, sections):
                continue
            if _is_in_sections(order, excludes):
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
