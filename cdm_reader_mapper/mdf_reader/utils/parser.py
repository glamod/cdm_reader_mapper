"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import csv
import logging

from dataclasses import dataclass, replace
from copy import deepcopy
from itertools import zip_longest

import numpy as np
import pandas as pd
import xarray as xr

from .. import properties
from ..schemas.schemas import read_schema, SchemaDict
from .utilities import convert_dtypes

from .convert_and_decode import Converters, Decoders


@dataclass(frozen=True)
class ParserConfig:
    imodel: str
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


def _parse_fixed_width(
    line: str,
    i: int,
    header: dict,
    elements: dict,
    sections: list | None,
    excludes: list | None,
    out: dict,
) -> int:
    section_length = header.get("length", properties.MAX_FULL_REPORT_WIDTH)
    delimiter = header.get("delimiter")
    sentinel = header.get("sentinel")

    bad_sentinel = sentinel is not None and not _validate_sentinel(i, line, sentinel)
    section_end = i + section_length

    line_len = len(line)
    delim_len = len(delimiter) if delimiter else 0

    for spec in elements.values():
        field_length = spec.get("field_length", 0)
        index = spec.get("index")

        j = i if bad_sentinel else i + field_length
        if j > section_end:
            j = section_end

        if (
            not spec.get("ignore")
            and _is_in_sections(index, sections)
            and not _is_in_sections(index, excludes)
        ):
            if i < j:
                value = line[i:j]
                if not value.strip() or value == spec.get("missing_value"):
                    value = True
            else:
                value = False

            out[index] = value

        if (
            delimiter
            and j + delim_len <= line_len
            and line[j : j + delim_len] == delimiter
        ):
            j += delim_len

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


def _parse_line_with_config(
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

        if spec["is_delimited"]:
            parse_func = _parse_delimited
        else:
            parse_func = _parse_fixed_width

        i = parse_func(
            line,
            i,
            header,
            elements,
            sections,
            excludes,
            out,
        )

    return out


def parse_pandas(
    df: pd.DataFrame, config: ParserConfig, sections: list | None, excludes: list | None
) -> pd.DataFrame:
    """Parse text lines into a pandas DataFrame."""
    col = df.columns[0]
    records = df[col].map(
        lambda line: _parse_line_with_config(
            line,
            config,
            sections,
            excludes,
        )
    )
    records = records.to_list()
    return pd.DataFrame.from_records(records)


def parse_netcdf(
    ds: xr.Dataset,
    config: ParserConfig,
    sections: list | None,
    excludes: list | None,
) -> pd.DataFrame:
    """Parse netcdf arrays into a pandas DataFrame."""

    def replace_empty_strings(series: pd.Series) -> pd.Series:
        if series.dtype == "object":
            series = series.str.decode("utf-8")
            series = series.str.strip()
            series = series.map(lambda x: True if x == "" else x)
        return series

    excludes = excludes or []

    missing_values = []
    attrs = {}
    renames = {}
    disables = []

    is_in_sections = _is_in_sections

    for order, ospec in config.order_specs.items():
        if not is_in_sections(order, sections):
            continue
        if is_in_sections(order, excludes):
            continue

        header = ospec.get("header", {})
        if header.get("disable_read") is True:
            disables.append(order)
            continue

        for element, espec in ospec.get("elements", {}).items():
            if espec.get("ignore"):
                continue

            index = espec.get("index")

            if element in ds.data_vars or element in ds.dims:
                renames[element] = index
            elif element in ds.attrs:
                attrs[index] = ds.attrs[element]
            else:
                missing_values.append(index)

    df = ds[renames.keys()].to_dataframe().reset_index()
    df = df[renames.keys()]

    df = df.rename(columns=renames)
    attrs = {k: v.replace("\n", "; ") for k, v in attrs.items()}
    df = df.assign(**attrs)

    if disables:
        df[disables] = np.nan

    df = df.apply(replace_empty_strings)

    if missing_values:
        df[missing_values] = False

    return df


def build_parser_config(
    imodel: str | None = None,
    ext_schema_path: str | None = None,
    ext_schema_file: str | None = None,
) -> ParserConfig:
    """Build ParserConfig from a normalized schema."""
    schema: SchemaDict = read_schema(
        imodel=imodel, ext_schema_path=ext_schema_path, ext_schema_file=ext_schema_file
    )

    # Flatten parsing order
    orders = [
        order
        for group in schema["header"]["parsing_order"]
        for section_list in group.values()
        for order in section_list
    ]
    olength = len(orders)

    # Initialize ParserConfig containers
    dtypes: dict = {}
    validation: dict = {}
    order_specs: dict = {}
    disable_reads: list[str] = []
    converters: dict = {}
    converter_kwargs: dict = {}
    decoders: dict = {}

    for order in orders:
        section = schema["sections"][order]
        header = section["header"]

        # Normalize field_layout in-place
        field_layout = header.get("field_layout") or (
            "delimited" if header.get("delimiter") else "fixed_width"
        )
        header = {**header, "field_layout": field_layout}

        elements = section.get("elements", {})

        if header.get("disable_read"):
            disable_reads.append(order)

        # Build element specs
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
                "field_length": meta.get(
                    "field_length", properties.MAX_FULL_REPORT_WIDTH
                ),
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
                converters[index] = conv_func
            conv_args = {
                k: meta.get(k)
                for k in properties.data_type_conversion_args.get(ctype, [])
            }
            if conv_args:
                converter_kwargs[index] = conv_args
            encoding = meta.get("encoding")
            if encoding:
                dec_func = Decoders(ctype, encoding).decoder()
                if dec_func:
                    decoders[index] = dec_func

            # Validation
            validation[index] = {}
            if ctype:
                validation[index]["column_type"] = ctype
            for k in ("valid_min", "valid_max", "codetable"):
                if meta.get(k) is not None:
                    validation[index][k] = meta[k]

        # Save section config
        order_specs[order] = {
            "header": header,
            "elements": element_specs,
            "is_delimited": header.get("format") == "delimited",
        }

    # Convert dtypes & parse_dates
    dtypes, parse_dates = convert_dtypes(dtypes)

    return ParserConfig(
        imodel=schema.get("imodel"),
        order_specs=order_specs,
        disable_reads=disable_reads,
        dtypes=dtypes,
        parse_dates=parse_dates,
        convert_decode={
            "converter_dict": converters,
            "converter_kwargs": converter_kwargs,
            "decoder_dict": decoders,
        },
        validation=validation,
        encoding=schema["header"].get("encoding", "utf-8"),
    )


def update_xr_config(ds: xr.Dataset, config: ParserConfig) -> ParserConfig:
    new_order_specs = deepcopy(config.order_specs)
    new_validation = deepcopy(config.validation)

    for order, ospecs in new_order_specs.items():
        elements = ospecs["elements"]

        for element, especs in elements.items():
            if (
                element not in ds.data_vars
                and element not in ds.attrs
                and element not in ds.dims
            ):
                especs["ignore"] = True
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
        config,
        order_specs=new_order_specs,
        validation=new_validation,
    )


def update_pd_config(pd_kwargs: dict, config: ParserConfig) -> ParserConfig:
    if "encoding" in pd_kwargs and pd_kwargs["encoding"]:
        return replace(config, encoding=pd_kwargs["encoding"])
    return config
