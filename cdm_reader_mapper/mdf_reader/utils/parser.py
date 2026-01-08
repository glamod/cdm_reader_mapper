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
    order_specs: dict
    disable_reads: list[str]
    dtypes: dict
    parse_dates: list[str]
    convert_decode: dict
    validation: dict
    encoding: str
    columns: pd.Index | pd.MultiIndex | None = None


def _get_index(section: str, order: str, length: int) -> str | tuple[str, str]:
    return section if length == 1 else (order, section)


def _get_ignore(section_dict: dict) -> bool:
    ignore = section_dict.get("ignore", False)
    if isinstance(ignore, str):
        ignore = ignore.lower() in {"true", "1", "yes"}
    return bool(ignore)


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
    sections: set | None,
    excludes: set,
    out: dict,
) -> int:
    section_length = header.get("length", properties.MAX_FULL_REPORT_WIDTH)
    delimiter = header.get("delimiter")
    sentinel = header.get("sentinel")

    section_end = i + section_length
    bad_sentinel = sentinel is not None and not line.startswith(sentinel, i)
    line_len = len(line)
    delim_len = len(delimiter) if delimiter else 0

    for spec in elements.values():
        field_length = spec.get("field_length", 0)
        index = spec.get("index")
        ignore = spec.get("ignore", False)
        missing_value = spec.get("missing_value")

        missing = True
        j = i if bad_sentinel else i + field_length
        if j > section_end:
            missing = False
            j = section_end

        if not ignore:
            key = index[0] if isinstance(index, tuple) else index
            if (sections is None or key in sections) and key not in excludes:
                if i < j:
                    value = line[i:j]
                    if not value.strip() or value == missing_value:
                        value = True
                else:
                    value = False if missing else True

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
    sections: set | None,
    excludes: set,
    out: dict,
) -> int:
    delimiter = header["delimiter"]
    fields = next(csv.reader([line[i:]], delimiter=delimiter))

    for element, value in zip_longest(elements.keys(), fields):
        index = elements[element]["index"]
        key = index[0] if isinstance(index, tuple) else index

        if (sections is None or key in sections) and key not in excludes:
            out[index] = value.strip() if value is not None else None

    return len(line)


def _parse_line_with_config(
    line: str,
    config: ParserConfig,
    sections: set | None,
    excludes: set,
) -> dict:
    i = 0
    out = {}
    max_width = properties.MAX_FULL_REPORT_WIDTH

    for order, spec in config.order_specs.items():
        header = spec["header"]
        elements = spec["elements"]

        if header.get("disable_read"):
            if order not in excludes:
                out[order] = line[i : i + max_width]
            i += header.get("length", max_width)
            continue

        if spec["is_delimited"]:
            i = _parse_delimited(line, i, header, elements, sections, excludes, out)
        else:
            i = _parse_fixed_width(line, i, header, elements, sections, excludes, out)

    return out


def parse_pandas(
    df: pd.DataFrame,
    config: ParserConfig,
    sections: list | None,
    excludes: list | None,
) -> pd.DataFrame:
    col = df.columns[0]

    sections = set(sections) if sections is not None else None
    excludes = set(excludes) if excludes else set()

    parse = _parse_line_with_config
    records = df[col].map(lambda line: parse(line, config, sections, excludes))
    return pd.DataFrame.from_records(records)


def parse_netcdf(
    ds: xr.Dataset,
    config: ParserConfig,
    sections: list | None,
    excludes: list | None,
) -> pd.DataFrame:
    sections = set(sections) if sections is not None else None
    excludes = set(excludes) if excludes else set()

    missing_values = []
    attrs = {}
    renames = {}
    disables = []

    data_vars = ds.data_vars
    dims = ds.dims
    ds_attrs = ds.attrs

    for order, ospec in config.order_specs.items():
        if sections is not None and order not in sections:
            continue
        if order in excludes:
            continue

        header = ospec.get("header", {})
        if header.get("disable_read") is True:
            disables.append(order)
            continue

        for element, espec in ospec.get("elements", {}).items():
            if espec.get("ignore"):
                continue

            index = espec["index"]

            if element in data_vars or element in dims:
                renames[element] = index
            elif element in ds_attrs:
                attrs[index] = ds_attrs[element]
            else:
                missing_values.append(index)

    df = ds[list(renames)].to_dataframe().reset_index()
    df = df[list(renames)].rename(columns=renames)

    if attrs:
        df = df.assign(**{k: v.replace("\n", "; ") for k, v in attrs.items()})

    if disables:
        df[disables] = np.nan

    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        s = df[col].str.decode("utf-8").str.strip()
        df[col] = s.map(lambda x: True if x == "" else x)

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

    orders = [
        order
        for group in schema["header"]["parsing_order"]
        for section_list in group.values()
        for order in section_list
    ]
    olength = len(orders)

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

        field_layout = header.get("field_layout") or (
            "delimited" if header.get("delimiter") else "fixed_width"
        )
        header = {**header, "field_layout": field_layout}

        elements = section.get("elements", {})

        if header.get("disable_read"):
            disable_reads.append(order)

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

            dtype = properties.pandas_dtypes.get(ctype)
            if dtype is not None:
                dtypes[index] = dtype

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

            validation[index] = {}
            if ctype:
                validation[index]["column_type"] = ctype
            for k in ("valid_min", "valid_max", "codetable"):
                if meta.get(k) is not None:
                    validation[index][k] = meta[k]

        order_specs[order] = {
            "header": header,
            "elements": element_specs,
            "is_delimited": header.get("format") == "delimited",
        }

    dtypes, parse_dates = convert_dtypes(dtypes)

    return ParserConfig(
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
