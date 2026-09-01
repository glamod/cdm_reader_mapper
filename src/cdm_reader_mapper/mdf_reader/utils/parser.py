"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations
import csv
import logging
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import zip_longest
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import xarray as xr

from .. import properties
from ..schemas.schemas import SchemaDict, read_schema
from .convert_and_decode import Converters, Decoders
from .utilities import convert_dtypes


class OrderSpec(TypedDict):
    """
    Parsing specification for a single section.

    Defines the header configuration, element layout, and parsing mode
    (fixed-width or delimited) for a section.
    """

    header: dict[str, Any]
    elements: dict[str, dict[str, Any]]
    is_delimited: bool


@dataclass(frozen=True)
class ParserConfig:
    """
    Configuration for dataset parsing.

    Attributes
    ----------
    order_specs : dict
        Column ordering specifications.
    disable_reads : list[str]
        Columns or sources to skip during parsing.
    dtypes : dict
        Column data type mappings.
    parse_dates : list[str]
        Columns to parse as datetimes.
    convert_decode : dict
        Value conversion or decoding rules.
    validation : dict
        Validation rules for parsed data.
    encoding : str
        Text encoding used when reading input data.
    columns : pd.Index or pd.MultiIndex or None, optional
        Explicit column index to apply. If None, inferred from input.
    """

    order_specs: dict[str, OrderSpec]
    disable_reads: list[str]
    dtypes: dict[Any, Any]
    parse_dates: list[str]
    convert_decode: dict[Any, Any]
    validation: dict[Any, Any]
    encoding: str
    columns: pd.Index | pd.MultiIndex | None = None


def _get_index(section: str, order: str, length: int) -> str | tuple[str, str]:
    """
    Build an index key based on section count.

    Parameters
    ----------
    section : str
        Name of the section being indexed.
    order : str
        Order identifier used when multiple sections exist.
    length : int
        Number of elements in the section group.

    Returns
    -------
    str or tuple of str and str
        If `length == 1`, returns `section`.
        Otherwise returns a tuple `(order, section)`.
    """
    return section if length == 1 else (order, section)


def _get_ignore(section_dict: dict[str, Any]) -> bool:
    """
    Determine whether a section should be ignored.

    Parameters
    ----------
    section_dict : dict
        Configuration dictionary for a section. May contain an "ignore"
        key as a boolean or string representation of a boolean.

    Returns
    -------
    bool
        True if the section should be ignored, otherwise False.
    """
    ignore = section_dict.get("ignore", False)
    if isinstance(ignore, str):
        ignore = ignore.lower() in {"true", "1", "yes"}
    return bool(ignore)


def _convert_dtype_to_default(dtype: str | None) -> str | None:
    """
    Normalize deprecated or aliased dtype strings.

    Parameters
    ----------
    dtype : str or None
        Input dtype specification, possibly deprecated or aliased.

    Returns
    -------
    str or None
        Normalized dtype string. May map deprecated float/int aliases to
        standardized internal representations.

    Notes
    -----
    Logs a warning when deprecated dtype formats are converted.
    """
    if dtype is None:
        return None
    elif dtype == "float":
        return dtype
    elif dtype == "int":
        return properties.pandas_int
    elif "float" in dtype.lower():
        logging.warning("Set column type from deprecated %s to float.", dtype)
        return "float"
    elif "int" in dtype.lower():
        logging.warning("Set column type from deprecated %s to int.", dtype)
        return properties.pandas_int
    return dtype


def _parse_fixed_width(
    line: str,
    i: int,
    header: dict[str, Any],
    elements: dict[str, dict[str, Any]],
    sections: set[str] | None,
    excludes: set[str],
    out: dict[Any, Any],
) -> int:
    """
    Parse a fixed-width section of a line into an output dictionary.

    Parameters
    ----------
    line : str
        Input line to parse.
    i : int
        Current parsing position in the line.
    header : dict
        Section header metadata including length, delimiter, and sentinel.
    elements : dict
        Field definitions for the section.
    sections : set of str or None
        Optional subset of sections to include in parsing.
    excludes : set of str
        Section keys to exclude from parsing.
    out : dict
        Output dictionary to populate with parsed values.

    Returns
    -------
    int
        Updated index position after parsing the section.
    """
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
                value: str | bool
                if i < j:
                    value = line[i:j]
                    if not value.strip() or value == missing_value:
                        value = True
                else:
                    value = False if missing else True

                out[index] = value

        if delimiter and j + delim_len <= line_len and line[j : j + delim_len] == delimiter:
            j += delim_len

        i = j

    return i


def _parse_delimited(
    line: str,
    i: int,
    header: dict[str, Any],
    elements: dict[str, dict[str, Any]],
    sections: set[str] | None,
    excludes: set[str],
    out: dict[Any, Any],
) -> int:
    """
    Parse a delimiter-separated section of a line into an output dictionary.

    Parameters
    ----------
    line : str
        Input line to parse.
    i : int
        Current parsing position in the line.
    header : dict
        Header metadata including delimiter definition.
    elements : dict
        Field definitions for the section.
    sections : set of str or None
        Optional subset of sections to include.
    excludes : set of str
        Section keys to exclude from parsing.
    out : dict
        Output dictionary to populate with parsed values.

    Returns
    -------
    int
        Final position in the line after parsing (typically end of line).
    """
    delimiter = header["delimiter"]
    fields = next(csv.reader([line[i:]], delimiter=delimiter))

    for element, value in zip_longest(elements.keys(), fields):
        index = elements[element]["index"]
        key = index[0] if isinstance(index, tuple) else index

        if (sections is None or key in sections) and key not in excludes:
            out[index] = value.strip() if value is not None else None

    return len(line)


def _parse_line(
    line: str,
    order_specs: dict[str, OrderSpec],
    sections: set[str] | None,
    excludes: set[str],
) -> dict[str, str]:
    """
    Parse a line using the provided parser configuration.

    Parameters
    ----------
    line : str
        Input line to parse.
    order_specs : dict
        Ordered specification of parsing rules for each section.
    sections : set of str or None
        Optional subset of sections to include in parsing.
    excludes : set of str
        Section keys to exclude from parsing.

    Returns
    -------
    dict
        Dictionary mapping parsed section keys to extracted values.
    """
    i = 0
    out = {}
    max_width = properties.MAX_FULL_REPORT_WIDTH

    for order, spec in order_specs.items():
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
    order_specs: dict[str, OrderSpec],
    sections: Iterable[str] | None = None,
    excludes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Parse a pandas DataFrame containing raw record lines.

    Each row of the input DataFrame is expected to contain a single
    fixed-width or delimiter-separated record, which is parsed according
    to the provided order specifications.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with exactly one column (column index ``0``),
        where each row contains a raw record string.
    order_specs : dict[str, OrderSpec]
        Mapping of section names to parsing specifications. Each specification
        defines the header configuration, element layout, and parsing mode
        for a section.
    sections : iterable of str or None
        Section names to include. If None, all sections are parsed.
    excludes : iterable of str or None
        Section names to exclude from parsing.

    Returns
    -------
    pandas.DataFrame
        DataFrame constructed from parsed records. Columns are derived
        from element indices and may be strings or tuples.

    Notes
    -----
    - Ignored elements (``ignore=True``) are skipped.
    - Disabled sections (``disable_read=True``) are included as raw strings in the output.
    - Missing elements are filled with ``False``.
    - Object-type columns are stripped, decoded from UTF-8 if necessary, and empty
      strings are replaced with ``True``.
    - No type conversion is performed at this stage.

    Examples
    --------
    Example ``order_specs`` structure::

        order_specs = {
            "core": {
                "header": {
                    "sentinel": None,
                    "length": 108,
                },
                "elements": {
                    "YR": {
                        "index": ("core", "YR"),
                        "field_length": 4,
                        "ignore": False,
                        "column_type": "Int64",
                        "missing_value": None,
                    },
                    "MO": {
                        "index": ("core", "MO"),
                        "field_length": 2,
                        "ignore": False,
                        "column_type": "Int64",
                        "missing_value": None,
                    },
                },
                "is_delimited": False,
            }
        }
    """
    col = df.columns[0]

    sections = set(sections) if sections is not None else None
    excludes = set(excludes) if excludes else set()

    records = df[col].map(lambda line: _parse_line(line, order_specs, sections, excludes))
    return pd.DataFrame.from_records(records.to_list(), index=records.keys())


def parse_netcdf(
    ds: xr.Dataset,
    order_specs: dict[str, OrderSpec],
    sections: Iterable[str] | None = None,
    excludes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Parse an xarray Dataset into a pandas DataFrame based on order specifications.

    This function converts an xarray Dataset into a tabular pandas DataFrame
    according to parsing rules defined in `order_specs`. Data variables, dimensions,
    and global attributes are mapped to columns as specified, with ignored or missing
    elements handled automatically.

    Parameters
    ----------
    ds : xarray.Dataset
        Input Dataset containing data variables, dimensions, and attributes.
    order_specs : dict[str, OrderSpec]
        Mapping of section names to parsing specifications. Each specification
        defines the header configuration, element layout, and parsing mode
        for a section.
    sections : iterable of str or None
        Section names to include. If None, all sections are parsed.
    excludes : iterable of str or None
        Section names to exclude from parsing.

    Returns
    -------
    pandas.DataFrame
        DataFrame constructed from the Dataset according to the parsing specification.
        Columns are derived from element indices. Missing fields are filled with
        False, disabled sections with NaN, and empty strings are converted to True.

    Notes
    -----
    - Variables, dimensions, and global attributes in `ds` are mapped to columns
      according to the element `index`.
    - Ignored elements (`ignore=True`) are skipped.
    - Disabled sections (`disable_read=True`) are added as columns filled with NaN.
    - Missing elements are added as columns filled with False.
    - Object-type columns are decoded from UTF-8, stripped, and empty strings
      replaced with True.

    Examples
    --------
    Example ``order_specs`` structure::

        order_specs = {
            "global_attributes": {
                "header": {
                    "disable_read": True,
                },
                "elements": {
                    "title": {
                        "index": ("global_attributes", "title"),
                        "ignore": False,
                        "column_type": "str",
                        "missing_value": None,
                    },
                    "institution": {
                        "index": ("global_attributes", "institution"),
                        "ignore": False,
                        "column_type": "str",
                        "missing_value": None,
                    },
                },
                "is_delimited": False,
            }
        }
    """
    sections = set(sections) if sections is not None else None
    excludes = set(excludes) if excludes else set()

    missing_values = []
    attrs = {}
    renames = {}
    disables = []

    data_vars = ds.data_vars
    dims = ds.dims
    coords = ds.coords
    ds_attrs = ds.attrs

    for order, ospec in order_specs.items():
        if order in excludes or (sections is not None and order not in sections):
            continue

        header = ospec.get("header", {})
        if header.get("disable_read") is True:
            disables.append(order)
            continue

        for element, espec in ospec.get("elements", {}).items():
            if espec.get("ignore"):
                continue

            index = espec["index"]

            if element in data_vars or element in dims or element in coords:
                renames[element] = index
            elif element in ds_attrs:
                attrs[index] = ds_attrs[element]
            else:
                missing_values.append(index)

    df = ds[list(renames)].to_dataframe().reset_index()
    df = df[list(renames)].rename(columns=renames)

    if disables:
        df[disables] = np.nan

    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        s = df[col].str.decode("utf-8").str.strip()
        df[col] = s.map(lambda x: True if x == "" else x)

    for k, v in attrs.items():
        df[k] = v.replace("\n", "; ")

    if missing_values:
        df[missing_values] = False

    return df


def build_parser_config(
    imodel: str | None = None,
    ext_schema_path: str | Path | None = None,
    ext_schema_file: str | Path | None = None,
) -> ParserConfig:
    """
    Build a ParserConfig from a normalized schema definition.

    This function reads a schema definition and constructs a fully populated
    :py:class:`ParserConfig` instance. The resulting configuration contains
    parsing order specifications, data types, converters, decoders, validation
    rules, and encoding information required to parse raw input records.

    Parameters
    ----------
    imodel : str or None, optional
        Internal model identifier used to locate the schema.
    ext_schema_path : str or Path, optional
        Path to an external schema directory.
    ext_schema_file : str or Path, optional
        Filename of an external schema definition.

    Returns
    -------
    ParserConfig
        Fully initialized parser configuration derived from the schema.

    Notes
    -----
    - Section parsing order is derived from ``schema["header"]["parsing_order"]``.
    - Sections marked with ``disable_read=True`` are recorded in
      ``ParserConfig.disable_reads``.
    - Elements marked as ignored or disabled are excluded from dtype,
      conversion, and validation setup.
    - Column indices may be strings or tuples depending on the number of
      sections in the schema.
    - Deprecated or aliased column types are normalized via
      ``_convert_dtype_to_default``.
    - Converter and decoder functions are resolved dynamically based on
      column type and encoding.
    - Validation rules may include value ranges and code tables, as defined
      in the schema.
    """
    schema: SchemaDict = read_schema(
        imodel=imodel,
        ext_schema_path=ext_schema_path,
        ext_schema_file=ext_schema_file,
    )

    orders = [order for group in schema["header"]["parsing_order"] for section_list in group.values() for order in section_list]
    olength = len(orders)

    dtypes: dict[Any, Any] = {}
    validation: dict[Any, dict[str, Any]] = {}
    order_specs: dict[str, OrderSpec] = {}
    disable_reads: list[str] = []
    converters: dict[Any, Any] = {}
    converter_kwargs: dict[Any, dict[str, Any]] = {}
    decoders: dict[Any, Any] = {}

    for order in orders:
        section = schema["sections"][order]
        header = section["header"]
        elements = section.get("elements", {})

        if header.get("disable_read"):
            disable_reads.append(order)

        element_specs: dict[str, dict[str, Any]] = {}
        for name, meta in elements.items():
            index = _get_index(name, order, olength)
            ignore = _get_ignore(meta)

            element_specs[name] = {
                "index": index,
                "ignore": ignore,
                "missing_value": meta.get("missing_value"),
                "field_length": meta.get("field_length", properties.MAX_FULL_REPORT_WIDTH),
            }

            if ignore or meta.get("disable_read", False):
                continue

            ctype = _convert_dtype_to_default(meta.get("column_type"))
            if ctype is None:
                continue

            dtype = properties.pandas_dtypes.get(ctype)
            if dtype is not None:
                dtypes[index] = dtype

            converters[index] = Converters(ctype).converter()

            conv_args = {k: meta.get(k) for k in properties.data_type_conversion_args.get(ctype, [])}
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

        order_specs[order] = OrderSpec(
            header=header,
            elements=element_specs,
            is_delimited=header.get("format") == "delimited",
        )

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
    """
    Update a ParserConfig instance using metadata from an xarray Dataset.

    This function adjusts the parser configuration based on the contents of
    the provided Dataset. Elements not present in the Dataset are marked as
    ignored, and validation rules marked as ``"__from_file__"`` are populated
    from Dataset variable attributes when available.

    Parameters
    ----------
    ds : xarray.Dataset
        Input Dataset containing data variables, dimensions, and attributes.
    config : ParserConfig
        Existing parser configuration.

    Returns
    -------
    ParserConfig
        Updated parser configuration with modified order specifications and
        validation rules derived from the Dataset.
    """
    new_order_specs = deepcopy(config.order_specs)
    new_validation = deepcopy(config.validation)

    for ospecs in new_order_specs.values():
        elements = ospecs["elements"]

        for element, especs in elements.items():
            if element not in ds.data_vars and element not in ds.attrs and element not in ds.dims and element not in ds.coords:
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


def update_pd_config(pd_kwargs: dict[str, Any], config: ParserConfig) -> ParserConfig:
    """
    Update a ParserConfig instance using pandas keyword arguments.

    Currently, only the ``encoding`` option is supported. If an encoding
    is provided in ``pd_kwargs``, a new ParserConfig instance is returned
    with the updated encoding. Otherwise, the original configuration is
    returned unchanged.

    Parameters
    ----------
    pd_kwargs : dict[str, Any]
        Keyword arguments intended for pandas I/O functions.
    config : ParserConfig
        Existing parser configuration.

    Returns
    -------
    ParserConfig
        Updated parser configuration if applicable, otherwise the original
        configuration.
    """
    if "encoding" in pd_kwargs and pd_kwargs["encoding"]:
        return replace(config, encoding=pd_kwargs["encoding"])
    return config
