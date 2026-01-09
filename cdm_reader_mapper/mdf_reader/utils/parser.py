"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import csv
import logging

from dataclasses import dataclass, replace
from copy import deepcopy
from itertools import zip_longest
from typing import TypedDict, Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr

from .. import properties
from ..schemas.schemas import read_schema, SchemaDict
from .utilities import convert_dtypes

from .convert_and_decode import Converters, Decoders


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

    Parameters
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

    order_specs: OrderSpec
    disable_reads: list[str]
    dtypes: dict
    parse_dates: list[str]
    convert_decode: dict
    validation: dict
    encoding: str
    columns: pd.Index | pd.MultiIndex | None = None


def _get_index(section: str, order: str, length: int) -> str | tuple[str, str]:
    """Build an index key based on section count."""
    return section if length == 1 else (order, section)


def _get_ignore(section_dict: dict[str, Any]) -> bool:
    """Determine whether a section should be ignored."""
    ignore = section_dict.get("ignore", False)
    if isinstance(ignore, str):
        ignore = ignore.lower() in {"true", "1", "yes"}
    return bool(ignore)


def _convert_dtype_to_default(dtype: str | None) -> str | None:
    """Normalize deprecated or aliased dtype strings."""
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
    header: dict[str, Any],
    elements: dict[str, dict[str, Any]],
    sections: set | None,
    excludes: set,
    out: dict[Any, Any],
) -> int:
    """Parse a fixed-width section of a line into an output dictionary."""
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
    header: dict[str, Any],
    elements: dict[str, dict[str, Any]],
    sections: set | None,
    excludes: set,
    out: dict[Any, Any],
) -> int:
    """Parse a delimiter-separated section of a line into an output dictionary."""
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
    sections: set | None,
    excludes: set,
) -> dict[str, dict[Any, Any]]:
    """Parse a line using the provided parser configuration."""
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

    Notes
    -----
    - Ignored elements (``ignore=True``) are skipped.
    - Disabled sections (``disable_read=True``) are included as raw strings in the output.
    - Missing elements are filled with ``False``.
    - Object-type columns are stripped, decoded from UTF-8 if necessary, and empty
      strings are replaced with ``True``.
    - No type conversion is performed at this stage.
    """
    col = df.columns[0]

    sections = set(sections) if sections is not None else None
    excludes = set(excludes) if excludes else set()

    records = df[col].map(
        lambda line: _parse_line(line, order_specs, sections, excludes)
    )
    return pd.DataFrame.from_records(records)


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

    Notes
    -----
    - Variables, dimensions, and global attributes in `ds` are mapped to columns
      according to the element `index`.
    - Ignored elements (`ignore=True`) are skipped.
    - Disabled sections (`disable_read=True`) are added as columns filled with NaN.
    - Missing elements are added as columns filled with False.
    - Object-type columns are decoded from UTF-8, stripped, and empty strings
      replaced with True.
    """
    sections = set(sections) if sections is not None else None
    excludes = set(excludes) if excludes else set()

    missing_values = []
    attrs = {}
    renames = {}
    disables = []

    data_vars = ds.data_vars
    dims = ds.dims
    ds_attrs = ds.attrs

    for order, ospec in order_specs.items():
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
        print(df[col])
        print(df[col].str)
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
    ext_schema_path : str or None, optional
        Path to an external schema directory.
    ext_schema_file : str or None, optional
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

    orders = [
        order
        for group in schema["header"]["parsing_order"]
        for section_list in group.values()
        for order in section_list
    ]
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
                "field_length": meta.get(
                    "field_length", properties.MAX_FULL_REPORT_WIDTH
                ),
            }

            if ignore or meta.get("disable_read", False):
                continue

            ctype = _convert_dtype_to_default(meta.get("column_type"))
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
