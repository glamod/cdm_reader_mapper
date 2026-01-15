"""
Manage data model schema files.

Functions to manage data model
schema files and objects according to the
requirements of the data reader tool

"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from cdm_reader_mapper.common.json_dict import collect_json_files, combine_dicts

from .. import properties


class SectionDict(TypedDict, total=False):
    """
    Schema definition for a single section within a report.

    Attributes
    ----------
    header : dict, optional
        Metadata or configuration for the section header.
    elements : dict, optional
        Dictionary of elements/fields contained within the section.
    """

    header: dict
    elements: dict


class SchemaHeaderDict(TypedDict, total=False):
    """
    Schema definition for the report header.

    Attributes
    ----------
    parsing_order : list[dict], optional
        List of dictionaries defining the order in which header fields are parsed.
    delimiter : str, optional
        Delimiter used to separate fields in the header.
    field_layout : str, optional
        Layout or format of the fields (e.g., fixed width, CSV).
    format : str, optional
        General format type of the header.
    encoding : str, optional
        Text encoding for the header, e.g., 'utf-8'.
    multiple_reports_per_line : bool, optional
        Whether multiple reports may appear on a single line.
    """

    parsing_order: list[dict]
    delimiter: str
    field_layout: str
    format: str
    encoding: str
    multiple_reports_per_line: bool


class SchemaDict(TypedDict, total=False):
    """
    Complete schema definition for a report.

    Attributes
    ----------
    header : SchemaHeaderDict, optional
        Configuration for the report header.
    sections : dict[str, SectionDict], optional
        Mapping of section names to section schemas.
    elements : dict, optional
        Mapping of element names to their attributes.
    name : list[Path], optional
        List of Path objects representing schema files or sources.
    imodel : str | None, optional
        Name of the internal data model, if applicable.
    """

    header: SchemaHeaderDict
    sections: dict[str, SectionDict]
    elements: dict
    name: list[Path]
    imodel: str | None


def _resolve_schema_files(
    *,
    imodel: str | None = None,
    ext_schema_path: str | None = None,
    ext_schema_file: str | None = None,
) -> list[Path]:
    """Determine which schema file(s) to use based on the input parameters."""
    if ext_schema_file:
        path = Path(ext_schema_file)
        if not path.is_file():
            raise FileNotFoundError(f"Can't find input schema file {ext_schema_file}")
        return [path]

    if ext_schema_path:
        schema_path = Path(ext_schema_path).resolve()
        path = schema_path / f"{schema_path.name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Can't find input schema path {ext_schema_path}")
        return [path]

    if imodel:
        parts = imodel.split("_")
        model = parts[0]
        if model not in properties.supported_data_models:
            raise ValueError(f"Input data model {model} not supported")

        return collect_json_files(*parts, base=f"{properties._base}.schemas")

    raise ValueError(
        "One of 'imodel', 'ext_schema_path', or 'ext_schema_file' must be set"
    )


def _normalize_schema(schema: SchemaDict) -> SchemaDict:
    """Normalize a schema dictionary by ensuring it has sections and a parsing order."""
    header = schema.get("header", {})
    sections = schema.get("sections")
    elements = schema.get("elements")

    if not sections:
        if not elements:
            raise KeyError("Schema has no sections and no elements")
        level = properties.dummy_level
        dummy_header = {
            k: header[k] for k in ("delimiter", "field_layout", "format") if k in header
        }
        sections = {level: {"header": dummy_header, "elements": elements}}
        schema = {k: v for k, v in schema.items() if k != "elements"}

    header = {
        **header,
        "parsing_order": header.get("parsing_order") or [{"s": list(sections.keys())}],
    }

    return {**schema, "header": header, "sections": sections}


def read_schema(
    imodel: str | None = None,
    ext_schema_path: str | None = None,
    ext_schema_file: str | None = None,
) -> SchemaDict:
    """
    Load and normalize a data model schema.

    Reads a data model schema file into a dictionary and
    normalizes it by adding the information required by
    the parser.

    Parameters
    ----------
    imodel: str, optional
        Name of internally available input data model.
        e.g. icoads_r300_d704
    ext_schema_path: str, optional
        The path to the external input data model schema file.
        The schema file must have the same name as the directory.
        One of ``imodel`` and ``ext_schema_path`` or ``ext_schema_file`` must be set.
    ext_schema_file: str, optional
        The external input data model schema file.
        One of ``imodel`` and ``ext_schema_path`` or ``ext_schema_file`` must be set.

    Returns
    -------
    SchemaDict
        Data model schema
    """
    schema_files = _resolve_schema_files(
        imodel=imodel,
        ext_schema_path=ext_schema_path,
        ext_schema_file=ext_schema_file,
    )

    raw_schema = combine_dicts(schema_files, base=f"{properties._base}.schemas")

    enriched = {
        **raw_schema,
        "name": schema_files,
    }

    return _normalize_schema(enriched)
