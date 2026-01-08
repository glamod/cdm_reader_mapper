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
    header: dict
    elements: dict


class SchemaHeaderDict(TypedDict, total=False):
    parsing_order: list[dict]
    delimiter: str
    field_layout: str
    format: str
    encoding: str
    multiple_reports_per_line: bool


class SchemaDict(TypedDict, total=False):
    header: SchemaHeaderDict
    sections: dict[str, SectionDict]
    elements: dict
    name: list[Path]
    imodel: str | None


def _resolve_schema_files(
    *,
    imodel: str | None,
    ext_schema_path: str | None,
    ext_schema_file: str | None,
) -> list[Path]:
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
    imodel: str | None,
    ext_schema_path: str | None,
    ext_schema_file: str | None,
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
