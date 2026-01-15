from __future__ import annotations

import pytest
import json

from cdm_reader_mapper.mdf_reader.schemas.schemas import (
    _resolve_schema_files,
    _normalize_schema,
    read_schema,
)


@pytest.fixture
def tmp_schema_file(tmp_path):
    schema_data = {
        "header": {"delimiter": ","},
        "sections": {"sec1": {"elements": {"a": 1, "b": 2}}},
    }
    path = tmp_path / "schema"
    path.mkdir(exist_ok=True)
    file_path = tmp_path / "schema" / "schema.json"
    file_path.write_text(json.dumps(schema_data))
    return file_path, schema_data


def test_resolve_schema_file_by_file(tmp_schema_file):
    file_path, _ = tmp_schema_file
    result = _resolve_schema_files(ext_schema_file=str(file_path))
    assert isinstance(result, list)
    assert result[0] == file_path


def test_resolve_schema_file_by_path(tmp_path):
    dir_path = tmp_path / "myschema"
    dir_path.mkdir()
    schema_file = dir_path / "myschema.json"
    schema_file.write_text(json.dumps({"header": {}}))

    result = _resolve_schema_files(ext_schema_path=str(dir_path))
    assert len(result) == 1
    assert result[0] == schema_file.resolve()


def test_resolve_schema_file_missing_file(tmp_path):
    missing_file = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        _resolve_schema_files(ext_schema_file=str(missing_file))


def test_resolve_schema_file_missing_path(tmp_path):
    missing_dir = tmp_path / "nonexistent_dir"
    with pytest.raises(FileNotFoundError):
        _resolve_schema_files(ext_schema_path=str(missing_dir))


def test_resolve_schema_file_no_input():
    with pytest.raises(ValueError):
        _resolve_schema_files()


def test_normalize_schema_with_sections():
    schema = {
        "header": {"delimiter": ","},
        "sections": {"sec1": {"elements": {"a": 1}}},
    }
    result = _normalize_schema(schema)
    assert "sections" in result
    assert result["header"]["parsing_order"] == [{"s": ["sec1"]}]


def test_normalize_schema_missing_sections_and_elements():
    schema = {"header": {"delimiter": ","}}
    with pytest.raises(KeyError):
        _normalize_schema(schema)


def test_normalize_schema_preserves_existing_parsing_order():
    schema = {
        "header": {"delimiter": ",", "parsing_order": [{"s": ["sec1"]}]},
        "sections": {"sec1": {"elements": {"x": 1}}},
    }
    result = _normalize_schema(schema)
    assert result["header"]["parsing_order"] == [{"s": ["sec1"]}]


def test_read_schema_with_imodel():
    result = read_schema(imodel="icoads")
    assert isinstance(result, dict)
    assert "header" in result
    assert "sections" in result
    assert "name" in result


def test_read_schema_with_ext_file(tmp_schema_file):
    file_path, _ = tmp_schema_file

    result = read_schema(ext_schema_file=str(file_path))
    assert isinstance(result, dict)
    assert "sections" in result
    assert result["sections"]["sec1"]["elements"] == {"a": 1, "b": 2}
    assert result["name"] == [file_path]


def test_read_schema_with_ext_path(tmp_schema_file):
    file_path, _ = tmp_schema_file
    result = read_schema(ext_schema_path=str(file_path.parent))
    assert isinstance(result, dict)
    assert "sections" in result
    assert result["sections"]["sec1"]["elements"] == {"a": 1, "b": 2}
    assert result["name"] == [file_path]


def test_read_schema_requires_input():
    with pytest.raises(ValueError):
        read_schema(imodel=None, ext_schema_path=None, ext_schema_file=None)
