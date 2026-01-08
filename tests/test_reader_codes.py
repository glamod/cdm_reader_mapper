from __future__ import annotations

import pytest
from pathlib import Path
import json

from cdm_reader_mapper.mdf_reader.codes.codes import read_table


@pytest.fixture
def tmp_json_file(tmp_path: Path) -> tuple[Path, dict]:
    """Create a temporary JSON file and return path and data."""
    data = {"A": {"value": 1}, "B": {"value": 2}}
    file_path = tmp_path / "test_table.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")
    return file_path, data


def test_read_table_with_imodel():
    result = read_table("ICOADS.c99.SEALUMI", imodel="icoads_r300_d781")
    assert isinstance(result, dict)
    assert result == {"0": "no", "1": "yes", "9": "missing", "8": "unknown"}


def test_read_table_with_external_file(tmp_json_file):
    file_path, expected_data = tmp_json_file
    result = read_table("test_table", ext_table_path=str(file_path.parent))
    assert isinstance(result, dict)
    assert result == expected_data


def test_read_table_with_missing_file():
    with pytest.raises(FileNotFoundError):
        read_table("nonexistent_table", ext_table_path="tmp")


def test_read_table_requires_input():
    with pytest.raises(ValueError):
        read_table("table_without_path_or_model")
