from __future__ import annotations
import hashlib
import importlib
import json
import logging
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests

from cdm_reader_mapper.common.getting_files import (
    _check_md5s,
    _file_md5_checksum,
    _get_file,
    _get_remote_file,
    _rm_tree,
    _with_md5_suffix,
    get_path,
    load_file,
)
from cdm_reader_mapper.common.io_files import get_filename
from cdm_reader_mapper.common.json_dict import (
    collect_json_files,
    combine_dicts,
    open_json_file,
)
from cdm_reader_mapper.common.logging_hdlr import init_logger


def compute_md5(content: bytes) -> str:
    """Helper to get MD5 of bytes."""
    return hashlib.md5(content, usedforsecurity=False).hexdigest()  # noqa: S324


def get_remote_bytes(url: str) -> bytes:
    """Fetch the content of a remote URL as bytes."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}.")

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.content


def create_structure(root: Path, structure):
    """
    Recursively create files and directories from a declarative structure list.

    Structure syntax:
    - "file.txt" ? creates file
    - ("dirname", [ ... ]) ? creates subdirectory with children
    """
    for item in structure:
        if isinstance(item, str):
            (root / item).write_text("x")
        else:
            dirname, children = item
            sub = root / dirname
            sub.mkdir()
            create_structure(sub, children)


def create_temp_file(suffix: str) -> tuple[Path, str, Path]:
    """Create a temporary file and return (file_path, suffix, expected_md5_path)."""
    tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = Path(tmp_file.name)
    md5_path = tmp_path.with_suffix(tmp_path.suffix + ".md5")
    tmp_file.close()
    return tmp_path, suffix, md5_path


@pytest.fixture
def tmp_json_file(tmp_path):
    data = {"a": 1, "b": 2}
    file_path = tmp_path / "test.json"
    file_path.write_text(json.dumps(data))
    return file_path, data


def test_init_logger_returns_logger():
    logger = init_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_init_logger_levels(caplog):
    logger = logging.getLogger("level_module")
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    with caplog.at_level(logging.DEBUG, logger="level_module"):
        logger.debug("Debug message")
        logger.info("Info message")

    logger.removeHandler(stream_handler)

    assert "Debug message" in caplog.text
    assert "Info message" in caplog.text


def test_warning_logged(caplog):
    logger = logging.getLogger("lowercase_module")
    logger.setLevel(logging.WARNING)

    with caplog.at_level(logging.WARNING, logger="lowercase_module"):
        logger.warning("Warning message")

    assert "Warning message" in caplog.text


def test_init_logger_file(tmp_path):
    log_file = tmp_path / "test.log"
    logger = init_logger("file_module", fn=str(log_file))
    logger.info("File log message")

    assert log_file.exists()
    with Path(log_file).open(encoding="utf-8") as f:
        content = f.read()
    assert "File log message" in content


def test_open_json_file(tmp_json_file):
    file_path, data = tmp_json_file
    result = open_json_file(file_path)
    assert result == data


def test_open_json_file_with_pathlib(tmp_path):
    data = {"x": 42}
    file_path = tmp_path / "data.json"
    file_path.write_text(json.dumps(data))
    result = open_json_file(file_path)
    assert result == data


def test_collect_json_files_basic(tmp_path):
    base_pkg = tmp_path / "idir0"
    base_pkg.mkdir()
    (base_pkg / "__init__.py").write_text("")
    (base_pkg / "idir0.json").write_text(json.dumps({"x": 1}))
    (base_pkg / "idir0_release.json").write_text(json.dumps({"y": 2}))

    sys.path.insert(0, str(tmp_path))

    importlib.import_module("idir0")

    files = list((base_pkg).glob("*.json"))
    names = [f.name for f in files]

    assert "idir0.json" in names
    assert "idir0_release.json" in names


def test_collect_json_files_with_args(tmp_path):
    base_pkg = tmp_path / "idir1"
    base_pkg.mkdir()
    (base_pkg / "__init__.py").write_text("")

    idir_pkg = base_pkg / "idir1"
    idir_pkg.mkdir()
    (idir_pkg / "__init__.py").write_text("")
    (idir_pkg / "idir1.json").write_text(json.dumps({"x": 1}))

    release_pkg = idir_pkg / "release"
    release_pkg.mkdir()
    (release_pkg / "__init__.py").write_text("")
    (release_pkg / "idir1_release.json").write_text(json.dumps({"y": 2}))

    sys.path.insert(0, str(tmp_path))

    importlib.import_module("idir1.idir1.release")

    files = collect_json_files("idir1", "release", base="idir1")
    names = [f.name for f in files]

    assert "idir1.json" in names
    assert "idir1_release.json" in names


def test_collect_json_files_missing_dir(tmp_path):
    package_dir = tmp_path / "missing_pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("")

    sys.path.insert(0, str(tmp_path))
    files = collect_json_files("missing_pkg")

    assert files == []


def test_combine_dicts_single_file(tmp_path):
    module_name = "temp_module_single"
    module_path = tmp_path / module_name
    module_path.mkdir()
    (module_path / "__init__.py").write_text("")

    file_path = module_path / "data.json"
    file_path.write_text(json.dumps({"a": 1}))

    sys.path.insert(0, str(tmp_path))

    combined = combine_dicts(str(file_path))
    assert combined == {"a": 1}

    sys.path.pop(0)


def test_combine_dicts_multiple_files(tmp_path):
    module_name = "temp_module_multi"
    module_path = tmp_path / module_name
    module_path.mkdir()
    (module_path / "__init__.py").write_text("")

    file1 = module_path / "file1.json"
    file1.write_text(json.dumps({"a": 1}))
    file2 = module_path / "file2.json"
    file2.write_text(json.dumps({"b": 2}))

    sys.path.insert(0, str(tmp_path))

    combined = combine_dicts([str(file1), str(file2)])
    assert combined == {"a": 1, "b": 2}

    sys.path.pop(0)


def test_combine_dicts(tmp_path):
    base_file = tmp_path / "base.json"
    base_file.write_text(json.dumps({"a": 1}))

    sub_file = tmp_path / "sub.json"
    sub_file.write_text(json.dumps({"b": 2}))

    combined = combine_dicts([base_file, sub_file])
    assert combined == {"a": 1, "b": 2}


@pytest.mark.parametrize(
    "pattern, extension, expected_filename",
    [
        (["a", "b"], "txt", "a_b.txt"),
        (["a", "", "c"], "psv", "a_c.psv"),
        (["x"], ".csv", "x.csv"),
        ([], "log", ""),
    ],
)
def test_get_filename_basic(tmp_path, pattern, extension, expected_filename):
    expected = str(tmp_path / expected_filename) if expected_filename else ""
    assert get_filename(pattern, path=str(tmp_path), extension=extension) == expected


@pytest.mark.parametrize(
    "pattern, separator, expected",
    [
        (["part1", "part2"], "-", "part1-part2"),
        (["part1", "part2"], "_", "part1_part2"),
        (["single"], "_", "single"),
        (["", "only"], "-", "only"),
        ([], "-", ""),
    ],
)
def test_get_filename_separator(pattern, separator, expected):
    result = get_filename(pattern, extension="", separator=separator)
    assert Path(result).name == expected


@pytest.mark.parametrize(
    "extension, normalized",
    [
        ("txt", ".txt"),
        (".csv", ".csv"),
        ("psv", ".psv"),
        ("", ""),
    ],
)
def test_get_filename_extension_normalization(tmp_path, extension, normalized):
    result = get_filename(["file"], path=str(tmp_path), extension=extension)
    assert result.endswith(normalized)


@pytest.mark.parametrize(
    "pattern, expected_name",
    [
        (["data", "2024"], "data_2024.pq"),
        (["", "A", "B"], "A_B.pq"),
        (["only"], "only.pq"),
    ],
)
def test_get_filename_name_part(pattern, expected_name):
    out = get_filename(pattern)
    assert out.endswith(expected_name)


@pytest.mark.parametrize(
    "content",
    [
        b"hello world",
        b"",
        b"1234567890" * 1000,
    ],
)
def test_file_md5_checksum(tmp_path, content):
    file_path = tmp_path / "testfile.txt"
    file_path.write_bytes(content)

    expected = compute_md5(content)

    result = _file_md5_checksum(file_path)
    assert result == expected


def test_get_remote_file_real(tmp_path):
    lfile = tmp_path / "README.md"
    name = Path("README.md")
    base_url = "https://raw.githubusercontent.com/glamod/cdm-testdata/main"

    local_file, _ = _get_remote_file(lfile, base_url, name)

    assert lfile.exists()
    assert lfile.stat().st_size > 0
    assert Path(local_file) == lfile


def test_get_remote_file_raises(tmp_path):
    lfile = tmp_path / "README.md"
    name = Path("README.md")
    base_url = "htttps://raw.githubusercontent.com/glamod/cdm-testdata/main"
    with pytest.raises(ValueError):
        _get_remote_file(lfile, base_url, name)


@pytest.mark.parametrize("content", [b"hello world", b"1234567890"])
def test_check_md5s_correct(tmp_path, content):
    f = tmp_path / "testfile.txt"
    f.write_bytes(content)
    md5 = compute_md5(content)
    assert _check_md5s(f, md5) is True
    assert f.exists()


def test_check_md5s_error_mode(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("data")
    wrong_md5 = "deadbeefdeadbeefdeadbeefdeadbeef"
    with pytest.raises(OSError, match="do not match"):
        _check_md5s(f, wrong_md5, mode="error")

    assert not f.exists()


def test_check_md5s_warning_mode(tmp_path):
    f = tmp_path / "file2.txt"
    f.write_text("data")
    wrong_md5 = "deadbeefdeadbeefdeadbeefdeadbeef"

    with pytest.warns(UserWarning, match="do not match"):
        _check_md5s(f, wrong_md5, mode="warning")

    assert not f.exists()


@pytest.mark.parametrize(
    "original,suffix,expected",
    [
        (Path("file.txt"), ".txt", Path("file.txt.md5")),
        (Path("archive.tar.gz"), ".hash", Path("archive.tar.hash.md5")),
        (Path("noext"), ".x", Path("noext.x.md5")),
        create_temp_file(".bin"),
    ],
)
def test_with_md5_suffix(original, suffix, expected):
    assert _with_md5_suffix(original, suffix) == expected


@pytest.mark.parametrize(
    "structure",
    [
        [],
        ["a.txt"],
        ["a.txt", ("sub", ["b.txt", "c.txt"])],
        [("x", [("y", ["deep.txt"])])],
    ],
)
def test_rm_tree(tmp_path, structure):
    base = tmp_path / "root"
    base.mkdir()

    create_structure(base, structure)

    assert base.exists()
    _rm_tree(base)
    assert not base.exists()


def test_get_file_real(tmp_path):
    f = "header-icoads_r302_d792_2022-02-01_subset.psv"
    name = Path(f)
    base_url = "https://raw.githubusercontent.com/glamod/cdm-testdata/main/icoads/r302/d792/cdm_tables"
    cache_dir = tmp_path / "cache"

    out_file = _get_file(
        name=name,
        suffix=".psv",
        url=base_url,
        cache_dir=cache_dir,
        clear_cache=True,
        within_drs=False,
    )

    assert out_file.exists()
    assert out_file.stat().st_size > 0
    assert out_file.parent == cache_dir
    assert out_file.name == f

    remote_bytes = get_remote_bytes(f"{base_url}/{name.name}")
    assert compute_md5(out_file.read_bytes()) == compute_md5(remote_bytes)

    assert list(cache_dir.rglob("*.md5")) == []


@pytest.mark.parametrize("within_drs", [True, False])
@pytest.mark.parametrize("cache", [True, False])
def test_load_file_real(tmp_path, within_drs, cache):
    base_url = "https://github.com/glamod/cdm-testdata"
    drs = "icoads/r302/d792/cdm_tables"
    f = "header-icoads_r302_d792_2022-02-01_subset.psv"
    cache_dir = tmp_path / "cache"

    local_file = load_file(
        name=str(Path(drs) / f),
        github_url=base_url,
        branch="main",
        cache=cache,
        cache_dir=str(cache_dir),
        clear_cache=True,
        within_drs=within_drs,
    )

    if cache:
        assert local_file.exists()
        expected_parent = cache_dir if not within_drs else cache_dir / Path(drs)
        assert local_file.parent == expected_parent
        assert local_file.name == f
        assert list(cache_dir.rglob("*.md5")) == []
    else:
        assert not local_file.exists()

    remote_bytes = get_remote_bytes(f"{base_url}/raw/main/{drs}/{f}")
    if cache:
        assert compute_md5(local_file.read_bytes()) == compute_md5(remote_bytes)


def test_load_file_invalid_url():
    with pytest.raises(ValueError):
        load_file(name="file.txt", github_url="ftp://malicious-site.com")


def test_get_path_existing_file(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("hello")

    result = get_path(str(file_path))

    assert isinstance(result, Path)
    assert result.exists()
    assert result == file_path


def test_get_path_missing_file_module_not_found(tmp_path, caplog):
    """Test get_path when the file is missing and _files raises ModuleNotFoundError."""
    missing_file = tmp_path / "missing.txt"
    caplog.set_level(logging.WARNING)

    result = get_path(str(missing_file))

    assert result is None
    assert any("No module named" in msg for msg in caplog.messages)
