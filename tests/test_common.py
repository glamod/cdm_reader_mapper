from __future__ import annotations

import pytest

import hashlib
import importlib
import json
import logging
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

from io import StringIO
from pathlib import Path
import pyarrow.parquet as pq

from urllib.parse import urlparse

import requests


from cdm_reader_mapper.common.select import (
    _split_df,
    _split_by_index_df,
    _split_by_boolean_df,
    _split_by_column_df,
    split_by_boolean,
    split_by_boolean_true,
    split_by_boolean_false,
    split_by_column_entries,
    split_by_index,
)
from cdm_reader_mapper.common.replace import replace_columns

from cdm_reader_mapper.common.logging_hdlr import init_logger
from cdm_reader_mapper.common.json_dict import (
    open_json_file,
    collect_json_files,
    combine_dicts,
)
from cdm_reader_mapper.common.io_files import get_filename
from cdm_reader_mapper.common.inspect import _count_by_cat, get_length, count_by_cat
from cdm_reader_mapper.common.getting_files import (
    _file_md5_checksum,
    _get_remote_file,
    _check_md5s,
    _with_md5_suffix,
    _rm_tree,
    _get_file,
    load_file,
    get_path,
)
from cdm_reader_mapper.common.iterators import (
    ProcessFunction,
    ParquetStreamReader,
    _sort_chunk_outputs,
    _initialize_storage,
    _write_chunks_to_disk,
    _parquet_generator,
    _process_chunks,
    _prepare_readers,
    parquet_stream_from_iterable,
    is_valid_iterator,
    ensure_parquet_reader,
    process_disk_backed,
    _process_function,
    process_function,
)


def make_parser(text, **kwargs):
    """Helper: create a TextFileReader similar to user code."""
    buffer = StringIO(text)
    return pd.read_csv(buffer, chunksize=2, **kwargs)


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


def dummy_func(x):
    return 2 * x


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["x", "y", "z", "x", "y"],
            "C": [True, False, True, False, True],
        },
        index=[10, 11, 12, 13, 14],
    )


@pytest.fixture
def sample_reader():
    text = "10,1,x,True\n11,2,y,False\n12,3,z,True\n13,4,x,False\n14,5,y,True"
    reader = make_parser(text, names=["A", "B", "C"])
    return reader


@pytest.fixture
def sample_df_multi():
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["x", "y", "z", "x", "y"],
            "C": [True, False, True, False, True],
        },
        index=[10, 11, 12, 13, 14],
    )


@pytest.fixture
def sample_reader_multi():
    text = "10,1,x,True\n11,2,y,False\n12,3,z,True\n13,4,x,False\n14,5,y,True"
    reader = make_parser(text, names=[("A", "a"), ("B", "b"), ("C", "c")])
    return reader


@pytest.fixture
def empty_df():
    return pd.DataFrame(columns=["A", "B", "C"])


@pytest.fixture
def empty_reader():
    return make_parser("", names=["A", "B", "C"])


@pytest.fixture
def boolean_mask():
    return pd.DataFrame(
        {
            "mask1": [False, True, False, True, False],
            "mask2": [True, False, True, False, True],
        },
        index=[10, 11, 12, 13, 14],
    )


@pytest.fixture
def boolean_mask_true():
    return pd.DataFrame(
        {
            "mask1": [True, True, False, True, False],
            "mask2": [True, False, True, False, True],
        },
        index=[10, 11, 12, 13, 14],
    )


@pytest.fixture
def tmp_json_file(tmp_path):
    data = {"a": 1, "b": 2}
    file_path = tmp_path / "test.json"
    file_path.write_text(json.dumps(data))
    return file_path, data


def test_split_df(sample_df):
    mask = pd.Series([True, False, False, True, False], index=sample_df.index)
    selected, rejected, _, _ = _split_df(sample_df, mask, return_rejected=True)
    assert list(selected.index) == [10, 13]
    assert list(rejected.index) == [11, 12, 14]


def _test_split_df_false_mask(sample_df):
    mask = pd.Series([False, False, False, False, False], index=sample_df.index)
    selected, rejected, _, _ = _split_df(sample_df, mask, return_rejected=True)
    assert list(selected.index) == [10, 13]
    assert list(rejected.index) == [11, 12, 14]


def test_split_df_multiindex(sample_df):
    mask = pd.Series([True, False, False, True, False], index=sample_df.index)
    sample_df.columns = pd.MultiIndex.from_tuples(
        [
            ("A", "a"),
            (
                "B",
                "b",
            ),
            ("C", "c"),
        ]
    )
    selected, rejected, _, _ = _split_df(sample_df, mask, return_rejected=True)
    assert list(selected.index) == [10, 13]
    assert list(rejected.index) == [11, 12, 14]


@pytest.mark.parametrize(
    "column,boolean,expected_selected,expected_rejected",
    [
        ("C", True, [10, 12, 14], [11, 13]),
        ("C", False, [11, 13], [10, 12, 14]),
    ],
)
def test_split_by_boolean_df(
    sample_df, column, boolean, expected_selected, expected_rejected
):
    mask = sample_df[[column]]
    selected, rejected, _, _ = _split_by_boolean_df(
        sample_df, mask, boolean=boolean, return_rejected=True
    )
    assert list(selected.index) == expected_selected
    assert list(rejected.index) == expected_rejected


def test_split_by_boolean_df_empty_mask(sample_df):
    mask = pd.DataFrame(columns=sample_df.columns)
    selected, rejected, _, _ = _split_by_boolean_df(
        sample_df, mask, boolean=True, return_rejected=True
    )
    assert list(selected.index) == list(sample_df.index)
    assert rejected.empty


@pytest.mark.parametrize(
    "col,values,return_rejected,expected_selected,expected_rejected",
    [
        ("B", ["x", "z"], True, [10, 12, 13], [11, 14]),
        ("B", ["missing"], True, [], [10, 11, 12, 13, 14]),
        ("B", ["x", "z"], False, [10, 12, 13], []),
    ],
)
def test_split_by_column_df(
    sample_df, col, values, return_rejected, expected_selected, expected_rejected
):
    selected, rejected, _, _ = _split_by_column_df(
        sample_df, col, values, return_rejected=return_rejected
    )
    assert list(selected.index) == expected_selected
    assert list(rejected.index) == expected_rejected


@pytest.mark.parametrize(
    "index_list,inverse,return_rejected,expected_selected,expected_rejected",
    [
        ([11, 13], False, True, [11, 13], [10, 12, 14]),
        ([11, 13], False, False, [11, 13], []),
        ([11, 13], True, True, [10, 12, 14], [11, 13]),
    ],
)
def test_split_by_index_df(
    sample_df,
    index_list,
    inverse,
    return_rejected,
    expected_selected,
    expected_rejected,
):
    selected, rejected, _, _ = _split_by_index_df(
        sample_df, index_list, inverse=inverse, return_rejected=return_rejected
    )
    assert list(selected.index) == expected_selected
    assert list(rejected.index) == expected_rejected


@pytest.mark.parametrize("TextFileReader", [False, True])
def test_split_by_index_basic(sample_df, sample_reader, TextFileReader):
    if TextFileReader:
        data = sample_reader
    else:
        data = sample_df
    selected, rejected, _, _ = split_by_index(data, [11, 13], return_rejected=True)

    if TextFileReader:
        selected = selected.read()
        rejected = rejected.read()

    assert list(selected.index) == [11, 13]
    assert list(rejected.index) == [10, 12, 14]


def test_split_by_index_multiindex(sample_reader_multi):
    selected, rejected, _, _ = split_by_index(
        sample_reader_multi, [11, 13], return_rejected=True
    )

    selected = selected.read()
    rejected = rejected.read()

    assert list(selected.index) == [11, 13]
    assert list(rejected.index) == [10, 12, 14]


@pytest.mark.parametrize("TextFileReader", [False, True])
def test_split_by_column_entries_basic(sample_df, sample_reader, TextFileReader):
    if TextFileReader:
        data = sample_reader
    else:
        data = sample_df

    selected, rejected, _, _ = split_by_column_entries(
        data, {"B": ["y"]}, return_rejected=True
    )

    if TextFileReader:
        selected = selected.read()
        rejected = rejected.read()

    assert list(selected.index) == [11, 14]
    assert list(rejected.index) == [10, 12, 13]


@pytest.mark.parametrize(
    "inverse, reset_index, exp_selected_idx, exp_rejected_idx",
    [
        (False, False, [], [10, 11, 12, 13, 14]),
        (False, True, [], [10, 11, 12, 13, 14]),
        (True, False, [10, 11, 12, 13, 14], []),
        (True, True, [10, 11, 12, 13, 14], []),
    ],
)
@pytest.mark.parametrize("chunked", [False, True])
def test_split_by_boolean_basic_true(
    sample_df,
    sample_reader,
    boolean_mask,
    inverse,
    reset_index,
    exp_selected_idx,
    exp_rejected_idx,
    chunked,
):
    if chunked:
        data = sample_reader
    else:
        data = sample_df

    selected, rejected, _, _ = split_by_boolean(
        data,
        boolean_mask,
        boolean=True,
        inverse=inverse,
        reset_index=reset_index,
        return_rejected=True,
    )

    exp_selected = sample_df.loc[exp_selected_idx]
    exp_rejected = sample_df.loc[exp_rejected_idx]

    if reset_index is True:
        exp_selected = exp_selected.reset_index(drop=True)
        exp_rejected = exp_rejected.reset_index(drop=True)

    if chunked:
        selected = selected.read()
        rejected = rejected.read()

    pd.testing.assert_frame_equal(selected, exp_selected)
    pd.testing.assert_frame_equal(rejected, exp_rejected)


@pytest.mark.parametrize(
    "inverse, reset_index, exp_selected_idx, exp_rejected_idx",
    [
        (False, False, [], [10, 11, 12, 13, 14]),
        (False, True, [], [10, 11, 12, 13, 14]),
        (True, False, [10, 11, 12, 13, 14], []),
        (True, True, [10, 11, 12, 13, 14], []),
    ],
)
@pytest.mark.parametrize("chunked", [False, True])
def test_split_by_boolean_basic_false(
    sample_df,
    sample_reader,
    boolean_mask,
    inverse,
    reset_index,
    exp_selected_idx,
    exp_rejected_idx,
    chunked,
):
    if chunked:
        data = sample_reader
    else:
        data = sample_df

    selected, rejected, _, _ = split_by_boolean(
        data,
        boolean_mask,
        boolean=False,
        inverse=inverse,
        reset_index=reset_index,
        return_rejected=True,
    )

    exp_selected = sample_df.loc[exp_selected_idx]
    exp_rejected = sample_df.loc[exp_rejected_idx]

    if reset_index is True:
        exp_selected = exp_selected.reset_index(drop=True)
        exp_rejected = exp_rejected.reset_index(drop=True)

    if chunked:
        selected = selected.read()
        rejected = rejected.read()

    pd.testing.assert_frame_equal(selected, exp_selected)
    pd.testing.assert_frame_equal(rejected, exp_rejected)


@pytest.mark.parametrize("TextFileReader", [False, True])
def test_split_by_boolean_true_basic(
    sample_df, sample_reader, boolean_mask_true, TextFileReader
):
    if TextFileReader:
        data = sample_reader
    else:
        data = sample_df

    selected, rejected, _, _ = split_by_boolean_true(
        data, boolean_mask_true, return_rejected=True
    )

    if TextFileReader:
        selected = selected.read()
        rejected = rejected.read()

    assert list(selected.index) == [10]
    assert list(rejected.index) == [11, 12, 13, 14]


@pytest.mark.parametrize("TextFileReader", [False, True])
def test_split_by_boolean_false_basic(
    sample_df, sample_reader, boolean_mask, TextFileReader
):
    if TextFileReader:
        data = sample_reader
    else:
        data = sample_df

    selected, rejected, _, _ = split_by_boolean_false(
        data, boolean_mask, return_rejected=True
    )

    if TextFileReader:
        selected = selected.read()
        rejected = rejected.read()

    assert list(selected.index) == []
    assert list(rejected.index) == [10, 11, 12, 13, 14]


@pytest.mark.parametrize("TextFileReader", [False, True])
def test_split_by_index_empty(empty_df, empty_reader, TextFileReader):
    if TextFileReader:
        data = empty_reader
    else:
        data = empty_df

    selected, rejected, _, _ = split_by_index(data, [0, 1], return_rejected=True)

    if TextFileReader:
        selected = selected.read()
        rejected = rejected.read()

    assert selected.empty
    assert rejected.empty


@pytest.mark.parametrize("TextFileReader", [False, True])
def test_split_by_column_empty(empty_df, empty_reader, TextFileReader):
    if TextFileReader:
        data = empty_reader
    else:
        data = empty_df

    selected, rejected, _, _ = split_by_column_entries(
        data, {"A": [1]}, return_rejected=True
    )

    if TextFileReader:
        selected = selected.read()
        rejected = rejected.read()

    assert selected.empty
    assert rejected.empty


@pytest.mark.parametrize("TextFileReader", [False, True])
def test_split_by_boolean_empty(empty_df, empty_reader, TextFileReader):
    if TextFileReader:
        data = empty_reader
    else:
        data = empty_df

    mask = empty_df.astype(bool)
    selected, rejected, _, _ = split_by_boolean(
        data, mask, boolean=True, return_rejected=True
    )

    if TextFileReader:
        selected = selected.read()
        rejected = rejected.read()

    assert selected.empty
    assert rejected.empty


def test_basic_replacement_df():
    df_l = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
    df_r = pd.DataFrame({"id": [1, 2], "x": [100, 200]})

    out = replace_columns(df_l, df_r, pivot_c="id", rep_c="x")
    assert out["x"].tolist() == [100, 200]


def test_basic_replacement_textfilereader():
    parser_l = make_parser("id,x\n1,10\n2,20")
    parser_r = make_parser("id,x\n1,100\n2,200")

    out = replace_columns(parser_l, parser_r, pivot_c="id", rep_c="x")
    out = out.read()
    assert out["x"].tolist() == [100, 200]


def test_rep_map_different_names():
    df_l = pd.DataFrame({"id": [1, 2], "a": [1, 2]})
    df_r = pd.DataFrame({"id": [1, 2], "b": [10, 20]})

    out = replace_columns(df_l, df_r, pivot_c="id", rep_map={"a": "b"})
    assert out["a"].tolist() == [10, 20]


def test_missing_pivot_raises():
    df_l = pd.DataFrame({"id": [1]})
    df_r = pd.DataFrame({"id": [1]})

    with pytest.raises(ValueError):
        replace_columns(df_l, df_r, rep_c="x")


def test_missing_replacement_raises():
    df_l = pd.DataFrame({"id": [1]})
    df_r = pd.DataFrame({"id": [1]})

    with pytest.raises(ValueError):
        replace_columns(df_l, df_r, pivot_c="id")


def test_missing_source_col_raises():
    df_l = pd.DataFrame({"id": [1], "a": [10]})
    df_r = pd.DataFrame({"id": [1]})

    with pytest.raises(ValueError):
        replace_columns(df_l, df_r, pivot_c="id", rep_map={"a": "missing"})


def test_index_reset():
    df_l = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
    df_r = pd.DataFrame({"id": [1, 2], "x": [100, 200]})

    out = replace_columns(df_l, df_r, pivot_c="id", rep_c="x")
    assert list(out.index) == [0, 1]


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
    with open(log_file, encoding="utf-8") as f:
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
        (["a", "b"], "txt", "a-b.txt"),
        (["a", "", "c"], "psv", "a-c.psv"),
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
    assert os.path.basename(result) == expected


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
        (["data", "2024"], "data-2024.psv"),
        (["", "A", "B"], "A-B.psv"),
        (["only"], "only.psv"),
    ],
)
def test_get_filename_name_part(pattern, expected_name):
    out = get_filename(pattern)
    assert out.endswith(expected_name)


@pytest.mark.parametrize(
    "data, expected",
    [
        (["a", "b", "a"], {"a": 2, "b": 1}),
        ([np.nan, "x", np.nan], {"nan": 2, "x": 1}),
        ([], {}),
        ([1, 2, 1, 3], {1: 2, 2: 1, 3: 1}),
        ([None, "a", None], {"nan": 2, "a": 1}),
    ],
)
def test_count_by_cat_i(data, expected):
    series = pd.DataFrame(data, columns=["test"])
    assert _count_by_cat(series, ["test"])["test"] == expected


@pytest.mark.parametrize(
    "data, columns, expected",
    [
        (pd.DataFrame({"A": ["x", "y", "x"]}), "A", {"A": {"x": 2, "y": 1}}),
        (
            pd.DataFrame({"A": ["x", "y", "x"], "B": [1, 2, np.nan]}),
            ["A", "B"],
            {"A": {"x": 2, "y": 1}, "B": {1: 1, 2: 1, "nan": 1}},
        ),
        (
            pd.DataFrame({"C": ["a", "a", "b"]}),
            "C",
            {"C": {"a": 2, "b": 1}},
        ),
        (pd.DataFrame(columns=["D"]), ["D"], {"D": {}}),
    ],
)
def test_count_by_cat_dataframe(data, columns, expected):
    result = count_by_cat(data, columns)
    assert result == expected


def test_count_by_cat_single_column_string():
    df = pd.DataFrame({"A": [1, 2, 2, np.nan]})
    result = count_by_cat(df, "A")
    assert result == {"A": {1: 1, 2: 2, "nan": 1}}


def test_count_by_cat_textfilereader():
    text = "A,B\n1,x\n2,y\n2,x\nnan,z"
    parser = make_parser(text)

    result = count_by_cat(parser, ["A", "B"])
    expected = {
        "A": {1: 1, 2: 2, "nan": 1},
        "B": {"x": 2, "y": 1, "z": 1},
    }
    assert result == expected


@pytest.mark.parametrize(
    "data, expected_len",
    [
        (pd.DataFrame({"A": [1, 2, 3]}), 3),
        (make_parser("A,B\n1,x\n2,y\n3,z"), 3),
    ],
)
def test_get_length_inspect(data, expected_len):
    assert get_length(data) == expected_len


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
        name=os.path.join(drs, f),
        github_url=base_url,
        branch="main",
        cache=cache,
        cache_dir=cache_dir,
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


def test_get_path_missing_file(tmp_path, caplog):
    missing_file = tmp_path / "missing.txt"
    caplog.set_level(logging.WARNING)

    result = get_path(str(missing_file))

    assert result is None
    assert any(
        "No module named" in msg or "Cannot treat" in msg for msg in caplog.messages
    )


def test_class_process_function_basic():
    df = pd.DataFrame({"a": [1, 2, 3]})

    pf = ProcessFunction(data=df, func=dummy_func)

    assert isinstance(pf, ProcessFunction)
    pd.testing.assert_frame_equal(pf.data, df)
    assert pf.func is dummy_func
    assert pf.func_args == ()
    assert pf.func_kwargs == {}


def test_class_process_function_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError, match="not callable"):
        ProcessFunction(data=df, func="invalid_function")


def test_class_process_function_tuple():
    df = pd.DataFrame({"a": [1, 2, 3]})

    pf = ProcessFunction(data=df, func=dummy_func, func_args=10)

    assert pf.func_args == (10,)


def test_class_process_function_extra():
    df = pd.DataFrame({"a": [1, 2, 3]})

    pf = ProcessFunction(df, dummy_func, extra=123, flag=True)

    assert pf.kwargs == {"extra": 123, "flag": True}


def make_chunks():
    return [
        pd.DataFrame({"a": [1, 2]}),
        pd.DataFrame({"a": [3, 4]}),
    ]


def chunk_generator():
    yield from make_chunks()


def test_init_with_iterator():
    reader = ParquetStreamReader(iter(make_chunks()))
    assert isinstance(reader, ParquetStreamReader)


def test_init_with_factory():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))
    assert isinstance(reader, ParquetStreamReader)


def test_init_invalid_source():
    with pytest.raises(TypeError):
        ParquetStreamReader(source=123)


def test_iteration_over_chunks():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))

    chunks = list(reader)

    assert len(chunks) == 2
    assert chunks[0]["a"].iloc[0] == 1
    assert chunks[1]["a"].iloc[-1] == 4


def test_next_raises_stop_iteration():
    reader = ParquetStreamReader(lambda: iter([]))

    with pytest.raises(StopIteration):
        next(reader)


def test_prepend_pushes_chunk_to_front():
    chunks = make_chunks()
    reader = ParquetStreamReader(lambda: iter(chunks))

    first = next(reader)
    reader.prepend(first)

    again = next(reader)

    pd.testing.assert_frame_equal(first, again)


def test_get_chunk_returns_next_chunk():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))

    chunk = reader.get_chunk()

    assert isinstance(chunk, pd.DataFrame)
    assert len(chunk) == 2


def test_read_concatenates_all_chunks():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))

    df = reader.read()

    assert len(df) == 4
    assert df["a"].tolist() == [1, 2, 3, 4]


def test_read_empty_stream_returns_empty_dataframe():
    reader = ParquetStreamReader(lambda: iter([]))

    df = reader.read()

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_copy_creates_independent_stream():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))

    reader_copy = reader.copy()

    original_first = next(reader)
    copy_first = next(reader_copy)

    pd.testing.assert_frame_equal(original_first, copy_first)


def test_copy_closed_stream_raises():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))
    reader.close()

    with pytest.raises(ValueError):
        reader.copy()


def test_empty_returns_true_if_empty():
    reader = ParquetStreamReader(lambda: iter([]))
    assert reader.empty() is True


def test_empty_returns_false_if_not_empty():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))
    assert reader.empty() is False


def test_reset_index_continuous_index():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))

    new_reader = reader.reset_index(drop=True)

    df = new_reader.read()

    assert df.index.tolist() == [0, 1, 2, 3]


def test_reset_index_keeps_old_index_column():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))

    new_reader = reader.reset_index(drop=False)
    df = new_reader.read()

    assert "index" in df.columns
    assert df.index.tolist() == [0, 1, 2, 3]


def test_reset_index_closed_stream_raises():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))
    reader.close()

    with pytest.raises(ValueError):
        reader.reset_index()


def test_next_on_closed_stream_raises():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))
    reader.close()

    with pytest.raises(ValueError):
        next(reader)


def test_context_manager_closes_stream():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))

    with reader as r:
        chunk = next(r)
        assert len(chunk) == 2

    with pytest.raises(ValueError):
        next(reader)


@pytest.mark.parametrize(
    "outputs,capture_meta,expected_data_len,expected_meta_len",
    [
        ((pd.DataFrame({"a": [1]}),), False, 1, 0),
        ((pd.DataFrame({"a": [1]}), "meta"), True, 1, 1),
        (([pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})],), False, 2, 0),
        (("meta1", "meta2"), True, 0, 2),
    ],
)
def test_sort_chunk_outputs_parametrized(
    outputs, capture_meta, expected_data_len, expected_meta_len
):
    data, meta = _sort_chunk_outputs(
        outputs,
        capture_meta=capture_meta,
        requested_types=(pd.DataFrame,),
    )

    assert len(data) == expected_data_len
    assert len(meta) == expected_meta_len


def make_df_0():
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})


def make_series_0():
    return pd.Series([1, 2, 3], name="my_series")


@pytest.mark.parametrize(
    "inputs,expected_schema_types",
    [
        ([make_df_0()], [(pd.DataFrame, make_df_0().columns)]),
        ([make_series_0()], [(pd.Series, "my_series")]),
        (
            [make_df_0(), make_series_0()],
            [
                (pd.DataFrame, make_df_0().columns),
                (pd.Series, "my_series"),
            ],
        ),
        (
            [make_df_0(), make_df_0()],
            [
                (pd.DataFrame, make_df_0().columns),
                (pd.DataFrame, make_df_0().columns),
            ],
        ),
    ],
)
def test_initialize_storage_valid(inputs, expected_schema_types):
    temp_dirs, schemas = _initialize_storage(inputs)

    try:
        # Correct number of temp dirs created
        assert len(temp_dirs) == len(inputs)

        # Ensure they are TemporaryDirectory instances
        assert all(isinstance(td, tempfile.TemporaryDirectory) for td in temp_dirs)

        # Check schemas
        assert len(schemas) == len(expected_schema_types)

        for (actual_type, actual_meta), (exp_type, exp_meta) in zip(
            schemas, expected_schema_types
        ):
            assert actual_type is exp_type

            if exp_type is pd.DataFrame:
                assert list(actual_meta) == list(exp_meta)
            else:
                assert actual_meta == exp_meta

    finally:
        # Clean up temp dirs to avoid ResourceWarning
        for td in temp_dirs:
            td.cleanup()


def test_initialize_storage_empty():
    temp_dirs, schemas = _initialize_storage([])

    assert temp_dirs == []
    assert schemas == []


@pytest.mark.parametrize(
    "invalid_input",
    [
        [123],
        ["string"],
        [object()],
        [make_df_0(), 42],
    ],
)
def test_initialize_storage_invalid_type_raises(invalid_input):
    with pytest.raises(TypeError, match="Unsupported data type"):
        _initialize_storage(invalid_input)


def make_df_1():
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})


def make_series_1():
    return pd.Series([10, 20], name="s")


def read_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


@pytest.mark.parametrize(
    "batch",
    [
        [make_df_1()],
        [make_series_1()],
        [make_df_1(), make_df_1()],
        [make_df_1(), make_series_1()],
    ],
)
def test_write_chunks_creates_files(batch):
    temp_dirs = [tempfile.TemporaryDirectory() for _ in batch]

    try:
        _write_chunks_to_disk(batch, temp_dirs, chunk_counter=0)

        for i, _ in enumerate(batch):
            expected_file = Path(temp_dirs[i].name) / "part_00000.parquet"
            assert expected_file.exists()

    finally:
        for td in temp_dirs:
            td.cleanup()


@pytest.mark.parametrize(
    "counter,expected_name",
    [
        (0, "part_00000.parquet"),
        (1, "part_00001.parquet"),
        (42, "part_00042.parquet"),
        (1234, "part_01234.parquet"),
    ],
)
def test_chunk_counter_format(counter, expected_name):
    batch = [make_df_1()]
    temp_dirs = [tempfile.TemporaryDirectory()]

    try:
        _write_chunks_to_disk(batch, temp_dirs, chunk_counter=counter)

        expected_file = Path(temp_dirs[0].name) / expected_name
        assert expected_file.exists()

    finally:
        temp_dirs[0].cleanup()


def test_series_written_as_dataframe():
    s = make_series_1()
    temp_dirs = [tempfile.TemporaryDirectory()]

    try:
        _write_chunks_to_disk([s], temp_dirs, chunk_counter=0)

        file_path = Path(temp_dirs[0].name) / "part_00000.parquet"
        df = read_parquet(file_path)

        # Series becomes single-column dataframe
        assert list(df.columns) == ["s"]
        assert df["s"].tolist() == [10, 20]

    finally:
        temp_dirs[0].cleanup()


def test_index_is_preserved():
    df = make_df_1()
    df.index = ["x", "y"]

    temp_dirs = [tempfile.TemporaryDirectory()]

    try:
        _write_chunks_to_disk([df], temp_dirs, chunk_counter=0)

        file_path = Path(temp_dirs[0].name) / "part_00000.parquet"
        result = read_parquet(file_path)

        assert list(result.index) == ["x", "y"]

    finally:
        temp_dirs[0].cleanup()


def test_multiple_chunk_writes():
    batch = [make_df_1()]
    temp_dirs = [tempfile.TemporaryDirectory()]

    try:
        _write_chunks_to_disk(batch, temp_dirs, chunk_counter=0)
        _write_chunks_to_disk(batch, temp_dirs, chunk_counter=1)

        file0 = Path(temp_dirs[0].name) / "part_00000.parquet"
        file1 = Path(temp_dirs[0].name) / "part_00001.parquet"

        assert file0.exists()
        assert file1.exists()

    finally:
        temp_dirs[0].cleanup()


def test_mismatched_temp_dirs_raises_index_error():
    batch = [make_df_1(), make_df_1()]
    temp_dirs = [tempfile.TemporaryDirectory()]  # only one dir

    try:
        with pytest.raises(IndexError):
            _write_chunks_to_disk(batch, temp_dirs, chunk_counter=0)
    finally:
        temp_dirs[0].cleanup()


def write_parquet(path: Path, df: pd.DataFrame):
    df.to_parquet(path, index=True)


def make_df(values, columns=("a",)):
    return pd.DataFrame(values, columns=columns)


def test_parquet_generator_dataframe():
    temp_dir = tempfile.TemporaryDirectory()

    try:
        df1 = make_df([[1], [2]])
        df2 = make_df([[3], [4]])

        write_parquet(Path(temp_dir.name) / "part_00000.parquet", df1)
        write_parquet(Path(temp_dir.name) / "part_00001.parquet", df2)

        gen = _parquet_generator(
            temp_dir=temp_dir,
            data_type=pd.DataFrame,
            schema=df1.columns,
        )

        outputs = list(gen)

        assert len(outputs) == 2
        pd.testing.assert_frame_equal(outputs[0], df1)
        pd.testing.assert_frame_equal(outputs[1], df2)

    finally:
        # Generator should already cleanup, but ensure no crash
        if Path(temp_dir.name).exists():
            temp_dir.cleanup()


def test_parquet_generator_series():
    temp_dir = tempfile.TemporaryDirectory()

    try:
        df1 = make_df([[10], [20]])
        df2 = make_df([[30], [40]])

        write_parquet(Path(temp_dir.name) / "part_00000.parquet", df1)
        write_parquet(Path(temp_dir.name) / "part_00001.parquet", df2)

        gen = _parquet_generator(
            temp_dir=temp_dir,
            data_type=pd.Series,
            schema="my_series",
        )

        outputs = list(gen)

        assert len(outputs) == 2
        assert isinstance(outputs[0], pd.Series)
        assert outputs[0].name == "my_series"
        assert outputs[0].tolist() == [10, 20]
        assert outputs[1].tolist() == [30, 40]

    finally:
        if Path(temp_dir.name).exists():
            temp_dir.cleanup()


def test_files_are_read_sorted():
    temp_dir = tempfile.TemporaryDirectory()

    try:
        df1 = make_df([[1]])
        df2 = make_df([[2]])

        # Intentionally reversed names
        write_parquet(Path(temp_dir.name) / "part_00001.parquet", df2)
        write_parquet(Path(temp_dir.name) / "part_00000.parquet", df1)

        gen = _parquet_generator(
            temp_dir=temp_dir,
            data_type=pd.DataFrame,
            schema=df1.columns,
        )

        outputs = list(gen)

        # Should be sorted lexicographically
        assert outputs[0]["a"].iloc[0] == 1
        assert outputs[1]["a"].iloc[0] == 2

    finally:
        if Path(temp_dir.name).exists():
            temp_dir.cleanup()


def test_empty_directory_yields_nothing():
    temp_dir = tempfile.TemporaryDirectory()

    gen = _parquet_generator(
        temp_dir=temp_dir,
        data_type=pd.DataFrame,
        schema=None,
    )

    outputs = list(gen)
    assert outputs == []


def test_cleanup_after_full_iteration():
    temp_dir = tempfile.TemporaryDirectory()

    df = make_df([[1]])
    write_parquet(Path(temp_dir.name) / "part_00000.parquet", df)

    gen = _parquet_generator(
        temp_dir=temp_dir,
        data_type=pd.DataFrame,
        schema=df.columns,
    )

    list(gen)

    # Directory should be removed after generator finishes
    assert not Path(temp_dir.name).exists()


def test_cleanup_on_partial_iteration():
    temp_dir = tempfile.TemporaryDirectory()

    df1 = make_df([[1]])
    df2 = make_df([[2]])

    write_parquet(Path(temp_dir.name) / "part_00000.parquet", df1)
    write_parquet(Path(temp_dir.name) / "part_00001.parquet", df2)

    gen = _parquet_generator(
        temp_dir=temp_dir,
        data_type=pd.DataFrame,
        schema=df1.columns,
    )

    next(gen)  # consume one element
    gen.close()  # trigger generator finalization

    assert not Path(temp_dir.name).exists()


def make_reader(chunks):
    return ParquetStreamReader(lambda: iter(chunks))


def df(val):
    return pd.DataFrame({"a": [val]})


def test_process_chunks_data_only():
    readers = [make_reader([df(1), df(2)])]

    def func(x):
        return x * 2

    result = _process_chunks(
        readers=readers,
        func=func,
        requested_types=(pd.DataFrame,),
        static_args=[],
        static_kwargs={},
        non_data_output="first",
        non_data_proc=None,
        non_data_proc_args=(),
        non_data_proc_kwargs={},
    )

    data_reader = result[0]
    output = data_reader.read()

    assert output["a"].tolist() == [2, 4]


def test_metadata_only_first_chunk():
    readers = [make_reader([df(1), df(2)])]

    def func(x):
        return x, f"meta_{x['a'].iloc[0]}"

    result = _process_chunks(
        readers=readers,
        func=func,
        requested_types=(pd.DataFrame,),
        static_args=[],
        static_kwargs={},
        non_data_output="first",
        non_data_proc=None,
        non_data_proc_args=(),
        non_data_proc_kwargs={},
    )

    data_reader, meta = result

    assert data_reader.read()["a"].tolist() == [1, 2]
    assert meta == "meta_1"  # only first chunk captured


def test_metadata_accumulation():
    readers = [make_reader([df(1), df(2)])]

    def func(x):
        return x, x["a"].iloc[0]

    result = _process_chunks(
        readers=readers,
        func=func,
        requested_types=(pd.DataFrame,),
        static_args=[],
        static_kwargs={},
        non_data_output="acc",
        non_data_proc=None,
        non_data_proc_args=(),
        non_data_proc_kwargs={},
    )

    _, meta = result

    assert meta == [1, 2]


def test_non_data_proc_applied_helper():
    readers = [make_reader([df(1), df(2)])]

    def func(x):
        return x, x["a"].iloc[0]

    def processor(meta):
        return sum(meta)

    result = _process_chunks(
        readers=readers,
        func=func,
        requested_types=(pd.DataFrame,),
        static_args=[],
        static_kwargs={},
        non_data_output="acc",
        non_data_proc=processor,
        non_data_proc_args=(),
        non_data_proc_kwargs={},
    )

    _, meta = result

    assert meta == 3


def test_only_metadata_output():
    readers = [make_reader([df(1), df(2)])]

    def func(x):
        return x["a"].iloc[0]

    result = _process_chunks(
        readers=readers,
        func=func,
        requested_types=(pd.DataFrame,),
        static_args=[],
        static_kwargs={},
        non_data_output="acc",
        non_data_proc=None,
        non_data_proc_args=(),
        non_data_proc_kwargs={},
    )

    # Should return metadata only
    assert result == [1, 2]


def test_empty_iterable_raises():
    readers = [make_reader([])]

    def func(x):
        return x

    with pytest.raises(ValueError, match="Iterable is empty"):
        _process_chunks(
            readers=readers,
            func=func,
            requested_types=(pd.DataFrame,),
            static_args=[],
            static_kwargs={},
            non_data_output="first",
            non_data_proc=None,
            non_data_proc_args=(),
            non_data_proc_kwargs={},
        )


def test_invalid_type_raises():
    readers = [make_reader(["not_df"])]

    def func(x):
        return x

    with pytest.raises(TypeError):
        _process_chunks(
            readers=readers,
            func=func,
            requested_types=(pd.DataFrame,),
            static_args=[],
            static_kwargs={},
            non_data_output="first",
            non_data_proc=None,
            non_data_proc_args=(),
            non_data_proc_kwargs={},
        )


def test_multiple_readers():
    r1 = make_reader([df(1), df(2)])
    r2 = make_reader([df(10), df(20)])

    def func(x, y):
        return x + y

    result = _process_chunks(
        readers=[r1, r2],
        func=func,
        requested_types=(pd.DataFrame,),
        static_args=[],
        static_kwargs={},
        non_data_output="first",
        non_data_proc=None,
        non_data_proc_args=(),
        non_data_proc_kwargs={},
    )

    data_reader = result[0]
    output = data_reader.read()

    assert output["a"].tolist() == [11, 22]


def make_reader_2(values=None):
    if values is None:
        values = []
    return ParquetStreamReader(lambda: iter(values))


def make_df_2(val):
    return pd.DataFrame({"a": [val]})


def test_base_reader_only():
    base = make_reader_2([make_df_2(1)])

    readers, args, kwargs = _prepare_readers(
        reader=base,
        func_args=[],
        func_kwargs={},
        makecopy=False,
    )

    assert readers == [base]
    assert args == []
    assert kwargs == {}


@pytest.mark.parametrize(
    "func_args,expected_reader_count,expected_static_len",
    [
        ([], 1, 0),
        ([123], 1, 1),
        ([make_reader_2()], 2, 0),
        ([make_reader_2(), 999], 2, 1),
    ],
)
def test_func_args_separation(func_args, expected_reader_count, expected_static_len):
    base = make_reader_2([make_df_2(1)])

    readers, args, kwargs = _prepare_readers(
        reader=base,
        func_args=func_args,
        func_kwargs={},
        makecopy=False,
    )

    assert len(readers) == expected_reader_count
    assert len(args) == expected_static_len
    assert kwargs == {}


def test_func_kwargs_separation():
    base = make_reader_2([make_df_2(1)])
    reader_kw = make_reader_2([make_df_2(2)])

    readers, args, kwargs = _prepare_readers(
        reader=base,
        func_args=[],
        func_kwargs={"r": reader_kw, "x": 42},
        makecopy=False,
    )

    assert len(readers) == 2
    assert args == []
    assert kwargs == {"x": 42}


def test_reader_ordering():
    base = make_reader_2()
    r1 = make_reader_2()
    r2 = make_reader_2()

    readers, _, _ = _prepare_readers(
        reader=base,
        func_args=[r1],
        func_kwargs={"k": r2},
        makecopy=False,
    )

    assert readers[0] is base
    assert readers[1] is r1
    assert readers[2] is r2


def test_makecopy_false_preserves_identity():
    base = make_reader_2()
    r1 = make_reader_2()

    readers, _, _ = _prepare_readers(
        reader=base,
        func_args=[r1],
        func_kwargs={},
        makecopy=False,
    )

    assert readers[0] is base
    assert readers[1] is r1


def test_makecopy_true_creates_copies():
    base = make_reader_2([make_df_2(1)])
    r1 = make_reader_2([make_df_2(2)])

    readers, _, _ = _prepare_readers(
        reader=base,
        func_args=[r1],
        func_kwargs={},
        makecopy=True,
    )

    # Copies should not be the same object
    assert readers[0] is not base
    assert readers[1] is not r1

    # But should behave identically
    assert readers[0].read()["a"].tolist() == [1]
    assert readers[1].read()["a"].tolist() == [2]


def test_empty_args_and_kwargs():
    base = make_reader_2()

    readers, args, kwargs = _prepare_readers(
        reader=base,
        func_args=[],
        func_kwargs={},
        makecopy=False,
    )

    assert readers == [base]
    assert args == []
    assert kwargs == {}


def make_df_3(val):
    return pd.DataFrame({"a": [val]})


def make_series_3(val, name="s"):
    return pd.Series([val], name=name)


def reader_from_list(items):
    return iter(items)


@pytest.mark.parametrize(
    "input_data,requested_types",
    [
        ([make_df_3(1), make_df_3(2)], (pd.DataFrame,)),
        ([make_series_3(10), make_series_3(20)], (pd.Series,)),
    ],
)
def test_basic_processing(input_data, requested_types):
    def func(x):
        return x

    result = process_disk_backed(
        reader=reader_from_list(input_data),
        func=func,
        requested_types=requested_types,
    )

    # First element is a generator
    gen = result[0]

    output = list(gen)
    assert all(isinstance(o, requested_types) for o in output)

    if isinstance(output[0], pd.DataFrame):
        assert [row["a"].iloc[0] for row in output] == [
            df["a"].iloc[0] for df in input_data if isinstance(df, pd.DataFrame)
        ]
    else:
        assert [o.iloc[0] for o in output] == [
            s.iloc[0] for s in input_data if isinstance(s, pd.Series)
        ]


def test_non_data_first_mode():
    def func(df):
        return df, df["a"].iloc[0]

    result = process_disk_backed(
        reader=reader_from_list([make_df_3(1), make_df_3(2)]),
        func=func,
        non_data_output="first",
    )

    gen, meta = result

    # Only first chunk captured
    assert meta == 1
    output = list(gen)
    assert [row["a"].iloc[0] for row in output] == [1, 2]


def test_non_data_acc_mode():
    def func(df):
        return df, df["a"].iloc[0]

    result = process_disk_backed(
        reader=reader_from_list([make_df_3(1), make_df_3(2)]),
        func=func,
        non_data_output="acc",
    )

    gen, meta = result
    assert meta == [1, 2]

    output = list(gen)
    assert [row["a"].iloc[0] for row in output] == [1, 2]


def test_non_data_proc_applied_function():
    def func(df):
        return df, df["a"].iloc[0]

    def processor(meta, factor):
        return [x * factor for x in meta]

    result = process_disk_backed(
        reader=reader_from_list([make_df_3(1), make_df_3(2)]),
        func=func,
        non_data_output="acc",
        non_data_proc=processor,
        non_data_proc_args=(10,),
        non_data_proc_kwargs={},
    )

    gen, meta = result
    assert meta == [10, 20]

    output = list(gen)
    assert [row["a"].iloc[0] for row in output] == [1, 2]


def test_func_args_kwargs():
    def func(df, val, extra=0):
        return df * val + extra

    result = process_disk_backed(
        reader=reader_from_list([make_df_3(1), make_df_3(2)]),
        func=func,
        func_args=[2],
        func_kwargs={"extra": 5},
    )

    gen = result[0]
    output = list(gen)
    assert [row["a"].iloc[0] for row in output] == [1 * 2 + 5, 2 * 2 + 5]


def test_empty_iterator_raises():
    def func(x):
        return x

    with pytest.raises(ValueError, match="Iterable is empty"):
        process_disk_backed(
            reader=reader_from_list([]),
            func=func,
        )


def test_requested_types_single_type():
    def func(x):
        return x

    input_data = [make_df_3(1)]
    # requested_types as single type
    result = process_disk_backed(
        reader=reader_from_list(input_data),
        func=func,
        requested_types=pd.DataFrame,
    )

    gen = result[0]
    output = list(gen)
    assert all(isinstance(o, pd.DataFrame) for o in output)


def test_parquet_stream_from_iterable_dataframe():
    dfs = [make_df_3(1), make_df_3(2)]
    reader = parquet_stream_from_iterable(dfs)

    assert isinstance(reader, ParquetStreamReader)
    output = list(reader)
    assert all(isinstance(df, pd.DataFrame) for df in output)
    assert [df["a"].iloc[0] for df in output] == [1, 2]


def test_parquet_stream_from_iterable_series():
    series_list = [make_series_3(10), make_series_3(20)]
    reader = parquet_stream_from_iterable(series_list)

    assert isinstance(reader, ParquetStreamReader)
    output = list(reader)
    assert all(isinstance(s, pd.Series) for s in output)
    assert [s.iloc[0] for s in output] == [10, 20]


def test_parquet_stream_from_iterable_empty_raises():
    with pytest.raises(ValueError, match="Iterable is empty"):
        parquet_stream_from_iterable([])


def test_parquet_stream_from_iterable_mixed_types_raises():
    dfs = [make_df_3(1), make_series_3(2)]
    with pytest.raises(TypeError, match="All chunks must be of the same type"):
        parquet_stream_from_iterable(dfs)


def test_parquet_stream_from_iterable_wrong_type_first_raises():
    with pytest.raises(
        TypeError, match="Iterable must contain pd.DataFrame or pd.Series"
    ):
        parquet_stream_from_iterable([123, 456])


def test_ensure_parquet_reader_returns_existing_reader():
    reader = parquet_stream_from_iterable([make_df_3(1)])
    result = ensure_parquet_reader(reader)
    assert result is reader


def test_ensure_parquet_reader_converts_iterator():
    dfs = [make_df_3(1), make_df_3(2)]
    iterator = iter(dfs)
    result = ensure_parquet_reader(iterator)
    assert isinstance(result, ParquetStreamReader)
    output = list(result)
    assert [df["a"].iloc[0] for df in output] == [1, 2]


def test_ensure_parquet_reader_returns_non_iterator_unchanged():
    obj = 123
    result = ensure_parquet_reader(obj)
    assert result == 123


@pytest.mark.parametrize(
    "value,expected",
    [
        (iter([1, 2, 3]), True),  # iterator
        ((x for x in range(5)), True),  # generator expression
        ([1, 2, 3], False),  # list
        ((1, 2, 3), False),  # tuple
        (123, False),  # int
        ("abc", False),  # string
        (None, False),  # None
    ],
)
def test_is_valid_iterator(value, expected):
    assert is_valid_iterator(value) is expected


def test_non_process_function_returns():
    val = 123
    assert _process_function(val) == val


def test_dataframe_calls_func_directly():
    df = make_df_3(5)

    called = {}

    def func(d):
        called["data"] = d
        return d["a"].iloc[0] * 2

    pf = ProcessFunction(df, func)
    result = _process_function(pf)

    assert result == 10
    assert called["data"] is df


def test_series_calls_func_directly():
    s = make_series_3(7)

    def func(x):
        return x.iloc[0] + 3

    pf = ProcessFunction(s, func)
    result = _process_function(pf)
    assert result == 10


def test_xarray_dataset_direct_call():
    ds = xr.Dataset({"a": ("x", [1, 2])})

    def func(x):
        return x["a"].sum().item()

    pf = ProcessFunction(ds, func)
    result = _process_function(pf)
    assert result == 3


def test_iterator_of_dataframes_disk_backed():
    dfs = [make_df_3(1), make_df_3(2)]
    it = iter(dfs)

    def func(df):
        return df["a"].iloc[0] * 10

    pf = ProcessFunction(it, func, non_data_output="acc")
    result = _process_function(pf)
    assert result == [10, 20]


def test_list_of_dataframes_disk_backed():
    dfs = [make_df_3(3), make_df_3(4)]

    def func(df):
        return df["a"].iloc[0] * 2

    pf = ProcessFunction(dfs, func, non_data_output="acc")
    result = _process_function(pf)
    assert result == [6, 8]


def test_data_only_returns_first():
    dfs = [make_df_3(1)]
    pf = ProcessFunction(dfs, lambda df: df)
    result = _process_function(pf, data_only=True)
    assert isinstance(result, ParquetStreamReader)


def test_unsupported_type_raises():
    pf = ProcessFunction(12345, lambda x: x)
    with pytest.raises(TypeError, match="Unsupported data type"):
        _process_function(pf)


def test_basic_dataframe_decorator():
    @process_function()
    def func(df):
        return df * 2

    df = make_df_3(3)
    result = func(df)
    assert isinstance(result, pd.DataFrame)
    assert result["a"].iloc[0] == 6


def test_iterable_returns_disk_backed():
    @process_function()
    def func(dfs):
        return dfs

    dfs = [make_df_3(1), make_df_3(2)]
    result = func(dfs)

    assert isinstance(result, list)
    assert len(result) == 2

    pd.testing.assert_frame_equal(result[0], pd.DataFrame({"a": [1]}))
    pd.testing.assert_frame_equal(result[1], pd.DataFrame({"a": [2]}))


def test_data_only_returns_generator_only():
    @process_function(data_only=True)
    def func(dfs):
        return dfs

    dfs = [make_df_3(1)]
    result = func(dfs)

    assert isinstance(result, list)
    assert len(result) == 1

    pd.testing.assert_frame_equal(result[0], pd.DataFrame({"a": [1]}))


def test_postprocessing_not_callable_raises():
    @process_function(postprocessing={"func": 123, "kwargs": []})
    def func(df):
        return df

    df = make_df_3(1)
    with pytest.raises(ValueError, match="is not callable"):
        func(df)
