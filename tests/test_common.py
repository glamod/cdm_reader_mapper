from __future__ import annotations

import pytest
import json
import logging
import sys
import pandas as pd

from io import StringIO


from cdm_reader_mapper.common.select import (
    _select_rows_by_index,
    _split_by_index,
    _ensure_empty_df_consistent,
    _split_by_boolean_mask,
    _split_by_column_values,
    _split_by_index_values,
    _split,
    split_by_boolean,
    split_by_boolean_true,
    split_by_boolean_false,
    split_by_column_entries,
    split_by_index,
)
from cdm_reader_mapper.common.replace import replace_columns
from cdm_reader_mapper.common.pandas_TextParser_hdlr import (
    make_copy,
    restore,
    is_not_empty,
    get_length,
)
from cdm_reader_mapper.common.logging_hdlr import init_logger
from cdm_reader_mapper.common.json_dict import (
    open_json_file,
    collect_json_files,
    combine_dicts,
)


def make_parser(text, **kwargs):
    """Helper: create a TextFileReader similar to user code."""
    buffer = StringIO(text)
    return pd.read_csv(buffer, chunksize=2, **kwargs)


def make_broken_parser(text: str):
    """Return a pandas TextFileReader that will fail in make_copy."""
    parser = pd.read_csv(StringIO(text), chunksize=2)
    parser.handles.handle = None
    return parser


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
def empty_df():
    return pd.DataFrame(columns=["A", "B", "C"])


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


@pytest.mark.parametrize(
    "index_list,inverse,reset_index,expected",
    [
        ([10, 12], False, False, [10, 12]),
        ([10, 12], True, False, [11, 13, 14]),
        ([10, 12], False, True, [0, 1]),
    ],
)
def test_select_rows_by_index(sample_df, index_list, inverse, reset_index, expected):
    df = sample_df
    selected = _select_rows_by_index(
        df, index_list, inverse=inverse, reset_index=reset_index
    )
    assert list(selected.index) == expected


def test_select_rows_by_index_empty_df(empty_df):
    selected = _select_rows_by_index(empty_df, [0])
    assert selected.empty


@pytest.mark.parametrize(
    "index_list,inverse,return_rejected,expected_selected,expected_rejected",
    [
        ([11, 13], False, True, [11, 13], [10, 12, 14]),
        ([11, 13], True, True, [10, 12, 14], [11, 13]),
        ([11, 13], False, False, [11, 13], []),
    ],
)
def test_split_by_index(
    sample_df,
    index_list,
    inverse,
    return_rejected,
    expected_selected,
    expected_rejected,
):
    selected, rejected = _split_by_index(
        sample_df, index_list, inverse=inverse, return_rejected=return_rejected
    )
    assert list(selected.index) == expected_selected
    assert list(rejected.index) == expected_rejected


def test_split_by_index_empty_df(empty_df):
    selected, rejected = _split_by_index(empty_df, [0], return_rejected=True)
    assert selected.empty
    assert rejected.empty


def test_ensure_empty_df_consistent(sample_df):
    empty = pd.DataFrame(columns=sample_df.columns)
    result = _ensure_empty_df_consistent(empty, sample_df)
    for col in sample_df.columns:
        assert result[col].dtype == sample_df[col].dtype
    assert "_prev_index" in result.__dict__


def test_ensure_empty_df_consistent_non_empty(sample_df):
    result = _ensure_empty_df_consistent(sample_df, sample_df)
    assert result.equals(sample_df)


@pytest.mark.parametrize(
    "column,boolean,expected_selected,expected_rejected",
    [
        ("C", True, [10, 12, 14], [11, 13]),
        ("C", False, [11, 13], [10, 12, 14]),
    ],
)
def test_split_by_boolean_mask(
    sample_df, column, boolean, expected_selected, expected_rejected
):
    mask = sample_df[[column]]
    selected, rejected = _split_by_boolean_mask(
        sample_df, mask, boolean=boolean, return_rejected=True
    )
    assert list(selected.index) == expected_selected
    assert list(rejected.index) == expected_rejected


def test_split_by_boolean_mask_empty_mask(sample_df):
    mask = pd.DataFrame(columns=sample_df.columns)
    selected, rejected = _split_by_boolean_mask(
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
def test_split_by_column_values(
    sample_df, col, values, return_rejected, expected_selected, expected_rejected
):
    selected, rejected = _split_by_column_values(
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
def test_split_by_index_values(
    sample_df,
    index_list,
    inverse,
    return_rejected,
    expected_selected,
    expected_rejected,
):
    selected, rejected = _split_by_index_values(
        sample_df, index_list, inverse=inverse, return_rejected=return_rejected
    )
    assert list(selected.index) == expected_selected
    assert list(rejected.index) == expected_rejected


def test_split_wrapper_index(sample_df):
    selected, rejected = _split(
        sample_df, _split_by_index_values, [11, 13], return_rejected=True
    )
    assert list(selected.index) == [11, 13]
    assert list(rejected.index) == [10, 12, 14]


def test_split_wrapper_column(sample_df):
    selected, rejected = _split(
        sample_df, _split_by_column_values, "B", ["y"], return_rejected=True
    )
    assert list(selected.index) == [11, 14]
    assert list(rejected.index) == [10, 12, 13]


def test_split_wrapper_boolean(sample_df, boolean_mask):
    selected, rejected = _split(
        sample_df,
        _split_by_boolean_mask,
        boolean_mask[["mask1"]],
        True,
        return_rejected=True,
    )
    assert list(selected.index) == [11, 13]
    assert list(rejected.index) == [10, 12, 14]


def test_split_by_index_public(sample_df):
    selected, rejected = split_by_index(sample_df, [11, 13], return_rejected=True)
    assert list(selected.index) == [11, 13]
    assert list(rejected.index) == [10, 12, 14]


def test_split_by_column_entries_public(sample_df):
    selected, rejected = split_by_column_entries(
        sample_df, {"B": ["y"]}, return_rejected=True
    )
    assert list(selected.index) == [11, 14]
    assert list(rejected.index) == [10, 12, 13]


def test_split_by_boolean_public(sample_df, boolean_mask):
    selected, rejected = split_by_boolean(
        sample_df, boolean_mask, boolean=False, return_rejected=True
    )
    assert list(selected.index) == [10, 11, 12, 13, 14]
    assert rejected.empty
    selected, rejected = split_by_boolean(
        sample_df, boolean_mask, boolean=True, return_rejected=True
    )
    assert selected.empty
    assert list(rejected.index) == [10, 11, 12, 13, 14]


def test_split_by_boolean_true_public(sample_df, boolean_mask_true):
    selected, rejected = split_by_boolean_true(
        sample_df, boolean_mask_true, return_rejected=True
    )
    assert list(selected.index) == [10]
    assert list(rejected.index) == [11, 12, 13, 14]


def test_split_by_boolean_false_public(sample_df, boolean_mask):
    selected, rejected = split_by_boolean_false(
        sample_df, boolean_mask, return_rejected=True
    )
    assert list(selected.index) == [10, 11, 12, 13, 14]
    assert rejected.empty


def test_split_by_index_empty(empty_df):
    selected, rejected = split_by_index(empty_df, [0, 1], return_rejected=True)
    assert selected.empty
    assert rejected.empty


def test_split_by_column_empty(empty_df):
    selected, rejected = split_by_column_entries(
        empty_df, {"A": [1]}, return_rejected=True
    )
    assert selected.empty
    assert rejected.empty


def test_split_by_boolean_empty(empty_df):
    mask = empty_df.astype(bool)
    selected, rejected = split_by_boolean(
        empty_df, mask, boolean=True, return_rejected=True
    )
    assert selected.empty
    assert rejected.empty


def test_basic_replacement():
    df_l = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
    df_r = pd.DataFrame({"id": [1, 2], "x": [100, 200]})

    out = replace_columns(df_l, df_r, pivot_c="id", rep_c="x")
    assert out["x"].tolist() == [100, 200]


def test_rep_map_different_names():
    df_l = pd.DataFrame({"id": [1, 2], "a": [1, 2]})
    df_r = pd.DataFrame({"id": [1, 2], "b": [10, 20]})

    out = replace_columns(df_l, df_r, pivot_c="id", rep_map={"a": "b"})
    assert out["a"].tolist() == [10, 20]


def test_missing_pivot_returns_none():
    df_l = pd.DataFrame({"id": [1]})
    df_r = pd.DataFrame({"id": [1]})

    assert replace_columns(df_l, df_r, rep_c="x") is None


def test_missing_replacement_returns_none():
    df_l = pd.DataFrame({"id": [1]})
    df_r = pd.DataFrame({"id": [1]})

    assert replace_columns(df_l, df_r, pivot_c="id") is None


def test_missing_source_col_returns_none():
    df_l = pd.DataFrame({"id": [1], "a": [10]})
    df_r = pd.DataFrame({"id": [1]})

    assert replace_columns(df_l, df_r, pivot_c="id", rep_map={"a": "missing"}) is None


def test_index_reset():
    df_l = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
    df_r = pd.DataFrame({"id": [1, 2], "x": [100, 200]})

    out = replace_columns(df_l, df_r, pivot_c="id", rep_c="x")
    assert list(out.index) == [0, 1]


def test_make_copy_basic():
    parser = make_parser("a,b\n1,2\n3,4\n")
    cp = make_copy(parser)

    assert cp is not None

    expected = pd.DataFrame({"a": [1, 3], "b": [2, 4]})

    assert cp.get_chunk().equals(expected)
    assert parser.get_chunk().equals(expected)


def test_make_copy_failure_memory():
    parser = make_broken_parser("a,b\n1,2\n")
    cp = make_copy(parser)
    assert cp is None


def test_restore_basic():
    parser = make_parser("a,b\n1,2\n3,4\n")
    parser.get_chunk()

    restored = restore(parser)
    assert restored is not None

    expected = pd.DataFrame({"a": [1, 3], "b": [2, 4]})
    assert restored.get_chunk().equals(expected)


def test_restore_failure_memory():
    parser = make_broken_parser("a,b\n1,2\n")
    restored = restore(parser)
    assert restored is None


def test_is_not_empty_true():
    parser = make_parser("a,b\n1,2\n")
    assert is_not_empty(parser) is True


def test_is_not_empty_false():
    parser = make_parser("a,b\n")
    assert is_not_empty(parser) is False


def test_is_not_empty_failure_make_copy_memory():
    parser = make_broken_parser("a,b\n1,2\n")
    result = is_not_empty(parser)
    assert result is None


def test_get_length_basic():
    parser = make_parser("a,b\n1,2\n3,4\n5,6\n")
    assert get_length(parser) == 3


def test_get_length_empty():
    parser = make_parser("a,b\n")
    assert get_length(parser) == 0


def test_get_length_failure_due_to_bad_line():
    parser = make_parser("a,b\n1,2\n1,2,3\n")
    assert get_length(parser) is None


def test_get_length_failure_make_copy_memory():
    parser = make_broken_parser("a,b\n1,2\n")
    result = get_length(parser)
    assert result is None


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
    sys.modules.pop("idir", None)
    base_pkg = tmp_path / "idir"
    base_pkg.mkdir()
    (base_pkg / "__init__.py").write_text("")
    (base_pkg / "idir.json").write_text(json.dumps({"x": 1}))
    (base_pkg / "idir_release.json").write_text(json.dumps({"y": 2}))

    sys.path.insert(0, str(tmp_path))
    import importlib

    importlib.import_module("idir")

    files = list((base_pkg).glob("*.json"))
    names = [f.name for f in files]

    assert "idir.json" in names
    assert "idir_release.json" in names


def test_collect_json_files_with_args(tmp_path):
    sys.modules.pop("idir", None)
    sys.modules.pop("idir.idir", None)
    sys.modules.pop("idir,idir,release", None)
    base_pkg = tmp_path / "idir"
    base_pkg.mkdir()
    (base_pkg / "__init__.py").write_text("")

    idir_pkg = base_pkg / "idir"
    idir_pkg.mkdir()
    (idir_pkg / "__init__.py").write_text("")
    (idir_pkg / "idir.json").write_text(json.dumps({"x": 1}))

    release_pkg = idir_pkg / "release"
    release_pkg.mkdir()
    (release_pkg / "__init__.py").write_text("")
    (release_pkg / "idir_release.json").write_text(json.dumps({"y": 2}))

    sys.path.insert(0, str(tmp_path))
    import importlib

    importlib.import_module("idir.idir.release")

    files = collect_json_files("idir", "release", base="idir")
    names = [f.name for f in files]

    assert "idir.json" in names
    assert "idir_release.json" in names


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
