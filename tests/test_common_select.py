from __future__ import annotations

import pandas as pd
import pytest

from cdm_reader_mapper.common.iterators import ParquetStreamReader
from cdm_reader_mapper.common.select import (
    _split_by_boolean_df,
    _split_by_column_df,
    _split_by_index_df,
    _split_df,
    split_by_boolean,
    split_by_boolean_false,
    split_by_boolean_true,
    split_by_column_entries,
    split_by_index,
)


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
def sample_psr():
    df1 = pd.DataFrame({"A": [1, 2], "B": ["x", "y"], "C": [True, False]}, index=[10, 11])
    df2 = pd.DataFrame({"A": [3], "B": ["z"], "C": [True]}, index=[12])
    df3 = pd.DataFrame({"A": [4, 5], "B": ["x", "y"], "C": [False, True]}, index=[13, 14])
    return ParquetStreamReader([df1, df2, df3])


@pytest.fixture
def sample_df_multi():
    return pd.DataFrame(
        {
            ("A", "a"): [1, 2, 3, 4, 5],
            ("B", "b"): ["x", "y", "z", "x", "y"],
            ("C", "c"): [True, False, True, False, True],
        },
        index=[10, 11, 12, 13, 14],
    )


@pytest.fixture
def sample_psr_multi():
    df1 = pd.DataFrame(
        {("A", "a"): [1, 2], ("B", "b"): ["x", "y"], ("C", "c"): [True, False]},
        index=[10, 11],
    )
    df2 = pd.DataFrame({("A", "a"): [3], ("B", "b"): ["z"], ("C", "c"): [True]}, index=[12])
    df3 = pd.DataFrame(
        {("A", "a"): [4, 5], ("B", "b"): ["x", "y"], ("C", "c"): [False, True]},
        index=[13, 14],
    )
    return ParquetStreamReader([df1, df2, df3])


@pytest.fixture
def empty_df():
    return pd.DataFrame(columns=["A", "B", "C"])


@pytest.fixture
def empty_psr():
    return ParquetStreamReader([pd.DataFrame(columns=["A", "B", "C"])])


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


def test_split_df_multiindex(sample_df_multi):
    mask = pd.Series([True, False, False, True, False], index=sample_df_multi.index)
    selected, rejected, _, _ = _split_df(sample_df_multi, mask, return_rejected=True)
    assert list(selected.index) == [10, 13]
    assert list(rejected.index) == [11, 12, 14]


@pytest.mark.parametrize(
    "column,boolean,expected_selected,expected_rejected",
    [
        ("C", True, [10, 12, 14], [11, 13]),
        ("C", False, [11, 13], [10, 12, 14]),
    ],
)
def test_split_by_boolean_df(sample_df, column, boolean, expected_selected, expected_rejected):
    mask = sample_df[[column]]
    selected, rejected, _, _ = _split_by_boolean_df(sample_df, mask, boolean=boolean, return_rejected=True)
    assert list(selected.index) == expected_selected
    assert list(rejected.index) == expected_rejected


def test_split_by_boolean_df_empty_mask(sample_df):
    mask = pd.DataFrame(columns=sample_df.columns)
    selected, rejected, _, _ = _split_by_boolean_df(sample_df, mask, boolean=True, return_rejected=True)
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
def test_split_by_column_df(sample_df, col, values, return_rejected, expected_selected, expected_rejected):
    selected, rejected, _, _ = _split_by_column_df(sample_df, col, values, return_rejected=return_rejected)
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
def test_split_by_index_helper(
    sample_df,
    index_list,
    inverse,
    return_rejected,
    expected_selected,
    expected_rejected,
):
    selected, rejected, _, _ = _split_by_index_df(sample_df, index_list, inverse=inverse, return_rejected=return_rejected)
    assert list(selected.index) == expected_selected
    assert list(rejected.index) == expected_rejected


def test_split_by_index_df(sample_df):
    selected, rejected, _, _ = split_by_index(sample_df, [11, 13], return_rejected=True)

    assert list(selected.index) == [11, 13]
    assert list(rejected.index) == [10, 12, 14]


def test_split_by_index_prs(sample_psr):
    selected, rejected, _, _ = split_by_index(sample_psr, [11, 13], return_rejected=True)

    selected = selected.read()
    rejected = rejected.read()

    assert list(selected.index) == [11, 13]
    assert list(rejected.index) == [10, 12, 14]


def test_split_by_index_multiindex_df(sample_df_multi):
    selected, rejected, _, _ = split_by_index(sample_df_multi, [11, 13], return_rejected=True)

    assert list(selected.index) == [11, 13]
    assert list(rejected.index) == [10, 12, 14]


def test_split_by_index_multiindex_psr(sample_psr_multi):
    selected, rejected, _, _ = split_by_index(sample_psr_multi, [11, 13], return_rejected=True)

    selected = selected.read()
    rejected = rejected.read()

    assert list(selected.index) == [11, 13]
    assert list(rejected.index) == [10, 12, 14]


def test_split_by_column_entries_df(sample_df):
    selected, rejected, _, _ = split_by_column_entries(sample_df, {"B": ["y"]}, return_rejected=True)

    assert list(selected.index) == [11, 14]
    assert list(rejected.index) == [10, 12, 13]


def test_split_by_column_entries_psr(sample_psr):
    selected, rejected, _, _ = split_by_column_entries(sample_psr, {"B": ["y"]}, return_rejected=True)

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
def test_split_by_boolean_df_true(
    sample_df,
    boolean_mask,
    inverse,
    reset_index,
    exp_selected_idx,
    exp_rejected_idx,
):
    selected, rejected, _, _ = split_by_boolean(
        sample_df,
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
def test_split_by_boolean_psr_true(
    sample_psr,
    boolean_mask,
    inverse,
    reset_index,
    exp_selected_idx,
    exp_rejected_idx,
):
    selected, rejected, _, _ = split_by_boolean(
        sample_psr.copy(),
        boolean_mask,
        boolean=True,
        inverse=inverse,
        reset_index=reset_index,
        return_rejected=True,
    )

    exp = sample_psr.copy().read()
    exp_selected = exp.loc[exp_selected_idx]
    exp_rejected = exp.loc[exp_rejected_idx]

    if reset_index is True:
        exp_selected = exp_selected.reset_index(drop=True)
        exp_rejected = exp_rejected.reset_index(drop=True)

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
def test_split_by_boolean_df_false(
    sample_df,
    boolean_mask,
    inverse,
    reset_index,
    exp_selected_idx,
    exp_rejected_idx,
):
    selected, rejected, _, _ = split_by_boolean(
        sample_df,
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
def test_split_by_boolean_psr_false(
    sample_psr,
    boolean_mask,
    inverse,
    reset_index,
    exp_selected_idx,
    exp_rejected_idx,
):
    selected, rejected, _, _ = split_by_boolean(
        sample_psr.copy(),
        boolean_mask,
        boolean=False,
        inverse=inverse,
        reset_index=reset_index,
        return_rejected=True,
    )

    exp = sample_psr.copy().read()
    exp_selected = exp.loc[exp_selected_idx]
    exp_rejected = exp.loc[exp_rejected_idx]

    if reset_index is True:
        exp_selected = exp_selected.reset_index(drop=True)
        exp_rejected = exp_rejected.reset_index(drop=True)

    selected = selected.read()
    rejected = rejected.read()

    pd.testing.assert_frame_equal(selected, exp_selected)
    pd.testing.assert_frame_equal(rejected, exp_rejected)


def test_split_by_boolean_true_df(sample_df, boolean_mask_true):
    selected, rejected, _, _ = split_by_boolean_true(sample_df, boolean_mask_true, return_rejected=True)

    assert list(selected.index) == [10]
    assert list(rejected.index) == [11, 12, 13, 14]


def test_split_by_boolean_true_psr(sample_psr, boolean_mask_true):
    selected, rejected, _, _ = split_by_boolean_true(sample_psr, boolean_mask_true, return_rejected=True)

    selected = selected.read()
    rejected = rejected.read()

    assert list(selected.index) == [10]
    assert list(rejected.index) == [11, 12, 13, 14]


def test_split_by_boolean_false_df(sample_df, boolean_mask):
    selected, rejected, _, _ = split_by_boolean_false(sample_df, boolean_mask, return_rejected=True)

    assert list(selected.index) == []
    assert list(rejected.index) == [10, 11, 12, 13, 14]


def test_split_by_boolean_false_psr(sample_psr, boolean_mask):
    selected, rejected, _, _ = split_by_boolean_false(sample_psr, boolean_mask, return_rejected=True)

    selected = selected.read()
    rejected = rejected.read()

    assert list(selected.index) == []
    assert list(rejected.index) == [10, 11, 12, 13, 14]


def test_split_by_index_empty_df(empty_df):
    selected, rejected, _, _ = split_by_index(empty_df, [0, 1], return_rejected=True)

    assert selected.empty
    assert rejected.empty


def test_split_by_index_empty_psr(empty_psr):
    selected, rejected, _, _ = split_by_index(empty_psr, [0, 1], return_rejected=True)

    selected = selected.read()
    rejected = rejected.read()

    assert selected.empty
    assert rejected.empty


def test_split_by_column_empty_df(empty_df):
    selected, rejected, _, _ = split_by_column_entries(empty_df, {"A": [1]}, return_rejected=True)

    assert selected.empty
    assert rejected.empty


def test_split_by_column_empty_psr(empty_psr):
    selected, rejected, _, _ = split_by_column_entries(empty_psr, {"A": [1]}, return_rejected=True)

    selected = selected.read()
    rejected = rejected.read()

    assert selected.empty
    assert rejected.empty


def test_split_by_boolean_empty_df(empty_df):
    mask = pd.DataFrame(columns=["A", "B", "C"], dtype=bool)
    selected, rejected, _, _ = split_by_boolean(empty_df, mask, boolean=True, return_rejected=True)

    assert selected.empty
    assert rejected.empty


def test_split_by_boolean_empty_psr(empty_psr):
    mask = ParquetStreamReader([pd.DataFrame(columns=["A", "B", "C"], dtype=bool)])
    selected, rejected, _, _ = split_by_boolean(empty_psr, mask, boolean=True, return_rejected=True)

    selected = selected.read()
    rejected = rejected.read()

    assert selected.empty
    assert rejected.empty
