from __future__ import annotations

import pandas as pd
import pytest

from cdm_reader_mapper import DataBundle


@pytest.fixture
def sample_df():
    """Simple DataFrame for testing."""
    data = pd.DataFrame(
        {
            "A": [19, 26, 27, 41, 91],
            "B": [0, 1, 2, 3, 4],
        }
    )
    mask = pd.DataFrame(
        {
            "A": [True, True, True, False, True],
            "B": [True, True, True, False, False],
        }
    )
    return DataBundle(data=data, mask=mask)


@pytest.mark.parametrize(
    "func, args, idx_exp, idx_rej",
    [
        ("select_where_all_true", [], [0, 1, 2], [3, 4]),
        ("select_where_all_false", [], [3], [0, 1, 2, 4]),
        ("select_where_index_isin", [[0, 2, 4]], [0, 2, 4], [1, 3]),
        ("select_where_entry_isin", [{"A": [26, 41]}], [1, 3], [0, 2, 4]),
    ],
)
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_select_operators(
    sample_df,
    func,
    args,
    idx_exp,
    idx_rej,
    reset_index,
    inverse,
):
    result = getattr(sample_df, func)(*args, reset_index=reset_index, inverse=inverse)

    expected = sample_df.data
    expected_mask = sample_df.mask
    selected = result.data
    selected_mask = result.mask

    if inverse is False:
        idx = expected.index.isin(idx_exp)
    else:
        idx = expected.index.isin(idx_rej)

    expected = expected[idx]
    expected_mask = expected_mask[idx]

    if reset_index is True:
        expected = expected.reset_index(drop=True)
        expected_mask = expected_mask.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected, selected)
    pd.testing.assert_frame_equal(expected_mask, selected_mask)


@pytest.mark.parametrize(
    "func, args, idx_exp, idx_rej",
    [
        ("split_by_boolean_true", [], [0, 1, 2], [3, 4]),
        ("split_by_boolean_false", [], [3], [0, 1, 2, 4]),
        ("split_by_index", [[0, 2, 4]], [0, 2, 4], [1, 3]),
        ("split_by_column_entries", [{"A": [26, 41]}], [1, 3], [0, 2, 4]),
    ],
)
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_split_operators(
    sample_df,
    func,
    args,
    idx_exp,
    idx_rej,
    reset_index,
    inverse,
):
    result = getattr(sample_df, func)(*args, reset_index=reset_index, inverse=inverse)

    expected = sample_df.data
    expected_mask = sample_df.mask
    selected = result[0].data
    selected_mask = result[0].mask
    rejected = result[1].data
    rejected_mask = result[1].mask

    if inverse is False:
        idx1 = expected.index.isin(idx_exp)
        idx2 = expected.index.isin(idx_rej)
    else:
        idx1 = expected.index.isin(idx_rej)
        idx2 = expected.index.isin(idx_exp)

    expected1 = expected[idx1]
    expected2 = expected[idx2]
    expected_mask1 = expected_mask[idx1]
    expected_mask2 = expected_mask[idx2]

    if reset_index is True:
        expected1 = expected1.reset_index(drop=True)
        expected2 = expected2.reset_index(drop=True)
        expected_mask1 = expected_mask1.reset_index(drop=True)
        expected_mask2 = expected_mask2.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected1, selected)
    pd.testing.assert_frame_equal(expected2, rejected)
    pd.testing.assert_frame_equal(expected_mask1, selected_mask)
    pd.testing.assert_frame_equal(expected_mask2, rejected_mask)


def test_inspect_count_by_cat(sample_df):
    result = sample_df.unique(columns=("A"))
    assert result == {"A": {19: 1, 26: 1, 27: 1, 41: 1, 91: 1}}


def test_replace_columns(sample_df):
    df_corr = pd.DataFrame(
        {
            "A_new": [101, 201, 301, 401, 501],
            "B": range(5),
        }
    )
    result = sample_df.replace_columns(
        df_corr,
        subset=["B", "A"],
        rep_map={"A": "A_new"},
        pivot_l="B",
        pivot_r="B",
    )
    expected = pd.DataFrame(
        {
            "A": [101, 201, 301, 401, 501],
            "B": [0, 1, 2, 3, 4],
        }
    )

    pd.testing.assert_frame_equal(result.data, expected)
