from __future__ import annotations

import pytest  # noqa

import pandas as pd
from io import StringIO
from pandas.io.parsers import TextFileReader
from unittest.mock import Mock

from cdm_reader_mapper.common.select import (
    split_dataframe_by_boolean, 
    split_dataframe_by_column_entries,
    split_dataframe_by_index,
    split_parser,
    split,
    split_by_boolean,
    split_by_boolean_true,
    split_by_boolean_false,
    split_by_column_entries,
    split_by_index,
)


@pytest.mark.parametrize(
    "data, mask_data, boolean, return_rejected, expected_selected, expected_rejected",
    [
        # Simple True selection
        (
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            {"A": [True, False, True], "B": [True, True, True]},
            True,
            True,
            pd.DataFrame({"A": [1, 3], "B": [4, 6]}),
            pd.DataFrame({"A": [2], "B": [5]}),
        ),
        # Simple False selection
        (
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            {"A": [True, False, True], "B": [True, True, True]},
            False,
            True,
            pd.DataFrame({"A": [2], "B": [5]}),
            pd.DataFrame({"A": [1, 3], "B": [4, 6]}),
        ),
        # All True, return_rejected=False
        (
            {"A": [1, 2], "B": [3, 4]},
            {"A": [True, True], "B": [True, True]},
            True,
            False,
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            pd.DataFrame({"A": pd.Series(dtype=int), "B": pd.Series(dtype=int)}),
        ),
        # Mask contains NaN
        (
            {"A": [10, 20, 30], "B": [40, 50, 60]},
            {"A": [True, None, False], "B": [True, None, True]},
            True,
            True,
            pd.DataFrame({"A": [10], "B": [40]}),
            pd.DataFrame({"A": [20, 30], "B": [50, 60]}),
        ),
        # Empty mask DataFrame
        (
            {"A": [], "B": []},
            {"A": [], "B": []},
            True,
            True,
            pd.DataFrame({"A": pd.Series(dtype=float), "B": pd.Series(dtype=float)}),
            pd.DataFrame({"A": pd.Series(dtype=float), "B": pd.Series(dtype=float)}),
        ),
    ]
)
def test_split_dataframe_by_boolean(
    data, mask_data, boolean, return_rejected, expected_selected, expected_rejected
):
    df = pd.DataFrame(data)
    mask = pd.DataFrame(mask_data)

    selected, rejected = split_dataframe_by_boolean(
        df, mask, boolean=boolean, return_rejected=return_rejected, reset_index=True
    )

    pd.testing.assert_frame_equal(selected, expected_selected)
    pd.testing.assert_frame_equal(rejected, expected_rejected)
    
    # Check _prev_index attribute exists
    assert hasattr(selected, "_prev_index")
    if return_rejected:
        assert hasattr(rejected, "_prev_index")
        

@pytest.mark.parametrize(
    "data, col, values, return_rejected, expected_selected, expected_rejected",
    [
        # Single value match
        (
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            "A",
            [2],
            True,
            pd.DataFrame({"A": [2], "B": [5]}),
            pd.DataFrame({"A": [1, 3], "B": [4, 6]}),
        ),
        # Multiple values match
        (
            {"A": [1, 2, 3, 4], "B": [10, 20, 30, 40]},
            "A",
            [2, 4],
            True,
            pd.DataFrame({"A": [2, 4], "B": [20, 40]}),
            pd.DataFrame({"A": [1, 3], "B": [10, 30]}),
        ),
        # No matches
        (
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            "A",
            [99],
            True,
            pd.DataFrame({"A": pd.Series(dtype=int), "B": pd.Series(dtype=int)}),
            pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
        ),
        # return_rejected=False
        (
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            "A",
            [1, 3],
            False,
            pd.DataFrame({"A": [1, 3], "B": [4, 6]}),
            pd.DataFrame({"A": pd.Series(dtype=int), "B": pd.Series(dtype=int)}),
        ),
        # Empty DataFrame input
        (
            {"A": [], "B": []},
            "A",
            [1],
            True,
            pd.DataFrame({"A": pd.Series(dtype=float), "B": pd.Series(dtype=float)}),
            pd.DataFrame({"A": pd.Series(dtype=float), "B": pd.Series(dtype=float)}),
        ),
    ]
)
def test_split_dataframe_by_column_entries(
    data, col, values, return_rejected, expected_selected, expected_rejected
):
    df = pd.DataFrame(data)
    selected, rejected = split_dataframe_by_column_entries(
        df, col, values, return_rejected=return_rejected, reset_index=True
    )

    pd.testing.assert_frame_equal(selected, expected_selected)
    pd.testing.assert_frame_equal(rejected, expected_rejected)  


@pytest.mark.parametrize(
    "data, index, return_rejected, expected_selected, expected_rejected",
    [
        # Single index selection
        (
            {"A": [10, 20, 30], "B": [1, 2, 3]},
            [1],
            True,
            pd.DataFrame({"A": [20], "B": [2]}),
            pd.DataFrame({"A": [10, 30], "B": [1, 3]}),
        ),
        # Multiple indices selection
        (
            {"A": [10, 20, 30, 40], "B": [1, 2, 3, 4]},
            [0, 2],
            True,
            pd.DataFrame({"A": [10, 30], "B": [1, 3]}),
            pd.DataFrame({"A": [20, 40], "B": [2, 4]}),
        ),
        # Index not present (empty selection)
        (
            {"A": [10, 20, 30], "B": [1, 2, 3]},
            [99],
            True,
            pd.DataFrame({"A": pd.Series(dtype=int), "B": pd.Series(dtype=int)}),
            pd.DataFrame({"A": [10, 20, 30], "B": [1, 2, 3]}),
        ),
        # return_rejected=False
        (
            {"A": [10, 20, 30], "B": [1, 2, 3]},
            [0, 2],
            False,
            pd.DataFrame({"A": [10, 30], "B": [1, 3]}),
            pd.DataFrame({"A": pd.Series(dtype=int), "B": pd.Series(dtype=int)}),
        ),
        # Empty DataFrame input
        (
            {"A": [], "B": []},
            [0],
            True,
            pd.DataFrame({"A": pd.Series(dtype=float), "B": pd.Series(dtype=float)}),
            pd.DataFrame({"A": pd.Series(dtype=float), "B": pd.Series(dtype=float)}),
        ),
    ]
)
def test_split_dataframe_by_index(
    data, index, return_rejected, expected_selected, expected_rejected
):
    df = pd.DataFrame(data)
    selected, rejected = split_dataframe_by_index(
        df, index, return_rejected=return_rejected, reset_index=True
    )

    pd.testing.assert_frame_equal(selected, expected_selected)
    pd.testing.assert_frame_equal(rejected, expected_rejected)


@pytest.mark.parametrize(
    "index, expected_sel_idx",
    [
        ([0], [0]),
        ([1, 2], [1, 2]),
        ([], []),
    ]
)
def test_split_with_split_dataframe_by_index(index, expected_sel_idx):
    df = pd.DataFrame({"x": [10, 20, 30]})

    sel, rej = split(
        data=[df],
        func=split_dataframe_by_index,
        index=index,
        return_rejected=True,
    )

    assert list(sel.index) == expected_sel_idx
    assert list(rej.index) == [i for i in df.index if i not in expected_sel_idx]

    assert hasattr(sel, "_prev_index")
    assert hasattr(rej, "_prev_index")
    
@pytest.mark.parametrize(
    "values, expected_sel_idx",
    [
        ([1], [0]),
        ([2, 3], [1, 2]),
        ([999], []),
    ]
)
def test_split_with_split_dataframe_by_column_entries(values, expected_sel_idx):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

    sel, rej = split(
        data=[df],
        func=split_dataframe_by_column_entries,
        col="a",
        values=values,
        return_rejected=True,
    )

    assert list(sel.index) == expected_sel_idx

    expected_rej_idx = [i for i in df.index if i not in expected_sel_idx]
    assert list(rej.index) == expected_rej_idx

    assert hasattr(sel, "_prev_index")
    assert hasattr(rej, "_prev_index")


@pytest.mark.parametrize(
    "mask_df, boolean, expected_sel_idx",
    [
        # boolean=True ? all True
        (pd.DataFrame({"m1": [True, False], "m2": [True, True]}), True, [0]),
        # boolean=False ? any False
        (pd.DataFrame({"m1": [True, False], "m2": [True, False]}), False, [1]),
    ]
)
def test_split_with_split_dataframe_by_boolean(mask_df, boolean, expected_sel_idx):
    df = pd.DataFrame({"x": [10, 20]})

    sel, rej = split(
        data=[df, mask_df],
        func=split_dataframe_by_boolean,
        boolean=boolean,
        return_rejected=True,
    )

    assert list(sel.index) == expected_sel_idx

    expected_rej_idx = [i for i in df.index if i not in expected_sel_idx]
    assert list(rej.index) == expected_rej_idx

    assert hasattr(sel, "_prev_index")
    assert hasattr(rej, "_prev_index")
    
@pytest.mark.parametrize(
    "mask_df, boolean, expected_sel_idx",
    [
        # boolean=True ? all True
        (pd.DataFrame({"m1": [True, False], "m2": [True, True]}), True, [0]),

        # boolean=False ? any False
        (pd.DataFrame({"m1": [True, False], "m2": [True, False]}), False, [1]),
    ]
)
def test_split_by_boolean_functional(mask_df, boolean, expected_sel_idx):
    df = pd.DataFrame({"x": [10, 20]})

    sel, rej = split_by_boolean(
        df,
        mask_df,
        boolean,
        return_rejected=True,
    )

    assert list(sel.index) == expected_sel_idx
    expected_rej_idx = [i for i in df.index if i not in expected_sel_idx]
    assert list(rej.index) == expected_rej_idx

    assert hasattr(sel, "_prev_index")
    assert hasattr(rej, "_prev_index")


def test_split_by_boolean_empty_mask_selects_all():
    df = pd.DataFrame({"x": [10, 20]})
    mask = pd.DataFrame()  # empty mask

    sel, rej = split_by_boolean(df, mask, boolean=True, return_rejected=True)

    assert sel.equals(df)
    assert rej.empty
    assert list(rej.columns) == list(df.columns)
    
@pytest.mark.parametrize("boolean", [True, False])
def test_split_by_boolean_empty_selection_dtypes(boolean):
    df = pd.DataFrame({"x": [10, 20]})

    # Construct masks that force an empty selection
    if boolean:
        mask = pd.DataFrame({"m": [False, False]})  # no row is all True
    else:
        mask = pd.DataFrame({"m": [True, True]})  # no row has any False

    sel, rej = split_by_boolean(df, mask, boolean, return_rejected=True)

    # One of them must be empty
    assert sel.empty or rej.empty

    if sel.empty:
        assert list(sel.dtypes) == list(df.dtypes)
    if rej.empty:
        assert list(rej.dtypes) == list(df.dtypes)


def test_split_by_boolean_reset_index():
    df = pd.DataFrame({"x": [10, 20]})
    mask = pd.DataFrame({"m": [True, False]})

    sel, rej = split_by_boolean(df, mask, True, reset_index=True, return_rejected=True)

    assert list(sel.index) == [0]     # reset index
    assert list(rej.index) == [0]     # also reset
    

@pytest.mark.parametrize(
    "mask_df, expected_sel_idx",
    [
        # boolean=True ? select rows where *all* mask columns are True
        (pd.DataFrame({"m1": [True, False], "m2": [True, True]}), [0]),
        (pd.DataFrame({"m1": [True, True], "m2": [True, True]}), [0, 1]),
    ]
)
def test_split_by_boolean_true_functional(mask_df, expected_sel_idx):
    df = pd.DataFrame({"x": [10, 20]})

    sel1, rej1 = split_by_boolean_true(df, mask_df, return_rejected=True)

    # Should match the base function with boolean=True
    sel2, rej2 = split_by_boolean(df, mask_df, True, return_rejected=True)

    assert sel1.equals(sel2)
    assert rej1.equals(rej2)
    assert list(sel1.index) == expected_sel_idx

def test_split_by_boolean_true_empty_mask_selects_all():
    df = pd.DataFrame({"x": [10, 20]})
    mask = pd.DataFrame()  # empty mask

    sel, rej = split_by_boolean_true(df, mask, return_rejected=True)

    assert sel.equals(df)
    assert rej.empty
    assert list(rej.columns) == list(df.columns)


def test_split_by_boolean_true_empty_selection_dtypes():
    df = pd.DataFrame({"x": [10, 20]})

    # A mask that yields no rows where all values are True
    mask = pd.DataFrame({"m1": [False, False]})

    sel, rej = split_by_boolean_true(df, mask, return_rejected=True)

    assert sel.empty or rej.empty

    if sel.empty:
        assert list(sel.dtypes) == list(df.dtypes)
    if rej.empty:
        assert list(rej.dtypes) == list(df.dtypes)


def test_split_by_boolean_true_reset_index():
    df = pd.DataFrame({"x": [10, 20]})
    mask = pd.DataFrame({"m": [True, False]})

    sel, rej = split_by_boolean_true(
        df,
        mask,
        reset_index=True,
        return_rejected=True,
    )

    assert list(sel.index) == [0]
    assert list(rej.index) == [0]

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [10, 20, 30, 40]
    })

@pytest.fixture
def sample_mask():
    return pd.DataFrame({
        "A": [True, False, True, True],
        "B": [True, True, False, True]
    })

@pytest.mark.parametrize(
    "func, boolean, expected_selected_indices",
    [
        (split_by_boolean_true, True, [0, 3]),  # only rows where all True
        (split_by_boolean_false, False, [1, 2])  # rows where any False
    ]
)
@pytest.mark.parametrize(
    "return_rejected", [False, True]
)
def test_split_by_boolean(sample_df, sample_mask, func, boolean, expected_selected_indices, return_rejected):
    selected, rejected = func(
        sample_df,
        sample_mask,
        reset_index=True,
        return_rejected=return_rejected
    )

    # Check selected rows
    assert list(selected.index) == list(range(len(expected_selected_indices)))  # reset_index=True
    assert selected.shape[0] == len(expected_selected_indices)

    # Check _prev_index exists
    assert hasattr(selected, "_prev_index")
    if return_rejected:
        assert hasattr(rejected, "_prev_index")
        assert rejected.shape[0] + selected.shape[0] == sample_df.shape[0]
    else:
        # rejected should be empty
        assert rejected.empty
        assert all(col in rejected.columns for col in sample_df.columns)
        
@pytest.mark.parametrize(
    "selection, expected_selected_idx",
    [
        ({"city": ["London"]}, [0, 3]),
        ({"city": ["Paris", "Berlin"]}, [1, 2]),
    ]
)
def test_split_by_column_entries(selection, expected_selected_idx):
    df = pd.DataFrame({
        "city": ["London", "Paris", "Berlin", "London"],
        "value": [10, 20, 30, 40],
    })

    sel, rej = split_by_column_entries(df, selection, return_rejected=True)

    assert list(sel.index) == expected_selected_idx
    assert sel["city"].tolist() == df.loc[expected_selected_idx, "city"].tolist()

    expected_rej_idx = [i for i in df.index if i not in expected_selected_idx]
    assert list(rej.index) == expected_rej_idx

    assert hasattr(sel, "_prev_index")
    assert hasattr(rej, "_prev_index")

def test_split_by_column_entries_reset_index():
    df = pd.DataFrame({
        "city": ["A", "B", "A"],
        "value": [1, 2, 3],
    })

    sel, rej = split_by_column_entries(
        df,
        {"city": ["A"]},
        reset_index=True,
        return_rejected=True,
    )

    assert list(sel.index) == [0, 1]     # reset
    assert list(rej.index) == [0]        # reset

def test_split_by_column_entries_inverse():
    df = pd.DataFrame({
        "city": ["A", "B", "C"],
        "value": [1, 2, 3],
    })

    # Normally "A" ? index [0]
    # With inverse=True ? everything except [0]
    sel, rej = split_by_column_entries(
        df,
        {"city": ["A"]},
        inverse=True,
        return_rejected=True,
    )

    assert list(sel.index) == [1, 2]   # inverse selection
    assert list(rej.index) == [0]
    
def test_split_by_column_entries_empty_rejected_dtype_preserved():
    df = pd.DataFrame({
        "city": ["A", "A"],
        "value": [1, 2],
    })

    sel, rej = split_by_column_entries(
        df,
        {"city": ["A"]},
        return_rejected=False,
    )

    assert sel.equals(df)
    assert rej.empty
    assert list(rej.columns) == list(df.columns)


def test_split_by_column_entries_empty_selection_dtype_preserved():
    df = pd.DataFrame({
        "city": ["A", "A"],
        "value": [1, 2],
    })

    sel, rej = split_by_column_entries(
        df,
        {"city": ["Z"]},  # no match
        return_rejected=True,
    )

    assert sel.empty
    assert list(sel.dtypes) == list(df.dtypes)
    assert list(rej.index) == list(df.index)
    

@pytest.mark.parametrize(
    "index, expected_selected",
    [
        ([0], [0]),
        ([1, 3], [1, 3]),
        ([], []),
    ]
)
def test_split_by_index_basic(index, expected_selected):
    df = pd.DataFrame({"x": [10, 20, 30, 40]})

    sel, rej = split_by_index(df, index, return_rejected=True)

    assert list(sel.index) == expected_selected
    expected_rejected = [i for i in df.index if i not in expected_selected]
    assert list(rej.index) == expected_rejected

    assert hasattr(sel, "_prev_index")
    assert hasattr(rej, "_prev_index")

def test_split_by_index_reset_index():
    df = pd.DataFrame({"x": [10, 20, 30]})

    sel, rej = split_by_index(df, [0, 2], reset_index=True, return_rejected=True)

    assert list(sel.index) == [0, 1]
    assert list(rej.index) == [0]


def test_split_by_index_inverse():
    df = pd.DataFrame({"x": [10, 20, 30]})

    sel, rej = split_by_index(df, [1], inverse=True, return_rejected=True)

    assert list(sel.index) == [0, 2]     # everything except index 1
    assert list(rej.index) == [1]


def test_split_by_index_empty_rejected_dtype_preserved():
    df = pd.DataFrame({"x": [10, 20]})

    sel, rej = split_by_index(df, [0, 1], return_rejected=False)

    assert sel.equals(df)
    assert rej.empty
    assert list(rej.columns) == ["x"]
    assert rej["x"].dtype == df["x"].dtype
    
def test_split_by_index_empty_selection():
    df = pd.DataFrame({"x": [10, 20]})

    sel, rej = split_by_index(df, [], return_rejected=True)

    assert sel.empty
    assert list(sel.dtypes) == list(df.dtypes)

    assert list(rej.index) == list(df.index)
    
        
