from __future__ import annotations

import pandas as pd
import pytest

from cdm_reader_mapper import read, test_data
from cdm_reader_mapper.common.pandas_TextParser_hdlr import make_copy

from ._results import cdm_header, correction_df

data_dict = dict(test_data.test_icoads_r300_d721)


def _get_data(TextParser, **kwargs):
    if TextParser is True:
        kwargs["chunksize"] = 3
    return read(**data_dict, imodel="icoads_r300_d721", **kwargs)


@pytest.mark.parametrize(
    "func,args,idx_exp,idx_rej,skwargs",
    [
        ("select_where_all_true", [], [0, 1, 2, 3, 4], [], {"sections": ["c99_data"]}),
        ("select_where_all_false", [], [], [0, 1, 2, 3, 4], {"sections": ["c99_data"]}),
        ("select_where_index_isin", [[0, 2, 4]], [0, 2, 4], [1, 3], {}),
        ("select_where_entry_isin", [{("c1", "B1"): [26, 41]}], [1, 3], [0, 2, 4], {}),
    ],
)
@pytest.mark.parametrize("TextParser", [False, True])
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def ztest_select_operators(
    func, args, idx_exp, idx_rej, skwargs, TextParser, reset_index, inverse
):
    data = _get_data(TextParser, **skwargs)
    result = getattr(data, func)(*args, reset_index=reset_index, inverse=inverse)
    expected = data.data
    expected_mask = data.mask
    selected = result.data
    selected_mask = result.mask

    if TextParser is True:
        expected = make_copy(expected).read()
        selected = make_copy(selected).read()
        expected_mask = make_copy(expected_mask).read()
        selected_mask = make_copy(selected_mask).read()

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
    "func,args,idx_exp,idx_rej,skwargs",
    [
        ("split_by_boolean_true", [], [0, 1, 2, 3, 4], [], {"sections": ["c99_data"]}),
        ("split_by_boolean_false", [], [], [0, 1, 2, 3, 4], {"sections": ["c99_data"]}),
        ("split_by_index", [[0, 2, 4]], [0, 2, 4], [1, 3], {}),
        ("split_by_column_entries", [{("c1", "B1"): [26, 41]}], [1, 3], [0, 2, 4], {}),
    ],
)
@pytest.mark.parametrize("TextParser", [False, True])
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_split_operators(
    func, args, idx_exp, idx_rej, skwargs, TextParser, reset_index, inverse
):
    data = _get_data(TextParser, **skwargs)
    result = getattr(data, func)(*args, reset_index=reset_index, inverse=inverse)
    expected = data.data
    expected_mask = data.mask
    selected = result[0].data
    selected_mask = result[0].mask
    rejected = result[1].data
    rejected_mask = result[1].mask

    if TextParser is True:
        expected = make_copy(expected).read()
        selected = make_copy(selected).read()
        rejected = make_copy(rejected).read()
        expected_mask = make_copy(expected_mask).read()
        selected_mask = make_copy(selected_mask).read()
        rejected_mask = make_copy(rejected_mask).read()

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

    print(selected)
    print(rejected)

    pd.testing.assert_frame_equal(expected1, selected)
    pd.testing.assert_frame_equal(expected2, rejected)
    pd.testing.assert_frame_equal(expected_mask1, selected_mask)
    pd.testing.assert_frame_equal(expected_mask2, rejected_mask)


@pytest.mark.parametrize("TextParser", [True, False])
def test_inspect_count_by_cat(TextParser):
    data = _get_data(TextParser)
    result = data.unique(columns=("c1", "B1"))
    assert result == {("c1", "B1"): {19: 1, 26: 1, 27: 1, 41: 1, 91: 1}}


def test_replace():
    result = cdm_header.replace_columns(
        correction_df,
        subset="header",
        pivot_c="report_id",
        rep_c=["primary_station_id", "primary_station_id.isChange"],
    )
    cdm_header[("header", "primary_station_id")] = [
        "MASKSTID2",
        "MASKSTID2",
        "MASKSTID",
        "MASKSTID2",
        "MASKSTID2",
    ]
    pd.testing.assert_frame_equal(cdm_header.data, result.data)
