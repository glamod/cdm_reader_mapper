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


@pytest.mark.parametrize("TextParser", [False, True])
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_select_where_all_true(TextParser, reset_index, inverse):
    data = _get_data(TextParser, sections=["c99_data"])
    result = data.select_where_all_true(reset_index=reset_index, inverse=inverse)
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
        expected = expected[:5]
        expected_mask = expected_mask[:5]
    else:
        expected = expected[:0]
        expected_mask = expected_mask[:0]

    if reset_index is True:
        expected = expected.reset_index(drop=True)
        expected_mask = expected_mask.reset_index(drop=True)

    print(selected_mask)
    pd.testing.assert_frame_equal(expected, selected)
    pd.testing.assert_frame_equal(expected_mask, selected_mask)


@pytest.mark.parametrize("TextParser", [False, True])
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_select_where_all_false(TextParser, reset_index, inverse):
    data = _get_data(TextParser, sections=["c99_data"])
    result = data.select_where_all_false(reset_index=reset_index, inverse=inverse)
    expected = data.data
    selected = result.data

    if TextParser is True:
        expected = make_copy(expected).read()
        selected = make_copy(selected).read()

    if inverse is False:
        expected = expected[:0]
    else:
        expected = expected[:5]

    if reset_index is True:
        expected = expected.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected, selected)


@pytest.mark.parametrize("TextParser", [False, True])
@pytest.mark.parametrize("reset_index", [True, False])
@pytest.mark.parametrize("inverse", [True, False])
def test_select_where_index_isin(TextParser, reset_index, inverse):
    data = _get_data(TextParser)
    result = data.select_where_index_isin(
        [0, 2, 4], reset_index=reset_index, inverse=inverse
    )
    expected = data.data
    selected = result.data

    if TextParser is True:
        expected = make_copy(expected).read()
        selected = make_copy(selected).read()

    if inverse is False:
        idx = expected.index.isin([0, 2, 4])
    else:
        idx = expected.index.isin([1, 3])

    expected = expected[idx]

    if reset_index is True:
        expected = expected.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected, selected)


@pytest.mark.parametrize("TextParser", [False, True])
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_select_where_entry_isin(TextParser, reset_index, inverse):
    data = _get_data(TextParser)
    selection = {("c1", "B1"): [26, 41]}
    result = data.select_where_entry_isin(
        selection, reset_index=reset_index, inverse=inverse
    )
    expected = data.data
    selected = result.data

    if TextParser is True:
        expected = make_copy(expected).read()
        selected = make_copy(selected).read()

    if inverse is False:
        idx = expected.index.isin([1, 3])
    else:
        idx = expected.index.isin([0, 2, 4])

    expected = expected[idx]

    if reset_index is True:
        expected = expected.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected, selected)


@pytest.mark.parametrize("TextParser", [False, True])
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_split_by_boolean_true(TextParser, reset_index, inverse):
    data = _get_data(TextParser, sections=["c99_data"])
    result = data.split_by_boolean_true(reset_index=reset_index, inverse=inverse)
    expected = data.data
    selected = result[0].data
    rejected = result[1].data

    if TextParser is True:
        expected = make_copy(expected).read()
        selected = make_copy(selected).read()
        rejected = make_copy(rejected).read()

    if inverse is False:
        expected1 = expected[:5]
        expected2 = expected[:0]
    else:
        expected1 = expected[:0]
        expected2 = expected[:5]

    if reset_index is True:
        expected1 = expected1.reset_index(drop=True)
        expected2 = expected2.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected1, selected)
    pd.testing.assert_frame_equal(expected2, rejected)


@pytest.mark.parametrize("TextParser", [False, True])
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_split_by_boolean_false(TextParser, reset_index, inverse):
    data = _get_data(TextParser, sections=["c99_data"])
    result = data.split_by_boolean_false(reset_index=reset_index, inverse=inverse)
    expected = data.data
    selected = result[0].data
    rejected = result[1].data

    if TextParser is True:
        expected = make_copy(expected).read()
        selected = make_copy(selected).read()
        rejected = make_copy(rejected).read()

    if inverse is False:
        expected1 = expected[:0]
        expected2 = expected[:5]
    else:
        expected1 = expected[:5]
        expected2 = expected[:0]

    if reset_index is True:
        expected1 = expected1.reset_index(drop=True)
        expected2 = expected2.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected1, selected)
    pd.testing.assert_frame_equal(expected2, rejected)


@pytest.mark.parametrize("TextParser", [False, True])
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_split_by_index(TextParser, reset_index, inverse):
    data = _get_data(TextParser)
    result = data.split_by_index([0, 2, 4], reset_index=reset_index, inverse=inverse)
    expected = data.data
    selected = result[0].data
    rejected = result[1].data

    if TextParser is True:
        expected = make_copy(expected).read()
        selected = make_copy(selected).read()
        rejected = make_copy(rejected).read()

    if inverse is False:
        idx1 = expected.index.isin([0, 2, 4])
        idx2 = expected.index.isin([1, 3])
    else:
        idx1 = expected.index.isin([1, 3])
        idx2 = expected.index.isin([0, 2, 4])

    expected1 = expected[idx1]
    expected2 = expected[idx2]

    if reset_index is True:
        expected1 = expected1.reset_index(drop=True)
        expected2 = expected2.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected1, selected)
    pd.testing.assert_frame_equal(expected2, rejected)


@pytest.mark.parametrize("TextParser", [False, True])
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_split_by_column_entries(TextParser, reset_index, inverse):
    data = _get_data(TextParser)
    selection = {("c1", "B1"): [26, 41]}
    result = data.split_by_column_entries(
        selection, reset_index=reset_index, inverse=inverse
    )
    expected = data.data
    selected = result[0].data
    rejected = result[1].data

    if TextParser is True:
        expected = make_copy(expected).read()
        selected = make_copy(selected).read()
        rejected = make_copy(rejected).read()

    if inverse is False:
        idx1 = expected.index.isin([1, 3])
        idx2 = expected.index.isin([0, 2, 4])
    else:
        idx1 = expected.index.isin([0, 2, 4])
        idx2 = expected.index.isin([1, 3])

    expected1 = expected[idx1]
    expected2 = expected[idx2]

    if reset_index is True:
        expected1 = expected1.reset_index(drop=True)
        expected2 = expected2.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected1, selected)
    pd.testing.assert_frame_equal(expected2, rejected)


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
