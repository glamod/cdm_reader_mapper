from __future__ import annotations

import pandas as pd
import pytest

from cdm_reader_mapper import read, test_data
from cdm_reader_mapper.common.pandas_TextParser_hdlr import make_copy

from ._results import cdm_header, correction_df

data_dict = dict(test_data.test_icoads_r300_d721)


def _get_data(TextParser, **kwargs):
    if TextParser is True:
        kwargs["chunksize"] = 10000
    return read(**data_dict, imodel="icoads_r300_d721", **kwargs)


@pytest.mark.parametrize("TextParser", [True, False])
@pytest.mark.parametrize("reset_index", [True, False])
@pytest.mark.parametrize("inverse", [True, False])
def test_select_true(TextParser, reset_index, inverse):
    data = _get_data(TextParser, sections=["c99_data"])
    result = data.select_true(reset_index=reset_index, inverse=inverse)
    expected = data.data
    selected = result.data

    if TextParser is True:
        expected = make_copy(expected).read()
        selected = make_copy(selected).read()

    if inverse is False:
        expected = expected[:5]
    else:
        expected = expected[:0]

    if reset_index is True:
        expected = expected.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected, selected)


@pytest.mark.parametrize("TextParser", [True, False])
@pytest.mark.parametrize("reset_index", [True, False])
@pytest.mark.parametrize("inverse", [True, False])
def test_select_false(TextParser, reset_index, inverse):
    data = _get_data(TextParser, sections=["c99_data"])
    result = data.select_false(reset_index=reset_index, inverse=inverse)
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
def test_select_from_index(TextParser, reset_index, inverse):
    data = _get_data(TextParser)
    result = data.select_from_index([0, 2, 4], reset_index=reset_index, inverse=inverse)
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


@pytest.mark.parametrize("TextParser", [True, False])
@pytest.mark.parametrize("reset_index", [True, False])
@pytest.mark.parametrize("inverse", [True, False])
def test_select_from_list(TextParser, reset_index, inverse):
    data = _get_data(TextParser)
    selection = {("c1", "B1"): [26, 41]}
    result = data.select_from_list(selection, reset_index=reset_index, inverse=inverse)
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
