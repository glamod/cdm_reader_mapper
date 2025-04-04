from __future__ import annotations

import numpy as np
import pandas as pd
import pytest  # noqa

from cdm_reader_mapper import read, test_data
from cdm_reader_mapper.common.pandas_TextParser_hdlr import make_copy

data_dict = dict(test_data.test_icoads_r300_d700)


def _get_data(TextParser, **kwargs):
    if TextParser is True:
        kwargs["chunksize"] = 3
    return read(**data_dict, imodel="icoads_r300_d700", **kwargs)


@pytest.mark.parametrize("TextParser", [True, False])
def test_index(TextParser):
    data = _get_data(TextParser)
    result = data.index
    np.testing.assert_equal(list(result), [0, 1, 2, 3, 4])


@pytest.mark.parametrize("TextParser", [True, False])
def test_size(TextParser):
    data = _get_data(TextParser)
    result = data.size
    np.testing.assert_equal(result, 1430)


@pytest.mark.parametrize("TextParser", [True, False])
def test_shape(TextParser):
    data = _get_data(TextParser)
    result = data.shape
    np.testing.assert_equal(result, (5, 286))


@pytest.mark.parametrize("TextParser", [True, False])
def test_dropna(TextParser):
    data = _get_data(TextParser)
    result = data.dropna(how="any")
    if TextParser:
        result = make_copy(result).read()
    expected = pd.DataFrame(columns=data.columns)
    expected = expected.astype(data.dtypes)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("TextParser", [True, False])
def test_rename(TextParser):
    data = _get_data(TextParser)
    _renames = {("core", "MO"): ("core", "MONTH")}
    result = data.rename(columns=_renames)
    expected = data.data
    if TextParser:
        result = make_copy(result).read()
        expected = make_copy(expected).read()
    expected = expected.rename(columns=_renames)
    expected = expected.astype(result.dtypes)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("TextParser", [True, False])
def test_inplace(TextParser):
    data = _get_data(TextParser)
    _renames = {("core", "MO"): ("core", "MONTH")}
    data_ = data.copy()
    data_.rename(columns=_renames, inplace=True)
    result = data_.data
    expected = data.data
    if TextParser:
        result = make_copy(result).read()
        expected = make_copy(expected).read()
    expected = expected.rename(columns=_renames)
    expected = expected.astype(result.dtypes)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("TextParser", [False])
def test_iloc(TextParser):
    data = _get_data(TextParser)
    result = data.iloc[[1, 3, 4]]
    expected = data.data
    if TextParser:
        result = make_copy(result).read()
        expected = make_copy(expected).read()
    expected = expected.loc[[1, 3, 4]]
    pd.testing.assert_frame_equal(result, expected)
