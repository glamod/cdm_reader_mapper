from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdm_reader_mapper.mdf_reader.utils.validators import (
    _is_true,
    _is_false,
    validate_datetime,
    validate_numeric,
    validate_str,
    validate_codes,
    validate,
)


@pytest.fixture
def sample_series():
    return pd.Series(["20200101", "bad", None, "20221231"], dtype="object")


@pytest.fixture
def numeric_series():
    return pd.Series(["1", "2", "3", "False", "bad"], dtype="object")


@pytest.fixture
def code_series():
    return pd.Series(["A", "B", "C", None, "X"], dtype="object")


def test_is_true_false():
    assert _is_true(True) is True
    assert _is_true(False) is False
    assert _is_false(False) is True
    assert _is_false(True) is False
    assert _is_true(1) is False
    assert _is_false(0) is False


def test_validate_datetime(sample_series):
    result = validate_datetime(sample_series)
    expected = pd.Series([True, False, True, True])
    pd.testing.assert_series_equal(result, expected)


def test_validate_numeric(numeric_series):
    result = validate_numeric(numeric_series, 1, 3)
    expected = pd.Series([True, True, True, False, False])
    pd.testing.assert_series_equal(result, expected)


def test_validate_str(numeric_series):
    result = validate_str(numeric_series)
    expected = pd.Series([True] * len(numeric_series), dtype="boolean")
    pd.testing.assert_series_equal(result, expected)


def test_validate_codes(code_series):
    codes = ["A", "B", "C"]
    result = validate_codes(code_series, codes, "str")
    expected = pd.Series([True, True, True, True, False])
    pd.testing.assert_series_equal(result, expected)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "NUM": ["1", "2", "bad", np.nan, "5"],
            "KEY": ["0", "1", "2", "9", np.nan],
            "STR": ["foo", "bar", "baz", "", np.nan],
            "DATE": ["20220101", "20220202", "bad_date", np.nan, "20220505"],
            "BOOL": ["True", "False", "TRUE", "FALSE", None],
        }
    )


@pytest.fixture
def attributes():
    return {
        "NUM": {"column_type": "int", "valid_min": 1, "valid_max": 5},
        "KEY": {"column_type": "key", "codetable": "ICOADS.C0.A"},
        "STR": {"column_type": "str"},
        "DATE": {"column_type": "datetime"},
        "BOOL": {"column_type": "int"},  # treat boolean literals as numeric override
    }


def test_validate_all_columns(sample_df, attributes):
    mask = validate(
        sample_df, imodel="icoads", ext_table_path=None, attributes=attributes
    )

    expected_num = [True, True, False, True, True]
    assert mask["NUM"].tolist() == expected_num

    expected_key = [True, True, True, False, True]
    assert mask["KEY"].tolist() == expected_key

    expected_key = [True, True, True, True, True]
    assert mask["STR"].tolist() == expected_key

    expected_date = [True, True, False, True, True]
    assert mask["DATE"].tolist() == expected_date

    expected_bool = [True, False, False, False, True]
    assert mask["BOOL"].tolist() == expected_bool
