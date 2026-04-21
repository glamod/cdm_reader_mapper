from __future__ import annotations
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from cdm_reader_mapper.cdm_mapper.utils.conversions import (
    _convert_array_general_from_str,
    _convert_array_general_to_str,
    _convert_datetime_from_str,
    _convert_datetime_to_str,
    _convert_float_array_from_str,
    _convert_float_array_to_str,
    _convert_float_from_str,
    _convert_float_to_str,
    _convert_integer_array_from_str,
    _convert_integer_array_to_str,
    _convert_integer_from_str,
    _convert_integer_to_str,
    _convert_str_array_from_str,
    _convert_str_array_to_str,
    _convert_str_from_str,
    _convert_str_to_str,
    # _convert_column,
    # _convert_columns,
    # convert_from_str_df,
    # convert_from_str_series,
    # convert_to_str_df,
    # convert_to_str_series,
)


@pytest.mark.parametrize(
    "data, dtype, exp",
    [
        (["1,2,3"], int, [[1, 2, 3]]),
        (["{1,2,3}"], int, [[1, 2, 3]]),
        ([[1, 2, 3]], int, [[1, 2, 3]]),
        ([["1", "2", "3"]], int, [[1, 2, 3]]),
        (["1, 2, 3"], int, [[1, 2, 3]]),
        (["{1}"], int, [[1]]),
        (["1"], int, [[1]]),
        ([""], str, [pd.NA]),
        (["{abc}"], str, [["abc"]]),
        ([None], int, [pd.NA]),
        ([pd.NA], int, [pd.NA]),
        (["null"], "Int64", [pd.NA]),
        (["1,null,3"], "Int64", [[1, pd.NA, 3]]),
        ([[]], int, [pd.NA]),
        ([np.nan], int, [pd.NA]),
        (["1,2", [3, 4], None], int, [[1, 2], [3, 4], pd.NA]),
    ],
)
def test_convert_array_general_from_str(data, dtype, exp):
    series = pd.Series(data)
    result = _convert_array_general_from_str(series, "null", dtype)
    expected = pd.Series(exp)

    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, dtype, exp",
    [
        ([[1, 2, 3]], int, ["{1,2,3}"]),
        (["[1,2,3]"], int, ["{1,2,3}"]),
        ([1], int, ["{1}"]),
        (["1"], int, ["{1}"]),
        ([[]], int, ["null"]),
        ([None], int, ["null"]),
        ([pd.NA], int, ["null"]),
        ([np.nan], int, ["null"]),
        ([""], str, ["null"]),
        (["abc"], str, ["{abc}"]),
        ([[1.1, 2.2]], float, ["{1.1,2.2}"]),
        ([[1, 0]], bool, ["{True,False}"]),
        ([[1, pd.NA, 3]], "Int64", ["{1,null,3}"]),
        ([["", "", ""]], str, ["{null,null,null}"]),
        ([[1, 2], np.array([3, 4]), None], int, ["{1,2}", "{3,4}", "null"]),
    ],
)
def test_convert_array_general_to_str(exp, dtype, data):
    series = pd.Series(data)
    result = _convert_array_general_to_str(series, "null", dtype)
    expected = pd.Series(exp)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, exp",
    [
        ([1, None, [1, 2], "abc"], ["1", "null", "[1, 2]", "abc"]),
        ([[], "", np.nan], ["[]", "", "null"]),
    ],
)
def test_convert_str_to_str(data, exp):
    series = pd.Series(data)
    result = _convert_str_to_str(series, "null")
    expected = pd.Series(exp)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, exp",
    [
        (["1", "null", "[1, 2]", "abc"], ["1", pd.NA, "[1, 2]", "abc"]),
        (["[]", "", "NA"], ["[]", "", "NA"]),
    ],
)
def test_convert_str_from_str(data, exp):
    series = pd.Series(data)
    result = _convert_str_from_str(series, "null")
    expected = pd.Series(exp)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, exp",
    [
        (["a"], ["{a}"]),
        ([["a", "b"]], ["{a,b}"]),
        ([[]], ["null"]),
        ([None, np.nan], ["null", "null"]),
        ([["a", "", "c"]], ["{a,null,c}"]),
        ([""], ["null"]),
    ],
)
def test_convert_str_array_to_str(data, exp):
    series = pd.Series(data)
    result = _convert_str_array_to_str(series, "null")
    expected = pd.Series(exp)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, exp",
    [
        (["{a,b,c}"], [["a", "b", "c"]]),
        (["{}"], pd.NA),
        (["null"], pd.NA),
        (["{a,,c}"], [["a", pd.NA, "c"]]),
        (["{x}"], [["x"]]),
        (["not an array"], [["not an array"]]),
    ],
)
def test_convert_str_array_from_str(data, exp):
    series = pd.Series(data)
    result = _convert_str_array_from_str(series, "null")
    expected = pd.Series(exp)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, exp",
    [
        ([1, 2, 3], ["1", "2", "3"]),
        ([1.0, 2.5, 3.9], ["1", "2", "3"]),
        ([np.nan, None, pd.NA], ["null", "null", "null"]),
        ([1, 2.2, None, 4], ["1", "2", "null", "4"]),
        ([1, "abc", 3], ["1", "null", "3"]),
    ],
)
def test_convert_integer_to_str(data, exp):
    series = pd.Series(data)
    result = _convert_integer_to_str(series, "null")
    expected_series = pd.Series(exp)
    pd.testing.assert_series_equal(result, expected_series)


@pytest.mark.parametrize(
    "data, exp",
    [
        (
            ["1", "2", "3"],
            [1, 2, 3],
        ),
        ([1.0, np.nan, 3.0], [1, pd.NA, 3]),
        (["1", "abc", "3"], [1, pd.NA, 3]),
        (["1", "null", "3"], [1, pd.NA, 3]),
        (["", "2"], [pd.NA, 2]),
    ],
)
def test_convert_integer_from_str(data, exp):
    series = pd.Series(data)
    result = _convert_integer_from_str(series, "null")
    expected = pd.Series(exp, dtype="Int64")
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, exp",
    [
        ([1, 2, 3], ["{1}", "{2}", "{3}"]),
        ([[1, 2], [3, 4]], ["{1,2}", "{3,4}"]),
        ([[]], ["null"]),
        ([None, np.nan, pd.NA], ["null", "null", "null"]),
        ([[1, None, 3], [4, 5]], ["{1,null,3}", "{4,5}"]),
        ([0], ["{0}"]),
    ],
)
def test_convert_integer_array_to_str(data, exp):
    series = pd.Series(data)
    result = _convert_integer_array_to_str(series, "null")
    expected = pd.Series(exp)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, exp",
    [
        (["{1,2,3}", "{4,5}"], [[1, 2, 3], [4, 5]]),
        (["{}"], pd.NA),
        (["null"], pd.NA),
        (["{1,null,3}"], [[1, pd.NA, 3]]),
        (["{0}"], [[0]]),
    ],
)
def test_convert_integer_array_from_str(data, exp):
    series = pd.Series(data)
    result = _convert_integer_array_from_str(series, "null")
    expected = pd.Series(exp)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, decimal_places, exp",
    [
        ([1.234, 5.678, 9.0], 2, ["1.23", "5.68", "9.00"]),
        ([1, 2, 3], 1, ["1.0", "2.0", "3.0"]),
        ([np.nan, None, pd.NA], 2, ["null", "null", "null"]),
        ([1.2, None, 3.4], 1, ["1.2", "null", "3.4"]),
        ([1.9, 2.1], 0, ["2", "2"]),
        ([-1.234, -5.678], 2, ["-1.23", "-5.68"]),
        ([], 2, []),
    ],
)
def test_convert_float_to_str(data, decimal_places, exp):
    series = pd.Series(data)
    result = _convert_float_to_str(series, "null", decimal_places)
    expected = pd.Series(exp)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, exp",
    [
        (["1.23", "5.68", "9.00"], [1.23, 5.68, 9.0]),
        (["1", "2", "3"], [1.0, 2.0, 3.0]),
        (["null", "1.5"], [pd.NA, 1.5]),
        (["abc", "2.5"], [pd.NA, 2.5]),
        (["", "3.1"], [pd.NA, 3.1]),
        (["1.1", "", "abc", "5.5"], [1.1, pd.NA, pd.NA, 5.5]),
        ([], []),
    ],
)
def test_convert_float_from_str(data, exp):
    series = pd.Series(data)
    result = _convert_float_from_str(series, "null")
    expected = pd.Series(exp, dtype="Float64")
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, exp",
    [
        ([1.2, 3.4], ["{1.2}", "{3.4}"]),
        ([[1.2, 3.4]], ["{1.2,3.4}"]),
        ([1.2], ["{1.2}"]),
        ([np.nan], ["null"]),
        ([None], ["null"]),
        ([[]], ["null"]),
        (["[1.2,3.4]"], ["{1.2,3.4}"]),
        (["[1.2]"], ["{1.2}"]),
        ([""], ["null"]),
    ],
)
def test_convert_float_array_to_str(data, exp):
    series = pd.Series(data)
    result = _convert_float_array_to_str(series, "null")
    expected = pd.Series(exp, dtype=object)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, exp",
    [
        (["{1.2,3.4}"], [[1.2, 3.4]]),
        (["{1.2}"], [[1.2]]),
        (["{}"], pd.NA),
        ([np.nan], pd.NA),
        ([None], pd.NA),
        ([[]], pd.NA),
    ],
)
def test_convert_float_array_from_str(data, exp):
    series = pd.Series(data)
    result = _convert_float_array_from_str(series, "null")
    expected = pd.Series(exp, dtype=object)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, exp",
    [
        (
            [datetime(2023, 1, 1, 12, 0, 0), "2022-01-01 00:00:00", np.nan],
            ["2023-01-01 12:00:00", "2022-01-01 00:00:00", "null"],
        ),
        (
            [np.nan, None, pd.NA],
            ["null", "null", "null"],
        ),
        (
            [datetime(2021, 7, 4, 9, 30), datetime(2020, 12, 31, 23, 59, 59)],
            ["2021-07-04 09:30:00", "2020-12-31 23:59:59"],
        ),
        (
            [],
            [],
        ),
    ],
)
def test_convert_datetime_to_str(data, exp):
    series = pd.Series(data)
    result = _convert_datetime_to_str(series, "null")
    expected = pd.Series(exp, dtype=object)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, exp",
    [
        (
            ["2023-01-01 12:00:00", "2022-01-01 00:00:00"],
            [pd.Timestamp("2023-01-01 12:00:00"), pd.Timestamp("2022-01-01 00:00:00")],
        ),
        (
            ["not-a-date", "2022-01-01 00:00:00"],
            [pd.NaT, pd.Timestamp("2022-01-01 00:00:00")],
        ),
        (
            [datetime(2021, 7, 4, 9, 30), np.nan],
            [pd.Timestamp("2021-07-04 09:30:00"), pd.NaT],
        ),
        (
            [],
            [],
        ),
    ],
)
def test_convert_datetime_from_str(data, exp):
    series = pd.Series(data)
    result = _convert_datetime_from_str(series, "null")
    expected = pd.Series(exp, dtype="datetime64[ns]")
    pd.testing.assert_series_equal(result, expected)
