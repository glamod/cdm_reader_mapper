from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

from cdm_reader_mapper.cdm_mapper.utils.conversions import (
    convert_integer,
    convert_float,
    convert_datetime,
    convert_str,
    convert_integer_array,
    convert_str_array,
)


@pytest.mark.parametrize(
    "input_series, null_label, expected_series",
    [
        # All integers
        (pd.Series([1, 2, 3]), "NA", pd.Series(["1", "2", "3"], dtype=object)),
        # Floats that are whole numbers
        (pd.Series([1.0, 2.0, 3.0]), "NA", pd.Series(["1", "2", "3"], dtype=object)),
        # Floats with decimals
        (
            pd.Series([1.1, 2.5, 3.9]),
            "NA",
            pd.Series(["1", "2", "3"], dtype=object),
        ),  # truncated
        # Strings that are numbers
        (
            pd.Series(["4", "5.0", "6.7"]),
            "NA",
            pd.Series(["4", "5", "6"], dtype=object),
        ),
        # NaN values
        (
            pd.Series([np.nan, 2, "3"]),
            "NULL",
            pd.Series(["NULL", "2", "3"], dtype=object),
        ),
        # Invalid strings
        (
            pd.Series(["abc", "123", "4.5"]),
            "NA",
            pd.Series(["NA", "123", "4"], dtype=object),
        ),
        # Empty Series
        (pd.Series([], dtype=object), "NA", pd.Series([], dtype=object)),
    ],
)
def test_convert_integer(input_series, null_label, expected_series):
    result = convert_integer(input_series, null_label)

    pd.testing.assert_series_equal(
        result.reset_index(drop=True),
        expected_series.reset_index(drop=True),
        check_names=False,
        check_dtype=True,  # ensure all output elements are strings
    )


@pytest.mark.parametrize(
    "input_series, null_label, decimal_places, expected_series",
    [
        (
            pd.Series([1.0, 2.5, 3.14159]),
            "NA",
            2,
            pd.Series(["1.00", "2.50", "3.14"], dtype=object),
        ),
        (
            pd.Series([0.1234, 5.6789]),
            "NA",
            3,
            pd.Series(["0.123", "5.679"], dtype=object),
        ),
        (pd.Series([1, 2, 3]), "NA", 1, pd.Series(["1.0", "2.0", "3.0"], dtype=object)),
        (
            pd.Series([np.nan, 2.5, 3]),
            "NULL",
            2,
            pd.Series(["NULL", "2.50", "3.00"], dtype=object),
        ),
        (pd.Series(["abc", 1.23]), "NA", 1, pd.Series(["NA", "1.2"], dtype=object)),
        (pd.Series([], dtype=float), "NA", 2, pd.Series([], dtype=object)),
    ],
)
def test_convert_float(input_series, null_label, decimal_places, expected_series):
    result = convert_float(input_series, null_label, decimal_places)

    pd.testing.assert_series_equal(
        result.reset_index(drop=True),
        expected_series.reset_index(drop=True),
        check_names=False,
        check_dtype=True,
    )


@pytest.mark.parametrize(
    "input_series, null_label, expected_series",
    [
        # Datetime objects
        (
            pd.Series(
                [
                    pd.Timestamp("2024-01-01 12:30:00"),
                    pd.Timestamp("2023-12-31 23:59:59"),
                ]
            ),
            "NA",
            pd.Series(["2024-01-01 12:30:00", "2023-12-31 23:59:59"], dtype=object),
        ),
        # String dates (should be returned as-is)
        (
            pd.Series(["2024-01-01 12:30:00", "2023-12-31 23:59:59"]),
            "NA",
            pd.Series(["2024-01-01 12:30:00", "2023-12-31 23:59:59"], dtype=object),
        ),
        # Mixed datetime, string, and NaN
        (
            pd.Series(
                [pd.Timestamp("2024-01-01 12:30:00"), "2023-12-31 23:59:59", np.nan]
            ),
            "NULL",
            pd.Series(
                ["2024-01-01 12:30:00", "2023-12-31 23:59:59", "NULL"], dtype=object
            ),
        ),
        # Empty series
        (
            pd.Series([], dtype="datetime64[ns]"),
            "NA",
            pd.Series([], dtype=object),
        ),
    ],
)
def test_convert_datetime(input_series, null_label, expected_series):
    result = convert_datetime(input_series, null_label)

    pd.testing.assert_series_equal(
        result.reset_index(drop=True),
        expected_series.reset_index(drop=True),
        check_names=False,
        check_dtype=True,
    )


@pytest.mark.parametrize(
    "input_series, null_label, expected_series",
    [
        # Regular strings
        (pd.Series(["a", "b", "c"]), "NA", pd.Series(["a", "b", "c"], dtype=object)),
        # Integers (should be converted to strings)
        (pd.Series([1, 2, 3]), "NA", pd.Series(["1", "2", "3"], dtype=object)),
        # Floats (converted to strings)
        (
            pd.Series([1.1, 2.5, 3.0]),
            "NA",
            pd.Series(["1.1", "2.5", "3.0"], dtype=object),
        ),
        # Lists (should be converted to string representation)
        (
            pd.Series([[1, 2], ["a", "b"]]),
            "NA",
            pd.Series(["[1, 2]", "['a', 'b']"], dtype=object),
        ),
        # NaNs replaced by null_label
        (
            pd.Series([np.nan, "hello", None]),
            "NULL",
            pd.Series(["NULL", "hello", "NULL"], dtype=object),
        ),
        # Empty series
        (pd.Series([], dtype=object), "NA", pd.Series([], dtype=object)),
    ],
)
def test_convert_str(input_series, null_label, expected_series):
    result = convert_str(input_series, null_label)

    pd.testing.assert_series_equal(
        result.reset_index(drop=True),
        expected_series.reset_index(drop=True),
        check_names=False,
        check_dtype=True,
    )


@pytest.mark.parametrize(
    "input_series, null_label, expected_series",
    [
        # Single integers
        (pd.Series([1, 2, 3]), None, pd.Series(["{1}", "{2}", "{3}"], dtype=object)),
        # Single floats that are whole numbers
        (
            pd.Series([1.0, 2.0, 3.0]),
            None,
            pd.Series(["{1}", "{2}", "{3}"], dtype=object),
        ),
        # Single floats with decimals (truncated)
        (
            pd.Series([1.9, 2.5, 3.7]),
            None,
            pd.Series(["{1}", "{2}", "{3}"], dtype=object),
        ),
        # Lists of integers
        (
            pd.Series([[1, 2], [3, 4], [5]]),
            None,
            pd.Series(["{1,2}", "{3,4}", "{5}"], dtype=object),
        ),
        # Lists of floats (whole numbers)
        (
            pd.Series([[1.0, 2.0], [3.0, 4.0]]),
            None,
            pd.Series(["{1,2}", "{3,4}"], dtype=object),
        ),
        # Lists with invalid entries
        (
            pd.Series([[1, "a", 3], None, [np.nan]]),
            "NA",
            pd.Series(["{1,3}", "NA", "NA"], dtype=object),
        ),
        # Strings representing lists
        (
            pd.Series(["[1,2,3]", "[4,5]"]),
            None,
            pd.Series(["{1,2,3}", "{4,5}"], dtype=object),
        ),
        # Single None and NaN values
        (pd.Series([None, np.nan]), "NULL", pd.Series(["NULL", "NULL"], dtype=object)),
        # Empty Series
        (pd.Series([], dtype=float), None, pd.Series([], dtype=object)),
    ],
)
def test_convert_integer_array(input_series, null_label, expected_series):
    result = convert_integer_array(input_series, null_label)

    pd.testing.assert_series_equal(
        result.reset_index(drop=True),
        expected_series.reset_index(drop=True),
        check_names=False,
    )


@pytest.mark.parametrize(
    "input_series, null_label, expected_series",
    [
        # String arrays
        (pd.Series([["a", "b", "c"]]), "NA", pd.Series(["{a,b,c}"])),
        (pd.Series(["x"]), "NA", pd.Series(["{x}"])),
        (pd.Series([None]), "NULL", pd.Series(["NULL"])),
        (pd.Series(['["p","q","r"]']), "NA", pd.Series(["{p,q,r}"])),
        (pd.Series([[], None]), "NA", pd.Series(["NA", "NA"])),
        (pd.Series([], dtype=object), "NA", pd.Series([], dtype=object)),
    ],
)
def test_convert_str_array(input_series, null_label, expected_series):
    result = convert_str_array(input_series, null_label=null_label)
    pd.testing.assert_series_equal(
        result.reset_index(drop=True),
        expected_series.reset_index(drop=True),
        check_names=False,
    )
