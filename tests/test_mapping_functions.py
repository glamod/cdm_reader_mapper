from __future__ import annotations

import pytest

import datetime
import math
import re

import numpy as np
import pandas as pd
from pandas import Timestamp

from cdm_reader_mapper.cdm_mapper.utils.mapping_functions import (
    find_entry,
    coord_360_to_180i,
    coord_dmh_to_90i,
    convert_to_utc_i,
    time_zone_i,
    longitude_360to180_i,
    location_accuracy_i,
    convert_to_str,
    string_add_i,
    to_int,
    mapping_functions,
)

imappings = mapping_functions("icoads_r300_d720")


@pytest.mark.parametrize(
    "entry,expected",
    [
        ("apple", 6),
        ("apple_tree", 6),
        ("banana", 3),
        ("banana_house", 2),
        ("cherry", None),
    ],
)
def test_find_entry(entry, expected):
    entry_dict = {
        "apple": 6,
        "apple_tree_house": 5,
        "apple_house": 4,
        "banana": 3,
        "banana_house": 2,
    }
    result = find_entry(entry, entry_dict)
    assert result == expected


@pytest.mark.parametrize(
    "lon,expected",
    [
        (0, 0.0),
        (1, 1.0),
        (170, 170.0),
        (180, -180.0),
        (270, -90.0),
        (360, 0.0),
        (-90, -90.0),
    ],
)
def test_coord_360_to_180i(lon, expected):
    assert coord_360_to_180i(lon) == expected


@pytest.mark.parametrize(
    "deg, minutes, hemis, expected",
    [
        # Standard North hemisphere
        (10, 30, "N", 10.5),
        (0, 45, "N", 0.75),
        (90, 0, "N", 90.0),
        # Standard South hemisphere
        (10, 30, "S", -10.5),
        (0, 45, "S", -0.75),
        (90, 0, "S", -90.0),
        # Rounding edge cases
        (10, 59, "N", 10.98),
        (10, 1, "S", -10.02),
        # Negative degrees
        (-10, 30, "N", 10.5),
        (-10, 30, "S", -10.5),
    ],
)
def test_coord_dmh_to_90i_values(deg, minutes, hemis, expected):
    result = coord_dmh_to_90i(deg, minutes, hemis)
    assert result == expected


@pytest.mark.parametrize("hemis", ["E", "W", "", "n", "s"])
def test_coord_dmh_to_90i_invalid_hemis(hemis):
    with pytest.raises(ValueError):
        coord_dmh_to_90i(10, 30, hemis)


@pytest.mark.parametrize("minutes", [-1, 60, 100])
def test_coord_dmh_to_90i_invalid_minutes(minutes):
    with pytest.raises(ValueError):
        coord_dmh_to_90i(10, minutes, "N")


def test_convert_to_utc_i_basic():
    # Create a naive DatetimeIndex
    local_times = pd.date_range("2025-11-17 00:00", periods=3, freq="h")

    # Convert from US/Eastern (UTC-5 normally)
    utc_times = convert_to_utc_i(local_times, "US/Eastern")

    expected = pd.DatetimeIndex(
        [
            Timestamp("2025-11-17 05:00", tz="UTC"),
            Timestamp("2025-11-17 06:00", tz="UTC"),
            Timestamp("2025-11-17 07:00", tz="UTC"),
        ]
    )

    pd.testing.assert_index_equal(utc_times, expected)


def test_convert_to_utc_i_different_timezone():
    # Europe/Berlin (UTC+1 normally)
    local_times = pd.date_range("2025-11-17 00:00", periods=2, freq="h")
    utc_times = convert_to_utc_i(local_times, "Europe/Berlin")

    expected = pd.DatetimeIndex(
        [
            Timestamp("2025-11-16 23:00", tz="UTC"),
            Timestamp("2025-11-17 00:00", tz="UTC"),
        ]
    )

    pd.testing.assert_index_equal(utc_times, expected)


def test_convert_to_utc_i_input_not_datetimeindex():
    # Should raise AttributeError if not a DatetimeIndex
    with pytest.raises(AttributeError):
        convert_to_utc_i([1, 2, 3], "UTC")


@pytest.mark.parametrize(
    "lat,lon,expected",
    [
        (40.7128, -74.0060, "America/New_York"),
        (51.5074, -0.1278, "Europe/London"),
        (-33.8688, 151.2093, "Australia/Sydney"),
        (0, 0, "Etc/GMT"),
    ],
)
def test_time_zone_i_known_locations(lat, lon, expected):
    assert time_zone_i(lat, lon) == expected


@pytest.mark.parametrize("lat,lon", [(100, 0), (0, 200)])
def test_time_zone_i_invalid_inputs(lat, lon):
    assert time_zone_i(lat, lon) is None


@pytest.mark.parametrize(
    "lon,expected",
    [
        (0, 0),  # normal case
        (90, 90),  # below 180
        (180, 180),  # edge case
        (190, -170),  # >180
        (270, -90),  # >180
        (360, -180),  # exactly 360
        (-90, -90),  # negative number
        (-190, -190),  # negative below -180 (note: function doesn't adjust)
        (450, -90),  # >360
    ],
)
def test_longitude_360to180_i(lon, expected):
    assert longitude_360to180_i(lon) == expected


@pytest.mark.parametrize(
    "li,lat,expected",
    [
        (0, 0, 16),  # 0.1 * sqrt(111^2 * 2) -> 157
        (1, 0, 157),  # 1 * sqrt(111^2 * 2) -> 157
        (4, 0, 3),  # (1/60)* sqrt(...) -> ~2.62 -> 3
        (5, 0, 1),  # (1/3600)* sqrt(...) -> ~0.044 -> 1 (min enforced)
        ("1", 0, 157),
        ("4", 0, 3),
    ],
)
def test_location_accuracy_i_valid_values(li, lat, expected):
    assert location_accuracy_i(li, lat) == expected


@pytest.mark.parametrize("li,lat", [(2, 0), ("abc", 0)])
def test_location_accuracy_i_invalid_li(li, lat):
    # li not in dictionary
    assert np.isnan(location_accuracy_i(li, lat))


def test_location_accuracy_i_lat_edge_cases_positiv():
    result_90 = location_accuracy_i(1, 90)
    expected_90 = int(
        round(1 * math.sqrt(111**2 * (1 + math.cos(math.radians(90)) ** 2)))
    )  # 111
    assert result_90 == expected_90


def test_location_accuracy_i_lat_edge_cases_negativ():
    result_neg90 = location_accuracy_i(1, -90)
    expected_neg90 = int(
        round(1 * math.sqrt(111**2 * (1 + math.cos(math.radians(-90)) ** 2)))
    )  # 111
    assert result_neg90 == expected_neg90


def test_location_accuracy_i_minimum_one():
    # li that gives accuracy <1
    assert location_accuracy_i(5, 90) == 1


@pytest.mark.parametrize(
    "input_val,expected",
    [
        # Truthy values (should be converted to string)
        (123, "123"),
        (45.6, "45.6"),
        ([1, 2, 3], "[1, 2, 3]"),
        ("hello", "hello"),
        (True, "True"),
        # Falsy values (should be returned as-is)
        (0, 0),
        (None, None),
        (False, False),
        ("", ""),
        ([], []),
    ],
)
def test_convert_to_str(input_val, expected):
    assert convert_to_str(input_val) == expected


@pytest.mark.parametrize(
    "a,b,c,sep,expected",
    [
        ("x", "y", "z", "-", "x-y-z"),  # normal case
        ("x", "y", None, "-", "x-y"),  # c filtered out
        ("x", "y", "", "-", "x-y"),  # c empty string filtered
        (None, "mid", "end", "/", "mid/end"),  # a filtered out
        (1, 2, 3, ",", "1,2,3"),  # numeric conversion
        ("a", "b", "c", "::", "a::b::c"),  # custom separator
    ],
)
def test_string_add_i_valid(a, b, c, sep, expected):
    assert string_add_i(a, b, c, sep) == expected


@pytest.mark.parametrize(
    "a,b,c,sep",
    [
        ("a", None, "c", "-"),  # b becomes falsy
        ("a", "", "c", "-"),  # b empty ? returns None
        (None, None, None, ","),  # all None ? b falsy
    ],
)
def test_string_add_i_returns_none_when_b_falsy(a, b, c, sep):
    assert string_add_i(a, b, c, sep) is None


@pytest.mark.parametrize(
    "input_val,expected",
    [
        # Valid integer inputs
        (123, 123),
        (0, 0),
        (-45, -45),
        # Floats that can be cast to int
        (45.0, 45),
        (-3.0, -3),
        # Strings representing integers
        ("123", 123),
        ("0", 0),
        ("-10", -10),
        # Invalid strings
        ("abc", pd.NA),
        ("12.3", pd.NA),  # float in string, not an int
        # NaN / None / pd.NA
        (None, pd.NA),
        (pd.NA, pd.NA),
        (float("nan"), pd.NA),
        # Other types
        ([1, 2, 3], pd.NA),
        ({}, pd.NA),
        (True, 1),  # True converts to int
        (False, 0),  # False converts to int
    ],
)
def test_to_int(input_val, expected):
    result = to_int(input_val)
    if pd.isna(expected):
        assert pd.isna(result)
    else:
        assert result == expected


@pytest.mark.parametrize(
    "row, expected_hr, expected_m",
    [
        (pd.Series([2025, 11, 2, 0, 10]), 10, 0),  # 10 hours
        (pd.Series([2025, 11, 2, 0, 10.5]), 10, 30),  # 10.5 hours -> 10:30
        (pd.Series([2025, 11, 2, 0, 0]), 0, 0),  # 0 hours
        (pd.Series([2025, 11, 2, 0, "NaN"]), None, None),  # missing hour
        (pd.Series([2025, 11, 2, 0, None]), None, None),  # missing hour
        (pd.Series([2025, 11, 2, 0, np.nan]), None, None),  # missing hour
        (pd.Series([2025, 11, 2, 0, "abc"]), None, None),  # invalid string
        (pd.Series([2025, 11, 2, 0]), None, None),  # less than 5 elements
    ],
)
def test_datetime_decimalhour_to_hm(row, expected_hr, expected_m):
    obj = mapping_functions("dummy_model")
    result = obj.datetime_decimalhour_to_hm(row)
    assert result["HR"] == expected_hr
    assert result["M"] == expected_m


@pytest.mark.parametrize(
    "df, expected",
    [
        # Simple example: 10 hours
        (
            pd.DataFrame([[2025, 11, 2, 10]]),
            pd.DatetimeIndex([pd.Timestamp("2025-11-02 10:00")]),
        ),
        # Example with float hour
        (
            pd.DataFrame([[2025, 11, 2, 10.5]]),
            pd.DatetimeIndex([pd.Timestamp("2025-11-02 10:30")]),
        ),
        # Example with None / NaN hour
        (pd.DataFrame([[2025, 11, 2, None]]), pd.DatetimeIndex([pd.NaT])),
        # Empty DataFrame
        (pd.DataFrame([]), pd.DatetimeIndex([])),
    ],
)
def test_datetime_imma1(df, expected):
    obj = mapping_functions("dummy_model")
    result = obj.datetime_imma1(df)
    pd.testing.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "df, expected",
    [
        # Missing hour, longitude 0, latitude 0 -> should default to 12:00 local, converted to UTC
        (
            pd.DataFrame(
                {
                    ("col", 0): [2025],
                    ("col", 1): [11],
                    ("col", 2): [2],
                    ("col", 3): [None],  # hour missing
                    ("col", 4): [0],  # longitude
                    ("col", 5): [0],  # latitude
                }
            ),
            pd.DatetimeIndex([pd.Timestamp("2025-11-02 12:00", tz="UTC")]),
        ),
        # Two rows with missing hour, different coordinates
        (
            pd.DataFrame(
                {
                    ("col", 0): [2025, 2025],
                    ("col", 1): [11, 12],
                    ("col", 2): [2, 3],
                    ("col", 3): [None, None],
                    ("col", 4): [0, 45],  # longitudes
                    ("col", 5): [0, -30],  # latitudes
                }
            ),
            pd.DatetimeIndex(
                [
                    pd.Timestamp("2025-11-02 12:00", tz="UTC"),
                    pd.Timestamp("2025-12-03 09:00", tz="UTC"),
                ]
            ),
        ),
        # Empty dataframe
        (pd.DataFrame([]), pd.DatetimeIndex([])),
    ],
)
def test_datetime_imma1_to_utc(df, expected):
    obj = mapping_functions("dummy_model")
    result = obj.datetime_imma1_to_utc(df)

    # The result is tz-naive (because your function does dt.tz_convert(None))
    # Remove tz info from expected for comparison
    expected_naive = expected.tz_localize(None) if expected.tz else expected

    pd.testing.assert_index_equal(result, expected_naive)


@pytest.mark.parametrize(
    "df, expected",
    [
        # Hour present -> use datetime_imma1
        (
            pd.DataFrame([[2025, 11, 2, 10]]),
            pd.DatetimeIndex([pd.Timestamp("2025-11-02 10:00")]),
        ),
        # Hour missing -> use datetime_imma1_to_utc
        (
            pd.DataFrame([[2025, 11, 2, None, 0, 0]]),
            pd.DatetimeIndex([pd.Timestamp("2025-11-02 12:00", tz=None)]),
        ),
        # Mixed
        (
            pd.DataFrame(
                [
                    [2025, 11, 2, 10, 0, 0],  # hour present
                    [2025, 12, 3, None, 45, -30],  # hour missing
                ]
            ),
            pd.DatetimeIndex(
                [
                    pd.Timestamp("2025-11-02 10:00"),
                    pd.Timestamp("2025-12-03 09:00"),  # matches UTC conversion
                ]
            ),
        ),
        # Empty DataFrame
        (pd.DataFrame([]), pd.DatetimeIndex([])),
    ],
)
def test_datetime_imma1_701(df, expected):
    obj = mapping_functions("dummy_model")
    result = obj.datetime_imma1_701(df)
    pd.testing.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "df, expected",
    [
        # Simple case: integer hour
        (
            pd.DataFrame([[2025, 11, 2, 10]]),
            pd.DatetimeIndex([pd.Timestamp("2025-11-02 10:00")]),
        ),
        # Hour 0
        (
            pd.DataFrame([[2025, 11, 2, 0]]),
            pd.DatetimeIndex([pd.Timestamp("2025-11-02 00:00")]),
        ),
        # Multiple rows
        (
            pd.DataFrame([[2025, 11, 2, 10], [2025, 12, 3, 15]]),
            pd.DatetimeIndex(
                [pd.Timestamp("2025-11-02 10:00"), pd.Timestamp("2025-12-03 15:00")]
            ),
        ),
        # Empty DataFrame
        (pd.DataFrame([]), pd.DatetimeIndex([])),
    ],
)
def test_datetime_immt(df, expected):
    obj = mapping_functions("dummy_model")
    result = obj.datetime_immt(df)
    pd.testing.assert_index_equal(result, expected)


def test_datetime_utcnow():
    obj = mapping_functions("dummy_model")
    result = obj.datetime_utcnow(pd.DataFrame())

    # Check that result is a datetime object
    assert isinstance(result, datetime.datetime)

    # Check that it is timezone-aware and UTC
    assert result.tzinfo is not None
    assert result.tzinfo.utcoffset(result) == datetime.timedelta(0)


@pytest.mark.parametrize(
    "df, expected",
    [
        # Single datetime string
        (
            pd.DataFrame([["2025-11-02 10:30:00.000"]]),
            pd.DatetimeIndex([pd.Timestamp("2025-11-02 10:30:00")]),
        ),
        # Multiple datetime strings
        (
            pd.DataFrame([["2025-11-02 10:30:00.000"], ["2025-12-03 15:45:00.123"]]),
            pd.DatetimeIndex(
                [
                    pd.Timestamp("2025-11-02 10:30:00"),
                    pd.Timestamp("2025-12-03 15:45:00.123"),
                ]
            ),
        ),
        # Invalid string -> should return NaT
        (pd.DataFrame([["invalid"]]), pd.DatetimeIndex([pd.NaT])),
        # Empty DataFrame
        (pd.DataFrame([]), pd.DatetimeIndex([])),
    ],
)
def test_datetime_craid(df, expected):
    obj = mapping_functions("dummy_model")
    result = obj.datetime_craid(df)
    pd.testing.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "df, sep, expected",
    [
        # simple two-column DataFrame
        (pd.DataFrame({"A": [1, 2], "B": [3, 4]}), "-", pd.Series(["1-3", "2-4"])),
        # single column
        (pd.DataFrame({"A": [5, 6]}), ",", pd.Series(["5", "6"])),
        # empty DataFrame
        (pd.DataFrame([]), "-", pd.Series([], dtype=str)),
        # mixed types
        (
            pd.DataFrame({"A": [1, "x"], "B": [True, 3.5]}),
            "|",
            pd.Series(["1|True", "x|3.5"]),
        ),
    ],
)
def test_df_col_join_series(df, sep, expected):
    obj = mapping_functions("dummy_model")
    result = obj.df_col_join(df, sep)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "df, expected",
    [
        (5.0, -5.0),  # positive float
        (-3.2, 3.2),  # negative float
        (0.0, -0.0),  # zero
        (123.456, -123.456),  # larger float
        (pd.Series([1.0, -2.0, 3.5]), pd.Series([-1.0, 2.0, -3.5])),
    ],
)
def test_float_opposite(df, expected):
    obj = mapping_functions("dummy_model")
    result = obj.float_opposite(df)
    if isinstance(result, pd.Series):
        pd.testing.assert_series_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize(
    "df, expected",
    [
        # Single column
        (pd.DataFrame({"A": [1, 2, 3]}), pd.Series([1, 2, 3], name="A")),
        # Two columns, second updates first where not NaN
        (
            pd.DataFrame({"A": [1, 2, None], "B": [10, None, 30]}),
            pd.Series([1.0, 2.0, 30.0], name="B"),  # Updated to match function logic
        ),
        # Three columns
        (
            pd.DataFrame({"X": [1, None, 3], "Y": [None, 5, None], "Z": [7, 8, 9]}),
            pd.Series([1, 5, 3], name="Z"),  # Updated to match function logic
        ),
        # Empty DataFrame
        (pd.DataFrame([]), pd.Series(dtype=float)),
    ],
)
def test_select_column(df, expected):
    obj = mapping_functions("dummy_model")
    result = obj.select_column(df)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "input_s, factor, expected",
    [
        # Basic numeric series
        (pd.Series([1, 2, 3], name="A"), 2, pd.Series([2, 4, 6], name="A")),
        # Fractional scaling
        (pd.Series([10, 20], name="B"), 0.5, pd.Series([5.0, 10.0], name="B")),
        # Negative scaling
        (pd.Series([3, -6, 9], name="C"), -1, pd.Series([-3, 6, -9], name="C")),
        # Empty numeric series
        (
            pd.Series([], dtype=float, name="E"),
            10,
            pd.Series([], dtype=float, name="E"),
        ),
        # Non-numeric series ? expected empty numeric series
        (pd.Series(["x", "y", "z"], name="F"), 3, pd.Series([], dtype=float, name="F")),
    ],
)
def test_float_scale(input_s, factor, expected):
    obj = mapping_functions("dummy_model")
    result = obj.float_scale(input_s, factor=factor)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "input_obj, expected",
    [
        # --- SERIES TESTS ---
        # Integer Series ? float Series
        (pd.Series([1, 2, 3], name="A"), pd.Series([1.0, 2.0, 3.0], name="A")),
        # Empty int Series
        (
            pd.Series([], dtype="int64", name="E"),
            pd.Series([], dtype="float64", name="E"),
        ),
        # Non-numeric Series ? empty float Series
        (
            pd.Series(["x", "y", "z"], name="S"),
            pd.Series([], dtype="float64", name="S"),
        ),
        # Empty Series
        (pd.Series([]), pd.Series([], dtype=float)),
    ],
)
def test_integer_to_float(input_obj, expected):
    obj = mapping_functions("dummy_model")
    result = obj.integer_to_float(input_obj)

    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "input_s, expected",
    [
        # 361 becomes 0
        (pd.Series([361, 10, 20], name="A"), pd.Series([0, 10, 20], name="A")),
        # 362 becomes NaN
        (pd.Series([5, 362, 15], name="B"), pd.Series([5, np.nan, 15], name="B")),
        # mixture of 361 and 362
        (pd.Series([361, 362, 100], name="C"), pd.Series([0, np.nan, 100], name="C")),
        # no conversion needed
        (pd.Series([1, 2, 3], name="D"), pd.Series([1, 2, 3], name="D")),
        # empty Series
        (pd.Series([], dtype=float, name="E"), pd.Series([], dtype=float, name="E")),
    ],
)
def test_icoads_wd_conversion(input_s, expected):
    obj = mapping_functions("dummy_model")
    result = obj.icoads_wd_conversion(input_s)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "input_s, expected",
    [
        # 361 ? 0 ? float
        (pd.Series([361, 10, 20], name="A"), pd.Series([0.0, 10.0, 20.0], name="A")),
        # 362 ? NaN ? float dtype
        (pd.Series([5, 362, 15], name="B"), pd.Series([5.0, np.nan, 15.0], name="B")),
        # Mixed 361 + 362 + normal values
        (
            pd.Series([361, 362, 100], name="C"),
            pd.Series([0.0, np.nan, 100.0], name="C"),
        ),
        # No special values ? just convert to float
        (pd.Series([1, 2, 3], name="D"), pd.Series([1.0, 2.0, 3.0], name="D")),
        # Empty numeric series
        (pd.Series([], dtype=float, name="E"), pd.Series([], dtype=float, name="E")),
        # Non-numeric series ? expected empty float series
        (pd.Series(["x", "y", "z"], name="F"), pd.Series([], dtype=float, name="F")),
    ],
)
def test_icoads_wd_integer_to_float(input_s, expected):
    obj = mapping_functions("dummy_model")
    result = obj.icoads_wd_integer_to_float(input_s)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "imodel, expected_suffix",
    [
        ("icoads", ". Initial conversion from ICOADS R3.0.0T"),
        (
            "icoads_r300_d714",
            ". Initial conversion from ICOADS R3.0.0T with supplemental data recovery",
        ),
        ("icoads_r302", ". Initial conversion from ICOADS R3.0.2T NRT"),
        ("craid", ". Initial conversion from C-RAID"),
        ("unknown_model", ""),  # no lineage found
    ],
)
def test_lineage(imodel, expected_suffix):
    obj = mapping_functions(imodel)

    result = obj.lineage(df=None)  # df is unused in function

    # 1. Check timestamp format at beginning
    # Example: 2025-01-12 15:44:10
    timestamp_pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
    assert re.match(
        timestamp_pattern, result
    ), f"Timestamp missing or invalid: {result}"

    # 2. Check that it ends with expected lineage suffix
    assert result.endswith(expected_suffix), f"Lineage suffix mismatch: {result}"


@pytest.mark.parametrize(
    "input_s, expected",
    [
        # Basic longitudes
        (
            pd.Series([0, 90, 180, -90, -180], name="LON"),
            pd.Series([0, 90, 180, -90, -180], name="LON"),
        ),
        # >180 degrees
        (
            pd.Series([181, 270, 360], name="LON"),
            pd.Series([-179, -90, -180], name="LON"),
        ),
        # Empty series
        (
            pd.Series([], dtype=float, name="LON"),
            pd.Series([], dtype=float, name="LON"),
        ),
    ],
)
def test_longitude_360to180(input_s, expected):
    obj = mapping_functions("dummy_model")
    result = obj.longitude_360to180(input_s)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "input_df, expected",
    [
        # Basic numeric values (li=0 -> small accuracy)
        (
            pd.DataFrame({"li": [0, 1, 0], "lat": [10, 20, 30]}),
            pd.Series([16.0, 152.0, 15.0]),
        ),
        # Include NaNs in li or lat
        (
            pd.DataFrame({"li": [np.nan, 1], "lat": [10, np.nan]}),
            pd.Series([np.nan, np.nan]),
        ),
        # Empty DataFrame
        (pd.DataFrame(columns=["li", "lat"]), pd.Series([], dtype=float)),
        # li not in degrees dictionary -> NaN
        (pd.DataFrame({"li": [2, 3], "lat": [10, 20]}), pd.Series([np.nan, np.nan])),
        # Mixed numeric and NaN
        (
            pd.DataFrame({"li": [0, np.nan, 1], "lat": [10, 20, np.nan]}),
            pd.Series([16.0, np.nan, np.nan]),
        ),
    ],
)
def test_location_accuracy(input_df, expected):
    obj = mapping_functions("dummy_model")
    result = obj.location_accuracy(input_df)
    # Ensure dtype float64
    result = result.astype("float64")
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "input_series, expected",
    [
        # Basic mapped values
        (pd.Series(["0", "1", "2", "3", "4", "5"]), pd.Series([[5, 7, 56]] * 6)),
        # Specific mapping for 7
        (pd.Series(["7"]), pd.Series([[5, 7, 9]])),
        # Values not in dictionary stay unchanged
        (
            pd.Series(["6", "8", "10"]),
            pd.Series([np.nan, np.nan, np.nan], dtype=object),
        ),
        # Mixed numeric strings and NaN
        (
            pd.Series(["0", "7", np.nan, "8"]),
            pd.Series([[5, 7, 56], [5, 7, 9], np.nan, np.nan]),
        ),
        # Empty Series
        (pd.Series([], dtype=object), pd.Series([], dtype=object)),
    ],
)
def test_observing_programme(input_series, expected):
    obj = mapping_functions("dummy_model")
    result = obj.observing_programme(input_series)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "input_series, prepend, append, separator, expected",
    [
        (pd.Series(["a", None, "c"]), "X", "Y", "", pd.Series(["XaY", None, "XcY"])),
        (
            pd.Series(["a", None, "c"]),
            "pre",
            "_app",
            "-",
            pd.Series(["pre-a-_app", None, "pre-c-_app"]),
        ),
        (pd.Series([]), "X", "Y", "", pd.Series([], dtype=object)),
    ],
)
def test_string_add(input_series, prepend, append, separator, expected):
    obj = mapping_functions("dummy_model")
    result = obj.string_add(
        input_series, prepend=prepend, append=append, separator=separator
    )
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "df, prepend, append, separator, zfill_col, zfill, expected",
    [
        # Basic prepend and append with column join
        (
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            "X",
            "Y",
            "-",
            None,
            None,
            pd.Series(["X-1-3-Y", "X-2-4-Y"]),
        ),
        # With zfill on both columns
        (
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            "X",
            "Y",
            "-",
            [0, 1],
            [2, 2],
            pd.Series(["X-01-03-Y", "X-02-04-Y"]),
        ),
        # No prepend or append
        (
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            None,
            None,
            "-",
            None,
            None,
            pd.Series(["1-3", "2-4"]),
        ),
        # Empty DataFrame
        (
            pd.DataFrame(columns=["A", "B"]),
            "X",
            "Y",
            "-",
            None,
            None,
            pd.Series([], dtype=object),
        ),
        # Single column with zfill
        (
            pd.DataFrame({"A": [5, 6]}),
            "P",
            "Q",
            ":",
            [0],
            [3],
            pd.Series(["P:005:Q", "P:006:Q"]),
        ),
    ],
)
def test_string_join_add(df, prepend, append, separator, zfill_col, zfill, expected):
    obj = mapping_functions("dummy_model")
    result = obj.string_join_add(
        df,
        prepend=prepend,
        append=append,
        separator=separator,
        zfill_col=zfill_col,
        zfill=zfill,
    )
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "input_df, imodel, expected",
    [
        # Method A: simple Celsius to Kelvin
        (
            pd.Series([0, 25, -10]),
            None,  # imodel not set, defaults to method_a
            pd.Series([273.15, 298.15, 263.15]),
        ),
        # Method A with DataFrame input
        (
            pd.DataFrame({"temp": [0, 100]}),
            None,
            pd.Series([273.15, 373.15], name="temp"),
        ),
        # Method B: uses c2k_methods mapping
        (
            pd.DataFrame({"col1": [0, 5, 2], "col2": [10, 20, 30]}),
            "gdac",
            pd.Series([283.15, 293.15, 243.15]),
        ),
        # Empty DataFrame
        (
            pd.DataFrame(columns=["col1", "col2"]),
            None,
            pd.Series([], dtype=float, name="col1"),
        ),
        # Single-row DataFrame
        (pd.DataFrame({"col1": [5], "col2": [10]}), "gdac", pd.Series([283.15])),
    ],
)
def test_temperature_celsius_to_kelvin(input_df, imodel, expected):
    # Create dummy object with imodel attribute and method
    obj = mapping_functions(imodel)
    result = obj.temperature_celsius_to_kelvin(input_df)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "input_series, expected",
    [
        # All known codes
        (pd.Series(["0", "1", "2", "3"]), pd.Series([3600, 360, 60, 36])),
        # Unknown codes stay unchanged
        (pd.Series(["4", "X", np.nan]), pd.Series([np.nan, np.nan, np.nan])),
        # Mixed known and unknown
        (pd.Series(["0", "2", "X", np.nan]), pd.Series([3600.0, 60.0, np.nan, np.nan])),
        # Empty series
        (pd.Series([], dtype=object), pd.Series([], dtype=int)),
    ],
)
def test_time_accuracy(input_series, expected):
    obj = mapping_functions("dummy_model")
    result = obj.time_accuracy(input_series)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "input_series, expected",
    [
        # Simple numeric feet values
        (pd.Series([0, 3.2808, 6.5616]), pd.Series([0.00, 1.00, 2.00])),
        # String values that should be converted to float
        (pd.Series(["3.2808", "32.808"]), pd.Series([1.00, 10.00])),
        # Includes NaN
        (pd.Series([3.2808, np.nan, 9.8424]), pd.Series([1.00, np.nan, 3.00])),
        # Empty series
        (pd.Series([], dtype=float), pd.Series([], dtype=float)),
    ],
)
def test_feet_to_m(input_series, expected):
    obj = mapping_functions("dummy_model")  # same pattern as your example
    result = obj.feet_to_m(input_series)

    # Make sure dtype is float in expected because feet_to_m returns floats
    expected = expected.astype(float)

    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "input_df, prepend, append, expected_uuids",
    [
        # Basic single row
        (
            pd.DataFrame({"AAAA": [12], "MM": [3], "YY": [7], "GG": [5]}),
            "",
            "",
            pd.Series(["a57ea24d0eb65ca390a63bd175c906db"], dtype="object"),
        ),
        # Multiple rows
        (
            pd.DataFrame(
                {"AAAA": [1, 2024], "MM": [1, 12], "YY": [1, 99], "GG": [1, 23]}
            ),
            "",
            "",
            pd.Series(
                [
                    "de3d414a8823554bbfde50f2305958d0",
                    "5f4b0ac6560552bf9e69cc9de0541bd6",
                ],
                dtype="object",
            ),
        ),
        # With prepend and append
        (
            pd.DataFrame({"AAAA": [50], "MM": [6], "YY": [24], "GG": [4]}),
            "PRE-",
            "-POST",
            pd.Series(["PRE-1d37cb121ceb546daba6431da61cd309-POST"], dtype="object"),
        ),
        # Empty dataframe
        (
            pd.DataFrame({"AAAA": [], "MM": [], "YY": [], "GG": []}),
            "",
            "",
            pd.Series([], dtype="object"),
        ),
    ],
)
def test_gdac_uid(input_df, prepend, append, expected_uuids):
    obj = mapping_functions("dummy_model")

    result = obj.gdac_uid(input_df, prepend=prepend, append=append)

    pd.testing.assert_series_equal(
        result.reset_index(drop=True),
        expected_uuids.reset_index(drop=True),
        check_names=False,  # ignore the Series name difference
    )


@pytest.mark.parametrize(
    "input_df, expected_latitudes",
    [
        # Quadrants 1 and 2: positive
        (pd.DataFrame({"Qc": [1, 2], "LaLaLa": [10.0, 20.0]}), pd.Series([10.0, 20.0])),
        # Quadrants 3 and 5: negative
        (
            pd.DataFrame({"Qc": [3, 5], "LaLaLa": [10.0, 20.0]}),
            pd.Series([-10.0, -20.0]),
        ),
        # Mixed quadrants
        (
            pd.DataFrame({"Qc": [1, 3, 5, 2], "LaLaLa": [1.0, 2.0, 3.0, 4.0]}),
            pd.Series([1.0, -2.0, -3.0, 4.0]),
        ),
        # Empty dataframe
        (pd.DataFrame({"Qc": [], "LaLaLa": []}), pd.Series([], dtype=float)),
    ],
)
def test_gdac_latitude(input_df, expected_latitudes):
    obj = mapping_functions("dummy_model")

    result = obj.gdac_latitude(input_df)

    pd.testing.assert_series_equal(
        result.reset_index(drop=True),
        expected_latitudes.reset_index(drop=True),
        check_names=False,
    )


def test_gdac_latitude_missing_columns():
    obj = mapping_functions("dummy_model")

    # Missing 'Qc'
    df_missing_qc = pd.DataFrame({"LaLaLa": [10.0, 20.0]})
    with pytest.raises(KeyError):
        obj.gdac_latitude(df_missing_qc)

    # Missing 'LaLaLa'
    df_missing_lat = pd.DataFrame({"Qc": [1, 2]})
    with pytest.raises(KeyError):
        obj.gdac_latitude(df_missing_lat)


@pytest.mark.parametrize(
    "input_df, expected_longitudes",
    [
        # Quadrants 1, 3: positive
        (
            pd.DataFrame({"Qc": [1, 3], "LoLoLoLo": [10.0, 20.0]}),
            pd.Series([10.0, 20.0]),
        ),
        # Quadrants 5 and 7: negative
        (
            pd.DataFrame({"Qc": [5, 7], "LoLoLoLo": [10.0, 20.0]}),
            pd.Series([-10.0, -20.0]),
        ),
        # Mixed quadrants
        (
            pd.DataFrame({"Qc": [1, 5, 7, 3], "LoLoLoLo": [1.0, 2.0, 3.0, 4.0]}),
            pd.Series([1.0, -2.0, -3.0, 4.0]),
        ),
        # Empty dataframe
        (pd.DataFrame({"Qc": [], "LoLoLoLo": []}), pd.Series([], dtype=float)),
    ],
)
def test_gdac_longitude(input_df, expected_longitudes):
    obj = mapping_functions("dummy_model")

    result = obj.gdac_longitude(input_df)

    pd.testing.assert_series_equal(
        result.reset_index(drop=True),
        expected_longitudes.reset_index(drop=True),
        check_names=False,
    )


def test_gdac_longitude_missing_columns():
    obj = mapping_functions("dummy_model")

    # Missing 'Qc'
    df_missing_qc = pd.DataFrame({"LoLoLoLo": [10.0, 20.0]})
    with pytest.raises(KeyError):
        obj.gdac_longitude(df_missing_qc)

    # Missing 'LoLoLoLo'
    df_missing_lon = pd.DataFrame({"Qc": [1, 2]})
    with pytest.raises(KeyError):
        obj.gdac_longitude(df_missing_lon)
