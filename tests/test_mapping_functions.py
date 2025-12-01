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
        (10, 30, "N", 10.5),
        (0, 45, "N", 0.75),
        (90, 0, "N", 90.0),
        (10, 30, "S", -10.5),
        (0, 45, "S", -0.75),
        (90, 0, "S", -90.0),
        (10, 59, "N", 10.98),
        (10, 1, "S", -10.02),
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
    local_times = pd.date_range("2025-11-17 00:00", periods=3, freq="h")

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
        (0, 0),
        (90, 90),
        (180, 180),
        (190, -170),
        (270, -90),
        (360, -180),
        (-90, -90),
        (-190, -190),
        (450, -90),
    ],
)
def test_longitude_360to180_i(lon, expected):
    assert longitude_360to180_i(lon) == expected


@pytest.mark.parametrize(
    "li,lat,expected",
    [
        (0, 0, 16),
        (1, 0, 157),
        (4, 0, 3),
        (5, 0, 1),
        ("1", 0, 157),
        ("4", 0, 3),
    ],
)
def test_location_accuracy_i_valid_values(li, lat, expected):
    assert location_accuracy_i(li, lat) == expected


@pytest.mark.parametrize("li,lat", [(2, 0), ("abc", 0)])
def test_location_accuracy_i_invalid_li(li, lat):
    assert np.isnan(location_accuracy_i(li, lat))


def test_location_accuracy_i_lat_edge_cases_positiv():
    result_90 = location_accuracy_i(1, 90)
    expected_90 = int(
        round(1 * math.sqrt(111**2 * (1 + math.cos(math.radians(90)) ** 2)))
    )
    assert result_90 == expected_90


def test_location_accuracy_i_lat_edge_cases_negativ():
    result_neg90 = location_accuracy_i(1, -90)
    expected_neg90 = int(
        round(1 * math.sqrt(111**2 * (1 + math.cos(math.radians(-90)) ** 2)))
    )
    assert result_neg90 == expected_neg90


def test_location_accuracy_i_minimum_one():
    assert location_accuracy_i(5, 90) == 1


@pytest.mark.parametrize(
    "input_val,expected",
    [
        (123, "123"),
        (45.6, "45.6"),
        ([1, 2, 3], "[1, 2, 3]"),
        ("hello", "hello"),
        (True, "True"),
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
        ("x", "y", "z", "-", "x-y-z"),
        ("x", "y", None, "-", "x-y"),
        ("x", "y", "", "-", "x-y"),
        (None, "mid", "end", "/", "mid/end"),
        (1, 2, 3, ",", "1,2,3"),
        ("a", "b", "c", "::", "a::b::c"),
    ],
)
def test_string_add_i_valid(a, b, c, sep, expected):
    assert string_add_i(a, b, c, sep) == expected


@pytest.mark.parametrize(
    "a,b,c,sep",
    [
        ("a", None, "c", "-"),
        ("a", "", "c", "-"),
        (None, None, None, ","),
    ],
)
def test_string_add_i_returns_none_when_b_falsy(a, b, c, sep):
    assert string_add_i(a, b, c, sep) is None


@pytest.mark.parametrize(
    "input_val,expected",
    [
        (123, 123),
        (0, 0),
        (-45, -45),
        (45.0, 45),
        (-3.0, -3),
        ("123", 123),
        ("0", 0),
        ("-10", -10),
        ("abc", pd.NA),
        ("12.3", pd.NA),
        (None, pd.NA),
        (pd.NA, pd.NA),
        (float("nan"), pd.NA),
        ([1, 2, 3], pd.NA),
        ({}, pd.NA),
        (True, 1),
        (False, 0),
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
        (pd.Series([2025, 11, 2, 0, 10]), 10, 0),
        (pd.Series([2025, 11, 2, 0, 10.5]), 10, 30),
        (pd.Series([2025, 11, 2, 0, 0]), 0, 0),
        (pd.Series([2025, 11, 2, 0, "NaN"]), None, None),
        (pd.Series([2025, 11, 2, 0, None]), None, None),
        (pd.Series([2025, 11, 2, 0, np.nan]), None, None),
        (pd.Series([2025, 11, 2, 0, "abc"]), None, None),
        (pd.Series([2025, 11, 2, 0]), None, None),
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
        (
            pd.DataFrame([[2025, 11, 2, 10]]),
            pd.DatetimeIndex([pd.Timestamp("2025-11-02 10:00")]),
        ),
        (
            pd.DataFrame([[2025, 11, 2, 10.5]]),
            pd.DatetimeIndex([pd.Timestamp("2025-11-02 10:30")]),
        ),
        (pd.DataFrame([[2025, 11, 2, None]]), pd.DatetimeIndex([pd.NaT])),
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
        (
            pd.DataFrame(
                {
                    ("col", 0): [2025],
                    ("col", 1): [11],
                    ("col", 2): [2],
                    ("col", 3): [None],
                    ("col", 4): [0],
                    ("col", 5): [0],
                }
            ),
            pd.DatetimeIndex([pd.Timestamp("2025-11-02 12:00", tz="UTC")]),
        ),
        (
            pd.DataFrame(
                {
                    ("col", 0): [2025, 2025],
                    ("col", 1): [11, 12],
                    ("col", 2): [2, 3],
                    ("col", 3): [None, None],
                    ("col", 4): [0, 45],
                    ("col", 5): [0, -30],
                }
            ),
            pd.DatetimeIndex(
                [
                    pd.Timestamp("2025-11-02 12:00", tz="UTC"),
                    pd.Timestamp("2025-12-03 09:00", tz="UTC"),
                ]
            ),
        ),
        (pd.DataFrame([]), pd.DatetimeIndex([])),
    ],
)
def test_datetime_imma1_to_utc(df, expected):
    obj = mapping_functions("dummy_model")
    result = obj.datetime_imma1_to_utc(df)

    expected_naive = expected.tz_localize(None) if expected.tz else expected

    pd.testing.assert_index_equal(result, expected_naive)


@pytest.mark.parametrize(
    "df, expected",
    [
        (
            pd.DataFrame([[2025, 11, 2, 10]]),
            pd.DatetimeIndex([pd.Timestamp("2025-11-02 10:00")]),
        ),
        (
            pd.DataFrame([[2025, 11, 2, None, 0, 0]]),
            pd.DatetimeIndex([pd.Timestamp("2025-11-02 12:00", tz=None)]),
        ),
        (
            pd.DataFrame(
                [
                    [2025, 11, 2, 10, 0, 0],
                    [2025, 12, 3, None, 45, -30],
                ]
            ),
            pd.DatetimeIndex(
                [
                    pd.Timestamp("2025-11-02 10:00"),
                    pd.Timestamp("2025-12-03 09:00"),
                ]
            ),
        ),
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
        (
            pd.DataFrame([[2025, 11, 2, 10]]),
            pd.DatetimeIndex([pd.Timestamp("2025-11-02 10:00")]),
        ),
        (
            pd.DataFrame([[2025, 11, 2, 0]]),
            pd.DatetimeIndex([pd.Timestamp("2025-11-02 00:00")]),
        ),
        (
            pd.DataFrame([[2025, 11, 2, 10], [2025, 12, 3, 15]]),
            pd.DatetimeIndex(
                [pd.Timestamp("2025-11-02 10:00"), pd.Timestamp("2025-12-03 15:00")]
            ),
        ),
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

    assert isinstance(result, datetime.datetime)

    assert result.tzinfo is not None
    assert result.tzinfo.utcoffset(result) == datetime.timedelta(0)


@pytest.mark.parametrize(
    "df, expected",
    [
        (
            pd.DataFrame([["2025-11-02 10:30:00.000"]]),
            pd.DatetimeIndex([pd.Timestamp("2025-11-02 10:30:00")]),
        ),
        (
            pd.DataFrame([["2025-11-02 10:30:00.000"], ["2025-12-03 15:45:00.123"]]),
            pd.DatetimeIndex(
                [
                    pd.Timestamp("2025-11-02 10:30:00"),
                    pd.Timestamp("2025-12-03 15:45:00.123"),
                ]
            ),
        ),
        (pd.DataFrame([["invalid"]]), pd.DatetimeIndex([pd.NaT])),
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
        (pd.DataFrame({"A": [1, 2], "B": [3, 4]}), "-", pd.Series(["1-3", "2-4"])),
        (pd.DataFrame({"A": [5, 6]}), ",", pd.Series(["5", "6"])),
        (pd.DataFrame([]), "-", pd.Series([], dtype=str)),
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
        (5.0, -5.0),
        (-3.2, 3.2),
        (0.0, -0.0),
        (123.456, -123.456),
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
        (pd.DataFrame({"A": [1, 2, 3]}), pd.Series([1, 2, 3], name="A")),
        (
            pd.DataFrame({"A": [1, 2, None], "B": [10, None, 30]}),
            pd.Series([1.0, 2.0, 30.0], name="B"),
        ),
        (
            pd.DataFrame({"X": [1, None, 3], "Y": [None, 5, None], "Z": [7, 8, 9]}),
            pd.Series([1, 5, 3], name="Z"),
        ),
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
        (pd.Series([1, 2, 3], name="A"), 2, pd.Series([2, 4, 6], name="A")),
        (pd.Series([10, 20], name="B"), 0.5, pd.Series([5.0, 10.0], name="B")),
        (pd.Series([3, -6, 9], name="C"), -1, pd.Series([-3, 6, -9], name="C")),
        (
            pd.Series([], dtype=float, name="E"),
            10,
            pd.Series([], dtype=float, name="E"),
        ),
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
        (pd.Series([1, 2, 3], name="A"), pd.Series([1.0, 2.0, 3.0], name="A")),
        (
            pd.Series([], dtype="int64", name="E"),
            pd.Series([], dtype="float64", name="E"),
        ),
        (pd.Series([]), pd.Series([], dtype=float)),
    ],
)
def test_integer_to_float(input_obj, expected):
    obj = mapping_functions("dummy_model")
    result = obj.integer_to_float(input_obj)

    pd.testing.assert_series_equal(result, expected)


def test_integer_to_float_raises():
    obj = mapping_functions("dummy_model")
    with pytest.raises(ValueError):
        obj.integer_to_float(pd.Series(["x", "y", "z"], name="S"))


@pytest.mark.parametrize(
    "input_s, expected",
    [
        (pd.Series([361, 10, 20], name="A"), pd.Series([0, 10, 20], name="A")),
        (pd.Series([5, 362, 15], name="B"), pd.Series([5, np.nan, 15], name="B")),
        (pd.Series([361, 362, 100], name="C"), pd.Series([0, np.nan, 100], name="C")),
        (pd.Series([1, 2, 3], name="D"), pd.Series([1, 2, 3], name="D")),
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
        (pd.Series([361, 10, 20], name="A"), pd.Series([0.0, 10.0, 20.0], name="A")),
        (pd.Series([5, 362, 15], name="B"), pd.Series([5.0, np.nan, 15.0], name="B")),
        (
            pd.Series([361, 362, 100], name="C"),
            pd.Series([0.0, np.nan, 100.0], name="C"),
        ),
        (pd.Series([1, 2, 3], name="D"), pd.Series([1.0, 2.0, 3.0], name="D")),
        (pd.Series([], dtype=float, name="E"), pd.Series([], dtype=float, name="E")),
    ],
)
def test_icoads_wd_integer_to_float(input_s, expected):
    obj = mapping_functions("dummy_model")
    result = obj.icoads_wd_integer_to_float(input_s)
    pd.testing.assert_series_equal(result, expected)


def test_icoads_wd_integer_to_float_raises():
    obj = mapping_functions("dummy_model")
    with pytest.raises(ValueError):
        obj.icoads_wd_integer_to_float(pd.Series(["x", "y", "z"], name="F"))


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
        ("unknown_model", ""),
    ],
)
def test_lineage(imodel, expected_suffix):
    obj = mapping_functions(imodel)

    result = obj.lineage(df=None)

    timestamp_pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
    assert re.match(
        timestamp_pattern, result
    ), f"Timestamp missing or invalid: {result}"

    assert result.endswith(expected_suffix), f"Lineage suffix mismatch: {result}"


@pytest.mark.parametrize(
    "input_s, expected",
    [
        (
            pd.Series([0, 90, 180, -90, -180], name="LON"),
            pd.Series([0, 90, 180, -90, -180], name="LON"),
        ),
        (
            pd.Series([181, 270, 360], name="LON"),
            pd.Series([-179, -90, -180], name="LON"),
        ),
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
        (
            pd.DataFrame({"li": [0, 1, 0], "lat": [10, 20, 30]}),
            pd.Series([16.0, 152.0, 15.0]),
        ),
        (
            pd.DataFrame({"li": [np.nan, 1], "lat": [10, np.nan]}),
            pd.Series([np.nan, np.nan]),
        ),
        (pd.DataFrame(columns=["li", "lat"]), pd.Series([], dtype=float)),
        (pd.DataFrame({"li": [2, 3], "lat": [10, 20]}), pd.Series([np.nan, np.nan])),
        (
            pd.DataFrame({"li": [0, np.nan, 1], "lat": [10, 20, np.nan]}),
            pd.Series([16.0, np.nan, np.nan]),
        ),
    ],
)
def test_location_accuracy(input_df, expected):
    obj = mapping_functions("dummy_model")
    result = obj.location_accuracy(input_df)
    result = result.astype("float64")
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "input_series, expected",
    [
        (pd.Series(["0", "1", "2", "3", "4", "5"]), pd.Series([[5, 7, 56]] * 6)),
        (pd.Series(["7"]), pd.Series([[5, 7, 9]])),
        (
            pd.Series(["6", "8", "10"]),
            pd.Series([np.nan, np.nan, np.nan], dtype=object),
        ),
        (
            pd.Series(["0", "7", np.nan, "8"]),
            pd.Series([[5, 7, 56], [5, 7, 9], np.nan, np.nan]),
        ),
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
        (
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            "X",
            "Y",
            "-",
            None,
            None,
            pd.Series(["X-1-3-Y", "X-2-4-Y"]),
        ),
        (
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            "X",
            "Y",
            "-",
            [0, 1],
            [2, 2],
            pd.Series(["X-01-03-Y", "X-02-04-Y"]),
        ),
        (
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            None,
            None,
            "-",
            None,
            None,
            pd.Series(["1-3", "2-4"]),
        ),
        (
            pd.DataFrame(columns=["A", "B"]),
            "X",
            "Y",
            "-",
            None,
            None,
            pd.Series([], dtype=object),
        ),
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
        (
            pd.Series([0, 25, -10]),
            None,
            pd.Series([273.15, 298.15, 263.15]),
        ),
        (
            pd.DataFrame({"temp": [0, 100]}),
            None,
            pd.Series([273.15, 373.15], name="temp"),
        ),
        (
            pd.DataFrame({"col1": [0, 5, 2], "col2": [10, 20, 30]}),
            "gdac",
            pd.Series([283.15, 293.15, 243.15]),
        ),
        (
            pd.DataFrame(columns=["col1", "col2"]),
            None,
            pd.Series([], dtype=float, name="col1"),
        ),
        (pd.DataFrame({"col1": [5], "col2": [10]}), "gdac", pd.Series([283.15])),
    ],
)
def test_temperature_celsius_to_kelvin(input_df, imodel, expected):
    obj = mapping_functions(imodel)
    result = obj.temperature_celsius_to_kelvin(input_df)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "input_series, expected",
    [
        (pd.Series(["0", "1", "2", "3"]), pd.Series([3600, 360, 60, 36])),
        (pd.Series(["4", "X", np.nan]), pd.Series([np.nan, np.nan, np.nan])),
        (pd.Series(["0", "2", "X", np.nan]), pd.Series([3600.0, 60.0, np.nan, np.nan])),
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
        (pd.Series([0, 3.2808, 6.5616]), pd.Series([0.00, 1.00, 2.00])),
        (pd.Series(["3.2808", "32.808"]), pd.Series([1.00, 10.00])),
        (pd.Series([3.2808, np.nan, 9.8424]), pd.Series([1.00, np.nan, 3.00])),
        (pd.Series([], dtype=float), pd.Series([], dtype=float)),
    ],
)
def test_feet_to_m(input_series, expected):
    obj = mapping_functions("dummy_model")
    result = obj.feet_to_m(input_series)

    expected = expected.astype(float)

    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "input_df, prepend, append, expected_uuids",
    [
        (
            pd.DataFrame({"AAAA": [12], "MM": [3], "YY": [7], "GG": [5]}),
            "",
            "",
            pd.Series(["a57ea24d0eb65ca390a63bd175c906db"], dtype="object"),
        ),
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
        (
            pd.DataFrame({"AAAA": [50], "MM": [6], "YY": [24], "GG": [4]}),
            "PRE-",
            "-POST",
            pd.Series(["PRE-1d37cb121ceb546daba6431da61cd309-POST"], dtype="object"),
        ),
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
        check_names=False,
    )


@pytest.mark.parametrize(
    "input_df, expected_latitudes",
    [
        (pd.DataFrame({"Qc": [1, 2], "LaLaLa": [10.0, 20.0]}), pd.Series([10.0, 20.0])),
        (
            pd.DataFrame({"Qc": [3, 5], "LaLaLa": [10.0, 20.0]}),
            pd.Series([-10.0, -20.0]),
        ),
        (
            pd.DataFrame({"Qc": [1, 3, 5, 2], "LaLaLa": [1.0, 2.0, 3.0, 4.0]}),
            pd.Series([1.0, -2.0, -3.0, 4.0]),
        ),
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

    df_missing_qc = pd.DataFrame({"LaLaLa": [10.0, 20.0]})
    with pytest.raises(KeyError):
        obj.gdac_latitude(df_missing_qc)

    df_missing_lat = pd.DataFrame({"Qc": [1, 2]})
    with pytest.raises(KeyError):
        obj.gdac_latitude(df_missing_lat)


@pytest.mark.parametrize(
    "input_df, expected_longitudes",
    [
        (
            pd.DataFrame({"Qc": [1, 3], "LoLoLoLo": [10.0, 20.0]}),
            pd.Series([10.0, 20.0]),
        ),
        (
            pd.DataFrame({"Qc": [5, 7], "LoLoLoLo": [10.0, 20.0]}),
            pd.Series([-10.0, -20.0]),
        ),
        (
            pd.DataFrame({"Qc": [1, 5, 7, 3], "LoLoLoLo": [1.0, 2.0, 3.0, 4.0]}),
            pd.Series([1.0, -2.0, -3.0, 4.0]),
        ),
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

    df_missing_qc = pd.DataFrame({"LoLoLoLo": [10.0, 20.0]})
    with pytest.raises(KeyError):
        obj.gdac_longitude(df_missing_qc)

    df_missing_lon = pd.DataFrame({"Qc": [1, 2]})
    with pytest.raises(KeyError):
        obj.gdac_longitude(df_missing_lon)
