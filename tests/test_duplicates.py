from __future__ import annotations

import pytest  # noqa
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from cdm_reader_mapper.cdm_mapper import duplicate_check

from ._duplicates import (
    compare_kwargs_,
    df_craid,
    df_icoads,
    exp1,
    exp2,
    exp3,
    exp4,
    exp5,
    exp6,
    exp7,
    exp8,
    method_kwargs_,
)


@pytest.mark.parametrize(
    "method, method_kwargs, compare_kwargs, ignore_columns, ignore_entries, offsets, expected",
    [
        (None, None, None, None, None, None, exp1),
        (
            None,
            None,
            None,
            None,
            {"primary_station_id": ["SHIP", "MASKSTID"]},
            None,
            exp2,
        ),
        (
            None,
            None,
            None,
            None,
            {"station_speed": "null", "station_course": "null"},
            None,
            exp7,
        ),
        (
            None,
            None,
            None,
            None,
            {
                "primary_station_id": ["SHIP", "MASKSTID"],
                "station_speed": "null",
                "station_course": "null",
            },
            None,
            exp8,
        ),
        (None, method_kwargs_, None, None, None, None, exp1),
        (None, None, compare_kwargs_, None, None, None, exp3),
        (None, None, None, ["primary_station_id"], None, None, exp4),
        (
            None,
            None,
            None,
            None,
            None,
            {"latitude": 1.0, "longitude": 1.0, "report_timestamp": 360},
            exp5,
        ),
        ("Block", {"left_on": "report_timestamp"}, None, None, None, None, exp6),
    ],
)
def test_duplicates_flag(
    method,
    method_kwargs,
    compare_kwargs,
    ignore_columns,
    ignore_entries,
    offsets,
    expected,
):
    if method is None:
        method = "SortedNeighbourhood"
    DupDetect = duplicate_check(
        df_icoads,
        method=method,
        method_kwargs=method_kwargs,
        compare_kwargs=compare_kwargs,
        ignore_columns=ignore_columns,
        ignore_entries=ignore_entries,
        offsets=offsets,
    )
    DupDetect.flag_duplicates()
    result = DupDetect.result
    assert_array_equal(result["duplicate_status"], expected["duplicate_status"])
    assert_array_equal(result["report_quality"], expected["report_quality"])
    assert_array_equal(result["duplicates"], expected["duplicates"])


def test_duplicates_remove():
    DupDetect = duplicate_check(
        df_icoads,
        ignore_entries={
            "primary_station_id": ["SHIP", "MASKSTID"],
            "station_speed": "null",
            "station_course": "null",
        },
    )
    DupDetect.remove_duplicates()
    expected = DupDetect.data.iloc[[0, 1, 2, 4, 6, 8, 10, 12, 15, 17, 18]]
    assert_frame_equal(expected, DupDetect.result)


def test_duplicates_craid():
    DupDetect = duplicate_check(df_craid, ignore_columns="primary_station_id")
    DupDetect.flag_duplicates()
    assert_array_equal(DupDetect.result["duplicate_status"], [0] * 10)
    assert_array_equal(DupDetect.result["report_quality"], [2] * 10)
    assert_array_equal(DupDetect.result["duplicates"], ["null"] * 10)
