from __future__ import annotations

import pytest  # noqa

import pandas as pd

from cdm_reader_mapper.duplicates.duplicates import (
    convert_series,
    add_history,
    add_duplicates,
    add_report_quality,
    set_comparer,
    remove_ignores,
    change_offsets,
    duplicate_check,
    DupDetect,
    reindex_nulls,
    Comparer,
)
from cdm_reader_mapper.duplicates._duplicate_settings import (
    _compare_kwargs,
    _histories,
    _method_kwargs,
    Compare,
)


def test_convert_series_basic():
    df = pd.DataFrame({"a": ["1", "2", "3"], "b": ["10.5", "20.5", "30.5"]})
    conversion = {"a": "int", "b": "float"}

    expected = pd.DataFrame({"a": [1, 2, 3], "b": [10.5, 20.5, 30.5]})

    result = convert_series(df, conversion)
    pd.testing.assert_frame_equal(result, expected)


def test_convert_series_null_replacement():
    df = pd.DataFrame({"a": ["1", "null", "3"], "b": ["null", "2.5", "null"]})
    conversion = {"a": "float", "b": "float"}

    expected = pd.DataFrame({"a": [1.0, 9999.0, 3.0], "b": [9999.0, 2.5, 9999.0]})

    result = convert_series(df, conversion)
    pd.testing.assert_frame_equal(result, expected)


def test_convert_series_date_to_float():
    df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02", "2023-01-03"]})
    conversion = {"date": "convert_date_to_float"}

    result = convert_series(df, conversion)
    expected = pd.DataFrame({"date": [0.0, 86400.0, 172800.0]})

    pd.testing.assert_frame_equal(result, expected)


def test_convert_series_mixed():
    df = pd.DataFrame(
        {
            "num": ["1", "null", "3"],
            "val": ["10.5", "20.5", "null"],
            "date": ["2023-01-01", "null", "2023-01-03"],
        }
    )
    conversion = {"num": "Int64", "val": "float", "date": "convert_date_to_float"}

    result = convert_series(df, conversion)
    expected = pd.DataFrame(
        {
            "num": [1, 9999, 3],
            "val": [10.5, 20.5, 9999.0],
            "date": [0.0, 9999.0, 172800.0],
        }
    )

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_add_history_basic():
    df = pd.DataFrame({"value": [10, 20, 30], "history": ["", "", ""]})

    updated_df = add_history(df, [0, 2])

    for idx in [0, 2]:
        for msg in _histories.values():
            assert msg in updated_df.loc[idx, "history"]

    assert updated_df.loc[1, "history"] == ""


def test_add_history_creates_column():
    df = pd.DataFrame({"value": [1, 2, 3]})

    updated_df = add_history(df, [0])
    for msg in _histories.values():
        assert msg in updated_df.loc[0, "history"]
    assert updated_df.loc[1, "history"] == ""


def test_add_duplicates_strings():
    df = pd.DataFrame({"report_id": ["A", "B", "C", "D"]})

    dups = pd.DataFrame(
        {
            0: [["B", "C"], ["D"]],
        },
        index=[0, 2],
    )

    updated = add_duplicates(df, dups)

    assert updated.loc[0, "duplicates"] == "{B,C}"
    assert updated.loc[2, "duplicates"] == "{D}"
    assert updated.loc[1, "duplicates"] == ""
    assert updated.loc[3, "duplicates"] == ""


def test_add_duplicates_indices():
    df = pd.DataFrame({"report_id": ["A", "B", "C", "D"]})

    dups = pd.DataFrame({0: [[1, 2], [3]]}, index=[0, 2])

    updated = add_duplicates(df, dups)

    assert updated.loc[0, "duplicates"] == "{B,C}"
    assert updated.loc[2, "duplicates"] == "{D}"


@pytest.mark.parametrize(
    "initial, indexes_bad, expected",
    [
        ([0, 0, 0], [1], [0, 1, 0]),
        ([0, 0, 2], [0, 2], [1, 0, 1]),
        ([1, 2, 3], [], [1, 2, 3]),
    ],
)
def test_add_report_quality(initial, indexes_bad, expected):
    df = pd.DataFrame({"report_quality": initial})
    result = add_report_quality(df, indexes_bad)
    pd.testing.assert_series_equal(
        result["report_quality"],
        pd.Series(expected, name="report_quality"),
        check_dtype=False,
    )


def test_set_comparer():
    compare_dict = {
        "col1": {"method": "exact"},
        "col2": {"method": "numeric", "kwargs": {"method": "step", "offset": 0.1}},
        "col3": {"method": "date2"},
    }
    comparer = set_comparer(compare_dict)
    assert isinstance(comparer, Compare)
    assert comparer.conversion["col2"] is float
    assert comparer.conversion["col3"] == "convert_date_to_float"


def test_remove_ignores():
    dic = {"a": 1, "b": ["x", "y"], "c": "z"}
    filtered = remove_ignores(dic, ["b", "c"])
    assert "b" not in filtered
    assert "c" not in filtered
    assert "a" in filtered


def test_change_offsets():
    dic = {"col1": {"kwargs": {"offset": 0.1}}, "col2": {"kwargs": {"offset": 0.2}}}
    new_offsets = {"col1": 0.5}
    updated = change_offsets(dic, new_offsets)
    assert updated["col1"]["kwargs"]["offset"] == 0.5
    assert updated["col2"]["kwargs"]["offset"] == 0.2


def test_duplicate_check_basic():
    df = pd.DataFrame(
        {
            "report_id": ["A", "B"],
            "primary_station_id": ["S1", "S2"],
            "longitude": [10, 20],
            "latitude": [50, 60],
            "report_timestamp": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "station_speed": [5, 6],
            "station_course": [100, 200],
        }
    )
    detector = duplicate_check(df, method="SortedNeighbourhood")
    assert isinstance(detector, DupDetect)
    assert detector.data.shape[0] == df.shape[0]


def test_reindex_nulls_orders_by_null_count():
    df = pd.DataFrame({"a": ["null", 1, "null", 2], "b": ["null", 2, 3, "null"]})
    result = reindex_nulls(df)

    expected_order = [1, 2, 3, 0]
    assert list(result.index) == expected_order


def test_reindex_nulls_empty_df():
    df = pd.DataFrame()
    result = reindex_nulls(df)
    assert result.equals(df)


def test_comparer_basic():
    df = pd.DataFrame(
        {
            "report_id": ["A", "B", "C"],
            "primary_station_id": [1, 1, 2],
            "longitude": [0.1, 0.15, 0.2],
            "latitude": [51.0, 51.01, 52.0],
            "report_timestamp": pd.to_datetime(
                ["2023-01-01 00:00", "2023-01-01 00:01", "2023-01-02 00:00"]
            ),
            "station_speed": [10.0, 12.0, 8.0],
            "station_course": [90, 180, 270],
        }
    )

    comp = Comparer(
        data=df,
        method="SortedNeighbourhood",
        method_kwargs=_method_kwargs,
        compare_kwargs=_compare_kwargs,
        convert_data=True,
    )

    assert isinstance(comp.data, pd.DataFrame)
    assert isinstance(comp.compared, pd.DataFrame)
    assert "primary_station_id" in comp.compared.columns


@pytest.fixture
def dummy_data():
    df = pd.DataFrame(
        {
            "report_id": ["A", "B", "C", "D"],
            "primary_station_id": [1, 1, 2, 1],
            "longitude": [0.1, 0.1, 0.2, 0.1],
            "latitude": [51.0, 51.0, 52.0, 51.0],
            "report_timestamp": pd.to_datetime(
                [
                    "2023-01-01 00:00",
                    "2023-01-01 00:00",
                    "2023-01-02 00:00",
                    "2023-01-01 00:00",
                ]
            ),
            "station_speed": [10.0, 10.0, 8.0, 10.0],
            "station_course": [90, 90, 180, 90],
            "report_quality": 2,
        }
    )
    df.index = [0, 1, 2, 3]
    return df


@pytest.fixture
def compared_matrix(dummy_data):
    index = pd.MultiIndex.from_tuples(
        [
            (0, 1),
            (2, 3),
        ]
    )
    compared = pd.DataFrame(
        {
            "primary_station_id": [True, False],
            "longitude": [True, False],
            "latitude": [True, False],
            "station_speed": [False, False],
            "station_course": [False, False],
        },
        index=index,
    )
    return compared


def test_get_duplicates_limit_and_equal_musts(dummy_data, compared_matrix):
    dd = DupDetect(dummy_data, compared_matrix, "SortedNeighbourhood", {}, {})

    # Test default limit
    matches_default = dd.get_duplicates(keep="first", limit=0.5)
    assert len(matches_default) > 0

    # Test custom limit
    matches_custom = dd.get_duplicates(keep="first", limit=0.5)
    assert len(matches_custom) > 0

    # Test equal_musts as string
    matches_eq_str = dd.get_duplicates(keep="first", equal_musts="primary_station_id")
    assert all(matches_eq_str["primary_station_id"])

    # Test equal_musts as list
    matches_eq_list = dd.get_duplicates(
        keep="first", equal_musts=["primary_station_id", "longitude"]
    )
    assert all(matches_eq_list["primary_station_id"])
    assert all(matches_eq_list["longitude"])


def test_get_duplicates_keep_integer(dummy_data, compared_matrix):
    dd = DupDetect(dummy_data, compared_matrix, "SortedNeighbourhood", {}, {})

    # Using integer as keep
    dd.get_duplicates(keep=0)
    assert hasattr(dd, "matches")


def test_flag_duplicates_variants(dummy_data, compared_matrix):
    dd = DupDetect(dummy_data, compared_matrix, "SortedNeighbourhood", {}, {})

    # keep="last"
    flagged_last = dd.flag_duplicates(keep="last")
    assert flagged_last["duplicate_status"].isin([0, 1, 3]).all()

    # ensure 'duplicates' and 'history' columns exist
    assert "duplicates" in flagged_last.columns
    assert "history" in flagged_last.columns

    # keep integer index
    flagged_int = dd.flag_duplicates(keep=0)
    assert flagged_int["duplicate_status"].isin([0, 1, 3]).all()


def test_remove_duplicates_variants(dummy_data, compared_matrix):
    dd = DupDetect(dummy_data, compared_matrix, "SortedNeighbourhood", {}, {})

    # keep="first"
    result_first = dd.remove_duplicates(keep="first")
    remaining_ids_first = set(result_first["report_id"])
    assert "A" in remaining_ids_first

    # keep="last"
    result_last = dd.remove_duplicates(keep="last")
    remaining_ids_last = set(result_last["report_id"])
    assert "D" in remaining_ids_last

    # keep integer index
    result_int = dd.remove_duplicates(keep=0)
    remaining_ids_int = set(result_int["report_id"])
    assert "A" in remaining_ids_int


def test_get_total_score(dummy_data, compared_matrix):
    dd = DupDetect(dummy_data, compared_matrix, "SortedNeighbourhood", {}, {})
    dd._total_score()
    # Score should be computed
    assert hasattr(dd, "score")
    assert dd.score.min() >= 0
    assert dd.score.max() <= 1
