from __future__ import annotations

import pandas as pd
import pytest

from cdm_reader_mapper.common.iterators import ParquetStreamReader
from cdm_reader_mapper.duplicates.duplicates import DupDetect

from cdm_reader_mapper import DataBundle

YR = ("core", "YR")
MO = ("core", "MO")
DY = ("core", "DY")
HR = ("core", "HR")
PT = ("c1", "PT")
ID = ("core", "ID")


@pytest.fixture
def sample_db_df():
    data = pd.DataFrame(
        {
            "A": [19, 26, 27, 41, 91],
            "B": [0, 1, 2, 3, 4],
        }
    )
    mask = pd.DataFrame(
        {
            "A": [True, True, True, False, True],
            "B": [True, True, True, False, False],
        }
    )
    return DataBundle(data=data, mask=mask)


@pytest.fixture
def sample_db_df_multi():
    data = pd.DataFrame(
        {
            ("A", "a"): [19, 26, 27, 41, 91],
            ("B", "b"): [0, 1, 2, 3, 4],
        }
    )
    mask = pd.DataFrame(
        {
            ("A", "a"): [True, True, True, False, True],
            ("B", "b"): [True, True, True, False, False],
        }
    )
    return DataBundle(data=data, mask=mask)


@pytest.fixture
def sample_db_psr():
    data1 = pd.DataFrame({"A": [19, 26], "B": [0, 1]}, index=[0, 1])
    data2 = pd.DataFrame({"A": [27, 41, 91], "B": [2, 3, 4]}, index=[2, 3, 4])
    data = ParquetStreamReader([data1, data2])

    mask1 = pd.DataFrame({"A": [True, True], "B": [True, True]}, index=[0, 1])
    mask2 = pd.DataFrame(
        {"A": [True, False, True], "B": [True, False, False]}, index=[2, 3, 4]
    )
    mask = ParquetStreamReader([mask1, mask2])

    return DataBundle(data=data, mask=mask)


@pytest.fixture
def sample_db_psr_multi():
    data1 = pd.DataFrame({("A", "a"): [19, 26], ("B", "b"): [0, 1]}, index=[0, 1])
    data2 = pd.DataFrame(
        {("A", "a"): [27, 41, 91], ("B", "b"): [2, 3, 4]}, index=[2, 3, 4]
    )
    data = ParquetStreamReader([data1, data2])

    mask1 = pd.DataFrame(
        {("A", "a"): [True, True], ("B", "b"): [True, True]}, index=[0, 1]
    )
    mask2 = pd.DataFrame(
        {("A", "a"): [True, False, True], ("B", "b"): [True, False, False]},
        index=[2, 3, 4],
    )
    mask = ParquetStreamReader([mask1, mask2])

    return DataBundle(data=data, mask=mask)


@pytest.fixture
def sample_data():
    return pd.DataFrame({"C": [20, 21, 22, 23, 24]})


@pytest.fixture
def sample_mask():
    return pd.DataFrame({"C": [True, False, True, False, False]})


def test_len_df(sample_db_df):
    assert len(sample_db_df) == 5


def test_len_psr(sample_db_psr):
    assert len(sample_db_psr) == 5


def test_print_df(sample_db_df, capsys):
    print(sample_db_df)

    captured = capsys.readouterr()

    assert captured.out.strip() != ""

    for col in sample_db_df.columns:
        assert col in captured.out


def test_print_psr(sample_db_psr, capsys):
    print(sample_db_psr)

    captured = capsys.readouterr()

    assert captured.out.strip() != ""


def test_copy_df(sample_db_df):
    db_cp = sample_db_df.copy()

    pd.testing.assert_frame_equal(sample_db_df.data, db_cp.data)
    pd.testing.assert_frame_equal(sample_db_df.mask, db_cp.mask)


def test_copy_psr(sample_db_psr):
    db_cp = sample_db_psr.copy()
    pd.testing.assert_frame_equal(sample_db_psr.data.read(), db_cp.data.read())
    pd.testing.assert_frame_equal(sample_db_psr.mask.read(), db_cp.mask.read())


def test_add_df(sample_db_df):
    sample_data = sample_db_df.data
    sample_mask = sample_db_df.mask

    db = DataBundle()
    db_add = db.add({"data": sample_data})

    pd.testing.assert_frame_equal(db_add.data, sample_data)

    db = DataBundle()
    db_add = db.add({"mask": sample_mask})

    pd.testing.assert_frame_equal(db_add.mask, sample_mask)

    db = DataBundle()
    db_add = db.add({"data": sample_data, "mask": sample_mask})

    pd.testing.assert_frame_equal(db_add.data, sample_data)
    pd.testing.assert_frame_equal(db_add.mask, sample_mask)


def test_add_psr_data(sample_db_psr):
    sample_data = sample_db_psr.data

    db = DataBundle()
    db_add = db.add({"data": sample_data})

    pd.testing.assert_frame_equal(db_add.data.read(), sample_data.read())


def test_add_psr_mask(sample_db_psr):
    sample_mask = sample_db_psr.mask

    db = DataBundle()
    db_add = db.add({"mask": sample_mask})

    pd.testing.assert_frame_equal(db_add.mask.read(), sample_mask.read())


def test_add_psr_both(sample_db_psr):
    sample_data = sample_db_psr.data
    sample_mask = sample_db_psr.mask

    db = DataBundle()
    db_add = db.add({"data": sample_data, "mask": sample_mask})

    pd.testing.assert_frame_equal(db_add.data.read(), sample_data.read())
    pd.testing.assert_frame_equal(db_add.mask.read(), sample_mask.read())


def test_stack_v_df(sample_db_df):
    sample_data = sample_db_df.data.copy()
    sample_mask = sample_db_df.mask.copy()

    db = DataBundle(data=sample_data, mask=sample_mask)

    sample_db_df.stack_v(db, inplace=True)

    expected_data = pd.concat([sample_data, db.data], ignore_index=True)
    expected_mask = pd.concat([sample_mask, db.mask], ignore_index=True)

    pd.testing.assert_frame_equal(sample_db_df.data, expected_data)
    pd.testing.assert_frame_equal(sample_db_df.mask, expected_mask)


def test_stack_v_psr(sample_db_psr):
    sample_data = sample_db_psr.data
    sample_mask = sample_db_psr.mask

    db = DataBundle(data=sample_data, mask=sample_mask)

    with pytest.raises(ValueError):
        sample_db_psr.stack_v(db)


def test_stack_h_df(sample_db_df, sample_data, sample_mask):
    db = DataBundle(data=sample_data, mask=sample_mask)
    expected_data = pd.concat([sample_db_df.data, db.data], axis=1)
    expected_mask = pd.concat([sample_db_df.mask, db.mask], axis=1)

    sample_db_df.stack_h(db, inplace=True)

    pd.testing.assert_frame_equal(sample_db_df.data, expected_data)
    pd.testing.assert_frame_equal(sample_db_df.mask, expected_mask)


def test_stack_h_psr(sample_db_psr):
    sample_data = sample_db_psr.data
    sample_mask = sample_db_psr.mask

    db = DataBundle(data=sample_data, mask=sample_mask)

    with pytest.raises(ValueError):
        sample_db_psr.stack_h(db)


@pytest.mark.parametrize(
    "func, args, idx_exp, idx_rej",
    [
        ("select_where_all_true", [], [0, 1, 2], [3, 4]),
        ("select_where_all_false", [], [3], [0, 1, 2, 4]),
        ("select_where_index_isin", [[0, 2, 4]], [0, 2, 4], [1, 3]),
        ("select_where_entry_isin", [{"A": [26, 41]}], [1, 3], [0, 2, 4]),
    ],
)
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_select_operators_df(
    sample_db_df,
    func,
    args,
    idx_exp,
    idx_rej,
    reset_index,
    inverse,
):
    result = getattr(sample_db_df, func)(
        *args, reset_index=reset_index, inverse=inverse
    )

    data = sample_db_df.data
    mask = sample_db_df.mask

    selected_data = result.data
    selected_mask = result.mask

    if inverse is False:
        idx = data.index.isin(idx_exp)
    else:
        idx = data.index.isin(idx_rej)

    expected_data = data[idx]
    expected_mask = mask[idx]

    if reset_index is True:
        expected_data = expected_data.reset_index(drop=True)
        expected_mask = expected_mask.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected_data, selected_data)
    pd.testing.assert_frame_equal(expected_mask, selected_mask)


@pytest.mark.parametrize(
    "func, args, idx_exp, idx_rej",
    [
        ("select_where_all_true", [], [0, 1, 2], [3, 4]),
        ("select_where_all_false", [], [3], [0, 1, 2, 4]),
        ("select_where_index_isin", [[0, 2, 4]], [0, 2, 4], [1, 3]),
        ("select_where_entry_isin", [{"A": [26, 41]}], [1, 3], [0, 2, 4]),
    ],
)
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_select_operators_psr(
    sample_db_psr,
    func,
    args,
    idx_exp,
    idx_rej,
    reset_index,
    inverse,
):
    result = getattr(sample_db_psr, func)(
        *args, reset_index=reset_index, inverse=inverse
    )

    data = sample_db_psr.data.read()
    mask = sample_db_psr.mask.read()

    selected_data = result.data.read()
    selected_mask = result.mask.read()

    if inverse is False:
        idx = data.index.isin(idx_exp)
    else:
        idx = data.index.isin(idx_rej)

    expected_data = data[idx]
    expected_mask = mask[idx]

    if reset_index is True:
        expected_data = expected_data.reset_index(drop=True)
        expected_mask = expected_mask.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected_data, selected_data)
    pd.testing.assert_frame_equal(expected_mask, selected_mask)


@pytest.mark.parametrize(
    "func, args, idx_exp, idx_rej",
    [
        ("split_by_boolean_true", [], [0, 1, 2], [3, 4]),
        ("split_by_boolean_false", [], [3], [0, 1, 2, 4]),
        ("split_by_index", [[0, 2, 4]], [0, 2, 4], [1, 3]),
        ("split_by_column_entries", [{"A": [26, 41]}], [1, 3], [0, 2, 4]),
    ],
)
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_split_operators_df(
    sample_db_df,
    func,
    args,
    idx_exp,
    idx_rej,
    reset_index,
    inverse,
):
    result = getattr(sample_db_df, func)(
        *args, reset_index=reset_index, inverse=inverse
    )

    data = sample_db_df.data
    mask = sample_db_df.mask

    selected_data = result[0].data
    selected_mask = result[0].mask
    rejected_data = result[1].data
    rejected_mask = result[1].mask

    if inverse is False:
        idx1 = data.index.isin(idx_exp)
        idx2 = data.index.isin(idx_rej)
    else:
        idx1 = data.index.isin(idx_rej)
        idx2 = data.index.isin(idx_exp)

    expected_data1 = data[idx1]
    expected_data2 = data[idx2]
    expected_mask1 = mask[idx1]
    expected_mask2 = mask[idx2]

    if reset_index is True:
        expected_data1 = expected_data1.reset_index(drop=True)
        expected_data2 = expected_data2.reset_index(drop=True)
        expected_mask1 = expected_mask1.reset_index(drop=True)
        expected_mask2 = expected_mask2.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected_data1, selected_data)
    pd.testing.assert_frame_equal(expected_data2, rejected_data)
    pd.testing.assert_frame_equal(expected_mask1, selected_mask)
    pd.testing.assert_frame_equal(expected_mask2, rejected_mask)


@pytest.mark.parametrize(
    "func, args, idx_exp, idx_rej",
    [
        ("split_by_boolean_true", [], [0, 1, 2], [3, 4]),
        ("split_by_boolean_false", [], [3], [0, 1, 2, 4]),
        ("split_by_index", [[0, 2, 4]], [0, 2, 4], [1, 3]),
        ("split_by_column_entries", [{"A": [26, 41]}], [1, 3], [0, 2, 4]),
    ],
)
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_split_operators_psr(
    sample_db_psr,
    func,
    args,
    idx_exp,
    idx_rej,
    reset_index,
    inverse,
):
    result = getattr(sample_db_psr, func)(
        *args, reset_index=reset_index, inverse=inverse
    )

    data = sample_db_psr.data.read()
    mask = sample_db_psr.mask.read()

    selected_data = result[0].data.read()
    selected_mask = result[0].mask.read()
    rejected_data = result[1].data.read()
    rejected_mask = result[1].mask.read()

    if inverse is False:
        idx1 = data.index.isin(idx_exp)
        idx2 = data.index.isin(idx_rej)
    else:
        idx1 = data.index.isin(idx_rej)
        idx2 = data.index.isin(idx_exp)

    expected_data1 = data[idx1]
    expected_data2 = data[idx2]
    expected_mask1 = mask[idx1]
    expected_mask2 = mask[idx2]

    if reset_index is True:
        expected_data1 = expected_data1.reset_index(drop=True)
        expected_data2 = expected_data2.reset_index(drop=True)
        expected_mask1 = expected_mask1.reset_index(drop=True)
        expected_mask2 = expected_mask2.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected_data1, selected_data)
    pd.testing.assert_frame_equal(expected_data2, rejected_data)
    pd.testing.assert_frame_equal(expected_mask1, selected_mask)
    pd.testing.assert_frame_equal(expected_mask2, rejected_mask)


def test_split_by_index_df_multi(sample_db_df_multi):
    result = sample_db_df_multi.split_by_column_entries({("A", "a"): [26, 41]})

    data = sample_db_df_multi.data
    mask = sample_db_df_multi.mask

    selected_data = result[0].data
    selected_mask = result[0].mask
    rejected_data = result[1].data
    rejected_mask = result[1].mask

    idx1 = data.index.isin([1, 3])
    idx2 = data.index.isin([0, 2, 4])

    expected_data1 = data[idx1]
    expected_data2 = data[idx2]
    expected_mask1 = mask[idx1]
    expected_mask2 = mask[idx2]

    pd.testing.assert_frame_equal(expected_data1, selected_data)
    pd.testing.assert_frame_equal(expected_data2, rejected_data)
    pd.testing.assert_frame_equal(expected_mask1, selected_mask)
    pd.testing.assert_frame_equal(expected_mask2, rejected_mask)


def test_split_by_index_psr_multi(sample_db_psr_multi):
    result = sample_db_psr_multi.split_by_column_entries({("A", "a"): [26, 41]})

    data = sample_db_psr_multi.data.read()
    mask = sample_db_psr_multi.mask.read()

    selected_data = result[0].data.read()
    selected_mask = result[0].mask.read()
    rejected_data = result[1].data.read()
    rejected_mask = result[1].mask.read()

    idx1 = data.index.isin([1, 3])
    idx2 = data.index.isin([0, 2, 4])

    expected_data1 = data[idx1]
    expected_data2 = data[idx2]
    expected_mask1 = mask[idx1]
    expected_mask2 = mask[idx2]

    pd.testing.assert_frame_equal(expected_data1, selected_data)
    pd.testing.assert_frame_equal(expected_data2, rejected_data)
    pd.testing.assert_frame_equal(expected_mask1, selected_mask)
    pd.testing.assert_frame_equal(expected_mask2, rejected_mask)


def test_unique_df(sample_db_df):
    result = sample_db_df.unique(columns=("A"))
    assert result == {"A": {19: 1, 26: 1, 27: 1, 41: 1, 91: 1}}


def test_unique_psr(sample_db_psr):
    result = sample_db_psr.unique(columns=("A"))
    assert result == {"A": {19: 1, 26: 1, 27: 1, 41: 1, 91: 1}}


def test_replace_columns_all_df(sample_db_df):
    df_corr = pd.DataFrame(
        {
            "A_new": [101, 201, 301, 401, 501],
            "B": range(5),
        }
    )
    result = sample_db_df.replace_columns(
        df_corr,
        rep_map={"A": "A_new"},
        pivot_l="B",
        pivot_r="B",
    )
    expected = pd.DataFrame(
        {
            "A": [101, 201, 301, 401, 501],
            "B": [0, 1, 2, 3, 4],
        }
    )

    pd.testing.assert_frame_equal(result.data, expected)


def test_replace_columns_subset_df(sample_db_df):
    df_corr = pd.DataFrame(
        {
            "A_new": [101, 201, 301, 401, 501],
            "B": range(5),
        }
    )
    result = sample_db_df.replace_columns(
        df_corr,
        subset=["A", "B"],
        rep_map={"A": "A_new"},
        pivot_l="B",
        pivot_r="B",
    )
    expected = pd.DataFrame(
        {
            "A": [101, 201, 301, 401, 501],
            "B": [0, 1, 2, 3, 4],
        }
    )

    pd.testing.assert_frame_equal(result.data, expected)


def test_replace_columns_all_psr(sample_db_psr):
    df_corr = pd.DataFrame(
        {
            "A_new": [101, 201, 301, 401, 501],
            "B": range(5),
        }
    )

    with pytest.raises(TypeError, match="Data must be a pd.DataFrame or pd.Series"):
        sample_db_psr.replace_columns(
            df_corr,
            rep_map={"A": "A_new"},
            pivot_l="B",
            pivot_r="B",
        )


def test_correct_datetime_df():
    data = pd.DataFrame({YR: [1899], MO: [1], DY: [1], HR: [0]})

    db = DataBundle(
        data=data,
        imodel="icoads_r300_d201",
    )

    result = db.correct_datetime()
    expected = pd.DataFrame({YR: [1898], MO: [12], DY: [31], HR: [0]})

    pd.testing.assert_frame_equal(result.data, expected)


def test_correct_datetime_psr():
    df1 = pd.DataFrame({YR: [1899], MO: [1], DY: [1], HR: [0]})

    db = DataBundle(
        data=ParquetStreamReader([df1]),
        imodel="icoads_r300_d201",
    )

    result = db.correct_datetime()

    expected = pd.DataFrame({YR: [1898], MO: [12], DY: [31], HR: [0]})
    pd.testing.assert_frame_equal(result.data.read(), expected)


def test_validate_datetime_df():
    data = pd.DataFrame({YR: [2023, 2023], MO: [1, 1], DY: [1, 2], HR: [12, None]})

    db = DataBundle(
        data=data,
        imodel="icoads",
    )

    result = db.validate_datetime()

    pd.testing.assert_series_equal(result, pd.Series([True, False]))


def test_validate_datetime_psr():
    df1 = pd.DataFrame({YR: [2023], MO: [1], DY: [1], HR: [12]}, index=[0])
    df2 = pd.DataFrame({YR: [2023], MO: [1], DY: [2], HR: [None]}, index=[1])

    db = DataBundle(
        data=ParquetStreamReader([df1, df2]),
        imodel="icoads",
    )

    result = db.validate_datetime()

    pd.testing.assert_series_equal(result.read(), pd.Series([True, False]))


def test_correct_pt_df():
    data = pd.DataFrame({PT: [None, "7", None]})

    db = DataBundle(
        data=data,
        imodel="icoads_r300_d993",
    )

    result = db.correct_pt()

    expected = pd.DataFrame({PT: ["5", "7", "5"]})
    pd.testing.assert_frame_equal(result.data, expected)


def test_correct_pt_psr():
    df1 = pd.DataFrame({PT: [None, "7", None]})

    db = DataBundle(
        data=ParquetStreamReader([df1]),
        imodel="icoads_r300_d993",
    )

    result = db.correct_pt()

    expected = pd.DataFrame({PT: ["5", "7", "5"]})
    pd.testing.assert_frame_equal(result.data.read(), expected)


def test_validate_id_df():
    data = pd.DataFrame({ID: ["12345", "ABCDE", None]})

    db = DataBundle(
        data=data,
        imodel="icoads_r300_d201",
    )

    result = db.validate_id()

    pd.testing.assert_series_equal(result, pd.Series([True, False, True]))


def test_validate_id_psr():
    df1 = pd.DataFrame({ID: ["12345"]}, index=[0])
    df2 = pd.DataFrame({ID: ["ABCDE"]}, index=[1])
    df3 = pd.DataFrame({ID: [None]}, index=[2])

    db = DataBundle(
        data=ParquetStreamReader([df1, df2, df3]),
        imodel="icoads_r300_d201",
    )

    result = db.validate_id()

    pd.testing.assert_series_equal(result.read(), pd.Series([True, False, True]))


def test_map_model_df():
    data = pd.DataFrame(
        {
            ("c1", "PT"): ["2", "4", "9", "21"],
            ("c98", "UID"): ["5012", "8960", "0037", "1000"],
            ("c1", "LZ"): ["1", None, None, "3"],
        }
    )

    db = DataBundle(
        data=data,
        imodel="icoads_r302",
    )

    result = db.map_model()

    expected = pd.DataFrame(
        {
            ("header", "report_id"): [
                "ICOADS-302-5012",
                "ICOADS-302-8960",
                "ICOADS-302-0037",
                "ICOADS-302-1000",
            ],
            ("header", "duplicate_status"): ["4", "4", "4", "4"],
            ("header", "platform_type"): ["2", "33", "32", "45"],
            ("header", "location_quality"): ["2", "0", "0", "0"],
            ("header", "source_id"): ["null", "null", "null", "null"],
        }
    )

    pd.testing.assert_frame_equal(result.data[expected.columns], expected)


def test_map_model_psr():
    data = pd.DataFrame(
        {
            ("c1", "PT"): ["2", "4", "9", "21"],
            ("c98", "UID"): ["5012", "8960", "0037", "1000"],
            ("c1", "LZ"): ["1", None, None, "3"],
        }
    )

    db = DataBundle(
        data=ParquetStreamReader([data]),
        imodel="icoads_r302",
    )

    result = db.map_model()

    expected = pd.DataFrame(
        {
            ("header", "report_id"): [
                "ICOADS-302-5012",
                "ICOADS-302-8960",
                "ICOADS-302-0037",
                "ICOADS-302-1000",
            ],
            ("header", "duplicate_status"): ["4", "4", "4", "4"],
            ("header", "platform_type"): ["2", "33", "32", "45"],
            ("header", "location_quality"): ["2", "0", "0", "0"],
            ("header", "source_id"): ["null", "null", "null", "null"],
        }
    )

    pd.testing.assert_frame_equal(result.data.read()[expected.columns], expected)


def test_duplicate_check_single_index():
    data = pd.DataFrame(
        {
            "report_id": ["A", "B", "C", "D", "E", "F"],
            "primary_station_id": ["S1", "S1", "S2", "S2", "S1", "S1"],
            "longitude": [0.1, 0.1, 0.2, 0.1, 0.1, 0.1],
            "latitude": [51.0, 51.2, 52.0, 51.0, 51.0, 51.0],
            "report_timestamp": pd.to_datetime(
                [
                    "2023-01-01 00:00",
                    "2023-01-01 00:00",
                    "2023-01-01 00:00",
                    "2023-01-01 00:00",
                    "2023-01-01 00:00",
                    "2023-01-01 00:00",
                ]
            ),
            "station_speed": [10.0, 10.0, 8.0, 10.0, 8.0, 10.0],
            "station_course": [90, 90, 180, 90, 60, 90],
            "report_quality": 2,
            ("header", "duplicates"): "",
            ("header", "duplicate_status"): 4,
            ("header", "history"): "",
        }
    )

    db = DataBundle(
        data=data,
    )

    db_dupdetect = db.duplicate_check()

    assert hasattr(db_dupdetect, "DupDetect")
    detector = db_dupdetect.DupDetect

    assert isinstance(detector, DupDetect)
    assert detector.data.shape[0] == data.shape[0]

    duplicates = db_dupdetect.get_duplicates()

    assert isinstance(duplicates, pd.DataFrame)

    pd.testing.assert_index_equal(duplicates.index, pd.MultiIndex.from_tuples([(5, 0)]))

    flagged = db_dupdetect.flag_duplicates()

    pd.testing.assert_series_equal(
        flagged.data["duplicates"],
        pd.Series(["{F}", "", "", "", "", "{A}"], name="duplicates"),
    )
    pd.testing.assert_series_equal(
        flagged.data["duplicate_status"],
        pd.Series([1, 0, 0, 0, 0, 3], name="duplicate_status"),
    )

    removed = db_dupdetect.remove_duplicates()

    pd.testing.assert_frame_equal(data.iloc[[0, 1, 2, 3, 4]], removed.data)


def test_duplicate_check_multi_index():
    data = pd.DataFrame(
        {
            ("header", "report_id"): ["A", "B", "C", "D", "E", "F"],
            ("header", "primary_station_id"): ["S1", "S1", "S2", "S2", "S1", "S1"],
            ("header", "longitude"): [0.1, 0.1, 0.2, 0.1, 0.1, 0.1],
            ("header", "latitude"): [51.0, 51.2, 52.0, 51.0, 51.0, 51.0],
            ("header", "report_timestamp"): pd.to_datetime(
                [
                    "2023-01-01 00:00",
                    "2023-01-01 00:00",
                    "2023-01-01 00:00",
                    "2023-01-01 00:00",
                    "2023-01-01 00:00",
                    "2023-01-01 00:00",
                ]
            ),
            ("header", "station_speed"): [10.0, 10.0, 8.0, 10.0, 8.0, 10.0],
            ("header", "station_course"): [90, 90, 180, 90, 60, 90],
            ("header", "report_quality"): 2,
            ("header", "duplicates"): "",
            ("header", "duplicate_status"): 4,
            ("header", "history"): "",
        }
    )

    db = DataBundle(
        data=data,
        mode="tables",
    )

    db_dupdetect = db.duplicate_check()

    assert hasattr(db_dupdetect, "DupDetect")
    detector = db_dupdetect.DupDetect

    assert isinstance(detector, DupDetect)
    assert detector.data.shape[0] == data.shape[0]

    duplicates = db_dupdetect.get_duplicates()

    assert isinstance(duplicates, pd.DataFrame)

    pd.testing.assert_index_equal(duplicates.index, pd.MultiIndex.from_tuples([(5, 0)]))

    flagged = db_dupdetect.flag_duplicates()

    pd.testing.assert_series_equal(
        flagged.data[("header", "duplicates")],
        pd.Series(["{F}", "", "", "", "", "{A}"], name=("header", "duplicates")),
    )
    pd.testing.assert_series_equal(
        flagged.data[("header", "duplicate_status")],
        pd.Series([1, 0, 0, 0, 0, 3], name=("header", "duplicate_status")),
    )

    removed = db_dupdetect.remove_duplicates()

    pd.testing.assert_frame_equal(data.iloc[[0, 1, 2, 3, 4]], removed.data)
