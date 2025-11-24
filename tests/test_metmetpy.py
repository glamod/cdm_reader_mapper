from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdm_reader_mapper.metmetpy import properties
from cdm_reader_mapper.metmetpy.datetime.correction_functions import dck_201_icoads
from cdm_reader_mapper.metmetpy.datetime.model_datetimes import (
    datetime_decimalhour_to_hm,
    to_datetime,
    from_datetime,
    icoads,
)
from cdm_reader_mapper.metmetpy.platform_type.correction_functions import (
    is_num,
    overwrite_data,
    fill_value,
    deck_717_gdac,
    deck_700_icoads,
    deck_892_icoads,
    deck_792_icoads,
    deck_992_icoads,
)

YR = properties.metadata_datamodels["year"]["icoads"]
MO = properties.metadata_datamodels["month"]["icoads"]
DY = properties.metadata_datamodels["day"]["icoads"]
HR = properties.metadata_datamodels["hour"]["icoads"]

datetime_cols = [YR, MO, DY, HR]


@pytest.mark.parametrize(
    "decimal_hour,expected",
    [
        (0.0, (0, 0)),
        (1.5, (1, 30)),
        (13.75, (13, 45)),
        (23.99, (23, 59)),
        (12.0, (12, 0)),
    ],
)
def test_datetime_decimalhour_to_hm(decimal_hour, expected):
    result = datetime_decimalhour_to_hm(decimal_hour)
    assert result == expected


def test_icoads_to_datetime_basis():
    df = pd.DataFrame(
        {
            YR: [2000, 2001],
            MO: [1, 2],
            DY: [10, 15],
            HR: [12.5, 6.25],
        }
    )

    dt = icoads(df, "to_datetime")

    assert isinstance(dt, pd.Series)
    assert dt.dtype == "datetime64[ns]"

    assert dt.iloc[0].year == 2000
    assert dt.iloc[0].month == 1
    assert dt.iloc[0].day == 10
    assert dt.iloc[0].hour == 12
    assert dt.iloc[0].minute == 30

    assert dt.iloc[1].hour == 6
    assert dt.iloc[1].minute == 15


def test_icoads_to_datetime_missing_values():
    df = pd.DataFrame(
        {
            YR: [2000, None, 2002],
            MO: [1, 2, 3],
            DY: [10, None, 20],
            HR: [12.5, None, 18.75],
        }
    )

    dt = icoads(df, "to_datetime")

    assert dt.isna().tolist() == [False, True, False]


def test_icoads_from_datetime():
    ds = pd.Series(
        [
            pd.Timestamp("2000-01-10 12:30"),
            pd.Timestamp("2001-02-15 06:15"),
            pd.Timestamp("2002-03-20 18:45"),
        ]
    )

    df = icoads(ds, "from_datetime")

    assert list(df.columns) == [YR, MO, DY, HR]

    assert df.loc[0, YR] == 2000
    assert df.loc[0, MO] == 1
    assert df.loc[0, DY] == 10
    assert abs(df.loc[0, HR] - 12.5) < 1e-6


def test_icoads_roundtrip():
    df_in = pd.DataFrame(
        {
            YR: [2000, 2001],
            MO: [1, 2],
            DY: [10, 15],
            HR: [12.5, 6.25],
        }
    )

    dt = icoads(df_in, "to_datetime")
    df_out = icoads(dt, "from_datetime")

    pd.testing.assert_frame_equal(df_in, df_out, check_dtype=False)


def test_icoads_invalid_conversion():
    df = pd.DataFrame(
        {
            YR: [2000],
            MO: [1],
            DY: [10],
            HR: [12.5],
        }
    )

    with pytest.raises(ValueError):
        icoads(df, "bad_conversion.")


def test_icoads_from_datetime_wrong_input_type():
    df = pd.DataFrame(
        {
            YR: [2000],
            MO: [1],
            DY: [10],
            HR: [12.5],
        }
    )

    with pytest.raises(ValueError):
        icoads(df, "from_datetime.")


def test_icoads_to_datetime_wrong_input_type():
    s = pd.Series([pd.Timestamp("200-01-01")])

    with pytest.raises(TypeError):
        icoads(s, "to_datetime")


def test_icoads_to_datetime_missing_columns():
    df = pd.DataFrame({YR: [2000]})

    result = icoads(df, "to_datetime")

    assert isinstance(result, pd.Series)
    assert result.dtype == "datetime64[ns]"
    assert result.isna().all()


def test_icoads_from_datetime_empty_series():
    s = pd.Series([], dtype="datetime64[ns]")

    result = icoads(s, "from_datetime")

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_to_datetime_basis():
    df = pd.DataFrame(
        {
            YR: [2000, 2001],
            MO: [1, 2],
            DY: [10, 15],
            HR: [12.5, 6.25],
        }
    )

    dt = to_datetime(df)

    assert isinstance(dt, pd.Series)
    assert dt.dtype == "datetime64[ns]"

    assert dt.iloc[0].year == 2000
    assert dt.iloc[0].month == 1
    assert dt.iloc[0].day == 10
    assert dt.iloc[0].hour == 12
    assert dt.iloc[0].minute == 30

    assert dt.iloc[1].hour == 6
    assert dt.iloc[1].minute == 15


def test_from_datetime_basis():
    ds = pd.Series(
        [
            pd.Timestamp("2000-01-10 12:30"),
            pd.Timestamp("2001-02-15 06:15"),
            pd.Timestamp("2002-03-20 18:45"),
        ]
    )

    df = from_datetime(ds)

    assert list(df.columns) == [YR, MO, DY, HR]

    assert df.loc[0, YR] == 2000
    assert df.loc[0, MO] == 1
    assert df.loc[0, DY] == 10
    assert abs(df.loc[0, HR] - 12.5) < 1e-6


def test_to_datetime_raises_error():
    with pytest.raises(ValueError):
        to_datetime(pd.DataFrame(), "invalid_model")


def test_from_datetime_raises_error():
    with pytest.raises(ValueError):
        to_datetime(pd.Series(), "invalid_model")


def test_dck_201_icoads():
    data = pd.DataFrame(
        {YR: [1899, 1900, 1899], MO: [1, 2, 3], DY: [1, 15, 1], HR: [0, 12, 0]}
    )

    expected = pd.DataFrame(
        {
            YR: [1898, 1900, 1899],
            MO: [12, 2, 2],
            DY: [31, 15, 28],
            HR: [0, 12, 0],
        }
    )

    result = dck_201_icoads(data.copy())

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize(
    "x, exp",
    [
        ("123", True),
        ("0", True),
        ("12.3", False),
        ("abc", False),
        (None, False),
        (1, False),
        (12.3, False),
    ],
)
def test_is_num(x, exp):
    assert is_num(x) is exp


def test_overwrite_data():
    df = pd.DataFrame({"PT": [1, 2, 3, 4], "VAL": [10, 20, 30, 40]})

    loc = df["PT"] > 2
    result = overwrite_data(df.copy(), loc, "VAL", 999)

    expected = pd.DataFrame({"PT": [1, 2, 3, 4], "VAL": [10, 20, 999, 999]})

    pd.testing.assert_frame_equal(result, expected)

    result2 = overwrite_data(df.copy(), loc, "NON_EXIST", 0)
    pd.testing.assert_frame_equal(result2, df)

    result3 = overwrite_data(df.copy(), [False, False, False, False], "VAL", 0)
    pd.testing.assert_frame_equal(result3, df)


def test_fill_value():
    s = pd.Series([1, 2, 3, None, 2])

    result = fill_value(s, fill_value=99, fillna=True)
    expected = pd.Series([1, 2, 3, 99, 2])
    pd.testing.assert_series_equal(result, expected, check_dtype=False)

    result2 = fill_value(s, fill_value=0, self_condition_value=2)
    expected2 = pd.Series([1, 0, 3, None, 0])
    pd.testing.assert_series_equal(result2, expected2)

    df = pd.DataFrame({"mask": [True, False, True, False, True]})
    result3 = fill_value(
        s, fill_value=-1, out_condition=df, out_condition_values={"mask": True}
    )
    expected3 = pd.Series([-1, 2, -1, None, -1])
    pd.testing.assert_series_equal(result3, expected3)


def test_deck_717_gdac():
    pt_col = properties.metadata_datamodels["platform"]["gdac"]

    df = pd.DataFrame({pt_col: [np.nan, 0, 1, np.nan], "N": [1, np.nan, 2, np.nan]})

    expected = pd.DataFrame({pt_col: ["7", "9", 1, "7"], "N": [1, np.nan, 2, np.nan]})

    result = deck_717_gdac(df.copy())

    pd.testing.assert_frame_equal(result, expected)


def test_deck_700_icoads():
    id_col = properties.metadata_datamodels["id"]["icoads"]
    sid_col = properties.metadata_datamodels["source"]["icoads"]
    pt_col = properties.metadata_datamodels["platform"]["icoads"]

    data = pd.DataFrame(
        {
            id_col: ["12345", "54321", "00001", "ABCDE"],
            sid_col: ["147", "147", "999", "147"],
            pt_col: [pd.NA, "5", "5", "5"],
        }
    )

    expected = pd.DataFrame(
        {
            id_col: ["12345", "54321", "00001", "ABCDE"],
            sid_col: ["147", "147", "999", "147"],
            pt_col: ["7", "6", "5", "5"],
        }
    )

    result = deck_700_icoads(data.copy())

    pd.testing.assert_frame_equal(result, expected)


def test_deck_892_icoads():
    ID = properties.metadata_datamodels["id"]["icoads"]
    SID = properties.metadata_datamodels["source"]["icoads"]
    PT = properties.metadata_datamodels["platform"]["icoads"]

    columns = [ID, SID, PT]

    data = pd.DataFrame(
        [
            ["12345", "29", "5"],
            ["12345", "29", "6"],
            ["12345", "30", "5"],
            ["1234", "29", "5"],
        ],
        columns=columns,
    )

    expected = pd.DataFrame(
        [
            ["12345", "29", "6"],
            ["12345", "29", "6"],
            ["12345", "30", "5"],
            ["1234", "29", "5"],
        ],
        columns=columns,
    )

    result = deck_892_icoads(data.copy())

    pd.testing.assert_frame_equal(result, expected)


def test_deck_792_icoads_basic():
    id_col = properties.metadata_datamodels["id"]["icoads"]
    sid_col = properties.metadata_datamodels["source"]["icoads"]
    pt_col = properties.metadata_datamodels["platform"]["icoads"]

    data = pd.DataFrame(
        {
            id_col: ["123456", "12345", "7123456", "2345678"],
            sid_col: ["103", "103", "103", "103"],
            pt_col: ["5", "5", "5", "5"],
        }
    )

    expected = pd.DataFrame(
        {
            id_col: ["123456", "12345", "7123456", "2345678"],
            sid_col: ["103", "103", "103", "103"],
            pt_col: ["6", "5", "5", "5"],
        }
    )

    result = deck_792_icoads(data.copy())
    pd.testing.assert_frame_equal(result, expected)


def test_deck_792_icoads_no_change():
    id_col = properties.metadata_datamodels["id"]["icoads"]
    sid_col = properties.metadata_datamodels["source"]["icoads"]
    pt_col = properties.metadata_datamodels["platform"]["icoads"]

    data = pd.DataFrame(
        {
            id_col: ["12345", "7123456", "56789"],
            sid_col: ["104", "103", "103"],
            pt_col: ["5", "6", "5"],
        }
    )

    expected = data.copy()
    result = deck_792_icoads(data.copy())
    pd.testing.assert_frame_equal(result, expected)


def test_deck_992_icoads_basic():
    id_col = properties.metadata_datamodels["id"]["icoads"]
    sid_col = properties.metadata_datamodels["source"]["icoads"]
    pt_col = properties.metadata_datamodels["platform"]["icoads"]

    data = pd.DataFrame(
        {
            id_col: ["6202222", "123456", "7123456", "2345678"],
            sid_col: ["114", "114", "114", "114"],
            pt_col: ["5", "5", "5", "5"],
        }
    )

    expected = pd.DataFrame(
        {
            id_col: ["6202222", "123456", "7123456", "2345678"],
            sid_col: ["114", "114", "114", "114"],
            pt_col: ["4", "6", "5", "5"],
        }
    )

    result = deck_992_icoads(data.copy())
    pd.testing.assert_frame_equal(result, expected)
