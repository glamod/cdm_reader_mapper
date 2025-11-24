from __future__ import annotations

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
