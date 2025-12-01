from __future__ import annotations

import numpy as np
import pandas as pd
import logging
import pytest

from io import StringIO

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
from cdm_reader_mapper.metmetpy.correct import (
    _correct_dt,
    _correct_pt,
    correct_datetime,
    correct_pt,
)
from cdm_reader_mapper.metmetpy.validate import (
    _get_id_col,
    _get_patterns,
    validate_id,
    validate_datetime,
)

YR = properties.metadata_datamodels["year"]["icoads"]
MO = properties.metadata_datamodels["month"]["icoads"]
DY = properties.metadata_datamodels["day"]["icoads"]
HR = properties.metadata_datamodels["hour"]["icoads"]
ID = properties.metadata_datamodels["id"]["icoads"]
SID = properties.metadata_datamodels["source"]["icoads"]
PT = properties.metadata_datamodels["platform"]["icoads"]

datetime_cols = [YR, MO, DY, HR]


@pytest.fixture
def sample_df():
    return pd.DataFrame({"A": [1, 2, 3]})


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
    data = pd.DataFrame(
        {
            ID: ["12345", "54321", "00001", "ABCDE"],
            SID: ["147", "147", "999", "147"],
            PT: [pd.NA, "5", "5", "5"],
        }
    )

    expected = pd.DataFrame(
        {
            ID: ["12345", "54321", "00001", "ABCDE"],
            SID: ["147", "147", "999", "147"],
            PT: ["7", "6", "5", "5"],
        }
    )

    result = deck_700_icoads(data.copy())

    pd.testing.assert_frame_equal(result, expected)


def test_deck_892_icoads():
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
    data = pd.DataFrame(
        {
            ID: ["123456", "12345", "7123456", "2345678"],
            SID: ["103", "103", "103", "103"],
            PT: ["5", "5", "5", "5"],
        }
    )

    expected = pd.DataFrame(
        {
            ID: ["123456", "12345", "7123456", "2345678"],
            SID: ["103", "103", "103", "103"],
            PT: ["6", "5", "5", "5"],
        }
    )

    result = deck_792_icoads(data.copy())
    pd.testing.assert_frame_equal(result, expected)


def test_deck_792_icoads_no_change():
    data = pd.DataFrame(
        {
            ID: ["12345", "7123456", "56789"],
            SID: ["104", "103", "103"],
            PT: ["5", "6", "5"],
        }
    )

    expected = data.copy()
    result = deck_792_icoads(data.copy())
    pd.testing.assert_frame_equal(result, expected)


def test_deck_992_icoads_basic():
    data = pd.DataFrame(
        {
            ID: ["6202222", "123456", "7123456", "2345678"],
            SID: ["114", "114", "114", "114"],
            PT: ["5", "5", "5", "5"],
        }
    )

    expected = pd.DataFrame(
        {
            ID: ["6202222", "123456", "7123456", "2345678"],
            SID: ["114", "114", "114", "114"],
            PT: ["4", "6", "5", "5"],
        }
    )

    result = deck_992_icoads(data.copy())
    pd.testing.assert_frame_equal(result, expected)


def test_correct_dt():
    data = pd.DataFrame(
        {
            YR: [1899, 1900, 1899],
            MO: [1, 2, 3],
            DY: [1, 15, 1],
            HR: [0, 12, 0],
        }
    )

    expected = pd.DataFrame(
        {
            YR: [1898, 1900, 1899],
            MO: [12, 2, 2],
            DY: [31, 15, 28],
            HR: [0, 12, 0],
        }
    )

    correction_method = {"201": {"function": "dck_201_icoads"}}

    result = _correct_dt(
        data.copy(),
        data_model="icoads",
        dck="201",
        correction_method=correction_method,
    )

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_correct_pt_fillna():
    pt_col = "platform"
    data = pd.DataFrame({pt_col: [None, "5", None, "3"]})

    fix_methods = {
        "201": {
            "method": "fillna",
            "fill_value": "7",
        }
    }

    expected = pd.DataFrame({pt_col: ["7", "5", "7", "3"]})

    result = _correct_pt(
        data.copy(),
        imodel="icoads",
        dck="201",
        pt_col=pt_col,
        fix_methods=fix_methods,
    )

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_correct_pt_function():
    data = pd.DataFrame(
        {
            ID: ["12345", "99999"],
            SID: ["147", "999"],
            PT: ["5", "5"],
        }
    )

    fix_methods = {
        "700": {
            "method": "function",
            "function": "deck_700_icoads",
        }
    }

    expected = pd.DataFrame(
        {
            ID: ["12345", "99999"],
            SID: ["147", "999"],
            PT: ["6", "5"],
        }
    )

    result = _correct_pt(
        data.copy(),
        imodel="icoads",
        dck="700",
        pt_col=PT,
        fix_methods=fix_methods,
    )

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_correct_pt_no_fix_for_deck():
    pt_col = "PT"
    data = pd.DataFrame({pt_col: ["1", None, "3"]})

    fix_methods = {"999": {"method": "fillna", "fill_value": "7"}}

    result = _correct_pt(
        data.copy(),
        imodel="icoads",
        dck="888",
        pt_col=pt_col,
        fix_methods=fix_methods,
    )

    pd.testing.assert_frame_equal(result, data)


def test_correct_pt_missing_platform_column():
    data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    fix_methods = {"201": {"method": "fillna", "fill_value": "7"}}

    result = _correct_pt(
        data.copy(),
        imodel="icoads",
        dck="201",
        pt_col="PT",
        fix_methods=fix_methods,
    )

    pd.testing.assert_frame_equal(result, data)


def test_correct_pt_raises_unknown_method():
    data = pd.DataFrame({"PT": ["1", "2"]})

    fix_methods = {"201": {"method": "not_a_method"}}

    with pytest.raises(ValueError, match="not implemented"):
        _correct_pt(
            data.copy(),
            imodel="icoads",
            dck="201",
            pt_col="PT",
            fix_methods=fix_methods,
        )


def test_correct_pt_fillna_missing_fillvalue():
    data = pd.DataFrame({"PT": ["1", None]})

    fix_methods = {"201": {"method": "fillna"}}

    with pytest.raises(ValueError, match='requires "fill_value"'):
        _correct_pt(
            data.copy(),
            imodel="icoads",
            dck="201",
            pt_col="PT",
            fix_methods=fix_methods,
        )


def test_correct_pt_missing_function_name():
    data = pd.DataFrame({"PT": ["1", "2"]})

    fix_methods = {"700": {"method": "function"}}

    with pytest.raises(ValueError, match='requires "function" name'):
        _correct_pt(
            data.copy(),
            imodel="icoads",
            dck="700",
            pt_col="PT",
            fix_methods=fix_methods,
        )


def test_correct_pt_missing_function_object():
    data = pd.DataFrame({"PT": ["1", "2"]})

    fix_methods = {"700": {"method": "function", "function": "NO_SUCH_FUNC"}}

    with pytest.raises(ValueError, match="not found"):
        _correct_pt(
            data.copy(),
            imodel="icoads",
            dck="700",
            pt_col="PT",
            fix_methods=fix_methods,
        )


@pytest.mark.parametrize(
    "data_input,imodel,expected",
    [
        (
            pd.DataFrame({YR: [1899], MO: [1], DY: [1], HR: [0]}),
            "icoads_r300_d201",
            pd.DataFrame({YR: [1898], MO: [12], DY: [31], HR: [0]}),
        ),
        (
            pd.DataFrame({YR: [1900], MO: [1], DY: [1], HR: [12]}),
            "icoads_r300_d201",
            pd.DataFrame({YR: [1900], MO: [1], DY: [1], HR: [12]}),
        ),
    ],
)
def test_correct_datetime(data_input, imodel, expected):
    result = correct_datetime(data_input.copy(), imodel, log_level="CRITICAL")
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_correct_datetime_textfilereader():
    csv_text = "1899,1,1,0\n1900,1,1,12"

    expected = pd.DataFrame({YR: [1898, 1900], MO: [12, 1], DY: [31, 1], HR: [0, 12]})

    parser = pd.read_csv(
        StringIO(csv_text), chunksize=2, header=None, names=datetime_cols, dtype=int
    )

    result = correct_datetime(parser, "icoads_r300_d201").read()

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )


@pytest.mark.parametrize(
    "data_input,imodel,expected",
    [
        (
            pd.DataFrame({PT: [None, "7", None]}),
            "icoads_r300_d993",
            pd.DataFrame({PT: ["5", "7", "5"]}),
        ),
        (
            pd.DataFrame(
                {
                    ID: ["12345", "99999"],
                    SID: ["147", "999"],
                    PT: ["5", "5"],
                }
            ),
            "icoads_r300_d700",
            pd.DataFrame(
                {
                    ID: ["12345", "99999"],
                    SID: ["147", "999"],
                    PT: ["6", "5"],
                }
            ),
        ),
    ],
)
def test_correct_pt_dataframe(data_input, imodel, expected):
    """Test correct_pt with DataFrame input."""
    result = correct_pt(data_input.copy(), imodel, log_level="CRITICAL")
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize(
    "csv_text,names,imodel,expected",
    [
        (
            "\n7\n\n",
            [PT],
            "icoads_r300_d993",
            pd.DataFrame({PT: ["5", "7", "5"]}),
        ),
        (
            "5,12345,147\n5,99999,999\n7,123,999",
            [PT, ID, SID],
            "icoads_r300_d700",
            pd.DataFrame(
                {
                    PT: ["6", "5", "7"],
                    ID: ["12345", "99999", "123"],
                    SID: ["147", "999", "999"],
                }
            ),
        ),
    ],
)
def test_correct_pt_textfilereader(csv_text, names, imodel, expected):
    """Test correct_pt with TextFileReader input."""
    parser = pd.read_csv(
        StringIO(csv_text),
        chunksize=2,
        header=None,
        names=names,
        dtype=object,
        skip_blank_lines=False,
    )
    result = (
        correct_pt(parser, imodel, log_level="CRITICAL").read().reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_get_id_col_not_defined():
    logger = logging.getLogger("test_logger")
    df = pd.DataFrame({"X": [1, 2, 3]})
    result = _get_id_col(df, "unknown_model", logger)
    assert result is None


def test_get_id_col_missing_in_data():
    logger = logging.getLogger("test_logger")
    df = pd.DataFrame({"X": [1, 2, 3]})
    result = _get_id_col(df, "icoads", logger)
    assert result is None


def test_get_id_col_single_column_present():
    logger = logging.getLogger("test_logger")
    df = pd.DataFrame({("core", "ID"): [1, 2, 3], ("other", "ID"): [4, 5, 6]})
    result = _get_id_col(df, "icoads", logger)
    assert result == ("core", "ID")


def test_get_patterns_empty_dict():
    logger = logging.getLogger("test_logger")
    dck_id_model = {"valid_patterns": {}}
    blank = False
    dck = "123"
    data_model_files = ["dummy.json"]

    result = _get_patterns(dck_id_model, blank, dck, data_model_files, logger)
    assert result == [".*?"]


def test_get_patterns_with_patterns():
    logger = logging.getLogger("test_logger")
    dck_id_model = {"valid_patterns": {"p1": "A.*", "p2": "B.*"}}
    blank = False
    dck = "123"
    data_model_files = ["dummy.json"]

    result = _get_patterns(dck_id_model, blank, dck, data_model_files, logger)
    assert set(result) == {"A.*", "B.*"}


def test_get_patterns_with_blank_true():
    logger = logging.getLogger("test_logger")
    dck_id_model = {"valid_patterns": {"p1": "A.*"}}
    blank = True
    dck = "123"
    data_model_files = ["dummy.json"]

    result = _get_patterns(dck_id_model, blank, dck, data_model_files, logger)
    assert "A.*" in result
    assert "^$" in result
    assert len(result) == 2


def test_get_patterns_empty_and_blank_true():
    logger = logging.getLogger("test_logger")
    dck_id_model = {"valid_patterns": {}}
    blank = True
    dck = "123"
    data_model_files = ["dummy.json"]

    result = _get_patterns(dck_id_model, blank, dck, data_model_files, logger)
    assert ".*?" in result
    assert "^$" in result
    assert len(result) == 2


@pytest.mark.parametrize(
    "data_input, imodel, blank, expected",
    [
        (
            pd.DataFrame({ID: ["12345", "ABCDE"]}),
            "icoads_r300_d201",
            False,
            pd.Series([True, False], name=ID),
        ),
        (
            pd.DataFrame({ID: ["12345", ""]}),
            "icoads_r300_d201",
            True,
            pd.Series([True, True], name=ID),
        ),
        (
            pd.DataFrame({ID: ["12345", ""]}),
            "icoads_r300_d201",
            False,
            pd.Series([True, True], name=ID),
        ),
    ],
)
def test_validate_id_dataframe(data_input, imodel, blank, expected):
    result = validate_id(data_input.copy(), imodel, blank=blank, log_level="CRITICAL")
    pd.testing.assert_series_equal(result, expected, check_dtype=False)


def test_validate_id_textfilereader():
    csv_text = "12345\nABCDE\n\n"
    parser = pd.read_csv(
        StringIO(csv_text),
        chunksize=2,
        header=None,
        names=[ID],
        dtype=object,
        skip_blank_lines=False,
    )
    result = validate_id(parser, "icoads_r300_d201", blank=False, log_level="CRITICAL")
    expected = pd.Series([True, False, True], name=ID)
    pd.testing.assert_series_equal(
        result.reset_index(drop=True), expected, check_dtype=False
    )


@pytest.mark.parametrize(
    "data_input, expected",
    [
        (
            pd.DataFrame({YR: [2023, 2023], MO: [1, 1], DY: [1, 2], HR: [12, 13]}),
            pd.Series([True, True]),
        ),
        (
            pd.DataFrame({YR: [2023, 2023], MO: [1, 1], DY: [1, 2], HR: [12, None]}),
            pd.Series([True, False]),
        ),
        (
            pd.DataFrame(
                {YR: [2023, None], MO: [1, None], DY: [1, None], HR: [12, None]}
            ),
            pd.Series([True, False]),
        ),
    ],
)
def test_validate_datetime_dataframe(data_input, expected):
    result = validate_datetime(data_input.copy(), "icoads", log_level="CRITICAL")
    pd.testing.assert_series_equal(
        result.reset_index(drop=True), expected, check_dtype=False
    )


@pytest.mark.parametrize(
    "csv_text, expected",
    [
        ("2023,1,1,12\n2023,1,2,13\n\n", pd.Series([True, True, False])),
        ("2023,1,1,12\n2023,1,2,\n\n", pd.Series([True, False, False])),
    ],
)
def test_validate_datetime_textfilereader(csv_text, expected):
    parser = pd.read_csv(
        StringIO(csv_text),
        chunksize=2,
        header=None,
        names=datetime_cols,
        dtype=object,
        skip_blank_lines=False,
    )
    result = validate_datetime(parser, "icoads", log_level="CRITICAL")
    pd.testing.assert_series_equal(
        result.reset_index(drop=True), expected, check_dtype=False
    )
