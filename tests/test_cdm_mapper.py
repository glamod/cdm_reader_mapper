from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from io import StringIO

from cdm_reader_mapper.cdm_mapper.mapper import (
    _check_input_data_type,
    _is_empty,
    _drop_duplicated_rows,
    _get_nested_value,
    _decimal_places,
    _transform,
    _code_table,
    _default,
    _fill_value,
    _extract_input_data,
    _column_mapping,
    _convert_dtype,
    _table_mapping,
    _map_and_convert,
    # _prepare_cdm_tables,  # no tests yet
    # _process_chunk,  # no tests yet
    # _finalize_output,  # no tests yet
    # _normalize_input_data,  # no tests yet
    map_model,
)

from cdm_reader_mapper.common import logging_hdlr
from cdm_reader_mapper.common.json_dict import open_json_file

from cdm_reader_mapper.cdm_mapper.properties import cdm_tables
from cdm_reader_mapper.cdm_mapper.utils.mapping_functions import mapping_functions
from cdm_reader_mapper.cdm_mapper.tables.tables import get_imodel_maps, get_cdm_atts

from cdm_reader_mapper.data import test_data


@pytest.fixture
def sample_df():
    return pd.DataFrame({"A": [1, 2]})


@pytest.fixture
def sample_df_empty():
    return pd.DataFrame()


@pytest.fixture
def sample_tfr():
    csv_data = "A\n1\n2"
    return pd.read_csv(StringIO(csv_data), chunksize=1)


@pytest.fixture
def sample_tfr_empty():
    csv_data = "A\n"
    return pd.read_csv(StringIO(csv_data), chunksize=1)


@pytest.fixture
def sample_string():
    return "A"


@pytest.fixture
def imodel_maps():
    return get_imodel_maps("icoads", "r300", "d720", cdm_tables=["header"])


@pytest.fixture
def imodel_functions():
    return mapping_functions("icoads_r300_d720")


@pytest.fixture
def data_header():
    return pd.DataFrame(
        data={
            ("c1", "PT"): ["2", "4", "9", "21"],
            ("c98", "UID"): ["5012", "8960", "0037", "1000"],
            ("c1", "LZ"): ["1", None, None, "3"],
        }
    )


@pytest.fixture
def data_header_expected():
    return pd.DataFrame(
        data={
            ("header", "report_id"): [
                "ICOADS-30-5012",
                "ICOADS-30-8960",
                "ICOADS-30-0037",
                "ICOADS-30-1000",
            ],
            ("header", "duplicate_status"): ["4", "4", "4", "4"],
            ("header", "platform_type"): ["2", "33", "32", "45"],
            ("header", "location_quality"): ["2", "0", "0", "0"],
            ("header", "source_id"): ["null", "null", "null", "null"],
        }
    )


def _map_model_test_data(data_model, encoding="utf-8", select=None, **kwargs):
    source = test_data[f"test_{data_model}"]["mdf_data"]
    info = open_json_file(test_data[f"test_{data_model}"]["mdf_info"])
    df = pd.read_csv(source, dtype=info["dtypes"], encoding=encoding)
    if ":" in df.columns[0]:
        df.columns = pd.MultiIndex.from_tuples(col.split(":") for col in df.columns)
    result = map_model(df, data_model, **kwargs)
    if not select:
        select = cdm_tables
    for cdm_table in select:
        expected = pd.read_csv(
            test_data[f"test_{data_model}"][f"cdm_{cdm_table}"],
            delimiter="|",
            dtype="object",
            na_values=None,
            keep_default_na=False,
        )
        result_table = result[cdm_table].copy()
        result_table = result_table.dropna()

        if "record_timestamp" in expected.columns:
            expected = expected.drop("record_timestamp", axis=1)
            result_table = result_table.drop("record_timestamp", axis=1)
        if "history" in expected.columns:
            expected = expected.drop("history", axis=1)
            result_table = result_table.drop("history", axis=1)

        pd.testing.assert_frame_equal(result_table, expected)


def test_check_input_data_type_df_non_empty(sample_df):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    result = _check_input_data_type(sample_df, logger)

    assert result == [sample_df]


def test_check_input_data_type_df_empty(sample_df_empty):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    result = _check_input_data_type(sample_df_empty, logger)

    assert result is None


def test_check_input_data_type_textfilereader_non_empty(sample_tfr):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    result = _check_input_data_type(sample_tfr, logger)

    assert result is sample_tfr


def test_check_input_data_type_textfilereader_empty(sample_tfr_empty):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    result = _check_input_data_type(sample_tfr_empty, logger)

    assert result is None


def test_check_input_data_type_invalid_type(sample_string):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    result = _check_input_data_type(sample_string, logger)

    assert result is None


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, True),
        (pd.DataFrame(), True),
        (pd.DataFrame({"a": [1]}), False),
        (123, False),
        ("string", False),
        ([], True),
        ({}, True),
        ("", True),
    ],
)
def test_is_empty(value, expected):
    assert _is_empty(value) is expected


def test_drop_duplicated_rows():
    data = pd.DataFrame(
        data={"col1": [1, 2, 3, 4, 3], "col2": [[5, 9], [6, 9], [7, 9], [8, 9], [7, 9]]}
    )
    result = _drop_duplicated_rows(data)
    expected = pd.DataFrame(
        data={"col1": [1, 2, 3, 4], "col2": [[5, 9], [6, 9], [7, 9], [8, 9]]}
    )
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", 1000),
        ("2", 2000),
        ("3", None),
        (["4", "5"], 5000),
    ],
)
def test_get_nested_value(value, expected):
    mapping_table = {"1": 1000, "2": 2000, "4": {"5": 5000}}
    assert _get_nested_value(mapping_table, value) == expected


def test_get_nested_value_none():
    assert _get_nested_value(None, "4") is None


@pytest.mark.parametrize(
    "decimal_places,expected",
    [(None, 5), (4, 4), ("4", 5)],
)
def test_decimal_places(decimal_places, expected):
    assert _decimal_places(decimal_places) == expected


def test_transform(imodel_functions):
    series = pd.Series(data={"a": 1, "b": 2, "c": np.nan}, index=["a", "b", "c"])
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    result = _transform(series, imodel_functions, "integer_to_float", {}, logger)
    expected = pd.Series(data={"a": 1.0, "b": 2.0, "c": np.nan}, index=["a", "b", "c"])
    pd.testing.assert_series_equal(result, expected)


def test_transform_notfound(imodel_functions):
    series = pd.Series(data={"a": 1, "b": 2, "c": np.nan}, index=["a", "b", "c"])
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    result = _transform(series, imodel_functions, "invalid_function", {}, logger)
    pd.testing.assert_series_equal(result, series)


def test_code_table():
    series = pd.Series(
        data={"a": "1", "b": "2", "c": np.nan},
        index=["a", "b", "c"],
        name=("test", "data"),
    )
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    result = _code_table(series, "icoads_r300_d721", "baro_units", logger)
    expected = pd.Series(
        data={"a": 1001.0, "b": 1004.0, "c": np.nan},
        index=["a", "b", "c"],
    )
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "default,length,expected",
    [
        (5, 1, [5]),
        (5, 5, [5, 5, 5, 5, 5]),
        ([5], 5, [[5], [5], [5], [5], [5]]),
        ([5, 6], 5, [[5, 6], [5, 6], [5, 6], [5, 6], [5, 6]]),
    ],
)
def test_default(default, length, expected):
    assert _default(default, length) == expected


@pytest.mark.parametrize(
    "series,fill_value,expected",
    [
        (
            pd.Series(data={"a": 1, "b": 2, "c": 3}, index=["a", "b", "c"]),
            None,
            pd.Series(data={"a": 1, "b": 2, "c": 3}, index=["a", "b", "c"]),
        ),
        (
            pd.Series(data={"a": 1, "b": None, "c": np.nan}, index=["a", "b", "c"]),
            5,
            pd.Series(data={"a": 1, "b": 5, "c": 5.0}, index=["a", "b", "c"]),
        ),
    ],
)
def test_fill_value(series, fill_value, expected):
    result = _fill_value(series, fill_value)
    if isinstance(result, pd.Series):
        pd.testing.assert_series_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize(
    "value,atts,expected",
    [
        (5, {"data_type": "numeric", "decimal_places": 2}, "5.00"),
        (5, None, np.nan),
        (5, {"data_type": "invalid"}, 5),
        ("5", {"data_type": "int"}, "5"),
    ],
)
def test_convert_dtype(value, atts, expected):
    idata = pd.Series(value)
    result = _convert_dtype(idata, atts)
    if isinstance(result, pd.Series):
        pd.testing.assert_series_equal(result, pd.Series(expected))
    else:
        assert result, expected


@pytest.mark.parametrize(
    "column, elements, default, use_default, exp",
    [
        ("duplicate_status", None, 4, True, [4, 4, 4, 4]),
        ("platform_type", [("c1", "PT")], None, False, "idata"),
        ("report_id", [("c98", "UID")], None, False, "idata"),
        ("latitude", [("core", "LAT")], None, True, [None, None, None, None]),
        ("location_quality", [("c1", "LZ")], None, False, "idata"),
    ],
)
def test_extract_input_data(
    imodel_maps, data_header, column, elements, default, use_default, exp
):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    result = _extract_input_data(
        data_header,
        elements,
        data_header.columns,
        default,
        logger,
    )
    assert isinstance(result, tuple)

    assert result[1] is use_default

    if exp == "idata":
        exp = data_header[elements[0]]
    elif isinstance(exp, list):
        exp = pd.Series(exp)

    pd.testing.assert_series_equal(result[0], exp)


@pytest.mark.parametrize(
    "column, expected",
    [
        ("duplicate_status", [4, 4, 4, 4]),
        ("platform_type", [2, 33, 32, 45]),
        (
            "report_id",
            ["ICOADS-30-5012", "ICOADS-30-8960", "ICOADS-30-0037", "ICOADS-30-1000"],
        ),
        ("location_quality", [2.0, "0", "0", "0"]),
        ("latitude", [None, None, None, None]),
    ],
)
def test_column_mapping(imodel_maps, imodel_functions, data_header, column, expected):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    mapping_column = imodel_maps["header"][column]
    column_atts = get_cdm_atts("header")["header"][column]
    result, _ = _column_mapping(
        data_header,
        mapping_column,
        imodel_functions,
        column_atts,
        None,
        data_header.columns,
        column,
        logger,
    )
    pd.testing.assert_series_equal(result, pd.Series(expected, name=column))


def test_table_mapping(
    imodel_maps, imodel_functions, data_header, data_header_expected
):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    table_atts = get_cdm_atts("header")["header"]
    result = _table_mapping(
        data_header,
        imodel_maps["header"],
        table_atts,
        data_header.columns,
        "null",
        imodel_functions,
        None,
        True,
        False,
        False,
        logger,
    )
    expected = data_header_expected["header"]
    pd.testing.assert_frame_equal(result[expected.columns], expected)


def test_map_and_convert(data_header, data_header_expected):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    result = _map_and_convert(
        "icoads",
        "r300",
        "d720",
        data=data_header,
        cdm_subset=["header"],
        logger=logger,
    )
    pd.testing.assert_frame_equal(
        result[data_header_expected.columns], data_header_expected
    )


def test_map_model_icoads(data_header, data_header_expected):
    result = map_model(
        data_header,
        "icoads_r300_d720",
        cdm_subset=["header"],
    )
    pd.testing.assert_frame_equal(
        result[data_header_expected.columns], data_header_expected
    )


def test_map_model_pub47():
    pub47_csv = test_data["test_pub47"]["source"]
    df = pd.read_csv(
        pub47_csv,
        delimiter="|",
        dtype="object",
        header=0,
        na_values="MSNG",
    ).head(5)

    result = map_model(
        df,
        "pub47",
        drop_missing_obs=False,
        drop_duplicates=False,
    )

    columns = [
        ("header", "station_name"),
        ("header", "platform_sub_type"),
        ("header", "primary_station_id"),
        ("header", "station_record_number"),
        ("header", "report_duration"),
        ("observations-at", "sensor_automation_status"),
        ("observations-at", "z_coordinate"),
        ("observations-at", "observation_height_above_station_surface"),
        ("observations-at", "sensor_id"),
        ("observations-dpt", "sensor_automation_status"),
        ("observations-dpt", "z_coordinate"),
        ("observations-dpt", "observation_height_above_station_surface"),
        ("observations-dpt", "sensor_id"),
        ("observations-slp", "sensor_automation_status"),
        ("observations-slp", "z_coordinate"),
        ("observations-slp", "observation_height_above_station_surface"),
        ("observations-slp", "sensor_id"),
        ("observations-sst", "sensor_automation_status"),
        ("observations-sst", "z_coordinate"),
        ("observations-sst", "observation_height_above_station_surface"),
        ("observations-sst", "sensor_id"),
        ("observations-wbt", "sensor_automation_status"),
        ("observations-wbt", "z_coordinate"),
        ("observations-wbt", "observation_height_above_station_surface"),
        ("observations-wbt", "sensor_id"),
        ("observations-wd", "sensor_automation_status"),
        ("observations-wd", "z_coordinate"),
        ("observations-wd", "observation_height_above_station_surface"),
        ("observations-wd", "sensor_id"),
        ("observations-ws", "sensor_automation_status"),
        ("observations-ws", "z_coordinate"),
        ("observations-ws", "observation_height_above_station_surface"),
        ("observations-ws", "sensor_id"),
    ]
    result = result[columns]

    exp = np.array(
        [
            [
                "DIMLINGTON",
                "FS AQUARIUS",
                "CMA CGM SWORDFISH",
                "ZENITH LEADER",
                "MAERSK KENSINGTON",
            ],
            ["null", "27", "null", "30", "4"],
            ["03380", "2AAY7", "2ABB2", "2ACU6", "2AEC7"],
            ["0", "6", "2", "3", "4"],
            ["9", "11", "null", "15", "11"],
            ["1", "5", "5", "5", "2"],
            ["null", "17.5", "30.0", "27.0", "32.0"],
            ["null", "17.5", "30.0", "27.0", "32.0"],
            ["AT", "AT", "AT", "AT", "AT"],
            ["1", "5", "5", "5", "2"],
            ["null", "17.5", "30.0", "27.0", "32.0"],
            ["null", "17.5", "30.0", "27.0", "32.0"],
            ["HUM", "HUM", "HUM", "HUM", "HUM"],
            ["1", "5", "5", "5", "2"],
            ["null", "17.5", "30.0", "27.0", "32.0"],
            ["null", "17.5", "30.0", "27.0", "32.0"],
            ["SLP", "SLP", "SLP", "SLP", "SLP"],
            ["1", "5", "5", "5", "2"],
            ["null", "null", "null", "-5.5", "-7.0"],
            ["null", "null", "null", "-5.5", "-7.0"],
            ["SST", "SST", "SST", "SST", "SST"],
            ["1", "5", "5", "5", "2"],
            ["null", "17.5", "30.0", "27.0", "32.0"],
            ["null", "17.5", "30.0", "27.0", "32.0"],
            ["HUM", "HUM", "HUM", "HUM", "HUM"],
            ["1", "5", "5", "5", "2"],
            ["null", "21.0", "30.0", "3.5", "9.0"],
            ["null", "21.0", "30.0", "3.5", "9.0"],
            ["WSPD", "WSPD", "WSPD", "WSPD", "WSPD"],
            ["1", "5", "5", "5", "2"],
            ["null", "21.0", "30.0", "3.5", "9.0"],
            ["null", "21.0", "30.0", "3.5", "9.0"],
            ["WSPD", "WSPD", "WSPD", "WSPD", "WSPD"],
        ]
    )
    expected = pd.DataFrame(
        data=exp.T,
        columns=pd.MultiIndex.from_tuples(columns),
    )

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data_model",
    [
        "icoads_r300_d714",
        "icoads_r300_d701",
        "icoads_r300_d706",
        "icoads_r300_d705",
        "icoads_r300_d702",
        "icoads_r300_d707",
        "icoads_r302_d794",
        "icoads_r300_d704",
        "icoads_r300_d721",
        "icoads_r300_d730",
        "icoads_r300_d781",
        "icoads_r300_d703",
        "icoads_r300_d201",
        "icoads_r300_d892",
        "icoads_r300_d700",
        "icoads_r302_d792",
        "icoads_r302_d992",
        "craid",
        "gdac",
    ],
)
def test_map_model_test_data(data_model):
    _map_model_test_data(data_model)


def test_map_model_test_data_mixed():
    _map_model_test_data("icoads_r300_mixed", encoding="cp1252")


def test_map_model_test_data_select():
    _map_model_test_data(
        "icoads_r300_d714",
        select=["header", "observations-sst"],
        cdm_subset=["header", "observations-sst"],
    )
