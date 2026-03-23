from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdm_reader_mapper.cdm_mapper.mapper import (
    _is_empty,
    _drop_duplicated_rows,
    _get_nested_value,
    _transform,
    _code_table,
    _default,
    _fill_value,
    _extract_input_data,
    _column_mapping,
    _table_mapping,
    _prepare_cdm_tables,
    map_model,
)

from cdm_reader_mapper.common import logging_hdlr
from cdm_reader_mapper.common.json_dict import open_json_file

from cdm_reader_mapper.cdm_mapper.properties import cdm_tables
from cdm_reader_mapper.cdm_mapper.reader import read_tables
from cdm_reader_mapper.cdm_mapper.tables.tables import get_imodel_maps, get_cdm_atts
from cdm_reader_mapper.cdm_mapper.utils.mapping_functions import mapping_functions

from cdm_reader_mapper.data import test_data


from cdm_reader_mapper.cdm_mapper.writer import write_tables

from cdm_reader_mapper.cdm_mapper.writer import write_tables

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
    data = pd.DataFrame(
        {
            ("header", "report_id"): [
                "ICOADS-300-5012",
                "ICOADS-300-8960",
                "ICOADS-300-0037",
                "ICOADS-300-1000",
            ],
            ("header", "duplicate_status"): [4, 4, 4, 4],
            ("header", "platform_type"): [2, 33, 32, 45],
            ("header", "location_quality"): [2, 0, 0, 0],
            ("header", "source_id"): [pd.NA, pd.NA, pd.NA, pd.NA],
        }
    )
    return data.astype(
        {
            ("header", "report_id"): object,
            ("header", "duplicate_status"): "Int64",
            ("header", "platform_type"): "Int64",
            ("header", "location_quality"): "Int64",
            ("header", "source_id"): object,
        }
    )


def _map_model_test_data(
    data_model, encoding="utf-8", select=None, chunksize=None, **kwargs
):
    source = test_data[f"test_{data_model}"]["mdf_data"]
    
    mdf_info = test_data[f"test_{data_model}"]["mdf_info"]
    if mdf_info is None:
        dtypes = object
    else:
        info = open_json_file(test_data[f"test_{data_model}"]["mdf_info"])
        dtypes = info["dtypes"]

    delimiter = test_data[f"test_{data_model}"]["delimiter"]

    df = pd.read_csv(
        source,
        dtype=dtypes,
        chunksize=chunksize,
        delimiter=delimiter,
        encoding=encoding,
    )

    if chunksize is None and ":" in df.columns[0]:
        df.columns = pd.MultiIndex.from_tuples(col.split(":") for col in df.columns)

    result = map_model(df, data_model, **kwargs)

    if chunksize:
        result = result.read()

    if not select:
        select = cdm_tables
        
    for cdm_table in select:
        result_table = result[cdm_table].copy()
        result_table = result_table.dropna(how="all")
        result_table = result_table.reset_index(drop=True)
        
        try:
          expected_table = read_tables(
            test_data[f"test_{data_model}"][f"cdm_{cdm_table}"].parent,
            data_format="parquet",
            extension="pq",
            suffix="*",
            cdm_subset=cdm_table,
          )[cdm_table]
        except ValueError:
            expected_table =pd.DataFrame()

        if result_table.empty and expected_table.empty:
            continue          

        if "record_timestamp" in expected_table.columns:
            expected_table = expected_table.drop("record_timestamp", axis=1)
            result_table = result_table.drop("record_timestamp", axis=1)
        if "history" in expected_table.columns:
            expected_table = expected_table.drop("history", axis=1)
            result_table = result_table.drop("history", axis=1)

        pd.testing.assert_frame_equal(result_table, expected_table)


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
        (["4", "6"], None),
        ([], None),
    ],
)
def test_get_nested_value(value, expected):
    mapping_table = {"1": 1000, "2": 2000, "4": {"5": 5000}}
    assert _get_nested_value(mapping_table, value) == expected


def test_get_nested_value_none():
    assert _get_nested_value(None, "4") is None
    assert _get_nested_value({"1", 1000}, ["1", "x"]) is None


@pytest.mark.parametrize(
    "table",
    [
        "header",
        "observations-at",
        "observations-dpt",
        "observations-slp",
        "observations-sst",
        "observations-wbt",
        "observations-wd",
        "observations-ws",
    ],
)
@pytest.mark.parametrize("is_list", [True, False])
def test_prepare_cdm_tables(table, is_list):
    if is_list is True:
        table_in = [table]
    else:
        table_in = table

    result = _prepare_cdm_tables(table_in)

    assert isinstance(result, dict)
    assert list(result.keys()) == [table]


def test_prepare_cdm_tables_invalid():
    result = _prepare_cdm_tables("invalid")
    assert result == {}


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
    "column, elements, default, use_default, exp",
    [
        ("duplicate_status", None, 4, True, [4, 4, 4, 4]),
        ("platform_type", [("c1", "PT")], None, False, "idata"),
        ("report_id", [("c98", "UID")], None, False, "idata"),
        ("latitude", [("core", "LAT")], None, True, [None, None, None, None]),
        ("location_quality", [("c1", "LZ")], None, False, "idata"),
    ],
)
def test_extract_input_data(data_header, column, elements, default, use_default, exp):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    result = _extract_input_data(
        data_header,
        elements,
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


def test_extract_input_data_empty():
    test_data = pd.DataFrame({"a": []})
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    result = _extract_input_data(
        test_data,
        ["a"],
        "null",
        logger,
    )
    assert isinstance(result, tuple)

    assert result[1] is True

    pd.testing.assert_series_equal(result[0], pd.Series([]))


@pytest.mark.parametrize(
    "column, dtype, expected",
    [
        ("duplicate_status", "Int64", [4, 4, 4, 4]),
        ("platform_type", "Int64", [2, 33, 32, 45]),
        (
            "report_id",
            str,
            [
                "ICOADS-300-5012",
                "ICOADS-300-8960",
                "ICOADS-300-0037",
                "ICOADS-300-1000",
            ],
        ),
        ("location_quality", "Int64", [2, 0, 0, 0]),
        ("latitude", "Float64", [pd.NA, pd.NA, pd.NA, pd.NA]),
    ],
)
def test_column_mapping(
    imodel_maps, imodel_functions, data_header, column, dtype, expected
):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    mapping_column = imodel_maps["header"][column]
    column_atts = get_cdm_atts("header")["header"][column]
    result = _column_mapping(
        data_header,
        mapping_column,
        imodel_functions,
        column_atts,
        None,
        column,
        logger,
    )
    pd.testing.assert_series_equal(
        result, pd.Series(expected, name=column, dtype=dtype)
    )


def test_history_column_mapping(imodel_maps, imodel_functions, data_header):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    mapping_column = imodel_maps["header"]["history"]
    column_atts = get_cdm_atts("header")["header"]["history"]
    result = _column_mapping(
        data_header,
        mapping_column,
        imodel_functions,
        column_atts,
        None,
        "history",
        logger,
    )
    assert result.str.contains("Initial conversion from ICOADS R3.0.0T").all()


def test_column_mapping_subset(imodel_maps, imodel_functions, data_header):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    column = "platform_type"
    mapping_column = imodel_maps["header"][column]
    column_atts = get_cdm_atts("header")["header"][column]
    result = _column_mapping(
        data_header,
        mapping_column,
        imodel_functions,
        column_atts,
        ["new_platform_type"],
        column,
        logger,
    )
    expected = data_header[("c1", "PT")].rename(column).astype("Int64")
    pd.testing.assert_series_equal(result, expected)


def test_table_mapping_basic(
    imodel_maps, imodel_functions, data_header, data_header_expected
):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    table_atts = get_cdm_atts("header")["header"]
    results = _table_mapping(
        data_header,
        imodel_maps["header"],
        table_atts,
        imodel_functions,
        None,
        True,
        False,
        False,
        logger,
    )
    expected = data_header_expected["header"]
    result = results[expected.columns]

    pd.testing.assert_frame_equal(result, expected)


def test_table_mapping_empty(imodel_maps, imodel_functions, data_header):
    logger = logging_hdlr.init_logger(__name__, level="INFO")
    result = _table_mapping(
        data_header,
        imodel_maps["header"],
        {},
        imodel_functions,
        None,
        True,
        False,
        False,
        logger,
    )
    pd.testing.assert_frame_equal(result, pd.DataFrame(index=data_header.index))


def test_map_model_icoads(data_header, data_header_expected):
    result = map_model(
        data_header,
        "icoads_r300_d720",
        cdm_subset=["header"],
    )
    c = ("header", "duplicate_status")
    pd.testing.assert_frame_equal(
        result[data_header_expected.columns], data_header_expected
    )


def test_map_model_raises(data_header):
    with pytest.raises(ValueError, match="is not defined"):
        map_model(data_header, None)
    with pytest.raises(TypeError, match="Input data model type is not supported"):
        map_model(data_header, ["icoads_r300_d720"])
    with pytest.raises(ValueError, match="not supported"):
        map_model(data_header, "icaods_r300_d720")


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

    dtypes = {
        ("header", "station_name"): str,
        ("header", "platform_sub_type"): "Int64",
        ("header", "primary_station_id"): str,
        ("header", "station_record_number"): "Int64",
        ("header", "report_duration"): "Int64",
        ("observations-at", "sensor_automation_status"): "Int64",
        ("observations-at", "z_coordinate"): "Float64",
        ("observations-at", "observation_height_above_station_surface"): "Float64",
        ("observations-at", "sensor_id"): str,
        ("observations-dpt", "sensor_automation_status"): "Int64",
        ("observations-dpt", "z_coordinate"): "Float64",
        ("observations-dpt", "observation_height_above_station_surface"): "Float64",
        ("observations-dpt", "sensor_id"): str,
        ("observations-slp", "sensor_automation_status"): "Int64",
        ("observations-slp", "z_coordinate"): "Float64",
        ("observations-slp", "observation_height_above_station_surface"): "Float64",
        ("observations-slp", "sensor_id"): str,
        ("observations-sst", "sensor_automation_status"): "Int64",
        ("observations-sst", "z_coordinate"): "Float64",
        ("observations-sst", "observation_height_above_station_surface"): "Float64",
        ("observations-sst", "sensor_id"): str,
        ("observations-wbt", "sensor_automation_status"): "Int64",
        ("observations-wbt", "z_coordinate"): "Float64",
        ("observations-wbt", "observation_height_above_station_surface"): "Float64",
        ("observations-wbt", "sensor_id"): str,
        ("observations-wd", "sensor_automation_status"): "Int64",
        ("observations-wd", "z_coordinate"): "Float64",
        ("observations-wd", "observation_height_above_station_surface"): "Float64",
        ("observations-wd", "sensor_id"): str,
        ("observations-ws", "sensor_automation_status"): "Int64",
        ("observations-ws", "z_coordinate"): "Float64",
        ("observations-ws", "observation_height_above_station_surface"): "Float64",
        ("observations-ws", "sensor_id"): str,
    }

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
            [pd.NA, 27, pd.NA, 30, 4],
            ["03380", "2AAY7", "2ABB2", "2ACU6", "2AEC7"],
            [0, 6, 2, 3, 4],
            [9, 11, pd.NA, 15, 11],
            [1, 5, 5, 5, 2],
            [np.nan, 17.5, 30.0, 27.0, 32.0],
            [np.nan, 17.5, 30.0, 27.0, 32.0],
            ["AT", "AT", "AT", "AT", "AT"],
            [1, 5, 5, 5, 2],
            [np.nan, 17.5, 30.0, 27.0, 32.0],
            [np.nan, 17.5, 30.0, 27.0, 32.0],
            ["HUM", "HUM", "HUM", "HUM", "HUM"],
            [1, 5, 5, 5, 2],
            [np.nan, 17.5, 30.0, 27.0, 32.0],
            [np.nan, 17.5, 30.0, 27.0, 32.0],
            ["SLP", "SLP", "SLP", "SLP", "SLP"],
            [1, 5, 5, 5, 2],
            [np.nan, np.nan, np.nan, -5.5, -7.0],
            [np.nan, np.nan, np.nan, -5.5, -7.0],
            ["SST", "SST", "SST", "SST", "SST"],
            [1, 5, 5, 5, 2],
            [np.nan, 17.5, 30.0, 27.0, 32.0],
            [np.nan, 17.5, 30.0, 27.0, 32.0],
            ["HUM", "HUM", "HUM", "HUM", "HUM"],
            [1, 5, 5, 5, 2],
            [np.nan, 21.0, 30.0, 3.5, 9.0],
            [np.nan, 21.0, 30.0, 3.5, 9.0],
            ["WSPD", "WSPD", "WSPD", "WSPD", "WSPD"],
            [1, 5, 5, 5, 2],
            [np.nan, 21.0, 30.0, 3.5, 9.0],
            [np.nan, 21.0, 30.0, 3.5, 9.0],
            ["WSPD", "WSPD", "WSPD", "WSPD", "WSPD"],
        ]
    )
    expected = pd.DataFrame(
        data=exp.T,
        columns=pd.MultiIndex.from_tuples(columns),
    )
    expected = expected.astype(dtypes)

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
        "marob",
    ],
)
def test_map_model_test_data_basic(data_model):
    _map_model_test_data(data_model)


def test_map_model_test_data_mixed():
    _map_model_test_data("icoads_r300_mixed", encoding="cp1252")


def test_map_model_test_data_select():
    _map_model_test_data(
        "icoads_r300_d714",
        select=["header", "observations-sst"],
        cdm_subset=["header", "observations-sst"],
    )


def test_map_model_test_data_chunksize():
    _map_model_test_data(
        "icoads_r300_d714",
        chunksize=2,
    )
