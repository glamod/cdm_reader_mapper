from __future__ import annotations

import numpy as np
import pandas as pd
import pytest  # noqa

from cdm_reader_mapper.cdm_mapper.mapper import map_model

from cdm_reader_mapper.data import test_data


def test_map_model():
    pub47_csv = test_data["test_pub47"]["source"]  # noqa
    df = pd.read_csv(
        pub_47_csv,
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
