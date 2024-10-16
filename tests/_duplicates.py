from __future__ import annotations

from cdm_reader_mapper.cdm_mapper import read_tables

from ._results import result_data


def _manipulate_header(df):
    # Duplicate : Different report_id's
    # Failure in data set;
    # each report needs a specific report_id
    df.loc[5] = df.loc[4]
    df.loc[5, "report_id"] = "ICOADS-302-N688EY"
    df.loc[5, "report_quality"] = 2

    # No Duplicate: Lat and Lon values differ to much
    # valid is .5 degrees
    df.loc[6] = df.loc[4]
    df.loc[6, "report_id"] = "ICOADS-302-N688EZ"
    df.loc[6, "latitude"] = -65.80
    df.loc[6, "longitude"] = 21.20
    df.loc[6, "report_quality"] = 2

    # Duplicate: report timestamp differs no enough
    # valid is 60 seconds
    df.loc[7] = df.loc[1]
    df.loc[7, "report_id"] = "ICOADS-302-N688DT"
    df.loc[7, "report_timestamp"] = "2022-02-01 00:01:00"
    df.loc[7, "report_quality"] = 2

    # No Duplicate: report timestamp differs to much
    # valid is 60 seconds
    df.loc[8] = df.loc[1]
    df.loc[8, "report_id"] = "ICOADS-302-N688DU"
    df.loc[8, "report_timestamp"] = "2022-02-02 00:00:00"
    df.loc[8, "report_quality"] = 2

    # Duplicate : Different report_id's
    # Failure in data set
    df.loc[9] = df.loc[2]
    df.loc[9, "report_id"] = "ICOADS-302-N688DW"
    df.loc[9, "report_quality"] = 2

    # Duplicate : Different report_id's
    # Failure in data set
    # each report needs a specific report_id
    df.loc[10] = df.loc[3]
    df.loc[10, "report_id"] = "ICOADS-302-N688EF"
    df.loc[10, "latitude"] = 66.00
    df.loc[10, "longitude"] = 8.50
    df.loc[10, "report_quality"] = 2

    # Duplicate: Lat and Lon values differ not enough
    # valid is .5 degrees
    df.loc[11] = df.loc[3]
    df.loc[11, "report_id"] = "ICOADS-302-N688EE"
    df.loc[11, "latitude"] = 66.05
    df.loc[11, "longitude"] = 8.15
    df.loc[11, "report_quality"] = 2

    # No Duplicate: primary_station_id differs
    df.loc[12] = df.loc[3]
    df.loc[12, "report_id"] = "ICOADS-302-N688ED"
    df.loc[12, "primary_station_id"] = "MASKSTIP"
    df.loc[12, "report_quality"] = 2

    # Duplicate: Lat and Lon values differ not enough
    # valid is .5 degrees
    df.loc[13] = df.loc[3]
    df.loc[13, "report_id"] = "ICOADS-302-N688EC"
    df.loc[13, "latitude"] = 65.95
    df.loc[13, "longitude"] = 8.05
    df.loc[13, "report_quality"] = 2

    # Duplicate: ignore primary_station_id SHIP
    df.loc[14] = df.loc[3]
    df.loc[14, "report_id"] = "ICOADS-302-N688EG"
    df.loc[14, "primary_station_id"] = "SHIP"
    df.loc[14, "report_quality"] = 2

    # No Duplicate: Lat and Lon values differ to much
    # valid is .5 degrees
    df.loc[15] = df.loc[4]
    df.loc[15, "report_id"] = "ICOADS-302-N688EV"
    df.loc[15, "latitude"] = 65.60
    df.loc[15, "longitude"] = -21.40
    df.loc[15, "report_quality"] = 2

    # Duplicate: Lat and Lon values differ not enough
    # valid is .5 degrees
    df.loc[16] = df.loc[4]
    df.loc[16, "report_id"] = "ICOADS-302-N688EW"
    df.loc[16, "latitude"] = 65.90
    df.loc[16, "longitude"] = -21.10
    df.loc[16, "report_quality"] = 2

    # No Duplicate:
    df.loc[17] = df.loc[1]
    df.loc[17, "report_id"] = "ICOADS-302-N688EK"
    df.loc[17, "station_course"] = 316.0

    # No Duplicate:
    df.loc[18] = df.loc[1]
    df.loc[18, "report_id"] = "ICOADS-302-N688EL"
    df.loc[18, "station_speed"] = 4.0

    # Duplicate:
    df.loc[19] = df.loc[1]
    df.loc[19, "report_id"] = "ICOADS-302-N688EM"
    df.loc[19, "station_course"] = "null"

    # Duplicate:
    df.loc[20] = df.loc[1]
    df.loc[20, "report_id"] = "ICOADS-302-N688EN"
    df.loc[20, "station_speed"] = "null"
    return df


def _get_test_data(imodel):
    exp_name = f"expected_{imodel}"
    exp_data = getattr(result_data, exp_name)
    data_path = exp_data.get("cdm_table")
    return read_tables(
        data_path,
        tb_id=f"{imodel}*",
        cdm_subset="header",
    )


exp1 = {
    "duplicate_status": [0, 1, 1, 1, 1, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 0, 0, 0],
    "report_quality": [1, 1, 0, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1],
    "duplicates": [
        "null",
        "{ICOADS-302-N688DT}",
        "{ICOADS-302-N688DW}",
        "{ICOADS-302-N688EC,ICOADS-302-N688EE}",
        "{ICOADS-302-N688EW,ICOADS-302-N688EY}",
        "{ICOADS-302-N688EI}",
        "null",
        "{ICOADS-302-N688DS}",
        "null",
        "{ICOADS-302-N688DV}",
        "null",
        "{ICOADS-302-N688EH}",
        "null",
        "{ICOADS-302-N688EH}",
        "null",
        "null",
        "{ICOADS-302-N688EI}",
        "null",
        "null",
        "null",
        "null",
    ],
}

exp2 = {
    "duplicate_status": [0, 1, 1, 3, 1, 3, 0, 3, 0, 3, 0, 3, 1, 3, 3, 0, 3, 0, 0, 0, 0],
    "report_quality": [1, 1, 0, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1],
    "duplicates": [
        "null",
        "{ICOADS-302-N688DT}",
        "{ICOADS-302-N688DW}",
        "{ICOADS-302-N688ED}",
        "{ICOADS-302-N688EW,ICOADS-302-N688EY}",
        "{ICOADS-302-N688EI}",
        "null",
        "{ICOADS-302-N688DS}",
        "null",
        "{ICOADS-302-N688DV}",
        "null",
        "{ICOADS-302-N688ED}",
        "{ICOADS-302-N688EC,ICOADS-302-N688EE,ICOADS-302-N688EG,ICOADS-302-N688EH}",
        "{ICOADS-302-N688ED}",
        "{ICOADS-302-N688ED}",
        "null",
        "{ICOADS-302-N688EI}",
        "null",
        "null",
        "null",
        "null",
    ],
}

exp3 = {
    "duplicate_status": [1, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3],
    "report_quality": [1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1],
    "duplicates": [
        "{ICOADS-302-N688DS,ICOADS-302-N688DT,ICOADS-302-N688DV,ICOADS-302-N688DW,ICOADS-302-N688EC,ICOADS-302-N688EE,ICOADS-302-N688EF,ICOADS-302-N688EH,ICOADS-302-N688EI,ICOADS-302-N688EK,ICOADS-302-N688EL,ICOADS-302-N688EM,ICOADS-302-N688EN,ICOADS-302-N688EV,ICOADS-302-N688EW,ICOADS-302-N688EY,ICOADS-302-N688EZ}",
        "{ICOADS-302-N688DR}",
        "{ICOADS-302-N688DR}",
        "{ICOADS-302-N688DR}",
        "{ICOADS-302-N688DR}",
        "{ICOADS-302-N688DR}",
        "{ICOADS-302-N688DR}",
        "{ICOADS-302-N688DR}",
        "null",
        "{ICOADS-302-N688DR}",
        "{ICOADS-302-N688DR}",
        "{ICOADS-302-N688DR}",
        "null",
        "{ICOADS-302-N688DR}",
        "null",
        "{ICOADS-302-N688DR}",
        "{ICOADS-302-N688DR}",
        "{ICOADS-302-N688DR}",
        "{ICOADS-302-N688DR}",
        "{ICOADS-302-N688DR}",
        "{ICOADS-302-N688DR}",
    ],
}

exp4 = {
    "duplicate_status": [0, 1, 1, 1, 1, 3, 0, 3, 0, 3, 0, 3, 3, 3, 3, 0, 3, 0, 0, 0, 0],
    "report_quality": [1, 1, 0, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
    "duplicates": [
        "null",
        "{ICOADS-302-N688DT}",
        "{ICOADS-302-N688DW}",
        "{ICOADS-302-N688EC,ICOADS-302-N688ED,ICOADS-302-N688EE,ICOADS-302-N688EG}",
        "{ICOADS-302-N688EW,ICOADS-302-N688EY}",
        "{ICOADS-302-N688EI}",
        "null",
        "{ICOADS-302-N688DS}",
        "null",
        "{ICOADS-302-N688DV}",
        "null",
        "{ICOADS-302-N688EH}",
        "{ICOADS-302-N688EH}",
        "{ICOADS-302-N688EH}",
        "{ICOADS-302-N688EH}",
        "null",
        "{ICOADS-302-N688EI}",
        "null",
        "null",
        "null",
        "null",
    ],
}

exp5 = {
    "duplicate_status": [0, 1, 1, 1, 1, 3, 0, 3, 0, 3, 3, 3, 0, 3, 0, 3, 3, 0, 0, 0, 0],
    "report_quality": [1, 1, 0, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1],
    "duplicates": [
        "null",
        "{ICOADS-302-N688DT}",
        "{ICOADS-302-N688DW}",
        "{ICOADS-302-N688EC,ICOADS-302-N688EE,ICOADS-302-N688EF}",
        "{ICOADS-302-N688EV,ICOADS-302-N688EW,ICOADS-302-N688EY}",
        "{ICOADS-302-N688EI}",
        "null",
        "{ICOADS-302-N688DS}",
        "null",
        "{ICOADS-302-N688DV}",
        "{ICOADS-302-N688EH}",
        "{ICOADS-302-N688EH}",
        "null",
        "{ICOADS-302-N688EH}",
        "null",
        "{ICOADS-302-N688EI}",
        "{ICOADS-302-N688EI}",
        "null",
        "null",
        "null",
        "null",
    ],
}

exp6 = {
    "duplicate_status": [0, 0, 1, 1, 1, 3, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 0, 0, 0],
    "report_quality": [1, 1, 0, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1],
    "duplicates": [
        "null",
        "null",
        "{ICOADS-302-N688DW}",
        "{ICOADS-302-N688EC,ICOADS-302-N688EE}",
        "{ICOADS-302-N688EW,ICOADS-302-N688EY}",
        "{ICOADS-302-N688EI}",
        "null",
        "null",
        "null",
        "{ICOADS-302-N688DV}",
        "null",
        "{ICOADS-302-N688EH}",
        "null",
        "{ICOADS-302-N688EH}",
        "null",
        "null",
        "{ICOADS-302-N688EI}",
        "null",
        "null",
        "null",
        "null",
    ],
}

exp7 = {
    "duplicate_status": [0, 1, 1, 1, 1, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 0, 3, 3],
    "report_quality": [1, 1, 0, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1],
    "duplicates": [
        "null",
        "{ICOADS-302-N688DT,ICOADS-302-N688EM,ICOADS-302-N688EN}",
        "{ICOADS-302-N688DW}",
        "{ICOADS-302-N688EC,ICOADS-302-N688EE}",
        "{ICOADS-302-N688EW,ICOADS-302-N688EY}",
        "{ICOADS-302-N688EI}",
        "null",
        "{ICOADS-302-N688DS}",
        "null",
        "{ICOADS-302-N688DV}",
        "null",
        "{ICOADS-302-N688EH}",
        "null",
        "{ICOADS-302-N688EH}",
        "null",
        "null",
        "{ICOADS-302-N688EI}",
        "null",
        "null",
        "{ICOADS-302-N688DS}",
        "{ICOADS-302-N688DS}",
    ],
}

exp8 = {
    "duplicate_status": [0, 1, 1, 3, 1, 3, 0, 3, 0, 3, 0, 3, 1, 3, 3, 0, 3, 0, 0, 3, 3],
    "report_quality": [1, 1, 0, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1],
    "duplicates": [
        "null",
        "{ICOADS-302-N688DT,ICOADS-302-N688EM,ICOADS-302-N688EN}",
        "{ICOADS-302-N688DW}",
        "{ICOADS-302-N688ED}",
        "{ICOADS-302-N688EW,ICOADS-302-N688EY}",
        "{ICOADS-302-N688EI}",
        "null",
        "{ICOADS-302-N688DS}",
        "null",
        "{ICOADS-302-N688DV}",
        "null",
        "{ICOADS-302-N688ED}",
        "{ICOADS-302-N688EC,ICOADS-302-N688EE,ICOADS-302-N688EG,ICOADS-302-N688EH}",
        "{ICOADS-302-N688ED}",
        "{ICOADS-302-N688ED}",
        "null",
        "{ICOADS-302-N688EI}",
        "null",
        "null",
        "{ICOADS-302-N688DS}",
        "{ICOADS-302-N688DS}",
    ],
}

method_kwargs_ = {
    "left_on": "report_timestamp",
    "window": 7,
    "block_on": ["primary_station_id"],
}

compare_kwargs_ = {
    "primary_station_id": {"method": "exact"},
    "report_timestamp": {
        "method": "date2",
        "kwargs": {"method": "gauss", "offset": 60.0},
    },
}
df_icoads = _get_test_data("icoads_r302_d792")
df_icoads = _manipulate_header(df_icoads)

df_craid = _get_test_data("craid")
