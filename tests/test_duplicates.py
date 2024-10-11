from __future__ import annotations

import pytest  # noqa
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from cdm_reader_mapper.cdm_mapper import duplicate_check, read_tables

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
    df.loc[5, "report_id"] = "ICOADS-302-N688EY"
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


df = _get_test_data("icoads_r302_d792")
df = _manipulate_header(df)
DupDetect = duplicate_check(df)


def test_duplicates_flag():
    DupDetect.flag_duplicates()
    result = DupDetect.result
    assert_array_equal(
        result["duplicate_status"], [0, 1, 1, 3, 1, 3, 0, 3, 0, 3, 0, 3, 1, 3, 3]
    )
    assert_array_equal(
        result["report_quality"], [1, 1, 0, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1]
    )
    assert_array_equal(
        result["duplicates"],
        [
            "null",
            "{ICOADS-302-N688DT}",
            "{ICOADS-302-N688DW}",
            "{ICOADS-302-N688ED}",
            "{ICOADS-302-N688EY}",
            "{ICOADS-302-N688EI}",
            "null",
            "{ICOADS-302-N688DS}",
            "null",
            "{ICOADS-302-N688DV}",
            "null",
            "{ICOADS-302-N688ED}",
            "{ICOADS-302-N688EE,ICOADS-302-N688EC,ICOADS-302-N688EG,ICOADS-302-N688EH}",
            "{ICOADS-302-N688ED}",
            "{ICOADS-302-N688ED}",
        ],
    )


def test_duplicates_remove():
    DupDetect.remove_duplicates()
    expected = DupDetect.data.iloc[[0, 1, 2, 3, 4, 6, 8, 10, 12]].reset_index(drop=True)
    assert_frame_equal(expected, DupDetect.result)


def test_duplicates_craid():
    df = _get_test_data("craid")
    DupDetect = duplicate_check(df, ignore_columns="primary_station_id")
    DupDetect.flag_duplicates()
    assert_array_equal(DupDetect.result["duplicate_status"], [0] * 10)
    assert_array_equal(DupDetect.result["report_quality"], [2] * 10)
    assert_array_equal(DupDetect.result["duplicates"], ["null"] * 10)
