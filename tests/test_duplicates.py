from __future__ import annotations

import numpy as np
import pytest  # noqa

from cdm_reader_mapper.cdm_mapper import duplicate_check, read_tables

from ._results import result_data


def _manipulate_header(df):
    # Duplicate : Different report_id's
    # Failure in data set;
    # each report needs a specific report_id
    df.loc[5] = df.loc[4]
    df.loc[5, "report_id"] = "ICOADS-302-N688EY"

    # No Duplicate: Lat and Lon values differ to much
    # valid is .5 degrees
    df.loc[6] = df.loc[4]  # no duplicate
    df.loc[5, "report_id"] = "ICOADS-302-N688EY"
    df.loc[6, "latitude"] = -65.80
    df.loc[6, "longitude"] = 21.20

    # Duplicate: report timestamp differs no enough
    # valid is 60 seconds
    df.loc[7] = df.loc[1]  # duplicate
    df.loc[7, "report_id"] = "ICOADS-302-N688DT"
    df.loc[7, "report_timestamp"] = "2022-02-01 00:01:00"

    # No Duplicate: report timestamp differs to much
    # valid is 60 seconds
    df.loc[8] = df.loc[1]  # no duplicate
    df.loc[8, "report_id"] = "ICOADS-302-N688DU"
    df.loc[8, "report_timestamp"] = "2022-02-02 00:00:00"

    # Duplicate : Different report_id's
    # Failure in data set
    df.loc[9] = df.loc[2]  # duplicate
    df.loc[9, "report_id"] = "ICOADS-302-N688DW"

    # Duplicate : Different report_id's
    # Failure in data set
    # each report needs a specific report_id
    df.loc[10] = df.loc[3]  # no duplicate
    df.loc[10, "report_id"] = "ICOADS-302-N688EF"
    df.loc[10, "latitude"] = 66.00
    df.loc[10, "longitude"] = 8.50

    # Duplicate: Lat and Lon values differ not enough
    # valid is .5 degrees
    df.loc[11] = df.loc[3]  # duplicate
    df.loc[11, "report_id"] = "ICOADS-302-N688EE"
    df.loc[11, "latitude"] = 66.05
    df.loc[11, "longitude"] = 8.15

    # No Duplicate: primary_station_id differs
    df.loc[12] = df.loc[3]
    df.loc[12, "report_id"] = "ICOADS-302-N688ED"
    df.loc[12, "primary_station_id"] = "MASKSTIP"

    # Duplicate: Lat and Lon values differ not enough
    # valid is .5 degrees
    df.loc[13] = df.loc[3]  # duplicate
    df.loc[13, "report_id"] = "ICOADS-302-N688EC"
    df.loc[13, "latitude"] = 65.95
    df.loc[13, "longitude"] = 8.05
    return df


def test_duplicates_header():
    expected_data = result_data.expected_103_792
    data_path = expected_data.get("cdm_table")
    df = read_tables(
        data_path,
        tb_id="103-792*",
        cdm_subset="header",
    )
    df = _manipulate_header(df)
    DupDetect = duplicate_check(df)
    DupDetect.flag_duplicates()
    np.testing.assert_array_equal(
        DupDetect.result["duplicate_status"], [0, 1, 1, 1, 1, 3, 0, 3, 0, 3, 0, 3, 0, 3]
    )
    np.testing.assert_array_equal(
        DupDetect.result["duplicates"],
        [
            "null",
            "ICOADS-302-N688DT",
            "ICOADS-302-N688DW",
            "ICOADS-302-N688EE, ICOADS-302-N688EC",
            "ICOADS-302-N688EY",
            "ICOADS-302-N688EI",
            "null",
            "ICOADS-302-N688DS",
            "null",
            "ICOADS-302-N688DV",
            "null",
            "ICOADS-302-N688EH",
            "null",
            "ICOADS-302-N688EH",
        ],
    )
