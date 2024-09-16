from __future__ import annotations

import pytest  # noqa

from cdm_reader_mapper.cdm_mapper import duplicate_check, read_tables

from ._results import result_data


def _manipulate_header(df):
    df.loc[5] = df.loc[4]
    df.loc[5, "report_id"] = "ICOADS-302-N688EY"  # duplicate
    df.loc[6] = df.loc[4]
    df.loc[6, "latitude"] = -65.80  # no duplicate
    df.loc[6, "longitude"] = 21.20
    df.loc[7] = df.loc[1]
    df.loc[7, "report_timestamp"] = "2022-02-01 00:01:00"  # duplicate
    df.loc[8] = df.loc[1]
    df.loc[8, "report_timestamp"] = "2022-02-02 00:00:00"  # no duplicate
    df.loc[9] = df.loc[2]
    df.loc[9, "report_id"] = "ICOADS-302-N688DW"  # duplicate
    df.loc[10] = df.loc[3]
    df.loc[10, "latitude"] = 8.50  # no duplicate
    df.loc[10, "longitude"] = 66.00
    df.loc[11] = df.loc[3]
    df.loc[11, "latitude"] = 8.15  # duplicate
    df.loc[11, "longitude"] = 66.05
    df.loc[12] = df.loc[3]
    df.loc[12, "primary_station_id"] = "MASKSTIP"  # no duplicate
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
    return DupDetect.remove_duplicates()
