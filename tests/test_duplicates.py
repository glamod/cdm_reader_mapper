from __future__ import annotations

import pytest  # noqa

from cdm_reader_mapper.cdm_mapper import duplicate_check
from cdm_reader_mapper.cdm_mapper import read_tables

from ._results import result_data


def _manipulate_header(df):
    df.loc[5] = df.loc[4]
    df.loc[5, "report_id"] = "ICOADS-302-N688EY"
    df.loc[5, "source_record_id"] = "N688EY"
    df.loc[6] = df.loc[4]
    df.loc[6, "latitude"] = -65.80
    df.loc[6, "longitude"] = 21.20
    df.loc[7] = df.loc[0]
    df.loc[7, "source_record_id"] = "N688DP"
    df.loc[8] = df.loc[1]
    df.loc[8, "report_timestamp"] = "2022-02-02 00:00:00"
    df.loc[9] = df.loc[2]
    df.loc[9, "report_id"] = "ICOADS-302-N688DW"
    df.loc[10] = df.loc[3]
    df.loc[10, "latitude"] = 9.10
    df.loc[10, "longitude"] = 68.00
    return df


def _manipulate_observations(df):
    df.loc[3] = df.loc[0]
    df.loc[3, "observation_id"] = "ICOADS-302-N688DP-SST"
    df.loc[3, "source_id"] = "ICOADS-3-0-2T-103-792-2022-3"
    df.loc[4] = df.loc[0]
    df.loc[4, "latitude"] = -71.30
    df.loc[5] = df.loc[1]
    df.loc[5, "source_id"] = "ICOADS-3-0-2T-103-792-2022-3"
    df.loc[6] = df.loc[1]
    df.loc[6, "date_time"] = "2022-02-02 00:00:00"
    df.loc[7] = df.loc[2]
    df.loc[7, "observation_id"] = "ICOADS-302-N688DW-SST"
    df.loc[8] = df.loc[2]
    df.loc[8, "latitude"] = 73.20
    df.loc[8, "longitude"] = 34.00
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


def test_duplicates_observations():
    expected_data = result_data.expected_103_792
    data_path = expected_data.get("cdm_table")
    df = read_tables(
        data_path,
        tb_id="103-792*",
        cdm_subset="observations-sst",
    )
    df = _manipulate_observations(df)
    DupDetect = duplicate_check(df, cdm_name="observations")
    return DupDetect.remove_duplicates()    