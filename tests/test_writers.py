from __future__ import annotations

import pandas as pd
import pytest  # noqa

from cdm_reader_mapper.cdm_mapper import read_tables, write_tables

from ._results import result_data

imodel = "icoads_r300_d714"
exp = f"expected_{imodel}"
expected_data = getattr(result_data, exp)
output = read_tables(
    expected_data["cdm_table"],
    suffix=f"{imodel}*",
)


def test_write_data():
    write_tables(output, suffix=f"{imodel}_all")
    output_ = read_tables(".", suffix=f"{imodel}_all")
    pd.testing.assert_frame_equal(output, output_)


@pytest.mark.parametrize("table", ["header", "observations-sst"])
def test_write_tables(table):
    write_tables(output, suffix=f"{imodel}_{table}_all", cdm_subset=table)
    output_table = read_tables(".", suffix=f"{imodel}_{table}_all")
    output_origi = output[table].dropna(how="all").reset_index(drop=True)
    pd.testing.assert_frame_equal(output_origi, output_table[table])


def test_write_fns():
    write_tables(
        output, prefix="prefix", suffix=f"{imodel}_all", extension="csv", delimiter=","
    )
    output_ = read_tables(
        ".", prefix="prefix", suffix=f"{imodel}_all", extension="csv", delimiter=","
    )
    pd.testing.assert_frame_equal(output, output_)


def test_write_filename():
    write_tables(output, filename=f"{imodel}_filename_all")
    output_ = read_tables(".", suffix=f"{imodel}_filename_all")
    pd.testing.assert_frame_equal(output, output_)


def test_write_filename_dict():
    filename_dict = {
        "header": f"{imodel}_filename_dict_all",
        "observations-sst": f"observations-sst-{imodel}_filename_dict_all.psv",
    }
    write_tables(output, filename=filename_dict)
    output_ = read_tables(".", suffix=f"{imodel}_filename_dict_all")
    pd.testing.assert_frame_equal(output[filename_dict.keys()], output_)


def test_write_col_subset():
    table = "header"
    columns = ["report_id", "latitude", "longitude"]
    write_tables(
        output,
        suffix=f"{imodel}_{table}_all",
        cdm_subset=table,
        col_subset={table: columns},
    )
    output_table = read_tables(".", suffix=f"{imodel}_{table}_all")
    output_origi = output[table][columns].dropna(how="all").reset_index(drop=True)
    pd.testing.assert_frame_equal(output_origi, output_table[table])
