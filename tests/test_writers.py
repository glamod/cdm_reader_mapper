from __future__ import annotations

import pandas as pd
import pytest  # noqa

from cdm_reader_mapper import read_tables

from ._results import result_data

imodel = "icoads_r300_d714"
exp = f"expected_{imodel}"
expected_data = getattr(result_data, exp)
output = read_tables(
    expected_data["cdm_table"],
    suffix=f"{imodel}*",
)


def test_write_data():
    output.write_tables(suffix=f"{imodel}_all")
    output_ = read_tables(".", suffix=f"{imodel}_all")
    pd.testing.assert_frame_equal(output.tables, output_.tables)


@pytest.mark.parametrize("table", ["header", "observations-sst"])
def test_write_tables(table):
    output.write_tables(suffix=f"{imodel}_{table}_all", cdm_subset=table)
    output_table = read_tables(".", suffix=f"{imodel}_{table}_all", cdm_subset=table)
    output_origi = output.tables[table].dropna(how="all").reset_index(drop=True)
    pd.testing.assert_frame_equal(output_origi, output_table.tables[table])


def test_write_fns():
    output.write_tables(
        prefix="prefix", suffix=f"{imodel}_all", extension="csv", delimiter=","
    )
    output_ = read_tables(
        ".", prefix="prefix", suffix=f"{imodel}_all", extension="csv", delimiter=","
    )
    pd.testing.assert_frame_equal(output.tables, output_.tables)


def test_write_filename():
    output.write_tables(filename=f"{imodel}_filename_all")
    output_ = read_tables(".", suffix=f"{imodel}_filename_all")
    pd.testing.assert_frame_equal(output.tables, output_.tables)


def test_write_filename_dict():
    filename_dict = {
        "header": f"{imodel}_filename_dict_all",
        "observations-sst": f"observations-sst-{imodel}_filename_dict_all.psv",
    }
    output.write_tables(filename=filename_dict)
    output_ = read_tables(".", suffix=f"{imodel}_filename_dict_all")
    pd.testing.assert_frame_equal(output.tables[filename_dict.keys()], output_.tables)


def test_write_col_subset():
    table = "header"
    columns = ["report_id", "latitude", "longitude"]
    output.write_tables(
        suffix=f"{imodel}_{table}_all",
        cdm_subset=table,
        col_subset={table: columns},
    )
    output_table = read_tables(".", suffix=f"{imodel}_{table}_all")
    output_origi = (
        output.tables[table][columns].dropna(how="all").reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(output_origi, output_table.tables[table])
