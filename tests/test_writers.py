from __future__ import annotations

import pandas as pd
import pytest  # noqa

from cdm_reader_mapper import read

from ._results import result_data

imodel = "icoads_r300_d714"
exp = f"expected_{imodel}"
expected_data = getattr(result_data, exp)
db_exp = read(dict(expected_data)["cdm_table"], suffix=f"{imodel}*", mode="tables")


def test_write_data():
    db_exp.write(suffix=f"{imodel}_all")
    db_res = read(".", suffix=f"{imodel}_all", mode="tables")
    pd.testing.assert_frame_equal(db_exp.data, db_res.data)


@pytest.mark.parametrize("table", ["header", "observations-sst"])
def test_write_tables(table):
    db_exp.write(suffix=f"{imodel}_{table}_all", cdm_subset=table)
    db_res = read(".", suffix=f"{imodel}_{table}_all", cdm_subset=table, mode="tables")
    table_exp = db_exp[table].dropna(how="all").reset_index(drop=True)
    pd.testing.assert_frame_equal(table_exp, db_res[table])


def test_write_fns():
    db_exp.write(
        prefix="prefix", suffix=f"{imodel}_all", extension="csv", delimiter=","
    )
    db_res = read(
        ".",
        prefix="prefix",
        suffix=f"{imodel}_all",
        extension="csv",
        delimiter=",",
        mode="tables",
    )
    pd.testing.assert_frame_equal(db_exp.data, db_res.data)


def test_write_filename():
    db_exp.write(filename=f"{imodel}_filename_all")
    db_res = read(".", suffix=f"{imodel}_filename_all", mode="tables")
    pd.testing.assert_frame_equal(db_exp.data, db_res.data)


def test_write_filename_dict():
    filename_dict = {
        "header": f"{imodel}_filename_dict_all",
        "observations-sst": f"observations-sst-{imodel}_filename_dict_all.psv",
    }
    db_exp.write(filename=filename_dict)
    db_res = read(".", suffix=f"{imodel}_filename_dict_all", mode="tables")
    pd.testing.assert_frame_equal(db_exp[filename_dict.keys()], db_res.data)


def test_write_col_subset():
    table = "header"
    columns = ["report_id", "latitude", "longitude"]
    db_exp.write(
        suffix=f"{imodel}_{table}_all",
        cdm_subset=table,
        col_subset={table: columns},
    )
    db_res = read(".", suffix=f"{imodel}_{table}_all", mode="tables")
    table_exp = db_exp[table][columns].dropna(how="all").reset_index(drop=True)
    pd.testing.assert_frame_equal(table_exp, db_res[table])
