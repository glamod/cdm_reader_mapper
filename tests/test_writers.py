from __future__ import annotations

import pandas as pd
import pytest  # noqa

from cdm_reader_mapper import read_tables

from ._results import result_data

imodel = "icoads_r300_d714"
exp = f"expected_{imodel}"
expected_data = getattr(result_data, exp)
db_exp = read_tables(
    dict(expected_data)["cdm_table"],
    suffix=f"{imodel}*",
)


def test_write_data():
    db_exp.write_data(suffix=f"{imodel}_all")
    db_res = read_tables(".", suffix=f"{imodel}_all")
    pd.testing.assert_frame_equal(db_exp.data, db_res.data)


@pytest.mark.parametrize("table", ["header", "observations-sst"])
def test_write_tables(table):
    db_exp.write_data(suffix=f"{imodel}_{table}_all", cdm_subset=table)
    db_res = read_tables(".", suffix=f"{imodel}_{table}_all", cdm_subset=table)
    tables_exp = db_exp.data.copy()
    table_exp = tables_exp[table].dropna(how="all").reset_index(drop=True)
    tables_res = db_res.data.copy()
    pd.testing.assert_frame_equal(table_exp, tables_res[table])


def test_write_fns():
    db_exp.write_data(
        prefix="prefix", suffix=f"{imodel}_all", extension="csv", delimiter=","
    )
    db_res = read_tables(
        ".", prefix="prefix", suffix=f"{imodel}_all", extension="csv", delimiter=","
    )
    pd.testing.assert_frame_equal(db_exp.data, db_res.data)


def test_write_filename():
    db_exp.write_data(filename=f"{imodel}_filename_all")
    db_res = read_tables(".", suffix=f"{imodel}_filename_all")
    pd.testing.assert_frame_equal(db_exp.data, db_res.data)


def test_write_filename_dict():
    filename_dict = {
        "header": f"{imodel}_filename_dict_all",
        "observations-sst": f"observations-sst-{imodel}_filename_dict_all.psv",
    }
    db_exp.write_data(filename=filename_dict)
    db_res = read_tables(".", suffix=f"{imodel}_filename_dict_all")
    tables_exp = db_exp.data.copy()
    pd.testing.assert_frame_equal(tables_exp[filename_dict.keys()], db_res.data)


def test_write_col_subset():
    table = "header"
    columns = ["report_id", "latitude", "longitude"]
    db_exp.write_data(
        suffix=f"{imodel}_{table}_all",
        cdm_subset=table,
        col_subset={table: columns},
    )
    db_res = read_tables(".", suffix=f"{imodel}_{table}_all")
    tables_exp = db_exp.data.copy()
    table_exp = tables_exp[table][columns].dropna(how="all").reset_index(drop=True)
    tables_res = db_res.data.copy()
    pd.testing.assert_frame_equal(table_exp, tables_res[table])
