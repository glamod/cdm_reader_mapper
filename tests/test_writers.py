from __future__ import annotations

import pandas as pd
import pytest  # noqa

from cdm_reader_mapper import read, test_data
from cdm_reader_mapper.cdm_mapper.properties import cdm_tables


@pytest.fixture(scope="session")
def db_exp():
    imodel = "icoads_r300_d714"
    pattern = f"test_{imodel}"
    for table in cdm_tables:
        cdm_path = test_data[pattern][f"cdm_{table}"].parent

    db = read(cdm_path, suffix=f"{imodel}*", extension="psv", mode="tables")
    db.imodel = imodel
    print(db)
    return db


def test_write_data(tmp_path, db_exp):
    db_exp.write(out_dir=tmp_path, suffix=f"{db_exp.imodel}_all")
    db_res = read(tmp_path, suffix=f"{db_exp.imodel}_all", mode="tables")
    pd.testing.assert_frame_equal(db_exp.data, db_res.data)


def test_write_header(tmp_path, db_exp):
    table = "header"
    db_exp.write(
        out_dir=tmp_path,
        suffix=f"{db_exp.imodel}_{table}_all",
        extension="psv",
        cdm_subset=table,
    )
    db_res = read(
        tmp_path,
        suffix=f"{db_exp.imodel}_{table}_all",
        cdm_subset=table,
        extension="psv",
        mode="tables",
    )

    table_exp = db_exp[table].dropna(how="all").reset_index(drop=True)
    pd.testing.assert_frame_equal(table_exp, db_res[table])


def test_write_observations(tmp_path, db_exp):
    table = "observations-sst"
    db_exp.write(
        out_dir=tmp_path, suffix=f"{db_exp.imodel}_{table}_all", cdm_subset=table
    )
    db_res = read(
        tmp_path, suffix=f"{db_exp.imodel}_{table}_all", cdm_subset=table, mode="tables"
    )
    table_exp = db_exp[table].dropna(how="all").reset_index(drop=True)
    pd.testing.assert_frame_equal(table_exp, db_res[table])


def test_write_fns(tmp_path, db_exp):
    db_exp.write(
        out_dir=tmp_path,
        prefix="prefix",
        suffix=f"{db_exp.imodel}_all",
        extension="csv",
        delimiter=",",
    )
    db_res = read(
        tmp_path,
        prefix="prefix",
        suffix=f"{db_exp.imodel}_all",
        extension="csv",
        delimiter=",",
        mode="tables",
    )
    pd.testing.assert_frame_equal(db_exp.data, db_res.data)


def test_write_filename(tmp_path, db_exp):
    db_exp.write(out_dir=tmp_path, filename=f"{db_exp.imodel}_filename_all")
    db_res = read(tmp_path, suffix=f"{db_exp.imodel}_filename_all", mode="tables")
    pd.testing.assert_frame_equal(db_exp.data, db_res.data)


def test_write_filename_dict_header(tmp_path, db_exp):
    filename_dict = {
        "header": f"{db_exp.imodel}_filename_dict_all",
    }
    db_exp.write(out_dir=tmp_path, filename=filename_dict)
    db_res = read(tmp_path, suffix=f"{db_exp.imodel}_filename_dict_all", mode="tables")
    table_exp = db_exp["header"].dropna(how="all").reset_index(drop=True)
    pd.testing.assert_frame_equal(table_exp, db_res.data["header"])


def test_write_filename_dict_observations(tmp_path, db_exp):
    filename_dict = {
        "observations-sst": f"observations-sst-{db_exp.imodel}_filename_dict_all.psv",
    }
    db_exp.write(out_dir=tmp_path, filename=filename_dict, extension="psv")
    db_res = read(
        tmp_path,
        suffix=f"{db_exp.imodel}_filename_dict_all",
        mode="tables",
        extension="psv",
    )
    table_exp = db_exp["observations-sst"].dropna(how="all").reset_index(drop=True)
    pd.testing.assert_frame_equal(table_exp, db_res.data["observations-sst"])


def test_write_col_subset(tmp_path, db_exp):
    table = "header"
    columns = ["report_id", "latitude", "longitude"]
    db_exp.write(
        out_dir=tmp_path,
        suffix=f"{db_exp.imodel}_{table}_all",
        cdm_subset=table,
        col_subset={table: columns},
    )
    db_res = read(tmp_path, suffix=f"{db_exp.imodel}_{table}_all", mode="tables")
    table_exp = db_exp[table][columns].dropna(how="all").reset_index(drop=True)
    pd.testing.assert_frame_equal(table_exp, db_res[table])
