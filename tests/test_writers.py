from __future__ import annotations

import os

import pandas as pd
import pytest  # noqa

from cdm_reader_mapper import read, test_data
from cdm_reader_mapper.cdm_mapper.properties import cdm_tables


@pytest.fixture(scope="session")
def db_tables():
    imodel = "icoads_r300_d714"
    pattern = f"test_{imodel}"
    for table in cdm_tables:
        cdm_path = test_data[pattern][f"cdm_{table}"].parent

    db = read(cdm_path, suffix=f"{imodel}*", extension="psv", mode="tables")
    db.imodel = imodel
    return db


@pytest.fixture(scope="session")
def db_data():
    imodel = "icoads_r300_d714"
    pattern = f"test_{imodel}"

    data_file = test_data[pattern]["mdf_data"]
    info_file = test_data[pattern]["mdf_info"]

    db = read(data_file, info_file=info_file, mode="data")
    db.imodel = imodel
    return db


def test_write_data_csv(tmp_path, db_data):
    db_data.write(out_dir=tmp_path, data_format="csv")
    db_res = read(
        os.path.join(tmp_path, "data.csv"),
        info_file=os.path.join(tmp_path, "info.json"),
        data_format="csv",
        mode="data",
    )
    pd.testing.assert_frame_equal(db_data.data, db_res.data)


def test_write_tables_csv(tmp_path, db_tables):
    db_tables.write(out_dir=tmp_path, suffix=f"{db_tables.imodel}_all")
    db_res = read(tmp_path, suffix=f"{db_tables.imodel}_all", mode="tables")
    pd.testing.assert_frame_equal(db_tables.data, db_res.data)


def test_write_header(tmp_path, db_tables):
    table = "header"
    db_tables.write(
        out_dir=tmp_path,
        suffix=f"{db_tables.imodel}_{table}_all",
        extension="psv",
        cdm_subset=table,
    )
    db_res = read(
        tmp_path,
        suffix=f"{db_tables.imodel}_{table}_all",
        cdm_subset=table,
        extension="psv",
        mode="tables",
    )

    table_exp = db_tables[table].dropna(how="all").reset_index(drop=True)
    pd.testing.assert_frame_equal(table_exp, db_res[table])


def test_write_observations(tmp_path, db_tables):
    table = "observations-sst"
    db_tables.write(
        out_dir=tmp_path, suffix=f"{db_tables.imodel}_{table}_all", cdm_subset=table
    )
    db_res = read(
        tmp_path,
        suffix=f"{db_tables.imodel}_{table}_all",
        cdm_subset=table,
        mode="tables",
    )
    table_exp = db_tables[table].dropna(how="all").reset_index(drop=True)
    pd.testing.assert_frame_equal(table_exp, db_res[table])


def test_write_fns(tmp_path, db_tables):
    db_tables.write(
        out_dir=tmp_path,
        prefix="prefix",
        suffix=f"{db_tables.imodel}_all",
        extension="csv",
        delimiter=",",
    )
    db_res = read(
        tmp_path,
        prefix="prefix",
        suffix=f"{db_tables.imodel}_all",
        extension="csv",
        delimiter=",",
        mode="tables",
    )
    pd.testing.assert_frame_equal(db_tables.data, db_res.data)


def test_write_filename(tmp_path, db_tables):
    db_tables.write(out_dir=tmp_path, filename=f"{db_tables.imodel}_filename_all")
    db_res = read(tmp_path, suffix=f"{db_tables.imodel}_filename_all", mode="tables")
    pd.testing.assert_frame_equal(db_tables.data, db_res.data)


def test_write_filename_dict_header(tmp_path, db_tables):
    filename_dict = {
        "header": f"{db_tables.imodel}_filename_dict_all",
    }
    db_tables.write(out_dir=tmp_path, filename=filename_dict)
    db_res = read(
        tmp_path, suffix=f"{db_tables.imodel}_filename_dict_all", mode="tables"
    )
    table_exp = db_tables["header"].dropna(how="all").reset_index(drop=True)
    pd.testing.assert_frame_equal(table_exp, db_res.data["header"])


def test_write_filename_dict_observations(tmp_path, db_tables):
    filename_dict = {
        "observations-sst": f"observations-sst-{db_tables.imodel}_filename_dict_all.psv",
    }
    db_tables.write(out_dir=tmp_path, filename=filename_dict, extension="psv")
    db_res = read(
        tmp_path,
        suffix=f"{db_tables.imodel}_filename_dict_all",
        mode="tables",
        extension="psv",
    )
    table_exp = db_tables["observations-sst"].dropna(how="all").reset_index(drop=True)
    pd.testing.assert_frame_equal(table_exp, db_res.data["observations-sst"])


def test_write_col_subset(tmp_path, db_tables):
    table = "header"
    columns = ["report_id", "latitude", "longitude"]
    db_tables.write(
        out_dir=tmp_path,
        suffix=f"{db_tables.imodel}_{table}_all",
        cdm_subset=table,
        col_subset={table: columns},
    )
    db_res = read(tmp_path, suffix=f"{db_tables.imodel}_{table}_all", mode="tables")
    table_exp = db_tables[table][columns].dropna(how="all").reset_index(drop=True)
    pd.testing.assert_frame_equal(table_exp, db_res[table])


def test_write_data_parquet(tmp_path, db_data):
    db_data.write(out_dir=tmp_path, data_format="parquet")
    db_res = read(
        os.path.join(tmp_path, "data.parquet"), data_format="parquet", mode="data"
    )
    pd.testing.assert_frame_equal(db_data.data, db_res.data)


def test_write_data_feather(tmp_path, db_data):
    db_data.write(out_dir=tmp_path, data_format="feather")
    db_res = read(
        os.path.join(tmp_path, "data.feather"), data_format="feather", mode="data"
    )
    pd.testing.assert_frame_equal(db_data.data, db_res.data)


def test_write_tables_parquet(tmp_path, db_tables):
    db_tables.write(
        out_dir=tmp_path, suffix=f"{db_tables.imodel}_all", data_format="parquet"
    )
    db_res = read(
        tmp_path, suffix=f"{db_tables.imodel}_all", mode="tables", data_format="parquet"
    )
    pd.testing.assert_frame_equal(db_tables.data, db_res.data)


def test_write_tables_feather(tmp_path, db_tables):
    db_tables.write(
        out_dir=tmp_path, suffix=f"{db_tables.imodel}_all", data_format="feather"
    )
    db_res = read(
        tmp_path, suffix=f"{db_tables.imodel}_all", mode="tables", data_format="feather"
    )
    pd.testing.assert_frame_equal(db_tables.data, db_res.data)
