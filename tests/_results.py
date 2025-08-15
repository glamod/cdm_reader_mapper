"""cdm_reader_mapper testing suite result files."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from urllib.error import HTTPError

import pandas as pd

from cdm_reader_mapper import read
from cdm_reader_mapper.common import load_file

cdm_tables = [
    "header-{}.psv",
    "observations-at-{}.psv",
    "observations-dpt-{}.psv",
    "observations-slp-{}.psv",
    "observations-sst-{}.psv",
    "observations-wbt-{}.psv",
    "observations-wd-{}.psv",
    "observations-ws-{}.psv",
]

_base = Path(__file__).parent


class result_data:
    """Expected results for cdm_reader_mapper testing suite"""

    def __init__(self):
        self.name = "CDM reader mapper result testing data."

    @property
    def expected_icoads_r300_d714(self):
        return self._get_data_dict(
            "2010-07-01_subset",
            "icoads_r300_d714",
        )

    @property
    def expected_icoads_r300_d701(self):
        return self._get_data_dict(
            "1845-04-01_subset",
            "icoads_r300_d701",
        )

    @property
    def expected_icoads_r300_d705(self):
        return self._get_data_dict(
            "1938-04-01_subset",
            "icoads_r300_d705",
        )

    @property
    def expected_icoads_r300_d781(self):
        return self._get_data_dict(
            "1987-09-01_subset",
            "icoads_r300_d781",
        )

    @property
    def expected_icoads_r300_d706(self):
        return self._get_data_dict(
            "1919-03-01_subset",
            "icoads_r300_d706",
        )

    @property
    def expected_icoads_r300_d702(self):
        return self._get_data_dict(
            "1873-01-01_subset",
            "icoads_r300_d702",
        )

    @property
    def expected_icoads_r300_d707(self):
        return self._get_data_dict(
            "1916-04-01_subset",
            "icoads_r300_d707",
        )

    @property
    def expected_icoads_r302_d794(self):
        return self._get_data_dict(
            "2022-11-01_subset",
            "icoads_r302_d794",
        )

    @property
    def expected_icoads_r300_d704(self):
        return self._get_data_dict(
            "1878-10-01_subset",
            "icoads_r300_d704",
        )

    @property
    def expected_icoads_r300_d721(self):
        return self._get_data_dict(
            "1862-06-01_subset",
            "icoads_r300_d721",
        )

    @property
    def expected_icoads_r300_d730(self):
        return self._get_data_dict(
            "1776-10-01_subset",
            "icoads_r300_d730",
        )

    @property
    def expected_icoads_r300_d703(self):
        return self._get_data_dict(
            "1979-09-01_subset",
            "icoads_r300_d703",
        )

    @property
    def expected_icoads_r300_d201(self):
        return self._get_data_dict(
            "1913-11-01_subset",
            "icoads_r300_d201",
        )

    @property
    def expected_icoads_r300_d892(self):
        return self._get_data_dict(
            "1996-02-01_subset",
            "icoads_r300_d892",
        )

    @property
    def expected_icoads_r300_d700(self):
        return self._get_data_dict(
            "2002-08-01_subset",
            "icoads_r300_d700",
        )

    @property
    def expected_icoads_r300_mixed(self):
        return self._get_data_dict(
            "1899-01-02_subset",
            "icoads_r300_mixed",
        )

    @property
    def expected_icoads_r302_d792(self):
        return self._get_data_dict(
            "2022-02-01_subset",
            "icoads_r302_d792",
        )

    @property
    def expected_icoads_r302_d992(self):
        return self._get_data_dict(
            "2022-01-01_subset",
            "icoads_r302_d992",
        )

    @property
    def expected_gdac(self):
        return self._get_data_dict(
            "2003-02-01_subset",
            "gdac",
        )

    @property
    def expected_craid(self):
        return self._get_data_dict(
            "2004-12-20_subset",
            "craid",
        )

    def __getitem__(self, attr):
        return getattr(self, attr)

    def _load_file(self, ifile):
        try:
            return load_file(ifile)
        except HTTPError:
            return pd.DataFrame()

    def _get_data_dict(self, data_file, data_model):
        drs = data_model.split("_")

        data_file = f"{data_model}_{data_file}"
        data = f"data_{data_file}.csv"
        info = f"info_{data_file}.json"
        mask = f"mask_{data_file}.csv"
        vaid = f"vaid_{data_file}.csv"
        vadt = f"vadt_{data_file}.csv"

        for cdm_table in cdm_tables:
            name = cdm_table.format(data_file)
            path = load_file(os.path.join(*drs, "cdm_tables", name)).parent

        try:
            info = load_file(os.path.join(*drs, "output", info))
        except OSError as exception:
            raise exception
        except Exception as exception:
            logging.warning(exception)
            info = None

        return {
            "data": self._load_file(os.path.join(*drs, "output", data)),
            "mask": self._load_file(os.path.join(*drs, "output", mask)),
            "info": info,
            "cdm_table": path,
            "vaid": self._load_file(os.path.join(*drs, "validation", vaid)),
            "vadt": self._load_file(os.path.join(*drs, "validation", vadt)),
        }


result_data = result_data()

cdm_header = read(
    result_data.expected_icoads_r302_d792["cdm_table"],
    cdm_subset=["header"],
    mode="tables",
)
correction_file = list((_base / "corrections").glob("2022-02.txt.gz"))[0]
correction_df = pd.read_csv(
    correction_file,
    delimiter="|",
    names=["report_id", "primary_station_id", "primary_station_id.isChange"],
)
