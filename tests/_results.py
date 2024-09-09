"""cdm_reader_mapper testing suite result files."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.error import HTTPError

import pandas as pd

from cdm_reader_mapper.cdm_mapper import read_tables
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

for correction_file in (_base / "results").glob("2022-02.txt.gz"):
    break

correction_df = pd.read_csv(
    correction_file,
    delimiter="|",
    dtype="object",
    header=None,
    usecols=[0, 1, 2],
    names=["report_id", "primary_station_id", "primary_station_id.isChange"],
    quotechar=None,
    quoting=3,
)

table_df = read_tables((_base / "results"), "test", cdm_subset=["header"])
table_df.set_index("report_id", inplace=True, drop=False)


class result_data:
    """Expected results for cdm_reader_mapper testing suite"""

    def __init__(self):
        self.name = "CDM reader mapper result testing data."

    @property
    def expected_063_714(self):
        return self._get_data_dict(
            "063-714_2010-07-01_subset",
            "714",
            "imma1",
        )

    @property
    def expected_069_701_type1(self):
        return self._get_data_dict(
            "069-701_type1_1845-04-01_subset",
            "701",
            "imma1",
        )

    @property
    def expected_069_701_type2(self):
        return self._get_data_dict(
            "069-701_type2_1845-04-01_subset",
            "701",
            "imma1",
        )

    @property
    def expected_085_705(self):
        return self._get_data_dict(
            "085-705_1938-04-01_subset",
            "705",
            "imma1",
        )

    @property
    def expected_143_781(self):
        return self._get_data_dict(
            "143-781_1987-09-01_subset",
            "781",
            "imma1",
        )

    @property
    def expected_084_706(self):
        return self._get_data_dict(
            "084-706_1919-03-01_subset",
            "706",
            "imma1",
        )

    @property
    def expected_096_702(self):
        return self._get_data_dict(
            "096-702_1873-01-01_subset",
            "702",
            "imma1",
        )

    @property
    def expected_098_707(self):
        return self._get_data_dict(
            "098-707_1916-04-01_subset",
            "707",
            "imma1",
        )

    @property
    def expected_103_794(self):
        return self._get_data_dict(
            "103-794_2022-11-01_subset",
            "794",
            "imma1",
        )

    @property
    def expected_125_704(self):
        return self._get_data_dict(
            "125-704_1878-10-01_subset",
            "704",
            "imma1",
        )

    @property
    def expected_125_721(self):
        return self._get_data_dict(
            "125-721_1862-06-01_subset",
            "721",
            "imma1",
        )

    @property
    def expected_133_730(self):
        return self._get_data_dict(
            "133-730_1776-10-01_subset",
            "730",
            "imma1",
        )

    @property
    def expected_144_703(self):
        return self._get_data_dict(
            "144-703_1979-09-01_subset",
            "703",
            "imma1",
        )

    @property
    def expected_091_201(self):
        return self._get_data_dict(
            "091-201_1913-11-01_subset",
            "201",
            "imma1",
        )

    @property
    def expected_077_892(self):
        return self._get_data_dict(
            "077-892_1996-02-01_subset",
            "892",
            "imma1",
        )

    @property
    def expected_147_700(self):
        return self._get_data_dict(
            "147-700_2002-08-01_subset",
            "700",
            "imma1",
        )

    @property
    def expected_103_792(self):
        return self._get_data_dict(
            "103-792_2022-02-01_subset",
            "792",
            "imma1",
        )

    @property
    def expected_114_992(self):
        return self._get_data_dict(
            "114-992_2022-01-01_subset",
            "992",
            "imma1",
        )

    @property
    def expected_mix_out(self):
        return self._get_data_dict(
            "mix-out_20030201",
            "gcc",
            "immt",
        )

    @property
    def expected_craid(self):
        return self._get_data_dict(
            "1260810_2004-12-20_subset",
            "raid",
            "c",
        )

    def __getitem__(self, attr):
        return getattr(self, attr)

    def _load_file(self, ifile):
        try:
            return load_file(ifile)
        except HTTPError:
            return pd.DataFrame()

    def _get_data_dict(self, data_file, deck, dm):
        drs = f"{dm}_{deck}"

        data = f"data_{data_file}.csv"
        mask = f"mask_{data_file}.csv"
        vaid = f"vaid_{data_file}.csv"
        vadt = f"vadt_{data_file}.csv"

        for cdm_table in cdm_tables:
            name = cdm_table.format(data_file)
            path = load_file(os.path.join(drs, "cdm_tables", name)).parent

        return {
            "data": self._load_file(os.path.join(drs, "output", data)),
            "mask": self._load_file(os.path.join(drs, "output", mask)),
            "cdm_table": path,
            "vaid": self._load_file(os.path.join(drs, "validation", vaid)),
            "vadt": self._load_file(os.path.join(drs, "validation", vadt)),
        }


result_data = result_data()
