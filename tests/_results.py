"""cdm_reader_mapper testing suite result files."""
from __future__ import annotations

from glob import glob
from pathlib import Path

import pandas as pd
import pytest

from cdm_reader_mapper.cdm_mapper import read_tables
from cdm_reader_mapper.common.local import get_files

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
    "Expected results for cdm_reader_mapper testing suite"

    def __init__(self):
        self._data_path = _base / "results"

        self.expected_063_714_nosupp = self._get_data_dict("063-714_nosupp")
        self.expected_069_701_type1_nosupp = self._get_data_dict("069-701_type1_nosupp")
        self.expected_069_701_type2_nosupp = self._get_data_dict("069-701_type2_nosupp")
        self.expected_085_705_nosupp = self._get_data_dict("085-705_nosupp")
        self.expected_143_781_nosupp = self._get_data_dict("143-781_nosupp")
        self.expected_084_706_nosupp = self._get_data_dict("084-706_nosupp")
        self.expected_096_702_nosupp = self._get_data_dict("096-702_nosupp")
        self.expected_098_707_nosupp = self._get_data_dict("098-707_nosupp")
        self.expected_103_794_nosupp = self._get_data_dict("103-794_nosupp")
        self.expected_125_704_nosupp = self._get_data_dict("125-704_nosupp")
        self.expected_125_721_nosupp = self._get_data_dict("125-721_nosupp")
        self.expected_133_730_nosupp = self._get_data_dict("133-730_nosupp")
        self.expected_144_703_nosupp = self._get_data_dict("144-703_nosupp")
        self.expected_091_201_nosupp = self._get_data_dict("091-201_nosupp")
        self.expected_077_892_nosupp = self._get_data_dict("077-892_nosupp")
        self.expected_147_700_nosupp = self._get_data_dict("147-700_nosupp")
        self.expected_103_792_nosupp = self._get_data_dict("103-792_nosupp")
        self.expected_114_992_nosupp = self._get_data_dict("114-992_nosupp")
        self.expected_063_714_c99 = self._get_data_dict("063-714_c99")
        self.expected_063_714_cdms = self._get_data_dict("063-714_cdms")
        self.expected_063_714_chunk = self._get_data_dict("063-714_chunk")

    def __getitem__(cls, attr):
        return getattr(cls, attr)

    def _get_data_dict(self, suffix):
        path = self._data_path / suffix
        data = pd.DataFrame()
        mask = pd.DataFrame()
        vaid = pd.DataFrame()
        vadt = pd.DataFrame()
        for data in path.glob("data_*.csv"):
            break
        for mask in path.glob("mask_*.csv"):
            break
        for vaid in path.glob("vaid_*.csv"):
            break
        for vadt in path.glob("vadt_*.csv"):
            break
        return {
            "data": data,
            "mask": mask,
            "cdm_table": path,
            "vaid": vaid,
            "vadt": vadt,
        }


result_data = result_data()
