"""Common Data Model (CDM) reader and mapper test data."""

from __future__ import annotations

import os


class test_data:
    """CDM test data."""

    def __init__(self):
        self.data_path = os.path.dirname(os.path.abspath(__file__))

        self.test_063_714 = self._get_data_dict(
            "063-714_2010-07_subset.imma", "imma1_d714"
        )
        self.test_069_701 = self._get_data_dict(
            "069-701_1845-04_subset.imma", "imma1_d701"
        )
        self.test_084_706 = self._get_data_dict(
            "084-706_1919-03_subset.imma", "imma1_d705-707"
        )
        self.test_085_705 = self._get_data_dict(
            "085-705_1938-04_subset.imma", "imma1_d705-707"
        )
        self.test_096_702 = self._get_data_dict(
            "096-702_1873-01_subset.imma", "imma1_d702"
        )
        self.test_098_707 = self._get_data_dict(
            "098-707_1916-04_subset.imma", "imma1_d705-707"
        )
        self.test_103_794 = self._get_data_dict(
            "103-794_2021-11_subset.imma", "imma1_nodt"
        )
        self.test_125_704 = self._get_data_dict(
            "125-704_1878-10_subset.imma", "imma1_d704"
        )
        self.test_125_721 = self._get_data_dict(
            "125-721_1862-06_subset.imma", "imma1_d721"
        )
        self.test_133_730 = self._get_data_dict(
            "133-730_1776-10_subset.imma", "imma1_d730"
        )
        self.test_143_781 = self._get_data_dict(
            "143-781_1987-09_subset.imma", "imma1_d781"
        )
        self.test_144_703 = self._get_data_dict(
            "144-703_1979-09_subset.imma", "imma1"
        )
        self.test_gcc_mix = self._get_data_dict("mix_out_200302.immt", "gcc_immt")

    def _get_data_dict(self, data_file, schema):
        return {
            "source": os.path.join(self.data_path, data_file),
            "data_model": schema,
        }


test_data = test_data()
