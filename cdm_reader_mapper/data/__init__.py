"""Common Data Model (CDM) reader and mapper test data."""

from __future__ import annotations

from cdm_reader_mapper.common.getting_files import load_file


class test_data:
    """CDM test data."""

    def __init__(self):
        self.test_063_714 = self._get_data_dict(
            "063-714_2010-07-01_subset.imma", "imma1_d714", "714",
        )
        self.test_069_701 = self._get_data_dict(
            "069-701_1845-04-01_subset.imma", "imma1_d701", "701",
        )
        self.test_084_706 = self._get_data_dict(
            "084-706_1919-03-01_subset.imma", "imma1_d705-707", "706",
        )
        self.test_085_705 = self._get_data_dict(
            "085-705_1938-04-01_subset.imma", "imma1_d705-707", "705",
        )
        self.test_096_702 = self._get_data_dict(
            "096-702_1873-01-01_subset.imma", "imma1_d702", "702",
        )
        self.test_098_707 = self._get_data_dict(
            "098-707_1916-04-01_subset.imma", "imma1_d705-707", "707"
        )
        self.test_103_794 = self._get_data_dict(
            "103-794_2021-11-01_subset.imma", "imma1_nodt", "794",
        )
        self.test_125_704 = self._get_data_dict(
            "125-704_1878-10-01_subset.imma", "imma1_d704", "704",
        )
        self.test_125_721 = self._get_data_dict(
            "125-721_1862-06-01_subset.imma", "imma1_d721", "721",
        )
        self.test_133_730 = self._get_data_dict(
            "133-730_1776-10-01_subset.imma", "imma1_d730", "730",
        )
        self.test_143_781 = self._get_data_dict(
            "143-781_1987-09-01_subset.imma", "imma1_d781", "781",
        )
        self.test_144_703 = self._get_data_dict(
            "144-703_1979-09-01_subset.imma", "imma1", "703",
        )
        #self.test_gcc_mix = self._get_data_dict("mix_out_20030201.immt", "gcc_immt")

    def __getitem__(self, attr):
        """Make class subscriptable."""
        try:
            return getattr(self, attr)
        except AttributeError as err:
            raise KeyError(attr) from err

    def _get_data_dict(self, data_file, schema, deck):
        data_file_ = f"imma1_d{deck}/input/{data_file}"
        return {
            "source": load_file(data_file_),
            "data_model": schema,
        }


test_data = test_data()
