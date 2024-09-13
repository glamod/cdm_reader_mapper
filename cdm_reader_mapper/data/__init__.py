"""Common Data Model (CDM) reader and mapper test data."""

from __future__ import annotations

from cdm_reader_mapper.common import load_file


class test_data:
    """CDM test data."""

    def __init__(self):
        self.name = "CDM reader mapper input testing data."

    @property
    def test_063_714(self):
        """IMMA1 deck 714 test data."""
        return self._get_data_dict(
            "063-714_2010-07-01_subset.imma",
            "icoads_r300_d714",
            "714",
            "imma1",
            "icoads_r3000",
        )

    @property
    def test_069_701(self):
        """IMMA1 deck 701 test data."""
        return self._get_data_dict(
            "069-701_1845-04-01_subset.imma",
            "icoads_r300_d701",
            "701",
            "imma1",
            "icoads_r3000",
        )

    @property
    def test_084_706(self):
        """IMMA1 deck 706 test data."""
        return self._get_data_dict(
            "084-706_1919-03-01_subset.imma",
            "icoads_r300_d706",
            "706",
            "imma1",
            "icoads_r3000",
        )

    @property
    def test_085_705(self):
        """IMMA1 deck 705 test data."""
        return self._get_data_dict(
            "085-705_1938-04-01_subset.imma",
            "icoads_r300_d705",
            "705",
            "imma1",
            "icoads_r3000",
        )

    @property
    def test_096_702(self):
        """IMMA1 deck 702 test data."""
        return self._get_data_dict(
            "096-702_1873-01-01_subset.imma",
            "icoads_r300_d702",
            "702",
            "imma1",
            "icoads_r3000",
        )

    @property
    def test_098_707(self):
        """IMMA1 deck 707 test data."""
        return self._get_data_dict(
            "098-707_1916-04-01_subset.imma",
            "icoads_r300_d707",
            "707",
            "imma1",
            "icoads_r3000",
        )

    @property
    def test_103_794(self):
        """IMMA1 deck 794 test data."""
        return self._get_data_dict(
            "103-794_2022-11-01_subset.imma",
            "icoads_r302_794",
            "794",
            "imma1",
            "icoads_r3000_NRT",
        )

    @property
    def test_125_704(self):
        """IMMA1 deck 704 test data."""
        return self._get_data_dict(
            "125-704_1878-10-01_subset.imma",
            "icoads_r300_d704",
            "704",
            "imma1",
            "icoads_r3000",
        )

    @property
    def test_125_721(self):
        """IMMA1 deck 721 test data."""
        return self._get_data_dict(
            "125-721_1862-06-01_subset.imma",
            "icoads_r300_d721",
            "721",
            "imma1",
            "icoads_r3000",
        )

    @property
    def test_133_730(self):
        """IMMA1 deck 730 test data."""
        return self._get_data_dict(
            "133-730_1776-10-01_subset.imma",
            "icoads_r300_d730",
            "730",
            "imma1",
            "icoads_r3000",
        )

    @property
    def test_143_781(self):
        """IMMA1 deck 781 test data."""
        return self._get_data_dict(
            "143-781_1987-09-01_subset.imma",
            "icoads_r300_d781",
            "781",
            "imma1",
            "icoads_r3000",
        )

    @property
    def test_144_703(self):
        """IMMA1 deck 703 test data."""
        return self._get_data_dict(
            "144-703_1979-09-01_subset.imma",
            "icoads_r300_d703",
            "703",
            "imma1",
            "icoads_r3000",
        )

    @property
    def test_091_201(self):
        """IMMA1 deck 201 test data."""
        return self._get_data_dict(
            "091-201_1913-11-01_subset.imma",
            "icoads_r300_d201",
            "201",
            "imma1",
            "icoads_r3000",
        )

    @property
    def test_077_892(self):
        """IMMA1 deck 892 test data."""
        return self._get_data_dict(
            "077-892_1996-02-01_subset.imma",
            "icoads_r300_d892",
            "892",
            "imma1",
            "icoads_r3000",
        )

    @property
    def test_147_700(self):
        """IMMA1 deck 700 test data."""
        return self._get_data_dict(
            "147-700_2002-08-01_subset.imma",
            "icoads_r300_d700",
            "700",
            "imma1",
            "icoads_r3000",
        )

    @property
    def test_103_792(self):
        """IMMA1 deck 792 test data."""
        return self._get_data_dict(
            "103-792_2022-02-01_subset.imma",
            "icoads_r302_d792",
            "792",
            "imma1",
            "icoads_r3000_NRT",
        )

    @property
    def test_114_992(self):
        """IMMA1 deck 992 test data."""
        return self._get_data_dict(
            "114-992_2022-01-01_subset.imma",
            "icoads_r302_d992",
            "992",
            "imma1",
            "icoads_r3000_NRT",
        )

    @property
    def test_gcc_mix(self):
        """IMMAT deck ??? test data."""
        return self._get_data_dict(
            "mix-out_20030201.immt",
            "gcc",
            "gcc",
            "immt",
            "gdac_r0000",
        )

    @property
    def test_craid_1260810(self):
        """C-RAID 1260810 test data."""
        return self._get_data_dict(
            "1260810_2004-12-20_subset.nc",
            "craid",
            "raid",
            "c",
            "c_raid",
        )

    def __getitem__(self, attr):
        """Make class subscriptable."""
        try:
            return getattr(self, attr)
        except AttributeError as err:
            raise KeyError(attr) from err

    def _get_data_dict(self, data_file, schema, deck, dm, ds):
        return {
            "source": load_file(f"{dm}_{deck}/input/{data_file}"),
            "data_model": schema,
            "dm": dm,
            "ds": ds,
            "deck": deck,
        }


test_data = test_data()
