"""Common Data Model (CDM) reader and mapper test data."""

from __future__ import annotations

from cdm_reader_mapper.common import load_file


class test_data:
    """CDM test data."""

    def __init__(self):
        self.name = "CDM reader mapper input testing data."

    @property
    def test_icoads_r300_d714(self):
        """IMMA1 deck 714 test data."""
        return self._get_data_dict(
            "2010-07-01_subset.imma",
            "icoads_r300_d714",
        )

    @property
    def test_icoads_r300_d701(self):
        """IMMA1 deck 701 test data."""
        return self._get_data_dict(
            "1845-04-01_subset.imma",
            "icoads_r300_d701",
        )

    @property
    def test_icoads_r300_d701_type1(self):
        """IMMA1 deck 701 type1 test data."""
        return self._get_data_dict(
            "1845-04-01_subset.imma",
            "icoads_r300_d701",
        )

    @property
    def test_icoads_r300_d701_type2(self):
        """IMMA1 deck 701 type2 test data."""
        return self._get_data_dict(
            "1845-04-01_subset.imma",
            "icoads_r300_d701",
        )

    @property
    def test_icoads_r300_d706(self):
        """IMMA1 deck 706 test data."""
        return self._get_data_dict(
            "1919-03-01_subset.imma",
            "icoads_r300_d706",
        )

    @property
    def test_icoads_r300_d705(self):
        """IMMA1 deck 705 test data."""
        return self._get_data_dict(
            "1938-04-01_subset.imma",
            "icoads_r300_d705",
        )

    @property
    def test_icoads_r300_d702(self):
        """IMMA1 deck 702 test data."""
        return self._get_data_dict(
            "1873-01-01_subset.imma",
            "icoads_r300_d702",
        )

    @property
    def test_icoads_r300_d707(self):
        """IMMA1 deck 707 test data."""
        return self._get_data_dict(
            "1916-04-01_subset.imma",
            "icoads_r300_d707",
        )

    @property
    def test_icoads_r302_d794(self):
        """IMMA1 deck 794 test data."""
        return self._get_data_dict(
            "2022-11-01_subset.imma",
            "icoads_r302_d794",
        )

    @property
    def test_icoads_r300_d704(self):
        """IMMA1 deck 704 test data."""
        return self._get_data_dict(
            "1878-10-01_subset.imma",
            "icoads_r300_d704",
        )

    @property
    def test_icoads_r300_d721(self):
        """IMMA1 deck 721 test data."""
        return self._get_data_dict(
            "1862-06-01_subset.imma",
            "icoads_r300_d721",
        )

    @property
    def test_icoads_r300_d730(self):
        """IMMA1 deck 730 test data."""
        return self._get_data_dict(
            "1776-10-01_subset.imma",
            "icoads_r300_d730",
        )

    @property
    def test_icoads_r300_d781(self):
        """IMMA1 deck 781 test data."""
        return self._get_data_dict(
            "1987-09-01_subset.imma",
            "icoads_r300_d781",
        )

    @property
    def test_icoads_r300_d703(self):
        """IMMA1 deck 703 test data."""
        return self._get_data_dict(
            "1979-09-01_subset.imma",
            "icoads_r300_d703",
        )

    @property
    def test_icoads_r300_d201(self):
        """IMMA1 deck 201 test data."""
        return self._get_data_dict(
            "1913-11-01_subset.imma",
            "icoads_r300_d201",
        )

    @property
    def test_icoads_r300_d892(self):
        """IMMA1 deck 892 test data."""
        return self._get_data_dict(
            "1996-02-01_subset.imma",
            "icoads_r300_d892",
        )

    @property
    def test_icoads_r300_d700(self):
        """IMMA1 deck 700 test data."""
        return self._get_data_dict(
            "2002-08-01_subset.imma",
            "icoads_r300_d700",
        )

    @property
    def test_icoads_r302_d792(self):
        """IMMA1 deck 792 test data."""
        return self._get_data_dict(
            "2022-02-01_subset.imma",
            "icoads_r302_d792",
        )

    @property
    def test_icoads_r302_d992(self):
        """IMMA1 deck 992 test data."""
        return self._get_data_dict(
            "2022-01-01_subset.imma",
            "icoads_r302_d992",
        )

    @property
    def test_gcc(self):
        """IMMAT deck ??? test data."""
        return self._get_data_dict(
            "20030201.immt",
            "gcc",
        )

    @property
    def test_craid(self):
        """C-RAID 1260810 test data."""
        return self._get_data_dict(
            "2004-12-20_subset.nc",
            "craid",
        )

    def __getitem__(self, attr):
        """Make class subscriptable."""
        try:
            return getattr(self, attr)
        except AttributeError as err:
            raise KeyError(attr) from err

    def _get_data_dict(self, data_file, data_model):
        drs = "/".join(data_model.split("_"))
        return {
            "source": load_file(f"{drs}/input/{data_model}_{data_file}"),
        }


test_data = test_data()
