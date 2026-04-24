"""Common Data Model (CDM) reader and mapper test data."""

from __future__ import annotations
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from requests.exceptions import HTTPError  # type: ignore[import-untyped]

from cdm_reader_mapper.cdm_mapper.properties import cdm_tables
from cdm_reader_mapper.common import load_file


__all__ = ["test_data"]


class LazyDataDict(dict[Any, Any]):
    """
    Create an empty dictionary and store the loader / mapping.

    Parameters
    ----------
    loader : Callable[[Any], Any]
            Function used to load a value from a key.
    items : Mapping[Any, Any]
            Mapping from keys to the *resource* used by `loader` (e.g. a file path).
    attrs : Any, optional
            Reserved for future use - currently ignored.
    """

    def __init__(self, loader: Callable[[Any], Any], items: Mapping[Any, Any], attrs: Any = None) -> None:
        super().__init__()
        self._loader = loader
        self._items = items

    def __getitem__(self, key: Any) -> Any:
        """Make class subscriptable."""
        if key not in self:
            path = self._items[key]
            self[key] = self._loader(path)
        return super().__getitem__(key)


class TestData:
    """CDM test data."""

    def __init__(self) -> None:
        self.name = "CDM reader mapper input testing data."

    @property
    def test_icoads_r300_d714(self) -> LazyDataDict:
        """IMMA1 deck 714 test data."""
        return self._get_data_dict(
            "2010-07-01_subset",
            "icoads_r300_d714",
            "imma",
        )

    @property
    def test_icoads_r300_d701(self) -> LazyDataDict:
        """IMMA1 deck 701 test data."""
        return self._get_data_dict(
            "1845-04-01_subset",
            "icoads_r300_d701",
            "imma",
        )

    @property
    def test_icoads_r300_d706(self) -> LazyDataDict:
        """IMMA1 deck 706 test data."""
        return self._get_data_dict(
            "1919-03-01_subset",
            "icoads_r300_d706",
            "imma",
        )

    @property
    def test_icoads_r300_d705(self) -> LazyDataDict:
        """IMMA1 deck 705 test data."""
        return self._get_data_dict(
            "1938-04-01_subset",
            "icoads_r300_d705",
            "imma",
        )

    @property
    def test_icoads_r300_d702(self) -> LazyDataDict:
        """IMMA1 deck 702 test data."""
        return self._get_data_dict(
            "1873-01-01_subset",
            "icoads_r300_d702",
            "imma",
        )

    @property
    def test_icoads_r300_d707(self) -> LazyDataDict:
        """IMMA1 deck 707 test data."""
        return self._get_data_dict(
            "1916-04-01_subset",
            "icoads_r300_d707",
            "imma",
        )

    @property
    def test_icoads_r300_mixed(self) -> LazyDataDict:
        """IMMA1 mixed test data."""
        return self._get_data_dict(
            "1899-01-02_subset",
            "icoads_r300_mixed",
            "imma",
        )

    @property
    def test_icoads_r302_d794(self) -> LazyDataDict:
        """IMMA1 deck 794 test data."""
        return self._get_data_dict(
            "2022-11-01_subset",
            "icoads_r302_d794",
            "imma",
        )

    @property
    def test_icoads_r300_d704(self) -> LazyDataDict:
        """IMMA1 deck 704 test data."""
        return self._get_data_dict(
            "1878-10-01_subset",
            "icoads_r300_d704",
            "imma",
        )

    @property
    def test_icoads_r300_d721(self) -> LazyDataDict:
        """IMMA1 deck 721 test data."""
        return self._get_data_dict(
            "1862-06-01_subset",
            "icoads_r300_d721",
            "imma",
        )

    @property
    def test_icoads_r300_d730(self) -> LazyDataDict:
        """IMMA1 deck 730 test data."""
        return self._get_data_dict(
            "1776-10-01_subset",
            "icoads_r300_d730",
            "imma",
        )

    @property
    def test_icoads_r300_d781(self) -> LazyDataDict:
        """IMMA1 deck 781 test data."""
        return self._get_data_dict(
            "1987-09-01_subset",
            "icoads_r300_d781",
            "imma",
        )

    @property
    def test_icoads_r300_d703(self) -> LazyDataDict:
        """IMMA1 deck 703 test data."""
        return self._get_data_dict(
            "1979-09-01_subset",
            "icoads_r300_d703",
            "imma",
        )

    @property
    def test_icoads_r300_d201(self) -> LazyDataDict:
        """IMMA1 deck 201 test data."""
        return self._get_data_dict(
            "1913-11-01_subset",
            "icoads_r300_d201",
            "imma",
        )

    @property
    def test_icoads_r300_d892(self) -> LazyDataDict:
        """IMMA1 deck 892 test data."""
        return self._get_data_dict(
            "1996-02-01_subset",
            "icoads_r300_d892",
            "imma",
        )

    @property
    def test_icoads_r300_d700(self) -> LazyDataDict:
        """IMMA1 deck 700 test data."""
        return self._get_data_dict(
            "2002-08-01_subset",
            "icoads_r300_d700",
            "imma",
        )

    @property
    def test_icoads_r302_d792(self) -> LazyDataDict:
        """IMMA1 deck 792 test data."""
        return self._get_data_dict(
            "2022-02-01_subset",
            "icoads_r302_d792",
            "imma",
        )

    @property
    def test_icoads_r302_d992(self) -> LazyDataDict:
        """IMMA1 deck 992 test data."""
        return self._get_data_dict(
            "2022-01-01_subset",
            "icoads_r302_d992",
            "imma",
        )

    @property
    def test_gdac(self) -> LazyDataDict:
        """IMMT test data."""
        return self._get_data_dict(
            "2003-02-01_subset",
            "gdac",
            "immt",
        )

    @property
    def test_craid(self) -> LazyDataDict:
        """C-RAID 1260810 test data."""
        return self._get_data_dict("2004-12-20_subset", "craid", "nc")

    @property
    def test_marob(self) -> LazyDataDict:
        """MAROB (DWD database) test data."""
        return self._get_data_dict("2026-04-07_subset", "marob")

    @property
    def test_cmems(self) -> LazyDataDict:
        """CMEMS (copernicusmarine) test data."""
        return self._get_data_dict("2017-05-02_subset", "cmems", "nc")

    @property
    def test_pub47(self) -> dict[str, Any]:
        """Pub47 v202501 test data."""
        return {"source": load_file("Pub47/v202501/pub47_2022_01.csv")}

    def __getitem__(self, attr: str) -> Any:
        """Make class subscriptable."""
        try:
            return getattr(self, attr)
        except AttributeError as err:
            raise KeyError(attr) from err

    def _load_file(self, ifile: str) -> Path | None:
        try:
            return load_file(ifile)
        except HTTPError as err:
            logging.warning(err)
            return None
        except OSError as err:
            raise err

    def _get_data_dict(self, data_file: str, data_model: str, source_ext: str = "csv") -> LazyDataDict:
        drs = "/".join(data_model.split("_"))
        data_dict = {
            "source": f"{drs}/input/{data_model}_{data_file}.{source_ext}",
            "mdf_data": f"{drs}/output/data_{data_model}_{data_file}.pq",
            "mdf_mask": f"{drs}/output/mask_{data_model}_{data_file}.pq",
            "vadt": f"{drs}/validation/vadt_{data_model}_{data_file}.csv",
            "vaid": f"{drs}/validation/vaid_{data_model}_{data_file}.csv",
        }
        for cdm_table in cdm_tables:
            cdm_table_file = f"{drs}/cdm_tables/{cdm_table}-{data_model}_{data_file}.pq"
            data_dict[f"cdm_{cdm_table}"] = cdm_table_file

        return LazyDataDict(self._load_file, data_dict)


test_data = TestData()
