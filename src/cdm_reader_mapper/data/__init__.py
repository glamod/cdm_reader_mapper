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
        """
        Initialize LazyDataDict instance.

        Parameters
        ----------
        loader : Callable[[Any], Any]
            Function used to load a value from a key.
        items : Mapping[Any, Any]
            Mapping from keys to the *resource* used by `loader` (e.g. a file path).
        attrs : Any, optional
            Reserved for future use - currently ignored.
        """
        super().__init__()
        self._loader = loader
        self._items = items

    def __getitem__(self, key: Any) -> Any:
        """
        Retrieve an item from the DataBundle.

        Parameters
        ----------
        key : Any
            Key used to access the underlying data or attribute.

        Returns
        -------
        Any
            The value associated with the given key.
        """
        if key not in self:
            path = self._items[key]
            self[key] = self._loader(path)
        return super().__getitem__(key)


class TestData:
    """
    Accessor for bundled CDM test data.

    Attributes
    ----------
    name : str
        Human-readable description of the test data collection.

    Notes
    -----
    Data is loaded lazily via `LazyDataDict`.
    Files are only accessed when explicitly requested.
    """

    def __init__(self) -> None:
        """Initialize the TestData container."""
        self.name = "CDM reader mapper input testing data."

    def __getitem__(self, attr: str) -> Any:
        """
        Retrieve a test dataset by attribute name.

        Parameters
        ----------
        attr : str
            Name of the dataset attribute to access.

        Returns
        -------
        Any
            The dataset associated with the given attribute.

        Raises
        ------
        KeyError
            If the requested attribute does not exist.
        """
        try:
            return getattr(self, attr)
        except AttributeError as err:
            raise KeyError(attr) from err

    def _load_file(self, ifile: str) -> Path | None:
        """
        Load a file from disk or remote source.

        Parameters
        ----------
        ifile : str
            File path or identifier to load.

        Returns
        -------
        pathlib.Path or None
            Path to the loaded file, or None if loading failed due to
            an HTTP error.

        Raises
        ------
        OSError
            If a local file system error occurs during loading.
        """
        try:
            return load_file(ifile)
        except HTTPError as err:
            logging.warning(err)
            return None
        except OSError as err:
            raise err

    def _get_data_dict(self, data_file: str, data_model: str, source_ext: str = "csv") -> LazyDataDict:
        """
        Construct a lazy-loading dictionary of dataset file paths.

        Parameters
        ----------
        data_file : str
            Identifier for the dataset (typically a timestamped subset).
        data_model : str
            Name of the data model (e.g., `icoads_r300_d714`).
        source_ext : str, default: csv
            File extension of the source input data.

        Returns
        -------
        LazyDataDict
            Dictionary-like object mapping dataset components to file paths,
            including source data, MDF outputs, validation files, and CDM tables.

        Notes
        -----
        The returned dictionary includes keys such as:
        - `source`: raw input file
        - `mdf_data` / `mdf_mask`: processed MDF outputs
        - `vadt` / `vaid`: validation outputs
        - `cdm_*`: CDM table files
        """
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

    @property
    def test_icoads_r300_d714(self) -> LazyDataDict:
        """
        Retrieve IMMA1 deck 714 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "2010-07-01_subset",
            "icoads_r300_d714",
            "imma",
        )

    @property
    def test_icoads_r300_d701(self) -> LazyDataDict:
        """
        Retrieve IMMA1 deck 701 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "1845-04-01_subset",
            "icoads_r300_d701",
            "imma",
        )

    @property
    def test_icoads_r300_d706(self) -> LazyDataDict:
        """
        Retrieve IMMA1 deck 706 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "1919-03-01_subset",
            "icoads_r300_d706",
            "imma",
        )

    @property
    def test_icoads_r300_d705(self) -> LazyDataDict:
        """
        Retrieve IMMA1 deck 705 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "1938-04-01_subset",
            "icoads_r300_d705",
            "imma",
        )

    @property
    def test_icoads_r300_d702(self) -> LazyDataDict:
        """
        Retrieve IMMA1 deck 702 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "1873-01-01_subset",
            "icoads_r300_d702",
            "imma",
        )

    @property
    def test_icoads_r300_d707(self) -> LazyDataDict:
        """
        Retrieve IMMA1 deck 707 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "1916-04-01_subset",
            "icoads_r300_d707",
            "imma",
        )

    @property
    def test_icoads_r300_mixed(self) -> LazyDataDict:
        """
        Retrieve IMMA1 mixed test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "1899-01-02_subset",
            "icoads_r300_mixed",
            "imma",
        )

    @property
    def test_icoads_r302_d794(self) -> LazyDataDict:
        """
        Retrieve IMMA1 deck 794 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "2022-11-01_subset",
            "icoads_r302_d794",
            "imma",
        )

    @property
    def test_icoads_r300_d704(self) -> LazyDataDict:
        """
        Retrieve IMMA1 deck 704 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "1878-10-01_subset",
            "icoads_r300_d704",
            "imma",
        )

    @property
    def test_icoads_r300_d721(self) -> LazyDataDict:
        """
        Retrieve IMMA1 deck 721 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "1862-06-01_subset",
            "icoads_r300_d721",
            "imma",
        )

    @property
    def test_icoads_r300_d730(self) -> LazyDataDict:
        """
        Retrieve IMMA1 deck 730 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "1776-10-01_subset",
            "icoads_r300_d730",
            "imma",
        )

    @property
    def test_icoads_r300_d781(self) -> LazyDataDict:
        """
        Retrieve IMMA1 deck 781 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "1987-09-01_subset",
            "icoads_r300_d781",
            "imma",
        )

    @property
    def test_icoads_r300_d703(self) -> LazyDataDict:
        """
        Retrieve IMMA1 deck 703 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "1979-09-01_subset",
            "icoads_r300_d703",
            "imma",
        )

    @property
    def test_icoads_r300_d201(self) -> LazyDataDict:
        """
        Retrieve IMMA1 deck 201 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "1913-11-01_subset",
            "icoads_r300_d201",
            "imma",
        )

    @property
    def test_icoads_r300_d892(self) -> LazyDataDict:
        """
        Retrieve IMMA1 deck 892 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "1996-02-01_subset",
            "icoads_r300_d892",
            "imma",
        )

    @property
    def test_icoads_r300_d700(self) -> LazyDataDict:
        """
        Retrieve IMMA1 deck 700 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "2002-08-01_subset",
            "icoads_r300_d700",
            "imma",
        )

    @property
    def test_icoads_r302_d792(self) -> LazyDataDict:
        """
        Retrieve IMMA1 deck 792 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "2022-02-01_subset",
            "icoads_r302_d792",
            "imma",
        )

    @property
    def test_icoads_r302_d992(self) -> LazyDataDict:
        """
        Retrieve IMMA1 deck 992 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "2022-01-01_subset",
            "icoads_r302_d992",
            "imma",
        )

    @property
    def test_gdac(self) -> LazyDataDict:
        """
        Retrieve IMMT test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict(
            "2003-02-01_subset",
            "gdac",
            "immt",
        )

    @property
    def test_craid(self) -> LazyDataDict:
        """
        Retrieve C-RAID 1260810 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict("2004-12-20_subset", "craid", "nc")

    @property
    def test_marob(self) -> LazyDataDict:
        """
        Retrieve MAROB (DWD database) test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict("2026-04-07_subset", "marob")

    @property
    def test_cmems(self) -> LazyDataDict:
        """
        Retrieve CMEMS (copernicusmarine) test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return self._get_data_dict("2017-05-02_subset", "cmems", "nc")

    @property
    def test_pub47(self) -> dict[str, Any]:
        """
        Retrieve Pub47 v202501 test dataset.

        Returns
        -------
        LazyDataDict
            Lazy-loading dictionary containing file paths for this dataset.
        """
        return {"source": load_file("Pub47/v202501/pub47_2022_01.csv")}


test_data = TestData()
