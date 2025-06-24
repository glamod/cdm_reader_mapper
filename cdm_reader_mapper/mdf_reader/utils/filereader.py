"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import csv
import logging
import os
from copy import deepcopy
from io import StringIO

import pandas as pd
import xarray as xr

from .. import properties
from ..schemas import schemas
from .configurator import Configurator
from .utilities import validate_path


class FileReader:
    """Class to read marine-meteorological data."""

    def __init__(
        self,
        source,
        imodel=None,
        ext_schema_path=None,
        ext_schema_file=None,
        ext_table_path=None,
        year_init=None,
        year_end=None,
    ):
        # 0. VALIDATE INPUT
        if not imodel and not ext_schema_path:
            logging.error(
                "A valid input data model name or path to data model must be provided"
            )
            return
        if not os.path.isfile(source):
            logging.error(f"Can't find input data file {source}")
            return
        if not validate_path("ext_schema_path", ext_schema_path):
            return

        self.source = source
        self.imodel = imodel
        self.year_init = year_init
        self.year_end = year_end
        self.ext_table_path = ext_table_path

        # 1. GET DATA MODEL
        # Schema reader will return empty if cannot read schema or is not valid
        # and will log the corresponding error
        # multiple_reports_per_line error also while reading schema
        logging.info("READING DATA MODEL SCHEMA FILE...")
        if ext_schema_path or ext_schema_file:
            self.schema = schemas.read_schema(
                ext_schema_path=ext_schema_path, ext_schema_file=ext_schema_file
            )
        else:
            self.schema = schemas.read_schema(imodel=imodel)

    def _adjust_schema(self, ds, dtypes) -> dict:
        sections = deepcopy(self.schema["sections"])
        for section in sections.keys():
            elements = sections[section]["elements"]
            for data_var in elements.keys():
                not_in_data_vars = data_var not in ds.data_vars
                not_in_glb_attrs = data_var not in ds.attrs
                not_in_data_dims = data_var not in ds.dims
                if not_in_data_vars and not_in_glb_attrs and not_in_data_dims:
                    del self.schema["sections"][section]["elements"][data_var]
                    continue
                for attr, value in elements[data_var].items():
                    if value != "__from_file__":
                        continue
                    if attr in ds[data_var].attrs:
                        self.schema["sections"][section]["elements"][data_var][attr] = (
                            ds[data_var].attrs[attr]
                        )
                    else:
                        del self.schema["sections"][section]["elements"][data_var][attr]

    def _select_years(self, df) -> pd.DataFrame:
        def get_years_from_datetime(date):
            try:
                return date.year
            except AttributeError:
                return date

        if self.year_init is None and self.year_end is None:
            return df

        data_model = self.imodel.split("_")[0]
        dates = df[properties.year_column[data_model]]
        years = dates.apply(lambda x: get_years_from_datetime(x))
        years = years.astype(int)

        mask = pd.Series([True] * len(years))
        if self.year_init:
            mask[years < self.year_init] = False
        if self.year_end:
            mask[years > self.year_end] = False

        index = mask[mask].index
        return df.iloc[index].reset_index(drop=True)

    def _read_pandas(self, **kwargs) -> pd.DataFrame | pd.io.parsers.TextFileReader:
        if (enc := kwargs.get("encoding")) is not None:
            logging.info(f"Reading with encoding = {enc}")
        return pd.read_fwf(
            self.source,
            header=None,
            quotechar="\0",
            escapechar="\0",
            dtype=object,
            skip_blank_lines=False,
            **kwargs,
        )

    def _read_netcdf(self, **kwargs) -> xr.Dataset:
        ds = xr.open_mfdataset(self.source, **kwargs)
        self._adjust_schema(ds, ds.dtypes)
        return ds.squeeze()

    def _read_sections(
        self,
        TextParser,
        order,
        valid,
        open_with,
    ) -> pd.DataFrame:
        if open_with == "pandas":
            df = Configurator(
                df=TextParser, schema=self.schema, order=order, valid=valid
            ).open_pandas()
        elif open_with == "netcdf":
            df = Configurator(
                df=TextParser, schema=self.schema, order=order, valid=valid
            ).open_netcdf()
        else:
            raise ValueError("open_with has to be one of ['pandas', 'netcdf']")

        self.columns = df.columns
        return self._select_years(df)

    def get_configurations(self, order, valid) -> dict:
        """DOCUMENTATION."""
        config_dict = Configurator(
            schema=self.schema, order=order, valid=valid
        ).get_configuration()
        for attr, val in config_dict["self"].items():
            setattr(self, attr, val)
        del config_dict["self"]
        return config_dict

    def open_data(
        self,
        order,
        valid,
        chunksize,
        open_with="pandas",
        encoding: str | None = None,
    ) -> pd.DataFrame | pd.io.parsers.TextFileReader:
        """DOCUMENTATION."""
        encoding = encoding or self.schema["header"].get("encoding")
        if open_with == "netcdf":
            TextParser = self._read_netcdf()
        elif open_with == "pandas":
            TextParser = self._read_pandas(
                encoding=encoding,
                widths=[properties.MAX_FULL_REPORT_WIDTH],
                skiprows=self.skiprows,
                chunksize=chunksize,
            )
        else:
            raise ValueError("open_with has to be one of ['pandas', 'netcdf']")

        if isinstance(TextParser, pd.DataFrame) or isinstance(TextParser, xr.Dataset):
            return self._read_sections(TextParser, order, valid, open_with=open_with)
        else:
            data_buffer = StringIO()
            for i, df_ in enumerate(TextParser):
                df = self._read_sections(df_, order, valid, open_with=open_with)
                df.to_csv(
                    data_buffer,
                    header=False,
                    mode="a",
                    encoding=encoding,
                    index=False,
                    quoting=csv.QUOTE_NONE,
                    sep=properties.internal_delimiter,
                    quotechar="\0",
                    escapechar="\0",
                )
            data_buffer.seek(0)
            data = pd.read_csv(
                data_buffer,
                names=df.columns,
                chunksize=self.chunksize,
                dtype=object,
                parse_dates=self.parse_dates,
                delimiter=properties.internal_delimiter,
                quotechar="\0",
                escapechar="\0",
            )
            return data
