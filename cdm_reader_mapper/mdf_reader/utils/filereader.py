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

    def _adjust_schema(self, ds, dtypes):
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
                    if value == "__from_file__":
                        if attr in ds[data_var].attrs:
                            self.schema["sections"][section]["elements"][data_var][
                                attr
                            ] = ds[data_var].attrs[attr]
                        else:
                            del self.schema["sections"][section]["elements"][data_var][
                                attr
                            ]

    def _select_years(self, df):
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

    def _read_pandas(self, **kwargs):
        return pd.read_fwf(
            self.source,
            header=None,
            quotechar="\0",
            escapechar="\0",
            dtype=object,
            skip_blank_lines=False,
            **kwargs,
        )

    def _read_netcdf(self, **kwargs):
        ds = xr.open_mfdataset(self.source, **kwargs)
        self._adjust_schema(ds, ds.dtypes)
        return ds.squeeze()

    def _read_sections(
        self,
        TextParser,
        order,
        valid,
        open_with,
        configurations={},
    ):
        if open_with == "pandas":
            df_total = TextParser.apply(
                lambda x: Configurator(
                    df=x,
                    schema=self.schema,
                    order=order,
                    valid=valid,
                ).open_pandas(configurations, self.imodel, self.ext_table_path),
                axis=1,
            )
        elif open_with == "netcdf":
            df_total = Configurator(
                df=TextParser, schema=self.schema, order=order, valid=valid
            ).open_netcdf(configurations)
        else:
            raise ValueError("open_with has to be one of ['pandas', 'netcdf']")

        if configuration.get("validate") is True:
            columns = df_total.columns
            half = len(columns) / 2
            df = df_total.iloc[:, : int(half)]
            mask = df_total.iloc[:, int(half) :]
        else:
            mask = pd.DataFrame()
        self.columns = df.columns
        df = df.where(df.notnull(), None)
        return df, mask

    def get_configurations(self, order, valid):
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
        convert=True,
        decode=True,
        validate=True,
        configurations={},
        open_with="pandas",
    ):
        """DOCUMENTATION."""
        configurations["convert"] = convert
        configurations["decode"] = decode
        configurations["validate"] = validate
        if open_with == "netcdf":
            TextParser = self._read_netcdf()
        elif open_with == "pandas":
            TextParser = self._read_pandas(
                encoding=self.schema["header"].get("encoding"),
                widths=[properties.MAX_FULL_REPORT_WIDTH],
                skiprows=self.skiprows,
                chunksize=chunksize,
            )
        else:
            raise ValueError("open_with has to be one of ['pandas', 'netcdf']")

        if isinstance(TextParser, pd.DataFrame) or isinstance(TextParser, xr.Dataset):
            df, mask = self._read_sections(
                TextParser,
                order,
                valid,
                open_with=open_with,
                configurations=configurations,
            )
            return df, mask
        else:
            data_buffer = StringIO()
            missings_buffer = StringIO()
            isna_buffer = StringIO()
            for i, df_ in enumerate(TextParser):
                df, missing_values = self._read_sections(
                    df_, order, valid, open_with=open_with
                )
                df_isna = df.isna()
                missing_values.to_csv(
                    missings_buffer,
                    header=False,
                    mode="a",
                    encoding="utf-8",
                    index=False,
                )
                df_isna.to_csv(
                    isna_buffer,
                    header=False,
                    mode="a",
                    index=False,
                    quoting=csv.QUOTE_NONE,
                    sep=properties.internal_delimiter,
                    quotechar="\0",
                    escapechar="\0",
                )
                df.to_csv(
                    data_buffer,
                    header=False,
                    mode="a",
                    encoding="utf-8",
                    index=False,
                    quoting=csv.QUOTE_NONE,
                    sep=properties.internal_delimiter,
                    quotechar="\0",
                    escapechar="\0",
                )
            missings_buffer.seek(0)
            self.missing_values = pd.read_csv(
                missings_buffer,
                names=missing_values.columns,
                chunksize=None,
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
            isna_buffer.seek(0)
            isna = pd.read_csv(
                isna_buffer,
                names=df.columns,
                chunksize=self.chunksize,
                delimiter=properties.internal_delimiter,
                quotechar="\0",
                escapechar="\0",
            )
            return data, isna
