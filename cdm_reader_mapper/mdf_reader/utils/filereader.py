"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import logging
import os
from copy import deepcopy

import polars as pl
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
                    if value != "__from_file__":
                        continue
                    if attr in ds[data_var].attrs:
                        self.schema["sections"][section]["elements"][data_var][attr] = (
                            ds[data_var].attrs[attr]
                        )
                    else:
                        del self.schema["sections"][section]["elements"][data_var][attr]

    def _select_years(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.year_init is None and self.year_end is None:
            return df

        if self.imodel is None:
            logging.error("Selection of years is not supported for custom schema")
            return df

        data_model = self.imodel.split("_")[0]
        dates = df.get_column(properties.year_column[data_model])
        if dates.dtype == pl.Datetime:
            years = dates.dt.year()
        else:
            years = dates.cast(pl.Int64, strict=False)

        mask = pl.repeat(True, df.height, eager=True)
        if self.year_init and self.year_end:
            mask = years.is_between(self.year_init, self.year_end, closed="both")
        elif self.year_init:
            mask = years.ge(self.year_init)
        elif self.year_end:
            mask = years.le(self.year_end)

        return df.filter(mask)

    def _read_text(self, **kwargs):
        return pd.read_fwf(
            self.source,
            header=None,
            quotechar="\0",
            escapechar="\0",
            dtype=object,
            skip_blank_lines=False,
            widths=[properties.MAX_FULL_REPORT_WIDTH],
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
        format,
    ):
        if format == "text":
            df, mask_df = Configurator(
                df=TextParser, schema=self.schema, order=order, valid=valid
            ).open_text()
        elif format == "netcdf":
            df, mask_df = Configurator(
                df=TextParser, schema=self.schema, order=order, valid=valid
            ).open_netcdf()
        else:
            raise ValueError("format has to be one of ['text', 'netcdf']")

        # missing_values = df.select(["index", "missing_values"]).pipe(set_missing_values)
        df = df.pipe(self._select_years)

        self.columns = df.columns
        # Replace None with NaN - is this necessary for polars?
        # df = df.where(df.notnull(), np.nan)
        return df, mask_df

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
        chunksize,
        format="text",
    ):
        """DOCUMENTATION."""
        encoding = self.schema["header"].get("encoding")
        if format == "netcdf":
            TextParser = self._read_netcdf()
        # NOTE: Chunking - polars does have pl.read_csv_batched, but batch_size
        # is not respected: https://github.com/pola-rs/polars/issues/19978
        # alternative: lazy?
        elif format == "text":
            TextParser = self._read_text(
                encoding=encoding,
                skiprows=self.skiprows,
                chunksize=chunksize,
            )
        else:
            raise ValueError("format has to be one of ['text', 'netcdf']")

        return TextParser
        # if isinstance(TextParser, (pl.DataFrame, xr.Dataset)):
        #     df, mask_df = self._read_sections(TextParser, order, valid, open_with=open_with)
        #     return df, mask_df
        # else:
        #     data_buffer = StringIO()
        #     for i, df_ in enumerate(TextParser):
        #         df = self._read_sections(df_, order, valid, open_with=open_with)
        #         df.to_csv(
        #             data_buffer,
        #             header=False,
        #             mode="a",
        #             encoding="utf-8",
        #             index=False,
        #             quoting=csv.QUOTE_NONE,
        #             sep=properties.internal_delimiter,
        #             quotechar="\0",
        #             escapechar="\0",
        #         )
        #     data_buffer.seek(0)
        #     data = pd.read_csv(
        #         data_buffer,
        #         names=df.columns,
        #         chunksize=self.chunksize,
        #         dtype=object,
        #         parse_dates=self.parse_dates,
        #         delimiter=properties.internal_delimiter,
        #         quotechar="\0",
        #         escapechar="\0",
        #     )
        #     return data
