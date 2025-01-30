"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import logging
import os
from copy import deepcopy

import polars as pl
import xarray as xr

from .. import properties
from ..schemas import schemas
from ..validate import validate
from .configurator import Configurator
from .utilities import (
    convert_entries,
    create_mask,
    decode_entries,
    set_missing_values,
    validate_path,
)


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

    # def _read_pandas(self, **kwargs):
    #     return pd.read_fwf(
    #         self.source,
    #         header=None,
    #         quotechar="\0",
    #         escapechar="\0",
    #         dtype=object,
    #         skip_blank_lines=False,
    #         **kwargs,
    #     )
    #
    def _read_fwf_polars(self, **kwargs):
        if "chunksize" in kwargs:
            logging.warning("Chunking not supported by polars reader")
            # batch_size = kwargs["chunksize"]
            del kwargs["chunksize"]
            # return pl.read_csv_batched(
            #     self.source,
            #     has_header=False,
            #     separator="\0",
            #     new_columns=["full_str"],
            #     quote_char="\0",
            #     infer_schema_length=0,
            #     batch_size=batch_size,
            #     **kwargs,
            # )
        return pl.read_csv(
            self.source,
            has_header=False,
            separator="\0",
            new_columns=["full_str"],
            quote_char="\0",
            infer_schema=False,
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
    ):
        if open_with == "polars":
            df = Configurator(
                df=TextParser, schema=self.schema, order=order, valid=valid
            ).open_polars()
        # elif open_with == "pandas":
        #     df = Configurator(df=TextParser, schema=self.schema, order=order, valid=valid).open_pandas()
        elif open_with == "netcdf":
            df = Configurator(
                df=TextParser, schema=self.schema, order=order, valid=valid
            ).open_netcdf()
        else:
            raise ValueError("open_with has to be one of ['polars', 'netcdf']")

        missing_values = df.select(["index", "missing_values"]).pipe(set_missing_values)
        df = df.drop("missing_values").pipe(self._select_years)

        self.columns = df.columns
        # Replace None with NaN - is this necessary for polars?
        # df = df.where(df.notnull(), np.nan)
        return df, missing_values

    def get_configurations(self, order, valid):
        """DOCUMENTATION."""
        config_dict = Configurator(
            schema=self.schema, order=order, valid=valid
        ).get_configuration()
        for attr, val in config_dict["self"].items():
            setattr(self, attr, val)
        del config_dict["self"]
        return config_dict

    def convert_and_decode_df(
        self,
        df,
        converter_dict,
        converter_kwargs,
        decoder_dict,
    ):
        """DOCUMENTATION."""
        for section in converter_dict.keys():
            if section not in df.columns:
                continue
            if section in decoder_dict.keys():
                decoded = decode_entries(
                    df[section],
                    decoder_dict[section],
                )
                # decoded.index = df[section].index
                df = df.with_columns(decoded.alias(section))

            converted = convert_entries(
                df[section],
                converter_dict[section],
                **converter_kwargs[section],
            )
            # converted.index = df[section].index
            df = df.with_columns(converted.alias(section))
        return df

    def validate_df(self, df, isna=None):
        """DOCUMENTATION."""
        mask = create_mask(df, isna, missing_values=self.missing_values)
        return validate(
            data=df,
            mask0=mask,
            imodel=self.imodel,
            ext_table_path=self.ext_table_path,
            schema=self.schema,
            disables=self.disable_reads,
        )

    def open_data(
        self,
        order,
        valid,
        # chunksize,
        open_with="polars",
    ):
        """DOCUMENTATION."""
        if open_with == "netcdf":
            TextParser = self._read_netcdf()
        # NOTE: Chunking - polars does have pl.read_csv_batched, but batch_size
        # is not respected: https://github.com/pola-rs/polars/issues/19978
        # alternative: lazy?
        elif open_with == "polars":
            TextParser = self._read_fwf_polars(
                encoding=self.schema["header"].get("encoding"),
                skip_rows=self.skiprows,
                # chunksize=chunksize,
            )
        else:
            raise ValueError("open_with has to be one of ['polars', 'netcdf']")

        # if isinstance(TextParser, (pl.DataFrame, xr.Dataset)):
        df, self.missing_values = self._read_sections(
            TextParser, order, valid, open_with=open_with
        )
        return df, df.select(pl.all().is_null())
        # else:
        #     data_buffer = StringIO()
        #     missings_buffer = StringIO()
        #     isna_buffer = StringIO()
        #     df = pl.DataFrame()
        #     missing_values = pl.DataFrame()
        #     for _, df_ in enumerate(TextParser):
        #         df, missing_values = self._read_sections(df_, order, valid, open_with=open_with)
        #         df_isna = df.select(pl.all().is_null())
        #         # utf-8 is default for polars
        #         missing_values.drop("index").write_csv(
        #             missings_buffer,
        #             include_header=False,
        #         )
        #         df_isna.drop("index").write_csv(
        #             isna_buffer,
        #             include_header=False,
        #             separator=properties.internal_delimiter,
        #             quote_char="\0",
        #         )
        #         df.drop("index").write_csv(
        #             data_buffer,
        #             include_header=False,
        #             separator=properties.internal_delimiter,
        #             quote_char="\0",
        #         )
        #     missings_buffer.seek(0)
        #     self.missing_values = pl.read_csv(
        #         missings_buffer,
        #         columns=missing_values.columns,
        #     )
        #     data_buffer.seek(0)
        #     data = pl.read_csv(
        #         data_buffer,
        #         names=df.columns,
        #         chunksize=self.chunksize,
        #         dtype=object,
        #         parse_dates=self.parse_dates,
        #         delimiter=properties.internal_delimiter,
        #         quotechar="\0",
        #         escapechar="\0",
        #     )
        #     isna_buffer.seek(0)
        #     isna = pl.read_csv(
        #         isna_buffer,
        #         columns=df.columns,
        #         separator=properties.internal_delimiter,
        #         quote_char="\0",
        #     )
        #     return data, isna
