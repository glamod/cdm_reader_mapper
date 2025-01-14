"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import csv
import json
import logging
import os
from copy import deepcopy
from io import StringIO

import numpy as np
import pandas as pd
import xarray as xr

from cdm_reader_mapper.common import pandas_TextParser_hdlr

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
    ):
        if open_with == "pandas":
            df = TextParser.apply(
                lambda x: Configurator(
                    df=x, schema=self.schema, order=order, valid=valid
                ).open_pandas(),
                axis=1,
            )
        elif open_with == "netcdf":
            df = Configurator(
                df=TextParser, schema=self.schema, order=order, valid=valid
            ).open_netcdf()
        else:
            raise ValueError("open_with has to be one of ['pandas', 'netcdf']")

        missing_values_ = df["missing_values"]
        del df["missing_values"]
        df = self._select_years(df)
        missing_values = set_missing_values(pd.DataFrame(missing_values_), df)
        self.columns = df.columns
        df = df.where(df.notnull(), np.nan)
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
                print(decoder_dict)
                decoded = decode_entries(
                    df[section],
                    decoder_dict[section],
                )
                decoded.index = df[section].index
                df[section] = decoded

            converted = convert_entries(
                df[section],
                converter_dict[section],
                **converter_kwargs[section],
            )
            converted.index = df[section].index
            df[section] = converted
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

    def dump_atts(self, out_atts, out_path):
        """Dump attributes to atts.json."""
        if not isinstance(self.data, pd.io.parsers.TextFileReader):
            data = [self.data]
            valid = [self.mask]
        else:
            data = pandas_TextParser_hdlr.make_copy(self.data)
            valid = pandas_TextParser_hdlr.make_copy(self.mask)
        logging.info(f"WRITING DATA TO FILES IN: {out_path}")
        for i, (data_df, valid_df) in enumerate(zip(data, valid)):
            header = False
            mode = "a"
            out_atts_json = {}
            if i == 0:
                mode = "w"
                cols = [x for x in data_df]
                if isinstance(cols[0], tuple):
                    header = [":".join(x) for x in cols]
                    out_atts_json = {
                        ":".join(x): out_atts.get(x) for x in out_atts.keys()
                    }
                else:
                    header = cols
                    out_atts_json = out_atts
            kwargs = {
                "header": header,
                "mode": mode,
                "encoding": "utf-8",
                "index": True,
                "index_label": "index",
                "escapechar": "\0",
            }
            data_df.to_csv(os.path.join(out_path, "data.csv"), **kwargs)
            valid_df.to_csv(os.path.join(out_path, "mask.csv"), **kwargs)

            with open(os.path.join(out_path, "atts.json"), "w") as fileObj:
                json.dump(out_atts_json, fileObj, indent=4)

    def open_data(
        self,
        order,
        valid,
        chunksize,
        open_with="pandas",
    ):
        """DOCUMENTATION."""
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
            df, self.missing_values = self._read_sections(
                TextParser, order, valid, open_with=open_with
            )
            return df, df.isna()
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
