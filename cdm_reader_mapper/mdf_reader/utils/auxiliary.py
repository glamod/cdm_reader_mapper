"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import csv
import json
import logging
import os
from io import StringIO

import numpy as np
import pandas as pd

from cdm_reader_mapper.common import pandas_TextParser_hdlr
from cdm_reader_mapper.common.local import get_files

from .. import properties
from ..schema import schemas
from ..validate import validate
from . import converters, decoders


def convert_float_format(out_dtypes):
    """DOCUMENTATION."""
    out_dtypes_ = {}
    for k, v in out_dtypes.items():
        if "float" in v:
            v = "float"
        out_dtypes_[k] = v
    return out_dtypes_


def convert_dtypes(dtypes):
    """DOCUMENTATION."""
    dtypes = convert_float_format(dtypes)
    parse_dates = []
    for i, element in enumerate(list(dtypes)):
        if dtypes[element] == "datetime":
            parse_dates.append(i)
            dtypes[element] = "object"
    return dtypes, parse_dates


def validate_arg(arg_name, arg_value, arg_type):
    """Validate input argument is as expected type.

    Parameters
    ----------
    arg_name : str
        Name of the argument
    arg_value : arg_type
        Value fo the argument
    arg_type : type
        Type of the argument

    Returns
    -------
    boolean:
        Returns True if type of `arg_value` equals `arg_type`
    """
    if arg_value and not isinstance(arg_value, arg_type):
        logging.error(
            f"Argument {arg_name} must be {arg_type}, input type is {type(arg_value)}"
        )
        return False
    return True


def validate_path(arg_name, arg_value):
    """Validate input argument is an existing directory.

    Parameters
    ----------
    arg_name : str
        Name of the arguemnt
    arg_value : str
        Value of the argument

    Returns
    -------
    boolean
        Returns True if `arg_name` is an existing directory.
    """
    if arg_value and not os.path.isdir(arg_value):
        logging.error(f"{arg_name} could not find path {arg_value}")
        return False
    return True


class Configurator:
    """Class for configurating MDF reader information."""

    def __init__(
        self,
        df,
        schema={},
        order=[],
        valid=[],
    ):
        self.order = order
        self.valid = valid
        self.schema = schema
        self.str_line = df.iloc[0].values[0]

    def _add_field_length(self, element_section, index):
        if "field_length" in element_section.keys():
            field_length = element_section["field_length"]
        else:
            field_length = properties.MAX_FULL_REPORT_WIDTH
            self.length = None
        return index + field_length

    def _validate_sentinal(self, i, section):
        slen = len(self.sentinal)
        str_start = self.str_line[i : i + slen]
        if str_start != self.sentinal:
            self.length = 0
            return i
        else:
            self.sentinal = None
            return self._add_field_length(section, i)

    def _validate_delimited(self, i, j, delimiter, section):
        i = self._skip_delimiter(self.str_line, i, delimiter)
        if self.delimiter_format == "delimited":
            self.delimiters = delimiter
            self.mode = "csv"
            return i, j
        elif self.field_layout == "fixed_width":
            j = self._add_field_length(section, i)
            return i, j
        return None, None

    def _skip_delimiter(self, line, index, delimiter):
        length = len(line)
        while True:
            if index == length:
                break
            if line[index] == delimiter:
                index += 1
            break
        return index

    def _get_index(self, section):
        if len(self.order) == 1:
            return section
        else:
            return (self.order, section)

    def _get_ignore(self):
        if self.order in self.valid:
            ignore = self.sections_dict.get("ignore")
        else:
            ignore = True
        if isinstance(ignore, str):
            ignore = eval(ignore)
        return ignore

    def _get_borders(self, i, j):
        if self.sentinal is not None:
            j = self._validate_sentinal(i, self.sections_dict)
        elif self.delimiter is None:
            j = self._add_field_length(self.sections_dict, i)
        else:
            i, j = self._validate_delimited(i, j, self.delimiter, self.sections_dict)
            self.missing = False
        return i, j

    def _adjust_right_borders(self, j, k):
        if self.length is None:
            self.length = j - k
        if j - k > self.length:
            self.missing = False
            j = k + self.length
        return j, k

    def _get_dtypes(self):
        return properties.pandas_dtypes.get(self.sections_dict.get("column_type"))

    def _get_converters(self):
        return converters.get(self.sections_dict.get("column_type"))

    def _get_conv_kwargs(self):
        return {
            converter_arg: self.sections_dict.get(converter_arg)
            for converter_arg in properties.data_type_conversion_args.get(
                self.sections_dict.get("column_type")
            )
        }

    def _get_decoders(self):
        return decoders.get(self.sections_dict["encoding"]).get(
            self.sections_dict.get("column_type")
        )

    def build(self):
        """Build configuration dictionary."""
        missings = []
        names_fwf = []
        names_csv = []
        lengths = []
        convert = {}
        decode = {}
        na_values = {}
        kwargs = {}
        dtypes = {}
        self.delimiters = None
        first_col_skip = 0
        first_col_name = None
        disable_reads = []
        i = 0
        j = 0
        for order in self.order:
            self.order = order
            header = self.schema["sections"][order]["header"]
            self.sentinal = header.get("sentinal")
            self.sentinal_length = header.get("sentinal_length")
            self.delimiter = header.get("delimiter")
            self.field_layout = header.get("field_layout")
            self.delimiter_format = header.get("format")
            disable_read = header.get("disable_read")
            if disable_read is True:
                names_fwf += [order]
                lengths += [(i, properties.MAX_FULL_REPORT_WIDTH)]
                disable_reads += [order]
                continue
            sections = self.schema["sections"][order]["elements"]
            k = i
            for section in sections.keys():
                self.length = header.get("length")
                self.mode = "fwf"
                self.missing = True
                self.sections_dict = sections[section]
                index = self._get_index(section)
                ignore = self._get_ignore()

                encoding = sections[section].get("encoding")
                na_values[index] = sections[section].get("missing_value")

                i, j = self._get_borders(i, j)

                if i is None:
                    logging.error(
                        f"Delimiter is set to {self.delimiter}. Please specify either format or field_layout in your header schema {header}."
                    )
                    return

                j, k = self._adjust_right_borders(j, k)

                if ignore is not True:
                    if self.mode == "fwf":
                        names_fwf += [index]
                        lengths += [(i, j)]
                    elif self.mode == "csv":
                        names_csv += [index]
                        first_col_skip = i - 1
                        if first_col_name is None:
                            first_col_name = index

                    dtypes[index] = self._get_dtypes()
                    convert[index] = self._get_converters()
                    kwargs[index] = self._get_conv_kwargs()
                    if encoding is not None:
                        decode[index] = self._get_decoders()

                if i == j and self.missing is True:
                    missings.append(index)
                i = j

        dtypes, parse_dates = convert_dtypes(dtypes)

        return {
            "fwf": {
                "names": names_fwf,
                "colspecs": lengths,
                "na_values": na_values,
            },
            "csv": {
                "names": names_csv,
                "delimiter": self.delimiters,
                "first_col_name": first_col_name,
                "first_col_skip": first_col_skip,
            },
            "concat": {
                "dtype": dtypes,
            },
            "convert_decode": {
                "converter_dict": convert,
                "converter_kwargs": kwargs,
                "decoder_dict": decode,
                "dtype": dtypes,
            },
            "self": {
                "dtypes": dtypes,
                "disable_reads": disable_reads,
                "parse_dates": parse_dates,
                "missings": missings,
                "na_values": na_values,
                "delimiters": self.delimiters,
            },
        }


class _FileReader:
    def __init__(
        self,
        source,
        data_model=None,
        data_model_path=None,
    ):
        # 0. VALIDATE INPUT
        if not data_model and not data_model_path:
            logging.error(
                "A valid data model name or path to data model must be provided"
            )
            return
        if not os.path.isfile(source):
            logging.error(f"Can't find input data file {source}")
            return
        if not validate_path("data_model_path", data_model_path):
            return

        self.source = source
        self.data_model = data_model

        # 1. GET DATA MODEL
        # Schema reader will return empty if cannot read schema or is not valid
        # and will log the corresponding error
        # multiple_reports_per_line error also while reading schema
        logging.info("READING DATA MODEL SCHEMA FILE...")
        self.schema = schemas.read_schema(schema_name=data_model)
        if not self.schema:
            return
        if self.data_model:
            model_path = f"{properties._base}.code_tables.{self.data_model}"
            self.code_tables_path = get_files(model_path)
            self.imodel = data_model
        else:
            self.code_tables_path = os.path.join(data_model_path, "code_tables")
            self.imodel = data_model_path

    def _convert_entries(self, series, converter_func, **kwargs):
        return converter_func(series, **kwargs)

    def _decode_entries(self, series, decoder_func):
        return decoder_func(series)

    def _get_configurations(
        self,
        order,
        valid,
    ):
        df = self._read_pandas_fwf(
            encoding=self.schema["header"].get("encoding"),
            widths=[properties.MAX_FULL_REPORT_WIDTH],
            skiprows=self.skiprows,
            nrows=1,
        )
        config_dict = Configurator(
            df, schema=self.schema, order=order, valid=valid
        ).build()

        for attr, val in config_dict["self"].items():
            setattr(self, attr, val)
        del config_dict["self"]

        return config_dict

    def _read_pandas_fwf(
        self,
        **kwargs,
    ):
        return pd.read_fwf(
            self.source,
            header=None,
            quotechar="\0",
            escapechar="\0",
            dtype=object,
            skip_blank_lines=False,
            **kwargs,
        )

    def _read_pandas_csv(
        self,
        delimiter=";",
        first_col_skip=0,
        first_col_name=None,
        **kwargs,
    ):
        def skip_first_col(col):
            return col[first_col_skip:]
        print(delimiter)
        return pd.read_csv(
            self.source,
            header=None,
            delimiter=delimiter,
            quotechar="\0",
            escapechar="\0",
            dtype=object,
            skip_blank_lines=False,
            converters={first_col_name: skip_first_col},
            **kwargs,
        )

    def _concat_dataframes(
        self,
        series,
        chunksize=None,
        dtype=None,
        **kwargs,
    ):
        if series[0] is None:
            return series[1]

        if series[1] is None:
            return series[0]

        if isinstance(series[0], pd.DataFrame):
            return pd.concat(series, **kwargs)

        data_buffer = StringIO()

        for df1, df2 in zip(*series):
            df = pd.concat([df1, df2], **kwargs)
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
        data_buffer.seek(0)
        return pd.read_csv(
            data_buffer,
            names=df.columns,
            chunksize=chunksize,
            delimiter=properties.internal_delimiter,
            quotechar="\0",
            escapechar="\0",
            dtype=object,
        )

    def _convert_and_decode_df(
        self,
        df,
        converter_dict,
        converter_kwargs,
        decoder_dict,
    ):
        self.missing = df.isna()
        for section in converter_dict.keys():
            if section in decoder_dict.keys():
                df[section] = self._decode_entries(
                    df[section],
                    decoder_dict[section],
                )

            df[section] = self._convert_entries(
                df[section],
                converter_dict[section],
                **converter_kwargs[section],
            )
        return df

    def _create_mask(self, df):
        if not hasattr(self, "missing"):
            self.missing = df.isna()
        valid = df.notna()
        mask = self.missing | valid
        mask[self.missings] = np.nan
        return mask

    def _validate_df(self, df):
        mask = self._create_mask(df)
        mask = validate(
            df,
            mask,
            self.schema,
            self.code_tables_path,
            disables=self.disable_reads,
        )
        return mask

    def _dump_atts(self, out_atts, out_path):
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
