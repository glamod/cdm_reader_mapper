"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import csv
import json
import logging
import os
from io import StringIO

import pandas as pd

from cdm_reader_mapper.common.getting_files import get_files

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
        df=pd.DataFrame(),
        schema={},
        order=[],
        valid=[],
    ):
        self.orders = order
        self.valid = valid
        self.schema = schema
        if len(df) > 0:
            self.str_line = df.iloc[0]
        else:
            self.str_line = ""

    def _add_field_length(self, index):
        if "field_length" in self.sections_dict.keys():
            field_length = self.sections_dict["field_length"]
        else:
            field_length = properties.MAX_FULL_REPORT_WIDTH
        return index + field_length

    def _validate_sentinal(self, i):
        slen = len(self.sentinal)
        str_start = self.str_line[i : i + slen]
        if str_start != self.sentinal:
            self.length = 0
            return i
        else:
            self.sentinal = None
            return self._add_field_length(i)

    def _validate_delimited(self, i, j):
        i = self._skip_delimiter(self.str_line, i)
        if self.delimiter_format == "delimited":
            j = self._next_delimiter(self.str_line, i)
            return i, j
        elif self.field_layout == "fixed_width":
            j = self._add_field_length(i)
            return i, j
        return None, None

    def _skip_delimiter(self, line, index):
        length = len(line)
        while True:
            if index == length:
                break
            if line[index] == self.delimiter:
                index += 1
            break
        return index

    def _next_delimiter(self, line, index):
        while True:
            if index == len(line):
                break
            if line[index] == self.delimiter:
                break
            index += 1
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
            j = self._validate_sentinal(i)
        elif self.delimiter is None:
            j = self._add_field_length(i)
        else:
            i, j = self._validate_delimited(i, j)
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

    def _convert_entries(self, series, converter_func, **kwargs):
        return converter_func(series, **kwargs)

    def _decode_entries(self, series, decoder_func):
        return decoder_func(series)

    def get_configuration(self):
        """Get ICOADS data model specific information."""
        disable_reads = []
        dtypes = {}
        convert = {}
        kwargs = {}
        decode = {}
        for order in self.orders:
            self.order = order
            header = self.schema["sections"][order]["header"]
            disable_read = header.get("disable_read")
            if disable_read is True:
                disable_reads.append(order)
                continue

            sections = self.schema["sections"][order]["elements"]
            for section in sections.keys():
                self.sections_dict = sections[section]
                encoding = sections[section].get("encoding")
                index = self._get_index(section)
                ignore = self._get_ignore()
                if ignore is not True:
                    dtypes[index] = self._get_dtypes()
                    convert[index] = self._get_converters()
                    kwargs[index] = self._get_conv_kwargs()
                    if encoding is not None:
                        decode[index] = self._get_decoders()

        dtypes, parse_dates = convert_dtypes(dtypes)
        return {
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
            },
        }

    def open_file(self):
        """Build configuration dictionary."""
        missings = []
        self.delimiter = None
        i = 0
        j = 0
        data_dict = {}
        dtypes = {}
        for order in self.orders:
            self.order = order
            header = self.schema["sections"][order]["header"]
            self.sentinal = header.get("sentinal")
            self.sentinal_length = header.get("sentinal_length")
            self.delimiter = header.get("delimiter")
            self.field_layout = header.get("field_layout")
            self.delimiter_format = header.get("format")
            disable_read = header.get("disable_read")
            if disable_read is True:
                data_dict[order] = self.str_line[i : properties.MAX_FULL_REPORT_WIDTH]
                continue
            sections = self.schema["sections"][order]["elements"]
            k = i
            for section in sections.keys():
                self.length = header.get("length")
                self.missing = True
                self.sections_dict = sections[section]
                index = self._get_index(section)
                ignore = self._get_ignore()
                na_value = sections[section].get("missing_value")

                i, j = self._get_borders(i, j)

                if i is None:
                    logging.error(
                        f"Delimiter is set to {self.delimiter}. Please specify either format or field_layout in your header schema {header}."
                    )
                    return

                j, k = self._adjust_right_borders(j, k)

                if ignore is not True:
                    data_dict[index] = self.str_line[i:j]
                    dtypes[index] = self._get_dtypes()

                    if not data_dict[index].strip():
                        data_dict[index] = None
                    if data_dict[index] == na_value:
                        data_dict[index] = None

                if i == j and self.missing is True:
                    missings.append(index)

                i = j

        df = pd.Series(data_dict)
        df["missings"] = missings
        return df


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

    def _get_configurations(self, order, valid):
        config_dict = Configurator(
            schema=self.schema, order=order, valid=valid
        ).get_configuration()
        for attr, val in config_dict["self"].items():
            setattr(self, attr, val)
        del config_dict["self"]
        return config_dict

    def _set_missing_values(self, df, ref):
        explode_ = df.explode("missings")
        explode_["index"] = explode_.index
        explode_["values"] = True
        pivots_ = explode_.pivot_table(
            columns="missings",
            index="index",
            values="values",
        )
        missings = pd.DataFrame(data=pivots_, columns=ref.columns, index=ref.index)
        return missings.notna()

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

    def _read_sections(
        self,
        TextParser,
        order,
        valid,
    ):
        df = TextParser.apply(
            lambda x: Configurator(
                df=x, schema=self.schema, order=order, valid=valid
            ).open_file(),
            axis=1,
        )
        missings_ = df["missings"]
        del df["missings"]
        missings = self._set_missing_values(pd.DataFrame(missings_), df)
        self.columns = df.columns
        return df, missings

    def _open_data(
        self,
        order,
        valid,
    ):
        TextParser = self._read_pandas(
            encoding=self.schema["header"].get("encoding"),
            widths=[properties.MAX_FULL_REPORT_WIDTH],
            skiprows=self.skiprows,
        )
        df, self.missings = self._read_sections(TextParser, order, valid)
        return df

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
        if not hasattr(self, "valid"):
            self.valid = df.notna()
        mask = self.missing | self.valid
        mask[self.missings] = None
        return mask

    def _validate_df(self, df):
        mask = self._create_mask(df)
        return validate(
            df,
            mask,
            self.schema,
            self.code_tables_path,
            disables=self.disable_reads,
        )

    def _dump_atts(self, out_atts, out_path):
        """Dump attributes to atts.json."""
        data = [self.data]
        valid = [self.mask]
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
