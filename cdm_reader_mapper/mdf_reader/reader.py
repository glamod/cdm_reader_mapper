"""Common Data Model (CDM) MDF reader."""

from __future__ import annotations

import ast
import csv
import logging
import os
from io import StringIO as StringIO

import pandas as pd

from cdm_reader_mapper.common.json_dict import open_json_file
from cdm_reader_mapper.common.pandas_TextParser_hdlr import make_copy
from cdm_reader_mapper.core.databundle import DataBundle

from . import properties
from .utils.filereader import FileReader
from .utils.utilities import adjust_dtype, remove_boolean_values, validate_arg
from .utils.validators import validate


class MDFFileReader(FileReader):
    """Class to represent reader output.

    Attributes
    ----------
    data : pd.DataFrame or pd.io.parsers.TextFileReader
        a pandas.DataFrame or pandas.io.parsers.TextFileReader
        with the output data
    mask : pd.DataFrame or pd.io.parsers.TextFileReader
        a pandas.DataFrame or pandas.io.parsers.TextFileReader
        with the output data validation mask
    attrs : dict
        a dictionary with the output data elements attributes
    """

    def __init__(self, *args, **kwargs):
        FileReader.__init__(self, *args, **kwargs)

    def _convert_and_decode(
        self,
        df,
        converter_dict,
        converter_kwargs,
        decoder_dict,
    ) -> pd.DataFrame:
        for section in converter_dict.keys():
            if section not in df.columns:
                continue
            if section in decoder_dict.keys():
                decoded = decoder_dict[section](df[section])
                decoded.index = df[section].index
                df[section] = decoded

            converted = converter_dict[section](
                df[section], **converter_kwargs[section]
            )
            converted.index = df[section].index
            df[section] = converted
        return df

    def _validate(self, df) -> pd.DataFrame:
        return validate(
            data=df,
            imodel=self.imodel,
            ext_table_path=self.ext_table_path,
            schema=self.schema,
            disables=self.disable_reads,
        )

    def convert_and_decode_entries(
        self,
        data,
        convert=True,
        decode=True,
        converter_dict=None,
        converter_kwargs=None,
        decoder_dict=None,
    ) -> pd.DataFrame | pd.io.parsers.TextFileReader:
        """Convert and decode data entries by using a pre-defined data model.

        Overwrite attribute `data` with converted and/or decoded data.

        Parameters
        ----------
        data: pd.DataFrame or pd.io.parsers.TextFileReader
          Data to convert and decode.
        convert: bool, default: True
          If True convert entries by using a pre-defined data model.
        decode: bool, default: True
          If True decode entries by using a pre-defined data model.
        converter_dict: dict of {Hashable: func}, optional
          Functions for converting values in specific columns.
          If None use information from a pre-defined data model.
        converter_kwargs: dict of {Hashable: kwargs}, optional
          Key-word arguments for converting values in specific columns.
          If None use information from a pre-defined data model.
        decoder_dict: dict, optional
          Functions for decoding values in specific columns.
          If None use information from a pre-defined data model.
        """
        if converter_dict is None:
            converter_dict = self.configurations["convert_decode"]["converter_dict"]
        if converter_kwargs is None:
            converter_kwargs = self.configurations["convert_decode"]["converter_kwargs"]
        if decoder_dict is None:
            decoder_dict = self.configurations["convert_decode"]["decoder_dict"]
        if not (convert and decode):
            self.dtypes = "object"
            return data
        if convert is not True:
            converter_dict = {}
            converter_kwargs = {}
        if decode is not True:
            decoder_dict = {}

        if isinstance(data, pd.DataFrame):
            data = self._convert_and_decode(
                data,
                converter_dict,
                converter_kwargs,
                decoder_dict,
            )
        else:
            data_buffer = StringIO()
            TextParser = make_copy(data)
            for i, df_ in enumerate(TextParser):
                df = self._convert_and_decode(
                    df_,
                    converter_dict,
                    converter_kwargs,
                    decoder_dict,
                )
                df.to_csv(
                    data_buffer,
                    header=False,
                    mode="a",
                    encoding=self.encoding,
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
                delimiter=properties.internal_delimiter,
                quotechar="\0",
                escapechar="\0",
            )
        return data

    def validate_entries(
        self, data, validate
    ) -> pd.DataFrame | pd.io.parsers.TextFileReader:
        """Validate data entries by using a pre-defined data model.

        Fill attribute `valid` with boolean mask.
        """
        if validate is not True:
            mask = pd.DataFrame()
        elif isinstance(data, pd.DataFrame):
            mask = self._validate(data)
        else:
            data_buffer = StringIO()
            TextParser_ = make_copy(data)
            for i, df_ in enumerate(TextParser_):
                mask_ = self._validate(df_)
                mask_.to_csv(
                    data_buffer,
                    header=False,
                    mode="a",
                    encoding=self.encoding,
                    index=False,
                )
            data_buffer.seek(0)
            mask = pd.read_csv(
                data_buffer,
                names=df_.columns,
                chunksize=self.chunksize,
            )
        return mask

    def remove_boolean_values(
        self, data
    ) -> pd.DataFrame | pd.io.parsers.TextFileReader:
        """DOCUMENTATION"""
        if isinstance(data, pd.DataFrame):
            data = data.map(remove_boolean_values)
            dtype = adjust_dtype(self.dtypes, data)
            return data.astype(dtype)
        else:
            data_buffer = StringIO()
            TextParser = make_copy(data)
            for i, df_ in enumerate(TextParser):
                df = df_.map(remove_boolean_values)
                dtype = adjust_dtype(self.dtypes, df)
                date_columns = []
                df.to_csv(
                    data_buffer,
                    header=False,
                    mode="a",
                    encoding=self.encoding,
                    index=False,
                    quoting=csv.QUOTE_NONE,
                    sep=properties.internal_delimiter,
                    quotechar="\0",
                    escapechar="\0",
                )
            date_columns = []
            for i, element in enumerate(list(dtype)):
                if dtype.get(element) == "datetime":
                    date_columns.append(i)
            dtype = adjust_dtype(dtype, df)
            data_buffer.seek(0)
            data = pd.read_csv(
                data_buffer,
                names=df.columns,
                chunksize=self.chunksize,
                dtype=dtype,
                parse_dates=date_columns,
                delimiter=properties.internal_delimiter,
                quotechar="\0",
                escapechar="\0",
            )
        return data

    def read(
        self,
        chunksize=None,
        sections=None,
        skiprows=0,
        convert=True,
        decode=True,
        converter_dict=None,
        converter_kwargs=None,
        validate=True,
        encoding: str | None = None,
        **kwargs,
    ) -> DataBundle:
        """Read data from disk.

        Parameters
        ----------
        chunksize : int, optional
          Number of reports per chunk.
        sections : list, optional
          List with subset of data model sections to output, optional
          If None read pre-defined data model sections.
        skiprows : int
          Number of initial rows to skip from file, default: 0
        convert: bool, default: True
          If True convert entries by using a pre-defined data model.
        decode: bool, default: True
          If True decode entries by using a pre-defined data model.
        converter_dict: dict of {Hashable: func}, optional
          Functions for converting values in specific columns.
          If None use information from a pre-defined data model.
        converter_kwargs: dict of {Hashable: kwargs}, optional
          Key-word arguments for converting values in specific columns.
          If None use information from a pre-defined data model.
        validate: bool, default: True
          Validate data entries by using a pre-defined data model.
        encoding: str, optional
          Encoding of the input file, overrides the value in the imodel schema
        """
        # 0. VALIDATE INPUT
        if not validate_arg("sections", sections, list):
            return
        if not validate_arg("chunksize", chunksize, int):
            return
        if not validate_arg("skiprows", skiprows, int):
            return

        self.chunksize = chunksize
        self.skiprows = skiprows

        # 2. READ AND VALIDATE DATA
        logging.info(f"EXTRACTING DATA FROM MODEL: {self.imodel}")
        # 2.1. Subset data model sections to requested sections
        parsing_order = self.schema["header"].get("parsing_order")
        sections_ = [x.get(y) for x in parsing_order for y in x]
        read_sections_list = [y for x in sections_ for y in x]
        if sections is None:
            sections = read_sections_list

        # 2.2 Homogenize input data to an iterable with dataframes:
        # a list with a single dataframe or a pd.io.parsers.TextFileReader
        logging.info("Getting data string from source...")
        self.configurations = self.get_configurations(read_sections_list, sections)
        self.encoding = encoding or self.encoding
        data = self.open_data(
            read_sections_list,
            sections,
            # INFO: Set default as "pandas" to account for custom schema
            open_with=properties.open_file.get(self.imodel, "pandas"),
            encoding=self.encoding,
            chunksize=chunksize,
        )

        # 2.3. Extract, read and validate data in same loop
        logging.info("Extracting and reading sections")
        data = self.convert_and_decode_entries(
            data,
            convert=convert,
            decode=decode,
        )
        mask = self.validate_entries(data, validate)

        # 3. Create output DataBundle object
        logging.info("Create an output DataBundle object")
        data = self.remove_boolean_values(data)
        return DataBundle(
            data=data,
            columns=self.columns,
            dtypes=self.dtypes,
            parse_dates=self.parse_dates,
            encoding=self.encoding,
            mask=mask,
            imodel=self.imodel,
        )


def read_mdf(
    source,
    imodel=None,
    ext_schema_path=None,
    ext_schema_file=None,
    ext_table_path=None,
    year_init=None,
    year_end=None,
    encoding: str | None = None,
    **kwargs,
) -> DataBundle:
    """Read data files compliant with a user specific data model.

    Reads a data file to a pandas DataFrame using a pre-defined data model.
    Read data is validates against its data model producing a boolean mask
    on output.

    The data model needs to be input to the module as a named model
    (included in the module) or as the path to a valid data model.

    Parameters
    ----------
    source: str
        The file (including path) to be read.
    imodel: str, optional
        Name of internally available input data model.
        e.g. icoads_r300_d704
    ext_schema_path: str, optional
        The path to the external input data model schema file.
        The schema file must have the same name as the directory.
        One of ``imodel`` and ``ext_schema_path`` or ``ext_schema_file`` must be set.
    ext_schema_file: str, optional
        The external input data model schema file.
        One of ``imodel`` and ``ext_schema_path`` or ``ext_schema_file`` must be set.
    ext_table_path: str, optional
        The path to the external input data model code tables.
    year_init: str or int, optional
        Left border of time axis.
    year_end: str or int, optional
        Right border of time axis.
    encoding : str, optional
        The encoding of the input file. Overrides the value in the imodel schema file.

    Returns
    -------
    cdm_reader_mapper.DataBundle

    See Also
    --------
    read: Read either original marine-meteorological or MDF data or CDM tables from disk.
    read_data : Read MDF data and validation mask from disk.
    read_tables : Read CDM tables from disk.
    write: Write either MDF data or CDM tables to disk.
    write_data : Write MDF data and validation mask to disk.
    write_tables : Write CDM tables to disk.
    """

    def get_list_element(lst, idx):
        try:
            return lst[idx]
        except IndexError:
            return None

    logging.basicConfig(
        format="%(levelname)s\t[%(asctime)s](%(filename)s)\t%(message)s",
        level=logging.INFO,
        datefmt="%Y%m%d %H:%M:%S",
        filename=None,
    )
    return MDFFileReader(
        source=source,
        imodel=imodel,
        ext_schema_path=ext_schema_path,
        ext_schema_file=ext_schema_file,
        ext_table_path=ext_table_path,
        year_init=year_init,
        year_end=year_end,
    ).read(encoding=encoding, **kwargs)


def read_data(
    source,
    mask=None,
    info=None,
    imodel=None,
    col_subset=None,
    encoding: str | None = None,
    **kwargs,
) -> DataBundle:
    """Read MDF data which is already on a pre-defined data model.

    Parameters
    ----------
    source: str
        The data file (including path) to be read.
    mask: str, optional
        The validation file (including path) to be read.
    info: str, optional
        The information file (including path) to be read.
    imodel: str, optional
        Name of internally available input data model.
        e.g. icoads_r300_d704
    col_subset: str, tuple or list, optional
        Specify the section or sections of the file to write.

        - For multiple sections of the tables:
          e.g col_subset = [columns0,...,columnsN]

        - For a single section:
          e.g. list type object col_subset = [columns]

        Column labels could be both string or tuple.
    encoding : str, optional
        The encoding of the input file. Overrides the value in the imodel schema file.

    Returns
    -------
    cdm_reader_mapper.DataBundle

    See Also
    --------
    read: Read original marine-meteorological data as well as MDF data or CDM tables from disk.
    read_mdf : Read original marine-meteorological data from disk.
    read_tables : Read CDM tables from disk.
    write: Write both MDF data or CDM tables to disk.
    write_data : Write MDF data and validation mask to disk.
    write_tables : Write CDM tables to disk.
    """

    def _update_column_labels(columns):
        columns_ = []
        for col in columns:
            try:
                col_ = ast.literal_eval(col)
            except SyntaxError:
                col_ = tuple(col.split(":"))
            except ValueError:
                col_ = col
            columns_.append(col_)
        return columns_

    def _read_csv(ifile, col_subset=None, **kwargs):
        if ifile is None:
            return pd.DataFrame()
        if not os.path.isfile(ifile):
            return pd.DataFrame()
        df = pd.read_csv(ifile, delimiter=",", **kwargs)
        df.columns = _update_column_labels(df.columns)
        if col_subset is not None:
            df = df[col_subset]
        return df

    if info is None:
        info_dict = {}
    else:
        info_dict = open_json_file(info)

    dtype = info_dict.get("dtypes", "object")
    parse_dates = info_dict.get("parse_dates", False)

    data = _read_csv(
        source,
        col_subset=col_subset,
        dtype=dtype,
        parse_dates=parse_dates,
        encoding=encoding,
    )
    mask = _read_csv(mask, col_subset=col_subset)
    return DataBundle(
        data=data,
        columns=data.columns,
        dtypes=dtype,
        parse_dates=parse_dates,
        mask=mask,
        imodel=imodel,
        encoding=encoding,
    )
