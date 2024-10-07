"""Common Data Model (CDM) reader."""

from __future__ import annotations

import csv
import logging
from io import StringIO as StringIO

import pandas as pd

from cdm_reader_mapper.common.pandas_TextParser_hdlr import make_copy

from . import properties
from .schemas import schemas
from .utils.auxiliary import _FileReader, validate_arg, validate_path


class MDFFileReader(_FileReader):
    """Class to represent reader output.

    Attributes
    ----------
    data : pd.DataFrame or pd.io.parsers.TextFileReader
        a pandas.DataFrame or pandas.io.parsers.TextFileReader
        with the output data
    attrs : dict
        a dictionary with the output data elements attributes
    mask : pd.DataFrame or pd.io.parsers.TextFileReader
        a pandas.DataFrame or pandas.io.parsers.TextFileReader
        with the output data validation mask
    """

    def __init__(self, *args, **kwargs):
        _FileReader.__init__(self, *args, **kwargs)

    def convert_and_decode_entries(
        self,
        convert=True,
        decode=True,
        converter_dict=None,
        converter_kwargs=None,
        decoder_dict=None,
        dtype=None,
    ):
        """Convert and decode data entries by using a pre-defined data model.

        Overwrite attribute `data` with converted and/or decoded data.

        Parameters
        ----------
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
        dtype: dtype or dict of {Hashable: dtype}, optional
          Data type(s) to apply to either the whole dataset or individual columns.
          If None use information from a pre-defined data model.
          Use only if data is read with chunksizes.
        """
        if converter_dict is None:
            converter_dict = self.configurations["convert_decode"]["converter_dict"]
        if converter_kwargs is None:
            converter_kwargs = self.configurations["convert_decode"]["converter_kwargs"]
        if decoder_dict is None:
            decoder_dict = self.configurations["convert_decode"]["decoder_dict"]
        if dtype is None:
            dtype = self.configurations["convert_decode"]["dtype"]
        if convert is not True:
            converter_dict = {}
            converter_kwargs = {}
        if decode is not True:
            decoder_dict = {}

        if isinstance(self.data, pd.DataFrame):
            dtype = self._adjust_dtype(dtype, self.data)
            data = self._convert_and_decode_df(
                self.data,
                converter_dict,
                converter_kwargs,
                decoder_dict,
            )
            self.data = data.astype(dtype)
        else:
            data_buffer = StringIO()
            TextParser = make_copy(self.data)
            for i, df_ in enumerate(TextParser):
                df = self._convert_and_decode_df(
                    df_,
                    converter_dict,
                    converter_kwargs,
                    decoder_dict,
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
            date_columns = []
            for i, element in enumerate(list(dtype)):
                if dtype.get(element) == "datetime":
                    date_columns.append(i)
            dtype = self._adjust_dtype(dtype, df)
            data_buffer.seek(0)
            self.data = pd.read_csv(
                data_buffer,
                names=df.columns,
                chunksize=self.chunksize,
                dtype=dtype,
                parse_dates=date_columns,
                delimiter=properties.internal_delimiter,
                quotechar="\0",
                escapechar="\0",
            )
        return self

    def validate_entries(
        self,
    ):
        """Validate data entries by using a pre-defined data model.

        Fill attribute `valid` with boolean mask.
        """
        if isinstance(self.data, pd.DataFrame):
            self.mask = self._validate_df(self.data, isna=self.isna)

        else:
            data_buffer = StringIO()
            TextParser_ = make_copy(self.data)
            TextParser_isna_ = make_copy(self.isna)
            for i, (df_, isna_) in enumerate(zip(TextParser_, TextParser_isna_)):
                mask_ = self._validate_df(df_, isna=isna_)
                mask_.to_csv(
                    data_buffer,
                    header=False,
                    mode="a",
                    encoding="utf-8",
                    index=False,
                )
            data_buffer.seek(0)
            self.mask = pd.read_csv(
                data_buffer,
                names=df_.columns,
                chunksize=self.chunksize,
            )
        return self

    def read(
        self,
        chunksize=None,
        sections=None,
        skiprows=0,
        out_path=None,
        convert=True,
        decode=True,
        converter_dict=None,
        converter_kwargs=None,
        validate=True,
        **kwargs,
    ):
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
        out_path : str, optional
          Path to dump output data, valid mask and attributes.
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
        """
        # 0. VALIDATE INPUT
        if not validate_arg("sections", sections, list):
            return
        if not validate_arg("chunksize", chunksize, int):
            return
        if not validate_arg("skiprows", skiprows, int):
            return
        if not validate_path("out_path", out_path):
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
        self.configurations = self._get_configurations(read_sections_list, sections)
        self.data, self.isna = self._open_data(
            read_sections_list,
            sections,
            # INFO: Set default as "pandas" to account for custom schema
            open_with=properties.open_file.get(self.imodel, "pandas"),
            chunksize=chunksize,
        )

        # 2.3. Extract, read and validate data in same loop
        # logging.info("Extracting and reading sections")

        if convert or decode:
            self.convert_and_decode_entries(
                convert=convert,
                decode=decode,
            )

        if validate is True:
            self.validate_entries()
        else:
            self.mask = pd.DataFrame()

        # 3. CREATE OUTPUT DATA ATTRIBUTES
        logging.info("CREATING OUTPUT DATA ATTRIBUTES FROM DATA MODEL")
        data_columns = (
            [x for x in self.data]
            if isinstance(self.data, pd.DataFrame)
            else self.data.orig_options["names"]
        )
        out_atts = schemas.df_schema(data_columns, self.schema)

        # 4. OUTPUT TO FILES IF REQUESTED
        if out_path:
            self._dump_atts(out_atts, out_path)
        self.attrs = out_atts
        return self


# END AUX FUNCTIONS -----------------------------------------------------------


def read(
    source,
    imodel=None,
    ext_schema_path=None,
    ext_schema_file=None,
    ext_table_path=None,
    year_init=None,
    year_end=None,
    **kwargs,
):
    """Read data files compliant with a user specific data model.

    Reads a data file to a pandas DataFrame using a pre-defined data model.
    Read data is validates against its data model producing a boolean mask
    on output.

    The data model needs to be input to the module as a named model
    (included in the module) or as the path to a valid data model.

    Parameters
    ----------
    source: str
        The file (including path) to be read
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

    Returns
    -------
    MDFFileReader
        Containing data (``data``), validation mask (``mask``)
        and attributes (``attrs``) corresponding to the information
        from ``source``.
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
    ).read(**kwargs)
