"""Common Data Model (CDM) reader."""

from __future__ import annotations

import logging
from io import StringIO as StringIO

import pandas as pd

from .schema import schemas
from .utils.auxiliary import _FileReader, validate_arg, validate_path


class MDFFileReader(_FileReader):
    """Class to represent reader output.

    Parameters
    ----------
    source : str
        The file path to read
    data_model : str, optional
        Name of internally available data model
    data_model_path : str, optional
        Path to external data model


    Attributes
    ----------
    data : pd.DataFrame
        a pandas.DataFrame with the output data
    attrs : dict
        a dictionary with the output data elements attributes
    mask : pd.DataFrame
        a pandas.DataFrame with the output data validation mask
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
        """
        if converter_dict is None:
            converter_dict = self.configurations["convert_decode"]["converter_dict"]
        if converter_kwargs is None:
            converter_kwargs = self.configurations["convert_decode"]["converter_kwargs"]
        if decoder_dict is None:
            decoder_dict = self.configurations["convert_decode"]["decoder_dict"]
        if convert is not True:
            converter_dict = {}
            converter_kwargs = {}
        if decode is not True:
            decoder_dict = {}

        dtype = self.configurations["convert_decode"]["dtype"]
        self.data = self._convert_and_decode_df(
            self.data,
            converter_dict,
            converter_kwargs,
            decoder_dict,
        )
        self.data = self.data.astype(dtype)
        return self

    def validate_entries(
        self,
    ):
        """Validate data entries by using a pre-defined data model.

        Fill attribute `valid` with boolean mask.
        """
        self.mask = self._validate_df(self.data)
        return self

    def read(
        self,
        sections=None,
        skiprows=0,
        out_path=None,
        convert=True,
        decode=True,
        converter_dict=None,
        converter_kwargs=None,
        decoder_kwargs=None,
        validate=True,
        **kwargs,
    ):
        """Read data from disk.

        Parameters
        ----------
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
        decoder_dict: dict, optional
          Functions for decoding values in specific columns.
          If None use information from a pre-defined data model.
        validate: bool, default: True
          Validate data entries by using a pre-defined data model.
        """
        # 0. VALIDATE INPUT
        if not validate_arg("sections", sections, list):
            return
        if not validate_arg("skiprows", skiprows, int):
            return
        if not validate_path("out_path", out_path):
            return

        self.skiprows = skiprows

        # 2. READ AND VALIDATE DATA
        logging.info(f"EXTRACTING DATA FROM MODEL: {self.imodel}")
        # 2.1. Subset data model sections to requested sections
        parsing_order = self.schema["header"].get("parsing_order")
        sections_ = [x.get(y) for x in parsing_order for y in x]
        read_sections_list = [y for x in sections_ for y in x]
        if sections is None:
            sections = read_sections_list

        # 2.2 Homogeneize input data to an iterable with dataframes:
        # a list with a single dataframe
        logging.info("Getting data string from source...")
        # self.configurations = self._get_configurations(read_sections_list, sections)
        self.configurations = self._get_configurations(read_sections_list, sections)
        
        self.data = self._open_data(
            read_sections_list,
            sections,
            open_with=properties.open_file[self.imodel],
            chunksize=chunksize,
        )

        ## 2.3. Extract, read and validate data in same loop
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
    data_model=None,
    data_model_path=None,
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
    source : str
        The file path to read
    data_model : str, optional
        Name of internally available data model
    data_model_path : str, optional
        Path to external data model

    Returns
    -------
    MDFFileReader
        Containing data (``data``), validation mask (``mask``)
        and attributes (``attrs``) corresponding to the information
        from ``source``.
    """
    logging.basicConfig(
        format="%(levelname)s\t[%(asctime)s](%(filename)s)\t%(message)s",
        level=logging.INFO,
        datefmt="%Y%m%d %H:%M:%S",
        filename=None,
    )
    return MDFFileReader(
        source=source, data_model=data_model, data_model_path=data_model_path
    ).read(**kwargs)
