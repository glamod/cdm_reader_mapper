"""Common Data Model (CDM) reader."""

from __future__ import annotations

import csv
import json
import logging
import os
from io import StringIO as StringIO

import pandas as pd

from cdm_reader_mapper.common import pandas_TextParser_hdlr
from cdm_reader_mapper.common.local import get_files

from . import properties
from .get_sections import get_sections
from .import_data import import_data
from .read_sections import read_sections
from .schema import schemas
from .validate import validate


def convert_float_format(out_dtypes):
    """DOCUMENTATION."""
    out_dtypes_ = {}
    for k, v in out_dtypes.items():
        if "float" in v:
            v = "float"
        out_dtypes_[k] = v
    return out_dtypes_


class MDFFileReader:
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

    def __init__(self, data=None, out_atts=None, valid=None):
        self.data = data
        self.attrs = out_atts
        self.mask = valid


# AUX FUNCTIONS ---------------------------------------------------------------
def ERV(TextParser, read_sections_list, schema, code_tables_path):
    """Extract, read and validate input data.

    Parameters
    ----------
    TextParser : list or pandas.io.parsers.TextFileReader
        The data to extract and read
    read_sections_list : list
        List with subset of data model sections to output
    schema : dict
        Data model schema
    code_tables_path : str
        Path to data model code tables


    Returns
    -------
    data: pandas.DataFrame or pandas.io.parsers.TextFileReader
        Contains the input data extracted and read
    valid: pandas.DataFrame, pandas.io.parsers.TextFileReader
        Contains a boolean mask with the data validation output
    """
    data_buffer = StringIO()
    valid_buffer = StringIO()

    for i_chunk, string_df in enumerate(TextParser):
        # 1. Get a DF with 1 column per sections:
        # - only sections requested, ignore rest
        # - requested NA sections as NaN columns
        # - columns(sections) order as in read_sections_list

        sections_df = get_sections(string_df, schema, read_sections_list)
        # 2. Read elements from sections
        # Along data chunks, resulting data types
        # may vary if gaps, keep track of data dtypes: v1.0
        # This has now been solved by working with Intxx pandas dtypes (nullable integers)
        # Sections are parsed in the same order as sections_df.columns

        [data_df, valid_df, out_dtypes] = read_sections(sections_df, schema)
        out_dtypes = convert_float_format(out_dtypes)
        # 3. Validate data elements

        valid_df = validate(data_df, valid_df, schema, code_tables_path)
        # 4. Save to buffer
        # Writing options from quoting on to prevent data with special characters, like commas, etc, to be quoted
        # https://stackoverflow.com/questions/21147058/pandas-to-csv-output-quoting-issue
        data_df.to_csv(
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

        valid_df.to_csv(
            valid_buffer, header=False, mode="a", encoding="utf-8", index=False
        )

    # Create the output
    # WE'LL NEED TO POSPROCESS THIS WHEN READING MULTIPLE REPORTS PER LINE, IF EVER...
    data_buffer.seek(0)
    valid_buffer.seek(0)
    logging.info("Wrapping output....")
    # Chunksize from the imported TextParser if it is a pd.io.parsers.TextFileReader
    # (source is either pd.io.parsers.TextFileReader or a file with chunksize specified on input):
    # This way it supports direct chunksize property inheritance if the input source was a pd.io.parsers.TextFileReader
    chunksize = (
        TextParser.orig_options["chunksize"]
        if isinstance(TextParser, pd.io.parsers.TextFileReader)
        else None
    )

    # 'datetime' is not a valid pandas dtype: Only on output (on reading) will be then converted (via parse_dates) to datetime64[ns] type,
    # cannot specify 'datetime' (of any kind) here: would fail, need to change to 'object' and tell the date parser where it is
    date_columns = (
        []
    )  # Needs to be the numeric index of the column, as seems not to be able to work with tupples....
    for i, element in enumerate(list(out_dtypes)):
        if out_dtypes.get(element) == "datetime":
            date_columns.append(i)
            out_dtypes.update({element: "object"})

    data = pd.read_csv(
        data_buffer,
        names=data_df.columns,
        chunksize=chunksize,
        dtype=out_dtypes,
        parse_dates=date_columns,
        delimiter=properties.internal_delimiter,
        quotechar="\0",
        escapechar="\0",
    )
    valid = pd.read_csv(valid_buffer, names=data_df.columns, chunksize=chunksize)

    return data, valid


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


def dump_atts(data, valid, out_atts, out_path):
    """Dump attributes to atts.json."""
    enlisted = False
    if not isinstance(data, pd.io.parsers.TextFileReader):
        data = [data]
        valid = [valid]
        enlisted = True
    logging.info(f"WRITING DATA TO FILES IN: {out_path}")
    for i, (data_df, valid_df) in enumerate(zip(data, valid)):
        header = False
        mode = "a"
        if i == 0:
            mode = "w"
            cols = [x for x in data_df]
            if isinstance(cols[0], tuple):
                header = [":".join(x) for x in cols]
                out_atts_json = {":".join(x): out_atts.get(x) for x in out_atts.keys()}
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
        if enlisted:
            data = data[0]
            valid = valid[0]
        else:
            data_ref = data.handles.handle
            valid_ref = valid.handles.handle
            data = pandas_TextParser_hdlr.restore(data_ref, data.orig_options)
            valid = pandas_TextParser_hdlr.restore(valid_ref, valid.orig_options)

        with open(os.path.join(out_path, "atts.json"), "w") as fileObj:
            json.dump(out_atts_json, fileObj, indent=4)

        return data, valid, out_atts


# END AUX FUNCTIONS -----------------------------------------------------------


def read(
    source,
    data_model=None,
    data_model_path=None,
    sections=None,
    chunksize=None,
    skiprows=0,
    out_path=None,
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
    sections : list, optional
        List with subset of data model sections to outpu (default is
        all)
    chunksize : int, optional
        Number of reports per chunk (default is
        no chunking)
    skiprows : int
        Number of initial rows to skip from file (default is 0)
    out_path : str, optional
        Path to output data, valid mask and attributes (default is
        no output)

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

    # 0. VALIDATE INPUT
    if not data_model and not data_model_path:
        logging.error("A valid data model name or path to data model must be provided")
        return
    if not os.path.isfile(source):
        logging.error(f"Can't find input data file {source}")
        return
    if not validate_arg("sections", sections, list):
        return
    if not validate_arg("chunksize", chunksize, int):
        return
    if not validate_arg("skiprows", skiprows, int):
        return
    if not validate_path("data_model_path", data_model_path):
        return
    if not validate_path("out_path", out_path):
        return

    # 1. GET DATA MODEL
    # Schema reader will return empty if cannot read schema or is not valid
    # and will log the corresponding error
    # multiple_reports_per_line error also while reading schema
    logging.info("READING DATA MODEL SCHEMA FILE...")
    schema = schemas.read_schema(
        schema_name=data_model, ext_schema_path=data_model_path
    )
    if not schema:
        return
    if data_model:
        model_path = f"{properties._base}.code_tables.{data_model}"
        code_tables_path = get_files(model_path)
    else:
        code_tables_path = os.path.join(data_model_path, "code_tables")

    # 2. READ AND VALIDATE DATA
    imodel = data_model if data_model else data_model_path
    logging.info(f"EXTRACTING DATA FROM MODEL: {imodel}")

    # 2.1. Subset data model sections to requested sections
    encoding = schema["header"].get("encoding")
    parsing_order = schema["header"].get("parsing_order")
    if not sections:
        sections = [x.get(y) for x in parsing_order for y in x]
        read_sections_list = [y for x in sections for y in x]
    else:
        read_sections_list = sections

    # 2.2 Homogeneize input data to an iterable with dataframes:
    # a list with a single dataframe or a pd.io.parsers.TextFileReader
    logging.info("Getting data string from source...")
    TextParser = import_data(
        source, encoding=encoding, chunksize=chunksize, skiprows=skiprows
    )
    # 2.3. Extract, read and validate data in same loop
    logging.info("Extracting and reading sections")
    data, valid = ERV(TextParser, read_sections_list, schema, code_tables_path)

    # 3. CREATE OUTPUT DATA ATTRIBUTES
    logging.info("CREATING OUTPUT DATA ATTRIBUTES FROM DATA MODEL")
    data_columns = (
        [x for x in data]
        if isinstance(data, pd.DataFrame)
        else data.orig_options["names"]
    )
    out_atts = schemas.df_schema(data_columns, schema)

    # 4. OUTPUT TO FILES IF REQUESTED
    if out_path:
        data, valid, out_atts = dump_atts(data, valid, out_atts, out_path)

    # 5. RETURN DATA
    return MDFFileReader(data, out_atts, valid)
