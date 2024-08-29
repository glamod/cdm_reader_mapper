"""
metmetpy platfrom_type correction package.

Created on Tue Jun 25 09:00:19 2019

Corrects the platform type field of data from a given data model. To account
for dataframes stored in TextParsers and for eventual use of data columns other
than those to be fixed (dependencies) in this or other metmetpy modules,
the input and output are the full data set.

Correction to apply is data model and deck specific and is registered in
./lib/data_model.json: multiple decks in input data are not supported.

The ones in imma1 (only available so far) come from
Liz's construct_monthly_files.R. PT corrections are rather simple with no
dependencies other than dck and can be basically classified in:

    - for a set of decks, set missing PT to known type 5.
    - for a set of decks, set PT=4,5 to 99: state nan. This decks are mainly
        buoys, misc (rigs, etc...) Why?, is it just to filter out from the
        processing ship data from decks where you do not expect to have them? This
        does not apply here, it is not an error of the metadata per se, we will
        select PT on a deck specific basis, SO THIS IS OBVIOUSLY NOT APPLIED HERE
    - for a set of sid-dck (2), with ship data, numeric id thought to be buoy
        (moored-6 of drifting-7, ?): set to 6,7? which, not really important so far,
        we just want to make sure it is not flagged as a ship....


Reference names of different metadata fields used in the metmetpy modules
and its location column|(section,column) in a data model are
registered in ../properties.py in metadata_datamodels.

If the data model is not available in ./lib or in metadata_models, the module
will return with no output (will break full processing downstream of its
invocation) logging an error.

@author: iregon
"""

from __future__ import annotations

import json
from io import StringIO

import pandas as pd

from cdm_reader_mapper.common import logging_hdlr
from cdm_reader_mapper.common.getting_files import get_files

from .. import properties
from . import gdac_r0000, icoads_r3000, icoads_r3000_NRT
from .correction_functions import fill_value

_base = f"{properties._base}.platform_type"
_files = get_files(_base)

_fix_function = {
    "gdac_r0000": gdac_r0000,
    "icoads_r3000": icoads_r3000,
    "icoads_r3000_NRT": icoads_r3000_NRT,
}


def correct_it(data, dataset, data_model, deck, pt_col, fix_methods, log_level="INFO"):
    """DOCUMENTATION."""
    logger = logging_hdlr.init_logger(__name__, level=log_level)

    deck_fix = fix_methods.get(deck)
    if not deck_fix:
        logger.info(
            f"No platform type fixes to apply to deck {deck} \
                    data from dataset {dataset}"
        )
        return data
    elif not isinstance(pt_col, list):
        pt_col = [pt_col]

    pt_col = [col for col in pt_col if col in data.columns]
    if not pt_col:
        data_columns = list(data.columns)
        logger.info(f"No platform type found. Selected columns are {data_columns}")
        return data
    elif len(pt_col) == 1:
        pt_col = pt_col[0]

    #    Find fix method
    if deck_fix.get("method") == "fillna":
        fillvalue = deck_fix.get("fill_value")
        logger.info(f"Filling na values with {fillvalue}")
        data[pt_col] = fill_value(data[pt_col], fillvalue)
        return data
    elif deck_fix.get("method") == "function":
        transform = deck_fix.get("function")
        logger.info(f"Applying fix function {transform}")
        fix_function = _fix_function[dataset]
        trans = getattr(fix_function, transform)
        return trans(data)
    else:
        logger.error(
            'Platform type fix method "{}" not implemented'.format(
                deck_fix.get("method")
            )
        )
        return


def correct(data, dataset, data_model, deck, log_level="INFO"):
    """Apply ICOADS deck specific platform ID corrections.

    Parameters
    ----------
    data: pd.DataFrame or pd.io.parsers.TextFileReader
        Input dataset.
    dataset: str
        Name of metmetpy specific data model.
    data_model: str
        Name of the ICOADS data model.
    deck: str
        Name of the ICOADS model deck.
    log_level: str
      level of logging information to save.
      Default: INFO

    Returns
    -------
    pd.DataFrame or pd.io.parsers.TextFileReader
        a pandas.DataFrame or pandas.io.parsers.TextFileReader
        with the adjusted data
    """
    logger = logging_hdlr.init_logger(__name__, level=log_level)

    fix_file = _files.glob(f"{dataset}.json")
    fix_file = [f for f in fix_file]
    if not fix_file:
        logger.warning(f"Dataset {dataset} not included in platform library")
        return data
    else:
        with open(fix_file[0]) as fileObj:
            fix_methods = json.load(fileObj)

    pt_col = properties.metadata_datamodels["platform"].get(data_model)

    if not pt_col:
        logger.error(
            f"Data model {data_model} platform column not defined in\
                     properties file"
        )
        return

    if isinstance(data, pd.DataFrame):
        data = correct_it(
            data, dataset, data_model, deck, pt_col, fix_methods, log_level="INFO"
        )
        return data
    elif isinstance(data, pd.io.parsers.TextFileReader):
        read_params = [
            "chunksize",
            "names",
            "dtype",
            "parse_dates",
            "date_parser",
            "infer_datetime_format",
        ]
        read_dict = {x: data.orig_options.get(x) for x in read_params}
        buffer = StringIO()
        for df in data:
            df = correct_it(
                df, dataset, data_model, deck, pt_col, fix_methods, log_level="INFO"
            )
            df.to_csv(buffer, header=False, index=False, mode="a")

        buffer.seek(0)
        return pd.read_csv(buffer, **read_dict)
