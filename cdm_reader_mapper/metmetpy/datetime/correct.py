"""
metmetpy correction package.

Created on Tue Jun 25 09:00:19 2019

Corrects datetime fields from a given deck in a data model.

To account for dataframes stored in TextParsers and for eventual use of data columns other
than those to be fixed in this or other metmetpy modules,
the input and output are the full data set.

Correctionsare data model and deck specific and are registered
in ./lib/data_model.json: multiple decks in the same input data are not
supported.

Reference names of different metadata fields used in the metmetpy modules
and its location column|(section,column) in a data model are
registered in ../properties.py in metadata_datamodels.

If the data model is not available in ./lib it is assumed to no corrections are
needed.
If the data model is not available in metadata_models, the module
will return with no output (will break full processing downstream of its
invocation) logging an error.


@author: iregon
"""

from __future__ import annotations

import json
import os
from io import StringIO

import pandas as pd

from cdm_reader_mapper.common import logging_hdlr, pandas_TextParser_hdlr
from cdm_reader_mapper.common.getting_files import get_files

from .. import properties
from . import correction_functions

_base = f"{properties._base}.datetime"
_files = get_files(_base)


def correct_it(data, data_model, deck, log_level="INFO"):
    """DOCUMENTAION."""
    logger = logging_hdlr.init_logger(__name__, level=log_level)

    # 1. Optional deck specific corrections
    for correction_method_file in _files.glob(f"{data_model}.json"):
        break
    if not os.path.isfile(correction_method_file):
        logger.info(f"No datetime corrections {data_model}")
    else:
        with open(correction_method_file) as fileObj:
            correction_method = json.load(fileObj)
        datetime_correction = correction_method.get(deck, {}).get("function")
        if not datetime_correction:
            logger.info(
                f"No datetime correction to apply to deck {deck} data from data\
                        model {data_model}"
            )
        else:
            logger.info(f'Applying "{datetime_correction}" datetime correction')
            try:
                # trans = eval("datetime_functions_mdl." + datetime_correction)
                trans = getattr(correction_functions, datetime_correction)
                trans(data)
            except Exception:
                logger.error("Applying correction ", exc_info=True)
                return

    return data


def correct(data, data_model, deck, log_level="INFO"):
    """DOCUMENTATION."""
    logger = logging_hdlr.init_logger(__name__, level=log_level)
    for replacements_method_file in _files.glob(f"{data_model}.json"):
        break
    if not os.path.isfile(replacements_method_file):
        logger.warning(f"Data model {data_model} has no replacements in library")
        logger.warning(
            "Module will proceed with no attempt to apply id\
                       replacements".format()
        )

    data = correct_it(data, data_model, deck, log_level="INFO")
    return data
