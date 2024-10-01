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

from io import StringIO

import pandas as pd

from cdm_reader_mapper.common import logging_hdlr, pandas_TextParser_hdlr
from cdm_reader_mapper.common.json_dict import collect_json_files, combine_dicts

from .. import properties
from . import correction_functions

_base = f"{properties._base}.datetime"


def correct_it(data, data_model, dck, correction_method, log_level="INFO"):
    """DOCUMENTATION."""
    logger = logging_hdlr.init_logger(__name__, level=log_level)

    # 1. Optional deck specific corrections
    datetime_correction = correction_method.get(dck, {}).get("function")
    if not datetime_correction:
        logger.info(
            f"No datetime correction to apply to deck {dck} data from data\
                        model {data_model}"
        )
    else:
        logger.info(f'Applying "{datetime_correction}" datetime correction')
        try:
            trans = getattr(correction_functions, datetime_correction)
            trans(data)
        except Exception:
            logger.error("Applying correction ", exc_info=True)
            return

    return data


def correct(data, imodel, log_level="INFO"):
    """Apply ICOADS deck specific datetime corrections.

    Parameters
    ----------
    data: pd.DataFrame or pd.io.parsers.TextFileReader
        Input dataset.
    imodel: str
        Name of internally available data model.
        e.g. icoads_d300_704
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
    mrd = imodel.split("_")
    if len(mrd) < 3:
        logger.warning(f"Dataset {imodel} has to deck information.")
        return data
    dck = mrd[2]

    replacements_method_files = collect_json_files(*mrd, base=_base)

    if len(replacements_method_files) == 0:
        logger.warning(f"Data model {imodel} has no replacements in library")
        logger.warning("Module will proceed with no attempt to apply id replacements")
        return data

    correction_method = combine_dicts(replacements_method_files, base=_base)

    if isinstance(data, pd.DataFrame):
        data = correct_it(data, imodel, dck, correction_method, log_level="INFO")
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
        data_ = pandas_TextParser_hdlr.make_copy(data)
        for df in data_:
            df = correct_it(df, imodel, dck, correction_method, log_level="INFO")
            df.to_csv(buffer, header=False, index=False, mode="a")
        buffer.seek(0)
        return pd.read_csv(buffer, **read_dict)
