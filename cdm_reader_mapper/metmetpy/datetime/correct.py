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

from cdm_reader_mapper.common import logging_hdlr
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


def correct(data, data_model, log_level="INFO"):
    """Apply ICOADS deck specific datetime corrections.

    Parameters
    ----------
    data: pd.DataFrame or pd.io.parsers.TextFileReader
        Input dataset.
    data_model: str
        Name of internally available data model.
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
    mrd = data_model.split("_")
    if len(mrd) < 3:
        logger.warning(f"Dataset {data_model} has to deck information.")
        return data
    dck = mrd[2]

    replacements_method_files = collect_json_files(*mrd, base=_base)

    if len(replacements_method_files) == 0:
        logger.warning(f"Data model {data_model} has no replacements in library")
        logger.warning("Module will proceed with no attempt to apply id replacements")
        return data

    data = correct_it(data, data_model, dck, correction_method, log_level="INFO")
    return data
