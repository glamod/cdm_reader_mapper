"""
Initial metmetpy correction package.

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
from collections.abc import Iterable
from typing import Any

import pandas as pd

from ..common import logging_hdlr
from ..common.iterators import ProcessFunction, process_function
from ..common.json_dict import collect_json_files, combine_dicts
from . import properties
from .datetime import correction_functions as corr_f_dt
from .platform_type import correction_functions as corr_f_pt


_base = f"{properties._base}"


def _correct_dt(
    data: pd.DataFrame,
    imodel: str,
    dck: str,
    correction_method: dict[str, dict[str, str]],
    log_level: str = "INFO",
) -> pd.DataFrame:
    """
    Apply deck-specific datetime corrections to a dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Data to be corrected.
    imodel : str
        Name of the ICOADS data model.
    dck : str
        Name of the ICOADS deck.
    correction_method : dict
        DECK-specific correction methods.
    log_level : str
        Logging level (e.g., ``logging.INFO`` or ``'INFO'``).

    Returns
    -------
    pd.DataFrame
        Data containing corrected datetime information.

    Raises
    ------
    TypeError
        If data is a pd.Series.
    AttributeError
        If correction function is not found.
    RuntimeError
        If datetime correction could not be executed.

    Notes
    -----
    - Log INFO if there is no datetime correction to apply to `dck`.
    - Log INFO what datetime correction is applied.
    """
    logger = logging_hdlr.init_logger(__name__, level=log_level)

    if isinstance(data, pd.Series):
        raise TypeError("pd.Series is not supported now.")

    # 1. Optional deck specific corrections
    datetime_correction = correction_method.get(dck, {}).get("function")
    if not datetime_correction:
        logger.info("No datetime correction to apply to deck %s data from data model %s.", dck, imodel)
        return data

    logger.info('Applying "%s" datetime correction', datetime_correction)
    try:
        trans = getattr(corr_f_dt, datetime_correction)
    except AttributeError as err:
        raise AttributeError(f"Correction function '{datetime_correction}' not found.") from err

    try:
        return trans(data)
    except Exception as err:
        raise RuntimeError("func '{trans.__name__}' could not be executed") from err


def _correct_pt(
    data: pd.DataFrame,
    imodel: str,
    dck: str,
    pt_col: str | list[str],
    correction_method: dict[str, dict[str, Any]],
    log_level: str = "INFO",
) -> pd.DataFrame:
    """
    Apply platform-type corrections for a given deck.

    Parameters
    ----------
    data : pd.DataFrame
        Data to be corrected.
    imodel : str
        Name of the ICOADS data model.
    dck : str
        Name of the ICOADS deck.
    pt_col : str or list of str
        Name(s) of the columns containing platform-type information.
    correction_method : dict
        DECK-specific correction methods.
    log_level : str
        Logging level (e.g., ``logging.INFO`` or ``'INFO'``).

    Returns
    -------
    pd.DataFrame
        Data containing corrected platform-type information.

    Raises
    ------
    TypeError
        - If `data` is a pd.Series.
    ValueError
        - If `correction_method` does not contain all required information.
        - If correction function do not contain all required information.
        - If DECK-specific method from `correction_method` is not implemented.

    Notes
    -----
    - Log INFO if there is no platform-type correction to apply to `dck`.
    - LOG INFO if `pt_col` is not found in `data`.
    """
    logger = logging_hdlr.init_logger(__name__, level=log_level)

    if isinstance(data, pd.Series):
        raise TypeError("pd.Series is not supported now.")

    deck_fix = correction_method.get(dck)
    if not deck_fix:
        logger.info("No platform type fixes to apply to deck %s data from dataset %s", dck, imodel)
        return data

    if not isinstance(pt_col, list):
        pt_col = [pt_col]

    pt_col = [col for col in pt_col if col in data.columns]
    if not pt_col:
        logger.info("No platform type found. Selected columns are %s", data.columns)
        return data

    if len(pt_col) == 1:
        pt_col = pt_col[0]

    method = deck_fix.get("method")

    if method == "fillna":
        fillvalue = deck_fix.get("fill_value")
        if fillvalue is None:
            raise ValueError(f'Platform fix "fillna" for deck {dck} requires "fill_value".')
        logger.info("Filling na values with %s", fillvalue)
        data[pt_col] = corr_f_pt.fill_value(data[pt_col], fillvalue)
        return data

    if method == "function":
        transform = deck_fix.get("function")
        if transform is None:
            raise ValueError(f'Platform fix "function" for deck {dck} requires "function" name.')
        logger.info("Applying fix function %s", transform)
        if not hasattr(corr_f_pt, transform):
            raise ValueError(f'Platform fix function "{transform}" not found.')
        trans = getattr(corr_f_pt, transform)
        return trans(data)

    raise ValueError(f'Platform type fix method "{method}" not implemented for deck {dck}.')


@process_function(data_only=True)
def correct_datetime(
    data: pd.DataFrame | Iterable[pd.DataFrame],
    imodel: str,
    log_level: str = "INFO",
    base: str | None = None,
) -> pd.DataFrame | Iterable[pd.DataFrame]:
    """
    Apply ICOADS deck specific datetime corrections.

    Parameters
    ----------
    data : pandas.DataFrame or Iterable[pd.DataFrame]
        Input dataset.
    imodel : str
        Name of internally available data model, e.g. icoads_d300_704.
    log_level : str, default: INFO
        Level of logging information to save.
    base : str, optional
        Base path for datetime correction metadata.
        If None use internal correction path.

    Returns
    -------
    pandas.DataFrame or Iterable[pd.DataFrame]
        A pandas.DataFrame or Iterable[pd.DataFrame] with the adjusted data.

    Raises
    ------
    ValueError
        If `_correct_dt` raises an error during correction.
    TypeError
        If `data` is not a pd.DataFrame or an Iterable[pd.DataFrame].
        If `data` is a pd.Series.
    """
    logger = logging_hdlr.init_logger(__name__, level=log_level)

    if base is None:
        base = f"{_base}.datetime"

    mrd = imodel.split("_")
    if len(mrd) < 3:
        logger.warning("Dataset %s has no deck information.", imodel)
        return data

    dck = mrd[2]

    replacements_method_files = collect_json_files(*mrd, base=base)
    if len(replacements_method_files) == 0:
        logger.warning("Data model %s has no replacements in library", imodel)
        logger.warning("Module will proceed with no attempt to apply id replacements")
        return data

    if isinstance(data, pd.Series):
        raise TypeError("pd.Series is not supported now.")

    correction_method = combine_dicts(replacements_method_files, base=base)

    return ProcessFunction(
        data=data,
        func=_correct_dt,
        func_kwargs={
            "imodel": imodel,
            "dck": dck,
            "correction_method": correction_method,
            "log_level": log_level,
        },
        makecopy=False,
    )


@process_function(data_only=True)
def correct_pt(
    data: pd.DataFrame | Iterable[pd.DataFrame],
    imodel: str,
    log_level: str = "INFO",
    base: str | None = None,
) -> pd.DataFrame | Iterable[pd.DataFrame]:
    """
    Apply ICOADS deck specific platform ID corrections.

    Parameters
    ----------
    data : pandas.DataFrame or Iterable[pd.DataFrame]
        Input dataset.
    imodel : str
        Name of internally available data model, e.g. icoads_d300_704.
    log_level : str, default: INFO
          Level of logging information to save.
    base : str, optional
        Base path for datetime correction metadata.
        If None use internal correction path.

    Returns
    -------
    pandas.DataFrame or Iterable[pd.DataFrame]
        A pandas.DataFrame or Iterable[pd.DataFrame] with the adjusted data.

    Raises
    ------
    ValueError
        If `_correct_pt` raises an error during correction.
        If platform column is not defined in properties file.
    TypeError
        If `data` is not a pd.DataFrame or an Iterable[pd.DataFrame].
        If `data` is a pd.Series.
    """
    logger = logging_hdlr.init_logger(__name__, level=log_level)

    if base is None:
        base = f"{_base}.platform_type"

    mrd = imodel.split("_")
    if len(mrd) < 3:
        logger.warning("Dataset %s has no deck information.", imodel)
        return data

    dck = mrd[2]

    fix_files = collect_json_files(*mrd, base=base)

    if len(fix_files) == 0:
        logger.warning("Dataset %s not included in platform library", imodel)
        return data

    if isinstance(data, pd.Series):
        raise TypeError("pd.Series is not supported now.")

    correction_method = combine_dicts(fix_files, base=base)
    pt_col = properties.metadata_datamodels["platform"].get(mrd[0])

    return ProcessFunction(
        data=data,
        func=_correct_pt,
        func_kwargs={
            "imodel": imodel,
            "dck": dck,
            "pt_col": pt_col,
            "correction_method": correction_method,
            "log_level": log_level,
        },
        makecopy=False,
    )
