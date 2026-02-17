"""
metmetpy validation package.

Created on Tue Jun 25 09:00:19 2019

Validates the datetime fields of a data model:
    -1. extracts or creates the datetime field of a data model as defined
    in submodule model_datetimes.
    -2. validates to False where NaT: no datetime or conversion to datetime failure

Validation is data model specific.

Output is a boolean series.

Does not account for input dataframes/series stored in TextParsers: as opposed
to correction modules, the output is only a boolean series which is external
to the input data ....

If the datetime conversion (or extraction) for a given data model is not
available in submodule model_datetimes, the module
will return with no output (will break full processing downstream of its
invocation) logging an error.

Reference names of different metadata fields used in the metmetpy modules
and its location column|(section,column) in a data model are
registered in ../properties.py in metadata_datamodels.

NaN, NaT: will validate to False.

Validates ID field in a pandas dataframe against a list of regex patterns.
Output is a boolean series.

Does not account for input dataframes/series stored in TextParsers: as opposed
to correction modules, the output is only a boolean series which is external
to the input data ....

Validations are dataset and deck specific following patterns stored in
 ./lib/dataset.json.: multiple decks in input data are not supported.

If the dataset is not available in the lib, the module
will return with no output (will break full processing downstream of its
invocation) logging an error.

ID corrections assume that the id field read from the source has
been white space stripped. Care must be taken that the way a data model
is read before input to this module, is coherent to the way patterns are
defined for that data model.

NaN: will validate to true if blank pattern ('^$') in list, otherwise to False.

If patterns:{} for dck (empty but defined in data model file),
will warn and validate all to True, with NaN to False

@author: iregon
"""

from __future__ import annotations

import logging
import re

from typing import Iterable

import pandas as pd

from ..common import logging_hdlr
from ..common.iterators import ProcessFunction, process_function
from ..common.json_dict import collect_json_files, combine_dicts

from . import properties
from .datetime import model_datetimes

_base = f"{properties._base}.station_id"


def _get_id_col(
    data: pd.DataFrame,
    imodel: str,
) -> int | list[int] | None:
    """Retrieve the ID column(s) for a given data model from the metadata."""
    id_col = properties.metadata_datamodels["id"].get(imodel)
    if not id_col:
        raise ValueError(
            f"Data model {imodel} ID column not defined in properties file."
        )

    if not isinstance(id_col, list):
        id_col = [id_col]

    id_col = [col for col in id_col if col in data.columns]
    if not id_col:
        raise ValueError(
            f"No ID columns found. Selected columns are {list(data.columns)}"
        )

    if len(id_col) == 1:
        id_col = id_col[0]

    return id_col


def _get_patterns(
    dck_id_model: dict,
    blank: bool,
    dck: str,
    data_model_files: list[str],
    logger: logging.logger,
) -> list[str]:
    """Generate a list of validation patterns for a given deck.."""
    pattern_dict = dck_id_model.get("valid_patterns")

    if not pattern_dict:
        logger.warning(
            f'Input dck "{dck}" validation patterns are empty in file {data_model_files}'
        )
        logger.warning("Adding match-all regex to validation patterns")
        patterns = [".*?"]
    else:
        patterns = list(pattern_dict.values())

    if blank:
        patterns.append("^$")
        logger.warning("Setting valid blank pattern option to true")
        logger.warning("NaN values will validate to True")

    return patterns


def _validate_id(data, mrd, combined_compiled, na_values):
    """Helper function to validate ID."""
    id_col = _get_id_col(data, mrd[0])
    if id_col is None:
        raise ValueError("No ID conversion columns found.")

    id_series = data[id_col]

    return id_series.str.match(combined_compiled, na=na_values)


def _validate_datetime(data: pd.DataFrame | pd.Series, model: str):
    """Helper function to validate datetime."""
    data_model_datetime = model_datetimes.to_datetime(data, model)

    if len(data_model_datetime) == 0:
        raise ValueError(
            f"No columns found for datetime conversion. Selected columns are {list(data.columns)}."
        )
    return data_model_datetime.notna()


@process_function(data_only=True)
def validate_id(
    data: pd.DataFrame | pd.Series | Iterable[pd.DataFrame, pd.Series],
    imodel: str,
    blank: bool = False,
    log_level: str = "INFO",
) -> pd.Series | None:
    """
    Validate ID column(s) in a dataset against deck-specific patterns.

    Parameters
    ----------
    data : pd.DataFrame, pd.Series, or Iterable[pd.DataFrame, pd.Series]
        Input dataset or series containing ID values.
    imodel : str
        Name of internally available data model, e.g., "icoads_r300_d201".
    blank : bool, optional
        If True, empty values are considered valid. Default is False.
    log_level : str, optional
        Logging level. Default is "INFO".

    Returns
    -------
    pd.Series or None
        Boolean Series indicating whether each ID is valid.
        Returns None if validation cannot be performed due to missing data,
        columns, or deck definitions.

    Raises
    ------
    TypeError
        If `data` is not a pd.DataFrame or a pd.Series or an Iterable[pd.DataFrame | pd.Series].
    Value Error
        If dataset `imodel` has no deck information.
        If no ID conversion columns found.
        If input deck is not defined in ID library files.
    FilenotFounderror
        If dataset `imodel` has no ID deck library.

    Notes
    -----
    - Uses `_get_id_col` to determine which column(s) contain IDs.
    - Uses `_get_patterns` to get regex patterns for the deck.
    - Empty values match "^$" pattern if `blank=True`.
    """
    logger = logging_hdlr.init_logger(__name__, level=log_level)

    mrd = imodel.split("_")
    if len(mrd) < 3:
        raise ValueError(f"Dataset {imodel} has no deck information.")

    dck = mrd[2]

    data_model_files = collect_json_files(*mrd, base=_base)

    if len(data_model_files) == 0:
        raise FileNotFoundError(f'Input dataset "{imodel}" has no ID deck library')

    id_models = combine_dicts(data_model_files, base=_base)

    dck_id_model = id_models.get(dck)
    if not dck_id_model:
        raise ValueError(f'Input dck "{dck}" not defined in file {data_model_files}')

    patterns = _get_patterns(dck_id_model, blank, dck, data_model_files, logger)

    na_values = True if "^$" in patterns else False
    combined_compiled = re.compile("|".join(patterns))

    return ProcessFunction(
        data=data,
        func=_validate_id,
        func_kwargs={
            "mrd": mrd,
            "combined_compiled": combined_compiled,
            "na_values": na_values,
        },
        makecopy=False,
    )


@process_function(data_only=True)
def validate_datetime(
    data: pd.DataFrame | pd.Series | Iterable[pd.DataFrame, pd.Series],
    imodel: str,
    blank: bool = False,
    log_level: str = "INFO",
) -> pd.Series:
    """Validate datetime columns in a dataset according to the specified model.

    Parameters
    ----------
    data : pd.DataFrame, pd.Series, or Iterable[pd.DataFrame, pd.Series]
        Input dataset or series containing ID values.
    imodel : str
        Name of internally available data model, e.g., "icoads_r300_d201".
    blank : bool, optional
        If True, empty values are considered valid. Default is False.
    log_level : str, optional
        Logging level. Default is "INFO".

    Returns
    -------
    pd.Series or None
        Boolean Series indicating whether each ID is valid.
        Returns None if validation cannot be performed due to missing data,
        columns, or deck definitions.

    Raises
    ------
    TypeError
        If `data` is not a pd.DataFrame or a pd.Series or an Iterable[pd.DataFrame | pd.Series].
    ValueError
        If no columns found for datetime conversion.
    """
    model = imodel.split("_")[0]

    return ProcessFunction(
        data=data,
        func=_validate_datetime,
        func_kwargs={"model": model},
        makecopy=False,
    )
