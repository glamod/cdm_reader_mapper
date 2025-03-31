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

import re

import pandas as pd

from cdm_reader_mapper.common import logging_hdlr, pandas_TextParser_hdlr
from cdm_reader_mapper.common.json_dict import collect_json_files, combine_dicts

from . import properties
from .datetime import model_datetimes

_base = f"{properties._base}.station_id"


def _get_id_col(data, imodel, logger) -> int | list | None:
    id_col = properties.metadata_datamodels["id"].get(imodel)
    if not id_col:
        logger.error(
            f"Data model {imodel} ID column not defined in\
                     properties file"
        )
        return
    if not isinstance(id_col, list):
        id_col = [id_col]

    id_col = [col for col in id_col if col in data.columns]
    if not id_col:
        logger.error(f"No ID columns found. Selected columns are {list(data.columns)}")
        return
    if len(id_col) == 1:
        id_col = id_col[0]
    return id_col


def _get_patterns(dck_id_model, blank, dck, data_model_files, logger) -> list[str]:
    pattern_dict = dck_id_model.get("valid_patterns")

    if pattern_dict == {}:
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


def validate_id(data, imodel, blank=False, log_level="INFO") -> pd.Series:
    """DOCUMENTATION."""
    logger = logging_hdlr.init_logger(__name__, level=log_level)
    if isinstance(data, pd.io.parsers.TextFileReader):
        data = pandas_TextParser_hdlr.make_copy(data).read()
    elif not isinstance(data, (pd.DataFrame, pd.Series)):
        logger.error(
            f"Input data must be a pd.DataFrame or pd.Series.\
                     Input data type is {type(data)}"
        )
        return

    mrd = imodel.split("_")
    if len(mrd) < 3:
        logger.error(f"Dataset {imodel} has no deck information.")
        return
    dck = mrd[2]

    if isinstance(data, pd.io.parsers.TextFileReader):
        data = pandas_TextParser_hdlr.make_copy(data).read()

    id_col = _get_id_col(data, mrd[0], logger)
    if id_col is None:
        return

    idSeries = data[id_col]

    data_model_files = collect_json_files(*mrd, base=_base)

    if len(data_model_files) == 0:
        logger.error(f'Input dataset "{imodel}" has no ID deck library')
        return

    id_models = combine_dicts(data_model_files, base=_base)

    dck_id_model = id_models.get(dck)
    if not dck_id_model:
        logger.error(f'Input dck "{dck}" not defined in file {data_model_files}')
        return

    patterns = _get_patterns(dck_id_model, blank, dck, data_model_files, logger)

    na_values = True if "^$" in patterns else False
    combined_compiled = re.compile("|".join(patterns))

    return idSeries.str.match(combined_compiled, na=na_values)


def validate_datetime(data, imodel, log_level="INFO") -> pd.Series:
    """DOCUMENTATiON."""
    # dck input only to be consistent with other validators in the metmetpy module
    logger = logging_hdlr.init_logger(__name__, level=log_level)
    model = imodel.split("_")[0]

    if isinstance(data, pd.io.parsers.TextFileReader):
        data = pandas_TextParser_hdlr.make_copy(data).read()
    elif not isinstance(data, (pd.DataFrame, pd.Series)):
        logger.error(
            f"Input data must be a pd.DataFrame or pd.Series.\
                     Input data type is {type(data)}"
        )
        return

    data_model_datetime = model_datetimes.to_datetime(data, model)

    if not isinstance(data_model_datetime, pd.Series):
        logger.error(
            f'Data model "{model}" datetime conversor not defined in model_datetimes module"'
        )
        return
    elif len(data_model_datetime) == 0:
        data_columns = list(data.columns)
        logger.info(
            f"No columns found for datetime conversion. Selected columns are {data_columns}"
        )
        return
    return data_model_datetime.notna()
