"""
Validate ID field in a pandas DataFrame.

Created on Tue Jun 25 09:00:19 2019

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

NaN: wil validate to true if blank pattern ('^$') in list, otherwise to False.

If patterns:{} for dck (empty but defined in data model file),
will warn and validate all to True, with NaN to False

@author: iregon
"""

from __future__ import annotations

import json
import re

import pandas as pd

from cdm_reader_mapper.common import logging_hdlr
from cdm_reader_mapper.common.getting_files import get_files

from .. import properties

_base = f"{properties._base}.station_id"
_files = get_files(_base)


def validate(data, dataset, data_model, dck, sid=None, blank=False, log_level="INFO"):
    """DOCUMENTATION."""
    logger = logging_hdlr.init_logger(__name__, level=log_level)

    if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series):
        logger.error(
            f"Input data must be a pd.DataFrame or pd.Series.\
                     Input data type is {type(data)}"
        )
        return

    id_col = properties.metadata_datamodels["id"].get(data_model)
    if not id_col:
        logger.error(
            f"Data model {data_model} ID column not defined in\
                     properties file"
        )
        return
    elif not isinstance(id_col, list):
        id_col = [id_col]

    id_col = [col for col in id_col if col in data.columns]

    if not id_col:
        data_columns = list(data.columns)
        logger.info(f"No ID columns found. Selected columns are {data_columns}")
        return
    elif len(id_col) == 1:
        id_col = id_col[0]

    idSeries = data[id_col]

    for data_model_file in _files.glob(f"{dataset}.json"):
        break
    try:
        data_model_file
    except UnboundLocalError:
        logger.error(f'Input dataset "{dataset}" has no ID deck library')
        return

    with open(data_model_file) as fileObj:
        id_models = json.load(fileObj)

    dck_id_model = id_models.get(dck)
    if not dck_id_model:
        logger.error(f'Input dck "{dck}" not defined in file {data_model_file}')
        return

    pattern_dict = dck_id_model.get("valid_patterns")

    if pattern_dict == {}:
        logger.warning(
            f'Input dck "{dck}" validation patterns are empty in file {data_model_file}'
        )
        logger.warning("Adding match-all regex to validation patterns")
        patterns = [".*?"]
    else:
        patterns = list(pattern_dict.values())

    if blank:
        patterns.append("^$")
        logger.warning("Setting valid blank pattern option to true")
        logger.warning("NaN values will validate to True")

    na_values = True if "^$" in patterns else False
    combined_compiled = re.compile("|".join(patterns))

    return idSeries.str.match(combined_compiled, na=na_values)
