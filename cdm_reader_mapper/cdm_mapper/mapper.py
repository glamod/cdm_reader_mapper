"""
Map Common Data Model (CDM).

Created on Thu Apr 11 13:45:38 2019

Maps data contained in a pandas DataFrame (or pd.io.parsers.TextFileReader) to
the C3S Climate Data Store Common Data Model (CDM) header and observational
tables using the mapping information available in the tool's mapping library
for the input data model.

@author: iregon
"""

from __future__ import annotations

import pandas as pd

from cdm_reader_mapper.common import logging_hdlr, pandas_TextParser_hdlr

from . import properties
from ._mappings import map_and_convert


def map_model(
    data,
    imodel,
    cdm_subset=None,
    codes_subset=None,
    null_label="null",
    cdm_complete=True,
    log_level="INFO",
):
    """Map a pandas DataFrame to the CDM header and observational tables.

    Parameters
    ----------
    data: pandas.DataFrame, pd.parser.TextFileReader or io.String
      input data to map.
    imodel: str
      A specific mapping from generic data model to CDM, like map a SID-DCK from IMMA1’s core and attachments to
      CDM in a specific way.
      e.g. ``icoads_r300_d704``
    cdm_subset: list, optional
      subset of CDM model tables to map.
      Defaults to the full set of CDM tables defined for the imodel.
    codes_subset: list, optional
      subset of code mapping tables to map.
      Default to the full set of code mapping tables defined for the imodel.
    log_level: str
      level of logging information to save.
      Defaults to ‘DEBUG’.

    Returns
    -------
    cdm_tables: pandas.DataFrame
      DataFrame with MultiIndex columns (cdm_table, column_name).
    """
    logger = logging_hdlr.init_logger(__name__, level=log_level)
    imodel = imodel.split("_")
    # Check we have imodel registered, leave otherwise
    if imodel[0] not in properties.supported_data_models:
        logger.error("Input data model " f"{imodel[0]}" " not supported")
        return

    # Check input data type and content (empty?)
    # Make sure data is an iterable: this is to homogenize how we handle
    # dataframes and textreaders
    if isinstance(data, pd.DataFrame):
        logger.debug("Input data is a pd.DataFrame")
        if len(data) == 0:
            logger.error("Input data is empty")
            return
        else:
            data = [data]
    elif isinstance(data, pd.io.parsers.TextFileReader):
        logger.debug("Input is a pd.TextFileReader")
        not_empty = pandas_TextParser_hdlr.is_not_empty(data)
        if not not_empty:
            logger.error("Input data is empty")
            return
    else:
        logger.error("Input data type " f"{type(data)}" " not supported")
        return

    return map_and_convert(
        imodel[0],
        *imodel[1:],
        data=data,
        cdm_subset=cdm_subset,
        codes_subset=codes_subset,
        null_label=null_label,
        cdm_complete=cdm_complete,
        logger=logger,
    )
