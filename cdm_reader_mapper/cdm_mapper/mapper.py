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

from io import StringIO

import numpy as np
import pandas as pd

from cdm_reader_mapper.common import logging_hdlr, pandas_TextParser_hdlr

from . import properties
from .codes.codes_hdlr import codes_hdlr
from .mappings import mapping_functions
from .tables.tables_hdlr import tables_hdlr


def drop_duplicates(df):
    """Drop duplicates from list."""

    def list_to_tuple(v):
        if isinstance(v, list):
            v = tuple(v)
        return v

    def tuple_to_list(v):
        if isinstance(v, tuple):
            v = list(v)
        return v

    dtypes = df.dtypes
    df = df.map(list_to_tuple)
    df = df.drop_duplicates(ignore_index=True)
    df = df.map(tuple_to_list)
    return df.astype(dtypes)


def _map_to_df(m, x):
    if not isinstance(m, dict):
        return
    for x_ in x:
        if x_ in m.keys():
            v = m[x_]
            if isinstance(v, dict):
                m = v
                continue
            else:
                return v
        return


def _decimal_places(
    cdm_tables,
    decimal_places,
    cdm_key,
    table,
    imodel_functions,
):
    if decimal_places is not None:
        if isinstance(decimal_places, int):
            cdm_tables[table]["atts"][cdm_key].update(
                {"decimal_places": decimal_places}
            )
        else:
            cdm_tables[table]["atts"][cdm_key].update(
                {"decimal_places": getattr(imodel_functions, decimal_places)()}
            )
    return cdm_tables


def _write_csv_files(
    idata,
    mapping,
    logger,
    table,
    cols,
    imodel_functions,
    imodel_code_tables,
    cdm_tables,
):
    table_df_i = pd.DataFrame(
        index=idata.index, columns=mapping.keys()
    )  # We cannot predifine column based dtypes here!
    logger.debug(f"Table: {table}")
    for cdm_key, imapping in mapping.items():
        logger.debug(f"\tElement: {cdm_key}")
        isEmpty = False
        elements = imapping.get("elements")
        transform = imapping.get("transform")
        kwargs = imapping.get("kwargs", {})
        code_table = imapping.get("code_table")
        default = imapping.get("default")
        fill_value = imapping.get("fill_value")
        decimal_places = imapping.get("decimal_places")
        if elements:
            # make sure they are clean and conform to their atts (tie dtypes)
            # we'll only let map if row complete so mapping functions do not need to worry about handling NA
            logger.debug("\telements: {}".format(" ".join([str(x) for x in elements])))
            missing_els = [x for x in elements if x not in cols]
            if len(missing_els) > 0:
                logger.warning(
                    "Following elements from data model missing from input data: {} to map {} ".format(
                        ",".join([str(x) for x in missing_els]), cdm_key
                    )
                )
                continue
            notna_idx_idx = np.where(idata[elements].notna().all(axis=1))[0]
            logger.debug(f"\tnotna_idx_idx: {notna_idx_idx}")
            to_map = idata[elements].iloc[notna_idx_idx]
            # notna_idx = notna_idx_idx + idata.index[0]  # to account for parsers #original
            notna_idx = idata.index[notna_idx_idx]  # fix?
            if len(elements) == 1:
                to_map = to_map.iloc[:, 0]

            if len(to_map) == 0:
                isEmpty = True

        if transform and not isEmpty:
            logger.debug(f"\ttransform: {transform}")
            logger.debug("\tkwargs: {}".format(",".join(list(kwargs.keys()))))

            trans = getattr(imodel_functions, transform)
            logger.debug(f"\ttable_df_i Index: {table_df_i.index}")
            logger.debug(f"\tidata_i Index: {idata.index}")
            if elements:
                logger.debug(f"\tnotna_idx: {notna_idx}")
                table_df_i.loc[notna_idx, cdm_key] = trans(to_map, **kwargs)
            else:
                table_df_i[cdm_key] = trans(**kwargs)
        if code_table and not isEmpty:
            # https://stackoverflow.com/questions/45161220/how-to-map-a-pandas-dataframe-column-to-a-nested-dictionary?rq=1
            # Approach that does not work when it is not nested...so just try and assume not nested if fails
            # Prepare code_table
            table_map = imodel_code_tables.get(code_table)
            try:
                to_map = to_map.to_frame()
            except Exception:
                logger.warning(f"Could not convert {to_map} to frame.")

            to_map_str = to_map.astype(str)

            to_map_str.columns = ["_".join(col) for col in to_map_str.columns.values]
            table_df_i[cdm_key] = to_map_str.apply(
                lambda x: _map_to_df(table_map, x), axis=1
            )
        if elements and not isEmpty:
            table_df_i[cdm_key] = to_map
        if default is not None:  # (value = 0 evals to False!!)
            if isinstance(default, list):
                table_df_i[cdm_key] = [default] * len(table_df_i.index)
            else:
                table_df_i[cdm_key] = default

        if fill_value is not None:
            table_df_i[cdm_key] = table_df_i[cdm_key].fillna(value=fill_value)

        cdm_tables = _decimal_places(
            cdm_tables,
            decimal_places,
            cdm_key,
            table,
            imodel_functions,
        )

    if "observation_value" in table_df_i:
        table_df_i = table_df_i.dropna(subset=["observation_value"])

    table_df_i = drop_duplicates(table_df_i)
    table_df_i.to_csv(cdm_tables[table]["buffer"], header=False, index=False, mode="a")
    return cdm_tables


def _map(imodel, data, cdm_subset=None, codes_subset=None, log_level="INFO"):
    """
    Map to the C3S Climate Data Store Common Data Model (CDM).

    Maps a pandas DataFrame (or pd.io.parsers.TextFileReader) to the C3S Climate Data Store Common Data Model (CDM)
    header and observational tables using mapping information from the input data model (imodel).

    Parameters
    ----------
    imodel: a data model that can be
            1. A generic mapping from a defined data model, like IMMA1's core and attachments
               e.g. ``~/cdm-mapper/lib/mappings/icoads_r3000``
            2. A specific mapping from generic data model to CDM, like map a SID-DCK from IMMA1's core and attachments
               to CDM in a specific way. e.g. ``~/cdm-mapper/lib/mappings/icoads_r3000_d704``
    data: input data to map
        e.g. a pandas.Dataframe or io.parsers.TextFileReader objects or in-memory text streams (io.StringIO object).
    cdm_subset: subset of CDM model tables to map.
        Defaults to the full set of CDM tables defined for the imodel. Type: list.
    codes_subset: subset of code mapping tables to map.
        Defaults to the full set of code mapping tables. defined for the imodel. Type: list.
    log_level: level of logging information to save.
        Defaults to ‘DEBUG’. Type: string.

    Returns
    -------
    cdm_tables: a python dictionary with the ``cdm_table_name`` and ``cdm_table_object`` pairs.

    cdm_table_name: is the name of the CDM table i.e. ``header``, ``observations_at``, etc.
    cdm_table_object: is the python dictionary with the ``{data:cdm_table_object, atts:cdm_table_atts}`` pairs.

    1. cdm_table_object: is a python pandas DataFrame object with the CDM elements aligned in columns according
    to the order established by the imodel.

    2. cdm_table_atts: python dictionary with the CDM element attributes. These element attributes can be the
    elements encoding, decimal places or other characteristics specified in the imodel.
    """
    logger = logging_hdlr.init_logger(__name__, level=log_level)
    codes_imodel = codes_hdlr(imodel)
    tables_imodel = tables_hdlr()
    # Get imodel mapping pack
    # Read mappings to CDM from imodel
    try:
        # Read mappings to CDM from imodel
        imodel_maps = tables_imodel.load_tables_maps(imodel, cdm_subset=cdm_subset)
        if len(imodel_maps) < 1:
            logger.error(f"No mapping codes found for model {imodel}")
            return
        imodel_functions = mapping_functions(imodel)
        imodel_code_tables = codes_imodel.load_code_tables_maps(
            codes_subset=codes_subset
        )
        if imodel_code_tables is None:
            logger.warning(f"No code table mappings found for model {imodel}")
        elif len(imodel_code_tables) < 1:
            logger.warning(
                f"No code table mappings found for model {imodel} (not NoneType)"
            )
    except Exception:
        logger.error(f"Error loading {imodel} cdm mappings", exc_info=True)
        return

    if not imodel_maps:
        logger.error(f"Error loading {imodel} cdm mappings")
        return
    # Read CDM table attributes
    cdm_atts = tables_imodel.load_tables()
    # Check that imodel cdm tables are consistent with CDM tables (at least in naming....)
    not_in_tool = [x for x in imodel_maps.keys() if x not in cdm_atts.keys()]
    if len(not_in_tool) > 0:
        if any(not_in_tool):
            logger.error(
                "One or more tables registered in the data model is not supported by the tool: {}".format(
                    ",".join(not_in_tool)
                )
            )
            logger.info(
                "CDM tables registered in the tool in properties.py are: {}".format(
                    ",".join(properties.cdm_tables)
                )
            )
            return
    # Initialize dictionary to store temporal tables (buffer) and table attributes
    cdm_tables = {
        k: {"buffer": StringIO(), "atts": cdm_atts.get(k)} for k in imodel_maps.keys()
    }
    date_columns = {}
    for table, values in imodel_maps.items():
        date_columns[table] = [
            i
            for i, x in enumerate(list(values))
            if "timestamp" in cdm_atts.get(table, {}).get(x, {}).get("data_type")
        ]

    # Now map per iterable item, per table
    for idata in data:
        cols = [x for x in idata]
        for table, mapping in imodel_maps.items():
            cdm_tables = _write_csv_files(
                idata,
                mapping,
                logger,
                table,
                cols,
                imodel_functions,
                imodel_code_tables,
                cdm_tables,
            )

    for table in cdm_tables.keys():
        # Convert dtime to object to be parsed by the reader
        logger.debug(
            f"\tParse datetime by reader; Table: {table}; Columns: {date_columns[table]}"
        )
        cdm_tables[table]["buffer"].seek(0)
        cdm_tables[table]["data"] = pd.read_csv(
            cdm_tables[table]["buffer"],
            names=imodel_maps[table].keys(),
            parse_dates=date_columns[table],
        )
        cdm_tables[table]["buffer"].close()
        cdm_tables[table].pop("buffer")
    return cdm_tables


def map_model(imodel, data, cdm_subset=None, codes_subset=None, log_level="INFO"):
    """Map a pandas DataFrame to the CDM header and observational tables.

    Parameters
    ----------
    imodel: str
      a data model that can be of several types.
      1. A generic mapping from a defined data model, like IMMA1’s core and attachments.
      e.g. ``cdm/library/mappings/icoads_r3000``
      2. A specific mapping from generic data model to CDM, like map a SID-DCK from IMMA1’s core and attachments to
      CDM in a specific way.
      e.g. ``cdm/library/mappings/icoads_r3000_d704``
    data: pd.DataFrame, pd.parser.TextFileReader or io.String
      input data to map.
      Type: string.
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
    cdm_tables: dict
      a python dictionary with the ``{cdm_table_name: cdm_table_object}`` pairs.
    """
    logger = logging_hdlr.init_logger(__name__, level=log_level)
    # Check we have imodel registered, leave otherwise
    if imodel not in properties.supported_models:
        logger.error("Input data model " f"{imodel}" " not supported")
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

    # Map thing:
    return _map(
        imodel,
        data,
        cdm_subset=cdm_subset,
        codes_subset=codes_subset,
        log_level=log_level,
    )
