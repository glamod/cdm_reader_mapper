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

from copy import deepcopy
from io import StringIO

import numpy as np
import pandas as pd

from cdm_reader_mapper.common import logging_hdlr, pandas_TextParser_hdlr

from . import properties
from .codes.codes import get_code_table
from .tables.tables import get_cdm_atts, get_imodel_maps
from .utils.conversions import converters, iconverters_kwargs
from .utils.mapping_functions import mapping_functions


def drop_duplicates(df) -> pd.DataFrame:
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
    entry,
    decimal_places,
) -> int:
    if decimal_places is not None:

        if isinstance(decimal_places, int):
            entry["decimal_places"] = decimal_places
        else:
            entry["decimal_places"] = properties.default_decimal_places

    return entry


def _transform(
    series,
    imodel_functions,
    transform,
    kwargs,
    logger,
) -> pd.Series:
    logger.debug(f"\ttransform: {transform}")
    logger.debug("\tkwargs: {}".format(",".join(list(kwargs.keys()))))
    trans = getattr(imodel_functions, transform)
    return trans(series, **kwargs)


def _code_table(
    series,
    data_model,
    code_table,
    logger,
) -> pd.Series:
    table_map = get_code_table(*data_model.split("_"), code_table=code_table)
    try:
        series = series.to_frame()
    except Exception:
        logger.warning(f"Could not convert {series} to frame.")

    series_str = series.astype(str)

    series_str.columns = ["_".join(col) for col in series_str.columns.values]
    return series_str.apply(lambda x: _map_to_df(table_map, x), axis=1)


def _default(
    default,
    length,
) -> list | int:
    if isinstance(default, list):
        return [default] * length
    return default


def _fill_value(series, fill_value) -> pd.Series | int:
    if fill_value is None:
        return series
    if series is None:
        return fill_value
    return series.fillna(value=fill_value).infer_objects(copy=False)


def _map_data(
    series,
    transform,
    code_table,
    default,
    fill_value,
    imodel_functions,
    kwargs,
    length,
    logger,
) -> pd.Series:
    if (series is None or series.empty) and not transform:
        series = _default(default, length)
    elif transform:
        series = _transform(
            series,
            imodel_functions,
            transform,
            kwargs,
            logger=logger,
        )
    elif code_table:
        series = _code_table(
            series,
            imodel_functions.imodel,
            code_table,
            logger=logger,
        )
    return _fill_value(series, fill_value)


def _mapping(
    idata, imapping, imodel_functions, atts, codes_subset, cols, logger
) -> pd.DataFrame:
    elements = imapping.get("elements")
    transform = imapping.get("transform")
    kwargs = imapping.get("kwargs", {})
    code_table = imapping.get("code_table")
    default = imapping.get("default")
    fill_value = imapping.get("fill_value")
    decimal_places = imapping.get("decimal_places")

    if codes_subset:
        if code_table not in codes_subset:
            code_table = None

    to_map = None
    if elements:
        logger.debug("\telements: {}".format(" ".join([str(x) for x in elements])))
        missing_els = [x for x in elements if x not in cols]
        if len(missing_els) > 0:
            logger.warning(
                "Following elements from data model missing from input data: {} to map.".format(
                    ",".join([str(x) for x in missing_els])
                )
            )
            return _default(None, len(idata)), atts

        to_map = idata[elements]
        if len(elements) == 1:
            to_map = to_map.iloc[:, 0]

    data = _map_data(
        to_map,
        transform,
        code_table,
        default,
        fill_value,
        imodel_functions,
        kwargs,
        len(idata),
        logger,
    )
    atts = _decimal_places(atts, decimal_places)
    return data, atts


def _convert_dtype(data, atts, logger) -> pd.DataFrame:
    if atts is None:
        return np.nan
    itype = atts.get("data_type")
    if converters.get(itype):
        iconverter_kwargs = iconverters_kwargs.get(itype)
        if iconverter_kwargs:
            kwargs = {x: atts.get(x) for x in iconverter_kwargs}
        else:
            kwargs = {}
        data = converters.get(itype)(data, np.nan, **kwargs)
    return data


def _map_and_convert(
    idata,
    mapping,
    table,
    cols,
    null_label,
    imodel_functions,
    codes_subset,
    cdm_tables,
    cdm_complete,
    logger,
) -> pd.DataFrame:
    atts = deepcopy(cdm_tables[table]["atts"])
    columns = (
        [x for x in atts.keys() if x in idata.columns]
        if not cdm_complete
        else list(atts.keys())
    )
    table_df_i = pd.DataFrame(index=idata.index, columns=columns)

    logger.debug(f"Table: {table}")
    for column in columns:
        if column not in mapping.keys():
            continue
        logger.debug(f"\tElement: {column}")
        table_df_i[column], atts[column] = _mapping(
            idata,
            mapping[column],
            imodel_functions,
            atts[column],
            codes_subset,
            cols,
            logger,
        )
        table_df_i[column] = _convert_dtype(
            table_df_i[column], atts.get(column), logger
        )

    if "observation_value" in table_df_i:
        table_df_i = table_df_i.dropna(subset=["observation_value"])

    table_df_i.columns = pd.MultiIndex.from_product([[table], columns])
    table_df_i = drop_duplicates(table_df_i)
    table_df_i = table_df_i.fillna(null_label)
    table_df_i.to_csv(cdm_tables[table]["buffer"], header=False, index=False, mode="a")
    cdm_tables[table]["columns"] = table_df_i.columns
    return cdm_tables


def map_and_convert(
    data_model,
    *sub_models,
    data=pd.DataFrame(),
    cdm_subset=None,
    codes_subset=None,
    cdm_complete=True,
    null_label="null",
    logger=None,
) -> pd.DataFrame:
    """Map and convert MDF data to CDM tables."""
    if not cdm_subset:
        cdm_subset = properties.cdm_tables

    cdm_atts = get_cdm_atts(cdm_subset)

    imodel_maps = get_imodel_maps(data_model, *sub_models, cdm_tables=cdm_subset)

    imodel_functions = mapping_functions("_".join([data_model] + list(sub_models)))

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

    for idata in data:
        cols = [x for x in idata]
        for table, mapping in imodel_maps.items():
            cdm_tables = _map_and_convert(
                idata,
                mapping,
                table,
                cols,
                null_label,
                imodel_functions,
                codes_subset,
                cdm_tables,
                cdm_complete,
                logger,
            )

    table_list = []
    for table in cdm_tables.keys():
        # Convert dtime to object to be parsed by the reader
        logger.debug(
            f"\tParse datetime by reader; Table: {table}; Columns: {date_columns[table]}"
        )
        cdm_tables[table]["buffer"].seek(0)
        data = pd.read_csv(
            cdm_tables[table]["buffer"],
            names=cdm_tables[table]["columns"],
            na_values=[],
            dtype="object",
            keep_default_na=False,
        )
        cdm_tables[table]["buffer"].close()
        cdm_tables[table].pop("buffer")
        table_list.append(data)

    merged = pd.concat(table_list, axis=1, join="outer")
    return merged.reset_index(drop=True)


def map_model(
    data,
    imodel,
    cdm_subset=None,
    codes_subset=None,
    null_label="null",
    cdm_complete=True,
    log_level="INFO",
) -> pd.DataFrame:
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
    null_label: str
        String how to label non valid values in `data`.
        Default: null
    cdm_complete:
        If True map entire CDM tables list.
    log_level: str
        level of logging information to save.
        Default: DEBUG.

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
