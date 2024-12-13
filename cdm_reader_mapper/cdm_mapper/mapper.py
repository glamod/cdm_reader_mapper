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

import ast
from io import StringIO

import numpy as np
import pandas as pd

from cdm_reader_mapper.common import logging_hdlr, pandas_TextParser_hdlr

from . import properties
from .codes.codes import get_code_table
from .mappings import mapping_functions
from .tables.tables import get_cdm_atts, get_imodel_maps


def convert_integer(data, null_label):
    """
    Convert all elements that have 'int' as type attribute.

    Parameters
    ----------
    data: data tables to convert
    null_label: specified how nan are represented

    Returns
    -------
    data: data as int type
    """

    def _return_str(x, null_label):
        if pd.isna(x):
            return null_label
        return str(
            int(float(x))
        )  # ValueError: invalid literal for int() with base 10: '5.0'

    return data.apply(lambda x: _return_str(x, null_label))


def convert_float(data, null_label, decimal_places):
    """
    Convert all elements that have 'float' as type attribute.

    Parameters
    ----------
    data: data tables to convert
    null_label: specified how nan are represented
    decimal_places: number of decimal places

    Returns
    -------
    data: data as float type
    """

    def _return_str(x, null_label, format_float):
        if pd.isna(x):
            return null_label
        return format_float.format(x)

    format_float = "{:." + str(decimal_places) + "f}"
    return data.apply(lambda x: _return_str(x, null_label, format_float))


def convert_datetime(data, null_label):
    """
    Convert datetime objects in the format: "%Y-%m-%d %H:%M:%S".

    Parameters
    ----------
    data: date time elements
    null_label: specified how nan are represented

    Returns
    -------
    data: data as datetime objects
    """

    def _return_str(x, null_label):
        if pd.isna(x):
            return null_label
        if isinstance(x, str):
            return x
        return x.strftime("%Y-%m-%d %H:%M:%S")

    return data.apply(lambda x: _return_str(x, null_label))


def convert_varchar(data, null_label):
    """
    Convert string elements.

    Parameters
    ----------
    data: data tables to convert
    null_label: specified how nan are represented

    Returns
    -------
    data: data as string objects
    """

    def _return_str(x, null_label):
        if pd.isna(x):
            return null_label
        return str(x)

    return data.apply(lambda x: _return_str(x, null_label))


def convert_integer_array(data, null_label):
    """
    Convert a series of integer objects as array.

    Parameters
    ----------
    data: data tables to convert
    null_label: specified how nan are represented

    Returns
    -------
    data: array of int objects
    """
    return data.apply(convert_integer_array_i, null_label=null_label)


def convert_varchar_array(data, null_label):
    """
    Convert a series of string objects as array.

    Parameters
    ----------
    data: data tables to convert
    null_label: specified how nan are represented

    Returns
    -------
    data: array of varchar objects
    """
    return data.apply(convert_varchar_array_i)


def convert_integer_array_i(row, null_label=None):
    """
    Convert a series of integer objects.

    Parameters
    ----------
    row
    null_label

    Returns
    -------
    data: int
    """
    row = row if not isinstance(row, str) else ast.literal_eval(row)
    row = row if isinstance(row, list) else [row]
    str_row = [str(int(x)) for x in row if np.isfinite(x)]
    string = ",".join(filter(bool, str_row))
    if len(string) > 0:
        return "{" + string + "}"
    return null_label


def convert_varchar_array_i(row, null_label=None):
    """
    Convert a series of string objects.

    Parameters
    ----------
    row
    null_label

    Returns
    -------
    data: varchar
    """
    row = row if not isinstance(row, str) else ast.literal_eval(row)
    row = row if isinstance(row, list) else [row]
    str_row = [str(x) for x in row if np.isfinite(x)]
    string = ",".join(filter(bool, str_row))
    if len(string) > 0:
        return "{" + string + "}"
    return null_label


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
    entry,
    decimal_places,
):
    if decimal_places is not None:

        if isinstance(decimal_places, int):
            entry["decimal_places"] = decimal_places
        else:
            entry["decimal_places"] = properties.default_decimal_places

    return entry


def _transform(
    to_map,
    imodel_functions,
    transform,
    kwargs,
    logger,
):
    logger.debug(f"\ttransform: {transform}")
    logger.debug("\tkwargs: {}".format(",".join(list(kwargs.keys()))))
    trans = getattr(imodel_functions, transform)
    return trans(to_map, **kwargs)


def _code_table(
    to_map,
    data_model,
    code_table,
    logger,
):
    # https://stackoverflow.com/questions/45161220/how-to-map-a-pandas-dataframe-column-to-a-nested-dictionary?rq=1
    # Approach that does not work when it is not nested...so just try and assume not nested if fails
    # Prepare code_table
    table_map = get_code_table(*data_model.split("_"), code_table=code_table)
    try:
        to_map = to_map.to_frame()
    except Exception:
        logger.warning(f"Could not convert {to_map} to frame.")

    to_map_str = to_map.astype(str)

    to_map_str.columns = ["_".join(col) for col in to_map_str.columns.values]
    return to_map_str.apply(lambda x: _map_to_df(table_map, x), axis=1)


def _default(
    default,
    length,
):
    if isinstance(default, list):
        return [default] * length
    return default


def _write_csv_files(
    idata,
    mapping,
    logger,
    table,
    cols,
    imodel_functions,
    codes_subset,
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

        if codes_subset:
            if code_table not in codes_subset:
                code_table = None

        to_map = None
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

            to_map = idata[elements]
            if len(elements) == 1:
                to_map = to_map.iloc[:, 0]

            if len(to_map) == 0:
                isEmpty = True

        if transform and not isEmpty:
            table_df_i[cdm_key] = _transform(
                to_map,
                imodel_functions,
                transform,
                kwargs,
                logger=logger,
            )
        elif code_table and not isEmpty:
            table_df_i[cdm_key] = _code_table(
                to_map,
                imodel_functions.imodel,
                code_table,
                logger=logger,
            )
        elif elements and not isEmpty:
            table_df_i[cdm_key] = to_map
        elif default is not None:  # (value = 0 evals to False!!)
            table_df_i[cdm_key] = _default(
                default,
                len(table_df_i.index),
            )

        if fill_value is not None:
            table_df_i[cdm_key] = table_df_i[cdm_key].fillna(value=fill_value)

        cdm_tables[table]["atts"][cdm_key] = _decimal_places(
            cdm_tables[table]["atts"][cdm_key],
            decimal_places,
        )

    if "observation_value" in table_df_i:
        table_df_i = table_df_i.dropna(subset=["observation_value"])

    table_df_i = drop_duplicates(table_df_i)
    table_df_i.to_csv(cdm_tables[table]["buffer"], header=False, index=False, mode="a")
    return cdm_tables


def _convert_dtype(data, atts, null_label, logger):
    cdm_table = pd.DataFrame(index=data.index, columns=atts.keys(), dtype="object")
    for column in atts.keys():
        if column in data:
            itype = atts.get(column).get("data_type")
            if converters.get(itype):
                iconverter_kwargs = iconverters_kwargs.get(itype)
                if iconverter_kwargs:
                    kwargs = {x: atts.get(column).get(x) for x in iconverter_kwargs}
                else:
                    kwargs = {}
                cdm_table[column] = converters.get(itype)(
                    data[column], null_label, **kwargs
                )
            else:
                logger.error(f"No converter defined for element {column}")
        else:
            cdm_table[column] = null_label
    return cdm_table


def _map(
    data_model,
    *sub_models,
    data=pd.DataFrame(),
    cdm_subset=None,
    codes_subset=None,
    cdm_complete=True,
    null_label="null",
    logger=None,
):
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
            cdm_tables = _write_csv_files(
                idata,
                mapping,
                logger,
                table,
                cols,
                imodel_functions,
                codes_subset,
                cdm_tables,
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
            names=imodel_maps[table].keys(),
            parse_dates=date_columns[table],
        )
        cdm_tables[table]["buffer"].close()
        cdm_tables[table].pop("buffer")
        atts = cdm_tables[table]["atts"]

        cdm_table = _convert_dtype(data, atts, null_label, logger)
        columns = (
            [x for x in atts.keys() if x in data.columns]
            if not cdm_complete
            else list(atts.keys())
        )
        cdm_table = cdm_table[columns]
        cdm_table.columns = pd.MultiIndex.from_product([[table], columns])
        table_list.append(cdm_table)

    merged = pd.concat(table_list, axis=1, join="outer")
    merged = merged.reset_index(drop=True)
    return merged


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
    data: pd.DataFrame, pd.parser.TextFileReader or io.String
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
    cdm_tables: dict
      a python dictionary with the ``{cdm_table_name: cdm_table_object}`` pairs.
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

    # Map thing:
    return _map(
        imodel[0],
        *imodel[1:],
        data=data,
        cdm_subset=cdm_subset,
        codes_subset=codes_subset,
        null_label=null_label,
        cdm_complete=cdm_complete,
        logger=logger,
    )


converters = {
    "int": convert_integer,
    "numeric": convert_float,
    "varchar": convert_varchar,
    "timestamp with timezone": convert_datetime,
    "int[]": convert_integer_array,
    "varchar[]": convert_varchar_array,
}

iconverters_kwargs = {
    "numeric": ["decimal_places"],
    "numeric[]": ["decimal_places"],
}
