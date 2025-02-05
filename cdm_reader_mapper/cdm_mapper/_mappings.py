"""Map and convert functions."""

from __future__ import annotations

from copy import deepcopy
from io import StringIO

import numpy as np
import pandas as pd

from . import properties
from ._conversions import converters, iconverters_kwargs
from ._mapping_functions import mapping_functions
from .codes.codes import get_code_table
from .tables.tables import get_cdm_atts, get_imodel_maps


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
    # logger.debug(f"\ttransform: {transform}")
    # logger.debug("\tkwargs: {}".format(",".join(list(kwargs.keys()))))
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


def _fill_value(data, fill_value):
    if fill_value is None:
        return data
    if data is None:
        return fill_value
    return data.fillna(value=fill_value)


def _map_data(
    to_map,
    transform,
    code_table,
    elements,
    default,
    fill_value,
    isEmpty,
    imodel_functions,
    kwargs,
    length,
    logger,
):
    if transform and not isEmpty:
        data = _transform(
            to_map,
            imodel_functions,
            transform,
            kwargs,
            logger=logger,
        )
    elif code_table and not isEmpty:
        data = _code_table(
            to_map,
            imodel_functions.imodel,
            code_table,
            logger=logger,
        )
    elif elements and not isEmpty:
        data = to_map
    elif default is not None:  # (value = 0 evals to False!!)
        data = _default(
            default,
            length,
        )
    else:
        data = _default(
            None,
            length,
        )
    return _fill_value(data, fill_value)


def _mapping(idata, imapping, imodel_functions, atts, codes_subset, cols, logger):
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
        # logger.debug("\telements: {}".format(" ".join([str(x) for x in elements])))
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

        if len(to_map) == 0:
            isEmpty = True

    data = _map_data(
        to_map,
        transform,
        code_table,
        elements,
        default,
        fill_value,
        isEmpty,
        imodel_functions,
        kwargs,
        len(idata),
        logger,
    )
    atts = _decimal_places(atts, decimal_places)
    return data, atts


def _convert_dtype(data, atts, logger):
    if atts is None:
        return np.nan
    itype = atts.get("data_type")
    if converters.get(itype):
        iconverter_kwargs = iconverters_kwargs.get(itype)
        if iconverter_kwargs:
            kwargs = {x: atts.get(x) for x in iconverter_kwargs}
        else:
            kwargs = {}
        return converters.get(itype)(data, np.nan, **kwargs)
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
):
    atts = deepcopy(cdm_tables[table]["atts"])
    columns = (
        [x for x in atts.keys() if x in idata.columns]
        if not cdm_complete
        else list(atts.keys())
    )
    table_df_i = pd.DataFrame(index=idata.index, columns=columns)

    # logger.debug(f"Table: {table}")
    for column in columns:
        if column not in mapping.keys():
            continue
        else:
            # logger.debug(f"\tElement: {column}")
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
        # logger.debug(
        #    f"\tParse datetime by reader; Table: {table}; Columns: {date_columns[table]}"
        # )
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
