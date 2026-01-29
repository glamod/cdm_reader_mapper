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
from typing import Any

import numpy as np
import pandas as pd

from pandas.io.parsers import TextFileReader

from cdm_reader_mapper.common import logging_hdlr, pandas_TextParser_hdlr

from . import properties
from .codes.codes import get_code_table
from .tables.tables import get_cdm_atts, get_imodel_maps
from .utils.conversions import converters, iconverters_kwargs
from .utils.mapping_functions import mapping_functions


def _check_input_data_type(data, logger):
    """Check whether inpuit data type is valid."""

    def _log_and_return_empty(msg):
        logger.error(msg)

    if isinstance(data, pd.DataFrame):
        logger.debug("Input data is a pd.DataFrame")
        if data.empty:
            return _log_and_return_empty("Input data is empty")
        return [data]

    elif isinstance(data, TextFileReader):
        logger.debug("Input is a pd.TextFileReader")
        if not pandas_TextParser_hdlr.is_not_empty(data):
            return _log_and_return_empty("Input data is empty")

        return data

    return _log_and_return_empty("Input data type " f"{type(data)}" " not supported")


def _normalize_input_data(data, logger):
    """Return an iterator of DataFrames irrespective of input type."""
    data = _check_input_data_type(data, logger)

    if data is None:
        return iter(())

    if isinstance(data, list):
        return iter(data)

    return data


def _is_empty(value):
    """Check whether a value is considered empty."""
    if value is None:
        return True

    if hasattr(value, "empty"):
        return bool(value.empty)

    if not value:
        return True

    return False


def _drop_duplicated_rows(df) -> pd.DataFrame:
    """Drop duplicates from list."""
    list_cols = [
        col for col in df.columns if df[col].apply(lambda x: isinstance(x, list)).any()
    ]

    for col in list_cols:
        df[col] = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    df.drop_duplicates(ignore_index=True, inplace=True)

    for col in list_cols:
        if df[col].apply(lambda x: isinstance(x, tuple)).any():
            df[col] = df[col].apply(lambda x: list(x) if isinstance(x, tuple) else x)

    return df


def _get_nested_value(ndict, keys) -> Any | None:
    """Traverse nested dictionaries along a sequence of keys."""
    if not isinstance(ndict, dict):
        return

    current = ndict
    for key in keys:
        if not isinstance(current, dict):
            return
        if key not in current:
            return
        value = current[key]
        if isinstance(value, dict):
            current = value
            continue
        return value


def _convert_dtype(series, atts) -> pd.DataFrame:
    """Convert data to the type specified in `atts`."""
    if atts is None:
        return np.nan

    dtype = atts.get("data_type")
    if not dtype:
        return series

    converter = converters.get(dtype)
    if not converter:
        return series

    converter_keys = iconverters_kwargs.get(dtype)
    if converter_keys:
        kwargs = {key: atts.get(key) for key in converter_keys}
    else:
        kwargs = {}

    return converter(series, np.nan, **kwargs)


def _decimal_places(decimal_places) -> int:
    """Set the 'decimal_places' in the entry dictionary."""
    if decimal_places is None or not isinstance(decimal_places, int):
        return properties.default_decimal_places

    return decimal_places


def _transform(
    data,
    imodel_functions,
    transform,
    kwargs,
    logger,
) -> pd.Series:
    """Apply a transformation function from imodel_functions to a pandas Series."""
    logger.debug(f"Applying transform: {transform}")
    if kwargs:
        logger.debug(f"With kwargs: {', '.join(kwargs.keys())}")
    try:
        trans_func = getattr(imodel_functions, transform)
    except AttributeError:
        logger.error(f"Transform '{transform}' not found in imodel_functions")
        return data

    return trans_func(data, **kwargs)


def _code_table(
    data,
    data_model,
    code_table,
    logger,
) -> pd.Series:
    """Map values in a Series or DataFrame using a (possibly nested) code table."""
    logger.debug(f"Mapping code table: {code_table}")
    table_map = get_code_table(*data_model.split("_"), code_table=code_table)

    try:
        df = data.to_frame() if isinstance(data, pd.Series) else data.copy()
    except Exception:
        logger.warning(f"Could not convert {data} to a DataFrame.")
        return pd.Series([None] * len(data), index=data.index)

    df = df.astype(str)

    df.columns = [
        "_".join(col) if isinstance(col, tuple) else str(col) for col in df.columns
    ]

    def _map_col(col):
        return _get_nested_value(table_map, col.tolist())

    return df.apply(_map_col, axis=1)


def _default(
    default,
    length,
) -> list:
    """Return a list of a given length filled with the default value."""
    return [default] * length


def _fill_value(series, fill_value) -> pd.Series:
    """Fill missing values in series."""
    if fill_value is None:
        return series
    return series.fillna(value=fill_value).infer_objects(copy=False)


def _extract_input_data(idata, elements, default, logger):
    """Extract the relevant input data based on `elements`."""

    def _return_default():
        return pd.Series(_default(default, len(idata)), index=idata.index), True

    if not elements:
        return _return_default()

    logger.debug(f"\telements: {' '.join(map(str, elements))}")

    cols = idata.columns

    for e in elements:
        if e not in cols:
            logger.warning(f"Missing element from input data: {e}")
            return _return_default()

    data = idata[elements[0]] if len(elements) == 1 else idata[elements]

    if _is_empty(data):
        return _return_default()

    return data, False


def _column_mapping(
    idata,
    imapping,
    imodel_functions,
    atts,
    codes_subset,
    column,
    logger,
):
    """Map a column (or multiple elements) in input data according to mapping rules."""
    elements = imapping.get("elements")
    transform = imapping.get("transform")
    kwargs = imapping.get("kwargs", {})
    code_table = imapping.get("code_table")
    default = imapping.get("default")
    fill_value = imapping.get("fill_value")
    decimal_places = imapping.get("decimal_places")

    if codes_subset and code_table not in codes_subset:
        code_table = None

    data, used_default = _extract_input_data(
        idata,
        elements,
        default,
        logger,
    )

    if not used_default:
        if transform:
            data = _transform(
                data,
                imodel_functions,
                transform,
                kwargs,
                logger=logger,
            )
        elif code_table:
            data = _code_table(
                data,
                imodel_functions.imodel,
                code_table,
                logger=logger,
            )

    if not isinstance(data, pd.Series):
        data = pd.Series(data, index=idata.index, copy=False)

    data.name = column

    if fill_value is not None:
        data = _fill_value(data, fill_value)

    if atts:
        atts["decimal_places"] = _decimal_places(decimal_places)
        data = _convert_dtype(data, atts)

    return data


def _table_mapping(
    idata,
    mapping,
    atts,
    null_label,
    imodel_functions,
    codes_subset,
    cdm_complete,
    drop_missing_obs,
    drop_duplicates,
    logger,
) -> pd.DataFrame:
    columns = list(atts) if cdm_complete else [c for c in atts if c in idata.columns]
    out = {}

    for column in columns:
        if column not in mapping.keys():
            out[column] = pd.Series(
                [null_label] * len(idata), index=idata.index, name=column
            )
            continue

        logger.debug(f"\tElement: {column}")

        out[column] = _column_mapping(
            idata,
            mapping[column],
            imodel_functions,
            atts[column],
            codes_subset,
            column,
            logger,
        )

    if not out:
        return pd.DataFrame(index=idata.index)

    table_df = pd.DataFrame(out, index=idata.index)

    if drop_missing_obs is True and "observation_value" in table_df:
        table_df = table_df.dropna(subset=["observation_value"])

    if drop_duplicates:
        table_df = _drop_duplicated_rows(table_df)

    return table_df.fillna(null_label)


def _prepare_cdm_tables(cdm_subset):
    """Prepare table buffers and attributes for CDM tables."""
    if isinstance(cdm_subset, str):
        cdm_subset = [cdm_subset]

    cdm_atts = get_cdm_atts(cdm_subset)
    if not cdm_atts:
        return {}

    tables = {}
    for table, atts in cdm_atts.items():
        for col, meta in atts.items():
            meta["decimal_places"] = _decimal_places(meta.get("decimal_places"))
        tables[table] = {
            "buffer": StringIO(),
            "atts": atts,
        }

    return tables


def _process_chunk(
    idata,
    imodel_maps,
    imodel_functions,
    cdm_tables,
    null_label,
    codes_subset,
    cdm_complete,
    drop_missing_obs,
    drop_duplicates,
    logger,
    is_reader,
):
    """Process one chunk of input data."""
    for table, mapping in imodel_maps.items():
        logger.debug(f"Table: {table}")

        table_df = _table_mapping(
            idata=idata,
            mapping=mapping,
            atts=deepcopy(cdm_tables[table]["atts"]),
            null_label=null_label,
            imodel_functions=imodel_functions,
            codes_subset=codes_subset,
            cdm_complete=cdm_complete,
            drop_missing_obs=drop_missing_obs,
            drop_duplicates=drop_duplicates,
            logger=logger,
        )

        table_df.columns = pd.MultiIndex.from_product([[table], table_df.columns])

        if is_reader:
            table_df.to_csv(
                cdm_tables[table]["buffer"],
                header=False,
                index=False,
                mode="a",
            )
            cdm_tables[table]["columns"] = table_df.columns
        else:
            cdm_tables[table]["df"] = table_df.astype(object)


def _finalize_output(cdm_tables, logger):
    """Turn buffers into DataFrames and combine all tables."""
    final_tables = []

    for table, meta in cdm_tables.items():
        logger.debug(f"\tParse datetime by reader; Table: {table}")

        if "df" not in meta:
            meta["buffer"].seek(0)
            df = pd.read_csv(
                meta["buffer"],
                names=meta["columns"],
                na_values=[],
                dtype="object",
                keep_default_na=False,
            )
            meta["buffer"].close()
        else:
            df = meta.get("df", pd.DataFrame())

        final_tables.append(df)

    if not final_tables:
        return pd.DataFrame()

    return pd.concat(final_tables, axis=1, join="outer").reset_index(drop=True)


def _map_and_convert(
    data_model,
    *sub_models,
    data=None,
    cdm_subset=None,
    codes_subset=None,
    cdm_complete=True,
    drop_missing_obs=True,
    drop_duplicates=True,
    null_label="null",
    logger=None,
) -> pd.DataFrame:
    """Map and convert MDF data to CDM tables."""
    data_iter = _normalize_input_data(data, logger)

    if data_iter is None:
        return pd.DataFrame()

    if not cdm_subset:
        cdm_subset = properties.cdm_tables

    imodel_maps = get_imodel_maps(data_model, *sub_models, cdm_tables=cdm_subset)
    imodel_functions = mapping_functions("_".join([data_model] + list(sub_models)))

    cdm_tables = _prepare_cdm_tables(imodel_maps.keys())

    is_reader = isinstance(data_iter, TextFileReader)

    for idata in data_iter:
        _process_chunk(
            idata=idata,
            imodel_maps=imodel_maps,
            imodel_functions=imodel_functions,
            cdm_tables=cdm_tables,
            null_label=null_label,
            codes_subset=codes_subset,
            cdm_complete=cdm_complete,
            drop_missing_obs=drop_missing_obs,
            drop_duplicates=drop_duplicates,
            logger=logger,
            is_reader=is_reader,
        )

    return _finalize_output(cdm_tables, logger)


def map_model(
    data,
    imodel,
    cdm_subset=None,
    codes_subset=None,
    null_label="null",
    cdm_complete=True,
    drop_missing_obs=True,
    drop_duplicates=True,
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
    cdm_complete: bool
        If True map entire CDM tables list.
        Default: True
    drop_missing_obs: bool
        If True Drop observations without a valid observation value
        (e.g. no air_temperature value).
        Default: True
    drop_duplicates: bool
        If True drop duplicated rows.
        Default: True
    log_level: str
        level of logging information to save.
        Default: INFO.

    Returns
    -------
    cdm_tables: pandas.DataFrame
      DataFrame with MultiIndex columns (cdm_table, column_name).
    """
    logger = logging_hdlr.init_logger(__name__, level=log_level)
    imodel = imodel.split("_")
    if imodel[0] not in properties.supported_data_models:
        logger.error("Input data model " f"{imodel[0]}" " not supported")
        return

    return _map_and_convert(
        imodel[0],
        *imodel[1:],
        data=data,
        cdm_subset=cdm_subset,
        codes_subset=codes_subset,
        null_label=null_label,
        cdm_complete=cdm_complete,
        drop_missing_obs=drop_missing_obs,
        drop_duplicates=drop_duplicates,
        logger=logger,
    )
