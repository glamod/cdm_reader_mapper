"""
Map Common Data Model (CDM).

Created on Thu Apr 11 13:45:38 2019

Maps data contained in a pandas DataFrame (or Iterable[pd.DataFrame]) to
the C3S Climate Data Store Common Data Model (CDM) header and observational
tables using the mapping information available in the tool's mapping library
for the input data model.

@author: iregon
"""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, get_args

import pandas as pd

from cdm_reader_mapper.common import logging_hdlr
from cdm_reader_mapper.common.iterators import (
    ParquetStreamReader,
    ProcessFunction,
    process_function,
)

from . import properties
from .codes.codes import get_code_table
from .tables.tables import get_cdm_atts, get_imodel_maps
from .utils.conversions import convert_from_str_series
from .utils.mapping_functions import mapping_functions


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
    list_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, list)).any()]

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
        return None

    current = ndict
    for key in keys:
        if not isinstance(current, dict):
            return None

        value = current.get(key)
        if value is None:
            return None

        if isinstance(value, dict):
            current = value
        else:
            return value

    return None


def _transform(
    data,
    imodel_functions,
    transform,
    kwargs,
    logger,
) -> pd.Series:
    """Apply a transformation function from imodel_functions to a pandas Series."""
    logger.debug("Applying transform: %s", transform)

    if kwargs:
        logger.debug(f"With kwargs: {', '.join(kwargs.keys())}")
    try:
        trans_func = getattr(imodel_functions, transform)
    except AttributeError:
        logger.error("Transform '%s' not found in imodel_functions", transform)
        return data

    return trans_func(data, **kwargs)


def _code_table(
    data,
    data_model,
    code_table,
    logger,
) -> pd.Series:
    """Map values in a Series or DataFrame using a (possibly nested) code table."""
    logger.debug("Mapping code table: %s", code_table)
    table_map = get_code_table(*data_model.split("_"), code_table=code_table)

    df = data.to_frame() if isinstance(data, pd.Series) else data.copy()

    df = df.astype(str)

    df.columns = ["_".join(col) if isinstance(col, tuple) else str(col) for col in df.columns]

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

    def _return_default(bool):
        return pd.Series(_default(default, len(idata)), index=idata.index), bool

    if not elements:
        if default is None:
            bool = False
        else:
            bool = True
        return _return_default(bool)

    logger.debug(f"\telements: {' '.join(map(str, elements))}")

    cols = idata.columns

    for e in elements:
        if e not in cols:
            logger.warning("Missing element from input data: %s", e)
            return _return_default(True)

    data = idata[elements[0]] if len(elements) == 1 else idata[elements]

    if _is_empty(data):
        return _return_default(True)

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

    atts_combined = {**atts, **imapping}
    return convert_from_str_series(
        data,
        atts_combined,
    )


def _table_mapping(
    idata,
    mapping,
    atts,
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
        logger.debug("	Element: %s", column)

        out[column] = _column_mapping(
            idata,
            mapping.get(column, {}),
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

    return table_df


def _prepare_cdm_tables(cdm_subset):
    """Prepare table buffers and attributes for CDM tables."""
    if isinstance(cdm_subset, str):
        cdm_subset = [cdm_subset]

    cdm_atts = get_cdm_atts(cdm_subset)
    if not cdm_atts:
        return {}

    tables = {}
    for table, atts in cdm_atts.items():
        tables[table] = atts

    return tables


def _map_data_model(
    idata,
    imodel_maps,
    imodel_functions,
    cdm_tables,
    codes_subset,
    cdm_complete,
    drop_missing_obs,
    drop_duplicates,
    logger,
):
    """Process one chunk of input data."""
    if ":" in idata.columns[0]:
        idata.columns = pd.MultiIndex.from_tuples(col.split(":") for col in idata.columns)

    all_tables = []
    for table, table_atts in cdm_tables.items():
        logger.debug("Table: %s", table)
        table_maps = imodel_maps[table]
        table_df = _table_mapping(
            idata=idata,
            mapping=table_maps,
            atts=table_atts,
            imodel_functions=imodel_functions,
            codes_subset=codes_subset,
            cdm_complete=cdm_complete,
            drop_missing_obs=drop_missing_obs,
            drop_duplicates=drop_duplicates,
            logger=logger,
        )

        table_df.columns = pd.MultiIndex.from_product([[table], table_df.columns])
        all_tables.append(table_df)

    tables_df = pd.concat(all_tables, axis=1, join="outer").reset_index(drop=True)
    columns = tables_df.columns
    return tables_df, columns


def map_model(
    data: pd.DataFrame | Iterable[pd.DataFrame],
    imodel: str,
    cdm_subset: str | list[str] | None = None,
    codes_subset: str | list[str] | None = None,
    cdm_complete: bool = True,
    drop_missing_obs: bool = True,
    drop_duplicates: bool = True,
    log_level: str = "INFO",
) -> pd.DataFrame | ParquetStreamReader:
    """
    Map a pandas DataFrame to the CDM header and observational tables.

    Parameters
    ----------
    data: pandas.DataFrame or Iterable[pd.DataFrame]
        input data to map.
    imodel: str
        A specific mapping from generic data model to CDM, like map a SID-DCK from IMMA1’s core and attachments to
        CDM in a specific way.
        e.g. ``icoads_r300_d704``
    cdm_subset: str or list, optional
        subset of CDM model tables to map.
        Defaults to the full set of CDM tables defined for the imodel.
    codes_subset: str or list, optional
        subset of code mapping tables to map.
        Default to the full set of code mapping tables defined for the imodel.
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

    @process_function()
    def _map_model():
        return ProcessFunction(
            data=data,
            func=_map_data_model,
            func_kwargs={
                "imodel_maps": imodel_maps,
                "imodel_functions": imodel_functions,
                "cdm_tables": cdm_tables,
                "codes_subset": codes_subset,
                "cdm_complete": cdm_complete,
                "drop_missing_obs": drop_missing_obs,
                "drop_duplicates": drop_duplicates,
                "logger": logger,
            },
            makecopy=False,
        )

    logger = logging_hdlr.init_logger(__name__, level=log_level)

    if imodel is None:
        raise ValueError("Input data model 'imodel' is not defined.")

    if not isinstance(imodel, str):
        raise TypeError(f"Input data model type is not supported: {type(imodel)}")

    data_model = imodel.split("_")
    if data_model[0] not in get_args(properties.SupportedDataModels):
        raise ValueError(f"Input data model {data_model[0]} not supported")

    if not cdm_subset:
        cdm_subset = properties.cdm_tables

    imodel_maps = get_imodel_maps(*data_model, cdm_tables=cdm_subset)
    imodel_functions = mapping_functions(imodel)

    cdm_tables = _prepare_cdm_tables(imodel_maps.keys())

    results = _map_model()

    result, columns = tuple(results)

    if isinstance(result, pd.DataFrame):
        return result

    if isinstance(result, ParquetStreamReader):
        result.columns = columns
        return result

    raise ValueError(f"result mus be a pd.DataFrame or ParquetStreamReader, not {type(result)}.")
