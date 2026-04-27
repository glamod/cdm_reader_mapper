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
from collections.abc import Iterable, Sequence
from logging import Logger
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
from .utils.mapping_functions import MappingFunctions


def _is_empty(value: Any) -> bool:
    """
    Check whether a value is considered empty.

    Parameters
    ----------
    value : Any
        Value to be checked.

    Returns
    -------
    bool
        True if value is not empty else False.
    """
    if value is None:
        return True

    if hasattr(value, "empty"):
        return bool(value.empty)

    if not value:
        return True

    return False


def _drop_duplicated_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates from list."""
    list_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, list)).any()]

    for col in list_cols:
        df[col] = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    df.drop_duplicates(ignore_index=True, inplace=True)

    for col in list_cols:
        if df[col].apply(lambda x: isinstance(x, tuple)).any():
            df[col] = df[col].apply(lambda x: list(x) if isinstance(x, tuple) else x)

    return df


def _get_nested_value(ndict: dict[Any, Any] | None, keys: list[Any]) -> Any | None:
    """Traverse nested dictionaries along a sequence of keys."""
    if not isinstance(ndict, dict):
        return None

    for key in keys:
        value = ndict.get(key)
        if value is None:
            return None

        if isinstance(value, dict):
            ndict = value
        else:
            return value

    return None


def _transform(
    data: pd.DataFrame | pd.Series,
    imodel_functions: MappingFunctions,
    transform: str,
    kwargs: dict[str, Any],
    logger: Logger,
) -> pd.Series:
    """Apply a transformation function from imodel_functions to a pandas Series."""
    logger.debug("Applying transform: %s", transform)

    if kwargs:
        logger.debug("With kwargs: %s", ", ".join(kwargs.keys()))
    try:
        trans_func = getattr(imodel_functions, transform)
    except AttributeError:
        logger.error("Transform '%s' not found in imodel_functions", transform)
        return data

    return trans_func(data, **kwargs)


def _code_table(
    data: pd.DataFrame | pd.Series,
    data_model: str,
    code_table: str,
    logger: Logger,
) -> pd.Series:
    """Map values in a Series or DataFrame using a (possibly nested) code table."""
    logger.debug("Mapping code table: %s", code_table)
    table_map = get_code_table(*data_model.split("_"), code_table=code_table)

    df = data.to_frame() if isinstance(data, pd.Series) else data.copy()

    df = df.astype(str)

    df.columns = ["_".join(col) if isinstance(col, tuple) else str(col) for col in df.columns]

    def _map_col(col: Any) -> Any:
        return _get_nested_value(table_map, col.tolist())

    return df.apply(_map_col, axis=1)


def _default(
    default: Any,
    length: int,
) -> list[Any]:
    """Return a list of a given length filled with the default value."""
    return [default] * length


def _fill_value(series: pd.DataFrame | pd.Series, fill_value: Any | None) -> pd.Series:
    """Fill missing values in series."""
    if fill_value is None:
        return series
    return series.fillna(value=fill_value).infer_objects(copy=False)


def _extract_input_data(
    idata: pd.DataFrame | pd.Series, elements: list[str | tuple[str, str]], default: Any, logger: Logger
) -> tuple[pd.DataFrame | pd.Series, bool]:
    """Extract the relevant input data based on `elements`."""

    def _return_default(bool: bool) -> tuple[pd.Series, bool]:
        return pd.Series(_default(default, len(idata)), index=idata.index), bool

    if not elements:
        if default is None:
            bool = False
        else:
            bool = True
        return _return_default(bool)

    logger.debug("\telements: %s", " ".join(map(str, elements)))

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
    idata: pd.DataFrame | pd.Series,
    imapping: dict[str, Any],
    imodel_functions: MappingFunctions,
    atts: dict[str, Any],
    codes_subset: str | tuple[str, str] | list[str | tuple[str, str]],
    column: str | tuple[str, str],
    logger: Logger,
) -> pd.DataFrame | pd.Series:
    """Map a column (or multiple elements) in input data according to mapping rules."""
    elements: list[str | tuple[str, str]] = imapping.get("elements", [])
    transform: str | None = imapping.get("transform")
    kwargs: dict[str, Any] = imapping.get("kwargs", {})
    code_table: str | None = imapping.get("code_table")
    default: Any = imapping.get("default")
    fill_value: Any = imapping.get("fill_value")

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
    idata: pd.DataFrame | pd.Series,
    mapping: dict[str, Any],
    atts: dict[str, Any],
    imodel_functions: MappingFunctions,
    codes_subset: str | tuple[str, str] | list[str | tuple[str, str]],
    cdm_complete: bool,
    drop_missing_obs: bool,
    drop_duplicates: bool,
    logger: Logger,
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


def _prepare_cdm_tables(cdm_subset: str | tuple[str, str] | Sequence[str | tuple[str, str]]) -> dict[str, Any]:
    """Prepare table buffers and attributes for CDM tables."""
    if isinstance(cdm_subset, str):
        cdm_subset = [cdm_subset]

    cdm_atts = get_cdm_atts(cdm_subset)
    if not cdm_atts:
        return {}

    tables = {table: atts for table, atts in cdm_atts.items()}

    return tables


def _map_data_model(
    idata: pd.DataFrame,
    imodel_maps: dict[str, Any],
    imodel_functions: MappingFunctions,
    cdm_tables: dict[str, Any],
    codes_subset: str | tuple[str, str] | list[str | tuple[str, str]],
    cdm_complete: bool,
    drop_missing_obs: bool,
    drop_duplicates: bool,
    logger: Logger,
) -> tuple[pd.DataFrame, pd.Index | pd.MultiIndex]:
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
    imodel: str | None,
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
    def _map_model() -> ProcessFunction:
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
    imodel_functions = MappingFunctions(imodel)

    cdms = list(imodel_maps.keys())
    cdm_tables = _prepare_cdm_tables(cdms)

    results = _map_model()

    if not isinstance(results, tuple):
        raise TypeError(f"result is not a tuple, {type(results)}.")

    result, columns = tuple(results)

    if isinstance(result, pd.DataFrame):
        return result

    if isinstance(result, ParquetStreamReader):
        result.columns = columns
        return result

    raise ValueError(f"result mus be a pd.DataFrame or ParquetStreamReader, not {type(result)}.")
