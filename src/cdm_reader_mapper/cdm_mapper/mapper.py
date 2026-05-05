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
    """
    Drop duplicates from list.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to drop duplictaed rows.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with deleted duplicated rows.
    """
    list_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, list)).any()]

    for col in list_cols:
        df[col] = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    df.drop_duplicates(ignore_index=True, inplace=True)

    for col in list_cols:
        if df[col].apply(lambda x: isinstance(x, tuple)).any():
            df[col] = df[col].apply(lambda x: list(x) if isinstance(x, tuple) else x)

    return df


def _get_nested_value(ndict: dict[Any, Any] | None, keys: list[Any]) -> Any | None:
    """
    Traverse nested dictionaries along a sequence of keys.

    Parameters
    ----------
    ndict : dict or None
        Input dictionary that may contain nested values.
    keys : list of Any
        List of keys in `ndict` containing nested values.

    Returns
    -------
    Any | None
        Any if `ndict` is not a dictionary or none of `keys` is in `ndict`, otherwise single nested value.
    """
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
    """
    Apply a transformation function from imodel_functions to a pandas Series.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Input data to be transformed.
    imodel_functions : MappingFunctions
        Class containing mapping functions.
    transform : str
        Name of the mapping function.
    kwargs : dict
        Key-word arguments used by mapping function.
    logger : Logger
        Logger instance used for logging.

    Returns
    -------
    pd.Series
        Transformed Series.

    Notes
    -----
    Logs error if `transform` is not a valid mapping function name.
    """
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
    """
    Map values in a Series or DataFrame using a (possibly nested) code table.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Input data to be mapped.
    data_model : str
        Name of the data model (e.g. `icoads_r300_d701`).
    code_table : str
        Name of the code table (e.g. `at_units`).
    logger : Logger
        Logger instance used for logging.

    Returns
    -------
    pd.Series
        Series mapped by a code table.
    """
    logger.debug("Mapping code table: %s", code_table)
    table_map = get_code_table(*data_model.split("_"), code_table=code_table)

    df = data.to_frame() if isinstance(data, pd.Series) else data.copy()

    df = df.astype(str)

    df.columns = ["_".join(col) if isinstance(col, tuple) else str(col) for col in df.columns]

    def _map_col(col: Any) -> Any:
        """
        Map column using a code table.

        Parameters
        ----------
        col : Any
            Column values.

        Returns
        -------
        Any
            Mapped values.
        """
        return _get_nested_value(table_map, col.tolist())

    return df.apply(_map_col, axis=1)


def _default(
    default: Any,
    length: int,
) -> list[Any]:
    """
    Return a list of a given length filled with the default value.

    Parameters
    ----------
    default : Any
        Arbitrary default value.
    length : int
        Length of output list.

    Returns
    -------
    list
        List of length `length` containing value `default`.
    """
    return [default] * length


def _fill_value(data: pd.DataFrame | pd.Series, fill_value: Any | None) -> pd.DataFrame | pd.Series:
    """
    Fill missing values in series.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Input data to be filled with `fill_value`.
    fill_value : Any or None
        Fill value used to fill missing values in `data`.

    Returns
    -------
    pd.DataFrame or pd.Series
        Output data filled with `fill_value`.
    """
    if fill_value is None:
        return data
    return data.fillna(value=fill_value).infer_objects(copy=False)


def _extract_input_data(
    data: pd.DataFrame | pd.Series, elements: list[str | tuple[str, str]], default: Any, logger: Logger
) -> tuple[pd.DataFrame | pd.Series, bool]:
    """
    Extract the relevant input data based on `elements`.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        Input data containing relevant information.
    elements : list of str or list of tuple
        Elements to be extracted.
    default : Any
        Default value if `elements` is empty.
    logger : Logger
        Logger instance used for logging.

    Returns
    -------
    tuple of pd.DataFrame or tuple of pd.Series
        Data containing relevant information only.

    Notes
    -----
    - Logs `elements` as 'DEBUG'.
    - Logs WARNING if one of `elements` is not in columns of `data`.
    """

    def _return_default(bool: bool) -> tuple[pd.Series, bool]:
        """
        Create series containing default values.

        Parameters
        ----------
        bool : bool
            True if default value is defined, otherwise False.

        Returns
        -------
        tuple of pd.Series and bool
            Data containing default value and boolean value whether default value is defined.
        """
        return pd.Series(_default(default, len(data)), index=data.index), bool

    if not elements:
        if default is None:
            bool = False
        else:
            bool = True
        return _return_default(bool)

    logger.debug("\telements: %s", " ".join(map(str, elements)))

    cols = data.columns

    for e in elements:
        if e not in cols:
            logger.warning("Missing element from input data: %s", e)
            return _return_default(True)

    result = data[elements[0]] if len(elements) == 1 else data[elements]

    if _is_empty(result):
        return _return_default(True)

    return result, False


def _column_mapping(
    data: pd.DataFrame | pd.Series,
    imapping: dict[str, Any],
    imodel_functions: MappingFunctions,
    atts: dict[str, Any],
    codes_subset: str | tuple[str, str] | list[str | tuple[str, str]],
    column: str | tuple[str, str],
    logger: Logger,
) -> pd.Series:
    """
    Map a column (or multiple elements) in input data according to mapping rules.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Input data to be mapped.
    imapping : dict
        Mapping dictionary.
    imodel_functions : MappingFunctions
        Class containing mapping functions.
    atts : dict
        Attribute dictionary.
    codes_subset : str or tuple of str or list of str or list of tuple of str
        Subset to be mapped.
    column : str or tuple of str
        Output column name.
    logger : Logger
        Logger instance used for logging.

    Returns
    -------
    pd.Series
        Mapped output data.
    """
    elements: list[str | tuple[str, str]] = imapping.get("elements", [])
    transform: str | None = imapping.get("transform")
    kwargs: dict[str, Any] = imapping.get("kwargs", {})
    code_table: str | None = imapping.get("code_table")
    default: Any = imapping.get("default")
    fill_value: Any = imapping.get("fill_value")

    if codes_subset and code_table not in codes_subset:
        code_table = None

    extracted, used_default = _extract_input_data(
        data,
        elements,
        default,
        logger,
    )

    if not used_default:
        if transform:
            extracted = _transform(
                extracted,
                imodel_functions,
                transform,
                kwargs,
                logger=logger,
            )
        elif code_table:
            extracted = _code_table(
                extracted,
                imodel_functions.imodel,
                code_table,
                logger=logger,
            )

    if not isinstance(extracted, pd.Series):
        extracted = pd.Series(extracted, index=data.index, copy=False)

    extracted.name = column

    if fill_value is not None:
        extracted = _fill_value(extracted, fill_value)

    atts_combined = {**atts, **imapping}
    return convert_from_str_series(
        extracted,
        atts_combined,
    )


def _table_mapping(
    data: pd.DataFrame,
    mapping: dict[str, Any],
    atts: dict[str, Any],
    imodel_functions: MappingFunctions,
    codes_subset: str | tuple[str, str] | list[str | tuple[str, str]],
    cdm_complete: bool,
    drop_missing_obs: bool,
    drop_duplicates: bool,
    logger: Logger,
) -> pd.DataFrame:
    """
    Map a CDM table in input data according to mapping rules.

    Parameters
    ----------
    data : pd.DataFrame
        Input data to be mapped.
    mapping : dict
        Mapping dictionary containing column-specific mapping information.
    atts : dict
        Attribute dictionary.
    imodel_functions : MappingFunctions
        Class containing mapping functions.
    codes_subset : str or tuple of str or list of str or list of tuple of str
        Subset of codes to be mapped.
    cdm_complete : bool
        If True process complete CDM table.
    drop_missing_obs : bool
        If True drop lines if column `observation_value` is missing.
    drop_duplicates : bool
        If True drop duplicated rows.
    logger : Logger
        Logger instance used for logging.

    Returns
    -------
    pd.DataFrame
        Mapped CDM table.

    Notes
    -----
    Logs each entry in `columns` as DEBUG.
    """
    columns = list(atts) if cdm_complete else [c for c in atts if c in data.columns]
    out = {}

    for column in columns:
        logger.debug("	Element: %s", column)

        out[column] = _column_mapping(
            data,
            mapping.get(column, {}),
            imodel_functions,
            atts[column],
            codes_subset,
            column,
            logger,
        )

    if not out:
        return pd.DataFrame(index=data.index)

    table_df = pd.DataFrame(out, index=data.index)

    if drop_missing_obs is True and "observation_value" in table_df:
        table_df = table_df.dropna(subset=["observation_value"])

    if drop_duplicates:
        table_df = _drop_duplicated_rows(table_df)

    return table_df


def _prepare_cdm_tables(cdm_subset: str | Sequence[str]) -> dict[str, Any]:
    """
    Prepare table buffers and attributes for CDM tables.

    Parameters
    ----------
    cdm_subset : str or tuple of str or Sequence of str or Sequence of tuple of str
        Subset of CDM tables.

    Returns
    -------
    dict
        CDM table-specific mapping attributes.
    """
    if isinstance(cdm_subset, str):
        cdm_subset = [cdm_subset]

    cdm_atts = get_cdm_atts(cdm_subset)
    if not cdm_atts:
        return {}

    tables = {table: atts for table, atts in cdm_atts.items()}

    return tables


def _map_data_model(
    data: pd.DataFrame,
    imodel_maps: dict[str, Any],
    imodel_functions: MappingFunctions,
    cdm_tables: dict[str, Any],
    codes_subset: str | tuple[str, str] | list[str | tuple[str, str]],
    cdm_complete: bool,
    drop_missing_obs: bool,
    drop_duplicates: bool,
    logger: Logger,
) -> tuple[pd.DataFrame, pd.Index | pd.MultiIndex]:
    """
    Process one chunk of input data.

    Parameters
    ----------
    data : pd.DataFrame
        Input data to be mapped.
    imodel_maps : dict
        A imodel-specific mapping dictionary.
    imodel_functions : MappingFunctions
        Class containing mapping functions.
    cdm_tables : str or tuple of str or Sequence of str or Sequence of tuple of str
        Subset of CDM tables.
    codes_subset : str or tuple of str or list of str or list of tuple of str
        Subset of codes to be mapped.
    cdm_complete : bool
        If True process complete CDM table.
    drop_missing_obs : bool
        If True drop lines if column `observation_value` is missing.
    drop_duplicates : bool
        If True drop duplicated rows.
    logger : Logger
        Logger instance used for logging.

    Returns
    -------
    tuple of pd.DataFrame and pd.Index or pd.MultiIndex
        Data containing all requested CDM tables and their columns.
    """
    if ":" in data.columns[0]:
        data.columns = pd.MultiIndex.from_tuples(col.split(":") for col in data.columns)

    all_tables = []
    for table, table_atts in cdm_tables.items():
        logger.debug("Table: %s", table)
        table_maps = imodel_maps[table]
        table_df = _table_mapping(
            data=data,
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
    data : pandas.DataFrame or Iterable[pd.DataFrame]
        Input data to map.
    imodel : str
        A specific mapping from generic data model to CDM, like map a SID-DCK from IMMA1’s core
        and attachments to CDM in a specific way, e.g. ``icoads_r300_d704``.
    cdm_subset : str or list, optional
        Subset of CDM model tables to map.
        Defaults to the full set of CDM tables defined for the imodel.
    codes_subset : str or list, optional
        Subset of code mapping tables to map.
        Default to the full set of code mapping tables defined for the imodel.
    cdm_complete : bool, default: True
        If True map entire CDM tables list.
    drop_missing_obs : bool, default: True
        If True Drop observations without a valid observation value, e.g. no air_temperature value.
    drop_duplicates : bool, default: True
        If True drop duplicated rows.
    log_level : str, default: INFO
        Level of logging information to save.

    Returns
    -------
    cdm_tables: pandas.DataFrame
      DataFrame with MultiIndex columns (cdm_table, column_name).

    Raises
    ------
    ValueError
        - If `imodel` is not defined.
        - If first split entry ('_') of `imodel` is not defined.
        - If mapping does not return a DataFame.
    TypeError
        - If type of `imodel` is not supported.
        - If anything during mapping fails.
    """

    @process_function()
    def _map_model() -> ProcessFunction:
        """
        Map model for both DataFrames and ParquetStreamReader.

        Returns
        -------
        ProcessFunction
            Tuple of pd.DataFrame and its columns.
        """
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
