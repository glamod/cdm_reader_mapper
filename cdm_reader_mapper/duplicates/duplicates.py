"""Common Data Model (CDM) pandas duplicate check."""

from __future__ import annotations

import datetime
from copy import deepcopy

from typing import Iterable

import numpy as np
import pandas as pd
import recordlinkage as rl

from ._duplicate_settings import Compare, _compare_kwargs, _histories, _method_kwargs


def convert_series(df: pd.DataFrame, conversion: dict) -> pd.DataFrame:
    """Convert data types in Dataframe.

    Parameters
    ----------
    df: pandas.DataFrame
        Input DataFrame
    conversion: dict
        Conversion dictionary conating columns and
        new data type as key-value pairs.

    Returns
    -------
    pandas.DataFrame
    """

    def convert_date_to_float(date):
        date = date.astype("datetime64[ns]")
        return (date - date.min()) / np.timedelta64(1, "s")

    df = df.copy()
    df = df.mask(df == "null", np.nan)
    for column, method in conversion.items():
        try:
            df[column] = df[column].astype(method)
        except TypeError:
            df[column] = locals()[method](df[column])

    df = df.infer_objects(copy=False).fillna(9999.0)
    return df


def add_history(df: pd.DataFrame, indexes: Iterable[int]) -> pd.DataFrame:
    """
    Append duplicate information to the 'history' column of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing a 'history' column.
    indexes : list[int] or pd.Index
        Row indexes where history should be updated.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with updated 'history' column for the selected rows.

    Notes
    -----
    - If 'history' column does not exist, it will be created with empty strings.
    - Each message is prefixed with a UTC timestamp in "YYYY-MM-DD HH:MM:SS" format.
    """

    def _datetime_now():

        try:
            now = datetime.datetime.now(datetime.UTC)
        except AttributeError:
            now = datetime.datetime.utcnow()

        return now.strftime("%Y-%m-%d %H:%M:%S")

    df = df.copy()

    if "history" not in df.columns:
        df["history"] = ""

    history_tstmp = _datetime_now()
    addition = "".join([f"; {history_tstmp}. {add}" for add in _histories.items()])
    df.loc[indexes, "history"] = df.loc[indexes, "history"] + addition
    return df


def add_duplicates(df: pd.DataFrame, dups: pd.DataFrame) -> pd.DataFrame:
    """
    Add duplicate information to the DataFrame based on the `dups` table.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'report_id' column.
    dups : pd.DataFrame
        DataFrame where the index corresponds to rows in `df` and
        the values are lists of duplicate indices or duplicate IDs.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with a 'duplicates' column containing duplicates
        as a sorted string list, e.g., "{ID1,ID2}".

    Notes
    -----
    - If a row has no duplicates, its 'duplicates' column is left unchanged.
    - Supports duplicates represented either by IDs (str) or by indices (int) of `report_id`.
    """

    def _add_dups(row):
        idx = row.name
        if idx not in dups.index:
            return row

        dup_idx = dups.loc[idx].to_list()
        if isinstance(dup_idx[0][0], str):
            v_ = sorted(dup_idx[0])
        else:
            v_ = report_ids.iloc[dup_idx[0]]
            v_ = sorted(v_.tolist())
        row["duplicates"] = "{" + ",".join(v_) + "}"
        return row

    df = df.copy()

    if "duplicates" not in df.columns:
        df["duplicates"] = ""

    report_ids = df["report_id"]
    return df.apply(lambda x: _add_dups(x), axis=1)


def add_report_quality(df: pd.DataFrame, indexes_bad: Iterable[int]) -> pd.DataFrame:
    """
    Update the 'report_quality' column in a DataFrame for bad reports.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a 'report_quality' column.
    indexes_bad : iterable of int
        Row indices in the DataFrame to mark as bad quality (value=1).

    Returns
    -------
    pd.DataFrame
        DataFrame with updated 'report_quality' column.
    """
    df = df.copy()
    df["report_quality"] = df["report_quality"].astype(int)
    df.loc[indexes_bad, "report_quality"] = 1
    return df


class DupDetect:
    """
    Class to detect, flag, and remove duplicate entries in a DataFrame
    using a comparison matrix from recordlinkage.

    Parameters
    ----------
    data : pd.DataFrame
        Original dataset.
    compared : pd.DataFrame
        Comparison matrix of the dataset.
    method : str
        Duplicate detection method used for recordlinkage indexing.
    method_kwargs : dict
        Keyword arguments for recordlinkage indexing method.
    compare_kwargs : dict
        Keyword arguments used for recordlinkage.Compare.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        compared: pd.DataFrame,
        method: str,
        method_kwargs: dict,
        compare_kwargs: dict,
    ):
        self.data = data.copy()
        self.compared = compared
        self.method = method
        self.method_kwargs = method_kwargs
        self.compare_kwargs = compare_kwargs

    def _get_limit(self, limit: str | float | None) -> float:
        """
        Resolve the duplicate threshold limit.

        Parameters
        ----------
        limit : str or float
            'default', None, or a numeric limit.

        Returns
        -------
        float
            Threshold for total score to consider duplicates.
        """
        default_limit = 0.991
        if limit in ("default", None):
            return default_limit

        return float(limit)

    def _get_equal_musts(self) -> list[str]:
        """
        Identify columns that must be equal for duplicates.

        Returns
        -------
        list[str]
            Columns that must match exactly to consider duplicates.
        """
        equal_musts = []
        for value in self.compare_kwargs.keys():
            if not isinstance(value, list):
                value = [value]
            for v in value:
                if v in self.data.columns:
                    equal_musts.append(v)
        return equal_musts

    def _total_score(self) -> None:
        """Compute total similarity score for each row in `self.compared`."""
        pcmax = self.compared.shape[1]
        self.score = 1 - (abs(self.compared.sum(axis=1) - pcmax) / pcmax)

    def get_duplicates(
        self,
        keep: str | int = "first",
        limit: str | float | None = "default",
        equal_musts: str | list[str] | None = None,
        overwrite: bool = True,
    ) -> pd.DataFrame:
        """
        Identify duplicate matches based on the comparison matrix.

        Parameters
        ----------
        keep : str or int
            Which entry to keep: 'first', 'last', or a specific integer index.
        limit : str or float, optional
            Threshold of total similarity score to consider as duplicate.
        equal_musts : str or list[str], optional
            Columns that must exactly match.
        overwrite : bool
            Whether to recompute matches if already calculated.

        Returns
        -------
        pd.DataFrame
            DataFrame containing matched duplicates.
        """
        if keep == "first":
            self.drop = 0
            self.keep = -1
        elif keep == "last":
            self.drop = -1
            self.keep = 0
        elif not isinstance(keep, int):
            raise ValueError("keep has to be one of 'first', 'last' of integer value.")
        if overwrite is True:
            self._total_score()
            self.limit = self._get_limit(limit)
            cond = self.score >= self.limit
            if equal_musts is None:
                equal_musts = self._get_equal_musts()
            if isinstance(equal_musts, str):
                equal_musts = [equal_musts]
            for must in equal_musts:
                cond = cond & (self.compared[must])
            self.matches = self.compared[cond]
        return self.matches

    def flag_duplicates(
        self,
        keep: str | int = "first",
        limit: str | float | None = "default",
        equal_musts: str | list[str] | None = None,
    ) -> pd.DataFrame:
        r"""Get result dataset with flagged duplicates.

        Parameters
        ----------
        keep: str, ["first", "last"]
            Which entry should be kept in result dataset.
        limit: float, optional
            Limit of total score that as to be exceeded to be declared as a duplicate.
            Default: .991
        equal_musts: str or list, optional
            Hashable of column name(s) that must totally be equal to be declared as a duplicate.
            Default: All column names found in method_kwargs.

        Returns
        -------
        pandas.DataFrame
            Input DataFrame with flagged duplicates.
            Flags for ``duplicate_status``: see `duplicate_status`_
            Flags for ``report_quality``: see `quality_flag`_


        .. _duplicate_status: https://glamod.github.io/cdm-obs-documentation/tables/code_tables/duplicate_status/duplicate_status.html
        .. _quality_flag: https://glamod.github.io/cdm-obs-documentation/tables/code_tables/quality_flag/quality_flag.html
        """

        def _get_similars(drop, keeps):
            if drop[drop_] in keeps:
                drops = drop[drop_]
                keeps = drop[keep_]
                try:
                    return int(drops), int(keeps)
                except ValueError:
                    return drops, keeps

        def _get_duplicates(x, last):
            b = list(set(x[last].values))
            return pd.Series({"dups": b})

        def _delete_values_equal_keys(dictionary):
            dictionary_ = {}
            drops_ = []
            for k, v in dictionary.items():
                if k == v:
                    drops_.append(v)
                    continue
                dictionary_[k] = v
            return dictionary_, drops_

        def replace_keeps_and_drops(df, keep_):
            while True:
                df = df.sort_index()
                keeps = df[keep_].values
                replaces = df.apply(lambda x: _get_similars(x, keeps), axis=1)
                replaces = dict(replaces.dropna().values)
                replaces, drops_ = _delete_values_equal_keys(replaces)
                keys = replaces.keys()
                values = replaces.values()
                if len(drops_) > 0:
                    df = df.drop(drops_, axis="index")
                df[keep_] = df[keep_].replace(replaces)
                if not set(keys).intersection(values):
                    return df

        self.get_duplicates(keep=keep, limit=limit, equal_musts=equal_musts)
        result = self.data.copy()
        result["duplicate_status"] = 0
        if not hasattr(self, "matches"):
            self.get_matches(limit="default", equal_musts=equal_musts)

        indexes = self.matches.index
        indexes_df = indexes.to_frame()
        drop_ = indexes_df.columns[self.drop]
        keep_ = indexes_df.columns[self.keep]
        indexes_df = indexes_df.drop_duplicates(subset=[drop_])
        indexes_df = replace_keeps_and_drops(indexes_df, keep_)

        dup_keep = indexes_df.groupby(indexes_df[keep_]).apply(
            lambda x: _get_duplicates(x, drop_),
            include_groups=False,
        )
        dup_drop = indexes_df.groupby(indexes_df[drop_]).apply(
            lambda x: _get_duplicates(x, keep_),
            include_groups=False,
        )
        duplicates = pd.concat([dup_keep, dup_drop])

        indexes_good = indexes_df[keep_].values.tolist()
        indexes_bad = indexes_df[drop_].values.tolist()
        indexes = indexes_good + indexes_bad
        result.loc[indexes_good, "duplicate_status"] = 1
        result.loc[indexes_bad, "duplicate_status"] = 3
        result = add_report_quality(result, indexes_bad=indexes_bad)
        result = add_history(result, indexes)
        result = result.sort_index(ascending=True)
        self.result = add_duplicates(result, duplicates)
        self.data = self.data.sort_index(ascending=True)

        return self.result

    def remove_duplicates(
        self,
        keep: str | int = "first",
        limit: str | float | None = "default",
        equal_musts: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Remove duplicate entries from the dataset.

        Parameters
        ----------
        keep : str or int
            Which entry to keep ('first' or 'last').
        limit : str or float, optional
            Minimum similarity score to declare duplicates.
        equal_musts : str or list[str], optional
            Columns that must exactly match.

        Returns
        -------
        pd.DataFrame
            Dataset without duplicates.
        """
        self.get_duplicates(keep=keep, limit=limit, equal_musts=equal_musts)
        result = self.data.copy()
        drops = self.matches.index.get_level_values(self.drop)
        result = result.drop(drops)
        self.result = result.sort_index(ascending=True)
        self.data = self.data.sort_index(ascending=True)
        return self.result


def set_comparer(compare_dict) -> Compare:
    """
    Build a recordlinkage Compare object with optional conversion dictionary.

    Parameters
    ----------
    compare_dict: dict
        Dictionary of columns to compare, e.g.,
        {"column_name": {"method": "exact" | "numeric" | "date2", "kwargs": {...}}}

    Returns
    -------
    recordlinkage.Compare
        Compare object with added comparison methods and a 'conversion' attribute.
    """
    comparer = Compare()
    setattr(comparer, "conversion", {})
    for column, c_dict in compare_dict.items():
        try:
            method = c_dict["method"]
        except KeyError:
            raise KeyError(
                "compare_kwargs must be hierarchically ordered: {<column_name>: {'method': <compare_method>}}. 'method' not found"
            )
        try:
            kwargs = c_dict["kwargs"]
        except KeyError:
            kwargs = {}
        getattr(comparer, method)(
            column,
            column,
            label=column,
            **kwargs,
        )
        if method == "numeric":
            comparer.conversion[column] = float
        if method == "date":
            comparer.conversion[column] = "datetime64[ns]"
        if method == "date2":
            comparer.conversion[column] = "convert_date_to_float"

    return comparer


def remove_ignores(dic: dict, columns: str | list[str]) -> dict:
    """
    Remove dictionary entries where keys or values match ignored columns.

    Parameters
    ----------
    dic : dict
        Original dictionary to filter.
    columns : str or list[str]
        Column(s) to ignore.

    Returns
    -------
    dict
        Filtered dictionary without the ignored columns.
    """
    new_dict = {}
    if isinstance(columns, str):
        columns = [columns]
    for k, v in dic.items():
        if k in columns:
            continue
        if v in columns:
            continue
        if isinstance(v, list):
            v2 = [v_ for v_ in v if v_ not in columns]
            if len(v2) == 0:
                continue
            v = v2
        new_dict[k] = v
    return new_dict


def change_offsets(dic: dict, dic_o: dict) -> dict:
    """
    Update the 'offset' value in compare dictionary kwargs.

    Parameters
    ----------
    dic : dict
        Original compare dictionary.
    dic_o : dict
        Dictionary mapping column names to new offsets.

    Returns
    -------
    dict
        Updated compare dictionary with modified offsets.
    """
    for key in dic.keys():
        if key not in dic_o.keys():
            continue
        dic[key]["kwargs"]["offset"] = dic_o[key]
    return dic


def reindex_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex a DataFrame in ascending order based on the number of 'null' strings in each row.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Cells with the string "null" are counted as nulls.

    Returns
    -------
    pd.DataFrame
        DataFrame reindexed so that rows with fewer 'null' values appear first.
        Original row order is preserved for rows with the same null count.
    """

    def _count_nulls(row):
        return (row == "null").sum()

    nulls = df.apply(lambda x: _count_nulls(x), axis=1)
    if nulls.empty:
        return df
    indexes_ = list(zip(*sorted(zip(nulls.values, nulls.index))))
    return df.reindex(indexes_[1])


class Comparer:
    """
    Wrapper around recordlinkage.Compare to compute pairwise comparisons on a DataFrame.

    This class initializes a recordlinkage indexer and Compare object, optionally converting
    the data types before computing the comparisons.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to compare.
    method : str
        The indexing method from `recordlinkage.index`, e.g., 'SortedNeighbourhood'.
    method_kwargs : dict
        Keyword arguments to pass to the indexing method.
    compare_kwargs : dict
        Dictionary specifying columns and comparison methods for recordlinkage.Compare.
    pairs_df : list[pd.DataFrame], optional
        Optional pre-split DataFrames to pass to the indexer. Defaults to `[data]`.
    convert_data : bool, default False
        Whether to convert data using `compare_kwargs` conversion dictionary.
    """

    def __init__(
        self,
        data,
        method,
        method_kwargs,
        compare_kwargs,
        pairs_df=None,
        convert_data=False,
    ):
        indexer = getattr(rl.index, method)(**method_kwargs)
        comparer = set_comparer(compare_kwargs)
        if convert_data is True:
            data_ = convert_series(data, comparer.conversion)
        else:
            data_ = data.copy()
        if pairs_df is None:
            pairs_df = [data_]
        pairs = indexer.index(*pairs_df)
        self.compared = comparer.compute(pairs, data_)
        self.data = data_


def duplicate_check(
    data,
    method="SortedNeighbourhood",
    method_kwargs=None,
    compare_kwargs=None,
    table_name=None,
    ignore_columns=None,
    ignore_entries=None,
    offsets=None,
    reindex_by_null=True,
) -> DupDetect:
    """
    Run a duplicate check on a dataset using recordlinkage.

    Returns a DupDetect object.

    Parameters
    ----------
    data: pandas.DataFrame
        Dataset for duplicate check.
    method: str
        Duplicate check method for recordlinkage.
        Default: SortedNeighbourhood
    method_kwargs: dict, optional
        Keyword arguments for recordlinkage duplicate check.
        Default: _method_kwargs
    compare_kwargs: dict, optional
        Keyword arguments for recordlinkage.Compare object.
        Default: _compare_kwargs
    table_name: str, optional
        Name of the CDM table to be selected from data.
    ignore_columns: str or list, optional
        Name of data columns to be ignored for duplicate check.
    ignore_entries: dict, optional
        Key: Column name
        Value: value to be ignored
        E.g. offsets={"station_speed": null}
    offsets: dict, optional
        Change offsets for recordlinkage Compare object.
        Key: Column name
        Value: new offset
        E.g. offsets={"latitude": 0.1}
    reindex_by_null: bool, optional
        If True data is re-indexed in ascending order according to the number of nulls in each row.

    Returns
    -------
        cdm_reader_mapper.DupDetect
    """
    if reindex_by_null is True:
        data = reindex_nulls(data)

    index = data.index
    data.reset_index(drop=True)

    if table_name:
        data = data[table_name]
    if not method_kwargs:
        method_kwargs = deepcopy(_method_kwargs)
    if not compare_kwargs:
        compare_kwargs = deepcopy(_compare_kwargs)
    if ignore_columns:
        method_kwargs = remove_ignores(method_kwargs, ignore_columns)
        compare_kwargs = remove_ignores(compare_kwargs, ignore_columns)
    if offsets:
        compare_kwargs = change_offsets(compare_kwargs, offsets)

    Compared_ = Comparer(
        data=data,
        method=method,
        method_kwargs=method_kwargs,
        compare_kwargs=compare_kwargs,
        convert_data=True,
    )
    compared = Compared_.compared
    data_ = Compared_.data

    if ignore_entries is None:
        return DupDetect(data, compared, method, method_kwargs, compare_kwargs)

    compared = [compared]

    for column_, entry_ in ignore_entries.items():
        if isinstance(entry_, str):
            entry_ = [entry_]
        entries = data[column_].isin(entry_)

        d1 = data.mask(entries).dropna()
        d2 = data.where(entries).dropna()

        if d1.empty:
            continue
        if d2.empty:
            continue

        method_kwargs_ = remove_ignores(method_kwargs, column_)
        compare_kwargs_ = remove_ignores(compare_kwargs, column_)

        compared_ = Comparer(
            data=data_,
            method=method,
            method_kwargs=method_kwargs_,
            compare_kwargs=compare_kwargs_,
            pairs_df=[d2, d1],
        ).compared
        compared_[list(ignore_entries.keys())] = 1
        compared.append(compared_)

    compared = pd.concat(compared)
    data.set_index(index, inplace=True)
    return DupDetect(data, compared, method, method_kwargs, compare_kwargs)
