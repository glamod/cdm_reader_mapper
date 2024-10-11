"""Common Data Model (CDM) pandas duplicate check."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import recordlinkage as rl
from recordlinkage.compare import Numeric


def convert_series(df, conversion):
    """Convert data types in dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame
    conversion: dict
        Conversion dictionary conating columns and
        new data type as key-value pairs.

    Returns
    -------
    pd.DataFrame
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

    return df


class Date2(Numeric):
    """Copy of ``rl.compare.Numeric`` class."""

    pass


def date2(self, *args, **kwargs):
    """New method for ``rl.Compare`` object using ``Date2`` object."""
    compare = Date2(*args, **kwargs)
    self.add(compare)
    return self


rl.Compare.date2 = date2

_method_kwargs = {
    "left_on": "report_timestamp",
    "window": 5,
    "block_on": ["primary_station_id"],
}

_compare_kwargs = {
    "primary_station_id": {"method": "exact"},
    "longitude": {
        "method": "numeric",
        "kwargs": {"method": "gauss", "offset": 0.05},  # C-RAID: 0.005 -> 0.0005
    },
    "latitude": {
        "method": "numeric",
        "kwargs": {"method": "gauss", "offset": 0.05},  # C-RAID: 0.005 -> 0.0005
    },
    "report_timestamp": {
        "method": "date2",
        "kwargs": {"method": "gauss", "offset": 60.0},  # C-RAID: weniger
    },
}

_histories = {
    "duplicate_status": "Added duplicate information - flag",
    "duplicates": "Added duplicate information - duplicates",
}


def add_history(df, indexes):
    """Add duplicate information to history."""

    def _datetime_now():

        try:
            now = datetime.datetime.now(datetime.UTC)
        except AttributeError:
            now = datetime.datetime.utcnow()

        return now.strftime("%Y-%m-%d %H:%M:%S")

    indexes = list(indexes)
    history_tstmp = _datetime_now()
    addition = "".join([f"; {history_tstmp}. {add}" for add in _histories.items()])
    df.loc[indexes, "history"] = df.loc[indexes, "history"].apply(
        lambda x: x + addition
    )
    return df


def add_duplicates(df, dups):
    """Add duplicates to table."""

    for k, v in dups.items():
        v_ = df.loc[v, "report_id"]
        v_ = v_.to_list()
        df.loc[k, "duplicates"] = "{" + ",".join(v_) + "}"

    return df


def add_report_quality(df, indexes_bad):
    """Add report quality to table."""
    df["report_quality"] = df["report_quality"].astype(int)
    df.loc[indexes_bad, "report_quality"] = 1
    return df


def swap_dict_values(dic, v1, v2):
    """Swap two value in a dictionary."""
    dic[v1] = dic[v2]
    del dic[v2]
    for k, v in dic.items():
        if v2 in v:
            dic[k] = [v1]
    return dic


def expand_dict(dic, k, v):
    """Expand dictionary values."""
    if k not in dic.keys():
        dic[k] = [v]
    else:
        dic[k].append(v)
    return dic


class DupDetect:
    """Class for duplicate check.

    Parameters
    ----------
    data: pd.DataFrame
        Original dataset
    compared: pd.DataFrame
        Dataset after duplicate check.
    method: str
        Duplicate check method for recordlinkage.
    method_kwargs: dict
        Keyword arguments for recordlinkage duplicate check.
    compare_kwargs: dict
        Keyword arguments for recordlinkage.Compare object.
    """

    def __init__(self, data, compared, method, method_kwargs, compare_kwargs):
        self.data = data
        self.compared = compared
        self.method = method
        self.method_kwargs = method_kwargs
        self.compare_kwargs = compare_kwargs

    def _get_limit(self, limit):
        _limit = 0.991
        if limit == "default":
            return _limit
        if limit is None:
            return _limit
        return limit

    def _get_equal_musts(self):
        equal_musts = []
        for value in self.compare_kwargs.keys():
            if not isinstance(value, list):
                value = [value]
            for v in value:
                if v in self.data.columns:
                    equal_musts.append(v)
        return equal_musts

    def total_score(self):
        """Get total score of duplicate check."""
        pcmax = self.compared.shape[1]
        self.score = 1 - (abs(self.compared.sum(axis=1) - pcmax) / pcmax)

    def get_duplicates(
        self, keep="first", limit="default", equal_musts=None, overwrite=True
    ):
        """Get duplicate matches.

        Parameters
        ----------
        keep: str, ["first", "last"]
            Which entry shpould be kept in result dataset.
        limit: float, optional
            Limit of total score that as to be exceeded to be declared as a duplicate.
            Default: .991
        equal_musts: str or list, optional
            Hashable of column name(s) that must totally be equal to be declared as a duplicate.
            Default: All column names found in method_kwargs.
        overwrite: bool
            If True overwrite find duplicates again.

        Returns
        -------
        list
            List of tuples containing duplicate matches.
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
            self.total_score()
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
        self, keep="first", limit="default", equal_musts=None, overwrite=True
    ):
        r"""Get result dataset with flagged duplicates.

        Parameters
        ----------
        keep: str, ["first", "last"]
            Which entry shpould be kept in result dataset.
        limit: float, optional
            Limit of total score that as to be exceeded to be declared as a duplicate.
            Default: .991
        equal_musts: str or list, optional
            Hashable of column name(s) that must totally be equal to be declared as a duplicate.
            Default: All column names found in method_kwargs.
        overwrite: bool
            If True overwrite find duplicates again.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with flagged duplicates. \n
            Flags for ``duplicate_status``: see `duplicate_status`_  \n
            Flags for ``report_quality``: see `quality_flag`_


        .. _duplicate_status: https://glamod.github.io/cdm-obs-documentation/tables/code_tables/duplicate_status/duplicate_status.html
        .. _quality_flag: https://glamod.github.io/cdm-obs-documentation/tables/code_tables/quality_flag/quality_flag.html
        """
        self.get_duplicates(keep=keep, limit=limit, equal_musts=equal_musts)
        self.result = self.data.copy()
        self.result["duplicate_status"] = 0
        if not hasattr(self, "matches"):
            self.get_matches(limit="default", equal_musts=equal_musts)

        indexes_good = []
        indexes_bad = []
        duplicates = {}

        for index in self.matches.index:
            keep_ = index[self.keep]
            drop_ = index[self.drop]

            if drop_ in indexes_bad:
                continue

            if drop_ in indexes_good:
                indexes_good.remove(drop_)
                duplicates = swap_dict_values(duplicates, keep_, drop_)

            if keep_ not in indexes_good:
                indexes_good.append(keep_)
            if drop_ not in indexes_bad:
                indexes_bad.append(drop_)

            duplicates = expand_dict(duplicates, drop_, keep_)
            duplicates = expand_dict(duplicates, keep_, drop_)

        indexes = indexes_good + indexes_bad

        self.result.loc[indexes_good, "duplicate_status"] = 1
        self.result.loc[indexes_bad, "duplicate_status"] = 3

        self.result = add_report_quality(self.result, indexes_bad=indexes_bad)
        self.result = add_duplicates(self.result, duplicates)
        self.result = add_history(self.result, indexes)
        return self.result

    def remove_duplicates(
        self, keep="first", limit="default", equal_musts=None, overwrite=True
    ):
        """Get result dataset with deleted matches.

        Parameters
        ----------
        keep: str, ["first", "last"]
            Which entry shpould be kept in result dataset.
        limit: float, optional
            Limit of total score that as to be exceeded to be declared as a duplicate.
            Default: .991
        equal_musts: str or list, optional
            Hashable of column name(s) that must totally be equal to be declared as a duplicate.
            Default: All column names found in method_kwargs.
        overwrite: bool
            If True overwrite find duplicates again.

        Returns
        -------
        pd.DataFrame
            Input DataFrame without duplicates.
        """
        self.get_duplicates(keep=keep, limit=limit, equal_musts=equal_musts)
        self.result = self.data.copy()
        drops = [index[self.drop] for index in self.matches.index]
        self.result = self.result.drop(drops).reset_index(drop=True)
        return self.result


def set_comparer(compare_dict):
    """Set recordlinkage Comparer.

    Parameters
    ----------
    compare_dict: dict
        Keyword arguments for recordlinkage.Compare object.

    Returns
    -------
    recordlinkage.Compare object:
        recordlinkage.Compare object with added methods.
    """
    comparer = rl.Compare()
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


def remove_ignores(dic, columns):
    """Remove entries containing column names."""
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


def multiply_entries(data, columns, entries):
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(entries, str):
        entries = [entries]

    for column in columns:
        given_entries = data[column].drop_duplicates().values()
        for entry in entries:
            selected = data[column] == entry
            for given_entry in given_entries:
                renamed = selected.replace({entry: given_entry})


def duplicate_check(
    data,
    method="SortedNeighbourhood",
    method_kwargs=None,
    compare_kwargs=None,
    table_name=None,
    ignore_columns=None,
    ignore_entries=["SHIP", "MASKSTID"],
):
    """Duplicate check.

    Parameters
    ----------
    data: pd.DataFrame
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
    ignore_entries: str or list, optional
        Name of column entries to be ignored for duplicate check.
        Those values will be renamed and added to each block_on group in rl.index object.

    Returns
    -------
        DupDetect object
    """
    if table_name:
        data = data[table_name]
    if not method_kwargs:
        method_kwargs = _method_kwargs
    if not compare_kwargs:
        compare_kwargs = _compare_kwargs
    if ignore_columns:
        method_kwargs = remove_ignores(method_kwargs, ignore_columns)
        compare_kwargs = remove_ignores(compare_kwargs, ignore_columns)
    if isinstance(ignore_entries, str):
        ignore_entries = [ignore_entries]
    if ignore_entries is None:
        ignore_entries = []

    indexer = getattr(rl.index, method)(**method_kwargs)
    comparer = set_comparer(compare_kwargs)
    data_ = convert_series(data, comparer.conversion)
    pairs = indexer.index(data_)
    compared = [comparer.compute(pairs, data_)]

    block_ons = method_kwargs.get("block_on")
    if not block_ons is None:
        if not isinstance(block_ons, list):
            block_ons = [block_ons]
        for block_on in block_ons:
            for ignore_entry in ignore_entries:
                d1 = data.where(data[block_on] != ignore_entry).dropna()
                d2 = data.where(data[block_on] == ignore_entry).dropna()

                if d1.empty:
                    continue
                if d2.empty:
                    continue

                method_kwargs_ = remove_ignores(method_kwargs, block_on)
                compare_kwargs_ = remove_ignores(compare_kwargs, block_on)
                indexer = getattr(rl.index, method)(**method_kwargs_)
                pairs = indexer.index(d2, d1)
                comparer = set_comparer(compare_kwargs_)
                compared_ = comparer.compute(pairs, data_)
                compared_[block_ons] = 1
                compared.append(compared_)

    compared = pd.concat(compared)
    return DupDetect(data, compared, method, method_kwargs, compare_kwargs)
