"""Common Data Model (CDM) pandas duplicate check."""

from __future__ import annotations

import numpy as np
import recordlinkage as rl
from recordlinkage.compare import Numeric


def convert_series(df, conversion):
    def convert_date_to_float(date):
        date = date.astype("datetime64[ns]")
        return (date - date.min()) / np.timedelta64(1, "s")

    df = df.copy()
    for column, method in conversion.items():
        try:
            df[column] = df[column].astype(method)
        except TypeError:
            df[column] = locals()[method](df[column])

    return df


class NumericDate(Numeric):
    pass


def numericdate(self, *args, **kwargs):
    compare = NumericDate(*args, **kwargs)
    self.add(compare)
    return self


rl.Compare.numericdate = numericdate

_method_kwargs = {
    "header": {
        "left_on": "report_timestamp",
        "window": 5,
        "block_on": ["primary_station_id"],
    },
}

_compare_kwargs = {
    "header": {
        "primary_station_id": {"method": "exact"},
        "longitude": {
            "method": "numeric",
            "kwargs": {"method": "gauss", "offset": 0.05},
        },
        "latitude": {
            "method": "numeric",
            "kwargs": {"method": "gauss", "offset": 0.05},
        },
        "report_timestamp": {
            "method": "numericdate",
            "kwargs": {"method": "gauss", "offset": 60.0},
        },
    },
}


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
        if limit == "default":
            limit = 0.991
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
        return self

    def get_matches(self, limit="default", equal_musts=None):
        """Get duplicate matches.

        Parameters
        ----------
        limit: float, optional
            Limit of total score that as to be exceeded to be declared as a duplicate.
            Default: .75
        equal_musts: str or list, optional
            Hashable of column name(s) that must totally be equal to be declared as a duplicate.
            Default: All column names found in method_kwargs.
        """
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
        return self

    def flag_duplicates(self, keep="first", limit="default", equal_musts=None):
        """Get result dataset with flagged duplicates.

        Parameters
        ----------
        keep: str, ["first", "last"]
            Which entry shpould be kept in result dataset.
        limit: float, optional
            Limit of total score that as to be exceeded to be declared as a duplicate.
            Default: .75
        equal_musts: str or list, optional
            Hashable of column name(s) that must totally be equal to be declared as a duplicate.
            Default: All column names found in method_kwargs.
        """
        drop = 0
        if keep == "first":
            keep = -1
            drop = 0
        elif keep == "last":
            keep = 0
            drop = -1
        elif not isinstance(keep, int):
            raise ValueError("keep has to be one of 'first', 'last' of integer value.")

        self.result = self.data.copy()
        self.result["duplicate_status"] = 0
        if not hasattr(self, "matches"):
            self.get_matches(limit="default", equal_musts=None)

        drops = []
        for index in self.matches.index:
            if index[drop] in drops:
                continue

            report_id_drop = self.result.loc[index[drop], "report_id"]
            report_id_keep = self.result.loc[index[keep], "report_id"]
            duplicates_drop = self.result.loc[index[drop], "duplicates"]
            duplicates_keep = self.result.loc[index[keep], "duplicates"]

            self.result.loc[index[keep], "duplicate_status"] = 1
            if self.result.loc[index[keep], "duplicates"] == "null":
                self.result.loc[index[keep], "duplicates"] = report_id_drop
            else:
                self.result.loc[index[keep], "duplicates"] = (
                    f"{duplicates_keep}, {report_id_drop}"
                )

            self.result.loc[index[drop], "duplicate_status"] = 3
            if self.result.loc[index[drop], "duplicates"] == "null":
                self.result.loc[index[drop], "duplicates"] = report_id_keep
            else:
                self.result.loc[index[drop], "duplicates"] = (
                    f"{duplicates_drop}, {report_id_keep}"
                )
            drops.append(index[drop])

        return self

    def remove_duplicates(self, keep="first", limit="default", equal_musts=None):
        """Get result dataset with deleted matches.

        Parameters
        ----------
        keep: str, ["first", "last"]
            Which entry shpould be kept in result dataset.
        limit: float, optional
            Limit of total score that as to be exceeded to be declared as a duplicate.
            Default: .75
        equal_musts: str or list, optional
            Hashable of column name(s) that must totally be equal to be declared as a duplicate.
            Default: All column names found in method_kwargs.
        """
        drop = 0
        if keep == "first":
            drop = 0
        elif keep == "last":
            drop = -1
        elif not isinstance(keep, int):
            raise ValueError("keep has to be one of 'first', 'last' of integer value.")
        self.result = self.data.copy()
        if not hasattr(self, "matches"):
            self.get_matches(limit="default", equal_musts=None)
        for index in self.matches.index:
            if index[drop] in self.result.index:
                self.result = self.result.drop(index[drop])
        self.result = self.result.reset_index(drop=True)
        return self


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
        if method == "numericdate":
            comparer.conversion[column] = "convert_date_to_float"
    return comparer


def duplicate_check(
    data,
    method="SortedNeighbourhood",
    method_kwargs=None,
    compare_kwargs=None,
    cdm_name="header",
    table_name=None,
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
        Default: _method_kwargs[cdm_name]
    compare_kwargs: dict, optional
        Keyword arguments for recordlinkage.Compare object.
        Default: _compare_kwargs[cdm_name]
    cdm_name: str, ["header", "observations"]
        Name of CDM table.
        Use only if at least one of method_kwargs or compare_kwargs is None.
    table_name: str
        Name of the CDM table to be selected from data.

    Returns
    -------
        DupDetect object
    """
    if table_name:
        data = data[table_name]
    if not method_kwargs:
        method_kwargs = _method_kwargs[cdm_name]
    if not compare_kwargs:
        compare_kwargs = _compare_kwargs[cdm_name]

    indexer = getattr(rl.index, method)(**method_kwargs)
    pairs = indexer.index(data)
    comparer = set_comparer(compare_kwargs)
    data_ = convert_series(data, comparer.conversion)
    compared = comparer.compute(pairs, data_)
    return DupDetect(data, compared, method, method_kwargs, compare_kwargs)
