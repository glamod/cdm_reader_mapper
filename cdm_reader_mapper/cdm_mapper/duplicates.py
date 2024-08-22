"""Common Data Model (CDM) pandas duplicate check."""

from __future__ import annotations

import recordlinkage as rl

_method_kwargs = {
    "header": {
        "left_on": "report_timestamp",
        "window": 5,
        "block_on": ["report_id"],
    },
    "observations": {
        "left_on": "observation_id",
        "window": 5,
        "block_on": ["date_time"],
    },
}

_compare_kwargs = {
    "header": {
        "report_id": {"method": "exact"},
        "primary_station_id": {"method": "exact"},
        "longitude": {
            "method": "numeric",
            "kwargs": {"method": "gauss", "offset": 0.0},
        },
        "latitude": {
            "method": "numeric",
            "kwargs": {"method": "gauss", "offset": 0.0},
        },
        "report_timestamp": {"method": "exact"},
        "source_record_id": {"method": "exact"},
    },
    "observations": {
        "observation_id": {"method": "exact"},
        "report_id": {"method": "exact"},
        "longitude": {
            "method": "numeric",
            "kwargs": {"method": "gauss", "offset": 0.0},
        },
        "latitude": {
            "method": "numeric",
            "kwargs": {"method": "gauss", "offset": 0.0},
        },
        "date_time": {"method": "exact"},
        "source_id": {"method": "exact"},
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
            limit = 0.75
        return limit

    def _get_equal_musts(self):
        equal_musts = []
        for value in self.method_kwargs.values():
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

    def delete_matches(self, keep="first"):
        """Get result data set with deleted matches.

        Parameters
        ----------
        keep: str, ["first", "last"]
            Which entry shpould be kept in result dataset.
        """
        if keep == "first":
            keep = 0
        elif keep == "last":
            keep = -1
        elif not isinstance(keep, int):
            raise ValueError("keep has to be one of 'first', 'last' of integer value.")
        self.result = self.data.copy()
        for index in self.matches.index:
            self.result = self.result.drop(index[keep])
        return self

    def remove_duplicates(self, keep="first", limit="default", equal_musts=None):
        """Remove duplicates from dataset.

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
        self.total_score()
        self.get_matches(limit, equal_musts=equal_musts)
        self.delete_matches(keep)
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
    data = data.astype(comparer.conversion)
    compared = comparer.compute(pairs, data)
    return DupDetect(data, compared, method, method_kwargs, compare_kwargs)
