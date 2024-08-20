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
        "block_on": ["report_id"],
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
        "source_id": {"method": "exact"},
    },
}


class DupDetect:
    """DOCUMENTATION."""

    def __init__(self, data, compared, method_kwargs, compare_kwargs):
        self.data = data
        self.compared = compared
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
        """DOCUMENTATION."""
        pcmax = self.compared.shape[1]
        self.score = 1 - (abs(self.compared.sum(axis=1) - pcmax) / pcmax)

    def get_matches(self, limit="default", equal_musts=None):
        """DOCUMENTATION."""
        self.limit = self._get_limit(limit)
        cond = self.score >= self.limit
        if equal_musts is None:
            equal_musts = self._get_equal_musts()
        for must in equal_musts:
            cond = cond & (self.compared[must])
        self.matches = self.compared[cond]

    def delete_matches(self, keep="first"):
        """DOCUMENTATION."""
        if keep == "first":
            keep = 0
        elif keep == "last":
            keep = -1
        elif not isinstance(keep, int):
            raise ValueError("keep has to be one of 'first', 'last' of integer value.")
        self.result = self.data.copy()
        for index in self.matches.index:
            self.result = self.result.drop(index[keep])

    def remove_duplicates(self, keep="first", limit="default", equal_musts=None):
        """DOCUMENTATION."""
        self.total_score()
        self.get_matches(limit, equal_musts=equal_musts)
        self.delete_matches(keep)


def set_comparer(compare_dict):
    """DOCUMENTATION."""
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
    """DOCUMENTATION."""
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
    return DupDetect(data, compared, method_kwargs, compare_kwargs)
