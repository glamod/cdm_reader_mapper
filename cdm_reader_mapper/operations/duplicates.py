"""Common Data Model (CDM) pandas duplicate check."""

from __future__ import annotations

import recordlinkage as rl

_method_kwargs = {
    "header": {
        "left_on": "report_timestamp",
        "window": 5,
        "block_on": ["report_id"],
    },
    "observation": {
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
    "observation": {
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

    def __init__(self, data, compared):
        self.data = data
        self.compared = compared

    def _get_limit(self, limit):
        if limit == "default":
            limit = 0.75
        return limit

    def total_score(self):
        """DOCUMENTATION."""
        pcmax = self.compared.shape[1]
        self.score = 1 - (abs(self.compared.sum(axis=1) - pcmax) / pcmax)

    def get_matches(self, limit="default"):
        """DOCUMENTATION."""
        self.limit = self._get_limit(limit)
        self.matches = self.compared[self.score >= self.limit]

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

    def remove_duplicates(self, keep="first", limit="default"):
        """DOCUMENTATION."""
        self.total_score()
        self.get_matches(limit)
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
    method_kwargs={},
    compare_kwargs={},
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
    return DupDetect(data, compared)
