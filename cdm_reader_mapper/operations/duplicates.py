"""Common Data Model (CDM) pandas duplicate check."""

from __future__ import annotations

import recordlinkage as rl

_method_kwargs = {
    "left_on": "report_timestamp",
    "window": 5,
    "block_on": ["report_id"],
}

_compare_kwargs = {
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
}

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
    table_name=None,
):
    """DOCUMENTATION."""
    if table_name:
        data = data[table_name]
    if not method_kwargs:
        method_kwargs = _method_kwargs
    if not compare_kwargs:
        compare_kwargs = _compare_kwargs
    indexer = getattr(rl.index, method)(**method_kwargs)
    pairs = indexer.index(data)
    comparer = set_comparer(compare_kwargs)
    data = data.astype(comparer.conversion)
    return comparer.compute(pairs, data)
