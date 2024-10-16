"""Settings for duplicate check."""

from __future__ import annotations

from recordlinkage import Compare
from recordlinkage.compare import Numeric

_method_kwargs = {
    "left_on": "report_timestamp",
    "window": 5,
    "block_on": ["primary_station_id"],
}

_compare_kwargs = {
    "primary_station_id": {"method": "exact"},
    "longitude": {
        "method": "numeric",
        "kwargs": {"method": "step", "offset": 0.11},
    },
    "latitude": {
        "method": "numeric",
        "kwargs": {"method": "step", "offset": 0.11},
    },
    "report_timestamp": {
        "method": "date2",
        "kwargs": {"method": "gauss", "offset": 60.0},
    },
    "station_speed": {
        "method": "numeric",
        "kwargs": {"method": "step", "offset": 0.09},
    },
    "station_course": {
        "method": "numeric",
        "kwargs": {"method": "step", "offset": 0.9},
    },
}

_histories = {
    "duplicate_status": "Added duplicate information - flag",
    "duplicates": "Added duplicate information - duplicates",
}


class Date2(Numeric):
    """Copy of ``rl.compare.Numeric`` class."""

    pass


def date2(self, *args, **kwargs):
    """New method for ``rl.Compare`` object using ``Date2`` object."""
    compare = Date2(*args, **kwargs)
    self.add(compare)
    return self


Compare.date2 = date2
