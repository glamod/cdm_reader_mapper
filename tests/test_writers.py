from __future__ import annotations

import pytest  # noqa

from ._testing_cdm_suite import _testing_writers


def test_write_data():
    _testing_writers("icoads_r300_d714")
