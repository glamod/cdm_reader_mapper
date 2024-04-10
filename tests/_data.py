"""cdm_reader_mapper testing suite result files."""
from __future__ import annotations

import pytest  # noqa

from cdm_reader_mapper import mdf_reader, test_data


def _read_data(**kwargs):
    read_ = mdf_reader.read(**kwargs)
    return read_.data, read_.attrs, read_.mask


data_dict = test_data.test_063_714
data_df, attrs_df, mask_df = _read_data(**data_dict)
