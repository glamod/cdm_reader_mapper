"""cdm_reader_mapper testing suite result files."""

from __future__ import annotations

import pytest  # noqa

from cdm_reader_mapper import mdf_reader, test_data


def _read_data(**kwargs):
    read_ = mdf_reader.read(**kwargs)
    return read_.data, read_.attrs, read_.mask


data_dict = dict(test_data.test_icoads_r300_d721)
data_df, attrs_df, mask_df = _read_data(**data_dict, imodel="icoads_r300_d721")
data_pa, attrs_pa, mask_pa = _read_data(
    chunksize=10000, **data_dict, imodel="icoads_r300_d721"
)
