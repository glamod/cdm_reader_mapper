"""Common Data Model (CDM) reader and mapper test data."""

from __future__ import annotations

import os


class test_data:
    """CDM test data."""

    def __init__(self):
        self.data_path = os.path.dirname(os.path.abspath(__file__))

        self.test_063_714 = self._get_data_dict("063-714_2010-07_subset.imma", "imma1")
        self.test_069_701 = self._get_data_dict("069-701_1845-04_subset.imma", "imma1")
        self.test_125_704 = self._get_data_dict("125-704_1878-10_subset.imma", "imma1")
        self.test_133_730 = self._get_data_dict("133-730_1776-10_subset.imma", "imma1")

    def _get_data_dict(self, data_file, schema):
        return {
            "source": os.path.join(self.data_path, data_file),
            "data_model": schema,
        }


test_data = test_data()
