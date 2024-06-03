"""
Created on Thu Apr 11 13:45:38 2019.

Module to handle data models mappings to C3S Climate Data Store
Common Data Model (CMD) tables within the cdm tool.

@author: iregon
"""
from __future__ import annotations

import datetime
import json
import os
from copy import deepcopy

from cdm_reader_mapper.common import logging_hdlr
from cdm_reader_mapper.common.getting_files import get_files

from ..properties import _base


class codes_hdlr:
    """Class for readinf CDM mapper code tables."""

    def __init__(self, imodel, log_level="INFO"):
        self.imodel = imodel
        self.logger = logging_hdlr.init_logger(__name__, level=log_level)
        self._common = f"{_base}.codes"
        self._imodel = f"{self._common}.{imodel}"
        self._common_data = get_files(self._common)
        try:
            self._imodel_data = get_files(self._imodel)
        except ModuleNotFoundError:
            self._imodel_data = None
            self.logger.warning(f"No specific code mappings for model {imodel}")

    def load_code_tables_maps(self, codes_subset=None):
        """Load CDM code tables."""
        codes_common = self._common_data.glob("*.json")
        codes_common_dict = {
            os.path.basename(x).split(".")[0]: x for x in list(codes_common)
        }
        if self._imodel_data is not None:
            codes_imodel = self._imodel_data.glob("*.json")
            codes_imodel_dict = {
                os.path.basename(x).split(".")[0]: x for x in list(codes_imodel)
            }
        else:
            codes_imodel_dict = {}
        codes_dict = dict(codes_common_dict, **codes_imodel_dict)

        if codes_subset:
            not_in_cdm = [x for x in codes_subset if x not in codes_dict.keys()]
            if any(not_in_cdm):
                self.logger.error(
                    "A wrong code table was requested for in model {}: {}".format(
                        self._imodel, ",".join(not_in_cdm)
                    )
                )
                self.logger.info(
                    "code tables registered for model are: {}".format(
                        ",".join(list(codes_dict.keys()))
                    )
                )
                return
            remove_codes = [x for x in codes_dict.keys() if x not in codes_subset]
            for x in remove_codes:
                codes_dict.pop(x, None)

        codes = dict()
        for key in codes_dict.keys():
            with open(codes_dict.get(key)) as fileObj:
                codes[key] = json.load(fileObj)
            self.expand_integer_range_key(codes[key])
        return codes

    def expand_integer_range_key(self, d):
        """Expand integer ranges."""
        if isinstance(d, dict):
            for k, v in list(d.items()):
                if "range_key" in k[0:9]:
                    range_params = k[10:-1].split(",")
                    try:
                        lower = int(range_params[0])
                    except Exception as e:
                        print("Lower bound parsing error in range key: ", k)
                        print("Error is:")
                        print(e)
                        return
                    try:
                        upper = int(range_params[1])
                    except Exception as e:
                        if range_params[1] == "yyyy":
                            upper = datetime.date.today().year
                        else:
                            print("Upper bound parsing error in range key: ", k)
                            print("Error is:")
                            print(e)
                            return
                    if len(range_params) > 2:
                        try:
                            step = int(range_params[2])
                        except Exception as e:
                            print("Range step parsing error in range key: ", k)
                            print("Error is:")
                            print(e)
                            return
                    else:
                        step = 1
                    for i_range in range(lower, upper + 1, step):
                        deep_copy_value = deepcopy(
                            d[k]
                        )  # Otherwiserepetitions are linked and act as one!
                        d.update({str(i_range): deep_copy_value})
                    d.pop(k, None)
                else:
                    for k, v in d.items():
                        self.expand_integer_range_key(v)
