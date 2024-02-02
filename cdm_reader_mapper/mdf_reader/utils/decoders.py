"""pandas decoding operators."""

from __future__ import annotations

import string

import numpy as np
import pandas as pd

from .. import properties

# for importer, modname, ispkg in pkgutil.walk_packages(path=package.__path__,prefix=package.__name__+'.',onerror=lambda x: None):
#    print(modname.split(".")[-1])
# TO DECODE FROM OBJECT TO INTEGER
#
# Decodes input object type pd.series to a specified data type
#
# On missing data, the resulting DATA type in numerics will be as integer promotion to accomodate np.nan:
# Promotion dtype for storing NAs: integer	cast to float64
# (https://pandas.pydata.org/pandas-docs/version/0.22/gotchas.html#nan-integer-na-values-and-na-type-promotions)
#
# return base10.astype(self.dtype, casting = 'safe')
# safe casting specified, otherwise converts np.nan to some number depending on dtype.


class df_decoders:
    """DOCUMENTATION."""

    def __init__(self, dtype):
        # Return as object, conversion to actual type in converters only!
        self.dtype = "object"

    def signed_overpunch(self, data):
        """DOCUMENTATION."""
        decoded_numeric = np.vectorize(signed_overpunch_i, otypes=[float])(data)
        return pd.Series(decoded_numeric, dtype=self.dtype)

    def base36(self, data):
        """DOCUMENTATION."""
        # Caution: int(str(np.nan),36) ==> 30191
        base10 = [str(int(str(i), 36)) if i == i and i else np.nan for i in data]

        return pd.Series(base10, dtype=self.dtype)


decoders = dict()

decoders["signed_overpunch"] = dict()
for dtype in properties.numeric_types:
    decoders["signed_overpunch"][dtype] = df_decoders(dtype).signed_overpunch
decoders["signed_overpunch"]["key"] = df_decoders("key").signed_overpunch

decoders["base36"] = dict()
for dtype in properties.numeric_types:
    decoders["base36"][dtype] = df_decoders(dtype).base36
decoders["base36"]["key"] = df_decoders("key").base36


## Now add the file format specific decoders
# import pkgutil
# import importlib
# from cdm_reader_mapper.mdf_reader import fs_decoders
# package=fs_decoders
# for importer, modname, ispkg in pkgutil.walk_packages(path=package.__path__,prefix=package.__name__+'.',onerror=lambda x: None):
#    file_format = modname.split(".")[-1]
#    try:
#        file_format_decoders = importlib.import_module(modname, package=None).decoders
#        for decoder in file_format_decoders.keys():
#            decoders[".".join([file_format,decoder])] = file_format_decoders.get(decoder)
#    except Exception as e:
#        logging.error("Error loading {0} decoders: {1}".format(modname,e))
#
