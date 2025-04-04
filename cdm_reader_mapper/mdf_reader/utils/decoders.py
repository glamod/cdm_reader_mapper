"""pandas decoding operators."""

from __future__ import annotations

from .. import properties
from .utilities import convert_str_boolean

import pandas as pd


class df_decoders:
    """DOCUMENTATION."""

    def __init__(self, dtype):
        # Return as object, conversion to actual type in converters only!
        self.dtype = "object"

    def base36(self, data) -> pd.Series:
        """DOCUMENTATION."""

        def _base36(x):
            x = convert_str_boolean(x)
            if isinstance(x, bool):
                return x
            return str(int(str(x), 36))

        return data.apply(lambda x: _base36(x))


decoders = {"base36": {}}
for dtype in properties.numeric_types:
    decoders["base36"][dtype] = df_decoders(dtype).base36
decoders["base36"]["key"] = df_decoders("key").base36
