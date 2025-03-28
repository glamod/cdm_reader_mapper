"""pandas decoding operators."""

from __future__ import annotations

import logging

import polars as pl

from .. import properties


class df_decoders:
    """DOCUMENTATION."""

    def __init__(self, dtype):
        # Return as object, conversion to actual type in converters only!
        self.dtype = pl.String

    def _check_decode(
        self, data: pl.Series, decoded: pl.Series, threshold: int, method: str
    ):
        if (
            bad_decode := data.filter(decoded.is_null() & data.is_not_null())
        ).len() > 0:
            msg = f"Have {bad_decode.len()} values that failed to be {method} decoded"
            if bad_decode.len() <= threshold:
                msg += f": values = {', '.join(bad_decode)}"
            logging.warning(msg)
        return None

    def base36(self, data):
        """DOCUMENTATION."""
        # Caution: int(str(np.nan),36) ==> 30191
        decoded = (
            data.replace({"NaN": None})
            .str.strip_chars(" ")
            .replace({"": None})
            .str.to_integer(base=36, strict=False)
            .cast(self.dtype)
        )

        self._check_decode(data, decoded, 20, "base36")
        return decoded


decoders = dict()
decoders["base36"] = dict()
for dtype in properties.numeric_types:
    decoders["base36"][dtype] = df_decoders(dtype).base36
decoders["base36"][pl.Categorical] = df_decoders(pl.Categorical).base36
