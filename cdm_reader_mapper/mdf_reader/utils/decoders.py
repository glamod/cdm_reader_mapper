"""pandas decoding operators."""

from __future__ import annotations

import logging
import string

import numpy as np
import polars as pl

from .. import properties


def _get_overpunch_number():
    overpunch_number = {string.digits[i]: str(i) for i in range(0, 10)}
    overpunch_number.update(
        {string.ascii_uppercase[i]: str(i + 1) for i in range(0, 9)}
    )
    overpunch_number.update(
        {string.ascii_uppercase[i]: str(i - 8) for i in range(9, 18)}
    )
    overpunch_number.update({"{": str(0)})
    overpunch_number.update({"<": str(0)})
    overpunch_number.update({"}": str(0)})
    overpunch_number.update({"!": str(0)})
    return overpunch_number


def _get_overpunch_factor():
    overpunch_factor = {string.digits[i]: 1 for i in range(0, 10)}
    overpunch_factor.update({string.ascii_uppercase[i]: 1 for i in range(0, 9)})
    overpunch_factor.update({string.ascii_uppercase[i]: -1 for i in range(9, 18)})
    overpunch_factor.update({"}": -1})
    overpunch_factor.update({"!": -1})
    overpunch_factor.update({"{": 1})
    overpunch_factor.update({"<": 1})
    return overpunch_factor


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

    def signed_overpunch(self, data: pl.Series):
        """DOCUMENTATION."""
        # Blanks and np.nan as missing data
        # In TDF-11, mix of overpunch and no overpunch: include integers in dictionary
        # Define decoding dictionary: should do this smart-like: None where non-existing keys!!!!
        overpunch_number = _get_overpunch_number()
        overpunch_factor = _get_overpunch_factor()

        name = data.name
        decoded = (
            data.str.to_uppercase()  # QUESTION: Do I want to uppercase??
            .str.split("")
            .to_frame()
            .with_row_index("i")
            .explode(name)
            .with_columns(
                [
                    pl.col(name).replace(overpunch_number).alias("opn"),
                    (
                        pl.col(name)
                        .replace_strict(overpunch_factor, default=np.nan)
                        .alias("opf")
                    ),
                ]
            )
            .group_by("i")
            .agg(name, pl.col("opn"), pl.col("opf").product())
            .with_columns(
                [
                    pl.col(name).list.join(""),
                    pl.col("opn").list.join("").str.to_integer(strict=False),
                ]
            )
            .with_columns(
                (pl.col("opn") * pl.col("opf")).cast(self.dtype).alias("conv")
            )
            .fill_nan(None)
            .get_column("conv")
        )
        self._check_decode(data, decoded, 20, "signed_overpunch")
        return decoded

    def base36(self, data: pl.Series):
        """DOCUMENTATION."""
        # Caution: int(str(np.nan),36) ==> 30191
        decoded = (
            data.fill_nan(None)
            .str.strip_chars(" ")
            .replace({"": None})
            .str.to_integer(base=36, strict=False)
            .cast(self.dtype)
        )

        self._check_decode(data, decoded, 20, "base36")
        return decoded


decoders = dict()

decoders["signed_overpunch"] = dict()
for dtype in properties.numeric_types:
    decoders["signed_overpunch"][dtype] = df_decoders(dtype).signed_overpunch
decoders["signed_overpunch"]["key"] = df_decoders("key").signed_overpunch

decoders["base36"] = dict()
for dtype in properties.numeric_types:
    decoders["base36"][dtype] = df_decoders(dtype).base36
decoders["base36"]["key"] = df_decoders("key").base36
