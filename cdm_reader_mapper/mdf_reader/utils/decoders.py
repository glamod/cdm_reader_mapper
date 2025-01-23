"""pandas decoding operators."""

from __future__ import annotations

import string

import numpy as np
import pandas as pd

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


def _get_n(x, overpunch_number):
    return (
        "".join(list(map(lambda x: overpunch_number.get(x, np.nan), list(x))))
        if x == x
        else np.nan
    )


def _get_f(x, overpunch_factor):
    return (
        np.prod(list(map(lambda x: overpunch_factor.get(x, np.nan), list(x))))
        if x == x
        else np.nan
    )


def _get_converted(n, f):
    return f * int(n) if f and n and n == n and f == f else np.nan


def signed_overpunch_i(x):
    """DOCUMENTATION."""
    # Blanks and np.nan as missing data
    # In TDF-11, mix of overpunch and no overpunch: include integers in dictionary
    # Define decoding dictionary: should do this smart-like: None where non-existing keys!!!!
    overpunch_number = _get_overpunch_number()
    overpunch_factor = _get_overpunch_factor()
    try:
        n = _get_n(x, overpunch_number)
        f = _get_f(x, overpunch_factor)
        return _get_converted(n, f)
    except Exception as e:
        print(f"ERROR decoding element: {x}")
        print(e)
        print("Conversion sequence:")
        try:
            print(f"number base conversion: {n}")
        except Exception:
            print("number base conversion not defined")
        try:
            print(f"factor conversion: {f}")
        except Exception:
            print("factor conversion not defined")
        return np.nan


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
        data = (
            np.nan if isinstance(data, str) and (data.isspace() or not data) else data
        )
        return str(int(str(data), 36)) if data == data and data else np.nan


decoders = dict()

decoders["signed_overpunch"] = dict()
for dtype in properties.numeric_types:
    decoders["signed_overpunch"][dtype] = df_decoders(dtype).signed_overpunch
decoders["signed_overpunch"]["key"] = df_decoders("key").signed_overpunch

decoders["base36"] = dict()
for dtype in properties.numeric_types:
    decoders["base36"][dtype] = df_decoders(dtype).base36
decoders["base36"]["key"] = df_decoders("key").base36
