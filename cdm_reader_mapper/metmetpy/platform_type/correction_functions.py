"""
metmetpy correction functions.

Created on Tue Jun 25 09:07:05 2019

@author: iregon
"""

from __future__ import annotations

import pandas as pd


def fill_value(
    fill_serie,
    fill_value,
    self_condition_value=None,
    fillna=False,
    out_condition=pd.DataFrame(),
    out_condition_values=None,
    self_out_conditions="intersect",
):
    """DOCUMENTATION."""
    # Modes:
    #   - if not self_condition_value and not out_condition: force fillna = True
    #   - if self_condition: fillna as requested
    #   - if self_condition and out_coundition:
    #       fillna: as requested
    #       self_out_conditions: intersect(and) or join(or) as requested
    #           - out_condition need always to intersect between them
    #   - fillna always joins with self and out condition combination

    if not self_condition_value and not out_condition_values:
        return fill_serie.fillna(fill_value)

    msk_na = (
        fill_serie.isna() if fillna else pd.Series(index=fill_serie.index, data=False)
    )
    msk_self = (
        (fill_serie == self_condition_value)
        if self_condition_value
        else pd.Series(index=fill_serie.index, data=True)
    )

    if len(out_condition) > 0:
        condition_dataframe = out_condition
        condition_values = out_condition_values
        if isinstance(condition_dataframe, pd.Series):
            msk_out = condition_dataframe == list(condition_values.values())[0]
        else:
            msk_out = pd.concat(
                (condition_dataframe[k] == v for k, v in condition_values.items()),
                axis=1,
            ).all(axis=1)
    else:
        msk_out = pd.Series(index=fill_serie.index, data=True)
        self_out_conditions == "intersect"

    if self_out_conditions == "join":
        msk = pd.concat([msk_self, msk_out], axis=1).any(axis=1)
    else:
        msk = pd.concat([msk_self, msk_out], axis=1).all(axis=1)

    msk = pd.concat([msk, msk_na], axis=1).any(axis=1)

    return fill_serie.mask(msk, other=fill_value)
