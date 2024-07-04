"""
Common Data Model (CDM) pandas replacement operators.

Created on Wed Jul  3 09:48:18 2019

Replace columns from right dataframe into left dataframe

Replacement occurs on a pivot column, that might have the same name in both
dfs (pivot_c) or be different (pivot_l and pivot_r)

Can replace one or multiple columns and support multiindexing (tested only on left, so far...)

Replacement arguments:
    - rep_c : list or string of column name(s) to replace, they are the same name in left and right
    - rep_map: dictionary with {col_l:col_r...} if not the same

@author: iregon
"""

from __future__ import annotations

import pandas as pd

from cdm_reader_mapper.common import logging_hdlr


def replace_columns(
    df_l,
    df_r,
    pivot_c=None,
    pivot_l=None,
    pivot_r=None,
    rep_c=None,
    rep_map=None,
    log_level="INFO",
):
    """DOCUMENTATION."""
    logger = logging_hdlr.init_logger(__name__, level=log_level)
    df_l = df_l.copy()
    df_r = df_r.copy()
    # Check inargs
    if not isinstance(df_l, pd.DataFrame) or not isinstance(df_r, pd.DataFrame):
        logger.error("Input left and right data must be pandas dataframes")
        return
    if not pivot_c and not (pivot_l and pivot_r):
        logger.error("Pivot columns must be declared correctly")
        return
    elif pivot_c:
        pivot_l = pivot_c
        pivot_r = pivot_c
    # Now index on pivot
    df_l = df_l.set_index(pivot_l, drop=False)
    df_r = df_r.set_index(pivot_r, drop=False)

    if not rep_c and not rep_map:
        logger.error(
            "Replacement columns must be declared with a list (rep_c) or a dictionary (rep_map)"
        )
        return

    # Subsample right df to what's going to be replaced renaming to left and
    # making sure we have right cols replicated when they are used to be mapped to multiple cols on left
    if rep_c:
        rep_c = [rep_c] if not isinstance(rep_c, list) else rep_c
    rep_map = rep_map if rep_map else {x: x for x in rep_c}

    names_l = list(rep_map.keys())
    df_r_l = pd.DataFrame(columns=names_l)
    for i in names_l:
        df_r_l[i] = df_r[rep_map.get(i)]

    # And merge all data from right into left
    df_l.update(df_r_l)
    # Return with index reset to default
    df_l = df_l.reset_index(drop=True)
    return df_l
