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

from . import logging_hdlr


def replace_columns(
    df_l: pd.DataFrame,
    df_r: pd.DataFrame,
    pivot_c: str | None = None,
    pivot_l: str | None = None,
    pivot_r: str | None = None,
    rep_c: str | list[str] | None = None,
    rep_map: dict[str, str] | None = None,
    log_level: str = "INFO",
) -> pd.DataFrame | None:
    """
    Replace columns in one DataFrame using row-matching from another.

    Parameters
    ----------
    df_l : pandas.DataFrame
        The left DataFrame whose columns will be replaced.
    df_r : pandas.DataFrame
        The right DataFrame providing replacement values.
    pivot_c : str, optional
        A single pivot column present in both DataFrames.
        Overrides `pivot_l` and `pivot_r`.
    pivot_l : str, optional
        Pivot column in `df_l`. Used only when `pivot_c` is not supplied.
    pivot_r : str, optional
        Pivot column in `df_r`. Used only when `pivot_c` is not supplied.
    rep_c : str or list of str, optional
        One or more column names to replace in `df_l`.
        Ignored if `rep_map` is supplied.
    rep_map : dict, optional
        Mapping between left and right column names as `{left_col: right_col}`.
    log_level : str, optional
        Logging level to use.

    Returns
    -------
    pandas.DataFrame or None
        Updated DataFrame with replacements applied, or `None` if validation fails.

    Notes
    -----
    This function logs errors and returns `None` instead of raising exceptions.
    """
    logger = logging_hdlr.init_logger(__name__, level=log_level)

    # Check inargs
    if not isinstance(df_l, pd.DataFrame) or not isinstance(df_r, pd.DataFrame):
        logger.error("Input left and right data must be pandas DataFrames.")
        return None

    if pivot_c is not None:
        pivot_l = pivot_r = pivot_c

    if pivot_l is None or pivot_r is None:
        logger.error(
            "Pivot columns must be declared using `pivot_c` or both `pivot_l` and `pivot_r`."
        )
        return None

    if rep_map is None:
        if rep_c is None:
            logger.error(
                "Replacement columns must be declared using `rep_c` or `rep_map`."
            )
            return None
        if isinstance(rep_c, str):
            rep_c = [rep_c]
        rep_map = {col: col for col in rep_c}

    missing_cols = [src for src in rep_map.values() if src not in df_r.columns]
    if missing_cols:
        logger.error(
            f"Replacement source columns not found in right DataFrame: {missing_cols}."
        )
        return None

    out = df_l.copy()
    right_lookup = (
        df_r[[pivot_r, *rep_map.values()]]
        .set_index(pivot_r)
        .rename(columns={v: k for k, v in rep_map.items()})
    )

    # Align once using reindex (vectorized, C-level)
    aligned = right_lookup.reindex(out[pivot_l].values)

    # Assign columns directly (fastest path)
    for col in aligned.columns:
        out[col] = aligned[col].values

    return out
