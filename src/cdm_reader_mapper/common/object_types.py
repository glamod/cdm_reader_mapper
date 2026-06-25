"""Utility function for reading and writing files."""

from __future__ import annotations

import pandas as pd


def standardize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string columns to object dtype and replace NaNs with None.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be standardized.

    Returns
    -------
    pd.DataFrame
        The same DataFrame instance after the dtype conversion and NaN handling.
    """
    df = df.copy()
    string_cols = df.select_dtypes(include="string").columns
    df[string_cols] = df[string_cols].astype(object)
    object_cols = df.select_dtypes(include="object").columns
    df[object_cols] = df[object_cols].fillna(None)
    return df
