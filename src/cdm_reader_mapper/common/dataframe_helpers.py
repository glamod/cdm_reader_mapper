"""Utility function for reading and writing files."""

from __future__ import annotations
from typing import Any

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


def restore_columns(item: Any) -> Any:
    """
    Restore columns from string literals if `item` is a pandas DataFrame or Series.

    Parameters
    ----------
    item : Any
        Object to restore.

    Returns
    -------
    Any
        Restored object.
    """

    def _literal_eval(column: Any) -> Any:
        """
        Evaluate a string literal if possible.

        Parameters
        ----------
        column : Any
            Column that is possibly a string literal.

        Returns
        -------
        Any
            Evaluated column.
        """
        if not isinstance(column, str):
            return column
        try:
            from ast import literal_eval

            return literal_eval(column)
        except (ValueError, SyntaxError):
            return column

    if isinstance(item, pd.DataFrame):
        columns = item.columns
        new_columns = []
        for column in columns:
            column = _literal_eval(column)
            new_columns.append(column)

        if new_columns and all(isinstance(c, tuple) for c in new_columns) and len({len(c) for c in new_columns}) == 1:
            item.columns = pd.MultiIndex.from_tuples(new_columns)
        else:
            item.columns = new_columns

    if isinstance(item, pd.Series):
        item.name = _literal_eval(item.name)

    return item
