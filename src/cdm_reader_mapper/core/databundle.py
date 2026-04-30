"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd

from cdm_reader_mapper.cdm_mapper.mapper import map_model
from cdm_reader_mapper.common import (
    count_by_cat,
    replace_columns,
    split_by_boolean_false,
    split_by_boolean_true,
    split_by_column_entries,
    split_by_index,
)
from cdm_reader_mapper.common.iterators import is_valid_iterator
from cdm_reader_mapper.duplicates.duplicates import duplicate_check
from cdm_reader_mapper.metmetpy import (
    correct_datetime,
    correct_pt,
    validate_datetime,
    validate_id,
)

from ._utilities import _DataBundle, _copy
from .writer import write


class DataBundle(_DataBundle):
    r"""
    Class for manipulating the MDF data and mapping it to the CDM.

    Parameters
    ----------
    \*args : Any
        Positional arguments for initializing instance.
    \**kwargs : Any
        Keyword-arguments for initializing instance.

    Attributes
    ----------
    data : pd.DataFrame or Iterable[pd.DataFrame], optional
        MDF DataFrame.
    columns : pd.Index, pd.MultiIndex or list, optional
        Column labels of `data`
    dtypes : pd.Series or dict, optional
        Data types of `data`.
    parse_dates : list or bool, optional
        Information how to parse dates on `data`
    mask : pandas.DataFrame, optional
        MDF validation mask
    imodel : str, optional
        Name of the MFD/CDM data model.
    mode : {data, tables}, default: data
        Data mode.

    Examples
    --------
    Getting a :py:class:`~DataBundle` while reading data from disk.

    >>> from cdm_reader_mapper import read_mdf
    >>> db = read_mdf(source="file_on_disk", imodel="custom_model_name")

    Constructing a :py:class:`~DataBundle` from already read MDf data.

    >>> from cdm_reader_mapper import DataBundle
    >>> read = read_mdf(source="file_on_disk", imodel="custom_model_name")
    >>> data_ = read.data
    >>> mask_ = read.mask
    >>> db = DataBundle(data=data_, mask=mask_)

    Constructing a :py:class:`~DataBundle` from already read CDM data.

    >>> from cdm_reader_mapper import read_tables
    >>> tables = read_tables("path_to_files").data
    >>> db = DataBundle(data=tables, mode="tables")
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        r"""
        Initialize a DataBundle instance.

        Parameters
        ----------
        \*args : Any
          Positional arguments for initializing instance.
        \**kwargs : Any
          Keyword-arguments for initializing instance.
        """
        super().__init__(*args, **kwargs)

    def _return_db(self, db: DataBundle, inplace: bool) -> DataBundle | None:
        """
        Return the resulting DataBundle depending on inplace mode.

        Parameters
        ----------
        db : DataBundle
          The DataBundle instance containing updated data.
        inplace : bool
          If True modifications are applied in place and None is returned.

        Returns
        -------
        :py:class:`~DataBundle` or None
            Returns `db` if `inplace` is False, otherwise None.
        """
        if inplace is True:
            return None
        return db

    def _get_db(self, inplace: bool) -> DataBundle | None:
        """
        Retrieve the target DataBundle for modification.

        Parameters
        ----------
        inplace : bool
          If True return the current instance; otherwise return a copy.

        Returns
        -------
        :py:class:`~DataBundle` or None
          The DataBundle instance to operate on.
        """
        if inplace is True:
            return self
        return self.copy()

    def _stack(self, other: str | list[str], datasets: pd.DataFrame | list[pd.DataFrame], inplace: bool, **kwargs: Any) -> DataBundle | None:
        r"""
        Concatenate datasets from multiple DataBundle instances.

        Parameters
        ----------
        other : str or list or str
          Attribute name(s) of other DataBundle instances whose data should
          be stacked with the current instance.
        datasets : pd.DataFrame or list of pdataFrame]
          Dataset attribute name(s) to be concatenated (e.g., "data", "mask").
        inplace : bool
          If True modify the current instance in place.
        \**kwargs : Any
          Additional keyword-arguments for stacking DataFrames.

        Returns
        -------
        :py:class:`~DataBundle` or None
          Updated DataBundle if ``inplace`` is False, otherwise None.

        Raises
        ------
        ValueError
          If any dataset is an iterator instead of a pandas DataFrame.
        """
        db_cp = self._get_db(inplace)

        if not isinstance(other, list):
            other = [other]
        if not isinstance(datasets, list):
            datasets = [datasets]

        for data in datasets:
            data_attr = f"_{data}"
            df_cp = getattr(db_cp, data_attr, pd.DataFrame())

            if is_valid_iterator(df_cp):
                raise ValueError("Data must be a pd.DataFrame not a iterable of pd.DataFrames.")

            to_concat = [df_cp]
            to_concat.extend(getattr(o, data_attr) for o in other if hasattr(o, data_attr))

            if not any(d.empty for d in to_concat):
                concatenated = pd.concat(to_concat, **kwargs)
            else:
                concatenated = pd.DataFrame()

            concatenated = concatenated.reset_index(drop=True)

            setattr(db_cp, data_attr, concatenated)

        return self._return_db(db_cp, inplace)

    def add(self, addition: dict[str, pd.DataFrame | pd.Series], inplace: bool = False) -> DataBundle | None:
        """
        Adding information to a :py:class:`~DataBundle`.

        Parameters
        ----------
        addition : dict
             Additional elements to add to the :py:class:`~DataBundle`.
        inplace : bool, default: False
            If True add datasets in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with added datasets.

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle with added information or None if "inplace=True".

        Examples
        --------
        >>> tables = read_tables("path_to_files")
        >>> db = db.add({"data": tables})
        """
        db_ = self._get_db(inplace)
        for name, data in addition.items():
            data_cp = _copy(data)
            setattr(db_, f"_{name}", data_cp)
        return self._return_db(db_, inplace)

    def copy(self) -> DataBundle:
        """
        Make deep copy of a :py:class:`~DataBundle`.

        Returns
        -------
        :py:class:`~DataBundle`
              Copy of a DataBundle.

        Examples
        --------
        >>> db2 = db.copy()
        """
        db = DataBundle()
        for key, value in self.__dict__.items():
            value = _copy(value)
            setattr(db, key, value)
        return db

    def stack_v(
        self, other: str | list[str], datasets: str | Sequence[str] | Literal["data", "mask"] = ("data", "mask"), inplace: bool = False, **kwargs: Any
    ) -> DataBundle | None:
        r"""
        Stack multiple :py:class:`~DataBundle`'s vertically.

        Parameters
        ----------
        other : str or list of str
            List of other DataBundles to stack vertically.
        datasets : str or Sequence of str, default: (data, mask)
            List of datasets to be stacked.
        inplace : bool, default: False
            If True overwrite datasets in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with stacked datasets.
        \**kwargs : Any
            Additional keyword-arguments for stacking DataFrames vertically.

        Returns
        -------
        :py:class:`~DataBundle` or None
            Vertically stacked DataBundle or None if "inplace=True".

        See Also
        --------
        DataBundle.stack_h : Stack multiple DataBundle's horizontally.

        Notes
        -----
        * This is only working with pd.DataFrames, not with iterables of pd.DataFrames!
        * The DataFrames in the :py:class:`~DataBundle` have to have the same data columns!

        Examples
        --------
        >>> db = db1.stack_v(db2, datasets=["data", "mask"])
        """
        return self._stack(other, datasets, inplace, **kwargs)

    def stack_h(
        self, other: str | list[str], datasets: str | Sequence[str] | Literal["data", "mask"] = ("data", "mask"), inplace: bool = False, **kwargs: Any
    ) -> DataBundle | None:
        r"""
        Stack multiple :py:class:`~DataBundle`'s horizontally.

        Parameters
        ----------
        other : str or list of str
            List of other :py:class:`~DataBundle` to stack horizontally.
        datasets : str or list of str, default: [data, mask]
            List of datasets to be stacked.
        inplace : bool, default: False
            If True overwrite `datasets` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with stacked datasets.
        \**kwargs : Any
            Additional keyword-arguments for stacking DataFrames horizontally.

        Returns
        -------
        :py:class:`~DataBundle` or None
            Horizontally stacked DataBundle or None if ``inplace=True``.

        See Also
        --------
        DataBundle.stack_v : Stack multiple DataBundle's vertically.

        Notes
        -----
        * This is only working with pd.DataFrames, not with iterables of pd.DataFrames!
        * The DataFrames in the :py:class:`~DataBundle` may have different data columns!

        Examples
        --------
        >>> db = db1.stack_h(db2, datasets=["data", "mask"])
        """
        return self._stack(other, datasets, inplace, axis=1, join="outer", **kwargs)

    def select_where_all_true(self, inplace: bool = False, do_mask: bool = True, **kwargs: Any) -> DataBundle | None:
        r"""
        Select rows from :py:attr:`data` where all column entries in :py:attr:`mask` are True.

        Parameters
        ----------
        inplace : bool, default: False
            If True overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with valid values only in :py:attr:`data`.
        do_mask : bool, default: True
            If True also do selection on :py:attr:`mask`.
        \**kwargs : Any
            Additional keyword-arguments for splitting `data` where all entries are True.

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle containing rows where all column entries in :py:attr:`mask` are True or None if ``inplace=True``.

        See Also
        --------
        DataBundle.select_where_all_false : Select rows from `data` where all entries in `mask` are False.
        DataBundle.select_where_entry_isin : Select rows from `data` where column entries are in a specific value list.
        DataBundle.select_where_index_isin : Select rows from `data` within specific index list.

        Notes
        -----
        For more information see :py:func:`split_by_boolean_true`

        Examples
        --------
        Select without overwriting the old data.

        >>> db_selected = db.select_where_all_true()

        Select overwriting the old data.

        >>> db.select_where_all_true(inplace=True)
        >>> df_selected = db.data
        """
        db_ = self._get_db(inplace)
        _mask = _copy(db_._mask)
        db_._data, _, selected_idx, _ = split_by_boolean_true(db_._data, _mask, **kwargs)
        if do_mask is True:
            db_._mask, _, _, _ = split_by_index(db_._mask, selected_idx, **kwargs)
        return self._return_db(db_, inplace)

    def select_where_all_false(self, inplace: bool = False, do_mask: bool = True, **kwargs: Any) -> DataBundle | None:
        r"""
        Select rows from :py:attr:`data` where all column entries in :py:attr:`mask` are False.

        Parameters
        ----------
        inplace : bool, default: False
            If True overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with invalid values only in :py:attr:`data`.
        do_mask : bool, default: True
            If True also do selection on :py:attr:`mask`.
        \**kwargs : Any
            Additional keyword-arguments for splitting `data` where all entries are False.

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle containing rows where all column entries in :py:attr:`mask` are False or None if ``inplace=True``.

        See Also
        --------
        DataBundle.select_where_all_true : Select rows from `data` where all entries in `mask` are True.
        DataBundle.select_where_entry_isin : Select rows from `data` where column entries are in a specific value list.
        DataBundle.select_where_index_isin : Select rows from `data` within specific index list.

        Notes
        -----
        For more information see :py:func:`split_by_boolean_false`

        Examples
        --------
        Select without overwriting the old data.

        >>> db_selected = db.select_where_all_false()

        Select valid values only with overwriting the old data.

        >>> db.select_where_all_false(inplace=True)
        >>> df_selected = db.data
        """
        db_ = self._get_db(inplace)
        _mask = _copy(db_._mask)
        db_._data, _, selected_idx, _ = split_by_boolean_false(db_._data, _mask, **kwargs)
        if do_mask is True:
            db_._mask, _, _, _ = split_by_index(db_._mask, selected_idx, **kwargs)
        return self._return_db(db_, inplace)

    def select_where_entry_isin(
        self, selection: dict[str | tuple[str, str], Sequence[Any]], inplace: bool = False, do_mask: bool = True, **kwargs: Any
    ) -> DataBundle | None:
        r"""
        Select rows from :py:attr:`data` where column entries are in a specific value list.

        Parameters
        ----------
        selection : dict
            Keys: Column names in :py:attr:`data`.
            Values: Specific value list.
        inplace : bool, default: False
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with selected columns only in :py:attr:`data`.
        do_mask : bool, default: True
            If True also do selection on :py:attr:`mask`.
        \**kwargs : Any
            Additional keyword-arguments for splitting `data` where entries within a specific value list.

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle containing rows where column entries are in a specific value list or None if ``inplace=True``.

        See Also
        --------
        DataBundle.select_where_index_isin : Select rows from `data` within specific index list.
        DataBundle.select_where_all_true : Select rows from `data` where all entries in `mask` are True.
        DataBundle.select_where_all_false : Select rows from `data` where all entries in `mask` are False.

        Notes
        -----
        For more information see :py:func:`split_by_column_entries`

        Examples
        --------
        Select without overwriting the old data.

        >>> db_selected = db.select_where_entry_isin(
        ...     selection={("c1", "B1"): [26, 41]},
        ... )

        Select with overwriting the old data.

        >>> db.select_where_entry_isin(selection={("c1", "B1"): [26, 41]}, inplace=True)
        >>> df_selected = db.data
        """
        db_ = self._get_db(inplace)
        db_._data, _, selected_idx, _ = split_by_column_entries(db_._data, selection, **kwargs)
        if do_mask is True:
            db_._mask, _, _, _ = split_by_index(db_._mask, selected_idx, **kwargs)
        return self._return_db(db_, inplace)

    def select_where_index_isin(self, index: list[int], inplace: bool = False, do_mask: bool = True, **kwargs: Any) -> DataBundle | None:
        r"""
        Select rows from :py:attr:`data` where indexes within a specific index list.

        Parameters
        ----------
        index : list of int
            Specific index list.
        inplace : bool, default: False
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with selected rows only in :py:attr:`data`.
        do_mask : bool, default: True
            If True also do selection on :py:attr:`mask`.
        \**kwargs : Any
            Additional keyword-arguments for splitting `data` where indexes within a specific index list.

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle containing rows where indexes are within a specific index list or None if ``inplace=True``.

        See Also
        --------
        DataBundle.select_where_entry_isin : Select rows from `data` where column entries are in a specific value list.
        DataBundle.select_where_all_true : Select rows from `data` where all entries in `mask` are True.
        DataBundle.select_where_all_false : Select rows from `data` where all entries in `mask` are False.

        Notes
        -----
        For more information see :py:func:`split_by_index`

        Examples
        --------
        Select without overwriting the old data.

         >>> db_selected = db.select_where_index_isin([0, 2, 4])

        Select with overwriting the old data.

        >>> db.select_where_index_isin(index=[0, 2, 4], inplace=True)
        >>> df_selected = db.data
        """
        db_ = self._get_db(inplace)
        db_._data, _, selected_idx, _ = split_by_index(db_._data, index, **kwargs)
        if do_mask is True:
            db_._mask, _, _, _ = split_by_index(db_._mask, selected_idx, **kwargs)
        return self._return_db(db_, inplace)

    def split_by_boolean_true(self, do_mask: bool = True, **kwargs: Any) -> tuple[DataBundle, DataBundle]:
        r"""
        Split :py:attr:`data` by rows where all column entries in :py:attr:`mask` are True.

        Parameters
        ----------
        do_mask : bool, default: True
            If True also do selection on :py:attr:`mask`.
        \**kwargs : Any
            Additional keyword-arguments for splitting `data` where mask is False.

        Returns
        -------
        tuple
            First :py:class:`~DataBundle` including rows where all column entries in :py:attr:`mask` are True.
            Second :py:class:`~DataBundle` including rows where all column entries in :py:attr:`mask` are False.

        See Also
        --------
        DataBundle.split_by_boolean_false : Split `data` by rows where all entries in `mask` are False.
        DataBundle.split_by_column_entries : Split `data` by rows where column entries are in a specific value list.
        DataBundle.split_by_index : Split `data` by rows within specific index list.

        Notes
        -----
        For more information see :py:func:`split_by_boolean_true`

        Examples
        --------
        Split DataBundle.

        >>> db_true, db_false = db.split_by_boolean_true()
        """
        db1_ = self.copy()
        db2_ = self.copy()
        _mask = _copy(db1_._mask)
        db1_._data, db2_._data, selected_idx, _ = split_by_boolean_true(db1_._data, _mask, return_rejected=True, **kwargs)
        if do_mask is True:
            db1_._mask, db2_._mask, _, _ = split_by_index(db1_._mask, selected_idx, return_rejected=True, **kwargs)
        return db1_, db2_

    def split_by_boolean_false(self, do_mask: bool = True, **kwargs: Any) -> tuple[DataBundle, DataBundle]:
        r"""
        Split :py:attr:`data` by rows where all column entries in :py:attr:`mask` are False.

        Parameters
        ----------
        do_mask : bool, default: True
            If True also do selection on :py:attr:`mask`.
        \**kwargs : Any
            Additional keyword-arguments for splitting `data` where mask is False.

        Returns
        -------
        tuple
            First :py:class:`~DataBundle` including rows where all column entries in :py:attr:`mask` are False.
            Second :py:class:`~DataBundle` including rows where all column entries in :py:attr:`mask` are True.

        See Also
        --------
        DataBundle.split_by_boolean_false : Split `data` by rows where all entries in `mask` are True.
        DataBundle.split_by_column_entries : Split `data` by rows where column entries are in a specific value list.
        DataBundle.split_by_index : Split `data` by rows within specific index list.

        Notes
        -----
        For more information see :py:func:`split_by_boolean_false`

        Examples
        --------
        Split DataBundle.

        >>> db_false, db_true = db.split_by_boolean_false()
        """
        db1_ = self.copy()
        db2_ = self.copy()
        _mask = _copy(db1_._mask)
        db1_._data, db2_._data, selected_idx, _ = split_by_boolean_false(db1_._data, _mask, return_rejected=True, **kwargs)
        if do_mask is True:
            db1_._mask, db2_._mask, _, _ = split_by_index(db1_._mask, selected_idx, return_rejected=True, **kwargs)
        return db1_, db2_

    def split_by_column_entries(
        self, selection: dict[str | tuple[str, str], Sequence[Any]], do_mask: bool = True, **kwargs: Any
    ) -> tuple[DataBundle, DataBundle]:
        r"""
        Split :py:attr:`data` by rows where column entries are in a specific value list.

        Parameters
        ----------
        selection : dict
            Keys: Column names in :py:attr:`data`.
            Values: Specific value list.
        do_mask : bool, default: True
            If True also do selection on :py:attr:`mask`.
        \**kwargs : Any
            Additional keyword-arguments for splitting `data` by column entries.

        Returns
        -------
        tuple
            First :py:class:`~DataBundle` including rows where column entries are in a specific value list.
            Second :py:class:`~DataBundle` including rows where column entries are not in a specific value list.

        See Also
        --------
        DataBundle.split_by_index : Split `data` by rows within specific index list.
        DataBundle.split_by_boolean_true : Split `data` by rows where all entries in `mask` are True.
        DataBundle.split_by_boolean_false : Split `data` by rows where all entries in `mask` are False.

        Notes
        -----
        For more information see :py:func:`split_by_column_entries`

        Examples
        --------
        Split DataBundle.

        >>> db_isin, db_isnotin = db.split_by_column_entries(
        ...     selection={("c1", "B1"): [26, 41]},
        ... )
        """
        db1_ = self.copy()
        db2_ = self.copy()
        db1_._data, db2_._data, selected_idx, _ = split_by_column_entries(db1_._data, selection, return_rejected=True, **kwargs)
        if do_mask is True:
            db1_._mask, db2_._mask, _, _ = split_by_index(db1_._mask, selected_idx, return_rejected=True, **kwargs)
        return db1_, db2_

    def split_by_index(self, index: list[int], do_mask: bool = True, **kwargs: Any) -> tuple[DataBundle, DataBundle]:
        r"""
        Split :py:attr:`data` by rows within specific index list.

        Parameters
        ----------
        index : list of int
            Specific index list.
        do_mask : bool, default: True
            If True also do selection on :py:attr:`mask`.
        \**kwargs : Any
            Additional keyword-arguments for splitting `data` by index.

        Returns
        -------
        tuple
            First :py:class:`~DataBundle` including rows within specific index list.
            Second :py:class:`~DataBundle` including rows outside specific index list.

        See Also
        --------
        DataBundle.split_by_column_entries : Select columns from `data` with specific values.
        DataBundle.split_by_boolean_true : Split `data` by rows where all entries in `mask` are True.
        DataBundle.split_by_boolean_false : Split `data` by rows where all entries in `mask` are False.

        Notes
        -----
        For more information see :py:func:`split_by_index`

        Examples
        --------
        Split DataBundle.

         >>> db_isin, db_isnotin = db.split_by_index([0, 2, 4])
        """
        db1_ = self.copy()
        db2_ = self.copy()
        db1_._data, db2_._data, _, _ = split_by_index(db1_._data, index, return_rejected=True, **kwargs)
        if do_mask is True:
            db1_._mask, db2_._mask, _, _ = split_by_index(db1_._mask, index, return_rejected=True, **kwargs)
        return db1_, db2_

    def unique(self, **kwargs: Any) -> dict[str | tuple[str, str], dict[Any, int]]:
        r"""
        Get unique values of :py:attr:`data`.

        Parameters
        ----------
        \**kwargs : Any
            Additional keyword-arguments for getting unique values.

        Returns
        -------
        dict
            Dictionary with unique values.

        Notes
        -----
        For more information see :py:func:`unique`

        Examples
        --------
        >>> db.unique(columns=("c1", "B1"))
        """
        return count_by_cat(self._data, **kwargs)  # type: ignore[no-any-return]

    def replace_columns(self, df_corr: pd.DataFrame, subset: str | None = None, inplace: bool = False, **kwargs: Any) -> DataBundle | None:
        r"""
        Replace columns in :py:attr:`data`.

        Parameters
        ----------
        df_corr : pd.DataFrame
            Data to be inplaced.
        subset : str or list of str, optional
            Select subset by columns. This option is useful for multi-indexed :py:attr:`data`.
        inplace : bool, default: False
            If True overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with replaced column names in :py:attr:`data`.
        \**kwargs : Any
            Additional keyword-arguments for replacing columns.

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle with replaced column names or None if "inplace=True".

        Notes
        -----
        For more information see :py:func:`replace_columns`

        Examples
        --------
        >>> import pandas as pd
        >>> df_corr = pd.read_csv("correction_file_on_disk")
        >>> df_repl = db.replace_columns(df_corr)
        """
        if not isinstance(self._data, (pd.DataFrame, pd.Series)):
            raise TypeError("Data must be a pd.DataFrame or pd.Series, not a {type(self._data)}.")

        db_ = self._get_db(inplace)
        if subset is None:
            db_._data = replace_columns(df_l=db_._data, df_r=df_corr, **kwargs)
        else:
            db_._data[subset] = replace_columns(df_l=db_._data[subset], df_r=df_corr, **kwargs)
        db_._columns = db_._data.columns
        return self._return_db(db_, inplace)

    def correct_datetime(self, imodel: str | None = None, inplace: bool = False, **kwargs: Any) -> DataBundle | None:
        r"""
        Correct datetime information in :py:attr:`data`.

        Parameters
        ----------
        imodel : str, optional
          Name of the MFD/CDM data model.
        inplace : bool, default: False
            If True overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with datetime-corrected values in :py:attr:`data`.
        \**kwargs : Any
            Additional keyword-arguments for correcting datetime.

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle with corrected datetime information or None if "inplace=True".

        See Also
        --------
        DataBundle.correct_pt : Correct platform type information in `data`.
        DataBundle.validate_datetime: Validate datetime information in `data`.
        DataBundle.validate_id : Validate station id information in `data`.

        Notes
        -----
        For more information see :py:func:`correct_datetime`

        Examples
        --------
        >>> df_dt = db.correct_datetime()
        """
        imodel = imodel or self._imodel
        db_ = self._get_db(inplace)
        db_._data = correct_datetime(db_._data, imodel, **kwargs)
        return self._return_db(db_, inplace)

    def validate_datetime(self, imodel: str | None = None, **kwargs: Any) -> pd.DataFrame:
        r"""
        Validate datetime information in :py:attr:`data`.

        Parameters
        ----------
        imodel : str, optional
          Name of the MFD/CDM data model.
        \**kwargs : Any
            Additional keyword-arguments for validating datetime.

        Returns
        -------
        pd.DataFrame
            DataFrame containing True and False values for each index in :py:attr:`data`.
            True: All datetime information in :py:attr:`data` row are valid.
            False: At least one datetime information in :py:attr:`data` row is invalid.

        See Also
        --------
        DataBundle.validate_id : Validate station id information in `data`.
        DataBundle.correct_datetime : Correct datetime information in `data`.
        DataBundle.correct_pt : Correct platform type information in `data`.

        Notes
        -----
        For more information see :py:func:`validate_datetime`

        Examples
        --------
        >>> val_dt = db.validate_datetime()
        """
        imodel = imodel or self._imodel
        return validate_datetime(self._data, imodel, **kwargs)

    def correct_pt(self, imodel: str | None = None, inplace: bool = False, **kwargs: Any) -> DataBundle | None:
        r"""
        Correct platform type information in :py:attr:`data`.

        Parameters
        ----------
        imodel : str, optional
          Name of the MFD/CDM data model.
        inplace : bool, default: True
            If True overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with platform-corrected values in :py:attr:`data`.
        \**kwargs : Any
            Additional keyword-arguments for correcting platform type.

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle with corrected platform type information or None if "inplace=True".

        See Also
        --------
        DataBundle.correct_datetime : Correct datetime information in `data`.
        DataBundle.validate_id : Validate station id information in `data`.
        DataBundle.validate_datetime : Validate datetime information in `data`.

        Notes
        -----
        For more information see :py:func:`correct_pt`

        Examples
        --------
        >>> df_pt = db.correct_pt()
        """
        imodel = imodel or self._imodel
        db_ = self._get_db(inplace)
        db_._data = correct_pt(db_._data, imodel, **kwargs)
        return self._return_db(db_, inplace)

    def validate_id(self, imodel: str | None = None, **kwargs: Any) -> pd.DataFrame:
        r"""
        Validate station id information in :py:attr:`data`.

        Parameters
        ----------
        imodel : str, optional
          Name of the MFD/CDM data model.
        \**kwargs : Any
            Additional keyword-arguments for validating station id.

        Returns
        -------
        pd.DataFrame
            DataFrame containing True and False values for each index in :py:attr:`data`.
            True: All station ID information in :py:attr:`data` row are valid.
            False: At least one station ID information in :py:attr:`data` row is invalid.

        See Also
        --------
        DataBundle.validate_datetime : Validate datetime information in `data`.
        DataBundle.correct_pt : Correct platform type information in `data`.
        DataBundle.correct_datetime : Correct datetime information in `data`.

        Notes
        -----
        For more information see :py:func:`validate_id`

        Examples
        --------
        >>> val_dt = db.validate_id()
        """
        imodel = imodel or self._imodel
        return validate_id(self._data, imodel, **kwargs)

    def map_model(self, imodel: str | None = None, inplace: bool = False, **kwargs: Any) -> DataBundle | None:
        r"""
        Map :py:attr:`data` to the Common Data Model.

        Parameters
        ----------
        imodel : str, optional
          Name of the MFD/CDM data model.
        inplace : bool, default: False
            If True overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with :py:attr:`data` as CDM tables.
        \**kwargs : Any
            Additional keyword-arguments for mapping to CDM.

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle containing :py:attr:`data` mapped to the CDM or None if ``inplace=True``.

        Notes
        -----
        For more information see :py:func:`map_model`

        Examples
        --------
        >>> cdm_tables = db.map_model()
        """
        imodel = imodel or self._imodel
        db_ = self._get_db(inplace)
        _tables = map_model(db_._data, imodel, **kwargs)
        db_._mode = "tables"
        db_._columns = _tables.columns
        db_._data = _tables
        return self._return_db(db_, inplace)

    def write(
        self,
        dtypes: dict[str | tuple[str, str], str | type] | None = None,
        parse_dates: list[str | tuple[str, str]] | bool | None = None,
        encoding: str | None = None,
        mode: Literal["data", "tables"] | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Write :py:attr:`data` on disk.

        Parameters
        ----------
        dtypes : dict, optional
          Data types of `data`.
        parse_dates : list or bool, optional
          Information how to parse dates on `data`.
        encoding : str, optional
          The encoding of the input file. Overrides the value in the imodel schema file.
        mode : {data, tables}, optional
          Data mode.
        \**kwargs : Any
          Additional keword-arguments for writing data in disk.

        See Also
        --------
        write_data : Write MDF data and validation mask to disk.
        write_tables: Write CDM tables to disk.
        read: Read original marine-meteorological data as well as MDF data or CDM tables from disk.
        read_data: Read MDF data and validation mask from disk.
        read_mdf : Read original marine-meteorological data from disk.

        Notes
        -----
        If :py:attr:`mode` is "data" write data using :py:func:`write_data`.
        If :py:attr:`mode` is "tables" write data using :py:func:`write_tables`.

        Examples
        --------
        >>> db.write()
        read_tables : Read CDM tables from disk.
        """
        dtypes = dtypes or self._dtypes
        parse_dates = parse_dates or self._parse_dates
        encoding = encoding or self._encoding
        mode = mode or self._mode
        write(
            data=self._data,
            mask=self._mask,
            dtypes=dtypes,
            parse_dates=parse_dates,
            encoding=encoding,
            mode=mode,
            **kwargs,
        )

    def duplicate_check(self, inplace: bool = False, **kwargs: Any) -> DataBundle | None:
        r"""
        Duplicate check in :py:attr:`data`.

        Parameters
        ----------
        inplace : bool, default: False
            If True overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with :py:attr:`data` as CDM tables.
        \**kwargs : Any
            Additional keyword-arguments for duplicate check.

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle containing new :py:class:`~DupDetect` class for further duplicate check methods or None if "inplace=True".

        See Also
        --------
        DataBundle.get_duplicates : Get duplicate matches in `data`.
        DataBundle.flag_duplicates : Flag detected duplicates in `data`.
        DataBundle.remove_duplicates : Remove detected duplicates in `data`.

        Notes
        -----
        Following columns have to be provided:

          * `longitude`
          * `latitude`
          * `primary_station_id`
          * `report_timestamp`
          * `station_course`
          * `station_speed`

        This adds a new class :py:class:`~DupDetect` to :py:class:`~DataBundle`.
        This class is necessary for further duplicate check methods.

        For more information see :py:func:`duplicate_check`

        Examples
        --------
        >>> db.duplicate_check()
        """
        db_ = self._get_db(inplace)
        if db_._mode == "tables" and "header" in db_._data:
            data = db_._data["header"]
        else:
            data = db_._data
        db_.DupDetect = duplicate_check(data, **kwargs)
        return self._return_db(db_, inplace)

    def flag_duplicates(self, inplace: bool = False, **kwargs: Any) -> DataBundle | None:
        r"""
        Flag detected duplicates in :py:attr:`data`.

        Parameters
        ----------
        inplace : bool, default: False
            If True overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with :py:attr:`data` containing flagged duplicates.
        \**kwargs : Any
            Additional keyword-arguments for flagging duplicates.

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle containing duplicate flags in :py:attr:`data` or None if "inplace=True".

        Raises
        ------
        RuntimeError
            Before flagging duplicates, a duplictate check has to be done, :py:func:`DataBundle.duplicate_check`.

        See Also
        --------
        DataBundle.remove_duplicates : Remove detected duplicates in `data`.
        DataBundle.get_duplicates : Get duplicate matches in `data`.
        DataBundle.duplicate_check : Duplicate check in `data`.

        Notes
        -----
        For more information see :py:func:`DupDetect.flag_duplicates`

        Examples
        --------
        Flag duplicates without overwriting :py:attr:`data`.

        >>> flagged_tables = db.flag_duplicates()

        Flag duplicates with overwriting :py:attr:`data`.

        >>> db.flag_duplicates(inplace=True)
        >>> flagged_tables = db.data
        """
        db_ = self._get_db(inplace)

        if db_.DupDetect is None:
            raise RuntimeError("Before flagging duplicates, a duplictate check has to be done: 'db.duplicate_check()'")

        db_.DupDetect.flag_duplicates(**kwargs)

        if db_._mode == "tables" and "header" in db_._data:
            db_._data["header"] = db_.DupDetect.result
        else:
            db_._data = db_.DupDetect.result
        return self._return_db(db_, inplace)

    def get_duplicates(self, **kwargs: Any) -> pd.DataFrame:
        r"""
        Get duplicate matches in :py:attr:`data`.

        Parameters
        ----------
        \**kwargs : Any
            Additional keyword-arguments used for getting duplicates.

        Returns
        -------
        pd.DataFrame
            DataFrame containing duplicate matches.

        Raises
        ------
        RuntimeError
            Before getting duplicates, a duplictate check has to be done, :py:func:`DataBundle.duplicate_check`.

        See Also
        --------
        DataBundle.remove_duplicates : Remove detected duplicates in `data`.
        DataBundle.flag_duplicates : Flag detected duplicates in `data`.
        DataBundle.duplicate_check : Duplicate check in `data`.

        Notes
        -----
        For more information see :py:func:`DupDetect.get_duplicates`

        Examples
        --------
        >>> matches = db.get_duplicates()
        """
        if self.DupDetect is None:
            raise RuntimeError("Before getting duplicates, a duplictate check has to be done: 'db.duplicate_check()'")
        return self.DupDetect.get_duplicates(**kwargs)

    def remove_duplicates(self, inplace: bool = False, **kwargs: Any) -> DataBundle | None:
        r"""
        Remove detected duplicates in :py:attr:`data`.

        Parameters
        ----------
        inplace : bool, default: False
            If True overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with :py:attr:`data` containing no duplicates.
        \**kwargs : Any
            Additional keyword-arguments used to remove duplicates.

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle without duplicated rows or None if "inplace=True".

        Raises
        ------
        RuntimeError
            Before removing duplicates, a duplictate check has to be done, :py:func:`DataBundle.duplicate_check`.

        See Also
        --------
        DataBundle.flag_duplicates : Flag detected duplicates in `data`.
        DataBundle.get_duplicates : Get duplicate matches in `data`.
        DataBundle.duplicate_check : Duplicate check in `data`.

        Notes
        -----
        For more information see :py:func:`DupDetect.remove_duplicates`

        Examples
        --------
        Remove duplicates without overwriting :py:attr:`data`.

        >>> removed_tables = db.remove_duplicates()

        Remove duplicates with overwriting :py:attr:`data`.

        >>> db.remove_duplicates(inplace=True)
        >>> removed_tables = db.data
        """
        db_ = self._get_db(inplace)

        if db_.DupDetect is None:
            raise RuntimeError("Before removing duplicates, a duplictate check has to be done: 'db.duplicate_check()'")

        db_.DupDetect.remove_duplicates(**kwargs)
        header_ = db_.DupDetect.result
        if not isinstance(db_._data, pd.DataFrame):
            raise TypeError("data has unsupported type: {type(db_._data)}.")
        db_._data = db_._data[db_._data.index.isin(header_.index)]
        return self._return_db(db_, inplace)
