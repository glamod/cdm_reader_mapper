"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations

from io import StringIO as StringIO

import pandas as pd

from ._utilities import _DataBundle, _copy

from .writer import write

from cdm_reader_mapper.cdm_mapper.mapper import map_model
from cdm_reader_mapper.common import (
    count_by_cat,
    replace_columns,
    split_by_boolean_true,
    split_by_index,
    split_by_column_entries,
    split_by_boolean_false,
)
from cdm_reader_mapper.duplicates.duplicates import duplicate_check
from cdm_reader_mapper.metmetpy import (
    correct_datetime,
    correct_pt,
    validate_datetime,
    validate_id,
)


class DataBundle(_DataBundle):
    """Class for manipulating the MDF data and mapping it to the CDM.

    Parameters
    ----------
    data: pandas.DataFrame, optional
        MDF DataFrame.
    columns: list, optional
        Column labels of ``data``
    dtypes: dict, optional
        Data types of ``data``.
    parse_dates: list, optional
        Information how to parse dates on ``data``
    mask: pandas.DataFrame, optional
        MDF validation mask
    imodel: str, optional
        Name of the MFD/CDM data model.
    mode: str
        Data mode ("data" or "tables")
        Default: "data"

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def copy(self) -> DataBundle:
        """Make deep copy of a :py:class:`~DataBundle`.

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

    def add(self, addition, inplace=False) -> DataBundle | None:
        """Adding information to a :py:class:`~DataBundle`.

        Parameters
        ----------
        addition: dict
             Additional elements to add to the :py:class:`~DataBundle`.
        inplace: bool
            If ``True`` add datasets in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with added datasets.
            Default: False

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle with added information or None if ``inplace=True``.

        Examples
        --------
        >>> tables = read_tables("path_to_files")
        >>> db = db.add({"data": tables})
        """
        db_ = self._get_db(inplace)
        for name, data in addition.items():
            setattr(db_, f"_{name}", data)
        return self._return_db(db_, inplace)

    def stack_v(
        self, other, datasets=["data", "mask"], inplace=False, **kwargs
    ) -> DataBundle | None:
        """Stack multiple :py:class:`~DataBundle`'s vertically.

        Parameters
        ----------
        other: str, list
            List of other DataBundles to stack vertically.
        datasets: str, list
            List of datasets to be stacked.
            Default: ['data', 'mask']
        inplace: bool
            If ``True`` overwrite datasets in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with stacked datasets.
            Default: False

        Note
        ----
        The DataFrames in the :py:class:`~DataBundle` have to have the same data columns!

        Returns
        -------
        :py:class:`~DataBundle` or None
            Vertically stacked DataBundle or None if ``inplace=True``.

        Examples
        --------
        >>> db = db1.stack_v(db2, datasets=["data", "mask"])

        See Also
        --------
        DataBundle.stack_h : Stack multiple DataBundle's horizontally.
        """
        return self._stack(other, datasets, inplace, **kwargs)

    def stack_h(
        self, other, datasets=["data", "mask"], inplace=False, **kwargs
    ) -> DataBundle | None:
        """Stack multiple :py:class:`~DataBundle`'s horizontally.

        Parameters
        ----------
        other: str, list
            List of other :py:class:`~DataBundle` to stack horizontally.
        datasets: str, list
            List of datasets to be stacked.
            Default: ['data', 'mask']
        inplace: bool
            If ``True`` overwrite `datasets` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with stacked datasets.
            Default: False

        Note
        ----
        The DataFrames in the :py:class:`~DataBundle` may have different data columns!

        Examples
        --------
        >>> db = db1.stack_h(db2, datasets=["data", "mask"])

        Returns
        -------
        :py:class:`~DataBundle` or None
            Horizontally stacked DataBundle or None if ``inplace=True``.

        See Also
        --------
        DataBundle.stack_v : Stack multiple DataBundle's vertically.
        """
        return self._stack(other, datasets, inplace, axis=1, join="outer", **kwargs)

    def select_where_all_true(
        self, inplace=False, do_mask=True, **kwargs
    ) -> DataBundle | None:
        """Select rows from :py:attr:`data` where all column entries in :py:attr:`mask` are True.

        Parameters
        ----------
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with valid values only in :py:attr:`data`.
            Default: False
        do_mask: bool
            If ``True`` also do selection on :py:attr:`mask`.

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle containing rows where all column entries in :py:attr:`mask` are True or None if ``inplace=True``.

        Examples
        --------
        Select without overwriting the old data.

        >>> db_selected = db.select_where_all_true()

        Select overwriting the old data.

        >>> db.select_where_all_true(inplace=True)
        >>> df_selected = db.data

        See Also
        --------
        DataBundle.select_where_all_false : Select rows from `data` where all entries in `mask` are False.
        DataBundle.select_where_entry_isin : Select rows from `data` where column entries are in a specific value list.
        DataBundle.select_where_index_isin : Select rows from `data` within specific index list.

        Note
        ----
        For more information see :py:func:`split_by_boolean_true`
        """
        db_ = self._get_db(inplace)
        _mask = _copy(db_._mask)
        db_._data = split_by_boolean_true(db_._data, _mask, **kwargs)[0]
        if do_mask is True:
            _prev_index = db_._data.__dict__["_prev_index"]
            db_._mask = split_by_index(db_._mask, _prev_index, **kwargs)[0]
        return self._return_db(db_, inplace)

    def select_where_all_false(
        self, inplace=False, do_mask=True, **kwargs
    ) -> DataBundle | None:
        """Select rows from :py:attr:`data` where all column entries in :py:attr:`mask` are False.

        Parameters
        ----------
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with invalid values only in :py:attr:`data`.
            Default: False
        do_mask: bool
            If ``True`` also do selection on :py:attr:`mask`.

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle containing rows where all column entries in :py:attr:`mask` are False or None if ``inplace=True``.

        Examples
        --------
        Select without overwriting the old data.

        >>> db_selected = db.select_where_all_true()

        Select valid values only with overwriting the old data.

        >>> db.select_where_all_true(inplace=True)
        >>> df_selected = db.data

        See Also
        --------
        DataBundle.select_where_all_true : Select rows from `data` where all entries in `mask` are True.
        DataBundle.select_where_entry_isin : Select rows from `data` where column entries are in a specific value list.
        DataBundle.select_where_index_isin : Select rows from `data` within specific index list.

        Note
        ----
        For more information see :py:func:`split_by_boolean_false`
        """
        db_ = self._get_db(inplace)
        _mask = _copy(db_._mask)
        db_._data = split_by_boolean_false(db_._data, _mask, **kwargs)[0]
        if do_mask is True:
            _prev_index = db_._data.__dict__["_prev_index"]
            db_._mask = split_by_index(db_._mask, _prev_index, **kwargs)[0]
        return self._return_db(db_, inplace)

    def select_where_entry_isin(
        self, selection, inplace=False, do_mask=True, **kwargs
    ) -> DataBundle | None:
        """Select rows from :py:attr:`data` where column entries are in a specific value list.

        Parameters
        ----------
        selection: dict
            Keys: Column names in :py:attr:`data`.
            Values: Specific value list.
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with selected columns only in :py:attr:`data`.
            Default: False
        do_mask: bool
            If ``True`` also do selection on :py:attr:`mask`.

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle containing rows where column entries are in a specific value list or None if ``inplace=True``.

        Examples
        --------
        Select without overwriting the old data.

        >>> db_selected = db.select_from_list(
        ...     selection={("c1", "B1"): [26, 41]},
        ... )

        Select with overwriting the old data.

        >>> db.select_where_entry_isin(selection={("c1", "B1"): [26, 41]}, inplace=True)
        >>> df_selected = db.data

        See Also
        --------
        DataBundle.select_where_index_isin : Select rows from `data` within specific index list.
        DataBundle.select_where_all_true : Select rows from `data` where all entries in `mask` are True.
        DataBundle.select_where_all_false : Select rows from `data` where all entries in `mask` are False.

        Note
        ----
        For more information see :py:func:`split_by_column_entries`
        """
        db_ = self._get_db(inplace)
        db_._data = split_by_column_entries(db_._data, selection, **kwargs)[0]
        if do_mask is True:
            _prev_index = db_._data.__dict__["_prev_index"]
            db_._mask = split_by_index(db_._mask, _prev_index, **kwargs)[0]
        return self._return_db(db_, inplace)

    def select_where_index_isin(
        self, index, inplace=False, do_mask=True, **kwargs
    ) -> DataBundle | None:
        """Select rows from :py:attr:`data` where indexes within a specific index list.

        Parameters
        ----------
        index: list
            Specific index list.
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with selected rows only in :py:attr:`data`.
            Default: False
        do_mask: bool
            If ``True`` also do selection on :py:attr:`mask`.

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle containing rows where indexes are within a specific index list or None if ``inplace=True``.

        Examples
        --------
        Select without overwriting the old data.

         >>> db_selected = db.select_from_index([0, 2, 4])

        Select with overwriting the old data.

        >>> db.select_from_index(index=[0, 2, 4], inplace=True)
        >>> df_selected = db.data

        See Also
        --------
        DataBundle.select_where_entry_isin : Select rows from `data` where column entries are in a specific value list.
        DataBundle.select_where_all_true : Select rows from `data` where all entries in `mask` are True.
        DataBundle.select_where_all_false : Select rows from `data` where all entries in `mask` are False.

        Note
        ----
        For more information see :py:func:`split_by_index`
        """
        db_ = self._get_db(inplace)
        db_._data = split_by_index(db_._data, index, **kwargs)[0]
        if do_mask is True:
            _prev_index = db_._data.__dict__["_prev_index"]
            db_._mask = split_by_index(db_._mask, _prev_index, **kwargs)[0]
        return self._return_db(db_, inplace)

    def split_by_boolean_true(
        self, do_mask=True, **kwargs
    ) -> tuple[DataBundle, DataBundle]:
        """Split :py:attr:`data` by rows where all column entries in :py:attr:`mask` are True.

        Parameteers
        -----------
        do_mask: bool
            If ``True`` also do selection on :py:attr:`mask`.

        Returns
        -------
        tuple
            First :py:class:`~DataBundle` including rows where all column entries in :py:attr:`mask` are True.
            Second :py:class:`~DataBundle` including rows where all column entries in :py:attr:`mask` are False.

        Examples
        --------
        Split DataBundle.

        >>> db_true, db_false = db.split_where_all_true()

        See Also
        --------
        DataBundle.split_by_boolean_false : Split `data` by rows where all entries in `mask` are False.
        DataBundle.split_by_column_entries : Split `data` by rows where column entries are in a specific value list.
        DataBundle.split_by_index : Split `data` by rows within specific index list.

        Note
        ----
        For more information see :py:func:`split_by_boolean_true`
        """
        db1_ = self.copy()
        db2_ = self.copy()
        _mask = _copy(db1_._mask)
        db1_._data, db2_._data = split_by_boolean_true(
            db1_._data, _mask, return_rejected=True, **kwargs
        )
        if do_mask is True:
            _prev_index = db1_._data.__dict__["_prev_index"]
            db1_._mask, db2_._mask = split_by_index(
                db1_._mask, _prev_index, return_rejected=True, **kwargs
            )
        return db1_, db2_

    def split_by_boolean_false(
        self, do_mask=True, **kwargs
    ) -> tuple[DataBundle, DataBundle]:
        """Split :py:attr:`data` by rows where all column entries in :py:attr:`mask` are False.

        Parameteers
        -----------
        do_mask: bool
            If ``True`` also do selection on :py:attr:`mask`.

        Returns
        -------
        tuple
            First :py:class:`~DataBundle` including rows where all column entries in :py:attr:`mask` are False.
            Second :py:class:`~DataBundle` including rows where all column entries in :py:attr:`mask` are True.

        Examples
        --------
        Split DataBundle.

        >>> db_false, db_true = db.split_where_all_false()

        See Also
        --------
        DataBundle.split_by_boolean_false : Split `data` by rows where all entries in `mask` are True.
        DataBundle.split_by_column_entries : Split `data` by rows where column entries are in a specific value list.
        DataBundle.split_by_index : Split `data` by rows within specific index list.

        Note
        ----
        For more information see :py:func:`split_by_boolean_false`
        """
        db1_ = self.copy()
        db2_ = self.copy()
        _mask = _copy(db1_._mask)
        db1_._data, db2_._data = split_by_boolean_false(
            db1_._data, _mask, return_rejected=True, **kwargs
        )
        if do_mask is True:
            _prev_index = db1_._data.__dict__["_prev_index"]
            db1_._mask, db2_._mask = split_by_index(
                db1_._mask, _prev_index, return_rejected=True, **kwargs
            )
        return db1_, db2_

    def split_by_column_entries(
        self, selection, do_mask=True, **kwargs
    ) -> tuple[DataBundle, DataBundle]:
        """Split :py:attr:`data` by rows where column entries are in a specific value list.

        Parameters
        ----------
        selection: dict
            Keys: Column names in :py:attr:`data`.
            Values: Specific value list.
        do_mask: bool
            If ``True`` also do selection on :py:attr:`mask`.

        Returns
        -------
        tuple
            First :py:class:`~DataBundle` including rows where column entries are in a specific value list.
            Second :py:class:`~DataBundle` including rows where column entries are not in a specific value list.

        Examples
        --------
        Split DataBundle.

        >>> db_isin, db_isnotin = db.split_where_entry_isin(
        ...     selection={("c1", "B1"): [26, 41]},
        ... )

        See Also
        --------
        DataBundle.split_by_index : Split `data` by rows within specific index list.
        DataBundle.split_by_boolean_true : Split `data` by rows where all entries in `mask` are True.
        DataBundle.split_by_boolean_false : Split `data` by rows where all entries in `mask` are False.

        Note
        ----
        For more information see :py:func:`split_by_column_entries`
        """
        db1_ = self.copy()
        db2_ = self.copy()
        db1_._data, db2_._data = split_by_column_entries(
            db1_._data, selection, return_rejected=True, **kwargs
        )
        if do_mask is True:
            _prev_index = db1_._data.__dict__["_prev_index"]
            db1_._mask, db2_._mask = split_by_index(
                db1_._mask, _prev_index, return_rejected=True, **kwargs
            )
        return db1_, db2_

    def split_by_index(
        self, index, do_mask=True, **kwargs
    ) -> tuple[DataBundle, DataBundle]:
        """Split :py:attr:`data` by rows within specific index list.

        Parameters
        ----------
        index: list
            Specific index list.
        do_mask: bool
            If ``True`` also do selection on :py:attr:`mask`.


        Returns
        -------
        tuple
            First :py:class:`~DataBundle` including rows within specific index list.
            Second :py:class:`~DataBundle` including rows outside specific index list.


        Examples
        --------
        Split DataBundle.

         >>> db_isin, db_isnotin = db.select_from_index([0, 2, 4])

        See Also
        --------
        DataBundle.split_by_column_entries : Select columns from `data` with specific values.
        DataBundle.split_by_boolean_true : Split `data` by rows where all entries in `mask` are True.
        DataBundle.split_by_boolean_false : Split `data` by rows where all entries in `mask` are False.

        Note
        ----
        For more information see :py:func:`split_by_index`
        """
        db1_ = self.copy()
        db2_ = self.copy()
        db1_._data, db2_._data = split_by_index(
            db1_._data, index, return_rejected=True, **kwargs
        )
        if do_mask is True:
            db1_._mask, db2_._mask = split_by_index(
                db1_._mask, index, return_rejected=True, **kwargs
            )
        return db1_, db2_

    def unique(self, **kwargs) -> dict:
        """Get unique values of :py:attr:`data`.

        Returns
        -------
        dict
            Dictionary with unique values.

        Examples
        --------
        >>> db.unique(columns=("c1", "B1"))

        Note
        ----
        For more information see :py:func:`unique`
        """
        return count_by_cat(self._data, **kwargs)

    def replace_columns(
        self, df_corr, subset=None, inplace=False, **kwargs
    ) -> DataBundle | None:
        """Replace columns in :py:attr:`data`.

        Parameters
        ----------
        df_corr: pandas.DataFrame
            Data to be inplaced.
        subset: str, list, optional
            Select subset by columns. This option is useful for multi-indexed :py:attr:`data`.
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with replaced column names in :py:attr:`data`.
            Default: False

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle with replaced column names or None if ``inplace=True``.

        Examples
        --------
        >>> import pandas as pd
        >>> df_corr = pr.read_csv("correction_file_on_disk")
        >>> df_repl = db.replace_columns(df_corr)

        Note
        ----
        For more information see :py:func:`replace_columns`
        """
        db_ = self._get_db(inplace)
        if subset is None:
            db_._data = replace_columns(df_l=db_._data, df_r=df_corr, **kwargs)
        else:
            db_._data[subset] = replace_columns(
                df_l=db_._data[subset], df_r=df_corr, **kwargs
            )
        db_._columns = db_._data.columns
        return self._return_db(db_, inplace)

    def correct_datetime(self, inplace=False) -> DataBundle | None:
        """Correct datetime information in :py:attr:`data`.

        Parameters
        ----------
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with datetime-corrected values in :py:attr:`data`.
            Default: False

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle with corrected datetime information or None if ``inplace=True``.

        Examples
        --------
        >>> df_dt = db.correct_datetime()

        See Also
        --------
        DataBundle.correct_pt : Correct platform type information in `data`.
        DataBundle.validate_datetime: Validate datetime information in `data`.
        DataBundle.validate_id : Validate station id information in `data`.

        Note
        ----
        For more information see :py:func:`correct_datetime`
        """
        db_ = self._get_db(inplace)
        db_._data = correct_datetime(db_._data, db_._imodel)
        return self._return_db(db_, inplace)

    def validate_datetime(self) -> pd.DataFrame:
        """Validate datetime information in :py:attr:`data`.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing True and False values for each index in :py:attr:`data`.
            True: All datetime information in :py:attr:`data` row are valid.
            False: At least one datetime information in :py:attr:`data` row is invalid.

        Examples
        --------
        >>> val_dt = db.validate_datetime()

        See Also
        --------
        DataBundle.validate_id : Validate station id information in `data`.
        DataBundle.correct_datetime : Correct datetime information in `data`.
        DataBundle.correct_pt : Correct platform type information in `data`.

        Note
        ----
        For more information see :py:func:`validate_datetime`
        """
        return validate_datetime(self._data, self._imodel)

    def correct_pt(self, inplace=False) -> DataBundle | None:
        """Correct platform type information in :py:attr:`data`.

        Parameters
        ----------
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with platform-corrected values in :py:attr:`data`.
            Default: False

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle with corrected platform type information or None if ``inplace=True``.

        Examples
        --------
        >>> df_pt = db.correct_pt()

        See Also
        --------
        DataBundle.correct_datetime : Correct datetime information in `data`.
        DataBundle.validate_id : Validate station id information in `data`.
        DataBundle.validate_datetime : Validate datetime information in `data`.

        Note
        ----
        For more information see :py:func:`correct_pt`
        """
        db_ = self._get_db(inplace)
        db_._data = correct_pt(db_._data, db_._imodel)
        return self._return_db(db_, inplace)

    def validate_id(self, **kwargs) -> pd.DataFrame:
        """Validate station id information in :py:attr:`data`.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing True and False values for each index in :py:attr:`data`.
            True: All station ID information in :py:attr:`data` row are valid.
            False: At least one station ID information in :py:attr:`data` row is invalid.

        Examples
        --------
        >>> val_dt = db.validate_id()

        See Also
        --------
        DataBundle.validate_datetime : Validate datetime information in `data`.
        DataBundle.correct_pt : Correct platform type information in `data`.
        DataBundle.correct_datetime : Correct datetime information in `data`.

        Note
        ----
        For more information see :py:func:`validate_id`
        """
        return validate_id(self._data, self._imodel, **kwargs)

    def map_model(self, inplace=False, **kwargs) -> DataBundle | None:
        """Map :py:attr:`data` to the Common Data Model.

        Parameters
        ----------
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with :py:attr:`data` as CDM tables.
            Default: False

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle containing :py:attr:`data` mapped to the CDM or None if ``inplace=True``.

        Examples
        --------
        >>> cdm_tables = db.map_model()

        Note
        ----
        For more information see :py:func:`map_model`
        """
        db_ = self._get_db(inplace)
        _tables = map_model(db_._data, db_._imodel, **kwargs)
        db_._mode = "tables"
        db_._columns = _tables.columns
        db_._data = _tables
        return self._return_db(db_, inplace)

    def write(self, **kwargs) -> None:
        """Write :py:attr:`data` on disk.

        Examples
        --------
        >>> db.write()

        See Also
        --------
        write_data : Write MDF data and validation mask to disk.
        write_tables: Write CDM tables to disk.
        read: Read original marine-meteorological data as well as MDF data or CDM tables from disk.
        read_data: Read MDF data and validation mask from disk.
        read_mdf : Read original marine-meteorological data from disk.
        read_tables : Read CDM tables from disk.

        Note
        ----
        If :py:attr:`mode` is "data" write data using :py:func:`write_data`.
        If :py:attr:`mode` is "tables" write data using :py:func:`write_tables`.
        """
        write(
            data=self._data,
            mask=self._mask,
            dtypes=self._dtypes,
            parse_dates=self._parse_dates,
            encoding=self._encoding,
            mode=self._mode,
            **kwargs,
        )

    def duplicate_check(self, inplace=False, **kwargs) -> DataBundle | None:
        """Duplicate check in :py:attr:`data`.

        Parameters
        ----------
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with :py:attr:`data` as CDM tables.
            Default: False

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle containing new :py:class:`~DupDetect` class for further duplicate check methods or None if ``inplace=True``.

        Note
        ----
        Following columns have to be provided:

          * ``longitude``
          * ``latitude``
          * ``primary_station_id``
          * ``report_timestamp``
          * ``station_course``
          * ``station_speed``

        Note
        ----
        This adds a new class :py:class:`~DupDetect` to :py:class:`~DataBundle`.
        This class is necessary for further duplicate check methods.

        Examples
        --------
        >>> db.duplicate_check()

        See Also
        --------
        DataBundle.get_duplicates : Get duplicate matches in `data`.
        DataBundle.flag_duplicates : Flag detected duplicates in `data`.
        DataBundle.remove_duplicates : Remove detected duplicates in `data`.

        Note
        ----
        For more information see :py:func:`duplicate_check`
        """
        db_ = self._get_db(inplace)
        if db_._mode == "tables" and "header" in db_._data:
            data = db_._data["header"]
        else:
            data = db_._data
        db_.DupDetect = duplicate_check(data, **kwargs)
        return self._return_db(db_, inplace)

    def flag_duplicates(self, inplace=False, **kwargs) -> DataBundle | None:
        """Flag detected duplicates in :py:attr:`data`.

        Parameters
        ----------
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with :py:attr:`data` containing flagged duplicates.
            Default: False

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle containing duplicate flags in :py:attr:`data` or None if ``inplace=True``.

        Note
        ----
        Before flagging duplicates, a duplictate check has to be done, :py:func:`DataBundle.duplicate_check`.

        Examples
        --------
        Flag duplicates without overwriting :py:attr:`data`.

        >>> flagged_tables = db.flag_duplicates()

        Flag duplicates with overwriting :py:attr:`data`.

        >>> db.flag_duplicates(inplace=True)
        >>> flagged_tables = db.data

        See Also
        --------
        DataBundle.remove_duplicates : Remove detected duplicates in `data`.
        DataBundle.get_duplicates : Get duplicate matches in `data`.
        DataBundle.duplicate_check : Duplicate check in `data`.

        Note
        ----
        For more information see :py:func:`DupDetect.flag_duplicates`
        """
        db_ = self._get_db(inplace)
        db_.DupDetect.flag_duplicates(**kwargs)
        if db_._mode == "tables" and "header" in db_._data:
            db_._data["header"] = db_.DupDetect.result
        else:
            db_._data = db_.DupDetect.result
        return self._return_db(db_, inplace)

    def get_duplicates(self, **kwargs) -> list:
        """Get duplicate matches in :py:attr:`data`.

        Returns
        -------
        list
            List of tuples containing duplicate matches.

        Note
        ----
        Before getting duplicates, a duplictate check has to be done, :py:func:`DataBundle.duplicate_check`.

        Examples
        --------
        >>> matches = db.get_duplicates()

        See Also
        --------
        DataBundle.remove_duplicates : Remove detected duplicates in `data`.
        DataBundle.flag_duplicates : Flag detected duplicates in `data`.
        DataBundle.duplicate_check : Duplicate check in `data`.

        Note
        ----
        For more information see :py:func:`DupDetect.get_duplicates`
        """
        return self.DupDetect.get_duplicates(**kwargs)

    def remove_duplicates(self, inplace=False, **kwargs) -> DataBundle | None:
        """Remove detected duplicates in :py:attr:`data`.

        Parameters
        ----------
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with :py:attr:`data` containing no duplicates.
            Default: False

        Returns
        -------
        :py:class:`~DataBundle` or None
            DataBundle without duplictaed rows or None if ``inplace=True``.

        Note
        ----
        Before removing duplicates, a duplictate check has to be done, :py:func:`DataBundle.duplicate_check`.

        Examples
        --------
        Remove duplicates without overwriting :py:attr:`data`.

        >>> removed_tables = db.remove_duplicates()

        Remove duplicates with overwriting :py:attr:`data`.

        >>> db.remove_duplicates(inplace=True)
        >>> removed_tables = db.data

        See Also
        --------
        DataBundle.flag_duplicates : Flag detected duplicates in `data`.
        DataBundle.get_duplicates : Get duplicate matches in `data`.
        DataBundle.duplicate_check : Duplicate check in `data`.

        Note
        ----
        For more information see :py:func:`DupDetect.remove_duplicates`
        """
        db_ = self._get_db(inplace)
        db_.DupDetect.remove_duplicates(**kwargs)
        header_ = db_.DupDetect.result
        db_._data = db_._data[db_._data.index.isin(header_.index)]
        return self._return_db(db_, inplace)
