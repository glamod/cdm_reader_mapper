"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from cdm_reader_mapper.cdm_mapper.mapper import map_model
from cdm_reader_mapper.cdm_mapper.writer import write_tables
from cdm_reader_mapper.common import (
    count_by_cat,
    get_length,
    replace_columns,
    select_from_index,
    select_from_list,
    select_true,
)
from cdm_reader_mapper.duplicates.duplicates import duplicate_check
from cdm_reader_mapper.mdf_reader.writer import write_data
from cdm_reader_mapper.metmetpy import (
    correct_datetime,
    correct_pt,
    validate_datetime,
    validate_id,
)


class DataBundle:
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
    tables: pandas.DataFrame, optional
        CDM tables.

    Examples
    --------
    Getting a :py:class:`cdm_reader_mapper.DataBundle` while reading data from disk.

    >>> from cdm_reader_mapper import read_mdf
    >>> db = read_mdf(source="file_on_disk", imodel="custom_model_name")

    Constructing a :py:class:`cdm_reader_mapper.DataBundle` from already read MDf data.

    >>> from cdm_reader_mapper import DataBundle
    >>> read = read_mdf(source="file_on_disk", imodel="custom_model_name")
    >>> data_ = read.data
    >>> mask_ = read.mask
    >>> db = DataBundle(data=data_, mask=mask_)

    Constructing a :py:class:`cdm_reader_mapper.DataBundle` from already read CDM data.

    >>> from cdm_reader_mapper import read_tables
    >>> tables = read_tables("path_to_files")
    >>> db = DataBundle(tables=tables)
    """

    def __init__(
        self,
        data=None,
        columns=None,
        dtypes=None,
        parse_dates=None,
        encoding=None,
        mask=None,
        imodel=None,
        tables=None,
    ):
        self._data = data
        self._columns = columns
        self._dtypes = dtypes
        self._parse_dates = parse_dates
        self._encoding = encoding
        self._mask = mask
        self._imodel = imodel
        self._tables = tables

    def __len__(self):
        """Length of :py:attr:`data`."""
        return get_length(self.data)

    def __getitem__(self, item):
        """Make class subscriptable."""
        return getattr(self, item)

    def _return_property(self, property):
        if hasattr(self, property):
            return getattr(self, property)

    @property
    def data(self):
        """MDF pandas.DataFrame data."""
        return self._return_property("_data")

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def columns(self):
        """Column labels of :py:attr:`data`."""
        return self._return_property("_columns")

    @property
    def index(self):
        """Indexes of :py:attr:`data`."""
        return self._return_property("_index")

    @property
    def dtypes(self):
        """Dictionary of data types on :py:attr:`data`."""
        return self._return_property("_dtypes")

    @property
    def parse_dates(self):
        """Information of how to parse dates in :py:attr:`data`.

        See Also
        --------
        :py:func:pandas.`read_csv`
        """
        return self._return_property("_parse_dates")

    @property
    def encoding(self):
        """A string representing the encoding to use in the :py:attr:`data`.

        See Also
        --------
        :py:func:pandas.`to_csv`
        """
        return self._return_property("_encoding")

    @property
    def mask(self):
        """MDF pandas.DataFrame validation mask."""
        return self._return_property("_mask")

    @mask.setter
    def mask(self, value):
        self._mask = value

    @property
    def imodel(self):
        """Name of the MDF/CDM input model."""
        return self._return_property("_imodel")

    @imodel.setter
    def imodel(self, value):
        self._imodel = value

    @property
    def tables(self):
        """CDM tables."""
        return self._return_property("_tables")

    @tables.setter
    def tables(self, value):
        self._tables = value

    def add(self, addition):
        """Adding information to a :py:class:`~DataBundle`.

        Parameters
        ----------
        addition: dict
             Additional elements to add to the :py:class:`cdm_reader_mapper.DataBundle`.

        Examples
        --------
        >>> tables = read_tables("path_to_files")
        >>> db = db.add({"tables": tables})
        """
        for name, data in addition.items():
            setattr(self, f"_{name}", data)
        return self

    def stack_v(self, other, datasets=["data", "mask", "tables"], **kwargs):
        """Stack multiple :py:class:`cdm_reader_mapper.DataBundle`'s vertically.

        Parameters
        ----------
        other: str, list
            List of other DataBundles to stack vertically.
        datasets: str, list
            List of datasets to be stacked.
            Default: ['data', 'mask', 'tables']

        Note
        ----
        The DataFrames in the :py:class:`cdm_reader_mapper.DataBundle` have to have the same data columns!

        Examples
        --------
        >>> db = db1.stack_v(db2, datasets=["data", "mask"])

        See Also
        --------
        DataBundle.stack_h : Stack multiple DataBundle's horizontally.
        """
        if not isinstance(other, list):
            other = [other]
        if not isinstance(datasets, list):
            datasets = [datasets]
        for data in datasets:
            data = f"_{data}"
            self_data = getattr(self, data) if hasattr(self, data) else pd.DataFrame
            to_concat = [
                getattr(concat, data) for concat in other if hasattr(concat, data)
            ]
            if not to_concat:
                continue
            if not self_data.empty:
                to_concat = [self_data] + to_concat
            self_data = pd.concat(to_concat, **kwargs)
            setattr(self, data, self_data.reset_index(drop=True))
        return self

    def stack_h(self, other, datasets=["data", "mask", "tables"], **kwargs):
        """Stack multiple :py:class:`cdm_reader_mapper.DataBundle`'s horizontally.

        Parameters
        ----------
        other: str, list
            List of other :py:class:`cdm_reader_mapper.DataBundle` to stack horizontally.
        datasets: str, list
            List of datasets to be stacked
            Default: ['data', 'mask', 'tables']

        Note
        ----
        The DataFrames in the :py:class:`cdm_reader_mapper.DataBundle` may have different data columns!

        Examples
        --------
        >>> db = db1.stack_h(db2, datasets=["data", "mask"])

        See Also
        --------
        DataBundle.stack_v : Stack multiple DataBundle's vertically.
        """
        if not isinstance(other, list):
            other = [other]
        for data in datasets:
            data = f"_{data}"
            self_data = getattr(self, data) if hasattr(self, data) else pd.DataFrame()
            to_concat = [
                getattr(concat, data) for concat in other if hasattr(concat, data)
            ]
            if not to_concat:
                continue
            if not self_data.empty:
                to_concat = [self_data] + to_concat
            self_data = pd.concat(to_concat, axis=1, join="outer")
            setattr(self, data, self_data.reset_index(drop=True))
        return self

    def copy(self):
        """Make deep copy of a :py:class:`cdm_reader_mapper.DataBundle`.

        Examples
        --------
        >>> db2 = db.copy()
        """
        return deepcopy(self)

    def select_true(self, data="data", return_invalid=False, overwrite=True, **kwargs):
        """Select valid values from :py:attr:`data` via :py:attr:`mask`.

        Parameters
        ----------
        data: str, {'data', 'mask'}
            Name of the data to be selected.
            Default: data
        return_invalid: bool
            If True return invalid data additionally.
            Default: False
        overwrite: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`cdm_reader_mapper.DataBundle`
            else return list containing both DataFrame with true and DataFrame with invalid rows.
            Default: True

        Note
        ----
        If `return_invalid` is ``True`` this function returns two values.

        Note
        ----
        Use this for :py:attr:`data` only. It does not work for :py:attr:`tables`.

        Examples
        --------
        Select valid values only with overwriting the old data.

        >>> db.select_true()
        >>> true_values = db.data

        Select valid values only without overwriting the old data.

        >>> true_values, false_values = db.select_true(overwrite=False)

        See Also
        --------
        DataBundle.select_from_list : Select columns from `data` with specific values.
        DataBundle.select_from_index : Select rows of `data` with specific indexes.

        Note
        ----
        For more information see :py:func:`select_true`
        """
        _data = f"_{data}"
        _data, _invalid, _index = select_true(
            getattr(self, _data), self._mask, in_index=True, **kwargs
        )
        if overwrite is True:
            setattr(self, data, _data)
            setattr(self, "_index", _index)
            _return = self
        else:
            _return = _data
        if return_invalid is True:
            return _return, _invalid
        return _return

    def select_from_list(
        self, selection, data="data", return_invalid=False, overwrite=True, **kwargs
    ):
        """Select columns from :py:attr:`data` with specific values.

        Parameters
        ----------
        selection: dict
            Keys: columns to be selected.
            Values: values in keys to be selected
        data: str, {'data', 'mask'}
            Name of the data to be selected.
            Default: data
        return_invalid: bool
            If True return invalid data additionally.
            Default: False
        overwrite: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`cdm_reader_mapper.DataBundle`
            else return list containing both DataFrame with true and DataFrame with invalid entries.
            Default: True

        Note
        ----
        If `return_invalid` is ``True`` this function returns two values.

        Note
        ----
        Use this for :py:attr:`data` only. It does not work for :py:attr:`tables`.

        Examples
        --------
        Select specific columns with overwriting the old data.

        >>> db.select_from_list(selection={("c1", "B1"): [26, 41]})
        >>> true_values = db.selected

        Select specific columns without overwriting the old data.

        >>> true_values, false_values = db.select_from_list(
        ...     selection={("c1", "B1"): [26, 41]}, overwrite=False
        ... )

        See Also
        --------
        DataBundle.select_from_index : Select rows of `data` with specific indexes.
        DataBundle.select_true : Select valid values from `data` via `mask`.

        Note
        ----
        For more information see :py:func:`select_from_list`
        """
        _data = f"_{data}"
        _data, _invalid, _index = select_from_list(
            getattr(self, _data), selection, in_index=True, **kwargs
        )
        if overwrite is True:
            setattr(self, data, _data)
            setattr(self, "_index", _index)
            _return = self
        else:
            _return = _data
        if return_invalid is True:
            return _return, _invalid
        return _return

    def select_from_index(
        self, index, data="data", return_invalid=False, overwrite=True, **kwargs
    ):
        """Select rows of :py:attr:`data` with specific indexes.

        Parameters
        ----------
        index: list
            Indexes to be selected.
        data: str, {'data', 'mask'}
            Name of the data to be selected.
            Default: data
        return_invalid: bool
            If True return invalid data additionally.
            Default: False
        overwrite: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`cdm_reader_mapper.DataBundle`
            else return list containing both DataFrame with true and DataFrame with invalid entries.
            Default: True

        Note
        ----
        If `return_invalid` is ``True`` this function returns two values.

        Note
        ----
        Use this for :py:attr:`data` only. It does not work for :py:attr:`tables`.

        Examples
        --------
        Select specific columns with overwriting the old data.

        >>> db.select_from_index(index=[0, 2, 4])
        >>> true_values = db.selected

        Select specific columns without overwriting the old data.

        >>> true_values, false_values = db.select_from_index([0, 2, 4], overwrite=False)

        See Also
        --------
        DataBundle.select_from_list : Select columns from `data` with specific values.
        DataBundle.select_true : Select valid values from `data` via `mask`.

        Note
        ----
        For more information see :py:func:`select_from_index`
        """
        _data = f"_{data}"
        _data, _invalid = select_from_index(getattr(self, _data), index, **kwargs)
        if overwrite is True:
            setattr(self, data, _data)
            setattr(self, "_index", index)
            _return = self
        else:
            _return = _data
        if return_invalid is True:
            return _return, _invalid
        return _return

    def unique(self, **kwargs):
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

    def replace_columns(self, df_corr, **kwargs):
        """Replace columns in :py:attr:`data`.

        Parameters
        ----------
        df_corr: pandas.DataFrame
            Data to be inplaced.

        Examples
        --------
        >>> import pandas as pd
        >>> df_corr = pr.read_csv("corecction_file_on_disk")
        >>> db.replace_columns(df_corr)

        Note
        ----
        For more information see :py:func:`replace_columns`
        """
        self._data = replace_columns(df_l=self._data, df_r=df_corr, **kwargs)
        self._columns = self._data.columns
        return self

    def correct_datetime(self, overwrite=True):
        """Correct datetime information in :py:attr:`data`.

        Parameters
        ----------
        overwrite: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`cdm_reader_mapper.DataBundle`
            else return datetime-corretcted DataFrame.
            Default: True

        Examples
        --------
        >>> db.correct_datetime()

        See Also
        --------
        DataBundle.correct_pt : Correct platform type information in `data`.
        DataBundle.validate_datetime: Validate datetime information in `data`.
        DataBundle.validate_id : Validate station id information in `data`.

        Note
        ----
        For more information see :py:func:`correct_datetime`
        """
        _data = correct_datetime(self._data, self._imodel)
        if overwrite is True:
            self._data = _data
            return self
        return _data

    def validate_datetime(self):
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
        DataBundle.correct_pt : Correct platform type information in `tables`.

        Note
        ----
        For more information see :py:func:`validate_datetime`
        """
        return validate_datetime(self._data, self._imodel)

    def correct_pt(self, overwrite=True):
        """Correct platform type information in :py:attr:`data`.

        Parameters
        ----------
        overwrite: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`cdm_reader_mapper.DataBundle`
            else return platform-corretcted DataFrame.
            Default: True

        Examples
        --------
        >>> db.correct_pt()

        See Also
        --------
        DataBundle.correct_datetime : Correct datetime information in `data`.
        DataBundle.validate_id : Validate station id information in `data`.
        DataBundle.validate_datetime : Validate datetime information in `data`.

        Note
        ----
        For more information see :py:func:`correct_pt`
        """
        _data = correct_pt(self._data, self._imodel)
        if overwrite is True:
            self._data = _data
            return self
        return _data

    def validate_id(self, **kwargs):
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
        DataBundle.correct_pt : Correct platform type information in `tables`.
        DataBundle.correct_datetime : Correct datetime information in `tables`.

        Note
        ----
        For more information see :py:func:`validate_id`
        """
        return validate_id(self._data, self._imodel, **kwargs)

    def write_data(self, **kwargs):
        """Write MDF data on disk.

        Examples
        --------
        >>> db.write_data()

        See Also
        --------
        DataBundle.write_tables : Write MDF data on disk.
        read_mdf : Read original marine-meteorological data from disk.
        read_tables : Read CDM tables from disk.

        Note
        ----
        For more information see :py:func:`write_data`
        """
        write_data(
            self._data,
            mask=self._mask,
            dtypes=self._dtypes,
            parse_dates=self._parse_dates,
            encoding=self._encoding,
            **kwargs,
        )

    def map_model(self, **kwargs):
        """Map :py:attr:`data` to the Common Data Model.
        Write output to :py:attr:`tables`.

        Examples
        --------
        >>> db.map_model()

        Note
        ----
        For more information see :py:func:`map_model`
        """
        self._tables = map_model(self._data, self._imodel, **kwargs)
        return self

    def write_tables(self, **kwargs):
        """Write CDM tables on disk.

        Note
        ----
        Before writing CDM tables on disk, they have to be provided in :py:class:`cdm_reader_mapper.DataBundle`,
        e.g. with :py:func:`DataBundle.map_model`.

        Examples
        --------
        >>> db.write_tables()

        See Also
        --------
        DataBundle.write_mdf : Write MDF data on disk.
        read_tables : Read CDM tables from disk.
        read_mdf : Read original marine-meteorological data from disk.

        Note
        ----
        For more information see :py:func:`write_tables`
        """
        write_tables(self._tables, encoding=self._encoding, **kwargs)

    def duplicate_check(self, **kwargs):
        """Duplicate check in :py:attr:`tables`.

        Note
        ----
        Before processing the duplicate check, CDM tables have to be provided in :py:class:`cdm_reader_mapper.DataBundle`,
        e.g. with :py:func:`DataBundle.map_model`.

        Examples
        --------
        >>> db.duplicate_check()

        See Also
        --------
        DataBundle.get_duplicates : Get duplicate matches in `tables`.
        DataBundle.flag_duplicates : Flag detected duplicates in `tables`.
        DataBundle.remove_duplicates : Remove detected duplicates in `tables`.

        Note
        ----
        For more information see :py:func:`duplicate_check`
        """
        self.DupDetect = duplicate_check(self._tables["header"], **kwargs)
        return self

    def flag_duplicates(self, overwrite=True, **kwargs):
        """Flag detected duplicates in :py:attr:`tables`.

        Parameters
        ----------
        overwrite: bool
            If ``True`` overwrite :py:attr:`tables` in DataBundle
            else return DataFrame containing flagged duplicates.
            Default: True

        Note
        ----
        Before flagging duplicates, a duplictate check has to be done, :py:func:`DataBundle.duplicate_check`.

        Examples
        --------
        Flag duplicates with overwriting :py:attr:`tables`.

        >>> db.flag_duplicates()
        >>> flagged_tables = db.tables

        Flag duplicates without overwriting :py:attr:`tables`.

        >>> flagged_tables = db.flag_duplicates(overwrite=False)

        See Also
        --------
        DataBundle.remove_duplicates : Remove detected duplicates in `tables`.
        DataBundle.get_duplicates : Get duplicate matches in `tables`.
        DataBundle.duplicate_check : Duplicate check in `tables`.

        Note
        ----
        For more information see :py:func:`DupDetect.flag_duplicates`
        """
        self.DupDetect.flag_duplicates(**kwargs)
        df_ = self._tables.copy()
        df_["header"] = self.DupDetect.result
        if overwrite is True:
            self._tables = df_
            return self
        return df_

    def get_duplicates(self, **kwargs):
        """Get duplicate matches in :py:attr:`tables`.

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
        DataBundle.remove_duplicates : Remove detected duplicates in `tables`.
        DataBundle.flag_duplicates : Flag detected duplicates in `tables`.
        DataBundle.duplicate_check : Duplicate check in `tables`.

        Note
        ----
        For more information see :py:func:`DupDetect.get_duplicates`
        """
        return self.DupDetect.get_duplicates(**kwargs)

    def remove_duplicates(self, overwrite=True, **kwargs):
        """Remove detected duplicates in :py:attr:`tables`.

        Parameters
        ----------
        overwrite: bool
            If ``True`` overwrite :py:attr:`tables` in :py:class:`cdm_reader_mapper.DataBundle`
            else return DataFrame containing non-duplicate rows.
            Default: True

        Note
        ----
        Before removing duplicates, a duplictate check has to be done, :py:func:`DataBundle.duplicate_check`.

        Examples
        --------
        Remove duplicates with overwriting :py:attr:`tables`.

        >>> db.remove_duplicates()
        >>> removed_tables = db.tables

        Remove duplicates without overwriting :py:attr:`tables`.

        >>> removed_tables = db.remove_duplicates(overwrite=False)

        See Also
        --------
        DataBundle.flag_duplicates : Flag detected duplicates in `tables`.
        DataBundle.get_duplicates : Get duplicate matches in `tables`.
        DataBundle.duplicate_check : Duplicate check in `tables`.

        Note
        ----
        For more information see :py:func:`DupDetect.remove_duplicates`
        """
        self.DupDetect.remove_duplicates(**kwargs)
        df_ = self._tables.copy()
        header_ = self.DupDetect.result
        df_ = df_[df_.index.isin(header_.index)]
        if overwrite is True:
            self._tables = df_
            return self
        return df_
