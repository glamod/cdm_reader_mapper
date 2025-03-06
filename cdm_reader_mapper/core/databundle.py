"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from .writer import write

from cdm_reader_mapper.cdm_mapper.mapper import map_model
from cdm_reader_mapper.common import (
    count_by_cat,
    get_length,
    replace_columns,
    select_from_index,
    select_from_list,
    select_true,
)
from cdm_reader_mapper.duplicates.duplicates import duplicate_check
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
    mode: str
        Data mode ("data" or "tables")
        Default: "data"

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
    >>> tables = read_tables("path_to_files").data
    >>> db = DataBundle(data=tables, mode="tables")
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
        mode="data",
    ):
        self._data = data
        self._columns = columns
        self._dtypes = dtypes
        self._parse_dates = parse_dates
        self._encoding = encoding
        self._mask = mask
        self._imodel = imodel
        self._mode = mode

    def __len__(self):
        """Length of :py:attr:`data`."""
        return get_length(self.data)

    def __print__(self):
        """Print :py:attr:`data`."""
        print(self.data)

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

    @columns.setter
    def columns(self, value):
        self._columns = value

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
    def mode(self):
        """Data mode."""
        return self._return_property("_mode")

    @mode.setter
    def mode(self, value):
        self._mode = value

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

    def stack_v(self, other, datasets=["data", "mask"], **kwargs):
        """Stack multiple :py:class:`cdm_reader_mapper.DataBundle`'s vertically.

        Parameters
        ----------
        other: str, list
            List of other DataBundles to stack vertically.
        datasets: str, list
            List of datasets to be stacked.
            Default: ['data', 'mask']

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

    def stack_h(self, other, datasets=["data", "mask"], **kwargs):
        """Stack multiple :py:class:`cdm_reader_mapper.DataBundle`'s horizontally.

        Parameters
        ----------
        other: str, list
            List of other :py:class:`cdm_reader_mapper.DataBundle` to stack horizontally.
        datasets: str, list
            List of datasets to be stacked
            Default: ['data', 'mask']

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

    def select_true(self, overwrite=False, **kwargs):
        """Select valid values from :py:attr:`data` via :py:attr:`mask`.

        Parameters
        ----------
        overwrite: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`cdm_reader_mapper.DataBundle`
            else return list containing both DataFrame with true and DataFrame with invalid rows.
            Default: False

        Examples
        --------
        Select valid values only without overwriting the old data.

        >>> true_values, false_values = db.select_true()

        Select valid values only with overwriting the old data.

        >>> db.select_true(overwrite=True)
        >>> true_values = db.data

        See Also
        --------
        DataBundle.select_from_list : Select columns from `data` with specific values.
        DataBundle.select_from_index : Select rows of `data` with specific indexes.

        Note
        ----
        For more information see :py:func:`select_true`
        """
        selected = select_true(self._data, self._mask, **kwargs)
        if overwrite is True:
            self._data = selected[0]
            return self
        return selected

    def select_from_list(self, selection, overwrite=False, **kwargs):
        """Select columns from :py:attr:`data` with specific values.

        Parameters
        ----------
        selection: dict
            Keys: columns to be selected.
            Values: values in keys to be selected
        overwrite: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`cdm_reader_mapper.DataBundle`
            else return list containing both DataFrame with true and DataFrame with invalid entries.
            Default: False

        Examples
        --------
        Select specific columns without overwriting the old data.

        >>> true_values, false_values = db.select_from_list(
        ...     selection={("c1", "B1"): [26, 41]},
        ... )

        Select specific columns with overwriting the old data.

        >>> db.select_from_list(selection={("c1", "B1"): [26, 41]}, overwrite=True)
        >>> true_values = db.selected

        See Also
        --------
        DataBundle.select_from_index : Select rows of `data` with specific indexes.
        DataBundle.select_true : Select valid values from `data` via `mask`.

        Note
        ----
        For more information see :py:func:`select_from_list`
        """
        selected = select_from_list(self._data, selection, **kwargs)
        if overwrite is True:
            self._data = selected[0]
            return self
        return selected

    def select_from_index(self, index, overwrite=False, **kwargs):
        """Select rows of :py:attr:`data` with specific indexes.

        Parameters
        ----------
        index: list
            Indexes to be selected.
        overwrite: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`cdm_reader_mapper.DataBundle`
            else return list containing both DataFrame with true and DataFrame with invalid entries.
            Default: False

        Examples
        --------
        Select specific columns without overwriting the old data.

         >>> true_values, false_values = db.select_from_index([0, 2, 4])

        Select specific columns with overwriting the old data.

        >>> db.select_from_index(index=[0, 2, 4], overwrite=True)
        >>> true_values = db.selected

        See Also
        --------
        DataBundle.select_from_list : Select columns from `data` with specific values.
        DataBundle.select_true : Select valid values from `data` via `mask`.

        Note
        ----
        For more information see :py:func:`select_from_index`
        """
        selected = select_from_index(self._data, index, **kwargs)
        if overwrite is True:
            self._data = selected[0]
            return self
        return selected

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

    def replace_columns(self, df_corr, overwrite=False, **kwargs):
        """Replace columns in :py:attr:`data`.

        Parameters
        ----------
        df_corr: pandas.DataFrame
            Data to be inplaced.
        overwrite: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`cdm_reader_mapper.DataBundle`
            else return pd.DataFrame with replaced columns.
            Default: False

        Examples
        --------
        >>> import pandas as pd
        >>> df_corr = pr.read_csv("corecction_file_on_disk")
        >>> df_repl = db.replace_columns(df_corr)

        Note
        ----
        For more information see :py:func:`replace_columns`
        """
        _data = replace_columns(df_l=self._data, df_r=df_corr, **kwargs)
        if overwrite is True:
            self._data = _data
            self._columns = self._data.columns
            return self
        return _data

    def correct_datetime(self, overwrite=False):
        """Correct datetime information in :py:attr:`data`.

        Parameters
        ----------
        overwrite: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`cdm_reader_mapper.DataBundle`
            else return datetime-corretcted DataFrame.
            Default: False

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
        DataBundle.correct_pt : Correct platform type information in `data`.

        Note
        ----
        For more information see :py:func:`validate_datetime`
        """
        return validate_datetime(self._data, self._imodel)

    def correct_pt(self, overwrite=False):
        """Correct platform type information in :py:attr:`data`.

        Parameters
        ----------
        overwrite: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`cdm_reader_mapper.DataBundle`
            else return platform-corretcted DataFrame.
            Default: False

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
        DataBundle.correct_pt : Correct platform type information in `data`.
        DataBundle.correct_datetime : Correct datetime information in `data`.

        Note
        ----
        For more information see :py:func:`validate_id`
        """
        return validate_id(self._data, self._imodel, **kwargs)

    def map_model(self, overwrite=False, **kwargs):
        """Map :py:attr:`data` to the Common Data Model.
        Write output to :py:attr:`data`.

        Parameters
        ----------
        overwrite: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`cdm_reader_mapper.DataBundle`
            else return CDM tables.
            Default: False

        Examples
        --------
        >>> cdm_tables = db.map_model()

        Note
        ----
        For more information see :py:func:`map_model`
        """
        _tables = map_model(self._data, self._imodel, **kwargs)
        if overwrite is True:
            self._mode = "tables"
            self.columns = _tables.columns
            self._data = _tables
            return self
        return _tables

    def write(self, **kwargs):
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

    def duplicate_check(self, **kwargs):
        """Duplicate check in :py:attr:`data`.

        Note
        ----
        Following columns have to be provided:

          * ``longitude``
          * ``latitude``
          * ``primary_station_id``
          * ``report_timestamp``
          * ``station_course``
          * ``station_speed``

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
        if self._mode == "tables" and "header" in self._data:
            data = self._data["header"]
        else:
            data = self._data
        self.DupDetect = duplicate_check(data, **kwargs)
        return self

    def flag_duplicates(self, overwrite=False, **kwargs):
        """Flag detected duplicates in :py:attr:`data`.

        Parameters
        ----------
        overwrite: bool
            If ``True`` overwrite :py:attr:`data` in DataBundle
            else return DataFrame containing flagged duplicates.
            Default: False

        Note
        ----
        Before flagging duplicates, a duplictate check has to be done, :py:func:`DataBundle.duplicate_check`.

        Examples
        --------
        Flag duplicates without overwriting :py:attr:`data`.

        >>> flagged_tables = db.flag_duplicates()

        Flag duplicates with overwriting :py:attr:`data`.

        >>> db.flag_duplicates(overwrite=True)
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
        self.DupDetect.flag_duplicates(**kwargs)
        df_ = self._data.copy()
        if self._mode == "tables" and "header" in self._data:
            df_["header"] = self.DupDetect.result
        else:
            df_ = self.DupDetect.result
        if overwrite is True:
            self._data = df_
            return self
        return df_

    def get_duplicates(self, **kwargs):
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

    def remove_duplicates(self, overwrite=False, **kwargs):
        """Remove detected duplicates in :py:attr:`data`.

        Parameters
        ----------
        overwrite: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`cdm_reader_mapper.DataBundle`
            else return DataFrame containing non-duplicate rows.
            Default: False

        Note
        ----
        Before removing duplicates, a duplictate check has to be done, :py:func:`DataBundle.duplicate_check`.

        Examples
        --------
        Remove duplicates without overwriting :py:attr:`data`.

        >>> removed_tables = db.remove_duplicates()

        Remove duplicates with overwriting :py:attr:`data`.

        >>> db.remove_duplicates(overwrite=True)
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
        self.DupDetect.remove_duplicates(**kwargs)
        df_ = self._data.copy()
        header_ = self.DupDetect.result
        df_ = df_[df_.index.isin(header_.index)]
        if overwrite is True:
            self._data = df_
            return self
        return df_
