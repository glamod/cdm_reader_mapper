"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from cdm_reader_mapper.cdm_mapper.mapper import map_model
from cdm_reader_mapper.cdm_mapper.table_writer import write_tables
from cdm_reader_mapper.duplicates.duplicates import duplicate_check
from cdm_reader_mapper.mdf_reader.write import write as write_mdf
from cdm_reader_mapper.metmetpy.datetime.correct import correct as correct_datetime
from cdm_reader_mapper.metmetpy.datetime.validate import validate as validate_datetime
from cdm_reader_mapper.metmetpy.platform_type.correct import correct as correct_pt
from cdm_reader_mapper.metmetpy.station_id.validate import validate as validate_id
from cdm_reader_mapper.operations import inspect, replace, select


class DataBundle:
    """Class for manipulating the MDF data and mapping it to the CDM.

    Parameters
    ----------
    data: pandas.DataFrame, optional
        MDF DataFrame.
    columns: list, optional
        Column labels of ``data``
    dtypes: list, optional
        Data types of ``data``.
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
        mask=None,
        imodel=None,
        tables=None,
    ):
        self._data = data
        self._columns = columns
        self._dtypes = dtypes
        self._parse_dates = parse_dates
        self._mask = mask
        self._imodel = imodel
        self._tables = tables

    def __len__(self):
        """Length of :py:attr:`data`."""
        return inspect.get_length(self.data)

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
    def dtypes(self):
        """Dictionary of data types on :py:attr:`data`."""
        return self._return_property("_dtypes")

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
    def selected(self):
        """Selection of :py:attr:`data`.

        This property is set if ``overwrite`` is ``False`` in one of the selection methods:

        * :py:func:`DataBundle.select_true`
        * :py:func:`DataBundle.select_from_list`
        * :py:func:`DataBundle.select_from_index`
        """
        return self._return_property("_selected")

    @property
    def deselected(self):
        """Non-selection of :py:attr:`data`.

        This property is set if ``overwrite`` is ``False`` in one of the selection methods:

        * :py:func:`DataBundle.select_true`
        * :py:func:`DataBundle.select_from_list`
        """
        return self._return_property("_deselected")

    @property
    def tables(self):
        """CDM tables."""
        return self._return_property("_tables")

    @tables.setter
    def tables(self, value):
        self._tables = value

    @property
    def tables_dups_flagged(self):
        """Flagged duplicates of :py:attr:`tables`.

        This property is set if ``overwrite`` is ``False`` in :py:func:`DataBundle.flag_duplicates`.
        """
        return self._return_property("_tables_dups_flagged")

    @property
    def tables_dups_removed(self):
        """Removed duplicates of :py:attr:`tables`.

        This property is set if ``overwrite`` is ``False`` in :py:func:`DataBundle.remove_duplicates`.
        """
        return self._return_property("_tables_dups_removed")

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

    def select_true(self, overwrite=True, **kwargs):
        """Select valid values from :py:attr:`data` via :py:attr:`mask`.

        Parameters
        ----------
        overwrite: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`cdm_reader_mapper.DataBundle`
            else set new attribute both :py:attr:`selected` containing valid value
            and :py:attr:`deselected` containing invalid data.
            Default: True

        Note
        ----
        Use this for :py:attr:`data` only. It does not work for :py:attr:`tables`.

        Examples
        --------
        Select valid values only with overwriting the old data.

        >>> db.select_true()
        >>> true_values = db.data

        Select valid values only without overwriting the old data.

        >>> db.select_true(overwrite=False)
        >>> true_values = db.selected
        >>> false_values = db.deselected

        See Also
        --------
        DataBundle.select_from_list : Select columns from `data` with specific values.
        DataBundle.select_from_index : Select rows of `data` with specific indexes.

        Note
        ----
        For more information see :py:func:`select_true`
        """
        selected = select.select_true(self._data, self._mask, **kwargs)
        if overwrite is True:
            self._data = selected[0]
        else:
            self._selected = selected[0]
        self._deselected = selected[1]
        return self

    def select_from_list(self, selection, overwrite=True, **kwargs):
        """Select columns from :py:attr:`data` with specific values.

        Parameters
        ----------
        selection: dict
            Keys: columns to be selected.
            Values: values in keys to be selected
        overwrite: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`cdm_reader_mapper.DataBundle`
            else set new attribute both :py:attr:`selected` containing valid value
            and :py:attr:`deselected` containing invalid data.
            Default: True

        Note
        ----
        Use this for :py:attr:`data` only. It does not work for :py:attr:`tables`.

        Examples
        --------
        Select specific columns with overwriting the old data.

        >>> db.select_from_list(selection={("c1", "B1"): [26, 41]})
        >>> true_values = db.selected

        Select specific columns without overwriting the old data.

        >>> db.select_from_list(selection={("c1", "B1"): [26, 41]}, overwrite=False)
        >>> true_values = db.selected
        >>> false_values = db.deselected

        See Also
        --------
        DataBundle.select_from_index : Select rows of `data` with specific indexes.
        DataBundle.select_true : Select valid values from `data` via `mask`.

        Note
        ----
        For more information see :py:func:`select_from_list`
        """
        selected = select.select_from_list(self._data, selection, **kwargs)
        if overwrite is True:
            self._data = selected[0]
        else:
            self._selected = selected[0]
        self._deselected = selected[1]
        return self

    def select_from_index(self, index, overwrite=True, **kwargs):
        """Select rows of :py:attr:`data` with specific indexes.

        Parameters
        ----------
        index: list
            Indexes to be selected.
        overwrite: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`cdm_reader_mapper.DataBundle`
            else set new attribute both :py:attr:`selected` containing valid value
            and :py:attr:`deselected` containing invalid data.
            Default: True

        Note
        ----
        Use this for :py:attr:`data` only. It does not work for :py:attr:`tables`.

        Examples
        --------
        Select specific columns with overwriting the old data.

        >>> db.select_from_index(index=[0, 2, 4])
        >>> true_values = db.selected

        Select specific columns without overwriting the old data.

        >>> db.select_from_index([0, 2, 4], overwrite=False)
        >>> true_values = db.selected
        >>> false_values = db.deselected

        See Also
        --------
        DataBundle.select_from_list : Select columns from `data` with specific values.
        DataBundle.select_true : Select valid values from `data` via `mask`.

        Note
        ----
        For more information see :py:func:`select_from_index`
        """
        selected = select.select_from_index(self._data, index, **kwargs)
        if overwrite is True:
            self._data = selected
        else:
            self._selected = selected
        return self

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
        return inspect.count_by_cat(self._data, **kwargs)

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
        self._data = replace.replace_columns(df_l=self._data, df_r=df_corr, **kwargs)
        self._columns = self._data.columns
        return self

    def correct_datetime(self):
        """Correct datetime information in :py:attr:`data`.

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
        self._data = correct_datetime(self._data, self._imodel)
        return self

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

    def correct_pt(self):
        """Correct platform type information in :py:attr:`data`.


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
        self._data = correct_pt(self._data, self._imodel)
        return self

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

    def write_mdf(self, **kwargs):
        """Write MDF data on disk.

        Examples
        --------
        >>> db.write_mdf()

        See Also
        --------
        DataBundle.write_tables : Write MDF data on disk.
        read_mdf : Read original marine-meteorological data from disk.
        read_tables : Read CDM tables from disk.

        Note
        ----
        For more information see :py:func:`write_mdf`
        """
        write_mdf(self._data, mask=self._mask, **kwargs)

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
        write_tables(self._tables, **kwargs)

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
            If ``True`` overwrite :py:attr:`tables` in DataBundle.
            Else set new attribute :py:attr:`tables_dups_flagged` containing flagged duplicates.
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

        >>> db.flag_duplicates()
        >>> flagged_tables = db.tables_dups_flagged

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
        else:
            self._tables_dups_flagged = df_
        return self

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
            If ``True`` overwrite :py:attr:`tables` in :py:class:`cdm_reader_mapper.DataBundle`.
            Else set new attribute :py:attr:`tables_dups_removed` containing flagged duplicates.
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

        >>> db.remove_duplicates()
        >>> removed_tables = db.tables_dups_removed

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
        else:
            self._tables_dups_removed = df_
        return self
