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

    def __getattr__(self, attr):
        """Apply attribute to :py:attr:`data` if attribute is not defined for :py:class:`~DataBundle` ."""

        def method(*args, **kwargs):
            return attr(*args, **kwargs)

        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(f"DataBundle object has no attribute {attr}.")

        _data = "_data"
        if not hasattr(self, _data):
            raise NameError("'data' is not defined in DataBundle object.")
        _df = getattr(self, _data)
        attr = getattr(_df, attr)
        if not callable(attr):
            return attr

        return method

    def __repr__(self):
        """Return a string representation for :py:attr:`data`."""
        return self._data.__repr__()

    def __setitem__(self, item, value):
        """Make class support item assignment for :py:attr:`data`."""
        if isinstance(item, str):
            if hasattr(self, item):
                setattr(self, item, value)
        else:
            self._data.__setitem__(item, value)

    def __getitem__(self, item):
        """Make class subscriptable."""
        if isinstance(item, str):
            if hasattr(self, item):
                return getattr(self, item)
        return self._data.__getitem__(item)

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
        return self._data.columns

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

    def _stack(self, other, datasets, inplace, **kwargs):
        if not isinstance(other, list):
            other = [other]
        db = self.copy()
        if not isinstance(datasets, list):
            datasets = [datasets]
        for data in datasets:
            _data = f"_{data}"
            _df = getattr(db, _data) if hasattr(db, _data) else pd.DataFrame()
            to_concat = [
                getattr(concat, _data) for concat in other if hasattr(concat, _data)
            ]
            if not to_concat:
                continue
            if not _df.empty:
                to_concat = [_df] + to_concat
            _df = pd.concat(to_concat, **kwargs)
            _df = _df.reset_index(drop=True)
            if inplace is True:
                setattr(self, f"_{data}", _df)

        if inplace is True:
            return self
        return db

    def add(self, addition):
        """Adding information to a :py:class:`~DataBundle`.

        Parameters
        ----------
        addition: dict
             Additional elements to add to the :py:class:`~DataBundle`.

        Examples
        --------
        >>> tables = read_tables("path_to_files")
        >>> db = db.add({"data": tables})
        """
        for name, data in addition.items():
            setattr(self, f"_{name}", data)
        return self

    def stack_v(self, other, datasets=["data", "mask"], inplace=False, **kwargs):
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
            else return stacked datasets.
            Default: False

        Note
        ----
        The DataFrames in the :py:class:`~DataBundle` have to have the same data columns!

        Examples
        --------
        >>> db = db1.stack_v(db2, datasets=["data", "mask"])

        See Also
        --------
        DataBundle.stack_h : Stack multiple DataBundle's horizontally.
        """
        return self._stack(other, datasets, inplace, **kwargs)

    def stack_h(self, other, datasets=["data", "mask"], inplace=False, **kwargs):
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
            else return stacked datasets.
            Default: False

        Note
        ----
        The DataFrames in the :py:class:`~DataBundle` may have different data columns!

        Examples
        --------
        >>> db = db1.stack_h(db2, datasets=["data", "mask"])

        See Also
        --------
        DataBundle.stack_v : Stack multiple DataBundle's vertically.
        """
        return self._stack(other, datasets, inplace, axis=1, join="outer", **kwargs)

    def copy(self):
        """Make deep copy of a :py:class:`~DataBundle`.

        Examples
        --------
        >>> db2 = db.copy()
        """
        return deepcopy(self)

    def select_true(self, inplace=False, **kwargs):
        """Select valid values from :py:attr:`data` via :py:attr:`mask`.

        Parameters
        ----------
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return list containing both DataFrame with true and DataFrame with invalid rows.
            Default: False

        Examples
        --------
        Select valid values only without overwriting the old data.

        >>> true_values, false_values = db.select_true()

        Select valid values only with overwriting the old data.

        >>> db.select_true(inplace=True)
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
        if inplace is True:
            self._data = selected[0]
            return self
        return selected

    def select_from_list(self, selection, inplace=False, **kwargs):
        """Select columns from :py:attr:`data` with specific values.

        Parameters
        ----------
        selection: dict
            Keys: columns to be selected.
            Values: values in keys to be selected
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return list containing both DataFrame with true and DataFrame with invalid entries.
            Default: False

        Examples
        --------
        Select specific columns without overwriting the old data.

        >>> true_values, false_values = db.select_from_list(
        ...     selection={("c1", "B1"): [26, 41]},
        ... )

        Select specific columns with overwriting the old data.

        >>> db.select_from_list(selection={("c1", "B1"): [26, 41]}, inplace=True)
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
        if inplace is True:
            self._data = selected[0]
            return self
        return selected

    def select_from_index(self, index, inplace=False, **kwargs):
        """Select rows of :py:attr:`data` with specific indexes.

        Parameters
        ----------
        index: list
            Indexes to be selected.
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return list containing both DataFrame with true and DataFrame with invalid entries.
            Default: False

        Examples
        --------
        Select specific columns without overwriting the old data.

         >>> true_values, false_values = db.select_from_index([0, 2, 4])

        Select specific columns with overwriting the old data.

        >>> db.select_from_index(index=[0, 2, 4], inplace=True)
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
        if inplace is True:
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

    def replace_columns(self, df_corr, subset=None, inplace=False, **kwargs):
        """Replace columns in :py:attr:`data`.

        Parameters
        ----------
        df_corr: pandas.DataFrame
            Data to be inplaced.
        subset: str, list, optional
            Select subset by columns. This option is useful for multi-indexed :py:attr:`data`.
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
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
        _data = self._data.copy()
        if subset is not None:
            _data[subset] = replace_columns(df_l=_data[subset], df_r=df_corr, **kwargs)
        else:
            _data = replace_columns(df_l=_data, df_r=df_corr, **kwargs)

        if inplace is True:
            self._data = _data
            self._columns = self._data.columns
            return self

        db = self.copy()
        db._data = _data
        db._columns = _data.columns
        return db

    def correct_datetime(self, inplace=False):
        """Correct datetime information in :py:attr:`data`.

        Parameters
        ----------
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
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
        db = self.copy()
        _data = correct_datetime(db._data, db._imodel)
        if inplace is True:
            self._data = _data
            return self
        db._data = _data
        return db

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

    def correct_pt(self, inplace=False):
        """Correct platform type information in :py:attr:`data`.

        Parameters
        ----------
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
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
        db = self.copy()
        _data = correct_pt(db._data, db._imodel)
        if inplace is True:
            self._data = _data
            return self
        db._data = _data
        return db

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

    def map_model(self, inplace=False, **kwargs):
        """Map :py:attr:`data` to the Common Data Model.
        Write output to :py:attr:`data`.

        Parameters
        ----------
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return CDM tables.
            Default: False

        Examples
        --------
        >>> cdm_tables = db.map_model()

        Note
        ----
        For more information see :py:func:`map_model`
        """
        db = self.copy()
        _tables = map_model(db._data, db._imodel, **kwargs)
        if inplace is True:
            db_ = self
        else:
            db_ = db
        db_._mode = "tables"
        db_._columns = _tables.columns
        db_._data = _tables
        return db_

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

    def flag_duplicates(self, inplace=False, **kwargs):
        """Flag detected duplicates in :py:attr:`data`.

        Parameters
        ----------
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
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
        db = self.copy()
        db.DupDetect.flag_duplicates(**kwargs)
        if db._mode == "tables" and "header" in db._data:
            db._data["header"] = db.DupDetect.result
        else:
            db._data = db.DupDetect.result
        if inplace is True:
            self._data = db._data
            return self
        return db

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

    def remove_duplicates(self, inplace=False, **kwargs):
        """Remove detected duplicates in :py:attr:`data`.

        Parameters
        ----------
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
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
        db = self.copy()
        db.DupDetect.remove_duplicates(**kwargs)
        header_ = db.DupDetect.result
        db._data = db._data[db._data.index.isin(header_.index)]
        if inplace is True:
            self._data = db._data
            return self
        return db
