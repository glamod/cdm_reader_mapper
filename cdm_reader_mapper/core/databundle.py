"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations

import csv

from copy import deepcopy
from io import StringIO as StringIO

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
from cdm_reader_mapper.common.pandas_TextParser_hdlr import make_copy
from cdm_reader_mapper.mdf_reader import properties


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
        if self._data is not None:
            return get_length(self._data)
        raise KeyError("data is not defined.")

    def __getattr__(self, attr):
        """Apply attribute to :py:attr:`data` if attribute is not defined for :py:class:`~DataBundle` ."""

        class SubscriptableMethod:
            def __init__(self, func):
                self.func = func

            def __getitem__(self, item):
                return self.func(item)

            def __call__(self, *args, **kwargs):
                return self.func(*args, **kwargs)

        def method(*args, **kwargs):
            try:
                return attr_func(*args, **kwargs)
            except TypeError:
                return attr_func[args]
            except Exception:
                raise ValueError(f"{attr} is neither callable nor subscriptable.")

        def reader_method(*args, **kwargs):
            data_buffer = StringIO()
            TextParser = make_copy(data)
            chunksize = TextParser.chunksize
            inplace = kwargs.get("inplace", False)
            for df_ in TextParser:
                nonlocal attr_func
                attr_func = getattr(df_, attr)
                if not callable(attr_func):
                    return attr_func
                result_df = method(*args, **kwargs)
                if result_df is None:
                    result_df = df_
                if result_df is None:
                    result_df = df_
                result_df.to_csv(
                    data_buffer,
                    header=False,
                    mode="a",
                    encoding=self.encoding,
                    index=False,
                    quoting=csv.QUOTE_NONE,
                    sep=properties.internal_delimiter,
                    quotechar="\0",
                    escapechar="\0",
                )
                data_buffer.seek(0)
            TextParser = pd.read_csv(
                data_buffer,
                names=result_df.columns,
                chunksize=chunksize,
                dtype=self.dtypes,
                delimiter=properties.internal_delimiter,
                quotechar="\0",
                escapechar="\0",
            )
            if inplace:
                self._data = TextParser
                return self
            return TextParser

        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(f"DataBundle object has no attribute {attr}.")

        _data = "_data"
        if not hasattr(self, _data):
            raise NameError("'data' is not defined in DataBundle object.")
        data = getattr(self, _data)
        if isinstance(data, pd.DataFrame):
            attr_func = getattr(data, attr)
            if not callable(attr_func):
                return attr_func
            return SubscriptableMethod(method)
        elif isinstance(data, pd.io.parsers.TextFileReader):
            TextParser = make_copy(data)
            first_chunk = next(TextParser)
            attr_func = getattr(first_chunk, attr)
            if not callable(attr_func):
                return attr_func
            return SubscriptableMethod(reader_method)
        raise TypeError("'data' is neither a DataFrame nor a TextFileReader object.")

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

    def _get_db(self, inplace):
        if inplace is True:
            return self
        return self.copy()

    def _stack(self, other, datasets, inplace, **kwargs):
        db_ = self._get_db(inplace)
        if not isinstance(other, list):
            other = [other]
        if not isinstance(datasets, list):
            datasets = [datasets]
        for data in datasets:
            _data = f"_{data}"
            _df = getattr(db_, _data) if hasattr(db_, _data) else pd.DataFrame()
            to_concat = [
                getattr(concat, _data) for concat in other if hasattr(concat, _data)
            ]
            if not to_concat:
                continue
            if not _df.empty:
                to_concat = [_df] + to_concat
            _df = pd.concat(to_concat, **kwargs)
            _df = _df.reset_index(drop=True)
            setattr(self, f"_{data}", _df)

        return db_

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
            else return a copy of :py:class:`~DataBundle` with stacked datasets.
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
            else return a copy of :py:class:`~DataBundle` with stacked datasets.
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
        db = DataBundle()
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                value = deepcopy(value)
            elif isinstance(value, pd.DataFrame):
                value = value.copy()
            elif isinstance(value, pd.io.parsers.TextFileReader):
                value = make_copy(value)
            setattr(db, key, value)
        return db

    def select_true(self, mask=False, return_invalid=False, inplace=False, **kwargs):
        """Select valid values from :py:attr:`data` via :py:attr:`mask`.

        Parameters
        ----------
        mask: bool
            If ``True`` select also valid values from :py:attr:`mask`
            Default: False
        return_invalid: bool
            If True return invalid data additionally.
            Default: False
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with valid values only in :py:attr:`data`.
            Default: False

        Note
        ----
        If `return_invalid` is ``True`` this function returns two values.

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
        db_ = self._get_db(inplace)
        selected = select_true(db_._data, db_._mask, **kwargs)
        db_._data = selected[0]
        if mask is True:
            db_._mask = select_from_index(db_._mask, db_.index, **kwargs)[0]
        if return_invalid is True:
            return db_, selected[1]
        return db_

    def select_from_list(
        self, selection, mask=False, return_invalid=False, inplace=False, **kwargs
    ):
        """Select columns from :py:attr:`data` with specific values.

        Parameters
        ----------
        selection: dict
            Keys: columns to be selected.
            Values: values in keys to be selected
        mask: bool
            If ``True`` select also valid values from :py:attr:`mask`
            Default: False
        return_invalid: bool
            If True return invalid data additionally.
            Default: False
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with selected columns only in :py:attr:`data`.
            Default: False

        Note
        ----
        If `return_invalid` is ``True`` this function returns two values.

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
        db_ = self._get_db(inplace)
        selected = select_from_list(db_._data, selection, **kwargs)
        db_._data = selected[0]
        if mask is True:
            db_._mask = select_from_index(db_._mask, db_.index, **kwargs)[0]
        if return_invalid is True:
            return db_, selected[1]
        return db_

    def select_from_index(
        self, index, mask=False, return_invalid=False, inplace=False, **kwargs
    ):
        """Select rows of :py:attr:`data` with specific indexes.

        Parameters
        ----------
        index: list
            Indexes to be selected.
        mask: bool
            If ``True`` select also valid values from :py:attr:`mask`
            Default: False
        return_invalid: bool
            If True return invalid data additionally.
            Default: False
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with selected rows only in :py:attr:`data`.
            Default: False

        Note
        ----
        If `return_invalid` is ``True`` this function returns two values.

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
        db_ = self._get_db(inplace)
        selected = select_from_index(db_._data, index, **kwargs)
        db_._data = selected[0]
        if mask is True:
            db_._mask = select_from_index(db_._mask, index, **kwargs)[0]
        if return_invalid is True:
            return db_, selected[1]
        return db_

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
            else return a copy of :py:class:`~DataBundle` with replaced column names in :py:attr:`data`.
            Default: False

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
        return db_

    def correct_datetime(self, inplace=False):
        """Correct datetime information in :py:attr:`data`.

        Parameters
        ----------
        inplace: bool
            If ``True`` overwrite :py:attr:`data` in :py:class:`~DataBundle`
            else return a copy of :py:class:`~DataBundle` with datetime-corrected values in :py:attr:`data`.
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
        db_ = self._get_db(inplace)
        db_._data = correct_datetime(db_._data, db_._imodel)
        return db_

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
            else return a copy of :py:class:`~DataBundle` with platform-corrected values in :py:attr:`data`.
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
        db_ = self._get_db(inplace)
        db_._data = correct_pt(db_._data, db_._imodel)
        return db_

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
            else return a copy of :py:class:`~DataBundle` with :py:attr:`data` as CDM tables.
            Default: False

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
            else return a copy of :py:class:`~DataBundle` with :py:attr:`data` containing flagged duplicates.
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
        db_ = self._get_db(inplace)
        db_.DupDetect.flag_duplicates(**kwargs)
        if db_._mode == "tables" and "header" in db_._data:
            db_._data["header"] = db_.DupDetect.result
        else:
            db_._data = db_.DupDetect.result
        return db_

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
            else return a copy of :py:class:`~DataBundle` with :py:attr:`data` containing no duplicates.
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
        db_ = self._get_db(inplace)
        db_.DupDetect.remove_duplicates(**kwargs)
        header_ = db_.DupDetect.result
        db_._data = db_._data[db_._data.index.isin(header_.index)]
        return db_
