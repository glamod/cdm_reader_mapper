"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import pandas as pd

from cdm_reader_mapper.cdm_mapper.mapper import map_model
from cdm_reader_mapper.common import (
    count_by_cat,
    get_length,
    replace_columns,
    split_by_boolean_false,
    split_by_boolean_true,
    split_by_column_entries,
    split_by_index,
)
from cdm_reader_mapper.common.iterators import ParquetStreamReader, is_valid_iterator
from cdm_reader_mapper.metmetpy import (
    correct_datetime,
    correct_pt,
    validate_datetime,
    validate_id,
)

from ._utilities import (
    SubscriptableMethod,
    _copy,
    _normalize_data_input,
    _normalize_mask_input,
    _validate_mode,
    combine_attribute_values,
    reader_method,
)
from .writer import write


properties = {
    "data",
    "columns",
    "dtypes",
    "mask",
    "imodel",
    "mode",
    "parse_dates",
    "encoding",
}


class DataBundle:
    r"""
    Container for tabular data and associated metadata.

    This class wraps either an in-memory `pd.DataFrame` or a
    `ParquetStreamReader` for chunked, disk-backed processing. It provides
    a unified interface for accessing DataFrame-like attributes and methods,
    transparently handling streaming data where required.

    Parameters
    ----------
    data : pandas.DataFrame or Iterable[pandas.DataFrame] or ParquetStreamReader, optional
        Input data. If an iterable is provided, it is converted into a
        `ParquetStreamReader` for streaming.
    columns : pandas.Index or pandas.MultiIndex or list, optional
        Column labels used when initializing empty data.
    dtypes : pandas.Series or dict, optional
        Data types for columns.
    parse_dates : list or bool, optional
        Instructions for parsing dates.
    encoding : str, optional
        Encoding associated with the data.
    mask : pandas.DataFrame or Iterable[pandas.DataFrame] or ParquetStreamReader, optional
        Boolean mask aligned with `data`. If not provided, an empty mask is created.
    imodel : str, optional
        Name of the input data model.
    mode : {"data", "tables"}, default "data"
        Data representation mode.

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
        data: pd.DataFrame | Iterable[pd.DataFrame] | None = None,
        columns: pd.Index | pd.MultiIndex | list[Any] | None = None,
        dtypes: pd.Series | dict[str | tuple[str, str], Any] | None = None,
        parse_dates: list[Any] | bool | None = None,
        encoding: str | None = None,
        mask: pd.DataFrame | Iterable[pd.DataFrame] | None = None,
        imodel: str | None = None,
        mode: Literal["data", "tables"] = "data",
    ) -> None:
        """
        Initialization of a DataBundle instance.

        Parameters
        ----------
        data : pandas.DataFrame or Iterable[pandas.DataFrame] or ParquetStreamReader, optional
            Input data. If an iterable is provided, it is converted into a
            `ParquetStreamReader` for streaming.
        columns : pandas.Index or pandas.MultiIndex or list, optional
            Column labels used when initializing empty data.
        dtypes : pandas.Series or dict, optional
            Data types for columns.
        parse_dates : list or bool, optional
            Instructions for parsing dates.
        encoding : str, optional
            Encoding associated with the data.
        mask : pandas.DataFrame or Iterable[pandas.DataFrame] or ParquetStreamReader, optional
            Boolean mask aligned with `data`. If not provided, an empty mask is created.
        imodel : str, optional
            Name of the input data model.
        mode : {"data", "tables"}, default "data"
            Data representation mode.

        Raises
        ------
        ValueError
            If `mode` is invalid.
        TypeError
            If `data` and/or `mask` has an unsupported type.
        """
        _validate_mode(mode)

        data = _normalize_data_input(data, columns, dtypes)
        mask = _normalize_mask_input(mask, data)

        self._data: pd.DataFrame | ParquetStreamReader = data
        self._columns = columns
        self._dtypes = dtypes
        self._parse_dates = parse_dates
        self._encoding = encoding
        self._mask: pd.DataFrame | ParquetStreamReader = mask
        self._imodel = imodel
        self._mode = mode

    def __len__(self) -> int:
        """
        Length of :py:attr:`data`.

        Returns
        -------
        int
            Number of rows in the underlying data.

        Raises
        ------
        TypeError
             If the computed length is not an integer.
        """
        length = get_length(self._data)
        if isinstance(length, int):
            return length
        raise TypeError(f"Length is not an integer: {length}, {type(length)}")

    def __getattr__(self, attr: str) -> Any:
        """
        Apply attribute to :py:attr:`data` if attribute is not defined for :py:class:`~DataBundle` .

        Parameters
        ----------
        attr : str
            Name of the attribute.

        Returns
        -------
        Any
            Attribute value, callable wrapper, or computed result.

        Raises
        ------
        AttributeError
            If the attribute does not exist.
        ValueError
            If the data stream is empty.
        TypeError
            If the underlying data type is unsupported.
        """
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(f"DataBundle object has no attribute {attr}.")

        data = self._data

        if isinstance(data, pd.DataFrame):
            attr_func = getattr(data, attr)
            if not callable(attr_func):
                return attr_func
            return SubscriptableMethod(attr_func)

        if isinstance(data, ParquetStreamReader):
            # This allows db.read(), db.close(), db.get_chunk() to work
            if hasattr(data, attr):
                return getattr(data, attr)

            data = data.copy()

            try:
                first_chunk = data.get_chunk()
            except (StopIteration, ValueError) as err:
                raise ValueError("Cannot access attribute on empty data stream.") from err

            if not hasattr(first_chunk, attr):
                # Restore state before raising error
                data.prepend(first_chunk)
                raise AttributeError(f"DataFrame chunk has no attribute '{attr}'.")

            attr_value = getattr(first_chunk, attr)

            if callable(attr_value):
                # METHOD CALL (e.g., .dropna(), .fillna())
                # Put the chunk BACK so the reader_method sees the full stream.
                data.prepend(first_chunk)

                def wrapped_reader_method(*args: Any, **kwargs: Any) -> ParquetStreamReader | None:
                    return reader_method(data, attr, *args, **kwargs)

                return SubscriptableMethod(wrapped_reader_method)
            else:
                # PROPERTY ACCESS (e.g., .shape, .dtypes)
                # DO NOT put the chunk back yet. Pass the 'first_value'
                # and the 'data' iterator (which is now at chunk 2) to the combiner.
                # The combiner will consume the rest.
                return combine_attribute_values(attr_value, data, attr)

        raise TypeError(f"'data' is {type(data)}, expected DataFrame or ParquetStreamReader.")

    def __repr__(self) -> str:
        """
        Return a string representation for :py:attr:`data`.

        Returns
        -------
        str
            String representation for the underlying data.
        """
        return self._data.__repr__()

    def __setitem__(self, item: Any, value: Any) -> None:
        """
        Make class support item assignment for :py:attr:`data`.

        Parameters
        ----------
        item : Any
            Column name or property key.
        value : Any
            Value to assign.
        """
        if isinstance(item, str) and item in properties:
            setattr(self, item, value)
        else:
            self._data[item] = value

    def __getitem__(self, item: Any) -> Any:
        """
        Make class subscriptable.

        Parameters
        ----------
        item : Any
            Key or column name.

        Returns
        -------
        Any
            Item `item` of underlying data.
        """
        if isinstance(item, str):
            if hasattr(self, item):
                return getattr(self, item)
        return self._data.__getitem__(item)

    def _return_property(self, property: str) -> Any:
        """
        Return an internal property if it exists.

        Parameters
        ----------
        property : str
            Name of the attribute.

        Returns
        -------
        Any
            Internal property `property`.
        """
        if hasattr(self, property):
            return getattr(self, property)

    @property
    def data(self) -> pd.DataFrame | ParquetStreamReader:
        """
        Underlying MDF data.

        Returns
        -------
        pd.DataFrame or ParquetStreamReader
            Underlying MDf data.
        """
        return self._return_property("_data")

    @data.setter
    def data(self, value: pd.DataFrame | ParquetStreamReader) -> None:
        """
        Set the underlying MDF data.

        Parameters
        ----------
        value : pandas.DataFrame or ParquetStreamReader
            Value to be set.
        """
        self._data = value

    @property
    def columns(self) -> pd.Index | pd.MultiIndex:
        """
        Column labels of :py:attr:`data`.

        Returns
        -------
        pd.Index or pd.MultiIndex
            Column labels of the underlying MDf data.
        """
        return self._data.columns

    @columns.setter
    def columns(self, value: pd.Index | pd.MultiIndex | list[Any]) -> None:
        """
        Set column labels of the underlying MDF data.

        Parameters
        ----------
        value : pandas.Index or pandas.MultiIndex or list
            Value to be set.
        """
        self._columns = value

    @property
    def dtypes(self) -> pd.Series | dict[str, Any] | None:
        """
        Dictionary of data types on :py:attr:`data`.

        Returns
        -------
        pd.Series or dict or None
            Data types of underlying MDF data.
        """
        return self._return_property("_dtypes")

    @property
    def parse_dates(self) -> list[Any] | bool | None:
        """
        Information of how to parse dates in :py:attr:`data`.

        Returns
        -------
        list or bool or None
            Information of how to parse dates in underlying MDF data.

        See Also
        --------
        :py:func:`pd.read_csv` : Read CSV file using pandas.
        """
        parse_dates_ = self._return_property("_parse_dates")
        if parse_dates_ is None:
            return None
        if isinstance(parse_dates_, (list, bool)):
            return parse_dates_
        raise TypeError(f"parse_dates has type {type(parse_dates_)}; expected list[Any], bool, or None.")

    @property
    def encoding(self) -> str | None:
        """
        A string representing the encoding to use in the :py:attr:`data`.

        Returns
        -------
        str or None
            String representing the encoding to use in the underlying MDF data.

        See Also
        --------
        :py:func:`pd.to_csv` : Write data with encoding to CSV file.
        """
        encoding_ = self._return_property("_encoding")
        if encoding_ is None:
            return None
        if isinstance(encoding_, str):
            return encoding_
        raise TypeError(f"encoding has type {type(encoding_)}; expected str or None.")

    @property
    def mask(self) -> pd.DataFrame | ParquetStreamReader:
        """
        MDF validation mask.

        Returns
        -------
        pd.DataFrame or ParquetStreamReader
            Validation mask of the underlying MDF data.
        """
        return self._return_property("_mask")

    @mask.setter
    def mask(self, value: pd.DataFrame | ParquetStreamReader) -> None:
        """
        Set the validation mask of underlying MDF data.

        Parameters
        ----------
        value : pd.DataFrame or ParquetStreamReader
            Value to be set.
        """
        self._mask = value

    @property
    def imodel(self) -> str | None:
        """
        Name of the MDF/CDM input model.

        Returns
        -------
        str or None
            Name of the MDF/CDM input model if available.
        """
        imodel_ = self._return_property("_imodel")
        if imodel_ is None:
            return None
        if isinstance(imodel_, str):
            return imodel_
        raise TypeError(f"imodel has type {type(imodel_)}; expected str or None.")

    @imodel.setter
    def imodel(self, value: str) -> None:
        """
        Set the data model name of underlying MDF data.

        Parameters
        ----------
        value : str
            Value to be set.
        """
        self._imodel = value

    @property
    def mode(self) -> str:
        """
        Data mode.

        Returns
        -------
        str
            Current data mode.

        Raises
        ------
        TypeError
            If mode of the underlying data is not a string.
        """
        mode_ = self._return_property("_mode")
        if isinstance(mode_, str):
            return mode_
        raise TypeError(f"mode_ has type {type(mode_)}; expected str.")

    @mode.setter
    def mode(self, value: Literal["data", "tables"]) -> None:
        """
        Set the data mode name of underlying data.

        Parameters
        ----------
        value : {'data', 'tables'}
            Value to be set

        Raises
        ------
        ValueError
            If `value` is not one of `data` or `tables`.
        """
        if value not in ("data", "tables"):
            raise ValueError("value must be one of 'data' or 'tables'.")
        self._mode = value

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

    def _stack(self, other: DataBundle | Sequence[DataBundle], datasets: str | Sequence[str], inplace: bool, **kwargs: Any) -> DataBundle | None:
        r"""
        Concatenate datasets from multiple DataBundle instances.

        Parameters
        ----------
        other : :py:class:`~DataBundle` or Sequence of :py:class:`~DataBundle`
            Other DataBundle instances whose data should be stacked with the current instance.
        datasets : str or Sequence of str
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

        if isinstance(other, DataBundle):
            other = [other]
        if isinstance(datasets, str):
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

        if db_cp is None:
            return None

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
        if db_ is None:
            return None
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
        self,
        other: DataBundle | Sequence[DataBundle],
        datasets: str | Sequence[str] | Literal["data", "mask"] = ("data", "mask"),
        inplace: bool = False,
        **kwargs: Any,
    ) -> DataBundle | None:
        r"""
        Stack multiple :py:class:`~DataBundle`'s vertically.

        Parameters
        ----------
        other : :py:class:`~DataBundle` or Sequence of :py:class:`~DataBundle`
            List of other :py:class:`~DataBundle` to stack vertically.
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
        self,
        other: DataBundle | Sequence[DataBundle],
        datasets: str | Sequence[str] | Literal["data", "mask"] = ("data", "mask"),
        inplace: bool = False,
        **kwargs: Any,
    ) -> DataBundle | None:
        r"""
        Stack multiple :py:class:`~DataBundle`'s horizontally.

        Parameters
        ----------
        other : :py:class:`~DataBundle` or Sequence of :py:class:`~DataBundle`
            List of other :py:class:`~DataBundle` to stack horizontally.
        datasets : str or Sequence of str, default: [data, mask]
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
        if db_ is None:
            return None
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
        if db_ is None:
            return None
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
        if db_ is None:
            return None
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
        if db_ is None:
            return None
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
        if db_ is None:
            return None
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
        if db_ is None:
            return None
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
        if db_ is None:
            return None
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
        if db_ is None:
            return None
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
