"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from cdm_reader_mapper.cdm_mapper.mapper import map_model
from cdm_reader_mapper.cdm_mapper.table_writer import write_tables
from cdm_reader_mapper.duplicates.duplicates import duplicate_check
from cdm_reader_mapper.metmetpy.datetime.correct import correct as correct_datetime
from cdm_reader_mapper.metmetpy.datetime.validate import validate as validate_datetime
from cdm_reader_mapper.metmetpy.platform_type.correct import correct as correct_pt
from cdm_reader_mapper.metmetpy.station_id.validate import validate as validate_id
from cdm_reader_mapper.operations import inspect, replace, select


class DataBundle:
    """Class for manipulating the MDF data and mapping it to the CDM.

    Parameters
    ----------
    MDFFileReader: MDFFileReader object, optional
        MDF object
    tables: pd.DataFrame, optional
        CDM tables
    data: pd.DataFrame, optional
        MDF DataFrame
    mask: pd.DataFrame, optional
        MDF validation mask

    Examples
    --------
    Getting a DataBundle while reading data from disk.

    >>> from cdm_reader_mapper import read_mdf
    >>> db = read_mdf(source="file_on_disk", imodel="custom_model_name")

    Constructing a DataBundle from already read MDf data.

    >>> from cdm_reader_mapper import DataBundle
    >>> read = read_mdf(source="file_on_disk", imodel="custom_model_name")
    >>> data_ = read.data
    >>> mask_ = read.mask
    >>> db = DataBundle(data=data_, mask=mask_)

    Constructing a DataBundle from already read CDM data.

    >>> from cdm_reader_mapper import read_tables
    >>> tables = read_tables("path_to_files")
    >>> db = DataBundle(tables=tables)
    """

    def __init__(self, MDFFileReader=None, tables=None, data=None, mask=None):
        if MDFFileReader is not None:
            self.data = MDFFileReader.data
            self.columns = MDFFileReader.columns
            self.dtypes = MDFFileReader.dtypes
            self.attrs = MDFFileReader.attrs
            self.parse_dates = MDFFileReader.parse_dates
            self.mask = MDFFileReader.mask
            self.imodel = MDFFileReader.imodel
        if tables is not None:
            self.tables = tables
        if data is not None:
            self.data = data
            self.columns = data.columns
            self.dtypes = data.dtypes
        if mask is not None:
            self.mask = mask

    def __len__(self):
        """Length of ``data``."""
        return inspect.get_length(self.data)

    def _return_property(self, property):
        if hasattr(self, property):
            return property

    @property
    def data(self):
        """MDF pandas.DataFrame data."""
        return self.return_property("data")

    @property
    def columns(self):
        """Column labels of ``data``."""
        return self.return_property("columns")

    @property
    def dtypes(self):
        """Dictionary of data types on ``data``."""
        return self.return_property("dtypes")

    @property
    def attrs(self):
        """Dictionary of attributes on ``data``."""
        return self.return_property("attrs")

    @property
    def mask(self):
        """MDF pandas.DataFrame validation mask."""
        return self.return_property("mask")

    @property
    def imodel(self):
        """Name of the MDF/CDM input model."""
        return self.return_property("imodel")

    @property
    def selected(self):
        """Selection of ``data``.

        This property is set if overwrite is False in one of the selection methods:

        * ``select_true``
        * ``select_from_list``
        * ``select_from_index``
        """
        return self.return_property("selected")

    @property
    def deselected(self):
        """Non-selection of ``data``.

        This property is set if overwrite is False in one of the selection methods:

        * ``select_true``
        * ``select_from_list``
        """
        return self.return_property("deselected")

    @property
    def tables(self):
        """CDM tables."""
        return self.return_property("tables")

    @property
    def tables_dups_flagged(self):
        """Flagged duplicates of `tables``.

        This property is set if overwrite is False in ``flag_duplicates``.
        """
        return self.return_property("tables_dups_flagged")

    @property
    def tables_dups_removed(self):
        """Removed duplicates of `tables``.

        This property is set if overwrite is False in ``remove_duplicates``.
        """
        return self.return_property("tables_dups_removed")

    def add(self, addition):
        """Adding information to a DataBundle.

        Parameters
        ----------
        addition: dict
             Additional elements to add to the ``DataBundle``.

        Examples
        --------
        >>> tables = read_tables("path_to_files")
        >>> db = db.add({"tables": tables})
        """
        for name, data in addition.items():
            setattr(self, name, data)
        return self

    def stack_v(self, other, datasets=["data", "mask", "tables"], **kwargs):
        """Stack multiple DataBundle's vertically.

        Parameters
        ----------
        other: str, list
            List of other DataBundles to stack vertically.
        datasets: str, list
            List of datasets to be stacked.
            Default: ['data', 'mask', 'tables']

        Note
        ----
        The DataFrames in the DataBundles have to have the same data columns!

        Examples
        --------
        >>> db = db1.stack_v(db2, datasets=["data", "mask"])
        """
        if not isinstance(other, list):
            other = [other]
        if not isinstance(datasets, list):
            datasets = [datasets]
        for data in datasets:
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
        """Stack multiple DataBundle's horizontally.

        Parameters
        ----------
        other: str, list
            List of other DataBundles to stack horizontally.
        datasets: str, list
            List of datasets to be stacked
            Default: ['data', 'mask', 'tables']

        Note
        ----
        The DataFrames in the DataBundles may have different data columns!

        Examples
        --------
        >>> db = db1.stack_h(db2, datasets=["data", "mask"])
        """
        if not isinstance(other, list):
            other = [other]
        for data in datasets:
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
        """Make deep copy of a DataBundle.

        Examples
        --------
        >>> db2 = db.copy()
        """
        return deepcopy(self)

    def select_true(self, overwrite=True, **kwargs):
        """Select valid values from ``data`` via ``mask``.

        Parameters
        ----------
        overwrite: bool
            If ``True`` overwrite ``data`` in ``DataBundle
            Else set new attribute both ``selected`` containing valid value
            and ``deselected`` containing invalid data.
            Default: True

        Note
        ----
        Use this for ``data`` only. It does not work for ``tables```.

        Examples
        --------
        Select valid values only with overwriting the old data.

        >>> db.select_true()
        >>> true_values = db.data

        Select valid values only without overwriting the old data.

        >>> db.select_true(overwrite=False)
        >>> true_values = db.selected
        >>> false_values = db.deselected
        """
        selected = select.select_true(self.data, self.mask, **kwargs)
        if overwrite is True:
            self.data = selected[0]
        else:
            self.selected = selected[0]
        self.deselected = selected[1]
        return self

    def select_from_list(self, selection, overwrite=True, **kwargs):
        """Select columns from ``data`` with specific values.

        Parameters
        ----------
        selection: dict
            Keys: columns to be selected.
            Values: values in keys to be selected
        overwrite: bool
            If ``True`` overwrite ``data`` in ``DataBundle
            Else set new attribute both ``selected`` containing valid value
            and ``deselected`` containing invalid data.
            Default: True

        Note
        ----
        Use this for ``data`` only. It does not work for ``tables```.

        Examples
        --------
        Select specific columns with overwriting the old data.

        >>> db.select_from_list(selection={("c1", "B1"): [26, 41]})
        >>> true_values = db.selected

        Select specific columns without overwriting the old data.

        >>> db.select_from_list(selection={("c1", "B1"): [26, 41]}, overwrite=False)
        >>> true_values = db.selected
        >>> false_values = db.deselected
        """
        selected = select.select_from_list(self.data, selection, **kwargs)
        if overwrite is True:
            self.data = selected[0]
        else:
            self.selected = selected[0]
        self.deselected = selected[1]
        return self

    def select_from_index(self, index, overwrite=True, **kwargs):
        """Select rows of ``data`` with specific indexes.

        Parameters
        ----------
        index: list
            Indexes to be selected.
        overwrite: bool
            If ``True`` overwrite ``data`` in ``DataBundle``
            Else set new attribute both ``selected`` containing valid value
            and ``deselected`` containing invalid data.
            Default: True

        Note
        ----
        Use this for ``data`` only. It does not work for ``tables```.

        Examples
        --------
        Select specific columns with overwriting the old data.

        >>> db.select_from_index(index=[0, 2, 4])
        >>> true_values = db.selected

        Select specific columns without overwriting the old data.

        >>> db.select_from_index([0, 2, 4], overwrite=False)
        >>> true_values = db.selected
        >>> false_values = db.deselected
        """
        selected = select.select_from_index(self.data, index, **kwargs)
        if overwrite is True:
            self.data = selected
        else:
            self.selected = selected
        return self

    def unique(self, **kwargs):
        """Get unique values of ``data``.

        Returns
        -------
        dict
            Dictionary with unique values.

        Examples
        --------
        >>> db.unique(columns=("c1", "B1"))
        """
        return inspect.count_by_cat(self.data, **kwargs)

    def replace_columns(self, df_corr, **kwargs):
        """Replace columns in ``data``.

        Parameters
        ----------
        df_corr: pd.DataFrame
            Data to be inplaced.

        Examples
        --------
        >>> import pandas as pd
        >>> df_corr = pr.read_csv("corecction_file_on_disk")
        >>> db.replace_columns(df_corr)
        """
        self.data = replace.replace_columns(df_l=self.data, df_r=df_corr, **kwargs)
        self.columns = self.data.columns
        return self

    def correct_datetime(self):
        """Correct datetime information in ``data``.

        Examples
        --------
        >>> db.correct_datetime()
        """
        self.data = correct_datetime(self.data, self.imodel)
        return self

    def validate_datetime(self):
        """Validate datetime information in ``data``.

        Returns
        -------
        pd.DataFrame
            DataFrame containing True and False values for each index in ``data``.
            True: All datetime information in ``data`` row are valid.
            False: At least one datetime information in ``data`` row is invalid.

        Examples
        --------
        >>> val_dt = db.validate_datetime()
        """
        return validate_datetime(self.data, self.imodel)

    def correct_pt(self):
        """Correct platform type information in ``data``.


        Examples
        --------
        >>> db.correct_pt()
        """
        self.data = correct_pt(self.data, self.imodel)
        return self

    def validate_id(self, **kwargs):
        """Validate station id information in ``data``.

        Returns
        -------
        pd.DataFrame
            DataFrame containing True and False values for each index in ``data``.
            True: All station ID information in ``data`` row are valid.
            False: At least one station ID information in ``data`` row is invalid.

        Examples
        --------
        >>> val_dt = db.validate_id()
        """
        return validate_id(self.data, self.imodel, **kwargs)

    def map_model(self, **kwargs):
        """Map ``data`` to the Common Data Model.
        Write output in ``DataBundle.tables``.

        Examples
        --------
        >>> db.map_model()
        """
        self.tables = map_model(self.data, self.imodel, **kwargs)
        return self

    def write_tables(self, **kwargs):
        """Write CDM tables on disk.

        Note
        ----
        Before writing CDM tables on disk, they have to be provided in ``DataBundle``, e.g. with ``map_model``.

        Examples
        --------
        >>> db.map_model()
        """
        write_tables(self.tables, **kwargs)

    def duplicate_check(self, **kwargs):
        """Duplicate check.

        Note
        ----
        Before processing the duplicate check, CDM tables have to be provided in ``DataBundle``, e.g. with ``map_model``.

        Examples
        --------
        >>> db.duplicate_check()
        """
        self.DupDetect = duplicate_check(self.tables["header"], **kwargs)
        return self

    def flag_duplicates(self, overwrite=True, **kwargs):
        """Flag detected duplicates in ``tables``.

        Parameters
        ----------
        overwrite: bool
            If ``True`` overwrite ``tables`` in ``DataBundle``.
            Else set new attribute ``tables_dups_flagged`` containing flagged duplicates.
            Default: True

        Note
        ----
        Before flagging duplicates, a duplictate check has to be done, ``duplicate_check``.

        Examples
        --------
        Flag duplicates with overwriting ``tables``.

        >>> db.flag_duplicates()
        >>> flagged_tables = db.tables

        Flag duplicates without overwriting ``tables``.

        >>> db.flag_duplicates()
        >>> flagged_tables = db.tables_dups_flagged
        """
        self.DupDetect.flag_duplicates(**kwargs)
        df_ = self.tables.copy()
        df_["header"] = self.DupDetect.result
        if overwrite is True:
            self.tables = df_
        else:
            self.tables_dups_flagged = df_
        return self

    def get_duplicates(self, **kwargs):
        """Get duplicate matches in ``tables``.

        Returns
        -------
        list
            List of tuples containing duplicate matches.

        Note
        ----
        Before getting duplicates, a duplictate check has to be done, ``duplicate_check``.

        Examples
        --------
        >>> matches = db.get_duplicates()
        """
        return self.DupDetect.get_duplicates(**kwargs)

    def remove_duplicates(self, overwrite=True, **kwargs):
        """Remove detected duplicates in ``tables``.

        Parameters
        ----------
        overwrite: bool
            If ``True`` overwrite ``tables`` in ``DataBundle``.
            Else set new attribute ``tables_dups_removed`` containing flagged duplicates.
            Default: True

        Note
        ----
        Before removing duplicates, a duplictate check has to be done, ``duplicate_check``.

        Examples
        --------
        Remove duplicates with overwriting ``tables``.

        >>> db.remove_duplicates()
        >>> removed_tables = db.tables

        Remove duplicates without overwriting ``tables``.

        >>> db.remove_duplicates()
        >>> removed_tables = db.tables_dups_removed
        """
        self.DupDetect.remove_duplicates(**kwargs)
        df_ = self.tables.copy()
        header_ = self.DupDetect.result
        df_ = df_[df_.index.isin(header_.index)]
        if overwrite is True:
            self.tables = df_
        else:
            self.tables_dups_removed = df_
        return self
