"""Common Data Model (CDM) class."""

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

    Attributes
    ----------
    data: pd.DataFrame or pd.io.parsers.TextFileReader
        MDF data
    columns:
        The column labels of ``data``.
    dtypes: dict
        Data types of ``data``.
    attrs: dict
        ``data`` elements attributes.
    parse_dates: bool, list of Hashable, list of lists or dict of {Hashable : list}
        Parameter used in pandas.read_csv.
    mask: pd.DataFrame or pd.io.parsers.TextFileReader
        MDF validation mask
    imodel: str
        Name of the CDM input model.
    """

    def __init__(self, MDFFileReader=None, cdm_tables=None, data=None, mask=None):
        if MDFFileReader is not None:
            self.data = MDFFileReader.data
            self.columns = MDFFileReader.columns
            self.dtypes = MDFFileReader.dtypes
            self.attrs = MDFFileReader.attrs
            self.parse_dates = MDFFileReader.parse_dates
            self.mask = MDFFileReader.mask
            self.imodel = MDFFileReader.imodel
        if cdm_tables is not None:
            self.cdm = cdm_tables
        if data is not None:
            self.data = data
            self.columns = data.columns
            self.dtypes = data.dtypes
        if mask is not None:
            self.mask = mask

    def __len__(self):
        """Length of ``data``."""
        return inspect.get_length(self.data)

    def add(self, addition):
        """Adding information to a DataBundle."""
        for name, data in addition.items():
            setattr(self, name, data)
        return self

    def stack_v(self, other, datasets=["data", "mask", "cdm"], **kwargs):
        """Stack multiple DataBundle's vertically."""
        if not isinstance(other, list):
            other = [other]
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

    def stack_h(self, other, datasets=["data", "mask", "cdm"], **kwargs):
        """Stack multiple DataBundle's horizontally."""
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
        """Make deep copy of a DataBundle."""
        return deepcopy(self)

    def select_true(self, overwrite=True, **kwargs):
        """Select valid values from ``data`` via ``mask``."""
        selected = select.select_true(self.data, self.mask, **kwargs)
        if overwrite is True:
            self.data = selected[0]
        else:
            self.selected = selected[0]
        self.deselected = selected[1]
        return self

    def select_from_list(self, selection, overwrite=True, **kwargs):
        """Select columns of ``data`` from list of column names."""
        selected = select.select_from_list(self.data, selection, **kwargs)
        if overwrite is True:
            self.data = selected[0]
        else:
            self.selected = selected[0]
        self.deselected = selected[1]
        return self

    def select_from_index(self, index, overwrite=True, **kwargs):
        """Select columns of ``data`` from list of column names."""
        selected = select.select_from_index(self.data, index, **kwargs)
        if overwrite is True:
            self.data = selected
        else:
            self.selected = selected
        return self

    def unique(self, **kwargs):
        """Get unique values of ``data``."""
        return inspect.count_by_cat(self.data, **kwargs)

    def replace_columns(self, df_corr, **kwargs):
        """Replace columns in ``data``."""
        self.data = replace.replace_columns(df_l=self.data, df_r=df_corr, **kwargs)
        self.columns = self.data.columns
        return self

    def correct_datetime(self):
        """Correct datetime information in ``data``."""
        self.data = correct_datetime(self.data, self.imodel)
        return self

    def validate_datetime(self):
        """Validate datetime information in ``data``."""
        return validate_datetime(self.data, self.imodel)

    def correct_pt(self):
        """Correct platform type information in ``data``."""
        self.data = correct_pt(self.data, self.imodel)
        return self

    def validate_id(self, **kwargs):
        """Validate station id information in ``data``."""
        return validate_id(self.data, self.imodel, **kwargs)

    def map_model(self, **kwargs):
        """Map ``data`` to the Common Data Model."""
        self.cdm = map_model(self.data, self.imodel, **kwargs)
        return self

    def write_tables(self, **kwargs):
        """Write CDM tables on disk."""
        write_tables(self.cdm, **kwargs)

    def duplicate_check(self, **kwargs):
        """Duplicate check."""
        self.DupDetect = duplicate_check(self.cdm["header"], **kwargs)
        return self

    def flag_duplicates(self, overwrite=True, **kwargs):
        """Flag detected duplicates in ``cdm``."""
        self.DupDetect.flag_duplicates(**kwargs)
        df_ = self.cdm.copy()
        df_["header"] = self.DupDetect.result
        if overwrite is True:
            self.cdm = df_
        else:
            self.cdm_dups_flagged = df_
        return self

    def get_duplicates(self, **kwargs):
        """Get duplicate matches in ``cdm``."""
        return self.DupDetect.get_duplicates(**kwargs)

    def remove_duplicates(self, overwrite=True, **kwargs):
        """Remove detected duplicates in ``cdm``."""
        self.DupDetect.remove_duplicates(**kwargs)
        df_ = self.cdm.copy()
        header_ = self.DupDetect.result
        df_ = df_[df_.index.isin(header_.index)]
        if overwrite is True:
            self.cdm = df_
        else:
            self.cdm_dups_removed = df_
        return self
