"""Common Data Model (CDM) class."""

from __future__ import annotations

from cdm_reader_mapper.cdm_mapper.mapper import map_model
from cdm_reader_mapper.cdm_mapper.table_writer import cdm_to_ascii
from cdm_reader_mapper.duplicates.duplicates import duplicate_check
from cdm_reader_mapper.metmetpy.datetime.correct import correct as correct_datetime
from cdm_reader_mapper.metmetpy.datetime.validate import validate as validate_datetime
from cdm_reader_mapper.metmetpy.platform_type.correct import correct as correct_pt
from cdm_reader_mapper.metmetpy.station_id.validate import validate as validate_id
from cdm_reader_mapper.operations import inspect, replace, select


class CDM:
    """Class for mapping to the CDM and manipulating the data.

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

    def __init__(self, MDFFileReader=None, cdm_tables=None):
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

    def __len__(self):
        """Length of ``data``."""
        return inspect.get_length(self.data)

    def select_true(self, **kwargs):
        """Select valid values from ``data`` via ``mask``."""
        self.data = select.select_true(self.data, self.mask, **kwargs)
        self.columns = self.data.columns
        return self

    def select_from_list(self, selection, **kwargs):
        """Select columns of ``data`` from list of column names."""
        self.data = select.select_from_list(self.data, selection, **kwargs)
        self.columns = self.data.columns
        return self

    def select_from_index(self, index, **kwargs):
        """Select columns of ``data`` from list of column names."""
        self.data = select.select_from_index(self.data, index, **kwargs)
        self.columns = self.data.columns
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
        cdm_to_ascii(self.cdm, **kwargs)

    def duplicate_check(self, **kwargs):
        """Duplicate check."""
        self.DupDetect = duplicate_check(self.cdm["header"], **kwargs)
        return self

    def flag_duplicates(self, overwrite=True, **kwargs):
        """Flag detected duplicates in ``cdm``."""
        self.DupDetect.flag_duplicates(**kwargs)
        if overwrite is True:
            self.cdm = self.DupDetect.result
        else:
            self.cdm_dups_flagged = self.DupDetect.result
        return self

    def get_duplicates(self, **kwargs):
        """Get duplicate matches in ``cdm``."""
        self.DupDetect.get_duplicates(**kwargs)

    def remove_duplicates(self, overwrite=True, **kwargs):
        """Remove detected duplicates in ``cdm``."""
        self.DupDetect.remove_duplicates(**kwargs)
        if overwrite is True:
            self.cdm = self.DupDetect.result
        else:
            self.cdm_dups_removed = self.DupDetect.result
        return self
