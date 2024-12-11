"""Common Data Model (CDM) class."""

from cdm_reader_mapper.operations import (select, inspect, replace)
from cdm_reader_mapper.metmetpy.datetime import correct as correct_datetime
from cdm_reader_mapper.metmetpy.datetime import validate as validate_datetime
from cdm_reader_mapper.metmetpy.platform_type import correct as correct_pt
from cdm_reader_mapper.metmetpy.station_id import validate as validate_id
from cdm_reader_mapper.cdm_mapper import map_model
from cdm_reader_mapper.cdm_mapper import write_tables


def class CDM:
    """Class for mapping to the CDM and manipulating the data.

    Attributes
    ----------
    data: pd.DataFrame or pd.io.parsers.TextFileReader
        MDF data
    mask: pd.DataFrame or pd.io.parsers.TextFileReader
        MDF validation mask
    attrs: dict
        Data elements attributes.
    imodel: str
        Name of the CDM input model.
        
    """
    
    def __init__(self, data, mask, attrs, imodel):
        self.data = data
        self.mask = mask
        self.attrs = attrs
        self.imodel = imodel
        
    def __len__(self):
        return inspect.get_length(self.data)
        
    def select_true(self, **kwargs):
        self.data = select.select_true(self.data, self.mask, **kwargs)
        return self
        
    def select_from_list(self, selection, **kwargs):
        self.data = select.select_from_list(self.data, selection, **kwargs)
        return self
        
    def select_from_index(self, index, **kwargs):
        self.data = select.select_from_index(self.data, index, **kwargs)
        return self
        
    def unique(self, **kwargs):
        return inspect.count_by_cat(self.data, **kwargs)
        
    def replace_columns(self, df_corr, **kwargs)
        self.data = replace.replace_columns(df_l=self.data, df_r=df_corr, **kwargs)
        
    self.correct_datetime(self):
        self.data = correct_datetime(self.data, self.imodel)
        return self
        
    self.validate_datetime(self):
        return validate_datetime(self.data, self.imodel)
        
    self.correct_pt(self):
        self.data = correct_pt(self.data, self.imodel)
        return self
        
    self.validate_id(self, **kwargs):
        return validate_id(self.data, self.imodel, **kwargs)
        
    self.map_model(self, **kwargs):
        self.cdm = map_model(self.data, self.imodel, **kwargs)
        return self
        
    self.write_tables(self, **kwargs):
        write_tables(self.cdm, **kwargs)
        
    

