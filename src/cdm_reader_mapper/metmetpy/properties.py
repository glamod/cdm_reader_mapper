"""
metmetpy properties.

Created on Wed Jul 10 09:18:41 2019

@author: iregon
"""

_base = "cdm_reader_mapper.metmetpy"

metadata_datamodels = {}

metadata_datamodels["deck"] = {}
metadata_datamodels["deck"]["icoads"] = ("c1", "DCK")
metadata_datamodels["deck"]["gdac"] = "717"

metadata_datamodels["source"] = {}
metadata_datamodels["source"]["icoads"] = ("c1", "SID")
metadata_datamodels["source"]["gdac"] = "005"

metadata_datamodels["platform"] = {}
metadata_datamodels["platform"]["icoads"] = ("c1", "PT")
metadata_datamodels["platform"]["gdac"] = "OP"
metadata_datamodels["platform"]["cdm"] = ("header", "platform_type")

metadata_datamodels["id"] = {}
metadata_datamodels["id"]["icoads"] = ("core", "ID")
metadata_datamodels["id"]["gdac"] = "ID"
metadata_datamodels["id"]["craid"] = ("drifter_characteristics", "DRIFTER_WMO_NUMBER")
metadata_datamodels["id"]["cdm"] = ("header", "primary_station_id")

metadata_datamodels["year"] = {}
metadata_datamodels["year"]["icoads"] = ("core", "YR")
metadata_datamodels["year"]["gdac"] = "YR"

metadata_datamodels["month"] = {}
metadata_datamodels["month"]["icoads"] = ("core", "MO")
metadata_datamodels["month"]["gdac"] = "MO"

metadata_datamodels["day"] = {}
metadata_datamodels["day"]["icoads"] = ("core", "DY")
metadata_datamodels["day"]["gdac"] = "DY"

metadata_datamodels["hour"] = {}
metadata_datamodels["hour"]["icoads"] = ("core", "HR")
metadata_datamodels["hour"]["gdac"] = "GG"
