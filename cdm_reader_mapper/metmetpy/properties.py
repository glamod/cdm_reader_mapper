"""
metmetpy properties.

Created on Wed Jul 10 09:18:41 2019

@author: iregon
"""

_base = "cdm_reader_mapper.metmetpy"

metadata_datamodels = {}

metadata_datamodels["deck"] = {}
metadata_datamodels["deck"]["icoads"] = ("c1", "DCK")
metadata_datamodels["deck"]["gcc"] = "717"

metadata_datamodels["source"] = {}
metadata_datamodels["source"]["icoads"] = ("c1", "SID")
metadata_datamodels["source"]["gcc"] = "005"

metadata_datamodels["platform"] = {}
metadata_datamodels["platform"]["icoads"] = ("c1", "PT")
metadata_datamodels["platform"]["gcc"] = "OP"
metadata_datamodels["platform"]["cdm"] = ("header", "platform_type")

metadata_datamodels["id"] = {}
metadata_datamodels["id"]["icoads"] = ("core", "ID")
metadata_datamodels["id"]["gcc"] = "ID"
metadata_datamodels["id"]["craid"] = ("drifter_characteristics", "DRIFTER_WMO_NUMBER")
metadata_datamodels["id"]["cdm"] = ("header", "primary_station_id")

metadata_datamodels["year"] = {}
metadata_datamodels["year"]["icoads"] = ("core", "YR")
metadata_datamodels["year"]["gcc"] = "YR"

metadata_datamodels["month"] = {}
metadata_datamodels["month"]["icoads"] = ("core", "MO")
metadata_datamodels["month"]["gcc"] = "MO"

metadata_datamodels["day"] = {}
metadata_datamodels["day"]["icoads"] = ("core", "DY")
metadata_datamodels["day"]["gcc"] = "DY"

metadata_datamodels["hour"] = {}
metadata_datamodels["hour"]["icoads"] = ("core", "HR")
metadata_datamodels["hour"]["gcc"] = "GG"

metadata_datamodels["datetime"] = {}
metadata_datamodels["datetime"]["cdm"] = ("header", "report_timestamp")

metadata_datamodels["lon"] = {}
metadata_datamodels["lon"]["icoads"] = ("core", "LON")
metadata_datamodels["lon"]["gcc"] = "LON"
metadata_datamodels["lon"]["craid"] = ("drifter_measurements", "LONGITUDE")

metadata_datamodels["lat"] = {}
metadata_datamodels["lat"]["icoads"] = ("core", "LAT")
metadata_datamodels["lat"]["gcc"] = "LAT"
metadata_datamodels["lat"]["craid"] = ("drifter_measurements", "LATITUDE")

metadata_datamodels["uid"] = {}
metadata_datamodels["uid"]["icoads"] = ("c98", "UID")
