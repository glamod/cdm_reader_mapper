"""
metmetpy properties.

Created on Wed Jul 10 09:18:41 2019

@author: iregon
"""

_base = "cdm_reader_mapper.metmetpy"

metadata_datamodels = {}

metadata_datamodels["deck"] = {}
metadata_datamodels["deck"]["imma1"] = ("c1", "DCK")
metadata_datamodels["deck"]["immt"] = "717"

metadata_datamodels["source"] = {}
metadata_datamodels["source"]["imma1"] = ("c1", "SID")
metadata_datamodels["source"]["immt"] = "005"

metadata_datamodels["platform"] = {}
metadata_datamodels["platform"]["imma1"] = ("c1", "PT")
metadata_datamodels["platform"]["cdm"] = ("header", "platform_type")
metadata_datamodels["platform"]["immt"] = "OP"
metadata_datamodels["platform"]["cdm"] = ("header", "platform_type")

metadata_datamodels["id"] = {}
metadata_datamodels["id"]["imma1"] = ("core", "ID")
metadata_datamodels["id"]["cdm"] = ("header", "primary_station_id")
metadata_datamodels["id"]["immt"] = "ID"
metadata_datamodels["id"]["cdm"] = ("header", "primary_station_id")

metadata_datamodels["year"] = {}
metadata_datamodels["year"]["imma1"] = ("core", "YR")
metadata_datamodels["year"]["immt"] = "YR"

metadata_datamodels["month"] = {}
metadata_datamodels["month"]["imma1"] = ("core", "MO")
metadata_datamodels["month"]["immt"] = "MO"

metadata_datamodels["day"] = {}
metadata_datamodels["day"]["imma1"] = ("core", "DY")
metadata_datamodels["day"]["immt"] = "DY"

metadata_datamodels["hour"] = {}
metadata_datamodels["hour"]["imma1"] = ("core", "HR")
metadata_datamodels["hour"]["immt"] = "GG"

metadata_datamodels["datetime"] = {}
metadata_datamodels["datetime"]["cdm"] = ("header", "report_timestamp")

metadata_datamodels["lon"] = {}
metadata_datamodels["lon"]["imma1"] = ("core", "LON")

metadata_datamodels["lat"] = {}
metadata_datamodels["lat"]["imma1"] = ("core", "LAT")

metadata_datamodels["id"] = {}
metadata_datamodels["id"]["imma1"] = ("core", "ID")

metadata_datamodels["uid"] = {}
metadata_datamodels["uid"]["imma1"] = ("c98", "UID")
