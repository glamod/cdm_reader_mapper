"""Duplicate check for test data."""

from cdm_reader_mapper.common.json_dict import open_json_file
from cdm_reader_mapper import DataBundle

import pandas as pd

ifile = "test_in.json"

idict = open_json_file(ifile)

data = pd.DataFrame(idict)
data.columns = [c.lower() for c in data.columns]

db = DataBundle(tables=data)

db.duplicate_check()
df_flagged = db.flag_duplicates(overwrite=False)

df_flagged.to_csv("test_out.json")
