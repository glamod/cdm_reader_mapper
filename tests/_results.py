"""cdm_reader_mapper testing suite result files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cdm_reader_mapper.cdm_mapper import read_tables

_base = Path(__file__).parent

for correction_file in (_base / "results").glob("2022-02.txt.gz"):
    break

correction_df = pd.read_csv(
    correction_file,
    delimiter="|",
    dtype="object",
    header=None,
    usecols=[0, 1, 2],
    names=["report_id", "primary_station_id", "primary_station_id.isChange"],
    quotechar=None,
    quoting=3,
)

table_df = read_tables((_base / "results"), "test", cdm_subset=["header"])
table_df.set_index("report_id", inplace=True, drop=False)
