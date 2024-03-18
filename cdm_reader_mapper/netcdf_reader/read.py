"""Common data model (CDM) netcdf reader."""

from __future__ import annotations

from xarray import open_mfdataset

def read(
  source,
  data_mode=None,
  data_mode_path=None,
  **kwargs,
):
  ds = open_mfdataset(source)
  return ds