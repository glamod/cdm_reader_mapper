"""Common data model (CDM) netcdf reader."""

from __future__ import annotations

from xarray import open_mfdataset


class NetCDFFileReader:
    def __init__(self, source):
        self.source = source

    def read(self):
        ds = open_mfdataset(self.source)
        self.data = ds.to_dataframe().reset_index()
        self.attrs = ds.attrs
        self.mask = self.data.copy()
        self.dtypes = ds.dtypes
        return self


def read(
    source,
    data_mode=None,
    data_mode_path=None,
    **kwargs,
):
    return NetCDFFileReader(source).read()
