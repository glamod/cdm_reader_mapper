"""Common Data Model (CDM) DataBundle class."""

from __future__ import annotations

from cdm_reader_mapper.cdm_mapper.reader import read_tables
from cdm_reader_mapper.mdf_reader.reader import read_mdf, read_data


def read(
    data=None,
    mode="mdf",
    inp_dir=None,
    imodel=None,
    ext_schema_path=None,
    ext_schema_file=None,
    ext_table_path=None,
    year_init=None,
    year_end=None,
    mask=None,
    info=None,
    col_subset=None,
    cdm_subset=None,
    prefix=None,
    suffix=None,
    extension="psv",
    delimiter="|",
    na_values=None,
    **kwargs,
):
    """Read either original marine-meteorological data or MDF data or CDM tables from disk.

    Parameters
    ----------
    data: str, optional
        The file (including path) to be read.
    mode: str, ["mdf", "data", "tables"]
        Data mode.
        Default: "mdf"
    inp_dir: str, optional
        Path to the input file(s).
        Use only if ``mode`` is "tables" and `data` is None.
    imodel: str, optional
        Name of internally available input data model.
        e.g. icoads_r300_d704
        Use only if ``mode`` is "mdf" or "data".
    ext_schema_path: str, optional
        The path to the external input data model schema file.
        The schema file must have the same name as the directory.
        One of ``imodel`` and ``ext_schema_path`` or ``ext_schema_file`` must be set.
        Use only if ``mode`` is "mdf".
    ext_schema_file: str, optional
        The external input data model schema file.
        One of ``imodel`` and ``ext_schema_path`` or ``ext_schema_file`` must be set.
        Use only if ``mode`` is "mdf".
    ext_table_path: str, optional
        The path to the external input data model code tables.
        Use only if ``mode`` is "mdf".
    year_init: str or int, optional
        Left border of time axis.
        Use only if ``mode`` is "mdf".
    year_end: str or int, optional
        Right border of time axis.
        Use only if ``mode`` is "mdf".
    mask: str, optional
        The validation file (including path) to be read.
        Use only if ``mode`` is "data".
    info: str, optional
        The information file (including path) to be read.
        Use only if ``mode`` is "data".
    col_subset: str, tuple or list, optional
        Specify the section or sections of the file to write.

        - For multiple sections of the tables:
          e.g col_subset = [columns0,...,columnsN]

        - For a single section:
          e.g. list type object col_subset = [columns]

        Column labels could be both string or tuple.
        Use only if ``mode`` is "data" or "tables".
    cdm_subset: str or list, optional
        Specifies a subset of tables or a single table.

        - For multiple subsets of tables:
          This function returns a pandas.DataFrame that is multi-index at
          the columns, with (table-name, field) as column names. Tables are merged via the report_id field.

        - For a single table:
          This function returns a pandas.DataFrame with a simple indexing for the columns.

        Use only if ``mode`` is "tables".
    prefix: str, optional
        Prefix of file name structure: ``<prefix>-<table>-*<suffix>.<extension>``.
        Use only if ``mode`` is "tables".
    suffix: str, optional
        Suffix of file name structure: ``<prefix>-<table>-*<suffix>.<extension>``.
        Use only if ``mode`` is "tables".
    extension: str
        Extension of file name structure: ``<prefix>-<table>-*<suffix>.<extension>``.
        Default: psv
        Use only if ``mode`` is "tables".
    delimiter: str
        Character or regex pattern to treat as the delimiter while reading with pandas.read_csv.
        Default: '|'
        Use only if ``mode`` is "tables".
    na_values: Hashable, Iterable of Hashable or dict of {Hashable: Iterable}, optional
        Additional strings to recognize as Na/NaN while reading input file with pandas.read_csv.
        For more details see: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        Use only if ``mode`` is "tables".

    See Also
    --------
    read_mdf : Read original marine-meteorological data from disk.
    read_data : Read MDF data and validation mask from disk.
    read_tables : Read CDM tables from disk.
    write: Write either MDF data or CDM tables on disk.
    write_data : Write MDF data and validation mask to disk.
    write_tables: Write CDM tables to disk.

    Note
    ----
    If `mode` is "mdf" use :py:func:`read_mdf`.
    If `mode` is "data" use :py:func:`read_data`.
    If `mode` is "tables" use :py:func:`read_tables`.

    """
    if mode == "mdf":
        return read_mdf(
            data,
            imodel=imodel,
            ext_schema_path=ext_schema_path,
            ext_schema_file=ext_schema_file,
            ext_table_path=ext_table_path,
            year_init=year_init,
            year_end=year_end,
            **kwargs,
        )
    elif mode == "data":
        return read_data(
            data,
            mask=mask,
            info=info,
            imodel=imodel,
            col_subset=col_subset,
            **kwargs,
        )
    elif mode == "tables":
        return read_tables(
            data=data,
            inp_dir=inp_dir,
            prefix=prefix,
            suffix=suffix,
            extension=extension,
            cdm_subset=cdm_subset,
            col_subset=col_subset,
            delimiter=delimiter,
            na_values=na_values,
        )
    else:
        raise ValueError(
            f"No valid mode: {mode}. Choose one of ['mdf', 'data', 'tables']"
        )
