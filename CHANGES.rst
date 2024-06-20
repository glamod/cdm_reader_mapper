
=========
Changelog
=========

0.4.0 (unpublished)
-------------------
Contributors to this version: Ludwig Lierhammer (:user:`ludwiglierhammer`) and Joseph Siddons (:user:`jtsiddons`)

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* ``common.getting_files.load_file``: optionally, load data within data reference syntax (:pull:`41`)
* ``common.getting_files.load_file``: optionally, clear cache directory (:pull:`45`)

Breaking changes
^^^^^^^^^^^^^^^^
* set chunksize from 10000 to 3 in testing suite (:pull:`35`)
* ``cdm_mapper``: read header column ``location_quality`` from ``(c1, LZ)`` and set fill_value to ``0`` (:issue:`36`, :pull:`37`)
* ``cdm_mapper``: set default value of header column ``report_quality`` to ``2`` (:issue:`36`, :pull:`37`)

Internal changes
^^^^^^^^^^^^^^^^
* replace deprecated ``datetime.datetime.utcnow()`` with ``datetime.datetime.now(datetime.UTC)`` (see: https://github.com/python/cpython/issues/103857) (:pull:`39`, :pull:`43`)
* make use of ``cdm-testdata`` release ``v2024.06.07`` https://github.com/glamod/cdm-testdata/releases/tag/v2024.06.07 (:issue:`44`, :pull:`45`)
* migration to ``setup-micromamba``: https://github.com/mamba-org/provision-with-micromamba#migration-to-setup-micromamba (:pull:`48`)
* update actions to use Node.js 20: https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#example-using-versioned-actions (:pull:`48`)

Bug fixes
^^^^^^^^^
* indexing working with user-given chunksize (:pull:`35`)
* fix reading of custom schema in ``mdf_reader.read`` (:pull:`40`)
* ensure ``format`` schema field for delimited files is passed correctly, avoiding ``"...Please specify either format or field_layout in your header schema..."`` error (:pull:`40`)


0.3.0 (2024-05-17)
------------------
Contributors to this version: Ludwig Lierhammer (:user:`ludwiglierhammer`) and Joseph Siddons (:user:`jtsiddons`)

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* ``mdf_reader``: read C-RAID netCDF buoy data (:issue:`13`, :pull:`24`, :pull:`28`)
* adding both GCC IMMT and C-RAID netCDF data to ``test_data`` (:pull:`24`, :pull:`28`)
* ``cdm_mapper``: adding C-RAID mapping and code tables (:issue:`13`, :pull:`28`)
* ``cdm_mapper``: add ``load_tables`` to ``__init.py__`` (:pull:`32`)

Breaking changes
^^^^^^^^^^^^^^^^
* adding tests for IMMT and C-Raid data (:issue:`26`, :pull:`24`, :pull:`28`)
* ``cdm_mapper.map_model``: drop dulicated lines in pd.DataFrame before writing CDM table on disk (:pull:`28`)
* add pyarrow (see: https://github.com/pandas-dev/pandas/issues/54466) to requirements
* solving pyarrow-snappy issue (see: openforcefield/openff-nagl#106) (:issue:`33`, :pull:`28`, :pull:`34`)

Internal changes
^^^^^^^^^^^^^^^^
* do not diferentiate between tuple and single column names (:pull:`24`)
* ``metmetpy``: Do not raise erros if ``validate_datetime``, ``correct_datetime``, ``correct_pt`` and/or ``validate_id`` do not find any entries (:pull:`24`)
* get rid of warnings (:issue:`9`, :pull:`27`)
* adding python 3.12 to testing suite (:pull:`29`)
* set time out for testing suite to 10 minutes (:pull:`29`)

Bug fixes
^^^^^^^^^^
* ``cdm_mapper``: set debugging logger into if statement (:pull:`24`)
* ``cdm_mapper``: do not use code table ``qc_flag`` with ``report_id`` (:pull:`24`)
* ``metmetpy``: fixing ICOADS 30000 NRT functions for ``pandas>=2.2.0`` (:pull:`31`)
* ``cdm_mapper.read_tables``: if table not available return empty ``pd.DataFrame`` (:pull:`32`)


0.2.0 (2024-03-15)
------------------
Contributors to this version: Ludwig Lierhammer (:user:`ludwiglierhammer`) and Joseph Siddons (:user:`jtsiddons`)

Breaking changes
^^^^^^^^^^^^^^^^
* move converters and decoders from ``common`` to ``mdf_reader/utils`` (:pull:`3`)
* delete redundant functions from ``cdm_reader_mapper.common``
* ``cdm_reader_mapper``: import common (__init__.py)
* remove unused modules from ``metmetpy``
* ``cdm_reader_mapper.mdf_reader`` split data_models into code_tables and schema
* logging: Allow for use of log file (:pull:`6`)
* cannot use as command-line tool anymore (:pull:`22`)
* outsource input and result data to `cdm-testdata` (:issue:`16`, :pull:`21`)

Internal changes
^^^^^^^^^^^^^^^^
* adding tests to cdm_reader_mapper testing suite (:issue:`12`, :pull:`2`, :pull:`20`, :pull:`22`)
* adding testing result data (:pull:`4`)
* use slugify insted of unidecde for licening reasons
* remove pip install instruction (:pull:`2`)
* ``HISTORY.rst`` has been renamed ``CHANGES.rst``, to follow `xclim`-like conventions (:pull:`7`).
* speed up mapping functions with `swifter` (:pull:`4`)
* ``mdf_reader``: adding auxiliary functions and classes (:pull:`4`)
* ``mdf_reader``: read tables line-by-line (:pull:`20`)

Bug fixes
^^^^^^^^^
* Fixed an issue with missing ``conda`` dependencies in the ``cdm_reader_mapper`` documentation (:pull:`14`)


0.1.0 (2024-01-16)
------------------
Contributors to this version: Ludwig Lierhammer (:user:`ludwiglierhammer`)

Breaking changes
^^^^^^^^^^^^^^^^
* combine `mdf_reader <https://github.com/glamod/mdf_reader/tree/backup>`_ , `cdm-mapper <https://github.com/glamod/cdm-mapper>`_, `pandas_operations <https://github.com/glamod/pandas_operations>`_ and `metmetpy <https://github.com/glamod/metmetpy>`_
* optionally: use ``cdm_reader_mapper`` as a command-line interface tool

Internal changes
^^^^^^^^^^^^^^^^
* make use of ``pre-commit``
* prepare for ``pandas>=2.1.0``
* use ``setuptools_scm`` for automatic updating of version numbers
