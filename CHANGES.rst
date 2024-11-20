
=========
Changelog
=========

1.0.2 (2024-11-13)
------------------
Contributors to this version: Ludwig Lierhammer (:user:`ludwiglierhammer`)

Announcements
^^^^^^^^^^^^^
* New PyPi Classifiers:

  * Development Status :: 5 - Production/Stable
  * Development Status :: Intended Audience :: Science/Research
  * License :: OSI Approved :: Apache Software License
  * Operating System :: OS Independent

1.0.1 (2024-11-08)
------------------
Contributors to this version: Ludwig Lierhammer (:user:`ludwiglierhammer`)

Announcements
^^^^^^^^^^^^^
* set package version to v1.0.1

1.0.0 (2024-11-08)
------------------
Contributors to this version: Ludwig Lierhammer (:user:`ludwiglierhammer`)

Announcements
^^^^^^^^^^^^^
* Final version used for GLAMOD marine processing release 7.0

Bug fixes
^^^^^^^^^
* ``cdm_mapper``: Two reports that describe each other as best duplicates are not flagged as duplicates (DupDetect) (:pull:`149`)
* ``cdm_mapper``: Reindex only if null values available (DupDetect) (:pull:`153`)

0.4.3 (2024-10-23)
------------------
Contributors to this version: Ludwig Lierhammer (:user:`ludwiglierhammer`)

Announcements
^^^^^^^^^^^^^
* First release on pypi (:issue:`17`)
* First release on zenodo (:issue:`18`)

0.4.2 (2024-10-23)
------------------
Contributors to this version: Ludwig Lierhammer (:user:`ludwiglierhammer`)

Announcements
^^^^^^^^^^^^^
* Testing first release on pypi (:issue:`17`)
* Testing first release on zenodo (:issue:`18`)

0.4.1 (2024-10-23)
------------------
Contributors to this version: Ludwig Lierhammer (:user:`ludwiglierhammer`)

Announcements
^^^^^^^^^^^^^
* Testing first release on pypi (:issue:`17`)
* Testing first release on zenodo (:issue:`18`)

0.4.0 (2024-10-23)
-------------------
Contributors to this version: Ludwig Lierhammer (:user:`ludwiglierhammer`) and Joseph Siddons (:user:`jtsiddons`)

Announcements
^^^^^^^^^^^^^
* Now under Apache v2.0 license (:pull:`69`)

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* ``common.getting_files.load_file``: optionally, load data within data reference syntax (:pull:`41`)
* ``common.getting_files.load_file``: optionally, clear cache directory (:pull:`45`)
* reworked readthedocs documentation for gathered ``cdm_reader_mapper`` package (:issue:`19`, :pull:`83`)
* ``mdf_reader``: new validation function for datetime objects (:pull:`89`)
* ``mdf_reader``: select time period with new arguments ``year_init`` ad ``year_end`` (:pull:`98`)
* ``cdm_mapper``: duplicate check using ``recordlinkage`` (:pull:`81`)
* ``mdf_reader.read``: optionally, set left and right time bounds (``year_init`` and ``year_end``) (:issue:`11`, :pull:`97`)
* ``mdf_reader.read``: optionally, set both external schema and code table paths and external schema file (:issue:`47`, :pull:`111`)
* ``cdm_mapper``: Change both columns history and report_quality during duplicate_check (:pull:`112`)
* ``cdm_mapper``: optionally, set column names to be ignored while duplicate check (:pull:`115`)
* ``cdm_mapper``: optionally, set offset values for duplicate_check (:pull:`119`)
* ``cdm_mapper``: optionally, set column entries to be ignored while duplicate_check (:pull:`119`)
* ``cdm_mapper``: add both column names ``station_speed`` and ``station_course`` to default duplicate check list (:pull:`119`)
* ``cdm_mapper``: optionally, re-index data in ascending order according to the number of nulls in each row (:pull:`119`)

Breaking changes
^^^^^^^^^^^^^^^^
* set chunksize from 10000 to 3 in testing suite (:pull:`35`)
* ``cdm_mapper``: read header column ``location_quality`` from ``(c1, LZ)`` and set fill_value to ``0`` (:issue:`36`, :pull:`37`)
* ``cdm_mapper``: set default value of header column ``report_quality`` to ``2`` (:issue:`36`, :pull:`37`)
* reading C-RAID data: set decimal places according to input file data precision (:pull:`60`)
* always convert data types of both ``int`` and ``float`` in schemas into default data types (:issue:`59`, :pull:`60`)
* ``cdm_mapper.map_model``: call function without input parameter ``data_atts`` (:issue:`66`, :pull:`67`)
* ``decimal_places`` information is moved from ``mdf_reader.schema`` to ``cdm_mapper.tables``; ``decimal_places`` in  user-given schemas will be ignored (:issue:`66`, :pull:`67`)
* ``cdm_mapper`` does not need any attribute information from ``mdf_reader`` (:issue:`66`, :pull:`67`)
* ``cdm_mapper``: map ICOADS wind direction data (``361`` -> ``0``; ``362`` -> ``np.nan``) (:pull:`82`)
* ``cdm_mapper``: set fill_value to ``UNKNOWN`` for C-RAID's ``primary_station_id`` (:pull:`93`)
* ``cdm_mapper``: map C-RAID quality flags to CDM quality flags (:pull:`94`)
* ``mdf_reader``: summarize schema and code tables (:issue:`11`, :pull:`97`)
* ``mdf_reader``: rename ``c_raid`` to ``craid``, ``gcc_immt`` to ``gcc`` and ``imma1`` to ``icoads`` (:issue:`11`, :pull:`97`)
* ``cdm_mapper``: summarize tables and code tables (:issue:`11`, :pull:`97`)
* ``cdm_mapper``: rename ``c_raid`` to ``craid`` and ``gcc_mapping`` to ``gcc`` (:issue:`11`, :pull:`97`)
* ``metmetpy``: rename ``immt`` to ``gcc`` and ``imma`` to ``icoads`` (:issue:`11`, :pull:`97`)
* ``cdm_mapper.map_model``: use standardized imodel_name as <data_model>_<release>_<deck> (e.g. icoads_r300_d701) (:issue:`11`, :pull:`97`)
* ``mdf_reader.read``: use standardized imodel_name as <data_model>_<release>_<deck> (e.g. icoads_r300_d701) (:issue:`11`, :pull:`97`)
* ``mdf_reader``: (``core``, ``VS``) set column_type to ``key`` for all ICOADS decks (:issue:`11`, :pull:`97`)
* ``cdm_mapper``: rename pub47_noc mapping to pub47 (:pull:`102`)
* Note by each function call: rename ``data_model`` into ``imodel`` e.g. imodel=icoads_r300_d704 (:pull:`103`)
* ``cdm_mapper.map_model``: call with (data, imodel=imodel) (:pull:`103`)
* ``mdf_reader.read``: call with (source, imodel=imodel) (:pull:`103`)
* Re-order arguments to ``mdf_reader.validate``, and create argument for ``ext_table_path`` (:pull:`105`)
* ``operations``: delete corrections module (:pull:`104`)
* ``cdm_mapper``: duplicate check is available for header table only (:pull:`115`)
* ``cdm_mapper``: set report_quality to ``1`` for bad duplicates (:pull:`115`)
* ``cdm_mapper``: set default primary_station_id to ``4`` for C-RAID mapping (:issue:`117`, :pull:`121`)
* renamed some element names in ``icoads_r300_d730`` schema for consistency (``InsName`` to ``InstName``, ``InsPlace`` to ``InstPlace``, ``InsLand`` to ``InstLand``, ``No_data_entry`` to ``NumArchiveSet``) (:pull:`110`)

Internal changes
^^^^^^^^^^^^^^^^
* replace deprecated ``datetime.datetime.utcnow()`` with ``datetime.datetime.now(datetime.UTC)`` (see: https://github.com/python/cpython/issues/103857) (:pull:`39`, :pull:`43`)
* make use of ``cdm-testdata`` release ``v2024.06.07`` https://github.com/glamod/cdm-testdata/releases/tag/v2024.06.07 (:issue:`44`, :pull:`45`)
* migration to ``setup-micromamba``: https://github.com/mamba-org/provision-with-micromamba#migration-to-setup-micromamba (:pull:`48`)
* update actions to use Node.js 20: https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#example-using-versioned-actions (:pull:`48`)
* ``mdf_reader.auxiliary.utils``: rename variable for missing values to ``missing_values`` (:pull:`56`)
* add ``pre-commit`` hooks: ``codespell``, ``pylint`` and ``vulture`` (:pull:`56`)
* use ``pytest.parametrize`` for testing suite (:pull:`61`)
* use ``ast.literal_eval`` instead of ``eval`` (:pull:`64`)
* remove unused code tables in ``mdf_reader`` (:issue:`10`, :pull:`65`)
* ``cdm_mapper.mappings``: use ``datetime`` to convert ``float`` into hours and minutes.
* add FOSSA license scanning to github workflows (:pull:`80`)
* add ``cdm_reader_mapper`` author list including ORCID iD's (:pull:`38`, :pull:`49`)
* ``mdf_reader``: replace empty strings with missing values (:pull:`89`)
* ``metmetpy``: use function ``overwrite_data`` in all platform type correction functions (:pull:`89`)
* rename ``data_model`` into ``imodel`` (:pull:`103`)
* implement assertion tests for module operations (:pull:`104`)
* ``cdm_mapper``: put settings for duplicate check in _duplicate_settings (:pull:`119`)
* ``cdm_mapper``: use pandas.apply function instead of for loops in duplicate_check (:pull:`119`)
* adding some more duplicate checks to testing suite (:pull:`119`)
* ``cdm_mapper``: re-adding conserderation of indexes of nan values during transformation (:pull:`125`)

Bug fixes
^^^^^^^^^
* indexing working with user-given chunksize (:pull:`35`)
* fix reading of custom schema in ``mdf_reader.read`` (:pull:`40`)
* ensure ``format`` schema field for delimited files is passed correctly, avoiding ``"...Please specify either format or field_layout in your header schema..."`` error (:pull:`40`)
* there is a loss of data precision due to data type conversion. Hence, use default data types of both ``int`` and ``float`` (:issue:`59`, :pull:`60`)
* reading C-RAID data: adjust datetime formats to read dates into ``MDFFileReader`` (:pull:`60`)
* ensure external code tables are used when using an external schema in ``mdf_reader.read`` (:pull:`105`)
* update readme and example Jupyter notebooks to :pull:`103` (:pull:`110`)
* restructure ``CLIWOC_datamodel`` Jupyter notebook to add an example of data model construction (:pull:`110`)
* remove ``create_data_model.ipynb`` example Jupyter notebook (:pull:`110`)


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
* ``cdm_mapper.map_model``: drop duplicated lines in pd.DataFrame before writing CDM table on disk (:pull:`28`)
* add pyarrow (see: https://github.com/pandas-dev/pandas/issues/54466) to requirements
* solving pyarrow-snappy issue (see: openforcefield/openff-nagl#106) (:issue:`33`, :pull:`28`, :pull:`34`)

Internal changes
^^^^^^^^^^^^^^^^
* do not differentiate between tuple and single column names (:pull:`24`)
* ``metmetpy``: Do not raise errors if ``validate_datetime``, ``correct_datetime``, ``correct_pt`` and/or ``validate_id`` do not find any entries (:pull:`24`)
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
* use slugify instead of unidecde for licening reasons
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
