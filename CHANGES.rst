
=========
Changelog
=========

0.2.0 (unpublished)
-------------------
Contributors to this version: Ludwig Lierhammer (:user: `ludwiglierhammer`) and Joseph Siddons (:user: `jtsiddons`)

Breaking changes
^^^^^^^^^^^^^^^^
* move converters and decoders from `common` to `mdf_reader/utils` (:pull: 3)
* delete redundant functions from `cdm_reader_mapper.common`
* `cdm_reader_mapper`: import common (__init__.py)
* remove unused modules from `metmetpy`
* `cdm_reader_mapper.mdf_reader` split data_models into code_tables and schema
* logging: Allow for use of log file (:pull: 6)

Internal changes
^^^^^^^^^^^^^^^^
* adding tests to cdm_reader_mapper testing suite (:pull: 2)
* use slugify insted of unidecde for licening reasons
* remove pip install instruction (:pull: 5)
* `HISTORY.rst` has been renamed `CHANGES.rst`, to follow `xclim`-like conventions (:pull: 7).

0.1.0 (2023-01-16)
------------------
Contributors to this version: Ludwig Lierhammer (:user: `ludwiglierhammer`)

Breaking changes
^^^^^^^^^^^^^^^^
* combine `mdf_reader <https://github.com/glamod/mdf_reader/tree/backup>`_ , `cdm-mapper <https://github.com/glamod/cdm-mapper>`_, `pandas_operations <https://github.com/glamod/pandas_operations>`_ and `metmetpy <https://github.com/glamod/metmetpy>`_
* optionally: use ``cdm_reader_mapper`` as a command-line interface tool

Internal changes
^^^^^^^^^^^^^^^^
* make use of pre-commit
* prepare for pandas>=2.1.0
* use ``setuptools_scm`` for automatic updating of version numbers
