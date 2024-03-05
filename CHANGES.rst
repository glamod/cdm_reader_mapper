
=========
Changelog
=========

0.2.0 (unpublished)
-------------------

* use slugify insted of unidecde for licening reasons
* delete redundatn functions from cdm_reader_mapper.common
* cdm_reader_mapepr: import common (__init__.py)
* remove unused modules from metmetpy
* adding tests to cdm_reader_mapper testing suite
* cdm_reader_mapper.mdf_reader split data_models into code_tables and schema
* move converters and decoders from common to mdf_reader/utils

0.1.0 (2023-01-16)
------------------

* combine ``mdf_reader`` and ``cdm-mapper``
* prepare for pandas>=2.1.0
* make use of pre-commit
* make pypi release
* use ``setuptools_scm`` for automatic updating of version numbers
* optionally: use ``cdm_reader_mapper`` as a command-line interface tool
