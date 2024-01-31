====================================================
Common Data Model reader and mapper: ``cdm`` toolbox
====================================================

+----------------------------+-----------------------------------------------------+
| Versions                   | |pypi| |versions|                                   |
+                            +                                                     +
|                            | |tag| |release|                                     |
+----------------------------+-----------------------------------------------------+
| Documentation and Support  | |docs|                                              |
+----------------------------+-----------------------------------------------------+
| Open Source                | |license| |zenodo|                                  |
+----------------------------+-----------------------------------------------------+
| Coding Standards           | |black| |ruff| |pre-commit| |fossa|                 |
+----------------------------+-----------------------------------------------------+
| Development Status         | |status| |build| |coveralls|                        |
+----------------------------+-----------------------------------------------------+
| Funding                    | |funding|                                           |
+----------------------------+-----------------------------------------------------+

The ``cdm`` toolbox is a python_ tool designed for both:

* to read data files compliant with a user specified `data model`_
* map observed variables and its associated metadata from a `data
model`_or models combination to the `C3S CDS Common Data Model`_ (CDM) format

It was developed to read the IMMA_ (International Maritime Meteorological Archive) data format, but it has been enhanced to account for meteorological data formats in the case of:

* Data that is stored in a human-readable manner: “ASCII” format.
* Data that is organized in single line reports
* Reports that have a coherent internal structure and can be modelised.
* Reports that have a fixed width or field delimited types
* Reports that can be organized in sections, in which case each section can be of different types (fixed width of delimited)

Installation
------------

You can install the package directly with pip:

.. code-block:: console

     pip install cdm

If you want to contribute, I recommend cloning the repository and installing the package in development mode, e.g.

.. code-block:: console

    git clone https://github.com/glamod/cdm_reader_mapper
    cd cdm_reader_mapper
    pip install -e .

This will install the package but you can still edit it and you don't need the package in your :code:`PYTHONPATH`


Run a test
----------

Read imma data with the `cdm.read()` and copy the data attributes:

.. code-block:: console

    import cdm

    data = cdm.tests.read_imma1_buoys_nosupp()

    imma_data = cdm.read(filepath, data_model = 'imma1',sections = ['core','c1','c98'])

    data_raw = imma_data.data.copy()

    attributes = imma_data.attrs.copy()

Map this data to a CDM build for the same deck (in this case deck 704: US Marine Metereological Journal collection of data):

.. code-block:: console

    name_of_model = 'icoads_r3000_d704'

    cdm_dict = cdm.map_model(
                        name_of_model,
                        data_raw,
                        attributes,
                        cdm_subset = None,
                        log_level = 'DEBUG',
    )


For more details on how to use the ``reader`` tool see the following `jupyter notebooks`_.
For more details on how to use the ``mapper`` tool see the following `jupyter notebook`_.

For a detailed guide on how to build a cdm and write the output of the `cdm.map_model()` function in ascii see the `user guide`_.

.. hyperlinks

.. _C3S CDS Common Data Model: https://git.noc.ac.uk/brecinosrivas/cdm-mapper/-/blob/master/docs/cdm_latest.pdf

.. _data model: https://cds.climate.copernicus.eu/toolbox/doc/how-to/15_how_to_understand_the_common_data_model/15_how_to_understand_the_common_data_model.html

.. _IMMA: https://icoads.noaa.gov/e-doc/imma/R3.0-imma1.pdf

.. _jupyter notebooks: https://github.com/glamod/cdm_reader_mapper/tree/main/docs/example_notebooks

.. _python: https://www.python.org

.. |build| image:: https://github.com/glamod/cdm_reader_mapper/actions/workflows/ci.yml/badge.svg
        :target: https://github.com/glamod/cdm_reader_mapper/actions/workflows/ci.yml
        :alt: Build Status

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :target: https://github.com/psf/black
        :alt: Python Black

.. |coveralls| image:: https://codecov.io/gh/glamod/cdm_reader_mapper/branch/main/graph/badge.svg
	      :target: https://codecov.io/gh/glamod/cdm_reader_mapper
	      :alt: Coveralls

.. |docs| image:: https://readthedocs.org/projects/cdm_reader_mapper/badge/?version=latest
        :target: https://cdm-reader-mapper.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. |fossa| image:: https://app.fossa.com/api/projects/git%2Bgithub.com%2Fglamod%2Fcdm_reader_mapper.svg?type=shield
        :target: https://app.fossa.com/projects/git%2Bgithub.com%2Fglamod%2Fcdm_reader_mapper?ref=badge_shield
        :alt: FOSSA

.. |funding| image:: https://img.shields.io/badge/Powered%20by-Copernicus-blue.svg
        :target: https://climate.copernicus.eu/
        :alt: Funding

.. |license| image:: https://img.shields.io/github/license/glamod/cdm_reader_mapper.svg
        :target: https://github.com/glamod/cdm_reader_mapper/blob/main/LICENSE
        :alt: License

.. |pre-commit| image:: https://results.pre-commit.ci/badge/github/glamod/cdm_reader_mapper/main.svg
        :target: https://results.pre-commit.ci/latest/github/glamod/cdm_reader_mapper/main
        :alt: pre-commit.ci status

.. |pypi| image:: https://img.shields.io/pypi/v/cdm_reader_mapper.svg
        :target: https://pypi.python.org/pypi/cdm_reader_mapper
        :alt: Python Package Index Build

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
        :target: https://github.com/astral-sh/ruff
        :alt: Ruff

.. |status| image:: https://www.repostatus.org/badges/latest/wip.svg
        :target: https://www.repostatus.org/#wip
        :alt: Project Status: WIP: Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.

.. |release| image:: https://img.shields.io/github/v/release/glamod/cdm_reader_mapper.svg
        :target: https://github.com/glamod/cdm_reader_mapper/releases
        :alt: Release

.. |tag| image:: https://img.shields.io/github/v/tag/glamod/cdm_reader_mapper.svg
        :target: https://github.com/glamod/cdm_reader_mapper/tags
        :alt: Tag

.. |versions| image:: https://img.shields.io/pypi/pyversions/cdm_reader_mapper.svg
        :target: https://pypi.python.org/pypi/cdm_reader_mapper
        :alt: Supported Python Versions

.. |zenodo| image:: https://img.shields.io/badge/zenodo-package_or_version_not_found-red
        :target: https://zenodo.org/cdm_reader_mapper
 	      :alt: DOI
