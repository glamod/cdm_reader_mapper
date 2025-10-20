.. module:: cdm_reader_mapper

.. mdf_reader documentation master file, created by
   sphinx-quickstart on Fri Apr 16 14:18:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

Marine ``cdm_reader_mapper`` toolbox documentation
--------------------------------------------------

The **cdm_reader_mapper** toolbox is a python3_ tool designed for

* reading original marine-meteorological data files compliant with a user specified data model (:ref:`data-models`) into a Marine Data Format (MDF) file.
* mapping observed meteorological variables and its associated metadata from a data model (:ref:`data-models`) to the C3S CDS Common Data Model (CDM_) format or **imodel** as called in this tool.
* to detect and flag or remove duplicated observations

It was developed with the initial idea of reading data from the International Comprehensive Ocean-Atmosphere Data Set (ICOADS_) stored in the International Maritime Meteorological Archive (IMMA_) data format. In the meanwhile, it can read data C-RAID_ Copernicus in situ project too.

The tool has been further enhanced to account for any marine meteorological data format, provided that this data meets the following specifications:

-	Data is stored in a human-readable manner: ASCII_.
- Data is stored in a Network Common Data Format: NetCDF_.
-	Data is organized in single line reports (e.g. rows of observations separated by a delimiter like .csv).
-	Reports have a coherent internal structure that can be modelized.
-	Reports are fixed width or field delimited types.
-	Reports can be organized in sections, in which case each section can be of different types (fixed width of delimited).

Several data models have been added to the tool including both the IMMA and the C-RAID schema.

.. note:: **Data from other data models than those already available can be read, providing that this data meets the basic specifications listed above. A data model can be built externally and fed into the tool.**

The **read_mdf** function uses the information provided in a `data model`_ to read meteorological data into a so-called **cdm_reader_mapper.DataBundle** object. This object includes several main attributes:

* `data`: meteorogical data stored as a pandas.DataFrame_, with the column names and data types set according to each data element’s description specified in the data model or **schema**.
* `mask`: mask of `data` that validates data elements against the **schema** provided.

The reader allows for basic transformations of the data. This feature includes `basic numeric data decoding`_ (base36, signed_overpunch) and numeric data conversion (scale and offset).

In addition, the **cdm_reader_mapper.DataBundle** object has several main method functions:

* :py:func:`DataBundle.map_model`: map observed variables and its associated metadata from a data model or models combination to the standardized C3S CDS Common Data Model (CDM_) format.
* :py:func:`DataBundle.duplicate_check`: detect duplicated observations
* :py:func:`DataBundle.flag_duplicates`: flag detected duplicated observations
* :py:func:`DataBundle.remove_duplicates`: remove detected duplicated observations
* :py:func:`DataBundle.write`: save both observational MDF files as a coma-separated list and observational standardized CDM tables as pipe-seperated lists

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   About <readme>
   tool-set-up
   tool-overview-reader
   tool-overview-mapper
   tool-overview-databundle
   getting-started
   data-models
   how-to-build-a-data-model
   how-to-read-csv
   how-to-register-a-new-data-model-mapping
   cdm-tables-mapping-files-and-descriptors
   example_notebooks/read_overview.ipynb
   example_notebooks/CLIWOC_datamodel.ipynb
   example_notebooks/CDM_mapper_example_deck704.ipynb
   contributing
   authors
   api
   changes


About
-----

:Version: |pypi|

:Citation: |zenodo|

:License: |license|

|newline|

|newline|

|tab| |tab| |tab| |tab| |tab| |tab| |tab| |tab| |tab| |tab| |tab| |logo_c3s| |logo_ICOADS| |logo_copernicus|

|tab| |tab| |tab| |tab| |tab| |tab| |tab| |tab| |tab| |tab| |tab| |tab| |tab| |tab| |logo_DWD| |logo_NOC|

.. include:: hyperlinks.rst

.. include:: ../README.rst
    :start-after: hyperlinks

.. |newline| raw:: html

   <br>

.. |tab| raw:: html

   &emsp;  <!-- HTML em-space -->
