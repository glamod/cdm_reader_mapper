.. mdf_reader documentation master file, created by
   sphinx-quickstart on Fri Apr 16 14:18:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

Marine data file reader-mapper toolbox documentation
----------------------------------------------------

The **cdm_reader_mapper** toolbox is a python3_ tool designed for both

* to read data files compliant with a user specified data model (**mdf_reader**).
* to map observed meteorological variables and its associated metadata from a data model (schema_) to the C3S CDS Common Data Model (CDM_) format or **imodel** as called in this tool (**cdm_mapper**)

It was developed with the initial idea of reading data from the International Comprehensive Ocean-Atmosphere Data Set (ICOADS_) stored in the International Maritime Meteorological Archive (IMMA_) data format. In the meanwhile, it can read data C-RAID_ Copernicus in situ project too.

The tool has been further enhanced to account for any marine meteorological data format, provided that this data meets the following specifications:

-	Data is stored in a human-readable manner: ASCII_.
- Data is stored in a Network Common Data Format: NetCDF_.
-	Data is organized in single line reports (e.g. rows of observations separated by a delimiter like .csv).
-	Reports have a coherent internal structure that can be modelized.
-	Reports are fixed width or field delimited types.
-	Reports can be organized in sections, in which case each section can be of different types (fixed width of delimited).


The **mdf_reader** uses the information provided in a `data model`_ to read meteorological data into a python pandas.DataFrame_, with the column names and data types set according to each data element’s description specified in the data model or **schema**. In addition to reading, the **mdf_reader** validates data elements against the **schema** provided.

The reader allows for basic transformations of the data. This feature includes `basic numeric data decoding`_ (base36, signed_overpunch) and numeric data conversion (scale and offset).

Several data models have been added to the tool including both the IMMA and the C-RAID schema.

.. note:: **Data from other data models than those already available can be read, providing that this data meets the basic specifications listed above. A data model can be built externally and fed into the tool.**

After reading the data into a pandas.DataFrame, the **cdm_mapper** will map observed variables and its associated metadata from a data model or models combination to the standardized C3S CDS Common Data Model (CDM_) format.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   About <readme>
   tool-set-up
   tool-overview_reader
   tool-overview_mapper
   tool-overview_metmetpy
   getting-started
   data-models
   how-to-build-a-data-model
   how_to_read_csv
   how-to-register-a-new-data-model-mapping.rst
   cdm-tables-mapping-files-and-descriptors.rst
   example_notebooks/mdf_reader_test_overview.ipynb
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

|logo_c3s| |logo_NOC| |logo_ICOADS| |logo_copernicus|

.. include:: hyperlinks.rst

.. include:: ../README.rst
    :start-after: hyperlinks
