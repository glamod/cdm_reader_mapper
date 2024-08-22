.. mdf_reader documentation master file, created by
   sphinx-quickstart on Fri Apr 16 14:18:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

Marine data file reader-mapper toolbox documentation
----------------------------------------------------

The **cdm_reader_mapper** toolbox is a python3_ tool designed for both

* to read data files compliant with a user specified data model (**mdf_reader**).
* to map observed meteorological variables and its associated metadata from a data model (schema_) to the C3S CDS Common Data Model (CDM_) format or **imodel** as called in this tool (**cdm_mapper**)

It was developed with the initial idea of reading data from the International Comprehensive Ocean-Atmosphere Data Set (ICOADS_) stored in the International Maritime Meteorological Archive (IMMA_) data format.

The tool has been further enhanced to account for any marine meteorological data format, provided that this data meets the following specifications:

-	Data is stored in a human-readable manner: ASCII_.
- Data is stored in a Network Common Data Foramt: NetCDF_.
-	Data is organized in single line reports (e.g. rows of observations separated by a delimiter like .csv).
-	Reports have a coherent internal structure that can be modelized.
-	Reports are fixed width or field delimited types.
-	Reports can be organized in sections, in which case each section can be of different types (fixed width of delimited).


The **mdf_reader** uses the information provided in a `data model`_ to read meteorological data into a python pandas.DataFrame_, with the column names and data types set according to each data element’s description specified in the data model or **schema**. In addition to reading, the **mdf_reader** validates data elements against the **schema** provided.

This tool outputs a python object with the following attributes:

1.	A pandas.DataFrame_ (DF) with the data values.
2.	A `boolean pandas`_ DF with the data validation mask.
3.	A dictionary_ with a simplified version of the input data model.

The reader allows for basic transformations of the data. This feature includes `basic numeric data decoding`_ (base36, signed_overpunch) and numeric data conversion (scale and offset).

Several data models have been added to the tool including the IMMA schema.

.. note:: **Data from other data models than those already available can be read, providing that this data meets the basic specifications listed above. A data model can be built externally and fed into the tool.**

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   About <readme>
   tool-set-up
   tool-overview_reader
   tool-overview_mapper
   getting-started
   data-models
   how-to-build-a-data-model
   how_to_read_csv
   how-to-register-a-new-data-model-mapping.rst
   cdm-tables-mapping-files-and-descriptors.rst
   example_notebooks/CLIWOC_datamodel.ipynb
   example_notebooks/CDM_mapper_example_deck704.ipynb
   contributing
   authors
   api
   changes


About
-----

:Version: |tag| |release|

:Citation: |zenodo|

:License: |license|

:Authors:
   David Berry, Irene Perez Gonzalez, Beatriz Recinos, Andreas Wernecke and Ludwig Lierhammer

|logo_c3s| |logo_NOC| |logo_ICOADS|

.. include:: hyperlinks.rst

.. include:: ../README.rst
    :start-after: hyperlinks
