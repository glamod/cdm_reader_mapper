.. _tool-overview-databundle:

Overview over the :class:`cdm_reader_mapper.DataBundle` class
=============================================================

Reading original meteorological/marine data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After reading meteorogical/marine data like ICOADS or C-RAID with the :func:`cdm_reader_mapper.read_mdf`, the function returns a so-called :class:`cdm_reader_mapper.DataBundle`. As input data a string representing the path to the original data and the name of the data model (``imodel``) is required. The original data is stored as :attr:`cdm_reader_mapper.DataBundle.data`. Next to the data there is a validation mask, called :attr:`cdm_reader_mapper.DataBundle.mask`. This mask validates the input data against the input data model scheme. For more information see chapter :ref:`data-models`.

.. code-block:: console

    from cdm_reader_mapper import read_mdf, test_data

    data_path = test_data.test_icoads_r300_d714.source
    imodel="icoads_r300_d714"

    db = read_mdf(source=data_path, imodel=imodel)

    #Original MDF data
    db.data

    #Validation mask
    db.mask

Validate :attr:`cdm_reader_mapper.DataBundle.data`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After reading the data, the method functions :func:`cdm_reader_mapper.DataBundle.validate_datetime` validates date time information in :attr:`cdm_reader_mapper.DataBundle.data`:

.. code-block:: console

    val_dt = db.validate_datetime()

Another validation method is to validate :attr:`cdm_reader_mapper.DataBundle.data` against station id names with :func:`cdm_reader_mapper.DataBundle.validate_id`:

.. code-block:: console

    val_id = db.validate_id()

Correct :attr:`cdm_reader_mapper.DataBundle.data`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After reading the data, in some cases, it is desired that the final CDM set of tables is composed of a combination of different data models/sources. Based on the IMMA1 reprocessing experience so far. This can be the case of adding data elements from a different data source (like adding WMO PUB 47 metadata). It is recommended to map both things separately and then make the appropriate replacements/additions based on the corresponding CDM element matching (i.e. ``primary_station_id``).

.. note:: Correcting data in the CDM format is only necessary for ICOADS data.

**cdm_reader_mapper.DataBundle** provides two functions for correcting data in the CDM format:

1. :func:`.DataBundle.correct_pt`
2. :func:`.DataBundle.correct_datetime`

The first function applies ICOADS deck specific platform ID corrections to the data, the second one ICOADS deck specific datetime corrections.

.. code-block:: console

    db_cor_pt = db.correct_pt()

    db_cor_dt = db.correct_datetime()

Manipulate :attr:`cdm_reader_mapper.DataBundle.data` and select subsets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more details how to manipulate a :class:`cdm_reader_mapper.DataBundle` or select subsets of it see :ref:`databundle`.

Map :attr:`cdm_reader_mapper.DataBundle.data` to the CDM_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now the meteorological data can be maqpped to the Common Data Model (CDM_) using the method function :func:`.DataBundle.map_model`:

.. code-block:: console

    db_cdm = db.map_model()

    cdm_tables = db_cdm.data

.. note:: Set ``inplace`` to True to overwrite :attr:`cdm_reader_mapper.DataBundle.data`:

.. code-block:: console

   db.map_model(inplace=True)

   cdm_tables = db.data

For more information how the mapping is working, please see :ref:`tool-overview-mapper` and/or :ref:`how-to-register-a-new-data-model-mapping`.

.. include:: hyperlinks.rst
