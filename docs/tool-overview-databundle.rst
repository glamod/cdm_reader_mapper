.. currentmodule:: cdm_reader_mapper

.. _tool-overview-databundle:

Overview over the :py:class:`cdm_reader_mapper.DataBundle` class
================================================================

Reading original meteorological/marine data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After reading meteorogical/marine data like ICOADS or C-RAID with the :py:class:`cdm_reader_mapper.read_mdf`, the function returns a so-called :py:class:`cdm_reader_mapper.DataBundle`. As input data a string representing the path to the original data and the name of the data model (``imodel``) is required. The original data is stored as :py:attr:`DataBundle.data`. Next to the data there is a validation mask, called :py:attr:`DataBundle.mask`. This mask validates the input data against the input data model scheme. For more information see chapter :ref:`data-models`.

.. code-block:: console

    from cdm_reader_mapper import read_mdf, test_data

    data_path = test_data.test_icoads_r300_d714.source
    imodel="icoads_r300_d714"

    db = read_mdf(source=data_path, imdel=imodel)

    #Original MDF data
    db.data

    #Validation mask
    db.mask

Validate ``data``
^^^^^^^^^^^^^^^^^

After reading the data, the method functions :py:func:`DataBundle.validate_datetime` validates date time information in ``data``:

.. code-block:: console

    val_dt = db.validate_datetime()

Another validation method is to validate ``data`` against station id names with :py:func:`DataBundle.validate_id`:

.. code-block:: console

    val_id = db.validate_id()

Manipulate ``data`` and select subsets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more details how to manipulate :py:class:`cdm_reader_mapper.DataBundle` see :ref:`manipulation`.
For more details how to select subsets of :py:class:`cdm_reader_mapper.DataBundle` see and :ref:`selection`.

Map ``data`` to the CDM_
^^^^^^^^^^^^^^^^^^^^^^^^

Now the meteorological data can be maqpped to the Common Data Model (CDM_) using the method function :py:func:`DataBundle.map_model`:

.. code-block:: console

    db.map_model()

    cdm_tables = db.tables

The mapped data will be stored as a class attribute called :py:attr:`DataBundel.tables`.
For more information how the mapping is working, please see :ref:`tool-overview_mapper` and/or :ref:`how-to-register-a-new-data-model-mapping`.

CDM_ correction
^^^^^^^^^^^^^^^

After mapping to the CDM format, in some cases, it is desired that the final CDM set of tables is composed of a combination of different data models/sources. Based on the IMMA1 reprocessing experience so far. This can be the case of adding data elements from a different data source (like adding WMO PUB 47 metadata). It is recommended to map both things separately and then make the appropriate replacements/additions based on the corresponding CDM element matching (i.e. ``primary_station_id``).

.. note:: Correcting data in the CDM format is only necessary for ICOADS data.

**cdm_reader_mapper.DataBundle** provides two functions for correcting data in the CDM format:

1. :py:func:`DataBundle.correct_pt`
2. :py:func:`DataBundle.correct_datetime`

The first function applies ICOADS deck specific platform ID corrections to the data, the second one ICOADS deck specific datetime corrections.

.. code-block:: console

    db.correct_pt()

    db.correct_datetime()

:ref:`dupdetect`
^^^^^^^^^^^^^^^^

After mapping to the CDM format it is useful to check if :py:attr:`DataBundle.tables` contains any duplicates. The duplicate checker included in the ``cdm_reader_mapper`` toolbos is based on python record linkage toolkit RecordLinkage_.

The first step is to call the method function :py:func:`DataBundle.duplicate_check`. This function scans :py:attr:`DataBundle.tables` for any duplicates and stores the result to :py:attr:`DataBundle.DupDetect`.

.. code-block:: console

    db.duplicate_check()

Afterwards their are two options how to deal with the detected duplicates:

1. :py:func:`DataBundle.flag_duplicates`
2. :py:func:`DataBundle.remove_duplicates`

The first function flags the detected duplicates. For more information about the flags see `CDM code tables for duplicate_status`_ and `CDM code tables for report_quality`_. The second function removes the detected duplicates.

.. include:: hyperlinks.rst
