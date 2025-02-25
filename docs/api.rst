.. currentmodule:: cdm_reader_mapper

.. _api:

"""""""""""""
API reference
"""""""""""""

This page provides an auto-generated summary of the ``cdm_reader_mapper`` API.

.. _read_data:

Read data from disk
===================

.. autosummary::
   :toctree: generated/

   read
   read_mdf
   read_data
   read_tables

.. _databundle:

DataBundle
==========

.. autosummary::
   :toctree: generated/

   DataBundle

.. _information

DataBundle's method functions
------------------------------

Information
^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   DataBundle.unique

.. _manipulation:

Manipulation
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   DataBundle.add

   DataBundle.copy

   DataBundle.replace_columns

   DataBundle.stack_h

   DataBundle.stack_v

.. _selection:

Selection
^^^^^^^^^

.. autosummary::
   :toctree: generated/

   DataBundle.select_from_index

   DataBundle.select_from_list

   DataBundle.select_true

.. _validation:

Validation
^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   DataBundle.validate_datetime

   DataBundle.validate_id

.. _cdm_tables:


Map data to CDM tables
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   DataBundle.map_model

.. _cdm_correction

Correction
^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   DataBundle.correct_datetime

   DataBundle.correct_pt

.. _duplicate_check:

Duplicate check
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   DataBundle.duplicate_check

   DataBundle.get_duplicates

   DataBundle.flag_duplicates

   DataBundle.remove_duplicates

Write data on disk
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   DataBundle.write

.. _properties

DataBundle's property attributes
--------------------------------

.. autosummary::
   :toctree: generated/

   DataBundle.data

   DataBundle.columns

   DataBundle.dtypes

   DataBundle.parse_dates

   DataBundle.mask

   DataBundle.imodel

.. _useful_func:

Useful functions
================

.. autosummary::
   :toctree: generated/

   correct_datetime
   correct_pt
   unique
   replace_columns
   validate_datetime
   validate_id
   select_true
   select_from_index
   select_from_list
   map_model
   duplicate_check
   write

.. _dupdetect:

DupDetect
=========

.. autosummary::
   :toctree: generated/

   DupDetect

Check Duplicates
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   DupDetect.get_duplicates

   DupDetect.flag_duplicates

   DupDetect.remove_duplicates
