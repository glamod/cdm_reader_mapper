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

   read_data

   read_mdf

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

   DataBundle.split_by_boolean_true

   DataBundle.split_by_boolean_false

   DataBundle.split_by_column_entries

   DataBundle.split_by_index

   DataBundle.select_where_all_true

   DataBundle.select_where_all_false

   DataBundle.select_where_entry_isin

   DataBundle.select_where_index_isisn

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

   DataBundle.flag_duplicates

   DataBundle.get_duplicates

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

   DataBundle.columns

   DataBundle.data

   DataBundle.dtypes

   DataBundle.encoding

   DataBundle.imodel

   DataBundle.mask

   DataBundle.mode

   DataBundle.parse_dates

.. _useful_func:

Useful functions
================

.. autosummary::
   :toctree: generated/

   correct_datetime

   correct_pt

   duplicate_check

   map_model

   read

   read_data

   read_mdf

   read_tables

   replace_columns

   split_by_boolean

   split_by_boolea_false

   split_by_boolean_true

   split_by_column_entries

   split_by_index

   unique

   validate_datetime

   validate_id

   write

   write_data

   write_tables

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

   DupDetect.flag_duplicates

   DupDetect.get_duplicates

   DupDetect.remove_duplicates
