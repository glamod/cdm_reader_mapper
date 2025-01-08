.. currentmodule:: cdm_reader_mapper

.. _api:

"""""""""""""
API reference
"""""""""""""

This page provides an auto-generated summary of the ``cdm_reader_mapper`` API.

Read data from disk
===================

.. autosummary::
   :toctree: generated/

   read_mdf
   read_tables

DataBundle
==========

.. autosummary::
   :toctree: generated/

   DataBundle

DataBundle's method functions
------------------------------

Information
^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   DataBundle.unique

Manipulation
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   DataBundle.add

   DataBundle.copy

   DataBundle.replace_columns

   DataBundle.stack_h

   DataBundle.stack_v

Selection
^^^^^^^^^

.. autosummary::
   :toctree: generated/

   DataBundle.select_from_index

   DataBundle.select_from_list

   DataBundle.select_true

Correction
^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   DataBundle.correct_datetime

   DataBundle.correct_pt

Validation
^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   DataBundle.validate_datetime

   DataBundle.validate_id

CDM tables
^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   DataBundle.map_model

   DataBundle.write_tables

Duplicate check
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   DataBundle.duplicate_check

   DataBundle.get_duplicates

   DataBundle.flag_duplicates

   DataBundle.remove_duplicates


DataBundle's property attributes
--------------------------------

MDF data
^^^^^^^^

.. autosummary::
   :toctree: generated/

   DataBundle.data

   DataBundle.columns

   DataBundle.dtypes

   DataBundle.attrs

   DataBundle.mask

   DataBundle.imodel

   DataBundle.selected

   DataBundle.deselected

CDM tables
^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   DataBundle.tables

   DataBundle.tables_dups_flagged

   DataBundle.tables_dups_removed


Useful functions
================

.. autosummary::
   :toctree: generated/

   duplicate_check
   map_model
   write_tables
   correct_datetime
   correct_pt
   unique
   replace_columns
   validate_datetime
   validate_id
   select_true
   select_from_index
   select_from_list

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
