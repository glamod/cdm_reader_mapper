.. cdm documentation master file, created by
   sphinx-quickstart on Fri Apr 16 14:18:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

.. _getting-started:

Getting started
===============

1. Read an IMMA file
~~~~~~~~~~~~~~~~~~~~

You can test the tool very easy by using a sample data set that comes with the repository. For this you need to run the following code:

.. code-block:: console

   from cdm_reader_mapper import read_mdf
   from cdm_reader_mapper.test_data import test_icoads_r300_d704 as test_data

   filepath = test_data.source
   imodel = "icoads_r300_d704"

   db = read_mdf(filepath, imodel=imodel)


2. Read subsection of an IMMA file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also read subsections from the IMMA test file:

.. code-block:: console

   db_sub = read_mdf(filepath, imodel=imodel, sections = ["core", "c1", "c98"])

3. Map this data to a CDM build for the same deck
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case deck 704: US Marine Meteorological Journal collection of data code:

.. code-block:: console

    db.map_model()

    cdm_tables = db.tables.copy()

4. Detect duplicated observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detect and flag duplicated observations without overwriting the original CDM tables:

.. code-block:: console

    db.duplicate_check()
    db.flag_duplicates(overwrite=False)

    flagged_tables = db.tables_dups_flagged.copy()

    db.remove_duplicates(overwrite=False)

    removed_tables = db.tables_dups_removed.copy()

5. Write the output
~~~~~~~~~~~~~~~~~~~
This writes the output to an ascii file with a pipe delimited format using the following function:

.. code-block:: console

    db.write_tables()
