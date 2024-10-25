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

   from cdm_reader_mapper import mdf_reader
   from cdm_reader_mapper.test_data import test_069_701 as test_data

   filepath = test_data.source
   data_model = test_data.data_model

   data = mdf_reader.read(filepath, data_model=data_model)

or simplify the command by passing `test_data`:

.. code-block:: console

  data = mdf_reader.read(**test_data)

2. Read subsection of an IMMA file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also read subsections from the IMMA test file:

.. code-block:: console

   imma_data = mfd_reader.read(filepath, data_model=data_model, sections = ["core", "c1", "c98"])

3. Map this data to a CDM build for the same deck
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case deck 704: US Marine Meteorological Journal collection of data code:

.. code-block:: console

    from cdm_reader_mapper import cdm_mapper

    name_of_model = 'icoads_r3000_d704'

    cdm_dict = cdm_mapper.map_model(
                  name_of_model,
                  data_raw.data,
                  attributes,
                  cdm_subset = None,
                  log_level = 'DEBUG',
    )

4. Write the output
~~~~~~~~~~~~~~~~~~~
This writes the output to an ascii file with a pipe delimited format using the following function:

.. code-block:: console

    from cdm_reader_mapper import cdm_mapper

    cdm_mapper.cdm_to_ascii(
            cdm_dict,
    )
