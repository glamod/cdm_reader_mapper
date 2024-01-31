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

   from cdm.test_data import test_069_701 as test_data

   filepath = test_data.source
   data_model = test_data.data_model

   data = cdm.read(filepath, data_model=data_model)

or simplify the command by passing `test_data`:

.. code-block:: console

  data = cdm.read(**test_data)

2. Read subsection of an IMMA file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also read subsections from the IMMA test file:

.. code-block:: console

   imma_data = cdm.read(filepath, data_model=data_model, sections = ["core", "c1", "c98"])

3. Map this data to a CDM build for the same deck
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case deck 704: US Marine Meteorological Journal collection of data code:

.. code-block:: console

    name_of_model = 'icoads_r3000_d704'

    cdm_dict = cdm.map_model(
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

    cdm.cdm_to_ascii(
            cdm_dict,
            delimiter = '|',
            extension = 'psv',
            null_label = 'null',
            out_dir = None,
            suffix = None,
            prefix = None,
            log_level = 'INFO',
    )

For more details and an overview of the tool check out the following python notebook:

- example_notebooks/CDM_mapper_example_deck704.ipynb

f. Run **cdm** as an command-line interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also run the **cdm** toolbox as a command-line interface. To call the function from a terminal type:

   cdm <your-file-path> --data_model imma1 --out_path <yout-output-path>

For more details how to run the command-line interface please call the helper function:

.. code-block:: console

   cdm -h
