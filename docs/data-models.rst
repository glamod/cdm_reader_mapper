.. cdm documentation master file, created by
   sphinx-quickstart on Fri Apr 16 14:18:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

.. _data-models:

===========
Data Models
===========

Schema
======

The schema file gathers a collection of descriptors that enables the **cdm** toolbox to access and extract meaningful units of information for each element.

Valid schemas files are json files that the tool accesses and stores internally as dictionaries. The basename of the schema file must be the same as the data model directory and its extension ``.json``

.. figure:: _static/images/schema.png
    :width: 45%

    Data model directory

There are two levels of information in the schema:

   1. **General** information on the data format layout, that helps the tool decide which approach to follow in order to access the data content. This information is included in the **header block** at the top of the schema (see figure below).


   2. **Specific** information on the data elements and, optionally, on the sections. In the case that the data model has its report elements organised in one or multiple sections (as shown in the figure below). This information is included in the **elements block** of the schema.

.. figure:: _static/images/new_schema.png
    :width: 80%

    Content inside a ``schema.json`` file.

The **cdm** toolbox supports reading and validation of both internal and external schemas:

- An **internal data model** has its schema registered within the tool. To read and validate data from these models, we only need to pass its reference name to the reader and validation modules, using the argument ``data_model``. A list of the reference names for internally supported data models can be access via the tool's function::

   import cdm
   cdm.properties.supported_data_models()

- An **external data model** is a data format that is unknown to the tool. If the data model meets the specifications for which the tool was built, then a model can be built externally and fed into it for both functions data reading and model validation using the argument ``data_model_path``::

   model_path = 'cdm/data_models/library/imma1_d701'
   data_file_path = 'cdm/data/069-701_1845-04_subset.imma'
   data = cdm.read(data_file_path, data_model_path= model_path)

.. _code-tables:

Code tables
===========

.. figure:: _static/images/elements.png
    :width: 80%

    Element content inside a ``schema.json`` file.

Elements defined in the data model ``schema.json`` with an element attribute ``"column_type": "key"`` are linked to a code table in the data model through a codetable descriptor in the schema (e.g. ``"codetable": "ICOADS.C99.FORM"``). Code tables contain the ``key:value`` pairs and are stored as individual ``.json`` files in the ``data_models/schema/code_tables`` subdirectory.

The content of a code table translating a ship-log report type into its real meaning (``ICOADS.C99.FORM.json``) can be seen in text below::

     {
     " 1": "daily",
     " 2": "reports more than once a day"
     }

This code table is part of the ``imma1_d701`` data model included in this tool.

The following range of code table structures are currently supported:

- Simple code tables: code tables with a list of ``key:value`` pairs.
- Nested code tables: code tables with multiple (2 or more) keys mapping to a value ``-> key(1):…:key(n):value.``
- Range-keyed code tables: code tables (simple or multi-keyed) where one or more keys is a (integer) range of values.

Code tables can be imported as python dictionaries directly using the json package. To be fully read by the tool, however, keys in **range-keyed code tables** need to be expanded and access to all code tables is managed in the application through a **code table manager module**.

The following commands typed in a python console, show how to access code table templates to create new code tables::

      template_names = cdm.code_tables.templates()

To copy a template to edit::

      cdm.code_tables.copy_template(template_name,out_path=file_path)

or::

      cdm.code_tables.copy_template(template_name,out_dir=dir_path)


Common features
---------------
As code tables are stored as ``.json`` files, the json syntax rules must be met when they are generated. See the following `link <https://www.w3schools.com/js/js_json_syntax.asp>`_ to a basic introduction to json syntax.

To create code tables it is important to highlight that:

- String values must be written with double quotes
- Keys must be strings
- Values can be strings, numbers, objects (JSON objects), arrays, booleans (``true|false``) or ``null``.
- Due to the way range keyed tables are parsed, keys cannot have the string ``range_key`` as initial substring (unless they are range keys).

Simple code tables
------------------

Simple code tables are built using a single json object (enclosed in curly braces) with the ``key:value`` pairs separated by commas like the following example for a weather visibility indicator, the file name is ``visibility_ind.json``::

      {
         " ": "Not measured",
         "0": "Measured",
         "1": "Fog present"
      }

Nested code tables
------------------

Nested code tables are included to deal with situations when a coded element's encoding, varies according to an indicator (contained in a different element in the data) or/and changes along time (different code table versions). Instead of storing these tables in separate files, the tool allows to create nested code tables.

The following ``.json`` file example shows a code table with 2 levels of indexing. It is built as a single **json object** in which the values of the ``key:value`` pairs of the outer indexing level are simple code tables, instead of individual values.

Nested table (named: ``visibility.json``) example::

      {
         "0":
             {"90":"<0.05 km",
              "91":"0.05 km",
              "92":"0.2 km",
              "93":"0.5 km",
              "94":"1 km",
              "95":"2 km",
              "96":"4 km",
              "97":"10 km",
              "98":"20 km",
              "99":"50 km or more"},
         "1":
             {"90":"<0.05 km",
              "91":"0.05 km",
              "92":"0.2 km",
              "93":"Fog present, no visibility reported",
              "94":"1 km",
              "95":"2 km",
              "96":"4 km",
              "97":"10 km",
              "98":"20 km",
              "99":"50 km or more"}
      }

This type of nested code table requires an additional ``.keys`` (named: ``visibility.keys``) file with the following format::

      {
         "('core1','VIS')" : ["('core1','VIS I')","('core1','VIS')"]
      }

This **code_table** can be called from the ``schema.json`` by setting the element descriptor ``column_type`` to ``key`` in the following way::

       "VIS": {
                    "description": "Visibility",
                    "field_length": 2,
                    "column_type": "key",
                    "codetable": "visibility"
                }

Note that only the **nested code table** ``visibility`` is called not the .keys, and we do not require the ``.json`` extension.

The data file schema provides the ``element:codetable`` correspondence. However, to map the element to its value in the code table, it is necessary to know the elements in the data file from which the outer keys are derived. Each nested table ``table_name.json`` has a companion ``.json`` file ``table_name.keys`` with a set of ``key:value`` pairs. The key is the actual element the table decodes and the value is a list with the complete set of key elements, from outer to inner.

As a single table can be potentially used to code different data file elements, a key must be provided for every element wishing to be decoded with a nested table (even if it is unique)

Range-keyed code tables
-----------------------

Range-keyed code tables can be any a simple or a nested type of code table. This term will apply if any of its ``key:value`` pairs is a range, like a period of years (1910-1945) or simply an integer interval (1-10).

Instead of building the table repeating each of the ``key:value`` pairs for every value in the range, the corresponding range key pairs are defined as range (init, end [, step]):value in the json file. The code table manager will identify this special type of key and will expand the keys in the dictionary as is read internally.

Range keys rules and use:

   - Only integer ranges are currently supported
   - Parameter step is optional. Defaults to 1.
   - In ranges that apply to a range of years, the keyword yyyy can be used in the place of the end parameter. It will expand the period to the current year.

Example of a Range-key nested table named: ``ICOADS.CO.VS.json`` is shown below::

      {
         "range_key(1750,1967)":
              {
                "0":"0 knots;[0.0,0.0,0.0] ms-1",
                "1":"1-3 knots;[0.51444,1.02888,1.54332] ms-1",
                "2":"4-6 knots;[2.05776,2.5722,3.08664] ms-1",
                "3":"7-9 knots;[3.60108,4.11552,4.62996] ms-1",
                "4":"10-12 knots;[5.1444,5.65884,6.17328] ms-1",
                "5":"13-15 knots;[6.68772,7.20216,7.7166] ms-1",
                "6":"16-18 knots;[8.23104,8.74548,9.25992] ms-1",
                "7":"19-21 knots;[9.77436,10.2888,10.8032] ms-1",
                "8":"22-24 knots;[11.3177,11.8321,12.3466] ms-1",
                "9":"over 24 knots;[12.3466,12.861,null] ms-1"
              },
         "range_key(1968,yyyy)":
              {
                "0":"0 knots;[0.0,0.0,0.0] ms-1",
                "1":"1-5 knots;[0.51444,1.54332,2.5722] ms-1",
                "2":"6-10 knots;[3.08664,4.11552,5.1444] ms-1",
                "3":"11-15 knots;[5.65884,6.68772,7.7166] ms-1",
                "4":"16-20 knots;[8.23104,9.25992,10.2888] ms-1",
                "5":"21-25 knots;[10.8032,11.8321,12.861] ms-1",
                "6":"26-30 knots;[13.3754,14.4043,15.4332] ms-1",
                "7":"31-35 knots;[15.9476,16.9765,18.0054] ms-1",
                "8":"36-40 knots;[18.5198,19.5487,20.5776] ms-1",
                "9":"over 40 knots;[21.092,22.1209,null] ms-1"
              }
      }

As is nested the corresponding ``ICOADS.CO.VS.keys`` file looks as follows::

      {
         "('core','VS')" : ["('core','YR')","('core','VS')"]
      }
