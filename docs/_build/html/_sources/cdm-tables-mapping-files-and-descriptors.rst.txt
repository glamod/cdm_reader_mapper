.. _cdm-tables-mapping-files-and-descriptors:

========================================
CDM tables mapping files and descriptors
========================================

The following section details the mapping sequence that the ``cdm_mapper`` tool follows to map meteorological data to a CDM element.

We will use **part** of the ``header.json`` python dictionary from the ``icoads_r300`` IMMA1 model to explain how we map an element. In the table below we explain all **elements** attributes and/or descriptors, that are needed in each python dictionary or ``.json`` file, for a successful mapping of the input meteorological data.

Below we see content from a ``header.json`` file::

    {
        "report_id": {
            "sections": "c98",
            "elements": "UID",
            "transform": "string_add",
            "kwargs":{"prepend":"ICOADS-30","separator":"-"}
        },
        "application_area": {
            "default": [1,7,10,11]
        },
        "observing_programme": {
            "sections": "c1",
            "elements": "PT",
            "transform": "observing_programme"
        },
        "report_type": {
            "default": 0
        },
        "platform_type": {
            "sections": "c1",
            "elements": "PT",
            "code_table": "platform_type"
        },
        "platform_sub_type": {
            "sections": "c1",
            "elements": "PT",
            "code_table": "platform_sub_type"
        },
        "location_accuracy": {
            "sections": "core",
            "elements": ["LI","LAT"],
            "transform": "location_accuracy",
            "decimal_places": 0
        },
        "station_speed": {
            "sections": "core",
            "elements": ["YR","VS"],
            "code_table": "ship_speed_ms",
            "decimal_places":1
        },
        "source_id": {
            "sections":["c1","c1","core","core"],
            "elements": ["SID","DCK","YR","MO"],
            "transform": "string_join_add",
            "kwargs":{"prepend":"ICOADS-3-0-0T","separator":"-","zfill_col":[0,3],"zfill":[3,2]}
        },
        "source_record_id": {
            "sections": "c98",
            "elements": "UID"
        }
    }

Descriptors
===========

=============================  ===============================================================================
 **Descriptor variable name**   **Function**
-----------------------------  -------------------------------------------------------------------------------
 ``elements``                   | String or list of strings with the element name (s) to map in the CDM table.
                                | e.g. ``report_id`` information is store in the ``imma1`` **schema**
                                | as ``UID``, this will be the variable name assigned to the ``element``
                                | attribute of ``report_id``.
 ``sections``                   | String or list of strings with the section name(s) from which the element(s)
                                | to be map will come from.
                                | - Use a single string to define a unique section if all the elements are
                                | located in the same section, e.g. ``location_accuracy``: the variables
                                | ``["LI","LAT"]`` come from a single section ``core`` in the IMMA1 model.
                                | - Use a list of strings to declare variables that come from multiple
                                |   sections and elements. e.g. ``source_id``
                                | - Always respect the order of the sections in the original **schema**.
 ``default``                    | Assigns a default value to the CDM element.
 ``fill_value``                 | Value to assign for missing data (NA/NaN). Datetime objects not supported.
 ``transform``                  | Name of the function to be used to perform the mapping of a specific element.
                                | This function must be defined in the ``mapping_functions`` class of the
                                | ``imodel.py`` module in order to be access by the mapper tool.
 ``kwars``                      | Keyword arguments of a transform function if any.
                                | Type dictionary with the format: {``keyword``:``value``,...,}
 ``code_table``                 | Code table name in the imodel mapping library needed to perform the mapping
                                | a particular element. Type: string.
 ``decimal_places``             | Number of decimal places to keep when printing an element.
                                | Type: integer
                                | value, a function name used to estimate this figure.
                                | Such function should be defined in the same way as the transform function
                                | but these cannot take keyword arguments. ``decimal_places = 0`` for integer
                                | elements defined as numeric in CDM or the element will be printed with
                                | default number of decimal places.
=============================  ===============================================================================

Mapping sequence
================
The mapper parses the mapping file element by element and takes the following steps:

a. Clean imodel data
    Remove any missing ``elements`` from the imodel. This preliminary step makes the definition of mapping functions easier, as no NaN handling needs to be added to the functions and integer fields casted to float by NA/NaN presence is reverted.

b. Map CDM element in the following order:
        1.	If ``transform``: eval function and apply with elements and|or ``kwargs`` as appropriate
        2.	Else if ``code_table``: map imodel elements using the defined ``code_table``
        3.	Else if ``elements``: assign imodel elements to CDM element
        4.	Else if ``value``: assign value to CDM element

c. **Fill CDM element NA/NaN values using default if defined**

d. **Define the number of decimal places in the CDM element attributes, so this gets pass to the table writer if ``decimal_places`` is provided**

Defining mapping functions
==========================

In the file ``imodel.py`` the user can define any function to **transform** any element in the data model. The python file needs to be accompanied with ``__init__.py`` file so all the functions written in ``imodel.py`` can be imported by the **cdm_mapper** toolbox.

.. note:: Remember that any new python dependency that you ``import`` the top of your ``imodel.py`` must be installed also in your python environment.

The **cdm_mapper** follows a set of rules that need to be taken into account when it comes to adding functions to the ``imodel.py`` script.

- The **cdm-mapper** only parses elements to the transforming function (e.g. Year, day or hour) or ``code_table`` mapping (e.g. platform_subtype), where none of the elements to be map (e.g. Year, day, hour or platform_subtype) have missing values.

- The output of all functions in ``imodel.py`` must respect the element type defined in the imodel mapper.

.. _cdm-code-tables:

Code tables
===========

Elements defined in the **imodel.json** files (e.g. elements inside ``header.json``) with the attribute ``code_table`` have an specific "key" that links the element variable to its corresponding numerical code defined in the C3S CDM. Code tables contain the ``key:value`` pairs and are stored as individual ``.json`` files in the ``lib/mappings/imodel/code_tables`` subdirectory.

The content of a code table translating ``platform_sub_type`` information into the appropriate CDM syntax's (``platform_sub_type.json``) can be seen in text below::

     {
         "7": 69
     }

This code table is part of the ``icoads_r300`` data model included in this tool.

The following range of code table structures are currently supported:

- Simple code tables: code tables with a list of ``key:value`` pairs.
- Nested code tables: code tables with multiple (2 or more) keys mapping to a value ``-> key(1):…:key(n):value.``
- Range-keyed code tables: code tables (simple or multi-keyed) where one or more keys is a (integer) range of values.

For more information on code tables and their structure check out the `cdm_reader_mapper tool - code tables <https://cdm-reader-mapper.readthedocs.io/en/mdf_reader/data-models.html#code-tables>`_ information.

The code table above, is use by the ``icoads_r300`` imodel to map ``platform_sub_type`` information to the C3s CDM format, this is done in  the following section of the ``header.json`` file::

    "platform_sub_type": {
            "sections": "c1",
            "elements": "PT",
            "code_table": "platform_sub_type"
    }

The "key" in this case, will be the value read from the ICOADS section ``c1`` and element ``PT``, for key values equal to 7 a 69 code will be assigned.

Code tables can be also used for simple transformations of the elements, depending on the medata data to map. e.g. The case of deck 701, where we expand ship names to the ships original full name. We do this by reading meta data information from the ``c99`` ICOADS supplemental data attachment. The imodel for deck 701 provides a code table to transform the names into the ships original name format recorded in the original ship logbook::

     "station_name": {
            "sections": "core",
            "elements": "ID",
            "code_table": "ship_names"
     }
