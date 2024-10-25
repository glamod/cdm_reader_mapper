.. _how-to-register-a-new-data-model-mapping:

How to register a new data model mapping
========================================

Using ``icoads_r300`` imodel as a reference, a data model (imodel) mapping can be understood in this context as:

- A generic mapping from a defined data model.
    e.g. the IMMA data model (``imodel = icoads_r300``), which in this case maps core ICOADS variables into the CDM format.
- A specific mapping from a generic data model.
    e.g. the case of mapping a specific ICOADS collection, where the ``imodel`` is built for a specific source and deck (``imodel = imma1_sid-dck``).
- A combination of multiple data models in a single CDM mapping.
    e.g. the case of mapping ICOADS.CORE variables and variables stored in the ICOADS supplemental data attachment (c99), this type of model is also source and deck dependent (``imodel = imma1_supp_model_sid-dck``).

This is a summary of the steps needed to add an imodel to the mapping tool:

1. Copy the mapping structure of an already existing mapping (e.g. ``cdm_reader_mapper/cdm_mapper/tables/icoads/r300``) to the same folder and re-name it according to the ``imodel`` that you are going to add.

2. Create a copy for each of the **CDM tables** in your imodel. To access the **CDM tables** templates available in the tool type::

    table_list = cdm_reader_mapper.cdm_mapper.properties.cdm_tables

3. Edit the mapping files (``*.json``) and create the mappings to CDM elements (refer to :ref:`cdm-tables-mapping-files-and-descriptors` for more information), this can be achieved using one of the following approaches:

    - Direct mapping from an imodel element.
    - Direct mapping via ``code_tables`` that can take one or multiple imodel elements.
    - Assignment of a default value.
    - Optionally, add any other mapping functions needed for the new data model to the ``cdm_mapper.mappings`` module:

    a.	imodel elements attributes
    b.	parameterization with input keyword arguments
    c.	one or multiple imodel elements
    d.	transforming function from/to coded elements other than direct mapping with code tables can be defined here (i.e.: imodel key to CDM value or imodel value to CDM key)

4. Create :ref:`cdm-code-tables` to transform variables if these required a specific key to translate the information into the CDM.

.. note:: Click on next for a detail description on steps 3 and 4.

.. note:: For any questions, please leave us a comment on the `issue tracker`_.

.. include:: hyperlinks

.. _issue tracker: https://github.com/glamod/cdm_reader_mapper/issues
