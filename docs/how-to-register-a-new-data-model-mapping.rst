.. _how-to-register-a-new-data-model-mapping:

How to register a new data model mapping
========================================

Using ``icoads_r3000`` imodel as a reference, a data model (imodel) mapping can be understood in this context as:

- A generic mapping from a defined data model.
    e.g. the IMMA data model (``imodel = icoads_r3000``), which in this case maps core ICOADS variables into the CDM format.
- A specific mapping from a generic data model.
    e.g. the case of mapping a specific ICOADS collection, where the ``imodel`` is built for a specific source and deck (``imodel = imma1_sid-dck``).
- A combination of multiple data models in a single CDM mapping.
    e.g. the case of mapping ICOADS.CORE variables and variables stored in the ICOADS supplemental data attachment (c99), this type of model is also source and deck dependent (``imodel = imma1_supp_model_sid-dck``).

This is a summary of the steps needed to add an imodel to the mapping tool:

1. Copy the template mapping structure in the mappings library directory (``~/cdm-mapper/lib/mappings/template``) to the same folder and re-name it according to the ``imodel`` that you are going to add:

.. figure:: _static/images/imodel_dir.png
    :width: 100%

    Directory structure of an imodel, showing the ``icoads_r3000`` .imma data model as an example.

        - The ``imodel.py`` module hosts the ``mapping_functions`` class. These the functions used by the tool to map imodel elements to CDM elements (if required). All transform functions have to be defined under this class, so the mapper tool can access them.
        - Additionally, an ``__init__.py`` file needs to be added, so python can recognise the imodel directory as a module and this can be use by the tool.

2. Create a copy of the ``template.json`` file for each of the **CDM tables** in your imodel. To access the **CDM tables** templates available in the tool type::

    table_list = cdm.properties.cdm_tables

   Your imodel should be looking something like the figure above.

3. Edit the mapping files (``*.json``) and create the mappings to CDM elements (refer to :ref:`cdm-tables-mapping-files-and-descriptors` for more information), this can be achieved using one of the following approaches:

    - Direct mapping from an imodel element.
    - Direct mapping via ``code_tables`` that can take one or multiple imodel elements.
    - Assignment of a default value.
    - Any other mapping including any combination of the following can be achieved using functions defined in the ``imodel.py`` module:

    a.	imodel elements attributes
    b.	parameterization with input keyword arguments
    c.	one or multiple imodel elements
    d.	transforming function from/to coded elements other than direct mapping with code tables can be defined here (i.e.: imodel key to CDM value or imodel value to CDM key)

4. Create :ref:`cdm-code-tables` to transform variables if these required a specific key to translate the information into the CDM.

.. note:: Click on next for a detail description on steps 3 and 4.

.. warning:: After finishing your new ``imodel``, don't forget to deactivate and then re-activate your python virtualenv or to reset your jupyter-notebook kernel, so the cdm tool recognises your new mapper. Alternatively you can pass to the main ``cdm.map_model`` function the directory path where you have stored your cdm imodel mapper (see API Reference for more information).
