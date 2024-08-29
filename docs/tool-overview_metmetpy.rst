``metmetpy`` overview
=======================

After mapping to the CDM format, in some cases, it is desired that the final CDM set of tables is composed of a combination of different data models/sources. Based on the IMMA1 reprocessing experience so far. This can be the case of adding data elements from a different data source (like adding WMO PUB 47 metadata). It is recommended to map both things separately and then make the appropriate replacements/additions based on the corresponding CDM element matching (i.e. ``primary_station_id``). This part is taken over from **metmetpy**.

.. note:: Correcting data in the CDM format is only necessary for ICOADS data.

**metmetpy** provides two functions for correcting data in the CDM format:

1. ``metmetpy.correct_pt.correct``
2. ``metmetpy.correct_datetime.correct``

The first function applies ICOADS deck specific platform ID corrections to the data, the second one ICOADS deck specific datetime corrections.

.. note:: In further releases **metmetpy** will be included in the **cdm_mapper** module.
