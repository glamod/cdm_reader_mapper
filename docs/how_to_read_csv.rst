.. mdf_reader documentation master file, created by
   sphinx-quickstart on Fri Apr 16 14:18:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

.. _how-to-read-a-simple-csv:

=============================
How to read a simple csv file
=============================

1. Have a clear understanding of the structure of the data file you wish to read in, namely delimiter used, the variables contained per column, the units, acceptable limits for each variable etc.

2. As described in section ref:`_how-to-build-a-data-model`, create a valid directory tree where the model you will create (**(mymodel)**) should be saved. This can be placed either in the mdf_reader library directory (``~/mdf_reader/data_models/library/``) or in a user defined path which will be provided into the mdf_reader at a later step.

3. Create the **schema** file under ``../library/mymodel/mymodel.json`` or ``path/mymodel/mymodel.json``.
For example for a data file without sections, stored in comma delimeted csv format, that contains 8 columns (year, month, day, longitude, latitude, wind speed, sea surface temperature and sea level pressure respectively) like the following:

+------+------+------+---------+--------+--------+---------+---------+
| YR   | MO   | DY   | LON     | LAT    | W      | SST     | SLP     |
+------+------+------+---------+--------+--------+---------+---------+
| 2007 | 2    | 3    | 255.708 | 9.829  | 5.902  | 301.323 | 1008.57 |
| 2007 | 2    | 4    | 255.682 | 12.691 | 5.222  | 300.971 | 1009.14 |
| 2007 | 7    | 2    | 242.764 | 32.707 | 1.3    | 296.556 | 1010.37 |
| 2007 | 7    | 3    | 242.158 | 32.943 | 4.792  | 294.558 | 1009.92 |
| 2007 | 7    | 4    | 240.329 | 34.218 | 3.739  | 290.405 | 1008.32 |
| 2007 | 7    | 5    | 239.889 | 34.377 | 5.629  | 288.273 | 1007.38 |
| 2007 | 7    | 6    | 240.054 | 34.38  | 4.322  | 289.752 | 1008.54 |
| 2007 | 7    | 7    | 240.011 | 34.229 | 4.717  | 288.624 | 1010.59 |
| 2007 | 7    | 8    | 240.054 | 34.394 | 3.924  | 290.584 | 1008.85 |
| 2007 | 7    | 9    | 241.352 | 33.8   | 2.852  | 293.248 | 1010.51 |
| 2007 | 7    | 10   | 241.499 | 33.804 | 2.308  | 293.03  | 1011.51 |
+------+------+------+---------+--------+--------+---------+---------+

The basic schema would look like this::

       {
         "header":
         {
          "filed_layout":"delimited",
          "delimiter":","
        },
          "elements":
        {
            "YR": {
            "description": "year UTC",
            "column_type": "uint16",
            "valid_max": 2008,
            "valid_min": 2006,
            "units": "year",
            "missing_value":"MSNG"
            },
            "MO": {
            "description": "month UTC",
            "field_length": 2,
            "column_type": "uint8",
            "valid_max": 12,
            "valid_min": 1,
            "units": "month",
            "missing_value":"MSNG"
            },
            "DY": {
            "description": "day UTC",
            "field_length": 2,
            "column_type": "uint8",
            "valid_max": 31,
            "valid_min": 1,
            "units": "day",
            "missing_value":"MSNG"
            },
            "lon": {
            "description":"LON",
            "field_length": 6,
            "column_type": "float32",
            "valid_max": 359.99,
            "valid_min": 0.0,
            "scale": 1,
            "decimal_places": 2,
            "units": "degrees",
            "missing_value":"MSNG"
            },
            "lat": {
            "description": "LAT",
            "field_length": 5,
            "column_type": "float32",
            "valid_max": 90.0,
            "valid_min": -90.0,
            "scale": 1,
            "decimal_places": 2,
            "units": "degrees",
            "missing_value":"MSNG"
            },
            "W": {
            "description": "wind speed",
            "field_length": 4,
            "column_type": "float32",
            "valid_max": 99.9,
            "valid_min": 0.0,
            "scale": 1,
            "decimal_places": 2,
            "units": "metres per second",
            "missing_value":"MSNG"
            },
            "SST": {
            "description": "sea surface temperature",
            "field_length": 5,
            "column_type": "float32",
            "valid_max": 999.9,
            "valid_min": -999.9,
            "scale": 1,
            "decimal_places": 2,
            "units": "degree Kelvin",
            "missing_value":"MSNG"
            },
            "SLP": {
            "description": "sea level pressure",
            "field_length": 6,
            "column_type": "float32",
            "valid_max": 1074.6,
            "valid_min": 870.0,
            "scale": 1,
            "decimal_places": 2,
            "units": "hectopascal",
            "missing_value":"MSNG"
            }
        }
       }

in which the file format information are given in the header and information about the data at each column are given in the ``elements``; details on setting up the element blocks are given in :ref:`schema-element-block`. Note that the elements in the data are parsed in the order they are declared in the schema.

In case an element expects a numeric value but is given letter type input then the data are set to missing. However, if the input is numeric even if it's given as string it is read in.

In case the user would like to skip a column/element, they can use ``ignore`` in the ``elements`` e.g. as::

      "SST": {
      "description": "sea surface temperature",
      "ignore": "True"
      },
