r"""
Import data as a pandas TextParser object.

Created on Fri Jan 10 13:17:43 2020
FUNCTION TO PREPARE SOURCE DATA TO WHAT GET_SECTIONS() EXPECTS:

   AN ITERABLE WITH DATAFRAMES

INPUT IS NOW ONLY A FILE PATH. COULD OPTIONALLY GET OTHER TYPE OBJECTS...
OUTPUT IS AN ITERABLE, DEPENDING ON CHUNKSIZE BEING SET:

   - a single dataframe in a list
   - a pd.io.parsers.textfilereader

WITH BASICALLY 1 RECORD (ONE OR MULTIPLE REPORTS) IN ONE LINE
delimiter="\t" option in pandas.read_fwf avoids white spaces at tails
to be stripped
@author: iregon

OPTIONS IN OLD DEVELOPMENT:
   1. DLMT: delimiter = ',' default
   names = [ (x,y) for x in schema['sections'].keys() for y in schema['sections'][x]['elements'].keys()]
   missing = { x:schema['sections'][x[0]]['elements'][x[1]].get('missing_value') for x in names }
   TextParser = pd.read_csv(source,header = None, delimiter = delimiter, encoding = 'utf-8',
   dtype = 'object', skip_blank_lines = False, chunksize = chunksize,
   skiprows = skiprows, names = names, na_values = missing)
   2. FWF:# delimiter = '\t' so that it reads blanks as blanks, otherwise reads as empty: NaN
   this applies mainly when reading elements from sections, but we leave it also here
   TextParser = pd.read_fwf(source,widths=[FULL_WIDTH],header = None, skiprows = skiprows, delimiter="\t", chunksize = chunksize)
"""

from __future__ import annotations

import os

import pandas as pd

from . import properties


def import_data(source, encoding=None, chunksize=None, skiprows=None):
    """Import data as a pd.TextParser object.

    Returns an iterable object with a pandas dataframe from
    an input data source. The pandas dataframe has a report
    per row and a single column with the full report as a
    block string.
    Currently only supports a data file path as source data,
    but could be easily extended to accept a different
    source object.

    Parameters
    ----------
    source: str
        Path to data file

    encoding: dict, optional
        Encoding dictionary passed to function
        ``pd.read_fwf``.

    chunksize : int, optional
        Number of lines to chunk the input data into
        passed to function ``pd.read_fwf``.

    skiprows : int, optional
        Number of lines to skip from input file
        passed to function ``pd.read_fwf``.

    Returns
    -------
    iterable
        List of with a single pandas dataframe
        or pandas.io.parsers.textfilereader


    """
    if os.path.isfile(source):
        TextParser = pd.read_fwf(
            source,
            encoding=encoding,
            widths=[properties.MAX_FULL_REPORT_WIDTH],
            header=None,
            delimiter="\t",
            skiprows=skiprows,
            chunksize=chunksize,
            quotechar="\0",
            escapechar="\0",
        )
        if not chunksize:
            TextParser = [TextParser]
        return TextParser
    else:
        print("Error")
        return
