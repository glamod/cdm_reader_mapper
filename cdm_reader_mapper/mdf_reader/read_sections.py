r"""
Read data sections.

Created on Fri Jan 10 13:17:43 2020
Extracts and reads (decodes, scales, etc...) the elements of data sections.
Each column of the input dataframe is a section with all its elements stored
as a single string.
Working on a section by section basis, this module uses the data model
information provided in the schema to split the elements, decode and scale them
where appropriate and ensure its data type consistency.
Output is a dataframe with columns as follows depending on the data model
structure:

   1) Data model with sections (1 or more): [(section0,element0),.......(sectionN,elementM)]
   2) Data model with no sections[element0...element1]

DEV NOTES:
1) the 'quoted' issue: in version 1.0:
Writing options from quoting on to prevent supp buoy data to be quoted:
maybe this happenned because buoy data has commas, and pandas makes its own decission about
how to write that.....
https://stackoverflow.com/questions/21147058/pandas-to-csv-output-quoting-issue
quoting=csv.QUOTE_NONE was failing when a section is empty (or just one record in a section,...)
sections_df[section].to_csv(section_buffer,header=False, encoding = 'utf-8',index = False,quoting=csv.QUOTE_NONE,escapechar="\\",sep="\t")
But we were still experiencing problems when reading fully empty sections, now
we only write to the section buffer reports that are not empty. We afterwards
recover the indexes....
# @author: iregon
"""

from __future__ import annotations

import csv
from io import StringIO as StringIO

import pandas as pd

from cdm_reader_mapper.common.converters import converters
from cdm_reader_mapper.common.decoders import decoders

from . import properties


def extract_fixed_width(section_serie_bf, section_schema):
    """DOCUMENTATION."""
    # Read section elements descriptors
    section_names = section_schema["elements"].keys()
    section_widths = list(
        map(
            lambda x: x if x else properties.MAX_FULL_REPORT_WIDTH,
            [section_schema["elements"][i].get("field_length") for i in section_names],
        )
    )
    section_missing = {
        i: section_schema["elements"][i].get("missing_value")
        if section_schema["elements"][i].get("disable_white_strip") is True
        else [
            section_schema["elements"][i].get("missing_value"),
            " "
            * section_schema["elements"][i].get(
                "field_length", properties.MAX_FULL_REPORT_WIDTH
            ),
        ]
        for i in section_names
    }
    section_elements = pd.read_fwf(
        section_serie_bf,
        widths=section_widths,
        header=None,
        names=section_names,
        na_values=section_missing,
        encoding="utf-8",
        dtype="object",
        skip_blank_lines=False,
        quotechar="\0",
        escapechar="\0",
    )
    return section_elements


def extract_delimited(section_serie_bf, section_schema):
    """DOCUMENTATION."""
    delimiter = section_schema["header"].get("delimiter")
    section_names = section_schema["elements"].keys()
    section_missing = {
        x: section_schema["elements"][x].get("missing_value") for x in section_names
    }
    section_elements = pd.read_csv(
        section_serie_bf,
        header=None,
        delimiter=delimiter,
        encoding="utf-8",
        dtype="object",
        skip_blank_lines=False,
        names=section_names,
        na_values=section_missing,
        quotechar="\0",
        escapechar="\0",
    )

    return section_elements


def read_data(section_df, section_schema):
    """DOCUMENTATION."""
    section_names = section_df.columns
    section_dtypes = {
        i: section_schema["elements"][i]["column_type"] for i in section_names
    }
    encoded = [
        (x) for x in section_names if "encoding" in section_schema["elements"][x]
    ]
    section_encoding = {i: section_schema["elements"][i]["encoding"] for i in encoded}
    section_valid = pd.DataFrame(
        index=section_df.index, columns=section_df.columns, dtype=object
    )

    for element in section_dtypes.keys():
        missing = section_df[element].isna()
        if element in encoded:
            section_df[element] = decoders.get(section_encoding.get(element)).get(
                section_dtypes.get(element)
            )(section_df[element])
        kwargs = {
            converter_arg: section_schema["elements"][element].get(converter_arg)
            for converter_arg in properties.data_type_conversion_args.get(
                section_dtypes.get(element)
            )
        }
        section_df[element] = converters.get(section_dtypes.get(element))(
            section_df[element], **kwargs
        )

        section_valid[element] = missing | section_df[element].notna()
    return section_df, section_valid


def read_sections(sections_df, schema):
    """Read sections from pd.DataFrame.

    Returns a pandas dataframe with a report per row
    and the report sections split along the columns.
    Each section is a block string and only the sections
    listed in read_sections parameter are output.

    Parameters
    ----------
    sections_df : pd.DataFrame
        Pandas dataframe with a column per report sections.
        The sections in the columns as a block strings.

    schema : dict
        Data source data model schema

    Returns
    -------
    data : pd.DataFrame
        Dataframe with the report section elements split
        along the columns. Both multiindex and regular
        are possible.

    mask : pd.DataFrame
        Dataframe with the report section elements split
        along the columns. Both multiindex and regular
        are possible.

    dtypes : dict
        Dictionary with pandas data types for each of the
        output elements

    """
    multiindex = (
        True
        if len(sections_df.columns) > 1
        or sections_df.columns[0] != properties.dummy_level
        else False
    )
    data_df = pd.DataFrame(index=sections_df.index)
    valid_df = pd.DataFrame(index=sections_df.index)
    out_dtypes = dict()

    for section in sections_df.columns:
        print(f"Reading section {section}")
        section_schema = schema["sections"].get(section)
        disable_read = section_schema.get("header").get("disable_read")

        if not disable_read:
            field_layout = section_schema.get("header").get("field_layout")
            ignore = [
                i
                for i in section_schema["elements"].keys()
                if section_schema["elements"][i].get("ignore")
            ]  # evals to True if set and true, evals to False if not set or set and false
            # Get rid of false delimiters in fixed_width
            delimiter = section_schema["header"].get("delimiter")
            if delimiter and field_layout == "fixed_width":
                sections_df[section] = sections_df[section].str.replace(delimiter, "")

            section_buffer = StringIO()
            # Here indices are lost, have to give the real ones, those in section_strings:
            # we'll see if we do that in the caller module or here....
            # Only pass records with data to avoid the hassle of dealing with
            # how the NaN rows are written and then read!
            notna_idx = sections_df[sections_df[section].notna()].index
            sections_df[section].loc[notna_idx].to_csv(
                section_buffer,
                header=False,
                encoding="utf-8",
                index=False,
                quoting=csv.QUOTE_NONE,
                quotechar="\0",
                escapechar="\0",
                sep=properties.internal_delimiter,
            )
            section_buffer.seek(0)
            # Get the individual elements as objects
            if field_layout == "fixed_width":
                section_elements_obj = extract_fixed_width(
                    section_buffer, section_schema
                )
            elif field_layout == "delimited":
                section_elements_obj = extract_delimited(section_buffer, section_schema)

            section_elements_obj.drop(ignore, axis=1, inplace=True)

            # Read the objects to their data types and apply decoding, scaling and so on...
            # Give them their actual indexes back
            section_elements, section_valid = read_data(
                section_elements_obj, section_schema
            )

            section_elements.index = notna_idx
            section_valid.index = notna_idx
        else:
            section_elements = pd.DataFrame(sections_df[section], columns=[section])
            section_valid = pd.DataFrame(
                index=section_elements.index, data=True, columns=[section]
            )

        section_elements.columns = (
            [(section, x) for x in section_elements.columns]
            if multiindex
            else section_elements.columns
        )
        section_valid.columns = section_elements.columns
        data_df = pd.concat([data_df, section_elements], sort=False, axis=1)
        valid_df = pd.concat([valid_df, section_valid], sort=False, axis=1)

    # Do the dtypes after removing unwnated elements, etc..
    for section in sections_df.columns:
        section_schema = schema["sections"].get(section)
        if not section_schema.get("header").get("disable_read"):
            elements = [x[1] for x in data_df.columns if x[0] == section]
            if multiindex:
                out_dtypes.update(
                    {
                        (section, i): properties.pandas_dtypes.get(
                            section_schema["elements"][i].get("column_type")
                        )
                        for i in elements
                    }
                )
            else:
                out_dtypes.update(
                    {
                        i: properties.pandas_dtypes.get(
                            section_schema["elements"][i].get("column_type")
                        )
                        for i in elements
                    }
                )
        else:
            if multiindex:
                out_dtypes.update({(section, section): "object"})
            else:
                out_dtypes.update({section: "object"})
    return data_df, valid_df, out_dtypes
