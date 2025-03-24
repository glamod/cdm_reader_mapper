# noqa: D100
"""
Common Data Model (CDM) pandas selection operators.

Created on Wed Jul  3 09:48:18 2019

@author: iregon
"""
from __future__ import annotations

from io import StringIO

import pandas as pd

from cdm_reader_mapper.common import pandas_TextParser_hdlr

# Need to define a general thing for the parser() functions, like we did with
# the dataframe_apply_index(), because they are all the same but for the
# selection applied!!!!!

# The index of the resulting dataframe(s) is reinitialized here, it does not
# inherit from parent df
#
# data is a dataframe or a TextFileReader


def dataframe_apply_index(
    df,
    index_list,
    inverse=False,
    idx_in_offset=0,
    idx_out_offset=0,
):
    """Apply index to pandas DataFrame."""
    index = df.index.isin(index_list)

    if inverse is True:
        in_df = df[~index]
    else:
        in_df = df[index]

    in_df.index = range(idx_in_offset, idx_in_offset + len(in_df))

    return in_df


def select_bool(data, mask, boolean, inverse=False):
    """DOCUMENTATION."""

    # mask is a the full df/parser of which we only use col
    def dataframe(df, mask, boolean, inverse=False, idx_in_offset=0, idx_out_offset=0):
        # get the index values and pass to the general function
        # If a mask is empty, assume True (...)
        if boolean is True:
            global_mask = mask.all(axis=1)
        else:
            global_mask = ~(mask.any(axis=1))
        index = global_mask[global_mask.fillna(boolean)].index
        return dataframe_apply_index(
            df,
            index,
            inverse=inverse,
            idx_in_offset=idx_in_offset,
            idx_out_offset=idx_out_offset,
        )

    def parser(data_parser, mask_parser, boolean, inverse=False):
        mask_cp = pandas_TextParser_hdlr.make_copy(mask_parser)
        read_params = [
            "chunksize",
            "names",
            "dtype",
            "parse_dates",
            "date_parser",
            "infer_datetime_format",
        ]
        read_dict = {x: data_parser.orig_options.get(x) for x in read_params}
        in_buffer = StringIO()
        idx_in_offset = 0
        idx_out_offset = 0
        for df, mask_df in zip(data_parser, mask_cp):
            output = dataframe(
                df,
                mask_df,
                boolean,
                inverse=inverse,
                idx_in_offset=idx_in_offset,
                idx_out_offset=idx_out_offset,
            )
            output.to_csv(in_buffer, header=False, index=False, mode="a")
            idx_in_offset += len(output)

        mask_cp.close()
        in_buffer.seek(0)
        return pd.read_csv(in_buffer, **read_dict)

    if isinstance(data, pd.io.parsers.TextFileReader):
        return parser(data, mask, boolean, inverse=inverse)

    return dataframe(data, mask, boolean, inverse=inverse)


def select_true(data, mask, inverse=False):
    """DOCUMENTATION."""
    return select_bool(data, mask, True, inverse=inverse)


def select_false(data, mask, inverse=False):
    """DOCUMENTATION."""
    return select_bool(data, mask, False, inverse=inverse)


def select_from_list(data, selection, inverse=False):
    """DOCUMENTATION."""

    # selection is a dictionary like {col_name:[values to select]}
    def dataframe(
        df,
        col,
        values,
        inverse=False,
        idx_in_offset=0,
        idx_out_offset=0,
    ):
        # get the index values and pass to the general function
        in_df = df.loc[df[col].isin(values)]
        index = list(in_df.index)
        return dataframe_apply_index(
            df,
            index,
            inverse=inverse,
            idx_in_offset=idx_in_offset,
            idx_out_offset=idx_out_offset,
        )

    def parser(data_parser, col, values, inverse=False):
        read_params = [
            "chunksize",
            "names",
            "dtype",
            "parse_dates",
            "date_parser",
            "infer_datetime_format",
        ]
        read_dict = {x: data_parser.orig_options.get(x) for x in read_params}
        in_buffer = StringIO()
        idx_in_offset = 0
        idx_out_offset = 0
        for df in data_parser:
            output = dataframe(
                df,
                col,
                values,
                inverse=inverse,
                idx_in_offset=idx_in_offset,
                idx_out_offset=idx_out_offset,
            )
            output.to_csv(in_buffer, header=False, index=False, mode="a")
            idx_in_offset += len(output)

        in_buffer.seek(0)
        return pd.read_csv(in_buffer, **read_dict)

    col = list(selection.keys())[0]
    values = list(selection.values())[0]
    if isinstance(data, pd.io.parsers.TextFileReader):
        return parser(data, col, values, inverse=inverse)
    return dataframe(data, col, values, inverse=inverse)


def select_from_index(data, index, inverse=False):
    """DOCUMENTATION."""

    # index is a list of integer positions to select from data
    def dataframe(df, index, inverse=False, idx_in_offset=0, idx_out_offset=0):
        return dataframe_apply_index(
            df,
            index,
            inverse=inverse,
            idx_in_offset=idx_in_offset,
            idx_out_offset=idx_out_offset,
        )

    def parser(data_parser, index, inverse=False):
        read_params = [
            "chunksize",
            "names",
            "dtype",
            "parse_dates",
            "date_parser",
            "infer_datetime_format",
        ]
        read_dict = {x: data_parser.orig_options.get(x) for x in read_params}
        in_buffer = StringIO()

        for df in data_parser:
            output = dataframe(df, index, inverse=inverse)
            output.to_csv(in_buffer, header=False, index=False, mode="a")

        in_buffer.seek(0)
        return pd.read_csv(in_buffer, **read_dict)

    if isinstance(data, pd.io.parsers.TextFileReader):
        return parser(data, index, inverse=inverse)

    return dataframe(data, index, inverse=inverse)
