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
    reset_index=False,
    inverse=False,
):
    """Apply index to pandas DataFrame."""
    index = df.index.isin(index_list)

    if inverse is True:
        in_df = df[~index]
    else:
        in_df = df[index]

    if reset_index is True:
        in_df = in_df.reset_index(drop=True)

    return in_df


def dataframe_selection(
    df,
    index,
    reset_index=False,
    inverse=False,
    return_rejected=False,
):
    """Common dataframe selction fucntion."""
    out1 = dataframe_apply_index(
        df,
        index,
        reset_index=reset_index,
        inverse=inverse,
    )
    if return_rejected is True:
        index2 = [idx for idx in df.index if idx not in index]
        out2 = dataframe_apply_index(
            df,
            index2,
            reset_index=reset_index,
            inverse=inverse,
        )
        return out1, out2
    return out1, pd.DataFrame(columns=out1.columns)


def parser_apply_index():
    """Apply index for pandas.TextFileReader."""


def select_bool(
    data, mask, boolean, reset_index=False, inverse=False, return_rejected=False
):
    """DOCUMENTATION."""

    # mask is a the full df/parser of which we only use col
    def dataframe(
        df,
        mask,
        boolean,
        reset_index=False,
        inverse=False,
        return_rejected=False,
    ):
        # get the index values and pass to the general function
        # If a mask is empty, assume True (...)
        if boolean is True:
            global_mask = mask.all(axis=1)
        else:
            global_mask = ~(mask.any(axis=1))
        index = global_mask[global_mask.fillna(boolean)].index
        return dataframe_selection(
            df,
            index,
            reset_index=reset_index,
            inverse=inverse,
            return_rejected=return_rejected,
        )

    def parser(
        data_parser,
        mask_parser,
        boolean,
        reset_index=False,
        inverse=False,
        return_rejected=False,
    ):
        mask_cp = pandas_TextParser_hdlr.make_copy(mask_parser)
        read_params = [
            "chunksize",
            "names",
            "dtype",
            "parse_dates",
            "date_parser",
            "infer_datetime_format",
        ]
        write_dict = {"header": None, "mode": "a", "index": not reset_index}
        read_dict = {x: data_parser.orig_options.get(x) for x in read_params}
        buffer1 = StringIO()
        buffer2 = StringIO()
        for df, mask_df in zip(data_parser, mask_cp):
            out1, out2 = dataframe(
                df,
                mask_df,
                boolean,
                reset_index=reset_index,
                inverse=inverse,
                return_rejected=return_rejected,
            )
            out1.to_csv(buffer1, **write_dict)
            if return_rejected is True:
                out2.to_csv(buffer2, **write_dict)

        mask_cp.close()
        buffer1.seek(0)
        buffer2.seek(0)
        return pd.read_csv(buffer1, **read_dict), pd.read_csv(buffer2, **read_dict)

    if isinstance(data, pd.io.parsers.TextFileReader):
        return parser(
            data,
            mask,
            boolean,
            reset_index=reset_index,
            inverse=inverse,
            return_rejected=return_rejected,
        )

    return dataframe(
        data,
        mask,
        boolean,
        reset_index=reset_index,
        inverse=inverse,
        return_rejected=return_rejected,
    )


def select_true(data, mask, reset_index=False, inverse=False, return_rejected=False):
    """DOCUMENTATION."""
    return select_bool(
        data,
        mask,
        True,
        reset_index=reset_index,
        inverse=inverse,
        return_rejected=return_rejected,
    )


def select_false(data, mask, reset_index=False, inverse=False, return_rejected=False):
    """DOCUMENTATION."""
    return select_bool(
        data,
        mask,
        False,
        reset_index=reset_index,
        inverse=inverse,
        return_rejected=return_rejected,
    )


def select_from_list(
    data, selection, reset_index=False, inverse=False, return_rejected=False
):
    """DOCUMENTATION."""

    # selection is a dictionary like {col_name:[values to select]}
    def dataframe(
        df,
        col,
        values,
        reset_index=False,
        inverse=False,
        return_rejected=False,
    ):
        # get the index values and pass to the general function
        in_df = df.loc[df[col].isin(values)]
        index = list(in_df.index)
        return dataframe_selection(
            df,
            index,
            reset_index=reset_index,
            inverse=inverse,
            return_rejected=return_rejected,
        )

    def parser(
        data_parser,
        col,
        values,
        reset_index=False,
        inverse=False,
        return_rejected=False,
    ):
        read_params = [
            "chunksize",
            "names",
            "dtype",
            "parse_dates",
            "date_parser",
            "infer_datetime_format",
        ]
        write_dict = {"header": None, "mode": "a", "index": not reset_index}
        read_dict = {x: data_parser.orig_options.get(x) for x in read_params}
        buffer1 = StringIO()
        buffer2 = StringIO()
        for df in data_parser:
            out1, out2 = dataframe(
                df,
                col,
                values,
                reset_index=reset_index,
                inverse=inverse,
                return_rejected=return_rejected,
            )
            out1.to_csv(buffer1, **write_dict)
            if return_rejected is True:
                out2.to_csv(buffer2, **write_dict)

        buffer1.seek(0)
        buffer2.seek(0)
        return pd.read_csv(buffer1, **read_dict), pd.read_csv(buffer2, **read_dict)

    col = list(selection.keys())[0]
    values = list(selection.values())[0]
    if isinstance(data, pd.io.parsers.TextFileReader):
        return parser(
            data,
            col,
            values,
            reset_index=reset_index,
            inverse=inverse,
            return_rejected=return_rejected,
        )
    return dataframe(
        data,
        col,
        values,
        reset_index=reset_index,
        inverse=inverse,
        return_rejected=return_rejected,
    )


def select_from_index(
    data, index, reset_index=False, inverse=False, return_rejected=False
):
    """DOCUMENTATION."""

    # index is a list of integer positions to select from data
    def dataframe(
        df,
        index,
        reset_index=False,
        inverse=False,
        return_rejected=False,
    ):
        return dataframe_selection(
            df,
            index,
            reset_index=reset_index,
            inverse=inverse,
            return_rejected=return_rejected,
        )

    def parser(
        data_parser, index, reset_index=False, inverse=False, return_rejected=False
    ):
        read_params = [
            "chunksize",
            "names",
            "dtype",
            "parse_dates",
            "date_parser",
            "infer_datetime_format",
        ]
        write_dict = {"header": None, "mode": "a", "index": not reset_index}
        read_dict = {x: data_parser.orig_options.get(x) for x in read_params}
        buffer1 = StringIO()
        buffer2 = StringIO()
        for df in data_parser:
            out1, out2 = dataframe(
                df,
                index,
                reset_index=reset_index,
                inverse=inverse,
                return_rejected=return_rejected,
            )
            out1.to_csv(buffer1, **write_dict)
            if return_rejected is True:
                out2.to_csv(buffer2, **write_dict)

        buffer1.seek(0)
        buffer2.seek(0)
        return pd.read_csv(buffer1, **read_dict), pd.read_csv(buffer2, **read_dict)

    if isinstance(data, pd.io.parsers.TextFileReader):
        return parser(
            data,
            index,
            reset_index=reset_index,
            inverse=inverse,
            return_rejected=return_rejected,
        )

    return dataframe(
        data,
        index,
        reset_index=reset_index,
        inverse=inverse,
        return_rejected=return_rejected,
    )
