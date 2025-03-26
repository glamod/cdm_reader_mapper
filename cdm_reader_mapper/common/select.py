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
    idx_in_offset=0,
    idx_out_offset=0,
):
    """Apply index to pandas DataFrame."""
    index = df.index.isin(index_list)

    if inverse is True:
        in_df = df[~index]
    else:
        in_df = df[index]

    if reset_index is True:
        in_df.index = range(idx_in_offset, idx_in_offset + len(in_df))

    return in_df


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
        idx_in_offset=0,
        idx_out_offset=0,
    ):
        # get the index values and pass to the general function
        # If a mask is empty, assume True (...)
        if boolean is True:
            global_mask = mask.all(axis=1)
        else:
            global_mask = ~(mask.any(axis=1))
        index = global_mask[global_mask.fillna(boolean)].index
        out1 = dataframe_apply_index(
            df,
            index,
            reset_index=reset_index,
            inverse=inverse,
            idx_in_offset=idx_in_offset,
            idx_out_offset=idx_out_offset,
        )
        if return_rejected is True:
            out2 = dataframe_apply_index(
                df,
                index,
                reset_index=reset_index,
                inverse=inverse,
                idx_in_offset=idx_in_offset,
                idx_out_offset=idx_out_offset,
            )
            return out1, out2
        return out1, pd.DataFrame()

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
        read_dict = {x: data_parser.orig_options.get(x) for x in read_params}
        buffer1 = StringIO()
        buffer2 = StringIO()
        idx_in_offset = 0
        idx_out_offset = 0
        for df, mask_df in zip(data_parser, mask_cp):
            out1, out2 = dataframe(
                df,
                mask_df,
                boolean,
                reset_index=reset_index,
                inverse=inverse,
                return_rejected=return_rejected,
                idx_in_offset=idx_in_offset,
                idx_out_offset=idx_out_offset,
            )
            out1.to_csv(buffer1, header=False, mode="a")
            if return_rejected is True:
                out2.to_csv(buffer1, header=False, mode="a")
            idx_in_offset += len(out1)

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
        idx_in_offset=0,
        idx_out_offset=0,
    ):
        # get the index values and pass to the general function
        in_df = df.loc[df[col].isin(values)]
        index = list(in_df.index)
        out1 = dataframe_apply_index(
            df,
            index,
            reset_index=reset_index,
            inverse=inverse,
            idx_in_offset=idx_in_offset,
            idx_out_offset=idx_out_offset,
        )
        if return_rejected is True:
            out2 = dataframe_apply_index(
                df,
                index,
                reset_index=reset_index,
                inverse=inverse,
                idx_in_offset=idx_in_offset,
                idx_out_offset=idx_out_offset,
            )
            return out1, out2
        return out1, pd.DataFrame()

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
        read_dict = {x: data_parser.orig_options.get(x) for x in read_params}
        buffer1 = StringIO()
        buffer2 = StringIO()
        idx_in_offset = 0
        idx_out_offset = 0
        for df in data_parser:
            out1, out2 = dataframe(
                df,
                col,
                values,
                reset_index=reset_index,
                inverse=inverse,
                idx_in_offset=idx_in_offset,
                idx_out_offset=idx_out_offset,
            )
            out1.to_csv(buffer1, header=False, mode="a")
            idx_in_offset += len(out1)
            if return_rejected is True:
                out2.to_csv(buffer2, header=False, mode="a")

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
        idx_in_offset=0,
        idx_out_offset=0,
    ):
        out1 = dataframe_apply_index(
            df,
            index,
            reset_index=reset_index,
            inverse=inverse,
            idx_in_offset=idx_in_offset,
            idx_out_offset=idx_out_offset,
        )
        if return_rejected is True:
            out2 = dataframe_apply_index(
                df,
                index,
                reset_index=reset_index,
                inverse=inverse,
                idx_in_offset=idx_in_offset,
                idx_out_offset=idx_out_offset,
            )
            return out1, out2
        return out1, pd.DataFrame()

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
            out1.to_csv(buffer1, header=False, mode="a")
            if return_rejected is True:
                out2.to_csv(buffer2, header=False, mode="a")

        buffer1.seek(0)
        buffer2.seek(0)
        return pd.read_csv(buffer1, **read_dict), pd.read_csv(buffer2, **read_dict)

    if isinstance(data, pd.io.parsers.TextFileReader):
        return parser(data, index, reset_index=reset_index, inverse=inverse)

    return dataframe(data, index, reset_index=reset_index, inverse=inverse)
