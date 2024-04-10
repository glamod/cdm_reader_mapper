# noqa: D100
"""
Common Data Model (CDM) pandas selection operators.

Created on Wed Jul  3 09:48:18 2019

@author: iregon
"""
from __future__ import annotations

# Need to define a general thing for the parser() functions, like we did with
# the dataframe_apply_index(), because they are all the same but for the
# selection applied!!!!!

#    The index of the resulting dataframe(s) is reinitialized here, it does not
#    inherit from parent df
#
#    data is a dataframe


def dataframe_apply_index(
    df,
    index_list,
    out_rejected=False,
    in_index=False,
    idx_in_offset=0,
    idx_out_offset=0,
):
    """Apply index to pandas DataFrame."""
    index = df.index.isin(index_list)
    in_df = df[index]
    in_df.index = range(idx_in_offset, idx_in_offset + len(in_df))
    output = [in_df]
    if out_rejected:
        out_df = df[~index]
        out_df.index = range(idx_out_offset, idx_out_offset + len(out_df))
        output.append(out_df)
    if in_index:
        output.append(index_list)

    return output


def select_true(data, mask, out_rejected=False, in_index=False):
    """DOCUMENTATION."""

    #   mask is a the full df/parser of which we only use col
    def dataframe(
        df, mask, out_rejected=False, in_index=False, idx_in_offset=0, idx_out_offset=0
    ):
        # get the index values and pass to the general function
        # If a mask is empty, assume True (...)
        global_mask = mask.all(axis=1)
        index = global_mask[global_mask.fillna(True)].index
        return dataframe_apply_index(
            df,
            index,
            out_rejected=out_rejected,
            in_index=in_index,
            idx_in_offset=idx_in_offset,
            idx_out_offset=idx_out_offset,
        )

    output = dataframe(data, mask, out_rejected=out_rejected, in_index=in_index)

    if len(output) > 1:
        return output
    else:
        return output[0]


def select_from_list(data, selection, out_rejected=False, in_index=False):
    """DOCUMENTATION."""

    #   selection is a dictionary like {col_name:[values to select]}
    def dataframe(
        df,
        col,
        values,
        out_rejected=False,
        in_index=False,
        idx_in_offset=0,
        idx_out_offset=0,
    ):
        # get the index values and pass to the general function
        in_df = df.loc[df[col].isin(values)]
        # in_df = df.loc[df[col].isin([int(i) for i in values])]
        index = list(in_df.index)
        return dataframe_apply_index(
            df,
            index,
            out_rejected=out_rejected,
            in_index=in_index,
            idx_in_offset=idx_in_offset,
            idx_out_offset=idx_out_offset,
        )

    col = list(selection.keys())[0]
    values = list(selection.values())[0]

    output = dataframe(data, col, values, out_rejected=out_rejected, in_index=in_index)

    if len(output) > 1:
        return output
    else:
        return output[0]


def select_from_index(data, index, out_rejected=False):
    """DOCUMENTATION."""

    # index is a list of integer positions to select from data
    def dataframe(df, index, out_rejected=False, idx_in_offset=0, idx_out_offset=0):
        return dataframe_apply_index(
            df,
            index,
            out_rejected=out_rejected,
            idx_in_offset=idx_in_offset,
            idx_out_offset=idx_out_offset,
        )

    output = dataframe(data, index, out_rejected=out_rejected)

    if len(output) > 1:
        return output
    else:
        return output[0]
