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
    """Common pandas DataFrame selection function."""
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


def dataframe_selection_bool(df, mask, boolean, **kwargs):
        """DOCUMENTATION."""
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
            **kwargs,
        )
      
def dataframe_selection_from_list(df, col, values, **kwargs):
        """DOCUMENTATION."""
        # get the index values and pass to the general function
        in_df = df.loc[df[col].isin(values)]
        index = list(in_df.index)
        return dataframe_selection(
            df,
            index,
            **kwargs,
        )


def dataframe_selection_from_index(df, index, **kwargs):    
        """DOCUMENTATION."""
        return dataframe_selection(
            df,
            index,
            **kwargs,
        )    
    
    
def parser_selection(data, *args, func=None, mask=None, reset_index=False, inverse=False, return_rejected=False):
    """Common pandas TextFileReader selection function."""
    read_params = [
            "chunksize",
            "names",
            "dtype",
            "parse_dates",
            "date_parser",
            "infer_datetime_format",
    ]
    write_dict = {"header": None, "mode": "a", "index": not reset_index}
    read_dict = {x: data.orig_options.get(x) for x in read_params}
    buffer1 = StringIO()
    buffer2 = StringIO()
    if mask is None:
        zipped = data
    else:
        mask_cp = pandas_TextParser_hdlr.make_copy(mask)
        zipped = zip(data, mask_cp)
        
    for zip_ in zipped:
            if not isinstance(zip_, tuple):
                zip_ = [zip_]
            out1, out2 = func(
                *zip_,
                *args,
                reset_index=reset_index,
                inverse=inverse,
                return_rejected=return_rejected,
            )
            out1.to_csv(buffer1, **write_dict)
            if return_rejected is True:
                out2.to_csv(buffer2, **write_dict)
    
    if mask is not None:
        mask_cp.close()
    buffer1.seek(0)
    buffer2.seek(0)
    return pd.read_csv(buffer1, **read_dict), pd.read_csv(buffer2, **read_dict)    
    
def select(data, func, *args, **kwargs):
    """DOCUMENTATION."""
    if isinstance(data, pd.io.parsers.TextFileReader):
        return parser_selection(
            data,
            *args,
            func=func,
            **kwargs,
        )
    return func(data, *args, **kwargs)    

def select_bool(
    data, mask, boolean, reset_index=False, inverse=False, return_rejected=False
):
    """DOCUMENTATION.""" 
    func = dataframe_selection_bool
    return select(
        data,
        func,
        boolean,
        mask=mask,
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
    func = dataframe_selection_from_list
    col = list(selection.keys())[0]
    values = list(selection.values())[0]
    return select(
        data,
        func,
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
    func = dataframe_selection_from_index
    return select(
        data,
        func,
        index,
        reset_index=reset_index,
        inverse=inverse,
        return_rejected=return_rejected,        
    )
