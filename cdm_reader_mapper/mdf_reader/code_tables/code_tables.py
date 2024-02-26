"""
Manage data model code table files.

Functions to manage data model
code table files and objects according to the
requirements of the data reader tool

"""

from __future__ import annotations

import datetime
import json
import os
from copy import deepcopy

import numpy as np
import pandas as pd

try:
    from pandas.io.json._normalize import nested_to_record
except Exception:
    from pandas.io.json.normalize import nested_to_record

import ast


def read_table(table_path):
    """
    Read a data model code table file to a dictionary.

    It completes the code table to the full complexity
    the data reader expects, by appending information
    on secondary keys and expanding range keys.

    Arguments
    ---------
    table_path : str
        The file path of the code table.

    Returns
    -------
    dict
        Code table

    """
    with open(table_path, encoding="utf-8") as fileObj:
        table = json.load(fileObj)
    # Add keys for nested code tables
    keys_path = ".".join([".".join(table_path.split(".")[:-1]), "keys"])
    if os.path.isfile(keys_path):
        with open(keys_path) as fileObj:
            table_keys = json.load(fileObj)
            table["_keys"] = {}
            for x, y in table_keys.items():
                key = eval_dict_items(x)
                values = [eval_dict_items(k) for k in y]
                table["_keys"][key] = values
    # Expand range keys
    expand_integer_range_key(table)

    return table


def expand_integer_range_key(d):
    """DOCUMENTATION."""
    # Looping based on print_nested above
    if isinstance(d, dict):
        for k, v in list(d.items()):
            if "range_key" in k[0:9]:
                range_params = k[10:-1].split(",")
                try:
                    lower = int(range_params[0])
                except Exception as e:
                    print("Lower bound parsing error in range key: ", k)
                    print("Error is:")
                    print(e)
                    return
                try:
                    upper = int(range_params[1])
                except Exception as e:
                    if range_params[1] == "yyyy":
                        upper = datetime.date.today().year
                    else:
                        print("Upper bound parsing error in range key: ", k)
                        print("Error is:")
                        print(e)
                        return
                if len(range_params) > 2:
                    try:
                        step = int(range_params[2])
                    except Exception as e:
                        print("Range step parsing error in range key: ", k)
                        print("Error is:")
                        print(e)
                        return
                else:
                    step = 1
                for i_range in range(lower, upper + 1, step):
                    deep_copy_value = deepcopy(
                        d[k]
                    )  # Otherwiserepetitions are linked and act as one!
                    d.update({str(i_range): deep_copy_value})
                d.pop(k, None)
            else:
                for k, v in d.items():
                    expand_integer_range_key(v)


def eval_dict_items(item):
    """DOCUMENTATION."""
    try:
        return ast.literal_eval(item)
    except Exception:
        return item


def table_keys(table):
    """DOCUMENTATION."""
    separator = "∿"  # something hopefully not in keys...
    if table.get("_keys"):
        _table = deepcopy(table)
        _table.pop("_keys")
        keys = list(nested_to_record(_table, sep=separator).keys())

        return [x.split(separator) for x in keys]
    else:
        return list(table.keys())


def get_nested(table, *args):
    """DOCUMENTATION."""
    # HERE HAVE TO ADD WHICH ITEM TO GET FROM THE KEY: WE HAVE TO ADD VALUE, LOWER, ETC...TO THE CODE TABLES!!!
    # CAN BE AND OPTIONAL PARAMETER, LIKE: similarly, would have to add tbis to table_value_from_keys
    #    def get_nested(table,param = None,*args):
    #       nested_get_str = 'table'
    #       z = np.array([*args])
    #       for i,x in enumerate(z):
    #           nested_get_str += '.get(z[' + str(i) + '])'
    #       if param:
    #           nested_get_str += '.get(' + param + ')'
    #       try:
    #           return eval(nested_get_str)
    #       except:
    #           return None
    nested_get_str = "table"
    z = np.array([*args])
    for i, x in enumerate(z):
        nested_get_str += ".get(z[" + str(i) + "])"
    try:
        return eval(nested_get_str)
    except Exception:
        return None


def table_value_from_keys(table, df):
    """DOCUMENTATION."""
    # df is pd.DataFrame or Series
    v_nested_get = np.vectorize(
        get_nested
    )  # Because cannot directly vectorize a nested get, we build it in a function, and then vectorize it
    calling_str = "v_nested_get(table"
    if isinstance(df, pd.DataFrame):
        # return v_nested_get(table,[ df[x]  for x in df]) # This won't work
        for i, x in enumerate(df.columns):
            calling_str += (
                ",df[" + str(x) + "].astype(str)"
            )  # have to do likewise in not DataFrame!!!
        calling_str += ")"
        return eval(calling_str)
    else:
        return v_nested_get(table, df)
