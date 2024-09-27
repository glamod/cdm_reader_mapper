"""json dictionary manipulator."""

from __future__ import annotations

import ast
import datetime
import json
import logging
from copy import deepcopy

from .getting_files import get_path

try:
    from pandas.io.json._normalize import nested_to_record
except Exception:
    from pandas.io.json.normalize import nested_to_record


def open_json_file(ifile, encoding="utf-8"):
    """Open JSON file.

    Parameters
    ----------
    ifile: str
        JSON file on disk
    encoding: str, default: "utf-8"
        File encoding

    Returns
    -------
    dict
    """
    with open(ifile, encoding=encoding) as fileObj:
        json_dict = json.load(fileObj)
    return json_dict


def collect_json_files(idir, *args, base=None, name=None):
    """Collect available data_model release deck files.

    Parameters
    ----------
    idir: str
        The name of the input directory.
    args*: optional
        Optional data_model subdirectories for recursivel searching.
        E.g. r300 d701
    base: str
        JSON file base path.
    name: str, optional
        Name of the file to collect.
        Default is a combination of ``data_model`` and ``args``.

    Returns
    -------
    list
        List of available files
    """
    path = f"{base}.{idir}"
    data = get_path(path)
    if name is None:
        ifile = idir
    else:
        ifile = name
    list_of_files = list(data.glob(f"{ifile}.json"))

    i = 0
    while i < len(args):
        if name is None:
            ifile = f"{ifile}_{args[i]}"
        path = f"{path}.{args[i]}"
        data = get_path(path)
        arg_files = []
        if data:
            arg_files = list(data.glob(f"{ifile}.json"))
        if len(arg_files) != 0:
            list_of_files += arg_files
        i += 1
    return list_of_files


def combine_dicts(list_of_files, base=None):
    """Read list of json files and combine them to one dictionary.

    Parameters
    ----------
    list_of_files: str, list
        One or more JSON files on disk to be read.
        One or more JSON dictionaries.
    base: str
        JSON file base path.

    Returns
    -------
    dict
        Combined dictionary from read `list_of_files`.
    """

    def update_dict(old, new):
        keys = list(old.keys()) + list(new.keys())
        keys = list(set(keys))
        for key in keys:
            if key not in new.keys():
                continue
            elif key not in old.keys():
                old[key] = new[key]
            elif new[key] == "ignore":
                old.pop(key)
            elif isinstance(new[key], dict):
                old[key] = update_dict(old[key], new[key])
            else:
                old[key] = new[key]
        return old

    combined_dict = {}
    if isinstance(list_of_files, str):
        list_of_files = [list_of_files]
    for json_dict in list_of_files:
        if not isinstance(json_dict, dict):
            json_dict = open_json_file(json_dict)
        if "substitute" in json_dict.keys():
            data_model = json_dict["substitute"].get("data_model")
            release = json_dict["substitute"].get("release")
            deck = json_dict["substitute"].get("deck")
            new_list_of_files = collect_json_files(data_model, release, deck, base=base)
            new_list_of_files = [
                new_json_file
                for new_json_file in new_list_of_files
                if new_json_file not in list_of_files
            ]
            json_dict = combine_dicts(new_list_of_files)
        combined_dict = update_dict(combined_dict, json_dict)
    return combined_dict


def get_table_keys(table):
    """DOCUMENTATION."""
    separator = "?"  # something hopefully not in keys...
    if table.get("_keys"):
        _table = deepcopy(table)
        _table.pop("_keys")
        keys = list(nested_to_record(_table, sep=separator).keys())

        return [x.split(separator) for x in keys]
    else:
        return list(table.keys())


def eval_dict_items(item):
    """DOCUMENTATION."""
    try:
        return ast.literal_eval(item)
    except Exception:
        return item


def add_nested_keys(table, table_path):
    """DOCUMENTATION."""
    keys_path = table_path.with_suffix(".keys")
    if keys_path.is_file():
        with open(keys_path) as fileObj:
            table_keys = json.load(fileObj)
            table["_keys"] = {}
            for x, y in table_keys.items():
                key = eval_dict_items(x)
                values = [eval_dict_items(k) for k in y]
                table["_keys"][key] = values
    return table


def open_code_table(table_path):
    """DOCUMENTATION."""
    table = open_json_file(table_path)
    # Add keys for nested code tables
    table = add_nested_keys(table, table_path)
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
                    logging.error(f"Lower bound parsing error in range key: {k}")
                    logging.error("Error is:")
                    logging.error(e)
                    return
                try:
                    upper = int(range_params[1])
                except Exception as e:
                    if range_params[1] == "yyyy":
                        upper = datetime.date.today().year
                    else:
                        logging.error(f"Upper bound parsing error in range key: {k}")
                        logging.error("Error is:")
                        logging.error(e)
                        return
                if len(range_params) > 2:
                    try:
                        step = int(range_params[2])
                    except Exception as e:
                        logging.error(f"Range step parsing error in range key: {k}")
                        logging.error("Error is:")
                        logging.error(e)
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
