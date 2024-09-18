"""json dictionary manipulator."""

from __future__ import annotations

import json
import logging

from .. import properties
from .getting_file import get_files


def get_path(path):
    """Get path."""
    try:
        return get_files(path)
    except ModuleNotFoundError:
        logging.warning(f"No module named {path}")


def collect_json_files(data_model, module="schema", *args):
    """Collect available data_model release deck files.

    Parameters
    ----------
    data_model: str
        The name of the data model to read. This is for
        data models included in the tool
    module: str, default: schema
        Name of the module to get the files.
    args*: optional
        Optional data_model subdirectories.

    Returns
    -------
    list
        List of available files
    """
    if data_model not in properties.supported_data_models:
        logging.error(f"Input data model {data_model} not supported.")
        return
    path = f"{properties._base}.{module}.{data_model}"
    data = get_path(path)
    list_of_files = list(data.glob(f"{data_model}.json"))

    i = 0
    while i < len(args):
        data_model = f"{data_model}_{args[i]}"
        path = f"{path}.{args[i]}"
        data = get_path(path)
        arg_files = []
        if data:
            arg_files = list(data.glob(f"{data_model}.json"))
        if len(arg_files) == 0:
            logging.warning(f"Input {data_model} not supported.")
            list_of_files += arg_files
        i += 1
    return list_of_files


def combine_dicts(list_of_files):
    """Read list of json files and combine them to one dictionary.

    Parameters
    ----------
    list_of_files: str, list
        One or more json files on disk to be read.

    Returns
    -------
    dict
        Combined dictionary from read `list_of_files`.
    """

    def open_json_file(ifile):
        with open(ifile) as fileObj:
            json_dict = json.load(fileObj)
        return json_dict

    def update_dict(old, new):
        keys = list(old.keys()) + list(new.keys())
        keys = list(set(keys))
        for key in keys:
            if key not in new.keys():
                continue
            elif key not in old.keys():
                old[key] = new[key]
            elif isinstance(new[key], dict):
                old[key] = update_dict(old[key], new[key])
            else:
                old[key] = new[key]
        return old

    combined_dict = {}
    if isinstance(list_of_files, str):
        list_of_files = [list_of_files]
    for json_file in list_of_files:
        json_dict = open_json_file(json_file)
        if "substitute" in json_dict.keys():
            data_model = json_dict["substitute"].get("data_model")
            release = json_dict["substitute"].get("release")
            deck = json_dict["substitute"].get("deck")
            new_list_of_files = collect_json_files(data_model, release, deck)
            new_list_of_files = [
                new_json_file
                for new_json_file in new_list_of_files
                if new_json_file not in list_of_files
            ]
            json_dict = combine_dicts(new_list_of_files)
        combined_dict = update_dict(combined_dict, json_dict)
    return combined_dict
