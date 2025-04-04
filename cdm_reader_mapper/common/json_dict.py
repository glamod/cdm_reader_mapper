"""json dictionary manipulator."""

from __future__ import annotations

import json

from .getting_files import get_path


def open_json_file(ifile, encoding="utf-8") -> dict:
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


def collect_json_files(idir, *args, base=None, name=None) -> list[str]:
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


def combine_dicts(list_of_files, base=None) -> dict:
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
