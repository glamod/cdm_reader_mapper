"""JSON dictionary manipulator."""

from __future__ import annotations
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .getting_files import get_path


def open_json_file(ifile: str | Path, encoding: str = "utf-8") -> dict[Any, Any]:
    """
    Open a JSON file and return its contents as a dictionary.

    Parameters
    ----------
    ifile : str or Path
        Path to the JSON file.
    encoding : str, default 'utf-8'
        Encoding to use when reading the file.

    Returns
    -------
    dict
        Contents of the JSON file.
    """
    with Path(ifile).open(encoding=encoding) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object, got {type(data)}.")

    return data


def collect_json_files(idir: str, *args: str, base: str | None = None, name: str | None = None) -> list[Path]:
    r"""
    Collect JSON files recursively based on directory and optional subdirectories.

    Parameters
    ----------
    idir : str
        Base directory to search.
    \*args : str
        Optional subdirectory names for recursive searching.
    base : str, optional
        Base path to prepend to idir.
    name : str, optional
        Base file name to search. If None, defaults to idir.

    Returns
    -------
    list of Path
        List of matching JSON file paths.
    """
    path = f"{base}.{idir}" if base else idir
    data_dir = get_path(path)
    ifile = name or idir
    list_of_files = list(data_dir.glob(f"{ifile}.json")) if data_dir else []

    for arg in args:
        if name is None:
            ifile = f"{ifile}_{arg}"
        path = f"{path}.{arg}" if base else arg
        data_dir = get_path(path)
        arg_files = list(data_dir.glob(f"{ifile}.json")) if data_dir else []
        list_of_files.extend(arg_files)

    return list_of_files


def combine_dicts(list_of_files: str | Path | Sequence[str | Path | dict[str, Any]], base: str | None = None) -> dict[str, Any]:
    """
    Combine multiple JSON files or dictionaries into a single dictionary.

    Supports nested 'substitute' references to recursively load additional JSON files.

    Parameters
    ----------
    list_of_files : str, Path, list
        JSON file(s) or dictionaries to combine.
    base : str, optional
        Base path used when resolving substituted files.

    Returns
    -------
    dict
        Combined dictionary from all input files/dictionaries.
    """

    def update_dict(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
        """
        Update dictionary.

        Parameters
        ----------
        old : dict
            Dictionary to be updated.
        new : dict
            Dictionary used for updating.

        Returns
        -------
        dict
            Updated dictionary.
        """
        keys = set(old.keys()) | set(new.keys())
        for key in keys:
            if key not in new:
                continue
            elif key not in old:
                old[key] = new[key]
            elif new[key] == "ignore":
                old.pop(key, None)
            elif isinstance(new[key], dict):
                old[key] = update_dict(old.get(key, {}), new[key])
            else:
                old[key] = new[key]
        return old

    combined_dict: dict[str, Any] = {}
    if isinstance(list_of_files, (str, Path)):
        list_of_files = [list_of_files]

    for item in list_of_files:
        json_dict = item if isinstance(item, dict) else open_json_file(item)
        # Handle recursive substitution
        if "substitute" in json_dict:
            sub = json_dict["substitute"]
            data_model = sub.get("data_model")
            release = sub.get("release")
            deck = sub.get("deck")
            new_files = collect_json_files(data_model, release, deck, base=base)
            new_files = [f for f in new_files if f not in list_of_files]
            json_dict = combine_dicts(new_files, base=base)
        combined_dict = update_dict(combined_dict, json_dict)

    return combined_dict
