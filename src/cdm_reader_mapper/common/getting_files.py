"""Internal pandas local file operator."""

from __future__ import annotations
import hashlib
import logging
import os
import warnings
from importlib.resources import files as _files
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import requests  # type: ignore[import-untyped]
from platformdirs import user_cache_dir


_default_cache_dir_ = Path(user_cache_dir("cdm-testdata", ".cache"))


def _file_md5_checksum(f_name: str | Path) -> str:
    """
    Compute the MD5 checksum of a file.

    Parameters
    ----------
    f_name : str or Path-like
        File on disk computing MD5 checksum for.

    Returns
    -------
    str
        MD5 checksum of `f_name`.
    """
    hash_md5 = hashlib.md5()  # noqa: S324
    chunk_size = 8192

    with Path(f_name).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def _get_remote_file(lfile: str | Path, url: str, name: str | Path) -> tuple[Path, requests.models.Response]:
    """
    Download a remote file to a local path.

    Parameters
    ----------
    lfile : str or Path-like
        Destination path where the downloaded file will be saved.
    url : str
        Base URL of the remote location (must use HTTP or HTTPS).
    name : str or Path-like
        File name or relative path to append to the base URL.

    Returns
    -------
    tuple of Path and requests.models.Response
        A tuple containing:
        - The local file path (as a ``Path`` object)
        - The ``requests.models.Response`` object returned by the HTTP request

    Raises
    ------
    ValueError
        If the constructed URL does not use HTTP or HTTPS.
    """
    lfile = Path(lfile)
    name = Path(name)
    remote_url = "/".join((url.rstrip("/"), name.as_posix()))

    lfile.parent.mkdir(exist_ok=True, parents=True)
    logging.info("Attempting to fetch remote file: %s", name.as_posix())

    parsed = urlparse(remote_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}.")

    response = requests.get(remote_url, timeout=10)
    response.raise_for_status()

    lfile.write_bytes(response.content)

    return lfile, response


def _check_md5s(f: Path, md5: str, mode: Literal["error", "warning"] = "error") -> bool:
    """
    Verify the MD5 checksum of a file.

    Parameters
    ----------
    f : Path
        File to verify MD5 checksum.
    md5 : str
        MD5 checksum to verify `f`.
    mode : {error, literal}, default, error
        If error raises OSError if `md5` does not match MD5 checksum of `f`.
        If warning warns and returns False.

    Returns
    -------
    bool
        True if `md5` matches MD5 checksum of `f`., otherwise False.

    Raises
    ------
    OSError
        If mode is error and `md5` does not match MD5 checksum of `f`.

    Warns
    -----
    UserWarning
        If mode is warning and `md5` does not match MD5 checksum of `f`.
    """
    msg = f"{f.as_posix()} and md5 checksum do not match."
    md5_actual = _file_md5_checksum(f)
    if md5_actual.strip() != md5.strip():
        f.unlink()
        if mode == "error":
            raise OSError(msg)
        elif mode == "warning":
            warnings.warn(msg, stacklevel=2)
            return False
    return True


def _with_md5_suffix(name: Path, suffix: str) -> Path:
    """
    Return a new path with the given suffix followed by ``.md5``.

    Parameters
    ----------
    name : Path
        Original file path.
    suffix : str
        Suffix to apply before the .md5 extension.

    Returns
    -------
    Path
        A new Path object with the modified suffix.
    """
    return name.with_suffix(f"{suffix}.md5")


def _rm_tree(path: Path) -> None:
    """
    Recursively remove a directory and all its children.

    Parameters
    ----------
    path : Path
        The directory to remove.

    Raises
    ------
    OSError
        If a file or directory cannot be removed due to permission issues
        or other filesystem-related errors.
    """
    # https://stackoverflow.com/questions/50186904/pathlib-recursively-remove-directory
    if not path.is_dir():
        logging.warning("Could not clear cache. Directory %s does not exist.", path.name)
        return

    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        else:
            _rm_tree(child)
    path.rmdir()


def _get_file(
    name: Path,
    suffix: str,
    url: str,
    cache_dir: Path,
    clear_cache: bool,
    within_drs: bool,
) -> Path:
    """
    Retrieve a file into a cache directory.

    Parameters
    ----------
    name : Path
        Remote file path (relative to `url`).
    suffix : str
        Suffix used to construct the associated MD5 filename.
    url : str
        Base URL of the remote file repository.
    cache_dir : Path
        Local directory used for caching downloaded files.
    clear_cache : bool
        If True the entire cache directory is removed before downloading.
    within_drs : bool
        If True preserve the relative directory structure of `name` inside
        the cache, otherwise remove only the filename is used.

    Returns
    -------
    Path
        Path to the cached local file.
    """
    cache_dir = cache_dir.absolute()

    if clear_cache is True:
        _rm_tree(cache_dir)

    cache_dir.mkdir(exist_ok=True, parents=True)

    if within_drs is False:
        local_name = Path(name.name)
    else:
        local_name = name

    local_file = cache_dir / local_name
    md5_file = cache_dir / _with_md5_suffix(local_name, suffix)
    md5_name = _with_md5_suffix(name, suffix)

    _get_remote_file(md5_file, url, md5_name)

    with Path(md5_file).open() as f:
        remote_md5 = f.read()

    if not local_file.is_file():
        _get_remote_file(local_file, url, name)
        _check_md5s(local_file, remote_md5)
    else:
        if not _check_md5s(local_file, remote_md5, mode="warning"):
            _get_remote_file(local_file, url, name)
            _check_md5s(local_file, remote_md5)

    md5_file.unlink()
    return local_file


# idea copied from xclim that borrowed it from raven that borrowed it from xclim that borrowed it from xarray that was borrowed from Seaborn
def load_file(
    name: str | os.PathLike[str],
    github_url: str = "https://github.com/glamod/cdm-testdata",
    branch: str = "main",
    cache: bool = True,
    cache_dir: str | Path = _default_cache_dir_,
    clear_cache: bool = False,
    within_drs: bool = True,
) -> Path:
    """
    Load file from the online Github-like repository.

    Parameters
    ----------
    name : str or os.PathLike
        Name of the file containing the dataset.
    github_url : str
        URL to GitHub repository where the data is stored.
    branch : str, optional
        For GitHub-hosted files, the branch to download from.
    cache : bool
        If True, then cache data locally for use on subsequent calls.
    cache_dir : Path
        The directory in which to search for and write cached data.
    clear_cache : bool
        If True, clear cache directory.
    within_drs : bool
        If True, then download data within data reference syntax.

    Returns
    -------
    Path
        Destination path of the downloaded file.
    """
    name = Path(name)

    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    suffix = name.suffix
    name = name.with_suffix(suffix)

    if not github_url.lower().startswith("http"):
        raise ValueError(f"GitHub URL not safe: '{github_url}'.")

    url = f"{github_url}/raw/{branch}"

    local_file = _get_file(
        name=name,
        suffix=suffix,
        url=url,
        cache_dir=cache_dir,
        clear_cache=clear_cache,
        within_drs=within_drs,
    )

    if not cache:
        local_file.unlink()
    return local_file


def get_path(path: str | Path) -> Path | None:
    """
    Get path either from _files(path) or directly from file system.

    Parameters
    ----------
    path : str | Path
        If it points to an existing file on disk, that file is returned.
        Otherwise the value is interpreted as a module name or as
        ``<module>:<subpath>`` (e.g. ``"mypkg:templates/index.html"``).

    Returns
    -------
    Path | None
        The resolved path or ``None`` if the resource cannot be found.
    """
    p = Path(path)
    if p.exists():
        return p

    try:
        return Path(str(_files(str(p))))
    except ModuleNotFoundError:
        logging.warning("No module named %s.", path)

    return None
