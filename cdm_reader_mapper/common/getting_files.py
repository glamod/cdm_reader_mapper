"""pandas local file operator."""
from __future__ import annotations

import hashlib
import logging
import os
import warnings
from pathlib import Path
from urllib.request import urlretrieve

from platformdirs import user_cache_dir

try:
    from importlib.resources import files as _files
except ImportError:
    from importlib_resources import files as _files

_default_cache_dir_ = Path(user_cache_dir("cdm-testdata", ".cache"))


def _file_md5_checksum(f_name):
    hash_md5 = hashlib.md5()  # nosec
    with open(f_name, "rb") as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()


def _get_remote_file(lfile, url, name):
    url = "/".join((url, name.as_posix()))
    lfile.parent.mkdir(exist_ok=True, parents=True)
    msg = f"Attempting to fetch remote file: {name.as_posix()}"
    logging.info(msg)
    return urlretrieve(url, lfile)


def _check_md5s(f, md5, mode="error"):
    msg = f"{f.as_posix()} and md5 checksum do not match."
    md5_ = _file_md5_checksum(f)
    if md5_.strip() != md5.strip():
        f.unlink()
        if mode == "error":
            raise OSError(msg)
        elif mode == "warning":
            warnings.warn(msg)
            return
    return True


def _get_file(
    name: Path,
    suffix: str,
    url: str,
    cache_dir: Path,
):
    cache_dir = cache_dir.absolute()
    cache_dir.mkdir(exist_ok=True, parents=True)
    local_file = cache_dir / name
    md5_name = name.with_suffix(f"{suffix}.md5")
    md5_file = cache_dir / md5_name

    _get_remote_file(md5_file, url, md5_name)
    with open(md5_file) as f:
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


# idea copied from xclim that it raven that it borrowed from xclim that borrowed it from xarray that was borrowed from Seaborn
def load_file(
    name: str | os.PathLike,
    github_url: str = "https://github.com/glamod/cdm-testdata",
    branch: str = "main",
    cache: bool = True,
    cache_dir: Path = _default_cache_dir_,
):
    """Load file from the online Github-like repository.

    Parameters
    ----------
    name : str or os.PathLike
        Name of the file containing the dataset.
    github_url : str
        URL to GitHub repository where the data is stored.
    branch : str, optional
        For GitHub-hosted files, the branch to download from.
    cache_dir : Path
        The directory in which to search for and write cached data.
    cache : bool
        If True, then cache data locally for use on subsequent calls.

    Returns
    -------
    Path
    """
    if isinstance(name, str):
        name = Path(name)

    suffix = name.suffix
    name = name.with_suffix(suffix)

    if not github_url.lower().startswith("http"):
        raise ValueError(f"GitHub URL not safe: '{github_url}'.")
        return

    url = "/".join((github_url, "raw", branch))

    local_file = _get_file(
        name=name,
        suffix=suffix,
        url=url,
        cache_dir=cache_dir,
    )

    if not cache:
        local_file.unlink()
    return local_file


def get_files(anchor):
    """Get files."""
    return _files(anchor)
