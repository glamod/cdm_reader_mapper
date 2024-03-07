"""pandas local file operator."""
import hashlib
import os

from pathlib import Path
from platformdirs import user_cache_dir
from urllib.request import urlretrieve
try:
    from importlib.resources import files as _files
except ImportError:
    from importlib_resources import files as _files

_default_cache_dir_ = Path(user_cache_dir("cdm-testdata"))

def _file_md5_checksum(f_name):
    hash_md5 = hashlib.md5()  # nosec
    with open(f_name, "rb") as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()

def _get_remote_file(file, url, name):
    url = "/".join(url, name)
    urlretrieve(url, file)

def _get_file(
        name: Path,
        suffix: str,
        url: str,
        cache_dir: Path,
):
    cache_dir = cache_dir.absolute()
    local_file = cache_dir / name
    md5_name = name.with_suffix(f"{suffix}.md5")
    md5_file = cache_dir / md5_name

    if not local_file.is_file():
        _get_remote_file(local_file, url, name)
    
    if not md5_file.is_file():
        _get_remote_file(md5_file, url, md5_name)
    
    local_md5 = _file_md5_checksum(local_file)
    with open(md5_file) as f:
        remote_md5 = f.read()
    
    if local_md5.strip() != remote_md5.strip():
        local_file.unlink()
        raise OSError(f"{local_file.as_posix()} and md5 checksum do not match.")
    
    return local_file

def load_file(
        name: str | os.PathLike,
        github_url: str = "https://github.com/glamod/cdm-testdata",
        branch: str = "main",
        mode: str = "input",
        suffix: str = "imma",
        cache: bool = True,
        cache_dir: Path = _default_cache_dir_,
):
    """Load file from the online Github-like repository.
    
    Parameters
    ---------- 
    
    Returns
    -------
    None.
    """   
    if isinstance(name, str):
        name = Path(name)
        
    if not github_url.lower().startswith("http"):
        raise ValueError(f"GitHub URL not safe: '{github_url}'.")
    return

    url = "/".join(github_url, "raw", branch, mode)
    local_file = _get_file(
        name = name,
        suffix = suffix,
        url = url,
        cache_dir = cache_dir,
    )
    if not cache:
        local_file.unlink()
    return local_file

def get_files(anchor):
    """Get files."""
    return _files(anchor)