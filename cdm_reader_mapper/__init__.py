"""Common Data Model (CDM) reader and mapper package."""
from __future__ import annotations

from . import cdm_mapper  # noqa
from . import common  # noqa
from . import mdf_reader  # noqa
from . import metmetpy  # noqa
from . import operations  # noqa
from .data import test_data  # noqa


def _get_version():
    """Test."""
    __version__ = "unknown"
    try:
        from ._version import __version__
    except ImportError:
        pass
    return __version__


__author__ = """Ludwig Lierhammer"""
__email__ = "ludwiglierhammer@dwd.de"
__version__ = _get_version()
