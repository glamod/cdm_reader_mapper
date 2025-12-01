"""
Initialize logger.

Created on Wed Apr  3 08:45:03 2019

@author: iregon
"""

from __future__ import annotations

import logging
import os

LOG_FN = os.getenv("CDM_LOG_FILE", None)


def init_logger(module, level="INFO", fn=LOG_FN):
    """
    Initialize and configure a logger for a given module.

    Parameters
    ----------
    module : str
        Name of the module or logger.
    level : str, default 'INFO'
        Logging level as string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    fn : str or None, optional
        Optional filename to write logs to. If None, logs go to stdout.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    This function calls `logging.basicConfig` to configure the root logger.
    Repeated calls to this function may not reconfigure logging unless `reload(logging)` is used.
    """
    from importlib import reload

    reload(logging)

    level = logging.getLevelName(level.upper())
    logging_params = {
        "level": level,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }

    if fn is not None:
        logging_params["filename"] = fn

    logging.basicConfig(**logging_params)
    logging.info("Initialized basic logging configuration successfully")

    return logging.getLogger(module)
