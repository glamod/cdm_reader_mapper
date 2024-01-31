"""Console script for cdm_reader_mapper."""

from __future__ import annotations

import argparse
import sys

import cdm
from dask.distributed import Client


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        nargs="?",
        help="File path to read.",
    )
    parser.add_argument(
        "-dm",
        "--data_model",
        dest="data_model",
        nargs="?",
        help="Name of internally available data_model.",
    )
    parser.add_argument(
        "-dmp",
        "--data_model_path",
        dest="data_model_path",
        nargs="?",
        help="Path to externally available data_model.",
    )
    parser.add_argument(
        "-s",
        "--sections",
        dest="sections",
        nargs="+",
        help="List with subsets of data model sections.",
    )
    parser.add_argument(
        "-cs",
        "--chunksize",
        dest="chunksize",
        type=int,
        help="Numer of reports per chunk.",
    )
    parser.add_argument(
        "-sp",
        "--skiprows",
        dest="skiprows",
        type=int,
        default=0,
        help="Number of initial rows to skip from file.",
    )
    parser.add_argument(
        "-op",
        "--out_path",
        dest="out_path",
        nargs="?",
        help="Path to output data, valid mask and attributes.",
    )
    return parser


def _args_to_reader_mapper(args):
    return cdm.read(**vars(args))


def main():
    """Execute main routine for cdm_reader_mapper command-line interface."""
    parser = _parser()
    args = parser.parse_args()

    _args_to_reader_mapper(args)
    return 0


if __name__ == "__main__":
    with Client() as client:
        sys.exit(main())
