from __future__ import annotations

import os

import pytest  # noqa

from cdm_reader_mapper import test_data


# A. TESTS TO READ FROM DATA FROM DIFFERENT DATA MODELS WITH AND WITHOUT SUPP
# -----------------------------------------------------------------------------
def test_cli_imma1_buoys_nosupp():
    source = test_data.test_063_714["source"]
    data_model = test_data.test_063_714["data_model"]
    s = "mdf_reader " f"{source} " f"--data_model {data_model} " "--out_path ."
    os.system(s)


def test_cli_imma1_buoys_supp():
    source = test_data.test_063_714["source"]
    data_model = test_data.test_063_714["data_model"]
    supp_section = "c99"
    s = (
        "mdf_reader "
        f"{source} "
        f"--data_model {data_model} "
        f"--sections {supp_section} "
        "--out_path ."
    )
    os.system(s)


# B. TESTS TO TEST CHUNKING
# -----------------------------------------------------------------------------
# FROM FILE: WITH AND WITHOUT SUPPLEMENTAL


def test_cli_imma1_buoys_nosupp_chunks():
    source = test_data.test_063_714["source"]
    data_model = test_data.test_063_714["data_model"]
    chunksize = 10000
    s = (
        "mdf_reader "
        f"{source} "
        f"--data_model {data_model} "
        f"--chunksize {chunksize} "
        "--out_path ."
    )
    os.system(s)


def test_cli_imma1_buoys_supp_chunks():
    source = test_data.test_063_714["source"]
    data_model = test_data.test_063_714["data_model"]
    supp_section = "c99"
    chunksize = 10000
    s = (
        "mdf_reader "
        f"{source} "
        f"--data_model {data_model} "
        f"--sections {supp_section} "
        f"--chunksize {chunksize} "
        "--out_path ."
    )
    os.system(s)
