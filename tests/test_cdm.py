from __future__ import annotations

import pytest  # noqa

from cdm_reader_mapper import test_data

from ._testing_cdm_suite import _testing_suite


def test_read_imma1_buoys_nosupp():
    _testing_suite(
        **test_data.test_063_714,
        cdm_name="icoads_r3000_d714",
        out_path=".",
        suffix="063_714_nosupp",
    )


def test_read_imma1_buoys_supp():
    _testing_suite(
        **test_data.test_063_714,
        sections="c99",
        mapping=False,
    )


def test_read_imma1_701_type1():
    _testing_suite(
        **test_data.test_069_701,
        cdm_name="icoads_r3000_d701_type1",
        suffix="069-701_type1_nosupp",
    )


def test_read_imma1_701_type2():
    _testing_suite(
        **test_data.test_069_701,
        cdm_name="icoads_r3000_d701_type2",
        suffix="069-701_type2_nosupp",
    )


def test_read_imma1_702():
    _testing_suite(
        **test_data.test_096_702,
        cdm_name="icoads_r3000_d702",
        suffix="096-702_nosupp",
    )


def test_read_imma1_703():
    _testing_suite(
        **test_data.test_144_703,
        cdm_name="icoads_r3000",
        suffix="144-703_nosupp",
    )


def test_read_imma1_704():
    _testing_suite(
        **test_data.test_125_704,
        cdm_name="icoads_r3000_d704",
        suffix="125-704_nosupp",
    )


def test_read_imma1_705():
    _testing_suite(
        **test_data.test_085_705,
        cdm_name="icoads_r3000_d705-707",
        suffix="069-705_nosupp",
    )


def test_read_imma1_706():
    _testing_suite(
        **test_data.test_084_706,
        cdm_name="icoads_r3000_d705-707",
        suffix="084-706_nosupp",
    )


def test_read_imma1_707():
    _testing_suite(
        **test_data.test_098_707,
        cdm_name="icoads_r3000_d705-707",
        suffix="098-707_nosupp",
    )


def test_read_imma1_721():
    _testing_suite(
        **test_data.test_125_721,
        cdm_name="icoads_r3000_d721",
        suffix="125-721_nosupp",
    )


def test_read_imma1_730():
    _testing_suite(
        **test_data.test_133_730,
        cdm_name="icoads_r3000_d730",
        suffix="133-730_nosupp",
    )


def test_read_imma1_781():
    _testing_suite(
        **test_data.test_143_781,
        cdm_name="icoads_r3000_d781",
        suffix="069-781_nosupp",
    )


def test_read_imma1_794():
    _testing_suite(
        **test_data.test_103_794,
        cdm_name="icoads_r3000",
        suffix="103-794_nosupp",
    )


def test_read_immt_gcc():
    "NOT WORKING: cdm_mapper in original version"
    _testing_suite(
        **test_data.test_gcc_mix,
        cdm_name="gcc_mapping",
        suffix="mix_out_nosupp",
    )


# B. TESTS TO READ FROM DATA FROM DIFFERENT DATA MODELS WITH
# BOTH CDM AND CODE MAPPING TABLE SUBSET
# ----------------------------------------------------------


def test_read_imma1_buoys_cdm_subset():
    _testing_suite(
        **test_data.test_063_714,
        cdm_name="icoads_r3000_d714",
        cdm_subset=["header", "observations-sst"],
        suffix="063-714_cdms",
    )


def test_read_imma1_buoys_codes_subset():
    "NOT WORKING: codes_subset not in map_maodel"
    _testing_suite(
        **test_data.test_063_714,
        cdm_name="icoads_r3000_d714",
        codes_subset=["platform_sub_type", "wind_direction"],
        suffix="063-714_codes",
    )


# C. TESTS TO TEST CHUNKING
# -----------------------------------------------------------------------------
# FROM FILE: WITH AND WITHOUT SUPPLEMENTAL
def test_read_imma1_buoys_nosupp_chunks():
    "NOT WORKING: textfilereader"
    _testing_suite(
        **test_data.test_063_714,
        cdm_name="icoads_r3000_d714",
        suffix="063-714_chunk",
        chunksize=10000,
    )


def test_read_imma1_buoys_supp_chunks():
    "NOT WORKING: textfilereader"
    _testing_suite(
        **test_data.test_063_714,
        sections="c99",
        chunksize=10000,
        mapping=False,
    )
