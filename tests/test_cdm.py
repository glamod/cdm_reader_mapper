from __future__ import annotations

import pytest  # noqa

from cdm_reader_mapper import test_data

from ._testing_cdm_suite import _testing_suite
#from _testing_cdm_suite import _testing_suite

def test_read_imma1_buoys_nosupp():
    _testing_suite(
        **test_data.test_063_714,
        cdm_name="icoads_r3000_d714",
        out_path=".",
        suffix="063_714_nosupp",
    )

#test_read_imma1_buoys_nosupp()
def test_read_imma1_buoys_supp():
    _testing_suite(
        **test_data.test_063_714,
        sections="c99",
        suffix="063_714_nosupp",
        mapping=False,
    )


def test_read_imma1_701_type1():
    _testing_suite(
        **test_data.test_069_701,
        cdm_name="icoads_r3000_d701_type1",
        suffix="069_701_type1_nosupp",
    )
#test_read_imma1_701_type1()

def test_read_imma1_701_type2():
    _testing_suite(
        **test_data.test_069_701,
        cdm_name="icoads_r3000_d701_type2",
        suffix="069_701_type2_nosupp",
    )


def test_read_imma1_702():
    _testing_suite(
        **test_data.test_096_702,
        cdm_name="icoads_r3000_d702",
        suffix="096_702_nosupp",
    )


def test_read_imma1_703():
    _testing_suite(
        **test_data.test_144_703,
        cdm_name="icoads_r3000",
        suffix="144_703_nosupp",
    )


def test_read_imma1_704():
    _testing_suite(
        **test_data.test_125_704,
        cdm_name="icoads_r3000_d704",
        suffix="125_704_nosupp",
    )


def test_read_imma1_705():
    _testing_suite(
        **test_data.test_085_705,
        cdm_name="icoads_r3000_d705-707",
        suffix="085_705_nosupp",
    )


def test_read_imma1_706():
    _testing_suite(
        **test_data.test_084_706,
        cdm_name="icoads_r3000_d705-707",
        suffix="084_706_nosupp",
    )


def test_read_imma1_707():
    _testing_suite(
        **test_data.test_098_707,
        cdm_name="icoads_r3000_d705-707",
        suffix="098_707_nosupp",
    )


def test_read_imma1_721():
    _testing_suite(
        **test_data.test_125_721,
        cdm_name="icoads_r3000_d721",
        suffix="125_721_nosupp",
    )


def test_read_imma1_730():
    _testing_suite(
        **test_data.test_133_730,
        cdm_name="icoads_r3000_d730",
        suffix="133_730_nosupp",
    )


def test_read_imma1_781():
    _testing_suite(
        **test_data.test_143_781,
        cdm_name="icoads_r3000_d781",
        suffix="143_781_nosupp",
    )


def test_read_imma1_794():
    _testing_suite(
        **test_data.test_103_794,
        cdm_name="icoads_r3000",
        suffix="103_794_nosupp",
    )
#test_read_imma1_794()

# def test_read_immt_gcc():
#    "NOT WORKING: cdm_mapper in original version"
#    _testing_suite(
#        **test_data.test_gcc_mix,
#        cdm_name="gcc_mapping",
#        suffix="mix_out_nosupp",
#    )


# B. TESTS TO READ FROM DATA FROM DIFFERENT DATA MODELS WITH
# BOTH CDM AND CODE MAPPING TABLE SUBSET
# ----------------------------------------------------------


def test_read_imma1_buoys_cdm_subset():
    _testing_suite(
        **test_data.test_063_714,
        cdm_name="icoads_r3000_d714",
        cdm_subset=["header", "observations-sst"],
        suffix="063_714_nosupp",
    )


def test_read_imma1_buoys_codes_subset():
    "NOT WORKING: codes_subset not in map_model"
    _testing_suite(
        **test_data.test_063_714,
        cdm_name="icoads_r3000_d714",
        codes_subset=["platform_sub_type", "wind_direction"],
        suffix="063_714_nosupp",
    )


# C. TESTS TO TEST CHUNKING
# -----------------------------------------------------------------------------
# FROM FILE: WITH AND WITHOUT SUPPLEMENTAL
def test_read_imma1_buoys_nosupp_chunks():
    "NOT WORKING: textfilereader"
    _testing_suite(
        **test_data.test_063_714,
        cdm_name="icoads_r3000_d714",
        suffix="063_714_nosupp",
        chunksize=10000,
    )
#test_read_imma1_buoys_nosupp_chunks()

def test_read_imma1_buoys_supp_chunks():
    _testing_suite(
        **test_data.test_063_714,
        sections="c99",
        suffix="063_714_nosupp",
        chunksize=10000,
        mapping=False,
    )
#test_read_imma1_buoys_supp_chunks()