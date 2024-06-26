from __future__ import annotations

import pytest  # noqa

from cdm_reader_mapper import test_data

from ._testing_cdm_suite import _testing_suite


def test_read_imma1_714():
    _testing_suite(
        **dict(test_data.test_063_714),
        cdm_name="icoads_r3000_d714",
        out_path=".",
        suffix="063_714",
    )


def test_read_imma1_714_supp():
    _testing_suite(
        **dict(test_data.test_063_714),
        sections=["core", "c99"],
        suffix="063_714",
        mapping=False,
    )


def test_read_imma1_701_type1():
    _testing_suite(
        **dict(test_data.test_069_701),
        cdm_name="icoads_r3000_d701_type1",
        suffix="069_701_type1",
    )


def test_read_imma1_701_type2():
    _testing_suite(
        **dict(test_data.test_069_701),
        cdm_name="icoads_r3000_d701_type2",
        suffix="069_701_type2",
    )


def test_read_imma1_702():
    _testing_suite(
        **dict(test_data.test_096_702),
        cdm_name="icoads_r3000_d702",
        suffix="096_702",
    )


def test_read_imma1_703():
    _testing_suite(
        **dict(test_data.test_144_703),
        cdm_name="icoads_r3000",
        suffix="144_703",
    )


def test_read_imma1_704():
    _testing_suite(
        **dict(test_data.test_125_704),
        cdm_name="icoads_r3000_d704",
        suffix="125_704",
    )


def test_read_imma1_705():
    _testing_suite(
        **dict(test_data.test_085_705),
        cdm_name="icoads_r3000_d705-707",
        suffix="085_705",
    )


def test_read_imma1_706():
    _testing_suite(
        **dict(test_data.test_084_706),
        cdm_name="icoads_r3000_d705-707",
        suffix="084_706",
    )


def test_read_imma1_707():
    _testing_suite(
        **dict(test_data.test_098_707),
        cdm_name="icoads_r3000_d705-707",
        suffix="098_707",
    )


def test_read_imma1_721():
    _testing_suite(
        **dict(test_data.test_125_721),
        cdm_name="icoads_r3000_d721",
        suffix="125_721",
    )


def test_read_imma1_730():
    _testing_suite(
        **dict(test_data.test_133_730),
        cdm_name="icoads_r3000_d730",
        suffix="133_730",
    )


def test_read_imma1_781():
    _testing_suite(
        **dict(test_data.test_143_781),
        cdm_name="icoads_r3000_d781",
        suffix="143_781",
    )


def test_read_imma1_794():
    _testing_suite(
        **dict(test_data.test_103_794),
        cdm_name="icoads_r3000_NRT",
        suffix="103_794",
    )


def test_read_imma1_201():
    _testing_suite(
        **dict(test_data.test_091_201),
        cdm_name="icoads_r3000",
        suffix="091_201",
    )


def test_read_imma1_892():
    _testing_suite(
        **dict(test_data.test_077_892),
        cdm_name="icoads_r3000",
        suffix="077_892",
    )


def test_read_imma1_700():
    _testing_suite(
        **dict(test_data.test_147_700),
        cdm_name="icoads_r3000",
        suffix="147_700",
    )


def test_read_imma1_792():
    _testing_suite(
        **dict(test_data.test_103_792),
        cdm_name="icoads_r3000_NRT",
        suffix="103_792",
    )


def test_read_imma1_992():
    _testing_suite(
        **dict(test_data.test_114_992),
        cdm_name="icoads_r3000_NRT",
        suffix="114_992",
    )


def test_read_immt_gcc():
    _testing_suite(
        **dict(test_data.test_gcc_mix),
        cdm_name="gcc_mapping",
        suffix="mix_out",
    )


def test_read_craid_1260810():
    _testing_suite(
        **dict(test_data.test_craid_1260810),
        cdm_name="c_raid",
        suffix="craid",
    )


# B. TESTS TO READ FROM DATA FROM DIFFERENT DATA MODELS WITH
# BOTH CDM AND CODE MAPPING TABLE SUBSET
# ----------------------------------------------------------


def test_read_imma1_714_cdm_subset():
    _testing_suite(
        **dict(test_data.test_063_714),
        cdm_name="icoads_r3000_d714",
        cdm_subset=["header", "observations-sst"],
        suffix="063_714",
    )


def test_read_imma1_714_codes_subset():
    _testing_suite(
        **dict(test_data.test_063_714),
        cdm_name="icoads_r3000_d714",
        codes_subset=["platform_sub_type", "wind_direction"],
        suffix="063_714",
    )


# C. TESTS TO TEST CHUNKING
# -----------------------------------------------------------------------------
# FROM FILE: WITH AND WITHOUT SUPPLEMENTAL
def test_read_imma1_714_nosupp_chunks():
    _testing_suite(
        **dict(test_data.test_063_714),
        cdm_name="icoads_r3000_d714",
        suffix="063_714",
        chunksize=3,
    )


def test_read_imma1_714_supp_chunks():
    _testing_suite(
        **dict(test_data.test_063_714),
        sections=["c99"],
        suffix="063_714",
        chunksize=3,
        mapping=False,
    )
