from __future__ import annotations

import pytest  # noqa

from cdm_reader_mapper import cdm_mapper, mdf_reader, test_data

# A. TESTS TO READ FROM DATA FROM DIFFERENT DATA MODELS WITH AND WITHOUT SUPP
# -----------------------------------------------------------------------------


def test_read_imma1_buoys_nosupp(
    save_cdm=True,
):
    path_ = "."
    read_ = mdf_reader.read(**test_data.test_063_714, out_path=path_)
    data = read_.data
    attrs = read_.attrs
    output = cdm_mapper.map_model(
        "icoads_r3000_d714",
        data,
        attrs,
        log_level="DEBUG",
    )
    if save_cdm is True:
        cdm_mapper.cdm_to_ascii(output, suffix="test")
        cdm_mapper.read_tables(path_)
    assert output


def test_read_imma1_buoys_supp(plot_validation=False):
    supp_section = "c99"
    # supp_model = "cisdm_dbo_imma1"
    output = mdf_reader.read(
        **test_data.test_063_714,
        sections=[
            supp_section,
        ],
    )
    # if plot_validation:
    #    cdm.plot_model_validation(output)
    assert output


def test_read_imma1_701_type2():
    read_ = mdf_reader.read(**test_data.test_069_701)
    data = read_.data
    attrs = read_.attrs
    print(data)
    return
    assert cdm_mapper.map_model(
        "icoads_r3000_d701_type2",
        data,
        attrs,
        log_level="DEBUG",
    )


def test_read_imma1_702():
    read_ = mdf_reader.read(**test_data.test_096_702)
    data = read_.data
    attrs = read_.attrs
    assert cdm_mapper.map_model(
        "icoads_r3000_d702",
        data,
        attrs,
        log_level="DEBUG",
    )


def test_read_imma1_703():
    read_ = mdf_reader.read(**test_data.test_144_703)
    data = read_.data
    attrs = read_.attrs
    assert cdm_mapper.map_model(
        "icoads_r3000",
        data,
        attrs,
        log_level="DEBUG",
    )


def test_read_imma1_704():
    read_ = mdf_reader.read(**test_data.test_125_704)
    data = read_.data
    attrs = read_.attrs
    assert cdm_mapper.map_model(
        "icoads_r3000_d704",
        data,
        attrs,
        log_level="DEBUG",
    )


def test_read_imma1_705():
    read_ = mdf_reader.read(**test_data.test_085_705)
    data = read_.data
    attrs = read_.attrs
    assert cdm_mapper.map_model(
        "icoads_r3000_d705-707",
        data,
        attrs,
        log_level="DEBUG",
    )


def test_read_imma1_706():
    read_ = mdf_reader.read(**test_data.test_084_706)
    data = read_.data
    attrs = read_.attrs
    assert cdm_mapper.map_model(
        "icoads_r3000_d705-707",
        data,
        attrs,
        log_level="DEBUG",
    )


def test_read_imma1_707():
    read_ = mdf_reader.read(**test_data.test_098_707)
    data = read_.data
    attrs = read_.attrs
    assert cdm_mapper.map_model(
        "icoads_r3000_d705-707",
        data,
        attrs,
        log_level="DEBUG",
    )


def test_read_imma1_721():
    read_ = mdf_reader.read(**test_data.test_125_721)
    data = read_.data
    attrs = read_.attrs
    assert cdm_mapper.map_model(
        "icoads_r3000_d721",
        data,
        attrs,
        log_level="DEBUG",
    )


def test_read_imma1_730():
    read_ = mdf_reader.read(**test_data.test_133_730)
    data = read_.data
    attrs = read_.attrs
    assert cdm_mapper.map_model(
        "icoads_r3000_d730",
        data,
        attrs,
        log_level="DEBUG",
    )


def test_read_imma1_781():
    read_ = mdf_reader.read(**test_data.test_143_781)
    data = read_.data
    attrs = read_.attrs
    assert cdm_mapper.map_model(
        "icoads_r3000_d781",
        data,
        attrs,
        log_level="DEBUG",
    )

    
def test_read_imma1_794():
    read_ = mdf_reader.read(**test_data.test_103_794)
    data = read_.data
    attrs = read_.attrs
    assert cdm_mapper.map_model(
        "icoads_r3000_NRT",
        data,
        attrs,
        log_level="DEBUG",
    )

def test_read_immt_gcc():
    read_ = mdf_reader.read(**test_data.test_gcc_mix)
    data = read_.data
    attrs = read_.attrs
    assert cdm_mapper.map_model(
        "gcc_mapping",
        data,
        attrs,
        log_level="DEBUG",
    )
# B. TESTS TO READ FROM DATA FROM DIFFERENT DATA MODELS WITH
# BOTH CDM AND CODE MAPPING TABLE SUBSET
# ----------------------------------------------------------


def test_read_imma1_buoys_cdm_subset(
    plot_validation=False,
):
    read_ = mdf_reader.read(**test_data.test_063_714)
    data = read_.data
    attrs = read_.attrs
    assert cdm_mapper.map_model(
        "icoads_r3000_d714",
        data,
        attrs,
        cdm_subset=["header", "observations-sst"],
        log_level="DEBUG",
    )


def test_read_imma1_buoys_codes_subset(
    plot_validation=False,
):
    read_ = mdf_reader.read(**test_data.test_063_714)
    data = read_.data
    attrs = read_.attrs
    assert cdm_mapper.map_model(
        "icoads_r3000_d714",
        data,
        attrs,
        codes_subset=["platform_sub_type", "wind_direction"],
        log_level="DEBUG",
    )


# C. TESTS TO TEST CHUNKING
# -----------------------------------------------------------------------------
# FROM FILE: WITH AND WITHOUT SUPPLEMENTAL
def test_read_imma1_buoys_nosupp_chunks():
    chunksize = 10000
    assert mdf_reader.read(
        **test_data.test_063_714,
        chunksize=chunksize,
    )


def test_read_imma1_buoys_supp_chunks():
    chunksize = 10000
    supp_section = "c99"
    # supp_model = "cisdm_dbo_imma1"
    assert mdf_reader.read(
        **test_data.test_063_714,
        sections=[supp_section],
        chunksize=chunksize,
    )
    
test_read_imma1_701_type2()    

test_read_immt_gcc()