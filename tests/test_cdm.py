from __future__ import annotations

import pytest  # noqa

from cdm_reader_mapper import test_data

from ._testing_cdm_suite import _testing_suite


@pytest.mark.parametrize(
    "dname, cdm_name, cdm_subset, codes_subset, suffix, out_path, mapping, mdf_kwargs",
    [
        ("063_714", "icoads_r3000_d714", None, None, "063_714", ".", True, {}),
        #(
        #    "063_714",
        #    None,
        #    None,
        #    None,
        #    "063_714",
        #    None,
        #    False,
        #    {"sections": ["core", "c99"]},
        #),
        #(
        #    "069_701",
        #    "icoads_r3000_d701_type1",
        #    None,
        #    None,
        #    "069_701_type1",
        #    None,
        #    True,
        #    {},
        #),
        #(
        #    "069_701",
        #    "icoads_r3000_d701_type2",
        #    None,
        #    None,
        #    "069_701_type2",
        #    None,
        #    True,
        #    {},
        #),
        #("096_702", "icoads_r3000_d702", None, None, "096_702", None, True, {}),
        #("144_703", "icoads_r3000", None, None, "144_703", None, True, {}),
        #("125_704", "icoads_r3000_d704", None, None, "125_704", None, True, {}),
        #("085_705", "icoads_r3000_d705-707", None, None, "085_705", None, True, {}),
        #("084_706", "icoads_r3000_d705-707", None, None, "084_706", None, True, {}),
        #("098_707", "icoads_r3000_d705-707", None, None, "098_707", None, True, {}),
        #("125_721", "icoads_r3000_d721", None, None, "125_721", None, True, {}),
        #("133_730", "icoads_r3000_d730", None, None, "133_730", None, True, {}),
        #("143_781", "icoads_r3000_d781", None, None, "143_781", None, True, {}),
        #("103_794", "icoads_r3000_NRT", None, None, "103_794", None, True, {}),
        #("091_201", "icoads_r3000", None, None, "091_201", None, True, {}),
        #("077_892", "icoads_r3000", None, None, "077_892", None, True, {}),
        #("147_700", "icoads_r3000", None, None, "147_700", None, True, {}),
        #("103_792", "icoads_r3000_NRT", None, None, "103_792", None, True, {}),
        #("114_992", "icoads_r3000_NRT", None, None, "114_992", None, True, {}),
        #("gcc_mix", "gcc_mapping", None, None, "mix_out", None, True, {}),
        #("craid_1260810", "c_raid", None, None, "craid", None, True, {}),
        #(
        #    "063_714",
        #    "icoads_r3000_d714",
        #    ["header", "observations-sst"],
        #    None,
        #    "063_714",
        #    None,
        #    True,
        #    {},
        #),
        #(
        #    "063_714",
        #    "icoads_r3000_d714",
        #    None,
        #    ["platform_sub_type", "wind_direction"],
        #    "063_714",
        #    None,
        #    True,
        #    {},
        #),
        #(
        #    "063_714",
        #    "icoads_r3000_d714",
        #    None,
        #    None,
        #    "063_714",
        #    None,
        #    True,
        #    {"chunksize": 3},
        #),
        #(
        #    "063_714",
        #    "icoads_r3000_d714",
        #    None,
        #    None,
        #    "063_714",
        #    None,
        #    False,
        #    {"sections": ["c99"], "chunksize": 3},
        #),
    ],
)
def test_read_data(
    dname, cdm_name, cdm_subset, codes_subset, suffix, out_path, mapping, mdf_kwargs
):
    _testing_suite(
        **dict(getattr(test_data, f"test_{dname}")),
        cdm_name=cdm_name,
        cdm_subset=cdm_subset,
        codes_subset=codes_subset,
        suffix=suffix,
        out_path=out_path,
        mapping=mapping,
        **mdf_kwargs,
    )
