from __future__ import annotations

import pytest  # noqa

from cdm_reader_mapper import test_data

from ._testing_cdm_suite import _testing_suite


@pytest.mark.parametrize(
    "dname, cdm_name, cdm_subset, codes_subset, suffix, out_path, mapping, drops, mdf_kwargs",
    [
        (
            "icoads_r300_d714",
            "icoads_r300_d714",
            None,
            None,
            "063_714",
            ".",
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r300_d714",
            None,
            None,
            None,
            "063_714",
            None,
            False,
            None,
            {"sections": ["core", "c99"]},
        ),  # passing
        (
            "icoads_r300_d701",
            "icoads_r300_d701_type1",
            None,
            None,
            "069_701_type1",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r300_d701",
            "icoads_r300_d701_type2",
            None,
            None,
            "069_701_type2",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r300_d702",
            "icoads_r300_d702",
            None,
            None,
            "096_702",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r300_d703",
            "icoads_r300_d703",
            None,
            None,
            "144_703",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r300_d704",
            "icoads_r300_d704",
            None,
            None,
            "125_704",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r300_d705",
            "icoads_r300_d705",
            None,
            None,
            "085_705",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r300_d706",
            "icoads_r300_d706",
            None,
            None,
            "084_706",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r300_d707",
            "icoads_r300_d707",
            None,
            None,
            "098_707",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r300_d721",
            "icoads_r300_d721",
            None,
            None,
            "125_721",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r300_d730",
            "icoads_r300_d730",
            None,
            None,
            "133_730",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r300_d781",
            "icoads_r300_d781",
            None,
            None,
            "143_781",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r302_d794",
            "icoads_r302_d794",
            None,
            None,
            "103_794",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r300_d201",
            "icoads_r300_d201",
            None,
            None,
            "091_201",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r300_d892",
            "icoads_r300_d892",
            None,
            None,
            "077_892",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r300_d700",
            "icoads_r300_d700",
            None,
            None,
            "147_700",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r302_d792",
            "icoads_r302_d792",
            None,
            None,
            "103_792",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r302_d992",
            "icoads_r302_d992",
            None,
            None,
            "114_992",
            None,
            True,
            None,
            {},
        ),  # passing
        ("gcc", "gcc", None, None, "mix_out", None, True, None, {}),  # passing
        ("craid", "craid", None, None, "1260810", None, True, None, {}),  # passing
        (
            "icoads_r300_d714",
            "icoads_r300_d714",
            ["header", "observations-sst"],
            None,
            "063_714",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r300_d714",
            "icoads_r300_d714",
            None,
            ["platform_sub_type", "wind_direction"],
            "063_714",
            None,
            True,
            None,
            {},
        ),  # passing
        (
            "icoads_r300_d714",
            "icoads_r300_d714",
            None,
            None,
            "063_714",
            None,
            True,
            None,
            {"chunksize": 3},
        ),  # passing
        (
            "icoads_r300_d714",
            "icoads_r300_d714",
            None,
            None,
            "063_714",
            None,
            False,
            None,
            {"sections": ["c99"], "chunksize": 3},
        ),  # passing
        (
            "icoads_r300_d702",
            "icoads_r300_d702",
            None,
            None,
            "096_702",
            None,
            True,
            [0, 1, 2, 3, 4],
            {"year_init": 1874},
        ),  # passing
        (
            "icoads_r300_d702",
            "icoads_r300_d702",
            None,
            None,
            "096_702",
            None,
            True,
            [5, 6, 7, 8, 9],
            {"year_end": 1874},
        ),  # passing
        (
            "icoads_r300_d702",
            "icoads_r300_d702",
            None,
            None,
            "096_702",
            None,
            True,
            "all",
            {"year_init": 1874, "year_end": 1874},
        ),  # passing
        (
            "gcc",
            "gcc",
            None,
            None,
            "mix_out",
            None,
            True,
            [0, 1, 2, 3, 4],
            {"year_init": 2002},
        ),  # passing
        (
            "craid",
            "craid",
            None,
            None,
            "1260810",
            None,
            True,
            "all",
            {"year_end": 2003},
        ),  # passing
    ],
)
def test_read_data(
    dname,
    cdm_name,
    cdm_subset,
    codes_subset,
    suffix,
    out_path,
    mapping,
    drops,
    mdf_kwargs,
):
    _testing_suite(
        **dict(getattr(test_data, f"test_{dname}")),
        cdm_name=cdm_name,
        cdm_subset=cdm_subset,
        codes_subset=codes_subset,
        suffix=suffix,
        out_path=out_path,
        mapping=mapping,
        drops=drops,
        **mdf_kwargs,
    )
