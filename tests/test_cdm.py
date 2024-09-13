from __future__ import annotations

import pytest  # noqa

from cdm_reader_mapper import test_data

from ._testing_cdm_suite import _testing_suite


@pytest.mark.parametrize(
    "dname, cdm_name, cdm_subset, codes_subset, suffix, out_path, mapping, drops, mdf_kwargs",
    [
        ("096_702", "icoads_r3000_d702", None, None, "096_702", None, True, {}),
        (
            "096_702",
            "icoads_r3000_d702",
            None,
            None,
            "096_702",
            None,
            True,
            [0, 1, 2, 3, 4],
            {"year_init": 1874},
        ),
        (
            "096_702",
            "icoads_r3000_d702",
            None,
            None,
            "096_702",
            None,
            True,
            [5, 6, 7, 8, 9],
            {"year_end": 1874},
        ),
        (
            "096_702",
            "icoads_r3000_d702",
            None,
            None,
            "096_702",
            None,
            True,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            {"year_init": 1874, "year_end": 1874},
        ),
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
