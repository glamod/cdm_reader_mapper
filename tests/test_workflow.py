from __future__ import annotations

import os

import pytest  # noqa

from cdm_reader_mapper import test_data

from ._testing_workflow_suite import _testing_suite


@pytest.mark.parametrize(
    "imodel, cdm_subset, codes_subset, mapping, drops, mdf_kwargs",
    [
        ("icoads_r300_d714", None, None, True, None, {}),
        (
            "icoads_r300_d714",
            None,
            None,
            False,
            None,
            {"sections": ["core", "c99"]},
        ),
        ("icoads_r300_d701", None, None, True, None, {}),
        ("icoads_r300_d702", None, None, True, None, {}),
        ("icoads_r300_d703", None, None, True, None, {}),
        ("icoads_r300_d704", None, None, True, None, {}),
        ("icoads_r300_d705", None, None, True, None, {}),
        ("icoads_r300_d706", None, None, True, None, {}),
        ("icoads_r300_d707", None, None, True, None, {}),
        ("icoads_r300_d721", None, None, True, None, {}),
        ("icoads_r300_d730", None, None, True, None, {}),
        ("icoads_r300_d781", None, None, True, None, {}),
        ("icoads_r302_d794", None, None, True, None, {}),
        ("icoads_r300_d201", None, None, True, None, {}),
        ("icoads_r300_d892", None, None, True, None, {}),
        ("icoads_r300_d700", None, None, True, None, {}),
        ("icoads_r302_d792", None, None, True, None, {}),
        ("icoads_r302_d992", None, None, True, None, {}),
        ("gdac", None, None, True, None, {}),
        ("craid", None, None, True, None, {}),
        (
            "icoads_r300_d714",
            ["header", "observations-sst"],
            None,
            True,
            None,
            {},
        ),
        (
            "icoads_r300_d714",
            None,
            ["platform_sub_type", "ship_speed_ms"],
            True,
            None,
            {},
        ),
        ("icoads_r300_d714", None, None, True, None, {"chunksize": 3}),
        (
            "icoads_r300_d714",
            None,
            None,
            False,
            None,
            {"sections": ["c99"], "chunksize": 3},
        ),
        ("icoads_r300_d721", None, None, True, None, {"chunksize": 3}),
        (
            "icoads_r300_d702",
            None,
            None,
            True,
            [0, 1, 2, 3, 4],
            {"year_init": 1874},
        ),
        (
            "icoads_r300_d702",
            None,
            None,
            True,
            [5, 6, 7, 8, 9],
            {"year_end": 1874},
        ),
        (
            "icoads_r300_d702",
            None,
            None,
            True,
            "all",
            {"year_init": 1874, "year_end": 1874},
        ),
        (
            "gdac",
            None,
            None,
            True,
            [0, 1, 2, 3, 4],
            {"year_init": 2002},
        ),
        (
            "craid",
            None,
            None,
            True,
            "all",
            {"year_end": 2003},
        ),
        (
            "icoads_r300_d703",
            None,
            None,
            False,
            None,
            {
                "ext_schema_path": os.path.join(
                    ".", "cdm_reader_mapper", "mdf_reader", "schemas", "icoads"
                )
            },
        ),
        (
            "icoads_r300_d703",
            None,
            None,
            False,
            None,
            {
                "ext_table_path": os.path.join(
                    ".", "cdm_reader_mapper", "mdf_reader", "codes", "icoads"
                )
            },
        ),
        (
            "icoads_r300_d703",
            None,
            None,
            False,
            None,
            {
                "ext_table_path": os.path.join(
                    ".", "cdm_reader_mapper", "mdf_reader", "codes", "icoads"
                ),
                "ext_schema_path": os.path.join(
                    ".", "cdm_reader_mapper", "mdf_reader", "schemas", "icoads"
                ),
            },
        ),
        (
            "icoads_r300_d703",
            None,
            None,
            False,
            None,
            {
                "ext_schema_file": os.path.join(
                    ".",
                    "cdm_reader_mapper",
                    "mdf_reader",
                    "schemas",
                    "icoads",
                    "icoads.json",
                )
            },
        ),
        (
            "icoads_r300_mixed",
            None,
            None,
            False,
            None,
            {"encoding": "cp1252"},
        ),
    ],
)
def test_read_data(
    imodel,
    cdm_subset,
    codes_subset,
    mapping,
    drops,
    mdf_kwargs,
):
    _testing_suite(
        **dict(getattr(test_data, f"test_{imodel}")),
        imodel=imodel,
        cdm_subset=cdm_subset,
        codes_subset=codes_subset,
        mapping=mapping,
        drops=drops,
        **mdf_kwargs,
    )
