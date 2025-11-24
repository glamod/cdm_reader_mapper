from __future__ import annotations

import pytest  # noqa

from cdm_reader_mapper.cdm_mapper import properties
from cdm_reader_mapper.cdm_mapper.tables.tables import get_cdm_atts, get_imodel_maps


def _assert_dict_keys(d: dict, expected_keys: str | list | set):
    """Assert that the dictionary `d` is a dictionary and has exactly the keys in `expected_keys`."""
    assert isinstance(d, dict)

    actual_keys = set(d.keys())

    if isinstance(expected_keys, str):
        expected_keys_set = {expected_keys}
    else:
        expected_keys_set = set(expected_keys)

    assert actual_keys == expected_keys_set, (
        f"Unexpected keys: {actual_keys - expected_keys_set}, "
        f"Missing keys: {expected_keys_set - actual_keys}"
    )


@pytest.mark.parametrize(
    "cdm_tables",
    [
        None,
        [],
        "header",
        ["header", "observations-at"],
    ],
)
def test_get_cdm_atts(cdm_tables):
    expected_tables = properties.cdm_tables if cdm_tables is None else cdm_tables

    cdm_atts = get_cdm_atts(cdm_tables)
    _assert_dict_keys(cdm_atts, expected_tables)


@pytest.mark.parametrize(
    "dataset,cdm_tables",
    [
        ("icoads", ["header", "observations"]),
        ("icoads_r302", ["header"]),
        ("icoads_r302_d992", ["observations"]),
        ("icoads", []),
        ("icoads", None),
        ("icoads_r302", ["observations-at"]),
    ],
)
def test_get_imodel_maps(dataset, cdm_tables):
    expected_tables = properties.cdm_tables if cdm_tables is None else cdm_tables

    imaps = get_imodel_maps(dataset.split("_"), cdm_tables=cdm_tables)
    _assert_dict_keys(imaps, expected_tables)

    if "observations-at" in imaps:
        for v in imaps["observations-at"].values():
            elements = v.get("elements", [])
            assert isinstance(elements, list)
