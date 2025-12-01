from __future__ import annotations

import pandas as pd
import pytest
from io import StringIO

from cdm_reader_mapper import DataBundle


@pytest.fixture
def sample_df():
    """Simple DataFrame for testing."""
    data = pd.DataFrame({"A": [1, 2, None], "B": ["x", "y", "z"]})
    return DataBundle(data=data)


@pytest.fixture
def sample_text_reader():
    """Fixture that returns a TextFileReader."""
    csv_data = "A,B\n1,x\n2,y\n, z"
    data = pd.read_csv(StringIO(csv_data), chunksize=1)
    return DataBundle(data=data)


@pytest.mark.parametrize("fixture_name", ["sample_df", "sample_text_reader"])
def test_index(request, fixture_name):
    obj = request.getfixturevalue(fixture_name)
    assert list(obj.index) == [0, 1, 2]


@pytest.mark.parametrize("fixture_name", ["sample_df", "sample_text_reader"])
def test_size(request, fixture_name):
    obj = request.getfixturevalue(fixture_name)
    assert obj.size == 6


@pytest.mark.parametrize("fixture_name", ["sample_df", "sample_text_reader"])
def test_shape(request, fixture_name):
    obj = request.getfixturevalue(fixture_name)
    assert obj.shape == (3, 2)


@pytest.mark.parametrize("fixture_name", ["sample_df", "sample_text_reader"])
def test_dropna(request, fixture_name):
    obj = request.getfixturevalue(fixture_name)

    dropped = obj.dropna()
    if not isinstance(dropped, pd.DataFrame):
        dropped = dropped.read()

    assert dropped.shape == (2, 2)
    assert dropped["A"].isna().sum() == 0


@pytest.mark.parametrize("fixture_name", ["sample_df", "sample_text_reader"])
def test_rename(request, fixture_name):
    obj = request.getfixturevalue(fixture_name)

    renamed = obj.rename(columns={"A": "A_new"})
    if not isinstance(renamed, pd.DataFrame):
        renamed = renamed.read()

    assert "A_new" in renamed.columns
    assert "A" not in renamed.columns


@pytest.mark.parametrize("fixture_name", ["sample_df", "sample_text_reader"])
def test_rename_inplace(request, fixture_name):
    obj = request.getfixturevalue(fixture_name)

    obj.rename(columns={"A": "A_new"}, inplace=True)

    assert "A_new" in obj.columns


@pytest.mark.parametrize("fixture_name", ["sample_df", "sample_text_reader"])
def test_iloc(request, fixture_name):
    obj = request.getfixturevalue(fixture_name)

    if fixture_name == "sample_df":
        first_row = obj.iloc[0]
        assert first_row["A"] == 1
        assert first_row["B"] == "x"
    else:
        with pytest.raises(NotImplementedError):
            _ = obj.iloc[0]
