from __future__ import annotations

import pandas as pd
import pytest

from io import StringIO

from cdm_reader_mapper import DataBundle


def make_parser(text, **kwargs):
    """Helper: create a TextFileReader similar to user code."""
    buffer = StringIO(text)
    return pd.read_csv(buffer, chunksize=2, **kwargs)


@pytest.fixture
def sample_db_df():
    data = pd.DataFrame(
        {
            "A": [19, 26, 27, 41, 91],
            "B": [0, 1, 2, 3, 4],
        }
    )
    mask = pd.DataFrame(
        {
            "A": [True, True, True, False, True],
            "B": [True, True, True, False, False],
        }
    )
    return DataBundle(data=data, mask=mask)


@pytest.fixture
def sample_db_reader():
    text = "19,0\n26,1\n27,2\n41,3\n91,4"
    reader_data = make_parser(text, names=["A", "B"])

    text = "True,True\nTrue,True\nTrue,True\nFalse,False\nTrue,False"
    reader_mask = make_parser(text, names=["A", "B"])

    return DataBundle(data=reader_data, mask=reader_mask)


@pytest.fixture
def sample_db_reader_multi():
    text = "19,0\n26,1\n27,2\n41,3\n91,4"
    reader_data = make_parser(text, names=[("A", "a"), ("B", "b")])

    text = "True,True\nTrue,True\nTrue,True\nFalse,False\nTrue,False"
    reader_mask = make_parser(text, names=["A", "B"])

    return DataBundle(data=reader_data, mask=reader_mask)


@pytest.fixture
def sample_data():
    return pd.DataFrame({"C": [20, 21, 22, 23, 24]})


@pytest.fixture
def sample_mask():
    return pd.DataFrame({"C": [True, False, True, False, False]})


def test_len_df(sample_db_df):
    assert len(sample_db_df) == 5


def test_len_reader(sample_db_reader):
    assert len(sample_db_reader) == 5


def test_print_df(sample_db_df, capsys):
    print(sample_db_df)

    captured = capsys.readouterr()

    assert captured.out.strip() != ""

    for col in sample_db_df.columns:
        assert col in captured.out


def test_print_reader(sample_db_reader, capsys):
    print(sample_db_reader)

    captured = capsys.readouterr()

    assert captured.out.strip() != ""


def test_copy_df(sample_db_df):
    db_cp = sample_db_df.copy()

    pd.testing.assert_frame_equal(sample_db_df.data, db_cp.data)
    pd.testing.assert_frame_equal(sample_db_df.mask, db_cp.mask)


def test_copy_reader(sample_db_reader):
    db_cp = sample_db_reader.copy()

    pd.testing.assert_frame_equal(sample_db_reader.data.read(), db_cp.data.read())
    pd.testing.assert_frame_equal(sample_db_reader.mask.read(), db_cp.mask.read())


def test_add_df(sample_db_df):
    sample_data = sample_db_df.data
    sample_mask = sample_db_df.mask

    db = DataBundle()
    db_add = db.add({"data": sample_data})

    pd.testing.assert_frame_equal(db_add.data, sample_data)

    db = DataBundle()
    db_add = db.add({"mask": sample_mask})

    pd.testing.assert_frame_equal(db_add.mask, sample_mask)

    db = DataBundle()
    db_add = db.add({"data": sample_data, "mask": sample_mask})

    pd.testing.assert_frame_equal(db_add.data, sample_data)
    pd.testing.assert_frame_equal(db_add.mask, sample_mask)


def test_add_reader_data(sample_db_reader):
    sample_data = sample_db_reader.data

    db = DataBundle()
    db_add = db.add({"data": sample_data})

    pd.testing.assert_frame_equal(db_add.data.read(), sample_data.read())


def test_add_reader_mask(sample_db_reader):
    sample_mask = sample_db_reader.mask

    db = DataBundle()
    db_add = db.add({"mask": sample_mask})

    pd.testing.assert_frame_equal(db_add.mask.read(), sample_mask.read())


def test_add_reader_both(sample_db_reader):
    sample_data = sample_db_reader.data
    sample_mask = sample_db_reader.mask

    db = DataBundle()
    db_add = db.add({"data": sample_data, "mask": sample_mask})

    pd.testing.assert_frame_equal(db_add.data.read(), sample_data.read())
    pd.testing.assert_frame_equal(db_add.mask.read(), sample_mask.read())


def test_stack_v_df(sample_db_df):
    sample_data = sample_db_df.data.copy()
    sample_mask = sample_db_df.mask.copy()

    db = DataBundle(data=sample_data, mask=sample_mask)

    sample_db_df.stack_v(db)

    expected_data = pd.concat([sample_data, db.data], ignore_index=True)
    expected_mask = pd.concat([sample_mask, db.mask], ignore_index=True)

    pd.testing.assert_frame_equal(sample_db_df.data, expected_data)
    pd.testing.assert_frame_equal(sample_db_df.mask, expected_mask)


def test_stack_v_reader(sample_db_reader):
    sample_data = sample_db_reader.data
    sample_mask = sample_db_reader.mask

    db = DataBundle(data=sample_data, mask=sample_mask)

    with pytest.raises(ValueError):
        sample_db_reader.stack_v(db)


def test_stack_h_df(sample_db_df, sample_data, sample_mask):
    db = DataBundle(data=sample_data, mask=sample_mask)
    expected_data = pd.concat([sample_db_df.data, db.data], axis=1)
    expected_mask = pd.concat([sample_db_df.mask, db.mask], axis=1)

    sample_db_df.stack_h(db)

    pd.testing.assert_frame_equal(sample_db_df.data, expected_data)
    pd.testing.assert_frame_equal(sample_db_df.mask, expected_mask)


def test_stack_h_reader(sample_db_reader):
    sample_data = sample_db_reader.data
    sample_mask = sample_db_reader.mask

    db = DataBundle(data=sample_data, mask=sample_mask)

    with pytest.raises(ValueError):
        sample_db_reader.stack_h(db)


@pytest.mark.parametrize(
    "func, args, idx_exp, idx_rej",
    [
        ("select_where_all_true", [], [0, 1, 2], [3, 4]),
        ("select_where_all_false", [], [3], [0, 1, 2, 4]),
        ("select_where_index_isin", [[0, 2, 4]], [0, 2, 4], [1, 3]),
        ("select_where_entry_isin", [{"A": [26, 41]}], [1, 3], [0, 2, 4]),
    ],
)
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_select_operators_df(
    sample_db_df,
    func,
    args,
    idx_exp,
    idx_rej,
    reset_index,
    inverse,
):
    result = getattr(sample_db_df, func)(
        *args, reset_index=reset_index, inverse=inverse
    )

    data = sample_db_df.data
    mask = sample_db_df.mask

    selected_data = result.data
    selected_mask = result.mask

    if inverse is False:
        idx = data.index.isin(idx_exp)
    else:
        idx = data.index.isin(idx_rej)

    expected_data = data[idx]
    expected_mask = mask[idx]

    if reset_index is True:
        expected_data = expected_data.reset_index(drop=True)
        expected_mask = expected_mask.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected_data, selected_data)
    pd.testing.assert_frame_equal(expected_mask, selected_mask)


@pytest.mark.parametrize(
    "func, args, idx_exp, idx_rej",
    [
        ("select_where_all_true", [], [0, 1, 2], [3, 4]),
        ("select_where_all_false", [], [3], [0, 1, 2, 4]),
        ("select_where_index_isin", [[0, 2, 4]], [0, 2, 4], [1, 3]),
        ("select_where_entry_isin", [{"A": [26, 41]}], [1, 3], [0, 2, 4]),
    ],
)
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_select_operators_reader(
    sample_db_reader,
    func,
    args,
    idx_exp,
    idx_rej,
    reset_index,
    inverse,
):
    result = getattr(sample_db_reader, func)(
        *args, reset_index=reset_index, inverse=inverse
    )

    data = sample_db_reader.data.read()
    mask = sample_db_reader.mask.read()

    selected_data = result.data.read()
    selected_mask = result.mask.read()

    if inverse is False:
        idx = data.index.isin(idx_exp)
    else:
        idx = data.index.isin(idx_rej)

    expected_data = data[idx]
    expected_mask = mask[idx]

    if reset_index is True:
        expected_data = expected_data.reset_index(drop=True)
        expected_mask = expected_mask.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected_data, selected_data)
    pd.testing.assert_frame_equal(expected_mask, selected_mask)


@pytest.mark.parametrize(
    "func, args, idx_exp, idx_rej",
    [
        ("split_by_boolean_true", [], [0, 1, 2], [3, 4]),
        ("split_by_boolean_false", [], [3], [0, 1, 2, 4]),
        ("split_by_index", [[0, 2, 4]], [0, 2, 4], [1, 3]),
        ("split_by_column_entries", [{"A": [26, 41]}], [1, 3], [0, 2, 4]),
    ],
)
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_split_operators_df(
    sample_db_df,
    func,
    args,
    idx_exp,
    idx_rej,
    reset_index,
    inverse,
):
    result = getattr(sample_db_df, func)(
        *args, reset_index=reset_index, inverse=inverse
    )

    data = sample_db_df.data
    mask = sample_db_df.mask

    selected_data = result[0].data
    selected_mask = result[0].mask
    rejected_data = result[1].data
    rejected_mask = result[1].mask

    if inverse is False:
        idx1 = data.index.isin(idx_exp)
        idx2 = data.index.isin(idx_rej)
    else:
        idx1 = data.index.isin(idx_rej)
        idx2 = data.index.isin(idx_exp)

    expected_data1 = data[idx1]
    expected_data2 = data[idx2]
    expected_mask1 = mask[idx1]
    expected_mask2 = mask[idx2]

    if reset_index is True:
        expected_data1 = expected_data1.reset_index(drop=True)
        expected_data2 = expected_data2.reset_index(drop=True)
        expected_mask1 = expected_mask1.reset_index(drop=True)
        expected_mask2 = expected_mask2.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected_data1, selected_data)
    pd.testing.assert_frame_equal(expected_data2, rejected_data)
    pd.testing.assert_frame_equal(expected_mask1, selected_mask)
    pd.testing.assert_frame_equal(expected_mask2, rejected_mask)


@pytest.mark.parametrize(
    "func, args, idx_exp, idx_rej",
    [
        ("split_by_boolean_true", [], [0, 1, 2], [3, 4]),
        ("split_by_boolean_false", [], [3], [0, 1, 2, 4]),
        ("split_by_index", [[0, 2, 4]], [0, 2, 4], [1, 3]),
        ("split_by_column_entries", [{"A": [26, 41]}], [1, 3], [0, 2, 4]),
    ],
)
@pytest.mark.parametrize("reset_index", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_split_operators_reader(
    sample_db_reader,
    func,
    args,
    idx_exp,
    idx_rej,
    reset_index,
    inverse,
):
    result = getattr(sample_db_reader, func)(
        *args, reset_index=reset_index, inverse=inverse
    )

    data = sample_db_reader.data.read()
    mask = sample_db_reader.mask.read()

    selected_data = result[0].data.read()
    selected_mask = result[0].mask.read()
    rejected_data = result[1].data.read()
    rejected_mask = result[1].mask.read()

    if inverse is False:
        idx1 = data.index.isin(idx_exp)
        idx2 = data.index.isin(idx_rej)
    else:
        idx1 = data.index.isin(idx_rej)
        idx2 = data.index.isin(idx_exp)

    expected_data1 = data[idx1]
    expected_data2 = data[idx2]
    expected_mask1 = mask[idx1]
    expected_mask2 = mask[idx2]

    if reset_index is True:
        expected_data1 = expected_data1.reset_index(drop=True)
        expected_data2 = expected_data2.reset_index(drop=True)
        expected_mask1 = expected_mask1.reset_index(drop=True)
        expected_mask2 = expected_mask2.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected_data1, selected_data)
    pd.testing.assert_frame_equal(expected_data2, rejected_data)
    pd.testing.assert_frame_equal(expected_mask1, selected_mask)
    pd.testing.assert_frame_equal(expected_mask2, rejected_mask)


def test_split_by_index_multi(sample_db_reader_multi):
    result = sample_db_reader_multi.split_by_column_entries({("A", "a"): [26, 41]})

    data = sample_db_reader_multi.data.read()
    mask = sample_db_reader_multi.mask.read()

    selected_data = result[0].data.read()
    selected_mask = result[0].mask.read()
    rejected_data = result[1].data.read()
    rejected_mask = result[1].mask.read()

    idx1 = data.index.isin([1, 3])
    idx2 = data.index.isin([0, 2, 4])

    expected_data1 = data[idx1]
    expected_data2 = data[idx2]
    expected_mask1 = mask[idx1]
    expected_mask2 = mask[idx2]

    pd.testing.assert_frame_equal(expected_data1, selected_data)
    pd.testing.assert_frame_equal(expected_data2, rejected_data)
    pd.testing.assert_frame_equal(expected_mask1, selected_mask)
    pd.testing.assert_frame_equal(expected_mask2, rejected_mask)


def test_unique_df(sample_db_df):
    result = sample_db_df.unique(columns=("A"))
    assert result == {"A": {19: 1, 26: 1, 27: 1, 41: 1, 91: 1}}


def test_unique_reader(sample_db_reader):
    result = sample_db_reader.unique(columns=("A"))
    assert result == {"A": {19: 1, 26: 1, 27: 1, 41: 1, 91: 1}}


def test_replace_columns(sample_db_df):
    df_corr = pd.DataFrame(
        {
            "A_new": [101, 201, 301, 401, 501],
            "B": range(5),
        }
    )
    result = sample_db_df.replace_columns(
        df_corr,
        subset=["B", "A"],
        rep_map={"A": "A_new"},
        pivot_l="B",
        pivot_r="B",
    )
    expected = pd.DataFrame(
        {
            "A": [101, 201, 301, 401, 501],
            "B": [0, 1, 2, 3, 4],
        }
    )

    pd.testing.assert_frame_equal(result.data, expected)
