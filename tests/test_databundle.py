from __future__ import annotations

import pandas as pd
import pytest

from cdm_reader_mapper import DataBundle
from cdm_reader_mapper.common.iterators import ParquetStreamReader
from cdm_reader_mapper.core._utilities import SubscriptableMethod


YR = ("core", "YR")
MO = ("core", "MO")
DY = ("core", "DY")
HR = ("core", "HR")
PT = ("c1", "PT")
ID = ("core", "ID")


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
    return DataBundle(data=data, mask=mask, imodel="test_model", dtypes=data.dtypes)


@pytest.fixture
def sample_db_df_multi():
    data = pd.DataFrame(
        {
            ("A", "a"): [19, 26, 27, 41, 91],
            ("B", "b"): [0, 1, 2, 3, 4],
        }
    )
    mask = pd.DataFrame(
        {
            ("A", "a"): [True, True, True, False, True],
            ("B", "b"): [True, True, True, False, False],
        }
    )
    return DataBundle(data=data, mask=mask, imodel="test_model", dtypes=data.dtypes)


@pytest.fixture
def sample_db_psr():
    data1 = pd.DataFrame({"A": [19, 26], "B": [0, 1]}, index=[0, 1])
    data2 = pd.DataFrame({"A": [27, 41, 91], "B": [2, 3, 4]}, index=[2, 3, 4])
    data = ParquetStreamReader([data1, data2])

    mask1 = pd.DataFrame({"A": [True, True], "B": [True, True]}, index=[0, 1])
    mask2 = pd.DataFrame({"A": [True, False, True], "B": [True, False, False]}, index=[2, 3, 4])
    mask = ParquetStreamReader([mask1, mask2])

    return DataBundle(data=data, mask=mask, imodel="test_model", dtypes=data1.dtypes)


@pytest.fixture
def sample_db_psr_multi():
    data1 = pd.DataFrame({("A", "a"): [19, 26], ("B", "b"): [0, 1]}, index=[0, 1])
    data2 = pd.DataFrame({("A", "a"): [27, 41, 91], ("B", "b"): [2, 3, 4]}, index=[2, 3, 4])
    data = ParquetStreamReader([data1, data2])

    mask1 = pd.DataFrame({("A", "a"): [True, True], ("B", "b"): [True, True]}, index=[0, 1])
    mask2 = pd.DataFrame(
        {("A", "a"): [True, False, True], ("B", "b"): [True, False, False]},
        index=[2, 3, 4],
    )
    mask = ParquetStreamReader([mask1, mask2])

    return DataBundle(data=data, mask=mask, imodel="test_model", dtypes=data1.dtypes)


@pytest.fixture
def sample_data():
    return pd.DataFrame({"C": [20, 21, 22, 23, 24]})


@pytest.fixture
def sample_mask():
    return pd.DataFrame({"C": [True, False, True, False, False]})


def test_init_df(sample_db_df):
    db = sample_db_df
    assert isinstance(db.data, pd.DataFrame)
    assert isinstance(db.mask, pd.DataFrame)

    assert list(db.columns) == ["A", "B"]
    assert set(db.dtypes.index) == {"A", "B"}
    assert db.imodel == "test_model"
    assert db.mode == "data"

    pd.testing.assert_frame_equal(db.data, pd.DataFrame({"A": [19, 26, 27, 41, 91], "B": [0, 1, 2, 3, 4]}))
    pd.testing.assert_frame_equal(db.mask, pd.DataFrame({"A": [True, True, True, False, True], "B": [True, True, True, False, False]}))


def test_init_psr(sample_db_psr):
    db = sample_db_psr.copy()
    assert isinstance(db.data, ParquetStreamReader)
    assert isinstance(db.mask, ParquetStreamReader)

    assert list(db.columns) == ["A", "B"]
    assert set(db.dtypes.index) == {"A", "B"}
    assert db.imodel == "test_model"
    assert db.mode == "data"

    pd.testing.assert_frame_equal(db.data.read(), pd.DataFrame({"A": [19, 26, 27, 41, 91], "B": [0, 1, 2, 3, 4]}))
    pd.testing.assert_frame_equal(db.mask.read(), pd.DataFrame({"A": [True, True, True, False, True], "B": [True, True, True, False, False]}))


def test_init_iterators():
    data1 = pd.DataFrame([{"A": 1, "B": 2}])
    data2 = pd.DataFrame([{"A": 3, "B": 4}])

    mask1 = pd.DataFrame([{"A": True, "B": False}])
    mask2 = pd.DataFrame([{"A": False, "B": True}])

    db = DataBundle(
        data=[data1, data2],
        mask=[mask1, mask2],
    )

    assert isinstance(db.data, ParquetStreamReader)
    assert isinstance(db.mask, ParquetStreamReader)


def test_db_init_valueerror():
    with pytest.raises(ValueError):
        DataBundle(
            mode="invalid",
        )


def test_db_typerror():
    class Dummy:
        pass

    with pytest.raises(TypeError, match="'data' has unsupported type"):
        DataBundle(data=Dummy())


def test_getattr_pd(sample_db_df):
    db = sample_db_df

    sum_method = db.sum
    assert isinstance(sum_method, SubscriptableMethod)

    result = sum_method(axis=1)

    expected = pd.Series({0: 19, 1: 27, 2: 29, 3: 44, 4: 95})

    pd.testing.assert_series_equal(result, expected)

    columns = db.columns
    assert isinstance(columns, pd.Index)
    assert list(columns) == ["A", "B"]


def test_getattr_psr(sample_db_psr):
    db = sample_db_psr

    sum_method = db.sum
    assert isinstance(sum_method, SubscriptableMethod)

    result = sum_method(axis=1, process_kwargs={"non_data_output": "acc"}).read()

    expected = pd.Series({0: 19, 1: 27, 2: 29, 3: 44, 4: 95})

    pd.testing.assert_series_equal(result, expected)

    columns = db.columns
    assert isinstance(columns, pd.Index)
    assert list(columns) == ["A", "B"]

    assert db.attrs == {}


def test_getattr_attributeerror_object(sample_db_df):
    with pytest.raises(AttributeError, match="DataBundle object has no attribute"):
        _ = sample_db_df.__magic__


def test_getattr_valueerror(sample_db_psr):
    db = sample_db_psr.copy()
    db.read()

    with pytest.raises(ValueError, match="Cannot access attribute on empty data stream."):
        _ = db.some_attr


def test_getattr_attributeerror_chunk(sample_db_psr):
    with pytest.raises(AttributeError, match="DataFrame chunk has no attribute"):
        _ = sample_db_psr.invalid_attr


def test_repr_pd(sample_db_df):
    assert repr(sample_db_df) == repr(sample_db_df._data)


def test_repr_psr(sample_db_psr):
    assert repr(sample_db_psr) == repr(sample_db_psr._data)


def test_setitem_pd(sample_db_df):
    db = sample_db_df

    db["mode"] = "tables"
    assert db.mode == "tables"

    db["A"] = pd.Series([10, 20, 30, 40, 50], name="A")
    pd.testing.assert_series_equal(db._data["A"], pd.Series([10, 20, 30, 40, 50], name="A"))


def test_setitem_psr(sample_db_psr):
    db = sample_db_psr

    db["mode"] = "tables"
    assert db.mode == "tables"

    with pytest.raises(TypeError, match="'ParquetStreamReader' object does not support item assignment"):
        db["A"] = ParquetStreamReader([pd.Series([10, 20, 30, 40, 50], name="A")])


def test_getitem_pd(sample_db_df):
    result = sample_db_df["mode"]
    assert result == sample_db_df.mode

    result = sample_db_df["A"]
    pd.testing.assert_series_equal(result, sample_db_df["A"])

    with pytest.raises(KeyError):
        sample_db_df["non_existing_column"]


def test_getitem_psr(sample_db_psr):
    result = sample_db_psr["mode"]
    assert result == sample_db_psr.mode

    result = sample_db_psr["A"]
    pd.testing.assert_series_equal(result, sample_db_psr["A"])

    with pytest.raises(KeyError):
        sample_db_psr["non_existing_column"]


def test_return_property(sample_db_df):
    db = sample_db_df

    assert db.parse_dates == db._parse_dates
    assert db.encoding == db._encoding

    pd.testing.assert_frame_equal(db.data, db._data)
    pd.testing.assert_frame_equal(db.mask, db._mask)


def test_property_setters(sample_db_df):
    db = sample_db_df

    mask_df = pd.DataFrame([[True, False], [False, True]], columns=db._data.columns)
    db.mask = mask_df
    pd.testing.assert_frame_equal(db._mask, mask_df)

    db.imodel = "new_model"
    assert db._imodel == "new_model"

    db.mode = "tables"
    assert db._mode == "tables"


def test_columns_setter(sample_db_df):
    db = sample_db_df
    new_cols = ["x", "y"]
    db.columns = new_cols
    assert db._columns == new_cols


def test_pd_getitem(sample_db_df):
    db = sample_db_df

    a_col = db["A"]
    pd.testing.assert_series_equal(a_col, pd.Series([19, 26, 27, 41, 91], name="A"))

    df_slice = db._data[["A", "B"]]
    pd.testing.assert_frame_equal(df_slice, pd.DataFrame({"A": [19, 26, 27, 41, 91], "B": [0, 1, 2, 3, 4]}))


def test_psr_getitem(sample_db_psr):
    db = sample_db_psr

    a_col = db["A"]
    pd.testing.assert_series_equal(a_col, pd.Series([19, 26, 27, 41, 91], name="A"))

    with pytest.raises(TypeError, match="unhashable type: 'list'"):
        db._data[["A", "B"]]


def test_pd_setitem(sample_db_df):
    db = sample_db_df

    db["C"] = pd.Series([5, 6, 7, 8, 9], name="C")
    expected = pd.Series([5, 6, 7, 8, 9], name="C")
    pd.testing.assert_series_equal(db._data["C"], expected)

    db["A"] = pd.Series([7, 8, 9, 10, 11], name="A")
    expected = pd.Series([7, 8, 9, 10, 11], name="A")
    pd.testing.assert_series_equal(db._data["A"], expected)


def test_psr_setitem(sample_db_psr):
    db = sample_db_psr

    with pytest.raises(TypeError, match="'ParquetStreamReader' object does not support item assignment"):
        db["C"] = pd.Series([5, 6], name="C")


def test_len_df(sample_db_df):
    assert len(sample_db_df) == 5


def test_len_psr(sample_db_psr):
    assert len(sample_db_psr) == 5


def test_print_df(sample_db_df, capsys):
    print(sample_db_df)

    captured = capsys.readouterr()

    assert captured.out.strip() != ""

    for col in sample_db_df.columns:
        assert col in captured.out


def test_print_psr(sample_db_psr, capsys):
    print(sample_db_psr)

    captured = capsys.readouterr()

    assert captured.out.strip() != ""


def test_get_return_pd(sample_db_df):
    db = sample_db_df

    assert db._get_db(True) is db

    copy_db = db._get_db(False)
    assert copy_db is not db

    assert db._return_db(copy_db, inplace=True) is None

    assert db._return_db(copy_db, inplace=False) is copy_db


def test_get_return_psr(sample_db_psr):
    db = sample_db_psr

    assert db._get_db(True) is db

    copy_db = db._get_db(False)
    assert copy_db is not db

    assert db._return_db(copy_db, inplace=True) is None

    assert db._return_db(copy_db, inplace=False) is copy_db


def test_copy_df(sample_db_df):
    db_cp = sample_db_df.copy()

    pd.testing.assert_frame_equal(sample_db_df.data, db_cp.data)
    pd.testing.assert_frame_equal(sample_db_df.mask, db_cp.mask)


def test_copy_psr(sample_db_psr):
    db_cp = sample_db_psr.copy()
    pd.testing.assert_frame_equal(sample_db_psr.data.read(), db_cp.data.read())
    pd.testing.assert_frame_equal(sample_db_psr.mask.read(), db_cp.mask.read())


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


def test_add_psr_data(sample_db_psr):
    sample_data = sample_db_psr.data

    db = DataBundle()
    db_add = db.add({"data": sample_data})

    pd.testing.assert_frame_equal(db_add.data.read(), sample_data.read())


def test_add_psr_mask(sample_db_psr):
    sample_mask = sample_db_psr.mask

    db = DataBundle()
    db_add = db.add({"mask": sample_mask})

    pd.testing.assert_frame_equal(db_add.mask.read(), sample_mask.read())


def test_add_psr_both(sample_db_psr):
    sample_data = sample_db_psr.data
    sample_mask = sample_db_psr.mask

    db = DataBundle()
    db_add = db.add({"data": sample_data, "mask": sample_mask})

    pd.testing.assert_frame_equal(db_add.data.read(), sample_data.read())
    pd.testing.assert_frame_equal(db_add.mask.read(), sample_mask.read())


def test_stack_single(sample_db_df):
    db1 = sample_db_df
    db2 = sample_db_df

    stacked_db = db1._stack(other=db2, datasets="data", inplace=False)

    expected_df = pd.concat([db1.data, db2.data], ignore_index=True)
    pd.testing.assert_frame_equal(stacked_db.data, expected_df)


def test_stack_multiple(sample_db_df):
    db1 = sample_db_df
    db2 = sample_db_df

    stacked_db = db1._stack(other=[db2], datasets=["data"], inplace=False)

    expected_df = pd.concat([db1.data, db2.data], ignore_index=True)
    pd.testing.assert_frame_equal(stacked_db.data, expected_df)


def test_stack_inplace(sample_db_df):
    db1 = sample_db_df.copy()
    db2 = sample_db_df.copy()

    result = db1._stack(other=db2, datasets="data", inplace=True)

    assert result is None

    expected_df = pd.concat(
        [
            db2.data,
            db2.data,
        ],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(db1.data, expected_df)


def test_stack_missing_dataset(sample_db_df):
    db1 = sample_db_df
    db2 = sample_db_df

    result = db1._stack(other=db2, datasets="_mask", inplace=False)

    pd.testing.assert_frame_equal(result.data, db1.data)


def test_stack_invalid_iterator(sample_db_df):
    db1 = sample_db_df
    db2 = sample_db_df

    db1._data = iter([db1.data])

    with pytest.raises(ValueError, match="Data must be a pd.DataFrame"):
        db1._stack(other=db2, datasets="data", inplace=False)


def test_stack_v_df(sample_db_df):
    sample_data = sample_db_df.data.copy()
    sample_mask = sample_db_df.mask.copy()

    db = DataBundle(data=sample_data, mask=sample_mask)

    sample_db_df.stack_v(db, inplace=True)

    expected_data = pd.concat([sample_data, db.data], ignore_index=True)
    expected_mask = pd.concat([sample_mask, db.mask], ignore_index=True)

    pd.testing.assert_frame_equal(sample_db_df.data, expected_data)
    pd.testing.assert_frame_equal(sample_db_df.mask, expected_mask)


def test_stack_v_psr(sample_db_psr):
    sample_data = sample_db_psr.data
    sample_mask = sample_db_psr.mask

    db = DataBundle(data=sample_data, mask=sample_mask)

    with pytest.raises(ValueError):
        sample_db_psr.stack_v(db)


def test_stack_h_df(sample_db_df, sample_data, sample_mask):
    db = DataBundle(data=sample_data, mask=sample_mask)
    expected_data = pd.concat([sample_db_df.data, db.data], axis=1)
    expected_mask = pd.concat([sample_db_df.mask, db.mask], axis=1)

    sample_db_df.stack_h(db, inplace=True)

    pd.testing.assert_frame_equal(sample_db_df.data, expected_data)
    pd.testing.assert_frame_equal(sample_db_df.mask, expected_mask)


def test_stack_h_psr(sample_db_psr):
    sample_data = sample_db_psr.data
    sample_mask = sample_db_psr.mask

    db = DataBundle(data=sample_data, mask=sample_mask)

    with pytest.raises(ValueError):
        sample_db_psr.stack_h(db)


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
    result = getattr(sample_db_df, func)(*args, reset_index=reset_index, inverse=inverse)

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
def test_select_operators_psr(
    sample_db_psr,
    func,
    args,
    idx_exp,
    idx_rej,
    reset_index,
    inverse,
):
    result = getattr(sample_db_psr, func)(*args, reset_index=reset_index, inverse=inverse)

    data = sample_db_psr.data.read()
    mask = sample_db_psr.mask.read()

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
    result = getattr(sample_db_df, func)(*args, reset_index=reset_index, inverse=inverse)

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
def test_split_operators_psr(
    sample_db_psr,
    func,
    args,
    idx_exp,
    idx_rej,
    reset_index,
    inverse,
):
    result = getattr(sample_db_psr, func)(*args, reset_index=reset_index, inverse=inverse)

    data = sample_db_psr.data.read()
    mask = sample_db_psr.mask.read()

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


def test_split_by_index_df_multi(sample_db_df_multi):
    result = sample_db_df_multi.split_by_column_entries({("A", "a"): [26, 41]})

    data = sample_db_df_multi.data
    mask = sample_db_df_multi.mask

    selected_data = result[0].data
    selected_mask = result[0].mask
    rejected_data = result[1].data
    rejected_mask = result[1].mask

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


def test_split_by_index_psr_multi(sample_db_psr_multi):
    result = sample_db_psr_multi.split_by_column_entries({("A", "a"): [26, 41]})

    data = sample_db_psr_multi.data.read()
    mask = sample_db_psr_multi.mask.read()

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


def test_unique_psr(sample_db_psr):
    result = sample_db_psr.unique(columns=("A"))
    assert result == {"A": {19: 1, 26: 1, 27: 1, 41: 1, 91: 1}}


def test_replace_columns_all_df(sample_db_df):
    df_corr = pd.DataFrame(
        {
            "A_new": [101, 201, 301, 401, 501],
            "B": range(5),
        }
    )
    result = sample_db_df.replace_columns(
        df_corr,
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


def test_replace_columns_subset_df(sample_db_df):
    df_corr = pd.DataFrame(
        {
            "A_new": [101, 201, 301, 401, 501],
            "B": range(5),
        }
    )
    result = sample_db_df.replace_columns(
        df_corr,
        subset=["A", "B"],
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


def test_replace_columns_all_psr(sample_db_psr):
    df_corr = pd.DataFrame(
        {
            "A_new": [101, 201, 301, 401, 501],
            "B": range(5),
        }
    )

    with pytest.raises(TypeError, match="Data must be a pd.DataFrame or pd.Series"):
        sample_db_psr.replace_columns(
            df_corr,
            rep_map={"A": "A_new"},
            pivot_l="B",
            pivot_r="B",
        )


def test_correct_datetime_df():
    data = pd.DataFrame({YR: [1899], MO: [1], DY: [1], HR: [0]})

    db = DataBundle(
        data=data,
        imodel="icoads_r300_d201",
    )

    result = db.correct_datetime()
    expected = pd.DataFrame({YR: [1898], MO: [12], DY: [31], HR: [0]})

    pd.testing.assert_frame_equal(result.data, expected)


def test_correct_datetime_psr():
    df1 = pd.DataFrame({YR: [1899], MO: [1], DY: [1], HR: [0]})

    db = DataBundle(
        data=ParquetStreamReader([df1]),
        imodel="icoads_r300_d201",
    )

    result = db.correct_datetime()

    expected = pd.DataFrame({YR: [1898], MO: [12], DY: [31], HR: [0]})
    pd.testing.assert_frame_equal(result.data.read(), expected)


def test_validate_datetime_df():
    data = pd.DataFrame({YR: [2023, 2023], MO: [1, 1], DY: [1, 2], HR: [12, None]})

    db = DataBundle(
        data=data,
        imodel="icoads",
    )

    result = db.validate_datetime()

    pd.testing.assert_series_equal(result, pd.Series([True, False]))


def test_validate_datetime_psr():
    df1 = pd.DataFrame({YR: [2023], MO: [1], DY: [1], HR: [12]}, index=[0])
    df2 = pd.DataFrame({YR: [2023], MO: [1], DY: [2], HR: [None]}, index=[1])

    db = DataBundle(
        data=ParquetStreamReader([df1, df2]),
        imodel="icoads",
    )

    result = db.validate_datetime()

    pd.testing.assert_series_equal(result.read(), pd.Series([True, False]))


def test_correct_pt_df():
    data = pd.DataFrame({PT: [None, "7", None]})

    db = DataBundle(
        data=data,
        imodel="icoads_r300_d993",
    )

    result = db.correct_pt()

    expected = pd.DataFrame({PT: ["5", "7", "5"]})
    pd.testing.assert_frame_equal(result.data, expected)


def test_correct_pt_psr():
    df1 = pd.DataFrame({PT: [None, "7", None]})

    db = DataBundle(
        data=ParquetStreamReader([df1]),
        imodel="icoads_r300_d993",
    )

    result = db.correct_pt()

    expected = pd.DataFrame({PT: ["5", "7", "5"]}, dtype="object")
    pd.testing.assert_frame_equal(result.data.read(), expected)


def test_validate_id_df():
    data = pd.DataFrame({ID: ["12345", "ABCDE", None]})

    db = DataBundle(
        data=data,
        imodel="icoads_r300_d201",
    )

    result = db.validate_id()

    pd.testing.assert_series_equal(result, pd.Series([True, False, True]))


def test_validate_id_psr():
    df1 = pd.DataFrame({ID: ["12345"]}, index=[0])
    df2 = pd.DataFrame({ID: ["ABCDE"]}, index=[1])
    df3 = pd.DataFrame({ID: [None]}, index=[2])

    db = DataBundle(
        data=ParquetStreamReader([df1, df2, df3]),
        imodel="icoads_r300_d201",
    )

    result = db.validate_id()

    pd.testing.assert_series_equal(result.read(), pd.Series([True, False, True]))


def test_map_model_df():
    data = pd.DataFrame(
        {
            ("c1", "PT"): ["2", "4", "9", "21"],
            ("c98", "UID"): ["5012", "8960", "0037", "1000"],
            ("c1", "LZ"): ["1", None, None, "3"],
        }
    )

    db = DataBundle(
        data=data,
        imodel="icoads_r302",
    )

    result = db.map_model()

    expected = pd.DataFrame(
        {
            ("header", "report_id"): [
                "ICOADS-302-5012",
                "ICOADS-302-8960",
                "ICOADS-302-0037",
                "ICOADS-302-1000",
            ],
            ("header", "duplicate_status"): [4, 4, 4, 4],
            ("header", "platform_type"): [2, 33, 32, 45],
            ("header", "location_quality"): [2, 0, 0, 0],
        }
    )
    pd.testing.assert_frame_equal(result.data[expected.columns], expected, check_dtype=False)


def test_map_model_psr():
    data = pd.DataFrame(
        {
            ("c1", "PT"): ["2", "4", "9", "21"],
            ("c98", "UID"): ["5012", "8960", "0037", "1000"],
            ("c1", "LZ"): ["1", None, None, "3"],
        }
    )

    db = DataBundle(
        data=ParquetStreamReader([data]),
        imodel="icoads_r302",
    )

    result = db.map_model()

    expected = pd.DataFrame(
        {
            ("header", "report_id"): [
                "ICOADS-302-5012",
                "ICOADS-302-8960",
                "ICOADS-302-0037",
                "ICOADS-302-1000",
            ],
            ("header", "duplicate_status"): [4, 4, 4, 4],
            ("header", "platform_type"): [2, 33, 32, 45],
            ("header", "location_quality"): [2, 0, 0, 0],
        }
    )

    pd.testing.assert_frame_equal(result.data.read()[expected.columns], expected, check_dtype=False)
