from __future__ import annotations
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cdm_reader_mapper.common.iterators import ParquetStreamReader
from cdm_reader_mapper.core._utilities import (
    SubscriptableMethod,
    _DataBundle,
    _copy,
    combine_attribute_values,
    method,
    reader_method,
)


@pytest.fixture
def test_db_pd():
    df = pd.DataFrame({"a": [1, 3], "b": [2, 4]})
    return _DataBundle(
        data=df,
        columns=df.columns,
        dtypes=df.dtypes,
        imodel="test_model",
        mode="data",
    )


@pytest.fixture
def test_db_psr():
    df1 = pd.DataFrame({"a": 1, "b": 2}, index=[0])
    df2 = pd.DataFrame({"a": 3, "b": 4}, index=[1])
    psr = ParquetStreamReader([df1, df2])
    return _DataBundle(
        data=psr,
        columns=df1.columns,
        dtypes=df1.dtypes,
        imodel="test_model",
        mode="data",
    )


@pytest.fixture
def test_db_tuple():
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

    db1 = _DataBundle(data=df1, columns=df1.columns, dtypes=df1.dtypes)
    db2 = _DataBundle(data=df2, columns=df2.columns, dtypes=df2.dtypes)

    return db1, db2


def test_copy_dict():
    d = {"a": 1}
    assert d == _copy(d)


def test_copy_pandas():
    df = pd.DataFrame([{"a": 1, "b": 2}])
    pd.testing.assert_frame_equal(df, _copy(df))

    series = pd.Series([1, 2])
    pd.testing.assert_series_equal(series, _copy(series))


def test_copy_parquetstremareader():
    psr = ParquetStreamReader([pd.DataFrame([{"a": 1, "b": 2}]), pd.DataFrame([{"a": 3, "b": 4}])])
    copy = _copy(psr)

    pd.testing.assert_frame_equal(psr.read(), copy.read())


def test_copy_list():
    ll = [1, 2]
    assert ll == _copy(ll)


def test_copy_int():
    ii = 1
    assert ii == _copy(ii)


def test_method_callable():
    def f(x, y):
        return x + y

    result = method(f, 2, 3)

    assert result == 5


def test_method_subscriptable():
    data = {("a", "b"): 42}

    result = method(data, "a", "b")

    assert result == 42


def test_method_raises():
    class Dummy:
        pass

    obj = Dummy()

    with pytest.raises(ValueError, match="Attribute is neither callable nor subscriptable."):
        method(obj, 1)


def test_reader_method_callable(test_db_psr):
    result = reader_method(
        test_db_psr,
        test_db_psr.data,
        "sum",
        axis=1,
        process_kwargs={"non_data_output": "acc"},
    )

    expected = pd.Series({0: 3, 1: 7})

    pd.testing.assert_series_equal(result.read(), expected)


def test_reader_method_subscriptable(test_db_psr):
    result = reader_method(
        test_db_psr,
        test_db_psr.data,
        "__getitem__",
        "a",
    )

    expected = pd.Series({0: 1, 1: 3}, name="a")

    pd.testing.assert_series_equal(result.read(), expected)


def test_reader_method_none(test_db_psr):
    result = reader_method(
        test_db_psr,
        test_db_psr.data,
        "get",
        "false_attr",
    )

    assert result is None


def test_reader_method_inplace(test_db_psr):
    result = reader_method(
        test_db_psr,
        test_db_psr.data,
        "sum",
        axis=1,
        inplace=True,
        process_kwargs={"non_data_output": "acc"},
    )

    assert result is None

    assert hasattr(test_db_psr, "_data")
    assert test_db_psr._data is not None

    assert isinstance(test_db_psr._data, ParquetStreamReader)


def test_combine_attribute_values_index():
    first = pd.Index([1, 2])
    iterator = [
        SimpleNamespace(attr=pd.Index([2, 3])),
        SimpleNamespace(attr=pd.Index([3, 4])),
    ]

    result = combine_attribute_values(first, iterator, "attr")

    expected = pd.Index([1, 2, 3, 4])
    pd.testing.assert_index_equal(result, expected)


def test_combine_attribute_values_numeric():
    first = 10
    iterator = [SimpleNamespace(attr=5), SimpleNamespace(attr=3)]

    result = combine_attribute_values(first, iterator, "attr")

    assert result == 18


def test_combine_attribute_values_tuple():
    first = (2, 5)
    iterator = [SimpleNamespace(attr=(3, 5)), SimpleNamespace(attr=(4, 5))]

    result = combine_attribute_values(first, iterator, "attr")

    assert result == (9, 5)


def test_combine_attribute_values_list():
    first = [1, 2]
    iterator = [SimpleNamespace(attr=[3, 4]), SimpleNamespace(attr=[5])]

    result = combine_attribute_values(first, iterator, "attr")

    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5]))


def test_combine_attribute_values_ndarray():
    first = np.array([1, 2])
    iterator = [
        SimpleNamespace(attr=np.array([3, 4])),
        SimpleNamespace(attr=np.array([5])),
    ]

    result = combine_attribute_values(first, iterator, "attr")

    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5]))


def test_combine_attribute_values_pandas():
    first = pd.Series([1, 2], index=[0, 1])
    iterator = [
        SimpleNamespace(attr=pd.Series([3, 4], index=[2, 3])),
        SimpleNamespace(attr=pd.Series([5], index=[4])),
    ]

    result = combine_attribute_values(first, iterator, "attr")

    pd.testing.assert_series_equal(result, pd.Series([1, 2, 3, 4, 5]))


def test_combine_attribute_values_default():
    first = "a"
    iterator = [SimpleNamespace(attr="b"), SimpleNamespace(attr="c")]

    result = combine_attribute_values(first, iterator, "attr")

    assert result == ["a", "b", "c"]


def test_subscriptablemethod_call():
    def f(x, y):
        return x + y

    sm = SubscriptableMethod(f)

    result = sm(2, 3)
    assert result == 5


def test_subscriptablemethod_getitem_passes():
    data = {"a": 1, "b": 2}

    sm = SubscriptableMethod(data)
    result = sm["a"]

    assert result == 1
    result = sm["b"]
    assert result == 2


def test_subscriptablemethod_getitem_raises():
    def f(x):
        return x * 2

    sm = SubscriptableMethod(f)

    with pytest.raises(
        NotImplementedError,
        match="Calling subscriptable methods have not been implemented",
    ):
        sm[0]


def test_db_init_pd(test_db_pd):
    db = test_db_pd
    assert isinstance(db.data, pd.DataFrame)
    assert isinstance(db.mask, pd.DataFrame)

    assert list(db.columns) == ["a", "b"]
    assert set(db.dtypes.index) == {"a", "b"}
    assert db.imodel == "test_model"
    assert db.mode == "data"

    pd.testing.assert_frame_equal(db.data, pd.DataFrame({"a": [1, 3], "b": [2, 4]}))
    pd.testing.assert_frame_equal(db.mask, pd.DataFrame({"a": [True, True], "b": [True, True]}))


def test_db_init_psr(test_db_psr):
    db = test_db_psr.copy()
    assert isinstance(db.data, ParquetStreamReader)
    assert isinstance(db.mask, ParquetStreamReader)

    assert list(db.columns) == ["a", "b"]
    assert set(db.dtypes.index) == {"a", "b"}
    assert db.imodel == "test_model"
    assert db.mode == "data"

    pd.testing.assert_frame_equal(db.data.read(), pd.DataFrame({"a": [1, 3], "b": [2, 4]}))
    pd.testing.assert_frame_equal(db.mask.read(), pd.DataFrame({"a": [True, True], "b": [True, True]}))


def test_db_init_raises():
    with pytest.raises(ValueError):
        _DataBundle(
            mode="invalid",
        )


def test_db_init_iterators():
    data1 = pd.DataFrame([{"a": 1, "b": 2}])
    data2 = pd.DataFrame([{"a": 3, "b": 4}])

    mask1 = pd.DataFrame([{"a": True, "b": False}])
    mask2 = pd.DataFrame([{"a": False, "b": True}])

    db = _DataBundle(
        data=[data1, data2],
        mask=[mask1, mask2],
    )

    assert isinstance(db.data, ParquetStreamReader)
    assert isinstance(db.mask, ParquetStreamReader)


def test_db_len(test_db_pd, test_db_psr):
    assert len(test_db_pd) == 2
    assert len(test_db_psr) == 2


def test_db_getattr_pd(test_db_pd):
    db = test_db_pd

    sum_method = db.sum
    assert isinstance(sum_method, SubscriptableMethod)

    result = sum_method(axis=1)
    expected = pd.Series({0: 3, 1: 7})

    pd.testing.assert_series_equal(result, expected)

    columns = db.columns
    assert isinstance(columns, pd.Index)
    assert list(columns) == ["a", "b"]


def test_db_getattr_psr(test_db_psr):
    db = test_db_psr

    sum_method = db.sum
    assert isinstance(sum_method, SubscriptableMethod)

    result = sum_method(axis=1, process_kwargs={"non_data_output": "acc"}).read()

    expected = pd.Series({0: 3, 1: 7})

    pd.testing.assert_series_equal(result, expected)

    columns = db.columns
    assert isinstance(columns, pd.Index)
    assert list(columns) == ["a", "b"]

    assert db.attrs == {}


def test_db_getattr_raises_underscore(test_db_pd):
    with pytest.raises(AttributeError, match="DataBundle object has no attribute"):
        _ = test_db_pd.__magic__


def test_db_getattr_raises_invalid_type():
    class Dummy:
        pass

    db = _DataBundle(data=Dummy())
    with pytest.raises(TypeError, match="expected DataFrame or ParquetStreamReader"):
        _ = db.some_attr


def test_db_getattr_raises_empty(test_db_psr):
    db = test_db_psr.copy()
    db.read()

    with pytest.raises(ValueError, match="Cannot access attribute on empty data stream."):
        _ = db.some_attr


def test_db_getattr_raises_invalid_attr(test_db_psr):
    with pytest.raises(AttributeError, match="DataFrame chunk has no attribute"):
        _ = test_db_psr.invalid_attr


def test_db_repr_pd(test_db_pd):
    assert repr(test_db_pd) == repr(test_db_pd._data)


def test_db_repr_psr(test_db_psr):
    assert repr(test_db_psr) == repr(test_db_psr._data)


def test_setitem_pd(test_db_pd):
    db = test_db_pd

    db["mode"] = "tables"
    assert db.mode == "tables"

    db["a"] = pd.Series([10, 20], name="a")
    pd.testing.assert_series_equal(db._data["a"], pd.Series([10, 20], name="a"))


def test_setitem_psr(test_db_psr):
    db = test_db_psr

    db["mode"] = "tables"
    assert db.mode == "tables"

    with pytest.raises(TypeError, match="'ParquetStreamReader' object does not support item assignment"):
        db["a"] = ParquetStreamReader([pd.Series([10, 20], name="a")])


def test_getitem_pd(test_db_pd):
    result = test_db_pd["mode"]
    assert result == test_db_pd.mode

    result = test_db_pd["a"]
    pd.testing.assert_series_equal(result, test_db_pd["a"])

    with pytest.raises(KeyError):
        test_db_pd["non_existing_column"]


def test_getitem_psr(test_db_psr):
    result = test_db_psr["mode"]
    assert result == test_db_psr.mode

    result = test_db_psr["a"]
    pd.testing.assert_series_equal(result, test_db_psr["a"])

    with pytest.raises(KeyError):
        test_db_psr["non_existing_column"]


def test_columns_setter(test_db_pd):
    db = test_db_pd
    new_cols = ["x", "y"]
    db.columns = new_cols
    assert db._columns == new_cols


def test_return_property(test_db_pd):
    db = test_db_pd

    assert db.parse_dates == db._parse_dates
    assert db.encoding == db._encoding

    pd.testing.assert_frame_equal(db.data, db._data)
    pd.testing.assert_frame_equal(db.mask, db._mask)


def test_property_setters(test_db_pd):
    db = test_db_pd

    mask_df = pd.DataFrame([[True, False], [False, True]], columns=db._data.columns)
    db.mask = mask_df
    pd.testing.assert_frame_equal(db._mask, mask_df)

    db.imodel = "new_model"
    assert db._imodel == "new_model"

    db.mode = "tables"
    assert db._mode == "tables"


def test_db_getitem_pd(test_db_pd):
    db = test_db_pd

    a_col = db["a"]
    pd.testing.assert_series_equal(a_col, pd.Series([1, 3], name="a"))

    df_slice = db._data[["a", "b"]]
    pd.testing.assert_frame_equal(df_slice, db._data[["a", "b"]])


def test_db_getitem_psr(test_db_psr):
    db = test_db_psr

    a_col = db["a"]
    pd.testing.assert_series_equal(a_col, pd.Series([1, 3], name="a"))

    with pytest.raises(TypeError, match="unhashable type: 'list'"):
        db._data[["a", "b"]]


def test_db_setitem_pd(test_db_pd):
    db = test_db_pd

    db["c"] = pd.Series([5, 6], name="c")
    expected = pd.Series([5, 6], name="c")
    pd.testing.assert_series_equal(db._data["c"], expected)

    db["a"] = pd.Series([7, 8], name="a")
    expected = pd.Series([7, 8], name="a")

    pd.testing.assert_series_equal(db._data["a"], expected)


def test_db_setitem_psr(test_db_psr):
    db = test_db_psr

    with pytest.raises(TypeError, match="'ParquetStreamReader' object does not support item assignment"):
        db["c"] = pd.Series([5, 6], name="c")


def test_db_data_property_pd(test_db_pd):
    db = test_db_pd

    pd.testing.assert_frame_equal(db.data, db._data)

    new_df = pd.DataFrame({"x": [10, 20]})
    db.data = new_df
    pd.testing.assert_frame_equal(db.data, new_df)


def test_db_data_property_psr(test_db_psr):
    data1 = test_db_psr.copy().data.read()
    data2 = test_db_psr.copy()._data.read()

    pd.testing.assert_frame_equal(data1, data2)

    db = test_db_psr.copy()
    new_df = ParquetStreamReader([pd.DataFrame({"x": [10, 20]})])
    db.data = new_df.copy()
    pd.testing.assert_frame_equal(db.data.read(), new_df.read())


def test_db_get_return_pd(test_db_pd):
    db = test_db_pd

    assert db._get_db(True) is db

    copy_db = db._get_db(False)
    assert copy_db is not db

    assert db._return_db(copy_db, inplace=True) is None

    assert db._return_db(copy_db, inplace=False) is copy_db


def test_db_get_return_psr(test_db_psr):
    db = test_db_psr

    assert db._get_db(True) is db

    copy_db = db._get_db(False)
    assert copy_db is not db

    assert db._return_db(copy_db, inplace=True) is None

    assert db._return_db(copy_db, inplace=False) is copy_db


def test_stack_single(test_db_tuple):
    db1, db2 = test_db_tuple

    stacked_db = db1._stack(other=db2, datasets="data", inplace=False)

    expected_df = pd.concat([db1.data, db2.data], ignore_index=True)
    pd.testing.assert_frame_equal(stacked_db.data, expected_df)


def test_stack_multiple(test_db_tuple):
    db1, db2 = test_db_tuple

    stacked_db = db1._stack(other=[db2], datasets=["data"], inplace=False)

    expected_df = pd.concat([db1.data, db2.data], ignore_index=True)
    pd.testing.assert_frame_equal(stacked_db.data, expected_df)


def test_stack_inplace(test_db_tuple):
    db1, db2 = test_db_tuple

    result = db1._stack(other=db2, datasets="data", inplace=True)

    assert result is None

    expected_df = pd.concat(
        [
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            pd.DataFrame({"a": [5, 6], "b": [7, 8]}),
        ],
        ignore_index=True,
    )

    pd.testing.assert_frame_equal(db1.data, expected_df)


def test_stack_missing_dataset(test_db_tuple):
    db1, db2 = test_db_tuple

    result = db1._stack(other=db2, datasets="_mask", inplace=False)

    pd.testing.assert_frame_equal(result.data, db1.data)


def test_stack_invalid_iterator(test_db_tuple):
    db1, db2 = test_db_tuple

    db1._data = iter([db1.data])

    with pytest.raises(ValueError, match="Data must be a pd.DataFrame"):
        db1._stack(other=db2, datasets="data", inplace=False)
