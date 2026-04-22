from __future__ import annotations
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
import xarray as xr

from cdm_reader_mapper.common.iterators import (
    ParquetStreamReader,
    ProcessFunction,
    _initialize_storage,
    _parquet_generator,
    _prepare_readers,
    _process_chunks,
    _process_function,
    _sort_chunk_outputs,
    _write_chunks_to_disk,
    ensure_parquet_reader,
    is_valid_iterator,
    parquet_stream_from_iterable,
    process_disk_backed,
    process_function,
)


def dummy_func(x):
    return 2 * x


def test_class_process_function_basic():
    df = pd.DataFrame({"a": [1, 2, 3]})

    pf = ProcessFunction(data=df, func=dummy_func)

    assert isinstance(pf, ProcessFunction)
    pd.testing.assert_frame_equal(pf.data, df)
    assert pf.func is dummy_func
    assert pf.func_args == ()
    assert pf.func_kwargs == {}


def test_class_process_function_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError, match="not callable"):
        ProcessFunction(data=df, func="invalid_function")


def test_class_process_function_tuple():
    df = pd.DataFrame({"a": [1, 2, 3]})

    pf = ProcessFunction(data=df, func=dummy_func, func_args=10)

    assert pf.func_args == (10,)


def test_class_process_function_extra():
    df = pd.DataFrame({"a": [1, 2, 3]})

    pf = ProcessFunction(df, dummy_func, extra=123, flag=True)

    assert pf.kwargs == {"extra": 123, "flag": True}


def make_chunks():
    return [
        pd.DataFrame({"a": [1, 2]}),
        pd.DataFrame({"a": [3, 4]}),
    ]


def chunk_generator():
    yield from make_chunks()


def test_init_with_iterator():
    reader = ParquetStreamReader(iter(make_chunks()))
    assert isinstance(reader, ParquetStreamReader)


def test_init_with_factory():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))
    assert isinstance(reader, ParquetStreamReader)


def test_init_invalid_source():
    with pytest.raises(TypeError):
        ParquetStreamReader(source=123)


def test_iteration_over_chunks():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))

    chunks = list(reader)

    assert len(chunks) == 2
    assert chunks[0]["a"].iloc[0] == 1
    assert chunks[1]["a"].iloc[-1] == 4


def test_next_raises_stop_iteration():
    reader = ParquetStreamReader(lambda: iter([]))

    with pytest.raises(StopIteration):
        next(reader)


def test_prepend_pushes_chunk_to_front():
    chunks = make_chunks()
    reader = ParquetStreamReader(lambda: iter(chunks))

    first = next(reader)
    reader.prepend(first)

    again = next(reader)

    pd.testing.assert_frame_equal(first, again)


def test_get_chunk_returns_next_chunk():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))

    chunk = reader.get_chunk()

    assert isinstance(chunk, pd.DataFrame)
    assert len(chunk) == 2


def test_read_concatenates_all_chunks():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))

    df = reader.read()

    assert len(df) == 4
    assert df["a"].tolist() == [1, 2, 3, 4]


def test_read_empty_stream_returns_empty_dataframe():
    reader = ParquetStreamReader(lambda: iter([]))

    df = reader.read()

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_copy_creates_independent_stream():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))

    reader_copy = reader.copy()

    original_first = next(reader)
    copy_first = next(reader_copy)

    pd.testing.assert_frame_equal(original_first, copy_first)


def test_copy_closed_stream_raises():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))
    reader.close()

    with pytest.raises(ValueError):
        reader.copy()


def test_empty_returns_true_if_empty():
    reader = ParquetStreamReader(lambda: iter([]))
    assert reader.empty is True


def test_empty_returns_false_if_not_empty():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))
    assert reader.empty is False


def test_reset_index_continuous_index():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))

    new_reader = reader.reset_index(drop=True)

    df = new_reader.read()

    assert df.index.tolist() == [0, 1, 2, 3]


def test_reset_index_keeps_old_index_column():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))

    new_reader = reader.reset_index(drop=False)
    df = new_reader.read()

    assert "index" in df.columns
    assert df.index.tolist() == [0, 1, 2, 3]


def test_reset_index_closed_stream_raises():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))
    reader.close()

    with pytest.raises(ValueError):
        reader.reset_index()


def test_next_on_closed_stream_raises():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))
    reader.close()

    with pytest.raises(ValueError):
        next(reader)


def test_context_manager_closes_stream():
    reader = ParquetStreamReader(lambda: iter(make_chunks()))

    with reader as r:
        chunk = next(r)
        assert len(chunk) == 2

    with pytest.raises(ValueError):
        next(reader)


@pytest.mark.parametrize(
    "outputs,capture_meta,expected_data_len,expected_meta_len",
    [
        ((pd.DataFrame({"a": [1]}),), False, 1, 0),
        ((pd.DataFrame({"a": [1]}), "meta"), True, 1, 1),
        (([pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})],), False, 2, 0),
        (("meta1", "meta2"), True, 0, 2),
    ],
)
def test_sort_chunk_outputs_parametrized(outputs, capture_meta, expected_data_len, expected_meta_len):
    data, meta = _sort_chunk_outputs(
        outputs,
        capture_meta=capture_meta,
        requested_types=(pd.DataFrame,),
    )

    assert len(data) == expected_data_len
    assert len(meta) == expected_meta_len


def make_df_0():
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})


def make_series_0():
    return pd.Series([1, 2, 3], name="my_series")


@pytest.mark.parametrize(
    "inputs,expected_schema_types",
    [
        ([make_df_0()], [(pd.DataFrame, make_df_0().columns)]),
        ([make_series_0()], [(pd.Series, "my_series")]),
        (
            [make_df_0(), make_series_0()],
            [
                (pd.DataFrame, make_df_0().columns),
                (pd.Series, "my_series"),
            ],
        ),
        (
            [make_df_0(), make_df_0()],
            [
                (pd.DataFrame, make_df_0().columns),
                (pd.DataFrame, make_df_0().columns),
            ],
        ),
    ],
)
def test_initialize_storage_valid(inputs, expected_schema_types):
    temp_dirs, schemas = _initialize_storage(inputs)

    try:
        # Correct number of temp dirs created
        assert len(temp_dirs) == len(inputs)

        # Ensure they are TemporaryDirectory instances
        assert all(isinstance(td, tempfile.TemporaryDirectory) for td in temp_dirs)

        # Check schemas
        assert len(schemas) == len(expected_schema_types)

        for (actual_type, actual_meta), (exp_type, exp_meta) in zip(schemas, expected_schema_types, strict=True):
            assert actual_type is exp_type

            if exp_type is pd.DataFrame:
                assert list(actual_meta) == list(exp_meta)
            else:
                assert actual_meta == exp_meta

    finally:
        # Clean up temp dirs to avoid ResourceWarning
        for td in temp_dirs:
            td.cleanup()


def test_initialize_storage_empty():
    temp_dirs, schemas = _initialize_storage([])

    assert temp_dirs == []
    assert schemas == []


@pytest.mark.parametrize(
    "invalid_input",
    [
        [123],
        ["string"],
        [object()],
        [make_df_0(), 42],
    ],
)
def test_initialize_storage_invalid_type_raises(invalid_input):
    with pytest.raises(TypeError, match="Unsupported data type"):
        _initialize_storage(invalid_input)


def make_df_1():
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})


def make_series_1():
    return pd.Series([10, 20], name="s")


def read_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


@pytest.mark.parametrize(
    "batch",
    [
        [make_df_1()],
        [make_series_1()],
        [make_df_1(), make_df_1()],
        [make_df_1(), make_series_1()],
    ],
)
def test_write_chunks_creates_files(batch):
    temp_dirs = [tempfile.TemporaryDirectory() for _ in batch]

    try:
        _write_chunks_to_disk(batch, temp_dirs, chunk_counter=0)

        for i, _ in enumerate(batch):
            expected_file = Path(temp_dirs[i].name) / "part_00000.parquet"
            assert expected_file.exists()

    finally:
        for td in temp_dirs:
            td.cleanup()


@pytest.mark.parametrize(
    "counter,expected_name",
    [
        (0, "part_00000.parquet"),
        (1, "part_00001.parquet"),
        (42, "part_00042.parquet"),
        (1234, "part_01234.parquet"),
    ],
)
def test_chunk_counter_format(counter, expected_name):
    batch = [make_df_1()]
    temp_dirs = [tempfile.TemporaryDirectory()]

    try:
        _write_chunks_to_disk(batch, temp_dirs, chunk_counter=counter)

        expected_file = Path(temp_dirs[0].name) / expected_name
        assert expected_file.exists()

    finally:
        temp_dirs[0].cleanup()


def test_series_written_as_dataframe():
    s = make_series_1()
    temp_dirs = [tempfile.TemporaryDirectory()]

    try:
        _write_chunks_to_disk([s], temp_dirs, chunk_counter=0)

        file_path = Path(temp_dirs[0].name) / "part_00000.parquet"
        df = read_parquet(file_path)

        # Series becomes single-column dataframe
        assert list(df.columns) == ["s"]
        assert df["s"].tolist() == [10, 20]

    finally:
        temp_dirs[0].cleanup()


def test_index_is_preserved():
    df = make_df_1()
    df.index = ["x", "y"]

    temp_dirs = [tempfile.TemporaryDirectory()]

    try:
        _write_chunks_to_disk([df], temp_dirs, chunk_counter=0)

        file_path = Path(temp_dirs[0].name) / "part_00000.parquet"
        result = read_parquet(file_path)

        assert list(result.index) == ["x", "y"]

    finally:
        temp_dirs[0].cleanup()


def test_multiple_chunk_writes():
    batch = [make_df_1()]
    temp_dirs = [tempfile.TemporaryDirectory()]

    try:
        _write_chunks_to_disk(batch, temp_dirs, chunk_counter=0)
        _write_chunks_to_disk(batch, temp_dirs, chunk_counter=1)

        file0 = Path(temp_dirs[0].name) / "part_00000.parquet"
        file1 = Path(temp_dirs[0].name) / "part_00001.parquet"

        assert file0.exists()
        assert file1.exists()

    finally:
        temp_dirs[0].cleanup()


def test_mismatched_temp_dirs_raises_index_error():
    batch = [make_df_1(), make_df_1()]
    temp_dirs = [tempfile.TemporaryDirectory()]  # only one dir

    try:
        with pytest.raises(IndexError):
            _write_chunks_to_disk(batch, temp_dirs, chunk_counter=0)
    finally:
        temp_dirs[0].cleanup()


def write_parquet(path: Path, df: pd.DataFrame):
    df.to_parquet(path, index=True)


def make_df(values, columns=("a",)):
    return pd.DataFrame(values, columns=columns)


def test_parquet_generator_dataframe():
    temp_dir = tempfile.TemporaryDirectory()

    try:
        df1 = make_df([[1], [2]])
        df2 = make_df([[3], [4]])

        write_parquet(Path(temp_dir.name) / "part_00000.parquet", df1)
        write_parquet(Path(temp_dir.name) / "part_00001.parquet", df2)

        gen = _parquet_generator(
            temp_dir=temp_dir,
            data_type=pd.DataFrame,
            schema=df1.columns,
        )

        outputs = list(gen)

        assert len(outputs) == 2
        pd.testing.assert_frame_equal(outputs[0], df1)
        pd.testing.assert_frame_equal(outputs[1], df2)

    finally:
        # Generator should already cleanup, but ensure no crash
        if Path(temp_dir.name).exists():
            temp_dir.cleanup()


def test_parquet_generator_series():
    temp_dir = tempfile.TemporaryDirectory()

    try:
        df1 = make_df([[10], [20]])
        df2 = make_df([[30], [40]])

        write_parquet(Path(temp_dir.name) / "part_00000.parquet", df1)
        write_parquet(Path(temp_dir.name) / "part_00001.parquet", df2)

        gen = _parquet_generator(
            temp_dir=temp_dir,
            data_type=pd.Series,
            schema="my_series",
        )

        outputs = list(gen)

        assert len(outputs) == 2
        assert isinstance(outputs[0], pd.Series)
        assert outputs[0].name == "my_series"
        assert outputs[0].tolist() == [10, 20]
        assert outputs[1].tolist() == [30, 40]

    finally:
        if Path(temp_dir.name).exists():
            temp_dir.cleanup()


def test_files_are_read_sorted():
    temp_dir = tempfile.TemporaryDirectory()

    try:
        df1 = make_df([[1]])
        df2 = make_df([[2]])

        # Intentionally reversed names
        write_parquet(Path(temp_dir.name) / "part_00001.parquet", df2)
        write_parquet(Path(temp_dir.name) / "part_00000.parquet", df1)

        gen = _parquet_generator(
            temp_dir=temp_dir,
            data_type=pd.DataFrame,
            schema=df1.columns,
        )

        outputs = list(gen)

        # Should be sorted lexicographically
        assert outputs[0]["a"].iloc[0] == 1
        assert outputs[1]["a"].iloc[0] == 2

    finally:
        if Path(temp_dir.name).exists():
            temp_dir.cleanup()


def test_empty_directory_yields_nothing():
    temp_dir = tempfile.TemporaryDirectory()

    gen = _parquet_generator(
        temp_dir=temp_dir,
        data_type=pd.DataFrame,
        schema=None,
    )

    outputs = list(gen)
    assert outputs == []


def test_cleanup_after_full_iteration():
    temp_dir = tempfile.TemporaryDirectory()

    df = make_df([[1]])
    write_parquet(Path(temp_dir.name) / "part_00000.parquet", df)

    gen = _parquet_generator(
        temp_dir=temp_dir,
        data_type=pd.DataFrame,
        schema=df.columns,
    )

    list(gen)

    # Directory should be removed after generator finishes
    assert not Path(temp_dir.name).exists()


def test_cleanup_on_partial_iteration():
    temp_dir = tempfile.TemporaryDirectory()

    df1 = make_df([[1]])
    df2 = make_df([[2]])

    write_parquet(Path(temp_dir.name) / "part_00000.parquet", df1)
    write_parquet(Path(temp_dir.name) / "part_00001.parquet", df2)

    gen = _parquet_generator(
        temp_dir=temp_dir,
        data_type=pd.DataFrame,
        schema=df1.columns,
    )

    next(gen)  # consume one element
    gen.close()  # trigger generator finalization

    assert not Path(temp_dir.name).exists()


def make_reader(chunks):
    return ParquetStreamReader(lambda: iter(chunks))


def df(val):
    return pd.DataFrame({"a": [val]})


def test_process_chunks_data_only():
    readers = [make_reader([df(1), df(2)])]

    def func(x):
        return x * 2

    result = _process_chunks(
        readers=readers,
        func=func,
        requested_types=(pd.DataFrame,),
        static_args=[],
        static_kwargs={},
        non_data_output="first",
        non_data_proc=None,
        non_data_proc_args=(),
        non_data_proc_kwargs={},
    )

    data_reader = result[0]
    output = data_reader.read()

    assert output["a"].tolist() == [2, 4]


def test_metadata_only_first_chunk():
    readers = [make_reader([df(1), df(2)])]

    def func(x):
        return x, f"meta_{x['a'].iloc[0]}"

    result = _process_chunks(
        readers=readers,
        func=func,
        requested_types=(pd.DataFrame,),
        static_args=[],
        static_kwargs={},
        non_data_output="first",
        non_data_proc=None,
        non_data_proc_args=(),
        non_data_proc_kwargs={},
    )

    data_reader, meta = result

    assert data_reader.read()["a"].tolist() == [1, 2]
    assert meta == "meta_1"  # only first chunk captured


def test_metadata_accumulation():
    readers = [make_reader([df(1), df(2)])]

    def func(x):
        return x, x["a"].iloc[0]

    result = _process_chunks(
        readers=readers,
        func=func,
        requested_types=(pd.DataFrame,),
        static_args=[],
        static_kwargs={},
        non_data_output="acc",
        non_data_proc=None,
        non_data_proc_args=(),
        non_data_proc_kwargs={},
    )

    _, meta = result

    assert meta == [1, 2]


def test_non_data_proc_applied_helper():
    readers = [make_reader([df(1), df(2)])]

    def func(x):
        return x, x["a"].iloc[0]

    def processor(meta):
        return sum(meta)

    result = _process_chunks(
        readers=readers,
        func=func,
        requested_types=(pd.DataFrame,),
        static_args=[],
        static_kwargs={},
        non_data_output="acc",
        non_data_proc=processor,
        non_data_proc_args=(),
        non_data_proc_kwargs={},
    )

    _, meta = result

    assert meta == 3


def test_only_metadata_output():
    readers = [make_reader([df(1), df(2)])]

    def func(x):
        return x["a"].iloc[0]

    result = _process_chunks(
        readers=readers,
        func=func,
        requested_types=(pd.DataFrame,),
        static_args=[],
        static_kwargs={},
        non_data_output="acc",
        non_data_proc=None,
        non_data_proc_args=(),
        non_data_proc_kwargs={},
    )

    # Should return metadata only
    assert result == [1, 2]


def test_empty_iterable_raises():
    readers = [make_reader([])]

    def func(x):
        return x

    with pytest.raises(ValueError, match="Iterable is empty"):
        _process_chunks(
            readers=readers,
            func=func,
            requested_types=(pd.DataFrame,),
            static_args=[],
            static_kwargs={},
            non_data_output="first",
            non_data_proc=None,
            non_data_proc_args=(),
            non_data_proc_kwargs={},
        )


def test_invalid_type_raises():
    readers = [make_reader(["not_df"])]

    def func(x):
        return x

    with pytest.raises(TypeError):
        _process_chunks(
            readers=readers,
            func=func,
            requested_types=(pd.DataFrame,),
            static_args=[],
            static_kwargs={},
            non_data_output="first",
            non_data_proc=None,
            non_data_proc_args=(),
            non_data_proc_kwargs={},
        )


def test_multiple_readers():
    r1 = make_reader([df(1), df(2)])
    r2 = make_reader([df(10), df(20)])

    def func(x, y):
        return x + y

    result = _process_chunks(
        readers=[r1, r2],
        func=func,
        requested_types=(pd.DataFrame,),
        static_args=[],
        static_kwargs={},
        non_data_output="first",
        non_data_proc=None,
        non_data_proc_args=(),
        non_data_proc_kwargs={},
    )

    data_reader = result[0]
    output = data_reader.read()

    assert output["a"].tolist() == [11, 22]


def make_reader_2(values=None):
    if values is None:
        values = []
    return ParquetStreamReader(values)


def make_df_2(val):
    return pd.DataFrame({"a": [val]})


def test_base_reader_only():
    base = make_reader_2([make_df_2(1)])

    readers, args, kwargs = _prepare_readers(
        reader=base,
        func_args=[],
        func_kwargs={},
        makecopy=False,
    )

    assert readers == [base]
    assert args == []
    assert kwargs == {}


@pytest.mark.parametrize(
    "func_args,expected_reader_count,expected_static_len",
    [
        ([], 1, 0),
        ([123], 1, 1),
        ([make_reader_2()], 2, 0),
        ([make_reader_2(), 999], 2, 1),
    ],
)
def test_func_args_separation(func_args, expected_reader_count, expected_static_len):
    base = make_reader_2([make_df_2(1)])

    readers, args, kwargs = _prepare_readers(
        reader=base,
        func_args=func_args,
        func_kwargs={},
        makecopy=False,
    )

    assert len(readers) == expected_reader_count
    assert len(args) == expected_static_len
    assert kwargs == {}


def test_func_kwargs_separation():
    base = make_reader_2([make_df_2(1)])
    reader_kw = make_reader_2([make_df_2(2)])

    readers, args, kwargs = _prepare_readers(
        reader=base,
        func_args=[],
        func_kwargs={"r": reader_kw, "x": 42},
        makecopy=False,
    )

    assert len(readers) == 2
    assert args == []
    assert kwargs == {"x": 42}


def test_reader_ordering():
    base = make_reader_2()
    r1 = make_reader_2()
    r2 = make_reader_2()

    readers, _, _ = _prepare_readers(
        reader=base,
        func_args=[r1],
        func_kwargs={"k": r2},
        makecopy=False,
    )

    assert readers[0] is base
    assert readers[1] is r1
    assert readers[2] is r2


def test_makecopy_false_preserves_identity():
    base = make_reader_2()
    r1 = make_reader_2()

    readers, _, _ = _prepare_readers(
        reader=base,
        func_args=[r1],
        func_kwargs={},
        makecopy=False,
    )

    assert readers[0] is base
    assert readers[1] is r1


def test_makecopy_true_creates_copies():
    base = make_reader_2([make_df_2(1)])
    r1 = make_reader_2([make_df_2(2)])

    readers, _, _ = _prepare_readers(
        reader=base,
        func_args=[r1],
        func_kwargs={},
        makecopy=True,
    )

    # Copies should not be the same object
    assert readers[0] is not base
    assert readers[1] is not r1

    # But should behave identically
    assert readers[0].read()["a"].tolist() == [1]
    assert readers[1].read()["a"].tolist() == [2]


def test_empty_args_and_kwargs():
    base = make_reader_2()

    readers, args, kwargs = _prepare_readers(
        reader=base,
        func_args=[],
        func_kwargs={},
        makecopy=False,
    )

    assert readers == [base]
    assert args == []
    assert kwargs == {}


def make_df_3(val):
    return pd.DataFrame({"a": [val]})


def make_series_3(val, name="s"):
    return pd.Series([val], name=name)


def reader_from_list(items):
    return iter(items)


@pytest.mark.parametrize(
    "input_data,requested_types",
    [
        ([make_df_3(1), make_df_3(2)], (pd.DataFrame,)),
        ([make_series_3(10), make_series_3(20)], (pd.Series,)),
    ],
)
def test_basic_processing(input_data, requested_types):
    def func(x):
        return x

    result = process_disk_backed(
        reader=reader_from_list(input_data),
        func=func,
        requested_types=requested_types,
    )

    # First element is a generator
    gen = result[0]

    output = list(gen)
    assert all(isinstance(o, requested_types) for o in output)

    if isinstance(output[0], pd.DataFrame):
        assert [row["a"].iloc[0] for row in output] == [df["a"].iloc[0] for df in input_data if isinstance(df, pd.DataFrame)]
    else:
        assert [o.iloc[0] for o in output] == [s.iloc[0] for s in input_data if isinstance(s, pd.Series)]


def test_non_data_first_mode():
    def func(df):
        return df, df["a"].iloc[0]

    result = process_disk_backed(
        reader=reader_from_list([make_df_3(1), make_df_3(2)]),
        func=func,
        non_data_output="first",
    )

    gen, meta = result

    # Only first chunk captured
    assert meta == 1
    output = list(gen)
    assert [row["a"].iloc[0] for row in output] == [1, 2]


def test_non_data_acc_mode():
    def func(df):
        return df, df["a"].iloc[0]

    result = process_disk_backed(
        reader=reader_from_list([make_df_3(1), make_df_3(2)]),
        func=func,
        non_data_output="acc",
    )

    gen, meta = result
    assert meta == [1, 2]

    output = list(gen)
    assert [row["a"].iloc[0] for row in output] == [1, 2]


def test_non_data_proc_applied_function():
    def func(df):
        return df, df["a"].iloc[0]

    def processor(meta, factor):
        return [x * factor for x in meta]

    result = process_disk_backed(
        reader=reader_from_list([make_df_3(1), make_df_3(2)]),
        func=func,
        non_data_output="acc",
        non_data_proc=processor,
        non_data_proc_args=(10,),
        non_data_proc_kwargs={},
    )

    gen, meta = result
    assert meta == [10, 20]

    output = list(gen)
    assert [row["a"].iloc[0] for row in output] == [1, 2]


def test_func_args_kwargs():
    def func(df, val, extra=0):
        return df * val + extra

    result = process_disk_backed(
        reader=reader_from_list([make_df_3(1), make_df_3(2)]),
        func=func,
        func_args=[2],
        func_kwargs={"extra": 5},
    )

    gen = result[0]
    output = list(gen)
    assert [row["a"].iloc[0] for row in output] == [1 * 2 + 5, 2 * 2 + 5]


def test_empty_iterator_raises():
    def func(x):
        return x

    with pytest.raises(ValueError, match="Iterable is empty"):
        process_disk_backed(
            reader=reader_from_list([]),
            func=func,
        )


def test_requested_types_single_type():
    def func(x):
        return x

    input_data = [make_df_3(1)]
    # requested_types as single type
    result = process_disk_backed(
        reader=reader_from_list(input_data),
        func=func,
        requested_types=pd.DataFrame,
    )

    gen = result[0]
    output = list(gen)
    assert all(isinstance(o, pd.DataFrame) for o in output)


def test_parquet_stream_from_iterable_dataframe():
    dfs = [make_df_3(1), make_df_3(2)]
    reader = parquet_stream_from_iterable(dfs)

    assert isinstance(reader, ParquetStreamReader)
    output = list(reader)
    assert all(isinstance(df, pd.DataFrame) for df in output)
    assert [df["a"].iloc[0] for df in output] == [1, 2]


def test_parquet_stream_from_iterable_series():
    series_list = [make_series_3(10), make_series_3(20)]
    reader = parquet_stream_from_iterable(series_list)

    assert isinstance(reader, ParquetStreamReader)
    output = list(reader)
    assert all(isinstance(s, pd.Series) for s in output)
    assert [s.iloc[0] for s in output] == [10, 20]


def test_parquet_stream_from_iterable_empty_raises():
    with pytest.raises(ValueError, match="Iterable is empty"):
        parquet_stream_from_iterable([])


def test_parquet_stream_from_iterable_mixed_types_raises():
    dfs = [make_df_3(1), make_series_3(2)]
    with pytest.raises(TypeError, match="All chunks must be of the same type"):
        parquet_stream_from_iterable(dfs)


def test_parquet_stream_from_iterable_wrong_type_first_raises():
    with pytest.raises(TypeError, match="Iterable must contain pd.DataFrame or pd.Series"):
        parquet_stream_from_iterable([123, 456])


def test_ensure_parquet_reader_returns_existing_reader():
    reader = parquet_stream_from_iterable([make_df_3(1)])
    result = ensure_parquet_reader(reader)
    assert result is reader


def test_ensure_parquet_reader_converts_iterator():
    dfs = [make_df_3(1), make_df_3(2)]
    iterator = iter(dfs)
    result = ensure_parquet_reader(iterator)
    assert isinstance(result, ParquetStreamReader)
    output = list(result)
    assert [df["a"].iloc[0] for df in output] == [1, 2]


def test_ensure_parquet_reader_returns_non_iterator_unchanged():
    obj = 123
    result = ensure_parquet_reader(obj)
    assert result == 123


@pytest.mark.parametrize(
    "value,expected",
    [
        (iter([1, 2, 3]), True),  # iterator
        ((x for x in range(5)), True),  # generator expression
        ([1, 2, 3], False),  # list
        ((1, 2, 3), False),  # tuple
        (123, False),  # int
        ("abc", False),  # string
        (None, False),  # None
    ],
)
def test_is_valid_iterator(value, expected):
    assert is_valid_iterator(value) is expected


def test_non_process_function_returns():
    val = 123
    assert _process_function(val) == val


def test_dataframe_calls_func_directly():
    df = make_df_3(5)

    called = {}

    def func(d):
        called["data"] = d
        return d["a"].iloc[0] * 2

    pf = ProcessFunction(df, func)
    result = _process_function(pf)

    assert result == 10
    assert called["data"] is df


def test_series_calls_func_directly():
    s = make_series_3(7)

    def func(x):
        return x.iloc[0] + 3

    pf = ProcessFunction(s, func)
    result = _process_function(pf)
    assert result == 10


def test_xarray_dataset_direct_call():
    ds = xr.Dataset({"a": ("x", [1, 2])})

    def func(x):
        return x["a"].sum().item()

    pf = ProcessFunction(ds, func)
    result = _process_function(pf)
    assert result == 3


def test_iterator_of_dataframes_disk_backed():
    dfs = [make_df_3(1), make_df_3(2)]
    it = iter(dfs)

    def func(df):
        return df["a"].iloc[0] * 10

    pf = ProcessFunction(it, func, non_data_output="acc")
    result = _process_function(pf)
    assert result == [10, 20]


def test_list_of_dataframes_disk_backed():
    dfs = [make_df_3(3), make_df_3(4)]

    def func(df):
        return df["a"].iloc[0] * 2

    pf = ProcessFunction(dfs, func, non_data_output="acc")
    result = _process_function(pf)
    assert result == [6, 8]


def test_data_only_returns_first():
    dfs = [make_df_3(1)]
    pf = ProcessFunction(dfs, lambda df: df)
    result = _process_function(pf, data_only=True)
    assert isinstance(result, ParquetStreamReader)


def test_unsupported_type_raises():
    pf = ProcessFunction(12345, lambda x: x)
    with pytest.raises(TypeError, match="Unsupported data type"):
        _process_function(pf)


def test_basic_dataframe_decorator():
    @process_function()
    def func(df):
        return df * 2

    df = make_df_3(3)
    result = func(df)
    assert isinstance(result, pd.DataFrame)
    assert result["a"].iloc[0] == 6


def test_iterable_returns_disk_backed():
    @process_function()
    def func(dfs):
        return dfs

    dfs = [make_df_3(1), make_df_3(2)]
    result = func(dfs)

    assert isinstance(result, list)
    assert len(result) == 2

    pd.testing.assert_frame_equal(result[0], pd.DataFrame({"a": [1]}))
    pd.testing.assert_frame_equal(result[1], pd.DataFrame({"a": [2]}))


def test_data_only_returns_generator_only():
    @process_function(data_only=True)
    def func(dfs):
        return dfs

    dfs = [make_df_3(1)]
    result = func(dfs)

    assert isinstance(result, list)
    assert len(result) == 1

    pd.testing.assert_frame_equal(result[0], pd.DataFrame({"a": [1]}))


def test_postprocessing_not_callable_raises():
    @process_function(postprocessing={"func": 123, "kwargs": []})
    def func(df):
        return df

    df = make_df_3(1)
    with pytest.raises(ValueError, match="is not callable"):
        func(df)
