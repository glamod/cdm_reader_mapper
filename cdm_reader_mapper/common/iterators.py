"""Utilities for handling pandas TextParser objects safely."""

from __future__ import annotations

import tempfile

import inspect
import itertools

import pandas as pd
import xarray as xr

import pyarrow as pa
import pyarrow.parquet as pq

from functools import wraps

from pathlib import Path

from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Literal,
    Sequence,
)


class ProcessFunction:
    """Stores data and a callable function with optional arguments for processing."""

    def __init__(
        self,
        data: pd.DataFrame | pd.Series | Iterable[pd.DataFrame] | Iterable[pd.Series],
        func: Callable[..., Any],
        func_args: Any | list[Any] | tuple[Any] | None = None,
        func_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        self.data = data

        if not isinstance(func, Callable):
            raise ValueError(f"Function {func} is not callable.")

        self.func = func

        if func_args is None:
            func_args = ()

        if not isinstance(func_args, (list, tuple)):
            func_args = (func_args,)

        self.func_args = func_args

        if func_kwargs is None:
            func_kwargs = {}

        self.func_kwargs = func_kwargs

        self.kwargs = kwargs


class ParquetStreamReader:
    """A wrapper that mimics pandas.io.parsers.TextFileReader."""

    def __init__(
        self,
        source: (
            Iterator[pd.DataFrame | pd.Series]
            | Callable[[], Iterator[pd.DataFrame | pd.Series]]
        ),
    ):
        self._closed = False
        self._buffer: list[pd.DataFrame | pd.Series] = []

        if callable(source):
            # factory that produces a fresh iterator
            self._factory = source
        elif isinstance(source, Iterator):
            self._factory = lambda: source
        else:
            raise TypeError(
                "ParquetStreamReader expects an iterator or a factory callable."
            )

        self._generator = self._factory()

    def __iter__(self):
        """Allows: for df in reader: ..."""
        return self

    def __next__(self):
        """Allows: next(reader)"""
        if self._closed:
            raise ValueError("I/O operation on closed stream.")
        if self._buffer:
            return self._buffer.pop(0)
        return next(self._generator)

    def prepend(self, chunk: pd.DataFrame | pd.Series):
        """
        Push a chunk back onto the front of the stream.
        Useful for peeking at the first chunk without losing it.
        """
        # Insert at 0 ensures FIFO order (peeking logic)
        self._buffer.insert(0, chunk)

    def get_chunk(self):
        """
        Safe for Large Files.
        Returns the next single chunk from disk.
        (Note: 'size' is ignored here as chunks are pre-determined by the write step)
        """
        return next(self)

    def read(
        self,
    ):
        """
        WARNING: unsafe for Files > RAM.
        Reads ALL remaining data into memory at once.
        """
        # Consume the entire rest of the stream
        chunks = list(self)

        if not chunks:
            return pd.DataFrame()

        df = pd.concat(chunks)
        return df

    def copy(self):
        """Create an independent copy of the stream."""
        if self._closed:
            raise ValueError("Cannot copy a closed stream.")
        self._generator, new_gen = itertools.tee(self._generator)
        return ParquetStreamReader(new_gen)

    def empty(self):
        """Return True if stream is empty."""
        copy_stream = self.copy()

        try:
            next(copy_stream)
            return False
        except StopIteration:
            return True

    def reset_index(self, drop=False):
        """Reset indexes continuously."""
        if self._closed:
            raise ValueError("Cannot copy a closed stream.")

        offset = 0
        chunks = []

        for df in self:
            df = df.copy()
            n = len(df)

            indexes = range(offset, offset + n)
            df.index = indexes

            if drop is False:
                df.insert(0, "index", indexes)

            offset += n
            chunks.append(df)

        return ParquetStreamReader(lambda: iter(chunks))

    def close(self):
        """Close the stream and release resources."""
        self._closed = True

    def __enter__(self):
        """Allows: with ParquetStreamReader(...) as reader: ..."""
        return self

    def __exit__(self, *_):
        """Allows: with ParquetStreamReader(...) as reader: ..."""
        self.close()


def _sort_chunk_outputs(
    outputs: tuple, capture_meta: bool, requested_types: tuple[type, ...]
) -> tuple[list[pd.DataFrame | pd.Series], list[Any]]:
    """Separates DataFrames from metadata in the function output."""
    data, meta = [], []
    for out in outputs:
        if isinstance(out, requested_types):
            data.append(out)
        elif isinstance(out, list) and out and isinstance(out[0], requested_types):
            data.extend(out)
        elif capture_meta:
            # Only capture metadata from the first chunk
            meta.append(out)

    return data, meta


def _initialize_storage(
    first_batch: list[pd.DataFrame | pd.Series],
) -> tuple[list, list]:
    """Creates temp directories and captures schemas from the first chunk."""
    temp_dirs = []
    schemas = []

    for obj in first_batch:
        if isinstance(obj, pd.DataFrame):
            schemas.append((pd.DataFrame, obj.columns))
        elif isinstance(obj, pd.Series):
            schemas.append((pd.Series, obj.name))
        else:
            raise TypeError(
                f"Unsupported data type: {type(obj)}."
                "Use one of [pd.DataFrame, pd.Series]."
            )

        temp_dirs.append(tempfile.TemporaryDirectory())

    return temp_dirs, schemas


def _write_chunks_to_disk(
    batch: list[pd.DataFrame | pd.Series],
    temp_dirs: list[tempfile.TemporaryDirectory],
    chunk_counter: int,
) -> None:
    """Writes the current batch of DataFrames to their respective temp directories."""
    for i, data_out in enumerate(batch):
        if isinstance(data_out, pd.Series):
            data_out = data_out.to_frame()

        file_path = Path(temp_dirs[i].name) / f"part_{chunk_counter:05d}.parquet"

        table = pa.Table.from_pandas(data_out, preserve_index=True)
        pq.write_table(table, file_path, compression="snappy")


def _parquet_generator(
    temp_dir, data_type, schema
) -> Generator[pd.DataFrame | pd.Series]:
    """Yields DataFrames from a temp directory, restoring schema."""
    try:
        files = sorted(Path(temp_dir.name).glob("*.parquet"))

        for f in files:
            df = pd.read_parquet(f)

            if data_type is pd.Series:
                s = df.iloc[:, 0].copy()
                s.name = schema
                yield s
            else:
                yield df

    finally:
        temp_dir.cleanup()


def _process_chunks(
    readers: list[ParquetStreamReader],
    func: Callable[..., Any],
    requested_types: tuple[str],
    static_args: list[Any],
    static_kwargs: dict[str, Any],
    non_data_output: str,
    non_data_proc: Callable[..., Any] | None,
    non_data_proc_args: tuple[Any] | None,
    non_data_proc_kwargs: dict[str, Any] | None,
):
    """Process chunks."""
    # State variables
    temp_dirs = None
    schemas = None
    output_non_data: dict[int, list[Any]] = {}
    chunk_counter: int = 0

    for items in zip(*readers):

        if not isinstance(items[0], requested_types):
            raise TypeError(
                f"Unsupported data type in Iterable {items[0]}: {type(items[0])}"
                f"Requested types are: {requested_types} "
            )

        result = func(*items, *static_args, **static_kwargs)
        if not isinstance(result, tuple):
            result = (result,)

        # Sort outputs
        capture_meta = non_data_output == "acc" or chunk_counter == 0

        data, meta = _sort_chunk_outputs(result, capture_meta, requested_types)

        for i, meta in enumerate(meta):
            output_non_data.setdefault(i, []).append(meta)

        # Write DataFrames
        if data:
            if temp_dirs is None:
                temp_dirs, schemas = _initialize_storage(data)

            _write_chunks_to_disk(data, temp_dirs, chunk_counter)

        chunk_counter += 1

    if chunk_counter == 0:
        raise ValueError("Iterable is empty.")

    keys = list(output_non_data.keys())
    if len(keys) == 1:
        output_non_data = output_non_data[keys[0]]

    if isinstance(output_non_data, list) and len(output_non_data) == 1:
        output_non_data = output_non_data[0]

    if isinstance(non_data_proc, Callable):
        output_non_data = non_data_proc(
            output_non_data, *non_data_proc_args, **non_data_proc_kwargs
        )

    # If no data outputs at all
    if temp_dirs is None:
        return output_non_data

    final_iterators = [
        ParquetStreamReader(lambda d=d, t=t, s=s: _parquet_generator(d, t, s))
        for d, (t, s) in zip(temp_dirs, schemas)
    ]

    if isinstance(output_non_data, tuple):
        output_non_data = list(output_non_data)
    else:
        output_non_data = [output_non_data]

    return tuple(final_iterators + output_non_data)


def _prepare_readers(
    reader: Iterator[pd.DataFrame | pd.Series],
    func_args: Sequence[Any],
    func_kwargs: dict[str, Any],
    makecopy: bool,
) -> tuple[list[ParquetStreamReader], list[Any], dict[str, Any]]:
    """Prepare readers for chunking."""
    reader = ensure_parquet_reader(reader)

    args_reader = []
    args = []
    for arg in func_args:
        converted = ensure_parquet_reader(arg)
        if isinstance(converted, ParquetStreamReader):
            args_reader.append(converted)
        else:
            args.append(converted)

    kwargs = {}
    for k, v in func_kwargs.items():
        converted = ensure_parquet_reader(v)
        if isinstance(converted, ParquetStreamReader):
            args_reader.append(converted)
        else:
            kwargs[k] = converted

    readers = [reader] + args_reader

    if makecopy:
        readers = [r.copy() for r in readers]

    return readers, args, kwargs


def parquet_stream_from_iterable(
    iterable: Iterable[pd.DataFrame | pd.Series],
) -> ParquetStreamReader:
    """
    Stream an iterable of DataFrame/Series to parquet
    and return a disk-backed ParquetStreamReader.

    Memory usage remains constant.
    """
    iterator = iter(iterable)

    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("Iterable is empty.")

    if not isinstance(first, (pd.DataFrame, pd.Series)):
        raise TypeError("Iterable must contain pd.DataFrame or pd.Series objects.")

    temp_dir = tempfile.TemporaryDirectory()
    temp_dirs = [temp_dir]

    if isinstance(first, pd.DataFrame):
        data_type = pd.DataFrame
        schema = first.columns
    else:
        data_type = pd.Series
        schema = first.name
    _write_chunks_to_disk([first], temp_dirs, chunk_counter=0)

    for idx, chunk in enumerate(iterator, start=1):
        if not isinstance(chunk, type(first)):
            raise TypeError("All chunks must be of the same type.")

        _write_chunks_to_disk([chunk], temp_dirs, chunk_counter=idx)

    return ParquetStreamReader(lambda: _parquet_generator(temp_dir, data_type, schema))


def is_valid_iterator(reader: Any) -> bool:
    """Check if reader is a valid Iterable."""
    return isinstance(reader, Iterator)


def ensure_parquet_reader(obj: Any) -> Any:
    """Ensure obj is a ParquetStreamReader."""
    if isinstance(obj, ParquetStreamReader):
        return obj

    if is_valid_iterator(obj):
        return parquet_stream_from_iterable(obj)

    return obj


def process_disk_backed(
    reader: Iterator[pd.DataFrame | pd.Series],
    func: Callable[..., Any],
    func_args: Sequence[Any] | None = None,
    func_kwargs: dict[str, Any] | None = None,
    requested_types: type | tuple[type, ...] = (pd.DataFrame, pd.Series),
    non_data_output: Literal["first", "acc"] = "first",
    non_data_proc: Callable[..., Any] | None = None,
    non_data_proc_args: tuple[Any] | None = None,
    non_data_proc_kwargs: dict[str, Any] | None = None,
    makecopy: bool = True,
) -> tuple[Any, ...]:
    """
    Consumes a stream of DataFrames, processes them, and returns a tuple of
    results. DataFrames are cached to disk (Parquet) and returned as generators.
    """
    if func_args is None:
        func_args = ()
    if func_kwargs is None:
        func_kwargs = {}

    if not isinstance(requested_types, (list, tuple)):
        requested_types = (requested_types,)

    readers, static_args, static_kwargs = _prepare_readers(
        reader, func_args, func_kwargs, makecopy
    )

    if non_data_proc is not None:
        if not isinstance(non_data_proc, Callable):
            raise ValueError(f"Function {non_data_proc} is not callable.")

        if non_data_proc_args is None:
            non_data_proc_args = ()
        if non_data_proc_kwargs is None:
            non_data_proc_kwargs = {}

    return _process_chunks(
        readers,
        func,
        requested_types,
        static_args,
        static_kwargs,
        non_data_output,
        non_data_proc,
        non_data_proc_args,
        non_data_proc_kwargs,
    )


def _process_function(results, data_only=False):
    if not isinstance(results, ProcessFunction):
        return results

    data = results.data
    func = results.func
    args = results.func_args
    kwargs = results.func_kwargs

    if isinstance(data, (pd.DataFrame, pd.Series, xr.Dataset, xr.DataArray)):
        return func(data, *args, **kwargs)

    if is_valid_iterator(data) and not isinstance(data, ParquetStreamReader):
        data = parquet_stream_from_iterable(data)

    if isinstance(data, (list, tuple)):
        data = parquet_stream_from_iterable(data)

    if not isinstance(data, ParquetStreamReader):
        raise TypeError(f"Unsupported data type: {type(data)}")

    result = process_disk_backed(
        data,
        func,
        func_args=args,
        func_kwargs=kwargs,
        **results.kwargs,
    )

    if data_only is True:
        result = result[0]

    return result


def process_function(data_only=False, postprocessing=None):
    """Decorator to apply function to both pd.DataFrame and Iterable[pd.DataFrame]."""

    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            original_call = bound_args.arguments.copy()

            result_class = func(*args, **kwargs)
            results = _process_function(
                result_class,
                data_only=data_only,
            )

            if postprocessing is None:
                return results

            postproc_func = postprocessing.get("func")
            if not isinstance(postproc_func, Callable):
                raise ValueError(f"Function {postproc_func} is not callable.")
            postproc_list = postprocessing.get("kwargs", {})
            if isinstance(postproc_list, str):
                postproc_list = [postproc_list]

            postproc_kwargs = {k: original_call[k] for k in postproc_list}

            result_list = []
            for result in results:
                if isinstance(result, (pd.DataFrame, pd.Series, ParquetStreamReader)):
                    result = postproc_func(result, **postproc_kwargs)
                result_list.append(result)

            return tuple(result_list)

        return wrapper

    return decorator
