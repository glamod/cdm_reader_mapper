"""Utilities for handling pandas TextParser objects safely."""

from __future__ import annotations

import tempfile

import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

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


class ParquetStreamReader:
    """A wrapper that mimics pandas.io.parsers.TextFileReader."""

    def __init__(self, generator: Iterator[pd.DataFrame | pd.Series]):
        self._generator = generator
        self._closed = False
        self._buffer = []

    def __iter__(self):
        """Allows: for df in reader: ..."""
        return self

    def __next__(self):
        """Allows: next(reader)"""
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
        if self._closed:
            raise ValueError("I/O operation on closed file.")

        try:
            return next(self._generator)
        except StopIteration:
            raise ValueError("No more data to read (End of stream).")

    def read(self):
        """
        WARNING: unsafe for Files > RAM.
        Reads ALL remaining data into memory at once.
        """
        if self._closed:
            raise ValueError("I/O operation on closed file.")

        # Consume the entire rest of the stream
        chunks = list(self._generator)

        if not chunks:
            return pd.DataFrame()

        df = pd.concat(chunks)
        if df.index.has_duplicates:
            df = df.reset_index(drop=True)
        return df

    def close(self):
        """Close the stream and release resources."""
        if not self._closed:
            self._generator.close()
            self._closed = True

    def __enter__(self):
        """Allows: with ParquetStreamReader(...) as reader: ..."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Allows: with ParquetStreamReader(...) as reader: ..."""
        self.close()


def _sort_chunk_outputs(
    outputs: tuple, accumulators_initialized: bool, requested_types: tuple[type]
) -> tuple[list[pd.DataFrame | pd.Series], list[Any]]:
    """Separates DataFrames from metadata in the function output."""
    current_data = []
    new_metadata = []
    for out in outputs:
        if isinstance(out, requested_types):
            current_data.append(out)
        elif isinstance(out, list) and out and isinstance(out[0], requested_types):
            current_data.extend(out)
        elif not accumulators_initialized:
            # Only capture metadata from the first chunk
            new_metadata.append(out)

    return current_data, new_metadata


def _write_chunks_to_disk(current_data: list, temp_dirs: list, chunk_counter: int):
    """Writes the current batch of DataFrames to their respective temp directories."""
    for i, data_out in enumerate(current_data):
        if i < len(temp_dirs):
            if isinstance(data_out, pd.Series):
                data_out = data_out.to_frame()
            file_path = Path(temp_dirs[i].name) / f"part_{chunk_counter:05d}.parquet"
            data_out = data_out.reset_index()

            table = pa.Table.from_pandas(data_out, preserve_index=False)

            pq.write_table(table, file_path, compression="snappy")


def _initialize_storage(
    current_data: list[pd.DataFrame | pd.Series],
) -> tuple[list, list, list]:
    """Creates temp directories and captures schemas from the first chunk."""

    def _get_columns(data):
        if isinstance(data, pd.DataFrame):
            return type(data), data.columns
        if isinstance(data, pd.Series):
            return type(data), data.name
        raise TypeError(
            f"Unsupported data type: {type(data)}."
            "Use one of [pd.DataFrame, pd.Series]."
        )

    temp_dirs = []
    to_cleanup = []
    schemas = [_get_columns(df) for df in current_data]

    for _ in range(len(current_data)):
        t = tempfile.TemporaryDirectory()
        temp_dirs.append(t)
        to_cleanup.append(t)

    return temp_dirs, to_cleanup, schemas


def _parquet_generator(
    temp_dir_obj, data_type, schema
) -> Generator[pd.DataFrame | pd.Series]:
    """Yields DataFrames from a temp directory, restoring schema."""

    def _is_tuple_like(s):
        s = s.strip()
        return s.startswith("(") and s.endswith(")")

    if isinstance(schema, (tuple, list)):
        schema = [schema]

    try:
        files = sorted(Path(temp_dir_obj.name).glob("*.parquet"))
        for f in files:
            data = pd.read_parquet(f)
            idx = "('index', '')" if _is_tuple_like(data.columns[0]) else "index"
            if idx in data.columns:
                data = data.set_index(idx).rename_axis(None)
            if schema is not None:
                data.columns = schema

            if data_type == pd.Series:
                data = data.iloc[:, 0]
                if schema is None:
                    data.name = schema

            yield data
    finally:
        temp_dir_obj.cleanup()


def is_valid_iterable(reader: Any) -> bool:
    """Check if reader is a valid Iterable."""
    if not isinstance(reader, Iterator):
        return False
    if not isinstance(reader, Iterable):
        return False
    return True


def process_disk_backed(
    reader: Iterable[pd.DataFrame | pd.Series],
    func: Callable,
    func_args: Sequence[Any] | None = None,
    func_kwargs: dict[str, Any] | None = None,
    requested_types: type | list[type] | tuple[type] = (pd.DataFrame, pd.Series),
    non_data_output: Literal["first", "acc"] = "first",
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

    # State variables
    temp_dirs: list[tempfile.TemporaryDirectory] = []
    column_schemas = []
    output_non_data = {}
    directories_to_cleanup = []

    if not isinstance(requested_types, (list, tuple)):
        requested_types = (requested_types,)

    args_reader = []
    args = []
    for arg in func_args:
        if is_valid_iterable(arg):
            args_reader.append(arg)
        else:
            args.append(arg)

    kwargs = {}
    for k, v in func_kwargs.items():
        if is_valid_iterable(v):
            args_reader.append(v)
        else:
            kwargs[k] = v

    readers = [reader] + args_reader

    try:
        accumulators_initialized = False
        chunk_counter = 0

        for items in zip(*readers):
            if not isinstance(items[0], requested_types):
                raise TypeError(
                    "Unsupported data type in Iterable: {type(items[0])}"
                    "Requested types are: {requested_types} "
                )

            if makecopy:
                items = tuple(df.copy() for df in items)

            outputs = func(*items, *args, **kwargs)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            # Sort outputs
            accumulate_outputs = (
                accumulators_initialized if non_data_output != "acc" else False
            )
            current_data, new_meta = _sort_chunk_outputs(
                outputs, accumulate_outputs, requested_types
            )

            if new_meta:
                j = 0
                for meta in new_meta:
                    if j in output_non_data:
                        output_non_data[j].append(meta)
                    else:
                        output_non_data[j] = [meta]
                    j += 1

            # Initialize storage
            if not accumulators_initialized and current_data:
                temp_dirs, directories_to_cleanup, column_schemas = _initialize_storage(
                    current_data
                )
                accumulators_initialized = True

            # Write DataFrames
            if accumulators_initialized:
                _write_chunks_to_disk(current_data, temp_dirs, chunk_counter)

            chunk_counter += 1

        if chunk_counter == 0:
            raise ValueError("Iterable is empty.")

        if not accumulators_initialized:
            return output_non_data

        # Finalize Iterators
        final_iterators = [
            ParquetStreamReader(_parquet_generator(d, t, s))
            for d, (t, s) in zip(temp_dirs, column_schemas)
        ]

        # Transfer ownership to generators
        directories_to_cleanup.clear()

        return tuple(final_iterators + [output_non_data])

    finally:
        for d in directories_to_cleanup:
            d.cleanup()
