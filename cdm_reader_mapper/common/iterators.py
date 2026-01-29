"""Utilities for handling pandas TextParser objects safely."""

from __future__ import annotations

import tempfile

import pandas as pd

from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Iterator, Sequence


class ParquetStreamReader:
    """A wrapper that mimics pandas.io.parsers.TextFileReader."""

    def __init__(self, generator: Iterator[pd.DataFrame]):
        self._generator = generator
        self._closed = False
        self._buffer = []

    def __iter__(self):
        """Allows: for df in reader: ..."""
        return self

    def __next__(self):
        """Allows: next(reader)"""
        return next(self._generator)

    def prepend(self, chunk: pd.DataFrame):
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

        return pd.concat(chunks, ignore_index=True)

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
    outputs: tuple, accumulators_initialized: bool
) -> tuple[list[pd.DataFrame], list[Any]]:
    """Separates DataFrames from metadata in the function output."""
    current_dfs = []
    new_metadata = []

    for out in outputs:
        if isinstance(out, pd.DataFrame):
            current_dfs.append(out)
        elif isinstance(out, list) and out and isinstance(out[0], pd.DataFrame):
            current_dfs.extend(out)
        elif not accumulators_initialized:
            # Only capture metadata from the first chunk
            new_metadata.append(out)

    return current_dfs, new_metadata


def _write_chunks_to_disk(current_dfs: list, temp_dirs: list, chunk_counter: int):
    """Writes the current batch of DataFrames to their respective temp directories."""
    for i, df_out in enumerate(current_dfs):
        if i < len(temp_dirs):
            file_path = Path(temp_dirs[i].name) / f"part_{chunk_counter:05d}.parquet"
            df_out.to_parquet(
                file_path, engine="pyarrow", compression="snappy", index=False
            )


def _initialize_storage(current_dfs: list) -> tuple[list, list, list]:
    """Creates temp directories and captures schemas from the first chunk."""
    temp_dirs = []
    to_cleanup = []
    schemas = [df.columns for df in current_dfs]

    for _ in range(len(current_dfs)):
        t = tempfile.TemporaryDirectory()
        temp_dirs.append(t)
        to_cleanup.append(t)

    return temp_dirs, to_cleanup, schemas


def _parquet_generator(temp_dir_obj, schema) -> Generator[pd.DataFrame]:
    """Yields DataFrames from a temp directory, restoring schema."""
    try:
        files = sorted(Path(temp_dir_obj.name).glob("*.parquet"))
        for f in files:
            df = pd.read_parquet(f)
            if schema is not None:
                df.columns = schema
            yield df
    finally:
        temp_dir_obj.cleanup()


def process_disk_backed(
    reader: Iterable[pd.DataFrame],
    func: Callable,
    func_args: Sequence[Any] | None = None,
    func_kwargs: dict[str, Any] | None = None,
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
    output_non_df = []
    directories_to_cleanup = []

    try:
        accumulators_initialized = False
        chunk_counter = 0

        for df in reader:
            if makecopy:
                df = df.copy()

            outputs = func(df, *func_args, **func_kwargs)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            # Sort outputs
            current_dfs, new_meta = _sort_chunk_outputs(
                outputs, accumulators_initialized
            )
            if new_meta:
                output_non_df.extend(new_meta)

            # Initialize storage
            if not accumulators_initialized and current_dfs:
                temp_dirs, directories_to_cleanup, column_schemas = _initialize_storage(
                    current_dfs
                )
                accumulators_initialized = True

            # Write DataFrames
            if accumulators_initialized:
                _write_chunks_to_disk(current_dfs, temp_dirs, chunk_counter)

            chunk_counter += 1

        if not accumulators_initialized:
            return tuple(output_non_df)

        # Finalize Iterators
        final_iterators = [
            ParquetStreamReader(_parquet_generator(d, s))
            for d, s in zip(temp_dirs, column_schemas)
        ]

        # Transfer ownership to generators
        directories_to_cleanup.clear()

        return tuple(final_iterators + output_non_df)

    finally:
        for d in directories_to_cleanup:
            d.cleanup()
