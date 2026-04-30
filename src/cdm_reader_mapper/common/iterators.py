"""Utilities for handling pandas TextParser objects safely."""

from __future__ import annotations
import inspect
import itertools
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import (
    Any,
    Literal,
)

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr


class ProcessFunction:
    r"""
    Stores data and a callable function with optional arguments for processing.

    Parameters
    ----------
    data : pd.DataFrame, pd.Series, Iterable of pd.DataFrame or Iterable of pd.Series
        Input data to be processed.
    func : Callable
        A callable that will be applied to `data`.
    func_args : Any, list of Any or tuple of Any, optional
        Positional arguments to pass to `func`.
    func_kwargs : dict, optional
        Keyword arguments to pass to `func`.
    \**kwargs : Any
        Additional metadata or configuration parameters stored with the instance.
    """

    def __init__(
        self,
        data: pd.DataFrame | pd.Series | Iterable[pd.DataFrame] | Iterable[pd.Series],
        func: Callable[..., Any],
        func_args: Any | list[Any] | tuple[Any] | None = None,
        func_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        r"""
        Initialize a ProcessFunction instance.

        Parameters
        ----------
        data : pd.DataFrame, pd.Series, Iterable of pd.DataFrame or Iterable of pd.Series
          Input data to be processed.
        func : Callable
          A callable that will be applied to `data`.
        func_args : Any, list of Any or tuple of Any, optional
          Positional arguments to pass to `func`.
        func_kwargs : dict, optional
          Keyword arguments to pass to `func`.
        \**kwargs : Any
          Additional metadata or configuration parameters stored with the instance.

        Raises
        ------
        ValueError
            If `func` is not callable.
        """
        self.data = data

        if not callable(func):
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
    """
    A wrapper that mimics pandas.io.parsers.TextFileReader.

    Parameters
    ----------
    source : iterable or iterator or callable
            Data source yielding ``pandas.DataFrame`` or ``pandas.Series`` objects.
            If a callable is provided, it must return a fresh iterator each time
            it is called (useful for copying/resetting the stream).

    Attributes
    ----------
    columns : list or pandas.Index
        Column labels inferred from the first chunk.
    dtypes : dict or pandas.Series
        Data types inferred from the first chunk.
    attrs : dict
        User-defined metadata associated with the stream.

    Notes
    -----
    - The stream is consumed as it is iterated.
    - Use `copy()` to create an independent iterator.
    - Some operations (e.g., `read()`) load all data into memory and may not
      be suitable for large datasets.
    """

    def __init__(
        self,
        source: (
            list[pd.DataFrame | pd.Series]
            | tuple[pd.DataFrame | pd.Series]
            | Iterator[pd.DataFrame | pd.Series]
            | Callable[[], Iterator[pd.DataFrame | pd.Series]]
        ),
    ):
        """
        Initialize a ParquetStreamReader instance.

        Parameters
        ----------
        source : iterable or iterator or callable
            Data source yielding ``pandas.DataFrame`` or ``pandas.Series`` objects.
            If a callable is provided, it must return a fresh iterator each time
            it is called (useful for copying/resetting the stream).

        Raises
        ------
        TypeError
            If ``source`` is not an iterator, iterable, or callable returning an iterator.
        """
        self._closed = False
        self._buffer: list[pd.DataFrame | pd.Series] = []

        if isinstance(source, (tuple, list)):
            source = iter(source)

        if callable(source):
            # factory that produces a fresh iterator
            self._factory = source
        elif isinstance(source, Iterator):
            self._factory = lambda: source
        else:
            raise TypeError("ParquetStreamReader expects an iterator or a factory callable.")

        self._generator = self._factory()

        try:
            first = next(self._generator)
            if isinstance(first, pd.DataFrame):
                self.columns = first.columns
                self.dtypes = first.dtypes
            elif isinstance(first, pd.Series):
                self.columns = [first.name]
                self.dtypes = {first.name: first.dtype}
            self.prepend(first)
        except StopIteration:
            self.columns = []
            self.dtypes = {}

        self.attrs: dict[str, Any] = {}

    def __iter__(self) -> Iterator[pd.DataFrame | pd.Series]:
        """
        Return the iterator interface.

        Returns
        -------
        Iterator[pd.DataFrame | pd.Series]
            The stream itself.
        """
        return self

    def __next__(self) -> pd.DataFrame | pd.Series:
        """
        Return the next chunk from the stream.

        Returns
        -------
        pandas.DataFrame or pandas.Series
            The next chunk of data.

        Raises
        ------
        StopIteration
            If no more data is available.
        ValueError
            If the stream has been closed.
        """
        if self._closed:
            raise ValueError("I/O operation on closed stream.")
        if self._buffer:
            return self._buffer.pop(0)
        return next(self._generator)

    def __getitem__(self, item: str) -> Any:
        """
        Retrieve a value from the stream's metadata.

        Parameters
        ----------
        item : str
            Key in the `attrs` dictionary.

        Returns
        -------
        Any
            The stored metadata value.
        """
        return self.attrs[item]

    def __setitem__(self, item: str, value: Any) -> None:
        """
        Set a metadata attribute on the stream.

        Parameters
        ----------
        item : str
            Attribute name.
        value : Any
            Value to assign.
        """
        setattr(self, item, value)

    def prepend(self, chunk: pd.DataFrame | pd.Series) -> None:
        """
        Push a chunk back onto the front of the stream.

        Useful for peeking at the first chunk without losing it.

        Parameters
        ----------
        chunk : pandas.DataFrame or pandas.Series
            The chunk to prepend.
        """
        # Insert at 0 ensures FIFO order (peeking logic)
        self._buffer.insert(0, chunk)

    def get_chunk(self) -> pd.DataFrame | pd.Series:
        """
        Return the next available chunk.

        This is equivalent to calling ``next(reader)`` and is provided
        for API compatibility with pandas readers.

        Returns
        -------
        pandas.DataFrame or pandas.Series
            The next chunk of data.
        """
        return next(self)

    def read(self) -> pd.DataFrame:
        """
        Read all remaining data into a single DataFrame.

        This consumes the entire stream and concatenates all remaining
        chunks into one DataFrame.

        Returns
        -------
        pandas.DataFrame
            Concatenated result of all remaining chunks. Returns an empty
            DataFrame if the stream is exhausted.

        Warnings
        --------
        This operation loads all data into memory and may not be suitable
        for large datasets.
        """
        # Consume the entire rest of the stream
        chunks = list(self)

        if not chunks:
            return pd.DataFrame()

        return pd.concat(chunks)

    def copy(self) -> ParquetStreamReader:
        """
        Create an independent copy of the stream.

        Returns
        -------
        ParquetStreamReader
            A new stream reader with independent iteration state.

        Raises
        ------
        ValueError
            If the stream has been closed.
        """
        if self._closed:
            raise ValueError("Cannot copy a closed stream.")
        self._generator, new_gen = itertools.tee(self._generator)

        copy_stream = self.__class__.__new__(self.__class__)

        # Manually copy all internal state
        copy_stream._closed = self._closed
        copy_stream._buffer = self._buffer.copy()
        copy_stream._generator = new_gen
        copy_stream._factory = self._factory
        copy_stream.columns = self.columns.copy()
        copy_stream.dtypes = self.dtypes.copy()
        copy_stream.attrs = self.attrs.copy()

        return copy_stream

    @property
    def empty(self) -> bool:
        """
        Check whether the stream has any remaining data.

        Returns
        -------
        bool
            True if the stream is exhausted, False otherwise.

        Notes
        -----
        This method creates a temporary copy of the stream to check
        for remaining elements without consuming the original.
        """
        copy_stream = self.copy()

        try:
            next(copy_stream)
            return False
        except StopIteration:
            return True

    def reset_index(self, drop: bool = False) -> ParquetStreamReader:
        """
        Reset the index across all chunks to a continuous range.

        Parameters
        ----------
        drop : bool, default False
            If True, do not insert the old index as a column.
            If False, the new index is also inserted as a column named "index".

        Returns
        -------
        ParquetStreamReader
            A new stream reader with reindexed chunks.

        Raises
        ------
        ValueError
            If the stream has been closed.
        """
        if self._closed:
            raise ValueError("Cannot copy a closed stream.")

        offset = 0
        chunks: list[pd.DataFrame] = []

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

    def close(self) -> None:
        """Close the stream and release resources."""
        self._closed = True

    def __enter__(self) -> ParquetStreamReader:
        """
        Enter the runtime context for use in a ``with`` statement.

        Returns
        -------
        ParquetStreamReader
            The stream instance.
        """
        return self

    def __exit__(self, _exc_type: type | None, _exc_val: BaseException | None, _exc_tb: TracebackType | None) -> None:
        """
        Exit the runtime context and close the stream.

        Parameters
        ----------
        _exc_type : type or None
          The type of the exception raised within the context, if any.
          `None` if no exception occurred.
        _exc_val : BaseException or None
          The exception instance raised within the context, if any.
          `None` if no exception occurred.
        _exc_tb : TracebackType or None
          The traceback associated with the exception, if any.
          `None` if no exception occurred.
        """
        self.close()


def _sort_chunk_outputs(
    outputs: tuple[Any, ...], capture_meta: bool, requested_types: tuple[type, ...]
) -> tuple[list[pd.DataFrame | pd.Series], list[Any]]:
    """
    Separate DataFrames from metadata in the function output.

    Parameters
    ----------
    outputs : tuple of Any
        Tuple of objects returned by a processing function.
    capture_meta : bool
        If True, non-data outputs are collected as metadata. If False, they are ignored.
    requested_types : tuple of tuple
        Types that should be considered as valid data objects
        (e.g., `pd.DataFrame`, `pd.Series`).

    Returns
    -------
    tuple of list of pd.DataFrame or pd.Series and list of Any
        A tuple containing:
        - A list of extracted data objects (flattened if nested in lists)
        - A list of metadata objects (empty if `capture_meta` is False)
    """
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
) -> tuple[list[TemporaryDirectory[str]], list[tuple[type, Any]]]:
    """
    Create temp directories and captures schemas from the first chunk.

    Parameters
    ----------
    first_batch : list of pandas.DataFrame or pandas.Series
        The first batch of data objects used to initialize storage. Each element
        determines the schema and corresponding temporary storage.

    Returns
    -------
    tuple of list of TemporaryDirectory of str and list of tuple of type and Any
        A tuple containing:

        - A list of temporary directories, one for each object in `first_batch`.
        - A list of schema descriptors, where each entry is a tuple of:
          (object type, schema information). For DataFrames, the schema is the
          columns; for Series, it is the name.

    Raises
    ------
    TypeError
        If an element in `first_batch` is not a pandas DataFrame or Series.
    """
    temp_dirs: list[TemporaryDirectory[str]] = []
    schemas: list[tuple[type, Any]] = []

    for obj in first_batch:
        if isinstance(obj, pd.DataFrame):
            schemas.append((pd.DataFrame, obj.columns))
        elif isinstance(obj, pd.Series):
            schemas.append((pd.Series, obj.name))
        else:
            raise TypeError(f"Unsupported data type: {type(obj)}.Use one of [pd.DataFrame, pd.Series].")

        temp_dirs.append(TemporaryDirectory())

    return temp_dirs, schemas


def _write_chunks_to_disk(
    batch: list[pd.DataFrame | pd.Series],
    temp_dirs: list[TemporaryDirectory[str]],
    chunk_counter: int,
) -> None:
    """
    Write the current batch of DataFrames to their respective temp directories.

    Parameters
    ----------
    batch : list of pandas.DataFrame or pandas.Series
        A batch of data objects to be written to disk. Series objects are
        converted to single-column DataFrames before writing.
    temp_dirs : list of TemporaryDirectory of str
        Temporary directories corresponding to each element in `batch`.
        Each batch item is written into its matching directory.
    chunk_counter : int
        Sequential counter used to generate unique filenames for each chunk
        written to disk.
    """
    for i, data_out in enumerate(batch):
        if isinstance(data_out, pd.Series):
            data_out = data_out.to_frame()

        file_path = Path(temp_dirs[i].name) / f"part_{chunk_counter:05d}.parquet"

        table = pa.Table.from_pandas(data_out, preserve_index=True)
        pq.write_table(table, file_path, compression="snappy")


def _parquet_generator(temp_dir: TemporaryDirectory[str], data_type: type, schema: str | None) -> Generator[pd.DataFrame | pd.Series]:
    """
    Yield DataFrames from a temp directory, restoring schema.

    Parameters
    ----------
    temp_dir : TemporaryDirectory of str
        Temporary directory containing Parquet chunk files to be read.
        The directory is cleaned up after iteration completes (even if an
        exception occurs).
    data_type : type
        Expected output type for each chunk. If ``pd.Series``, each Parquet
        file is converted to a Series; otherwise, a DataFrame is returned.
    schema : str or None
        Metadata used to restore Series structure. When ``data_type`` is
        ``pd.Series``, this is used as the Series name.

    Returns
    -------
    Generator of pd.DataFrame or pd.Series
        A generator yielding reconstructed DataFrames or Series objects
        from the Parquet files stored in `temp_dir`.
    """
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
    requested_types: tuple[type, ...],
    static_args: list[Any],
    static_kwargs: dict[str, Any],
    non_data_output: str,
    non_data_proc: Callable[..., Any] | None,
    non_data_proc_args: tuple[Any] | None,
    non_data_proc_kwargs: dict[str, Any] | None,
) -> (
    tuple[ParquetStreamReader, ...]  # when data is produced
    | tuple[Any, ...]  # non-data outputs that are a tuple
    | Any  # single value or list (no data produced)
):
    """
    Process chunks.

    Parameters
    ----------
    readers : list of ParquetStreamReader
        Input stream readers providing chunked data to be processed in lockstep.
    func : Callable
        Function applied to each synchronized set of chunks from ``readers``.
    requested_types : tuple of type
        Types considered valid data outputs (e.g., pd.DataFrame, pd.Series).
        Used to separate data from metadata returned by `func`.
    static_args : list of Any
        Additional positional arguments passed unchanged to `func`.
    static_kwargs : dict
        Additional keyword arguments passed unchanged to `func`.
    non_data_output : str
        Controls how non-data outputs (metadata) are aggregated across chunks.
        Typically determines whether they are accumulated or processed differently.
    non_data_proc : Callable or None
        Optional function applied to aggregated non-data outputs after processing.
    non_data_proc_args : tuple of Any or None
        Positional arguments for `non_data_proc`.
    non_data_proc_kwargs : dict or None
        Keyword arguments for `non_data_proc`.

    Returns
    -------
    tuple of ParquetStreamReader or tuple of Any or Any
        If data outputs are produced:
            A tuple containing ParquetStreamReader objects (one per data stream)
            followed by processed non-data outputs.
        If no data outputs are produced:
            The aggregated non-data output, which may be a single value,
            list, or tuple depending on processing configuration.

    Raises
    ------
    TypeError
        If input chunks are not of the expected `requested_types`.
    ValueError
        If the input iterables are empty or schema initialization fails.
    """
    # State variables
    temp_dirs: list[TemporaryDirectory[str]] | None = None
    schemas: list[tuple[type, Any]] | None = None
    output_non_data_dict: dict[int, list[Any]] = {}
    chunk_counter: int = 0

    for items in zip(*readers, strict=True):
        if not isinstance(items[0], requested_types):
            raise TypeError(f"Unsupported data type in Iterable {items[0]}: {type(items[0])}Requested types are: {requested_types} ")

        result = func(*items, *static_args, **static_kwargs)

        if not isinstance(result, tuple):
            result = (result,)

        # Sort outputs
        capture_meta = non_data_output == "acc" or chunk_counter == 0

        data, meta = _sort_chunk_outputs(result, capture_meta, requested_types)

        for i, m in enumerate(meta):
            output_non_data_dict.setdefault(i, []).append(m)

        # Write DataFrames
        if data:
            if temp_dirs is None:
                temp_dirs, schemas = _initialize_storage(data)

            _write_chunks_to_disk(data, temp_dirs, chunk_counter)

        chunk_counter += 1

    if chunk_counter == 0:
        raise ValueError("Iterable is empty.")

    if len(output_non_data_dict) == 1:
        first_key = list(output_non_data_dict.keys())[0]
        output_non_data: Any = output_non_data_dict[first_key]
    else:
        output_non_data = output_non_data_dict

    if callable(non_data_proc):
        output_non_data = non_data_proc(
            output_non_data,
            *(non_data_proc_args or ()),
            **(non_data_proc_kwargs or {}),
        )

    if isinstance(output_non_data, list) and len(output_non_data) == 1:
        output_non_data = output_non_data[0]

    # If no data outputs at all
    if temp_dirs is None:
        return output_non_data

    if schemas is None:
        raise ValueError("Could not set schemas.")

    final_iterators: list[ParquetStreamReader] = [
        ParquetStreamReader(
            (
                lambda d=d, t=t, s=s: _parquet_generator(d, t, s)  # type: ignore[misc]
            )
        )
        for d, (t, s) in zip(temp_dirs, schemas, strict=True)
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
    """
    Prepare readers for chunking.

    Parameters
    ----------
    reader : Iterator of pandas.DataFrame or pandas.Series
        Primary input iterator to be converted into a ParquetStreamReader.
    func_args : Sequence of Any
        Positional arguments for the processing function. Any elements that
        can be converted into ParquetStreamReader instances are treated as
        additional data streams; others are passed through as static values.
    func_kwargs : dict
        Keyword arguments for the processing function. Values that can be
        converted into ParquetStreamReader instances are treated as streams;
        others are treated as static keyword arguments.
    makecopy : bool
        If True, all constructed ParquetStreamReader objects are copied so
        that iteration is independent across consumers.

    Returns
    -------
    tuple of list of ParquetStreamReader or list of Any or dict
        A tuple containing:
        - A list of ParquetStreamReader objects used for chunked iteration
          (including the primary reader and any detected in args/kwargs).
        - A list of non-stream positional arguments to be passed to the
          processing function.
        - A dictionary of non-stream keyword arguments to be passed to the
          processing function.

    Raises
    ------
    TypeError
        If the primary reader cannot be converted into a ParquetStreamReader.
    """
    reader = ensure_parquet_reader(reader)
    if not isinstance(reader, ParquetStreamReader):
        raise TypeError(f"reader is not a ParquetStreamReader: {type(reader)}")

    args_reader: list[ParquetStreamReader] = []
    args: list[Any] = []
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
    Stream an iterable of DataFrame/Series to parquet and return a disk-backed ParquetStreamReader.

    Memory usage remains constant.

    Parameters
    ----------
    iterable : Iterable pf pd.DataFrame or pd.Series
        An iterable of pandas DataFrame or Series objects to be streamed to disk.

    Returns
    -------
    ParquetStreamReader
        A disk-backed stream reader that lazily reads the provided iterable
        from Parquet files stored in a temporary directory.

    Raises
    ------
    ValueError
        If the input iterable is empty.
    TypeError
        If elements in the iterable are not pandas DataFrame or Series objects,
        or if mixed types are provided across chunks.
    """
    iterator = iter(iterable)

    try:
        first = next(iterator)
    except StopIteration as err:
        raise ValueError("Iterable is empty.") from err

    if not isinstance(first, (pd.DataFrame, pd.Series)):
        raise TypeError("Iterable must contain pd.DataFrame or pd.Series objects.")

    temp_dir = TemporaryDirectory()
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
    """
    Check if reader is a valid Iterable.

    Parameters
    ----------
    reader : Any
        Object to be checked for iterator compatibility.

    Returns
    -------
    bool
        True if `reader` is an instance of `Iterator`, otherwise False.
    """
    return isinstance(reader, Iterator)


def ensure_parquet_reader(obj: Any) -> Any:
    """
    Ensure obj is a ParquetStreamReader.

    Parameters
    ----------
    obj : Any
        Object that may represent a ParquetStreamReader, an iterator of
        pd.DataFrame or pd.Series objects, or a static value.

    Returns
    -------
    Any
        If `obj` is already a ParquetStreamReader, it is returned unchanged.
        If `obj` is an iterator, it is converted into a ParquetStreamReader.
        Otherwise, `obj` is returned as-is (treated as a static value).
    """
    if isinstance(obj, ParquetStreamReader):
        return obj

    if is_valid_iterator(obj):
        return parquet_stream_from_iterable(obj)

    return obj


def process_disk_backed(
    reader: Iterator[pd.DataFrame | pd.Series],
    func: Callable[..., Any],
    func_args: tuple[Any, ...] | None = None,
    func_kwargs: dict[str, Any] | None = None,
    requested_types: type | tuple[type, ...] = (pd.DataFrame, pd.Series),
    non_data_output: Literal["first", "acc"] = "first",
    non_data_proc: Callable[..., Any] | None = None,
    non_data_proc_args: tuple[Any, ...] | None = None,
    non_data_proc_kwargs: dict[str, Any] | None = None,
    makecopy: bool = True,
) -> tuple[Any, ...]:
    """
    Consume a stream of DataFrames, processes them, and returns a tuple of results.

    DataFrames are cached to disk (Parquet) and returned as generators.

    Parameters
    ----------
    reader : Iterator of pd.DataFrame or pd.Series
        Input stream of DataFrame or Series objects to be processed in chunks.
    func : Callable
        Function applied to each synchronized set of chunks from the stream.
        May return data objects (pd.DataFrame or pd.Series) and/or metadata.
    func_args : tuple of Any, optional
        Additional positional arguments passed to `func`. Defaults to empty tuple.
    func_kwargs : dict, optional
        Additional keyword arguments passed to `func`. Defaults to empty dict.
    requested_types : type or tuple of type, default (pd.DataFrame, pd.Series)
        Types treated as data outputs from `func`. All other outputs are
        treated as metadata.
    non_data_output : {"first", "acc"}, default "first"
        Strategy for handling non-data outputs:
        - "first": only the first chunk's metadata is kept
        - "acc": accumulate metadata across all chunks
    non_data_proc : Callable, optional
        Optional function applied to aggregated non-data outputs after processing.
    non_data_proc_args : tuple of Any, optional
        Positional arguments for `non_data_proc`.
    non_data_proc_kwargs : dict, optional
        Keyword arguments for `non_data_proc`.
    makecopy : bool, default True
        If True, ensures independent copies of input streams are used internally.

    Returns
    -------
    tuple of Any
        A tuple containing:
        - One or more ParquetStreamReader objects for chunked data outputs
          (if any data was produced)
        - Processed non-data outputs (metadata), optionally transformed by
          `non_data_proc`

    Raises
    ------
    ValueError
        If `non_data_proc` is provided but not callable.
    """
    if func_args is None:
        func_args = ()
    if func_kwargs is None:
        func_kwargs = {}

    if not isinstance(requested_types, (list, tuple)):
        requested_types = (requested_types,)

    readers, static_args, static_kwargs = _prepare_readers(reader, func_args, func_kwargs, makecopy)

    if non_data_proc is not None:
        if not callable(non_data_proc):
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


def _process_function(results: Any, data_only: bool = False) -> Any:
    """
    Execute a ProcessFunction or return the input unchanged.

    Parameters
    ----------
    results : Any
        Input object to be processed. If it is a `ProcessFunction` instance,
        it will be executed using its stored data, function, and arguments.
        Otherwise, it is returned unchanged.
    data_only : bool, default False
        If True, only the first element of the processed result (typically the
        data stream) is returned. If False, the full tuple of results is returned.

    Returns
    -------
    Any
        - If `results` is not a ProcessFunction, it is returned unchanged.
        - If it is a ProcessFunction, returns either:
            - A tuple containing processed data streams and metadata, or
            - A single data stream if `data_only=True`.
    """
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

    args_for_call: tuple[Any, ...] | None = tuple(args)

    result = process_disk_backed(
        data,
        func,
        func_args=args_for_call,
        func_kwargs=kwargs,
        **results.kwargs,
    )

    if data_only is True:
        result = result[0]

    return result


def process_function(
    data_only: bool = False,
    postprocessing: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """
    Decorator to apply function to both pd.DataFrame and Iterable[pd.DataFrame].

    Parameters
    ----------
    data_only : bool, default False
      If True, only the primary data output is returned from the processed result.
    postprocessing : dict, optional
      Optional configuration for a postprocessing step applied to each result.
      Expected keys:
      - "func": callable applied to each DataFrame/Series/stream output
      - "kwargs": list or dict of argument names taken from the original call

    Returns
    -------
    Callable
      A decorator that wraps a function so it can operate on both in-memory
      pandas objects and disk-backed ParquetStreamReader streams.

    Raises
    ------
    ValueError
      If a provided postprocessing function is not callable.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator that enables a function to operate on pandas objects and streamed Parquet data.

        Parameters
        ----------
        func : Callable
          Function to be wrapped so it can operate on both pandas objects
          and disk-backed ParquetStreamReader streams.

        Returns
        -------
        Callable
          Wrapped function that supports streaming and optional postprocessing.
        """
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            r"""
            Wrapper that executes the decorated function and handles streaming and optional postprocessing.

            Parameters
            ----------
            \*args : Any
              Positional arguments passed to the decorated function.
            \**kwargs : Any
              Keyword arguments passed to the decorated function.

            Returns
            -------
            Any
              Result of executing the decorated function, optionally processed
              through disk-backed streaming and postprocessing logic.
            """
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
            if not callable(postproc_func):
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
