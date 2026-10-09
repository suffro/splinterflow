"""The native storage backend (Phase 6A, decision 0012): the `FileBackedPageStore` contract on Weightsift's Rust core.

`NativePageStore` plans, reads and counts in the native extension `weightsift_native` (built from `native/` with PyO3
and maturin): the same plans as `plan_reads`, the same extents and read calls (positioned reads on a pool of native
threads, direct I/O), the same counters, with the GIL released while the core works. It adds an optional host-RAM
cache of rows under a strict byte budget (least recently used first; rows in use never evicted; a row loaded once
however many requests want it at the same time). The cache holds rows in blocks that evicted rows hand to the rows
admitted next, whatever their sizes (decision 0013); its pool of such blocks counts against the budget
(`cache_stats()["held_bytes"]`). `cache_block_bytes` defaults to the largest size dividing every segment's rows, at
least a megabyte (5.5 MiB for expert rows of 11 and 5.5 MiB).

Three ways to read:

  read_rows(segment, rows)    plan, read and gather on the native pool into pageable host memory (tests, audits,
                              hashing); the host cache is neither used nor filled
  stream(requests, slots)     one transfer job for several requests into a streamer's pinned staging slots, through
                              the host cache: hits are copied first, then the planned reads of the misses, piece by
                              piece; `PageStreamer` copies each piece to the device and releases its slot
  prefetch(requests)          rows soon asked for, loaded into the host cache in the background at the lowest
                              priority (a chunked experts call's later chunks); requests then hit them, or wait for
                              their load; every prefetched row is counted as used or wasted

The extension is optional: without it `NATIVE_AVAILABLE` is False, and the Python backend (`FileBackedPageStore`) is
the only one. Where bytes come from never changes what they are: every byte the native path delivers is the file's
(tested against the Python store and against the files).

`stats` has `IOStats`'s fields and meaning (requests, rows, logical bytes, physical bytes the reads returned, read
calls, extents, the 4 KiB blocks of the rows read from storage, io_ms, OS counters, ranges, per segment), plus the
host cache's counters (`as_dict()["host_cache"]`) and the native path's (`as_dict()["native"]`, prefetches included).
`io_ms` is the time the native readers had a read in flight. The OS's counters are read over windows of native
activity (a transfer, a `read_rows`, a prefetch, from its start to its close); overlapping windows merge, so each read
is counted once, background prefetch reads included.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from pathlib import Path

import torch

from awpmi.storage.fileio import DIRECT_ALIGNMENT, aligned_host_buffer, os_read_counters
from awpmi.storage.layout import AnySegment, ComposedSegment
from awpmi.storage.store import (
    DEFAULT_MAX_EXTENT_BYTES,
    DEFAULT_MAX_READ_BYTES,
    IOStats,
    PageStore,
    ReadPlan,
    check_rows,
)

try:  # the extension is optional
    import weightsift_native as _native
except ImportError:
    _native = None

NATIVE_AVAILABLE = _native is not None

_IO_FIELDS = ("requests", "rows", "logical_bytes", "physical_bytes", "read_calls", "extents", "blocks_4k")


class NativeIOStats:
    """The native engine's counters seen as an `IOStats`, plus the OS's own counters over windows of native activity.

    It holds the engine and the store's names, not the store: no reference cycle keeps a closed store (and whatever it
    references) alive until the garbage collector finds it.
    """

    def __init__(self, engine, file_keys: list[str], segment_names: list[str]) -> None:
        self._native = engine
        self._file_keys = file_keys
        self._segment_names = segment_names
        self.os_read_calls: int | None = 0
        self.os_read_bytes: int | None = 0
        self.record_ranges = False
        self._lock = threading.Lock()
        self._active = 0
        self._window: tuple[int, int] | None = None

    def begin(self) -> None:
        """Native activity starts (windows nest: the OS's counters are read when the first opens, the last closes)."""
        with self._lock:
            if self._active == 0:
                self._window = os_read_counters()
            self._active += 1

    def end(self) -> None:
        with self._lock:
            self._active -= 1
            if self._active == 0:
                self.count_os(self._window, os_read_counters())

    def _engine(self) -> dict:
        return self._native.stats()

    def __getattr__(self, name: str):
        if name in _IO_FIELDS:
            return self._engine()[name]
        raise AttributeError(name)

    @property
    def io_ms(self) -> float:
        return self._engine()["busy_ms"]

    @property
    def ranges(self) -> list[tuple[str, int, int]] | None:
        ranges = self._engine()["ranges"]
        if ranges is None:
            return None
        keys = self._file_keys
        return [(keys[file], offset, length) for file, offset, length in ranges]

    def reset(self, record_ranges: bool = False) -> None:
        with self._lock:
            self._native.reset_stats(record_ranges)
            self.os_read_calls = self.os_read_bytes = 0
            self.record_ranges = record_ranges
            if self._active:  # an open window restarts with the new counters
                self._window = os_read_counters()

    def count_os(self, before: tuple[int, int] | None, after: tuple[int, int] | None) -> None:
        if before is None or after is None or self.os_read_calls is None:
            self.os_read_calls = self.os_read_bytes = None
        else:
            self.os_read_calls += after[0] - before[0]
            self.os_read_bytes += after[1] - before[1]

    def as_dict(self) -> dict:
        engine = self._engine()
        names = self._segment_names
        data = {key: engine[key] for key in _IO_FIELDS}
        data["os_read_calls"] = self.os_read_calls
        data["os_read_bytes"] = self.os_read_bytes
        data["by_segment"] = {names[segment]: dict(entry) for segment, entry in engine["by_segment"].items()}
        if engine["ranges"] is not None:
            data["ranges"] = [list(r) for r in self.ranges]
        data["native"] = {
            key: engine[key]
            for key in (
                "busy_ms", "read_ms", "cache_copied_bytes", "gathered_bytes", "admitted_bytes", "fallback_rows", "prefetches",
                "prefetch_rows", "prefetch_bytes", "prefetch_blocks_4k", "submits", "submit_ms", "submit_cache_ms",
                "submit_plan_ms", "submit_start_ms",
            )
        }
        cache = self._native.cache_stats()
        if cache is not None:
            data["host_cache"] = cache
        return data


class NativePageStore(PageStore):
    """Segments in files, read by the native core (see the module docstring); `host_cache_bytes` > 0 adds the cache.

    Options are `FileBackedPageStore`'s, with the same checks and meaning.
    """

    native = True

    def __init__(
        self,
        files: Mapping[str, str | Path],
        segments: Mapping[str, AnySegment],
        direct: bool = True,
        alignment: int = DIRECT_ALIGNMENT,
        max_gap: int = 0,
        workers: int = 8,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
        max_extent_bytes: int = DEFAULT_MAX_EXTENT_BYTES,
        host_cache_bytes: int = 0,
        cache_block_bytes: int | None = None,
    ) -> None:
        if _native is None:
            raise RuntimeError("the native extension weightsift_native is not installed: `uv sync` builds it from native/")
        if direct and alignment % DIRECT_ALIGNMENT:
            raise ValueError(f"direct I/O needs a multiple of {DIRECT_ALIGNMENT}-byte alignment")
        if max_read_bytes % alignment or max_extent_bytes % alignment:
            raise ValueError("read and extent limits must be multiples of the alignment")
        missing = {file for segment in segments.values() for file in segment.files} - set(files)
        if missing:
            raise KeyError(f"segments refer to unknown files {sorted(missing)}")
        self.files = {key: Path(path) for key, path in files.items()}
        sizes = {key: path.stat().st_size for key, path in self.files.items()}
        for segment in segments.values():
            if isinstance(segment, ComposedSegment):
                if not segment.spans_within(sizes):
                    raise ValueError(f"{segment.name} has a span beyond the end of its file")
            elif segment.offset + segment.nbytes > sizes[segment.file]:
                raise ValueError(f"{segment.name} extends beyond {self.files[segment.file]}")
        self.segments = dict(segments)
        self._file_keys = list(self.files)
        file_ids = {key: index for index, key in enumerate(self._file_keys)}
        self._segment_names = list(self.segments)
        self._segment_ids = {name: index for index, name in enumerate(self._segment_names)}
        specs = []
        for name, segment in self.segments.items():
            if isinstance(segment, ComposedSegment):
                local = {key: index for index, key in enumerate(segment.files)}
                spans = [(local[file], offset) for row in segment.spans for file, offset in row]
                specs.append((name, [file_ids[key] for key in segment.files], segment.rows, segment.row_bytes, None, list(segment.part_bytes), spans))
            else:
                specs.append((name, [file_ids[segment.file]], segment.rows, segment.row_bytes, segment.offset, None, None))
        blocks = {} if cache_block_bytes is None else {"cache_block_bytes": cache_block_bytes}
        self._engine = _native.Engine(
            [(key, str(path)) for key, path in self.files.items()], specs, direct=direct, alignment=alignment,
            max_gap=max_gap, max_read_bytes=max_read_bytes, max_extent_bytes=max_extent_bytes, workers=workers,
            host_cache_bytes=host_cache_bytes, **blocks,
        )
        self.device = torch.device("cpu")
        self.direct = direct
        self.alignment = alignment
        self.max_gap = max_gap
        self.max_read_bytes = max_read_bytes
        self.max_extent_bytes = max_extent_bytes
        self.workers = workers
        self.host_cache_bytes = host_cache_bytes
        self.stats = NativeIOStats(self._engine, self._file_keys, self._segment_names)

    # Plans (for tests and audits: the same as the Python store's)

    def plan(self, segment: str, rows: torch.Tensor | None, positions: torch.Tensor | None = None) -> ReadPlan:
        info = self.segment(segment)
        rows = check_rows(rows, info)
        runs, extents, row_count, output_rows, _ = self._engine.plan(self._segment_ids[segment], _indices(rows), _indices(positions))
        as_tensor = lambda values, width: torch.tensor(values, dtype=torch.int64).reshape(-1, width)  # noqa: E731
        run_table = as_tensor([(o, n, out, e) for _, o, n, out, e in runs], 4)
        extent_table = as_tensor([(o, n) for _, o, n in extents], 2)
        return ReadPlan(
            info, rows, row_count, run_table, extent_table, self.alignment, info.files,
            torch.tensor([f for f, *_ in runs], dtype=torch.int64), torch.tensor([f for f, *_ in extents], dtype=torch.int64),
            output_rows,
        )

    def pieces(self, segment: str, rows: torch.Tensor | None, slot_bytes: int, positions: torch.Tensor | None = None) -> list:
        """The plan's staging pieces for slots of `slot_bytes`: [(extents (file, offset, length), parts (staging, length, output))]."""
        rows = check_rows(rows, self.segment(segment))
        return self._engine.pieces(self._segment_ids[segment], slot_bytes, _indices(rows), _indices(positions))

    # Reads

    def read_rows(self, segment: str, rows: torch.Tensor | None = None) -> torch.Tensor:
        info = self.segment(segment)
        rows = check_rows(rows, info)
        count = info.rows if rows is None else rows.numel()
        out = aligned_host_buffer(count * info.row_bytes, pin=False)
        self.stats.begin()
        try:
            self._engine.read_rows(self._segment_ids[segment], out.numpy(), _indices(rows), None)
        finally:
            self.stats.end()
        return out.view(count, info.row_bytes)

    def stream(self, requests: list[tuple[str, torch.Tensor | None, torch.Tensor | None]], slots: list[tuple[torch.Tensor, torch.Tensor | None]]) -> NativeTransfer:
        """A transfer of `requests` ((segment, rows, positions)) into `slots` ((staging, gather buffer or None)) through the
        host cache: iterate it for (slot index, copies, gathered bytes), release each slot once its copies finished."""
        converted = []
        for segment, rows, positions in requests:
            info = self.segment(segment)
            converted.append((self._segment_ids[segment], _indices(check_rows(rows, info)), _indices(positions)))
        buffers = [(staging.numpy(), None if compact is None else compact.numpy()) for staging, compact in slots]
        return NativeTransfer(self, converted, buffers)

    def prefetch(self, requests: list[tuple[str, torch.Tensor | None]]) -> NativePrefetch | None:
        """Start loading the rows of `requests` ((segment, rows)) into the host cache in the background: rows soon asked
        for. Returns a handle to close once they were used (None without a host cache: nothing to load into)."""
        if not self.host_cache_bytes:
            return None
        converted = [(self._segment_ids[segment], _indices(check_rows(rows, self.segment(segment)))) for segment, rows in requests]
        return NativePrefetch(self, converted)

    # The host cache

    def cache_stats(self) -> dict | None:
        """The host cache's counters, capacity, resident and peak bytes, entries (None without a cache)."""
        return self._engine.cache_stats()

    def set_admit(self, admit: bool) -> None:
        """Freeze (False) or resume (True) admission into the host cache: hits are still served."""
        self._engine.set_admit(admit)

    def clear_cache(self) -> None:
        self._engine.clear_cache()

    def cached_rows(self) -> list[tuple[str, int]]:
        """The rows the host cache holds, least recently used first."""
        return [(self._segment_names[segment], row) for segment, row in self._engine.cached_rows()]

    @property
    def resident_bytes(self) -> int:
        cache = self.cache_stats()
        return 0 if cache is None else cache["resident_bytes"]

    def close(self) -> None:
        """Stop the readers and release the host cache's memory (a closed store serves nothing again)."""
        self._engine.close()


class NativeTransfer:
    """A native transfer job: iterate for ready pieces, `release` slots, `close` (or leave the `with` block) at the end.

    Its window of native activity (for the OS's read counters) runs from its start to its close.
    """

    def __init__(self, store: NativePageStore, requests: list, buffers: list) -> None:
        self._store = store
        self._job = None
        store.stats.begin()
        try:
            self._job = store._engine.submit(requests, buffers, True)
        except BaseException:
            store.stats.end()
            raise
        self.pieces = self._job.pieces

    def __iter__(self) -> Iterator[tuple[int, list[tuple[int, int, int, int, int]], int]]:
        while (piece := self._job.next()) is not None:
            yield piece

    def release(self, slot: int) -> None:
        self._job.release(slot)

    def cancel(self) -> None:
        self._job.cancel()

    def close(self) -> None:
        if self._job is not None:
            self._job.close()
            self._job = None
            self._store.stats.end()

    def __enter__(self) -> NativeTransfer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NativePrefetch:
    """Rows being loaded into the host cache (`NativePageStore.prefetch`); `close` once they were used.

    Closing cancels whatever is still unread (a request waiting for such a row reads it itself) and waits until the
    prefetch's reads are over; its window of native activity runs from its start to its close.
    """

    def __init__(self, store: NativePageStore, requests: list) -> None:
        self._store = store
        self._prefetch = None
        store.stats.begin()
        try:
            self._prefetch = store._engine.prefetch(requests)
        except BaseException:
            store.stats.end()
            raise
        self.rows = self._prefetch.rows

    def wait(self) -> None:
        """Until every row is in the cache."""
        if self._prefetch is not None:
            self._prefetch.wait()

    def cancel(self) -> None:
        if self._prefetch is not None:
            self._prefetch.cancel()

    def close(self) -> None:
        if self._prefetch is not None:
            self._prefetch.close()
            self._prefetch = None
            self._store.stats.end()

    def __enter__(self) -> NativePrefetch:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _indices(values: torch.Tensor | None):
    """int64 indices as a contiguous NumPy array the extension reads through the buffer protocol (None stays None)."""
    if values is None:
        return None
    return values.reshape(-1).to("cpu", torch.int64).contiguous().numpy()
