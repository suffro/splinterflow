"""MaterializationBackend: page cache → page store → streamer, behind one call.

`materialize(segment, rows)` returns the requested rows (ascending) on the compute device as
raw bytes [n, row_bytes]:

  * a store that keeps its segments in memory on the compute device answers directly (the
    resident runtimes of Phases 1C and 2);
  * otherwise cached pages are taken from the `PageCache`, and the rest are fetched by the
    `PageStreamer` from the store and offered to the cache.

A whole-segment request is one cache entry; a request for some rows looks rows up one by one,
and also serves them from a cached whole segment. Every request is counted: rows and bytes
requested, served from the cache, fetched from storage; the store's and the streamer's own
counters give the physical bytes, reads and copies behind them.

`materialize(..., out=buffer)` writes the rows into a caller's device buffer instead (decision
0007: an expert layer's compact buffers). Cached rows are copied first, on the compute
stream, so those copies run while storage reads the rest ("hits first"); the rest are
streamed straight into their rows of the buffer, and a copy of each is offered to the cache.

On CUDA, cache entries are allocated from a memory pool of their own. Long-lived pages
scattered among short-lived buffers of varying sizes (an expert layer's compact buffers)
fragment the caching allocator: under a device budget, the first Phase 4A runs failed with
1.6 GiB reserved but unusable. In their own pool, pages reuse each other's memory and stay
within the cache's budget, and the rest of device memory remains contiguous for the working
buffers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from awpmi.storage.cache import PageCache
from awpmi.storage.layout import AnySegment
from awpmi.storage.store import PageStore
from awpmi.streaming.streamer import PageStreamer, Ticket, resolve_device

WHOLE_SEGMENT = -1


@dataclass
class MaterializationStats:
    requests: int = 0
    rows: int = 0
    requested_bytes: int = 0
    cache_hit_rows: int = 0
    cache_hit_bytes: int = 0
    fetched_rows: int = 0  # served by the store (not the cache)
    fetched_bytes: int = 0
    device_copy_bytes: int = 0  # bytes copied on the device to assemble a request from cached pages
    largest_request_bytes: int = 0  # the largest single request (rows × row bytes)

    def reset(self) -> None:
        self.__init__()

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class MaterializationBackend:
    def __init__(
        self,
        store: PageStore,
        device: torch.device | str,
        streamer: PageStreamer | None = None,
        cache: PageCache | None = None,
    ) -> None:
        self.store = store
        self.device = resolve_device(device)
        self.resident = store.in_memory and resolve_device(store.device) == self.device
        if not self.resident and streamer is None:
            raise ValueError("a store away from the compute device needs a streamer")
        if streamer is not None and streamer.device != self.device:
            raise ValueError("the streamer must deliver to the compute device")
        self.streamer = streamer
        self.cache = None if self.resident else cache
        self._cache_pool = torch.cuda.MemPool() if self.cache is not None and self.device.type == "cuda" else None
        self.stats = MaterializationStats()

    def _cache_copy(self, tensor: torch.Tensor) -> torch.Tensor:
        """A copy of `tensor` for the cache, allocated from the cache's own memory pool."""
        if self._cache_pool is None:
            return tensor.clone()
        with torch.cuda.use_mem_pool(self._cache_pool):
            return tensor.clone()

    def segment(self, name: str) -> AnySegment:
        return self.store.segment(name)

    # Requests

    def materialize(self, segment: str, rows: torch.Tensor | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
        """Rows of `segment` on the device ([n, row_bytes] uint8, ascending rows). Read-only.

        With `out` (contiguous uint8 [n, row_bytes] on the device), the rows are written into it
        and `out` is returned.
        """
        info = self.store.segment(segment)
        count = info.rows if rows is None else rows.numel()
        if out is not None and (out.dtype != torch.uint8 or tuple(out.shape) != (count, info.row_bytes) or not out.is_contiguous()):
            raise ValueError(f"out must be contiguous uint8 [{count}, {info.row_bytes}]")
        self.stats.requests += 1
        self.stats.rows += count
        self.stats.requested_bytes += count * info.row_bytes
        self.stats.largest_request_bytes = max(self.stats.largest_request_bytes, count * info.row_bytes)
        if self.resident:
            self.stats.fetched_rows += count
            self.stats.fetched_bytes += count * info.row_bytes
            data = self.store.read_rows(segment, rows)
            return data if out is None else out.copy_(data)
        if self.cache is None:
            return self._fetch(info, rows, out)
        whole = self.cache.peek((segment, WHOLE_SEGMENT))
        if rows is None:
            entry = self.cache.get((segment, WHOLE_SEGMENT), info.nbytes)
            if entry is not None:
                self._hit(info, info.rows)
                if out is None:
                    return entry
                self.stats.device_copy_bytes += info.nbytes
                return out.copy_(entry)
            data = self._fetch(info, None, out)
            if self.cache.can_admit(info.nbytes):
                self.cache.put((segment, WHOLE_SEGMENT), self._cache_copy(data))
            else:
                self.cache.bypass(1, info.nbytes)
            return data
        if whole is not None:
            self.cache.get((segment, WHOLE_SEGMENT), count * info.row_bytes)
            self._hit(info, count)
            index = rows.to(self.device)
            data = whole.index_select(0, index) if out is None else torch.index_select(whole, 0, index, out=out)
            self.stats.device_copy_bytes += count * info.row_bytes
            return data
        return self._rows_through_cache(info, rows, out)

    def materialize_many(self, requests: list[tuple[str, torch.Tensor | None, torch.Tensor | None]]) -> list[torch.Tensor]:
        """`materialize` of several requests, each (segment, rows, out), counted as that many requests.

        Without a device cache they go to the streamer together (`fetch_many`: one transfer for a native store, one
        fetch after the other otherwise); with one, request by request.
        """
        if self.resident or self.cache is not None or len(requests) < 2:
            return [self.materialize(segment, rows, out) for segment, rows, out in requests]
        fetches = []
        for segment, rows, out in requests:
            info = self.store.segment(segment)
            count = info.rows if rows is None else rows.numel()
            if out is not None and (out.dtype != torch.uint8 or tuple(out.shape) != (count, info.row_bytes) or not out.is_contiguous()):
                raise ValueError(f"out must be contiguous uint8 [{count}, {info.row_bytes}]")
            self.stats.requests += 1
            self.stats.rows += count
            self.stats.requested_bytes += count * info.row_bytes
            self.stats.largest_request_bytes = max(self.stats.largest_request_bytes, count * info.row_bytes)
            self.stats.fetched_rows += count
            self.stats.fetched_bytes += count * info.row_bytes
            fetches.append((segment, rows, out, None))
        return self.streamer.fetch_many(self.store, fetches)

    def prefetch_rows(self, requests: list[tuple[str, torch.Tensor | None]]):
        """A hint that the rows of `requests` ((segment, rows)) will be materialized soon.

        A store that can load them ahead (the native store, into its host cache, decision 0012) starts doing so in the
        background and returns a handle to `close` once they were used; otherwise None, and nothing happens. Bytes and
        results never depend on it.
        """
        prefetch = getattr(self.store, "prefetch", None)
        if self.resident or prefetch is None:
            return None
        return prefetch(requests)

    def _rows_through_cache(self, info: AnySegment, rows: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
        if not self.cache.holds(info.name) and not self.cache.can_admit(info.row_bytes):
            # No row of this segment is cached, and none could be: every row misses and is not kept.
            count = rows.numel()
            self.cache.count_misses(count, count * info.row_bytes)
            self.cache.bypass(count, count * info.row_bytes)
            return self._fetch(info, rows, out)
        row_list = rows.reshape(-1).tolist()
        cached = self.cache.get_many([(info.name, row) for row in row_list], info.row_bytes)
        missing = [k for k, entry in enumerate(cached) if entry is None]
        hits = len(row_list) - len(missing)
        self._hit(info, hits)
        if out is None:
            out = torch.empty(len(row_list), info.row_bytes, dtype=torch.uint8, device=self.device)
        # Hits first: their device copies are queued before storage is asked for the misses.
        for k, entry in enumerate(cached):
            if entry is not None:
                out[k].copy_(entry)
        self.stats.device_copy_bytes += hits * info.row_bytes
        if missing:
            positions = torch.tensor(missing, dtype=torch.int64)
            self._fetch(info, torch.tensor([row_list[k] for k in missing], dtype=torch.int64), out, positions)
            if self.cache.can_admit(info.row_bytes):
                for position in missing:
                    self.cache.put((info.name, row_list[position]), self._cache_copy(out[position]))
            else:
                self.cache.bypass(len(missing), len(missing) * info.row_bytes)
        return out

    def _hit(self, info: AnySegment, rows: int) -> None:
        self.stats.cache_hit_rows += rows
        self.stats.cache_hit_bytes += rows * info.row_bytes

    def _fetch(
        self, info: AnySegment, rows: torch.Tensor | None, out: torch.Tensor | None = None, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        count = info.rows if rows is None else rows.numel()
        self.stats.fetched_rows += count
        self.stats.fetched_bytes += count * info.row_bytes
        return self.streamer.fetch(self.store, info.name, rows, out, positions)

    def pin(self, segment: str) -> None:
        """Keep a whole segment resident in the cache (fetched now, never evicted)."""
        if self.cache is None:
            raise ValueError("pinning needs a page cache")
        info = self.store.segment(segment)
        key = (segment, WHOLE_SEGMENT)
        if key not in self.cache:
            data = self._fetch(info, None)
            self.cache.pin(key, self._cache_copy(data) if self._cache_pool is not None else data)
        else:
            self.cache.pin(key, self.cache.peek(key))

    def prefetch(self, segment: str, rows: torch.Tensor | None = None) -> Ticket:
        """Start fetching rows in the background (not counted as requested until used)."""
        if self.resident:
            raise ValueError("a resident store needs no prefetch")
        return self.streamer.prefetch(self.store, segment, rows)

    # Accounting

    def reset_stats(self, record_ranges: bool = False) -> None:
        self.stats.reset()
        self.store.stats.reset(record_ranges)
        if self.streamer is not None:
            self.streamer.stats.reset()
        if self.cache is not None:
            self.cache.stats.reset()

    def report(self) -> dict:
        """Everything counted since the last reset (device copy time is waited for)."""
        report = {"materialization": self.stats.as_dict(), "storage": self.store.stats.as_dict()}
        report["storage"]["io_ms"] = self.store.stats.io_ms
        if self.streamer is not None:
            report["transfer"] = self.streamer.stats.as_dict()
            report["transfer"]["copy_ms"] = self.streamer.stats.copy_ms()
        if self.cache is not None:
            report["cache"] = self.cache.stats.as_dict()
            report["cache"]["resident_bytes"] = self.cache.resident_bytes
        return report

    @property
    def device_resident_bytes(self) -> int:
        """Bytes kept on the compute device between requests (a resident store, or the cache)."""
        if self.resident:
            return self.store.resident_bytes
        return 0 if self.cache is None else self.cache.resident_bytes

    @property
    def host_resident_bytes(self) -> int:
        """Bytes kept in host memory between requests (an in-memory host store, the streamer's buffers)."""
        nbytes = 0 if self.resident else self.store.resident_bytes
        return nbytes + (0 if self.streamer is None else self.streamer.host_resident_bytes)
