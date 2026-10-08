"""PageStreamer: storage → pinned host staging → copy stream → device, with buffer reuse and prefetch.

Patterns adapted from Soup's layer streaming (decision 0006), without depending on it:

  * a ring of preallocated, page-aligned, pinned staging slots (two by default: double
    buffering), reused for every transfer;
  * a dedicated CUDA copy stream for host-to-device copies, so a copy overlaps whatever
    the compute stream is doing;
  * events: a slot is refilled only after the copy out of it has completed, and the
    compute stream waits for the copy of a tensor before it can use it;
  * prefetch: a background thread runs a transfer ahead of need and hands back a ticket.

`fetch(store, segment, rows)` for a file-backed store plans the read, splits its extents
into slot-sized pieces, and for each piece: waits for a free slot, reads the piece's
extents into it, gathers the requested rows (or copies a contiguous run as it is) and
issues the copy into the destination. The storage read of piece k+1 overlaps the copy of
piece k. Only the requested rows' bytes are copied to the device: the rest of an aligned
block stays in host staging. The returned tensor holds the rows in ascending order, as
raw bytes [n, row_bytes].

With `out` (a caller's device buffer of rows), the rows are written there instead, row i
of the request into row `positions[i]` (default i); the copy stream first waits for the
work already queued on the caller's stream, which may still be reading that buffer. A
piece whose runs land in separate places of the destination (a composed segment's spans,
scattered positions) is gathered once and copied in one transfer per contiguous range.

Runs of at least 256 KiB (an expert's tensors) are copied straight from pinned staging to
their destination: gathering them first would copy every byte once more on the host, which
cost a sixth of a streamed decode step in the Phase 4A profile (decision 0007). Smaller runs
(one row of a weight, a refinement record) are gathered, so that a piece needs few transfers.

Prefetch never materializes anything logically: a ticket's data is counted as consumed
only when the ticket is used, and as wasted when it is dropped.

A native store (`awpmi.storage.native`, Phase 6A, decision 0012) plans, reads, gathers and
serves its host cache itself, into `native_slots` pinned slots of the streamer's: one
transfer per call (`fetch_many` moves several requests in one), pieces delivered as soon as
they are ready, and the native readers keep reading into the other slots while a piece is
copied. The streamer issues each piece's copies on the copy stream (the same copies Python
would issue for the same plan) and hands a slot back once its copies finished, never
holding more than all but one, so the readers always have a slot to fill. That loop
(`_transfer_native`) is Phase 6B's integration point: a native copy issuer, or a device
decoder for a compressed representation, takes each piece's operations instead.
"""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import torch

from awpmi.storage.fileio import DIRECT_ALIGNMENT, aligned_host_buffer
from awpmi.storage.native import NativePageStore
from awpmi.storage.store import FileBackedPageStore, InMemoryPageStore, PageStore, ReadPlan, check_rows, gather_runs

DEFAULT_SLOT_BYTES = 8 << 20
DEFAULT_NATIVE_SLOTS = 4
DIRECT_COPY_BYTES = 256 << 10  # runs at least this long skip the host gather


@dataclass
class TransferStats:
    """Counters of one streamer; `reset` starts a new window. Device copy time is read with `copy_ms()`."""

    fetches: int = 0
    pieces: int = 0
    h2d_bytes: int = 0
    h2d_copies: int = 0
    gathered_bytes: int = 0  # bytes gathered in host memory before a copy
    prefetches: int = 0
    prefetched_bytes: int = 0
    consumed_bytes: int = 0
    wasted_bytes: int = 0
    _copy_events: list = field(default_factory=list)

    def reset(self) -> None:
        self.__init__()

    def copy_ms(self) -> float:
        """Device time of the host-to-device copies of this window (waits for them)."""
        total = 0.0
        for start, end in self._copy_events:
            end.synchronize()
            total += start.elapsed_time(end)
        return total

    def as_dict(self) -> dict:
        return {
            "fetches": self.fetches,
            "pieces": self.pieces,
            "h2d_bytes": self.h2d_bytes,
            "h2d_copies": self.h2d_copies,
            "gathered_bytes": self.gathered_bytes,
            "prefetches": self.prefetches,
            "prefetched_bytes": self.prefetched_bytes,
            "consumed_bytes": self.consumed_bytes,
            "wasted_bytes": self.wasted_bytes,
        }


@dataclass(frozen=True)
class _Piece:
    extents: torch.Tensor  # [k, 2] file offset, length (read back to back into a slot)
    files: torch.Tensor  # [k] index of each extent's file in the plan's files
    parts: torch.Tensor  # [p, 3] staging offset, length, destination offset


class _Slot:
    def __init__(self, nbytes: int, cuda: bool) -> None:
        self.staging = aligned_host_buffer(nbytes, DIRECT_ALIGNMENT, pin=cuda)
        self.compact = aligned_host_buffer(nbytes, DIRECT_ALIGNMENT, pin=cuda)
        self.free = torch.cuda.Event() if cuda else None
        self.used = False

    def wait_free(self) -> None:
        if self.free is not None and self.used:
            self.free.synchronize()


class Ticket:
    """A transfer started by `PageStreamer.prefetch`; `result()` hands its tensor to the caller's stream."""

    def __init__(self, streamer: PageStreamer, future: Future, nbytes: int) -> None:
        self._streamer = streamer
        self._future = future
        self.nbytes = nbytes
        self._settled = False

    def done(self) -> bool:
        return self._future.done()

    def result(self) -> torch.Tensor:
        tensor, ready = self._future.result()
        if not self._settled:
            self._settled = True
            self._streamer.stats.consumed_bytes += self.nbytes
        self._streamer._hand_over(tensor, ready)
        return tensor

    def discard(self) -> None:
        """Drop an unused prefetch: its bytes are counted as wasted."""
        if not self._settled:
            self._settled = True
            self._streamer.stats.wasted_bytes += self.nbytes
            self._future.cancel()


def resolve_device(device: torch.device | str) -> torch.device:
    """`device` with an explicit index for CUDA ("cuda" → "cuda:<current>"), so devices compare reliably."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


class PageStreamer:
    """Moves rows of a store's segments to `device`; see the module docstring."""

    def __init__(
        self, device: torch.device | str, slot_bytes: int = DEFAULT_SLOT_BYTES, slots: int = 2, native_slots: int = DEFAULT_NATIVE_SLOTS
    ) -> None:
        self.device = resolve_device(device)
        if slot_bytes % DIRECT_ALIGNMENT or slots < 1 or native_slots < 2:
            raise ValueError("slots must be at least one (two for a native store), of a multiple of the I/O alignment")
        self.cuda = self.device.type == "cuda"
        self.slot_bytes = slot_bytes
        self._slots = [_Slot(slot_bytes, self.cuda) for _ in range(slots)]
        self._next_slot = 0
        self.native_slots = native_slots
        self._native: list[_Slot] = []  # allocated by the first native transfer
        self.copy_stream = torch.cuda.Stream(self.device) if self.cuda else None
        self._lock = threading.Lock()
        self._prefetcher: ThreadPoolExecutor | None = None
        self.stats = TransferStats()

    @property
    def host_resident_bytes(self) -> int:
        """Pinned staging and gather buffers held by the streamer."""
        return sum(slot.staging.numel() + slot.compact.numel() for slot in [*self._slots, *self._native])

    # Synchronous path

    def fetch(
        self,
        store: PageStore,
        segment: str,
        rows: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rows of `segment` on the device, ready for use on the current stream ([n, row_bytes] uint8).

        With `out` (contiguous uint8 [m, row_bytes] on the device), request i is written to
        out[positions[i]] (default i) and `out` is returned.
        """
        self._check_out(store, segment, out, positions)
        with self._lock:
            tensor, ready = self._transfer(store, segment, rows, out, positions)
        self._hand_over(tensor, ready)
        return tensor

    def fetch_many(
        self, store: PageStore, requests: list[tuple[str, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]]
    ) -> list[torch.Tensor]:
        """`fetch` of several requests, each (segment, rows, out, positions); returns their tensors.

        A native store moves them in one transfer (its hits first, then each request's planned reads, one pipeline);
        any other store is fetched request by request, exactly as separate `fetch` calls.
        """
        if not isinstance(store, NativePageStore):
            return [self.fetch(store, segment, rows, out, positions) for segment, rows, out, positions in requests]
        for segment, _, out, positions in requests:
            self._check_out(store, segment, out, positions)
        with self._lock:
            tensors = self._transfer_native(store, requests)
            ready = self._record_ready()
        for tensor in tensors:
            self._hand_over(tensor, ready)
        return tensors

    def _check_out(self, store: PageStore, segment: str, out: torch.Tensor | None, positions: torch.Tensor | None) -> None:
        if out is not None:
            info = store.segment(segment)
            if out.dtype != torch.uint8 or out.dim() != 2 or out.shape[1] != info.row_bytes or not out.is_contiguous():
                raise ValueError(f"out must be contiguous uint8 [rows, {info.row_bytes}]")
            if resolve_device(out.device) != self.device:
                raise ValueError("out must be on the streamer's device")
        elif positions is not None:
            raise ValueError("positions need an output buffer")

    def _hand_over(self, tensor: torch.Tensor, ready) -> None:
        if ready is not None:
            current = torch.cuda.current_stream(self.device)
            current.wait_event(ready)
            tensor.record_stream(current)

    def _transfer(
        self, store: PageStore, segment: str, rows: torch.Tensor | None, out: torch.Tensor | None = None, positions=None
    ):
        if isinstance(store, NativePageStore):
            (tensor,) = self._transfer_native(store, [(segment, rows, out, positions)])
            return tensor, self._record_ready()
        self.stats.fetches += 1
        if store.in_memory and resolve_device(store.device) == self.device:
            return _place(store.read_rows(segment, rows), out, positions), None
        if isinstance(store, InMemoryPageStore) and store.device.type == "cpu":
            data = self._transfer_host_memory(store, segment, rows)
            return self._place_on_copy_stream(data, out, positions), self._record_ready()
        if not isinstance(store, FileBackedPageStore):
            data = self._copy_host_tensor(store.read_rows(segment, rows))
            return self._place_on_copy_stream(data, out, positions), self._record_ready()
        plan = store.plan(segment, rows, positions)
        store.count_plan(plan)
        if out is not None:
            if out.shape[0] < plan.output_rows:
                raise ValueError(f"out has {out.shape[0]} rows, the request needs {plan.output_rows}")
            dest = out.view(-1)
            if self.cuda:
                # The caller's stream may still be using this buffer (e.g. the previous layer's kernels).
                self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
                out.record_stream(self.copy_stream)
        elif self.cuda:
            with torch.cuda.stream(self.copy_stream):
                dest = torch.empty(plan.logical_bytes, dtype=torch.uint8, device=self.device)
        else:
            dest = torch.empty(plan.logical_bytes, dtype=torch.uint8)
        for piece in self._pieces(plan):
            self._move_piece(store, plan, piece, dest)
        result = out if out is not None else dest.view(plan.row_count, plan.segment.row_bytes)
        return result, self._record_ready()

    def _place_on_copy_stream(self, data: torch.Tensor, out: torch.Tensor | None, positions) -> torch.Tensor:
        if out is None:
            return data
        if not self.cuda:
            return _place(data, out, positions)
        self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.copy_stream):
            _place(data, out, positions)
        return out

    def _record_ready(self):
        if not self.cuda:
            return None
        ready = torch.cuda.Event()
        ready.record(self.copy_stream)
        return ready

    def _transfer_host_memory(self, store: InMemoryPageStore, segment: str, rows: torch.Tensor | None) -> torch.Tensor:
        """Rows of a host-memory store: gathered into pinned slots (or copied straight from pinned memory), then copied."""
        info = store.segment(segment)
        data = store.segment_bytes(segment)
        rows = check_rows(rows, info)
        count = info.rows if rows is None else rows.numel()
        store.count_gather(segment, count)
        if not self.cuda:
            return (data if rows is None else data.index_select(0, rows)).clone()
        with torch.cuda.stream(self.copy_stream):
            dest = torch.empty(count, info.row_bytes, dtype=torch.uint8, device=self.device)
        step = max(1, self.slot_bytes // info.row_bytes)
        pinned = data.is_pinned()
        for first in range(0, count, step):
            last = min(first + step, count)
            if rows is None and pinned:
                source, slot = data[first:last], None
            elif info.row_bytes * (last - first) <= self.slot_bytes:
                slot = self._slots[self._next_slot]
                self._next_slot = (self._next_slot + 1) % len(self._slots)
                slot.wait_free()
                source = slot.compact[: (last - first) * info.row_bytes].view(last - first, info.row_bytes)
                if rows is None:
                    source.copy_(data[first:last])
                else:
                    torch.index_select(data, 0, rows[first:last], out=source)
                self.stats.gathered_bytes += source.numel()
            else:  # a row larger than a slot, from pageable memory: copied synchronously
                source, slot = data[rows[first : first + 1] if rows is not None else slice(first, last)], None
            self._copy(dest[first:last], source, slot, asynchronous=slot is not None or pinned)
            self.stats.pieces += 1
        return dest

    def _copy(self, dest: torch.Tensor, source: torch.Tensor, slot: _Slot | None, asynchronous: bool) -> None:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.copy_stream):
            start.record()
            dest.copy_(source, non_blocking=asynchronous)
            end.record()
            if slot is not None:
                slot.free.record()
                slot.used = True
        self.stats._copy_events.append((start, end))
        self.stats.h2d_bytes += source.numel()
        self.stats.h2d_copies += 1

    def _copy_host_tensor(self, host: torch.Tensor) -> torch.Tensor:
        """A host tensor from any other store, copied directly (synchronously when it is pageable)."""
        if not self.cuda:
            return host.clone()
        with torch.cuda.stream(self.copy_stream):
            dest = torch.empty_like(host, device=self.device)
            dest.copy_(host)  # pageable source: synchronous with respect to the host
        self.stats.h2d_bytes += host.numel()
        self.stats.h2d_copies += 1
        return dest

    def _pieces(self, plan: ReadPlan) -> list[_Piece]:
        """Split a plan into slot-sized pieces: extents read back to back, and the parts of runs they hold."""
        limit = self.slot_bytes
        extents, runs, files = plan.extents, plan.runs, plan.extent_files
        if extents.numel() == 0:
            return []
        solo = [False] * extents.shape[0]
        if bool((extents[:, 1] > limit).any()):
            extents, runs, files, solo = _chunk_long_extents(extents, runs, files, limit)
        lengths = extents[:, 1].tolist()
        piece_of_extent, position = [], []
        piece, used, closed = 0, 0, False
        for length, alone in zip(lengths, solo):
            # A chunk of a long run gets a piece of its own: its parts may start mid-row.
            if used and (closed or alone or used + length > limit):
                piece, used = piece + 1, 0
            piece_of_extent.append(piece)
            position.append(used)
            used += length
            closed = alone
        piece_of_extent = torch.tensor(piece_of_extent, dtype=torch.int64)
        position = torch.tensor(position, dtype=torch.int64)
        extent_of_run = runs[:, 3]
        sources = position[extent_of_run] + runs[:, 0] - extents[extent_of_run, 0]
        parts = torch.stack([sources, runs[:, 1], runs[:, 2]], dim=1)
        piece_of_run = piece_of_extent[extent_of_run]
        count = piece + 1
        bounds = torch.arange(count + 1)
        extent_split = torch.searchsorted(piece_of_extent, bounds).tolist()
        run_split = torch.searchsorted(piece_of_run, bounds).tolist()
        return [
            _Piece(
                extents[extent_split[k] : extent_split[k + 1]],
                files[extent_split[k] : extent_split[k + 1]],
                parts[run_split[k] : run_split[k + 1]],
            )
            for k in range(count)
        ]

    def _move_piece(self, store: FileBackedPageStore, plan: ReadPlan, piece: _Piece, dest: torch.Tensor) -> None:
        slot = self._slots[self._next_slot]
        self._next_slot = (self._next_slot + 1) % len(self._slots)
        slot.wait_free()
        store.read_extents(plan, piece.extents, slot.staging, piece.files)
        self.stats.pieces += 1
        parts = piece.parts
        if parts.shape[0] == 0:  # a chunk that lies in a merged gap (max_gap > 0): read, nothing to move
            return
        if parts.shape[0] > 1:
            # Long runs go straight from staging; the rest is gathered below.
            direct = parts[:, 1] >= DIRECT_COPY_BYTES
            for source_offset, length, target in parts[direct].tolist():
                self._move(dest[target : target + length], slot.staging[source_offset : source_offset + length], slot)
            parts = parts[~direct]
            if parts.shape[0] == 0:
                return
        lengths = parts[:, 1]
        total = int(lengths.sum())
        if parts.shape[0] == 1:
            source = slot.staging[int(parts[0, 0]) : int(parts[0, 0]) + total]
        else:
            gather_runs(slot.staging, parts[:, 0], lengths, slot.compact[:total], plan.segment.row_bytes)
            source = slot.compact[:total]
            self.stats.gathered_bytes += total
        # One transfer per range of parts that are contiguous in the destination too.
        targets = parts[:, 2]
        breaks = (targets[1:] != targets[:-1] + lengths[:-1]).nonzero().squeeze(1) + 1
        starts = [0, *breaks.tolist(), parts.shape[0]]
        within = (torch.cumsum(lengths, 0) - lengths).tolist()
        for a, b in zip(starts[:-1], starts[1:]):
            first, size = int(targets[a]), int(lengths[a:b].sum())
            self._move(dest[first : first + size], source[within[a] : within[a] + size], slot)

    def _move(self, dest: torch.Tensor, source: torch.Tensor, slot: _Slot) -> None:
        if not self.cuda:
            dest.copy_(source)
        else:
            self._copy(dest, source, slot, asynchronous=True)

    # Native stores

    def _transfer_native(self, store: NativePageStore, requests: list) -> list[torch.Tensor]:
        """One native transfer of every request; returns each request's tensor (rows, or the caller's `out`)."""
        if not self._native:
            self._native = [_Slot(self.slot_bytes, self.cuda) for _ in range(self.native_slots)]
        dests, results = [], []
        for segment, rows, out, positions in requests:
            self.stats.fetches += 1
            info = store.segment(segment)
            count = info.rows if rows is None else rows.numel()
            if out is not None:
                needed = count if positions is None else (int(positions.max()) + 1 if count else 0)
                if out.shape[0] < needed:
                    raise ValueError(f"out has {out.shape[0]} rows, the request needs {needed}")
                if self.cuda:
                    # The caller's stream may still be using this buffer (e.g. the previous layer's kernels).
                    self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
                    out.record_stream(self.copy_stream)
                dests.append(out.view(-1))
                results.append(out)
            else:
                if self.cuda:
                    with torch.cuda.stream(self.copy_stream):
                        dest = torch.empty(count * info.row_bytes, dtype=torch.uint8, device=self.device)
                else:
                    dest = torch.empty(count * info.row_bytes, dtype=torch.uint8)
                dests.append(dest)
                results.append(dest.view(count, info.row_bytes))
        # The native readers may write any slot at once: every copy out of the previous transfer must be done.
        for slot in self._native:
            slot.wait_free()
        held: deque[int] = deque()
        with store.stream([(segment, rows, positions) for segment, rows, _, positions in requests], [(s.staging, s.compact) for s in self._native]) as job:
            for index, ops, gathered in job:
                slot = self._native[index]
                self.stats.pieces += 1
                self.stats.gathered_bytes += gathered
                for source, src, request, dst, length in ops:
                    buffer = slot.staging if source == 0 else slot.compact
                    self._move(dests[request][dst : dst + length], buffer[src : src + length], slot)
                if not self.cuda or not ops:
                    job.release(index)
                    continue
                held.append(index)
                while len(held) >= len(self._native) - 1:
                    oldest = held.popleft()
                    self._native[oldest].wait_free()
                    job.release(oldest)
        return results

    # Prefetch

    def prefetch(self, store: PageStore, segment: str, rows: torch.Tensor | None = None) -> Ticket:
        """Start a transfer in the background; the data is the caller's only through `Ticket.result()`."""
        if self._prefetcher is None:
            self._prefetcher = ThreadPoolExecutor(max_workers=1, thread_name_prefix="awpmi-prefetch")
        info = store.segment(segment)
        nbytes = info.nbytes if rows is None else rows.numel() * info.row_bytes
        self.stats.prefetches += 1
        self.stats.prefetched_bytes += nbytes

        def run():
            if self.cuda:
                torch.cuda.set_device(self.device)
            with self._lock:
                return self._transfer(store, segment, rows)

        return Ticket(self, self._prefetcher.submit(run), nbytes)

    def close(self) -> None:
        if self._prefetcher is not None:
            self._prefetcher.shutdown()
            self._prefetcher = None


def _place(data: torch.Tensor, out: torch.Tensor | None, positions) -> torch.Tensor:
    """Write rows `data` into out[positions] (default: the first rows of `out`); without `out`, `data` itself."""
    if out is None:
        return data
    if positions is None:
        out[: data.shape[0]].copy_(data)
    else:
        out.index_copy_(0, positions.reshape(-1).to(out.device, torch.int64), data)
    return out


def _chunk_long_extents(extents: torch.Tensor, runs: torch.Tensor, files: torch.Tensor, limit: int):
    """Split extents longer than `limit` into `limit`-sized chunks, and the runs they hold at chunk boundaries.

    Returns the new extents, runs and extent files, and which extents are chunks (to be moved alone).
    """
    new_extents, new_runs, new_files, solo = [], [], [], []
    run_list = runs.tolist()
    index = 0
    for extent_id, ((offset, length), file) in enumerate(zip(extents.tolist(), files.tolist())):
        mine = []
        while index < len(run_list) and run_list[index][3] == extent_id:
            mine.append(run_list[index])
            index += 1
        if length <= limit:
            new_extents.append((offset, length))
            new_files.append(file)
            solo.append(False)
            new_runs.extend((o, n, out, len(new_extents) - 1) for o, n, out, _ in mine)
            continue
        for chunk_start in range(offset, offset + length, limit):
            chunk_end = min(chunk_start + limit, offset + length)
            new_extents.append((chunk_start, chunk_end - chunk_start))
            new_files.append(file)
            solo.append(True)
            for run_offset, run_length, run_out, _ in mine:
                begin, end = max(run_offset, chunk_start), min(run_offset + run_length, chunk_end)
                if begin < end:
                    new_runs.append((begin, end - begin, run_out + begin - run_offset, len(new_extents) - 1))
    as_tensor = lambda rows, width: torch.tensor(rows, dtype=torch.int64).reshape(-1, width)  # noqa: E731
    return as_tensor(new_extents, 2), as_tensor(new_runs, 4), torch.tensor(new_files, dtype=torch.int64), solo
