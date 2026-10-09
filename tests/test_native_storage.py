"""Phase 6A (decision 0012): the native storage backend plans, reads and counts exactly as the Python backend, and its
host-RAM cache serves exact bytes under any budget, eviction and concurrency."""

from __future__ import annotations

import gc
import threading
import time
import weakref

import pytest
import torch

from awpmi.materialization.backend import MaterializationBackend
from awpmi.storage import fileio
from awpmi.storage.layout import ComposedSegment, Segment, safetensors_segments
from awpmi.storage.native import NATIVE_AVAILABLE, NativePageStore
from awpmi.storage.store import IO_BLOCK_BYTES, FileBackedPageStore, plan_reads
from awpmi.streaming.streamer import PageStreamer
from tests.conftest import DEVICES
from tests.test_composed_storage import composed  # noqa: F401  (fixture)
from tests.test_storage import checkpoint, expected_bytes, random_rows  # noqa: F401  (fixture)

pytestmark = pytest.mark.skipif(not NATIVE_AVAILABLE, reason="the native extension weightsift_native is not built")

IO_FIELDS = ("requests", "rows", "logical_bytes", "physical_bytes", "read_calls", "extents", "blocks_4k")


def both(files, segments, **options):
    host = options.pop("host_cache_bytes", 0)
    return FileBackedPageStore(files, segments, **options), NativePageStore(files, segments, host_cache_bytes=host, **options)


def assert_plans_equal(python, native):
    for field in ("runs", "extents", "run_files", "extent_files"):
        assert torch.equal(getattr(python, field), getattr(native, field)), field
    assert (python.row_count, python.output_rows, python.files, python.blocks_4k) == (native.row_count, native.output_rows, native.files, native.blocks_4k)


@pytest.mark.parametrize("alignment, max_gap, max_extent", [(4096, 0, 8 << 20), (512, 0, 8 << 20), (4096, 8192, 8 << 20), (4096, 0, 8192)])
def test_plans_equal_the_python_planner(tmp_path, alignment, max_gap, max_extent):
    path = tmp_path / "f.bin"
    path.write_bytes(bytes(40_000_000))
    generator = torch.Generator().manual_seed(alignment + max_gap + max_extent)
    for case in range(40):
        row_bytes = [1, 7, 292, 1152, 9000][case % 5]
        segment = Segment("s", "f", int(torch.randint(0, 10_000, (1,), generator=generator)), 3000, row_bytes, "U8", (row_bytes,))
        store = NativePageStore({"f": path}, {"s": segment}, direct=False, alignment=alignment, max_gap=max_gap, max_extent_bytes=max_extent)
        try:
            for fraction in (0.0, 0.002, 0.05, 0.5, 1.0):
                rows = random_rows(segment.rows, generator, fraction)
                for positions in (None, torch.randperm(rows.numel() + 5, generator=generator)[: rows.numel()]):
                    expected = plan_reads(segment, rows, alignment, max_gap, max_extent, positions)
                    assert_plans_equal(expected, store.plan("s", rows, positions))
            assert_plans_equal(plan_reads(segment, None, alignment, max_gap, max_extent), store.plan("s", None))
        finally:
            store.close()


@pytest.mark.parametrize("max_gap, max_extent", [(0, 8 << 20), (8192, 8 << 20), (0, 8192)])
def test_composed_plans_and_pieces_equal_the_python_ones(composed, max_gap, max_extent):  # noqa: F811
    segment, paths, _ = composed
    store = NativePageStore(paths, {segment.name: segment}, direct=True, max_gap=max_gap, max_extent_bytes=max_extent)
    streamer = PageStreamer("cpu", slot_bytes=8192)
    generator = torch.Generator().manual_seed(max_gap + max_extent)
    try:
        for fraction in (0.1, 0.4, 0.8, 1.0):
            rows = (torch.rand(segment.rows, generator=generator) < fraction).nonzero().squeeze(1)
            for positions in (None, torch.randperm(rows.numel() + 3, generator=generator)[: rows.numel()]):
                expected = plan_reads(segment, rows, IO_BLOCK_BYTES, max_gap, max_extent, positions)
                assert_plans_equal(expected, store.plan(segment.name, rows, positions))
                for slot_bytes in (8192, 64 << 10, 1 << 20):
                    streamer.slot_bytes = slot_bytes
                    python = [
                        (list(map(tuple, p.extents.tolist())), list(map(tuple, p.parts.tolist())), p.files.tolist())
                        for p in streamer._pieces(expected)
                    ]
                    native = store.pieces(segment.name, rows, slot_bytes, positions)
                    assert [(extents, parts) for extents, parts, _ in python] == [
                        ([(o, n) for _, o, n in extents], parts) for extents, parts in native
                    ]
                    assert [files for _, _, files in python] == [[f for f, _, _ in extents] for extents, _ in native]
    finally:
        store.close()


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("workers", [1, 8])
def test_reads_and_counters_equal_the_python_store(checkpoint, direct, workers):  # noqa: F811
    path, tensors = checkpoint
    python, native = both({"w": path}, safetensors_segments(path, "w"), direct=direct, workers=workers, max_read_bytes=8192)
    generator = torch.Generator().manual_seed(int(direct) * 10 + workers)
    try:
        for name, tensor in tensors.items():
            for fraction in (0.0, 0.01, 0.3, 1.0):
                rows = random_rows(tensor.shape[0], generator, fraction)
                python.stats.reset(record_ranges=True)
                native.stats.reset(record_ranges=True)
                got = native.read_rows(name, rows)
                assert torch.equal(got, python.read_rows(name, rows))
                assert torch.equal(got, expected_bytes(tensor, rows))
                for field in IO_FIELDS:
                    assert getattr(native.stats, field) == getattr(python.stats, field), field
                assert native.stats.ranges == python.stats.ranges
                if native.stats.os_read_calls is not None:
                    assert (native.stats.os_read_calls, native.stats.os_read_bytes) == (native.stats.read_calls, native.stats.physical_bytes)
            assert torch.equal(native.read_rows(name), expected_bytes(tensor, None))
    finally:
        python.close()
        native.close()


def test_no_hidden_reads(checkpoint):  # noqa: F811
    """The native readers read the planned extents, once each, and nothing else: the OS saw exactly those reads."""
    path, tensors = checkpoint
    store = NativePageStore({"w": path}, safetensors_segments(path, "w"), direct=True, workers=4, max_read_bytes=8192)
    generator = torch.Generator().manual_seed(3)
    size = path.stat().st_size
    try:
        for name in ("level_records", "base_records", "exact_rows", "experts"):
            for fraction in (0.01, 0.1, 0.5):
                rows = random_rows(tensors[name].shape[0], generator, fraction)
                plan = plan_reads(store.segment(name), rows)
                store.stats.reset(record_ranges=True)
                store.read_rows(name, rows)
                assert store.stats.ranges == [("w", o, n) for o, n in plan.extents.tolist()]
                requested = {b for o, n, _, _ in plan.runs.tolist() for b in range(o // IO_BLOCK_BYTES, (o + n - 1) // IO_BLOCK_BYTES + 1)}
                for offset, length in plan.extents.tolist():
                    last = min(offset + length, size)
                    assert set(range(offset // IO_BLOCK_BYTES, (last - 1) // IO_BLOCK_BYTES + 1)) <= requested
                if store.stats.os_read_calls is not None:
                    assert store.stats.os_read_calls == store.stats.read_calls
                    assert store.stats.os_read_bytes == store.stats.physical_bytes
    finally:
        store.close()


def test_bytes_outside_the_requested_rows_never_reach_the_output(checkpoint, composed):  # noqa: F811
    path, _ = checkpoint
    segments = safetensors_segments(path, "w")
    generator = torch.Generator().manual_seed(11)
    for name in ("level_records", "exact_rows"):
        segment = segments[name]
        rows = random_rows(segment.rows, generator, 0.05)
        store = NativePageStore({"w": path}, segments, direct=True)
        before = store.read_rows(name, rows).clone()
        store.close()
        raw = bytearray(path.read_bytes())
        kept = set(rows.tolist())
        for row in range(segment.rows):
            if row not in kept:
                start = segment.offset + row * segment.row_bytes
                raw[start : start + segment.row_bytes] = b"\xff" * segment.row_bytes
        path.write_bytes(bytes(raw))
        store = NativePageStore({"w": path}, segments, direct=True)
        try:
            assert torch.equal(store.read_rows(name, rows), before)
        finally:
            store.close()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("slot_bytes", [8192, 64 * 1024, 8 << 20])
def test_streamed_rows_and_transfers_equal_the_python_path(composed, checkpoint, device, slot_bytes):  # noqa: F811
    """Without a cache the native path moves the same rows in the same copies: equal outputs and transfer counters."""
    segment, paths, expected = composed
    path, tensors = checkpoint
    files = {**paths, "w": path}
    segments = {segment.name: segment, **safetensors_segments(path, "w")}
    python, native = both(files, segments, direct=True, max_extent_bytes=16384)
    generator = torch.Generator().manual_seed(slot_bytes)
    try:
        for name in (segment.name, "level_records", "exact_rows", "experts"):
            info = segments[name]
            truth = expected if name == segment.name else torch.cat([expected_bytes(tensors[name], None)])
            for fraction in (0.1, 0.5, 1.0):
                rows = random_rows(info.rows, generator, fraction)
                count = rows.numel()
                outputs = []
                for store in (python, native):
                    streamer = PageStreamer(device, slot_bytes=slot_bytes)
                    buffer = torch.zeros(count + 2, info.row_bytes, dtype=torch.uint8, device=device)
                    positions = torch.randperm(count + 2, generator=torch.Generator().manual_seed(count))[:count]
                    streamer.fetch(store, name, rows, out=buffer, positions=positions)
                    plain = streamer.fetch(store, name, rows)
                    outputs.append((buffer.cpu(), plain.cpu(), streamer.stats.as_dict()))
                    streamer.close()
                assert torch.equal(outputs[0][0], outputs[1][0]) and torch.equal(outputs[0][1], outputs[1][1])
                assert torch.equal(outputs[1][1], truth[rows] if rows is not None else truth)
                if device == "cuda":
                    for key in ("fetches", "pieces", "h2d_bytes", "h2d_copies", "gathered_bytes"):
                        assert outputs[0][2][key] == outputs[1][2][key], key
    finally:
        python.close()
        native.close()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("budget_rows", [0, 1, 5, 1000])
def test_the_host_cache_serves_exact_bytes_under_any_budget(composed, device, budget_rows):  # noqa: F811
    segment, paths, expected = composed
    store = NativePageStore(paths, {segment.name: segment}, direct=True, max_extent_bytes=16384, host_cache_bytes=budget_rows * segment.row_bytes)
    streamer = PageStreamer(device, slot_bytes=32 << 10, native_slots=3)
    backend = MaterializationBackend(store, device, streamer)
    generator = torch.Generator().manual_seed(budget_rows)
    try:
        requested = 0
        for step in range(40):
            rows = random_rows(segment.rows, generator, [0.1, 0.3, 0.7, 1.0][step % 4])
            out = torch.zeros(rows.numel(), segment.row_bytes, dtype=torch.uint8, device=device)
            backend.materialize(segment.name, rows, out=out)
            assert torch.equal(out.cpu(), expected[rows])
            requested += rows.numel()
            cache = store.cache_stats()
            if cache is not None:
                assert cache["resident_bytes"] <= cache["capacity_bytes"] and cache["peak_resident_bytes"] <= cache["capacity_bytes"]
        stats = store.stats.as_dict()
        assert stats["rows"] == requested and stats["logical_bytes"] == requested * segment.row_bytes
        cache = stats.get("host_cache")
        if budget_rows == 0:
            assert cache is None
        else:
            assert cache["lookups"] == requested == cache["hits"] + cache["waits"] + cache["misses"]
            # What the cache served was copied; what it missed was read: exactly the 4 KiB blocks of the missed rows
            # (the benchmarks' audit relation, with blocks counted for the rows read from storage only).
            assert stats["native"]["cache_copied_bytes"] == cache["hit_bytes"] + cache["wait_bytes"]
            assert 0 <= stats["blocks_4k"] * IO_BLOCK_BYTES - stats["physical_bytes"] < IO_BLOCK_BYTES * max(1, stats["requests"])
            assert stats["blocks_4k"] * IO_BLOCK_BYTES < cache["miss_bytes"] + 2 * 3 * IO_BLOCK_BYTES * cache["misses"] + 1
            if budget_rows >= segment.rows:
                assert cache["evictions"] == 0 and cache["inserts"] == segment.rows
            elif budget_rows > 1:
                assert cache["evictions"] > 0 and cache["hits"] > 0
        if device == "cuda":
            assert streamer.stats.h2d_bytes == requested * segment.row_bytes
    finally:
        store.close()
        streamer.close()


def test_the_cache_block_size_changes_no_byte_and_no_cache_decision(composed):  # noqa: F811
    """Phase 6B (decision 0013): the host cache holds rows in blocks that evicted rows hand to the next ones, whatever their
    sizes. With blocks smaller than a row the same requests give the same bytes, hits, misses, evictions and cached rows as
    one allocation per row (Phase 6A's memory), and everything the cache holds, its pool of blocks included, stays within
    its budget."""
    segment, paths, expected = composed  # rows of 12,388 bytes
    outcomes = {}
    for block in (1 << 30, 4096, 1000):  # one allocation per row; three blocks and a tail; twelve blocks and a tail
        store = NativePageStore(
            paths, {segment.name: segment}, direct=True, max_extent_bytes=16384, host_cache_bytes=5 * segment.row_bytes + 77,
            cache_block_bytes=block,
        )
        streamer = PageStreamer("cpu", slot_bytes=32 << 10, native_slots=3)
        backend = MaterializationBackend(store, "cpu", streamer)
        generator = torch.Generator().manual_seed(7)
        try:
            for step in range(30):
                rows = random_rows(segment.rows, generator, [0.1, 0.3, 0.7][step % 3])
                assert torch.equal(backend.materialize(segment.name, rows), expected[rows])
                cache = store.cache_stats()
                assert cache["held_bytes"] <= cache["capacity_bytes"] and cache["peak_held_bytes"] <= cache["capacity_bytes"]
            cache = store.cache_stats()
            assert cache["block_bytes"] == block
            outcomes[block] = (
                {k: cache[k] for k in ("lookups", "hits", "waits", "misses", "inserts", "evictions", "bypassed")}, store.cached_rows(),
            )
            if block < segment.row_bytes:
                assert cache["recycled_bytes"] > 0 and cache["allocated_bytes"] < cache["insert_bytes"]
        finally:
            store.close()
            streamer.close()
    assert outcomes[4096] == outcomes[1 << 30] == outcomes[1000]
    assert outcomes[1 << 30][0]["evictions"] > 10


@pytest.mark.parametrize("device", DEVICES)
def test_several_requests_in_one_transfer(checkpoint, device):  # noqa: F811
    path, tensors = checkpoint
    store = NativePageStore({"w": path}, safetensors_segments(path, "w"), direct=True, host_cache_bytes=1 << 20)
    backend = MaterializationBackend(store, device, PageStreamer(device, slot_bytes=64 << 10))
    try:
        for _ in range(2):  # cold, then from the cache
            rows = torch.tensor([0, 3, 4, 5])
            outs = [torch.zeros(4, store.segment(name).row_bytes, dtype=torch.uint8, device=device) for name in ("experts", "exact_rows")]
            backend.materialize_many([("experts", rows, outs[0]), ("exact_rows", rows, outs[1])])
            assert torch.equal(outs[0].cpu(), expected_bytes(tensors["experts"], rows))
            assert torch.equal(outs[1].cpu(), expected_bytes(tensors["exact_rows"], rows))
        assert backend.stats.requests == 4 and store.stats.requests == 4
        assert store.cache_stats()["hits"] == 8
    finally:
        store.close()


@pytest.mark.parametrize("device", DEVICES)
def test_prefetched_rows_are_served_from_the_cache(composed, device):  # noqa: F811
    segment, paths, expected = composed
    store = NativePageStore(paths, {segment.name: segment}, direct=True, max_extent_bytes=16384, host_cache_bytes=1 << 20)
    backend = MaterializationBackend(store, device, PageStreamer(device, slot_bytes=32 << 10, native_slots=3))
    try:
        rows = torch.tensor([1, 2, 5, 8, 13])
        backend.reset_stats()
        prefetch = backend.prefetch_rows([(segment.name, rows)])
        assert prefetch.rows == 5
        # Overlapping native activity: a transfer starts while the prefetch may still read; each read is counted once.
        out = backend.materialize(segment.name, torch.tensor([1, 2, 3]))
        prefetch.close()
        assert torch.equal(out.cpu(), expected[[1, 2, 3]])
        assert torch.equal(backend.materialize(segment.name, rows).cpu(), expected[rows])
        stats = store.stats.as_dict()
        cache = stats["host_cache"]
        # Rows 1 and 2 were prefetched (hits or waits), 3 a miss; then 1, 2, 5, 8, 13 all hit.
        assert (cache["lookups"], cache["misses"]) == (8, 1)
        assert (cache["prefetch_fills"], cache["prefetch_used"], cache["prefetch_wasted"]) == (5, 5, 0)
        assert stats["native"]["prefetch_rows"] == 5
        if stats["os_read_calls"] is not None:
            assert (stats["os_read_calls"], stats["os_read_bytes"]) == (stats["read_calls"], stats["physical_bytes"])
        blocks = stats["blocks_4k"] + stats["native"]["prefetch_blocks_4k"]
        assert 0 <= blocks * IO_BLOCK_BYTES - stats["physical_bytes"] < IO_BLOCK_BYTES * (stats["requests"] + stats["native"]["prefetches"])
    finally:
        store.close()
    plain = NativePageStore(paths, {segment.name: segment}, direct=True)
    try:
        assert MaterializationBackend(plain, "cpu", PageStreamer("cpu")).prefetch_rows([(segment.name, torch.tensor([1]))]) is None
    finally:
        plain.close()


def test_a_frozen_cache_serves_hits_and_admits_nothing(composed):  # noqa: F811
    segment, paths, expected = composed
    store = NativePageStore(paths, {segment.name: segment}, direct=True, host_cache_bytes=1 << 20)
    backend = MaterializationBackend(store, "cpu", PageStreamer("cpu"))
    try:
        backend.materialize(segment.name, torch.tensor([1, 2]))
        store.set_admit(False)
        out = backend.materialize(segment.name, torch.tensor([1, 2, 3]))
        assert torch.equal(out, expected[[1, 2, 3]])
        assert sorted(store.cached_rows()) == [(segment.name, 1), (segment.name, 2)]
        cache = store.cache_stats()
        assert (cache["hits"], cache["bypassed"], cache["admit"]) == (2, 1, False)
        store.clear_cache()
        assert store.cached_rows() == [] and store.resident_bytes == 0
    finally:
        store.close()


def test_threads_sharing_a_cache_load_each_row_once(composed):  # noqa: F811
    segment, paths, expected = composed
    store = NativePageStore(paths, {segment.name: segment}, direct=True, workers=8, host_cache_bytes=1 << 20)
    failures = []

    def run(seed: int) -> None:
        backend = MaterializationBackend(store, "cpu", PageStreamer("cpu", slot_bytes=16 << 10))
        generator = torch.Generator().manual_seed(seed)
        try:
            for _ in range(25):
                rows = random_rows(segment.rows, generator, 0.5)
                if not torch.equal(backend.materialize(segment.name, rows), expected[rows]):
                    failures.append(seed)
        except Exception as error:  # surfaced below
            failures.append(error)

    threads = [threading.Thread(target=run, args=(seed,)) for seed in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    try:
        assert not failures
        cache = store.cache_stats()
        assert cache["inserts"] <= segment.rows and cache["evictions"] == 0 and cache["aborted_fills"] == 0
        assert store.stats.as_dict()["native"]["fallback_rows"] == 0
    finally:
        store.close()


def test_the_native_store_refuses_what_the_python_store_refuses(checkpoint):  # noqa: F811
    path, _ = checkpoint
    segments = safetensors_segments(path, "w")
    with pytest.raises(ValueError):
        NativePageStore({"w": path}, segments, direct=True, alignment=512)
    with pytest.raises(KeyError):
        NativePageStore({"other": path}, {"x": Segment("x", "w", 0, 1, 1, "U8", (1,))})
    with pytest.raises(ValueError):
        NativePageStore({"w": path}, {"x": Segment("x", "w", path.stat().st_size - 10, 2, 8, "U8", (8,))})
    bad = ComposedSegment("c", 1, 8, "U8", (8,), (8,), ((("w", path.stat().st_size - 4),),))
    with pytest.raises(ValueError):
        NativePageStore({"w": path}, {"c": bad})
    store = NativePageStore({"w": path}, segments, direct=True)
    try:
        with pytest.raises(ValueError):
            store.read_rows("exact_rows", torch.tensor([3, 2]))
        with pytest.raises(IndexError):
            store.read_rows("exact_rows", torch.tensor([10_000]))
        with pytest.raises(KeyError):
            store.read_rows("nope")
    finally:
        store.close()


def test_read_errors_reach_python_as_os_errors(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(bytes(range(256)) * 400)
    segment = Segment("s", "f", 1000, 100, 512, "U8", (512,))
    python, native = both({"f": path}, {"s": segment}, direct=False, workers=2)
    try:
        expected = torch.frombuffer(bytearray(path.read_bytes()[1000 + 3 * 512 : 1000 + 4 * 512]), dtype=torch.uint8).view(1, 512)
        assert torch.equal(native.read_rows("s", torch.tensor([3])), expected)
        with open(path, "r+b") as handle:  # the file shrinks under both stores (each knows its size from opening)
            handle.truncate(2000)
        for rows in (torch.tensor([50, 60]), torch.tensor([0])):
            with pytest.raises(OSError, match="short read"):
                python.read_rows("s", rows)
            with pytest.raises(OSError, match="short read"):
                native.read_rows("s", rows)
    finally:
        python.close()
        native.close()


def test_native_reads_release_the_gil(tmp_path):
    """While a native read is in flight, another Python thread runs (the core never holds the GIL while it reads)."""
    path = tmp_path / "big.bin"
    path.write_bytes(bytes(48 << 20))
    segment = Segment("s", "f", 0, 48 << 10, 1024, "U8", (1024,))
    store = NativePageStore({"f": path}, {"s": segment}, direct=False, workers=1, max_read_bytes=4096, max_extent_bytes=4096)
    window = {}

    def read() -> None:
        window["start"] = time.perf_counter()
        store.read_rows("s", torch.arange(0, 48 << 10, 2))  # every other row: 24k separate extents
        window["end"] = time.perf_counter()

    thread = threading.Thread(target=read)
    ticks = []
    try:
        thread.start()
        while thread.is_alive():
            ticks.append(time.perf_counter())
        thread.join()
    finally:
        store.close()
    inside = [t for t in ticks if window["start"] < t < window["end"]]
    assert window["end"] - window["start"] > 0.01 and len(inside) > 1000


def test_transfers_cancel_and_the_engine_closes_cleanly(composed):  # noqa: F811
    segment, paths, expected = composed
    store = NativePageStore(paths, {segment.name: segment}, direct=True, max_extent_bytes=4096, host_cache_bytes=1 << 20)
    slots = [(fileio.aligned_host_buffer(8192, pin=False), fileio.aligned_host_buffer(8192, pin=False)) for _ in range(2)]
    try:
        with store.stream([(segment.name, None, None)], slots) as job:
            pieces = iter(job)
            index, _, _ = next(pieces)
            job.release(index)
            job.cancel()
            with pytest.raises(RuntimeError, match="cancelled"):
                next(pieces)
        cache = store.cache_stats()
        assert cache["resident_bytes"] % segment.row_bytes == 0 and cache["resident_bytes"] <= cache["capacity_bytes"]
        # The same rows are then served exactly (whatever the cancelled job had cached).
        assert torch.equal(MaterializationBackend(store, "cpu", PageStreamer("cpu")).materialize(segment.name, None), expected)
    finally:
        store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.read_rows(segment.name, torch.tensor([1]))


def test_a_closed_store_releases_its_cache_and_needs_no_collector(composed):  # noqa: F811
    """`close` gives the host cache's memory back at once, and the store is freed by reference counting alone: a store
    kept alive by a reference cycle held its whole cache across Phase 6A's first run (the working-set gate caught it)."""
    segment, paths, expected = composed
    store = NativePageStore(paths, {segment.name: segment}, direct=True, host_cache_bytes=1 << 20)
    backend = MaterializationBackend(store, "cpu", PageStreamer("cpu"))
    assert torch.equal(backend.materialize(segment.name, torch.tensor([1, 2, 3])), expected[[1, 2, 3]])
    assert store.resident_bytes == 3 * segment.row_bytes
    store.close()
    assert store.resident_bytes == 0 and store.cached_rows() == []
    collected = weakref.ref(store)
    gc.disable()
    try:
        del backend, store
        assert collected() is None
    finally:
        gc.enable()
