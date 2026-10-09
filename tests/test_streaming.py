"""Phase 3 transfer and materialization: streamer, page cache, backend, typed weights, expert groups, packs."""

from __future__ import annotations

import json
import zlib

import pytest
import torch
from safetensors.torch import save_file

from awpmi.materialization.backend import MaterializationBackend
from awpmi.materialization.weights import ExpertGroup, ExpertStore, WeightStore
from awpmi.storage.cache import HotnessPolicy, LRUPolicy, PageCache, SlotCache
from awpmi.storage.layout import row_bytes_of, safetensors_segments
from awpmi.storage.pack import MANIFEST, PackWriter, SourceFile, open_pack
from awpmi.storage.store import FileBackedPageStore, InMemoryPageStore
from awpmi.streaming.streamer import PageStreamer
from tests.conftest import DEVICES
from tests.test_storage import expected_bytes, make_tensors, random_rows


@pytest.fixture
def checkpoint(tmp_path):
    tensors = make_tensors(5)
    path = tmp_path / "weights.safetensors"
    save_file(tensors, str(path))
    return path, tensors


def file_store(path, **options) -> FileBackedPageStore:
    return FileBackedPageStore({"w": path}, safetensors_segments(path, "w"), **options)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("slot_bytes", [8192, 64 * 1024, 8 << 20])
def test_streamer_delivers_exactly_the_requested_rows(checkpoint, device, slot_bytes):
    path, tensors = checkpoint
    store = file_store(path, direct=True, max_extent_bytes=16384)
    streamer = PageStreamer(device, slot_bytes=slot_bytes)
    generator = torch.Generator().manual_seed(slot_bytes)
    try:
        for name, tensor in tensors.items():
            for rows in (None, random_rows(tensor.shape[0], generator, 0.02), random_rows(tensor.shape[0], generator, 0.6)):
                streamer.stats.reset()
                out = streamer.fetch(store, name, rows)
                assert out.device.type == device and out.dtype == torch.uint8
                assert torch.equal(out.cpu(), expected_bytes(tensor, rows))
                count = tensor.shape[0] if rows is None else rows.numel()
                if device == "cuda":
                    # Only the requested rows cross to the device, whatever the blocks read.
                    assert streamer.stats.h2d_bytes == count * store.segment(name).row_bytes
                    assert streamer.stats.copy_ms() >= 0
    finally:
        store.close()
        streamer.close()


@pytest.mark.parametrize("device", DEVICES)
def test_prefetch_tickets_count_consumed_and_wasted_bytes(checkpoint, device):
    path, tensors = checkpoint
    store = file_store(path)
    streamer = PageStreamer(device)
    rows = torch.tensor([1, 5, 6, 200])
    try:
        used = streamer.prefetch(store, "level_records", rows)
        dropped = streamer.prefetch(store, "exact_rows")
        assert torch.equal(used.result().cpu(), expected_bytes(tensors["level_records"], rows))
        assert torch.equal(used.result().cpu(), expected_bytes(tensors["level_records"], rows))  # idempotent
        dropped.discard()
        stats = streamer.stats
        assert stats.prefetches == 2
        assert stats.consumed_bytes == 4 * 292
        assert stats.wasted_bytes == tensors["exact_rows"].numel() * 2
        assert stats.prefetched_bytes == stats.consumed_bytes + stats.wasted_bytes
    finally:
        streamer.close()
        store.close()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("slot_bytes", [4096, 8 << 20])
def test_streamer_moves_rows_from_a_pack_loaded_in_host_memory(tmp_path, device, slot_bytes):
    tensors = make_tensors(4)
    writer = PackWriter(tmp_path / "pack", "test-kind")
    for name, tensor in tensors.items():
        writer.add_tensor(name, tensor)
    store = writer.write({}).load()
    assert store.device.type == "cpu" and store.in_memory
    streamer = PageStreamer(device, slot_bytes=slot_bytes)
    generator = torch.Generator().manual_seed(9)
    for name, tensor in tensors.items():
        for rows in (None, random_rows(tensor.shape[0], generator, 0.1)):
            streamer.stats.reset()
            out = streamer.fetch(store, name, rows)
            assert torch.equal(out.cpu(), expected_bytes(tensor, rows))
            if device == "cuda":
                count = tensor.shape[0] if rows is None else rows.numel()
                assert streamer.stats.h2d_bytes == count * store.segment(name).row_bytes
    assert store.segment("exact_rows").dtype == "BF16"


def test_streamer_serves_a_host_memory_store():
    tensors = make_tensors(2)
    device = DEVICES[-1]
    streamer = PageStreamer(device)
    rows = torch.tensor([0, 3, 4])
    out = streamer.fetch(InMemoryPageStore(tensors), "exact_rows", rows)
    assert torch.equal(out.cpu(), expected_bytes(tensors["exact_rows"], rows))


def test_lru_evicts_the_least_recently_used_page():
    cache = PageCache(3 * 4, LRUPolicy())
    page = lambda value: torch.full((4,), value, dtype=torch.uint8)  # noqa: E731
    for key in "abc":
        assert cache.put(key, page(ord(key)))
    assert cache.get("a", 4) is not None  # a is now the most recent
    assert cache.put("d", page(1))
    assert "b" not in cache and set("acd") <= {k for k in "abcd" if k in cache}
    assert cache.stats.evictions == 1 and cache.resident_bytes == 12


def test_hotness_keeps_the_hot_page_that_lru_would_evict():
    page = torch.zeros(4, dtype=torch.uint8)
    for policy, survivor in ((LRUPolicy(), False), (HotnessPolicy(half_life=100), True)):
        cache = PageCache(3 * 4, policy)
        cache.put("hot", page.clone())
        for _ in range(10):
            cache.get("hot", 4)
        cache.put("x", page.clone())
        cache.put("y", page.clone())
        cache.get("x", 4)
        cache.get("y", 4)
        cache.put("z", page.clone())  # evicts one of hot, x, y
        assert ("hot" in cache) == survivor


HOTNESS_SIMULATION = """
import json, torch
from awpmi.storage.cache import HotnessPolicy, PageCache
cache = PageCache(4 * 4, HotnessPolicy(half_life=64))
evicted = []
original = cache._evict
cache._evict = lambda key: (evicted.append(list(key)), original(key))
generator = torch.Generator().manual_seed(0)
for step in range(200):
    rows = sorted(set(torch.randint(0, 12, (3,), generator=generator).tolist()))
    keys = [(f"layers.{step % 3}.experts.weight", row) for row in rows]
    found = cache.get_many(keys, 4)
    for key, entry in zip(keys, found):
        if entry is None:
            cache.put(key, torch.zeros(4, dtype=torch.uint8))
print(json.dumps(evicted))
"""


def test_hotness_evictions_do_not_depend_on_the_process_hash_seed():
    """Pages inserted in the same access tie on hotness; the tie must not be broken by set iteration order."""
    import os
    import subprocess
    import sys

    outputs = []
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run([sys.executable, "-c", HOTNESS_SIMULATION], capture_output=True, text=True, env=env, check=True)
        outputs.append(result.stdout)
    assert len(json.loads(outputs[0])) > 50
    assert outputs[0] == outputs[1] == outputs[2]


def test_pinned_pages_stay_and_a_frozen_cache_admits_nothing():
    page = torch.zeros(4, dtype=torch.uint8)
    cache = PageCache(8, LRUPolicy())
    cache.pin("base", page.clone())
    assert cache.put("a", page.clone()) and cache.put("b", page.clone())
    assert "base" in cache and "a" not in cache and "b" in cache
    assert not cache.put("big", torch.zeros(5, dtype=torch.uint8))  # larger than the unpinned budget
    cache.admit = False
    assert not cache.put("c", page.clone()) and "c" not in cache
    assert cache.stats.bypassed == 2
    with pytest.raises(ValueError):
        cache.pin("other", torch.zeros(5, dtype=torch.uint8))


def _page(key, nbytes: int) -> torch.Tensor:
    """A page whose bytes say which key and size it is."""
    generator = torch.Generator().manual_seed(zlib.crc32(f"{key}/{nbytes}".encode()))
    return torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=generator)


def test_a_slot_cache_copies_pages_into_slots_of_their_size():
    cache = SlotCache({8: 2, 4: 3}, "cpu")
    assert cache.copies and cache.capacity_bytes == 2 * 8 + 3 * 4 and cache.slots == {4: 3, 8: 2}
    source = _page("a", 8).clone()
    assert cache.put("a", source)
    source.zero_()  # the cache holds a copy, not the caller's tensor
    assert torch.equal(cache.get("a", 8), _page("a", 8))
    for key in "bcd":
        assert cache.put(key, _page(key, 4))
    assert cache.put("e", _page("e", 8)) and cache.put("f", _page("f", 8))  # evicts "a", the only page of its size
    assert "a" not in cache and all(key in cache for key in "bcdef")  # the small pages were not touched
    assert cache.resident_bytes == cache.capacity_bytes == cache.peak_resident_bytes
    assert cache.stats.inserts == 6 and cache.stats.evictions == 1 and cache.stats.hits == 1
    assert cache.get("x", 5) is None and not cache.can_admit(5) and not cache.put("x", torch.zeros(5, dtype=torch.uint8))
    cache.admit = False
    assert not cache.put("g", _page("g", 4)) and "g" not in cache and cache.stats.bypassed == 2
    cache.clear()
    assert len(cache) == 0 and cache.resident_bytes == 0
    cache.admit = True
    assert cache.put("g", _page("g", 4)) and torch.equal(cache.get("g", 4), _page("g", 4))
    with pytest.raises(ValueError):
        SlotCache({8: 0}, "cpu")


def test_slot_cache_decisions_are_independent_lrus_and_every_hit_is_its_page():
    """Random lookups and puts of two page sizes: hits, misses and evictions equal one LRU PageCache per size with the
    same budget, and every hit returns its own page's bytes (slots freed by evictions, also within one call, reused)."""
    slots = {12: 3, 6: 4}
    cache = SlotCache(slots, "cpu")
    reference = {size: PageCache(size * count, LRUPolicy()) for size, count in slots.items()}
    generator = torch.Generator().manual_seed(3)
    for step in range(400):
        size = (12, 6)[step % 2]
        keys = [("layer", int(k)) for k in torch.randint(0, 6, (3,), generator=generator).unique()]
        found, expected = cache.get_many(keys, size), reference[size].get_many(keys, size)
        assert [f is None for f in found] == [e is None for e in expected]
        for key, entry in zip(keys, found):
            if entry is not None:
                assert torch.equal(entry, _page(key, size))
        for key, entry in zip(keys, found):
            if entry is None:
                assert cache.put(key, _page(key, size)) == reference[size].put(key, _page(key, size))
        assert sorted(map(str, (k for size_ in slots for k in reference[size_]._entries))) == sorted(
            map(str, (k for pages in cache._sizes.values() for k in pages._entries))
        )
    assert cache.stats.evictions == sum(r.stats.evictions for r in reference.values()) > 100
    assert cache.stats.hits == sum(r.stats.hits for r in reference.values()) > 100


def test_a_slot_cache_shares_its_budget_in_proportion_to_the_pages_bytes():
    cache = SlotCache.sized(1200, {8: 2 * 800, 4: 800}, "cpu")
    assert cache.slots == {8: 100, 4: 100} and cache.capacity_bytes == 1200
    with pytest.raises(ValueError):
        SlotCache.sized(10, {8: 1, 4: 100}, "cpu")


@pytest.mark.parametrize("device", DEVICES)
def test_backend_caches_pages_and_counts_what_it_serves(checkpoint, device):
    path, tensors = checkpoint
    store = file_store(path)
    backend = MaterializationBackend(store, device, PageStreamer(device), PageCache(1 << 20, LRUPolicy()))
    data = tensors["experts"]
    try:
        first = backend.materialize("experts", torch.tensor([1, 4]))
        assert torch.equal(first.cpu(), expected_bytes(data, torch.tensor([1, 4])))
        backend.reset_stats()
        second = backend.materialize("experts", torch.tensor([0, 1, 4, 5]))
        assert torch.equal(second.cpu(), expected_bytes(data, torch.tensor([0, 1, 4, 5])))
        report = backend.report()
        row = store.segment("experts").row_bytes
        assert report["materialization"]["cache_hit_rows"] == 2 and report["materialization"]["fetched_rows"] == 2
        assert report["storage"]["logical_bytes"] == 2 * row and report["cache"]["hits"] == 2
        # A pinned whole segment serves row requests without storage reads.
        backend.pin("level_records")
        backend.reset_stats()
        rows = torch.tensor([3, 9, 600])
        assert torch.equal(backend.materialize("level_records", rows).cpu(), expected_bytes(tensors["level_records"], rows))
        assert backend.report()["storage"]["requests"] == 0
        assert backend.device_resident_bytes >= store.segment("level_records").nbytes
    finally:
        store.close()


def test_backend_on_a_resident_store_passes_through():
    tensors = make_tensors(3)
    store = InMemoryPageStore(tensors)
    backend = MaterializationBackend(store, "cpu")
    rows = torch.tensor([2, 7])
    assert torch.equal(backend.materialize("exact_rows", rows), expected_bytes(tensors["exact_rows"], rows))
    assert backend.device_resident_bytes == store.resident_bytes
    with pytest.raises(ValueError):
        MaterializationBackend(InMemoryPageStore({"a": torch.zeros(2, 2)}), "meta")


@pytest.mark.parametrize("device", DEVICES)
def test_weight_and_expert_stores(checkpoint, device):
    path, tensors = checkpoint
    store = file_store(path)
    weights = WeightStore(MaterializationBackend(store, device, PageStreamer(device)))
    try:
        rows = torch.tensor([0, 2])
        assert torch.equal(weights.rows("exact_rows", rows).cpu(), tensors["exact_rows"][rows])
        experts = ExpertStore(weights, {"layer": ExpertGroup("layer", 6, {"w": "experts"})})
        loaded = experts.load("layer", torch.tensor([1, 3]))
        assert torch.equal(loaded["w"].cpu(), tensors["experts"][[1, 3]])
        buffer = torch.full_like(tensors["experts"], float("nan"), device=device)
        experts.fill("layer", torch.tensor([0, 5]), {"w": buffer})
        assert torch.equal(buffer[[0, 5]].cpu(), tensors["experts"][[0, 5]])
        assert bool(buffer[[1, 2, 3, 4]].isnan().all())
        with pytest.raises(ValueError):
            ExpertStore(weights, {"bad": ExpertGroup("bad", 5, {"w": "experts"})})
    finally:
        store.close()


def test_pack_round_trip_with_a_source_checkpoint(tmp_path, checkpoint):
    source_path, tensors = checkpoint
    source = SourceFile("example/model", "0" * 40, "weights.safetensors")
    writer = PackWriter(tmp_path / "pack", "test-kind")
    new = torch.arange(24, dtype=torch.float32).view(6, 4)
    writer.add_tensor("new", new)
    writer.add_source("checkpoint", source, source_path)
    writer.add_source_segment("exact", "checkpoint", "exact_rows", expected=tensors["exact_rows"])
    pack = writer.write({"note": "test"}, {"alignment": 4096})
    resolve = lambda requested: source_path  # noqa: E731
    reopened = open_pack(tmp_path / "pack", resolve=resolve)
    assert reopened.segments == pack.segments and reopened.metadata == {"note": "test"}
    store = reopened.store(direct=True)
    try:
        assert torch.equal(store.read_rows("exact"), row_bytes_of(tensors["exact_rows"]))
        assert torch.equal(store.read_rows("new").view(torch.float32).view(6, 4), new)
    finally:
        store.close()
    with pytest.raises(ValueError):
        bad = PackWriter(tmp_path / "bad", "test-kind")
        bad.add_source("checkpoint", source, source_path)
        bad.add_source_segment("exact", "checkpoint", "exact_rows", expected=tensors["exact_rows"] + 1)
        bad.write({})


def test_pack_detects_tampering(tmp_path, checkpoint):
    source_path, tensors = checkpoint
    writer = PackWriter(tmp_path / "pack", "test-kind")
    writer.add_tensor("records", tensors["level_records"])
    pack = writer.write({})
    data_file = pack.files["pack"]
    raw = bytearray(data_file.read_bytes())
    offset = pack.segments["records"].offset + 5 * 292 + 7
    raw[offset] ^= 0x01
    data_file.write_bytes(bytes(raw))
    open_pack(tmp_path / "pack", verify="size")  # sizes still match
    with pytest.raises(ValueError):
        open_pack(tmp_path / "pack")  # the segment hash does not
    with pytest.raises(ValueError):
        open_pack(tmp_path / "pack", verify="files")
    manifest = json.loads((tmp_path / "pack" / MANIFEST).read_text())
    manifest["files"]["pack"]["bytes"] += 1
    (tmp_path / "pack" / MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        open_pack(tmp_path / "pack", verify="size")
    with pytest.raises(FileExistsError):
        PackWriter(tmp_path / "pack", "test-kind").write({})
