"""Phase 6B (decision 0013): encoded packs store every row compressed by nvCOMP in independent chunks, and the GPU decodes
them back to exactly the source bytes, through every store and the host cache; corruption is caught."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from awpmi.materialization.backend import MaterializationBackend
from awpmi.storage.encoded import LARGE_ROW_BYTES, ROW_ALIGNMENT, Encoding, chunk_offsets, is_encoded, open_encoded, stored_sizes
from awpmi.storage.native import NATIVE_AVAILABLE
from awpmi.storage.pack import PackWriter, open_pack
from awpmi.streaming import nvcomp
from awpmi.streaming.streamer import PageStreamer

needs_gpu = pytest.mark.skipif(not (torch.cuda.is_available() and nvcomp.AVAILABLE), reason="encoded rows need CUDA and nvCOMP")


def weights(rows: int, shape: tuple[int, int], seed: int) -> torch.Tensor:
    """BF16 weights like a layer's (Gaussian: their exponents compress, their mantissas do not), with a few edge patterns."""
    generator = torch.Generator().manual_seed(seed)
    tensor = (torch.randn((rows, *shape), generator=generator) * 0.02).to(torch.bfloat16)
    flat = tensor.view(-1).view(torch.int16)
    flat[:6] = torch.tensor([0, -32768, 1, 0x7F7F, -0x0081, 0x0080], dtype=torch.int16)  # ±0, a subnormal, ±largest, smallest normal
    return tensor


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """A pack of two large segments (rows of 4 MiB and 2 MiB, Moonlight's 2:1) and a small one (rows of 4 KiB)."""
    directory = tmp_path_factory.mktemp("source")
    writer = PackWriter(directory / "pack", "moe-experts")
    writer.add_tensor("layer.big", weights(6, (1024, 2048), 1))
    writer.add_tensor("layer.half", weights(6, (1024, 1024), 2))
    writer.add_tensor("layer.small", weights(6, (2, 1024), 3))
    pack = writer.write({"groups": {"layer": {"experts": 6, "segments": {"a": "layer.big", "b": "layer.half", "c": "layer.small"}}}})
    return open_pack(pack.directory, verify="segments")


@pytest.fixture(scope="module")
def encoded(source, tmp_path_factory):
    if not (torch.cuda.is_available() and nvcomp.AVAILABLE):
        pytest.skip("encoded rows need CUDA and nvCOMP")
    from awpmi.streaming.codec import write_encoded_pack

    return write_encoded_pack(source, tmp_path_factory.mktemp("encoded") / "pack", chunk_bytes=512 << 10)


def test_stored_sizes_make_large_rows_whole_blocks():
    logical = {"big": 2 * LARGE_ROW_BYTES, "half": LARGE_ROW_BYTES, "small": 4096}
    needed = {"big": 1_400_000, "half": 690_001, "small": 2_900}
    sizes, block = stored_sizes(needed, logical)
    assert block % ROW_ALIGNMENT == 0 and block >= needed["big"] / 2 and block >= needed["half"]
    assert (sizes["big"], sizes["half"]) == (2 * block, block)
    assert sizes["small"] == ROW_ALIGNMENT  # small rows are padded alone
    assert stored_sizes({"small": 10}, {"small": 100}) == ({"small": ROW_ALIGNMENT}, 0)


def test_chunks_are_aligned_and_back_to_back():
    sizes = np.array([[5, 17, 1], [16, 16, 16]])
    starts, ends = chunk_offsets(sizes)
    assert starts.tolist() == [[0, 16, 48], [0, 16, 32]]
    assert ends.tolist() == [49, 48]


@needs_gpu
def test_an_encoded_pack_records_how_every_row_decodes(source, encoded):
    assert is_encoded(encoded.pack.directory) and not is_encoded(source.directory)
    again = open_encoded(encoded.pack.directory, verify="segments")  # every stored segment re-hashed
    assert again.encodings.keys() == {"layer.big", "layer.half", "layer.small"}
    block = encoded.metadata["encoding"]["block_bytes"]
    assert encoded.encodings["layer.big"].stored_row_bytes == 2 * encoded.encodings["layer.half"].stored_row_bytes == 2 * block
    for name, item in encoded.encodings.items():
        logical = source.segments[name]
        assert (item.logical.rows, item.logical.row_bytes, item.logical.dtype, item.logical.row_shape) == (
            logical.rows, logical.row_bytes, logical.dtype, logical.row_shape)
        assert item.codec == "ans" and item.options == {"data_type": "float16"}
        assert Encoding.from_json(name, json.loads(json.dumps(item.to_json()))).to_json() == item.to_json()
    assert encoded.metadata["groups"] == source.metadata["groups"]
    assert 0.5 < encoded.pack.manifest["packing"]["compressed_ratio"] < 0.85  # Gaussian BF16: only the exponents compress


@needs_gpu
@pytest.mark.parametrize("backend", ["python", pytest.param("native", marks=pytest.mark.skipif(not NATIVE_AVAILABLE, reason="no native extension")),
                                     pytest.param("native-cache", marks=pytest.mark.skipif(not NATIVE_AVAILABLE, reason="no native extension"))])
def test_decoded_rows_equal_the_source_rows(source, encoded, backend):
    from awpmi.streaming.codec import RowDecoder

    options = {"direct": True} if backend == "python" else {"direct": True, "host_cache_bytes": 3 * encoded.encodings["layer.big"].stored_row_bytes if backend == "native-cache" else 0}
    store = encoded.pack.store(backend="python" if backend == "python" else "native", **options)
    decoder = RowDecoder(encoded.encodings, "cuda")
    materializer = MaterializationBackend(store, "cuda", PageStreamer("cuda"), decoder=decoder)
    reference = source.store(direct=True)
    generator = torch.Generator().manual_seed(4)
    counted = {"hits": 0, "evictions": 0}
    try:
        for step in range(12):
            requests = []
            for name in ("layer.big", "layer.half", "layer.small"):
                rows = (torch.rand(6, generator=generator) < 0.6).nonzero().squeeze(1)
                if step % 4 == 0:
                    rows = torch.arange(6)
                requests.append((name, rows, None))
            outputs = materializer.materialize_many(requests)
            for (name, rows, _), out in zip(requests, outputs):
                assert materializer.segment(name).row_bytes == source.segments[name].row_bytes
                assert torch.equal(out.cpu(), reference.read_rows(name, rows)), (name, rows.tolist())
            report = materializer.report()  # checks every chunk's status
            m = report["materialization"]
            assert m["decoded_bytes"] == m["requested_bytes"] and m["fetched_bytes"] == m["stored_requested_bytes"]
            assert report["storage"]["logical_bytes"] == m["fetched_bytes"] == report["transfer"]["h2d_bytes"]
            if backend == "native-cache":
                cache = report["storage"]["host_cache"]
                assert cache["held_bytes"] <= cache["capacity_bytes"] and cache["peak_held_bytes"] <= cache["capacity_bytes"]
                counted = {k: counted[k] + cache[k] for k in counted}
            materializer.reset_stats()
        if backend == "native-cache":
            cache = store.cache_stats()
            assert counted["hits"] > 0 and counted["evictions"] > 0
            assert cache["block_bytes"] == encoded.metadata["encoding"]["block_bytes"]  # the rows split into no tails
    finally:
        store.close()
        reference.close()


@needs_gpu
def test_a_chunk_that_decodes_to_the_wrong_size_fails_the_check(source, encoded):
    """The guard behind the decoder: a chunk whose decompressed size is not the expected one is counted, and the next check
    raises (here a stored row whose first chunk header was zeroed: nvCOMP reports no error status for it, but decodes
    nothing). nvCOMP does not detect corrupted payloads: the pack's digests (`open_encoded(verify="files")`) and the
    benchmarks' audit of every decoded row do."""
    from awpmi.streaming.codec import RowDecoder

    item = encoded.encodings["layer.half"]
    decoder = RowDecoder({"layer.half": item}, "cuda")
    store = encoded.pack.store(direct=True)
    try:
        staging = store.read_rows("layer.half", torch.tensor([0])).cuda()
        staging[0, :32] = 0
        out = torch.empty(1, item.logical.row_bytes, dtype=torch.uint8, device="cuda")
        decoder.decode([("layer.half", torch.tensor([0]), np.array([staging.data_ptr()]), out)])
        with pytest.raises(RuntimeError, match="failed to decode"):
            decoder.check()
        decoder.check()  # the count was cleared
    finally:
        store.close()


@needs_gpu
def test_a_corrupted_stored_row_is_caught_by_the_packs_digests(encoded, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(encoded.pack.directory, copy)
    segment = encoded.pack.segments["layer.big"]
    path = copy / encoded.pack.manifest["files"][segment.file]["path"]
    data = bytearray(path.read_bytes())
    data[segment.offset + 100] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256|differs"):
        open_encoded(copy, verify="files")


def test_a_decoder_needs_cuda(source):
    if not nvcomp.AVAILABLE:
        pytest.skip("no nvCOMP")
    from awpmi.streaming.codec import RowDecoder

    with pytest.raises(ValueError):
        RowDecoder({}, "cpu")


@needs_gpu
@pytest.mark.parametrize("kind", ["pages", "slots"])
def test_a_device_cache_of_encoded_rows_decodes_its_hits_from_their_entries(source, encoded, kind):
    """Phase 6B: the device cache holds stored rows (copies in an allocator pool, or fixed slots); hits are decoded from
    their entries (nothing fetched), misses fetched, decoded and offered; every row exact, every byte counted, the budget
    held."""
    from awpmi.storage.cache import LRUPolicy, PageCache, SlotCache
    from awpmi.streaming.codec import RowDecoder

    if kind == "pages":
        budget = 4 * encoded.encodings["layer.big"].stored_row_bytes
        cache = PageCache(budget, LRUPolicy())
    else:
        cache = SlotCache({encoded.pack.segments[name].row_bytes: 2 for name in ("layer.big", "layer.half")}, "cuda")
        budget = cache.capacity_bytes
    store = encoded.pack.store(direct=True)
    materializer = MaterializationBackend(store, "cuda", PageStreamer("cuda"), cache, decoder=RowDecoder(encoded.encodings, "cuda"))
    reference = source.store(direct=True)
    generator = torch.Generator().manual_seed(9)
    hits = evictions = 0
    try:
        for step in range(16):
            requests = [(name, (torch.rand(6, generator=generator) < 0.5).nonzero().squeeze(1), None) for name in ("layer.big", "layer.half")]
            for (name, rows, _), out in zip(requests, materializer.materialize_many(requests)):
                assert torch.equal(out.cpu(), reference.read_rows(name, rows))
            report = materializer.report()
            m = report["materialization"]
            assert m["cache_hit_bytes"] + m["fetched_bytes"] == m["stored_requested_bytes"] and m["decoded_bytes"] == m["requested_bytes"]
            assert report["storage"]["logical_bytes"] == m["fetched_bytes"] == report["transfer"]["h2d_bytes"]
            assert cache.resident_bytes <= budget
            assert report["cache"]["hits"] == m["cache_hit_rows"]
            hits += m["cache_hit_rows"]
            evictions += report["cache"]["evictions"]
            materializer.reset_stats()
        assert hits > 0 and evictions > 0
        # With admission frozen, misses are decoded and not kept.
        cache.admit = False
        before = cache.resident_bytes
        rows = torch.arange(6)
        assert torch.equal(materializer.materialize("layer.half", rows).cpu(), reference.read_rows("layer.half", rows))
        assert cache.resident_bytes == before
    finally:
        store.close()
        reference.close()
