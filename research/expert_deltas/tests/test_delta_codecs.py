"""Codecs restore their input exactly; a page decodes from its own bytes alone."""

from __future__ import annotations

import numpy as np
import pytest

from expert_deltas import bits
from expert_deltas.codecs import Codec, Dictionary, PageFile, compress, decompress

CODECS = [Codec("none"), Codec("zstd", 1), Codec("zstd", 3), Codec("zstd", 19), Codec("lz4"), Codec("lz4hc", 9)]


def payloads():
    rng = np.random.default_rng(0)
    gaussian = rng.normal(0, 0.02, 50_000).astype(np.float32)
    weights = bits.patterns(__import__("torch").from_numpy(gaussian).to(__import__("torch").bfloat16))
    yield b""
    yield bytes(1)
    yield rng.integers(0, 256, 10_000, dtype=np.uint8).tobytes()
    yield weights.tobytes()
    yield bits.byte_split(weights)
    yield np.arange(1 << 16, dtype=np.uint32).astype(np.uint16).tobytes()


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.label)
def test_codecs_restore_exactly(codec):
    for data in payloads():
        assert decompress(compress(data, codec), len(data), codec) == data


def test_dictionary_round_trip_and_charge():
    rng = np.random.default_rng(1)
    samples = [rng.normal(0, 0.02, 2048).astype(np.float32).astype(np.float16).tobytes() for _ in range(400)]
    dictionary = Dictionary.train(4096, samples, level=3)
    assert 0 < dictionary.nbytes <= 4096
    codec = Codec("zstd", 3)
    for data in samples[:20]:
        assert decompress(compress(data, codec, dictionary), len(data), codec, dictionary) == data
    with pytest.raises(ValueError):
        compress(samples[0], Codec("lz4"), dictionary)


@pytest.mark.parametrize("codec", [Codec("zstd", 3), Codec("lz4"), Codec("lz4hc", 9)], ids=lambda c: c.label)
def test_a_page_decodes_from_its_own_bytes_only(codec):
    rng = np.random.default_rng(2)
    pages = {("expert", k): rng.normal(0, 1, 3000 + 7 * k).astype(np.float16).tobytes() for k in range(8)}
    file = PageFile.build(pages, codec, threads=4)
    assert file.raw_bytes == sum(len(v) for v in pages.values())
    for key, data in pages.items():
        entry = file.index[key]
        poisoned = bytearray(rng.integers(0, 256, len(file.blob), dtype=np.uint8).tobytes())
        poisoned[entry.offset : entry.offset + entry.length] = file.blob[entry.offset : entry.offset + entry.length]
        assert file.read(key, bytes(poisoned)) == data  # every other byte poisoned
    assert file.read_all(threads=3) == pages


def test_poisoning_a_page_is_noticed():
    """The guard behind the test above: corrupting the page's own bytes does change (or break) its decoding."""
    rng = np.random.default_rng(3)
    pages = {k: rng.normal(0, 1, 4000).astype(np.float16).tobytes() for k in range(3)}
    file = PageFile.build(pages, Codec("zstd", 3), threads=1)
    entry = file.index[1]
    poisoned = bytearray(file.blob)
    for i in range(entry.offset, entry.offset + entry.length):
        poisoned[i] ^= 0x5A
    try:
        changed = file.read(1, bytes(poisoned)) != pages[1]
    except Exception:  # noqa: BLE001 — a corrupt frame may fail to decode
        changed = True
    assert changed


@pytest.mark.parametrize("codec", [Codec("zstd", 3), Codec("zstd", 19), Codec("lz4"), Codec("lz4hc", 9)], ids=lambda c: c.label)
def test_batch_frames_equal_single_frames(codec):
    """The batch paths (python-zstandard's C threads, a thread pool for lz4) write and read the same frames as one call each."""
    from expert_deltas.codecs import compress_many, decompress_many

    rng = np.random.default_rng(4)
    pages = [rng.normal(0, 1, 1000 + 13 * k).astype(np.float16).tobytes() for k in range(9)] + [b""]
    for batch in (pages, pages[:-1]):  # with an empty page (per-frame path) and without (zstd's batch path)
        frames = compress_many(batch, codec, threads=4)
        assert frames == [compress(p, codec) for p in batch]
        assert decompress_many(frames, [len(p) for p in batch], codec, threads=4) == batch
        assert decompress_many(frames, [len(p) for p in batch], codec, threads=1) == batch


def test_measure_restores_deltas_and_flags_errors():
    """`measure` compares finished objects with the reference: a correct XOR restoration is exact, a wrong base is not."""
    from expert_deltas.compression import decode_throughput, measure

    rng = np.random.default_rng(5)

    def expert():
        return {"gate": rng.integers(0, 0x7F00, (32, 16)).astype(np.uint16), "up": rng.integers(0, 0x7F00, (32, 16)).astype(np.uint16),
                "down": rng.integers(0, 0x7F00, (16, 32)).astype(np.uint16)}

    weights = {e: expert() for e in range(3)}
    deltas = {e: {m: bits.xor_delta(weights[e][m], weights[0][m]) if e else weights[e][m] for m in weights[e]} for e in weights}

    def finish(e, m, p):
        return p if e == 0 else bits.xor_restore(p, weights[0][m])

    for kind in ("row", "rows16", "tensor", "expert"):
        for transform in ("raw", "byte_split", "planes"):
            entry = measure(deltas, weights, kind, transform, Codec("zstd", 3), threads=2, finish=finish)
            assert entry["exact"] and len(entry["per_object_stored"]) == 3
    wrong = measure(deltas, weights, "tensor", "planes", Codec("zstd", 3), threads=2, finish=lambda e, m, p: p)
    assert not wrong["exact"]
    timed = decode_throughput(deltas, "rows16", "planes", Codec("zstd", 3), 2, lambda e, m: None if e == 0 else weights[0][m], repeats=1)
    assert set(timed["throughput"]) == {"decompress_gb_s", "cpu_restore_gb_s"}


@pytest.mark.parametrize("transform", ["raw", "byte_split", "planes"])
def test_gpu_restore_equals_restore(transform):
    import torch

    from expert_deltas.compression import gpu_restore, restore, streams

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(6)
    for count in (8, 4096, 22528):
        p = rng.integers(0, 1 << 16, count, dtype=np.uint32).astype(np.uint16)
        parts = streams(p, transform)
        expected = restore(parts, count, transform).astype(np.int64)
        assert np.array_equal(gpu_restore(parts, count, transform, device).cpu().numpy().astype(np.int64), expected)
