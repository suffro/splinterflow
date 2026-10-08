"""Lossless codecs as published, and an addressable container of independently compressed pages (Phase 5C).

Codecs: Zstandard (python-zstandard, libzstd; optionally with a dictionary trained by its own `train_dictionary`) and
LZ4 (python-lz4's block format, default and high-compression modes). Nothing here compresses by itself: Weightsift only
chooses what bytes go in a page (`bits` transforms them exactly) and records where each page lands.

`PageFile` stores pages back to back with an index of (offset, length, raw length) per key. Decoding a page reads its
own byte range and nothing else (tested by poisoning every other byte); a dictionary, when used, is resident data
whose bytes are charged separately.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import lz4.block
import zstandard


@dataclass(frozen=True)
class Codec:
    """A codec setting: "zstd" (level), "lz4" (LZ4's default mode), "lz4hc" (high compression, level 1..12), "none"."""

    name: str
    level: int = 0

    @property
    def label(self) -> str:
        return self.name if self.name == "none" else f"{self.name}-{self.level}"

    def to_json(self) -> dict:
        return {"name": self.name, "level": self.level}


class Dictionary:
    """A trained Zstandard dictionary (its bytes are resident metadata, charged by the caller)."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.nbytes = len(data)
        self._dict = zstandard.ZstdCompressionDict(data)

    @classmethod
    def train(cls, size: int, samples: Sequence[bytes], level: int = 3, threads: int = -1) -> Dictionary:
        """zstd's own trainer (COVER, its parameters optimized by zstd) on `samples`, for compression at `level`."""
        trained = zstandard.train_dictionary(size, list(samples), level=level, threads=threads)
        return cls(trained.as_bytes())

    @property
    def zstd(self) -> zstandard.ZstdCompressionDict:
        return self._dict


class _Workers(threading.local):
    """Per-thread compressor objects (a ZstdCompressor is not thread-safe)."""

    def __init__(self) -> None:
        self.compressors: dict = {}
        self.decompressors: dict = {}


_WORKERS = _Workers()


def _zstd_compressor(level: int, dictionary: Dictionary | None) -> zstandard.ZstdCompressor:
    key = (level, id(dictionary))
    found = _WORKERS.compressors.get(key)
    if found is None:
        found = zstandard.ZstdCompressor(
            level=level, dict_data=None if dictionary is None else dictionary.zstd, write_checksum=False,
            write_content_size=True, write_dict_id=False,
        )
        _WORKERS.compressors[key] = found
    return found


def _zstd_decompressor(dictionary: Dictionary | None) -> zstandard.ZstdDecompressor:
    key = id(dictionary)
    found = _WORKERS.decompressors.get(key)
    if found is None:
        found = zstandard.ZstdDecompressor(dict_data=None if dictionary is None else dictionary.zstd)
        _WORKERS.decompressors[key] = found
    return found


def compress(data: bytes, codec: Codec, dictionary: Dictionary | None = None) -> bytes:
    if codec.name == "none":
        return bytes(data)
    if codec.name == "zstd":
        return _zstd_compressor(codec.level, dictionary).compress(data)
    if dictionary is not None:
        raise ValueError("dictionaries are zstd's")
    if codec.name == "lz4":
        return lz4.block.compress(data, mode="default", store_size=False)
    if codec.name == "lz4hc":
        return lz4.block.compress(data, mode="high_compression", compression=codec.level, store_size=False)
    raise ValueError(f"unknown codec {codec.name!r}")


def decompress(data: bytes, raw_length: int, codec: Codec, dictionary: Dictionary | None = None) -> bytes:
    if codec.name == "none":
        out = bytes(data)
    elif codec.name == "zstd":
        out = _zstd_decompressor(dictionary).decompress(data, max_output_size=raw_length)
    elif codec.name in ("lz4", "lz4hc"):
        out = lz4.block.decompress(data, uncompressed_size=raw_length)
    else:
        raise ValueError(f"unknown codec {codec.name!r}")
    if len(out) != raw_length:
        raise ValueError(f"decoded {len(out)} bytes, expected {raw_length}")
    return out


def compress_many(pages: Sequence[bytes], codec: Codec, dictionary: Dictionary | None = None, threads: int = 6) -> list[bytes]:
    """Each page compressed independently, as its own frame. zstd: python-zstandard's C-threaded batch API (one frame
    per page, the same frames `compress` writes); lz4: a thread pool (python-lz4 releases the GIL)."""
    if codec.name == "zstd" and len(pages) > 1:
        nonempty = [i for i, p in enumerate(pages) if len(p)]
        out = [compress(b"", codec, dictionary)] * len(pages)
        if nonempty:
            batch = _zstd_compressor(codec.level, dictionary).multi_compress_to_buffer([pages[i] for i in nonempty], threads=max(threads, 1))
            for k, i in enumerate(nonempty):
                out[i] = batch[k].tobytes()
        return out
    if threads <= 1 or len(pages) <= 1:
        return [compress(p, codec, dictionary) for p in pages]
    with ThreadPoolExecutor(threads) as pool:
        return list(pool.map(lambda p: compress(p, codec, dictionary), pages))


def decompress_many(frames: Sequence[bytes], raw_lengths: Sequence[int], codec: Codec, dictionary: Dictionary | None = None,
                    threads: int = 6) -> list[bytes]:
    """Inverse of `compress_many` (zstd: the C-threaded batch API; lengths checked)."""
    if codec.name == "zstd" and len(frames) > 1 and all(raw_lengths):
        batch = _zstd_decompressor(dictionary).multi_decompress_to_buffer(list(frames), threads=max(threads, 1))
        out = [batch[i].tobytes() for i in range(len(frames))]
        if [len(o) for o in out] != list(raw_lengths):
            raise ValueError("decoded lengths differ from the index")
        return out
    if threads <= 1 or len(frames) <= 1:
        return [decompress(f, n, codec, dictionary) for f, n in zip(frames, raw_lengths)]
    with ThreadPoolExecutor(threads) as pool:
        return list(pool.map(lambda pair: decompress(pair[0], pair[1], codec, dictionary), zip(frames, raw_lengths)))


@dataclass(frozen=True)
class PageEntry:
    offset: int
    length: int
    raw_length: int


@dataclass
class PageFile:
    """Independently compressed pages back to back (`blob`) and their index. `codec` and the dictionary are the file's."""

    codec: Codec
    blob: bytes
    index: dict[Hashable, PageEntry]
    dictionary: Dictionary | None = None
    timings: dict = field(default_factory=dict)

    @classmethod
    def build(cls, pages: dict[Hashable, bytes], codec: Codec, dictionary: Dictionary | None = None, threads: int = 6) -> PageFile:
        keys = list(pages)
        started = time.perf_counter()
        compressed = compress_many([pages[k] for k in keys], codec, dictionary, threads)
        elapsed = time.perf_counter() - started
        index, offset = {}, 0
        for key, data in zip(keys, compressed):
            index[key] = PageEntry(offset, len(data), len(pages[key]))
            offset += len(data)
        return cls(codec, b"".join(compressed), index, dictionary, {"compress_s": elapsed})

    @property
    def nbytes(self) -> int:
        return len(self.blob)

    @property
    def raw_bytes(self) -> int:
        return sum(entry.raw_length for entry in self.index.values())

    def page_bytes(self, key: Hashable) -> int:
        return self.index[key].length

    def read(self, key: Hashable, source: bytes | memoryview | None = None) -> bytes:
        """Decode one page from its own byte range of `source` (default: the file's blob) and nothing else."""
        entry = self.index[key]
        data = (self.blob if source is None else source)[entry.offset : entry.offset + entry.length]
        return decompress(bytes(data), entry.raw_length, self.codec, self.dictionary)

    def read_all(self, threads: int = 6) -> dict[Hashable, bytes]:
        keys = list(self.index)
        if threads <= 1:
            return {k: self.read(k) for k in keys}
        with ThreadPoolExecutor(threads) as pool:
            return dict(zip(keys, pool.map(self.read, keys)))
