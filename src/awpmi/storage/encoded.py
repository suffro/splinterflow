"""Encoded segments: an exact, compressed representation of a pack's rows, derived from it (Phase 6B, decision 0013).

An encoded pack stores every row of some source segments compressed in independent chunks. It is a Weightsift pack
(`open_pack`: files with their sha256, plain segments of stored rows that any store reads), whose metadata says, per
segment, what its rows decode to and how:

  logical       the source rows: count, bytes, dtype, shape, and each row's sha256 (what decoding must give back)
  codec         nvCOMP's codec and options (rANS in its float16 mode for BF16 rows: stage 6B2-A's probe chose it)
  chunk_bytes   a logical row is cut in chunks of this many bytes (the last may be shorter), each compressed alone
  chunk_sizes   [rows][chunks] compressed bytes; a row's chunks are back to back from the start of its stored row, each
                at a multiple of CHUNK_ALIGNMENT
  stored_row_bytes  every row of the segment padded to one size, a multiple of the pack's block (`stored_sizes`), so
                    that a cache whose block divides every row keeps whole blocks (the native host cache's default)

Nothing here compresses or decompresses: `awpmi.streaming.codec` does, on the GPU, through nvCOMP. Nothing here reads a
checkpoint either: the writer gets the source rows from a store. The source pack and the checkpoint it indexes stay the
reference; an encoded pack is checked against them row by row (`row_sha256`, and the runtime's audits).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from awpmi.storage.layout import SAFETENSORS_DTYPES
from awpmi.storage.pack import MANIFEST, Pack, open_pack

ENCODING_FORMAT = "weightsift-encoded-rows"
ENCODING_VERSION = 1
CHUNK_ALIGNMENT = 16  # nvCOMP's decompressors require at most 8-byte aligned inputs
ROW_ALIGNMENT = 4096


def round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


@dataclass(frozen=True)
class LogicalRows:
    """What a segment's rows decode to: a segment-like description (no file: the rows exist only once decoded)."""

    name: str
    rows: int
    row_bytes: int
    dtype: str
    row_shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return self.rows * self.row_bytes

    @property
    def torch_dtype(self) -> torch.dtype:
        return SAFETENSORS_DTYPES[self.dtype]

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.rows, *self.row_shape)


@dataclass(frozen=True, eq=False)
class Encoding:
    """How one segment's rows are stored (see the module docstring)."""

    logical: LogicalRows
    codec: str
    options: dict[str, Any]
    chunk_bytes: int
    stored_row_bytes: int
    chunk_sizes: np.ndarray  # int64 [rows, chunks]
    row_sha256: tuple[str, ...]
    offsets: np.ndarray = field(init=False, repr=False, compare=False)  # int64 [rows, chunks], within a stored row
    lengths: np.ndarray = field(init=False, repr=False, compare=False)  # int64 [chunks], uncompressed

    def __post_init__(self) -> None:
        rows, chunks = self.chunk_sizes.shape
        if rows != self.logical.rows or chunks != math.ceil(self.logical.row_bytes / self.chunk_bytes):
            raise ValueError(f"{self.logical.name}: a chunk table of {rows}x{chunks} for {self.logical.rows} rows of {self.logical.row_bytes} bytes")
        offsets, ends = chunk_offsets(self.chunk_sizes)
        if int(ends.max()) > self.stored_row_bytes:
            raise ValueError(f"{self.logical.name}: a row's chunks exceed its stored size")
        lengths = np.full(chunks, self.chunk_bytes, dtype=np.int64)
        lengths[-1] = self.logical.row_bytes - self.chunk_bytes * (chunks - 1)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "lengths", lengths)

    @property
    def chunks(self) -> int:
        return self.chunk_sizes.shape[1]

    def to_json(self) -> dict[str, Any]:
        return {
            "logical": {
                "rows": self.logical.rows, "row_bytes": self.logical.row_bytes, "dtype": self.logical.dtype,
                "row_shape": list(self.logical.row_shape), "row_sha256": list(self.row_sha256),
            },
            "codec": self.codec,
            "options": dict(self.options),
            "chunk_bytes": self.chunk_bytes,
            "stored_row_bytes": self.stored_row_bytes,
            "chunk_sizes": self.chunk_sizes.tolist(),
        }

    @classmethod
    def from_json(cls, name: str, data: dict[str, Any]) -> Encoding:
        logical = data["logical"]
        return cls(
            LogicalRows(name, int(logical["rows"]), int(logical["row_bytes"]), logical["dtype"], tuple(int(n) for n in logical["row_shape"])),
            data["codec"], dict(data["options"]), int(data["chunk_bytes"]), int(data["stored_row_bytes"]),
            np.asarray(data["chunk_sizes"], dtype=np.int64), tuple(logical["row_sha256"]),
        )


def chunk_offsets(chunk_sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where each chunk of each row starts in its stored row, and where each row's last chunk ends."""
    sizes = np.asarray(chunk_sizes, dtype=np.int64)
    aligned = (sizes + CHUNK_ALIGNMENT - 1) // CHUNK_ALIGNMENT * CHUNK_ALIGNMENT
    starts = np.cumsum(aligned, axis=1) - aligned
    return starts, starts[:, -1] + sizes[:, -1]


LARGE_ROW_BYTES = 1 << 20  # rows from this size share the pack's block (the native host cache's smallest default block)


def stored_sizes(needed: dict[str, int], logical: dict[str, int]) -> tuple[dict[str, int], int]:
    """Each segment's stored row size, and the pack's block (0 without large rows). A segment needs at least `needed` bytes
    per row (its largest encoded row). Large rows (logical rows of LARGE_ROW_BYTES or more) are stored in k_s blocks, k_s
    their logical row size over the largest size dividing every large logical row size (rows of 11 and 5.5 MiB: 2 and
    1), the block rounded up to ROW_ALIGNMENT: every large stored row is a whole number of blocks, and a cache whose block
    is the largest size dividing them splits none. Smaller rows are rounded up to ROW_ALIGNMENT alone."""
    large = [name for name in needed if logical[name] >= LARGE_ROW_BYTES]
    sizes = {name: round_up(needed[name], ROW_ALIGNMENT) for name in needed if name not in large}
    if not large:
        return sizes, 0
    common = 0
    for name in large:
        common = math.gcd(common, logical[name])
    multiples = {name: logical[name] // common for name in large}
    block = round_up(max(-(-needed[name] // multiples[name]) for name in large), ROW_ALIGNMENT)
    sizes.update({name: multiples[name] * block for name in large})
    return sizes, block


@dataclass
class EncodedPack:
    """An encoded pack: the stored rows (`pack`, a Weightsift pack) and how each segment's rows decode."""

    pack: Pack
    encodings: dict[str, Encoding]

    @property
    def metadata(self) -> dict[str, Any]:
        return self.pack.metadata

    @property
    def stored_ratio(self) -> float:
        """Stored bytes over the logical bytes they decode to (padding included)."""
        stored = sum(e.stored_row_bytes * e.logical.rows for e in self.encodings.values())
        return stored / sum(e.logical.nbytes for e in self.encodings.values())


def is_encoded(directory: str | Path) -> bool:
    manifest = json.loads((Path(directory) / MANIFEST).read_text(encoding="utf-8"))
    return manifest.get("metadata", {}).get("encoding", {}).get("format") == ENCODING_FORMAT


def open_encoded(directory: str | Path, verify: str = "size") -> EncodedPack:
    """Open an encoded pack (`verify` as `open_pack`'s: "size", "files" re-hashes its files, "segments" its segments)."""
    pack = open_pack(directory, verify=verify)
    encoding = pack.metadata.get("encoding", {})
    if encoding.get("format") != ENCODING_FORMAT or encoding.get("version") != ENCODING_VERSION:
        raise ValueError(f"{directory}: not an {ENCODING_FORMAT} v{ENCODING_VERSION} pack")
    encodings = {name: Encoding.from_json(name, data) for name, data in pack.metadata["encoded"].items()}
    for name, item in encodings.items():
        stored = pack.segments[name]
        if stored.rows != item.logical.rows or stored.row_bytes != item.stored_row_bytes:
            raise ValueError(f"{directory}: segment {name} does not hold its encoded rows")
    return EncodedPack(pack, encodings)
