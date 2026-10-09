"""Encoded rows on the GPU: nvCOMP compresses them into an encoded pack and decompresses them where they are used
(Phase 6B, decision 0013).

`RowDecoder` turns stored rows that are already on the device (just copied there by the streamer, or held by a device
cache) into their logical bytes, written straight into the caller's buffer: one batched nvCOMP launch for every chunk of
every row of a call, on the current stream, after the copies it reads (the streamer makes the current stream wait for
them). The chunk table of each row (`awpmi.storage.encoded`) gives every chunk's address in its stored row; the pointers
are built on the host and copied with the launch. Nothing synchronizes: each chunk's status, and every decompressed size
against the expected one, are counted on the device, and `check()` (one synchronization, e.g. once per step) raises if
any chunk failed. nvCOMP detects little by itself (measured: a zeroed header decodes to nothing with a success status, a
corrupted payload to wrong bytes of the right size), so the stored bytes' integrity is established before decoding,
like the checkpoint's: the encoded pack's files are re-hashed when it is opened (`open_encoded(verify="files")`), every
row is decoded and compared with its source when the pack is written (`write_encoded_pack`), and the benchmarks decode
every row again and compare it with the reference's digests before inference. Output sizes always come from the chunk
table, never from the stored bytes.

`write_encoded_pack` writes an encoded pack from a source pack's segments: every row read through a store, compressed
on the GPU in chunks of `chunk_bytes` (nvCOMP's batched compressor), decompressed again on the GPU and compared with the
source row, its chunk sizes recorded; then, the pack's block fixed (`stored_sizes`), every row compressed again (nvCOMP is
deterministic: the sizes must not change) and written at its slot. Nothing is converted, quantized or rounded.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from awpmi.storage.encoded import (
    CHUNK_ALIGNMENT,
    ENCODING_FORMAT,
    ENCODING_VERSION,
    ROW_ALIGNMENT,
    EncodedPack,
    Encoding,
    LogicalRows,
    chunk_offsets,
    open_encoded,
    round_up,
    stored_sizes,
)
from awpmi.storage.pack import FORMAT, FORMAT_VERSION, MANIFEST, Pack
from awpmi.streaming import nvcomp
from awpmi.streaming.streamer import resolve_device


def _decompress_options(codec: str, options: Mapping[str, Any]):
    if codec == "ans":
        return nvcomp.decompress_options(codec, backend=0, data_type=nvcomp.DATA_TYPES[options.get("data_type", "char")])
    return nvcomp.decompress_options(codec, backend=0)


def _compress_options(codec: str, options: Mapping[str, Any]):
    if codec == "ans":
        return nvcomp.compress_options(codec, type=0, data_type=nvcomp.DATA_TYPES[options.get("data_type", "char")])
    if codec == "gdeflate":
        return nvcomp.compress_options(codec, algorithm=int(options.get("algorithm", 1)))
    return nvcomp.compress_options(codec)


@dataclass
class DecodeStats:
    launches: int = 0
    rows: int = 0
    chunks: int = 0
    stored_bytes: int = 0  # bytes of the stored rows decoded
    decoded_bytes: int = 0  # logical bytes written

    def reset(self) -> None:
        self.__init__()

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class RowDecoder:
    """Decodes encoded rows on the GPU (see the module docstring)."""

    def __init__(self, encodings: Mapping[str, Encoding], device: torch.device | str) -> None:
        self.encodings = dict(encodings)
        self.device = resolve_device(device)
        if self.device.type != "cuda":
            raise ValueError("encoded rows are decoded on a CUDA device")
        self._options = {}
        for item in self.encodings.values():
            key = (item.codec, json.dumps(item.options, sort_keys=True))
            if key not in self._options:
                self._options[key] = _decompress_options(item.codec, item.options)
        self._errors = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stats = DecodeStats()

    def __contains__(self, segment: str) -> bool:
        return segment in self.encodings

    def logical(self, segment: str) -> LogicalRows:
        return self.encodings[segment].logical

    def decode(self, items: list[tuple[str, torch.Tensor, np.ndarray, torch.Tensor]]) -> None:
        """Decode, on the current stream, each item (segment, rows, sources, out): `rows` (int64) the rows' indices in the
        segment, `sources` the device addresses of their stored rows (int64 [n]), `out` a contiguous uint8 device tensor
        [n, logical row bytes] receiving row i in out[i]. One launch per codec."""
        groups: dict[tuple[str, str], list] = {}
        for segment, rows, sources, out in items:
            item = self.encodings[segment]
            count = rows.numel()
            if out.dtype != torch.uint8 or tuple(out.shape) != (count, item.logical.row_bytes) or not out.is_contiguous():
                raise ValueError(f"out must be contiguous uint8 [{count}, {item.logical.row_bytes}]")
            if count == 0:
                continue
            indices = rows.reshape(-1).to("cpu", torch.int64).numpy()
            starts = np.asarray(sources, dtype=np.int64).reshape(-1, 1)
            outputs = out.data_ptr() + np.arange(count, dtype=np.int64)[:, None] * item.logical.row_bytes
            table = np.stack([
                (starts + item.offsets[indices]).reshape(-1),
                item.chunk_sizes[indices].reshape(-1),
                (outputs + np.arange(item.chunks, dtype=np.int64)[None, :] * item.chunk_bytes).reshape(-1),
                np.broadcast_to(item.lengths, (count, item.chunks)).reshape(-1),
            ])
            groups.setdefault((item.codec, json.dumps(item.options, sort_keys=True)), []).append((table, item, count, out))
            self.stats.rows += count
            self.stats.chunks += table.shape[1]
            self.stats.stored_bytes += count * item.stored_row_bytes
            self.stats.decoded_bytes += count * item.logical.row_bytes
        for (codec, options), parts in groups.items():
            table = np.concatenate([p[0] for p in parts], axis=1)
            chunks = table.shape[1]
            largest = max(p[1].chunk_bytes for p in parts)
            options_struct = self._options[(codec, options)]
            temp = nvcomp.decompress_temp_bytes(codec, options_struct, chunks, largest, int(table[3].sum()))
            # Pageable host memory: the copy returns once the table is staged, so the array may go at once.
            arrays = torch.from_numpy(np.ascontiguousarray(table)).to(self.device, non_blocking=True)
            scratch = torch.empty(max(temp, 1), dtype=torch.uint8, device=self.device)
            actual = torch.empty(chunks, dtype=torch.int64, device=self.device)
            statuses = torch.empty(chunks, dtype=torch.int32, device=self.device)
            nvcomp.decompress(codec, options_struct, arrays, scratch, actual, statuses)
            self._errors += (statuses != 0).sum() + (actual != arrays[3]).sum()
            self.stats.launches += 1
            for _, _, _, out in parts:
                out.record_stream(torch.cuda.current_stream(self.device))

    def check(self) -> None:
        """Raise if any chunk decoded since the last check failed (synchronizes)."""
        failed = int(self._errors.item())
        if failed:
            self._errors.zero_()
            raise RuntimeError(f"{failed} encoded chunks failed to decode")


def encode_rows(codec: str, options: Mapping[str, Any], rows: torch.Tensor, chunk_bytes: int) -> list[list[torch.Tensor]]:
    """Each row of `rows` (uint8 [n, row bytes] on the device) in chunks of `chunk_bytes`, each compressed alone: the
    compressed chunks per row. Decompressed again on the GPU and compared with the row (raises if it differs)."""
    count, row_bytes = rows.shape
    chunks = [rows[i, start : start + chunk_bytes] for i in range(count) for start in range(0, row_bytes, chunk_bytes)]
    per_row = len(chunks) // count
    packed = nvcomp.compress(codec, _compress_options(codec, options), chunks)
    # The round trip, on the GPU: every chunk decoded into a fresh buffer and compared with the source row.
    back = torch.empty_like(rows)
    table = np.array([
        [p.data_ptr() for p in packed],
        [p.numel() for p in packed],
        [back.data_ptr() + i * row_bytes + start for i in range(count) for start in range(0, row_bytes, chunk_bytes)],
        [c.numel() for c in chunks],
    ], dtype=np.int64)
    arrays = torch.from_numpy(table).to(rows.device)
    decompress = _decompress_options(codec, options)
    scratch = torch.empty(max(nvcomp.decompress_temp_bytes(codec, decompress, len(packed), chunk_bytes, int(table[3].sum())), 1), dtype=torch.uint8, device=rows.device)
    actual = torch.empty(len(packed), dtype=torch.int64, device=rows.device)
    statuses = torch.empty(len(packed), dtype=torch.int32, device=rows.device)
    nvcomp.decompress(codec, decompress, arrays, scratch, actual, statuses)
    if bool((statuses != 0).any()) or not torch.equal(actual, arrays[3]) or not torch.equal(back, rows):
        raise RuntimeError("a compressed row does not decode to its source bytes")
    return [packed[i * per_row : (i + 1) * per_row] for i in range(count)]


def _sha256_rows(rows: np.ndarray) -> list[str]:
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in rows]


def write_encoded_pack(
    source: Pack,
    directory: str | Path,
    codec: str = "ans",
    options: Mapping[str, Any] | None = None,
    chunk_bytes: int = 1 << 20,
    rows_per_batch: int = 8,
    store_options: Mapping[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> EncodedPack:
    """Encode every segment of `source` (rows read through its store) into an encoded pack at `directory` (see the module
    docstring). Every row is compressed on the GPU, checked to decode to its source bytes, and its sha256 recorded."""
    options = dict(options or {"data_type": "float16"})
    directory = Path(directory)
    if (directory / MANIFEST).exists():
        raise FileExistsError(f"{directory} already holds a pack")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    store = source.store(**(store_options or {"direct": True}))
    device = torch.device("cuda", torch.cuda.current_device())
    names = sorted(source.segments)
    groups = source.metadata.get("groups", {})
    group_of = {segment: key for key, group in groups.items() for segment in group["segments"].values()}
    hasher = ThreadPoolExecutor(4)

    def batches(name: str):
        info = source.segments[name]
        for first in range(0, info.rows, rows_per_batch):
            rows = torch.arange(first, min(first + rows_per_batch, info.rows))
            yield rows, store.read_rows(name, rows)

    try:
        # Pass 1: chunk sizes and digests, every row checked.
        sizes: dict[str, np.ndarray] = {}
        digests: dict[str, list[str]] = {}
        for name in names:
            tables, hashes = [], []
            for _, host in batches(name):
                hashes.append(hasher.submit(_sha256_rows, host.numpy().copy()))
                packed = encode_rows(codec, options, host.to(device), chunk_bytes)
                tables.extend([[p.numel() for p in row] for row in packed])
            sizes[name] = np.asarray(tables, dtype=np.int64)
            digests[name] = [digest for future in hashes for digest in future.result()]
            if progress:
                progress(f"measured {name}")
        needed = {name: int(chunk_offsets(sizes[name])[1].max()) for name in names}
        stored, block = stored_sizes(needed, {name: source.segments[name].row_bytes for name in names})
        # Pass 2: the rows written at their slots, one file per group of segments (e.g. one experts layer).
        files: dict[str, dict] = {}
        segments: dict[str, dict] = {}
        layout: dict[str, list[str]] = {}
        for name in names:
            layout.setdefault(group_of.get(name, name), []).append(name)
        for number, (group, members) in enumerate(sorted(layout.items())):
            key = f"rows-{number:03d}"
            path = directory / f"{key}.bin"
            file_hash = hashlib.sha256()
            offset = 0
            with open(path, "wb") as handle:
                for name in members:
                    info = source.segments[name]
                    segment_hash = hashlib.sha256()
                    row_index = 0
                    for _, host in batches(name):
                        packed = encode_rows(codec, options, host.to(device), chunk_bytes)
                        for row in packed:
                            if [p.numel() for p in row] != sizes[name][row_index].tolist():
                                raise RuntimeError(f"{name} row {row_index}: compressed sizes changed between passes")
                            data = np.zeros(stored[name], dtype=np.uint8)
                            for start, chunk in zip(chunk_offsets(sizes[name][row_index : row_index + 1])[0][0], row):
                                data[start : start + chunk.numel()] = chunk.cpu().numpy()
                            handle.write(data.tobytes())
                            file_hash.update(data.tobytes())
                            segment_hash.update(data.tobytes())
                            row_index += 1
                    segments[name] = {
                        "file": key, "offset": offset, "rows": info.rows, "row_bytes": stored[name], "dtype": "U8", "row_shape": [stored[name]],
                        "sha256": segment_hash.hexdigest(),
                    }
                    offset += info.rows * stored[name]
            files[key] = {"path": path.name, "bytes": path.stat().st_size, "sha256": file_hash.hexdigest()}
            if progress:
                progress(f"wrote {path.name} ({group})")
    finally:
        store.close()
        hasher.shutdown()
    encodings = {
        name: Encoding(
            LogicalRows(name, source.segments[name].rows, source.segments[name].row_bytes, source.segments[name].dtype, tuple(source.segments[name].row_shape)),
            codec, options, chunk_bytes, stored[name], sizes[name], tuple(digests[name]),
        )
        for name in names
    }
    source_manifest = (Path(source.directory) / MANIFEST).read_bytes()
    logical = sum(e.logical.nbytes for e in encodings.values())
    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "kind": source.kind,
        "files": files,
        "segments": segments,
        "metadata": {
            **{k: v for k, v in source.metadata.items() if k == "groups"},
            "encoding": {
                "format": ENCODING_FORMAT, "version": ENCODING_VERSION, "library": {"nvcomp": nvcomp.version()}, "codec": codec,
                "options": options, "chunk_bytes": chunk_bytes, "chunk_alignment": CHUNK_ALIGNMENT, "row_alignment": ROW_ALIGNMENT,
                "block_bytes": block,
            },
            "encoded": {name: encodings[name].to_json() for name in names},
            "source": {"directory": str(source.directory), "kind": source.kind, "manifest_sha256": hashlib.sha256(source_manifest).hexdigest()},
        },
        "packing": {
            "logical_bytes": logical,
            "stored_bytes": sum(e.stored_row_bytes * e.logical.rows for e in encodings.values()),
            "compressed_bytes": int(sum(sizes[name].sum() for name in names)),
            "elapsed_s": time.perf_counter() - started,
        },
    }
    manifest["packing"]["stored_ratio"] = manifest["packing"]["stored_bytes"] / logical
    manifest["packing"]["compressed_ratio"] = manifest["packing"]["compressed_bytes"] / logical
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return open_encoded(directory, verify="size")
