"""Phase 6B, stage 6B2-A: can the GPU restore an exact compressed representation of Moonlight's experts fast enough to
pay for the copies it saves, and with which existing library (decision 0013)?

    python -m uv run --group gpu python benchmarks/gpu_codec_probe.py --output experiments/phase6b/codec/probe.json

The question is economic. Copying a decode token's 2.70 GB of routed experts to this GPU takes about 440 ms (PCIe 3.0 x8,
6.2 GB/s); a representation at 0.66 of the bytes would save about 150 ms of copies per token, if the GPU restores the
BF16 rows in well under that. One decode experts call moves 6 experts: 104 MB of BF16, 16.8 ms of copies.

The corpus is Phase 5C's (layer 26, experts 0..7: its codec probe's), and layer 9's experts 0..7, as Weightsift's expert
index stores them: an expert's gate and up rows (one index row of 11.5 MB) and its down rows (5.8 MB). Every row is read
three ways and compared (the index through Weightsift's store; positioned reads and safetensors' own loader, Phase 5C's
two paths), from files checked against the publisher's sha256 by direct reads.

  A  Phase 5C's exact formats, as Phase 5C's code writes them (python-zstandard 0.25, libzstd 1.5.7): bit planes or byte
     splits, per tensor / per 16-row page / per expert, one zstd frame per stream, at levels 19 and 1. nvCOMP's zstd
     decoder decompresses the frames on the GPU as libzstd wrote them (no re-encoding); every decoded frame is compared
     with the CPU's decoding, and the BF16 rows restored on the GPU with the checkpoint's.
  B  Codecs nvCOMP compresses itself, on the GPU, in independent chunks of each row: ANS in its float16 mode on the raw
     BF16 bytes; ANS, zstd and GDeflate on the rows' high bytes with the low bytes stored raw (a byte split); LZ4,
     GDeflate, Bitcomp and zstd on the raw bytes. Every chunk is decoded and compared.
  C  The CPU path's cost: Phase 5C's zstd decompression on threads and its plane merge (NumPy), per decode call; the
     "decompress on the CPU, copy BF16" path saves drive bytes, not copies.

Per format: stored bytes over BF16 (every frame or chunk, plus an 8-byte offset each), exactness, and the GPU time to
restore one decode call's rows with the compressed bytes already on the device (CUDA events, best and median of the
repeats), split into decompression and merge. Nothing else may run on the machine while it runs.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "research" / "expert_deltas"))

from expert_deltas import bits  # noqa: E402
from expert_deltas.codecs import Codec, compress_many, decompress_many  # noqa: E402
from expert_deltas.compression import blocks, streams  # noqa: E402
from expert_deltas.source import Checkpoint, verify_files  # noqa: E402

from awpmi.runtime import configure_reproducible_numerics  # noqa: E402
from awpmi.storage.pack import open_pack  # noqa: E402
from awpmi.tracing import environment_metadata  # noqa: E402

REPOSITORY, REVISION = "moonshotai/Moonlight-16B-A3B", "476b36a473d4467f94469414bef6cee75c9c8172"
INDEX = REPO_ROOT / "packs" / "moonlight-16b-a3b-expert-index"
CALL_EXPERTS = 6  # a decode experts call routes 6 experts


# nvCOMP's batched low-level C API (nvcomp64_5.dll, from the nvidia-libnvcomp wheel), through ctypes.


class _Opts(ctypes.Structure):
    """Every nvCOMP option struct is 64 bytes; unused bytes must be zero."""


def _opts(name: str, fields: list) -> type:
    used = sum(ctypes.sizeof(kind) for _, kind in fields)
    return type(name, (_Opts,), {"_fields_": [*fields, ("reserved", ctypes.c_char * (64 - used))]})


_INT, _SIZE, _U8 = ctypes.c_int, ctypes.c_size_t, ctypes.c_uint8
OPTIONS = {  # codec: (C name, compress options, decompress options)
    "zstd": ("Zstd", _opts("ZstdC", []), _opts("ZstdD", [("backend", _INT)])),
    "ans": ("ANS", _opts("ANSC", [("type", _INT), ("data_type", _INT), ("max_sub_chunk_count", _U8)]),
            _opts("ANSD", [("backend", _INT), ("data_type", _INT), ("max_sub_chunk_count", _U8)])),
    "gdeflate": ("Gdeflate", _opts("GdeflateC", [("algorithm", _INT)]), _opts("GdeflateD", [("backend", _INT)])),
    "lz4": ("LZ4", _opts("LZ4C", [("data_type", _INT), ("bitshuffle_mode", _INT)]),
            _opts("LZ4D", [("backend", _INT), ("sort_before_hw_decompress", _INT), ("data_type", _INT), ("bitshuffle_mode", _INT)])),
    "bitcomp": ("Bitcomp", _opts("BitcompC", [("algorithm", _INT), ("data_type", _INT)]), _opts("BitcompD", [("backend", _INT)])),
}
TYPE_CHAR, TYPE_UCHAR, TYPE_USHORT, TYPE_FLOAT16 = 0, 1, 3, 9


class Alignments(ctypes.Structure):
    _fields_ = [("input", ctypes.c_size_t), ("output", ctypes.c_size_t), ("temp", ctypes.c_size_t)]


class Properties(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("cudart_version", ctypes.c_uint32)]


class Nvcomp:
    def __init__(self) -> None:
        import nvidia.libnvcomp as package

        base = Path(package.__file__).parent / "bin"
        os.add_dll_directory(str(base))
        self.path = base / "nvcomp64_5.dll"
        self.lib = ctypes.CDLL(str(self.path))
        properties = Properties()
        self._check(self.lib.nvcompGetProperties(ctypes.byref(properties)), "nvcompGetProperties")
        self.version = properties.version
        self.cudart_version = properties.cudart_version

    @staticmethod
    def _check(status: int, what: str) -> None:
        if status != 0:
            raise RuntimeError(f"{what}: nvcompStatus {status}")

    def _fn(self, codec: str, suffix: str):
        return getattr(self.lib, f"nvcompBatched{OPTIONS[codec][0]}{suffix}")

    def alignments(self, codec: str, decompress: bool, opts) -> Alignments:
        out = Alignments()
        self._check(self._fn(codec, ("Decompress" if decompress else "Compress") + "GetRequiredAlignments")(opts, ctypes.byref(out)), "alignments")
        return out

    def compress(self, codec: str, opts, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
        """Each device chunk (uint8) compressed independently; returns the compressed chunks (device, uint8)."""
        n, sizes = len(chunks), [c.numel() for c in chunks]
        largest, total = max(sizes), sum(sizes)
        temp, max_out = ctypes.c_size_t(), ctypes.c_size_t()
        self._check(self._fn(codec, "CompressGetTempSizeAsync")(_SIZE(n), _SIZE(largest), opts, ctypes.byref(temp), _SIZE(total)), "temp")
        self._check(self._fn(codec, "CompressGetMaxOutputChunkSize")(_SIZE(largest), opts, ctypes.byref(max_out)), "max out")
        stride = -(-max_out.value // 256) * 256
        out = torch.empty(n * stride, dtype=torch.uint8, device="cuda")
        in_ptrs = torch.tensor([c.data_ptr() for c in chunks], dtype=torch.int64, device="cuda")
        in_bytes = torch.tensor(sizes, dtype=torch.int64, device="cuda")
        out_ptrs = torch.tensor([out.data_ptr() + k * stride for k in range(n)], dtype=torch.int64, device="cuda")
        out_bytes = torch.zeros(n, dtype=torch.int64, device="cuda")
        statuses = torch.zeros(n, dtype=torch.int32, device="cuda")
        scratch = torch.empty(max(temp.value, 1), dtype=torch.uint8, device="cuda")
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self._check(self._fn(codec, "CompressAsync")(
            ctypes.c_void_p(in_ptrs.data_ptr()), ctypes.c_void_p(in_bytes.data_ptr()), _SIZE(largest), _SIZE(n),
            ctypes.c_void_p(scratch.data_ptr()), _SIZE(temp.value), ctypes.c_void_p(out_ptrs.data_ptr()),
            ctypes.c_void_p(out_bytes.data_ptr()), opts, ctypes.c_void_p(statuses.data_ptr()), stream), "compress")
        torch.cuda.synchronize()
        if bool((statuses != 0).any()):
            raise RuntimeError(f"compression statuses {statuses.unique().tolist()}")
        lengths = out_bytes.tolist()
        return [out[k * stride : k * stride + lengths[k]].clone() for k in range(n)]

    def decompressor(self, codec: str, opts, frames: list[torch.Tensor], outputs: list[torch.Tensor], temp_scale: float = 1.0):
        """A batched decompression of device `frames` into device `outputs` (uint8, their exact sizes), ready to launch:
        returns (launch, check). `launch()` enqueues it on the current stream; `check()` synchronizes and verifies the
        statuses and the decompressed sizes."""
        n = len(frames)
        sizes = [o.numel() for o in outputs]
        largest, total = max(sizes), sum(sizes)
        temp = ctypes.c_size_t()
        self._check(self._fn(codec, "DecompressGetTempSizeAsync")(_SIZE(n), _SIZE(largest), opts, ctypes.byref(temp), _SIZE(total)), "temp")
        scratch = torch.empty(max(int(temp.value * temp_scale), 1), dtype=torch.uint8, device="cuda")
        in_ptrs = torch.tensor([f.data_ptr() for f in frames], dtype=torch.int64, device="cuda")
        in_bytes = torch.tensor([f.numel() for f in frames], dtype=torch.int64, device="cuda")
        out_ptrs = torch.tensor([o.data_ptr() for o in outputs], dtype=torch.int64, device="cuda")
        out_capacity = torch.tensor(sizes, dtype=torch.int64, device="cuda")
        out_bytes = torch.zeros(n, dtype=torch.int64, device="cuda")
        statuses = torch.zeros(n, dtype=torch.int32, device="cuda")
        function = self._fn(codec, "DecompressAsync")
        keep = (scratch, in_ptrs, in_bytes, out_ptrs, out_capacity, out_bytes, statuses, frames, outputs)

        def launch() -> None:
            stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            self._check(function(
                ctypes.c_void_p(in_ptrs.data_ptr()), ctypes.c_void_p(in_bytes.data_ptr()), ctypes.c_void_p(out_capacity.data_ptr()),
                ctypes.c_void_p(out_bytes.data_ptr()), _SIZE(n), ctypes.c_void_p(scratch.data_ptr()), _SIZE(scratch.numel()),
                ctypes.c_void_p(out_ptrs.data_ptr()), opts, ctypes.c_void_p(statuses.data_ptr()), stream), "decompress")

        def check() -> None:
            torch.cuda.synchronize()
            bad = (statuses != 0).nonzero().flatten().tolist()
            if bad:
                raise RuntimeError(f"decompression statuses {[int(statuses[k]) for k in bad[:5]]} at chunks {bad[:5]}")
            if out_bytes.tolist() != sizes:
                raise RuntimeError("decompressed sizes differ")

        launch.keep = keep  # the device arrays live as long as the launcher
        return launch, check, temp.value


# GPU merges (PyTorch operations on the device)


_SHIFTS = None


def merge_planes(planes: list[torch.Tensor], count: int) -> torch.Tensor:
    """Phase 5C's `merge_planes` on the device: sign (packed bits), exponent (bytes), m6..m0 (packed bits) -> int16 patterns."""
    global _SHIFTS
    if _SHIFTS is None:
        _SHIFTS = torch.arange(7, -1, -1, dtype=torch.uint8, device="cuda")
    packed = torch.stack([planes[0], *planes[2:]])  # [8, count/8]: sign, m6..m0
    unpacked = ((packed[:, :, None] >> _SHIFTS) & 1).reshape(8, -1)[:, :count].to(torch.int32)
    weights = torch.tensor([15, 6, 5, 4, 3, 2, 1, 0], dtype=torch.int32, device="cuda")
    out = (unpacked << weights[:, None]).sum(0, dtype=torch.int32) | (planes[1][:count].to(torch.int32) << 7)
    return out.to(torch.int16)


def merge_bytes(high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    """A byte split back to little-endian 16-bit patterns: low byte, then high byte."""
    return torch.stack((low, high), dim=1).reshape(-1)


# The corpus


def load_rows(layers: list[int], experts: list[int], verify: bool) -> tuple[dict, dict]:
    """{(layer, expert): {"gate_up": uint8 [11534336], "down": uint8 [5767168], "mats": {gate, up, down: uint16 patterns}}},
    every row equal through three read paths; and the provenance."""
    checkpoint = Checkpoint(REPOSITORY, REVISION)
    pack = open_pack(INDEX, verify="size")
    store = pack.store(direct=True)
    files = sorted({checkpoint.tensor(layer, e, m).file for layer in layers for e in experts for m in ("gate", "up", "down")})
    provenance = {"files": verify_files(checkpoint, files) if verify else "not verified (development)"}
    if verify and not all(entry["equal"] for entry in provenance["files"].values()):
        raise RuntimeError(f"checkpoint files differ from the publisher's sha256: {provenance['files']}")
    rows: dict = {}
    try:
        for layer in layers:
            index_rows = {
                kind: store.read_rows(f"model.layers.{layer}.mlp.experts.{kind}_proj", torch.tensor(experts, dtype=torch.int64))
                for kind in ("gate_up", "down")
            }
            for position, e in enumerate(experts):
                mats = {}
                for m in ("gate", "up", "down"):
                    tensor = checkpoint.tensor(layer, e, m)
                    positioned = checkpoint.read_patterns(tensor)
                    reference = bits.patterns(checkpoint.read_reference(tensor))
                    if not np.array_equal(positioned, reference):
                        raise RuntimeError(f"layer {layer} expert {e} {m}: positioned read != safetensors")
                    mats[m] = reference
                gate_up = np.concatenate([mats["gate"].reshape(-1), mats["up"].reshape(-1)]).view(np.uint8)
                down = mats["down"].reshape(-1).view(np.uint8)
                if not (np.array_equal(index_rows["gate_up"][position].numpy(), gate_up) and np.array_equal(index_rows["down"][position].numpy(), down)):
                    raise RuntimeError(f"layer {layer} expert {e}: the index row differs from the checkpoint's tensors")
                rows[(layer, e)] = {"gate_up": gate_up, "down": down, "mats": mats}
    finally:
        store.close()
    provenance["rows_sha256"] = {f"{layer}/{e}": {k: hashlib.sha256(v[k].tobytes()).hexdigest() for k in ("gate_up", "down")} for (layer, e), v in rows.items()}
    return rows, provenance


def call_groups(rows: dict) -> list[list[tuple[int, int]]]:
    """The decode calls measured: per layer, its first 6 experts (one call) and its last 6 (another)."""
    layers = sorted({layer for layer, _ in rows})
    groups = []
    for layer in layers:
        experts = sorted(e for l, e in rows if l == layer)
        groups.append([(layer, e) for e in experts[:CALL_EXPERTS]])
        groups.append([(layer, e) for e in experts[-CALL_EXPERTS:]])
    return groups


def timed(launch, repeats: int) -> list[float]:
    """Device milliseconds of `launch()` (enqueued on the current stream), each repeat apart."""
    out = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end))
    return out


def h2d_ms(nbytes: int, repeats: int) -> float:
    """Best device time of one pinned host-to-device copy of `nbytes` (the link's cost of moving them)."""
    host = torch.empty(nbytes, dtype=torch.uint8).pin_memory()
    device = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    times = timed(lambda: device.copy_(host, non_blocking=True), repeats)
    del host, device
    return min(times)


def economics(entry: dict, call_bf16: float, raw_ms: float, repeats: int) -> None:
    """Per decode call: the copy of the format's bytes against the BF16 copy, and what restoring them costs."""
    stored = entry["stored_ratio"] * call_bf16
    entry["call"] = {"bf16_bytes": call_bf16, "stored_bytes": stored, "h2d_bf16_ms": raw_ms, "h2d_stored_ms": h2d_ms(int(stored), repeats)}
    call = entry["call"]
    call["copy_saved_ms"] = raw_ms - call["h2d_stored_ms"]
    call["net_saved_ms_best"] = call["copy_saved_ms"] - entry["restore"]["best_ms"]
    call["net_saved_ms_median"] = call["copy_saved_ms"] - entry["restore"]["median_ms"]


def summary(times: list[float], bf16: int) -> dict:
    best, median = min(times), statistics.median(times)
    return {"best_ms": best, "median_ms": median, "best_gb_s": bf16 / best / 1e6, "median_gb_s": bf16 / median / 1e6}


# A: Phase 5C's frames, decoded by nvCOMP's zstd


def probe_5c(nv: Nvcomp, rows: dict, groups: list, kind: str, transform: str, codec: Codec, repeats: int, threads: int, temp_scale: float) -> dict:
    """Phase 5C's blocks of every expert, each stream its own zstd frame (Phase 5C's code), decoded on the GPU."""
    result = {"format": f"5c/{kind}/{transform}/{codec.label}", "block": kind, "transform": transform, "codec": codec.to_json()}
    jobs = []  # (expert key, block key, count, streams)
    for key, entry in rows.items():
        for block_key, patterns in blocks(entry["mats"], kind):
            jobs.append((key, block_key, patterns.size, streams(patterns, transform)))
    flat = [s for *_, parts in jobs for s in parts]
    started = time.perf_counter()
    frames = compress_many(flat, codec, None, threads)
    result["compress_cpu_s"] = time.perf_counter() - started
    raw_lengths = [len(s) for s in flat]
    bf16 = sum(v[k].nbytes for v in rows.values() for k in ("gate_up", "down"))
    result["frames"] = len(frames)
    result["frame_bytes"] = sum(len(f) for f in frames)
    result["stored_ratio"] = (result["frame_bytes"] + 8 * len(frames)) / bf16
    # The CPU's own decoding of the same frames (python-zstandard), the reference for the GPU's.
    cpu = decompress_many(frames, raw_lengths, codec, None, threads)
    if [bytes(c) for c in cpu] != flat:
        raise RuntimeError("the CPU decoding differs from the streams")
    alignment = nv.alignments("zstd", True, OPTIONS["zstd"][2](backend=0))
    result["nvcomp_alignments"] = {"input": alignment.input, "output": alignment.output, "temp": alignment.temp}
    exact, timings, merges = True, [], []
    starts = np.cumsum([0] + [len(parts) for *_, parts in jobs]).tolist()
    for group in groups:
        members = [k for k, (key, *_rest) in enumerate(jobs) if key in group]
        indices = [i for k in members for i in range(starts[k], starts[k + 1])]
        device_frames = [torch.from_numpy(np.frombuffer(frames[i], dtype=np.uint8).copy()).cuda() for i in indices]
        outputs = [torch.empty(raw_lengths[i], dtype=torch.uint8, device="cuda") for i in indices]
        launch, check, _ = nv.decompressor("zstd", OPTIONS["zstd"][2](backend=0), device_frames, outputs, temp_scale)
        launch()
        check()
        for i, out in zip(indices, outputs):
            exact &= bytes(out.cpu().numpy()) == flat[i]
        timings.extend(timed(launch, repeats))
        # The merge on the device, block by block (its decoded streams are already there), then every block checked.
        decoded, position = {}, 0
        for k in members:
            decoded[k] = outputs[position : position + len(jobs[k][3])]
            position += len(jobs[k][3])
        restored = {}

        def merge_all() -> None:
            for k in members:
                count, parts = jobs[k][2], decoded[k]
                if transform == "planes":
                    restored[k] = merge_planes(parts, count)
                elif transform == "byte_split":
                    restored[k] = merge_bytes(parts[0][:count], parts[0][count:])
                else:
                    restored[k] = parts[0]

        merges.extend(timed(merge_all, repeats))
        for k in members:
            key, block_key, _, _ = jobs[k]
            patterns = dict(blocks(rows[key]["mats"], kind))[block_key]
            exact &= np.array_equal(restored[k].contiguous().view(torch.int16).cpu().numpy().view(np.uint16), patterns)
        del device_frames, outputs, launch, check, restored
        torch.cuda.empty_cache()
    call_bf16 = statistics.mean(sum(rows[k][r].nbytes for k in g for r in ("gate_up", "down")) for g in groups)
    result["exact"] = bool(exact)
    result["decompress"] = summary(timings, call_bf16)
    result["merge"] = summary(merges, call_bf16)
    result["restore"] = summary([a + b for a, b in zip(timings, merges)], call_bf16)
    return result


# B: nvCOMP's own codecs, chunks of each index row


def split_row(row: np.ndarray, layout: str) -> list[tuple[str, np.ndarray]]:
    """A row's streams: raw (as stored), or byte split (high bytes, low bytes)."""
    if layout == "raw":
        return [("raw", row)]
    pairs = row.view(np.uint16)
    return [("high", (pairs >> 8).astype(np.uint8)), ("low", (pairs & 0xFF).astype(np.uint8))]


def probe_nvcomp(nv: Nvcomp, rows: dict, groups: list, name: str, codec: str, copts, dopts, layout: str, compressed_streams: set,
                 chunk: int, repeats: int) -> dict:
    """Each index row's streams in independent chunks; `compressed_streams` are compressed with `codec`, the others stored raw."""
    result = {"format": name, "codec": codec, "layout": layout, "chunk_bytes": chunk, "compressed_streams": sorted(compressed_streams)}
    bf16 = sum(v[k].nbytes for v in rows.values() for k in ("gate_up", "down"))
    stored, chunks_total = 0, 0
    per_row = {}  # (expert key, row) -> list of (stream, raw chunk device tensors, compressed device tensors)
    started = time.perf_counter()
    for key, entry in rows.items():
        for kind in ("gate_up", "down"):
            parts = []
            for stream, data in split_row(entry[kind], layout):
                device = torch.from_numpy(data.copy()).cuda()
                raw_chunks = [device[o : o + chunk] for o in range(0, device.numel(), chunk)]
                if stream in compressed_streams:
                    packed = nv.compress(codec, copts, raw_chunks)
                    stored += sum(p.numel() for p in packed) + 8 * len(packed)
                    chunks_total += len(packed)
                else:
                    packed = None
                    stored += device.numel()
                parts.append((stream, raw_chunks, packed))
            per_row[(key, kind)] = parts
    torch.cuda.synchronize()
    result["compress_gpu_s"] = time.perf_counter() - started
    result["chunks"] = chunks_total
    result["stored_ratio"] = stored / bf16
    exact, timings, merges = True, [], []
    for group in groups:
        frames, outputs, expected = [], [], []
        for key in group:
            for kind in ("gate_up", "down"):
                for stream, raw_chunks, packed in per_row[(key, kind)]:
                    if packed is None:
                        continue
                    frames.extend(packed)
                    outputs.extend(torch.empty_like(c) for c in raw_chunks)
                    expected.extend(raw_chunks)
        launch, check, _ = nv.decompressor(codec, dopts, frames, outputs)
        launch()
        check()
        exact &= all(torch.equal(o, e) for o, e in zip(outputs, expected))
        timings.extend(timed(launch, repeats))
        if layout == "split":
            # Restore each row from its high bytes (decoded, chunk by chunk) and its low bytes (stored raw).
            decoded = {}
            position = 0
            for key in group:
                for kind in ("gate_up", "down"):
                    for stream, raw_chunks, packed in per_row[(key, kind)]:
                        if packed is not None:
                            decoded[(key, kind)] = outputs[position : position + len(packed)]
                            position += len(packed)
            low = {(key, kind): torch.cat(per_row[(key, kind)][1][1]) for key in group for kind in ("gate_up", "down")}
            high = {k: torch.cat(v) for k, v in decoded.items()}
            restored = {}

            def merge_all():
                for k in high:
                    restored[k] = merge_bytes(high[k], low[k])

            merges.extend(timed(merge_all, repeats))
            for (key, kind), value in restored.items():
                exact &= np.array_equal(value.cpu().numpy(), rows[key][kind])
        del frames, outputs, expected, launch, check
        torch.cuda.empty_cache()
    call_bf16 = statistics.mean(sum(rows[k][r].nbytes for k in g for r in ("gate_up", "down")) for g in groups)
    result["exact"] = bool(exact)
    result["decompress"] = summary(timings, call_bf16)
    if merges:
        result["merge"] = summary(merges, call_bf16)
        result["restore"] = summary([a + b for a, b in zip(timings, merges)], call_bf16)
    else:
        result["restore"] = result["decompress"]
    return result


# C: the CPU's decoding of a decode call


def probe_cpu(rows: dict, groups: list, kind: str, transform: str, codec: Codec, threads: int, repeats: int) -> dict:
    result = {"format": f"cpu/{kind}/{transform}/{codec.label}", "threads": threads}
    timings_decompress, timings_merge = [], []
    for group in groups:
        jobs = [(key, block_key, p.size, streams(p, transform)) for key in group for block_key, p in blocks(rows[key]["mats"], kind)]
        flat = [s for *_, parts in jobs for s in parts]
        frames = compress_many(flat, codec, None, threads)
        lengths = [len(s) for s in flat]
        starts = np.cumsum([0] + [len(parts) for *_, parts in jobs]).tolist()
        for _ in range(repeats):
            started = time.perf_counter()
            decoded = decompress_many(frames, lengths, codec, None, threads)
            middle = time.perf_counter()
            with ThreadPoolExecutor(threads) as pool:
                if transform == "planes":
                    merged = list(pool.map(lambda k: bits.merge_planes(dict(zip(bits.PLANES, decoded[starts[k] : starts[k + 1]])), jobs[k][2]), range(len(jobs))))
                else:
                    merged = list(pool.map(lambda k: bits.byte_merge(decoded[starts[k]], jobs[k][2]), range(len(jobs))))
            ended = time.perf_counter()
            timings_decompress.append((middle - started) * 1e3)
            timings_merge.append((ended - middle) * 1e3)
        for k, (key, block_key, count, _) in enumerate(jobs):
            if not np.array_equal(merged[k], dict(blocks(rows[key]["mats"], kind))[block_key]):
                raise RuntimeError("the CPU restoration is not exact")
    call_bf16 = statistics.mean(sum(rows[k][r].nbytes for k in g for r in ("gate_up", "down")) for g in groups)
    result["decompress"] = summary(timings_decompress, call_bf16)
    result["merge"] = summary(timings_merge, call_bf16)
    result["restore"] = summary([a + b for a, b in zip(timings_decompress, timings_merge)], call_bf16)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[26, 9])
    parser.add_argument("--experts", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--skip-verify", action="store_true", help="skip the files' sha256 (development; recorded)")
    parser.add_argument("--only", nargs="*", default=None, help="run only formats whose name contains one of these (development)")
    args = parser.parse_args()
    flags = configure_reproducible_numerics()
    torch.cuda.init()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    nv = Nvcomp()
    import zstandard

    result = {
        "stage": "6B2-A", "layers": args.layers, "experts": args.experts, "call_experts": CALL_EXPERTS, "repeats": args.repeats,
        "environment": environment_metadata(REPO_ROOT, {"repository": REPOSITORY, "revision": REVISION}, flags),
        "nvcomp": {"library": nv.path.name, "version": nv.version, "cudart_version": nv.cudart_version},
        "zstandard": {"python": zstandard.__version__, "libzstd": zstandard.ZSTD_VERSION},
        "gpu": torch.cuda.get_device_name(),
    }
    started = time.perf_counter()
    rows, provenance = load_rows(args.layers, args.experts, not args.skip_verify)
    result["corpus"] = provenance
    result["corpus"]["load_s"] = time.perf_counter() - started
    groups = call_groups(rows)
    result["calls"] = [[list(k) for k in g] for g in groups]
    call_bf16 = statistics.mean(sum(rows[k][r].nbytes for k in g for r in ("gate_up", "down")) for g in groups)
    raw_ms = h2d_ms(int(call_bf16), args.repeats)
    result["h2d_bf16_call_ms"] = raw_ms
    wanted = lambda name: args.only is None or any(token in name for token in args.only)  # noqa: E731
    formats = []
    # A: Phase 5C's formats (level 19 first: the recorded settings), decoded by nvCOMP's zstd; the temp buffer as nvCOMP
    # asks (and 1.5x, its documented workaround for levels >= 18 with libzstd 1.5.6, if the first fails).
    for kind, transform, level in [("tensor", "planes", 19), ("rows16", "planes", 19), ("expert", "planes", 19), ("tensor", "byte_split", 19),
                                   ("tensor", "planes", 1), ("tensor", "byte_split", 1)]:
        name = f"5c/{kind}/{transform}/zstd-{level}"
        if not wanted(name):
            continue
        for scale in (1.0, 1.5):
            try:
                entry = probe_5c(nv, rows, groups, kind, transform, Codec("zstd", level), args.repeats, args.threads, scale)
                entry["temp_scale"] = scale
                economics(entry, call_bf16, raw_ms, args.repeats)
                break
            except RuntimeError as error:
                entry = {"format": name, "error": str(error), "temp_scale": scale}
        formats.append(entry)
        print(json.dumps({k: entry.get(k) for k in ("format", "stored_ratio", "exact", "restore", "call", "error")}), flush=True)
    # B: nvCOMP's own codecs on the index rows.
    ans_c, ans_d = OPTIONS["ans"][1], OPTIONS["ans"][2]
    specs = [
        ("nvcomp/ans-float16/raw/64k", "ans", ans_c(type=0, data_type=TYPE_FLOAT16), ans_d(backend=0, data_type=TYPE_FLOAT16), "raw", {"raw"}, 64 << 10),
        ("nvcomp/ans-float16/raw/256k", "ans", ans_c(type=0, data_type=TYPE_FLOAT16), ans_d(backend=0, data_type=TYPE_FLOAT16), "raw", {"raw"}, 256 << 10),
        ("nvcomp/ans-float16/raw/1m", "ans", ans_c(type=0, data_type=TYPE_FLOAT16), ans_d(backend=0, data_type=TYPE_FLOAT16), "raw", {"raw"}, 1 << 20),
        ("nvcomp/ans-float16/raw/2m", "ans", ans_c(type=0, data_type=TYPE_FLOAT16), ans_d(backend=0, data_type=TYPE_FLOAT16), "raw", {"raw"}, 2 << 20),
        ("nvcomp/ans-float16/raw/4m", "ans", ans_c(type=0, data_type=TYPE_FLOAT16), ans_d(backend=0, data_type=TYPE_FLOAT16), "raw", {"raw"}, 4 << 20),
        ("nvcomp/ans-float16/raw/12m", "ans", ans_c(type=0, data_type=TYPE_FLOAT16), ans_d(backend=0, data_type=TYPE_FLOAT16), "raw", {"raw"}, 12 << 20),
        ("nvcomp/ans/split-high/64k", "ans", ans_c(type=0, data_type=TYPE_CHAR), ans_d(backend=0, data_type=TYPE_CHAR), "split", {"high"}, 64 << 10),
        ("nvcomp/ans/split-high/256k", "ans", ans_c(type=0, data_type=TYPE_CHAR), ans_d(backend=0, data_type=TYPE_CHAR), "split", {"high"}, 256 << 10),
        ("nvcomp/zstd/split-high/64k", "zstd", OPTIONS["zstd"][1](), OPTIONS["zstd"][2](backend=0), "split", {"high"}, 64 << 10),
        ("nvcomp/gdeflate-entropy/split-high/64k", "gdeflate", OPTIONS["gdeflate"][1](algorithm=0), OPTIONS["gdeflate"][2](backend=0), "split", {"high"}, 64 << 10),
        ("nvcomp/ans/raw/64k", "ans", ans_c(type=0, data_type=TYPE_CHAR), ans_d(backend=0, data_type=TYPE_CHAR), "raw", {"raw"}, 64 << 10),
        ("nvcomp/zstd/raw/64k", "zstd", OPTIONS["zstd"][1](), OPTIONS["zstd"][2](backend=0), "raw", {"raw"}, 64 << 10),
        ("nvcomp/gdeflate-entropy/raw/64k", "gdeflate", OPTIONS["gdeflate"][1](algorithm=0), OPTIONS["gdeflate"][2](backend=0), "raw", {"raw"}, 64 << 10),
        ("nvcomp/lz4/raw/64k", "lz4", OPTIONS["lz4"][1](data_type=TYPE_CHAR, bitshuffle_mode=0), OPTIONS["lz4"][2](backend=0), "raw", {"raw"}, 64 << 10),
        ("nvcomp/bitcomp/raw/64k", "bitcomp", OPTIONS["bitcomp"][1](algorithm=0, data_type=TYPE_USHORT), OPTIONS["bitcomp"][2](backend=0), "raw", {"raw"}, 64 << 10),
    ]
    for name, codec, copts, dopts, layout, compressed, chunk in specs:
        if not wanted(name):
            continue
        try:
            entry = probe_nvcomp(nv, rows, groups, name, codec, copts, dopts, layout, compressed, chunk, args.repeats)
            economics(entry, call_bf16, raw_ms, args.repeats)
        except RuntimeError as error:
            entry = {"format": name, "error": str(error)}
        formats.append(entry)
        print(json.dumps({k: entry.get(k) for k in ("format", "stored_ratio", "exact", "restore", "call", "error")}), flush=True)
    # C: the CPU's decoding of Phase 5C's best formats.
    for kind, transform, level in [("tensor", "planes", 19), ("tensor", "byte_split", 1)]:
        name = f"cpu/{kind}/{transform}/zstd-{level}"
        if wanted(name):
            entry = probe_cpu(rows, groups[:2], kind, transform, Codec("zstd", level), args.threads, max(3, args.repeats // 3))
            formats.append(entry)
            print(json.dumps({k: entry.get(k) for k in ("format", "restore")}), flush=True)
    result["formats"] = formats
    result["elapsed_s"] = time.perf_counter() - started
    output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
