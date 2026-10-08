"""Compressing an expert's tensors in independently decodable blocks, and checking every reconstruction (Phase 5C).

A block is whole rows of one matrix ("row", "rows16", "rows64"), a matrix ("tensor"), or the expert's three tensors as
stored ("expert": down, gate, up, adjacent in the checkpoint). A block goes through an exact transform (`bits`: raw
bytes, byte split, bit planes: each plane its own frame) and a codec, one frame per stream. `measure` compresses every
block of every object, decodes every frame, restores the patterns, applies `finish` (for a delta: XOR or modular
restoration against its base), and compares the result with the reference patterns (the safetensors copy) bit for bit.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from expert_deltas import bits
from expert_deltas.codecs import Codec, Dictionary, compress_many, decompress_many
from expert_deltas.source import MATRICES

FILE_ORDER = ("down", "gate", "up")  # an expert's three tensors as stored (adjacent)
ROWS = {"row": 1, "rows16": 16, "rows64": 64}

Finish = Callable[[int, str, np.ndarray], np.ndarray]


def blocks(mats: dict[str, np.ndarray], kind: str) -> list[tuple[str, np.ndarray]]:
    """An expert's blocks of one kind, each a flat uint16 array."""
    if kind == "expert":
        return [("expert", np.concatenate([mats[m].reshape(-1) for m in FILE_ORDER]))]
    out = []
    for m in FILE_ORDER:
        a = mats[m]
        if kind == "tensor":
            out.append((m, a.reshape(-1)))
        else:
            step = ROWS[kind]
            out.extend((f"{m}:{r}", a[r : r + step].reshape(-1)) for r in range(0, a.shape[0], step))
    return out


def streams(p: np.ndarray, transform: str) -> list[bytes]:
    if transform == "raw":
        return [p.tobytes()]
    if transform == "byte_split":
        return [bits.byte_split(p)]
    if transform == "planes":
        return list(bits.split_planes(p).values())
    raise ValueError(transform)


def restore(parts: list[bytes], count: int, transform: str) -> np.ndarray:
    if transform == "raw":
        return np.frombuffer(parts[0], dtype=np.uint16).copy()
    if transform == "byte_split":
        return bits.byte_merge(parts[0], count)
    if transform == "planes":
        return bits.merge_planes(dict(zip(bits.PLANES, parts)), count)
    raise ValueError(transform)


def reassemble(decoded: dict[str, np.ndarray], shapes: dict[str, tuple[int, int]], kind: str) -> dict[str, np.ndarray]:
    if kind == "expert":
        flat, out, start = decoded["expert"], {}, 0
        for m in FILE_ORDER:
            n = shapes[m][0] * shapes[m][1]
            out[m] = flat[start : start + n].reshape(shapes[m])
            start += n
        return out
    if kind == "tensor":
        return {m: decoded[m].reshape(shapes[m]) for m in FILE_ORDER}
    step = ROWS[kind]
    return {m: np.concatenate([decoded[f"{m}:{r}"] for r in range(0, shapes[m][0], step)]).reshape(shapes[m]) for m in FILE_ORDER}


def decode_throughput(objects: dict[int, dict[str, np.ndarray]], kind: str, transform: str, codec: Codec, threads: int,
                      base_of: Callable[[int, str], np.ndarray | None] | None = None, repeats: int = 3, device=None) -> dict:
    """The costs a runtime pays per routed expert, timed separately (best of `repeats`; compressed once):

      decompress   every frame decoded: zstd's batch API on C threads (no Python copies), lz4 in a thread pool
      cpu_restore  each block's patterns restored and, for a XOR delta, undone against the matching rows of its base
                   (`base_of(e, matrix)`: the base's patterns, or None for an expert stored alone), numpy in a thread pool
      gpu_restore  (with `device`) the decoded streams copied host → device and restored there with PyTorch, one matrix
                   (or expert) at a time, XOR with a device-resident base; synchronized

    Throughputs in GB/s of BF16 weights out. The weights are not compared here (`measure` does; `gpu_restore` is tested
    against `restore`)."""
    from concurrent.futures import ThreadPoolExecutor

    import torch

    jobs = [(e, key, p.size, streams(p, transform)) for e, mats in objects.items() for key, p in blocks(mats, kind)]
    flat = [x for _, _, _, parts in jobs for x in parts]
    packed = compress_many(flat, codec, None, threads)
    lengths = [len(x) for x in flat]
    starts = np.cumsum([0] + [len(parts) for _, _, _, parts in jobs]).tolist()
    bf16 = sum(m.nbytes for mats in objects.values() for m in mats.values())
    shapes = {m: objects[next(iter(objects))][m].shape for m in MATRICES}
    unpacked = decompress_many(packed, lengths, codec, None, threads)
    timings: dict[str, float] = {}

    def best(name: str, run) -> None:
        for _ in range(repeats):
            started = time.perf_counter()
            run()
            elapsed = time.perf_counter() - started
            timings[name] = min(timings.get(name, elapsed), elapsed)

    if codec.name == "zstd":
        from expert_deltas.codecs import _zstd_decompressor

        best("decompress", lambda: _zstd_decompressor(None).multi_decompress_to_buffer(packed, threads=threads))
    else:
        best("decompress", lambda: decompress_many(packed, lengths, codec, None, threads))

    def cpu(index: int) -> None:
        e, key, count, _ = jobs[index]
        p = restore(unpacked[starts[index] : starts[index + 1]], count, transform)
        if base_of is None:
            return
        if kind == "expert":
            pieces = reassemble({"expert": p}, shapes, kind)
            for m in MATRICES:
                if base_of(e, m) is not None:
                    bits.xor_restore(pieces[m], base_of(e, m))
            return
        matrix = key.split(":")[0]
        base = base_of(e, matrix)
        if base is None:
            return
        rows = p.reshape(-1, shapes[matrix][1])
        first = 0 if kind == "tensor" else int(key.split(":")[1])
        bits.xor_restore(rows, base[first : first + rows.shape[0]])

    with ThreadPoolExecutor(threads) as pool:
        best("cpu_restore", lambda: list(pool.map(cpu, range(len(jobs)))))
    if device is not None:
        bases = {}
        if base_of is not None:
            for e in objects:
                for m in MATRICES:
                    base = base_of(e, m)
                    if base is not None:
                        bases[(e, m)] = torch.from_numpy(base.reshape(-1).view(np.int16)).to(device).to(torch.int32) & 0xFFFF
        groups: dict[tuple[int, str], list[int]] = {}
        for index, (e, key, _, _) in enumerate(jobs):
            groups.setdefault((e, "expert" if kind == "expert" else key.split(":")[0]), []).append(index)

        def run_gpu() -> None:
            for (e, name), indices in groups.items():
                if transform == "planes":  # a matrix's planes, block by block concatenated (whole bytes per block)
                    width = len(jobs[indices[0]][3])
                    parts = [b"".join(bytes(unpacked[starts[i] + k]) for i in indices) for k in range(width)]
                    out = gpu_restore(parts, sum(jobs[i][2] for i in indices), transform, device)
                else:
                    out = torch.cat([gpu_restore(unpacked[starts[i] : starts[i + 1]], jobs[i][2], transform, device) for i in indices])
                if name == "expert":
                    offset = 0
                    for m in FILE_ORDER:
                        n = shapes[m][0] * shapes[m][1]
                        if (e, m) in bases:
                            out[offset : offset + n] ^= bases[(e, m)]
                        offset += n
                elif (e, name) in bases:
                    out ^= bases[(e, name)]
            torch.cuda.synchronize(device)

        best("gpu_restore", run_gpu)
    return {"block": kind, "transform": transform, "codec": codec.to_json(), "bf16_bytes": bf16, "threads": threads,
            "delta": base_of is not None, "timings": timings,
            "throughput": {f"{k}_gb_s": bf16 / v / 1e9 for k, v in timings.items()}}


def gpu_restore(parts: list, count: int, transform: str, device):
    """`restore` on the device with PyTorch elementwise operations (the decoded streams copied host → device): int32
    patterns [count]. Equal to `restore` (tested)."""
    import torch

    def tensor(data) -> torch.Tensor:
        return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device)

    if transform == "raw":
        return tensor(parts[0]).view(torch.int16).to(torch.int32) & 0xFFFF
    if transform == "byte_split":
        both = tensor(parts[0]).to(torch.int32)
        return (both[:count] << 8) | both[count:]
    shifts = torch.arange(7, -1, -1, device=device, dtype=torch.int32)

    def unpack(data) -> torch.Tensor:
        return ((tensor(data).to(torch.int32)[:, None] >> shifts) & 1).reshape(-1)[:count]

    out = (unpack(parts[0]) << 15) | (tensor(parts[1]).to(torch.int32) << 7)
    for b, data in zip(range(6, -1, -1), parts[2:]):
        out |= unpack(data) << b
    return out


def measure(objects: dict[int, dict[str, np.ndarray]], reference: dict[int, dict[str, np.ndarray]], kind: str, transform: str,
            codec: Codec, threads: int, dictionary: Dictionary | None = None, finish: Finish | None = None) -> dict:
    """Every block of every object compressed alone, decoded, restored, finished and compared with `reference`.

    `objects` are what is stored per expert (the expert's patterns, or its delta); `finish(e, matrix, patterns)` turns a
    decoded object back into the expert's patterns (identity by default). Returns sizes (total and per object, in the
    order of `objects`), exactness, timings and throughputs (decompression; restoration and finishing, single thread).
    """
    jobs = []  # (object, block key, count, streams)
    for e, mats in objects.items():
        for key, p in blocks(mats, kind):
            jobs.append((e, key, p.size, streams(p, transform)))
    flat = [s for _, _, _, parts in jobs for s in parts]
    started = time.perf_counter()
    packed = compress_many(flat, codec, dictionary, threads)
    compress_s = time.perf_counter() - started
    lengths = [len(s) for s in flat]
    started = time.perf_counter()
    unpacked = decompress_many(packed, lengths, codec, dictionary, threads)
    decompress_s = time.perf_counter() - started
    first_object = next(iter(objects))
    first = sum(len(parts) for e, _, _, parts in jobs if e == first_object)
    started = time.perf_counter()
    decompress_many(packed[:first], lengths[:first], codec, dictionary, threads=1)
    decompress_1t_s = time.perf_counter() - started
    raw_first = sum(lengths[:first])
    position, decoded, stored = 0, {e: {} for e in objects}, {e: 0 for e in objects}
    started = time.perf_counter()
    for e, key, count, parts in jobs:
        decoded[e][key] = restore(unpacked[position : position + len(parts)], count, transform)
        stored[e] += sum(len(f) for f in packed[position : position + len(parts)])
        position += len(parts)
    shapes = {m: objects[first_object][m].shape for m in MATRICES}
    exact = True
    for e in objects:
        rebuilt = reassemble(decoded[e], shapes, kind)
        for m in MATRICES:
            weights = rebuilt[m] if finish is None else finish(e, m, rebuilt[m])
            exact &= bool(np.array_equal(weights, reference[e][m]))
    restore_s = time.perf_counter() - started
    bf16 = sum(m.nbytes for mats in objects.values() for m in mats.values())
    total = sum(stored.values())
    return {
        "block": kind, "transform": transform, "codec": codec.to_json(), "dictionary": None if dictionary is None else dictionary.nbytes,
        "blocks": len(jobs), "frames": len(flat), "bf16_bytes": bf16, "stream_bytes": sum(lengths), "stored_bytes": total,
        "ratio": total / bf16, "per_object_stored": [stored[e] for e in objects], "exact": bool(exact),
        "timings": {"compress_s": compress_s, "decompress_s": decompress_s, "decompress_1t_s": decompress_1t_s, "restore_and_check_s": restore_s},
        "throughput": {"decompress_gb_s": bf16 / decompress_s / 1e9, "decompress_1t_gb_s": raw_first / decompress_1t_s / 1e9,
                       "compress_mb_s": bf16 / compress_s / 1e6},
    }
