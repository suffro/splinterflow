"""Phase 6B, stage 6B2: one decode experts call's rows delivered to the GPU as BF16, path by path (decision 0013).

    uv run python benchmarks/encoded_io.py --output experiments/phase6b/io/encoded-io.json [--calls 40]

What the brief asks to compare, measured on the data path alone (no model; nothing else may run on the machine):

  bf16             Phase 6A's path: the expert index's BF16 rows, read by the native core into pinned staging and copied
                   to the device (2.70 GB per decode token over PCIe)
  encoded-gpu      the encoded pack's rows (nvCOMP rANS chunks, 0.67 of the bytes), read and copied the same way, then
                   decoded by nvCOMP on the GPU straight into the device buffers (`RowDecoder`)
  5c-gpu           Phase 5C's best exact format (bit planes, a zstd-19 frame per plane and tensor), its frames copied to the
                   device and decompressed by nvCOMP's zstd, the planes merged on the device (PyTorch operations)
  5c-cpu           the same frames decompressed on the CPU (zstd's C threads) and merged (NumPy, threads) into pinned memory,
                   then the BF16 rows copied to the device: fewer drive bytes, the full BF16 bytes over PCIe

Calls are a decode call's rows: one layer's 6 experts, both segments (104 MB of BF16), drawn with a fixed seed, the same
calls for every path. bf16 and encoded-gpu are measured cold (no host cache: every byte from the drive, direct I/O) and
warm (every row a hit of the native host cache, the warm-up pass not measured); the 5C paths hold their frames in host
memory (as host-cache hits would be), their best case. Per path: milliseconds per call until the rows are on the device
(synchronized), GB/s of BF16 delivered, bytes from the drive and over PCIe per call; every delivered row is compared with
the index's bytes (equal or the script fails).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "research" / "expert_deltas"))

from awpmi.runtime import configure_reproducible_numerics  # noqa: E402

NUMERICS = configure_reproducible_numerics()

from expert_deltas import bits  # noqa: E402
from expert_deltas.codecs import Codec, _zstd_decompressor, compress_many  # noqa: E402

from awpmi.materialization.backend import MaterializationBackend  # noqa: E402
from awpmi.models.moe import groups_from_pack  # noqa: E402
from awpmi.storage.encoded import open_encoded  # noqa: E402
from awpmi.storage.pack import open_pack  # noqa: E402
from awpmi.streaming import nvcomp  # noqa: E402
from awpmi.streaming.codec import RowDecoder  # noqa: E402
from awpmi.streaming.streamer import PageStreamer  # noqa: E402
from awpmi.tracing import environment_metadata  # noqa: E402

THREADS = 6


def draw_calls(groups: dict, count: int, seed: int) -> list[tuple[str, list[str], torch.Tensor]]:
    """(group, its segments in the index's order, 6 experts ascending) per call."""
    generator = torch.Generator().manual_seed(seed)
    keys = sorted(groups)
    calls = []
    for _ in range(count):
        group = groups[keys[int(torch.randint(0, len(keys), (1,), generator=generator))]]
        experts = torch.randperm(group.experts, generator=generator)[:6].sort().values
        calls.append((group.key, list(group.segments.values()), experts))
    return calls


def deliver(materializer: MaterializationBackend, segments: list[str], experts: torch.Tensor, outputs: dict[str, torch.Tensor]) -> None:
    materializer.materialize_many([(segment, experts, outputs[segment]) for segment in segments])


def measure_store(name: str, materializer: MaterializationBackend, calls: list, expected: dict, warm: bool) -> dict:
    """Every call delivered into device buffers; time per call (the rows on the device, synchronized)."""
    device = materializer.device
    segments = sorted({s for _, segs, _ in calls for s in segs})
    row_bytes = {s: materializer.segment(s).row_bytes for s in segments}
    outputs = {s: torch.empty(6, row_bytes[s], dtype=torch.uint8, device=device) for s in segments}
    # A first pass checks every delivered row against the index (and, warm, admits every row into the host cache); the
    # timed pass then runs the calls back to back: the checks' host time would leave the GPU idle between calls, and an
    # idle GPU drops its PCIe link to a slower generation (measured: the first copies after a pause run at a third).
    for _, segs, experts in calls:
        deliver(materializer, segs, experts, outputs)
        for segment in segs:
            got = outputs[segment].cpu().numpy()
            for k, expert in enumerate(experts.tolist()):
                if hashlib.sha256(got[k].tobytes()).hexdigest() != expected[segment][expert]:
                    raise RuntimeError(f"{name}: {segment} row {expert} differs from the index")
    if not warm:
        materializer.store.clear_cache() if hasattr(materializer.store, "clear_cache") else None
    torch.cuda.synchronize()
    materializer.reset_stats()
    times = []
    for _, segs, experts in calls:
        torch.cuda.synchronize()
        started = time.perf_counter()
        deliver(materializer, segs, experts, outputs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1e3)
    report = materializer.report()
    bf16 = statistics.mean(6 * sum(row_bytes[s] for s in segs) for _, segs, _ in calls)
    cache = report["storage"].get("host_cache")
    return {
        "path": name, "warm": warm, "calls": len(calls), "ms_mean": statistics.mean(times), "ms_median": statistics.median(times),
        "ms_p95": sorted(times)[int(0.95 * (len(times) - 1))], "bf16_gb_s": bf16 / statistics.mean(times) / 1e6,
        "drive_bytes_per_call": report["storage"]["physical_bytes"] / len(calls), "h2d_bytes_per_call": report["transfer"]["h2d_bytes"] / len(calls),
        "host_cache_hits": None if cache is None else cache["hits"], "host_cache_misses": None if cache is None else cache["misses"],
        "decoder": report.get("decoder"),
    }


def measure_5c(calls: list, index_store, expected: dict, device: torch.device, repeats: int) -> list[dict]:
    """Phase 5C's bit planes (zstd-19, a frame per plane and tensor) of each call's rows: on the GPU (nvCOMP zstd + a device
    merge) and on the CPU (zstd's threads + NumPy, then the BF16 rows copied). Every call's frames are compressed first;
    each path then runs the calls back to back (no idle GPU between them), and every restored row is checked after."""
    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
    from gpu_codec_probe import merge_planes  # the probe's device merge (PyTorch operations)

    codec = Codec("zstd", 19)
    options = nvcomp.decompress_options("zstd", backend=0)
    prepared = []
    for _, segs, experts in calls[:repeats]:
        # Each row's tensors as Phase 5C stores them: the gate/up row is two tensors, the down row one.
        blocks = []  # (segment, slot, tensor index, count, planes)
        for segment in segs:
            rows = index_store.read_rows(segment, experts).numpy()
            halves = 2 if "gate_up" in segment.split(".")[-1] else 1
            for slot in range(rows.shape[0]):
                for half, part in enumerate(np.split(rows[slot].view(np.uint16), halves)):
                    blocks.append((segment, slot, half, part.size, list(bits.split_planes(part).values())))
        flat = [p for *_, planes in blocks for p in planes]
        frames = compress_many(flat, codec, None, THREADS)
        prepared.append((segs, experts, blocks, frames, [len(f) for f in flat]))
    stored = sum(sum(len(f) for f in frames) for *_, frames, _ in prepared)
    bf16_total = sum(2 * count for _, _, blocks, _, _ in prepared for *_, count, _ in blocks)
    times = {"5c-gpu": [], "5c-cpu": []}
    restored_gpu, restored_cpu = [], []
    # GPU: the frames copied from pinned memory, nvCOMP's zstd, the device merge.
    for segs, experts, blocks, frames, lengths in prepared:
        host_frames = [torch.from_numpy(np.frombuffer(f, dtype=np.uint8).copy()).pin_memory() for f in frames]
        device_frames = [torch.empty(len(f), dtype=torch.uint8, device=device) for f in frames]
        decoded = [torch.empty(n, dtype=torch.uint8, device=device) for n in lengths]
        table = torch.tensor([[f.data_ptr() for f in device_frames], [len(f) for f in frames], [d.data_ptr() for d in decoded], lengths],
                             dtype=torch.int64, device=device)
        scratch = torch.empty(max(nvcomp.decompress_temp_bytes("zstd", options, len(frames), max(lengths), sum(lengths)), 1), dtype=torch.uint8, device=device)
        actual = torch.empty(len(frames), dtype=torch.int64, device=device)
        statuses = torch.empty(len(frames), dtype=torch.int32, device=device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for h, d in zip(host_frames, device_frames):
            d.copy_(h, non_blocking=True)
        nvcomp.decompress("zstd", options, table, scratch, actual, statuses)
        merged, position = [], 0
        for *_, count, planes in blocks:
            merged.append(merge_planes(decoded[position : position + len(planes)], count))
            position += len(planes)
        torch.cuda.synchronize()
        times["5c-gpu"].append((time.perf_counter() - started) * 1e3)
        if int((statuses != 0).sum()) or not torch.equal(actual.cpu(), torch.tensor(lengths)):
            raise RuntimeError("5c-gpu: a frame failed to decode")
        restored_gpu.append([m.cpu().numpy().view(np.uint16) for m in merged])
    # CPU: zstd's threads, NumPy merges on threads into pinned memory, then the BF16 rows over PCIe.
    largest = max(sum(2 * count for *_, count, _ in blocks) for _, _, blocks, _, _ in prepared)
    pinned = torch.empty(largest, dtype=torch.uint8).pin_memory()
    target = torch.empty(largest, dtype=torch.uint8, device=device)
    view = pinned.numpy()
    with ThreadPoolExecutor(THREADS) as pool:
        for segs, experts, blocks, frames, lengths in prepared:
            starts = np.cumsum([0] + [len(planes) for *_, planes in blocks]).tolist()
            offsets = np.cumsum([0] + [2 * count for *_, count, _ in blocks]).tolist()
            torch.cuda.synchronize()
            started = time.perf_counter()
            unpacked = _zstd_decompressor(None).multi_decompress_to_buffer(frames, threads=THREADS)

            def merge(k: int) -> None:
                planes = dict(zip(bits.PLANES, (unpacked[i].tobytes() for i in range(starts[k], starts[k + 1]))))
                view[offsets[k] : offsets[k + 1]] = bits.merge_planes(planes, blocks[k][3]).view(np.uint8)

            list(pool.map(merge, range(len(blocks))))
            target[: offsets[-1]].copy_(pinned[: offsets[-1]], non_blocking=True)
            torch.cuda.synchronize()
            times["5c-cpu"].append((time.perf_counter() - started) * 1e3)
            restored_cpu.append([target[offsets[k] : offsets[k + 1]].cpu().numpy().view(np.uint16) for k in range(len(blocks))])
    # Both restorations against the index's bytes.
    for (segs, experts, blocks, _, _), gpu, cpu in zip(prepared, restored_gpu, restored_cpu):
        for k, (segment, slot, half, count, _) in enumerate(blocks):
            row = index_store.read_rows(segment, experts[slot : slot + 1]).numpy()[0].view(np.uint16)
            part = np.split(row, 2 if "gate_up" in segment.split(".")[-1] else 1)[half]
            if not (np.array_equal(gpu[k], part) and np.array_equal(cpu[k], part)):
                raise RuntimeError(f"5c: {segment} expert {int(experts[slot])} restored wrongly")
    out = []
    for name, values in times.items():
        out.append({
            "path": name, "warm": True, "calls": len(values), "ms_mean": statistics.mean(values), "ms_median": statistics.median(values),
            "bf16_gb_s": bf16_total / len(values) / statistics.mean(values) / 1e6, "stored_ratio": stored / bf16_total,
            "h2d_bytes_per_call": (stored if name == "5c-gpu" else bf16_total) / len(values), "drive_bytes_per_call": stored / len(values),
            "note": "frames in host memory (a host cache's best case); drive bytes are the frames' had they been read",
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase6b-gpu.yaml"))
    parser.add_argument("--calls", type=int, default=40)
    parser.add_argument("--5c-calls", dest="calls_5c", type=int, default=6, help="calls for the Phase 5C paths (zstd-19 compresses slowly)")
    parser.add_argument("--seed", type=int, default=6)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    storage = config["storage"]
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_per_process_memory_fraction(int(config["gpu_budget_bytes"]) / torch.cuda.get_device_properties(device).total_memory, device)
    index = open_pack(REPO_ROOT / config["index"]["directory"], verify="size")
    encoded = open_encoded(REPO_ROOT / config["encoded"]["directory"], verify="size")
    groups = groups_from_pack(index)
    calls = draw_calls(groups, args.calls, args.seed)
    options = dict(direct=True, alignment=int(storage["alignment"]), max_gap=int(storage["max_gap"]), workers=int(storage["workers"]),
                   max_read_bytes=int(storage["max_read_bytes"]), max_extent_bytes=int(storage["max_extent_bytes"]))
    expected = {}
    reader = index.store(backend="native", **options)
    try:
        for _, segs, experts in calls:
            for segment in segs:
                rows = reader.read_rows(segment, experts).numpy()
                for k, expert in enumerate(experts.tolist()):
                    expected.setdefault(segment, {})[expert] = hashlib.sha256(rows[k].tobytes()).hexdigest()
    finally:
        reader.close()
    results = []

    def streamer() -> PageStreamer:
        return PageStreamer(device, int(storage["slot_bytes"]), int(storage["slots"]), native_slots=int(storage["native_slots"]))

    everything = 12_000_000_000
    for name, source, decoder in (("bf16", index, None), ("encoded-gpu", encoded.pack, RowDecoder(encoded.encodings, device))):
        for warm in (False, True):
            store = source.store(backend="native", host_cache_bytes=everything if warm else 0, **options)
            try:
                results.append(measure_store(name, MaterializationBackend(store, device, streamer(), decoder=decoder), calls, expected, warm))
            finally:
                store.close()
            print(json.dumps(results[-1]), flush=True)
    index_store = index.store(backend="native", **options)
    try:
        for entry in measure_5c(calls, index_store, expected, device, args.calls_5c):
            results.append(entry)
            print(json.dumps(entry), flush=True)
    finally:
        index_store.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "environment": environment_metadata(REPO_ROOT, {"repository": config["model"]["repository"], "revision": config["model"]["revision"]}, NUMERICS),
        "nvcomp": nvcomp.version(), "encoding": {**encoded.metadata["encoding"], "stored_ratio": encoded.stored_ratio},
        "calls": [[key, segs, experts.tolist()] for key, segs, experts in calls], "results": results,
    }, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
