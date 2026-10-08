"""Phase 6A, stage B: the native I/O core measured on its own, before inference (decision 0012).

    uv run python benchmarks/native_io.py --output experiments/phase6a/io/io-cuda.json [--requests 100] [--cuda] [--sweep]

Moonlight's expert index on the drive (`configs/phase4b-moonlight.yaml`), read three ways, each in its own pass:

  python   Phase 4B's path: `FileBackedPageStore` planned and read by Python (8 threads of positioned direct reads),
           through a `PageStreamer`
  native   `NativePageStore`: planned and read by the Rust core through the same streamer; `--cuda` adds the copies
  native-hit  the same requests again with every row in the native host cache (the cache-to-slot copy path)

Request patterns (the routing does not matter for I/O; experts are drawn with a fixed seed):

  decode   one layer's 6 experts, both segments (the gate/up rows, then the down rows): what a decode token's
           experts call materializes, 26 times per token
  prefill  one layer's 15 experts of one segment (a chunk of the Phase 4B call budget)
  layer    one layer's 64 experts, both segments: a long sequential read (the drive's ceiling)

Per pattern and path: GB/s of rows delivered, requests per second, wall and main-thread CPU milliseconds per request
(the Python and FFI overhead: what the GIL-holding thread spends, waits excluded), read calls and physical bytes per
request, and the OS's read counters against the store's. Also: the planning cost alone (Python's `plan_reads` and
`_pieces` against the native planner) on the decode requests, and with --sweep the native readers' threads (4, 8, 16)
against their read-call size (1 and 4 MiB) on the decode and layer patterns. The bytes delivered are compared with the Python path's
on every pattern (equal or the script fails). Nothing is digested: these are timings, measured with nothing else
running; the drive is read with direct I/O, so the OS page cache plays no part.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

from awpmi.runtime import configure_reproducible_numerics  # noqa: E402

NUMERICS = configure_reproducible_numerics()

from awpmi.models.moe import groups_from_pack  # noqa: E402
from awpmi.storage.fileio import os_read_counters  # noqa: E402
from awpmi.storage.native import NATIVE_AVAILABLE  # noqa: E402
from awpmi.storage.pack import open_pack  # noqa: E402
from awpmi.storage.store import plan_reads  # noqa: E402
from awpmi.streaming.streamer import PageStreamer  # noqa: E402
from awpmi.tracing import environment_metadata  # noqa: E402


def requests_for(pattern: str, groups: dict, generator: torch.Generator, count: int) -> list[list[tuple[str, torch.Tensor]]]:
    """`count` requests, each a list of (segment, rows) to fetch together."""
    keys = sorted(groups)
    out = []
    for _ in range(count):
        group = groups[keys[int(torch.randint(0, len(keys), (1,), generator=generator))]]
        experts = group.experts
        if pattern == "decode":
            rows = torch.randperm(experts, generator=generator)[:6].sort().values
            out.append([(segment, rows) for segment in group.segments.values()])
        elif pattern == "prefill":
            rows = torch.randperm(experts, generator=generator)[:15].sort().values
            segment = list(group.segments.values())[int(torch.randint(0, 2, (1,), generator=generator))]
            out.append([(segment, rows)])
        else:
            rows = torch.arange(experts)
            out.append([(segment, rows) for segment in group.segments.values()])
    return out


_WEIGHTS: dict = {}


def row_checksums(tensor: torch.Tensor) -> list[int]:
    """Per row, the sum of its int64 words weighted by their position (wrapping): sensitive to any byte or row moved.

    Row by row, so that the temporaries stay one row's size.
    """
    rows, row_bytes = tensor.shape
    words = row_bytes // 8
    key = (words, tensor.device)
    if key not in _WEIGHTS:
        _WEIGHTS[key] = torch.arange(1, words + 1, dtype=torch.int64, device=tensor.device)
    flat = tensor.view(torch.int64).reshape(rows, words)
    return [int((flat[r] * _WEIGHTS[key]).sum()) for r in range(rows)]


def run(store, streamer: PageStreamer, requests: list, device: torch.device, digest: bool) -> dict:
    """Fetch every request into preallocated buffers (as an experts call's buffers); per request wall and main-thread
    CPU time (CUDA's own waits included); bytes delivered and their checksums."""
    walls, cpus = [], []
    delivered = 0
    checksum = []
    buffers: dict = {}

    def out_for(segment: str, rows: torch.Tensor) -> torch.Tensor:
        row_bytes = store.segment(segment).row_bytes
        key = (row_bytes, rows.numel())
        if key not in buffers:
            buffers[key] = torch.empty((rows.numel(), row_bytes), dtype=torch.uint8, device=device)
        return buffers[key]

    store.stats.reset()
    before = os_read_counters()
    for request in requests:
        outs = [out_for(segment, rows) for segment, rows in request]
        wall, cpu = time.perf_counter(), time.thread_time()
        tensors = streamer.fetch_many(store, [(segment, rows, out, None) for (segment, rows), out in zip(request, outs)])
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        walls.append((time.perf_counter() - wall) * 1e3)
        cpus.append((time.thread_time() - cpu) * 1e3)
        delivered += sum(t.numel() for t in tensors)
        if digest:
            checksum.append([row_checksums(t) for t in tensors])
    elapsed = sum(walls) / 1e3  # the requests' own time: the checksums are computed outside it
    after = os_read_counters()
    stats = store.stats.as_dict()
    result = {
        "requests": len(requests),
        "delivered_gb": delivered / 1e9,
        "gb_per_s": delivered / 1e9 / elapsed,
        "requests_per_s": len(requests) / elapsed,
        "wall_ms": {"mean": statistics.mean(walls), "median": statistics.median(walls), "p95": sorted(walls)[int(0.95 * (len(walls) - 1))]},
        "main_thread_cpu_ms": {"mean": statistics.mean(cpus), "median": statistics.median(cpus)},
        "physical_bytes_per_request": stats["physical_bytes"] / len(requests),
        "read_calls_per_request": stats["read_calls"] / len(requests),
        "extents_per_request": stats["extents"] / len(requests),
        "drive_gb_per_s_busy": stats["physical_bytes"] / 1e9 / max(1e-9, stats.get("native", {}).get("busy_ms", 0) / 1e3) if "native" in stats else None,
        "os_reads_equal_store": None if before is None else (after[0] - before[0], after[1] - before[1]) == (stats["read_calls"], stats["physical_bytes"]),
        "transfer": streamer.stats.as_dict(),
    }
    if "host_cache" in stats:
        result["host_cache"] = {k: stats["host_cache"][k] for k in ("lookups", "hits", "misses", "evictions", "resident_bytes")}
    return result, checksum


def planning(pack, requests: list, slot_bytes: int, native_store) -> dict:
    """Planning alone, per decode request (both segments): Python's plan_reads + _pieces against the native planner."""
    streamer = PageStreamer("cpu", slot_bytes=slot_bytes)
    python_ms, native_ms = [], []
    for request in requests:
        started = time.perf_counter()
        for segment, rows in request:
            streamer._pieces(plan_reads(pack.segments[segment], rows))
        python_ms.append((time.perf_counter() - started) * 1e3)
        started = time.perf_counter()
        for segment, rows in request:
            native_store.pieces(segment, rows, slot_bytes)
        native_ms.append((time.perf_counter() - started) * 1e3)
    return {"python_ms_per_request": statistics.mean(python_ms), "native_ms_per_request": statistics.mean(native_ms)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase6a-native.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--requests", type=int, default=150, help="requests per pattern and path")
    parser.add_argument("--cuda", action="store_true", help="deliver to the GPU (copies included); default: host memory")
    parser.add_argument("--settle", type=float, default=3.0)
    parser.add_argument("--sweep", action="store_true", help="also sweep the native readers' threads and read-call size")
    args = parser.parse_args()
    if not NATIVE_AVAILABLE:
        print("the native extension is not built", file=sys.stderr)
        return 1
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    storage = config["storage"]
    pack = open_pack(REPO_ROOT / config["index"]["directory"], verify="size")
    groups = groups_from_pack(pack)
    device = torch.device("cuda", torch.cuda.current_device()) if args.cuda else torch.device("cpu")
    options = dict(
        direct=True, alignment=int(storage["alignment"]), max_gap=int(storage["max_gap"]), workers=int(storage["workers"]),
        max_read_bytes=int(storage["max_read_bytes"]), max_extent_bytes=int(storage["max_extent_bytes"]),
    )
    slot_bytes, slots, native_slots = int(storage["slot_bytes"]), int(storage["slots"]), int(storage.get("native_slots", 4))
    time.sleep(args.settle)
    results: dict = {
        "device": str(device), "options": options, "slot_bytes": slot_bytes, "slots": slots, "native_slots": native_slots,
        "arguments": {key: value for key, value in vars(args).items()},
        "environment": environment_metadata(REPO_ROOT, {"index": config["index"]["directory"]}, NUMERICS),
        "patterns": {},
    }
    for pattern in ("decode", "prefill", "layer"):
        count = args.requests if pattern != "layer" else max(8, args.requests // 15)
        requests = requests_for(pattern, groups, torch.Generator().manual_seed(6), count)
        entry = {}
        checks = {}
        cache_bytes = sum(pack.segments[s].row_bytes * rows.numel() for request in requests for s, rows in request) + (1 << 30)
        for path in ("python", "native", "native-hit"):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            backend = "python" if path == "python" else "native"
            store = pack.store(backend=backend, host_cache_bytes=cache_bytes if path == "native-hit" else 0, **options) if backend == "native" else pack.store(**options)
            streamer = PageStreamer(device, slot_bytes, slots, native_slots=native_slots)
            try:
                if path == "native-hit":
                    run(store, streamer, requests, device, digest=False)  # fill the cache (not measured)
                entry[path], checks[path] = run(store, streamer, requests, device, digest=True)
            finally:
                store.close()
                streamer.close()
            time.sleep(args.settle)
        if not (checks["python"] == checks["native"] == checks["native-hit"]):
            raise SystemExit(f"{pattern}: the native path delivered different bytes")
        entry["bytes_equal"] = True
        if pattern == "decode":
            native_store = pack.store(backend="native", **options)
            try:
                entry["planning"] = planning(pack, requests, slot_bytes, native_store)
            finally:
                native_store.close()
        results["patterns"][pattern] = entry
        print(pattern, json.dumps({p: {k: entry[p][k] for k in ("gb_per_s", "wall_ms", "main_thread_cpu_ms")} for p in ("python", "native", "native-hit")}), flush=True)
    if args.sweep:
        # The native readers' two knobs: threads (reads in flight) and the size of one read call.
        results["sweep"] = []
        for pattern in ("decode", "layer"):
            count = args.requests if pattern != "layer" else max(8, args.requests // 15)
            requests = requests_for(pattern, groups, torch.Generator().manual_seed(7), count)
            for workers in (4, 8, 16):
                for max_read in (1 << 20, 4 << 20):
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    store = pack.store(backend="native", **{**options, "workers": workers, "max_read_bytes": max_read})
                    streamer = PageStreamer(device, slot_bytes, slots, native_slots=native_slots)
                    try:
                        measured, _ = run(store, streamer, requests, device, digest=False)
                    finally:
                        store.close()
                        streamer.close()
                    results["sweep"].append({"pattern": pattern, "workers": workers, "max_read_bytes": max_read, "gb_per_s": measured["gb_per_s"],
                                             "wall_ms": measured["wall_ms"], "read_calls_per_request": measured["read_calls_per_request"]})
                    print("sweep", pattern, workers, max_read, round(measured["gb_per_s"], 3), flush=True)
                    time.sleep(args.settle)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
