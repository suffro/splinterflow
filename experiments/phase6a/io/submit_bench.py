"""Phase 6A, a follow-up measurement: where a native transfer's submit spends its time with a host cache (decision 0012).

    uv run python experiments/phase6a/io/submit_bench.py > experiments/phase6a/io/submit.txt

The profiles show 56-105 ms per decode token in `Engine.submit` with a host cache, against 3 ms without one. Here, on
synthetic data shaped like Moonlight's experts (64 rows of 11.5 MB and 64 of 5.8 MB in a temporary file, read with
direct I/O), decode-shaped transfers (6 rows of each segment) are submitted to the native store, and submit's own wall
time is measured apart from the transfer:

  filling, hits   a cache holding everything: while it fills, rows miss (and are admitted) or hit; then every row hits;
                  no eviction
  evicting        a cache of half the rows: about half the rows miss and evict others; with rows of one size every
                  eviction hands its memory to the new row ("recycled"), with two sizes about half cannot
  workers         the readers' thread count (a scheduling effect would depend on it)

Nothing else may run on the machine. Main-thread CPU is Windows' coarse thread clock (15.6 ms ticks), per transfer.
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

from awpmi.storage import fileio
from awpmi.storage.layout import Segment
from awpmi.storage.native import NativePageStore

ROWS = 64
GATE_UP, DOWN = 11_534_336, 5_767_168  # Moonlight's composed rows: an expert's gate and up, and its down
SLOT = 32 << 20


def main() -> int:
    directory = Path(tempfile.mkdtemp())
    path = directory / "experts.bin"
    with open(path, "wb") as handle:
        chunk = os.urandom(1 << 20)
        for _ in range((ROWS * (GATE_UP + DOWN)) // len(chunk) + 1):
            handle.write(chunk)
    segments = {
        "gate_up": Segment("gate_up", "f", 0, ROWS, GATE_UP, "U8", (GATE_UP,)),
        "down": Segment("down", "f", ROWS * GATE_UP, ROWS, DOWN, "U8", (DOWN,)),
    }
    slots = [(fileio.aligned_host_buffer(SLOT, pin=False), fileio.aligned_host_buffer(SLOT, pin=False)) for _ in range(4)]
    generator = torch.Generator().manual_seed(1)

    def run(store, names, count, label):
        submit, total = [], []
        before = store.cache_stats()
        cpu_started = time.thread_time()
        for _ in range(count):
            experts = torch.randperm(ROWS, generator=generator)[:6].sort().values
            started = time.perf_counter()
            transfer = store.stream([(name, experts, None) for name in names], slots)
            submit.append((time.perf_counter() - started) * 1e3)
            with transfer as job:
                for index, _, _ in job:
                    job.release(index)
            total.append((time.perf_counter() - started) * 1e3)
        cpu = (time.thread_time() - cpu_started) * 1e3 / count
        after = store.cache_stats()
        per = lambda key: (after[key] - before[key]) / count  # noqa: E731
        print(
            f"{label:44s} submit {statistics.mean(submit):6.3f} ms (median {statistics.median(submit):6.3f}); transfer {statistics.mean(total):6.2f} ms; "
            f"per transfer: hits {per('hits'):4.1f} misses {per('misses'):4.1f} evictions {per('evictions'):4.1f} recycled {per('recycled'):4.1f}; "
            f"main-thread CPU {cpu:5.2f} ms",
            flush=True,
        )

    both = ["gate_up", "down"]
    everything = ROWS * (GATE_UP + DOWN) + (64 << 20)
    for workers in (8, 2):
        store = NativePageStore({"f": path}, segments, direct=True, workers=workers, host_cache_bytes=everything)
        try:
            run(store, both, 40, f"workers {workers}, cache of everything: filling")
            run(store, both, 100, f"workers {workers}, cache of everything: hits")
        finally:
            store.close()
    for workers in (8, 2):
        for names, capacity, label in (
            (["gate_up"], ROWS * GATE_UP // 2, "one row size"),
            (both, ROWS * (GATE_UP + DOWN) // 2, "two row sizes"),
        ):
            store = NativePageStore({"f": path}, segments, direct=True, workers=workers, host_cache_bytes=capacity)
            try:
                run(store, names, 60, f"workers {workers}, half cache, {label}: warm-up")
                run(store, names, 100, f"workers {workers}, half cache, {label}: evicting")
            finally:
                store.close()
    path.unlink()
    directory.rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main())
