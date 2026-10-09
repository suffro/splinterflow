"""Phase 6B, stage 6B1: what a native transfer's submit costs with Moonlight's real routing and cache sizes (decision 0013).

    python experiments/phase6b/submit/submit_replay.py [--prompts 16] > experiments/phase6b/submit/replay-<build>.txt

No model and no GPU: the reference's routing (experiments/phase6a/baseline-run1, Phase 4B's 16 prompts, a prefill and 8
decode steps each) replayed through the native store over Moonlight's expert index, in the runtime's order (a decode call:
one transfer of its experts' gate/up rows then down rows; a prefill call over the 256 MiB budget: one transfer per
segment and chunk of 15 experts), with the admission freeze on prefills. Each transfer's pieces are consumed and released
as the streamer does (into pinned staging, no device copy). Per decode token: Python's time constructing the transfer
(what the profiles call `plan`), the engine's own time in submit (all, cache lookups and admissions, planning), and the
cache's counters. Nothing else may run on the machine.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]

from awpmi.storage import fileio  # noqa: E402
from awpmi.storage.pack import open_pack  # noqa: E402
from awpmi.tracing import read_jsonl  # noqa: E402

CALL_BUDGET = 268435456
SLOT = 32 << 20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--caches", type=float, nargs="+", default=[12e9, 4e9, 0])
    args = parser.parse_args()
    pack = open_pack(REPO / "packs" / "moonlight-16b-a3b-expert-index", verify="size")
    records = read_jsonl(REPO / "experiments" / "phase6a" / "baseline-run1" / "reference.jsonl.gz")
    prompts = sorted({r["prompt_id"] for r in records})[: args.prompts]
    by_key = {(r["prompt_id"], r["step"]): r for r in records}
    steps = max(r["step"] for r in records)
    layers = sorted({int(name.split(".")[2]) for name in pack.segments})
    segments = [(f"model.layers.{layer}.mlp.experts.gate_up_proj", f"model.layers.{layer}.mlp.experts.down_proj") for layer in layers]
    expert_row = pack.segments[segments[0][0]].row_bytes + pack.segments[segments[0][1]].row_bytes
    chunk = CALL_BUDGET // expert_row
    slots = [(fileio.aligned_host_buffer(SLOT, pin=True), fileio.aligned_host_buffer(SLOT, pin=True)) for _ in range(4)]
    for capacity in args.caches:
        store = pack.store(backend="native", direct=True, workers=8, host_cache_bytes=int(capacity))
        decode_python, decode_engine = [], []
        try:
            for prompt in prompts:
                for step in range(steps + 1):
                    if capacity:
                        store.set_admit(step > 0)
                    record = by_key[(prompt, step)]
                    before = store.stats.as_dict()["native"]
                    python_ms = 0.0
                    for (gate_up, down), experts in zip(segments, record["routed"]):
                        rows = torch.tensor(experts, dtype=torch.int64)
                        if len(experts) * expert_row > CALL_BUDGET:
                            transfers = [[(name, rows[first : first + chunk], None)] for name in (gate_up, down) for first in range(0, len(experts), chunk)]
                        else:
                            transfers = [[(gate_up, rows, None), (down, rows, None)]]
                        for requests in transfers:
                            started = time.perf_counter()
                            transfer = store.stream(requests, slots)
                            python_ms += (time.perf_counter() - started) * 1e3
                            with transfer as job:
                                for index, _, _ in job:
                                    job.release(index)
                    after = store.stats.as_dict()["native"]
                    if step > 0:
                        delta = {k: after[k] - before[k] for k in ("submit_ms", "submit_cache_ms", "submit_plan_ms", "submits")}
                        decode_python.append(python_ms)
                        decode_engine.append((delta["submit_ms"], delta["submit_cache_ms"], delta["submit_plan_ms"], delta["submits"]))
            cache = store.cache_stats()
            mean = statistics.mean
            print(
                f"cache {capacity / 1e9:5.1f} GB: per decode token, Python's submit {mean(decode_python):6.2f} ms "
                f"(median {statistics.median(decode_python):6.2f}); the engine's {mean(e[0] for e in decode_engine):6.2f} ms "
                f"(cache {mean(e[1] for e in decode_engine):6.2f}, plan {mean(e[2] for e in decode_engine):5.2f}) over "
                f"{mean(e[3] for e in decode_engine):.0f} submits"
                + ("" if cache is None else
                   f"; cache: hits {cache['hits']} misses {cache['misses']} evictions {cache['evictions']} recycled {cache['recycled']} "
                   f"allocated {cache['allocated_bytes'] / 1e9:.2f} GB released {cache['released_bytes'] / 1e9:.2f} GB "
                   f"block {cache['block_bytes']} held {cache['held_bytes'] / 1e9:.2f} GB"),
                flush=True,
            )
        finally:
            store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
