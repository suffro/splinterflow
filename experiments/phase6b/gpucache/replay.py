"""Phase 6B, stage 6B3-A, before building anything: what a GPU expert cache in front of the host tier would keep.

    python experiments/phase6b/gpucache/replay.py > experiments/phase6b/gpucache/replay.txt

The routing traces of Phase 6A's stage A (Phase 4B's 16 prompts and Phase 5A's 48, every MoE layer at every step),
replayed in the streamed model's order (layer by layer; per call, the gate/up rows then the down rows; a prefill's call in
chunks of the 256 MiB budget) through a device cache of a few GB, the device memory the 6 GB cap leaves (the streamed
process peaks at 3.4 GB). Policies: LRU (the deterministic baseline) and Phase 4B's hotness (decayed counts, half-life
1,024 accesses), each admitting everything or nothing during prefills (the host tier's freeze). Rows are BF16 (17.3 MB an
expert) or compressed at the measured ratio (`--ratio`, the device holding the compressed rows and decoding them on use).
Per configuration: the share of a decode step's expert bytes the device cache serves (no copy over PCIe), steady state
(after the first prompt). Arithmetic on recorded routing; nothing touches the GPU or the drive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

from awpmi.storage.cache import HotnessPolicy, LRUPolicy, PageCache  # noqa: E402
from awpmi.tracing import read_jsonl  # noqa: E402

TRACES = {
    "phase4b": "experiments/phase4b/moonlight-run1/reference.jsonl.gz",
    "phase5a": "experiments/phase5a/oracle-run1/capture.jsonl.gz",
}
GATE_UP, DOWN = 11_534_336, 5_767_168
CALL_BUDGET = 268_435_456


class _Row:
    """A cache entry of a given size (PageCache counts `numel() * element_size()`)."""

    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes

    def numel(self) -> int:
        return self.nbytes

    def element_size(self) -> int:
        return 1


def replay(records: list[dict], capacity: int, policy: str, freeze: bool, ratio: float) -> dict:
    sizes = {"gate_up": int(GATE_UP * ratio), "down": int(DOWN * ratio)}
    rows = {name: _Row(n) for name, n in sizes.items()}
    cache = PageCache(capacity, HotnessPolicy(1024.0) if policy == "hotness" else LRUPolicy())
    served = requested = 0
    first_prompt = records[0]["prompt_id"]
    for record in records:
        decode = record["step"] > 0
        cache.admit = not (freeze and not decode)
        steady = decode and record["prompt_id"] != first_prompt
        for layer, experts in enumerate(record["routed"]):
            chunk = CALL_BUDGET // (GATE_UP + DOWN)
            calls = [experts] if len(experts) * (GATE_UP + DOWN) <= CALL_BUDGET else [experts[i : i + chunk] for i in range(0, len(experts), chunk)]
            for call in calls:
                for name in ("gate_up", "down"):
                    for expert in call:
                        key = ((layer, name), expert)
                        hit = cache.get(key, sizes[name]) is not None
                        if steady:
                            requested += sizes[name]
                            served += sizes[name] * hit
                        if not hit and cache.can_admit(sizes[name]):
                            cache.put(key, rows[name])
    return {"served_fraction": served / requested if requested else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=0.67, help="compressed row size over BF16 (from the codec probe)")
    parser.add_argument("--capacities", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    args = parser.parse_args()
    for trace, path in TRACES.items():
        records = sorted((r for r in read_jsonl(REPO / path) if "routed" in r), key=lambda r: (r["prompt_id"], r["step"]))
        print(f"{trace}: {len({r['prompt_id'] for r in records})} prompts, {len(records)} steps", flush=True)
        for ratio, label in ((1.0, "bf16"), (args.ratio, f"compressed {args.ratio:.3f}")):
            for policy in ("lru", "hotness"):
                for freeze in (False, True):
                    cells = []
                    for gb in args.capacities:
                        result = replay(records, int(gb * 1e9), policy, freeze, ratio)
                        cells.append(f"{gb:.1f} GB {result['served_fraction']:.3f}")
                    print(f"  {label:17s} {policy:7s} {'freeze' if freeze else 'admit '}: " + "  ".join(cells), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
