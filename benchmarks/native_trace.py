"""Phase 6A, stage A: how many expert bytes Moonlight reads again, and what a host-RAM expert tier could keep (decision 0012).

    uv run python benchmarks/native_trace.py --output experiments/phase6a/trace

Reads routing traces recorded by earlier runs (the experts routed in every MoE layer at every step):

  phase4b  experiments/phase4b/moonlight-run1/reference.jsonl.gz   16 prompts x (prefill + 8 decode steps)
  phase5a  experiments/phase5a/oracle-run1/capture.jsonl.gz        48 prompts x (prefill + 16 decode steps)

and replays each in the order the streamed model requests expert rows (`moonlight_report.replay_cache`'s order: layer by
layer; per experts call, each expert-sliced segment in the index's order, one request per chunk of the call budget, the
experts ascending; prompts in order, one process). Every row request is an access of `row_bytes` bytes.

For every access the LRU stack distance in bytes (the bytes of the distinct rows requested since the row's previous
request) comes from a Fenwick tree over access times. An LRU cache of C bytes that admits every miss always holds the
longest prefix of the recency stack that fits in C, so an access hits iff its distance plus its own size is at most C:
one pass gives the exact LRU hit rate at every capacity. It is checked against a replay through `PageCache` at a few
capacities (equal hit counts on every step, or the script fails).

Also: bytes requested, compulsory (first-ever) and repeated, per phase; how much of a decode step's bytes were requested
by the previous step of the same layer, earlier in the prompt, or by an earlier prompt; and the previous-step predictor
(prefetch a layer's previous-step experts): its precision, and the share of an LRU cache's misses it would have covered.

Nothing here touches the GPU or the drive: it is arithmetic on recorded routing. Writes trace.json and trace.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from awpmi.storage.cache import LRUPolicy, PageCache
from awpmi.tracing import read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACES = {
    "phase4b": ("experiments/phase4b/moonlight-run1/reference.jsonl.gz", "experiments/phase4b/moonlight-run1/prompts.jsonl"),
    "phase5a": ("experiments/phase5a/oracle-run1/capture.jsonl.gz", "experiments/phase5a/oracle-run1/prompts.jsonl"),
}
CAPACITIES_GB = (0.5, 1, 2, 2.77, 4, 5.54, 6, 8, 10, 11.07, 12, 14, 16, 16.61, 20, 21.6, 24, 28.79)
CHECKED_CAPACITIES_GB = (2.77, 11.07, 16.61)


class Fenwick:
    """Prefix sums over access times (int64)."""

    def __init__(self, size: int) -> None:
        self.tree = np.zeros(size + 1, dtype=np.int64)

    def add(self, index: int, value: int) -> None:
        index += 1
        tree = self.tree
        while index < tree.size:
            tree[index] += value
            index += index & -index

    def prefix(self, index: int) -> int:
        """Sum over times [0, index)."""
        total = 0
        tree = self.tree
        while index > 0:
            total += int(tree[index])
            index -= index & -index
        return total


def accesses(steps: list[dict], segments: list[tuple[str, int]], call_budget: int | None) -> list[tuple]:
    """(step index, phase, layer, segment index, expert, bytes) in the backend's request order."""
    expert_row = sum(nbytes for _, nbytes in segments)
    out = []
    for index, record in enumerate(steps):
        for layer, experts in enumerate(record["routed"]):
            chunked = call_budget is not None and len(experts) * expert_row > call_budget
            size = call_budget // expert_row if chunked else max(1, len(experts))
            for segment, (_, nbytes) in enumerate(segments):
                for first in range(0, len(experts), size):
                    for expert in experts[first : first + size]:
                        out.append((index, record["phase"], layer, segment, expert, nbytes))
    return out


def stack_distances(stream: list[tuple]) -> np.ndarray:
    """Byte LRU stack distance of every access (-1 for a first access): distinct bytes requested since the previous one."""
    tree = Fenwick(len(stream))
    last: dict[tuple, int] = {}
    distances = np.full(len(stream), -1, dtype=np.int64)
    total = 0  # sum of the sizes at every row's latest access time
    for time, (_, _, layer, segment, expert, nbytes) in enumerate(stream):
        key = (layer, segment, expert)
        previous = last.get(key)
        if previous is not None:
            # Distinct rows accessed in (previous, time): each counted once, at its latest access.
            distances[time] = total - tree.prefix(previous + 1)
            tree.add(previous, -nbytes)
            total -= nbytes
        tree.add(time, nbytes)
        total += nbytes
        last[key] = time
    return distances


def replay(stream: list[tuple], capacity: int, steps: int) -> list[tuple[int, int]]:
    """(hits, misses) per step of an LRU `PageCache` of `capacity` bytes admitting every miss."""
    cache = PageCache(capacity, LRUPolicy())
    per_step = [[0, 0] for _ in range(steps)]
    rows: dict[int, object] = {}
    for index, _, layer, segment, expert, nbytes in stream:
        key = ((layer, segment), expert)
        if cache.get(key, nbytes) is None:
            per_step[index][1] += 1
            cache.put(key, rows.setdefault(nbytes, _Row(nbytes)))
        else:
            per_step[index][0] += 1
    return [tuple(x) for x in per_step]


class _Row:
    """A stand-in cache entry of `nbytes` (PageCache only asks numel and element_size)."""

    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes

    def numel(self) -> int:
        return self.nbytes

    def element_size(self) -> int:
        return 1


def analyze(name: str, steps: list[dict], prompts: list[dict], segments: list[tuple[str, int]], call_budget: int | None) -> dict:
    stream = accesses(steps, segments, call_budget)
    distances = stack_distances(stream)
    sizes = np.array([a[5] for a in stream], dtype=np.int64)
    step_of = np.array([a[0] for a in stream], dtype=np.int64)
    decode = np.array([a[1] == "decode" for a in stream])
    first_prompt = steps[0]["prompt_id"]
    prompt_of = np.array([steps[a[0]]["prompt_id"] for a in stream])
    steady = prompt_of != first_prompt
    phases = {"decode": decode, "prefill": ~decode}
    decode_steps = sum(1 for s in steps if s["phase"] == "decode")
    total_routed_bytes = len(steps[0]["routed"]) * 64 * sum(n for _, n in segments)

    # Bytes requested, compulsory and repeated; where a decode step's rows were last requested.
    requested = {phase: int(sizes[mask].sum()) for phase, mask in phases.items()}
    compulsory = {phase: int(sizes[mask & (distances < 0)].sum()) for phase, mask in phases.items()}
    origin = {"previous_step_same_layer": 0, "earlier_in_prompt": 0, "earlier_prompt": 0, "never": 0}
    last_seen: dict[tuple, tuple[int, int]] = {}  # row → (step index, prompt id) of its latest request
    for index, phase, layer, segment, expert, nbytes in stream:
        key = (layer, segment, expert)
        seen = last_seen.get(key)
        if phase == "decode":
            if seen is None:
                origin["never"] += nbytes
            elif seen[0] == index - 1 and seen[1] == steps[index]["prompt_id"]:
                origin["previous_step_same_layer"] += nbytes
            elif seen[1] == steps[index]["prompt_id"]:
                origin["earlier_in_prompt"] += nbytes
            else:
                origin["earlier_prompt"] += nbytes
        last_seen[key] = (index, steps[index]["prompt_id"])

    # Exact LRU hit rates at every capacity (stack distances), cold (every step) and steady (after the first prompt).
    curve = []
    for gb in CAPACITIES_GB:
        capacity = int(gb * 1e9)
        hit = (distances >= 0) & (distances + sizes <= capacity)
        entry = {"capacity_gb": gb, "fraction_of_experts": capacity / total_routed_bytes}
        for phase, mask in phases.items():
            for label, scope in (("cold", mask), ("steady", mask & steady)):
                entry[f"{phase}_{label}_hit_rate"] = float(hit[scope].sum() / max(1, scope.sum()))
                entry[f"{phase}_{label}_byte_hit_rate"] = float(sizes[scope & hit].sum() / max(1, sizes[scope].sum()))
        misses = sizes[decode & ~hit].sum()
        entry["decode_drive_gb_per_token"] = float(misses / max(1, decode_steps) / 1e9)
        curve.append(entry)

    # The check: a PageCache replay must give the same hits and misses on every step.
    checked = {}
    for gb in CHECKED_CAPACITIES_GB:
        capacity = int(gb * 1e9)
        hit = (distances >= 0) & (distances + sizes <= capacity)
        predicted = [(int(hit[step_of == k].sum()), int((~hit[step_of == k]).sum())) for k in range(len(steps))]
        measured = replay(stream, capacity, len(steps))
        checked[str(gb)] = predicted == measured
        if predicted != measured:
            raise SystemExit(f"{name}: stack distances disagree with the PageCache replay at {gb} GB")

    # The previous-step predictor (decode): prefetch a layer's previous-step experts.
    predictor = {"predicted": 0, "useful": 0, "actual": 0}
    previous: dict[int, set] = {}
    previous_prompt = None
    for record in steps:
        if record["prompt_id"] != previous_prompt:
            previous, previous_prompt = {}, record["prompt_id"]
        for layer, experts in enumerate(record["routed"]):
            current = set(experts)
            if record["phase"] == "decode" and layer in previous:
                predictor["predicted"] += len(previous[layer])
                predictor["useful"] += len(previous[layer] & current)
                predictor["actual"] += len(current)
            previous[layer] = current
    predictor["precision"] = predictor["useful"] / max(1, predictor["predicted"])
    predictor["recall"] = predictor["useful"] / max(1, predictor["actual"])
    # Of the misses of an LRU cache (decode, steady), the share a previous-step prefetch would have turned into hits.
    covered = []
    for gb in (0.0, 2.77, 11.07, 16.61):
        capacity = int(gb * 1e9)
        hit = (distances >= 0) & (distances + sizes <= capacity)
        miss = decode & steady & ~hit
        # A miss is covered when its row was requested by the previous step of the same layer and prompt.
        prior = np.zeros(len(stream), dtype=bool)
        seen: dict[tuple, int] = {}
        for time, (index, _, layer, segment, expert, _) in enumerate(stream):
            key = (layer, segment, expert)
            if key in seen and seen[key] == index - 1 and steps[index]["prompt_id"] == steps[index - 1]["prompt_id"]:
                prior[time] = True
            seen[key] = index
        covered.append({"capacity_gb": gb, "miss_bytes": int(sizes[miss].sum()), "covered_fraction": float(sizes[miss & prior].sum() / max(1, sizes[miss].sum()))})

    return {
        "trace": name,
        "prompts": len({s["prompt_id"] for s in steps}),
        "steps": len(steps),
        "decode_steps": decode_steps,
        "accesses": len(stream),
        "row_bytes": dict(segments),
        "call_budget_bytes": call_budget,
        "total_routed_bytes": total_routed_bytes,
        "requested_bytes": requested,
        "compulsory_bytes": compulsory,
        "repeated_bytes": {phase: requested[phase] - compulsory[phase] for phase in phases},
        "repeated_fraction": {phase: (requested[phase] - compulsory[phase]) / max(1, requested[phase]) for phase in phases},
        "decode_bytes_per_token": requested["decode"] / max(1, decode_steps),
        "decode_origin_fraction": {key: value / max(1, requested["decode"]) for key, value in origin.items()},
        "lru_curve": curve,
        "replay_check": checked,
        "previous_step_predictor": predictor,
        "previous_step_coverage_of_lru_misses": covered,
    }


def render(results: list[dict]) -> str:
    lines = ["# Phase 6A: repeated expert bytes and LRU capacity (recorded routing)", ""]
    for r in results:
        lines += [
            f"## {r['trace']}: {r['prompts']} prompts, {r['steps']} steps ({r['decode_steps']} decode), {r['accesses']:,} row requests",
            "",
            f"- Decode: {r['decode_bytes_per_token'] / 1e9:.3f} GB requested per token; "
            f"{r['repeated_fraction']['decode']:.3f} of decode bytes were requested before (prefill: {r['repeated_fraction']['prefill']:.3f}).",
            "- Where a decode step's bytes were last requested: "
            + ", ".join(f"{k.replace('_', ' ')} {v:.3f}" for k, v in r["decode_origin_fraction"].items()) + ".",
            f"- Previous-step predictor (decode): precision {r['previous_step_predictor']['precision']:.3f}, recall {r['previous_step_predictor']['recall']:.3f}.",
            "- Share of an LRU cache's decode misses (steady) a previous-step prefetch would cover: "
            + ", ".join(f"{c['capacity_gb']} GB {c['covered_fraction']:.3f}" for c in r["previous_step_coverage_of_lru_misses"]) + ".",
            f"- Stack distances equal a PageCache replay at {', '.join(r['replay_check'])} GB on every step: {all(r['replay_check'].values())}.",
            "",
            "| LRU capacity (GB) | of all experts | decode hits (cold) | decode hits (steady) | decode byte hits (steady) | prefill hits (steady) | drive GB per decode token |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in r["lru_curve"]:
            lines.append(
                f"| {c['capacity_gb']} | {c['fraction_of_experts']:.3f} | {c['decode_cold_hit_rate']:.3f} | {c['decode_steady_hit_rate']:.3f} | "
                f"{c['decode_steady_byte_hit_rate']:.3f} | {c['prefill_steady_hit_rate']:.3f} | {c['decode_drive_gb_per_token']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase4b-moonlight.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads((REPO_ROOT / config["index"]["directory"] / "manifest.json").read_text(encoding="utf-8"))
    first = next(iter(manifest["metadata"]["groups"].values()))
    segments = [(parameter, manifest["segments"][segment]["row_bytes"]) for parameter, segment in first["segments"].items()]
    call_budget = config.get("call_budget_bytes")
    results = []
    for name, (path, prompts_path) in TRACES.items():
        steps = sorted(read_jsonl(REPO_ROOT / path), key=lambda r: (r["prompt_id"], r["step"]))
        prompts = read_jsonl(REPO_ROOT / prompts_path)
        results.append(analyze(name, steps, prompts, segments, call_budget))
        print(f"{name}: done", flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "trace.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (output / "trace.md").write_text(render(results), encoding="utf-8")
    print(render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
