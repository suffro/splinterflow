"""Aggregate a Phase 6B run (decision 0013) and evaluate its gates (configs/phase6b-gpu.yaml).

    uv run python benchmarks/gpu_report.py experiments/phase6b/<run> [--compare experiments/phase6b/<other run>]

The run is `moonlight_runtime.py` on configs/phase6b-gpu.yaml: Phase 4B's benchmark with the routed experts as BF16 rows
or as encoded rows decoded on the GPU, native host caches and a device cache, against Phase 6A's baseline reference. Its
profiles are `moonlight_profile.py`'s, in the run directory (`profile-<configuration>-<cold|warm>[-<label>].json`; a label
marks a repeat, e.g. `seed2`). Writes summary.json and summary.md into the run directory. Exit code 0 iff every gate holds
(and, with --compare, both runs used the same source and native trees and have equal digests):

  correctness   every step of every configuration equal to the reference in every recorded digest; every audit clean
                (Phase 4B's and 6A's, and for encoded rows: decoded bytes = requested, the device cache's and the store's
                stored bytes = the requested rows' stored bytes)
  encoded       every row of the encoded pack, read from the drive and decoded on the GPU before inference, equal to the
                reference's row digests and to the pack's own record
  io_parity     native-stream's records and raw physical-I/O equal python-stream's (Phase 6A's gate, on 6B's engine)
  caches        on every step of every configuration with a device cache, or a host cache whose budget holds a transfer,
                each tier's measured hits and misses equal a replay of the reference's routing in the runtime's order:
                per experts call (or chunk), the device cache looked up first (an LRU per stored row size, as many
                slots as the run made), its misses transferred together through the host cache (an LRU over its rows'
                bytes: all looked up, then the misses admitted), then the device cache offered the misses; admissions
                frozen on prefill steps where configured; budgets held
  performance   the gates frozen in the config (`gates`), from the warm profiles: the 6B1 engine (its submit time per
                decode token, and decode with BF16 host caches against the Phase 6B baseline's profiles of the Phase 6A
                tree), encoded rows (against BF16 rows, same engine and host budget), the device cache (against the same
                host budget without it, within the GPU cap), decode graphs with PyTorch's NaN fills off (against the
                same configuration without them), and the best configuration against Phase 6A's 807 ms

Also the brief's metrics table (decode latency, tokens/s, drive and H2D bytes, copy and decode time, submit time, launches
per token from traced profiles, peak RAM and VRAM) and every repeated profile's spread.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from moonlight_report import _Row, quantiles  # noqa: E402
from native_report import correctness, io_parity  # noqa: E402

from awpmi.storage.cache import LRUPolicy, PageCache  # noqa: E402
from awpmi.tracing import read_jsonl  # noqa: E402


def segments_of(manifest: dict) -> list[tuple[str, int]]:
    """(parameter, row bytes) of an expert group, in the group's order (the order a call requests them)."""
    group = next(iter(manifest["metadata"]["groups"].values()))
    return [(parameter, manifest["segments"][segment]["row_bytes"]) for parameter, segment in group["segments"].items()]


def replay_tiers(reference: list[dict], prompts: list[dict], rows: list[tuple[str, int]], expert_bytes: int, host_capacity: int,
                 device_slots: dict[int, int] | None, freeze_prefill: bool, call_budget: int | None, steps: int,
                 num_prompts: int) -> list[dict]:
    """Per step, each tier's (hits, misses) in the runtime's order (module docstring). `rows`: the stored rows' sizes per
    parameter; `expert_bytes`: an expert's BF16 bytes (what a call's budget counts)."""
    by_key = {(r["prompt_id"], r["step"]): r for r in reference}
    sizes = dict(rows)
    entries = {nbytes: _Row(nbytes) for _, nbytes in rows}
    host = PageCache(host_capacity, LRUPolicy()) if host_capacity else None
    device = {size: PageCache(size * count, LRUPolicy()) for size, count in device_slots.items()} if device_slots else None
    per_step = []
    for prompt in prompts[:num_prompts]:
        for step in range(steps + 1):
            frozen = freeze_prefill and step == 0
            # (`is not None`: an empty PageCache is falsy.)
            for tier in ([host] if host is not None else []) + (list(device.values()) if device is not None else []):
                tier.admit = not frozen
            record = by_key[(prompt["prompt_id"], step)]
            before = {
                "host": (host.stats.hits, host.stats.misses) if host is not None else (0, 0),
                "device": tuple(sum(getattr(c.stats, k) for c in device.values()) for k in ("hits", "misses")) if device is not None else (0, 0),
            }
            for layer, experts in enumerate(record["routed"]):
                if call_budget is not None and len(experts) * expert_bytes > call_budget:
                    size = call_budget // expert_bytes
                    calls = [[(parameter, experts[first : first + size])] for parameter, _ in rows for first in range(0, len(experts), size)]
                else:
                    calls = [[(parameter, experts) for parameter, _ in rows]]
                for call in calls:
                    missing = []
                    for parameter, chosen in call:
                        for expert in chosen:
                            key = ((layer, parameter), expert)
                            if device is None or device[sizes[parameter]].get(key, sizes[parameter]) is None:
                                missing.append(key)
                    if host is not None and missing:
                        found = [host.get(key, sizes[key[0][1]]) for key in missing]
                        for key, entry in zip(missing, found):
                            if entry is None:
                                host.put(key, entries[sizes[key[0][1]]])
                    if device is not None:
                        for key in missing:
                            device[sizes[key[0][1]]].put(key, entries[sizes[key[0][1]]])
            after = {
                "host": (host.stats.hits, host.stats.misses) if host is not None else (0, 0),
                "device": tuple(sum(getattr(c.stats, k) for c in device.values()) for k in ("hits", "misses")) if device is not None else (0, 0),
            }
            per_step.append({tier: (after[tier][0] - before[tier][0], after[tier][1] - before[tier][1]) for tier in after})
    return per_step


def caches(config: dict, stream: dict, records: list[dict], reference: list[dict], prompts: list[dict], bf16: list[tuple[str, int]],
           encoded: list[tuple[str, int]] | None) -> dict:
    steps = int(config["prompts"]["decode_steps"])
    expert_bytes = sum(nbytes for _, nbytes in bf16)
    largest_call = max(len(experts) for r in reference for experts in r["routed"])
    margin = int(float(config["gates"]["max_working_set_above_cache_bytes"]))
    out = {}
    for entry in config["configurations"]:
        name = entry["name"]
        rows = encoded if entry.get("experts") == "encoded" else bf16
        stored_expert = sum(nbytes for _, nbytes in rows)
        mine = [r for r in records if r["configuration"] == name]
        # The host cache's budget as the run made it (`host_cache_experts` counts BF16 experts, whatever the rows).
        host_capacity = max((r["host_cache"]["capacity_bytes"] for r in mine if r.get("host_cache")), default=0)
        slots = (stream["configurations"].get(name) or {}).get("device_cache_slots")
        slots = {int(size): int(count) for size, count in slots.items()} if slots else None
        if not host_capacity and not slots:
            continue
        budget = entry.get("call_budget_bytes", config.get("call_budget_bytes"))
        if "call_budget_experts" in entry:
            budget = int(entry["call_budget_experts"]) * expert_bytes
        # The largest transfer: an unchunked call of as many experts as the budget allows (its rows at their stored size).
        transfer_bytes = min(largest_call, int(budget) // expert_bytes if budget else largest_call) * stored_expert
        result = {"host_capacity_bytes": host_capacity, "device_slots": slots, "freeze_prefill": bool(entry.get("freeze_prefill", False)), "steps": len(mine)}
        if host_capacity:
            result.update({
                "host_decode_hit_rate": _rate([r["host_cache"] for r in mine if r["phase"] == "decode"]),
                "host_evictions": sum(r["host_cache"]["evictions"] for r in mine),
                "host_bypassed": sum(r["host_cache"]["bypassed"] for r in mine),
                "host_within_budget": all(
                    max(r["host_cache"]["resident_bytes"], r["host_cache"]["peak_resident_bytes"], r["host_cache"].get("peak_held_bytes", 0)) <= host_capacity
                    for r in mine
                ),
                "max_working_set_bytes": max(r["system"]["process"]["resident_bytes"] for r in mine),
            })
            result["working_set_within_margin"] = result["max_working_set_bytes"] <= host_capacity + margin
        if slots:
            capacity = sum(size * count for size, count in slots.items())
            device = [r["cache"] for r in mine if r["phase"] == "decode"]
            hit = sum(c["hit_bytes"] for c in device)
            result.update({
                "device_capacity_bytes": capacity,
                "device_decode_byte_hit_rate": hit / max(1, hit + sum(c["miss_bytes"] for c in device)),
                "device_evictions": sum(r["cache"]["evictions"] for r in mine),
                "device_within_budget": all(r["cache_resident_bytes"] <= capacity for r in mine),
            })
        replayable = slots or host_capacity >= transfer_bytes
        if replayable:
            count = min(int(entry.get("num_prompts", len(prompts))), len(prompts))
            replay = replay_tiers(
                reference, prompts, rows, expert_bytes, host_capacity if host_capacity >= transfer_bytes else 0, slots,
                result["freeze_prefill"], int(budget) if budget else None, steps, count,
            )
            differing = 0
            for measured, expected in zip(mine, replay):
                if host_capacity >= transfer_bytes and (measured["host_cache"]["served"], measured["host_cache"]["misses"]) != expected["host"]:
                    differing += 1
                elif slots and (measured["cache"]["hits"], measured["cache"]["misses"]) != expected["device"]:
                    differing += 1
            result["replay_equals_measured"] = differing == 0 and len(replay) == len(mine)
            result["replay_differing_steps"] = differing
        else:
            result["replay_equals_measured"] = None  # a host cache smaller than one transfer: leases and loads make it bypass
        out[name] = result
    return out


def _rate(caches: list[dict]) -> float | None:
    lookups = sum(c["lookups"] for c in caches)
    return sum(c["served"] for c in caches) / lookups if lookups else None


def profiles(directory: Path) -> dict:
    """Per profile file: the brief's metrics."""
    out = {}
    for path in sorted(directory.glob("profile-*.json")):
        p = json.loads(path.read_text(encoding="utf-8"))
        decode, prefill = p["summary"].get("decode", {}), p["summary"].get("prefill", {})
        steps = [s for s in p["steps"] if s["phase"] == "decode" and not s["traced"]]
        traces = [t for t in p.get("traces", []) if t["phase"] == "decode"]
        decoder = [s["decoder"] for s in steps if s.get("decoder")]
        out[path.stem[len("profile-") :]] = {
            "configuration": p["configuration"],
            "warm": bool(p.get("warm")),
            "traced": bool(traces),
            "experts": p.get("experts", "bf16"),
            "host_cache_bytes": p.get("host_cache_bytes", 0),
            "device_cache_slots": p.get("device_cache_slots"),
            "seed": p.get("python_hash_seed"),
            "source_tree": (p.get("environment") or {}).get("source_tree_sha256"),
            "native_tree": (p.get("environment") or {}).get("native_tree_sha256"),
            "decode_ms": decode.get("wall_ms"),
            "decode_ms_median": decode.get("wall_ms_median"),
            "decode_ms_p95": decode.get("wall_ms_p95"),
            "tokens_per_s": decode.get("tokens_per_s"),
            "drive_gb_per_token": decode.get("physical_mb", 0) / 1e3,
            "h2d_gb_per_token": statistics.mean(s["h2d_bytes"] for s in steps) / 1e9 if steps else None,
            "h2d_device_ms": decode.get("h2d_device_ms"),
            "decode_host_ms": (decode.get("regions_ms") or {}).get("decode"),
            "decode_launches_per_token": statistics.mean(d["launches"] for d in decoder) if decoder else None,
            "submit_ms": (decode.get("engine_submit_ms") or {}).get(""),
            "host_cache_hit_rate": decode.get("host_cache_hit_rate"),
            "device_cache_byte_hit_rate": decode.get("device_cache_byte_hit_rate"),
            "regions_ms": decode.get("regions_ms"),
            "main_thread_cpu_ms": decode.get("main_thread_cpu_ms"),
            "peak_device_bytes": max(decode.get("peak_device_bytes") or 0, prefill.get("peak_device_bytes") or 0),
            "peak_reserved_bytes": max(decode.get("peak_reserved_bytes") or 0, prefill.get("peak_reserved_bytes") or 0) or None,
            "peak_resident_bytes": max(
                [(s.get("memory") or {}).get("peak_resident_bytes", 0) for s in p["steps"]] + [(p.get("memory_after_load") or {}).get("peak_resident_bytes", 0)]
            ),
            "kernel_launches": statistics.mean(t["kernel_launches"] for t in traces) if traces else None,
            "graph_launches": statistics.mean(t.get("graph_launches", 0) for t in traces) if traces else None,
            "device_kernels": statistics.mean(t["device_kernels"] for t in traces) if traces and "device_kernels" in traces[0] else None,
            "graphs_memory_bytes": (p.get("decode_graphs") or {}).get("memory_bytes"),
            "launch_cpu_ms": statistics.mean(t["launch_cpu_ms"] for t in traces) if traces else None,
            "gpu_idle_fraction": statistics.mean(t["gpu_idle_fraction"] for t in traces) if traces else None,
            "prefill_ms": prefill.get("wall_ms"),
        }
    return out


def warm_means(profiles_: dict) -> dict[str, dict]:
    """Per configuration, its un-traced warm profiles' decode means: the mean, each run, and the spread."""
    runs = defaultdict(list)
    for key, v in profiles_.items():
        if v["warm"] and not v["traced"] and v["decode_ms"]:
            runs[v["configuration"]].append(v["decode_ms"])
    return {
        name: {"mean": statistics.mean(values), "runs": values, "spread": (max(values) - min(values)) / statistics.mean(values) if len(values) > 1 else None}
        for name, values in runs.items()
    }


def gates(config: dict, summary: dict, baseline: dict | None) -> dict:
    gate = config["gates"]
    out = {}
    out["correctness"] = all(c["all_equal"] == c["steps"] and c["audit_failures"] == 0 for c in summary["correctness"].values())
    e = summary["encoded"]
    out["encoded"] = e is not None and not (e["differing"] or e["differing_from_pack"] or e["missing"]) and e["rows"] > 0
    parity = summary["io_parity"]
    out["io_parity"] = bool(parity["records_equal"] and parity["ranges_equal"])
    out["caches"] = all(
        c["replay_equals_measured"] is not False and c.get("host_within_budget", True) and c.get("device_within_budget", True)
        and c.get("working_set_within_margin", True)
        for c in summary["caches"].values()
    )
    warm = summary["warm"]
    ratio = lambda a, b: warm[a]["mean"] / warm[b]["mean"] if a in warm and b in warm else None  # noqa: E731
    perf = {}
    # 6B1: the engine's submit time, and decode with BF16 host caches against the baseline tree's profiles.
    submits = [v["submit_ms"] for v in summary["profiles"].values() if v["configuration"] in gate["engine_configurations"] and v["warm"] and v["submit_ms"] is not None]
    perf["submit_ms"] = max(submits) if submits else None
    perf["submit"] = perf["submit_ms"] is not None and perf["submit_ms"] <= float(gate["submit_ms_per_decode_token_max"])
    if baseline:
        mine = [warm[name]["mean"] for name in gate["engine_configurations"] if name in warm]
        theirs = [baseline[name]["mean"] for name in gate["engine_configurations"] if name in baseline]
        if len(mine) == len(theirs) == len(gate["engine_configurations"]):
            perf["engine_ratio"] = statistics.mean(mine) / statistics.mean(theirs)
            perf["engine"] = perf["engine_ratio"] <= float(gate["engine_decode_ratio_max"])
    # 6B2: encoded rows against BF16 rows, same engine and host budget.
    perf["encoded_ratio"] = ratio(*gate["encoded_pair"])
    perf["encoded"] = perf["encoded_ratio"] is not None and perf["encoded_ratio"] <= float(gate["encoded_decode_ratio_max"])
    # 6B3-A: the device cache against the same host budget without it, within the GPU cap.
    perf["device_cache_ratio"] = ratio(*gate["device_cache_pair"])
    reserved = [v["peak_reserved_bytes"] for v in summary["profiles"].values() if v["configuration"] == gate["device_cache_pair"][0] and v["peak_reserved_bytes"]]
    perf["device_cache_peak_reserved_bytes"] = max(reserved) if reserved else None
    perf["device_cache"] = (
        perf["device_cache_ratio"] is not None and perf["device_cache_ratio"] <= float(gate["device_cache_decode_ratio_max"])
        and perf["device_cache_peak_reserved_bytes"] is not None and perf["device_cache_peak_reserved_bytes"] <= int(float(config["gpu_budget_bytes"]))
    )
    # 6B4 (and the fills): the combined configuration against the device cache's.
    perf["fast_ratio"] = ratio(*gate["fast_pair"])
    perf["fast"] = perf["fast_ratio"] is not None and perf["fast_ratio"] <= float(gate["fast_decode_ratio_max"])
    # The phase: the best warm configuration against Phase 6A's.
    if warm:
        best = min(warm, key=lambda name: warm[name]["mean"])
        perf["best"] = best
        perf["best_ratio_to_phase6a"] = warm[best]["mean"] / float(gate["phase6a_decode_ms"])
        perf["phase"] = perf["best_ratio_to_phase6a"] <= float(gate["best_decode_ratio_to_phase6a_max"])
        perf["aspirational_met"] = perf["best_ratio_to_phase6a"] <= float(gate["aspirational_decode_ratio_to_phase6a"])
    out["performance"] = perf
    out["performance_pass"] = all(perf.get(k) for k in ("submit", "engine", "encoded", "device_cache", "fast", "phase"))
    return out


def render(summary: dict) -> str:
    mark = lambda ok: "PASS" if ok else ("n/a" if ok is None else "FAIL")  # noqa: E731
    f = lambda x, d=1: "–" if x is None else f"{x:.{d}f}"  # noqa: E731
    g = summary["gates"]
    p = g["performance"]
    lines = [f"# Phase 6B run {summary['run']}", "", "## Gates", "", "| Gate | Result |", "| --- | --- |"]
    for key in ("correctness", "encoded", "io_parity", "caches", "layering"):
        lines.append(f"| {key} | {mark(g.get(key))} |")
    lines += [
        f"| 6B1: engine submit ms per decode token (max over the engine configurations) | {f(p.get('submit_ms'), 2)} ({mark(p.get('submit'))}) |",
        f"| 6B1: BF16 host-cache decode against the baseline tree | {f(p.get('engine_ratio'), 3)} ({mark(p.get('engine'))}) |",
        f"| 6B2: encoded against BF16 rows, same host budget | {f(p.get('encoded_ratio'), 3)} ({mark(p.get('encoded'))}) |",
        f"| 6B3-A: device cache against the same host budget without it | {f(p.get('device_cache_ratio'), 3)}, peak reserved {f((p.get('device_cache_peak_reserved_bytes') or 0) / 1e9, 2)} GB ({mark(p.get('device_cache'))}) |",
        f"| 6B4: decode graphs and the fills off, against the same configuration without them | {f(p.get('fast_ratio'), 3)} ({mark(p.get('fast'))}) |",
        f"| Phase: best warm decode ({p.get('best')}) against Phase 6A's | {f(p.get('best_ratio_to_phase6a'), 3)} ({mark(p.get('phase'))}; aspirational {'met' if p.get('aspirational_met') else 'not met'}) |",
    ]
    lines += ["", "## Correctness", "", "| Configuration | Steps | Equal | Audit failures | Poisoned | Chunked calls |", "| --- | --- | --- | --- | --- | --- |"]
    for name, c in summary["correctness"].items():
        lines.append(f"| {name} | {c['steps']} | {c['all_equal']} | {c['audit_failures']} | {c['poisoned_steps']} | {c['chunked_calls']} |")
    e = summary["encoded"]
    if e:
        lines += ["", f"Encoded pack: {e['rows']} rows of {e['segments']} segments decoded on the GPU before inference ({e['stored_bytes'] / 1e9:.2f} GB stored, "
                  f"{e['logical_bytes'] / 1e9:.2f} GB restored, ratio {e['stored_ratio']:.4f}): differing from the reference {len(e['differing'])}, "
                  f"from the pack's record {len(e['differing_from_pack'])}, missing {len(e['missing'])}."]
    par = summary["io_parity"]
    lines += ["", f"I/O parity (native-stream against python-stream): records {par['records_equal']} over {par['steps']} steps; raw ranges {par['ranges_equal']}; "
              f"OS counters {par['os_counters_equal']}.", ""]
    lines += ["## Caches against a replay of the reference's routing", "",
              "| Configuration | Host (GB) | Device slots | Host decode hits | Device decode byte hits | Evictions (host / device) | Within budget | Replay = measured |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, c in summary["caches"].items():
        lines.append(
            f"| {name} | {c['host_capacity_bytes'] / 1e9:.2f} | {c['device_slots'] or '–'} | {f(c.get('host_decode_hit_rate'), 3)} | {f(c.get('device_decode_byte_hit_rate'), 3)} | "
            f"{c.get('host_evictions', '–')} / {c.get('device_evictions', '–')} | {c.get('host_within_budget', True) and c.get('device_within_budget', True)} | {c['replay_equals_measured']} |"
        )
    lines += ["", "## Performance (profiles; decode means over 4 prompts x 8 steps)", "",
              "| Profile | Decode ms/token (median, p95) | Tokens/s | Drive GB/token | H2D GB/token | Copy ms | Decode host ms | Submit ms | Host / device hits | Launches/token (kernels + graphs; kernels run) | Peak RAM / VRAM alloc / reserved (GB) | Prefill ms |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, v in summary["profiles"].items():
        lines.append(
            f"| {key} | {f(v['decode_ms'])} ({f(v['decode_ms_median'])}, {f(v['decode_ms_p95'])}) | {f(v['tokens_per_s'], 3)} | {f(v['drive_gb_per_token'], 3)} | "
            f"{f(v['h2d_gb_per_token'], 3)} | {f(v['h2d_device_ms'])} | {f(v['decode_host_ms'])} | {f(v['submit_ms'], 2)} | {f(v['host_cache_hit_rate'], 3)} / "
            f"{f(v['device_cache_byte_hit_rate'], 3)} | {f(v['kernel_launches'], 0)} + {f(v['graph_launches'], 0)}; {f(v['device_kernels'], 0)} | {f((v['peak_resident_bytes'] or 0) / 1e9, 2)} / "
            f"{f((v['peak_device_bytes'] or 0) / 1e9, 2)} / {f((v['peak_reserved_bytes'] or 0) / 1e9, 2)} | {f(v['prefill_ms'], 0)} |"
        )
    lines += ["", "Warm decode per configuration (mean of its runs; each run; spread):", ""]
    for name, w in sorted(summary["warm"].items(), key=lambda kv: kv[1]["mean"]):
        lines.append(f"- {name}: {w['mean']:.1f} ms ({', '.join(f'{x:.1f}' for x in w['runs'])}; spread {f(None if w['spread'] is None else 100 * w['spread'], 1)}%)")
    if summary.get("compare"):
        c = summary["compare"]
        lines += ["", f"Compared with {c['run']}: digests equal {c['digests_equal']}, same source tree {c['same_tree']}, same native tree {c['same_native_tree']}."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--compare", default=None)
    parser.add_argument("--baseline", nargs="*", default=[str(REPO_ROOT / "experiments/phase6b/baseline"), str(REPO_ROOT / "experiments/phase6b/baseline-repeat")],
                        help="directories of the Phase 6A tree's profiles (the 6B1 engine gate)")
    args = parser.parse_args()
    run = Path(args.run)
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    stream = json.loads((run / "stream_stage.json").read_text(encoding="utf-8"))
    records = read_jsonl(run / "records.jsonl.gz")
    reference = read_jsonl(run / "reference.jsonl.gz")
    ranges = read_jsonl(run / "ranges.jsonl.gz")
    prompts = read_jsonl(run / "prompts.jsonl")
    index = json.loads((run / "index.json").read_text(encoding="utf-8"))
    bf16 = segments_of(index["manifest"])
    encoded = None
    if index.get("encoded"):
        manifest = json.loads((REPO_ROOT / config["encoded"]["directory"] / "manifest.json").read_text(encoding="utf-8"))
        encoded = segments_of(manifest)
    audit = (index.get("encoded") or {}).get("audit")
    summary = {
        "run": run.as_posix(),
        "correctness": correctness(records),
        "encoded": None if audit is None else {**audit, "stored_ratio": index["encoded"]["stored_ratio"]},
        "io_parity": io_parity(records, ranges),
        "caches": caches(config, stream, records, reference, prompts, bf16, encoded),
        "profiles": profiles(run),
    }
    summary["warm"] = warm_means(summary["profiles"])
    baseline = {}
    for directory in args.baseline:
        for key, v in profiles(Path(directory)).items():
            if v["warm"] and not v["traced"] and v["decode_ms"]:
                baseline.setdefault(v["configuration"], []).append(v["decode_ms"])
    baseline = {name: {"mean": statistics.mean(values), "runs": values} for name, values in baseline.items()}
    summary["baseline_profiles"] = baseline
    summary["gates"] = gates(config, summary, baseline)
    layering = subprocess.run([sys.executable, "-m", "pytest", "-q", str(REPO_ROOT / "tests" / "test_layering.py")], capture_output=True, text=True)
    summary["gates"]["layering"] = layering.returncode == 0
    if args.compare:
        other = Path(args.compare)
        a = json.loads((run / "digest.json").read_text(encoding="utf-8"))
        b = json.loads((other / "digest.json").read_text(encoding="utf-8"))
        ea = json.loads((run / "environment.json").read_text(encoding="utf-8"))
        eb = json.loads((other / "environment.json").read_text(encoding="utf-8"))
        summary["compare"] = {
            "run": other.as_posix(),
            "digests_equal": a == b,
            "same_tree": ea["source_tree_sha256"] == eb["source_tree_sha256"],
            "same_native_tree": ea.get("native_tree_sha256") is not None and ea.get("native_tree_sha256") == eb.get("native_tree_sha256"),
        }
    (run / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (run / "summary.md").write_text(render(summary), encoding="utf-8")
    print(render(summary))
    g = summary["gates"]
    ok = all(g[k] for k in ("correctness", "encoded", "io_parity", "caches", "layering")) and g["performance_pass"]
    if args.compare:
        ok = ok and all(summary["compare"][k] for k in ("digests_equal", "same_tree", "same_native_tree"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
