"""Aggregate a Phase 6A run (decision 0012) and evaluate its gates (configs/phase6a-native.yaml).

    uv run python benchmarks/native_report.py experiments/phase6a/<run> [--compare experiments/phase6a/<other run>]
        [--baseline experiments/phase6a/baseline-run1] [--io experiments/phase6a/io/io-cuda.json ...]

The run is `moonlight_runtime.py` on configs/phase6a-native.yaml (Phase 4B's benchmark with Python and native
configurations against one reference), plus the profiles `moonlight_profile.py` wrote into it
(`profile-<configuration>-<cold|warm>[-repeat].json`). Writes summary.json and summary.md into the run directory. Exit
code 0 iff every gate holds (and, with --compare, both runs used the same source trees, the native core's included, and
have equal digests):

  correctness   every step of every configuration equal to the reference in every recorded digest; every audit clean
                (Phase 4B's storage audit, and the native host cache's: lookups, copied bytes, budget)
  io_parity     native-stream's records equal python-stream's in every field but timings and system, and its raw
                physical-I/O traces equal python-stream's
  host_cache    on every step of every host-cache configuration whose budget holds a call's rows, the measured hits and
                misses equal an LRU replay of the reference's routing in the native backend's order (per transfer: every
                row looked up, then the misses admitted; admission frozen on prefill steps where configured); the
                process's peak working set within the budget plus the configured margin
  performance   the best native configuration's mean decode step (warm profiles) at most the configured fraction of the
                best Python configuration's; the aspirational ratio against python-stream is reported

Also the comparison table of the brief (decode latency, tokens/s, drive bytes and throughput, cache hit rate, requests
and read amplification, main-thread CPU time, peak host and device memory, GPU idle and copy time), per configuration,
cold and warm, and the I/O microbenchmarks (`native_io.py`'s outputs, --io).
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

from moonlight_report import _Row, quantiles, summarize  # noqa: E402

from awpmi.storage.cache import LRUPolicy, PageCache  # noqa: E402
from awpmi.tracing import read_jsonl  # noqa: E402

# Record fields that describe the backend rather than the step (excluded from the I/O parity comparison).
BACKEND_FIELDS = ("configuration", "backend", "native", "host_cache", "timings_ms", "system")


def correctness(records: list[dict]) -> dict:
    by_config = defaultdict(list)
    for r in records:
        by_config[r["configuration"]].append(r)
    out = {}
    for name, rows in by_config.items():
        summary = summarize(rows)
        out[name] = {
            "steps": summary["steps"],
            "all_equal": summary["all_equal"],
            "audit_failures": summary["audit_failures"],
            "poisoned_steps": summary["poisoned_steps"],
            "chunked_calls": summary["chunked_calls"],
            "expert_outputs_checked_layers": summary["expert_outputs_checked_layers"],
            "os_counters_match": summary["os_counters_match"],
        }
    return out


def io_parity(records: list[dict], ranges: list[dict], python: str = "python-stream", native: str = "native-stream") -> dict:
    """native's records equal python's in every field but the backend's own, step by step; and their raw ranges."""
    strip = lambda r: {k: v for k, v in r.items() if k not in BACKEND_FIELDS}  # noqa: E731
    a = [strip(r) for r in records if r["configuration"] == python]
    b = [strip(r) for r in records if r["configuration"] == native]
    differing = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    fields = sorted({k for i in differing[:20] for k in a[i] if a[i][k] != b[i].get(k)})
    ra = [(r["prompt_id"], r["step"], r["ranges"]) for r in ranges if r["configuration"] == python]
    rb = [(r["prompt_id"], r["step"], r["ranges"]) for r in ranges if r["configuration"] == native]
    os_python = [(r["system"]["os_read_calls"], r["system"]["os_read_bytes"]) for r in records if r["configuration"] == python]
    os_native = [(r["system"]["os_read_calls"], r["system"]["os_read_bytes"]) for r in records if r["configuration"] == native]
    return {
        "steps": [len(a), len(b)],
        "records_equal": len(a) == len(b) and not differing,
        "differing_steps": len(differing),
        "differing_fields": fields,
        "ranges_steps": [len(ra), len(rb)],
        "ranges_equal": len(ra) == len(rb) and ra == rb,
        "os_counters_equal": os_python == os_native,
    }


def replay_native(reference: list[dict], prompts: list[dict], segments: list[tuple[str, int]], capacity: int, freeze_prefill: bool,
                  call_budget: int | None, steps: int, num_prompts: int, prefetch: bool = False) -> list[tuple[int, int]]:
    """(hits, misses) per step of an LRU cache driven in the native backend's order.

    Per experts call: unchunked, one transfer of every segment's rows (the gate/up rows, then the down rows); chunked,
    one transfer per segment and chunk. Per transfer every row is looked up first, then the misses are admitted in order
    (none on prefill steps when admission is frozen there). With `prefetch`, a chunked call first promotes its cached rows
    and admits the others, in the same order, without counting lookups (the native prefetch). Prompts in order, the
    cache persisting across them. A hit here is a native hit or wait.
    """
    by_key = {(r["prompt_id"], r["step"]): r for r in reference}
    cache = PageCache(capacity, LRUPolicy())
    expert_row = sum(nbytes for _, nbytes in segments)
    rows = {nbytes: _Row(nbytes) for _, nbytes in segments}
    per_step = []
    for prompt in prompts[:num_prompts]:
        for step in range(steps + 1):
            cache.admit = not (freeze_prefill and step == 0)
            record = by_key[(prompt["prompt_id"], step)]
            before = (cache.stats.hits, cache.stats.misses)
            for layer, experts in enumerate(record["routed"]):
                chunked = call_budget is not None and len(experts) * expert_row > call_budget
                if chunked:
                    size = call_budget // expert_row
                    transfers = [[(segment, e) for e in experts[first : first + size]] for segment, _ in segments for first in range(0, len(experts), size)]
                else:
                    transfers = [[(segment, e) for segment, _ in segments for e in experts]]
                sizes = dict(segments)
                if chunked and prefetch:
                    absent = []
                    for segment, _ in segments:
                        for e in experts:
                            key = ((layer, segment), e)
                            if cache.peek(key) is None:
                                absent.append((key, sizes[segment]))
                            else:
                                cache.policy.touch(key)
                    if cache.admit:
                        for key, nbytes in absent:
                            cache.put(key, rows[nbytes])
                for transfer in transfers:
                    found = [cache.get(((layer, segment), e), sizes[segment]) for segment, e in transfer]
                    for (segment, e), entry in zip(transfer, found):
                        if entry is None:
                            cache.put(((layer, segment), e), rows[sizes[segment]])
            per_step.append((cache.stats.hits - before[0], cache.stats.misses - before[1]))
    return per_step


def host_cache(config: dict, records: list[dict], reference: list[dict], prompts: list[dict], segments: list[tuple[str, int]]) -> dict:
    steps = int(config["prompts"]["decode_steps"])
    expert_row = sum(nbytes for _, nbytes in segments)
    largest_call = max(len(experts) for r in reference for experts in r["routed"])  # rows of one transfer, upper bound
    out = {}
    for entry in config["configurations"]:
        if entry.get("backend") != "native" or not (entry.get("host_cache_bytes") or entry.get("host_cache_experts")):
            continue
        name = entry["name"]
        mine = [r for r in records if r["configuration"] == name]
        capacity = int(entry["host_cache_experts"]) * expert_row if "host_cache_experts" in entry else int(float(entry["host_cache_bytes"]))
        budget = entry.get("call_budget_bytes", config.get("call_budget_bytes"))
        transfer_bytes = min(largest_call * expert_row, int(budget) if budget else largest_call * expert_row)
        measured = [(r["host_cache"]["served"], r["host_cache"]["misses"]) for r in mine]
        result = {
            "capacity_bytes": capacity,
            "freeze_prefill": bool(entry.get("freeze_prefill", False)),
            "steps": len(mine),
            "decode_hit_rate": _rate([r for r in mine if r["phase"] == "decode"]),
            "prefill_hit_rate": _rate([r for r in mine if r["phase"] == "prefill"]),
            "decode_drive_gb_per_token": quantiles(r["storage"]["physical_bytes"] / 1e9 for r in mine if r["phase"] == "decode"),
            "prefill_drive_gb": quantiles(r["storage"]["physical_bytes"] / 1e9 for r in mine if r["phase"] == "prefill"),
            "evictions": sum(r["host_cache"]["evictions"] for r in mine),
            "bypassed": sum(r["host_cache"]["bypassed"] for r in mine),
            "max_resident_bytes": max(r["host_cache"]["resident_bytes"] for r in mine),
            "max_peak_resident_bytes": max(r["host_cache"]["peak_resident_bytes"] for r in mine),
            "max_working_set_bytes": max(r["system"]["process"]["resident_bytes"] for r in mine),
            "within_budget": all(max(r["host_cache"]["resident_bytes"], r["host_cache"]["peak_resident_bytes"]) <= capacity for r in mine),
            "prefetch_rows": sum((r.get("native") or {}).get("prefetch_rows", 0) for r in mine),
            "prefetch_wasted": sum(r["host_cache"].get("prefetch_wasted", 0) for r in mine),
        }
        if capacity >= transfer_bytes:
            count = min(int(entry.get("num_prompts", len(prompts))), len(prompts))
            replay = replay_native(
                reference, prompts, segments, capacity, result["freeze_prefill"], int(budget) if budget else None, steps, count,
                prefetch=bool(entry.get("prefetch_chunks", False)),
            )
            result["replay_equals_measured"] = replay == measured
            result["replay_differing_steps"] = sum(a != b for a, b in zip(replay, measured))
        else:
            result["replay_equals_measured"] = None  # smaller than one transfer: leases and loads make it bypass
        out[name] = result
    return out


def _rate(rows: list[dict]) -> float | None:
    lookups = sum(r["host_cache"]["lookups"] for r in rows)
    return sum(r["host_cache"]["served"] for r in rows) / lookups if lookups else None


def profiles(run: Path) -> dict:
    """Per profile (configuration, cold or warm): the brief's comparison quantities."""
    out = {}
    for path in sorted(run.glob("profile-*.json")):
        p = json.loads(path.read_text(encoding="utf-8"))
        key = p["configuration"] + ("-warm" if p.get("warm") else "-cold") + ("-repeat" if path.stem.endswith("-repeat") else "")
        decode, prefill = p["summary"].get("decode", {}), p["summary"].get("prefill", {})
        steps = [s for s in p["steps"] if s["phase"] == "decode" and not s["traced"]]
        traces = [t for t in p.get("traces", []) if t["phase"] == "decode"]
        entry = {
            "configuration": p["configuration"],
            "backend": p.get("backend", "python"),
            "host_cache_bytes": p.get("host_cache_bytes", 0),
            "device_cache_bytes": p.get("cache_capacity_bytes", 0),
            "warm": bool(p.get("warm")),
            "decode_steps": len(steps),
            "decode_ms": decode.get("wall_ms"),
            "decode_ms_median": decode.get("wall_ms_median"),
            "decode_ms_p95": decode.get("wall_ms_p95"),
            "tokens_per_s": decode.get("tokens_per_s"),
            "drive_gb_per_token": decode.get("physical_mb", 0) / 1e3,
            "requested_gb_per_token": decode.get("requested_mb", 0) / 1e3 if decode.get("requested_mb") is not None else None,
            "read_calls_per_token": decode.get("read_calls"),
            "requests_per_token": decode.get("requests"),
            "extents_per_token": decode.get("extents"),
            "io_wait_ms": decode.get("regions_ms", {}).get("io"),
            "drive_busy_ms": decode.get("drive_busy_ms"),
            "drive_gb_per_s": decode.get("drive_gb_per_s_busy") or decode.get("drive_gb_per_s"),
            "host_cache_hit_rate": decode.get("host_cache_hit_rate"),
            "device_cache_hits_per_token": statistics.mean(s["cache_hits"] for s in steps) if steps and steps[0].get("cache_hits") is not None else None,
            "main_thread_cpu_ms": decode.get("main_thread_cpu_ms"),
            "h2d_device_ms": decode.get("h2d_device_ms"),
            "h2d_copies": decode.get("h2d_copies"),
            "regions_ms": decode.get("regions_ms"),
            "peak_device_bytes": decode.get("peak_device_bytes"),
            "peak_resident_bytes": max(
                [(s.get("memory") or {}).get("peak_resident_bytes", 0) for s in p["steps"]] + [(p.get("memory_after_load") or {}).get("peak_resident_bytes", 0)]
            ),
            "gpu_idle_fraction": statistics.mean(t["gpu_idle_fraction"] for t in traces) if traces else None,
            "gpu_busy_ms": statistics.mean(t["gpu_busy_ms"] for t in traces) if traces else None,
            "h2d_trace_ms": statistics.mean(t["device_ms_by_kind"].get("memcpy_h2d", 0.0) for t in traces) if traces else None,
            "prefill_ms": prefill.get("wall_ms"),
            "prefill_ms_by_length": prefill.get("by_length"),
            "prefill_drive_gb": prefill.get("physical_mb", 0) / 1e3 if prefill else None,
        }
        # Read amplification: physical bytes over the rows' bytes actually read from storage: what the device cache did not
        # serve, less what the host cache served (hits and waits), plus what prefetches read into it.
        read = sum(
            s.get("fetched_bytes", s.get("requested_bytes", 0)) - (s.get("native") or {}).get("cache_copied_bytes", 0)
            + (s.get("native") or {}).get("prefetch_bytes", 0)
            for s in steps
        )
        entry["read_amplification"] = sum(s["physical_bytes"] for s in steps) / read if read else None
        out[key] = entry
    return out


def gates(config: dict, summary: dict) -> dict:
    gate = config["gate"]
    out = {}
    correct = summary["correctness"]
    out["correctness"] = all(c["all_equal"] == c["steps"] and c["audit_failures"] == 0 for c in correct.values())
    parity = summary["io_parity"]
    out["io_parity"] = bool(parity["records_equal"] and parity["ranges_equal"])
    cache = summary["host_cache"]
    margin = int(float(gate["max_working_set_above_cache_bytes"]))
    out["host_cache"] = all(
        c["within_budget"] and c["replay_equals_measured"] is not False and c["max_working_set_bytes"] <= c["capacity_bytes"] + margin
        for c in cache.values()
    )
    # A configuration's warm decode step: the mean of its warm profiles (a repeat included).
    warm = defaultdict(list)
    backend = {}
    for v in summary["profiles"].values():
        if v["warm"] and v["decode_ms"]:
            warm[v["configuration"]].append(v["decode_ms"])
            backend[v["configuration"]] = v["backend"]
    means = {name: statistics.mean(values) for name, values in warm.items()}
    python = {name: ms for name, ms in means.items() if backend[name] == "python"}
    native = {name: ms for name, ms in means.items() if backend[name] == "native"}
    if python and native:
        best_python, best_native = min(python, key=python.get), min(native, key=native.get)
        ratio = native[best_native] / python[best_python]
        out["performance"] = ratio <= float(gate["max_native_decode_ratio_to_best_python"])
        out["performance_ratio"] = ratio
        out["best_python"], out["best_native"] = best_python, best_native
        stream = python.get("python-stream")
        out["aspirational_ratio"] = native[best_native] / stream if stream else None
        out["aspirational_met"] = out["aspirational_ratio"] is not None and out["aspirational_ratio"] <= float(gate["aspirational_decode_ratio"])
    else:
        out["performance"] = None
    return out


IO_FIELDS = ("requests", "gb_per_s", "wall_ms", "main_thread_cpu_ms", "physical_bytes_per_request", "read_calls_per_request",
             "extents_per_request", "drive_gb_per_s_busy", "os_reads_equal_store")


def io_benchmarks(paths: list[str]) -> dict:
    """`native_io.py`'s results, per file: each pattern's paths, the planning cost and the readers' sweep."""
    out = {}
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        patterns = {}
        for pattern, by_path in data["patterns"].items():
            entry = {"bytes_equal": by_path.get("bytes_equal"), "planning": by_path.get("planning")}
            for label in ("python", "native", "native-hit"):
                if label in by_path:
                    entry[label] = {key: by_path[label].get(key) for key in IO_FIELDS}
            patterns[pattern] = entry
        out[Path(path).stem] = {"device": data.get("device"), "patterns": patterns, "sweep": data.get("sweep", [])}
    return out


def render_io(io: dict) -> list[str]:
    f = lambda x, d=1: "–" if x is None else f"{x:.{d}f}"  # noqa: E731
    lines = []
    for name, result in io.items():
        lines += ["", f"## I/O microbenchmark `{name}` (delivered to {result['device']})", "",
                  "| Pattern | Path | GB/s delivered | Wall ms (mean, median, p95) | Main-thread CPU ms | Physical MB / request | Read calls / request | Drive GB/s busy | OS = store |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for pattern, entry in result["patterns"].items():
            for label in ("python", "native", "native-hit"):
                r = entry.get(label)
                if r is None:
                    continue
                wall = r["wall_ms"] or {}
                lines.append(
                    f"| {pattern} | {label} | {f(r['gb_per_s'], 2)} | {f(wall.get('mean'))} ({f(wall.get('median'))}, {f(wall.get('p95'))}) | "
                    f"{f((r['main_thread_cpu_ms'] or {}).get('mean'))} | {f((r['physical_bytes_per_request'] or 0) / 1e6)} | {f(r['read_calls_per_request'], 0)} | "
                    f"{f(r['drive_gb_per_s_busy'], 2)} | {r['os_reads_equal_store']} |"
                )
            if entry.get("planning"):
                p = entry["planning"]
                lines.append(f"| {pattern} | planning only (ms / request) | Python {p['python_ms_per_request']:.3f}, native {p['native_ms_per_request']:.3f} | | | | | | |")
        if result["sweep"]:
            lines += ["", "| Sweep pattern | Readers | Read call (MiB) | GB/s delivered | Wall ms (mean) |", "| --- | --- | --- | --- | --- |"]
            for s in result["sweep"]:
                lines.append(f"| {s['pattern']} | {s['workers']} | {s['max_read_bytes'] / 2**20:.0f} | {f(s['gb_per_s'], 2)} | {f(s['wall_ms']['mean'])} |")
    return lines


def render(summary: dict) -> str:
    mark = lambda ok: "PASS" if ok else ("n/a" if ok is None else "FAIL")  # noqa: E731
    g = summary["gates"]
    lines = [f"# Phase 6A run {summary['run']}", ""]
    lines += ["## Gates", "", "| Gate | Result |", "| --- | --- |"]
    for key in ("correctness", "io_parity", "host_cache", "performance"):
        lines.append(f"| {key} | {mark(g.get(key))} |")
    if g.get("performance_ratio") is not None:
        lines.append(f"| best native ({g['best_native']}) / best Python ({g['best_python']}) decode, warm | {g['performance_ratio']:.3f} |")
        lines.append(f"| best native / python-stream decode, warm (aspirational ≤ 0.80) | {g['aspirational_ratio']:.3f} ({'met' if g['aspirational_met'] else 'not met'}) |")
    lines.append(f"| layering | {mark(g.get('layering'))} |")
    lines += ["", "## Correctness", "", "| Configuration | Steps | Equal | Audit failures | Poisoned | Chunked calls | Expert outputs checked (layers) |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, c in summary["correctness"].items():
        lines.append(f"| {name} | {c['steps']} | {c['all_equal']} | {c['audit_failures']} | {c['poisoned_steps']} | {c['chunked_calls']} | {c['expert_outputs_checked_layers']} |")
    p = summary["io_parity"]
    lines += ["", f"I/O parity (native-stream against python-stream): records {p['records_equal']} over {p['steps']} steps "
              f"(differing fields: {p['differing_fields'] or 'none'}); raw ranges {p['ranges_equal']} over {p['ranges_steps']} steps; "
              f"OS counters {p['os_counters_equal']}.", ""]
    lines += ["## Host cache", "", "| Configuration | Budget (GB) | Decode hits | Prefill hits | Drive GB / decode token | Evictions | Bypassed | Max resident (GB) | Max working set (GB) | Replay = measured |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, c in summary["host_cache"].items():
        rate = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731
        lines.append(
            f"| {name} | {c['capacity_bytes'] / 1e9:.2f} | {rate(c['decode_hit_rate'])} | {rate(c['prefill_hit_rate'])} | {c['decode_drive_gb_per_token'].get('mean', 0):.3f} | "
            f"{c['evictions']} | {c['bypassed']} | {c['max_resident_bytes'] / 1e9:.2f} | {c['max_working_set_bytes'] / 1e9:.2f} | {c['replay_equals_measured']} |"
        )
    perf = summary["profiles"]
    if perf:
        lines += ["", "## Performance (profiles, un-instrumented; decode means over 4 prompts x 8 steps)", "",
                  "| Profile | Decode ms/token (median, p95) | Tokens/s | Drive GB/token | Drive GB/s | Host-cache hits | I/O wait ms | Main-thread CPU ms | Requests / read calls / extents per token | Read amplification | Peak RAM (GB) | Peak VRAM (GB) | GPU idle | H2D device ms | Prefill ms |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for key, v in perf.items():
            f = lambda x, d=1: "–" if x is None else f"{x:.{d}f}"  # noqa: E731
            lines.append(
                f"| {key} | {f(v['decode_ms'])} ({f(v['decode_ms_median'])}, {f(v['decode_ms_p95'])}) | {f(v['tokens_per_s'], 3)} | {f(v['drive_gb_per_token'], 3)} | "
                f"{f(v['drive_gb_per_s'], 2)} | {f(v['host_cache_hit_rate'], 3)} | {f(v['io_wait_ms'])} | {f(v['main_thread_cpu_ms'])} | "
                f"{f(v['requests_per_token'], 0)} / {f(v['read_calls_per_token'], 0)} / {f(v['extents_per_token'], 0)} | {f(v['read_amplification'], 4)} | "
                f"{f((v['peak_resident_bytes'] or 0) / 1e9, 2)} | {f((v['peak_device_bytes'] or 0) / 1e9, 2)} | {f(v['gpu_idle_fraction'], 2)} | {f(v['h2d_device_ms'])} | "
                f"{f(v['prefill_ms'], 0)} |"
            )
    if summary.get("io"):
        lines += render_io(summary["io"])
    if summary.get("compare"):
        c = summary["compare"]
        lines += ["", f"Compared with {c['run']}: digests equal {c['digests_equal']}, same source tree {c['same_tree']}, same native tree {c['same_native_tree']}."]
    if summary.get("baseline"):
        b = summary["baseline"]
        lines += ["", f"Python baseline {b['run']}: digests against Phase 4B run1: {b['equal_to_phase4b']}."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--compare", default=None)
    parser.add_argument("--baseline", default=None, help="the Python baseline run, compared with Phase 4B run1")
    parser.add_argument("--io", nargs="*", default=[], help="native_io.py results to tabulate")
    args = parser.parse_args()
    run = Path(args.run)
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    records = read_jsonl(run / "records.jsonl.gz")
    reference = read_jsonl(run / "reference.jsonl.gz")
    ranges = read_jsonl(run / "ranges.jsonl.gz")
    prompts = read_jsonl(run / "prompts.jsonl")
    manifest = json.loads((REPO_ROOT / config["index"]["directory"] / "manifest.json").read_text(encoding="utf-8"))
    first = next(iter(manifest["metadata"]["groups"].values()))
    segments = [(parameter, manifest["segments"][segment]["row_bytes"]) for parameter, segment in first["segments"].items()]
    summary = {
        "run": run.as_posix(),
        "correctness": correctness(records),
        "io_parity": io_parity(records, ranges),
        "host_cache": host_cache(config, records, reference, prompts, segments),
        "profiles": profiles(run),
    }
    if args.io:
        summary["io"] = io_benchmarks(args.io)
    summary["gates"] = gates(config, summary)
    layering = subprocess.run([sys.executable, "-m", "pytest", "-q", str(REPO_ROOT / "tests" / "test_layering.py")], capture_output=True, text=True)
    summary["gates"]["layering"] = layering.returncode == 0
    if args.compare:
        other = Path(args.compare)
        a = json.loads((run / "digest.json").read_text(encoding="utf-8"))
        b = json.loads((other / "digest.json").read_text(encoding="utf-8"))
        ea = json.loads((run / "environment.json").read_text(encoding="utf-8"))
        eb = json.loads((other / "environment.json").read_text(encoding="utf-8"))
        native_a, native_b = ea.get("native_tree_sha256"), eb.get("native_tree_sha256")
        summary["compare"] = {
            "run": other.as_posix(),
            "digests_equal": a == b,
            "same_tree": ea["source_tree_sha256"] == eb["source_tree_sha256"],
            "same_native_tree": native_a is not None and native_a == native_b,
        }
    if args.baseline:
        base = Path(args.baseline)
        mine = json.loads((base / "digest.json").read_text(encoding="utf-8"))
        theirs = json.loads((REPO_ROOT / "experiments" / "phase4b" / "moonlight-run1" / "digest.json").read_text(encoding="utf-8"))
        summary["baseline"] = {"run": base.as_posix(), "equal_to_phase4b": {k: mine.get(k) == theirs.get(k) for k in theirs if k.endswith("sha256")}}
    (run / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (run / "summary.md").write_text(render(summary), encoding="utf-8")
    print(render(summary))
    ok = summary["gates"]["correctness"] and summary["gates"]["io_parity"] and summary["gates"]["host_cache"] and summary["gates"]["layering"]
    ok = ok and summary["gates"].get("performance") is not False
    if args.compare:
        ok = ok and summary["compare"]["digests_equal"] and summary["compare"]["same_tree"] and summary["compare"]["same_native_tree"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
