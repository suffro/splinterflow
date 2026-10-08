"""Phase 5A2 report: correctness, the bound comparison, the certificates' bytes, cost, escalation and gate (decision 0010).

    uv run --project research/crown_expert_oracle python research/crown_expert_oracle/report.py <run> [--compare <run2>]

Reads `<run>` (the artifact's manifest, validate/compare/search records, the verifier's stage reports) and writes
`summary.json` and `summary.md` there. The escalation rule and the gate are the run's config (fixed before any CROWN run
on real data). With `--compare`, the two runs' digests (timings and costs excluded) must be identical.

Units. Structural bounds are lower bounds on Δ·y (real tier: the real computation's y; Δ = (W_w − W_j)⊙g, so a pair is
decided when the bound is positive). Certified margins are Phase 5A's pairwise-certificate margins (positive: certified).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import yaml

EXCLUDED_FIELDS = ("timings_ms", "cost", "system")


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def shards(run: Path, part: str) -> list[dict]:
    records, shard = [], 0
    while (run / f"{part}.{shard}.jsonl.gz").exists():
        records.extend(read_records(run / f"{part}.{shard}.jsonl.gz"))
        shard += 1
    return records


def manifest_digest(path: Path) -> str:
    """The artifact's manifest without the export's timings (its files' sha256 are in it)."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for sample in manifest["samples"]:
        sample.pop("timings_ms", None)
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()


def digest(records: list[dict]) -> str | None:
    if not records:
        return None
    h = hashlib.sha256()
    for record in records:
        kept = {k: v for k, v in record.items() if k not in EXCLUDED_FIELDS}
        if "log" in kept:  # the search's evaluation log carries no timing, but keep the rule explicit
            kept["log"] = [{k: v for k, v in entry.items() if k not in EXCLUDED_FIELDS} for entry in kept["log"]]
        if "attack" in kept:  # the attack's wall time
            kept["attack"] = {k: v for k, v in kept["attack"].items() if k != "ms"}
        h.update(json.dumps(kept, sort_keys=True).encode("utf-8") + b"\n")
    return h.hexdigest()


def clean(values) -> list[float]:
    return [v for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]


def median(values) -> float | None:
    values = clean(values)
    return statistics.median(values) if values else None


def quantile(values, q: float) -> float | None:
    values = sorted(clean(values))
    if not values:
        return None
    position = q * (len(values) - 1)
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def mean(values) -> float | None:
    values = clean(values)
    return sum(values) / len(values) if values else None


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 1e4 or (abs(value) < 1e-3 and value != 0):
        return f"{value:.3g}"
    return f"{value:.{digits}f}"


# The bound comparison


def point_label(record: dict) -> str:
    return "initial" if record["index"] < 0 else f"{(record['index'] + 1) / record['steps']:.2f}"


def comparison(compare: list[dict]) -> dict:
    """Per strategy, tier and schedule point: medians over samples and pairs, and how much of Phase 5A's realistic-to-ideal
    distance each method closes (structural bounds, real tier)."""
    table: dict = {}
    for record in compare:
        key = (record["strategy"], record["tier"], point_label(record))
        entry = table.setdefault(key, {"fraction": [], "truth": [], "rows": 0, "methods": {}, "phase5a": {}, "optimum": [], "attack": [],
                                       "attack_vs_truth": [], "closed": {}, "closed_optimum": [], "margin": {}})
        entry["fraction"].append(record["certified_fraction"])
        entry["truth"].extend(record["truth"])
        entry["rows"] += len(record["truth"])
        p5 = record["phase5a"]
        if record["tier"] == "real":
            realistic, ideal = p5["realistic"]["structural"], p5["ideal"]["structural"]
            entry["phase5a"].setdefault("realistic", []).extend(realistic)
            entry["phase5a"].setdefault("ideal", []).extend(ideal)
            optimum = record.get("reduced_optimum", [])
            entry["optimum"].extend(optimum)
            for j, (r, i, o) in enumerate(zip(realistic, ideal, optimum)):
                if None not in (r, i, o) and i > r:
                    entry["closed_optimum"].append((o - r) / (i - r))
            for method, values in record["crown"].items():
                entry["methods"].setdefault(method, []).extend(values["lower"])
                for j, (v, r, i) in enumerate(zip(values["lower"], realistic, ideal)):
                    if None not in (v, r, i) and i > r:
                        entry["closed"].setdefault(method, []).append((v - r) / (i - r))
            if "attack" in record:
                for j, value in record["attack"]["values"].items():
                    entry["attack"].append(value)
                    entry["attack_vs_truth"].append(value - record["truth"][int(j)])
        else:
            entry["phase5a"].setdefault("realistic", []).extend(p5["realistic"]["margin"])
            entry["phase5a"].setdefault("ideal", []).extend(p5["ideal"]["margin"])
            for method, values in record["crown"].items():
                entry["methods"].setdefault(method, []).extend(values["margin"])
                gains = [v - r for v, r in zip(values["margin"], p5["realistic"]["margin"]) if None not in (v, r)]
                entry["margin"].setdefault(method, []).extend(gains)
    summary = {}
    for (strategy, tier, label), entry in sorted(table.items(), key=lambda kv: (kv[0][0], kv[0][1], -1 if kv[0][2] == "initial" else float(kv[0][2]))):
        row = {"fraction": median(entry["fraction"]), "pairs": entry["rows"], "truth": median(entry["truth"]),
               "phase5a_realistic": median(entry["phase5a"].get("realistic", [])), "phase5a_ideal": median(entry["phase5a"].get("ideal", [])),
               "methods": {m: median(v) for m, v in entry["methods"].items()}}
        if tier == "real":
            row.update(optimum=median(entry["optimum"]), attack=median(entry["attack"]), attack_minus_truth=median(entry["attack_vs_truth"]),
                       closed={m: median(v) for m, v in entry["closed"].items()}, closed_by_optimum=median(entry["closed_optimum"]),
                       attack_below_zero=sum(1 for v in entry["attack"] if v < 0), attacks=len(entry["attack"]))
        else:
            row["margin_gain"] = {m: median(v) for m, v in entry["margin"].items()}
            row["margin_gain_max"] = {m: (max(clean(v)) if clean(v) else None) for m, v in entry["margin"].items()}
        summary[f"{strategy}|{tier}|{label}"] = row
    return summary


# The search


def search_summary(search: list[dict], manifest: dict) -> dict:
    cells: dict = {}
    for record in search:
        key = f"{record['strategy']}|{record['tier']}|{record['graph']}/{record['method']}"
        cells.setdefault(key, []).append(record)
    result = {}
    for key, records in sorted(cells.items()):
        fractions = [r["fraction"] for r in records]
        certified = [r for r in records if r["would_certify"]]
        unread = [r for r in records if r["would_certify"] and r["fraction"] < 1.0]
        phase5a = [r["phase5a_cell"]["fraction"] for r in records]
        bound_seconds = [r["cost"]["bound_ms"] / 1e3 for r in records]
        evaluations = [r["evaluations"] for r in records]
        result[key] = {
            "samples": len(records), "coverage": len(certified) / len(records), "certified_with_bytes_unread": len(unread) / len(records),
            "mean": mean(fractions), "median": median(fractions), "p90": quantile(fractions, 0.9), "p95": quantile(fractions, 0.95),
            "fallback": 1.0 - len(certified) / len(records), "phase5a_realistic_mean": mean(phase5a),
            "fewer_bytes_than_phase5a": sum(1 for r in records if r["fraction"] < r["phase5a_cell"]["fraction"] - 1e-12),
            "more_bytes_than_phase5a": sum(1 for r in records if r["fraction"] > r["phase5a_cell"]["fraction"] + 1e-12),
            "wrong_certified": sum(1 for r in records if r["certified"] and not r["winner_is_reference"]),
            "bound_seconds_per_token": median(bound_seconds), "bound_seconds_per_token_max": max(bound_seconds) if bound_seconds else None,
            "evaluations_per_token": median(evaluations),
            "seconds_per_evaluation": median([s / e for s, e in zip(bound_seconds, evaluations) if e]),
            "peak_device_bytes": max(r["cost"]["peak_device_bytes"] for r in records), "peak_rss_bytes": max(r["cost"]["peak_rss_bytes"] for r in records),
            "contenders_bounded": sum(sum(e.get("contenders_bounded", 0) for e in r["log"]) for r in records),
            "alpha_calls": sum(sum(e.get("alpha_calls", 0) for e in r["log"]) for r in records),
            "full_checks": sum(sum(1 for e in r["log"] if e.get("full")) for r in records),
            "capped_full_checks": sum(sum(1 for e in r["log"] if e.get("capped")) for r in records),
        }
    return result


def phase5a_baseline(manifest: dict) -> dict:
    """Phase 5A's own cells on the same samples (the export ran them): per strategy, tier and cell, mean fraction, coverage."""
    out: dict = {}
    for sample in manifest["samples"]:
        for strategy, data in sample["strategies"].items():
            for tier, tier_data in data["tiers"].items():
                for cell, value in tier_data["phase5a_cells"].items():
                    entry = out.setdefault(f"{strategy}|{tier}|{cell}", {"fractions": [], "certified": 0, "unread": 0})
                    entry["fractions"].append(value["fraction"])
                    entry["certified"] += int(bool(value["would_certify"]))
                    entry["unread"] += int(bool(value["would_certify"]) and value["fraction"] < 1.0)
    return {k: {"samples": len(v["fractions"]), "mean": mean(v["fractions"]), "median": median(v["fractions"]),
                "coverage": v["certified"] / len(v["fractions"]), "certified_with_bytes_unread": v["unread"] / len(v["fractions"])}
            for k, v in sorted(out.items())}


# Escalation and gate


def escalation(config: dict, compare_summary: dict, search: dict, manifest: dict) -> dict:
    rule = config["escalation"]
    primary = config["strategies"]["primary"]
    closed = []
    for key, row in compare_summary.items():
        strategy, tier, label = key.split("|")
        if strategy != primary or tier != "real" or label == "1.00":
            continue
        best = max((v for v in row.get("closed", {}).values() if v is not None), default=None)
        if best is not None:
            closed.append(best)
    gap_closed = median(closed)
    fewer = max((cell["fewer_bytes_than_phase5a"] for cell in search.values()), default=0)
    proceed = (gap_closed is not None and gap_closed >= rule["gap_closed"]) or fewer >= rule["fewer_bytes_samples"]
    return {"median_gap_closed_by_best_method": gap_closed, "threshold": rule["gap_closed"], "max_samples_with_fewer_bytes": fewer,
            "fewer_bytes_threshold": rule["fewer_bytes_samples"], "proceed_to_stage_3": proceed}


def gate(config: dict, search: dict, baseline: dict) -> dict:
    thresholds = config["gate"]
    certified = {k: v for k, v in search.items() if k.split("|")[1] == "certified"}
    structural = {k: v for k, v in search.items() if k.split("|")[1] == "real"}
    best_certified = min(certified.items(), key=lambda kv: kv[1]["mean"], default=(None, None))
    best_structural = min(structural.items(), key=lambda kv: kv[1]["mean"], default=(None, None))
    verdict, reasons = "FAIL", []
    if best_certified[1] is not None:
        c = best_certified[1]
        if c["mean"] <= thresholds["strong_pass"]["mean"] and c["coverage"] >= thresholds["strong_pass"]["coverage"]:
            verdict = "STRONG PASS"
        elif c["mean"] <= thresholds["pass"]["mean"] and c["coverage"] >= thresholds["pass"]["coverage"]:
            verdict = "PASS"
    s_key, s = best_structural
    if verdict == "FAIL" and s is not None:
        strategy = s_key.split("|")[0]
        ideal = baseline.get(f"{strategy}|real|ideal/ideal", {}).get("mean")
        realistic = baseline.get(f"{strategy}|real|realistic/realistic", {}).get("mean")
        promising = (s["mean"] <= thresholds["research_promising"]["mean"] and s["certified_with_bytes_unread"] >= thresholds["research_promising"]["coverage"]
                     and ideal is not None and s["mean"] <= ideal + thresholds["research_promising"]["ideal_distance"])
        too_little = realistic is not None and realistic - s["mean"] < thresholds["fail"]["improvement"]
        reasons.append(f"best structural cell {s_key}: mean {fmt(s['mean'])}, certified with bytes unread {fmt(s['certified_with_bytes_unread'])}, "
                       f"Phase 5A realistic {fmt(realistic)}, Phase 5A ideal {fmt(ideal)}")
        if promising:
            verdict = "RESEARCH-PROMISING"
        elif s["mean"] > thresholds["fail"]["mean"] or too_little:
            verdict = "FAIL"
            reasons.append("structural mean above the fail threshold" if s["mean"] > thresholds["fail"]["mean"] else "improvement over Phase 5A below threshold")
        else:
            verdict = "INCONCLUSIVE"
    slow = [k for k, v in search.items() if v["bound_seconds_per_token"] is not None and v["bound_seconds_per_token"] > thresholds["fail"]["seconds_per_token"]]
    if slow:
        reasons.append(f"verification above {thresholds['fail']['seconds_per_token']} s per token: {slow}")
    return {"verdict": verdict, "best_certified": best_certified[0], "best_structural": best_structural[0], "reasons": reasons,
            "runtime_cost_prohibitive": bool(slow) and all(k in slow for k in search)}


# Markdown


def markdown(summary: dict) -> str:
    lines = [f"# Phase 5A2 summary: {summary['run']}", ""]
    env = summary["environment"]
    lines += ["## Environment", "", f"- verifier: Python {env.get('python')}, torch {env.get('torch')} (CUDA {env.get('cuda')}), numpy {env.get('numpy')}, "
              f"auto_LiRPA {env.get('auto_LiRPA', {}).get('version')} @ `{env.get('auto_LiRPA', {}).get('commit', '')[:12]}` ({env.get('auto_LiRPA', {}).get('license')})",
              f"- GPU: {env.get('gpu')}; PYTHONHASHSEED {env.get('python_hash_seed')}; source tree `{str(env.get('source_tree_sha256'))[:12]}`, research tree `{str(env.get('research_tree_sha256'))[:12]}`", ""]
    v = summary["validation"]
    lines += ["## Correctness", "", "| Check | Result |", "| --- | --- |"]
    for key, value in v.items():
        lines.append(f"| {key} | {fmt(value) if not isinstance(value, (dict, list)) else json.dumps(value)} |")
    lines += ["", "## Bound comparison (median over samples and pairs)", "",
              "Real tier: lower bounds on Δ·y (structural); `closed` is the share of the distance from Phase 5A's realistic to its ideal bound that a method closes (median per pair).", "",
              "| Strategy | Point | Bytes | Truth | 5A ideal | Set optimum | Attack | 5A realistic | auto_LiRPA (median) | closed | optimum closes |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, row in summary["comparison"].items():
        strategy, tier, label = key.split("|")
        if tier != "real":
            continue
        methods = "; ".join(f"{m} {fmt(x)}" for m, x in row["methods"].items())
        closed = "; ".join(f"{m} {fmt(x)}" for m, x in row.get("closed", {}).items())
        lines.append(f"| {strategy} | {label} | {fmt(row['fraction'])} | {fmt(row['truth'])} | {fmt(row['phase5a_ideal'])} | {fmt(row.get('optimum'))} | "
                     f"{fmt(row.get('attack'))} | {fmt(row['phase5a_realistic'])} | {methods} | {closed} | {fmt(row.get('closed_by_optimum'))} |")
    lines += ["", "Certified tier: pairwise-certificate margins (logits; > 0 certifies a pair); gain = auto_LiRPA's margin − Phase 5A's realistic margin.", "",
              "| Strategy | Point | Bytes | 5A ideal | 5A realistic | auto_LiRPA (median) | gain median | gain max |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, row in summary["comparison"].items():
        strategy, tier, label = key.split("|")
        if tier != "certified":
            continue
        methods = "; ".join(f"{m} {fmt(x)}" for m, x in row["methods"].items())
        gain = "; ".join(f"{m} {fmt(x)}" for m, x in row["margin_gain"].items())
        gain_max = "; ".join(f"{m} {fmt(x)}" for m, x in row["margin_gain_max"].items())
        lines.append(f"| {strategy} | {label} | {fmt(row['fraction'])} | {fmt(row['phase5a_ideal'])} | {fmt(row['phase5a_realistic'])} | {methods} | {gain} | {gain_max} |")
    lines += ["", "## Certificates: routed-expert bytes (fraction of the BF16 bytes; a sample never certified costs its whole schedule)", "",
              "| Cell | Samples | Coverage | Certified, bytes unread | Mean | Median | p90 | p95 | Phase 5A realistic mean | Fewer bytes than 5A | s/token (bounds) | evaluations/token | peak VRAM GB |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, row in summary["search"].items():
        lines.append(f"| {key} | {row['samples']} | {fmt(row['coverage'])} | {fmt(row['certified_with_bytes_unread'])} | {fmt(row['mean'])} | {fmt(row['median'])} | "
                     f"{fmt(row['p90'])} | {fmt(row['p95'])} | {fmt(row['phase5a_realistic_mean'])} | {row['fewer_bytes_than_phase5a']} | "
                     f"{fmt(row['bound_seconds_per_token'])} | {fmt(row['evaluations_per_token'])} | {row['peak_device_bytes'] / 1e9:.2f} |")
    lines += ["", "Phase 5A's own cells on the same samples:", "", "| Cell | Samples | Coverage | Certified, bytes unread | Mean | Median |", "| --- | --- | --- | --- | --- | --- |"]
    for key, row in summary["phase5a_baseline"].items():
        lines.append(f"| {key} | {row['samples']} | {fmt(row['coverage'])} | {fmt(row['certified_with_bytes_unread'])} | {fmt(row['mean'])} | {fmt(row['median'])} |")
    e, g = summary["escalation"], summary["gate"]
    lines += ["", "## Escalation (config, fixed before the runs)", "", f"- median share of the realistic-to-ideal distance closed by the best method: "
              f"{fmt(e['median_gap_closed_by_best_method'])} (threshold {e['threshold']}); samples certified with fewer bytes than Phase 5A: "
              f"{e['max_samples_with_fewer_bytes']} (threshold {e['fewer_bytes_threshold']}): **{'proceed to stage 3' if e['proceed_to_stage_3'] else 'stop after stage 2'}**",
              "", "## Gate", "", f"**{g['verdict']}** (best certified cell: {g['best_certified']}; best structural cell: {g['best_structural']})", ""]
    lines += [f"- {reason}" for reason in g["reasons"]]
    lines += ["", "## Digests", ""] + [f"- {k}: `{v}`" for k, v in summary["digests"].items()]
    return "\n".join(lines) + "\n"


# Stage 1.5: the L2 probe (run.py --part l2)


def _min(values, rows=None) -> float | None:
    values = values if rows is None else [values[j] for j in rows]
    values = clean(values)
    return min(values) if values else None


def _first(states: list[dict], test) -> float | None:
    """The routed-byte fraction of the first state (schedule order) from which `test` holds at every later state."""
    holds = [bool(test(s)) for s in states]
    for position in range(len(states)):
        if all(holds[position:]):
            return states[position]["fraction"]
    return None


def _last(states: list[dict], test) -> float | None:
    """The fraction of the last state at which `test` holds (None if never)."""
    found = [s["fraction"] for s in states if test(s)]
    return max(found) if found else None


def _positive(values) -> bool:
    values = clean(values)
    return bool(values) and min(values) > 0


def l2_state_row(s: dict) -> dict:
    """One state: each quantity's minimum over the focus contenders (the runner-up, the tightest by Phase 5A's realistic
    bound, the tightest truly), and over all comparison contenders where one exists for all of them."""
    focus = s["focus"]
    row = {"index": s["index"], "fraction": s["fraction"], "focus": focus,
           "truth": _min(s["truth"], focus), "truth_all": _min(s["truth"]),
           "phase5a_realistic": _min(s["phase5a"]["realistic"]["margin"], focus), "phase5a_realistic_all": _min(s["phase5a"]["realistic"]["margin"]),
           "phase5a_ideal": _min(s["phase5a"]["ideal"]["margin"], focus), "phase5a_ideal_all": _min(s["phase5a"]["ideal"]["margin"]),
           "box_reduced_optimum": _min(s["box_reduced"], focus), "box_reduced_optimum_all": _min(s["box_reduced"]),
           "box_weight_optimum": _min(s["box_weight"], focus), "box_weight_optimum_all": _min(s["box_weight"]),
           "l2_reduced_lower": _min(s["l2_reduced_lower"], focus), "l2_reduced_lower_all": _min(s["l2_reduced_lower"]),
           "l2_reduced_attained": _min(s["l2_reduced_attained"], focus), "l2_reduced_attained_all": _min(s["l2_reduced_attained"]),
           "l2_weight_lower": _min(s["l2_weight_lower"], focus), "l2_weight_lower_all": _min(s["l2_weight_lower"]),
           "witness_box": _min(s["witness"]["box"]), "witness_l2": _min(s["witness"]["l2"]), "witness_resident": _min(s["witness"]["resident"]),
           "witness_pulled_rows": s["witness_pulled_rows"], "down_balls_not_nested": s["down_balls_not_nested"]}
    if "crown" in s:
        row["box_crown"] = _min(s["crown"]["box_reduced"], focus)
        row["box_crown_all"] = _min(s["crown"]["box_reduced"])
        row["l2_crown"] = _min(s["crown"]["l2_reduced"])  # computed for the focus contenders only, in focus order
        # Phase 5A's structural bound (its decomposed bound, no float slack): what a CROWN bound compares with.
        row["phase5a_realistic_structural"] = _min(s["phase5a"]["realistic"]["decomposed"], focus)
        cost = s["cost"]["l2_reduced"]
        row["l2_crown_seconds"] = cost["wall_ms"] / 1e3
        row["l2_crown_bound_seconds"] = cost["bound_ms"] / 1e3
        row["l2_crown_setup_seconds"] = cost["setup_ms"] / 1e3
        row["l2_crown_peak_device_bytes"] = cost["peak_device_bytes"]
        row["l2_crown_peak_rss_bytes"] = cost["peak_rss_bytes"]
        row["box_crown_seconds"] = s["cost"]["box_reduced"]["wall_ms"] / 1e3
        # Share of Phase 5A's realistic→ideal distance closed, per focus contender (the structural bounds: Phase 5A's
        # decomposed bound, as the escalation rule); and by the L2 set's attained optimum (what no verifier on the
        # reduced L2 set can exceed) and the resident witness (what no verifier on all the metadata can exceed).
        realistic, ideal = s["phase5a"]["realistic"]["decomposed"], s["phase5a"]["ideal"]["decomposed"]
        closed, closed_set, closed_resident = [], [], []
        for position, j in enumerate(focus):
            r, i = realistic[j], ideal[j]
            if None in (r, i) or i <= r:
                continue
            closed.append((s["crown"]["l2_reduced"][position] - r) / (i - r))
            closed_set.append((s["l2_reduced_attained"][j] - r) / (i - r))
            closed_resident.append((s["witness"]["resident"][position] - r) / (i - r))
        row.update(gap_closed_l2_crown=median(closed), gap_closed_l2_set_at_most=median(closed_set), gap_closed_resident_at_most=median(closed_resident),
                   gap_closed_l2_crown_all=closed, gap_closed_l2_set_all=closed_set)
    return row


def l2_summary(records: list[dict], config: dict, manifest: dict) -> dict:
    settings = config["l2"]
    by_sample: dict = {}
    for record in records:
        by_sample.setdefault(record["sample"], []).append(record)
    samples = {}
    for sample, states in sorted(by_sample.items()):
        states = sorted(states, key=lambda s: s["index"])
        rows = [l2_state_row(s) for s in states]
        entry = manifest["samples"][sample]
        cells = entry["strategies"][settings["strategy"]]["tiers"][settings["tier"]]["phase5a_cells"]
        # Q6: from which routed-byte fraction each bound decides every comparison pair, and up to which fraction a set
        # provably holds weights that flip a pair (a witness below zero).
        decided = {
            "phase5a_realistic": _first(states, lambda s: _positive(s["phase5a"]["realistic"]["margin"])),
            "phase5a_ideal": _first(states, lambda s: _positive(s["phase5a"]["ideal"]["margin"])),
            "box_set_optimum_reduced": _first(states, lambda s: _positive(s["box_reduced"])),
            "box_set_optimum_weights": _first(states, lambda s: _positive(s["box_weight"])),
            "l2_set_reduced_lower": _first(states, lambda s: _positive(s["l2_reduced_lower"])),
            "l2_set_reduced_attained": _first(states, lambda s: _positive(s["l2_reduced_attained"])),
            "l2_set_weights_lower": _first(states, lambda s: _positive(s["l2_weight_lower"])),
        }
        crown_states = [s for s in states if "crown" in s]
        decided_at_points = {
            "box_crown": _first(crown_states, lambda s: _positive(s["crown"]["box_reduced"])),
            "l2_crown": _first(crown_states, lambda s: _positive(s["crown"]["l2_reduced"])),
            "phase5a_realistic": _first(crown_states, lambda s: _positive(s["phase5a"]["realistic"]["margin"])),
        }
        flips = {kind: _last(states, lambda s, kind=kind: _min(s["witness"][kind]) is not None and _min(s["witness"][kind]) < 0) for kind in ("box", "l2", "resident")}
        flips["l2_set_reduced_attained"] = _last(states, lambda s: _min(s["l2_reduced_attained"]) is not None and _min(s["l2_reduced_attained"]) < 0)
        flips["box_set_reduced"] = _last(states, lambda s: _min(s["box_reduced"]) is not None and _min(s["box_reduced"]) < 0)
        points = [r for r in rows if "l2_crown" in r]
        closed = [v for r in points if r["fraction"] < states[-1]["fraction"] for v in r["gap_closed_l2_crown_all"]]
        closed_set = [v for r in points if r["fraction"] < states[-1]["fraction"] for v in r["gap_closed_l2_set_all"]]
        samples[sample] = {
            "prompt_id": entry["prompt_id"], "step": entry["step"], "gap": entry["gap"], "selection_index": entry.get("selection_index"),
            "states": len(states), "full_fraction": states[-1]["fraction"],
            "phase5a_cells": {k: {"fraction": v["fraction"], "would_certify": v["would_certify"]} for k, v in cells.items()},
            "decided_from": decided, "decided_from_at_points": decided_at_points, "flipping_weights_until": flips,
            "points": points, "gap_closed_l2_crown_median": median(closed), "gap_closed_l2_set_median": median(closed_set),
            "l2_crown_seconds_per_state": median([r["l2_crown_seconds"] for r in points]),
            "l2_crown_seconds_per_state_max": max((r["l2_crown_seconds"] for r in points), default=None),
            "l2_crown_peak_device_bytes": max((r["l2_crown_peak_device_bytes"] for r in points), default=None),
            "l2_crown_peak_rss_bytes": max((r["l2_crown_peak_rss_bytes"] for r in points), default=None),
            # Before full materialization only: with every row read both are the exact value (up to Phase 5A's slack).
            "l2_crown_above_phase5a_realistic": sum(1 for r in points if r["fraction"] < states[-1]["fraction"]
                                                    and None not in (r["l2_crown"], r["phase5a_realistic_structural"])
                                                    and r["l2_crown"] > r["phase5a_realistic_structural"]),
            "witness_pulled_rows": sum(sum(s["witness_pulled_rows"].values()) for s in states),
            "down_balls_not_nested": sum(s["down_balls_not_nested"] for s in states),
        }
    return {"samples": samples, "stop": l2_stops(samples, settings)}


def l2_stops(samples: dict, settings: dict) -> dict:
    """The brief's stop conditions (config `l2.stop`) and its proceed rule (`l2.proceed`), from the stage 1.5 records.
    STOP A (representation) is the probe's finding (probe_l2.json), recorded by hand in the decision; here B, C, D."""
    stop, proceed = settings["stop"], settings["proceed"]
    closed = [s["gap_closed_l2_crown_median"] for s in samples.values() if s["gap_closed_l2_crown_median"] is not None]
    closed_median = median(closed)
    near = stop["near_full_fraction"]
    # B: even the L2 set's attained optimum (no verifier on it can do better) and the resident witness keep a pair
    # flipped until about full materialization (a negative attained value at a fraction at or above `near`).
    set_flips = {k: s["flipping_weights_until"]["l2_set_reduced_attained"] for k, s in samples.items()}
    resident_flips = {k: s["flipping_weights_until"]["resident"] for k, s in samples.items()}
    l2_flips = {k: s["flipping_weights_until"]["l2"] for k, s in samples.items()}
    met_on = [k for k, v in l2_flips.items() if v is not None and v >= near]
    stop_b = len(met_on) == len(l2_flips)
    # C: L2 CROWN closes too little, decides only near full materialization, does not beat Phase 5A's realistic bound.
    l2_decided = {k: s["decided_from_at_points"]["l2_crown"] for k, s in samples.items()}
    beats = sum(s["l2_crown_above_phase5a_realistic"] for s in samples.values())
    stop_c = ((closed_median is None or closed_median < stop["gap_closed"]) and beats == 0
              and all(v is None or v >= near for v in l2_decided.values()))
    seconds = [s["l2_crown_seconds_per_state_max"] for s in samples.values() if s["l2_crown_seconds_per_state_max"] is not None]
    stop_d = bool(seconds) and max(seconds) > stop["seconds_per_state"]
    proceed_ok = (not stop_b and not stop_c and not stop_d and beats > 0
                  and (closed_median is not None and closed_median >= proceed["gap_closed"]
                       or any(v is not None and v < proceed["fraction"] for v in l2_decided.values())))
    return {"B_uncertainty_set": {"triggered": stop_b, "met_on_samples": met_on, "l2_set_weights_flip_until": l2_flips,
                                  "l2_reduced_set_flips_until": set_flips, "resident_set_flips_until": resident_flips, "near_full_fraction": near},
            "C_weak_improvement": {"triggered": stop_c, "gap_closed_median": closed_median, "threshold": stop["gap_closed"],
                                   "l2_crown_decides_from": l2_decided, "states_where_l2_crown_beats_phase5a_realistic": beats},
            "D_verifier_cost": {"triggered": stop_d, "l2_crown_seconds_per_state_max": max(seconds) if seconds else None, "threshold": stop["seconds_per_state"]},
            "proceed_to_stage_2": proceed_ok}


def l2_markdown(run: Path, summary: dict, validation: dict, environment: dict) -> str:
    lines = [f"# Phase 5A2 stage 1.5 (L2 probe) summary: {run}", "",
             f"- verifier: Python {environment.get('python')}, torch {environment.get('torch')}, auto_LiRPA {environment.get('auto_LiRPA', {}).get('version')} "
             f"@ `{environment.get('auto_LiRPA', {}).get('commit', '')[:12]}`; GPU {environment.get('gpu')}; PYTHONHASHSEED {environment.get('python_hash_seed')}", "",
             "## Correctness", "", "| Check | Result |", "| --- | --- |"]
    lines += [f"| {k} | {fmt(v) if not isinstance(v, (dict, list)) else json.dumps(v)} |" for k, v in validation.items()]
    lines += ["", "Units: lower bounds and attained values of Δ·y in real arithmetic (Δ = (W_w − W_j)⊙g; positive decides the pair). "
              "Per state, each quantity's minimum over the focus contenders (the runner-up, the tightest by Phase 5A's realistic bound, "
              "the tightest truly). Reduced sets: Phase 5A's activation enclosure × down's rows in their sets; weight sets: gate, up and down rows "
              "in their sets. `[a, b]`: a lower bound and an attained value of the set's minimum.", ""]
    for sample, s in summary["samples"].items():
        lines += [f"## Sample {sample}: prompt {s['prompt_id']}, step {s['step']}, gap {fmt(s['gap'])} (stage 2 position {s['selection_index']})", "",
                  "| Bytes | Truth | 5A realistic | Box CROWN | Box set (reduced) | Box set (weights) | L2 CROWN | L2 set (reduced) | L2 set (weights) | Resident witness | 5A ideal | Gap closed by L2 CROWN | L2 CROWN s | peak VRAM GB | peak RSS GB |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in s["points"]:
            lines.append(f"| {fmt(r['fraction'])} | {fmt(r['truth'])} | {fmt(r['phase5a_realistic'])} | {fmt(r['box_crown'])} | {fmt(r['box_reduced_optimum'])} | "
                         f"{fmt(r['box_weight_optimum'])} | {fmt(r['l2_crown'])} | [{fmt(r['l2_reduced_lower'])}, {fmt(r['l2_reduced_attained'])}] | "
                         f"[{fmt(r['l2_weight_lower'])}, {fmt(r['witness_l2'])}] | {fmt(r['witness_resident'])} | {fmt(r['phase5a_ideal'])} | "
                         f"{fmt(r['gap_closed_l2_crown'])} | {fmt(r['l2_crown_seconds'], 1)} | {r['l2_crown_peak_device_bytes'] / 1e9:.2f} | {r['l2_crown_peak_rss_bytes'] / 1e9:.2f} |")
        lines += ["", "Routed-byte fraction from which each decides every comparison pair (every state of the schedule; CROWN at the comparison points), "
                  "and the last fraction at which a set holds weights that flip a pair:", "", "| Quantity | Decides from | | Set | Flipping weights until |", "| --- | --- | --- | --- | --- |"]
        decided = list(s["decided_from"].items()) + [(f"{k} (points)", v) for k, v in s["decided_from_at_points"].items()]
        flips = list(s["flipping_weights_until"].items())
        for position in range(max(len(decided), len(flips))):
            d = decided[position] if position < len(decided) else ("", None)
            f = flips[position] if position < len(flips) else ("", None)
            lines.append(f"| {d[0]} | {fmt(d[1])} | | {f[0]} | {fmt(f[1])} |")
        lines += ["", f"Phase 5A's own cells on this sample: {json.dumps(s['phase5a_cells'])}; gap closed by L2 CROWN (median, before full read): "
                  f"{fmt(s['gap_closed_l2_crown_median'])}, by the L2 set's attained optimum at most {fmt(s['gap_closed_l2_set_median'])}; "
                  f"witness rows pulled toward the truth: {s['witness_pulled_rows']}; down balls not nested in earlier ones: {s['down_balls_not_nested']}.", ""]
    stop = summary["stop"]
    lines += ["## Stop conditions (config `l2`, the brief's §17–18)", ""]
    for key in ("B_uncertainty_set", "C_weak_improvement", "D_verifier_cost"):
        lines.append(f"- **{key}**: {'TRIGGERED' if stop[key]['triggered'] else 'not triggered'} — {json.dumps({k: v for k, v in stop[key].items() if k != 'triggered'})}")
    lines += [f"- STOP A (representation): the probe's finding, `probe_l2_A.json`.", f"- **Proceed to stage 2: {'yes' if stop['proceed_to_stage_2'] else 'no'}**", ""]
    return "\n".join(lines) + "\n"


def join_crown(records: list[dict], crown: list[dict]) -> list[dict]:
    """The comparison points' CROWN records (run.py --part l2crown) joined to the same states' set records."""
    by_state = {(c["sample"], c["index"]): c for c in crown}
    joined = []
    for record in records:
        c = by_state.get((record["sample"], record["index"]))
        if c is not None:
            if c["focus"] != record["focus"]:
                raise ValueError(f"state {record['sample']}/{record['index']}: the two parts chose different focus pairs")
            record = {**record, "crown": c["crown"], "cost": c["cost"], "crown_set_values": c["set_values"]}
        joined.append(record)
    return joined


def main_l2(run: Path) -> int:
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((run / "artifact" / "manifest.json").read_text(encoding="utf-8"))
    validate, sets_records, crown_records = shards(run, "validate"), shards(run, "l2"), shards(run, "l2crown")
    records = join_crown(sets_records, crown_records)
    reports = {part: json.loads((run / f"verifier_{part}_0.json").read_text(encoding="utf-8")) for part in ("l2", "l2crown") if (run / f"verifier_{part}_0.json").exists()}
    environment = next(iter(reports.values()))["environment"] if reports else {}
    failures = {part: report["failures"] for part, report in reports.items()}
    validation = {
        "samples validated": len(validate),
        "wrapper max relative error": max((r["wrapper_full_max_relative"] for r in validate), default=None),
        "sets without the true weights (boxes and L2 balls, every state)": sum(len(v) for r in validate for v in r["containment"].values()),
        "sets unchanged by poisoned unread bytes (boxes and L2 balls)": all(all(r["poisoning_unchanged"].values()) for r in validate) if validate else None,
        "assembly vs Phase 5A margins (max relative difference)": max((r["assembly_max_relative_difference"] for r in validate), default=None),
        "stage 1.5 states recorded (sets; CROWN points)": [len(sets_records), len(crown_records)],
        "stage 1.5 failures (a bound above an attained value or the truth, a witness outside its set or below its set's lower bound)": failures,
        "largest witness constraint excess": max((max(r["witness_largest_excess"].values()) for r in records), default=None),
        "reference reproduced bitwise at export": all(all(s["bitwise"].values()) for s in manifest["samples"]),
    }
    summary = {"run": str(run), "environment": environment, "validation": validation, **l2_summary(records, config, manifest),
               "parts": {part: {"timings_ms": report["timings_ms"], "peak_device_bytes": report["peak_device_bytes"], "peak_rss_bytes": report["peak_rss_bytes"]}
                         for part, report in reports.items()},
               "digests": {"validate": digest(validate), "l2": digest(sets_records), "l2crown": digest(crown_records),
                           "manifest": manifest_digest(run / "artifact" / "manifest.json")}}
    (run / "l2_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    text = l2_markdown(run, summary, validation, environment)
    (run / "l2_summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--compare", default=None)
    parser.add_argument("--l2", action="store_true", help="stage 1.5: summarize run.py --part l2 into l2_summary.{json,md}")
    args = parser.parse_args()
    run = Path(args.run)
    if args.l2:
        return main_l2(run)
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((run / "artifact" / "manifest.json").read_text(encoding="utf-8"))
    validate, compare, search = shards(run, "validate"), shards(run, "compare"), shards(run, "search")
    environment = {}
    for name in ("verifier_search_0.json", "verifier_compare_0.json", "verifier_validate_0.json"):
        if (run / name).exists():
            environment = json.loads((run / name).read_text(encoding="utf-8"))["environment"]
            break
    validation = {
        "samples validated": len(validate),
        "wrapper max relative error (full graph vs Phase 5A real forward)": max((r["wrapper_full_max_relative"] for r in validate), default=None),
        "sets without the true weights": sum(len(v) for r in validate for v in r["containment"].values()),
        "schedules whose sets only shrink": sum(1 for r in validate for v in r["monotone"].values() if v),
        "schedules checked": sum(len(r["monotone"]) for r in validate),
        "sets unchanged by poisoned unread bytes": all(all(r["poisoning_unchanged"].values()) for r in validate) if validate else None,
        "assembly vs Phase 5A margins (max relative difference)": max((r["assembly_max_relative_difference"] for r in validate), default=None),
        "comparison bounds above the true value": sum(v.get("violations_truth", 0) for r in compare for v in r["crown"].values()),
        "comparison bounds above an achievable value of their set (exact optimum; adversarial realization)": sum(v.get("violations_achievable", 0) for r in compare for v in r["crown"].values()),
        "certified tokens differing from the reference": sum(1 for r in search if r["certified"] and not r["winner_is_reference"]),
        "reference reproduced bitwise at export": all(all(s["bitwise"].values()) for s in manifest["samples"]),
    }
    compare_summary = comparison(compare)
    search_cells = search_summary(search, manifest)
    baseline = phase5a_baseline(manifest)
    summary = {
        "run": str(run), "environment": environment, "samples": [{k: s[k] for k in ("index", "prompt_id", "step", "gap", "token", "experts")} for s in manifest["samples"]],
        "validation": validation, "comparison": compare_summary, "search": search_cells, "phase5a_baseline": baseline,
        "escalation": escalation(config, compare_summary, search_cells, manifest), "gate": gate(config, search_cells, baseline),
        "digests": {"validate": digest(validate), "compare": digest(compare), "search": digest(search), "manifest": manifest_digest(run / "artifact" / "manifest.json")},
    }
    if args.compare:
        other = Path(args.compare)
        theirs = {"validate": digest(shards(other, "validate")), "compare": digest(shards(other, "compare")), "search": digest(shards(other, "search")),
                  "manifest": manifest_digest(other / "artifact" / "manifest.json")}
        summary["reproduced"] = {k: summary["digests"][k] == theirs[k] for k in theirs}
    (run / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (run / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
