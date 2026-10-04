"""Phase 5A report: correctness, ceilings, every strategy's bytes, the physical-I/O model and the decision gate.

    uv run python benchmarks/expert_oracle_report.py experiments/phase5a/<run> [--compare experiments/phase5a/<other run>]

Reads the run's samples.jsonl.gz (every sample: checks, ceilings, the certified and rn_even tiers' cells) and
real_cells.jsonl.gz (the real tier's cells), and writes summary.json and summary.md into the run directory.

A strategy's fraction on a sample is its bytes over the routed experts' BF16 bytes: the cell's if the sample's ceiling
holds in that tier (the cell then certifies, at worst with everything read), else its fallback (its whole schedule and
metadata: `fallback_bytes` in oracle_main_stage.json). Coverage is the share of samples certified (after any number of
bytes, the whole schedule included).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from awpmi.tracing import read_jsonl  # noqa: E402

GAP_BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, math.inf))
FLOOR_TERMS = ("norm_rounding", "y_rounding", "m_rounding", "R_rounding", "o_rounding", "down_accumulation", "z_rounding", "combine",
               "base_rounding", "lm_accumulation", "unread_neurons")
DECODE_GB = 2.70  # Phase 4B: routed-expert reads per decode token without a cache (0.0938 of 28.79 GB)
MOE_LAYERS = 26


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def at(q):
        return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]

    return {"n": len(values), "mean": statistics.fmean(values), "median": at(0.5), "p90": at(0.9), "p95": at(0.95), "min": ordered[0], "max": ordered[-1]}


def gap_bin(gap: float) -> str:
    for low, high in GAP_BINS:
        if low <= gap < high:
            return f"[{low:g}, {high:g})"
    return "?"


def load(run: Path) -> dict:
    from expert_oracle import shard_records

    data = {
        "config": yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")),
        "samples": shard_records(run, "samples"),
        "real_cells": shard_records(run, "real_cells"),
        "capture": read_jsonl(run / "capture.jsonl.gz"),
        "capture_stage": json.loads((run / "capture_stage.json").read_text(encoding="utf-8")),
        "digest": json.loads((run / "digest.json").read_text(encoding="utf-8")) if (run / "digest.json").exists() else {},
        "environment": json.loads((run / "environment.json").read_text(encoding="utf-8")),
    }
    data["stages"] = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(run.glob("oracle_*_stage.json"))}
    data["main_stage"] = data["stages"]["oracle_main_0_stage"]
    data["failures"] = json.loads((run / "failure.json").read_text(encoding="utf-8")) if (run / "failure.json").exists() else []
    return data


def correctness(data: dict) -> dict:
    samples, capture = data["samples"], data["capture"]
    compared = [r for r in capture if r.get("phase4b")]
    checks = defaultdict(int)
    for sample in samples:
        for key, value in sample["checks"].items():
            checks[key] += int(bool(value))
    tiers = {}
    for tier in samples[0]["ceilings"]:
        ceilings = [s["ceilings"][tier] for s in samples]
        cells = [c for s in samples for c in s["cells"] if c["tier"] == tier] + [c for s in data["real_cells"] for c in s["cells"] if c["tier"] == tier]
        realistic = [c for c in cells if c["bound"] == "realistic"]
        tiers[tier] = {
            "ceiling_violations": sum(sum(c["violations"].values()) for c in ceilings),
            "cell_violations": sum(sum(c["violations"].values()) for c in realistic),
            "wrong_would_certify": sum(1 for c in ceilings if c["would_certify"] and not c["candidate_is_reference"])
            + sum(1 for c in cells if c["would_certify"] and not c["candidate_is_reference"]),
            "certified_mismatches": sum(1 for c in ceilings if c["certified"] and not c["candidate_is_reference"])
            + sum(1 for c in cells if c["certified"] and not c["candidate_is_reference"]),
            "ceilings": len(ceilings),
            "cells": len(cells),
        }
    return {
        "samples": len(samples),
        "capture_steps": len(capture),
        "phase4b_compared_steps": len(compared),
        "phase4b_all_equal": all(all(r["phase4b"]["matches"].values()) for r in compared),
        "weights_equal_reference_rows": all(bool(stage.get("weights_equal_reference_rows")) for stage in data["stages"].values()),
        "reference_checks": {key: f"{value}/{len(samples)}" for key, value in checks.items()},
        "reference_checks_all": all(value == len(samples) for value in checks.values()),
        "tiers": tiers,
        "failures": data["failures"],
    }


def ceilings(data: dict) -> dict:
    samples = data["samples"]
    result = {}
    for tier in samples[0]["ceilings"]:
        rows = [(s, s["ceilings"][tier]) for s in samples]
        by_gap, by_length, by_step = defaultdict(list), defaultdict(list), defaultdict(list)
        for sample, ceiling in rows:
            by_gap[gap_bin(sample["gap"])].append(ceiling["would_certify"])
            by_length[sample["length"]].append(ceiling["would_certify"])
            by_step[sample["step"]].append(ceiling["would_certify"])
        terms, totals = defaultdict(list), []
        for sample, ceiling in rows:
            scale = ceiling.get("scale_lower")
            if ceiling["terms"] and scale:
                for name in FLOOR_TERMS:
                    if ceiling["terms"].get(name) is not None:
                        terms[name].append(ceiling["terms"][name] * scale)  # Δ·y units × q: logits
                totals.append(sum((ceiling["terms"].get(name) or 0.0) * scale for name in FLOOR_TERMS))
        margins = [c["margin"] for _, c in rows if c["margin"] is not None]
        result[tier] = {
            "coverage": statistics.fmean([c["would_certify"] for _, c in rows]) if rows else None,
            "by_gap": {key: {"n": len(v), "coverage": statistics.fmean(v)} for key, v in sorted(by_gap.items(), key=lambda kv: float(kv[0][1:].split(",")[0]))},
            "by_length": {str(k): {"n": len(v), "coverage": statistics.fmean(v)} for k, v in sorted(by_length.items())},
            "by_step": {str(k): {"n": len(v), "coverage": statistics.fmean(v)} for k, v in sorted(by_step.items())},
            "floor_terms_logits": {name: statistics.fmean(v) for name, v in terms.items() if v},
            "floor_total_logits": quantiles(totals),  # the tightest pair's uncertainty with every weight read
            "margin": quantiles(margins),
        }
    gaps = [s["gap"] for s in samples]
    result["gaps"] = quantiles(gaps)
    result["gap_bins"] = {gap_bin(low): sum(1 for g in gaps if gap_bin(g) == gap_bin(low)) for low, _ in GAP_BINS}
    return result


def strategy_table(data: dict, tier: str, bound: str, ordering: str, samples: list[dict] | None = None) -> dict:
    """Per strategy: every sample's fraction (the cell's, or the fallback's when the tier's ceiling fails), with breakdowns."""
    stage = data["main_stage"]
    routed = stage["routed_bytes"]
    samples = data["samples"] if samples is None else samples
    cells_by_sample = {}
    for record in (*data["samples"], *data["real_cells"]):
        for cell in record["cells"]:
            if (cell["tier"], cell["bound"], cell["ordering"]) == (tier, bound, ordering):
                cells_by_sample[(record["prompt_id"], record["step"], cell["strategy"])] = cell
    table = {}
    for name, info in stage["strategies"].items():
        fractions, certified_fractions, fallback_fractions, covered, by_gap, by_length, by_expert, per_expert = [], [], [], [], defaultdict(list), defaultdict(list), defaultdict(list), []
        physical = defaultdict(lambda: {"physical": 0, "logical": 0, "extents": [], "fractions": []})
        units, margins, unread = defaultdict(list), [], []
        breakdown = defaultdict(float)
        for sample in samples:
            ceiling = sample["ceilings"][tier] if sample.get("ceilings") else None
            cell = cells_by_sample.get((sample["prompt_id"], sample["step"], name))
            if cell is None and ceiling is not None and ceiling["would_certify"]:
                continue  # a cell not run (the real tier's subset)
            if cell is None:
                fraction, certified = info["fallback_bytes"] / routed, False
                breakdown["fallback_schedule"] += fraction
                modelled = info.get("fallback_physical", {})
            else:
                fraction, certified = cell["fraction"], cell["would_certify"]
                for key in ("gate", "up", "down", "levels", "metadata", "fallback"):
                    breakdown[key] += cell["bytes"][key] / routed
                modelled = cell["physical"]
            for scenario, values in physical_scenarios(info["family"], modelled).items():
                physical[scenario]["physical"] += values["physical_bytes"]
                physical[scenario]["logical"] += values["logical_bytes"]
                physical[scenario]["extents"].append(values["extents"])
                physical[scenario]["fractions"].append(values["physical_bytes"] / routed)
            if cell is not None:
                for key, value in cell["units_read"].items():
                    units[key].append(value)
                per_expert.extend(cell["per_expert_fraction"])
                if cell["margin"] is not None and certified:
                    margins.append(cell["margin"])
                if cell["unread_term"] is not None and certified:
                    unread.append(cell["unread_term"])
            fractions.append(fraction)
            covered.append(certified)
            (certified_fractions if certified else fallback_fractions).append(fraction)
            by_gap[gap_bin(sample["gap"])].append(fraction)
            by_length[sample["length"]].append(fraction)
            for expert in sample["experts"]:
                by_expert[expert].append(fraction)
        count = len(fractions)
        table[name] = {
            "family": info["family"], "samples": count,
            "coverage": statistics.fmean(covered) if covered else None,
            "fraction": quantiles(fractions),
            "fraction_certified": quantiles(certified_fractions),
            "fraction_fallback": quantiles(fallback_fractions),
            "fallback_share": 1.0 - statistics.fmean(covered) if covered else None,
            "breakdown": {key: value / count for key, value in breakdown.items()} if count else {},
            "by_gap": {key: quantiles(v)["mean"] for key, v in by_gap.items()},
            "by_length": {str(key): quantiles(v)["mean"] for key, v in sorted(by_length.items())},
            "by_expert_spread": quantiles([statistics.fmean(v) for v in by_expert.values()]),
            "per_expert_fraction": quantiles(per_expert),
            "physical": {
                scenario: {
                    "amplification": values["physical"] / max(1, values["logical"]),  # 4 KiB blocks over the bytes asked
                    "fraction": quantiles(values["fractions"]),  # modelled physical bytes over the routed BF16 bytes
                    "extents": quantiles(values["extents"]),
                }
                for scenario, values in physical.items()
            },
            "units_read": {key: statistics.fmean(v) for key, v in units.items()},
            "margin_when_certified": quantiles(margins),
            "unread_term_when_certified": quantiles(unread),
            "storage_multiplier": info["storage_multiplier"],
            "fallback_fraction": info["fallback_bytes"] / routed,
        }
    return table


def physical_scenarios(family: str, modelled: dict) -> dict[str, dict]:
    """A strategy's modelled reads: the checkpoint (A, B); for C the checkpoint (strided down columns) or a neuron-major
    copy, as alternatives; for D its level files and its exact rows from the checkpoint, together."""
    if not modelled:
        return {}
    if family == "neuron_major":
        return {name: modelled[name] for name in ("checkpoint", "neuron_major_copy") if name in modelled}
    if family == "refinement":
        parts = [modelled[name] for name in ("levels", "checkpoint") if name in modelled]
        return {"levels_and_checkpoint": {key: sum(p[key] for p in parts) for key in ("logical_bytes", "physical_bytes", "extents")}}
    return {"checkpoint": modelled["checkpoint"]}


def gate(config: dict, primary: dict, ideal: dict) -> dict:
    thresholds = config["gate"]
    best_name = min(primary, key=lambda n: primary[n]["fraction"]["mean"])
    best = primary[best_name]
    mean, coverage = best["fraction"]["mean"], best["coverage"]
    ideal_best_name = min(ideal, key=lambda n: ideal[n]["fraction"]["mean"])
    ideal_mean = ideal[ideal_best_name]["fraction"]["mean"]
    if mean <= thresholds["strong_pass_mean"] and coverage >= thresholds["strong_pass_coverage"]:
        verdict = "STRONG PASS"
    elif (mean <= thresholds["pass_mean"] and coverage >= thresholds["pass_coverage"]) or (mean > thresholds["pass_mean"] and ideal_mean <= thresholds["pass_mean"]):
        verdict = "PASS"
    elif mean <= thresholds["weak_mean"] and coverage >= thresholds["fail_coverage"]:
        verdict = "WEAK"
    else:
        verdict = "FAIL"
    return {
        "verdict": verdict, "best_strategy": best_name, "best_mean": mean, "best_coverage": coverage,
        "ideal_best_strategy": ideal_best_name, "ideal_best_mean": ideal_mean, "thresholds": thresholds,
    }


def projection(primary: dict, ideal: dict, real_ideal: dict | None) -> dict:
    """Q6: decode I/O if the oracle's fractions held in the runtime (a projection, not a measurement)."""

    def scaled(fraction: float) -> dict:
        return {
            "fraction": fraction,
            "last_layer_only_gb": DECODE_GB * (MOE_LAYERS - 1) / MOE_LAYERS + DECODE_GB / MOE_LAYERS * fraction,
            "every_layer_like_the_last_gb": DECODE_GB * fraction,
        }

    result = {"baseline_gb": DECODE_GB}
    for label, table in (("certified_realistic", primary), ("certified_ideal", ideal), ("real_ideal", real_ideal)):
        if table:
            best = min(table, key=lambda n: table[n]["fraction"]["mean"])
            result[label] = {"strategy": best, **scaled(table[best]["fraction"]["mean"])}
    return result


def summarize(run: Path, compare: Path | None) -> dict:
    data = load(run)
    config = data["config"]
    summary = {
        "run": str(run),
        "source_tree_sha256": data["environment"]["source_tree_sha256"],
        "python_hash_seed": data["environment"].get("python_hash_seed"),
        "correctness": correctness(data),
        "ceilings": ceilings(data),
    }
    tables = {}
    for tier, cells in config["oracle"]["cells"].items():
        steps = config["oracle"].get("cell_steps", {}).get(tier)
        subset = None if steps is None else [s for s in data["samples"] if s["step"] in set(steps)]
        for bound, ordering in cells:
            tables[f"{tier}/{bound}/{ordering}"] = strategy_table(data, tier, bound, ordering, subset)
    summary["tables"] = tables
    primary, ideal = tables["certified/realistic/realistic"], tables["certified/ideal/ideal"]
    summary["gate"] = gate(config, primary, ideal)
    summary["projection"] = projection(primary, ideal, tables.get("real/ideal/ideal"))
    summary["digest"] = data["digest"]
    if compare is not None:
        other = json.loads((compare / "digest.json").read_text(encoding="utf-8"))
        keys = [k for k in data["digest"] if k.endswith("sha256")]
        summary["reproducible"] = {k: data["digest"].get(k) == other.get(k) for k in keys}
        summary["reproducible"]["all"] = all(summary["reproducible"].values())
        other_environment = json.loads((compare / "environment.json").read_text(encoding="utf-8"))
        summary["reproducible"]["same_source_tree"] = other_environment["source_tree_sha256"] == data["environment"]["source_tree_sha256"]
    return summary


def percent(value) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def number(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render(summary: dict) -> str:
    lines = [f"# Phase 5A expert oracle — {Path(summary['run']).name}", ""]
    gate_ = summary["gate"]
    lines += [
        f"**Gate: {gate_['verdict']}.** Best realistic strategy {gate_['best_strategy']}: mean {number(gate_['best_mean'])} of the routed "
        f"bytes, coverage {percent(gate_['best_coverage'])}. Best ideal (certified tier): {gate_['ideal_best_strategy']}, mean "
        f"{number(gate_['ideal_best_mean'])}.", "",
    ]
    c = summary["correctness"]
    lines += ["## Correctness", "", "| Check | Result |", "| --- | --- |"]
    lines.append(f"| Capture equals Phase 4B's reference (shared steps) | {c['phase4b_compared_steps']} steps, all equal: {c['phase4b_all_equal']} |")
    lines.append(f"| Target layer's weights equal the reference's rows | {c['weights_equal_reference_rows']} |")
    for key, value in c["reference_checks"].items():
        lines.append(f"| Reference recomputation: {key} | {value} |")
    for tier, values in c["tiers"].items():
        lines.append(f"| {tier}: enclosure violations (ceilings, realistic cells) | {values['ceiling_violations']}, {values['cell_violations']} |")
        lines.append(f"| {tier}: wrong (would-)certified tokens | {values['wrong_would_certify']} |")
    if "reproducible" in summary:
        lines.append(f"| Reproducible (digests; same source tree) | {summary['reproducible']['all']}; {summary['reproducible']['same_source_tree']} |")
    tiers = [t for t in summary["ceilings"] if t not in ("gaps", "gap_bins")]
    lines += ["", "## Ceilings: every routed weight read", "", "| Tier | Coverage | Margin median (logits) |", "| --- | --- | --- |"]
    for tier in tiers:
        values = summary["ceilings"][tier]
        lines.append(f"| {tier} | {percent(values['coverage'])} | {number(values['margin'].get('median'))} |")
    lines += ["", "Coverage by the reference's top-2 gap:", "", "| Gap (logits) | Samples | " + " | ".join(tiers) + " |", "| --- | --- |" + " --- |" * len(tiers)]
    for key, values in summary["ceilings"]["certified"]["by_gap"].items():
        row = [summary["ceilings"][t]["by_gap"].get(key, {}).get("coverage") for t in tiers]
        lines.append(f"| {key} | {values['n']} | " + " | ".join(percent(v) for v in row) + " |")
    lines += ["", "Floor of the tightest pair (certified tier, mean, logits):", ""]
    lines.append(", ".join(f"{k} {number(v)}" for k, v in summary["ceilings"]["certified"]["floor_terms_logits"].items()))
    for key, table in summary["tables"].items():
        lines += ["", f"## {key}", "", "| Strategy | Coverage | Mean | Median | p90 | p95 | When certified | Physical (layout: fraction, 4 KiB amplification) | Storage |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, row in table.items():
            physical = "; ".join(f"{scenario}: {number(v['fraction'].get('mean'))}, {number(v['amplification'])}" for scenario, v in row["physical"].items())
            lines.append(
                f"| {name} | {percent(row['coverage'])} | {number(row['fraction'].get('mean'))} | {number(row['fraction'].get('median'))} | "
                f"{number(row['fraction'].get('p90'))} | {number(row['fraction'].get('p95'))} | {number(row['fraction_certified'].get('mean'))} | "
                f"{physical} | {number(row['storage_multiplier'], 2)}x |"
            )
    lines += ["", "## Projection (Q6)", ""]
    for key, value in summary["projection"].items():
        if isinstance(value, dict):
            lines.append(f"- {key}: {value['strategy']}, fraction {number(value['fraction'])}: last layer only {number(value['last_layer_only_gb'], 2)} GB, "
                         f"every layer alike {number(value['every_layer_like_the_last_gb'], 2)} GB (baseline {summary['projection']['baseline_gb']} GB)")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--compare", default=None)
    args = parser.parse_args()
    run = Path(args.run)
    summary = summarize(run, Path(args.compare) if args.compare else None)
    (run / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (run / "summary.md").write_text(render(summary), encoding="utf-8")
    print(render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
