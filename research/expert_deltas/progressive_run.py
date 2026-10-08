"""Stage 5C2-B: the structural certification oracle on a small Moonlight sample set (Phase 5C). Real arithmetic: a
structural diagnostic, never a certified BF16 result.

    python -m uv run python research/expert_deltas/progressive_run.py --output experiments/phase5c/progressive-run1 [--shard i --shards n]
    python -m uv run python research/expert_deltas/progressive_run.py --output experiments/phase5c/progressive-run1 --report

Samples (config `progressive.selection`): two per top-2 gap bin among Phase 5A run1's samples of prompts 0..23, evenly
spaced in (prompt, step) order. Per sample:

  - Phase 5A's reference recomputed (`awpmi.oracle.experts.decode_sample`) and compared with the capture bitwise
  - the routed experts as bit-plane pages (16 rows, each plane a zstd-19 frame), decoded back and compared with the
    checkpoint's bytes (the fallback's reconstruction)
  - Phase 5A's realistic D-q6+q4 schedule in real arithmetic (Phase 5A2's `export.schedule`: Phase 5A's code), and at
    every state Phase 5A's realistic bound on the comparison pairs (the reference token against its 64 nearest rows),
    with the state's byte fraction (metadata included)
  - the bit-plane cells: each configured schedule with the exact box sets (minimum = the set's, witnesses where < 0),
    and the greedy schedule with structured metadata at each `sketch_budgets` (a sound relaxation, its metadata charged)
  - for each cell: checkpoints, the first certifying checkpoint (bisection on the nearest rows, then every row), the last
    witnessed flip, poisoning invariance, soundness counters

Writes samples.<shard>.jsonl.gz; `--report` aggregates them into summary.json, summary.md and digest.json with the gate
(config `gate_5c2`).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # summaries print Unicode on any console

from awpmi.runtime import configure_reproducible_numerics  # noqa: E402

NUMERICS = configure_reproducible_numerics()

from awpmi.tracing import JsonlWriter, read_jsonl  # noqa: E402
from expert_deltas import bits, records  # noqa: E402
from expert_deltas.oracle import CODEC, ExpertPages, Sample, capture_tokens, greedy, load_capture, run_sample, sequential  # noqa: E402
from expert_deltas.progressive import Sketch, sketched_minimum  # noqa: E402
from expert_deltas.source import Checkpoint  # noqa: E402

LAYER = 26


def phase5a2_export():
    """Phase 5A2's export module (runs in this environment): Phase 5A's realistic schedule and its real-tier bounds."""
    path = records.REPO_ROOT / "research" / "crown_expert_oracle" / "export.py"
    spec = importlib.util.spec_from_file_location("phase5a2_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select(settings: dict, run: Path) -> list[dict]:
    """Two samples per gap bin among Phase 5A run1's samples of the evaluation prompts, evenly spaced (5A2's rule)."""
    rule = settings["selection"]
    found = []
    shard = 0
    while (records.REPO_ROOT / settings["phase5a_run"] / f"samples.{shard}.jsonl.gz").exists():
        found.extend(read_jsonl(records.REPO_ROOT / settings["phase5a_run"] / f"samples.{shard}.jsonl.gz"))
        shard += 1
    low_prompt, high_prompt = rule["prompts"]
    pool = [{"prompt_id": r["prompt_id"], "step": r["step"], "gap": r["gap"], "token": r["token"]} for r in found
            if low_prompt <= r["prompt_id"] <= high_prompt]
    bins = [float(b) for b in rule["gap_bins"]]
    chosen = []
    for low, high in zip(bins, bins[1:]):
        members = sorted((r for r in pool if low <= r["gap"] < high), key=lambda r: (r["prompt_id"], r["step"]))
        per_bin = int(rule["per_bin"])
        if len(members) < per_bin:
            raise ValueError(f"gap bin [{low}, {high}) holds {len(members)} samples")
        for k in range(per_bin):
            chosen.append({**members[int((k + 0.5) * len(members) / per_bin)], "gap_bin": [low, high]})
    return chosen


def phase5a_states(export, sample, expert_layer, lm, gain, eps, budget_step: float, rows: torch.Tensor) -> list[dict]:
    """Phase 5A's realistic D-q6+q4 states in real arithmetic and its realistic bound on `rows` at each (its own code)."""
    from awpmi.oracle import experts as oracle

    strategy = oracle.Strategy("D-q6+q4", "refinement", spec="q6+q4")
    arithmetic = oracle.REAL_ARITHMETIC
    weights = oracle.SampleWeights(expert_layer, sample.experts)
    certifier = oracle.Certifier(lm, gain, sample.reference["logits"], arithmetic, near=64, chunk_rows=4096)
    states, _ = export.schedule(sample, weights, strategy, arithmetic, certifier, gain, eps, budget_step)
    routed = len(sample.experts) * expert_layer.expert_bytes
    out = []
    for states_k in states:
        bounds = oracle.propagate(sample, strategy.knowledge(weights, states_k), arithmetic, oracle.BoundTier.REALISTIC, gain, eps, weights.column_norms)
        certifier.candidate = sample.token
        parts = export.real_components(certifier, bounds, rows)
        read = strategy.bytes_read(expert_layer, states_k)
        spent = sum(int(read[k].sum()) for k in ("gate", "up", "down", "levels")) + strategy.metadata_bytes(expert_layer, len(sample.experts))
        out.append({"fraction": spent / routed, "realistic": parts["decomposed"].tolist()})
    return out


def attribution(sample: Sample, token: int, rows: torch.Tensor, truth: torch.Tensor, neuron_pages: int, down_pages: int) -> dict:
    """Where the uncertainty left by the last mantissa planes sits (exact box minima on the comparison rows): every page
    one or two planes short of exact, then only gate and up, then only down; and how concentrated the remaining box
    terms are over input columns (gate, up), output rows and neurons (down), the shares of the largest 10%. A block shape
    can only save what its blocks leave unread: spread-out terms leave no small set of blocks to skip."""
    from expert_deltas.progressive import exact_minimum

    slots = len(sample.patterns)
    out = {}
    for short in (1, 2):
        level = 8 - short
        full, part = np.full((slots, neuron_pages), 8), np.full((slots, neuron_pages), level)
        full_d, part_d = np.full((slots, down_pages), 8), np.full((slots, down_pages), level)
        entry = {}
        for name, (n_steps, d_steps) in {"all": (part, part_d), "gate_up_only": (part, full_d), "down_only": (full, part_d)}.items():
            sets = sample.sets(n_steps, d_steps, seed=None)
            found = exact_minimum(sets, sample.x, sample.base, sample.delta(token, rows))
            entry[name] = {"min_gap": float((truth - found.value).max()), "median_gap": float((truth - found.value).median()),
                           "rows_negative": int((found.value < 0).sum())}
        sets = sample.sets(part, part_d, seed=None)
        j = int(torch.argmin(truth))
        delta = sample.delta(token, rows[j : j + 1])[0]
        columns, out_rows, neurons = [], [], []
        for slot, s in enumerate(sets):
            g_terms = (s.gate.half * sample.x.abs()[None, :]).sum(dim=0) + (s.up.half * sample.x.abs()[None, :]).sum(dim=0)  # per column
            columns.append(sample.weights[slot] * g_terms)
            act = exact_minimum([s], sample.x, sample.base * 0, delta[None, :]).activations[0]
            reach = torch.maximum(act.a[0].abs(), act.a[1].abs())
            terms = delta.abs()[:, None] * s.down.half * reach[None, :]  # [H, I]
            out_rows.append(sample.weights[slot] * terms.sum(dim=1))
            neurons.append(sample.weights[slot] * terms.sum(dim=0))

        def top_share(values: torch.Tensor) -> float:
            v = values.reshape(-1).sort(descending=True).values
            return float(v[: max(1, v.numel() // 10)].sum() / v.sum())

        entry["top10_share"] = {"gate_up_columns": top_share(torch.stack(columns).sum(dim=0)),
                                "down_rows": top_share(torch.cat(out_rows)), "down_neurons": top_share(torch.cat(neurons))}
        out[f"planes_unread_{short}"] = entry
    return out


def bases(capture: dict, lm: torch.Tensor, gain: torch.Tensor, calibration: list[int], k_max: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The layer's shared bases from the calibration prompts only: U (top right singular vectors of x) and V (of Δ: each
    calibration sample's reference token against its 64 nearest rows by its captured h's logits)."""
    prompts = capture["prompt_id"].tolist()
    cal = [n for n, p in enumerate(prompts) if p in calibration]
    _, _, vx = torch.linalg.svd(capture["x"][cal].to(device, torch.float64), full_matrices=False)
    deltas = []
    for n in cal[:: max(1, len(cal) // 96)]:
        h = capture["h"][n].to(device)
        logits = torch.cat([lm[i : i + 16384] @ h for i in range(0, lm.shape[0], 16384)]).float()
        order = torch.argsort(logits, descending=True, stable=True)
        deltas.append((lm[order[0]].to(torch.float64)[None, :] - lm[order[1:65]].to(torch.float64)) * gain[None, :])
    _, _, vd = torch.linalg.svd(torch.cat(deltas), full_matrices=False)
    return vx[:k_max].t().contiguous(), vd[:k_max].t().contiguous()


def run(config: dict, output: Path, shard: int, shards: int) -> int:
    settings, model = config["progressive"], config["model"]
    threads = int(config["structure"]["threads"])
    device = torch.device("cuda", torch.cuda.current_device())
    export = phase5a2_export()
    capture, check = load_capture(settings)
    if not check["equal"]:
        records.write_json(output / "failure.json", {"kind": "capture", **check})
        return 1
    checkpoint = Checkpoint(model["repository"], model["revision"])
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.activations import ACT2FN
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RMSNorm

    from awpmi.oracle import experts as oracle

    def tensor(name: str) -> torch.Tensor:
        with safe_open(str(checkpoint.path(checkpoint.weight_map[name])), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    lm = tensor("lm_head.weight").to(device)
    norm_weight = tensor("model.norm.weight").to(device)
    gain = norm_weight.to(torch.float64)
    hf_config = AutoConfig.from_pretrained(model["repository"], revision=model["revision"])
    norm = DeepseekV3RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps).to(device=device, dtype=torch.bfloat16)
    norm.weight.data.copy_(norm_weight)
    layer_patterns = {m: checkpoint.matrices(LAYER, m) for m in ("gate", "up", "down")}
    gate_up = torch.cat([bits.to_bfloat16(layer_patterns["gate"]), bits.to_bfloat16(layer_patterns["up"])], dim=1).contiguous().to(device)
    expert_layer = oracle.ExpertLayer(gate_up, bits.to_bfloat16(layer_patterns["down"]).contiguous().to(device), ACT2FN[hf_config.hidden_act])
    tokens = capture_tokens(settings)
    calibration = list(range(int(settings["calibration_range"][0]), int(settings["calibration_range"][1]) + 1))
    budgets = [float(b) for b in settings["sketch_budgets"]]
    k_max = max(int(round(b * 1024)) for b in budgets)
    U_all, V_all = bases(capture, lm, gain, calibration, k_max, device)
    chosen = select(settings, output)
    mine = chosen[shard::shards]
    prompts, steps = capture["prompt_id"].tolist(), capture["step"].tolist()
    started_all = time.perf_counter()
    with JsonlWriter(output / f"samples.{shard}.jsonl.gz") as log:
        for choice in mine:
            started = time.perf_counter()
            n = next(i for i in range(len(prompts)) if (prompts[i], steps[i]) == (choice["prompt_id"], choice["step"]))
            t = {k: capture[k][n].to(device) for k in ("x", "r", "S", "R", "m", "y", "h", "top_k_weights")}
            experts = capture["top_k_index"][n].tolist()
            sample5a = oracle.decode_sample(expert_layer, t["x"], t["r"], t["S"], experts, t["top_k_weights"], norm, lm)
            bitwise = {k: bool(torch.equal(sample5a.reference[k], t[k])) for k in ("R", "m", "y", "h")}
            bitwise["token"] = sample5a.token == tokens[(choice["prompt_id"], choice["step"])]
            entry = {**choice, "experts": experts, "bitwise": bitwise}
            if not all(bitwise.values()):
                entry["failure"] = "reference"
                log.write(entry)
                continue
            pages = [ExpertPages({m: layer_patterns[m][e] for m in ("gate", "up", "down")}, threads) for e in experts]
            decoded = [p.decode(threads) for p in pages]
            entry["pages_exact"] = all(np.array_equal(d[m], layer_patterns[m][e]) for d, e in zip(decoded, experts) for m in ("gate", "up", "down"))
            sample = Sample(capture["x"][n], capture["r"][n], capture["S"][n], experts, capture["top_k_weights"][n].tolist(), pages, lm, norm_weight, device)
            logits = sample5a.reference["logits"].to(torch.float32)
            order = torch.argsort(logits, descending=True, stable=True)
            rows = order[order != sample5a.token][:64]
            truth = sample.truth(rows, sample5a.token)
            comparison = [{"rows": rows.tolist(), "truth": truth.tolist(), **s}
                          for s in phase5a_states(export, sample5a, expert_layer, lm, gain, norm.variance_epsilon, float(settings["budget_step"]), rows)]
            comparison.sort(key=lambda s: s["fraction"])
            from expert_deltas.compression import measure

            alone = {e: {m: layer_patterns[m][e] for m in ("gate", "up", "down")} for e in experts}
            independent = measure(alone, alone, "expert", "planes", CODEC, threads)["stored_bytes"]
            routed = len(experts) * expert_layer.expert_bytes
            linf_bytes = sum(4 * (p.neurons * 2 + p.hidden) for p in pages)
            entry.update(token=sample5a.token, real_winner_agrees=bool((truth > 0).all()), routed_bytes=routed, best_independent_bytes=independent,
                         representation_bytes=sum(p.total for p in pages), phase5a_states=len(comparison),
                         phase5a_realistic_certifies_at=next((s["fraction"] for s in comparison if min(s["realistic"]) > 0), None), cells=[])
            entry["attribution"] = attribution(sample, sample5a.token, rows, truth, pages[0].neuron_planes.shape[0], pages[0].down_planes.shape[0])
            for name in settings["schedules"]:
                schedule = sequential(pages) if name == "sequential" else greedy(pages, sample)
                cell = run_sample(sample, pages, schedule, name, comparison, sample5a.token, settings, routed, linf_bytes, independent)
                cell["metadata"] = "linf"
                entry["cells"].append(cell)
            schedule = greedy(pages, sample)
            truths = [{m: bits.pattern_values(mats[m]) for m in mats} for mats in sample.patterns]
            for budget in budgets:
                k = int(round(budget * 1024))
                sketch = Sketch.build(U_all[:, :k], V_all[:, :k], truths)
                bases_bytes = 4 * (U_all.shape[0] * k * 2)  # U and V, float32, per layer (charged to the token in full)
                charged = linf_bytes + sketch.nbytes + bases_bytes

                def minimum(sets, x, base, delta, sketch=sketch):
                    return sketched_minimum(sets, x, base, delta, sketch)

                cell = run_sample(sample, pages, schedule, "greedy", comparison, sample5a.token, settings, routed, charged, independent,
                                  minimum=minimum, exact_sets=False)
                cell.update(metadata=f"sketch-{budget}", sketch_k=k, sketch_bytes=sketch.nbytes, bases_bytes=bases_bytes)
                entry["cells"].append(cell)
            entry["timings"] = {"sample_s": time.perf_counter() - started}
            log.write(entry)
            for cell in entry["cells"]:
                cert = cell["certificate"]
                print(f"({choice['prompt_id']}, {choice['step']}) gap {choice['gap']:.3f} {cell['schedule']}/{cell['metadata']}: "
                      f"{'certified at f_indep %.4f (f_raw %.4f)' % (cert['f_best_independent'], cert['f_raw']) if cert else 'never'}; "
                      f"last flip {cell['last_flip_f_raw']}", flush=True)
            del sample
            torch.cuda.empty_cache()
    from awpmi.storage.fileio import process_memory

    records.write_json(output / f"stage_{shard}.json", {"shard": shard, "shards": shards, "samples": [[c["prompt_id"], c["step"]] for c in mine],
                                                          "system": {"process_memory": process_memory(),
                                                                     "peak_device_bytes": torch.cuda.max_memory_allocated(device)},
                                                          "timings": {"total_s": time.perf_counter() - started_all}})
    return 0


# Report and gate


def report(config: dict, output: Path) -> dict:
    gate = config["gate_5c2"]
    entries = []
    shard = 0
    while (output / f"samples.{shard}.jsonl.gz").exists():
        entries.extend(read_jsonl(output / f"samples.{shard}.jsonl.gz"))
        shard += 1
    entries.sort(key=lambda e: (e["gap_bin"][0], e["prompt_id"], e["step"]))
    cells: dict[str, list[dict]] = {}
    for e in entries:
        for c in e["cells"]:
            cells.setdefault(f"{c['schedule']}/{c['metadata']}", []).append({**c, "sample": f"{e['prompt_id']}/{e['step']}", "gap": e["gap"]})
    violations = sum(1 for e in entries if not all(e["bitwise"].values()) or not e.get("pages_exact", False))
    for group in cells.values():
        for c in group:
            violations += sum(c["violations"].values()) + (0 if c["poisoning_invariant"] else 1)
    per_cell = {}
    for name, group in cells.items():
        fractions, raw, coverage, shares = [], [], 0, []
        for c in group:
            cert = c["certificate"]
            if cert is not None and cert["f_representation"] < 1.0:
                coverage += 1
            charged = cert["f_best_independent"] if cert else c["checkpoints"][-1]["f_best_independent"]
            fractions.append(charged)
            raw.append(cert["f_raw"] if cert else c["checkpoints"][-1]["f_raw"])
            middle = [p["distance_share_median"] for p in c["checkpoints"] if 0.38 <= p["f_raw"] <= 1.0]
            shares.append(float(np.median(middle)) if middle else None)
        per_cell[name] = {
            "samples": len(group), "structural_coverage": coverage / len(group), "mean_f_best_independent": float(np.mean(fractions)),
            "mean_f_raw": float(np.mean(raw)), "median_distance_share": float(np.median([s for s in shares if s is not None])),
            "certified": [{"sample": c["sample"], "gap": c["gap"], "f_best_independent": c["certificate"]["f_best_independent"] if c["certificate"] else None,
                           "f_representation": c["certificate"]["f_representation"] if c["certificate"] else None,
                           "candidate": c["certificate"]["candidate"] if c["certificate"] else None,
                           "last_flip_f_raw": c["last_flip_f_raw"], "last_flip_f_representation": c.get("last_flip_f_representation")} for c in group],
        }
        g = per_cell[name]
        certified = [c["certificate"] for c in group if c["certificate"] is not None and "physical" in c["certificate"]]
        g["physical_at_certificate"] = {
            layout: {"amplification": float(np.mean([x["physical"][layout]["physical_bytes"] / max(1, x["physical"][layout]["logical_bytes"]) for x in certified])),
                     "extents_per_token": float(np.mean([x["physical"][layout]["extents"] for x in certified]))}
            for layout in ("page_major", "plane_major")
        } if certified else None
        g["structural_pass"] = bool(violations == 0 and g["structural_coverage"] >= gate["min_structural_coverage"]
                                    and g["mean_f_best_independent"] <= gate["max_mean_fraction_of_best_independent"]
                                    and g["median_distance_share"] <= gate["max_distance_share_vs_phase5a"])
        g["strong"] = bool(g["structural_pass"] and g["mean_f_raw"] <= gate["strong_max_mean_fraction_of_bf16"] and g["mean_f_best_independent"] < 1.0)
    phase5a = [{"sample": f"{e['prompt_id']}/{e['step']}", "gap": e["gap"], "realistic_certifies_at": e.get("phase5a_realistic_certifies_at")} for e in entries]
    attributions = {}
    for short in ("planes_unread_1", "planes_unread_2"):
        found = [e["attribution"][short] for e in entries if "attribution" in e]
        if found:
            attributions[short] = {
                "samples_still_flipping": {k: sum(1 for f in found if f[k]["rows_negative"] > 0) for k in ("all", "gate_up_only", "down_only")},
                "median_gap": {k: float(np.median([f[k]["median_gap"] for f in found])) for k in ("all", "gate_up_only", "down_only")},
                "top10_share": {k: float(np.median([f["top10_share"][k] for f in found])) for k in ("gate_up_columns", "down_rows", "down_neurons")},
            }
    summary = {"samples": [{k: e[k] for k in ("prompt_id", "step", "gap", "gap_bin", "experts", "token")} for e in entries],
               "violations": violations, "cells": per_cell, "phase5a_realistic": phase5a, "attribution": attributions,
               "verdict": "STRUCTURAL PASS" if any(g["structural_pass"] for g in per_cell.values()) else "STRUCTURAL FAIL"}
    summary["digest"] = records.digest({"samples": entries})
    records.write_json(output / "summary.json", summary)
    lines = ["# Phase 5C2-B: structural certification oracle (real arithmetic; NOT a certified BF16 result)", "",
             f"Samples: {len(entries)}; soundness violations: {violations}; verdict: **{summary['verdict']}**.", "",
             "| Cell | coverage (certified with bytes unread) | mean bytes / best independent | mean bytes / BF16 | median distance share vs Phase 5A | pass |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, g in per_cell.items():
        lines.append(f"| {name} | {g['structural_coverage']:.3f} | {g['mean_f_best_independent']:.4f} | {g['mean_f_raw']:.4f} | {g['median_distance_share']:.3f} | {g['structural_pass']} |")
    lines += ["", "Modelled 4 KiB reads at the certifying state (amplification = physical / logical bytes; extents per token):", ""]
    for name, g in per_cell.items():
        if g["physical_at_certificate"]:
            lines.append(f"- {name}: " + "; ".join(f"{layout} {v['amplification']:.3f}, {v['extents_per_token']:.0f} extents"
                                                   for layout, v in g["physical_at_certificate"].items()))
    lines += ["", "Per sample (best independent fraction at certification; − never):", ""]
    for name, g in per_cell.items():
        lines.append(f"- {name}: " + ", ".join(f"{c['sample']} (gap {c['gap']:.2f}) {c['f_best_independent']:.3f}" if c["f_best_independent"] else f"{c['sample']} (gap {c['gap']:.2f}) −"
                                          for c in g["certified"]))
    lines += ["", "Uncertainty left by the last planes (comparison rows; samples whose set still flips a pair, median gap truth − minimum; "
              "share of the box terms in the largest 10% of columns, down rows, neurons):", ""]
    for short, a in attributions.items():
        lines.append(f"- {short}: flipping {a['samples_still_flipping']}, median gap {json.dumps({k: round(v, 3) for k, v in a['median_gap'].items()})}, "
                     f"top-10% shares {json.dumps({k: round(v, 3) for k, v in a['top10_share'].items()})}")
    lines += ["", "Phase 5A realistic (D-q6+q4, real tier) decides the comparison pairs from (fraction of BF16): " +
              ", ".join(f"{p['sample']} {p['realistic_certifies_at']:.3f}" if p["realistic_certifies_at"] else f"{p['sample']} never" for p in phase5a), ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    records.write_json(output / "digest.json", {"samples": summary["digest"], "excluded_fields": list(records.EXCLUDED_FIELDS)})
    print("\n".join(lines))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(records.REPO_ROOT / "configs" / "phase5c-expert-deltas.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--list", action="store_true", help="print the selected samples and exit")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.yaml").exists():
        config = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
    else:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        records.write_json(output / "environment.json", records.environment(config["model"], NUMERICS))
    if args.list:
        for c in select(config["progressive"], output):
            print(json.dumps(c))
        return 0
    if args.report:
        report(config, output)
        return 0
    return run(config, output, args.shard, args.shards)


if __name__ == "__main__":
    raise SystemExit(main())
