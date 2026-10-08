"""Stage 5C2-A: the progressive feasibility probe (Phase 5C). A structural diagnostic in real arithmetic.

    python -m uv run python research/expert_deltas/progressive_probe.py --output experiments/phase5c/progressive-probe

On Phase 5A2's three samples (config `progressive.probe_samples`: an exact tie, a gap of 3.88, a gap of 11.06) of
Moonlight's last MoE layer at decode, upstream exact:

  1. the capture's sha256 against Phase 5A run1's digest; the layer read from the checkpoint; Phase 5A's reference
     (`awpmi.oracle.experts.decode_sample`) recomputed and compared with the capture bitwise (R, m, y, h, the token)
  2. the routed experts stored as bit-plane pages (`expert_deltas.progressive`: 16-row pages, each plane a zstd-19
     frame); fallback parity: every page decoded and merged equals the checkpoint's bytes, and the reference recomputed
     with the decoded experts equals the capture bitwise
  3. a toy cut from the real weights (a few neurons and inputs): the closed-form optimum against enumeration
  4. structured metadata's precondition: how much of the experts' input x, and of the decision directions Δ, a
     calibration subspace (prompts disjoint from the samples') captures, at the metadata budgets
  5. per sample and schedule (sequential plane-major; realistic greedy after every page's sign and exponent): the state
     at every checkpoint (multiples of 1/64 of the routed BF16 bytes); the exact minimum of Δ·y over the set for Phase
     5A2's 64 comparison rows, against the truth and Phase 5A's realistic bound at a matched (larger or equal) byte
     fraction; a witness (weights in the set, by the real forward) wherever the minimum is negative; the first state
     whose certificate holds (candidate = the centre's argmax; its 64 nearest rows by centre logits by bisection, then
     every vocabulary row); poisoning checks of the unread bits
  6. the stage's gate (config `gate_5c2`, probe rule)

Writes progressive_probe.json and progressive_probe.md.
"""

from __future__ import annotations

import argparse
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

from expert_deltas import bits, records  # noqa: E402
from expert_deltas.oracle import (  # noqa: E402
    CODEC,
    ExpertPages,
    Sample,
    anisotropy,
    capture_tokens,
    comparison_records,
    greedy,
    load_capture,
    run_sample,
    sequential,
    toy_check,
)
from expert_deltas.source import Checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(records.REPO_ROOT / "configs" / "phase5c-expert-deltas.yaml"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    settings, model = config["progressive"], config["model"]
    threads = int(config["structure"]["threads"])
    device = torch.device("cuda", torch.cuda.current_device())
    started_all = time.perf_counter()
    result: dict = {"stage": "5C2-A", "config": settings, "environment": records.environment(model, NUMERICS), "timings": {}}
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    capture, check = load_capture(settings)
    result["capture"] = check
    if not check["equal"]:
        records.write_json(output / "failure.json", {"kind": "capture", **check})
        return 1
    layer_name = "model.layers.26"
    layer = int(layer_name.split(".")[-1])
    checkpoint = Checkpoint(model["repository"], model["revision"])
    from safetensors import safe_open

    def tensor(name: str) -> torch.Tensor:
        with safe_open(str(checkpoint.path(checkpoint.weight_map[name])), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    lm = tensor("lm_head.weight").to(device)
    norm_weight = tensor("model.norm.weight").to(device)
    gain = norm_weight.to(torch.float64)

    # 1. Phase 5A's reference, recomputed on the whole layer (as read from the checkpoint) and compared with the capture.
    from transformers import AutoConfig
    from transformers.activations import ACT2FN
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RMSNorm

    from awpmi.oracle import experts as oracle

    hf_config = AutoConfig.from_pretrained(model["repository"], revision=model["revision"])
    experts_count = checkpoint.experts(layer)
    layer_patterns = {m: checkpoint.matrices(layer, m) for m in ("gate", "up", "down")}

    def layer_tensors(patterns: dict[str, np.ndarray]):
        gate_up = torch.cat([bits.to_bfloat16(patterns["gate"]), bits.to_bfloat16(patterns["up"])], dim=1).contiguous().to(device)
        return gate_up, bits.to_bfloat16(patterns["down"]).contiguous().to(device)

    norm = DeepseekV3RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps).to(device=device, dtype=torch.bfloat16)
    norm.weight.data.copy_(norm_weight)
    act_fn = ACT2FN[hf_config.hidden_act]
    gate_up, down = layer_tensors(layer_patterns)
    expert_layer = oracle.ExpertLayer(gate_up, down, act_fn)
    tokens = capture_tokens(settings)
    comparisons = comparison_records(settings)
    prompts, steps = capture["prompt_id"].tolist(), capture["step"].tolist()
    chosen = [next(n for n in range(len(prompts)) if (prompts[n], steps[n]) == tuple(s)) for s in settings["probe_samples"]]
    references = {}
    for n in chosen:
        t = {k: capture[k][n].to(device) for k in ("x", "r", "S", "R", "m", "y", "h", "top_k_weights")}
        sample = oracle.decode_sample(expert_layer, t["x"], t["r"], t["S"], capture["top_k_index"][n].tolist(), t["top_k_weights"], norm, lm)
        bitwise = {k: bool(torch.equal(sample.reference[k], t[k])) for k in ("R", "m", "y", "h")}
        bitwise["token"] = sample.token == tokens[(prompts[n], steps[n])]
        references[n] = {"sample": sample, "bitwise": bitwise}
    result["reference_bitwise"] = {f"{prompts[n]}/{steps[n]}": references[n]["bitwise"] for n in chosen}
    if not all(all(r["bitwise"].values()) for r in references.values()):
        records.write_json(output / "failure.json", {"kind": "reference", "checks": result["reference_bitwise"]})
        return 1

    # 2. The routed experts as bit-plane pages; fallback parity.
    started = time.perf_counter()
    needed = sorted({e for n in chosen for e in capture["top_k_index"][n].tolist()})
    stored = {e: ExpertPages({m: layer_patterns[m][e] for m in ("gate", "up", "down")}, threads) for e in needed}
    decoded = {e: stored[e].decode(threads) for e in needed}
    pages_exact = all(np.array_equal(decoded[e][m], layer_patterns[m][e]) for e in needed for m in ("gate", "up", "down"))
    rebuilt = {m: layer_patterns[m].copy() for m in ("gate", "up", "down")}
    for e in needed:
        for m in ("gate", "up", "down"):
            rebuilt[m][e] = decoded[e][m]
    gate_up2, down2 = layer_tensors(rebuilt)
    parity = {}
    for n in chosen:
        t = {k: capture[k][n].to(device) for k in ("x", "r", "S", "R", "m", "y", "h", "top_k_weights")}
        again = oracle.decode_sample(oracle.ExpertLayer(gate_up2, down2, act_fn), t["x"], t["r"], t["S"], capture["top_k_index"][n].tolist(),
                                     t["top_k_weights"], norm, lm)
        parity[f"{prompts[n]}/{steps[n]}"] = all(bool(torch.equal(again.reference[k], t[k])) for k in ("R", "m", "y", "h")) and again.token == references[n]["sample"].token
    del gate_up2, down2
    result["fallback"] = {"pages_decode_to_checkpoint_bytes": pages_exact, "reference_from_decoded_pages": parity,
                          "experts": needed, "bytes": {str(e): stored[e].total for e in needed}}
    result["timings"]["pages_s"] = time.perf_counter() - started
    if not pages_exact or not all(parity.values()):
        records.write_json(output / "failure.json", {"kind": "fallback", **result["fallback"]})
        return 1
    del gate_up, down, expert_layer
    torch.cuda.empty_cache()

    # 3. A toy cut from the real weights.
    result["toy"] = toy_check({m: layer_patterns[m][needed[0]] for m in ("gate", "up", "down")}, capture["x"][chosen[0]].to(torch.float64), 3)

    # 4. Structured metadata's precondition.
    calibration = list(range(int(settings["calibration_range"][0]), int(settings["calibration_range"][1]) + 1))
    widths = torch.stack([bits.prefix_interval(torch.from_numpy(layer_patterns["gate"][e].astype(np.int64)), 9)[1]
                          - bits.prefix_interval(torch.from_numpy(layer_patterns["gate"][e].astype(np.int64)), 9)[0] for e in needed[:4]]).mean(dim=(0, 1)).to(device) * 0.5
    result["anisotropy"] = anisotropy(capture, lm, gain, calibration, chosen, [float(b) for b in settings["metadata_budgets"]], widths, device)

    # 5. Per sample and schedule.
    routed_bytes = 6 * 3 * layer_patterns["gate"][0].nbytes
    result["samples"] = []
    for n in chosen:
        started = time.perf_counter()
        experts = capture["top_k_index"][n].tolist()
        pages = [stored[e] for e in experts]
        sample = Sample(capture["x"][n], capture["r"][n], capture["S"][n], experts, capture["top_k_weights"][n].tolist(), pages, lm, norm_weight, device)
        metadata = sum(4 * (p.neurons * 2 + p.hidden) for p in pages)  # the resident L∞ bound of every row
        # The best independent exact baseline for these experts: each compressed alone, bit planes, zstd-19 (5C1's best).
        from expert_deltas.compression import measure

        alone = {e: {m: layer_patterns[m][e] for m in ("gate", "up", "down")} for e in experts}
        independent = measure(alone, alone, "expert", "planes", CODEC, threads)
        key = (prompts[n], steps[n])
        token = references[n]["sample"].token
        entry = {"prompt_id": prompts[n], "step": steps[n], "experts": experts, "token": token, "metadata_bytes": metadata,
                 "routed_bytes": routed_bytes, "best_independent_bytes": independent["stored_bytes"],
                 "representation_bytes": sum(p.total for p in pages), "schedules": []}
        for name, schedule in (("sequential", sequential(pages)), ("greedy", greedy(pages, sample))):
            found = run_sample(sample, pages, schedule, name, comparisons[key], token, settings, routed_bytes, metadata, independent["stored_bytes"])
            entry["schedules"].append(found)
            cert = found["certificate"]
            print(f"sample {key} {name}: certified {'at f_raw %.3f (f_indep %.3f)' % (cert['f_raw'], cert['f_best_independent']) if cert else 'never'}; "
                  f"last flip at {found['last_flip_f_raw']}", flush=True)
        entry["timings"] = {"sample_s": time.perf_counter() - started}
        result["samples"].append(entry)
        del sample
        torch.cuda.empty_cache()

    from awpmi.storage.fileio import process_memory

    result["system"] = {"process_memory": process_memory(), "peak_device_bytes": torch.cuda.max_memory_allocated(device)}
    result["gate"] = probe_gate(result, config["gate_5c2"])
    result["timings"]["total_s"] = time.perf_counter() - started_all
    result["digest"] = records.digest({k: v for k, v in result.items() if k != "environment"})
    records.write_json(output / "progressive_probe.json", result)
    (output / "progressive_probe.md").write_text(summary(result), encoding="utf-8")
    print(summary(result))
    return 0


def probe_gate(result: dict, gate: dict) -> dict:
    """The probe's rule (config gate_5c2): no violation; some sample with a margin certifies with at most the configured
    fraction of the best independent exact bytes, its sets materially tighter than Phase 5A's at matched bytes."""
    violations = sum(sum(s["violations"].values()) for e in result["samples"] for s in e["schedules"])
    violations += sum(1 for e in result["samples"] for s in e["schedules"] if not s["poisoning_invariant"])
    violations += 0 if result["toy"]["all_equal"] else 1
    best = []
    for e in result["samples"]:
        for s in e["schedules"]:
            cert = s["certificate"]
            shares = [c["distance_share_median"] for c in s["checkpoints"] if c["f_raw"] >= 0.38 and c["f_raw"] <= 1.0]
            best.append({"sample": f"{e['prompt_id']}/{e['step']}", "schedule": s["schedule"],
                         "certified_f_best_independent": None if cert is None else cert["f_best_independent"],
                         "certified_f_raw": None if cert is None else cert["f_raw"],
                         "median_distance_share": float(np.median(shares)) if shares else None, "last_flip_f_raw": s["last_flip_f_raw"]})
    proceed = any(
        b["certified_f_best_independent"] is not None and b["certified_f_best_independent"] <= gate["probe_proceed_max_fraction_of_best_independent"]
        and b["median_distance_share"] is not None and b["median_distance_share"] <= gate["max_distance_share_vs_phase5a"]
        for b in best
    )
    return {"violations": violations, "per_schedule": best, "proceed_to_5C2B": bool(violations == 0 and proceed)}


def summary(result: dict) -> str:
    lines = ["# Phase 5C2-A: progressive feasibility probe (real arithmetic; structural diagnostic, not a certified BF16 result)", ""]
    lines.append(f"Capture equal to Phase 5A run1: {result['capture']['equal']}. Reference recomputed bitwise: "
                 f"{all(all(v.values()) for v in result['reference_bitwise'].values())}. Pages decode to the checkpoint's bytes: "
                 f"{result['fallback']['pages_decode_to_checkpoint_bytes']}; reference from decoded pages: {result['fallback']['reference_from_decoded_pages']}.")
    lines.append(f"Toy cut from real weights, closed form = enumeration: {result['toy']['all_equal']}.")
    lines += ["", "## Structured metadata precondition (share outside a calibration subspace)", ""]
    for budget, v in result["anisotropy"]["x"].items():
        d = result["anisotropy"]["delta"][budget]
        lines.append(f"- budget {budget}: x (k={v['k']}): energy {v['energy_outside']:.3f}, box mass {v['box_mass_outside']:.3f}; "
                     f"Δ (k={d['k']}): energy {d['energy_outside']:.3f}, ℓ1 {d['l1_outside']:.3f}")
    for e in result["samples"]:
        lines += ["", f"## Sample prompt {e['prompt_id']} step {e['step']} (experts {e['experts']})", "",
                  f"Routed BF16 {e['routed_bytes']}, bit-plane pages {e['representation_bytes']} ({e['representation_bytes'] / e['routed_bytes']:.4f}), "
                  f"best independent {e['best_independent_bytes']} ({e['best_independent_bytes'] / e['routed_bytes']:.4f}), metadata {e['metadata_bytes']}."]
        for s in e["schedules"]:
            cert = s["certificate"]
            lines += ["", f"### {s['schedule']}: certificate {'at f_raw %.4f, f_indep %.4f, f_rep %.4f' % (cert['f_raw'], cert['f_best_independent'], cert['f_representation']) if cert else 'never'}; "
                      f"last witnessed flip at f_raw {s['last_flip_f_raw']}; poisoning invariant {s['poisoning_invariant']}; violations {s['violations']}", "",
                      "| f_raw | f_indep | min Δ·y over the set (64 rows) | truth min | Phase 5A realistic min (at f) | median distance share | rows < 0 |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for c in s["checkpoints"][:: max(1, len(s["checkpoints"]) // 16)]:
                lines.append(f"| {c['f_raw']:.4f} | {c['f_best_independent']:.4f} | {c['min_over_rows']:.4g} | {c['truth_min']:.4g} | "
                             f"{c['phase5a_realistic_min']:.4g} ({c['phase5a_fraction']:.3f}) | {c['distance_share_median']:.3f} | {c['rows_negative']} |")
    lines += ["", "## Gate (probe rule)", "", f"```\n{json.dumps(result['gate'], indent=1)}\n```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
