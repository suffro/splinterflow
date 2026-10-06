"""Phase 5A2: the CROWN oracle's inputs, exported from the Phase 5A capture (decision 0010).

Runs in the SPLINTERFLOW environment (the repository root's: torch 2.14, awpmi), not in the verifier's:

    PYTHONHASHSEED=1 uv run python research/crown_expert_oracle/export.py --output experiments/phase5a2/<name>

It writes everything the isolated verifier needs, and nothing it does not, into `<output>/artifact/`:

  weights.safetensors     per routed expert of the selected samples (`expert.<e>.<gate|up|down>.*`): the BF16 rows (the
                          truth: exact rows, validation), the q6 and q4 levels (int8 codes, float32 scales; decision 0003),
                          each level's remainder norms and the rows' own norms (resident metadata); the LM head and the final
                          norm's gain; the certificate's constants (`mu`)
  sample.<n>.safetensors  per sample: x, r, S, the routing weights, the reference's a (BF16) and the real forward's a and y
                          (float64), the reference's logits; per strategy and arithmetic tier, the activation enclosures of
                          every row level and, per budget index of Phase 5A's realistic schedule, the rows' levels and Phase
                          5A's enclosures and named error terms at that state; Phase 5A's bounds on the comparison pairs
  manifest.json           provenance, constants, samples, per-state scalars (candidate, nearest rows, bytes), Phase 5A's
                          cells on the same samples (the baseline), and the sha256 of every file

Everything Phase 5A already computes is computed by Phase 5A's code (`awpmi.oracle.experts`, `awpmi.bounds`): the
reference recomputation (checked bitwise against the capture), the decompositions, the realistic schedules, the bytes,
the enclosures and the pairwise certificate. The capture must equal Phase 5A run1's (its digests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from awpmi.runtime import configure_reproducible_numerics  # noqa: E402

NUMERICS_FLAGS = configure_reproducible_numerics()

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import expert_oracle as phase5a  # noqa: E402  (benchmarks/expert_oracle.py: the target's loader and checks)
from awpmi.bounds.floating import FLOAT64_UNIT_ROUNDOFF, gamma  # noqa: E402
from awpmi.bounds.operators import FP32_NORMAL_MIN  # noqa: E402
from awpmi.bounds.pairwise import exact_projection, mixture_bound, pairwise_certificate  # noqa: E402
from awpmi.bounds.residual import ReferenceNumerics  # noqa: E402
from awpmi.bounds.rounding import CERTIFIED, relative_rounding_error, rounding_error_upper, subnormal_floor  # noqa: E402
from awpmi.oracle import experts as oracle  # noqa: E402
from awpmi.tracing import canonical_digest, environment_metadata, read_jsonl, sha256_file, source_tree_sha256  # noqa: E402

FORMAT = "phase5a2-crown-artifact/1"
TIERS = {"certified": oracle.CERTIFIED_ARITHMETIC, "real": oracle.REAL_ARITHMETIC}
RESEARCH_TREE = Path("research") / "crown_expert_oracle"
RESEARCH_EXCLUDED = (".venv", "__pycache__", ".pytest_cache")


class ExportFailure(Exception):
    pass


def research_tree_sha256(repo_root: Path) -> str:
    """The verifier's code, lockfile and configuration (`research/crown_expert_oracle`, no environment or caches); the
    same rule as `source_tree_sha256`, which the verifier repeats (`crown_oracle.artifact.tree_sha256`)."""
    digest = hashlib.sha256()
    root = repo_root / RESEARCH_TREE
    files = sorted(p for p in root.rglob("*") if p.is_file() and not any(part in RESEARCH_EXCLUDED for part in p.relative_to(root).parts))
    for path in files:
        digest.update(path.relative_to(repo_root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


# Sample selection (fixed by the config before any CROWN run on real data)


def select_samples(records: list[dict], settings: dict) -> list[dict]:
    """Per gap bin, `per_bin` samples evenly spaced in (prompt_id, step) order; certifiable ones first from a gap on."""
    bins, per_bin = settings["gap_bins"], int(settings["per_bin"])
    chosen = []
    for low, high in zip(bins, bins[1:]):
        members = sorted((r for r in records if low <= r["gap"] < high), key=lambda r: (r["prompt_id"], r["step"]))
        if low >= float(settings["prefer_certifiable_from"]):
            certifiable = [r for r in members if r["ceilings"]["certified"]["certified"]]
            if len(certifiable) >= per_bin:
                members = certifiable
        if len(members) < per_bin:
            raise ExportFailure(f"gap bin [{low}, {high}) holds {len(members)} samples, fewer than {per_bin}")
        for k in range(per_bin):
            chosen.append({**members[int((k + 0.5) * len(members) / per_bin)], "gap_bin": [low, high]})
    return chosen


# Phase 5A's realistic schedule (run_cell's first steps, repeated here to expose every state)


def schedule(sample, weights, strategy, arithmetic, certifier, gain, eps, budget_step):
    """Phase 5A's realistic ordering of `strategy` in `arithmetic` (from its first state's tightest pair) and its budget
    grid: the initial state, and the state at each budget index (`run_cell`'s `state_at`)."""
    layer, device = weights.layer, sample.x.device
    routed = len(sample.experts) * layer.expert_bytes
    initial = strategy.initial(layer, len(sample.experts), device)
    bounds = oracle.propagate(sample, strategy.knowledge(weights, initial), arithmetic, oracle.BoundTier.REALISTIC, gain, eps, weights.column_norms)
    outcome = certifier.evaluate(bounds, full=True)
    units = strategy.units(layer, initial)
    contender = outcome.tightest if outcome.tightest >= 0 else sample.runner_up
    delta = oracle.pair_delta(certifier.lm_weight, gain, outcome.candidate, contender)
    gains = oracle.realistic_gains(units, strategy, weights, bounds, sample.weights, delta)
    ordered = oracle.order_units(units, gains, max(1, len(strategy.level_bits)))
    spent = torch.cumsum(ordered.nbytes, dim=0)
    total = int(spent[-1]) if len(ordered) else 0
    steps = math.ceil(total / (budget_step * routed)) if total else 0

    def state_at(index: int):
        if index < 0:
            return initial
        limit = min((index + 1) * budget_step * routed, total)
        count = int((spent <= limit + 0.5).sum())
        states = initial.copy()
        strategy.apply(states, ordered.take(torch.arange(count, device=device)))
        return states

    return [state_at(index) for index in range(-1, steps)], outcome


def levels_of(strategy) -> list[int]:
    """The row states a strategy uses, in order (UNKNOWN = −1, a level index, EXACT = 99)."""
    if strategy.family == "refinement":
        return [*range(len(strategy.level_bits)), oracle.EXACT]
    return [oracle.UNKNOWN, oracle.EXACT]


# Phase 5A's bounds on fixed pairs (the comparison) and the certificate's per-state inputs


def real_components(certifier, bounds, rows):
    """The real tier's pairwise lower bounds on Δ·y for `rows` (Certifier._real_margins' terms, separately)."""
    gain = certifier.norm_weight.to(torch.float64)
    winner = certifier.lm_weight[certifier.candidate].to(torch.float64)
    delta = (winner[None, :] - certifier.lm_weight.index_select(0, rows).to(torch.float64)) * gain[None, :]
    centre = bounds.y.center
    box = delta @ centre - delta.abs() @ bounds.y.radius_about(centre)
    part = mixture_bound(delta, bounds.mixture, exact_projection)
    decomposed = part.exact_and_read - part.unread
    absolute = delta.abs() @ (bounds.y.magnitude + centre.abs()) + part.absolute
    slack = 4.0 * gamma(2 * (delta.shape[1] + part.neurons) + 64, FLOAT64_UNIT_ROUNDOFF) * absolute
    return {"box": box, "decomposed": decomposed, "slack": slack, "margin": torch.maximum(box, decomposed) - slack}


def certified_components(certifier, bounds, rows):
    """The certified tier's pairwise certificate of `certifier.candidate` against `rows` (Certifier.margins' call)."""
    candidate = torch.tensor([certifier.candidate], device=rows.device)
    block = torch.cat([candidate, rows])
    magnitudes = certifier._magnitudes(bounds, block)
    result = pairwise_certificate(
        certifier.candidate, block, certifier.lm_weight.index_select(0, block), -magnitudes, magnitudes, certifier.norm_weight,
        bounds.y, bounds.norm.scale_lower, certifier.lm_gamma, CERTIFIED, mixture=bounds.mixture, projection=exact_projection,
    )
    return {"box": result.box_bound, "decomposed": result.decomposed_bound, "margin": result.margin}


def nearest_by_centre(certifier, bounds, count: int) -> tuple[int, list[int]]:
    """The candidate a runtime would choose (largest centre logit, lowest index) and the `count` rows nearest to it."""
    centres = certifier.centres(bounds)
    candidate = oracle.first_argmax(centres)
    order = torch.argsort(centres, descending=True, stable=True)
    order = order[order != candidate][:count]
    return candidate, [int(v) for v in order]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase5a2-crown-oracle.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=None, help="export only the first N selected samples (development)")
    args = parser.parse_args()
    output = Path(args.output)
    artifact = output / "artifact"
    artifact.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started_all = time.perf_counter()
    report: dict = {"timings_ms": {}}
    device = torch.device("cuda", torch.cuda.current_device())

    # The capture: regenerated by Phase 5A's capture stage, equal to Phase 5A run1's.
    capture_dir = REPO_ROOT / config["capture"]
    phase5a_dir = REPO_ROOT / config["phase5a_run"]
    expected = json.loads((phase5a_dir / "digest.json").read_text(encoding="utf-8"))
    capture_tensors = sha256_file(capture_dir / "capture.safetensors")
    capture_records = canonical_digest(read_jsonl(capture_dir / "capture.jsonl.gz"), phase5a.EXCLUDED_FIELDS)
    prompts_digest = canonical_digest(read_jsonl(capture_dir / "prompts.jsonl"))
    checks = {
        "capture_tensors_equal_phase5a": capture_tensors == expected["capture_tensors_sha256"],
        "capture_records_equal_phase5a": capture_records == expected["capture_records_sha256"],
        "prompts_equal_phase5a": prompts_digest == expected["prompts_sha256"],
    }
    report["capture"] = {"tensors_sha256": capture_tensors, "records_sha256": capture_records, "prompts_sha256": prompts_digest, **checks}
    if not all(checks.values()):
        raise ExportFailure(f"the capture differs from Phase 5A's: {checks}")

    # The samples, chosen from Phase 5A's records.
    records = phase5a.shard_records(phase5a_dir, "samples")
    chosen = select_samples(records, config["samples"])
    if args.samples is not None:
        chosen = chosen[: args.samples]
    raw = yaml.safe_load((capture_dir / "config.yaml").read_text(encoding="utf-8"))
    capture = load_file(str(capture_dir / "capture.safetensors"))
    captured = {(r["prompt_id"], r["step"]): r for r in read_jsonl(capture_dir / "capture.jsonl.gz")}
    layer_name = json.loads((capture_dir / "capture_stage.json").read_text(encoding="utf-8"))["target_layer"]
    index_of = {(int(p), int(s)): n for n, (p, s) in enumerate(zip(capture["prompt_id"].tolist(), capture["step"].tolist()))}

    # The target layer, as Phase 5A loads and checks it.
    started = time.perf_counter()
    gate_up, down, tensors, offsets = phase5a.load_target(raw, layer_name, device)
    rows_expected = json.loads((REPO_ROOT / raw["phase4b_reference"] / "reference_rows.json").read_text(encoding="utf-8"))
    experts_name = f"{layer_name}.mlp.experts"
    if not (phase5a.row_digests(gate_up) == rows_expected[f"{experts_name}.gate_up_proj"] and phase5a.row_digests(down) == rows_expected[f"{experts_name}.down_proj"]):
        raise ExportFailure("the target layer's weights differ from the Phase 4B reference's row digests")
    from transformers import AutoConfig
    from transformers.activations import ACT2FN
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MLP, DeepseekV3RMSNorm

    model_config = AutoConfig.from_pretrained(raw["model"]["repository"], revision=raw["model"]["revision"])
    layer = oracle.ExpertLayer(gate_up, down, ACT2FN[model_config.hidden_act])
    norm = DeepseekV3RMSNorm(model_config.hidden_size, eps=model_config.rms_norm_eps).to(device=device, dtype=torch.bfloat16)
    norm.weight.data.copy_(tensors["norm"])
    shared = DeepseekV3MLP(model_config, intermediate_size=model_config.moe_intermediate_size * model_config.n_shared_experts).to(device=device, dtype=torch.bfloat16)
    for projection in ("gate", "up", "down"):
        getattr(shared, f"{projection}_proj").weight.data.copy_(tensors[f"shared_{projection}"])
    lm_weight, gain = tensors["lm_head"], tensors["norm"]
    eps = norm.variance_epsilon
    report["timings_ms"]["load"] = (time.perf_counter() - started) * 1e3

    settings = config["oracle"]
    budget_step = float(settings["budget_step"])
    strategies = {s.name: s for s in oracle.strategies([1], ["q6+q4"])}
    chosen_strategies = [strategies[config["strategies"]["primary"]], strategies[config["strategies"]["spatial"]]]
    routed = layer.expert_bytes * capture["top_k_index"].shape[1]

    # The certificate's constants (pairwise_certificate's, decision 0005), for the verifier's assembly.
    r32, r16 = relative_rounding_error(torch.float32, CERTIFIED.elementwise), relative_rounding_error(torch.bfloat16, CERTIFIED.elementwise)
    xi = (1.0 + r32) ** 2 * (1.0 + r16) ** 2 - 1.0
    floor = FP32_NORMAL_MIN + subnormal_floor(torch.bfloat16)
    gain64 = gain.to(torch.float64)
    mu = (gain64.abs() * (1.0 + r32) + 1.0) * floor * (1.0 + r16) ** 2
    lm_gamma = ReferenceNumerics(lm_weight.dtype, oracle.CERTIFIED_ARITHMETIC.accumulation_unit_roundoff, lm_weight.shape[1]).accumulation_gamma
    constants = {
        "xi": xi, "lm_gamma": lm_gamma, "separations": 2.0, "neurons": int(capture["top_k_index"].shape[1] * layer.intermediate),
        "magnitude_inflation": (1.0 + 2.0 * gamma(lm_weight.shape[1] + 2, 2.0**-22)) * (1.0 + 2.0 * lm_gamma),
        "rms_norm_eps": eps, "routed_bytes": routed, "expert_bytes": layer.expert_bytes,
    }

    manifest: dict = {
        "format": FORMAT, "constants": constants, "target_layer": layer_name, "samples": [],
        "strategies": {s.name: {"family": s.family, "spec": s.spec, "levels": levels_of(s), "fallback_bytes": s.fallback_bytes(layer, 6),
                                "metadata_bytes": s.metadata_bytes(layer, 6), "storage_multiplier": s.storage_multiplier(layer)}
                       for s in chosen_strategies},
        "shapes": {"experts": layer.experts, "hidden": layer.hidden, "intermediate": layer.intermediate, "vocabulary": lm_weight.shape[0], "top_k": 6},
    }
    expert_ids: set[int] = set()

    for n_sample, choice in enumerate(chosen):
        sample_started = time.perf_counter()
        prompt_id, step = int(choice["prompt_id"]), int(choice["step"])
        n = index_of[(prompt_id, step)]
        values = {key: capture[key][n].to(device) for key in ("x", "r", "S", "R", "m", "y", "h", "top_k_weights")}
        experts = capture["top_k_index"][n].tolist()
        sample = oracle.decode_sample(layer, values["x"], values["r"], values["S"], experts, values["top_k_weights"], norm, lm_weight)
        with torch.inference_mode():
            shared_output = shared(values["x"].view(1, 1, -1)).reshape(-1)
        bitwise = {key: bool(torch.equal(sample.reference[key], values[key])) for key in ("R", "m", "y", "h")}
        bitwise["shared"] = bool(torch.equal(shared_output, values["S"]))
        bitwise["logits"] = phase5a.phase4b().logits_sha256(sample.reference["logits"]) == captured[(prompt_id, step)]["logits_sha256"]
        bitwise["token"] = sample.token == captured[(prompt_id, step)]["token"] == choice["token"]
        if not all(bitwise.values()):
            raise ExportFailure(f"the reference is not reproduced on ({prompt_id}, {step}): {bitwise}")
        expert_ids.update(sample.experts)
        weights = oracle.SampleWeights(layer, sample.experts)
        logits = sample.reference["logits"].to(torch.float32)
        comparison = torch.argsort(logits, descending=True, stable=True)
        comparison = comparison[comparison != sample.token][: int(settings["comparison_rows"])]
        tensors_out: dict[str, torch.Tensor] = {
            "x": sample.x.cpu(), "r": sample.residual.cpu(), "S": sample.shared.cpu(), "routing": sample.weights.cpu(),
            "experts": torch.tensor(sample.experts, dtype=torch.int64), "reference_a": sample.reference["a"].cpu(),
            "reference_logits": sample.reference["logits"].cpu(), "reference_y": sample.reference["y"].cpu(),
            "real_a": sample.real["a"].cpu(), "real_y": sample.real["y"].cpu(), "comparison_rows": comparison.cpu(),
        }
        entry: dict = {
            "index": n_sample, "prompt_id": prompt_id, "step": step, "length": choice["length"], "gap": choice["gap"], "gap_bin": choice["gap_bin"],
            "token": sample.token, "runner_up": sample.runner_up, "experts": list(sample.experts), "routing": [float(w) for w in sample.weights],
            "bitwise": bitwise, "ceilings": {}, "strategies": {},
            "phase5a_record": {"ceilings": choice["ceilings"],
                               "cells": [c for c in choice["cells"] if c["strategy"] in {s.name for s in chosen_strategies}]},
        }
        certifiers, ceiling_outcomes = {}, {}
        for tier, arithmetic in TIERS.items():
            certifier = oracle.Certifier(lm_weight, gain, sample.reference["logits"], arithmetic, near=int(settings["near_rows"]), chunk_rows=int(settings["chunk_rows"]))
            outcome, bounds, violations = oracle.ceiling(sample, weights, arithmetic, certifier, gain, eps)
            if arithmetic.certifies and sum(violations.values()):
                raise ExportFailure(f"enclosure violation at the ceiling of ({prompt_id}, {step})")
            entry["ceilings"][tier] = {"would_certify": outcome.certified, "candidate": outcome.candidate, "margin": phase5a.finite(outcome.margin),
                                       "terms": {k: phase5a.finite(v) for k, v in outcome.terms.items()}, "violations": violations}
            certifiers[tier], ceiling_outcomes[tier] = certifier, outcome

        for strategy in chosen_strategies:
            per_strategy: dict = {"tiers": {}, "phase5a_cells": {}}
            for tier, arithmetic in TIERS.items():
                certifier = certifiers[tier]
                states, first = schedule(sample, weights, strategy, arithmetic, certifier, gain, eps, budget_step)
                level_values = levels_of(strategy)
                # Activation enclosures per row level (an activation depends only on its own gate and up rows).
                per_level = {}
                for level in level_values:
                    filled = oracle.ExpertStates.filled(layer, len(experts), device, level, columns=False)
                    filled.down.fill_(oracle.EXACT)
                    b = oracle.propagate(sample, strategy.knowledge(weights, filled), arithmetic, oracle.BoundTier.REALISTIC, gain, eps, weights.column_norms)
                    per_level[level] = (b.a.lower, b.a.upper)
                tensors_out[f"{strategy.name}.{tier}.a_levels"] = torch.tensor(level_values, dtype=torch.int64)
                tensors_out[f"{strategy.name}.{tier}.a_lower"] = torch.stack([per_level[v][0] for v in level_values]).cpu()
                tensors_out[f"{strategy.name}.{tier}.a_upper"] = torch.stack([per_level[v][1] for v in level_values]).cpu()
                state_rows = {"gate": [], "up": [], "down": []}
                vectors: dict[str, list] = {}
                per_state = []
                for position, states in enumerate(states):
                    if not torch.equal(states.gate, states.up):
                        raise ExportFailure("gate and up rows move together in these strategies")
                    for name in ("gate", "up", "down"):
                        state_rows[name].append(states.matrix(name).to(torch.int8).cpu())
                    knowledge = strategy.knowledge(weights, states)
                    bounds = oracle.propagate(sample, knowledge, arithmetic, oracle.BoundTier.REALISTIC, gain, eps, weights.column_norms)
                    violations = oracle.enclosure_violations(bounds, sample, arithmetic.real)
                    if sum(violations.values()):
                        raise ExportFailure(f"enclosure violation in ({prompt_id}, {step}) {strategy.name} {tier} state {position - 1}")
                    # The activation enclosure of this state is its rows' levels' (checked, not assumed).
                    gathered_lo = torch.empty_like(bounds.a.lower)
                    gathered_hi = torch.empty_like(bounds.a.upper)
                    for level in level_values:
                        at = states.gate == level
                        gathered_lo = torch.where(at, per_level[level][0], gathered_lo)
                        gathered_hi = torch.where(at, per_level[level][1], gathered_hi)
                    if not (torch.equal(gathered_lo, bounds.a.lower) and torch.equal(gathered_hi, bounds.a.upper)):
                        raise ExportFailure("an activation enclosure depends on more than its own rows")
                    vectors.setdefault("y_lower", []).append(bounds.y.lower.cpu())
                    vectors.setdefault("y_upper", []).append(bounds.y.upper.cpu())
                    candidate, near = nearest_by_centre(certifier, bounds, int(settings["near_rows"]))
                    record = {"index": position - 1, "candidate": candidate, "near": near}
                    if not arithmetic.real:
                        vectors.setdefault("errors", []).append(sum(bounds.mixture.errors.values()).cpu())
                        vectors.setdefault("y_rounding", []).append(rounding_error_upper(bounds.y.magnitude, torch.bfloat16, CERTIFIED.elementwise).cpu())
                        vectors.setdefault("h_magnitude", []).append(bounds.norm.output.magnitude.cpu())
                        record["scale_lower"] = bounds.norm.scale_lower
                    read = strategy.bytes_read(layer, states)
                    record["bytes"] = {key: int(read[key].sum()) for key in ("gate", "up", "down", "levels", "unread")}
                    record["bytes"]["metadata"] = strategy.metadata_bytes(layer, len(experts))
                    record["certified_bytes"] = sum(record["bytes"][k] for k in ("gate", "up", "down", "levels", "metadata"))
                    record["certified_fraction"] = record["certified_bytes"] / routed
                    # Phase 5A's bounds on the comparison pairs (reference token against its nearest rows), realistic and ideal.
                    certifier.candidate = sample.token
                    rows = comparison.to(device)
                    for bound_tier in (oracle.BoundTier.REALISTIC, oracle.BoundTier.IDEAL):
                        b = bounds if bound_tier is oracle.BoundTier.REALISTIC else oracle.propagate(sample, knowledge, arithmetic, bound_tier, gain, eps, weights.column_norms)
                        parts = real_components(certifier, b, rows) if arithmetic.real else certified_components(certifier, b, rows)
                        for key, value in parts.items():
                            vectors.setdefault(f"compare.{bound_tier.value}.{key}", []).append(value.cpu())
                    per_state.append(record)
                for name, stacked in state_rows.items():
                    tensors_out[f"{strategy.name}.{tier}.states.{name}"] = torch.stack(stacked)
                for name, stacked in vectors.items():
                    tensors_out[f"{strategy.name}.{tier}.{name}"] = torch.stack(stacked)
                # Phase 5A's own cells on this sample (the baseline): realistic bounds and ordering, and ideal bounds and ordering.
                cells = {}
                for bound_name, ordering_name in (("realistic", "realistic"), ("ideal", "ideal")):
                    result = oracle.run_cell(sample, weights, strategy, arithmetic, oracle.BoundTier(bound_name), oracle.Ordering(ordering_name),
                                             certifier, budget_step, gain, eps, offsets, complete=ceiling_outcomes[tier])
                    cells[f"{bound_name}/{ordering_name}"] = {
                        "would_certify": result.would_certify, "certified": result.certified, "winner": result.winner, "checkpoint": result.checkpoint,
                        "fraction": result.fraction, "bytes": result.bytes, "margin": phase5a.finite(result.margin), "evaluations": result.evaluations,
                    }
                    if result.certified and result.winner != sample.token:
                        raise ExportFailure("Phase 5A certified a wrong token")
                # The schedule here is run_cell's: where its realistic cell certified, the state there is this export's.
                realistic = cells["realistic/realistic"]
                if realistic["would_certify"] and abs(per_state[realistic["checkpoint"] + 1]["certified_fraction"] - realistic["fraction"]) > 1e-12:
                    raise ExportFailure("the exported schedule is not Phase 5A's")
                per_strategy["tiers"][tier] = {"states": per_state, "first_state": {"candidate": first.candidate, "tightest": first.tightest, "margin": phase5a.finite(first.margin)},
                                              "phase5a_cells": cells}
            entry["strategies"][strategy.name] = per_strategy
        entry["timings_ms"] = {"export": (time.perf_counter() - sample_started) * 1e3}
        save_file({k: v.contiguous() for k, v in tensors_out.items()}, str(artifact / f"sample.{n_sample}.safetensors"))
        manifest["samples"].append(entry)
        print(f"export: sample {n_sample} ({prompt_id}, {step}) gap {choice['gap']:.3f} in {entry['timings_ms']['export'] / 1e3:.0f}s", flush=True)
        torch.cuda.empty_cache()

    # Weights: the routed experts of the samples, their levels and metadata; the LM head and the norm's gain.
    started = time.perf_counter()
    weights_out: dict[str, torch.Tensor] = {"lm_head": lm_weight.cpu(), "norm": gain.cpu(), "mu": mu.cpu()}
    for e in sorted(expert_ids):
        matrices = layer.matrices(e)
        decompositions = layer.levels(e, "q6+q4")
        expert_norms = layer.norms(e)
        for name in ("gate", "up", "down"):
            prefix = f"expert.{e}.{name}"
            weights_out[f"{prefix}.truth"] = matrices[name].contiguous().cpu()
            d = decompositions[name]
            for level_index, level in enumerate(d.levels):
                weights_out[f"{prefix}.q{level_index}.codes"] = level.codes.cpu()
                weights_out[f"{prefix}.q{level_index}.scales"] = level.scales.cpu()
                weights_out[f"{prefix}.r{level_index}.l2"] = d.remainder_norms[level_index].l2.cpu()
                weights_out[f"{prefix}.r{level_index}.linf"] = d.remainder_norms[level_index].linf.cpu()
            own = getattr(expert_norms, name)
            weights_out[f"{prefix}.norm.l2"] = own.l2.cpu()
            weights_out[f"{prefix}.norm.linf"] = own.linf.cpu()
    save_file({k: v.contiguous() for k, v in weights_out.items()}, str(artifact / "weights.safetensors"))
    report["timings_ms"]["weights"] = (time.perf_counter() - started) * 1e3

    manifest["experts"] = sorted(expert_ids)
    manifest["provenance"] = {
        "source_tree_sha256": source_tree_sha256(REPO_ROOT), "research_tree_sha256": research_tree_sha256(REPO_ROOT),
        "capture": report["capture"], "phase5a_run": config["phase5a_run"], "phase5a_digest": expected,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"), "levels_spec": "q6+q4",
    }
    manifest["files"] = {p.name: sha256_file(p) for p in sorted(artifact.glob("*.safetensors"))}
    (artifact / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    environment = environment_metadata(REPO_ROOT, {"repository": raw["model"]["repository"], "revision": raw["model"]["revision"], "dtype": "bfloat16", "device": "cuda"}, NUMERICS_FLAGS)
    environment["python_hash_seed"] = os.environ.get("PYTHONHASHSEED")
    environment["research_tree_sha256"] = manifest["provenance"]["research_tree_sha256"]
    (output / "export_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    report["timings_ms"]["total"] = (time.perf_counter() - started_all) * 1e3
    report["peak_device_bytes"] = torch.cuda.max_memory_allocated(device)
    report["artifact_bytes"] = sum(p.stat().st_size for p in artifact.iterdir())
    report["manifest_sha256"] = sha256_file(artifact / "manifest.json")
    (output / "export_stage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"exported {len(chosen)} samples, {len(expert_ids)} experts, {report['artifact_bytes'] / 1e9:.2f} GB to {artifact}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportFailure as failure:
        print(f"EXPORT FAILURE: {failure}", flush=True)
        raise SystemExit(1)
