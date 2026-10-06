"""Phase 5A2, stages 2–4: auto_LiRPA's bounds on Moonlight's routed experts, the certificates and the bytes (decision 0010).

    PYTHONHASHSEED=<n> uv run --project research/crown_expert_oracle python research/crown_expert_oracle/run.py \\
        --run experiments/phase5a2/<name> --part validate|compare|search [--samples 0,3] [--shard i --shards n]

Runs in the isolated verifier environment. Reads `<run>/artifact` (export.py), writes into `<run>`:

  validate  the wrapper (the graphs at the true weights against the exported real forward), every set of every state
            (the weight boxes and activation boxes contain the true values; along a schedule the boxes only shrink), the
            sets built without unread bytes (poisoned rows change nothing), and the certificate's assembly against Phase
            5A's own margins → validate.<shard>.jsonl
  compare   the bound comparison: at fixed points of each schedule, for the reference token against its nearest rows,
            auto_LiRPA's lower bounds by graph and method, Phase 5A's realistic and ideal bounds, the true value, the
            reduced set's exact optimum and adversarial realizations (an achievable value inside the set)
            → compare.<shard>.jsonl
  search    the certificates: Phase 5A's realistic schedules, the first budget at which auto_LiRPA's certificate holds
            (bisection on the nearest rows, then the whole vocabulary), its bytes, its cost → search.<shard>.jsonl
Each part also writes verifier_<part>_<shard>.json (environment, timings, peak memory) and stops at its first hard
failure (failure.json): a bound above a true value or a realization, a set without the true weights, a certified token
that is not the reference's.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import io
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from crown_oracle import attack, certify, sets  # noqa: E402
from crown_oracle.artifact import EXACT, UNKNOWN, Artifact, research_tree_sha256, source_tree_sha256  # noqa: E402
from crown_oracle.graph import (  # noqa: E402
    BoundCost, Bounder, ExpertsSuffix, ReducedSuffix, boxed_input, boxed_parameter, gpu_peak, gpu_reset, process_peak_rss,
    suffix_value,
)

torch.set_default_dtype(torch.float64)
torch.use_deterministic_algorithms(True)
EXCLUDED_FIELDS = ("timings_ms", "cost", "system")
MATRICES = ("gate", "up", "down")


class HardFailure(Exception):
    pass


def finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def floats(tensor: torch.Tensor) -> list:
    return [finite(v) for v in tensor.reshape(-1).tolist()]


class JsonlWriter:
    """Reproducible gzip JSONL (mtime 0), as awpmi.tracing writes it."""

    def __init__(self, path: Path) -> None:
        self.raw = open(path, "wb")
        self.gz = gzip.GzipFile(fileobj=self.raw, mode="wb", mtime=0)
        self.text = io.TextIOWrapper(self.gz, encoding="utf-8", newline="\n")

    def write(self, record: dict) -> None:
        self.text.write(json.dumps(record, sort_keys=True) + "\n")
        self.text.flush()

    def close(self) -> None:
        self.text.close()
        self.raw.close()


# A sample's data


class SampleData:
    """One exported sample: its inputs, the routed experts' matrices, the reference's and the real forward's values."""

    def __init__(self, artifact: Artifact, index: int, device: torch.device) -> None:
        self.entry = artifact.samples[index]
        self.index, self.device = index, device
        self.tensors = artifact.sample_tensors(index, device)
        self.experts = self.entry["experts"]
        self.matrices = {name: [artifact.matrix(e, name, device) for e in self.experts] for name in MATRICES}
        self.truth_bf16 = {name: torch.stack([m.truth for m in self.matrices[name]]) for name in MATRICES}  # validation only
        self.routing = self.tensors["routing"].to(torch.float64)
        self.base = self.tensors["r"].to(torch.float64) + self.tensors["S"].to(torch.float64)  # exact in float64 (Phase 5A: b)
        self.x = self.tensors["x"].to(torch.float64).reshape(1, -1)
        # Validation only: y_T = b + Σ_e w_e·D_e·a_ref with the true weights and the reference's activations, so that the
        # certified tier's structural part is T = Δ·y_T for any contender.
        down = self.truth_bf16["down"].to(torch.float64)
        self.reference_structural = self.base + torch.einsum("e,eh->h", self.routing, torch.einsum("ehi,ei->eh", down, self.tensors["reference_a"].to(torch.float64)))
        del down

    def truth(self, name: str) -> torch.Tensor:
        """The true weights of one matrix kind, float64 [K, R, C] (validation only)."""
        return self.truth_bf16[name].to(torch.float64)

    def rows(self, strategy: str, tier: str, index: int) -> dict[str, torch.Tensor]:
        """The rows' states [K, R] of each matrix at budget index `index` (−1: the initial state)."""
        return {name: self.tensors[f"{strategy}.{tier}.states.{name}"][index + 1].to(torch.int64) for name in MATRICES}

    def steps(self, strategy: str, tier: str) -> int:
        return self.tensors[f"{strategy}.{tier}.states.gate"].shape[0] - 1

    def state_record(self, strategy: str, tier: str, index: int) -> dict:
        return self.entry["strategies"][strategy]["tiers"][tier]["states"][index + 1]

    def weight_boxes(self, rows: dict[str, torch.Tensor], names=MATRICES) -> dict[str, sets.Box]:
        return {name: sets.expert_boxes(self.matrices[name], rows[name], self.device) for name in names}

    def activation_box(self, strategy: str, tier: str, gate_rows: torch.Tensor) -> sets.Box:
        prefix = f"{strategy}.{tier}"
        return sets.activation_box(self.tensors[f"{prefix}.a_levels"], self.tensors[f"{prefix}.a_lower"], self.tensors[f"{prefix}.a_upper"], gate_rows)

    def certified_state(self, strategy: str, index: int) -> certify.CertifiedState:
        prefix = f"{strategy}.certified"
        position = index + 1
        record = self.state_record(strategy, "certified", index)
        return certify.CertifiedState(self.tensors[f"{prefix}.y_lower"][position], self.tensors[f"{prefix}.y_upper"][position],
                                      self.tensors[f"{prefix}.errors"][position], self.tensors[f"{prefix}.y_rounding"][position],
                                      self.tensors[f"{prefix}.h_magnitude"][position], float(record["scale_lower"]))


# Graphs with their sets


def magnitudes(box: sets.Box) -> torch.Tensor:
    return torch.maximum(box.lower.abs(), box.upper.abs())


class StateBounds:
    """auto_LiRPA's bounds on one state of one schedule, any method, any rows.

    The property is separable over the routed experts: Δ·y = Δ·b + Σ_e Δ·(w_e·D_e·a_e), and no two experts share a variable
    (each has its own weights and activations; x and the router are exact), so the minimum of the sum over the set is the
    sum of the experts' minima. Each expert's graph is therefore bounded by auto_LiRPA on its own and the lower bounds are
    added to Δ·b (float64, exact b). For CROWN and CROWN-IBP this is the same computation as one graph of six experts (the
    backward pass is linear through the combine and the experts' subgraphs share no node); for α-CROWN it optimizes each
    expert's relaxation for its own term, the same objective, at a sixth of the memory (one graph of six exceeded the
    card)."""

    def __init__(self, sample: SampleData, strategy: str, tier: str, index: int, graph: str, settings: dict, cost: BoundCost,
                 optimum_for: torch.Tensor | None = None) -> None:
        """`optimum_for` (contenders' Δ, reduced graph): also compute the reduced set's exact optimum for them
        (`attack.reduced_optimum`, a diagnostic), before the boxes are released."""
        self.sample, self.graph, self.settings = sample, graph, settings
        self.optimum = None
        rows = sample.rows(strategy, tier, index)
        self.activations = sample.activation_box(strategy, tier, rows["gate"])
        zero = torch.zeros_like(sample.base)
        iterations = int(settings["alpha_iterations"])
        self.bounders = []
        if graph == "reduced":
            boxes = sample.weight_boxes(rows, names=("down",))
            down = boxes["down"]
            for e in range(len(sample.experts)):
                model = ReducedSuffix([boxed_parameter(down.centre[e], down.lower[e], down.upper[e])], [float(sample.routing[e])], zero)
                inputs = (boxed_input(self.activations.lower[e : e + 1], self.activations.upper[e : e + 1]),)
                self.bounders.append(Bounder(model, inputs, sample.device, alpha_iterations=iterations, cost=cost))
            self.a_magnitude = magnitudes(self.activations)
            if optimum_for is not None:
                self.optimum = attack.reduced_optimum(optimum_for, down.lower, down.upper, self.activations.lower, self.activations.upper,
                                                      sample.routing, sample.base)
            del down
        else:
            boxes = sample.weight_boxes(rows)
            for e in range(len(sample.experts)):
                parameters = [boxed_parameter(boxes[name].centre[e], boxes[name].lower[e], boxes[name].upper[e]) for name in MATRICES]
                model = ExpertsSuffix([parameters[0]], [parameters[1]], [parameters[2]], [float(sample.routing[e])], zero)
                self.bounders.append(Bounder(model, (sample.x,), sample.device, alpha_iterations=iterations, cost=cost))
            # |a| over the full graph's own set (its boxes, not Phase 5A's enclosure): |silu(g)| ≤ |g|, so |a| ≤ |g|⁺·|u|⁺.
            v = sample.x.reshape(-1).abs()
            g = torch.einsum("kih,h->ki", magnitudes(boxes["gate"]), v)
            u = torch.einsum("kih,h->ki", magnitudes(boxes["up"]), v)
            self.a_magnitude = g * u * (1.0 + 2.0**-40)
        self.down_magnitude = magnitudes(boxes["down"])
        del boxes
        gc.collect()
        self.chunk = int(settings["spec_chunk"][graph])

    def absolute(self, delta: torch.Tensor) -> torch.Tensor:
        """The structural expression's absolute mass per contender (for the computation slack)."""
        return certify.structural_absolute(delta, self.sample.base, self.sample.routing, self.down_magnitude, self.a_magnitude)

    def structural(self, delta: torch.Tensor, method: str) -> tuple[torch.Tensor, torch.Tensor]:
        """auto_LiRPA's lower bound on Δ·y per contender (Δ·b plus each expert's bound; α-CROWN one specification per
        call), and the expression's absolute mass."""
        chunk = 1 if method == "alpha-crown" else self.chunk
        lower = delta @ self.sample.base
        for bounder in self.bounders:
            lower = lower + bounder.lower(delta, method, chunk)
        return lower, self.absolute(delta)

    def close(self) -> None:
        # auto_LiRPA's graphs hold reference cycles: collect them before releasing the device memory.
        del self.bounders
        gc.collect()
        if self.sample.device.type == "cuda":
            torch.cuda.empty_cache()


# The true values (validation)


def true_structural(sample: SampleData, delta: torch.Tensor, tier: str) -> torch.Tensor:
    """The property at the true weights: Δ·y_real (real tier), or T = Δ·(b + Σ w·D·a_ref) with the reference's a (certified)."""
    if tier == "real":
        return delta @ sample.tensors["real_y"].to(torch.float64)
    return delta @ sample.reference_structural


def tolerance(values: torch.Tensor, absolute: torch.Tensor) -> torch.Tensor:
    """Float64 evaluation differences between two orders of the same exact expression: 2⁻⁴⁰ of its absolute mass."""
    return 2.0**-40 * absolute + 1e-300


# validate


def run_validate(artifact: Artifact, sample: SampleData, config: dict, writer, failures: list) -> None:
    settings = config["verifier"]
    record: dict = {"sample": sample.index, "prompt_id": sample.entry["prompt_id"], "step": sample.entry["step"]}
    # The wrapper: the full graph at the true weights equals the real forward Phase 5A computes (float64; another order).
    truth = {name: sample.truth(name) for name in MATRICES}
    y = suffix_value(sample.x, truth["gate"], truth["up"], truth["down"], sample.routing.tolist(), sample.base)
    real_y = sample.tensors["real_y"].to(torch.float64)
    scale = real_y.abs().max()
    record["wrapper_full_max_relative"] = float((y - real_y).abs().max() / scale)
    reduced_y = sample.base + torch.einsum("e,eh->h", sample.routing, torch.einsum("ehi,ei->eh", truth["down"], sample.tensors["real_a"].to(torch.float64)))
    record["wrapper_reduced_max_relative"] = float((reduced_y - real_y).abs().max() / scale)
    model = ExpertsSuffix([torch.nn.Parameter(t, requires_grad=False) for t in truth["gate"]],
                          [torch.nn.Parameter(t, requires_grad=False) for t in truth["up"]],
                          [torch.nn.Parameter(t, requires_grad=False) for t in truth["down"]], sample.routing.tolist(), sample.base).to(sample.device)
    with torch.no_grad():
        module_y = model(sample.x).reshape(-1)
    record["wrapper_module_max_relative"] = float((module_y - real_y).abs().max() / scale)
    if max(record["wrapper_full_max_relative"], record["wrapper_reduced_max_relative"], record["wrapper_module_max_relative"]) > 1e-12:
        failures.append({"kind": "wrapper", **record})
    # Every set of every state contains the true values; along a schedule the weight boxes only shrink.
    containment, inclusion, poisoned = {}, {}, {}
    real_a, reference_a = sample.tensors["real_a"].to(torch.float64), sample.tensors["reference_a"].to(torch.float64)
    for strategy in artifact.manifest["strategies"]:
        for tier in ("certified", "real"):
            key = f"{strategy}/{tier}"
            bad, shrinking, previous = [], True, None
            for index in range(-1, sample.steps(strategy, tier)):
                rows = sample.rows(strategy, tier, index)
                boxes = sample.weight_boxes(rows)
                for name in MATRICES:
                    if not boxes[name].contains(sample.truth_bf16[name]):
                        bad.append([index, name])
                activations = sample.activation_box(strategy, tier, rows["gate"])
                if not activations.contains(real_a if tier == "real" else reference_a):
                    bad.append([index, "a"])
                if previous is not None and not all(previous[name].includes(boxes[name]) for name in MATRICES):
                    shrinking = False
                previous = boxes
            containment[key], inclusion[key] = bad, shrinking
            if bad:
                failures.append({"kind": "set_without_truth", "sample": sample.index, "cell": key, "where": bad[:10]})
    # A set never depends on an unread byte: poison every row not read (random values, then NaN) and rebuild.
    strategy = next(iter(artifact.manifest["strategies"]))
    middle = sample.steps(strategy, "certified") // 2
    rows = sample.rows(strategy, "certified", middle)
    for name in MATRICES:
        clean = sets.expert_boxes(sample.matrices[name], rows[name], sample.device)
        for poison in ("random", "nan"):
            altered = []
            for k, m in enumerate(sample.matrices[name]):
                unread = (rows[name][k] != EXACT).unsqueeze(-1).to(sample.device)
                junk = torch.randn(m.truth.shape, device=sample.device, generator=torch.Generator(device=sample.device).manual_seed(k)).to(m.truth.dtype)
                if poison == "nan":
                    junk = torch.full_like(m.truth, math.nan)
                altered.append(type(m)(torch.where(unread, junk, m.truth), m.codes, m.scales, m.remainder_linf, m.remainder_l2, m.own_linf, m.own_l2))
            dirty = sets.expert_boxes(altered, rows[name], sample.device)
            poisoned[f"{name}/{poison}"] = bool(torch.equal(clean.lower, dirty.lower) and torch.equal(clean.upper, dirty.upper) and torch.equal(clean.centre, dirty.centre))
    if not all(poisoned.values()):
        failures.append({"kind": "set_reads_unread_bytes", "sample": sample.index, "poisoned": poisoned})
    record.update(containment=containment, monotone=inclusion, poisoning_unchanged=poisoned)
    # The certificate's assembly against Phase 5A's own margins on the comparison pairs (Phase 5A's structural bound).
    lm_weight, gain = artifact.tensor("lm_head", sample.device), artifact.tensor("norm", sample.device)
    mu = artifact.tensor("mu", sample.device).to(torch.float64)
    rows_compare = sample.tensors["comparison_rows"]
    worst = 0.0
    for strategy in artifact.manifest["strategies"]:
        prefix = f"{strategy}.certified"
        for index in range(-1, sample.steps(strategy, "certified")):
            state = sample.certified_state(strategy, index)
            _, _, _, delta = certify.pair_deltas(lm_weight, gain, sample.entry["token"], rows_compare)
            phase5a_decomposed = sample.tensors[f"{prefix}.compare.realistic.decomposed"][index + 1]
            structural = phase5a_decomposed + delta.abs() @ state.errors + delta.abs() @ state.y_rounding
            ours = certify.certified_margins(lm_weight, gain, mu, artifact.constants, state, sample.entry["token"], rows_compare,
                                             structural, torch.zeros_like(structural), 0.0)
            theirs = sample.tensors[f"{prefix}.compare.realistic.margin"][index + 1]
            scale_terms = delta.abs() @ (torch.maximum(state.y_lower.abs(), state.y_upper.abs()) + 1.0)
            worst = max(worst, float(((ours["margin"] - theirs).abs() / (scale_terms * state.scale_lower)).max()))
    record["assembly_max_relative_difference"] = worst
    if worst > 1e-9:
        failures.append({"kind": "assembly", "sample": sample.index, "difference": worst})
    writer.write(record)


# compare


def comparison_indices(steps: int, points: list[float]) -> list[int]:
    return sorted({-1 if p <= 0 else min(steps - 1, max(-1, round(p * steps) - 1)) for p in points})


def run_compare(artifact: Artifact, sample: SampleData, config: dict, writer, failures: list, totals: BoundCost) -> None:
    settings = config["verifier"]
    lm_weight, gain = artifact.tensor("lm_head", sample.device), artifact.tensor("norm", sample.device)
    mu = artifact.tensor("mu", sample.device).to(torch.float64)
    rows = sample.tensors["comparison_rows"]
    token = sample.entry["token"]
    _, _, _, delta = certify.pair_deltas(lm_weight, gain, token, rows)
    slack = float(settings["computation_slack"])
    graphs = {"certified": {settings["certified"]["graph"]: settings["certified"]["compare"]}, "real": settings["structural"]["compare"]}
    for strategy in artifact.manifest["strategies"]:
        for tier in ("certified", "real"):
            steps = sample.steps(strategy, tier)
            truth = true_structural(sample, delta, tier)
            for index in comparison_indices(steps, settings["comparison_points"]):
                started = time.perf_counter()
                record = {"sample": sample.index, "prompt_id": sample.entry["prompt_id"], "step": sample.entry["step"], "gap": sample.entry["gap"],
                          "strategy": strategy, "tier": tier, "index": index, "steps": steps,
                          "certified_fraction": sample.state_record(strategy, tier, index)["certified_fraction"], "rows": rows.tolist(),
                          "truth": floats(truth), "phase5a": {}, "crown": {}, "cost": {}}
                prefix = f"{strategy}.{tier}"
                for bound_tier in ("realistic", "ideal"):
                    record["phase5a"][bound_tier] = {key: floats(sample.tensors[f"{prefix}.compare.{bound_tier}.{key}"][index + 1])
                                                     for key in (("box", "decomposed", "margin") if tier == "certified" else ("box", "decomposed", "slack", "margin"))}
                    # Phase 5A's lower bound on the structural part T itself: its decomposed bound before the named rounding
                    # terms (certified tier; the real tier has none), comparable with auto_LiRPA's bound on T.
                    # (Certified tier: realistic only. The ideal propagation's own named terms are not exported; the ideal
                    # is compared through its margin there, where every term is its own.)
                    decomposed = sample.tensors[f"{prefix}.compare.{bound_tier}.decomposed"][index + 1]
                    if tier == "certified" and bound_tier == "realistic":
                        state_terms = sample.certified_state(strategy, index)
                        decomposed = decomposed + delta.abs() @ state_terms.errors + delta.abs() @ state_terms.y_rounding
                    if tier == "real" or bound_tier == "realistic":
                        record["phase5a"][bound_tier]["structural"] = floats(decomposed)
                # The focus pairs (α-CROWN's): the runner-up, the tightest by Phase 5A's realistic bound, the tightest truly.
                phase5a_margin = torch.tensor([v if v is not None else math.inf for v in record["phase5a"]["realistic"]["margin"]])
                focus = sorted({0, int(phase5a_margin.argmin()), int(truth.argmin())})[: int(settings["focus_pairs"])]
                record["focus"] = focus
                state = sample.certified_state(strategy, index) if tier == "certified" else None
                rows_states = sample.rows(strategy, tier, index)
                for graph, methods in graphs[tier].items():
                    cost = BoundCost()
                    bounds = StateBounds(sample, strategy, tier, index, graph, settings, cost, optimum_for=delta if graph == "reduced" else None)
                    for method in methods:
                        chosen = torch.tensor(focus, device=delta.device) if method == "alpha-crown" else torch.arange(delta.shape[0], device=delta.device)
                        part_lower, absolute = bounds.structural(delta[chosen], method)
                        lower = torch.full((delta.shape[0],), math.nan, dtype=torch.float64, device=delta.device)
                        lower[chosen] = part_lower
                        allowance = tolerance(truth[chosen], absolute) + slack * absolute
                        violations = int((part_lower > truth[chosen] + allowance).sum())
                        entry = {"lower": floats(lower), "violations_truth": violations, "rows_bounded": len(chosen)}
                        if tier == "certified":
                            margins = certify.certified_margins(lm_weight, gain, mu, artifact.constants, state, token, rows[chosen], part_lower, absolute, slack)
                        else:
                            y = sample.tensors[f"{prefix}.y_lower"][index + 1], sample.tensors[f"{prefix}.y_upper"][index + 1]
                            margins = certify.structural_margins(lm_weight, gain, token, rows[chosen], y[0], y[1], part_lower, absolute,
                                                                 int(artifact.constants["neurons"]), slack)
                        margin = torch.full_like(lower, math.nan)
                        margin[chosen] = margins["margin"]
                        entry["margin"] = floats(margin)
                        record["crown"][f"{graph}/{method}"] = entry
                        if violations:
                            failures.append({"kind": "bound_above_truth", "sample": sample.index, "strategy": strategy, "tier": tier, "index": index,
                                             "graph": graph, "method": method, "count": violations})
                    # The reduced set's exact optimum (a diagnostic: what a perfect verifier returns for that set).
                    if graph == "reduced":
                        record["reduced_optimum"] = floats(bounds.optimum)
                    record["cost"][graph] = cost.to_json()
                    totals.merge(cost)
                    bounds.close()
                # Achievable values of each graph's own set, against which its bounds are checked (a bound above one would be
                # unsound). The reduced graph's set (Phase 5A's activation enclosure, which also uses the rows' L2 remainder
                # norms, × the down boxes) attains its exact optimum at a vertex; the full graph's set (every weight in its L∞
                # box: a larger set) is searched by projected gradient (real arithmetic). An attack on the full graph's set is
                # not a point of the reduced set (its activations can leave Phase 5A's enclosure).
                if tier == "real":
                    record["attack"] = attack_values(sample, rows_states, delta, record, int(settings["attack_steps"]))
                for graph_method, entry in record["crown"].items():
                    graph = graph_method.split("/")[0]
                    if graph == "reduced":
                        achievable = {j: value for j, value in enumerate(record["reduced_optimum"]) if value is not None}
                    elif "attack" in record:
                        achievable = {int(j): value for j, value in record["attack"]["values"].items()}
                    else:
                        continue
                    bad = sum(1 for j, value in achievable.items() if entry["lower"][j] is not None
                              and entry["lower"][j] > value + 1e-9 * (1 + abs(value)))
                    entry["violations_achievable"] = bad
                    if bad:
                        failures.append({"kind": "bound_above_achievable_value", "sample": sample.index, "strategy": strategy, "tier": tier,
                                         "index": index, "cell": graph_method, "count": bad})
                record["timings_ms"] = {"point": (time.perf_counter() - started) * 1e3}
                writer.write(record)
                if failures:
                    raise HardFailure


def attack_values(sample: SampleData, rows: dict[str, torch.Tensor], delta: torch.Tensor, record: dict, steps: int) -> dict:
    """Projected-gradient searches of the full graph's weight boxes for the focus pairs (the runner-up, the tightest by
    Phase 5A's realistic bound and by the true value): the smallest Δ·y found per pair, each a point of the set."""
    boxes = sample.weight_boxes(rows)
    lower = [boxes[name].lower for name in MATRICES]
    upper = [boxes[name].upper for name in MATRICES]
    chosen = record["focus"]
    values, started = {}, time.perf_counter()
    for j in chosen:
        def objective(g, u, d, j=j):
            return suffix_value(sample.x, g, u, d, sample.routing.tolist(), sample.base) @ delta[j]
        # Starts: the runtime's centre, and the vertex a first-order adversary picks at the true weights.
        point = [sample.truth(name).requires_grad_(True) for name in MATRICES]
        grads = torch.autograd.grad(objective(*point), point)
        starts = [[boxes[name].centre for name in MATRICES], attack.vertex_starts(lower, upper, list(grads))]
        value, _ = attack.pgd_minimize(objective, lower, upper, starts, steps=steps)
        values[str(j)] = value
        del point, grads, starts
    del boxes, lower, upper
    gc.collect()
    torch.cuda.empty_cache()
    return {"rows": chosen, "values": values, "ms": (time.perf_counter() - started) * 1e3}


# search


def run_search(artifact: Artifact, sample: SampleData, config: dict, writer, failures: list, totals: BoundCost) -> None:
    settings = config["verifier"]
    lm_weight, gain = artifact.tensor("lm_head", sample.device), artifact.tensor("norm", sample.device)
    mu = artifact.tensor("mu", sample.device).to(torch.float64)
    token = sample.entry["token"]
    slack = float(settings["computation_slack"])
    neurons = int(artifact.constants["neurons"])
    routed = float(artifact.constants["routed_bytes"])
    cells = [("certified", settings["certified"]["graph"], m) for m in settings["certified"]["search"]]
    cells += [("real", graph, m) for graph, methods in settings["structural"]["search"].items() for m in methods]
    for strategy in artifact.manifest["strategies"]:
        fallback = artifact.manifest["strategies"][strategy]["fallback_bytes"]
        for tier, graph, method in cells:
            started = time.perf_counter()
            cost = BoundCost()
            steps = sample.steps(strategy, tier)
            log: list[dict] = []

            def evaluate(index: int, full: bool) -> dict:
                record = sample.state_record(strategy, tier, index)
                candidate, near = record["candidate"], torch.tensor(record["near"], device=sample.device)
                state = sample.certified_state(strategy, index) if tier == "certified" else None
                y = (sample.tensors[f"{strategy}.{tier}.y_lower"][index + 1], sample.tensors[f"{strategy}.{tier}.y_upper"][index + 1])
                bounds = StateBounds(sample, strategy, tier, index, graph, settings, cost)
                counts = {"bounded": 0, "alpha": 0}

                def assemble(rows, lower, absolute):
                    if tier == "certified":
                        return certify.certified_margins(lm_weight, gain, mu, artifact.constants, state, candidate, rows, lower, absolute, slack)["margin"]
                    return certify.structural_margins(lm_weight, gain, candidate, rows, y[0], y[1], lower, absolute, neurons, slack)["margin"]

                def bound(delta, use):
                    lower, absolute = bounds.structural(delta, use)
                    truth = true_structural(sample, delta, tier)
                    if bool((lower > truth + tolerance(truth, absolute) + slack * absolute).any()):
                        failures.append({"kind": "bound_above_truth", "sample": sample.index, "strategy": strategy, "tier": tier, "index": index, "cell": f"{graph}/{use}"})
                        raise HardFailure
                    return lower, absolute

                def failed(rows: torch.Tensor) -> tuple[bool, float]:
                    """Whether some contender of `rows` stays unsettled (chunk by chunk, stopping at the first), and the
                    smallest margin seen. "crown+alpha" bounds a contender CROWN leaves unsettled again by α-CROWN, and
                    takes the larger of the two valid lower bounds."""
                    smallest = math.inf
                    for start in range(0, rows.numel(), bounds.chunk):
                        part = rows[start : start + bounds.chunk]
                        _, _, _, delta = certify.pair_deltas(lm_weight, gain, candidate, part)
                        lower, absolute = bound(delta, "crown" if method == "crown+alpha" else method)
                        counts["bounded"] += int(part.numel())
                        margin = assemble(part, lower, absolute)
                        if method == "crown+alpha":
                            for j in (margin <= 0).nonzero().reshape(-1).tolist():
                                alpha, _ = bound(delta[j : j + 1], "alpha-crown")
                                counts["alpha"] += 1
                                lower[j] = torch.maximum(lower[j], alpha[0])
                                margin[j] = assemble(part[j : j + 1], lower[j : j + 1], absolute[j : j + 1])[0]
                                if float(margin[j]) <= 0:
                                    break
                        smallest = min(smallest, float(margin.min()))
                        if bool((margin <= 0).any()):
                            return True, smallest
                    return False, smallest

                near_failed, near_min = failed(near)
                outcome = {"index": index, "candidate": candidate, "near_failed": near_failed, "near_min": finite(near_min),
                           "certified": False, "full": False, "box_settled": None, "capped": False}
                if full and not near_failed:
                    outcome["full"] = True
                    unsettled = full_check_box(lm_weight, gain, mu, artifact, sample, tier, state, y, candidate, near, neurons)
                    outcome["box_settled"] = int(lm_weight.shape[0] - 1 - near.numel() - unsettled.numel())
                    outcome["unsettled_after_box"] = int(unsettled.numel())
                    if unsettled.numel() > int(settings["full_check_cap"]):
                        outcome["capped"] = True
                    else:
                        outcome["certified"] = not failed(unsettled)[0]
                outcome.update(contenders_bounded=counts["bounded"], alpha_calls=counts["alpha"])
                bounds.close()
                log.append(outcome)
                return outcome

            # Phase 5A's search (run_cell): the initial state, then a bisection on the nearest rows, then full checks.
            final = evaluate(-1, full=True)
            if not final["certified"]:
                low, high = 0, steps - 1
                while low < high:
                    middle = (low + high) // 2
                    if evaluate(middle, full=False)["near_failed"]:
                        low = middle + 1
                    else:
                        high = middle
                for index in range(low, steps):
                    final = evaluate(index, full=True)
                    if final["certified"]:
                        break
            certified = final["certified"]
            index = final["index"]
            state_record = sample.state_record(strategy, tier, index)
            fraction = state_record["certified_fraction"] if certified else fallback / routed
            winner = final["candidate"] if certified else -1
            if certified and tier == "certified" and winner != token:
                failures.append({"kind": "certified_mismatch", "sample": sample.index, "strategy": strategy, "cell": f"{graph}/{method}"})
            out = {"sample": sample.index, "prompt_id": sample.entry["prompt_id"], "step": sample.entry["step"], "gap": sample.entry["gap"],
                   "strategy": strategy, "tier": tier, "graph": graph, "method": method, "would_certify": certified,
                   "certified": certified and tier == "certified", "winner": winner, "winner_is_reference": winner == token,
                   "index": index if certified else None, "steps": steps, "fraction": fraction, "evaluations": len(log), "log": log,
                   "phase5a_cell": sample.entry["strategies"][strategy]["tiers"][tier]["phase5a_cells"]["realistic/realistic"],
                   "cost": cost.to_json(), "timings_ms": {"cell": (time.perf_counter() - started) * 1e3}}
            totals.merge(cost)
            writer.write(out)
            print(f"search: sample {sample.index} {strategy} {tier} {graph}/{method}: certified={certified} fraction={fraction:.3f} "
                  f"evaluations={len(log)} {out['timings_ms']['cell'] / 1e3:.0f}s", flush=True)
            if failures:
                raise HardFailure


def full_check_box(lm_weight, gain, mu, artifact, sample, tier, state, y, candidate, near, neurons) -> torch.Tensor:
    """Every row of the vocabulary but the candidate and the nearest rows: those the y box settles (a valid, weaker bound),
    in blocks; the rest are returned for auto_LiRPA."""
    vocabulary = lm_weight.shape[0]
    skip = torch.zeros(vocabulary, dtype=torch.bool, device=lm_weight.device)
    skip[candidate] = True
    skip[near] = True
    remaining = []
    for start in range(0, vocabulary, 4096):
        block = torch.arange(start, min(start + 4096, vocabulary), device=lm_weight.device)
        block = block[~skip[block]]
        if not block.numel():
            continue
        # The box bound alone: the structural bound set to −∞ so that max(box, ·) is the box.
        none = torch.full((block.numel(),), -math.inf, dtype=torch.float64, device=block.device)
        zero = torch.zeros_like(none)
        if tier == "certified":
            margin = certify.certified_margins(lm_weight, gain, mu, artifact.constants, state, candidate, block, none, zero, 0.0)["margin"]
        else:
            margin = certify.structural_margins(lm_weight, gain, candidate, block, y[0], y[1], none, zero, neurons, 0.0)["margin"]
        remaining.append(block[~(margin > 0)])
    return torch.cat(remaining) if remaining else torch.zeros(0, dtype=torch.int64, device=lm_weight.device)


# Driver


def environment(artifact: Artifact) -> dict:
    import auto_LiRPA
    import numpy

    return {
        "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": numpy.__version__,
        "auto_LiRPA": {"version": auto_LiRPA.__version__, "commit": "5a098e8f9fb5786a428a024981d833d303921f2d",
                       "repository": "https://github.com/Verified-Intelligence/auto_LiRPA", "license": "BSD-3-Clause"},
        "platform": platform.platform(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"), "source_tree_sha256": source_tree_sha256(REPO_ROOT),
        "research_tree_sha256": research_tree_sha256(REPO_ROOT), "artifact_manifest": artifact.manifest["provenance"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "default_dtype": str(torch.get_default_dtype()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--part", choices=["validate", "compare", "search"], required=True)
    parser.add_argument("--samples", default=None, help="comma-separated sample indices (default: all)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    run = Path(args.run)
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    artifact = Artifact(run / "artifact")
    device = torch.device("cuda")
    indices = list(range(len(artifact.samples))) if args.samples is None else [int(v) for v in args.samples.split(",")]
    indices = [i for position, i in enumerate(indices) if position % args.shards == args.shard]
    failures: list[dict] = []
    totals = BoundCost()
    report = {"environment": environment(artifact), "samples": indices, "timings_ms": {}}
    started = time.perf_counter()
    writer = JsonlWriter(run / f"{args.part}.{args.shard}.jsonl.gz")
    try:
        for index in indices:
            sample_started = time.perf_counter()
            sample = SampleData(artifact, index, device)
            gpu_reset(device)
            if args.part == "validate":
                run_validate(artifact, sample, config, writer, failures)
            elif args.part == "compare":
                run_compare(artifact, sample, config, writer, failures, totals)
            else:
                run_search(artifact, sample, config, writer, failures, totals)
            report["timings_ms"][f"sample_{index}"] = (time.perf_counter() - sample_started) * 1e3
            print(f"{args.part}: sample {index} done in {report['timings_ms'][f'sample_{index}'] / 1e3:.0f}s", flush=True)
            del sample
            torch.cuda.empty_cache()
            if failures:
                break
    except HardFailure:
        pass
    finally:
        writer.close()
    report["timings_ms"]["total"] = (time.perf_counter() - started) * 1e3
    report["cost"] = totals.to_json()
    report["peak_device_bytes"] = gpu_peak(device)
    report["peak_rss_bytes"] = process_peak_rss()
    report["failures"] = failures
    (run / f"verifier_{args.part}_{args.shard}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failures:
        (run / "failure.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"HARD FAILURE: {failures[:3]}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
