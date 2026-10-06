"""Phase 5A2, stage 1: can auto_LiRPA bound a weight-perturbed routed expert directly, and soundly? (decision 0010)

    uv run --project research/crown_expert_oracle python research/crown_expert_oracle/probe.py --output <dir> [--part A|B|all]

Part A, a toy expert (H = 3, I = 2) with exact checks, on the CPU:
  A1  one expert, all 18 weights in boxes: auto_LiRPA's lower and upper bounds on two properties (a random c·y and the
      pairwise difference y_0 − y_1) by IBP, CROWN-IBP, CROWN and alpha-CROWN, against every one of the 2¹⁸ box vertices,
      200,000 random points of the boxes, projected-gradient searches and the true weights;
  A2  two experts, nine weights each in a three-value set {lower, centre, upper} (the rest exact): the bounds against
      all 3⁹ = 19,683 discrete realizations;
  A3  every box of zero width: every method returns the exact value (full-materialization convergence);
  A4  one matrix perturbed at a time (gate, up, down): the bound widens, so each perturbation propagates;
  A5  the reduced graph (activation boxes from the exact interval ranges of each neuron, the down box): its bound
      against the closed-form optimum of its set, checked by vertex enumeration, and against the full graph's;
  A6  F.silu written directly: whether auto_LiRPA can parse it (it has no SiLU operator).
Part B, Moonlight's shapes (6 experts, H = 2,048, I = 1,408) with synthetic weights and q6-like boxes, on the GPU: the
  cost of each method and graph (setup, bound time per call, peak device memory and RAM, specifications per call), and
  the bounds against the true weights and a projected-gradient search.

Gate (the brief's): perturbations propagate through Linear → SiLU → multiply → Linear; every bound is sound; no
custom verification code was needed. Writes probe_<part>.json; exits 1 if the gate fails.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crown_oracle.attack import pgd_minimize, reduced_optimum  # noqa: E402
from crown_oracle.graph import (  # noqa: E402
    BoundCost, Bounder, ExpertsSuffix, ReducedSuffix, boxed_input, boxed_parameter, gpu_peak, gpu_reset, process_peak_rss,
    silu, suffix_value,
)

torch.set_default_dtype(torch.float64)
torch.use_deterministic_algorithms(True)
METHODS = ("ibp", "crown-ibp", "crown", "alpha-crown")
TOLERANCE = 1e-12  # relative: float64 evaluation of the same function in a different order


# Part A: the toy expert


def toy(seed: int, experts: int, hidden: int = 3, intermediate: int = 2, width: float = 0.2):
    """Random weights (the truth), boxes around a shifted centre that contain the truth, an input, a base and C."""
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape, scale=0.8):
        return torch.randn(*shape, generator=gen) * scale

    truth = {"gate": randn(experts, intermediate, hidden), "up": randn(experts, intermediate, hidden), "down": randn(experts, hidden, intermediate)}
    boxes = {}
    for name, value in truth.items():
        radius = width * value.abs().amax(dim=-1, keepdim=True).expand_as(value)
        shift = (torch.rand(value.shape, generator=gen) * 2 - 1) * 0.6 * radius  # the centre is not the truth
        centre = value + shift
        boxes[name] = (centre - radius, centre + radius, centre)
    x = randn(1, hidden, scale=1.0)
    base = randn(hidden, scale=0.1)
    routing = (0.73, 0.41)[:experts]
    C = torch.stack([randn(hidden, scale=1.0), torch.eye(hidden)[0] - torch.eye(hidden)[1]])
    return truth, boxes, x, base, routing, C


def build_full(boxes, routing, base):
    parameters = {name: [boxed_parameter(c[e], lo[e], hi[e]) for e in range(lo.shape[0])] for name, (lo, hi, c) in boxes.items()}
    return ExpertsSuffix(parameters["gate"], parameters["up"], parameters["down"], routing, base)


def evaluate(points: dict[str, torch.Tensor], x, routing, base, C) -> torch.Tensor:
    """C·y for a batch of weight realizations ([N, K, ...] each): [N, J]."""
    y = suffix_value(x, points["gate"], points["up"], points["down"], routing, base)
    return y @ C.T


def all_methods(model, inputs, C, cost):
    bounder = Bounder(model, inputs, torch.device("cpu"), alpha_iterations=50, cost=cost)
    results = {}
    for method in METHODS:
        lower, upper = bounder.bounds(C, method)
        results[method] = (lower, upper)
    return results


def soundness(results, low_values, high_values) -> dict:
    """Per method and property: the bounds, the extreme values found, whether lower ≤ min and upper ≥ max."""
    report = {}
    for method, (lower, upper) in results.items():
        rows = []
        for j in range(lower.numel()):
            tol = TOLERANCE * (1.0 + abs(float(low_values[j])) + abs(float(high_values[j])))
            rows.append({
                "lower": float(lower[j]), "upper": float(upper[j]), "min_found": float(low_values[j]), "max_found": float(high_values[j]),
                "sound": bool(float(lower[j]) <= float(low_values[j]) + tol and float(upper[j]) >= float(high_values[j]) - tol),
                "lower_gap": float(low_values[j] - lower[j]), "upper_gap": float(upper[j] - high_values[j]),
            })
        report[method] = rows
    return report


def part_a() -> dict:
    report: dict = {}
    cost = BoundCost()

    # A1: every weight of one expert in a box.
    truth, boxes, x, base, routing, C = toy(seed=11, experts=1)
    results = all_methods(build_full(boxes, routing, base), (x,), C, cost)
    names = list(boxes)
    sizes = [boxes[n][0].numel() for n in names]
    lows = torch.cat([boxes[n][0].reshape(-1) for n in names])
    highs = torch.cat([boxes[n][1].reshape(-1) for n in names])
    count = lows.numel()
    bits = torch.tensor(list(itertools.product((0.0, 1.0), repeat=count)))  # 2^18 vertices
    flat = lows + bits * (highs - lows)
    randoms = lows + torch.rand(200_000, count, generator=torch.Generator().manual_seed(5)) * (highs - lows)

    def unflatten(rows):
        out, start = {}, 0
        for n, size in zip(names, sizes):
            out[n] = rows[:, start : start + size].reshape(-1, *boxes[n][0].shape)
            start += size
        return out

    values = torch.cat([evaluate(unflatten(flat), x, routing, base, C), evaluate(unflatten(randoms), x, routing, base, C),
                        evaluate({n: truth[n].unsqueeze(0) for n in names}, x, routing, base, C)])
    searched_low, searched_high = [], []
    for j in range(C.shape[0]):
        for sign, store in ((1.0, searched_low), (-1.0, searched_high)):
            def objective(g, u, d, j=j, sign=sign):
                return sign * (suffix_value(x, g, u, d, routing, base) @ C[j])
            starts = [[boxes[n][2] for n in names], [truth[n] for n in names]]
            value, _ = pgd_minimize(objective, [boxes[n][0] for n in names], [boxes[n][1] for n in names], starts, steps=300)
            store.append(sign * value)
    low = torch.minimum(values.min(dim=0).values, torch.tensor(searched_low))
    high = torch.maximum(values.max(dim=0).values, torch.tensor(searched_high))
    report["A1"] = {"perturbed_weights": count, "vertices": int(bits.shape[0]), "random_points": 200_000, "pgd_runs": 4,
                    "truth_value": [float(v) for v in values[-1]], "methods": soundness(results, low, high)}

    # A2: two experts, nine weights each in {lower, centre, upper}, the rest exact.
    truth2, boxes2, x2, base2, routing2, C2 = toy(seed=23, experts=2)
    chosen = [("gate", 0, 0, 0), ("gate", 0, 1, 2), ("up", 0, 0, 1), ("up", 0, 1, 1), ("down", 0, 0, 0), ("down", 0, 2, 1),
              ("gate", 1, 1, 0), ("up", 1, 0, 2), ("down", 1, 1, 1)]
    exact = {n: (truth2[n].clone(), truth2[n].clone(), truth2[n].clone()) for n in truth2}
    for name, e, r, c in chosen:
        lo, hi, centre = boxes2[name]
        exact[name][0][e, r, c], exact[name][1][e, r, c], exact[name][2][e, r, c] = lo[e, r, c], hi[e, r, c], centre[e, r, c]
    results2 = all_methods(build_full(exact, routing2, base2), (x2,), C2, cost)
    grid = list(itertools.product((0, 1, 2), repeat=len(chosen)))
    points = {n: truth2[n].unsqueeze(0).repeat(len(grid), 1, 1, 1) for n in truth2}
    choice = torch.tensor(grid)
    for position, (name, e, r, c) in enumerate(chosen):
        options = torch.stack([exact[name][0][e, r, c], exact[name][2][e, r, c], exact[name][1][e, r, c]])
        points[name][:, e, r, c] = options[choice[:, position]]
    discrete = evaluate(points, x2, routing2, base2, C2)
    report["A2"] = {"perturbed_weights": len(chosen), "discrete_points": len(grid),
                    "methods": soundness(results2, discrete.min(dim=0).values, discrete.max(dim=0).values)}

    # A3: zero-width boxes: every method returns the exact value.
    zero = {n: (truth[n].clone(), truth[n].clone(), truth[n].clone()) for n in truth}
    results3 = all_methods(build_full(zero, routing, base), (x,), C, cost)
    exact_value = evaluate({n: truth[n].unsqueeze(0) for n in truth}, x, routing, base, C)[0]
    report["A3"] = {method: {"lower": [float(v) for v in lo], "upper": [float(v) for v in hi], "exact": [float(v) for v in exact_value],
                             "max_abs_error": float(torch.maximum((lo - exact_value).abs(), (hi - exact_value).abs()).max())}
                    for method, (lo, hi) in results3.items()}

    # A4: one matrix perturbed at a time.
    report["A4"] = {}
    for perturbed in ("gate", "up", "down"):
        partial = {n: (boxes[n] if n == perturbed else zero[n]) for n in truth}
        results4 = all_methods(build_full(partial, routing, base), (x,), C, cost)
        report["A4"][perturbed] = {method: {"widths": [float(v) for v in (hi - lo)]} for method, (lo, hi) in results4.items()}

    # A5: the reduced graph on the exact activation ranges, its closed-form optimum, and vertex enumeration.
    lo_g, hi_g = _linear_range(boxes["gate"], x)
    lo_u, hi_u = _linear_range(boxes["up"], x)
    lo_s, hi_s = _silu_range(lo_g, hi_g)
    corners = torch.stack([lo_s * lo_u, lo_s * hi_u, hi_s * lo_u, hi_s * hi_u])
    a_lo, a_hi = corners.min(dim=0).values, corners.max(dim=0).values
    downs = [boxed_parameter(boxes["down"][2][e], boxes["down"][0][e], boxes["down"][1][e]) for e in range(1)]
    reduced = ReducedSuffix(downs, routing, base)
    inputs = tuple(boxed_input(a_lo[e : e + 1], a_hi[e : e + 1]) for e in range(1))
    results5 = all_methods(reduced, inputs, C, cost)
    optimum = reduced_optimum(C, boxes["down"][0], boxes["down"][1], a_lo, a_hi, torch.tensor(routing), base)
    # Vertex enumeration of the reduced problem (a bilinear form attains its extremes at box vertices).
    a_count, d_count = a_lo.numel(), boxes["down"][0].numel()
    vertex_bits = torch.tensor(list(itertools.product((0.0, 1.0), repeat=a_count + d_count)))
    a_points = a_lo.reshape(-1) + vertex_bits[:, :a_count] * (a_hi - a_lo).reshape(-1)
    d_points = boxes["down"][0].reshape(-1) + vertex_bits[:, a_count:] * (boxes["down"][1] - boxes["down"][0]).reshape(-1)
    y = base + routing[0] * torch.einsum("nhi,ni->nh", d_points.reshape(-1, *boxes["down"][0].shape[1:]), a_points)
    enumerated = (y @ C.T).min(dim=0).values
    full_lower = {m: [float(v) for v in results[m][0]] for m in METHODS}
    report["A5"] = {
        "activation_ranges": {"lower": a_lo.tolist(), "upper": a_hi.tolist()},
        "reduced": {m: [float(v) for v in results5[m][0]] for m in METHODS},
        "full": full_lower,
        "reduced_optimum": [float(v) for v in optimum],
        "vertex_minimum": [float(v) for v in enumerated],
        "optimum_matches_enumeration": bool(torch.allclose(optimum, enumerated, rtol=1e-12, atol=1e-12)),
        "reduced_sound": all(float(results5[m][0][j]) <= float(optimum[j]) + 1e-12 for m in METHODS for j in range(C.shape[0])),
        "true_minimum_full_graph": [float(v) for v in low],
    }

    # A6: F.silu written directly.
    class DirectSilu(nn.Module):
        def __init__(self, gate):
            super().__init__()
            self.gate = gate
            self.bias = nn.Parameter(torch.zeros(gate.shape[0]), requires_grad=False)

        def forward(self, x):
            return F.silu(F.linear(x, self.gate, self.bias))

    try:
        bounder = Bounder(DirectSilu(boxed_parameter(boxes["gate"][2][0], boxes["gate"][0][0], boxes["gate"][1][0])), (x,), torch.device("cpu"))
        lower, upper = bounder.bounds(torch.eye(2), "crown")
        report["A6"] = {"parsed": True, "lower": lower.tolist(), "upper": upper.tolist()}
    except Exception as error:  # noqa: BLE001 - the outcome is the finding
        report["A6"] = {"parsed": False, "error": f"{type(error).__name__}: {str(error).splitlines()[0][:300]}"}
    report["cost"] = cost.to_json()

    sound = all(row["sound"] for case in ("A1", "A2") for rows in report[case]["methods"].values() for row in rows)
    converged = all(entry["max_abs_error"] <= 1e-12 for entry in report["A3"].values())
    propagates = all(all(w > 0 for w in entry["widths"]) for per_method in report["A4"].values() for entry in per_method.values())
    report["gate"] = {"sound": sound and report["A5"]["reduced_sound"], "zero_width_exact": converged, "propagates": propagates,
                      "closed_form_checked": report["A5"]["optimum_matches_enumeration"]}
    return report


def _linear_range(box, x):
    """Exact range of W·x over a box of W (x fixed): centre ± |x|·radius, per row."""
    lo, hi, _ = box
    centre, radius = (lo + hi) * 0.5, (hi - lo) * 0.5
    v = x.reshape(-1)
    mid = torch.einsum("kih,h->ki", centre, v)
    spread = torch.einsum("kih,h->ki", radius, v.abs())
    return mid - spread, mid + spread


def _silu_range(lo, hi):
    """Exact range of silu over [lo, hi] (decreasing then increasing, minimum at x* ≈ −1.27846)."""
    argmin = torch.tensor(-1.2784645427610738, device=lo.device)
    at_lo, at_hi = silu(lo), silu(hi)
    low = torch.where((lo <= argmin) & (hi >= argmin), silu(argmin).expand_as(lo), torch.minimum(at_lo, at_hi))
    return low, torch.maximum(at_lo, at_hi)


# Part B: Moonlight's shapes, synthetic weights


def q6_box(weight: torch.Tensor):
    """A q6-like coarse level of each row (symmetric int6, scale = row max / 31) and the box its remainder leaves."""
    scale = (weight.abs().amax(dim=-1, keepdim=True) / 31).to(torch.float32).to(torch.float64)
    centre = torch.round(weight / scale) * scale
    radius = (weight - centre).abs().amax(dim=-1, keepdim=True) * (1 + 1e-12)
    return centre - radius, centre + radius, centre


def part_b(device: torch.device, alpha_iterations: int) -> dict:
    experts, hidden, intermediate = 6, 2048, 1408
    gen = torch.Generator(device=device).manual_seed(7)
    truth = {
        "gate": torch.randn(experts, intermediate, hidden, device=device, generator=gen) * 0.02,
        "up": torch.randn(experts, intermediate, hidden, device=device, generator=gen) * 0.02,
        "down": torch.randn(experts, hidden, intermediate, device=device, generator=gen) * 0.02,
    }
    boxes = {name: q6_box(value) for name, value in truth.items()}
    x = torch.randn(1, hidden, device=device, generator=gen)
    base = torch.randn(hidden, device=device, generator=gen) * 0.5
    routing = [0.42, 0.31, 0.27, 0.2, 0.15, 0.12]
    C_all = torch.randn(8, hidden, device=device, generator=gen)
    true_y = suffix_value(x, truth["gate"], truth["up"], truth["down"], routing, base)
    true_values = C_all @ true_y
    report: dict = {"shapes": {"experts": experts, "hidden": hidden, "intermediate": intermediate},
                    "perturbed_weights": sum(v.numel() for v in truth.values()), "runs": []}

    def record(graph, method, specs, run):
        cost = BoundCost()
        entry = {"graph": graph, "method": method, "specifications": specs}
        try:
            bounder = run(cost)
            gpu_reset(device)
            lower = bounder.lower(C_all[:specs], method, chunk=specs)
            entry.update(lower=[float(v) for v in lower], truth=[float(v) for v in true_values[:specs]],
                         sound=bool((lower <= true_values[:specs] + 1e-9 * (1 + true_values[:specs].abs())).all()))
        except Exception as error:  # noqa: BLE001 - the outcome is the finding
            entry["error"] = f"{type(error).__name__}: {str(error).splitlines()[0][:300]}"
            entry["traceback"] = traceback.format_exc()[-2000:]
        entry["cost"] = cost.to_json()
        report["runs"].append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k not in ("traceback", "lower", "truth")}), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def full(cost):
        return Bounder(build_full(boxes, routing, base), (x,), device, alpha_iterations=alpha_iterations, cost=cost)

    # The activation boxes: each neuron's exact range under its gate and up boxes (independent rows).
    lo_g, hi_g = _linear_range(boxes["gate"], x)
    lo_u, hi_u = _linear_range(boxes["up"], x)
    lo_s, hi_s = _silu_range(lo_g, hi_g)
    corners = torch.stack([lo_s * lo_u, lo_s * hi_u, hi_s * lo_u, hi_s * hi_u])
    a_lo, a_hi = corners.min(dim=0).values, corners.max(dim=0).values
    del corners

    def reduced(cost):
        downs = [boxed_parameter(boxes["down"][2][e], boxes["down"][0][e], boxes["down"][1][e]) for e in range(experts)]
        inputs = tuple(boxed_input(a_lo[e : e + 1], a_hi[e : e + 1]) for e in range(experts))
        return Bounder(ReducedSuffix(downs, routing, base), inputs, device, alpha_iterations=alpha_iterations, cost=cost)

    for specs in (1, 4, 8):
        record("reduced", "crown-ibp", specs, reduced)
    record("reduced", "alpha-crown", 1, reduced)
    for specs in (1, 4):
        record("full", "crown-ibp", specs, full)
    record("full", "alpha-crown", 1, full)
    record("full", "crown", 1, full)

    optimum = reduced_optimum(C_all, boxes["down"][0], boxes["down"][1], a_lo, a_hi, torch.tensor(routing, device=device), base)
    report["reduced_optimum"] = [float(v) for v in optimum]

    # A projected-gradient search on the full graph for the first property: an achievable value inside the boxes.
    started = time.perf_counter()
    names = ("gate", "up", "down")

    def objective(g, u, d):
        return suffix_value(x, g, u, d, routing, base) @ C_all[0]

    value, _ = pgd_minimize(objective, [boxes[n][0] for n in names], [boxes[n][1] for n in names], [[boxes[n][2] for n in names]], steps=30)
    report["pgd"] = {"property": 0, "value": value, "truth": float(true_values[0]), "ms": (time.perf_counter() - started) * 1e3}
    report["peak_rss_bytes"] = process_peak_rss()
    lowers = [r for r in report["runs"] if "lower" in r]
    report["gate"] = {"sound": all(r["sound"] for r in lowers) and all(r["lower"][0] <= value + 1e-9 * (1 + abs(value)) for r in lowers)}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--part", choices=["A", "B", "all"], default="all")
    parser.add_argument("--alpha-iterations", type=int, default=20)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    environment = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
                   "auto_LiRPA": __import__("auto_LiRPA").__version__, "platform": platform.platform(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "python_hash_seed": os.environ.get("PYTHONHASHSEED")}
    passed = True
    for part in ("A", "B") if args.part == "all" else (args.part,):
        started = time.perf_counter()
        if part == "A":
            result = part_a()
        else:
            result = part_b(torch.device("cuda"), args.alpha_iterations)
        result["environment"] = environment
        result["wall_ms"] = (time.perf_counter() - started) * 1e3
        (output / f"probe_{part}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"part {part}: gate {result['gate']}", flush=True)
        passed = passed and all(result["gate"].values())
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
