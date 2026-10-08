"""Phase 5A2, stage 1.5: can auto_LiRPA use Phase 5A's L2 remainder norms, and do they help? (decision 0010)

    uv run --project research/crown_expert_oracle python research/crown_expert_oracle/probe_l2.py --output <dir> [--part A|B|all]

Part A, toys on the CPU (float64). Every lower bound is checked against its set's exact minimum (closed forms, or
`rowsets`: the activations' exact box and all its vertices) and against random points of the set; a bound above an
attained value is unsound.
  A1  one L2 ball on a whole weight, exact input (case A): CROWN against c·W₀·x − ε‖c‖₂‖x‖₂; random points of the ball;
      zero radius; shrinking radius
  A2  one ball per matrix (case B): gate, up and down of one expert, perturbed one kind at a time and all together
  A3  one ball per page (case C) and per row (case D), stacked (one weight node) or split (one linear per group), exact
      input: independence against the per-group closed form; the point with every group at its extreme at once
  A4  per-row balls in the expert: gate and up only (down exact), all three, and the reduced graph (the activations'
      exact box × per-row balls of down)
  A5  box against L2 on one q6-quantized toy expert: the exact minimum of each set (box, L2, both) and CROWN's bound on
      each where upstream is sound
Every case runs in auto_LiRPA's two interval modes for L2 roots (centre: upstream's default; box hull:
AUTOLIRPA_L2_DEBUG=1), with IBP, CROWN-IBP, CROWN and alpha-CROWN.
Part B, one expert at Moonlight's shape (synthetic weights, q6-like levels) on the GPU: the cost of per-row and per-page
  balls of down on the reduced graph (construction, bound time, device memory) and its bound against the L2 set's.

Writes probe_l2_<part>.json. No verification code: the graphs are PyTorch modules, every bound is auto_LiRPA's.
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
import probe  # noqa: E402  (stage 1's q6-like levels of synthetic weights)
from crown_oracle import rowsets  # noqa: E402
from crown_oracle.graph import BoundCost, Bounder, ExpertsSuffix, ReducedSuffix, boxed_input, boxed_parameter, leave, process_peak_rss, suffix_value  # noqa: E402
from crown_oracle.l2 import (  # noqa: E402
    ExactLinear, LinearExperts, LinearReduced, SplitLinear, StackedLinear, l2_interval_mode, l2_parameter, perturbed_count, uniform_in_balls,
)

torch.set_default_dtype(torch.float64)
torch.use_deterministic_algorithms(True)
CPU = torch.device("cpu")
METHODS = ("ibp", "crown-ibp", "crown", "alpha-crown")
MODES = {"centre": False, "box_hull": True}


def tol(value: torch.Tensor) -> torch.Tensor:
    return 1e-10 * (1.0 + value.abs())


def upward(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x * (1.0 + 2.0**-40), torch.full_like(x, math.inf))


def run(model_factory, inputs, C, methods=METHODS, modes=MODES) -> dict:
    """auto_LiRPA's lower and upper bounds of C·output per interval mode and method (a failure is recorded, not raised)."""
    out = {}
    for mode, hull in modes.items():
        entry = {}
        with l2_interval_mode(hull):
            try:
                bounder = Bounder(model_factory(), inputs, CPU, alpha_iterations=30)
            except Exception as error:  # noqa: BLE001 - the outcome is the finding
                out[mode] = {"construction_error": f"{type(error).__name__}: {str(error).splitlines()[0][:300]}"}
                continue
            for method in methods:
                try:
                    lower, upper = bounder.bounds(C, method)
                    entry[method] = {"lower": lower.tolist(), "upper": upper.tolist()}
                except Exception as error:  # noqa: BLE001
                    entry[method] = {"error": f"{type(error).__name__}: {str(error).splitlines()[0][:300]}",
                                     "where": traceback.format_exc().strip().splitlines()[-3][:200]}
        out[mode] = entry
    return out


def judge(results: dict, low: torch.Tensor, high: torch.Tensor, exact_low=None, exact_high=None) -> dict:
    """Per mode and method: sound (lower ≤ every attained value, upper ≥), exact (equal to the exact extremes when given),
    the gaps to them. `low`/`high`: the smallest and largest attained values (random points, witnesses, the truth)."""
    verdicts = {}
    for mode, entry in results.items():
        verdicts[mode] = {}
        for method, value in entry.items():
            if not isinstance(value, dict) or "lower" not in value:
                verdicts[mode][method] = {"ran": False, **(value if isinstance(value, dict) else {"error": str(value)})}
                continue
            lower, upper = torch.tensor(value["lower"]), torch.tensor(value["upper"])
            reference_low = low if exact_low is None else torch.minimum(low, exact_low)
            reference_high = high if exact_high is None else torch.maximum(high, exact_high)
            sound = bool((lower <= reference_low + tol(reference_low)).all() and (upper >= reference_high - tol(reference_high)).all())
            row = {"ran": True, "sound": sound, "lower": value["lower"], "upper": value["upper"],
                   "worst_lower_excess": float((lower - reference_low).max()), "worst_upper_shortfall": float((reference_high - upper).max())}
            if exact_low is not None:
                row["lower_gap_to_exact"] = (exact_low - lower).tolist()
                row["exact"] = bool(((lower - exact_low).abs() <= tol(exact_low)).all() and ((upper - exact_high).abs() <= tol(exact_high)).all())
            verdicts[mode][method] = row
    return verdicts


# Part A


class OneBall(nn.Module):
    """F.linear(x, W) with all of W in one L2 ball."""

    def __init__(self, centre: torch.Tensor, radius: float) -> None:
        super().__init__()
        self.weight = l2_parameter(centre, radius)
        self.bias = nn.Parameter(torch.zeros(centre.shape[0]), requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def a1() -> dict:
    gen = torch.Generator().manual_seed(1)
    W0, x, C = torch.randn(3, 4, generator=gen), torch.randn(1, 4, generator=gen), torch.randn(2, 3, generator=gen)
    report = {"radii": {}}
    for eps in (0.3, 0.15, 0.075, 0.0):
        exact_low = C @ W0 @ x.reshape(-1) - eps * torch.linalg.vector_norm(C, dim=1) * torch.linalg.vector_norm(x)
        exact_high = C @ W0 @ x.reshape(-1) + eps * torch.linalg.vector_norm(C, dim=1) * torch.linalg.vector_norm(x)
        points = uniform_in_balls(W0.unsqueeze(0), torch.tensor([eps]), 100_000, gen)[:, 0]
        values = torch.einsum("jr,nrc,c->nj", C, points, x.reshape(-1))
        results = run(lambda eps=eps: OneBall(W0, eps), (x,), C)
        report["radii"][str(eps)] = {"closed_form_lower": exact_low.tolist(), "random_min": values.min(0).values.tolist(),
                                     "verdicts": judge(results, values.min(0).values, values.max(0).values, exact_low, exact_high)}
    return report


def project_balls(points: list[torch.Tensor], centres: list[torch.Tensor], radii: list[float]) -> None:
    with torch.no_grad():
        for p, c, r in zip(points, centres, radii):
            d = p - c
            n = torch.linalg.vector_norm(d)
            if n > r:
                p.copy_(c + d * (r / n))


def pgd_balls(objective, centres: list[torch.Tensor], radii: list[float], starts: list[list[torch.Tensor]], steps: int = 300) -> float:
    """Projected normalized-gradient descent of a scalar over a product of L2 balls (one ball per tensor): the smallest
    value seen (an attained value)."""
    best = math.inf
    for start in starts:
        point = [s.clone().requires_grad_(True) for s in start]
        project_balls(point, centres, radii)
        for step in range(steps + 1):
            value = objective(*point)
            best = min(best, float(value.detach()))
            if step == steps:
                break
            grads = torch.autograd.grad(value, point)
            scale = 0.5 * (1.0 - step / steps) + 1e-3
            with torch.no_grad():
                for p, g, r in zip(point, grads, radii):
                    n = torch.linalg.vector_norm(g)
                    if n > 0 and r > 0:
                        p.sub_(scale * r * g / n)
            project_balls(point, centres, radii)
    return best


def a2() -> dict:
    """One expert, one ball per matrix (each holding its true matrix), perturbed kind by kind and all together."""
    gen = torch.Generator().manual_seed(2)
    H, I = 4, 3
    truth = {"gate": torch.randn(I, H, generator=gen), "up": torch.randn(I, H, generator=gen), "down": torch.randn(H, I, generator=gen)}
    centre = {n: v + 0.15 * torch.randn(v.shape, generator=gen) for n, v in truth.items()}
    radius = {n: float(torch.linalg.vector_norm(truth[n] - centre[n])) * 1.5 for n in truth}
    x, base, C = torch.randn(1, H, generator=gen), 0.1 * torch.randn(H, generator=gen), torch.randn(3, H, generator=gen)
    routing = [0.7]
    names = ("gate", "up", "down")
    report = {}
    for perturbed in (("down",), ("gate", "up"), ("gate", "up", "down")):
        def factory(perturbed=perturbed):
            mods = {n: (OneBall(centre[n], radius[n]) if n in perturbed else ExactLinear(truth[n])) for n in names}
            return LinearExperts([mods["gate"]], [mods["up"]], [mods["down"]], routing, base)
        results = run(factory, (x,), C)
        # Attained values: random points of the balls, projected-gradient searches (both signs), the truth.
        chosen = {n: (centre[n] if n in perturbed else truth[n]) for n in names}
        rad = {n: (radius[n] if n in perturbed else 0.0) for n in names}
        samples = {n: (uniform_in_balls(chosen[n].unsqueeze(0), torch.tensor([rad[n]]), 50_000, gen)[:, 0] if rad[n] > 0 else chosen[n].expand(50_000, *chosen[n].shape))
                   for n in names}
        values = suffix_value(x, samples["gate"].unsqueeze(1), samples["up"].unsqueeze(1), samples["down"].unsqueeze(1), routing, base) @ C.T
        low, high = values.min(0).values.clone(), values.max(0).values.clone()
        for j in range(C.shape[0]):
            for sign in (1.0, -1.0):
                def objective(g, u, d, j=j, sign=sign):
                    return sign * (suffix_value(x, g.unsqueeze(0), u.unsqueeze(0), d.unsqueeze(0), routing, base) @ C[j])
                found = sign * pgd_balls(objective, [chosen[n] for n in names], [rad[n] for n in names],
                                         [[chosen[n] for n in names], [truth[n] for n in names]])
                if sign > 0:
                    low[j] = min(float(low[j]), found)
                else:
                    high[j] = max(float(high[j]), found)
        exact = None
        if perturbed == ("down",):  # a exact: Δ·(b + w·D₀·a) ∓ w·ε‖Δ‖‖a‖ (one ball over all of D)
            a = rowsets.silu(truth["gate"] @ x.reshape(-1)) * (truth["up"] @ x.reshape(-1))
            centre_value = C @ (base + routing[0] * centre["down"] @ a)
            spread = routing[0] * radius["down"] * torch.linalg.vector_norm(C, dim=1) * torch.linalg.vector_norm(a)
            exact = (centre_value - spread, centre_value + spread)
        report["+".join(perturbed)] = {"attained_min": low.tolist(), "verdicts": judge(results, low, high, *(exact or (None, None)))}
    return report


def a3() -> dict:
    """Pages and rows of one weight, exact input: each group concretized over its own ball."""
    gen = torch.Generator().manual_seed(3)
    R, Cn = 6, 4
    W0, x, C = torch.randn(R, Cn, generator=gen), torch.randn(1, Cn, generator=gen), torch.randn(2, R, generator=gen)
    rho = 0.05 + 0.3 * torch.rand(R, generator=gen)
    v = x.reshape(-1)
    report = {}
    for label, size in (("pages_of_2", 2), ("rows", 1)):
        ranges = [(s, min(s + size, R)) for s in range(0, R, size)]
        eps = torch.stack([torch.linalg.vector_norm(rho[s:e]) for s, e in ranges])
        eps_up = upward(eps)
        # The per-group closed form and the joint ball's (what a single ball of radius ‖ρ‖ would give).
        per_group = sum(C[:, s:e] @ W0[s:e] @ v - eps_up[g] * torch.linalg.vector_norm(C[:, s:e], dim=1) * torch.linalg.vector_norm(v)
                        for g, (s, e) in enumerate(ranges))
        per_group_high = sum(C[:, s:e] @ W0[s:e] @ v + eps_up[g] * torch.linalg.vector_norm(C[:, s:e], dim=1) * torch.linalg.vector_norm(v)
                             for g, (s, e) in enumerate(ranges))
        joint = C @ W0 @ v - float(upward(torch.linalg.vector_norm(rho))) * torch.linalg.vector_norm(C, dim=1) * torch.linalg.vector_norm(v)
        # Random points of the product of the groups' balls, and the point with every group at its extreme for C[0].
        centres = torch.stack([W0[s:e] for s, e in ranges]) if size > 1 else W0.unsqueeze(1)
        points = uniform_in_balls(centres, eps, 50_000, gen).reshape(50_000, R, Cn)
        values = torch.einsum("jr,nrc,c->nj", C, points, v)
        extreme = W0.clone()
        for g, (s, e) in enumerate(ranges):
            direction = -torch.outer(C[0, s:e], v)
            extreme[s:e] = W0[s:e] + eps[g] * direction / torch.linalg.vector_norm(direction)
        at_extreme = float(C[0] @ extreme @ v)
        entry = {"per_group_closed_form": per_group.tolist(), "joint_ball_closed_form": joint.tolist(), "every_group_at_its_extreme": at_extreme,
                 "extreme_outside_one_ball_of_the_largest_radius": bool(float(torch.linalg.vector_norm(extreme - W0)) > float(eps.max()) * (1 + 1e-9))}
        for assembly, cls in (("stacked", StackedLinear), ("split", SplitLinear)):
            results = run(lambda cls=cls, size=size: cls(W0, rho, size), (x,), C, methods=("ibp", "crown-ibp", "crown"))
            low = torch.minimum(values.min(0).values, torch.tensor([at_extreme, math.inf]))
            entry[assembly] = judge(results, low, values.max(0).values, per_group, per_group_high)
        report[label] = entry
    return report


def toy_expert(seed: int, experts: int = 2, hidden: int = 6, intermediate: int = 4):
    """A small routed layer: true weights, their q6-like levels (stage 1's `probe.q6_box`: centre, the remainder's L∞
    box), each row's L2 remainder norm, an input, the base, routing weights and contenders' Δ."""
    gen = torch.Generator().manual_seed(seed)
    truth = {"gate": torch.randn(experts, intermediate, hidden, generator=gen), "up": torch.randn(experts, intermediate, hidden, generator=gen),
             "down": torch.randn(experts, hidden, intermediate, generator=gen)}
    levels = {}
    for name, value in truth.items():
        lo, hi, centre = probe.q6_box(value)
        l2 = upward(torch.linalg.vector_norm(value - centre, dim=-1))
        levels[name] = {"lower": lo, "upper": hi, "centre": centre, "l2": l2}
    x = torch.randn(1, hidden, generator=gen)
    base = 0.1 * torch.randn(hidden, generator=gen)
    routing = [0.6, 0.35][:experts]
    delta = torch.randn(3, hidden, generator=gen)
    return truth, levels, x, base, routing, delta


def toy_sets(levels, kind: str) -> dict[str, list[rowsets.RowSets]]:
    """Per matrix and expert: the rows' sets of one kind (box, l2, both)."""
    out = {}
    for name, level in levels.items():
        per_expert = []
        for e in range(level["centre"].shape[0]):
            centre, l2 = level["centre"][e], level["l2"][e]
            lower, upper = level["lower"][e], level["upper"][e]
            if kind == "box":
                per_expert.append(rowsets.RowSets((lower + upper) * 0.5, torch.full_like(l2, math.inf), lower, upper))
            elif kind == "l2":
                per_expert.append(rowsets.RowSets(centre, l2))
            else:
                per_expert.append(rowsets.RowSets(centre, l2, lower, upper))
        out[name] = per_expert
    return out


def exact_minimum(sets, x, routing, base, delta, down_exact=None):
    """min over the set of Δ·y per contender (rowsets' reduction: the activations' exact box, every vertex), and the
    value of a witness built at the best vertex (weights of the set; it must equal the minimum). `down_exact`: down known
    exactly (its rows are points)."""
    v = x.reshape(-1)
    total, attained = delta @ base, delta @ base
    for e in range(len(routing)):
        ranges = rowsets.activation_range(sets["gate"][e], sets["up"][e], v)
        a_lo, a_hi = ranges.a
        bits = torch.tensor(list(itertools.product((0.0, 1.0), repeat=a_lo.numel())))
        down = sets["down"][e] if down_exact is None else rowsets.RowSets(down_exact[e], torch.zeros(down_exact[e].shape[0]))
        values = torch.stack([rowsets.down_value(down, delta, a) for a in a_lo + bits * (a_hi - a_lo)])  # [V, J]
        best = values.min(dim=0)
        total = total + routing[e] * best.values
        found = []
        for j in range(delta.shape[0]):
            anchors = {"gate": sets["gate"][e].centre, "up": sets["up"][e].centre, "down": down.centre}
            w = rowsets.realize(sets["gate"][e], sets["up"][e], down, v, delta[j], bits[best.indices[j]], ranges, anchors)
            if w.largest_excess > 1e-12:
                raise AssertionError("a witness left its set")
            found.append(float(delta[j] @ (w.down @ w.a)))
        attained = attained + routing[e] * torch.tensor(found)
    return total, attained


def extremes(sets, x, routing, base, delta, down_exact=None) -> dict:
    """The exact minimum and maximum of Δ·y over the set, and the witnesses' values at both."""
    low, low_attained = exact_minimum(sets, x, routing, base, delta, down_exact)
    high, high_attained = exact_minimum(sets, x, routing, base, -delta, down_exact)  # max Δ·y = −min (−Δ)·y
    return {"low": low, "high": -high, "low_attained": low_attained, "high_attained": -high_attained}


def a4() -> dict:
    """Per-row balls in the expert graph (q6 centres, L2 remainder radii)."""
    truth, levels, x, base, routing, delta = toy_expert(4)
    sets = toy_sets(levels, "l2")
    report = {}
    names = ("gate", "up", "down")

    def random_values(perturbed, count=20_000, seed=5):
        gen = torch.Generator().manual_seed(seed)
        draws = {}
        for n in names:
            centre = levels[n]["centre"] if n in perturbed else truth[n]
            if n in perturbed:
                draws[n] = torch.stack([uniform_in_balls(centre[e].unsqueeze(1), levels[n]["l2"][e], count, gen)[:, :, 0] for e in range(centre.shape[0])], dim=1)
            else:
                draws[n] = centre.unsqueeze(0).expand(count, *centre.shape)
        return suffix_value(x, draws["gate"], draws["up"], draws["down"], routing, base) @ delta.T

    for label, perturbed in (("gate+up", ("gate", "up")), ("gate+up+down", names)):
        ex = extremes(sets, x, routing, base, delta, down_exact=None if "down" in perturbed else truth["down"])
        values = random_values(perturbed)
        entry = {"exact_min": ex["low"].tolist(), "witness_at_exact_min": ex["low_attained"].tolist(), "random_min": values.min(0).values.tolist()}
        for assembly, cls in (("stacked", StackedLinear), ("split", SplitLinear)):
            def factory(cls=cls, perturbed=perturbed):
                mods = {n: [(cls(levels[n]["centre"][e], levels[n]["l2"][e], 1) if n in perturbed else ExactLinear(truth[n][e])) for e in range(len(routing))] for n in names}
                return LinearExperts(mods["gate"], mods["up"], mods["down"], routing, base)
            results = run(factory, (x,), delta)
            entry[assembly] = judge(results, torch.minimum(values.min(0).values, ex["low_attained"]), torch.maximum(values.max(0).values, ex["high_attained"]),
                                    ex["low"], ex["high"])
        report[label] = entry
    # The reduced graph: the activations' exact box (from the gate and up rows' balls) × per-row balls of down.
    v = x.reshape(-1)
    boxes = [rowsets.activation_range(sets["gate"][e], sets["up"][e], v).a for e in range(len(routing))]
    ex = extremes(sets, x, routing, base, delta)
    entry = {"exact_min": ex["low"].tolist()}
    inputs = tuple(boxed_input(lo.reshape(1, -1), hi.reshape(1, -1)) for lo, hi in boxes)
    for assembly, cls in (("stacked", StackedLinear), ("split", SplitLinear)):
        results = run(lambda cls=cls: LinearReduced([cls(levels["down"]["centre"][e], levels["down"]["l2"][e], 1) for e in range(len(routing))], routing, base),
                      inputs, delta)
        entry[assembly] = judge(results, ex["low_attained"], ex["high_attained"], ex["low"], ex["high"])
    # One ball for all of down (a root parameter, no concatenation): its radius must hold every row's ball at once, so its
    # set is larger than the rows' product (soundness only, against the product's attained values).
    joint = [float(upward(torch.linalg.vector_norm(levels["down"]["l2"][e]))) for e in range(len(routing))]
    results = run(lambda: LinearReduced([OneBall(levels["down"]["centre"][e], joint[e]) for e in range(len(routing))], routing, base), inputs, delta)
    entry["one_ball_per_matrix"] = judge(results, ex["low_attained"], ex["high_attained"])
    report["reduced"] = entry
    return report


def a5() -> dict:
    """Box against L2 on one q6-quantized toy expert: the exact minimum of each set and CROWN on each where sound."""
    truth, levels, x, base, routing, delta = toy_expert(6)
    v = x.reshape(-1)
    report = {"truth": (delta @ suffix_value(x, truth["gate"], truth["up"], truth["down"], routing, base)).tolist()}
    found = {}
    for kind in ("box", "l2", "both"):
        sets = toy_sets(levels, kind)
        found[kind] = extremes(sets, x, routing, base, delta)
        widths = [rowsets.activation_range(sets["gate"][e], sets["up"][e], v).a for e in range(len(routing))]
        report[kind] = {"exact_min": found[kind]["low"].tolist(), "witness_at_exact_min": found[kind]["low_attained"].tolist(),
                        "activation_width_sum": [float((hi - lo).sum()) for lo, hi in widths]}

    def boxes_of(kind):
        sets = toy_sets(levels, kind)
        return tuple(boxed_input(lo.reshape(1, -1), hi.reshape(1, -1)) for lo, hi in
                     (rowsets.activation_range(sets["gate"][e], sets["up"][e], v).a for e in range(len(routing))))

    def box_param(n, e):
        return boxed_parameter(levels[n]["centre"][e], levels[n]["lower"][e], levels[n]["upper"][e])

    E = range(len(routing))
    ex = found["box"]
    # CROWN on the box set: stage 1's graph (every weight in its box), and the reduced graph on the box set's exact activations.
    box_full = run(lambda: ExpertsSuffix([box_param("gate", e) for e in E], [box_param("up", e) for e in E], [box_param("down", e) for e in E], routing, base),
                   (x,), delta, modes={"centre": False})
    box_reduced = run(lambda: ReducedSuffix([box_param("down", e) for e in E], routing, base), boxes_of("box"), delta, modes={"centre": False})
    report["box"]["crown_full"] = judge(box_full, ex["low_attained"], ex["high_attained"])
    report["box"]["crown_reduced"] = judge(box_reduced, ex["low_attained"], ex["high_attained"], ex["low"], ex["high"])
    # CROWN on the L2 set: the reduced graph on its exact activations, down's rows each a root (split) in its ball.
    ex = found["l2"]
    l2_reduced = run(lambda: LinearReduced([SplitLinear(levels["down"]["centre"][e], levels["down"]["l2"][e], 1) for e in E], routing, base),
                     boxes_of("l2"), delta)
    report["l2"]["crown_reduced"] = judge(l2_reduced, ex["low_attained"], ex["high_attained"], ex["low"], ex["high"])
    return report


def part_a() -> dict:
    report = {}
    for name, fn in (("A1", a1), ("A2", a2), ("A3", a3), ("A4", a4), ("A5", a5)):
        started = time.perf_counter()
        report[name] = fn()
        report[name]["wall_s"] = time.perf_counter() - started
        print(f"{name} done in {report[name]['wall_s']:.1f}s", flush=True)
    return report


# Part B


def part_b(device: torch.device) -> dict:
    """One expert at Moonlight's shape: the reduced graph with down's rows in L2 balls, grouped in pages or rows."""
    hidden, intermediate = 2048, 1408
    gen = torch.Generator(device=device).manual_seed(7)
    truth = {"gate": torch.randn(intermediate, hidden, device=device, generator=gen) * 0.02,
             "up": torch.randn(intermediate, hidden, device=device, generator=gen) * 0.02,
             "down": torch.randn(hidden, intermediate, device=device, generator=gen) * 0.02}
    levels = {}
    for name, value in truth.items():
        lo, hi, centre = probe.q6_box(value)
        levels[name] = {"centre": centre, "l2": upward(torch.linalg.vector_norm(value - centre, dim=-1)), "lower": lo, "upper": hi}
    x = torch.randn(1, hidden, device=device, generator=gen)
    base = torch.randn(hidden, device=device, generator=gen) * 0.5
    delta = torch.randn(8, hidden, device=device, generator=gen)
    routing = [0.42]
    v = x.reshape(-1)
    l2 = {n: rowsets.RowSets(levels[n]["centre"], levels[n]["l2"]) for n in truth}
    ranges = rowsets.activation_range(l2["gate"], l2["up"], v)
    a_lo, a_hi = ranges.a
    true_a = rowsets.silu(truth["gate"] @ v) * (truth["up"] @ v)
    truth_value = delta @ (base + routing[0] * truth["down"] @ true_a)
    M = delta @ levels["down"]["centre"]
    c = delta.abs() @ levels["down"]["l2"]
    lower_bound = delta @ base + routing[0] * rowsets.l2_decoupled(M, c, a_lo, a_hi)
    attained, _ = rowsets.l2_vertex_search(M, c, a_lo, a_hi)
    attained = delta @ base + routing[0] * attained
    report = {"shapes": {"hidden": hidden, "intermediate": intermediate}, "truth": truth_value.tolist(),
              "l2_set_decoupled_lower": lower_bound.tolist(), "l2_set_attained": attained.tolist(), "runs": []}
    for label, size in (("matrix", None), ("pages_128", 128), ("pages_16", 16), ("rows", 1)):
        entry = {"groups": label}
        cost = BoundCost()
        try:
            with l2_interval_mode(True):
                started = time.perf_counter()
                model = LinearReduced([SplitLinear(levels["down"]["centre"], levels["down"]["l2"], size)], routing, base).to(device)
                entry["perturbed_parameters"] = perturbed_count(model)
                entry["module_build_s"] = time.perf_counter() - started
                bounder = Bounder(model, (boxed_input(a_lo.reshape(1, -1), a_hi.reshape(1, -1)),), device, cost=cost)
                lower = bounder.lower(delta, "crown", chunk=8)
            entry.update(lower=lower.tolist(), sound=bool((lower <= attained + tol(attained)).all() and (lower <= truth_value + tol(truth_value)).all()),
                         setup_s=cost.setup_ms / 1e3, bound_s=cost.bound_ms / 1e3, peak_device_bytes=cost.peak_device_bytes, peak_rss_bytes=process_peak_rss())
        except Exception as error:  # noqa: BLE001 - the outcome is the finding
            entry["error"] = f"{type(error).__name__}: {str(error).splitlines()[0][:300]}"
        report["runs"].append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "lower"}), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--part", choices=["A", "B", "all"], default="all")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    environment = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
                   "auto_LiRPA": __import__("auto_LiRPA").__version__, "platform": platform.platform(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "python_hash_seed": os.environ.get("PYTHONHASHSEED")}
    for part in ("A", "B") if args.part == "all" else (args.part,):
        started = time.perf_counter()
        result = part_a() if part == "A" else part_b(torch.device("cuda"))
        result["environment"] = environment
        result["wall_ms"] = (time.perf_counter() - started) * 1e3
        (output / f"probe_l2_{part}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"part {part} written", flush=True)
    return 0


if __name__ == "__main__":
    leave(main())
