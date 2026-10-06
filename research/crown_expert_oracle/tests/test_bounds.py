"""auto_LiRPA's bounds on the verification graphs (Phase 5A2): sound against every tested realization, exact when no
weight is uncertain, widened by every perturbed matrix, and the reduced graph against its set's exact optimum."""

from __future__ import annotations

import itertools

import pytest
import torch

from crown_oracle.attack import pgd_minimize, reduced_optimum
from crown_oracle.graph import Bounder, ExpertsSuffix, ReducedSuffix, boxed_input, boxed_parameter, silu, suffix_value

CPU = torch.device("cpu")
METHODS = ("crown-ibp", "crown", "alpha-crown")


def problem(seed: int, experts: int = 2, hidden: int = 3, intermediate: int = 2, width: float = 0.15):
    gen = torch.Generator().manual_seed(seed)
    truth = {"gate": torch.randn(experts, intermediate, hidden, generator=gen), "up": torch.randn(experts, intermediate, hidden, generator=gen),
             "down": torch.randn(experts, hidden, intermediate, generator=gen)}
    boxes = {}
    for name, value in truth.items():
        radius = width * value.abs().amax(dim=-1, keepdim=True).expand_as(value)
        centre = value + (torch.rand(value.shape, generator=gen) * 2 - 1) * 0.5 * radius
        boxes[name] = (centre - radius, centre + radius, centre)
    x = torch.randn(1, hidden, generator=gen)
    base = torch.randn(hidden, generator=gen) * 0.1
    routing = [0.6, 0.35, 0.2][:experts]
    C = torch.randn(3, hidden, generator=gen)
    return truth, boxes, x, base, routing, C


def full_model(boxes, routing, base):
    params = {n: [boxed_parameter(c[e], lo[e], hi[e]) for e in range(lo.shape[0])] for n, (lo, hi, c) in boxes.items()}
    return ExpertsSuffix(params["gate"], params["up"], params["down"], routing, base)


def realizations(boxes, count: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    return {n: lo + torch.rand(count, *lo.shape, generator=gen) * (hi - lo) for n, (lo, hi, _) in boxes.items()}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bounds_contain_every_tested_realization(seed):
    truth, boxes, x, base, routing, C = problem(seed)
    bounder = Bounder(full_model(boxes, routing, base), (x,), CPU, alpha_iterations=20)
    points = realizations(boxes, 20_000, seed)
    # The vertices of the first expert's gate and the second's down, the other weights at random points.
    vertex_names = [("gate", 0), ("down", 1)]
    lows = torch.cat([boxes[n][0][e].reshape(-1) for n, e in vertex_names])
    highs = torch.cat([boxes[n][1][e].reshape(-1) for n, e in vertex_names])
    bits = torch.tensor(list(itertools.product((0.0, 1.0), repeat=lows.numel())))
    corners = lows + bits * (highs - lows)
    vertex_points = {n: v[: bits.shape[0]].clone() for n, v in realizations(boxes, bits.shape[0], seed + 100).items()}
    start = 0
    for n, e in vertex_names:
        size = boxes[n][0][e].numel()
        vertex_points[n][:, e] = corners[:, start : start + size].reshape(-1, *boxes[n][0][e].shape)
        start += size
    values = torch.cat([suffix_value(x, points["gate"], points["up"], points["down"], routing, base) @ C.T,
                        suffix_value(x, vertex_points["gate"], vertex_points["up"], vertex_points["down"], routing, base) @ C.T,
                        (suffix_value(x, truth["gate"], truth["up"], truth["down"], routing, base) @ C.T).unsqueeze(0)])
    for method in METHODS:
        lower, upper = bounder.bounds(C, method)
        assert bool((lower <= values.min(dim=0).values + 1e-12).all()), method
        assert bool((upper >= values.max(dim=0).values - 1e-12).all()), method


def test_lower_margin_never_exceeds_a_searched_realization():
    """The margin property: auto_LiRPA's lower bound on a pairwise difference is below the smallest value a
    projected-gradient search finds inside the boxes."""
    truth, boxes, x, base, routing, C = problem(7)
    pair = torch.zeros(1, C.shape[1])
    pair[0, 0], pair[0, 1] = 1.0, -1.0
    names = ("gate", "up", "down")

    def objective(g, u, d):
        return suffix_value(x, g, u, d, routing, base) @ pair[0]

    found, _ = pgd_minimize(objective, [boxes[n][0] for n in names], [boxes[n][1] for n in names], [[boxes[n][2] for n in names], [truth[n] for n in names]], steps=200)
    bounder = Bounder(full_model(boxes, routing, base), (x,), CPU, alpha_iterations=30)
    for method in METHODS:
        assert float(bounder.lower(pair, method, chunk=1)[0]) <= found + 1e-12, method


def test_exact_weights_give_the_exact_value():
    """Full materialization: every box a point, every method returns the graph's own value, which equals the batched
    forward and the module's forward."""
    truth, _, x, base, routing, C = problem(3)
    points = {n: (v, v, v) for n, v in truth.items()}
    model = full_model(points, routing, base)
    exact = suffix_value(x, truth["gate"], truth["up"], truth["down"], routing, base) @ C.T
    with torch.no_grad():
        module = model(x).reshape(-1) @ C.T
    assert torch.allclose(module, exact, rtol=1e-14, atol=1e-14)
    bounder = Bounder(model, (x,), CPU)
    for method in ("ibp", *METHODS):
        lower, upper = bounder.bounds(C, method)
        assert torch.allclose(lower, exact, rtol=1e-12, atol=1e-12), method
        assert torch.allclose(upper, exact, rtol=1e-12, atol=1e-12), method


@pytest.mark.parametrize("perturbed", ["gate", "up", "down"])
def test_each_weight_perturbation_propagates(perturbed):
    truth, boxes, x, base, routing, C = problem(4)
    partial = {n: (boxes[n] if n == perturbed else (truth[n], truth[n], truth[n])) for n in truth}
    bounder = Bounder(full_model(partial, routing, base), (x,), CPU)
    for method in METHODS:
        lower, upper = bounder.bounds(C, method)
        assert bool((upper - lower > 1e-6).all()), method


@pytest.mark.parametrize("seed", [5, 6])
def test_the_reduced_set_optimum_is_exact_and_bounds_auto_lirpa(seed):
    """`reduced_optimum` equals the minimum over every vertex of the reduced problem's boxes (a bilinear form attains its
    extremes at vertices), and auto_LiRPA's bounds on the reduced graph never exceed it."""
    gen = torch.Generator().manual_seed(seed)
    experts, hidden, intermediate = 2, 3, 2
    down_c = torch.randn(experts, hidden, intermediate, generator=gen)
    down_r = torch.rand(experts, hidden, 1, generator=gen).expand_as(down_c) * 0.3
    a_lo = torch.randn(experts, intermediate, generator=gen)
    a_hi = a_lo + torch.rand(experts, intermediate, generator=gen)
    routing, base = torch.tensor([0.7, 0.2]), torch.randn(hidden, generator=gen)
    delta = torch.randn(4, hidden, generator=gen)
    optimum = reduced_optimum(delta, down_c - down_r, down_c + down_r, a_lo, a_hi, routing, base)
    count = a_lo.numel() + down_c.numel()
    bits = torch.tensor(list(itertools.product((0.0, 1.0), repeat=count)))
    a = a_lo.reshape(-1) + bits[:, : a_lo.numel()] * (a_hi - a_lo).reshape(-1)
    d = (down_c - down_r).reshape(-1) + bits[:, a_lo.numel() :] * (2 * down_r).reshape(-1)
    a, d = a.reshape(-1, experts, intermediate), d.reshape(-1, experts, hidden, intermediate)
    y = base + torch.einsum("e,neh->nh", routing, torch.einsum("nehi,nei->neh", d, a))
    assert torch.allclose(optimum, (y @ delta.T).min(dim=0).values, rtol=1e-12, atol=1e-12)
    downs = [boxed_parameter(down_c[e], down_c[e] - down_r[e], down_c[e] + down_r[e]) for e in range(experts)]
    inputs = tuple(boxed_input(a_lo[e : e + 1], a_hi[e : e + 1]) for e in range(experts))
    bounder = Bounder(ReducedSuffix(downs, routing.tolist(), base), inputs, CPU, alpha_iterations=30)
    for method in METHODS:
        assert bool((bounder.lower(delta, method, chunk=4) <= optimum + 1e-12).all()), method


def test_the_experts_bound_separately_as_together():
    """Δ·y = Δ·b + Σ_e Δ·(w_e·D_e·a_e) with no variable shared between experts: auto_LiRPA's CROWN and CROWN-IBP bounds of
    the six-expert graph equal Δ·b plus the experts' own bounds (run.py's StateBounds), and α-CROWN's per-expert sum is
    still below the true minimum found by search."""
    truth, boxes, x, base, routing, C = problem(9, experts=3)
    together = Bounder(full_model(boxes, routing, base), (x,), CPU, alpha_iterations=30)
    zero = torch.zeros_like(base)
    apart = [Bounder(full_model({n: (lo[e : e + 1], hi[e : e + 1], c[e : e + 1]) for n, (lo, hi, c) in boxes.items()}, [routing[e]], zero), (x,), CPU,
                     alpha_iterations=30) for e in range(3)]
    for method in ("crown-ibp", "crown"):
        joint = together.lower(C, method, chunk=3)
        split = C @ base + sum(b.lower(C, method, chunk=3) for b in apart)
        assert torch.allclose(joint, split, rtol=1e-12, atol=1e-12), method
    names = ("gate", "up", "down")
    for j in range(C.shape[0]):
        def objective(g, u, d, j=j):
            return suffix_value(x, g, u, d, routing, base) @ C[j]
        found, _ = pgd_minimize(objective, [boxes[n][0] for n in names], [boxes[n][1] for n in names], [[boxes[n][2] for n in names]], steps=150)
        split = float(C[j] @ base) + sum(float(b.lower(C[j : j + 1], "alpha-crown", chunk=1)[0]) for b in apart)
        assert split <= found + 1e-12


def test_silu_is_the_composition():
    g = torch.linspace(-30, 30, 10001)
    assert torch.allclose(silu(g), torch.nn.functional.silu(g), rtol=1e-15, atol=1e-300)
