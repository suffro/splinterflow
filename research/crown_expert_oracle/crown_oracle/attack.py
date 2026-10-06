"""Adversarial realizations inside a weight set (Phase 5A2; validation and diagnostics only, never a certificate).

A realization is a point of the set: weights that agree with everything a runtime has read. Its property value is
achievable, so no sound bound over the set can exceed it:

    any sound lower bound  ≤  min over the set  ≤  the property at any realization.

`pgd_minimize` searches the box by projected sign-gradient descent (plain autograd on the forward graph). Every
realization found is also a soundness test of the bounds: an auto_LiRPA lower bound above one would be a violation.

`reduced_optimum` is the exact minimum of the reduced problem, min over a in its box and D in its box of
Σ_e w_e·Δ·D_e·a_e: for fixed a it is linear in each D entry, so D sits at an end of its interval, and what is left is a
concave piecewise-linear function of each a_i, minimized at an end of a_i's interval. It is the value a perfect verifier
would return for that set (a diagnostic of how far auto_LiRPA's relaxation is from it), not a bound used to certify.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


def pgd_minimize(objective: Callable[..., torch.Tensor], lower: list[torch.Tensor], upper: list[torch.Tensor],
                 starts: list[list[torch.Tensor]], steps: int = 100) -> tuple[float, list[torch.Tensor]]:
    """Projected sign-gradient descent of a scalar `objective(*weights)` over the box, from each start; the smallest value
    seen and its point. Steps shrink linearly from a quarter of each box's width."""
    best, best_point = math.inf, None
    widths = [(u - l) for l, u in zip(lower, upper)]
    for start in starts:
        point = [p.detach().clone().clamp_(l, u).requires_grad_(True) for p, l, u in zip(start, lower, upper)]
        for step in range(steps + 1):
            value = objective(*point)
            if float(value.detach()) < best:
                best, best_point = float(value.detach()), [p.detach().clone() for p in point]
            if step == steps:
                break
            grads = torch.autograd.grad(value, point)
            scale = 0.25 * (1.0 - step / steps) + 1e-3
            with torch.no_grad():
                for p, g, width, l, u in zip(point, grads, widths, lower, upper):
                    p.sub_(scale * width * torch.sign(g))
                    torch.maximum(torch.minimum(p, u), l, out=p)
    return best, best_point


def vertex_starts(lower: list[torch.Tensor], upper: list[torch.Tensor], gradient_signs: list[torch.Tensor]) -> list[torch.Tensor]:
    """The box vertex a first-order adversary picks: each entry at the end its gradient points away from."""
    return [torch.where(sign > 0, l, u) for l, u, sign in zip(lower, upper, gradient_signs)]


def reduced_optimum(delta: torch.Tensor, down_lower: torch.Tensor, down_upper: torch.Tensor, a_lower: torch.Tensor,
                    a_upper: torch.Tensor, routing: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """min over a ∈ [a_lower, a_upper], D ∈ [down_lower, down_upper] of Δ·(b + Σ_e w_e·D_e·a_e), per contender.

    delta [J, H]; down_* [K, H, I]; a_* [K, I]; routing [K]; base [H]. For a fixed a, the minimum over D of Δ_k·D_ki·a_i is
    a_i·min(Δ_k·D_lo, Δ_k·D_hi) for a_i ≥ 0 and a_i·max(·) for a_i < 0; summed over k this is f_i(a_i) = a_i·P_i for
    a_i ≥ 0 and a_i·Q_i for a_i < 0 (P ≤ Q), concave, so its minimum over [a_lo, a_hi] is at an end. Float64, no
    outward rounding: a diagnostic.
    """
    total = delta @ base
    positive, negative = delta.clamp_min(0.0), delta.clamp_max(0.0)
    for e in range(down_lower.shape[0]):
        P = positive @ down_lower[e] + negative @ down_upper[e]  # [J, I]: Σ_k min(Δ_k·D_lo,ki, Δ_k·D_hi,ki)
        Q = positive @ down_upper[e] + negative @ down_lower[e]  # Σ_k max(·, ·)
        at_lower = torch.where(a_lower[e] >= 0, a_lower[e] * P, a_lower[e] * Q)
        at_upper = torch.where(a_upper[e] >= 0, a_upper[e] * P, a_upper[e] * Q)
        total = total + routing[e] * torch.minimum(at_lower, at_upper).sum(dim=1)
    return total
