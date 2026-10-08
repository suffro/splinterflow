"""Row-product weight sets and the exact optima of the experts' decision over them (Phase 5A2, stage 1.5).

Diagnostics, never a certificate: what a perfect verifier would return for a set, and weights inside a set that a
runtime could not tell from the true ones (witnesses). No bound propagation: closed forms of one row at a time.

Every set is a product over rows: each row W_r of a routed expert's matrix lies in its own convex set S_r, independent of
every other row (the metadata is per row). `RowSets` holds one matrix's: the L2 ball of the row's current remainder
(about its current approximation c_r), optionally intersected with an elementwise box, and further L2 balls (the row's
own norm, earlier levels) that a witness must also satisfy. A set enters the decision only through support functions,

    h_r(v) = max_{W ∈ S_r} W·v = c_r·v + δ*_r(v)·v,   δ*_r(v) = argmax over the row's deviations

    L2 ball            δ* = ρ·v/‖v‖₂
    L2 ball ∩ box      δ* = clip(τ·v, l − c, u − c), τ ≥ 0 the largest with ‖δ*‖₂ ≤ ρ (KKT; the norm grows with τ)

The exact optimum (decision 0010's reduction). With x exact, g_i = G_i·x ranges over the whole interval
[−h_i(−x), h_i(x)] (one row, a convex set) and u_i likewise, independently of every other neuron; so a_i = silu(g_i)·u_i
ranges over an interval (the extremes of s·u over [s⁻, s⁺] × [u⁻, u⁺], s the range of silu on g's interval) and the
activations' set is exactly a box. For a fixed a each down row contributes min_{D_k ∈ S_k} Δ_k·D_k·a by itself, a concave
function of a, so the minimum of Δ·y over the set is attained at a vertex of the activation box, expert by expert:

    opt = Δ·b + Σ_e w_e · min over vertices a of φ_e(a),     φ_e(a) = Σ_k min_{D_k ∈ S_k} Δ_k·D_k·a

For L2 balls φ_e(a) = M_e·a − c_e·‖a‖₂ with M_e = C_eᵀΔ and c_e = Σ_k |Δ_k|·ρ_k. `l2_decoupled` minimizes the two terms
separately (a lower bound), `l2_vertex_search` reaches a vertex by monotone local search (its value is attained: an upper
bound), `vertex_enumeration` takes every vertex of a small box (exact). `Witness` builds the weights themselves.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import torch

SILU_ARGMIN = -1.2784645427610738  # x·σ(x) decreases on (−∞, x*] and increases after


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


SILU_MINIMUM = float(silu(torch.tensor(SILU_ARGMIN, dtype=torch.float64)))


@dataclass(frozen=True)
class RowSets:
    """One matrix's rows [R, C]: each in the L2 ball of `radius` about `centre`, ∩ [lower, upper] when a box is given;
    `extra` are further balls ((centre [R, C], radius [R]), ...) of the same rows. Radius 0: the row is exact."""

    centre: torch.Tensor
    radius: torch.Tensor
    lower: torch.Tensor | None = None
    upper: torch.Tensor | None = None
    extra: tuple[tuple[torch.Tensor, torch.Tensor], ...] = field(default=())

    @property
    def boxed(self) -> bool:
        return self.lower is not None

    def extreme(self, v: torch.Tensor, iterations: int = 80) -> torch.Tensor:
        """δ*(v) per row [R, C] (v [C] shared or [R, C] per row): the deviation from the centre maximizing δ·v."""
        v = v.expand_as(self.centre) if v.dim() == 1 else v
        radius = self.radius[:, None]
        norm = torch.linalg.vector_norm(v, dim=1, keepdim=True)
        if not self.boxed:
            return torch.where(norm > 0, radius * v / torch.where(norm > 0, norm, 1.0), 0.0)
        lo, hi = self.lower - self.centre, self.upper - self.centre
        floor = torch.clamp(torch.zeros_like(v), lo, hi)  # the box's point nearest the centre (τ = 0)
        corner = torch.where(v > 0, hi, torch.where(v < 0, lo, floor))
        fits = torch.linalg.vector_norm(corner, dim=1, keepdim=True) <= radius
        magnitude = v.abs()
        reach = torch.where(magnitude > 0, torch.maximum(lo.abs(), hi.abs()) / torch.where(magnitude > 0, magnitude, 1.0), 0.0)
        low = torch.zeros_like(radius)
        high = reach.amax(dim=1, keepdim=True)  # at τ = high every coordinate with v ≠ 0 is at its end: the corner
        for _ in range(iterations):
            middle = (low + high) * 0.5
            inside = torch.linalg.vector_norm(torch.clamp(middle * v, lo, hi), dim=1, keepdim=True) <= radius
            low = torch.where(inside, middle, low)
            high = torch.where(inside, high, middle)
        delta = torch.clamp(low * v, lo, hi)
        return torch.where(fits, corner, delta)

    def support(self, v: torch.Tensor) -> torch.Tensor:
        """h_r(v) = max over the row's set of W_r·v, [R]."""
        v = v.expand_as(self.centre) if v.dim() == 1 else v
        return (self.centre * v).sum(dim=1) + (self.extreme(v) * v).sum(dim=1)

    def excess(self, weights: torch.Tensor) -> torch.Tensor:
        """The largest constraint excess per row [R] (≤ 0: the row is in the set), each relative to its scale."""
        out = torch.linalg.vector_norm(weights - self.centre, dim=1) - self.radius
        if self.boxed:
            out = torch.maximum(out, (self.lower - weights).amax(dim=1))
            out = torch.maximum(out, (weights - self.upper).amax(dim=1))
        for centre, radius in self.extra:
            out = torch.maximum(out, torch.linalg.vector_norm(weights - centre, dim=1) - radius)
        return out

    def point(self, v: torch.Tensor, anchor: torch.Tensor, rounds: int = 3) -> tuple[torch.Tensor, int]:
        """Rows of the set (every constraint) with a large W·v: the current ball's (∩ box) maximizer; a row that leaves a
        further ball moves to the maximizer over the current ball ∩ that ball (`two_ball_max`), then into the box; a row
        still outside is pulled toward `anchor`. Radii are taken 2⁻⁴⁰ inside. Returns the rows and how many were pulled."""
        v = v.expand_as(self.centre) if v.dim() == 1 else v
        shrunk = RowSets(self.centre, self.radius * (1.0 - 2.0**-40), self.lower, self.upper, self.extra)
        rows = self.centre + shrunk.extreme(v)
        finite = [(c, r) for c, r in self.extra if bool(torch.isfinite(r).any())]
        for _ in range(rounds if finite else 0):
            excess = torch.stack([torch.where(torch.isfinite(r), torch.linalg.vector_norm(rows - c, dim=1) - r, -math.inf) for c, r in finite])
            worst, outside = excess.argmax(dim=0), excess.amax(dim=0) > 0
            if not bool(outside.any()):
                break
            centres = torch.stack([c for c, _ in finite]).gather(0, worst.view(1, -1, 1).expand(1, *rows.shape)).squeeze(0)
            radii = torch.stack([r for _, r in finite]).gather(0, worst.view(1, -1)).squeeze(0)
            paired = two_ball_max(self.centre, self.radius * (1.0 - 2.0**-40), centres, radii * (1.0 - 2.0**-40), v)
            if self.boxed:
                paired = torch.clamp(paired, self.lower, self.upper)
            rows = torch.where(outside.unsqueeze(-1), paired, rows)
        rows, lam = self.pull_back(rows, anchor)
        return rows, int((lam < 1).sum())

    def anchor(self, truth: torch.Tensor) -> torch.Tensor:
        """A point of the set per row for `point`'s last resort: halfway between the true row and the centre where that lies
        in the set (strictly inside the current ball), else the true row (a witness only has to lie in the set)."""
        midway = truth * 0.5 + self.centre * 0.5
        return torch.where((self.excess(midway) <= 0).unsqueeze(-1), midway, truth)

    def pull_back(self, weights: torch.Tensor, anchor: torch.Tensor, iterations: int = 60) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows outside the set moved toward `anchor` (a point of the set, row by row) to the last point inside: the
        largest λ ∈ [0, 1] with anchor + λ·(weights − anchor) in the set (it is convex). Returns the rows and λ."""
        outside = self.excess(weights) > 0
        lam = torch.ones(weights.shape[0], dtype=weights.dtype, device=weights.device)
        if not bool(outside.any()):
            return weights, lam
        if bool((self.excess(anchor) > 0).any()):
            raise ValueError("the anchor must lie in the set")
        low, high = torch.zeros_like(lam), torch.ones_like(lam)
        for _ in range(iterations):
            middle = (low + high) * 0.5
            inside = self.excess(anchor + middle[:, None] * (weights - anchor)) <= 0
            low, high = torch.where(inside, middle, low), torch.where(inside, high, middle)
        lam = torch.where(outside, low, lam)
        return anchor + lam[:, None] * (weights - anchor), lam


def two_ball_max(c1: torch.Tensor, r1: torch.Tensor, c2: torch.Tensor, r2: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """argmax W·v over {‖W − c1‖₂ ≤ r1} ∩ {‖W − c2‖₂ ≤ r2} per row (a non-empty intersection): c1 + r1·v̂ if it lies in the
    second ball, c2 + r2·v̂ if it lies in the first, else the best point of the spheres' intersection, the sphere about
    c1 + s·ê (ê = (c2 − c1)/D, s = (r1² − r2² + D²)/2D) of radius √(r1² − s²) in the hyperplane ⟂ ê, at v's direction
    there. c [R, C], r [R], v [R, C] or [C]."""
    v = v.expand_as(c1)
    unit = lambda t: t / torch.linalg.vector_norm(t, dim=1, keepdim=True).clamp_min(1e-300)  # noqa: E731
    v_hat = unit(v)
    first, second = c1 + r1[:, None] * v_hat, c2 + r2[:, None] * v_hat
    first_ok = torch.linalg.vector_norm(first - c2, dim=1) <= r2
    second_ok = torch.linalg.vector_norm(second - c1, dim=1) <= r1
    d = c2 - c1
    D = torch.linalg.vector_norm(d, dim=1).clamp_min(1e-300)
    e = d / D[:, None]
    s = (r1 * r1 - r2 * r2 + D * D) / (2.0 * D)
    across = torch.sqrt((r1 * r1 - s * s).clamp_min(0.0))
    across_direction = unit(v - (v * e).sum(dim=1, keepdim=True) * e)
    both = c1 + s[:, None] * e + across[:, None] * across_direction
    return torch.where(first_ok.unsqueeze(-1), first, torch.where(second_ok.unsqueeze(-1), second, both))


# The activations' set


@dataclass(frozen=True)
class ActivationRange:
    """The exact range of each neuron's g, u and a = silu(g)·u over the gate and up rows' sets, the rows reaching g's and
    u's ends (`points[name] = (rows at the lower end, rows at the upper end)`), and which ends realize a's extremes:
    g_for[side] (the g value) and u_end[side] (0 for u⁻, 1 for u⁺), side 0 for a⁻ and 1 for a⁺."""

    g: tuple[torch.Tensor, torch.Tensor]
    u: tuple[torch.Tensor, torch.Tensor]
    a: tuple[torch.Tensor, torch.Tensor]
    g_for: tuple[torch.Tensor, torch.Tensor]
    u_end: tuple[torch.Tensor, torch.Tensor]
    points: dict


def linear_range(rows: RowSets, x: torch.Tensor, anchor: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
    """The range of W_r·x over each row's set (x [C], exact) and rows reaching its ends. Without `anchor`: the current ball's
    (∩ box) range [−h(−x), h(x)], exact for it and a superset for the whole set (lower bounds use it); with `anchor`: the
    values at rows of the whole set (`RowSets.point`), attained (witnesses use them)."""
    if anchor is None:
        low_rows, high_rows = rows.centre + rows.extreme(-x), rows.centre + rows.extreme(x)
    else:
        (low_rows, _), (high_rows, _) = rows.point(-x, anchor), rows.point(x, anchor)
    return low_rows @ x, high_rows @ x, low_rows, high_rows


def activation_range(gate: RowSets, up: RowSets, x: torch.Tensor, anchors: dict | None = None) -> ActivationRange:
    """The activations' box over the gate and up rows' sets (`linear_range`'s two senses: without `anchors` the current
    balls' exact box; with them, the box between attained ends)."""
    g_lo, g_hi, g_low_rows, g_high_rows = linear_range(gate, x, None if anchors is None else anchors["gate"])
    u_lo, u_hi, u_low_rows, u_high_rows = linear_range(up, x, None if anchors is None else anchors["up"])
    at_lo, at_hi = silu(g_lo), silu(g_hi)
    straddles = (g_lo <= SILU_ARGMIN) & (g_hi >= SILU_ARGMIN)
    s_lo = torch.where(straddles, torch.full_like(at_lo, SILU_MINIMUM), torch.minimum(at_lo, at_hi))
    g_at_s_lo = torch.where(straddles, torch.full_like(g_lo, SILU_ARGMIN), torch.where(at_lo <= at_hi, g_lo, g_hi))
    s_hi = torch.maximum(at_lo, at_hi)
    g_at_s_hi = torch.where(at_hi >= at_lo, g_hi, g_lo)
    # The four corners of [s⁻, s⁺] × [u⁻, u⁺]: (s end, u end).
    corners = torch.stack([s_lo * u_lo, s_lo * u_hi, s_hi * u_lo, s_hi * u_hi])
    g_options = torch.stack([g_at_s_lo, g_at_s_lo, g_at_s_hi, g_at_s_hi])
    u_options = torch.tensor([0, 1, 0, 1], device=x.device).view(4, *([1] * g_lo.dim())).expand_as(corners)
    low, high = corners.argmin(dim=0, keepdim=True), corners.argmax(dim=0, keepdim=True)
    pick = lambda t, i: t.gather(0, i).squeeze(0)  # noqa: E731
    return ActivationRange((g_lo, g_hi), (u_lo, u_hi), (pick(corners, low), pick(corners, high)),
                           (pick(g_options, low), pick(g_options, high)), (pick(u_options, low), pick(u_options, high)),
                           {"gate": (g_low_rows, g_high_rows), "up": (u_low_rows, u_high_rows)})


# The reduced problem over L2 balls: φ(a) = M·a − c·‖a‖₂ on the vertices of [a⁻, a⁺]


def l2_decoupled(M: torch.Tensor, c: torch.Tensor, a_lo: torch.Tensor, a_hi: torch.Tensor) -> torch.Tensor:
    """Σ_i min(M_i·a⁻_i, M_i·a⁺_i) − c·‖max(|a⁻|, |a⁺|)‖₂: each term minimized by itself (≤ the optimum). M [J, I], c [J]."""
    linear = torch.minimum(M * a_lo, M * a_hi).sum(dim=-1)
    return linear - c * torch.linalg.vector_norm(torch.maximum(a_lo.abs(), a_hi.abs()), dim=-1)


def l2_value(M: torch.Tensor, c: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return (M * a).sum(dim=-1) - c * torch.linalg.vector_norm(a, dim=-1)


def l2_vertex_search(M: torch.Tensor, c: torch.Tensor, a_lo: torch.Tensor, a_hi: torch.Tensor, iterations: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """A vertex of the box with a small φ, per contender: from the linear minimizer and from the largest-norm vertex,
    repeat a ← argmin over vertices of (M − c·a/‖a‖)·a (φ never increases: ‖a'‖ ≥ a'·a/‖a‖). Returns φ at the best
    vertex found [J] (attained, so ≥ the optimum) and the vertex [J, I]."""
    starts = [torch.where(M >= 0, a_lo, a_hi).expand_as(M),
              torch.where(a_hi.abs() >= a_lo.abs(), a_hi, a_lo).expand_as(M).clone()]
    best_value, best_vertex = None, None
    for a in starts:
        a = a.clone()
        for _ in range(iterations):
            norm = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
            q = M - c[:, None] * torch.where(norm > 0, a / torch.where(norm > 0, norm, 1.0), 0.0)
            nxt = torch.where(q * a_lo <= q * a_hi, a_lo.expand_as(q), a_hi.expand_as(q))
            if torch.equal(nxt, a):
                break
            a = nxt
        value = l2_value(M, c, a)
        if best_value is None:
            best_value, best_vertex = value, a
        else:
            better = value < best_value
            best_value = torch.where(better, value, best_value)
            best_vertex = torch.where(better[:, None], a, best_vertex)
    return best_value, best_vertex


def vertex_enumeration(phi, a_lo: torch.Tensor, a_hi: torch.Tensor) -> torch.Tensor:
    """min over every vertex of a small box [a⁻, a⁺] ([I]) of phi(vertices [V, I]) → [V, J]: exact, [J]."""
    count = a_lo.numel()
    if count > 20:
        raise ValueError("too many vertices to enumerate")
    bits = torch.tensor(list(itertools.product((0.0, 1.0), repeat=count)), dtype=a_lo.dtype, device=a_lo.device)
    return phi(a_lo + bits * (a_hi - a_lo)).min(dim=0).values


def down_value(down: RowSets, delta: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """φ(a) = Σ_k min_{D_k ∈ S_k} Δ_k·D_k·a for each contender (delta [J, K], a [J, I] or [I]): the down rows at their
    extremes (row k minimizes Δ_k·D_k·a by itself: the maximizer of D_k·(−sign(Δ_k)·a))."""
    a = a.expand(delta.shape[0], -1) if a.dim() == 1 else a
    out = []
    for j in range(delta.shape[0]):
        direction = -torch.sign(delta[j])[:, None] * a[j][None, :]
        rows = down.centre + down.extreme(direction)
        out.append(delta[j] @ (rows @ a[j]))
    return torch.stack(out)


# Witnesses: weights in the set


@dataclass
class Witness:
    """Weights of one expert inside its sets (gate, up, down rows) and Δ's value of the expert's term with them."""

    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    a: torch.Tensor
    pulled_rows: int
    largest_excess: float


def realize(gate: RowSets, up: RowSets, down: RowSets, x: torch.Tensor, delta_j: torch.Tensor, sides: torch.Tensor,
            ranges: ActivationRange, anchors: dict[str, torch.Tensor]) -> Witness:
    """The weights realizing activation vertex `sides` ([I]: 0 for a⁻, 1 for a⁺) and the down rows at their extremes for
    it (`RowSets.point`), for one contender Δ_j. Each gate row moves along the segment between its rows at g's two ends
    (g is linear on it; the set is convex), each up row is one of its ends. `ranges` should come from
    `activation_range(..., anchors)` (rows of the whole sets); rows still outside are pulled toward `anchors` (points of
    the sets: the oracle passes `RowSets.anchor` of the true weights, as a witness only has to lie in the set)."""
    v = x.reshape(-1)
    pulled = 0
    minus_g, plus_g = ranges.points["gate"]
    g_lo, g_hi = ranges.g
    target = torch.where(sides.bool(), ranges.g_for[1], ranges.g_for[0])
    width = g_hi - g_lo
    lam = torch.where(width > 0, (target - g_lo) / torch.where(width > 0, width, 1.0), 0.0).clamp(0.0, 1.0)
    G = minus_g + lam[:, None] * (plus_g - minus_g)
    u_end = torch.where(sides.bool(), ranges.u_end[1], ranges.u_end[0])
    minus_u, plus_u = ranges.points["up"]
    U = torch.where(u_end.bool()[:, None], plus_u, minus_u)
    G, lg = gate.pull_back(G, anchors["gate"])
    U, lu = up.pull_back(U, anchors["up"])
    pulled += int((lg < 1).sum()) + int((lu < 1).sum())
    a = silu(G @ v) * (U @ v)
    direction = -torch.sign(delta_j)[:, None] * a[None, :]
    D, count = down.point(direction, anchors["down"])
    pulled += count
    excess = max(float(gate.excess(G).max()), float(up.excess(U).max()), float(down.excess(D).max()))
    return Witness(G, U, D, a, pulled, excess)
