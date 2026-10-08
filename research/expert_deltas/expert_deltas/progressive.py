"""Progressive bit-plane materialization of routed experts, and the exact optimum of the decision over what is read
(Phase 5C, stage 5C2). A structural diagnostic in real arithmetic: never a certified BF16 result.

Representation. Every row of a routed expert's gate, up and down matrices is stored as bit planes (`bits`: sign,
exponent, mantissa bits 6..0), in pages of 16 rows (a neuron page: 16 neurons' gate and up rows; a down page: 16 output
rows), each plane of a page its own zstd frame. A page is read step by step: step 1 its sign and exponent planes (prefix
t = 9), steps 2..8 one mantissa plane each (t = 10..16). Reading all of a page restores its rows exactly; the planes of
an XOR delta against any resident base give the same prefixes (`bits`), so the sets below hold for every such base.

Knowledge. A weight with prefix t lies in `bits.prefix_interval(pattern, t)`; an unread row (t = 0) lies in the box of
its resident L∞ bound when that metadata is held ([−n, n], n ≥ max|w|), else anywhere finite. The set of weights
consistent with every byte read and every resident byte is the product of these intervals: a box, independent across
elements. `read_view` builds it from the patterns with every unread bit replaced (poisoned) — the set never depends on
an unread bit (tested).

The decision. In real arithmetic the token is argmax W·(g⊙y)·q with q > 0 common to every logit, so w beats j iff
Δ·y > 0, Δ = (W_w − W_j)⊙g, y = r + S + Σ_e w_e·D_e·a_e, a_e = silu(G_e·x)⊙(U_e·x) (decision 0009's real tier). Over a
box set the minimum of Δ·y is separable and attained (`exact_minimum`):

    g_i, u_i range over [c·x − h·|x|, c·x + h·|x|] (each row its own box, x exact); a_i = silu(g_i)·u_i over the
    extremes of s·u on [s⁻, s⁺] × [u⁻, u⁺] (s = silu over g's interval; silu's minimum at g* when inside); the neurons
    are independent, so the activations' set is exactly a box. For a fixed a, each down element D_ki takes the end of
    its interval minimizing Δ_k·D_ki·a_i:  min_D Δ·D·a = M·a − H·|a|, M = C_Dᵀ·Δ, H = |Δ|ᵀ·h_D (centre and half-width
    of D's box). M_i·a_i − H_i·|a_i| is concave in a_i, so its minimum is at an end of a_i's interval, neuron by neuron.

    min over the set of Δ·y = Δ·(r + S) + Σ_e w_e·Σ_i min_{a_i ∈ {a_i⁻, a_i⁺}} (M_e,i·a_i − H_e,i·|a_i|)

Nothing in this computation is a relaxation: a perfect verifier given this information returns exactly this value. A
negative minimum comes with its minimizer (`witness`): weights inside the set whose real forward gives Δ·y < 0, a proof
that no sound verifier can certify the pair from this information. Float64 evaluation errs by far less than the slack
used (`SLACK`, relative to the magnitudes involved).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from expert_deltas import bits

SILU_ARGMIN = -1.2784645427610738  # x·σ(x) decreases on (−∞, g*] and increases after
# Relative float64 slack on a decision: the minimum's computation sums at most ~2·(H + I)·K terms per pair (≈ 4·10⁴), so
# its rounding stays below 4·γ_n ≈ 2⁻³⁴·… of the magnitudes involved; 2⁻³² of them is a generous margin.
SLACK = 2.0**-32
PAGE_ROWS = 16
STEP_PREFIX = (0, 9, 10, 11, 12, 13, 14, 15, 16)  # a page's prefix after 0..8 steps


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


SILU_MINIMUM = float(silu(torch.tensor(SILU_ARGMIN, dtype=torch.float64)))


# Sets from what is read


@dataclass(frozen=True)
class Box:
    """Per-element intervals of one matrix [R, C] (float64): centre and half-width."""

    centre: torch.Tensor
    half: torch.Tensor

    @property
    def lower(self) -> torch.Tensor:
        return self.centre - self.half

    @property
    def upper(self) -> torch.Tensor:
        return self.centre + self.half

    def contains(self, values: torch.Tensor, relative: float = 0.0) -> bool:
        slack = relative * (self.centre.abs() + self.half)
        return bool(((values >= self.lower - slack) & (values <= self.upper + slack)).all())


def read_view(patterns: torch.Tensor, prefix: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    """The patterns a runtime holds: each row's top `prefix[r]` bits, every other bit replaced by random bits (seed) or
    zeros. Building a set from this view instead of the true patterns must change nothing (`box`)."""
    p = patterns.to(torch.int64)
    keep = ((torch.ones_like(prefix, dtype=torch.int64) << (16 - prefix.to(torch.int64))) - 1)[:, None]  # unknown bits
    if seed is None:
        noise = torch.zeros_like(p)
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randint(0, 1 << 16, p.shape, generator=generator, dtype=torch.int64).to(p.device)
    return (p & ~keep & 0xFFFF) | (noise & keep)


def box(view: torch.Tensor, prefix: torch.Tensor, linf: torch.Tensor | None = None) -> Box:
    """The set of one matrix (patterns [R, C] as a read view, prefix [R] per row; optional resident L∞ bounds [R])."""
    lo, hi = bits.prefix_interval(view, prefix.to(torch.int64)[:, None].expand_as(view))
    if linf is not None:
        n = linf.to(torch.float64)[:, None]
        lo, hi = torch.maximum(lo, -n), torch.minimum(hi, n)
    return Box((lo + hi) * 0.5, (hi - lo) * 0.5)


def linf_bounds(patterns: torch.Tensor) -> torch.Tensor:
    """Resident metadata: each row's max |w|, rounded up to float32 (4 bytes per row)."""
    v = bits.pattern_values(patterns.to(torch.int64)).abs().amax(dim=1)
    single = v.to(torch.float32)
    up = torch.nextafter(single, torch.full_like(single, math.inf)).to(torch.float64)
    return torch.where(single.to(torch.float64) < v, up, single.to(torch.float64))


# The activations' exact range


@dataclass(frozen=True)
class Activations:
    g: tuple[torch.Tensor, torch.Tensor]
    u: tuple[torch.Tensor, torch.Tensor]
    a: tuple[torch.Tensor, torch.Tensor]
    g_for: tuple[torch.Tensor, torch.Tensor]  # the g value realizing a⁻ and a⁺
    u_high: tuple[torch.Tensor, torch.Tensor]  # whether a⁻ (a⁺) takes u's upper end


def activation_range(gate: Box, up: Box, x: torch.Tensor) -> Activations:
    """The exact range of each neuron's g, u and a = silu(g)·u over the gate and up boxes (x exact, float64 [H])."""
    ax = x.abs()
    g_c, g_h = gate.centre @ x, gate.half @ ax
    u_c, u_h = up.centre @ x, up.half @ ax
    return _activations_from((g_c - g_h, g_c + g_h), (u_c - u_h, u_c + u_h))


# The exact minimum of the decision


@dataclass(frozen=True)
class ExpertSet:
    gate: Box
    up: Box
    down: Box
    weight: float  # the routing weight (float32 value, exact in float64)


@dataclass(frozen=True)
class Minimum:
    value: torch.Tensor  # [J]: min over the set of Δ_j·y
    magnitude: torch.Tensor  # [J]: Σ of the absolute values involved (for the float64 slack)
    sides: list[torch.Tensor]  # per expert [J, I] bool: the minimizer's a_i end (True: a⁺)
    activations: list[Activations]

    @property
    def decided(self) -> torch.Tensor:
        """Pairs decided in real arithmetic (minimum above the float64 slack)."""
        return self.value > SLACK * self.magnitude


def exact_minimum(experts: list[ExpertSet], x: torch.Tensor, base: torch.Tensor, delta: torch.Tensor, chunk: int = 4096) -> Minimum:
    """min over the set of Δ_j·y for contenders `delta` [J, H] (float64), y = base + Σ_e w_e·D_e·a_e (module docstring)."""
    value = delta @ base
    magnitude = delta.abs() @ base.abs()
    sides, ranges = [], []
    for expert in experts:
        act = activation_range(expert.gate, expert.up, x)
        ranges.append(act)
        a_lo, a_hi = act.a
        part_value, part_mag, part_side = [], [], []
        for start in range(0, delta.shape[0], chunk):
            d = delta[start : start + chunk]
            M = d @ expert.down.centre  # [J, I]
            Hm = d.abs() @ expert.down.half  # [J, I]
            low = M * a_lo - Hm * a_lo.abs()
            high = M * a_hi - Hm * a_hi.abs()
            take_high = high < low
            part_value.append(torch.where(take_high, high, low).sum(dim=1))
            reach = torch.maximum(a_lo.abs(), a_hi.abs())
            part_mag.append((M.abs() + Hm) @ reach + d.abs() @ (expert.down.centre.abs() + expert.down.half) @ reach)
            part_side.append(take_high)
        value = value + expert.weight * torch.cat(part_value)
        magnitude = magnitude + expert.weight * torch.cat(part_mag)
        sides.append(torch.cat(part_side))
    return Minimum(value, magnitude, sides, ranges)


# Structured metadata: sketches through shared bases (a sound relaxation)

ROUNDING_32 = 2.0**-23  # a float64 value rounded to float32 errs by at most 2⁻²⁴ of its magnitude; doubled for safety


@dataclass(frozen=True)
class Sketch:
    """Resident metadata of the routed experts: a layer's bases U [H, k] (the experts' input) and V [H, k'] (decision
    directions), and per expert G·U, Up·U [I, k] and Vᵀ·D [k', I], all stored in float32 (here their float64 values).

    With x = U·c + x⊥ (c = Uᵀx, x⊥ = x − U·c, exact for any c), g = (G·U)·c + G·x⊥: the first term is known (up to the
    sketch's rounding), the second ranges over the box with |x⊥| instead of |x|. With Δ = V·d + Δ⊥ (d = VᵀΔ),
    Δᵀ·D·a = dᵀ·(Vᵀ·D)·a + Δ⊥ᵀ·D·a: the first known, the second over the box. Dropping the sketch's equality constraints
    on the rest gives a superset of the set: the minimum below is a lower bound of the set's (not attained)."""

    U: torch.Tensor
    V: torch.Tensor
    gate: list[torch.Tensor]
    up: list[torch.Tensor]
    down: list[torch.Tensor]
    nbytes: int  # the per-expert sketches (the bases are per layer, counted separately)

    @classmethod
    def build(cls, U: torch.Tensor, V: torch.Tensor, truths: list[dict[str, torch.Tensor]]) -> Sketch:
        U = U.to(torch.float32).to(torch.float64)
        V = V.to(torch.float32).to(torch.float64)

        def stored(t):
            return t.to(torch.float32).to(torch.float64)

        gate = [stored(t["gate"] @ U) for t in truths]
        up = [stored(t["up"] @ U) for t in truths]
        down = [stored(V.t() @ t["down"]) for t in truths]
        nbytes = sum(4 * (g.numel() + u.numel() + d.numel()) for g, u, d in zip(gate, up, down))
        return cls(U, V, gate, up, down, nbytes)


def sketched_minimum(experts: list[ExpertSet], x: torch.Tensor, base: torch.Tensor, delta: torch.Tensor, sketch: Sketch,
                     chunk: int = 4096) -> Minimum:
    """A lower bound of min Δ_j·y over the set intersected with the sketch's constraints (`Sketch`). No witness."""
    c = sketch.U.t() @ x
    rest = x - sketch.U @ c
    value = delta @ base
    magnitude = delta.abs() @ base.abs()
    sides, ranges = [], []
    for e, expert in enumerate(experts):
        known_g, known_u = sketch.gate[e] @ c, sketch.up[e] @ c
        error_g = ROUNDING_32 * (sketch.gate[e].abs() @ c.abs())
        error_u = ROUNDING_32 * (sketch.up[e].abs() @ c.abs())
        # The ranges of g and u: known part ± its rounding plus the box on x⊥, intersected with the box on x (both hold).
        through = activation_range(expert.gate, expert.up, rest)
        plain = activation_range(expert.gate, expert.up, x)
        g = (torch.maximum(plain.g[0], through.g[0] + known_g - error_g), torch.minimum(plain.g[1], through.g[1] + known_g + error_g))
        u = (torch.maximum(plain.u[0], through.u[0] + known_u - error_u), torch.minimum(plain.u[1], through.u[1] + known_u + error_u))
        act = _activations_from(g, u)
        ranges.append(act)
        a_lo, a_hi = act.a
        reach = torch.maximum(a_lo.abs(), a_hi.abs())
        part_value, part_mag, part_side = [], [], []
        for start in range(0, delta.shape[0], chunk):
            dl = delta[start : start + chunk]
            coefficients = dl @ sketch.V  # [J, k']
            orthogonal = dl - coefficients @ sketch.V.t()
            # Two lower bounds of min over D of Δᵀ·D·a for every a in the box: through the sketch, and the box alone.
            found = []
            for M, Hm in ((coefficients @ sketch.down[e] + orthogonal @ expert.down.centre,
                           orthogonal.abs() @ expert.down.half + ROUNDING_32 * (coefficients.abs() @ sketch.down[e].abs())),
                          (dl @ expert.down.centre, dl.abs() @ expert.down.half)):
                low = M * a_lo - Hm * a_lo.abs()
                high = M * a_hi - Hm * a_hi.abs()
                found.append((torch.where(high < low, high, low).sum(dim=1), high < low, (M.abs() + Hm) @ reach))
            use_sketch = found[0][0] >= found[1][0]
            part_value.append(torch.where(use_sketch, found[0][0], found[1][0]))
            part_side.append(torch.where(use_sketch[:, None], found[0][1], found[1][1]))
            part_mag.append(torch.maximum(found[0][2], found[1][2]) + dl.abs() @ (expert.down.centre.abs() + expert.down.half) @ reach)
        value = value + expert.weight * torch.cat(part_value)
        magnitude = magnitude + expert.weight * torch.cat(part_mag)
        sides.append(torch.cat(part_side))
    return Minimum(value, magnitude, sides, ranges)


def _activations_from(g: tuple[torch.Tensor, torch.Tensor], u: tuple[torch.Tensor, torch.Tensor]) -> Activations:
    """`activation_range`'s corner analysis from given ranges of g and u."""
    g_lo, g_hi = g
    u_lo, u_hi = u
    at_lo, at_hi = silu(g_lo), silu(g_hi)
    straddles = (g_lo <= SILU_ARGMIN) & (g_hi >= SILU_ARGMIN)
    s_lo = torch.where(straddles, torch.full_like(at_lo, SILU_MINIMUM), torch.minimum(at_lo, at_hi))
    g_at_s_lo = torch.where(straddles, torch.full_like(g_lo, SILU_ARGMIN), torch.where(at_lo <= at_hi, g_lo, g_hi))
    s_hi = torch.maximum(at_lo, at_hi)
    g_at_s_hi = torch.where(at_hi >= at_lo, g_hi, g_lo)
    corners = torch.stack([s_lo * u_lo, s_lo * u_hi, s_hi * u_lo, s_hi * u_hi])
    g_options = torch.stack([g_at_s_lo, g_at_s_lo, g_at_s_hi, g_at_s_hi])
    u_options = torch.tensor([False, True, False, True], device=g_lo.device)[:, None].expand_as(corners)
    low, high = corners.argmin(dim=0, keepdim=True), corners.argmax(dim=0, keepdim=True)

    def pick(t, i):
        return t.gather(0, i).squeeze(0)

    return Activations((g_lo, g_hi), (u_lo, u_hi), (pick(corners, low), pick(corners, high)),
                       (pick(g_options, low), pick(g_options, high)), (pick(u_options, low), pick(u_options, high)))


# Witnesses


@dataclass
class Witness:
    gate: list[torch.Tensor]
    up: list[torch.Tensor]
    down: list[torch.Tensor]
    value: float  # Δ_j·y with these weights, by the real forward
    inside: bool  # every weight within its interval


def witness(experts: list[ExpertSet], x: torch.Tensor, base: torch.Tensor, delta_j: torch.Tensor, minimum: Minimum, j: int) -> Witness:
    """The minimizer of contender j's pair as weights in the set, and Δ_j·y by the real forward with them."""
    sx = torch.sign(x)
    gates, ups, downs = [], [], []
    y = base.clone()
    inside = True
    for e, expert in enumerate(experts):
        act = minimum.activations[e]
        side = minimum.sides[e][j]
        g_lo_rows = expert.gate.centre - expert.gate.half * sx  # rows realizing g⁻ (and g⁺ below)
        g_hi_rows = expert.gate.centre + expert.gate.half * sx
        target = torch.where(side, act.g_for[1], act.g_for[0])
        width = act.g[1] - act.g[0]
        lam = torch.where(width > 0, (target - act.g[0]) / torch.where(width > 0, width, 1.0), 0.0).clamp(0.0, 1.0)
        G = g_lo_rows + lam[:, None] * (g_hi_rows - g_lo_rows)
        u_high = torch.where(side, act.u_high[1], act.u_high[0])
        U = torch.where(u_high[:, None], expert.up.centre + expert.up.half * sx, expert.up.centre - expert.up.half * sx)
        a = silu(G @ x) * (U @ x)
        D = expert.down.centre - expert.down.half * (torch.sign(delta_j)[:, None] * torch.sign(a)[None, :])
        inside &= expert.gate.contains(G, 1e-12) and expert.up.contains(U, 1e-12) and expert.down.contains(D, 1e-12)
        y = y + expert.weight * (D @ a)
        gates.append(G), ups.append(U), downs.append(D)
    return Witness(gates, ups, downs, float(delta_j @ y), inside)
