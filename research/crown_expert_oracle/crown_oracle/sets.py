"""Weight and activation sets from what a runtime could have read (Phase 5A2).

A row of a routed expert's matrix is in one of the states of Phase 5A's oracle: UNKNOWN (nothing read but its resident
norms), a level l of its precision refinement (levels 0..l read; the cumulative approximation A_l = Σ_{l'≤l} codes·scale
is exact in float64, decision 0003), or EXACT (its BF16 bytes read). Everything read constrains the row, so its set is
the intersection

    UNKNOWN    |W_rc| ≤ n_r                                   (the row's resident L∞ norm)
    level l    |W_rc| ≤ n_r  and  |W_rc − A_l',rc| ≤ ρ_l',r  for every l' ≤ l   (each level's resident remainder bound)
    EXACT      W_rc itself

with each end one float64 ulp outward. These are auto_LiRPA's elementwise boxes (`graph.boxed_parameter`). A runtime
holds no more than this: the codes, the scales and the norms (Phase 5A charges their bytes).

The builder never sees an unread value. It takes the BF16 rows through `read_view`, which keeps only the rows in state
EXACT and puts NaN everywhere else: a set built from it cannot depend on an unread byte (tested by poisoning them).
`contains` checks a set against the true weights afterwards, as validation only.

Activation sets (the reduced graph's inputs) are Phase 5A's activation enclosures per row level, exported by
`export.py` and gathered by each neuron's level: an activation depends only on its own gate and up rows (the export
checks this on every state).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from crown_oracle.artifact import EXACT, UNKNOWN, MatrixData


def _down(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x, torch.full_like(x, -math.inf))


def _up(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x, torch.full_like(x, math.inf))


def read_view(truth: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """The BF16 rows a runtime has read (state EXACT), in float64; NaN for every other row."""
    exact = (states == EXACT).unsqueeze(-1)
    return torch.where(exact, truth.to(torch.float64), torch.full((), math.nan, dtype=torch.float64, device=truth.device))


@dataclass(frozen=True)
class Box:
    lower: torch.Tensor
    upper: torch.Tensor
    centre: torch.Tensor  # the runtime's centre execution: A_l (or 0, or the exact value), clamped into the box

    def contains(self, values: torch.Tensor) -> bool:
        values = values.to(torch.float64)
        return bool(((self.lower <= values) & (values <= self.upper)).all())

    def includes(self, other: Box) -> bool:
        """Set inclusion: other ⊆ self."""
        return bool(((self.lower <= other.lower) & (other.upper <= self.upper)).all())

    @property
    def radius(self) -> torch.Tensor:
        return (self.upper - self.lower) * 0.5


def matrix_box(data: MatrixData, states: torch.Tensor, view: torch.Tensor) -> Box:
    """The set of one matrix of one expert ([R, C]) given its rows' states [R] and the read view of its BF16 rows."""
    if view.shape != data.truth.shape:
        raise ValueError("the read view must have the matrix's shape")
    device = view.device
    own = data.own_linf.to(device=device, dtype=torch.float64)[:, None]
    lower = (-own).expand_as(view).clone()
    upper = own.expand_as(view).clone()
    centre = torch.zeros_like(view)
    approximation = torch.zeros_like(view)
    for level, (codes, scales, rho) in enumerate(zip(data.codes, data.scales, data.remainder_linf)):
        approximation = approximation + codes.to(device=device, dtype=torch.float64) * scales.to(device=device, dtype=torch.float64)[:, None]
        read = ((states >= level) & (states != UNKNOWN)).unsqueeze(-1)  # levels 0..level read (an EXACT row read them all)
        radius = rho.to(device=device, dtype=torch.float64)[:, None]
        lower = torch.where(read, torch.maximum(lower, approximation - radius), lower)
        upper = torch.where(read, torch.minimum(upper, approximation + radius), upper)
        centre = torch.where((states == level).unsqueeze(-1), approximation, centre)
    lower, upper = _down(lower), _up(upper)
    exact = (states == EXACT).unsqueeze(-1)
    lower = torch.where(exact, view, lower)
    upper = torch.where(exact, view, upper)
    centre = torch.where(exact, view, torch.minimum(torch.maximum(centre, lower), upper))
    if bool(torch.isnan(lower).any() or torch.isnan(upper).any()):
        raise ValueError("a set needs a value that was not read")
    return Box(lower, upper, centre)


def expert_boxes(matrices: list[MatrixData], states: torch.Tensor, device) -> Box:
    """The boxes of one matrix kind across the routed experts: [K, R, C] (states [K, R])."""
    boxes = [matrix_box(m, states[k].to(device), read_view(m.truth.to(device), states[k].to(device))) for k, m in enumerate(matrices)]
    return Box(torch.stack([b.lower for b in boxes]), torch.stack([b.upper for b in boxes]), torch.stack([b.centre for b in boxes]))


def activation_box(levels: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, states: torch.Tensor) -> Box:
    """Phase 5A's activation enclosures per row level (levels [L] in reading order, lower/upper [L, K, I]) for neurons in
    states [K, I] (their gate rows' states): the intersection of the enclosures of every level a neuron has reached (each
    holds for every weight consistent with that level's reads, so for the true weights)."""
    box_lower = torch.full_like(lower[0], -math.inf)
    box_upper = torch.full_like(upper[0], math.inf)
    known = torch.zeros_like(states, dtype=torch.bool)
    for position, level in enumerate(levels.tolist()):
        reached = states >= level if level != EXACT else states == EXACT
        box_lower = torch.where(reached, torch.maximum(box_lower, lower[position]), box_lower)
        box_upper = torch.where(reached, torch.minimum(box_upper, upper[position]), box_upper)
        known |= states == level
    if not bool(known.all()):
        raise ValueError("a neuron's level has no exported enclosure")
    return Box(box_lower, box_upper, box_lower * 0.5 + box_upper * 0.5)
