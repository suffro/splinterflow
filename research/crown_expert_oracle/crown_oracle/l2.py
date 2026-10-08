"""L2 weight sets as auto_LiRPA graphs (Phase 5A2, stage 1.5).

Phase 5A's realistic bounds use each row's L2 remainder norm (Cauchy–Schwarz) next to its L∞ norm (Hölder); the boxes
of `sets.py` keep only the L∞ half. Here a group of rows is given to auto_LiRPA as its own L2 ball, with upstream
functionality only:

    W_G = C_G + δ_G,   ‖δ_G‖₂ ≤ ε_G        BoundedParameter(C_G, PerturbationLpNorm(norm=2, eps=ε_G))

A group is a whole matrix (cases A and B), a page of rows (C) or one row (D). The metadata bounds each row by itself
(‖δ_r‖₂ ≤ ρ_r), so a group's ball must contain the product of its rows' balls: ε_G = ‖ρ_G‖₂ (ρ_r for one row). Two
assemblies of the groups into one linear map, both plain graph construction:

    stacked   torch.cat of the groups' parameters, then one F.linear      (`StackedLinear`)
    split     one F.linear per group, the outputs concatenated            (`SplitLinear`)

No bound is computed here. What upstream auto_LiRPA 0.7.2 does with these graphs (`probe_l2.py`, stage 1.5):

  - `split` is accepted and its backward pass concretizes each group over its own ball (dual norm), so independent groups
    stay independent; `stacked` fails in the backward pass (BoundConcat reads axis 0 as the batch axis, and the
    concatenated weight gets no interval bounds).
  - With an exact input (gate and up after x; down after an exact a) CROWN's bound is the exact optimum.
  - An L2 root's interval bounds are its centre in upstream's default mode (PerturbationLpNorm.init). The product
    relaxation of a perturbed weight with an uncertain input reads them, so it treats the weight as fixed: the bounds of
    CROWN, CROWN-IBP and α-CROWN on down after an uncertain a are unsound (above attained values).
  - In the box-hull mode (AUTOLIRPA_L2_DEBUG=1) that relaxation is valid, but IBP of an L2-perturbed linear layer uses the
    hull's lower end as its centre and is unsound, and so is CROWN on a full graph whose first layer takes its interval
    bounds from IBP. Sound in the probe: the reduced graph (an activation box × split balls of down), whose relaxation
    then works on each ball's box hull (±ρ per entry): the L2 coupling survives only in the final concretization.
"""

from __future__ import annotations

import contextlib
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import BoundedParameter, PerturbationLpNorm

# auto_LiRPA's own switch for the interval bounds of an L2-perturbed root (PerturbationLpNorm.init): unset, they are
# its centre ("FIXME This causes confusing lower bound and upper bound"); set to 1, centre ± ε ("FIXME Experimental
# code. Need to change the IBP code also."). Upstream, unmodified; the probe measures both.
L2_DEBUG = "AUTOLIRPA_L2_DEBUG"


@contextlib.contextmanager
def l2_interval_mode(box_hull: bool):
    """Run auto_LiRPA with L2 roots' interval bounds at their centre (upstream's default) or their box hull."""
    previous = os.environ.get(L2_DEBUG)
    os.environ[L2_DEBUG] = "1" if box_hull else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(L2_DEBUG, None)
        else:
            os.environ[L2_DEBUG] = previous


def l2_parameter(centre: torch.Tensor, radius: float) -> BoundedParameter:
    """A weight known to lie in the L2 ball of `radius` about `centre` (the whole tensor is one ball)."""
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("an L2 radius must be finite and non-negative")
    return BoundedParameter(centre, PerturbationLpNorm(norm=2, eps=float(radius)), requires_grad=False)


def row_ranges(rows: int, size: int | None) -> list[tuple[int, int]]:
    """The groups of rows: one group (size None), pages of `size` rows, or single rows (size 1)."""
    if size is None:
        return [(0, rows)]
    if size < 1:
        raise ValueError("a page holds at least one row")
    return [(start, min(start + size, rows)) for start in range(0, rows, size)]


def group_radii(row_radius: torch.Tensor, ranges: list[tuple[int, int]]) -> list[float]:
    """ε_G = ‖ρ_G‖₂ per group, rounded up: the smallest ball about the group's centre holding every product of its rows'
    balls."""
    out = []
    for start, end in ranges:
        norm = float(torch.linalg.vector_norm(row_radius[start:end].to(torch.float64)))
        out.append(math.nextafter(norm * (1.0 + 2.0**-40), math.inf) if norm > 0 else 0.0)
    return out


def merge_exact(ranges: list[tuple[int, int]], radii: list[float]) -> tuple[list[tuple[int, int]], list[float]]:
    """Adjacent exact groups (radius 0) joined into one: fewer nodes, the same set."""
    merged_ranges, merged_radii = [], []
    for (start, end), radius in zip(ranges, radii):
        if radius == 0 and merged_radii and merged_radii[-1] == 0 and merged_ranges[-1][1] == start:
            merged_ranges[-1] = (merged_ranges[-1][0], end)
        else:
            merged_ranges.append((start, end))
            merged_radii.append(radius)
    return merged_ranges, merged_radii


class _Groups(nn.Module):
    """The groups of one weight matrix [R, C]: an exact group is a buffer (adjacent ones joined), any other an
    L2-perturbed parameter."""

    def __init__(self, centre: torch.Tensor, row_radius: torch.Tensor, size: int | None) -> None:
        super().__init__()
        ranges = row_ranges(centre.shape[0], size)
        self.ranges, self.radii = merge_exact(ranges, group_radii(row_radius, ranges))
        self.params = nn.ParameterList()
        self.kinds: list[tuple[str, int]] = []
        for n, ((start, end), radius) in enumerate(zip(self.ranges, self.radii)):
            piece = centre[start:end].clone()
            if radius > 0:
                self.kinds.append(("param", len(self.params)))
                self.params.append(l2_parameter(piece, radius))
            else:
                name = f"exact_{n}"
                self.register_buffer(name, piece)
                self.kinds.append(("buffer", n))

    def pieces(self) -> list[torch.Tensor]:
        return [self.params[i] if kind == "param" else getattr(self, f"exact_{i}") for kind, i in self.kinds]

    @property
    def perturbed(self) -> int:
        return len(self.params)


class StackedLinear(nn.Module):
    """F.linear(x, W) with W = torch.cat of its groups (one weight node) and an exact zero bias (auto_LiRPA's Gemm path,
    `graph._zero_bias`). The groups are joined along a leading singleton axis (BoundConcat's backward pass refuses
    axis 0, which auto_LiRPA reads as a batch axis), then that axis is dropped."""

    def __init__(self, centre: torch.Tensor, row_radius: torch.Tensor, size: int | None) -> None:
        super().__init__()
        self.groups = _Groups(centre, row_radius, size)
        self.bias = nn.Parameter(torch.zeros(centre.shape[0], dtype=centre.dtype, device=centre.device), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = self.groups.pieces()
        if len(pieces) == 1:
            weight = pieces[0]
        else:
            weight = torch.cat([p.unsqueeze(0) for p in pieces], dim=1).squeeze(0)
        return F.linear(x, weight, self.bias)


class SplitLinear(nn.Module):
    """F.linear(x, W) computed group by group (each group's weight is its own node), the outputs concatenated."""

    def __init__(self, centre: torch.Tensor, row_radius: torch.Tensor, size: int | None) -> None:
        super().__init__()
        self.groups = _Groups(centre, row_radius, size)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(end - start, dtype=centre.dtype, device=centre.device), requires_grad=False)
                                        for start, end in self.groups.ranges])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [F.linear(x, piece, bias) for piece, bias in zip(self.groups.pieces(), self.biases)]
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)


class ExactLinear(nn.Module):
    """F.linear(x, W) with W exact, or a parameter given by the caller (a box: `graph.boxed_parameter`)."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        if isinstance(weight, nn.Parameter):
            self.weight = weight
        else:
            self.register_buffer("weight", weight)
        self.bias = nn.Parameter(torch.zeros(weight.shape[0], dtype=weight.dtype, device=weight.device), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LinearExperts(nn.Module):
    """y = b + Σ_e w_e · down_e(silu(gate_e(x)) ⊙ up_e(x)) with one linear module per matrix (`graph.ExpertsSuffix`'s
    function; SiLU written g·σ(g))."""

    def __init__(self, gates, ups, downs, routing, base: torch.Tensor) -> None:
        super().__init__()
        if not len(gates) == len(ups) == len(downs) == len(routing):
            raise ValueError("one gate, up, down and routing weight per expert")
        self.gates, self.ups, self.downs = nn.ModuleList(gates), nn.ModuleList(ups), nn.ModuleList(downs)
        self.routing = [float(w) for w in routing]
        self.register_buffer("base", base.reshape(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base
        for gate, up, down, weight in zip(self.gates, self.ups, self.downs, self.routing):
            g = gate(x)
            y = y + weight * down(g * torch.sigmoid(g) * up(x))
        return y


class LinearReduced(nn.Module):
    """y = b + Σ_e w_e · down_e(a_e), one activation input a_e [1, I] per expert (`graph.ReducedSuffix`'s function)."""

    def __init__(self, downs, routing, base: torch.Tensor) -> None:
        super().__init__()
        if len(downs) != len(routing):
            raise ValueError("one down projection per routing weight")
        self.downs = nn.ModuleList(downs)
        self.routing = [float(w) for w in routing]
        self.register_buffer("base", base.reshape(-1))

    def forward(self, *activations: torch.Tensor) -> torch.Tensor:
        y = self.base
        for a, down, weight in zip(activations, self.downs, self.routing):
            y = y + weight * down(a)
        return y


def perturbed_count(model: nn.Module) -> int:
    return sum(1 for p in model.parameters() if isinstance(p, BoundedParameter) and p.ptb is not None)


def uniform_in_balls(centre: torch.Tensor, radius: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    """`count` points of the product of balls: each group's rows (the last two axes of `centre`, [G, rows, C]) drawn
    uniformly from its ball of `radius` [G], independently: [count, G, rows, C]."""
    groups, size = centre.shape[0], centre[0].numel()
    direction = torch.randn(count, groups, size, generator=generator, dtype=centre.dtype)
    direction = direction / torch.linalg.vector_norm(direction, dim=2, keepdim=True)
    scale = torch.rand(count, groups, 1, generator=generator, dtype=centre.dtype) ** (1.0 / size) * radius.reshape(1, groups, 1)
    return centre.unsqueeze(0) + (direction * scale).reshape(count, *centre.shape)
