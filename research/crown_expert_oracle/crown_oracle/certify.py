"""The two decisions of Phase 5A2, assembled around auto_LiRPA's structural bound.

STRUCTURAL CROWN DIAGNOSTIC (real arithmetic, not a certificate of the reference). The decision of the real computation
with the true weights is the sign of Δ·y (Δ = (W_w − W_j)⊙g; the norm's scale is common). A pair is settled when

    max(box, L) − slack > 0,      L = auto_LiRPA's lower bound on Δ·y over the weight set (or the reduced graph's set),

box = Δ·c − |Δ|·ρ over the real tier's enclosure of y (Phase 5A's), slack = float64 rounding (Phase 5A's real-tier
formula) plus `computation_slack` of the structural expression's absolute mass (auto_LiRPA computes in float64 without
directed rounding: an assumption, validated on every evaluation).

CERTIFIED-REFERENCE RESULT (the reference's BF16 arithmetic, Phase 5A's rounding model). y_ref = b + Σ_e w_e·D_e·a_e + η
(Phase 5A's mixture decomposition, decision 0009): a_e the reference's activation, inside Phase 5A's certified enclosure;
|η_k| ≤ N_k its named errors; then y's own rounding, the final norm's and the LM head's. auto_LiRPA bounds the structural
part T = Δ·(b + Σ_e w_e·D_e·a_e) over the down projections' weight sets and the certified activation enclosure (the
reduced graph); everything else is Phase 5A's pairwise certificate (`awpmi.bounds.pairwise.pairwise_certificate`, its
mixture branch) term for term, from the exported per-state vectors and constants:

    decomposed = L_T − |Δ|·N − |Δ|·ε_y,      lower = max(box, decomposed)
    bound      = lower − ξ·|Δ|·Y − γ_lm·(1 + ξ)·(|W_w| + |W_j|)·|g|·Y
    margin     = (bound − slack)·q⁻·(1 − 2⁻⁵⁰) − ↑(|W_w − W_j|·μ + γ_lm·(|W_w| + |W_j|)·μ + 2·U)

A pair is certified when margin > 0 (the faithful logits two grid spacings apart). The assembly repeats Phase 5A's code
(this environment imports no awpmi); `run.py --part validate` checks it against Phase 5A's own margins on the exported
comparison pairs, with Phase 5A's structural bound in place of auto_LiRPA's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

FLOAT64_UNIT_ROUNDOFF = 2.0**-53


def gamma(n: int, unit_roundoff: float) -> float:
    nu = n * unit_roundoff
    if nu >= 1.0:
        raise ValueError("no finite error bound")
    return math.nextafter(nu / (1.0 - nu), math.inf)


def next_up(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x, torch.full_like(x, math.inf))


def round_up_to_grid(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Smallest value of `dtype` ≥ x, as float64 (awpmi.bounds.floating.round_up_to_grid)."""
    candidate = x.to(dtype)
    too_low = candidate.to(torch.float64) < x
    candidate = torch.where(too_low, torch.nextafter(candidate, torch.full_like(candidate, math.inf)), candidate)
    return candidate.to(torch.float64)


def spacing_upper_bf16(magnitude: torch.Tensor) -> torch.Tensor:
    """Upper bound on the BF16 grid spacing at magnitudes ≤ `magnitude` (awpmi.bounds.rounding.spacing_upper)."""
    precision, subnormal = 8, -133
    magnitude = magnitude.to(torch.float64).abs()
    _, exponent = torch.frexp(magnitude)
    spacing = torch.ldexp(torch.ones_like(magnitude), exponent - precision)
    spacing = torch.where(magnitude > 0, torch.clamp_min(spacing, 2.0**subnormal), torch.full_like(spacing, 2.0**subnormal))
    return torch.where(torch.isfinite(magnitude), spacing, torch.full_like(spacing, math.inf))


def pair_deltas(lm_weight: torch.Tensor, gain: torch.Tensor, candidate: int, rows: torch.Tensor):
    """W_w, W_j (float64), W_w − W_j and Δ = (W_w − W_j)⊙g for contenders `rows`."""
    winner = lm_weight[candidate].to(torch.float64)
    others = lm_weight.index_select(0, rows).to(torch.float64)
    difference = winner[None, :] - others
    return winner, others, difference, difference * gain.to(torch.float64)[None, :]


def logit_magnitudes(lm_weight: torch.Tensor, rows: torch.Tensor, h_magnitude: torch.Tensor, inflation: float) -> torch.Tensor:
    """≥ |ℓ_j| for `rows` (Phase 5A's Certifier._magnitudes): |W_j|·|h|⁺ in binary32, inflated, up to the BF16 grid."""
    magnitude = round_up_to_grid(h_magnitude, torch.float32).to(torch.float32)
    total = (lm_weight.index_select(0, rows).to(torch.float32).abs() @ magnitude).to(torch.float64)
    return round_up_to_grid(next_up(total * inflation), lm_weight.dtype)


def structural_absolute(delta: torch.Tensor, base: torch.Tensor, routing: torch.Tensor, down_magnitude: torch.Tensor,
                        a_magnitude: torch.Tensor) -> torch.Tensor:
    """|Δ|·(|b| + Σ_e w_e·|D_e|⁺·|a_e|⁺): the absolute mass of the structural expression, per contender."""
    mass = base.abs() + torch.einsum("e,eh->h", routing.to(torch.float64), torch.einsum("ehi,ei->eh", down_magnitude, a_magnitude))
    return delta.abs() @ mass


@dataclass(frozen=True)
class CertifiedState:
    """Phase 5A's certified-tier quantities of one state (exported)."""

    y_lower: torch.Tensor
    y_upper: torch.Tensor
    errors: torch.Tensor  # Σ of the mixture's named error vectors (N)
    y_rounding: torch.Tensor  # ε_y
    h_magnitude: torch.Tensor
    scale_lower: float  # q⁻


def certified_margins(lm_weight, gain, mu, constants: dict, state: CertifiedState, candidate: int, rows: torch.Tensor,
                      structural: torch.Tensor, absolute_structural: torch.Tensor, computation_slack: float) -> dict[str, torch.Tensor]:
    """The certified margins of `candidate` against `rows`, given a lower bound `structural` on T per contender."""
    winner, others, difference, delta = pair_deltas(lm_weight, gain, candidate, rows)
    absolute_delta = delta.abs()
    Y = torch.maximum(state.y_lower.abs(), state.y_upper.abs())
    box = torch.minimum(delta * state.y_lower[None, :], delta * state.y_upper[None, :]).sum(dim=1)
    absolute_terms = absolute_delta @ (Y + state.y_lower.abs() + state.y_upper.abs())
    errors = absolute_delta @ state.errors
    y_rounding = absolute_delta @ state.y_rounding
    decomposed = structural - errors - y_rounding
    absolute_terms = absolute_terms + absolute_structural + errors + y_rounding
    lower = torch.maximum(box, decomposed)
    xi, lm_gamma = constants["xi"], constants["lm_gamma"]
    gain64 = gain.to(torch.float64)
    lm_mass = (winner.abs()[None, :] + others.abs()) * gain64.abs()[None, :]
    norm_rounding = xi * (absolute_delta @ Y)
    lm_accumulation = lm_gamma * (1.0 + xi) * (lm_mass @ Y)
    bound = lower - norm_rounding - lm_accumulation
    absolute_terms = absolute_terms + norm_rounding + lm_accumulation
    length = 2 * (delta.shape[1] + int(constants["neurons"])) + 64
    slack = 4.0 * gamma(length, FLOAT64_UNIT_ROUNDOFF) * absolute_terms + computation_slack * absolute_structural
    absolute = difference.abs() @ mu + lm_gamma * ((winner.abs()[None, :] + others.abs()) @ mu)
    magnitudes = logit_magnitudes(lm_weight, torch.cat([torch.tensor([candidate], device=rows.device), rows]), state.h_magnitude, constants["magnitude_inflation"])
    spacing = spacing_upper_bf16(torch.maximum(magnitudes[1:], magnitudes[0]))
    margin = (bound - slack) * state.scale_lower * (1.0 - 2.0**-50)
    margin = margin - next_up(absolute + constants["separations"] * spacing)
    return {"box": box, "decomposed": decomposed, "margin": margin, "norm_rounding": norm_rounding, "errors": errors, "y_rounding": y_rounding}


def structural_margins(lm_weight, gain, candidate: int, rows: torch.Tensor, y_lower: torch.Tensor, y_upper: torch.Tensor,
                       structural: torch.Tensor, absolute_structural: torch.Tensor, neurons: int, computation_slack: float) -> dict[str, torch.Tensor]:
    """The real computation's pairwise margins (Phase 5A's real tier, `Certifier._real_margins`) with auto_LiRPA's bound."""
    _, _, _, delta = pair_deltas(lm_weight, gain, candidate, rows)
    centre = y_lower * 0.5 + y_upper * 0.5
    radius = next_up(torch.maximum(y_upper - centre, centre - y_lower)).clamp_min(0.0)
    box = delta @ centre - delta.abs() @ radius
    magnitude = torch.maximum(y_lower.abs(), y_upper.abs())
    absolute = delta.abs() @ (magnitude + centre.abs()) + absolute_structural
    slack = 4.0 * gamma(2 * (delta.shape[1] + neurons) + 64, FLOAT64_UNIT_ROUNDOFF) * absolute + computation_slack * absolute_structural
    return {"box": box, "decomposed": structural, "margin": torch.maximum(box, structural) - slack}
