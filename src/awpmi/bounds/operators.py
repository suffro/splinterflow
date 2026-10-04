"""Verified propagation through the reference operations of a last MLP (or MoE block) and the final norm.

Each function maps enclosures of an operation's inputs to an enclosure of its output,
for the operation exactly as the reference executes it on BF16 tensors (`transformers`
5.18 LlamaMLP and LlamaRMSNorm, the decoder's residual addition, the numerics of
`awpmi.runtime`; Phase 5A: the same operations in DeepseekV3's experts, grouped_mm, and
DeepseekV3RMSNorm, which is LlamaRMSNorm's code):

  linear        F.linear: fp32 accumulation (`ReferenceNumerics`, any order), output rounded
  residual_add  a + b: the exact sum, rounded (fp32 opmath, then the output dtype)
  multiply      a * b: the exact product, rounded (BF16 × BF16 → BF16; BF16 × float32 → float32,
                the routing weights)
  reduce_sum    Σ_k t_k in fp32, then converted (the experts call's combine over its top-k)
  silu          F.silu: x / (1 + exp(−x)) evaluated in fp32, rounded
  rms_norm      LlamaRMSNorm: x in fp32, v = mean(x²), q = rsqrt(v + eps), n = rnd(x·q),
                output rnd(weight · n)

The output is rounded by the `RoundingModel` of the operation's kind (GEMM epilogue or
elementwise kernel, `awpmi.bounds.rounding`). Assumptions on binary32 elementwise
arithmetic, deliberately generous:

  * every fp32 operation of an elementwise kernel (a basic operation, `expf`, `rsqrtf`,
    `powf`, the scaling of a mean) has relative error ≤ FP32_OP_RELATIVE_ERROR = 2⁻¹⁸,
    i.e. 32 binary32 ulps. IEEE basic operations are within 1 ulp; the CUDA Math API
    documents at most 2 ulps for expf and rsqrtf and 4 for powf;
  * a result below the normal range may be flushed to zero (FP32_NORMAL_MIN allowance);
  * a reduction of n fp32 terms errs by at most γ_{n+2}(2⁻²²) relative to Σ|terms|, the
    accumulation model of decision 0001.

All float64 bound arithmetic is stepped outward (`next_down`/`next_up`, relative slack).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from awpmi.bounds.enclosure import Enclosure, UnboundedValue
from awpmi.bounds.floating import FLOAT64_BOUND_SLACK, FLOAT64_UNIT_ROUNDOFF, gamma, next_down, next_up
from awpmi.bounds.linear import absolute_mass_upper, matvec
from awpmi.bounds.residual import ReferenceNumerics
from awpmi.bounds.rounding import RoundingModel, round_enclosure

FP32_OP_RELATIVE_ERROR = 2.0**-18
FP32_NORMAL_MIN = 2.0**-126
# x² must stay far from the binary32 overflow threshold (2¹²⁸), also summed over a row.
RMS_NORM_MAX_INPUT = 2.0**55
# SiLU: |x|·e^x for x < −87 (where exp(−x) may overflow and the kernel returns −0) is below this.
SILU_ABSOLUTE_ERROR = 2.0**-110
# min_x x·σ(x) = −W(1/e) = −0.27846454276107379510…, attained at x* = −1 − W(1/e); a lower bound.
SILU_MINIMUM_LOWER = -0.2784645427610739
SILU_ARGMIN = -1.2784645427610738


def _check(*enclosures: Enclosure) -> None:
    for enclosure in enclosures:
        if not enclosure.finite:
            raise UnboundedValue("a non-finite bound cannot be propagated")


def _outward(lower: torch.Tensor, upper: torch.Tensor, relative: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Step [lower, upper] outward by a relative amount and one float64 ulp."""
    if relative:
        lower = lower - lower.abs() * relative
        upper = upper + upper.abs() * relative
    return next_down(lower), next_up(upper)


def _flushable(lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Include zero when a binary32 result could be flushed (|value| below the normal range)."""
    lower = torch.where((lower > 0) & (lower < FP32_NORMAL_MIN), torch.zeros_like(lower), lower)
    upper = torch.where((upper < 0) & (upper > -FP32_NORMAL_MIN), torch.zeros_like(upper), upper)
    return lower, upper


def _products(a_lo, a_hi, b_lo, b_hi) -> tuple[torch.Tensor, torch.Tensor]:
    """Bounds on a·b over the boxes (each float64 product rounds once: one ulp outward)."""
    products = torch.stack([a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi])
    return next_down(products.min(dim=0).values), next_up(products.max(dim=0).values)


def _rounded(lower, upper, dtype, model, provenance, label) -> Enclosure:
    lower, upper = round_enclosure(lower, upper, dtype, model)
    return Enclosure(lower, upper, provenance | {f"rounding:{label}"})


def linear(
    inputs: Enclosure,
    weight: torch.Tensor,
    numerics: ReferenceNumerics,
    model: RoundingModel,
    unread_mass: torch.Tensor | None = None,
    label: str = "linear",
) -> Enclosure:
    """F.linear(x, W) of the reference, for x in `inputs` (the last dim) and W = `weight` + unread part.

    `weight` (float64 [N, K]) holds the materialized entries, zeros elsewhere;
    `unread_mass[n]` ≥ Σ over unread entries |W_nk|·|x_k| bounds both the missing
    contribution and its share of the accumulation error. With centre c and radius ρ:

        centre  W·c        radius  |W|·ρ + unread + (γ_acc + γ₆₄)·(|W|·(|c| + ρ) + unread)

    Phase 5A: a batch of independent operations, weight [B, N, K] with inputs [K] or [B, K]
    (one experts call's routed experts), unread_mass [B, N].
    """
    _check(inputs)
    if weight.dtype != torch.float64 or weight.shape[-1] != numerics.reduction_length or weight.dim() not in (2, 3):
        raise ValueError("weight must be float64 [N, reduction_length] (or a batch of them)")
    center = inputs.center
    rho = inputs.radius_about(center)
    out_center = matvec(weight, center)
    mass = absolute_mass_upper(weight, next_up(center.abs() + rho))
    radius = absolute_mass_upper(weight, rho)
    if unread_mass is not None:
        mass = mass + unread_mass
        radius = radius + unread_mass
    radius = (radius + (numerics.accumulation_gamma + numerics.float64_gamma) * mass) * (1.0 + FLOAT64_BOUND_SLACK)
    lower, upper = next_down(out_center - radius), next_up(out_center + radius)
    return _rounded(lower, upper, numerics.output_dtype, model, inputs.provenance, label)


def residual_add(left: Enclosure, right: Enclosure, dtype: torch.dtype, model: RoundingModel, label: str = "add") -> Enclosure:
    _check(left, right)
    lower, upper = _flushable(*_outward(left.lower + right.lower, left.upper + right.upper))
    return _rounded(lower, upper, dtype, model, left.provenance | right.provenance, label)


def multiply(left: Enclosure, right: Enclosure, dtype: torch.dtype, model: RoundingModel, label: str = "mul") -> Enclosure:
    _check(left, right)
    lower, upper = _flushable(*_products(left.lower, left.upper, right.lower, right.upper))
    return _rounded(lower, upper, dtype, model, left.provenance | right.provenance, label)


def reduce_sum(
    terms: list[Enclosure] | Enclosure,
    reduction_unit_roundoff: float,
    dtype: torch.dtype,
    model: RoundingModel,
    label: str = "sum",
) -> Enclosure:
    """Σ_k terms[k] reduced in fp32 (any order), then converted to `dtype` (Phase 5A: an experts call's combine).

    transformers' grouped_mm combine: `weighted.view(T, K, H).sum(dim=1)` in float32, then
    `.to(bfloat16)`. The fp32 reduction of n terms errs by at most γ_{n+2}(u)·Σ|terms| (decision
    0001's accumulation model, as the norm's reduction); the conversion is one rounding to `dtype`.
    `terms` is a list, or one enclosure whose first dimension is summed.
    """
    if isinstance(terms, Enclosure):
        _check(terms)
        count = terms.lower.shape[0]
        lower, upper, magnitude = terms.lower.sum(dim=0), terms.upper.sum(dim=0), terms.magnitude.sum(dim=0)
        provenance = terms.provenance
    else:
        _check(*terms)
        if not terms:
            raise ValueError("nothing to sum")
        count = len(terms)
        lower = torch.stack([term.lower for term in terms]).sum(dim=0)
        upper = torch.stack([term.upper for term in terms]).sum(dim=0)
        magnitude = torch.stack([term.magnitude for term in terms]).sum(dim=0)
        provenance = frozenset().union(*(term.provenance for term in terms))
    if count < 1:
        raise ValueError("nothing to sum")
    # The float64 sums above (n terms each) err by at most γ_n(2⁻⁵³) relative to the magnitude.
    error = magnitude * (gamma(count + 2, reduction_unit_roundoff) + gamma(count + 2, FLOAT64_UNIT_ROUNDOFF))
    lower, upper = _outward(lower - error, upper + error, FLOAT64_BOUND_SLACK)
    return _rounded(lower, upper, dtype, model, provenance, label)


def _silu64(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.exp(-x))


def silu(inputs: Enclosure, dtype: torch.dtype, model: RoundingModel, label: str = "silu") -> Enclosure:
    """x·σ(x) decreases on (−∞, x*] and increases on [x*, ∞): its range is at the endpoints or x*."""
    _check(inputs)
    at_lower, at_upper = _silu64(inputs.lower), _silu64(inputs.upper)
    lower = torch.minimum(at_lower, at_upper)
    lower = torch.where((inputs.lower <= SILU_ARGMIN) & (inputs.upper >= SILU_ARGMIN), SILU_MINIMUM_LOWER, lower)
    upper = torch.maximum(at_lower, at_upper)
    # fp32 evaluation (exp, add, divide) and the float64 evaluation above; −0 for overflowed exp.
    lower, upper = _outward(lower, upper, FP32_OP_RELATIVE_ERROR + 2.0**-40)
    lower, upper = _flushable(lower - SILU_ABSOLUTE_ERROR, upper + SILU_ABSOLUTE_ERROR)
    return _rounded(lower, upper, dtype, model, inputs.provenance, label)


@dataclass(frozen=True)
class RMSNormBounds:
    """Every reference intermediate of LlamaRMSNorm that later bounds need."""

    scale_lower: float  # q = rsqrt(mean(x²) + eps) in fp32, a positive scalar
    scale_upper: float
    normalized: Enclosure  # n = rnd(x·q)
    output: Enclosure  # rnd(weight · n)


def rms_norm(
    inputs: Enclosure,
    weight: torch.Tensor,
    eps: float,
    model: RoundingModel,
    reduction_unit_roundoff: float,
    label: str = "norm",
) -> RMSNormBounds:
    """LlamaRMSNorm on one position (`inputs` over the hidden dim), output in `weight`'s dtype."""
    _check(inputs)
    if inputs.lower.dim() != 1 or inputs.lower.numel() != weight.numel():
        raise ValueError("one position of the hidden dimension is expected")
    if bool((inputs.magnitude > RMS_NORM_MAX_INPUT).any()):
        raise UnboundedValue("RMSNorm input too large for the binary32 variance")
    count = inputs.lower.numel()
    straddles = (inputs.lower <= 0) & (inputs.upper >= 0)
    square_lo = torch.where(straddles, 0.0, torch.minimum(inputs.lower.square(), inputs.upper.square()))
    square_hi = torch.maximum(inputs.lower.square(), inputs.upper.square())
    # fp32 squares (relative error, or flushed), then a fp32 sum of non-negative terms in any order.
    square_lo = (next_down(square_lo) * (1.0 - FP32_OP_RELATIVE_ERROR) - FP32_NORMAL_MIN).clamp_min(0.0)
    square_hi = next_up(square_hi) * (1.0 + FP32_OP_RELATIVE_ERROR) + FP32_NORMAL_MIN
    reduction = gamma(count + 2, reduction_unit_roundoff)
    summation = gamma(count + 2, FLOAT64_UNIT_ROUNDOFF)  # the float64 sums below
    total_lo = float(square_lo.sum()) * (1.0 - summation) * (1.0 - reduction) - count * FP32_NORMAL_MIN
    total_hi = float(square_hi.sum()) * (1.0 + summation) * (1.0 + reduction) + count * FP32_NORMAL_MIN
    # mean: scaling by 1/count; then + eps (eps rounded to fp32 first); then rsqrt.
    shrink, grow = 1.0 - FP32_OP_RELATIVE_ERROR, 1.0 + FP32_OP_RELATIVE_ERROR
    variance_lo = max(total_lo, 0.0) / count * shrink
    variance_hi = total_hi / count * grow
    shifted_lo = (variance_lo + eps * (1.0 - 2.0**-24)) * shrink * (1.0 - 2.0**-50)
    shifted_hi = (variance_hi + eps * (1.0 + 2.0**-24)) * grow * (1.0 + 2.0**-50)
    if not shifted_lo > 0.0 or not math.isfinite(shifted_hi):
        raise UnboundedValue("RMSNorm variance out of range")
    scale_lower = 1.0 / math.sqrt(shifted_hi) * shrink * (1.0 - 2.0**-50)
    scale_upper = 1.0 / math.sqrt(shifted_lo) * grow * (1.0 + 2.0**-50)
    scale_lo = torch.full_like(inputs.lower, scale_lower)
    scale_hi = torch.full_like(inputs.lower, scale_upper)
    product = _flushable(*_products(inputs.lower, inputs.upper, scale_lo, scale_hi))
    normalized = _rounded(*product, weight.dtype, model, inputs.provenance, f"{label}.normalized")
    gain = weight.to(torch.float64)
    scaled = _flushable(*_products(normalized.lower, normalized.upper, gain, gain))
    output = _rounded(*scaled, weight.dtype, model, normalized.provenance, label)
    return RMSNormBounds(scale_lower, scale_upper, normalized, output)
