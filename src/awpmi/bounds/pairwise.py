"""Scale-free pairwise certificate through the final RMSNorm (Phase 2, decision 0005).

Interval bounds treat every logit separately. Behind the final RMSNorm every logit
carries the same positive factor q = rsqrt(mean(y²) + eps) and the same rounding errors
of each h_k, so separate intervals count them twice and lose the cancellation of the
layers in between. For a candidate w against a contender j, with g the norm weight and
Δ_k = (W_wk − W_jk)·g_k, the reference's values satisfy (`awpmi.bounds.operators`)

    h_k = g_k·y_k·q·(1 + ξ_k) + μ_k,      |ξ_k| ≤ ξ, |μ_k| ≤ μ        (roundings of n_k and h_k)
    acc_w − acc_j = Σ_k (W_wk − W_jk)·h_k + e_w − e_j,      |e_i| ≤ γ_lm·Σ_k |W_ik|·|h_k|

so that acc_w − acc_j ≥ q·F − A with Y_k ≥ |y_k|, L ≤ Σ_k Δ_k·y_k and

    F = L − ξ·Σ_k |Δ_k|·Y_k − γ_lm·(1 + ξ)·Σ_k (|W_wk| + |W_jk|)·|g_k|·Y_k
    A = Σ_k |W_wk − W_jk|·μ + γ_lm·Σ_k (|W_wk| + |W_jk|)·μ

Two lower bounds L are used, the larger one per pair:

  box         Σ_k min(Δ_k·y_k⁻, Δ_k·y_k⁺) over the enclosure of y;
  decomposed  y_k = r_k + Σ_i W_d[k,i]·a_i + e_k + τ_k + ε_k: the residual, the MLP's
              down projection with its accumulation error e_k (|e_k| ≤ E_k), and the
              roundings of o_k (τ_k) and y_k (ε_k). With M = W_dᵀ·Δ,
                L = Δ·r + Σ_{i read} min(M_i·a_i⁻, M_i·a_i⁺) − ‖Δ‖₂·Σ_{i unread} C_i·|a_i|⁺
                    − Σ_k |Δ_k|·(E_k + T_k + Ε_k)
              where C_i ≥ ‖W_d[:, i]‖₂ (Cauchy–Schwarz on the unread columns). It keeps
              the cancellation of the down projection, which the box loses.

Each logit is rounded faithfully and independently (decision 0001), so ℓ_w > ℓ_j once
acc_w − acc_j > 2U, U the grid spacing at the two logits' magnitude; the pair is certified
when q⁻·F − A > 2U. That also settles every tie, whatever the indices. Float64
arithmetic is covered by a relative slack on the sum of the absolute values of all terms.

Phase 5A (decision 0009) adds a third lower bound, for a mixture of routed experts (an
`ExpertMixture`): y = b + Σ_e w_e·(K_e + X_e)·a_e + η, where b is exact (the residual plus the
shared experts), w_e > 0 the routing weights, K_e the known part of expert e's down
projection (exact or coarse values, 0 elsewhere), X_e the rest, a_e its activation and
|η_k| ≤ N_k the named additive errors (accumulations and roundings of the experts' outputs,
the routing-weight products, the combine, the MoE block's addition). With M_e = K_eᵀ·Δ,

    L = Δ·b + Σ_e w_e·[Σ_i min(M_e,i·a_i⁻, M_e,i·a_i⁺) − min(Σ_k |Δ_k|·ρ_e,k, ‖Δ‖₂·Γ_e)]
          − Σ_k |Δ_k|·N_k − Σ_k |Δ_k|·(y's rounding)

where ρ_e,k ≥ |(X_e·a)_k| (row form) and Γ_e ≥ ‖X_e·a‖₂ (column form, Σ_i ‖X_e[:, i]‖₂·|a_i|⁺) for
every a of the enclosure. Phase 2's `DownProjection` is the case of one expert of weight 1.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from awpmi.bounds.enclosure import Enclosure
from awpmi.bounds.floating import FLOAT64_UNIT_ROUNDOFF, gamma, next_up
from awpmi.bounds.operators import FP32_NORMAL_MIN
from awpmi.bounds.rounding import RoundingAssumptions, RoundingModel, relative_rounding_error, rounding_error_upper, subnormal_floor, spacing_upper


@dataclass(frozen=True)
class DownProjection:
    """What the decomposed bound needs to know about y = r + rnd(W_d·a) (one position)."""

    residual: torch.Tensor  # r, exact, float64 [H]
    read_weight: torch.Tensor  # W_d with unread columns zeroed, float64 [H, I]
    read: torch.Tensor  # bool [I]: columns read
    activation: Enclosure  # a over [I]; entries of unread neurons are ignored
    unread_activation: torch.Tensor  # |a_i| bound of unread neurons, float64 [I]
    column_norms: torch.Tensor  # C_i ≥ ‖W_d[:, i]‖₂, float64 [I]
    accumulation_error: torch.Tensor  # E_k ≥ |e_k|, float64 [H]
    output: Enclosure  # o, which also encloses its pre-rounding value


@dataclass(frozen=True)
class MixtureTerm:
    """One routed expert's share w·(K + X)·a of y (Phase 5A)."""

    weight: float  # the routing weight w > 0 (its float32 value, exact in float64)
    known: torch.Tensor  # K, float64 [H, I]: the known part of the down projection, 0 where unknown
    activation: Enclosure  # a over [I]
    row_remainder: torch.Tensor  # ρ_k ≥ |(X·a)_k| for every a in the enclosure, float64 [H]
    column_remainder: float | None = None  # Γ ≥ ‖X·a‖₂ for every a in the enclosure, or None


@dataclass(frozen=True)
class ExpertMixture:
    """y = base + Σ_e w_e·(K_e + X_e)·a_e + η with |η_k| ≤ Σ_name errors[name]_k (all but y's own rounding)."""

    base: torch.Tensor  # the exact part of y: residual plus shared experts, float64 [H]
    terms: tuple[MixtureTerm, ...]
    errors: dict[str, torch.Tensor]  # named per-coordinate error bounds, float64 [H] each


@dataclass(frozen=True)
class RankOneError:
    """An elementwise error bound of the form rows[j]·columns[i] (≥ |M̂_ji − M_ji|)."""

    rows: torch.Tensor  # [J]
    columns: torch.Tensor  # [I]


# M_e = K_eᵀ·Δ for a block of contenders, and a bound on its error: None (exact), elementwise [J, I], or rank one.
Projection = Callable[[torch.Tensor, MixtureTerm], tuple[torch.Tensor, "torch.Tensor | RankOneError | None"]]


def exact_projection(delta: torch.Tensor, term: MixtureTerm) -> tuple[torch.Tensor, None]:
    """Δ·K in float64; its rounding is covered by the certificate's slack on the absolute terms."""
    return delta @ term.known, None


def relax_mixture(mixture: ExpertMixture, keep: list[torch.Tensor]) -> ExpertMixture:
    """A weaker decomposition that keeps only the `keep` columns of each term's known part (bool [I] per term).

    A dropped column's K_i·a_i is moved into the base at the enclosure's centre c_i, and its spread |K_i|·ρ_i (ρ_i the
    radius about c_i) into an error on y, with the float64 rounding of the move: y is unchanged, only bounded less
    tightly. The certificate stays valid; it costs less when few columns are kept (the screen keeps none).
    """
    base = mixture.base.clone()
    spread = torch.zeros_like(base)
    terms = []
    for term, columns in zip(mixture.terms, keep):
        drop = ~columns
        a = term.activation
        centre = a.center
        radius = a.radius_about(centre)
        known = term.known[:, drop]
        moved = known @ centre[drop]
        mass = known.abs() @ centre[drop].abs()
        base = base + term.weight * moved
        # The dot products (n terms), the product by the weight and the addition: γ_{n+3}(2⁻⁵³) of the masses involved.
        rounding = gamma(int(drop.sum()) + 3, FLOAT64_UNIT_ROUNDOFF) * (term.weight * mass + base.abs())
        spread = spread + term.weight * (known.abs() @ radius[drop]) + rounding
        kept = Enclosure(a.lower[columns], a.upper[columns], a.provenance)
        terms.append(MixtureTerm(term.weight, term.known[:, columns], kept, term.row_remainder, term.column_remainder))
    errors = {**mixture.errors, "relaxed_columns": next_up(spread * (1.0 + 2.0**-40))}
    return ExpertMixture(base, tuple(terms), errors)


@dataclass(frozen=True)
class MixtureBound:
    """The mixture's part of the decomposed lower bound on Δ·y, per contender (before η and y's rounding)."""

    exact_and_read: torch.Tensor  # Δ·b + Σ_e w_e·Σ_i min(M_e,i·a_i⁻, M_e,i·a_i⁺) (projection errors subtracted)
    unread: torch.Tensor  # Σ_e w_e·min(Σ_k |Δ_k|·ρ_e,k, ‖Δ‖₂·Γ_e)
    absolute: torch.Tensor  # the absolute values behind both, for the float64 slack
    neurons: int  # Σ_e I_e (the dot products' length)


def mixture_bound(delta: torch.Tensor, mixture: ExpertMixture, projection: Projection = exact_projection) -> MixtureBound:
    """The mixture's terms of the decomposed bound (module docstring) for contenders `delta` [J, H] (float64)."""
    absolute_delta = delta.abs()
    delta_norm = next_up(torch.linalg.vector_norm(delta, dim=1) * (1.0 + 2.0**-40))
    read_part = delta @ mixture.base
    unread = torch.zeros_like(read_part)
    absolute = absolute_delta @ mixture.base.abs()
    neurons = 0
    for term in mixture.terms:
        a = term.activation
        row_form = absolute_delta @ term.row_remainder
        missing = row_form if term.column_remainder is None else torch.minimum(row_form, delta_norm * term.column_remainder)
        if a.lower.numel():
            mixed, error = projection(delta, term)  # M_e = K_eᵀ·Δ for every contender
            # Σ_i min(M_i·a_i⁻, M_i·a_i⁺) ≥ M·c − |M|·ρ for any c, ρ with [a⁻, a⁺] ⊆ c ± ρ.
            centre = a.center
            part = mixed @ centre - mixed.abs() @ a.radius_about(centre)
            absolute = absolute + term.weight * (mixed.abs() @ (centre.abs() + a.magnitude) + absolute_delta @ (term.known.abs() @ a.magnitude))
            if error is not None:  # |(M̂ − M)·a| ≤ error·|a|
                if isinstance(error, RankOneError):
                    projection_error = error.rows * (error.columns @ a.magnitude)
                else:
                    projection_error = error @ a.magnitude
                part = part - projection_error
                absolute = absolute + term.weight * projection_error
            read_part = read_part + term.weight * part
        absolute = absolute + term.weight * missing
        unread = unread + term.weight * missing
        neurons += a.lower.numel()
    return MixtureBound(read_part, unread, absolute, neurons)


@dataclass(frozen=True)
class PairwiseResult:
    certified: bool
    winner: int
    contenders: torch.Tensor  # rows j checked against the winner
    margin: torch.Tensor  # q⁻·F − A − 2U per contender (> 0: certified pair)
    box_bound: torch.Tensor  # L_box per contender
    decomposed_bound: torch.Tensor | None  # L_decomposed per contender
    terms: dict[str, float]  # the bound's terms for the tightest pair (provenance of the uncertainty)


def pairwise_certificate(
    winner: int,
    rows: torch.Tensor,
    lm_weight: torch.Tensor,
    logit_lower: torch.Tensor,
    logit_upper: torch.Tensor,
    norm_weight: torch.Tensor,
    y: Enclosure,
    scale_lower: float,
    lm_gamma: float,
    assumptions: RoundingAssumptions,
    down: DownProjection | None = None,
    mixture: ExpertMixture | None = None,
    projection: Projection = exact_projection,
) -> PairwiseResult:
    """Certify that `winner` is the reference argmax against every other row of `rows`.

    `lm_weight` holds the exact LM-head rows of `rows` (original dtype, aligned with
    `rows`); `logit_lower/upper` are those rows' certified logit intervals, aligned with
    `rows` (they bound the logits' magnitude); `lm_gamma` is the LM head's accumulation γ;
    `scale_lower` > 0 bounds the norm's q from below. `down` (Phase 2) or `mixture` (Phase
    5A) adds the decomposed bound; `projection` computes the mixture's M_e (exactly in float64
    by default, or with an error bound it returns).
    """
    if not scale_lower > 0.0:
        raise ValueError("the norm's scale must be bounded away from zero")
    if down is not None and mixture is not None:
        raise ValueError("one decomposition of y at a time")
    position = (rows == winner).nonzero()
    if position.numel() != 1:
        raise ValueError("the winner must be one of the rows")
    others = rows != winner
    contenders_ = rows[others]
    w64 = lm_weight[position[0, 0]].to(torch.float64)
    j64 = lm_weight[others].to(torch.float64)
    gain = norm_weight.to(torch.float64)
    difference = w64[None, :] - j64  # W_w − W_j
    delta = difference * gain[None, :]
    absolute_delta = delta.abs()
    magnitude = y.magnitude
    elementwise, gemm = assumptions.elementwise, assumptions.gemm

    # ξ: n = rnd16(rnd32(y·q)), then h = rnd16(rnd32(g·n)); μ: their absolute floors (or flushing).
    r32, r16 = relative_rounding_error(torch.float32, elementwise), relative_rounding_error(norm_weight.dtype, elementwise)
    xi = (1.0 + r32) ** 2 * (1.0 + r16) ** 2 - 1.0
    floor = FP32_NORMAL_MIN + subnormal_floor(norm_weight.dtype)
    mu = (gain.abs() * (1.0 + r32) + 1.0) * floor * (1.0 + r16) ** 2

    box = torch.minimum(delta * y.lower[None, :], delta * y.upper[None, :]).sum(dim=1)
    absolute_terms = absolute_delta @ (magnitude + y.lower.abs() + y.upper.abs())
    decomposed = None
    terms: dict[str, torch.Tensor] = {}
    if down is not None:
        mixed = delta @ down.read_weight  # M = W_dᵀ·Δ for every contender
        a = down.activation
        read_part = torch.where(down.read[None, :], torch.minimum(mixed * a.lower[None, :], mixed * a.upper[None, :]), 0.0)
        read_part = read_part.sum(dim=1)
        unread = torch.where(down.read, 0.0, down.column_norms * down.unread_activation).sum()
        delta_norm = next_up(torch.linalg.vector_norm(delta, dim=1) * (1.0 + 2.0**-40))
        rounding_o = rounding_error_upper(down.output.magnitude, norm_weight.dtype, gemm)
        rounding_y = rounding_error_upper(magnitude, norm_weight.dtype, elementwise)
        a_magnitude = torch.where(down.read, a.magnitude, 0.0)
        terms = {
            "residual_and_read_neurons": delta @ down.residual + read_part,
            "unread_neurons": delta_norm * unread,
            "down_accumulation": absolute_delta @ down.accumulation_error,
            "o_rounding": absolute_delta @ rounding_o,
            "y_rounding": absolute_delta @ rounding_y,
        }
        decomposed = terms["residual_and_read_neurons"] - terms["unread_neurons"] - terms["down_accumulation"]
        decomposed = decomposed - terms["o_rounding"] - terms["y_rounding"]
        absolute_terms = absolute_terms + absolute_delta @ down.residual.abs() + mixed.abs() @ a_magnitude
        absolute_terms = absolute_terms + absolute_delta @ (down.read_weight.abs() @ a_magnitude)
        absolute_terms = absolute_terms + terms["unread_neurons"] + absolute_delta @ (down.accumulation_error + rounding_o + rounding_y)
    neurons = 0
    if mixture is not None:
        part = mixture_bound(delta, mixture, projection)
        neurons = part.neurons
        terms = {"base_and_read_neurons": part.exact_and_read, "unread_neurons": part.unread}
        for name, vector in mixture.errors.items():
            terms[name] = absolute_delta @ vector
        terms["y_rounding"] = absolute_delta @ rounding_error_upper(magnitude, norm_weight.dtype, elementwise)
        decomposed = terms["base_and_read_neurons"] - terms["unread_neurons"]
        for name in (*mixture.errors, "y_rounding"):
            decomposed = decomposed - terms[name]
        absolute_terms = absolute_terms + part.absolute + sum(terms[name] for name in (*mixture.errors, "y_rounding"))
    lower_bound = box if decomposed is None else torch.maximum(box, decomposed)

    lm_mass = (w64.abs()[None, :] + j64.abs()) * gain.abs()[None, :]
    terms["norm_rounding"] = xi * (absolute_delta @ magnitude)
    terms["lm_accumulation"] = lm_gamma * (1.0 + xi) * (lm_mass @ magnitude)
    bound = lower_bound - terms["norm_rounding"] - terms["lm_accumulation"]
    absolute_terms = absolute_terms + terms["norm_rounding"] + terms["lm_accumulation"]
    # Float64 rounding of every sum and product above (at most 2(H + I) + 64 terms per dot product).
    length = 2 * (delta.shape[1] + (0 if down is None else down.read.numel()) + neurons) + 64
    slack = 4.0 * gamma(length, FLOAT64_UNIT_ROUNDOFF) * absolute_terms
    absolute = difference.abs() @ mu + lm_gamma * ((w64.abs()[None, :] + j64.abs()) @ mu)

    logit_magnitude = torch.maximum(logit_lower.abs(), logit_upper.abs())
    spacing = spacing_upper(torch.maximum(logit_magnitude[others], logit_magnitude[position[0, 0]]), lm_weight.dtype)
    separations = 2.0 if gemm is RoundingModel.FAITHFUL else 1.0
    # q ≥ q⁻ > 0 multiplies a positive F − slack; a non-positive one can never certify.
    margin = (bound - slack) * scale_lower * (1.0 - 2.0**-50)
    margin = margin - next_up(absolute + separations * spacing)
    certified = bool((margin > 0).all()) if contenders_.numel() else True
    tightest = int(margin.argmin()) if contenders_.numel() else None
    summary = {}
    if tightest is not None:
        summary = {name: float(value[tightest]) for name, value in terms.items()}
        summary.update(box=float(box[tightest]), slack=float(slack[tightest]), margin=float(margin[tightest]))
        if decomposed is not None:
            summary["decomposed"] = float(decomposed[tightest])
    return PairwiseResult(certified, winner, contenders_, margin, box, decomposed, summary)
