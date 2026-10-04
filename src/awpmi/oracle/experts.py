"""Phase 5A diagnostic oracle: AWPMI inside the routed experts of one MoE layer (decision 0009).

Nothing here is a runtime path. For one decode token whose upstream computation is exact (Mode A: everything before the
target layer's experts call, its router and its shared experts are the reference's), the oracle holds every weight of
the routed experts and simulates what a runtime would have read: it materializes them progressively in one of several
decompositions, propagates enclosures through the experts call, the MoE block, the residual, the final norm and the LM
head with the reference's own operations (`awpmi.bounds.operators`), and asks the pairwise certificate
(`awpmi.bounds.pairwise` with an `ExpertMixture`) whether the reference token is decided. Bytes are those of the units
read; no storage is involved.

The reference (transformers 5.18, `grouped_mm_experts_forward`, BF16, one decode token, top-k experts e):

    g_e, u_e = rnd(G_e·x), rnd(U_e·x)          one grouped GEMM over gate_up_proj, float32 accumulation
    s_e = rnd(silu(g_e)),  a_e = rnd(s_e·u_e)   elementwise kernels (float32 opmath)
    o_e = rnd(D_e·a_e)                           one grouped GEMM over down_proj
    z_e = fl32(o_e·w_e)                          BF16 output × float32 routing weight → float32
    R = rnd(Σ_e z_e)                             float32 sum over the top-k, then BF16 (the call's output)
    m = rnd(R + S)    y = rnd(r + m)             the shared experts' addition, the residual
    h = RMSNorm(y)    ℓ = rnd(W·h)               final norm and LM head; the token is argmax ℓ, lowest index on ties

Arithmetic tiers (how the reference's floating point is modelled):
  certified       faithful rounding, u = 2⁻²² (decisions 0001 and 0005): the only tier that may set `certified`
  certified_u24   EXPERIMENTAL / NOT CERTIFIED (ceilings only): faithful, an IEEE binary32 accumulator (u = 2⁻²⁴)
  rn_elementwise  EXPERIMENTAL / NOT CERTIFIED (ceilings only): round-to-nearest-even in elementwise kernels only
  rn_even         EXPERIMENTAL / NOT CERTIFIED: round-to-nearest-even everywhere (decision 0005's what-if)
  real            DIAGNOSTIC / NOT CERTIFIED: exact real arithmetic from the experts to y; the decision is the sign of
                  Δ·y (the norm's scale is common to every logit). It measures what a decomposition needs once the
                  reference's own rounding is no obstacle; its winner is the real computation's.
Bound tiers (what bounds the parts not read):
  realistic   resident metadata only (row and column norms, a level's remainder norms): Cauchy–Schwarz and Hölder
  ideal       DIAGNOSTIC: a missing part's true contribution |X·v| at the reference's input v
Orderings (in which order units are read; both fixed once per cell, from its first state):
  realistic   the largest bound contribution per byte to the tightest pair, from what has been read
  ideal       DIAGNOSTIC: the largest true contribution per byte to the reference's top-2 logit difference

Strategies (`strategies`): neuron pages with the whole down projection (A), neuron pages and down output-row pages (B,
several page sizes), neuron-major pages (C: a neuron's gate row, up row and down column), per-row precision
refinement of all three matrices (D: a coarse level of every row, then row by row up to the original BF16 rows).

A cell (a strategy under an arithmetic tier, a bound tier and an ordering) reads its units in order and looks for the
smallest byte budget (a grid of fractions of the routed BF16 bytes) at which the certificate holds. With realistic
bounds, more knowledge only narrows every enclosure, so certification is monotone in the budget and a binary search
finds the first budget exactly; with ideal bounds the search returns a budget where the certificate starts to hold.

Bytes: a unit's bytes are what the checkpoint (or a level) stores for it; the routed experts' metadata is charged to
every token; a fallback reads every byte not read yet in BF16 (decision 0003).
"""

from __future__ import annotations

import enum
import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from awpmi.bounds.enclosure import Enclosure
from awpmi.bounds.floating import FLOAT64_UNIT_ROUNDOFF, gamma, next_up, round_up_to_grid
from awpmi.bounds.linear import matvec
from awpmi.bounds.operators import FP32_NORMAL_MIN, RMSNormBounds, linear, multiply, reduce_sum, residual_add, rms_norm, silu
from awpmi.bounds.pairwise import (
    ExpertMixture,
    MixtureTerm,
    RankOneError,
    exact_projection,
    mixture_bound,
    pairwise_certificate,
    relax_mixture,
)
from awpmi.bounds.remainder import RowNormBounds, row_norm_bounds
from awpmi.bounds.residual import ReferenceNumerics
from awpmi.bounds.rounding import CERTIFIED, EXPERIMENTAL, EXPERIMENTAL_ELEMENTWISE, RoundingAssumptions, rounding_error_upper
from awpmi.decomposition import RefinementDecomposition
from awpmi.decomposition.accounting import IO_BLOCK_BYTES

UNKNOWN = -1  # a row's state: nothing read (its metadata only)
EXACT = 99  # the original row
METADATA_BYTES = 4  # float32
MATRICES = ("gate", "up", "down")


# Tiers


@dataclass(frozen=True)
class Arithmetic:
    name: str
    assumptions: RoundingAssumptions | None  # None: exact real arithmetic (diagnostic)
    accumulation_unit_roundoff: float
    certifies: bool

    @property
    def real(self) -> bool:
        return self.assumptions is None


CERTIFIED_ARITHMETIC = Arithmetic("certified", CERTIFIED, 2.0**-22, True)
RN_EVEN_ARITHMETIC = Arithmetic("rn_even", EXPERIMENTAL, 2.0**-22, False)
REAL_ARITHMETIC = Arithmetic("real", None, 0.0, False)
# Ceiling-only what-ifs (EXPERIMENTAL / NOT CERTIFIED), to attribute the floor: an IEEE binary32 accumulator (u = 2⁻²⁴
# instead of decision 0001's 2⁻²²), and round-to-nearest-even in the elementwise kernels only (Phase 2's open question).
U24_ARITHMETIC = Arithmetic("certified_u24", CERTIFIED, 2.0**-24, False)
RN_ELEMENTWISE_ARITHMETIC = Arithmetic("rn_elementwise", EXPERIMENTAL_ELEMENTWISE, 2.0**-22, False)
ARITHMETIC = {
    tier.name: tier
    for tier in (CERTIFIED_ARITHMETIC, U24_ARITHMETIC, RN_ELEMENTWISE_ARITHMETIC, RN_EVEN_ARITHMETIC, REAL_ARITHMETIC)
}


class BoundTier(enum.Enum):
    REALISTIC = "realistic"
    IDEAL = "ideal"


class Ordering(enum.Enum):
    REALISTIC = "realistic"
    IDEAL = "ideal"


# The reference


def experts_call_reference(gate_up, down, act_fn, hidden, top_k_index, top_k_weights) -> dict[str, torch.Tensor]:
    """transformers' `grouped_mm_experts_forward` (integrations/moe.py, 5.18) step by step, with its own functions.

    `gate_up` [E, 2I, H] and `down` [E, H, I] are the layer's whole parameters, so the call has the reference's shapes
    (offsets over all E experts). Returns, per (token, k) assignment in top-k order, g, u, s, a [T·K, I] and o [T·K, H]
    (BF16) and z [T·K, H] (float32), and the call's output [T, H] (BF16).
    """
    from transformers.integrations.moe import _grouped_linear

    num_top_k, num_tokens, hidden_dim = top_k_index.size(-1), hidden.size(0), hidden.size(-1)
    expert_ids = top_k_index.reshape(-1)
    expert_ids_g, perm = torch.sort(expert_ids)
    selected = hidden[perm // num_top_k]
    sample_weights_g = top_k_weights.reshape(-1)[perm]
    histc_input = expert_ids_g.float() if hidden.device.type in ("cpu", "mps") else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=gate_up.shape[0], min=0, max=gate_up.shape[0] - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
    projected = _grouped_linear(selected, gate_up, offsets, is_transposed=False)
    gate, up = projected.chunk(2, dim=-1)
    activation_function = act_fn(gate)
    activation = activation_function * up
    output = _grouped_linear(activation, down, offsets, is_transposed=False)
    weighted = output * sample_weights_g.unsqueeze(-1)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.size(0), device=perm.device)
    weighted = weighted[inverse]
    final = weighted.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden.dtype)
    return {
        "g": gate[inverse], "u": up[inverse], "s": activation_function[inverse], "a": activation[inverse],
        "o": output[inverse], "z": weighted, "output": final,
    }


# Weights and metadata


@dataclass(frozen=True)
class ExpertNorms:
    """Resident bound metadata of one expert (float32, rounded up): per row of gate, up and down; per column of down."""

    gate: RowNormBounds
    up: RowNormBounds
    down: RowNormBounds
    down_columns: torch.Tensor  # C_i ≥ ‖D[:, i]‖₂

    @property
    def nbytes(self) -> int:
        return self.gate.nbytes + self.up.nbytes + self.down.nbytes + self.down_columns.numel() * METADATA_BYTES


class ExpertLayer:
    """One MoE layer's routed experts, all held by the oracle; metadata and decompositions are built on demand."""

    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor, act_fn, cache_experts: int = 8) -> None:
        # `cache_experts` bounds the decompositions kept; float64 copies are never kept (a sample's stacks hold them).
        if gate_up.dim() != 3 or down.dim() != 3 or gate_up.shape[0] != down.shape[0] or gate_up.shape[1] != 2 * down.shape[2]:
            raise ValueError("expected gate_up [E, 2I, H] and down [E, H, I]")
        self.gate_up, self.down, self.act_fn = gate_up, down, act_fn
        self.experts, self.hidden, self.intermediate = down.shape[0], down.shape[1], down.shape[2]
        self.cache_experts = cache_experts
        self._norms: dict[int, ExpertNorms] = {}
        self._levels: OrderedDict[tuple[int, str], dict[str, RefinementDecomposition]] = OrderedDict()

    def matrices(self, expert: int) -> dict[str, torch.Tensor]:
        return {"gate": self.gate_up[expert, : self.intermediate], "up": self.gate_up[expert, self.intermediate :], "down": self.down[expert]}

    @property
    def expert_bytes(self) -> int:
        return sum(m.numel() * m.element_size() for m in self.matrices(0).values())

    @property
    def element_size(self) -> int:
        return self.down.element_size()

    def width(self, matrix: str) -> int:
        return self.hidden if matrix in ("gate", "up") else self.intermediate

    def rows(self, matrix: str) -> int:
        return self.intermediate if matrix in ("gate", "up") else self.hidden

    def float64(self, expert: int) -> dict[str, torch.Tensor]:
        return {name: m.to(torch.float64) for name, m in self.matrices(expert).items()}

    def norms(self, expert: int) -> ExpertNorms:
        if expert not in self._norms:
            m = self.float64(expert)
            columns = row_norm_bounds(m["down"].t().contiguous()).l2
            self._norms[expert] = ExpertNorms(row_norm_bounds(m["gate"]), row_norm_bounds(m["up"]), row_norm_bounds(m["down"]), columns)
        return self._norms[expert]

    def levels(self, expert: int, spec: str) -> dict[str, RefinementDecomposition]:
        """Per-row refinement decompositions (decision 0003) of the three matrices: rows are neurons (gate, up), outputs (down)."""
        key = (expert, spec)
        if key in self._levels:
            self._levels.move_to_end(key)
        else:
            self._levels[key] = {name: RefinementDecomposition.build(m, spec) for name, m in self.matrices(expert).items()}
            while len(self._levels) > self.cache_experts * 2:
                self._levels.popitem(last=False)
        return self._levels[key]


@dataclass(frozen=True)
class DecodeSample:
    """One decode token at the target layer: the oracle's inputs and the reference's values."""

    x: torch.Tensor  # the experts' input, BF16 [H]
    residual: torch.Tensor  # r, BF16 [H]
    shared: torch.Tensor  # S, the shared experts' output, BF16 [H]
    experts: tuple[int, ...]  # the router's choice, top-k order
    weights: torch.Tensor  # its routing weights, float32 [K]
    reference: dict[str, torch.Tensor]  # g, u, s, a, o, z ([K, ·]), R, m, y, n, h, logits: the reference's values
    real: dict[str, torch.Tensor]  # the same forward in exact real arithmetic (float64): the real tier's values
    token: int  # the reference's argmax (lowest index on ties)
    runner_up: int  # the largest other logit (lowest index on ties)


def decode_sample(layer: ExpertLayer, x, residual, shared, experts: Sequence[int], weights, norm: torch.nn.Module, lm_weight) -> DecodeSample:
    """The reference's values of every intermediate, by the reference's own operations and shapes (one decode token)."""
    with torch.inference_mode():
        index = torch.tensor([list(experts)], device=x.device)
        call = experts_call_reference(layer.gate_up, layer.down, layer.act_fn, x.view(1, -1), index, weights.view(1, -1))
        moe = call["output"].view(1, 1, -1) + shared.view(1, 1, -1)  # DeepseekV3MoE: routed + shared, BF16
        y = residual.view(1, 1, -1) + moe  # the decoder layer's residual addition
        hidden = y.to(torch.float32)
        scale = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
        normalized = (hidden * scale).to(y.dtype)  # RMSNorm's n, by its own operations
        h = norm(y)
        logits = torch.nn.functional.linear(h, lm_weight).reshape(-1)  # the LM head on the last position
    reference = {key: call[key] for key in ("g", "u", "s", "a", "o", "z")}
    reference.update(R=call["output"][0], m=moe.reshape(-1), y=y.reshape(-1), n=normalized.reshape(-1), h=h.reshape(-1), logits=logits)
    token = int(torch.argmax(logits))
    others = logits.clone()
    others[token] = -math.inf
    real = _real_forward(layer, x, residual, shared, experts, weights)
    return DecodeSample(x, residual, shared, tuple(int(e) for e in experts), weights, reference, real, token, int(torch.argmax(others)))


def _real_forward(layer: ExpertLayer, x, residual, shared, experts, weights) -> dict[str, torch.Tensor]:
    """The forward in exact real arithmetic (float64, every weight): the real tier's values."""
    values: dict[str, list] = {name: [] for name in ("g", "u", "s", "a", "o", "z")}
    x64 = x.to(torch.float64)
    y = residual.to(torch.float64) + shared.to(torch.float64)
    for slot, expert in enumerate(experts):
        m = layer.float64(int(expert))
        g, u = m["gate"] @ x64, m["up"] @ x64
        s = g / (1.0 + torch.exp(-g))
        a = s * u
        o = m["down"] @ a
        z = o * float(weights[slot])
        for name, value in zip(("g", "u", "s", "a", "o", "z"), (g, u, s, a, o, z)):
            values[name].append(value)
        y = y + z
    result = {name: torch.stack(v) for name, v in values.items()}
    result["y"] = y
    return result


# The routed experts of one sample


class SampleWeights:
    """The routed experts of one sample, stacked in top-k order (float64): exact matrices, metadata, and the levels of
    one decomposition at a time (`levels`)."""

    def __init__(self, layer: ExpertLayer, experts: Sequence[int]) -> None:
        self.layer, self.experts = layer, tuple(experts)
        self.truth = {name: torch.stack([layer.float64(e)[name] for e in experts]) for name in MATRICES}
        norms = [layer.norms(e) for e in experts]
        self.norms = {
            name: RowNormBounds(torch.stack([getattr(n, name).l2 for n in norms]).to(torch.float64), torch.stack([getattr(n, name).linf for n in norms]).to(torch.float64))
            for name in MATRICES
        }
        self.column_norms = torch.stack([n.down_columns for n in norms]).to(torch.float64)
        self.metadata_bytes = sum(n.nbytes for n in norms)
        self._spec: str | None = None
        self._levels: dict[str, tuple[list[torch.Tensor], list[RowNormBounds]]] = {}

    def levels(self, spec: str) -> dict[str, tuple[list[torch.Tensor], list[RowNormBounds]]]:
        """Per matrix: the cumulative approximation after each level ([K, rows, cols] float64) and its remainder's norms."""
        if spec != self._spec:
            self._levels = {}
            for name in MATRICES:
                decompositions = [self.layer.levels(e, spec)[name] for e in self.experts]
                count = decompositions[0].exact_state
                approximations = [torch.stack([_cumulative_level(d, level) for d in decompositions]) for level in range(count)]
                remainders = [
                    RowNormBounds(torch.stack([d.remainder_norms[level].l2 for d in decompositions]).to(torch.float64),
                                  torch.stack([d.remainder_norms[level].linf for d in decompositions]).to(torch.float64))
                    for level in range(count)
                ]
                self._levels[name] = (approximations, remainders)
            self._spec = spec
        return self._levels


# What is known of the routed experts


@dataclass
class ExpertStates:
    """What has been read of the routed experts: a state per row (UNKNOWN, a level index, EXACT) of each matrix,
    [K, rows] (top-k order), and the down columns known exactly (neuron-major pages)."""

    gate: torch.Tensor  # int64 [K, I]
    up: torch.Tensor  # int64 [K, I]
    down: torch.Tensor  # int64 [K, H] (output rows)
    columns: torch.Tensor  # bool [K, I]

    @classmethod
    def filled(cls, layer: ExpertLayer, slots: int, device, value: int = UNKNOWN, columns: bool = False) -> ExpertStates:
        def rows(n):
            return torch.full((slots, n), value, dtype=torch.int64, device=device)

        return cls(rows(layer.intermediate), rows(layer.intermediate), rows(layer.hidden), torch.full((slots, layer.intermediate), columns, device=device))

    def copy(self) -> ExpertStates:
        return ExpertStates(self.gate.clone(), self.up.clone(), self.down.clone(), self.columns.clone())

    def matrix(self, name: str) -> torch.Tensor:
        return {"gate": self.gate, "up": self.up, "down": self.down}[name]


@dataclass(frozen=True)
class MatrixKnowledge:
    """Known values of one matrix of each routed expert (exact, or a level's approximation; 0 where nothing is known)
    and the norms of the unknown part, batched over the experts."""

    known: torch.Tensor  # float64 [K, rows, cols]
    remainder: RowNormBounds  # float64 [K, rows]: ≥ the norms of the unknown part X (restricted to `support`)
    support: torch.Tensor | None  # bool [K, cols]: the columns where X may be non-zero (None: all)
    truth: torch.Tensor  # the exact matrices (float64): for the ideal tier and the orderings only, never a realistic bound


def remainder_mass(knowledge: MatrixKnowledge, magnitude: torch.Tensor, bound: BoundTier, reference_input: torch.Tensor, gamma_acc: float) -> torch.Tensor:
    """Realistic: ≥ Σ_c |X_rc|·|v_c| for every input |v| ≤ `magnitude` ([K, cols] or [cols]). Ideal: |X_r·v_ref| plus its
    accumulation share. [K, rows]."""
    if bound is BoundTier.IDEAL:
        remainder = knowledge.truth - knowledge.known
        reference = reference_input.to(torch.float64)
        return next_up(matvec(remainder, reference).abs() + gamma_acc * matvec(remainder.abs(), reference.abs()))
    v = magnitude.expand(knowledge.known.shape[0], -1) if magnitude.dim() == 1 else magnitude
    if knowledge.support is not None:
        v = torch.where(knowledge.support, v, 0.0)
    l2 = next_up(torch.linalg.vector_norm(v, dim=1) * (1.0 + 2.0**-40))
    l1 = next_up(v.sum(dim=1) * (1.0 + gamma(v.shape[1] + 1, FLOAT64_UNIT_ROUNDOFF)))
    return next_up(torch.minimum(knowledge.remainder.l2 * l2[:, None], knowledge.remainder.linf * l1[:, None]))


# Propagation


@dataclass(frozen=True)
class SampleBounds:
    gate: MatrixKnowledge
    up: MatrixKnowledge
    down: MatrixKnowledge
    g: Enclosure  # [K, I]
    u: Enclosure
    s: Enclosure
    a: Enclosure
    o: Enclosure  # [K, H]
    z: Enclosure
    down_mass: torch.Tensor  # [K, H]: the unknown part of down's share of o (≥ |X·a|)
    R: Enclosure
    m: Enclosure
    y: Enclosure
    norm: RMSNormBounds | None  # None in the real tier
    mixture: ExpertMixture


def propagate(
    sample: DecodeSample,
    knowledge: tuple[MatrixKnowledge, MatrixKnowledge, MatrixKnowledge],
    arithmetic: Arithmetic,
    bound: BoundTier,
    norm_weight: torch.Tensor,
    norm_eps: float,
    column_norms: torch.Tensor,
) -> SampleBounds:
    """Enclosures of every reference intermediate from what is known of the routed experts (top-k order, batched)."""
    real = arithmetic.real
    assumptions = CERTIFIED if real else arithmetic.assumptions  # the real tier's float64 grid makes every rounding exact
    gemm, elementwise = assumptions.gemm, assumptions.elementwise
    dtype = torch.float64 if real else sample.x.dtype
    single = torch.float64 if real else torch.float32
    unit = arithmetic.accumulation_unit_roundoff
    values = sample.real if real else sample.reference
    gate, up, down = knowledge
    hidden, slots = sample.x.numel(), down.known.shape[0]
    numerics_in = ReferenceNumerics(dtype, unit, hidden)
    numerics_down = ReferenceNumerics(dtype, unit, down.known.shape[2])
    x = Enclosure.exact(sample.x)
    g = linear(x, gate.known, numerics_in, gemm, remainder_mass(gate, x.magnitude, bound, sample.x, numerics_in.accumulation_gamma), label="gate")
    u = linear(x, up.known, numerics_in, gemm, remainder_mass(up, x.magnitude, bound, sample.x, numerics_in.accumulation_gamma), label="up")
    s = silu(g, dtype, elementwise)
    a = multiply(s, u, dtype, elementwise, label="activation")
    mass = remainder_mass(down, a.magnitude, bound, values["a"], numerics_down.accumulation_gamma)
    o = linear(a, down.known, numerics_down, gemm, mass, label="down")
    weights = sample.weights.to(torch.float64)
    z = multiply(o, Enclosure.exact(weights[:, None].expand(slots, hidden)), single, elementwise, label="weighted")
    R = reduce_sum(z, unit, dtype, elementwise, label="combine")
    m = residual_add(R, Enclosure.exact(sample.shared), dtype, elementwise, label="moe")
    y = residual_add(Enclosure.exact(sample.residual), m, dtype, elementwise, label="residual")
    base = sample.residual.to(torch.float64) + sample.shared.to(torch.float64)
    # r + S in float64 is exact unless their exponents are far apart: its rounding, at most 2⁻⁵³·|r + S|, is an error on y.
    errors: dict[str, torch.Tensor] = {"base_rounding": next_up(base.abs() * 2.0**-53 + 2.0**-1074)}
    norm = None
    if not real:
        # y = r + S + Σ_e w_e·D_e·a_e + η: everything in η but y's own rounding (the pairwise certificate adds that).
        accumulation = numerics_down.accumulation_gamma * (matvec(down.known.abs(), a.magnitude) + mass)
        errors["down_accumulation"] = (weights[:, None] * accumulation).sum(dim=0)
        errors["o_rounding"] = (weights[:, None] * rounding_error_upper(o.magnitude, sample.x.dtype, gemm)).sum(dim=0)
        errors["z_rounding"] = (rounding_error_upper(z.magnitude, torch.float32, elementwise) + FP32_NORMAL_MIN).sum(dim=0)
        errors["combine"] = z.magnitude.sum(dim=0) * gamma(slots + 2, unit)
        errors["R_rounding"] = rounding_error_upper(R.magnitude, sample.x.dtype, elementwise)
        errors["m_rounding"] = rounding_error_upper(m.magnitude, sample.x.dtype, elementwise)
        errors = {name: next_up(vector * (1.0 + 2.0**-30)) for name, vector in errors.items()}
        norm = rms_norm(y, norm_weight, norm_eps, elementwise, unit)
    terms = []
    for slot in range(slots):
        column = None
        if down.support is not None and bound is BoundTier.REALISTIC:  # column form: Γ = Σ_unread C_i·|a_i|⁺
            unread = torch.where(down.support[slot], a.magnitude[slot], 0.0)
            column = float(next_up(column_norms[slot] @ unread * (1.0 + 2.0**-40)))
        activation = Enclosure(a.lower[slot], a.upper[slot], a.provenance)
        terms.append(MixtureTerm(float(sample.weights[slot]), down.known[slot], activation, mass[slot], column))
    mixture = ExpertMixture(base, tuple(terms), errors)
    return SampleBounds(gate, up, down, g, u, s, a, o, z, mass, R, m, y, norm, mixture)


def enclosure_violations(bounds: SampleBounds, sample: DecodeSample, real: bool) -> dict[str, int]:
    """Values outside their enclosures, per intermediate: the reference's (or, real tier, the real forward's). Must be 0."""
    values = sample.real if real else sample.reference
    counts = {name: getattr(bounds, name).violations(values[name]) for name in ("g", "u", "s", "a", "o", "z", "y")}
    if not real:
        for name in ("R", "m"):
            counts[name] = getattr(bounds, name).violations(values[name])
        counts["n"] = bounds.norm.normalized.violations(values["n"])
        counts["h"] = bounds.norm.output.violations(values["h"])
    return counts


# The certificate


@dataclass
class Outcome:
    certified: bool
    candidate: int
    near_failed: bool  # a pair among the nearest rows failed (no full check)
    unsettled: int  # rows not certified against the candidate yet
    margin: float  # the smallest margin among the pairs checked at this evaluation (> 0: certified pairs)
    tightest: int  # the contender with that margin (−1: none checked)
    terms: dict[str, float]  # the tightest pair's terms (pairwise certificate)
    full_checks: int


class Float32Projection:
    """M = Δ·K in binary32, with a rank-one bound on its error: (γ_{H+2}(2⁻²²) + 2⁻²²)·‖Δ_j‖₂·‖K[:, i]‖₂.

    Both operands are rounded to binary32 (each within 2⁻²⁴ relative, so every product within 2⁻²² of the exact one);
    the binary32 sum of H products errs by at most γ_{H+2}(2⁻²²) of Σ_k |Δ_k|·|K_ki| ≤ ‖Δ‖₂·‖K[:, i]‖₂ (decision
    0001's accumulation model, any order). The binary32 copy of K and its column norms are kept for the next blocks.
    """

    def __init__(self) -> None:
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def __call__(self, delta: torch.Tensor, term: MixtureTerm) -> tuple[torch.Tensor, RankOneError]:
        key = id(term.known)
        if key not in self._cache:
            columns = next_up(torch.linalg.vector_norm(term.known, dim=0) * (1.0 + 2.0**-40))
            self._cache[key] = (term.known.to(torch.float32), columns)
        known32, columns = self._cache[key]
        mixed = (delta.to(torch.float32) @ known32).to(torch.float64)
        coefficient = gamma(delta.shape[1] + 2, 2.0**-22) + 2.0**-22
        rows = next_up(torch.linalg.vector_norm(delta, dim=1) * (coefficient * (1.0 + 2.0**-40)))
        return mixed, RankOneError(rows, columns)


def first_argmax(values: torch.Tensor) -> int:
    best = values.max()
    return int((values == best).nonzero()[0, 0])


class Certifier:
    """The pairwise certificate of one sample against every row of the vocabulary.

    The candidate is the row with the largest centre logit (lowest index on ties), as a runtime would choose it. The
    nearest rows (by the reference's logits: an efficiency filter, never part of a proof) are checked first, in
    float64; all others only when they all hold, with M in binary32 and its error bound. In the real tier, the full check
    goes in stages of decreasing relaxation (`relax_mixture`): a screen with no column in exact form, then the most
    uncertain columns, then every column. Each stage is a valid certificate; a row certified by one is not checked again
    within the evaluation.
    """

    partial_columns = 192

    def __init__(self, lm_weight: torch.Tensor, norm_weight: torch.Tensor, reference_logits: torch.Tensor, arithmetic: Arithmetic,
                 near: int = 64, chunk_rows: int = 4096) -> None:
        self.lm_weight, self.norm_weight, self.arithmetic = lm_weight, norm_weight, arithmetic
        self.order = torch.argsort(reference_logits.to(torch.float32), descending=True, stable=True)
        self.near, self.chunk_rows = near, chunk_rows
        self.vocabulary, self.hidden = lm_weight.shape
        unit = arithmetic.accumulation_unit_roundoff
        self.lm_gamma = 0.0 if arithmetic.real else ReferenceNumerics(lm_weight.dtype, unit, self.hidden).accumulation_gamma
        self.candidate = -1

    def centres(self, bounds: SampleBounds) -> torch.Tensor:
        """Centre logits (binary32): what a runtime ranks its candidate by. The real tier ranks W·(g⊙y) (q is common)."""
        vector = bounds.y.center * self.norm_weight.to(torch.float64) if bounds.norm is None else bounds.norm.output.center
        vector = vector.to(torch.float32)
        blocks = 4 * self.chunk_rows
        return torch.cat([self.lm_weight[start : start + blocks].to(torch.float32) @ vector for start in range(0, self.vocabulary, blocks)])

    def _magnitudes(self, bounds: SampleBounds, rows: torch.Tensor) -> torch.Tensor:
        """≥ |ℓ_j| for `rows`: |W_j|·|h|⁺ in binary32 (non-negative terms, inflated), its accumulation, up to the grid."""
        magnitude = round_up_to_grid(bounds.norm.output.magnitude, torch.float32).to(torch.float32)
        total = (self.lm_weight.index_select(0, rows).to(torch.float32).abs() @ magnitude).to(torch.float64)
        inflation = (1.0 + 2.0 * gamma(self.hidden + 2, 2.0**-22)) * (1.0 + 2.0 * self.lm_gamma)
        return round_up_to_grid(next_up(total * inflation), self.lm_weight.dtype)

    def margins(self, bounds: SampleBounds, mixture: ExpertMixture, rows: torch.Tensor, projection) -> tuple[torch.Tensor, dict[str, float]]:
        """Each row's margin against the candidate (> 0: a certified pair), in chunks; and the tightest pair's terms."""
        margins, terms = [], {}
        candidate = torch.tensor([self.candidate], device=rows.device)
        for start in range(0, rows.numel(), self.chunk_rows):
            part = rows[start : start + self.chunk_rows]
            if self.arithmetic.real:
                margin = self._real_margins(bounds, mixture, part, projection)
            else:
                block = torch.cat([candidate, part])
                magnitudes = self._magnitudes(bounds, block)
                result = pairwise_certificate(
                    self.candidate, block, self.lm_weight.index_select(0, block), -magnitudes, magnitudes, self.norm_weight,
                    bounds.y, bounds.norm.scale_lower, self.lm_gamma, self.arithmetic.assumptions, mixture=mixture,
                    projection=projection,
                )
                margin = result.margin  # in `part`'s order: the candidate is `block`'s first row
                if margin.numel() and float(margin.min()) < terms.get("margin", math.inf):
                    terms = dict(result.terms)
            margins.append(margin)
        return torch.cat(margins), terms

    def _real_margins(self, bounds: SampleBounds, mixture: ExpertMixture, rows: torch.Tensor, projection) -> torch.Tensor:
        """Real tier: a lower bound on Δ·y (Δ = (W_w − W_j)⊙g; q > 0 is common) less its float64 slack; > 0 decides the pair."""
        gain = self.norm_weight.to(torch.float64)
        winner = self.lm_weight[self.candidate].to(torch.float64)
        delta = (winner[None, :] - self.lm_weight.index_select(0, rows).to(torch.float64)) * gain[None, :]
        y = bounds.y
        centre = y.center
        box = delta @ centre - delta.abs() @ y.radius_about(centre)
        part = mixture_bound(delta, mixture, projection)
        decomposed = part.exact_and_read - part.unread
        if mixture.errors:
            decomposed = decomposed - delta.abs() @ sum(mixture.errors.values())
        absolute = delta.abs() @ (y.magnitude + centre.abs()) + part.absolute
        slack = 4.0 * gamma(2 * (delta.shape[1] + part.neurons) + 64, FLOAT64_UNIT_ROUNDOFF) * absolute
        return torch.maximum(box, decomposed) - slack

    def _stages(self, bounds: SampleBounds) -> list[tuple[ExpertMixture, object]]:
        """The real tier's enclosures are often tight enough for the screen; under rounding they never are (measured on
        Moonlight: nothing settled before the full stage), so the other tiers go straight to it."""
        projection = Float32Projection()
        if not self.arithmetic.real:
            return [(bounds.mixture, projection)]
        terms = bounds.mixture.terms
        screen = relax_mixture(bounds.mixture, [torch.zeros(t.activation.lower.numel(), dtype=torch.bool, device=t.known.device) for t in terms])
        keep = []
        for term in terms:
            spread = term.known.abs().sum(dim=0) * term.activation.radius_about(term.activation.center)
            chosen = torch.zeros_like(spread, dtype=torch.bool)
            chosen[torch.argsort(spread, descending=True, stable=True)[: self.partial_columns]] = True
            keep.append(chosen)
        return [(screen, exact_projection), (relax_mixture(bounds.mixture, keep), projection), (bounds.mixture, projection)]

    def evaluate(self, bounds: SampleBounds, full: bool = True) -> Outcome:
        """One certificate on fresh knowledge. `full=False` stops after the nearest rows (a necessary condition only)."""
        self.candidate = candidate = first_argmax(self.centres(bounds))
        settled = torch.zeros(self.vocabulary, dtype=torch.bool, device=self.lm_weight.device)
        settled[candidate] = True
        near = self.order[: self.near + 1]
        near = near[near != candidate]
        margins, terms = self.margins(bounds, bounds.mixture, near, exact_projection)
        settled[near[margins > 0]] = True
        position = int(margins.argmin())
        margin, tightest, checks = float(margins[position]), int(near[position]), 0
        if bool((margins <= 0).any()) or not full:
            unsettled = int((~settled).sum())
            return Outcome(False, candidate, bool((margins <= 0).any()), unsettled, margin, tightest, terms, checks)
        rows = (~settled).nonzero().squeeze(1)
        failing = torch.zeros(0, dtype=torch.bool, device=rows.device)
        if rows.numel():
            checks = 1
            for mixture, projection in self._stages(bounds):
                stage_margins, stage_terms = self.margins(bounds, mixture, rows, projection)
                settled[rows[stage_margins > 0]] = True
                failing = stage_margins <= 0
                if not bool(failing.any()):
                    break
                rows, stage_margins = rows[failing], stage_margins[failing]
                failing = torch.ones_like(rows, dtype=torch.bool)
            if bool(failing.any()):
                position = int(stage_margins.argmin())
                if float(stage_margins[position]) < margin:
                    margin, tightest, terms = float(stage_margins[position]), int(rows[position]), stage_terms
        unsettled = int((~settled).sum())
        return Outcome(unsettled == 0, candidate, False, unsettled, margin, tightest, terms, checks)


# Strategies

KINDS = ("neuron", "neuron_major", "down_rows", "refine_neuron", "refine_down")


@dataclass(frozen=True)
class Units:
    """Materialization units as parallel tensors: kind (index into KINDS), expert slot, index (neuron, page or row), the
    state the unit brings its rows to (`step`, refinements), and bytes."""

    kind: torch.Tensor
    slot: torch.Tensor
    index: torch.Tensor
    step: torch.Tensor
    nbytes: torch.Tensor

    def __len__(self) -> int:
        return int(self.kind.numel())

    @classmethod
    def of(cls, kind: str, slot: torch.Tensor | int, index: torch.Tensor, nbytes: torch.Tensor | int, step: torch.Tensor | int = EXACT) -> Units:
        index = index.to(torch.int64)

        def full(value):
            return value.to(torch.int64) if isinstance(value, torch.Tensor) else torch.full_like(index, int(value))

        return cls(torch.full_like(index, KINDS.index(kind)), full(slot), index, full(step), full(nbytes))

    @classmethod
    def concat(cls, parts: list[Units], device) -> Units:
        if not parts:
            empty = torch.zeros(0, dtype=torch.int64, device=device)
            return cls(empty, empty, empty, empty, empty)
        return cls(*(torch.cat([getattr(p, name) for p in parts]) for name in ("kind", "slot", "index", "step", "nbytes")))

    def take(self, positions: torch.Tensor) -> Units:
        return Units(self.kind[positions], self.slot[positions], self.index[positions], self.step[positions], self.nbytes[positions])


@dataclass(frozen=True)
class Strategy:
    """A decomposition of the routed experts into materialization units (module docstring)."""

    name: str
    family: str  # neuron_pages (A), down_rows (B), neuron_major (C), refinement (D)
    page_rows: int = 0  # B: down output rows per page
    spec: str | None = None  # D: the levels (decision 0003's syntax: "q8", "q6+q4", ...)

    @property
    def level_bits(self) -> tuple[int, ...]:
        from awpmi.decomposition.base import parse_spec

        return parse_spec(self.spec) if self.family == "refinement" else ()

    def initial(self, layer: ExpertLayer, slots: int, device) -> ExpertStates:
        states = ExpertStates.filled(layer, slots, device)
        if self.family == "neuron_pages":
            states.down.fill_(EXACT)  # the whole down projection, first
        elif self.family == "refinement":
            for name in MATRICES:
                states.matrix(name).fill_(0)  # every row's coarse level, first
        return states

    # Knowledge

    def knowledge(self, weights: SampleWeights, states: ExpertStates) -> tuple[MatrixKnowledge, ...]:
        result = []
        levels = weights.levels(self.spec) if self.family == "refinement" else None
        for name in MATRICES:
            truth, norms = weights.truth[name], weights.norms[name]
            if name == "down" and self.family == "neuron_major":
                known = torch.where(states.columns[:, None, :], truth, 0.0)
                result.append(MatrixKnowledge(known, norms, ~states.columns, truth))
                continue
            rows = states.matrix(name)
            exact = rows == EXACT
            known = torch.where(exact[:, :, None], truth, 0.0)
            l2 = torch.where(exact, 0.0, norms.l2)
            linf = torch.where(exact, 0.0, norms.linf)
            if levels is not None:
                approximations, remainders = levels[name]
                for level, (approximation, remainder) in enumerate(zip(approximations, remainders)):
                    at = rows == level
                    if bool(at.any()):
                        known = torch.where(at[:, :, None], approximation, known)
                        l2 = torch.where(at, remainder.l2, l2)
                        linf = torch.where(at, remainder.linf, linf)
            result.append(MatrixKnowledge(known, RowNormBounds(l2, linf), None, truth))
        return tuple(result)

    # Bytes (every row of a matrix has the same size in every state)

    def level_row_bytes(self, width: int) -> list[int]:
        """Bytes of one row of each stored level: bit-packed codes and a float32 scale (decisions 0003 and 0004)."""
        from awpmi.decomposition.packing import payload_bytes

        return [payload_bytes(width, bits) + METADATA_BYTES for bits in self.level_bits]

    def state_bytes(self, width: int, element: int) -> torch.Tensor:
        """Bytes read to bring one row from nothing to each state, as a lookup table indexed by state + 1 (UNKNOWN → 0):
        levels are cumulative; EXACT adds the BF16 row."""
        table = torch.zeros(EXACT + 2, dtype=torch.int64)
        total = 0
        for level, size in enumerate(self.level_row_bytes(width)):
            total += size
            table[level + 1] = total
        table[EXACT + 1] = total + width * element
        return table

    def next_states(self, states: torch.Tensor) -> torch.Tensor:
        if self.family != "refinement":
            return torch.full_like(states, EXACT)
        last = len(self.level_bits) - 1
        return torch.where(states >= last, torch.full_like(states, EXACT), states + 1)

    def bytes_read(self, layer: ExpertLayer, states: ExpertStates) -> dict[str, torch.Tensor]:
        """Per routed expert ([K] each): BF16 bytes per matrix (exact rows or columns), level bytes, BF16 bytes unread."""
        element = layer.element_size
        totals = {}
        unread = torch.zeros(states.gate.shape[0], dtype=torch.int64, device=states.gate.device)
        levels = torch.zeros_like(unread)
        for name in MATRICES:
            width = layer.width(name)
            if name == "down" and self.family == "neuron_major":
                columns = states.columns.sum(dim=1)
                totals[name] = columns * layer.hidden * element
                unread = unread + (layer.intermediate - columns) * layer.hidden * element
                continue
            rows = states.matrix(name)
            exact = (rows == EXACT).sum(dim=1)
            totals[name] = exact * width * element
            unread = unread + (rows.shape[1] - exact) * width * element
            if self.family == "refinement":
                table = self.state_bytes(width, element).to(rows.device)
                cumulative = table[rows + 1] - torch.where(rows == EXACT, width * element, 0)
                levels = levels + cumulative.sum(dim=1)
        totals["levels"] = levels
        totals["unread"] = unread
        return totals

    def metadata_bytes(self, layer: ExpertLayer, slots: int) -> int:
        """The routed experts' resident metadata, charged to the token: norms, and each non-exact level's remainder norms."""
        norms = (2 * (2 * layer.intermediate + layer.hidden) + layer.intermediate) * METADATA_BYTES  # l2 and l∞ per row; C_i
        levels = (2 * layer.intermediate + layer.hidden) * len(self.level_bits) * 2 * METADATA_BYTES
        return slots * (norms + levels)

    def fallback_bytes(self, layer: ExpertLayer, slots: int) -> int:
        """What a runtime following this strategy reads for a token it never certifies: its whole schedule (every level
        of every row, then every BF16 byte) and the metadata."""
        element = layer.element_size
        rows = sum(layer.rows(name) * self.state_bytes(layer.width(name), element)[EXACT + 1] for name in MATRICES)
        return slots * int(rows) + self.metadata_bytes(layer, slots)

    def storage_multiplier(self, layer: ExpertLayer) -> float:
        """Persistent bytes per BF16 expert byte: the checkpoint, plus a neuron-major copy or the levels it needs."""
        element = layer.element_size
        if self.family == "neuron_major":
            return 1.0 + 1.0 / 3.0  # the down projection again, transposed (gate and up rows are already neuron rows)
        if self.family == "refinement":
            shapes = [(layer.rows(name), layer.width(name)) for name in MATRICES]
            levels = sum(rows * sum(self.level_row_bytes(width)) for rows, width in shapes)
            metadata = sum(rows for rows, _ in shapes) * len(self.level_bits) * 2 * METADATA_BYTES
            return 1.0 + (levels + metadata) / sum(rows * width * element for rows, width in shapes)
        return 1.0

    # Units: every unit still to read from `states`, a refinement's later steps after its earlier ones

    def units(self, layer: ExpertLayer, states: ExpertStates) -> Units:
        element, parts = layer.element_size, []
        device = states.gate.device
        slots = states.gate.shape[0]
        for slot in range(slots):
            if self.family in ("neuron_pages", "down_rows"):
                parts.append(Units.of("neuron", slot, (states.gate[slot] != EXACT).nonzero().squeeze(1), 2 * layer.hidden * element))
            if self.family == "down_rows":
                pages = torch.arange(math.ceil(layer.hidden / self.page_rows), device=device)
                first = pages * self.page_rows
                unread = states.down[slot, first] != EXACT  # pages are read whole
                rows = (torch.clamp(first + self.page_rows, max=layer.hidden) - first)[unread]
                parts.append(Units.of("down_rows", slot, pages[unread], rows * layer.intermediate * element))
            if self.family == "neuron_major":
                parts.append(Units.of("neuron_major", slot, (~states.columns[slot]).nonzero().squeeze(1), 3 * layer.hidden * element))
        if self.family == "refinement":
            for kind, name, matrices in (("refine_neuron", "gate", 2), ("refine_down", "down", 1)):
                rows = states.matrix(name)
                table = self.state_bytes(layer.width(name), element).to(device)
                current = rows.clone()
                while True:
                    open_rows = (current != EXACT).nonzero()
                    if not open_rows.numel():
                        break
                    slot, index = open_rows[:, 0], open_rows[:, 1]
                    before = current[slot, index]
                    after = self.next_states(before)
                    parts.append(Units.of(kind, slot, index, matrices * (table[after + 1] - table[before + 1]), after))
                    current[slot, index] = after
        return Units.concat(parts, device)

    def apply(self, states: ExpertStates, units: Units) -> None:
        """Bring each unit's rows to the unit's state (a prefix of an order holds a row's steps in order)."""
        for code, kind in enumerate(KINDS):
            mask = units.kind == code
            if not bool(mask.any()):
                continue
            slot, index, step = units.slot[mask], units.index[mask], units.step[mask]
            if kind in ("neuron", "neuron_major"):
                states.gate[slot, index] = EXACT
                states.up[slot, index] = EXACT
                if kind == "neuron_major":
                    states.columns[slot, index] = True
            elif kind == "down_rows":
                offsets = torch.arange(self.page_rows, device=index.device)
                rows = (index[:, None] * self.page_rows + offsets[None, :]).reshape(-1)
                slots = slot[:, None].expand(-1, self.page_rows).reshape(-1)
                keep = rows < states.down.shape[1]
                states.down[slots[keep], rows[keep]] = EXACT
            elif kind == "refine_neuron":
                _advance(states.gate, slot, index, step)
                _advance(states.up, slot, index, step)
            else:
                _advance(states.down, slot, index, step)


def _advance(rows: torch.Tensor, slot: torch.Tensor, index: torch.Tensor, step: torch.Tensor) -> None:
    """rows[slot, index] = the furthest of `step` per row (a row may appear with several steps)."""
    order = torch.argsort(step, stable=True)  # ascending, so the last write of each row is its furthest step
    rows[slot[order], index[order]] = step[order]


def _cumulative_level(decomposition: RefinementDecomposition, state: int) -> torch.Tensor:
    values = torch.zeros(decomposition.out_features, decomposition.in_features, dtype=torch.float64, device=decomposition.weight.device)
    for level in decomposition.levels[: state + 1]:
        if level.bits:
            values += level.codes.to(torch.float64) * level.scales.to(torch.float64)[:, None]
    return values


def strategies(page_rows: Sequence[int], specs: Sequence[str]) -> list[Strategy]:
    """A, B (one per page size), C, D (one per level specification)."""
    result = [Strategy("A", "neuron_pages")]
    result += [Strategy(f"B{rows}", "down_rows", page_rows=rows) for rows in page_rows]
    result.append(Strategy("C", "neuron_major"))
    result += [Strategy(f"D-{spec}", "refinement", spec=spec) for spec in specs]
    return result


# Orderings


def _gather(units: Units, tables: dict[str, torch.Tensor]) -> torch.Tensor:
    """Each unit's entry in its kind's [K, n] table (refinements: [K, n, steps] by state)."""
    gains = torch.zeros(len(units), dtype=torch.float64, device=units.kind.device)
    for code, kind in enumerate(KINDS):
        mask = units.kind == code
        if bool(mask.any()):
            table = tables[kind]
            if table.dim() == 3:
                step = torch.where(units.step[mask] == EXACT, table.shape[2] - 1, units.step[mask] - 1)
                gains[mask] = table[units.slot[mask], units.index[mask], step]
            else:
                gains[mask] = table[units.slot[mask], units.index[mask]]
    return gains


def _page_sums(rows: torch.Tensor, page_rows: int) -> torch.Tensor:
    """[K, n] → [K, pages]."""
    pages = math.ceil(rows.shape[1] / page_rows)
    padded = torch.zeros(rows.shape[0], pages * page_rows, dtype=rows.dtype, device=rows.device)
    padded[:, : rows.shape[1]] = rows
    return padded.view(rows.shape[0], pages, page_rows).sum(dim=2)


def _level_reductions(levels, name: str, norm: torch.Tensor) -> torch.Tensor:
    """Per row and refinement step, the drop of the remainder bound (metadata norms × input norm [K]): [K, rows, steps]."""
    _, remainders = levels[name]
    per_level = [r.l2 for r in remainders]
    per_level.append(torch.zeros_like(per_level[0]))  # EXACT
    bound = torch.stack(per_level, dim=2) * norm[:, None, None]  # [K, rows, steps + 1]
    return (bound[:, :, :-1] - bound[:, :, 1:]).clamp_min(0.0)


def realistic_gains(units: Units, strategy: Strategy, weights: SampleWeights, bounds: SampleBounds, routing: torch.Tensor,
                    delta: torch.Tensor) -> torch.Tensor:
    """Each unit's share of the tightest pair's bound (Δ = (W_w − W_j)⊙g), from what has been read only."""
    w = routing.to(torch.float64)[:, None]
    column_norms = weights.column_norms
    radius = (bounds.a.upper - bounds.a.lower) * 0.5  # [K, I]
    mixed = matvec(bounds.down.known.transpose(1, 2), delta).abs()  # |K_eᵀ·Δ| [K, I]
    unknown_rows = bounds.down_mass > 0  # [K, H]
    reach = torch.linalg.vector_norm(torch.where(unknown_rows, delta[None, :], 0.0), dim=1)  # ‖Δ over unknown rows‖₂ [K]
    tables: dict[str, torch.Tensor] = {}
    tables["neuron"] = w * (mixed + reach[:, None] * column_norms) * radius
    tables["neuron_major"] = w * torch.linalg.vector_norm(delta) * column_norms * bounds.a.magnitude
    rows = w * delta.abs()[None, :] * bounds.down_mass  # [K, H]
    tables["down_rows"] = _page_sums(rows, strategy.page_rows) if strategy.page_rows else rows
    if strategy.family == "refinement":
        levels = weights.levels(strategy.spec)
        neuron_scale = (w * mixed * radius)[:, :, None]  # the current spread of a, shared out over the steps by the remainder drop
        gate = _level_reductions(levels, "gate", torch.ones(len(weights.experts), dtype=torch.float64, device=delta.device))
        share = gate / gate.sum(dim=2, keepdim=True).clamp_min(1e-300)
        tables["refine_neuron"] = neuron_scale * share
        drops = _level_reductions(levels, "down", torch.linalg.vector_norm(bounds.a.magnitude, dim=1))  # [K, H, steps]
        tables["refine_down"] = w[:, :, None] * delta.abs()[None, :, None] * drops
    return _gather(units, tables)


def ideal_gains(units: Units, strategy: Strategy, weights: SampleWeights, sample: DecodeSample, delta: torch.Tensor, real: bool) -> torch.Tensor:
    """Each unit's true contribution to the reference's top-2 difference Δ·y (diagnostic: it needs every weight)."""
    values = sample.real if real else sample.reference
    w = sample.weights.to(torch.float64)[:, None]
    down = weights.truth["down"]  # [K, H, I]
    a = values["a"].to(torch.float64)
    M = matvec(down.transpose(1, 2), delta).abs()  # |M_i| = |D_eᵀ·Δ|
    neuron = w * M * a.abs()  # w·|M_i·a_i|
    rows = w * (delta[None, :] * values["o"].to(torch.float64)).abs()  # w·|Δ_k·o_k|
    tables: dict[str, torch.Tensor] = {"neuron": neuron, "neuron_major": neuron, "down_rows": _page_sums(rows, strategy.page_rows) if strategy.page_rows else rows}
    if strategy.family == "refinement":
        levels = weights.levels(strategy.spec)
        x = sample.x.to(torch.float64)
        exact_g, exact_u = matvec(weights.truth["gate"], x), matvec(weights.truth["up"], x)
        exact_a = exact_g / (1.0 + torch.exp(-exact_g)) * exact_u
        errors_a, errors_o = [], []
        for gate, up, approximation in zip(levels["gate"][0], levels["up"][0], levels["down"][0]):
            g, u = matvec(gate, x), matvec(up, x)
            errors_a.append((g / (1.0 + torch.exp(-g)) * u - exact_a).abs())  # the neuron's true error at this level
            errors_o.append(matvec(down - approximation, a).abs())  # the output row's true remainder at this level
        errors_a.append(torch.zeros_like(exact_a))
        errors_o.append(torch.zeros_like(errors_o[0]))
        steps = len(errors_a) - 1
        tables["refine_neuron"] = torch.stack([w * M * (errors_a[s] - errors_a[s + 1]).clamp_min(0.0) for s in range(steps)], dim=2)
        tables["refine_down"] = torch.stack([w * delta.abs()[None, :] * (errors_o[s] - errors_o[s + 1]).clamp_min(0.0) for s in range(steps)], dim=2)
    return _gather(units, tables)


def order_units(units: Units, gains: torch.Tensor, steps: int = 1) -> Units:
    """Largest gain per byte first, ties in unit order; a row's later refinement step never before its earlier one.

    A step's score is capped by its row's earlier steps' (`steps` per row): with a stable sort and units listed step
    after step (`Strategy.units`), every prefix of the order holds a row's steps in order.
    """
    score = gains / units.nbytes.to(torch.float64)
    if steps > 1:
        for code in (KINDS.index("refine_neuron"), KINDS.index("refine_down")):
            mask = units.kind == code
            if not bool(mask.any()):
                continue
            slot, index, step = units.slot[mask], units.index[mask], units.step[mask]
            position = torch.where(step == EXACT, steps - 1, step - 1)
            table = torch.full((int(slot.max()) + 1, int(index.max()) + 1, steps), math.inf, dtype=torch.float64, device=score.device)
            table[slot, index, position] = score[mask]
            score[mask] = torch.cummin(table, dim=2).values[slot, index, position]
    return units.take(torch.argsort(-score, stable=True))


# Physical layout (modelled)


@dataclass(frozen=True)
class ExpertOffsets:
    """Where an expert's three tensors start in their checkpoint file (bytes): the published row-major layout."""

    file: str
    gate: int
    up: int
    down: int


def blocks_and_extents(ranges: list[tuple[str, int, int]]) -> tuple[int, int]:
    """4 KiB blocks touched by byte ranges [start, end) of each file, and the extents (runs of consecutive blocks)."""
    blocks: dict[str, set[int]] = {}
    for file, start, end in ranges:
        if end > start:
            blocks.setdefault(file, set()).update(range(start // IO_BLOCK_BYTES, (end - 1) // IO_BLOCK_BYTES + 1))
    count = sum(len(found) for found in blocks.values())
    extents = 0
    for found in blocks.values():
        ordered = sorted(found)
        extents += 1 + sum(1 for previous, current in zip(ordered, ordered[1:]) if current != previous + 1)
    return count, extents


def physical_estimate(strategy: Strategy, layer: ExpertLayer, experts: Sequence[int], states: ExpertStates,
                      offsets: dict[int, ExpertOffsets], fallback: bool) -> dict[str, dict[str, int]]:
    """Modelled direct I/O of the final state (with a fallback, every BF16 byte too): 4 KiB blocks and extents.

    checkpoint         the published files, row-major: a down column touches every row of its matrix
    neuron_major_copy  C only: a record per neuron (gate row, up row, down column), each expert's file 4 KiB-aligned
    levels             D only: a file per level and matrix, a record (codes, scale) per row, 4 KiB-aligned
    """
    element = layer.element_size
    row_bytes = {name: layer.width(name) * element for name in MATRICES}
    layouts: dict[str, list] = {"checkpoint": []}
    if strategy.family == "neuron_major":
        layouts["neuron_major_copy"] = []
    if strategy.family == "refinement":
        layouts["levels"] = []
    for slot, expert in enumerate(experts):
        place = offsets[expert]
        for name in MATRICES:
            start = getattr(place, name)
            if name == "down" and strategy.family == "neuron_major":
                if fallback or bool(states.columns[slot].any()):
                    layouts["checkpoint"].append((place.file, start, start + layer.hidden * row_bytes["down"]))
                continue
            rows = states.matrix(name)[slot]
            exact = torch.ones_like(rows, dtype=torch.bool) if fallback else rows == EXACT
            for row in exact.nonzero().squeeze(1).tolist():
                layouts["checkpoint"].append((place.file, start + row * row_bytes[name], start + (row + 1) * row_bytes[name]))
        if strategy.family == "neuron_major":
            columns = torch.ones_like(states.columns[slot]) if fallback else states.columns[slot]
            record = 3 * layer.hidden * element
            for neuron in columns.nonzero().squeeze(1).tolist():
                layouts["neuron_major_copy"].append((f"neuron_major/{expert}", neuron * record, (neuron + 1) * record))
        if strategy.family == "refinement":
            for name in MATRICES:
                rows = states.matrix(name)[slot]
                for level, record in enumerate(strategy.level_row_bytes(layer.width(name))):
                    reached = (rows >= level) & (rows != UNKNOWN)  # an exact row went through every level
                    for row in reached.nonzero().squeeze(1).tolist():
                        layouts["levels"].append((f"levels/{expert}/{name}/{level}", row * record, (row + 1) * record))
    result = {}
    for name, ranges in layouts.items():
        blocks, extents = blocks_and_extents(ranges)
        logical = sum(end - start for _, start, end in ranges)
        result[name] = {"logical_bytes": logical, "blocks_4k": blocks, "physical_bytes": blocks * IO_BLOCK_BYTES, "extents": extents}
    return result


# One cell


@dataclass
class CellResult:
    would_certify: bool  # the certificate held under the cell's tiers
    certified: bool  # would_certify in the certified arithmetic, the only one that certifies
    winner: int  # the certified (or would-be) token, −1 if none
    candidate_is_reference: bool  # the last candidate is the reference's token
    evaluations: int
    full_checks: int
    checkpoint: int  # the budget index reached (−1: the initial state)
    bytes: dict[str, int]
    fraction: float  # of the routed experts' BF16 bytes
    per_expert_fraction: list[float]
    units_read: dict[str, int]
    states: dict[str, dict[str, int]]
    margin: float
    unread_term: float
    physical: dict[str, dict[str, int]]
    violations: dict[str, int]
    reused_ceiling: bool  # the certificate at the whole schedule is the ceiling's (same knowledge)


def pair_delta(lm_weight: torch.Tensor, norm_weight: torch.Tensor, winner: int, contender: int) -> torch.Tensor:
    """Δ = (W_w − W_j)⊙g in float64."""
    return (lm_weight[winner].to(torch.float64) - lm_weight[contender].to(torch.float64)) * norm_weight.to(torch.float64)


def run_cell(
    sample: DecodeSample,
    weights: SampleWeights,
    strategy: Strategy,
    arithmetic: Arithmetic,
    bound: BoundTier,
    ordering: Ordering,
    certifier: Certifier,
    budget_step: float,
    norm_weight: torch.Tensor,
    norm_eps: float,
    offsets: dict[int, ExpertOffsets],
    complete: Outcome | None = None,
) -> CellResult:
    """The smallest budget (multiples of `budget_step` of the routed BF16 bytes, after the strategy's first reads) at which
    the certificate holds when the units are read in the cell's order (module docstring).

    `complete` is the ceiling's outcome in the same arithmetic tier: once every unit is read, the knowledge is the
    ceiling's (every row exact, no remainder, in either bound tier), and so is the certificate.
    """
    device = sample.x.device
    layer, experts = weights.layer, sample.experts
    routed = len(experts) * layer.expert_bytes
    real = arithmetic.real
    violations: dict[str, int] = {}
    evaluations = full_checks = 0
    reused_ceiling = False

    def evaluate(states: ExpertStates, full: bool):
        nonlocal evaluations, full_checks
        bounds = propagate(sample, strategy.knowledge(weights, states), arithmetic, bound, norm_weight, norm_eps, weights.column_norms)
        if bound is BoundTier.REALISTIC:
            for name, count in enclosure_violations(bounds, sample, real).items():
                violations[name] = violations.get(name, 0) + count
        outcome = certifier.evaluate(bounds, full=full)
        evaluations += 1
        full_checks += outcome.full_checks
        return bounds, outcome

    initial = strategy.initial(layer, len(experts), device)
    bounds, outcome = evaluate(initial, full=True)
    final_states, final_outcome, checkpoint = initial, outcome, -1
    if not outcome.certified:
        units = strategy.units(layer, initial)
        if ordering is Ordering.IDEAL:
            gains = ideal_gains(units, strategy, weights, sample, pair_delta(certifier.lm_weight, norm_weight, sample.token, sample.runner_up), real)
        else:
            contender = outcome.tightest if outcome.tightest >= 0 else sample.runner_up
            delta = pair_delta(certifier.lm_weight, norm_weight, outcome.candidate, contender)
            gains = realistic_gains(units, strategy, weights, bounds, sample.weights, delta)
        ordered = order_units(units, gains, max(1, len(strategy.level_bits)))
        spent = torch.cumsum(ordered.nbytes, dim=0)
        total = int(spent[-1]) if len(ordered) else 0
        steps = math.ceil(total / (budget_step * routed)) if total else 0

        def state_at(index: int) -> ExpertStates:
            limit = min((index + 1) * budget_step * routed, total)
            count = int((spent <= limit + 0.5).sum())
            states = initial.copy()
            strategy.apply(states, ordered.take(torch.arange(count, device=device)))
            return states

        # The first budget whose nearest pairs all hold (monotone with realistic bounds), then the full check there.
        low, high = 0, steps - 1
        while low < high:
            middle = (low + high) // 2
            _, probe = evaluate(state_at(middle), full=False)
            if probe.near_failed:
                low = middle + 1
            else:
                high = middle
        for index in range(low, steps):
            states = state_at(index)
            if complete is not None and index == steps - 1:
                outcome = complete  # the whole schedule read: the ceiling's knowledge and certificate
                reused_ceiling = True
            else:
                bounds, outcome = evaluate(states, full=True)
            final_states, final_outcome, checkpoint = states, outcome, index
            if outcome.certified:
                break
    states, outcome = final_states, final_outcome
    read = strategy.bytes_read(layer, states)
    would_certify = outcome.certified
    per_expert_read = read["gate"] + read["up"] + read["down"] + read["levels"]
    totals = {key: int(read[key].sum()) for key in ("gate", "up", "down", "levels")}
    totals["metadata"] = strategy.metadata_bytes(layer, len(experts))
    totals["fallback"] = 0 if would_certify else int(read["unread"].sum())
    totals["total"] = sum(totals.values())
    units_read = {
        "gate_rows": int((states.gate == EXACT).sum()), "up_rows": int((states.up == EXACT).sum()),
        "down_rows": int((states.down == EXACT).sum()), "down_columns": int(states.columns.sum()),
    }
    level_states: dict[str, dict[str, int]] = {}
    if strategy.family == "refinement":
        for name in MATRICES:
            values = states.matrix(name)
            level_states[name] = {str(int(v)): int((values == v).sum()) for v in torch.unique(values).tolist()}
    return CellResult(
        would_certify=would_certify,
        certified=would_certify and arithmetic.certifies,
        winner=outcome.candidate if would_certify else -1,
        candidate_is_reference=outcome.candidate == sample.token,
        evaluations=evaluations,
        full_checks=full_checks,
        checkpoint=checkpoint,
        bytes=totals,
        fraction=totals["total"] / routed,
        per_expert_fraction=(per_expert_read.to(torch.float64) / layer.expert_bytes).tolist(),
        units_read=units_read,
        states=level_states,
        margin=outcome.margin,
        unread_term=float(outcome.terms.get("unread_neurons", 0.0)) if outcome.terms else 0.0,
        physical=physical_estimate(strategy, layer, experts, states, offsets, not would_certify),
        violations=violations,
        reused_ceiling=reused_ceiling,
    )


def ceiling(sample: DecodeSample, weights: SampleWeights, arithmetic: Arithmetic, certifier: Certifier, norm_weight: torch.Tensor,
            norm_eps: float) -> tuple[Outcome, SampleBounds, dict[str, int]]:
    """Every routed weight known exactly: the certificate's ceiling under `arithmetic` (the floor of rounding alone)."""
    states = ExpertStates.filled(weights.layer, len(sample.experts), sample.x.device, EXACT, columns=True)
    knowledge = Strategy("ceiling", "neuron_pages").knowledge(weights, states)
    bounds = propagate(sample, knowledge, arithmetic, BoundTier.REALISTIC, norm_weight, norm_eps, weights.column_norms)
    violations = enclosure_violations(bounds, sample, arithmetic.real)
    return certifier.evaluate(bounds), bounds, violations
