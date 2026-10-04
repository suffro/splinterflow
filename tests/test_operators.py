"""Phase 2 operator bounds: soundness against the reference's own kernels, adversarial faithful
realizations, exact arithmetic, and guards.

Three independent checks per operator (roadmap §2.4):
  * brute force: every BF16 input inside the input enclosures (or extreme and random samples
    for wide ones) through the reference kernel itself, on CPU and CUDA;
  * adversarial faithful realizations: every rounding chosen up or down at random, every
    fp32 operation perturbed up to its assumed relative error, in float64/Decimal;
  * exact arithmetic (`Decimal`, `Fraction`) for the real-valued parts.
"""

from __future__ import annotations

import itertools
import math
from decimal import Decimal, getcontext
from fractions import Fraction

import pytest
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from awpmi.bounds import operators
from awpmi.bounds.enclosure import Enclosure, UnboundedValue
from awpmi.bounds.floating import round_down_to_grid, round_up_to_grid
from awpmi.bounds.operators import (
    FP32_OP_RELATIVE_ERROR,
    SILU_MINIMUM_LOWER,
    linear,
    multiply,
    residual_add,
    rms_norm,
    silu,
)
from awpmi.bounds.residual import FP32_ACCUMULATION_UNIT_ROUNDOFF, ReferenceNumerics
from awpmi.bounds.rounding import RoundingModel
from tests.conftest import DEVICES

BF16 = torch.bfloat16
MODELS = list(RoundingModel)


def grid_enclosure(generator, shape, scale: float, max_steps: int = 3, device="cpu") -> Enclosure:
    """Random BF16-grid enclosures [v, v + k ulps] with k in 0..max_steps."""
    values = (torch.randn(shape, generator=generator) * scale).to(BF16)
    upper = values.clone()
    for _ in range(max_steps):
        step = torch.rand(shape, generator=generator) < 0.6
        upper = torch.where(step, torch.nextafter(upper, torch.full_like(upper, math.inf)), upper)
    return Enclosure(values.to(torch.float64).to(device), upper.to(torch.float64).to(device))


def grid_points(lower: float, upper: float, limit: int = 64) -> list[float]:
    points, value = [], torch.tensor(lower, dtype=torch.float64).to(BF16)
    while float(value) <= upper and len(points) < limit:
        points.append(float(value))
        value = torch.nextafter(value, torch.tensor(math.inf, dtype=BF16))
    return points


def elementwise_cases(*enclosures: Enclosure) -> tuple[list[torch.Tensor], torch.Tensor]:
    """All combinations of grid points of the enclosures, element by element: input columns and element ids."""
    columns, owners = [[] for _ in enclosures], []
    for element in range(enclosures[0].lower.numel()):
        choices = [grid_points(float(e.lower[element]), float(e.upper[element])) for e in enclosures]
        for combination in itertools.product(*choices):
            for column, value in zip(columns, combination):
                column.append(value)
            owners.append(element)
    return [torch.tensor(c, dtype=torch.float64) for c in columns], torch.tensor(owners)


def assert_inside(values: torch.Tensor, enclosure: Enclosure, owners: torch.Tensor | None = None) -> None:
    values = values.to(torch.float64).cpu()
    lower, upper = enclosure.lower.cpu(), enclosure.upper.cpu()
    if owners is not None:
        lower, upper = lower[owners], upper[owners]
    outside = (values < lower) | (values > upper)
    assert not bool(outside.any()), (values[outside][:5], lower[outside][:5], upper[outside][:5])


# Brute force through the reference kernels


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model", MODELS)
def test_residual_add_and_multiply_enclose_the_reference_kernels(device, model):
    generator = torch.Generator().manual_seed(0)
    for scale in (1e-3, 1.0, 30.0):
        left = grid_enclosure(generator, 128, scale)
        right = grid_enclosure(generator, 128, scale * 3)
        (a, b), owners = elementwise_cases(left, right)
        a, b = a.to(BF16).to(device), b.to(BF16).to(device)
        assert_inside(a + b, residual_add(left, right, BF16, model), owners)
        assert_inside(a * b, multiply(left, right, BF16, model), owners)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model", MODELS)
def test_silu_encloses_the_reference_kernel(device, model):
    generator = torch.Generator().manual_seed(1)
    special = torch.tensor([-100.0, -88.0, -20.0, -1.2784645, -1.28125, -1.2734375, 0.0, 1e-30, 50.0, 1000.0])
    for scale in (0.5, 3.0, 40.0):
        inputs = grid_enclosure(generator, 256, scale, max_steps=4)
        inputs = Enclosure(torch.cat([inputs.lower, special.double()]), torch.cat([inputs.upper, special.double()]))
        (x,), owners = elementwise_cases(inputs)
        assert_inside(F.silu(x.to(BF16).to(device)), silu(inputs, BF16, model), owners)
    # An enclosure straddling the minimum, wide.
    wide = Enclosure(torch.tensor([-3.0, -1.5]).double(), torch.tensor([0.5, -1.0]).double())
    (x,), owners = elementwise_cases(wide)
    assert_inside(F.silu(x.to(BF16).to(device)), silu(wide, BF16, model), owners)


def rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float, module_class=LlamaRMSNorm):
    """LlamaRMSNorm.forward (or DeepseekV3RMSNorm's), plus its internal q and n, by the same operations on the same tensors."""
    module = module_class(weight.numel(), eps=eps).to(device=x.device, dtype=weight.dtype)
    module.weight.data.copy_(weight)
    with torch.inference_mode():
        output = module(x)
        hidden = x.to(torch.float32)
        scale = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + eps)
        normalized = (hidden * scale).to(x.dtype)
    return output, scale.squeeze(-1), normalized


def sample_rows(enclosure: Enclosure, generator, count: int) -> torch.Tensor:
    """Rows of grid points: all-lower, all-upper, alternating, then random endpoints and midpoints."""
    lower, upper = enclosure.lower.cpu(), enclosure.upper.cpu()
    middle = round_down_to_grid(lower * 0.5 + upper * 0.5, BF16)
    rows = [lower, upper, torch.where(torch.arange(lower.numel()) % 2 == 0, lower, upper)]
    for _ in range(count):
        pick = torch.randint(3, lower.shape, generator=generator)
        rows.append(torch.where(pick == 0, lower, torch.where(pick == 1, upper, middle)))
    return torch.stack(rows)


def deepseek_v3_rms_norm():
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RMSNorm

    return DeepseekV3RMSNorm


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("module", ["llama", "deepseek_v3"])
def test_rms_norm_encloses_the_reference_module(device, model, module):
    """Phase 5A: DeepseekV3RMSNorm (Moonlight's final norm) runs LlamaRMSNorm's operations; both are checked."""
    module_class = LlamaRMSNorm if module == "llama" else deepseek_v3_rms_norm()
    generator = torch.Generator().manual_seed(2)
    eps = 1e-5
    # Exhaustive on a tiny hidden size, sampled on the real ones (SmolLM2's 576, Moonlight's 2048).
    for width, scale, exhaustive in ((5, 2.0, True), (6, 0.01, True), (576, 20.0, False), (576, 1e-3, False), (2048, 6.0, False)):
        inputs = grid_enclosure(generator, width, scale, max_steps=2, device=device)
        weight = (torch.randn(width, generator=generator) * 0.8 + 0.2).to(BF16).to(device)
        bounds = rms_norm(inputs, weight, eps, model, FP32_ACCUMULATION_UNIT_ROUNDOFF)
        if exhaustive:
            choices = [grid_points(float(lo), float(hi)) for lo, hi in zip(inputs.lower.cpu(), inputs.upper.cpu())]
            rows = torch.tensor(list(itertools.product(*choices)), dtype=torch.float64)
        else:
            rows = sample_rows(inputs, generator, 200)
        output, scale_values, normalized = rms_norm_reference(rows.to(BF16).to(device)[:, None, :], weight, eps, module_class)
        for row in range(rows.shape[0]):
            assert_inside(output[row, 0], bounds.output)
            assert_inside(normalized[row, 0], bounds.normalized)
        scale_values = scale_values.to(torch.float64).reshape(-1)
        assert bool(((scale_values >= bounds.scale_lower) & (scale_values <= bounds.scale_upper)).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model", MODELS)
def test_linear_encloses_the_reference_gemm(device, model):
    generator = torch.Generator().manual_seed(3)
    for rows_, width in ((40, 96), (64, 576)):
        weight = (torch.randn(rows_, width, generator=generator) * 0.05).to(BF16).to(device)
        numerics = ReferenceNumerics(BF16, FP32_ACCUMULATION_UNIT_ROUNDOFF, width)
        inputs = grid_enclosure(generator, width, 2.0, max_steps=2, device=device)
        samples = sample_rows(inputs, generator, 60).to(BF16).to(device)
        actual = torch.stack([F.linear(s.view(1, 1, -1), weight).reshape(-1) for s in samples])
        full = linear(inputs, weight.to(torch.float64), numerics, model)
        # The same matrix with a third of its columns unread, bounded by their exact masses.
        unread = torch.rand(width, generator=generator).to(device) < 0.33
        read_weight = torch.where(unread, 0.0, weight.to(torch.float64))
        unread_mass = (weight.to(torch.float64).abs() * unread) @ inputs.magnitude
        partial = linear(inputs, read_weight, numerics, model, unread_mass=unread_mass * (1 + 1e-12))
        for row in actual:
            assert_inside(row, full)
            assert_inside(row, partial)
        assert bool((partial.width >= full.width).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model", MODELS)
def test_routing_weights_and_the_combine_enclose_grouped_mm_semantics(device, model):
    """Phase 5A: BF16 expert outputs × float32 routing weights (→ float32), summed over the top-k in float32, cast to BF16,
    as transformers' grouped_mm combine computes them, on every grid point of small enclosures."""
    generator = torch.Generator().manual_seed(9)
    for scale in (1e-3, 0.5, 40.0):
        terms = [grid_enclosure(generator, 32, scale, max_steps=2) for _ in range(3)]
        weights = (torch.rand(3, generator=generator) * 2.4 + 0.01).to(torch.float32)
        scaled = [multiply(term, Enclosure.exact(weights[k].double().expand(32)), torch.float32, model) for k, term in enumerate(terms)]
        combined = operators.reduce_sum(scaled, FP32_ACCUMULATION_UNIT_ROUNDOFF, BF16, model)
        values, owners = elementwise_cases(*terms)
        outputs = torch.stack(values).to(BF16).to(device)  # [k, cases]: the cases play the hidden dimension
        weighted = outputs * weights.to(device)[:, None]  # BF16 × float32 → float32, as `proj_out * sample_weights`
        for k in range(3):
            assert_inside(weighted[k], scaled[k], owners)
        # `weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)`, then `.to(bfloat16)`.
        assert_inside(weighted.view(1, 3, -1).sum(dim=1).to(BF16).reshape(-1), combined, owners)
        # Another order, by explicit additions.
        reversed_sum = (weighted[2] + weighted[1]) + weighted[0]
        assert_inside(reversed_sum.to(BF16), combined, owners)


def test_reduce_sum_encloses_adversarial_realizations():
    """Any fp32 summation order with every rounding up or down, then a faithful BF16 conversion."""
    import random

    rng = random.Random(10)
    generator = torch.Generator().manual_seed(10)
    for count in (2, 6, 8):
        terms = [Enclosure.exact((torch.randn(64, generator=generator) * 10.0 ** rng.uniform(-3, 2)).to(torch.float32).double()) for _ in range(count)]
        combined = operators.reduce_sum(terms, FP32_ACCUMULATION_UNIT_ROUNDOFF, BF16, RoundingModel.FAITHFUL)
        for _ in range(20):
            order = list(range(count))
            rng.shuffle(order)
            values = []
            for element in range(64):
                total = 0.0
                for k in order:
                    total = faithful_round(total + float(terms[k].lower[element]), torch.float32, rng.random() < 0.5)
                values.append(faithful_round(total, BF16, rng.random() < 0.5))
            assert_inside(torch.tensor(values), combined)
        # The exact real sum (Fraction) is inside too.
        exact = [float(sum(Fraction(float(t.lower[element])) for t in terms)) for element in range(64)]
        assert bool(((torch.tensor(exact).double() >= combined.lower) & (torch.tensor(exact).double() <= combined.upper)).all())


def test_reduce_sum_without_its_error_term_is_caught(monkeypatch):
    """Guard check: with no fp32 reduction error, an adversarial summation escapes (cancellation makes it visible)."""
    import random

    monkeypatch.setattr(operators, "gamma", lambda n, u: 0.0)
    rng = random.Random(11)
    escaped = 0
    for _ in range(200):
        big = rng.uniform(1.0, 2.0) * 2.0**20
        values = [big, 1.0 + rng.random() * 2.0**-10, -big]
        terms = [Enclosure.exact(torch.tensor([float(torch.tensor(v, dtype=torch.float32))]).double()) for v in values]
        combined = operators.reduce_sum(terms, FP32_ACCUMULATION_UNIT_ROUNDOFF, torch.float32, RoundingModel.FAITHFUL)
        total = faithful_round(faithful_round(values[0] + values[1], torch.float32, rng.random() < 0.5) + values[2], torch.float32, True)
        escaped += int(not float(combined.lower[0]) <= total <= float(combined.upper[0]))
    assert escaped > 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model", MODELS)
def test_a_batch_of_linear_operations_equals_each_one(device, model):
    """Phase 5A: `linear` on a batch of weights [B, N, K] (one experts call's routed experts) gives, bitwise, each weight's
    own enclosure, with one shared input or one input per weight."""
    generator = torch.Generator().manual_seed(12)
    numerics = ReferenceNumerics(BF16, FP32_ACCUMULATION_UNIT_ROUNDOFF, 96)
    weights = (torch.randn(3, 40, 96, generator=generator) * 0.05).to(BF16).to(torch.float64).to(device)
    unread = torch.rand(3, 40, generator=generator, dtype=torch.float64).to(device) * 0.01
    shared = grid_enclosure(generator, 96, 2.0, max_steps=2, device=device)
    own = [grid_enclosure(generator, 96, 1.0, max_steps=2, device=device) for _ in range(3)]
    batched_shared = linear(shared, weights, numerics, model, unread)
    batched_own = linear(Enclosure(torch.stack([e.lower for e in own]), torch.stack([e.upper for e in own])), weights, numerics, model, unread)
    for b in range(3):
        alone = linear(shared, weights[b], numerics, model, unread[b])
        assert torch.equal(batched_shared.lower[b], alone.lower) and torch.equal(batched_shared.upper[b], alone.upper)
        alone = linear(own[b], weights[b], numerics, model, unread[b])
        assert torch.equal(batched_own.lower[b], alone.lower) and torch.equal(batched_own.upper[b], alone.upper)


# Adversarial faithful realizations


def faithful_round(value: float, dtype: torch.dtype, up: bool) -> float:
    tensor = torch.tensor([value], dtype=torch.float64)
    return float((round_up_to_grid if up else round_down_to_grid)(tensor, dtype))


def perturb(value: float, rng, relative: float) -> float:
    """Any error within the assumed relative bound, extremes included."""
    choice = rng.random()
    factor = -1.0 if choice < 0.25 else 1.0 if choice < 0.5 else rng.uniform(-1, 1)
    return value * (1.0 + factor * relative * (1 - 1e-6))


def faithful_rms_norm(x: list[float], weight: list[float], eps: float, rng):
    """One faithful realization of LlamaRMSNorm within the operator module's assumptions."""
    fp32 = lambda v: faithful_round(v, torch.float32, rng.random() < 0.5)
    squares = [fp32(perturb(v * v, rng, FP32_OP_RELATIVE_ERROR)) for v in x]
    order = list(range(len(x)))
    rng.shuffle(order)
    total = 0.0
    for k in order:
        total = fp32(total + squares[k])
    variance = fp32(perturb(total / len(x), rng, FP32_OP_RELATIVE_ERROR))
    shifted = fp32(perturb(variance + float(torch.tensor(eps, dtype=torch.float32)), rng, FP32_OP_RELATIVE_ERROR))
    scale = fp32(perturb(1.0 / math.sqrt(shifted), rng, FP32_OP_RELATIVE_ERROR))
    normalized = [faithful_round(fp32(v * scale), BF16, rng.random() < 0.5) for v in x]
    output = [faithful_round(fp32(w * n), BF16, rng.random() < 0.5) for w, n in zip(weight, normalized)]
    return scale, normalized, output


def test_rms_norm_encloses_adversarial_faithful_realizations():
    import random

    rng = random.Random(4)
    generator = torch.Generator().manual_seed(4)
    for width, scale in ((7, 1.0), (64, 15.0), (576, 3.0)):
        inputs = grid_enclosure(generator, width, scale, max_steps=2)
        weight = (torch.randn(width, generator=generator) + 0.5).to(BF16)
        bounds = rms_norm(inputs, weight, 1e-5, RoundingModel.FAITHFUL, FP32_ACCUMULATION_UNIT_ROUNDOFF)
        for row in sample_rows(inputs, generator, 30):
            q, n, h = faithful_rms_norm(row.tolist(), weight.double().tolist(), 1e-5, rng)
            assert bounds.scale_lower <= q <= bounds.scale_upper
            assert_inside(torch.tensor(n), bounds.normalized)
            assert_inside(torch.tensor(h), bounds.output)


def test_silu_encloses_adversarial_faithful_realizations():
    import random

    rng = random.Random(5)
    generator = torch.Generator().manual_seed(5)
    inputs = grid_enclosure(generator, 512, 4.0, max_steps=3)
    bounds = silu(inputs, BF16, RoundingModel.FAITHFUL)
    (x,), owners = elementwise_cases(inputs)
    exact = (x / (1 + torch.exp(-x))).tolist()
    for _ in range(3):
        values = [faithful_round(perturb(v, rng, FP32_OP_RELATIVE_ERROR), BF16, rng.random() < 0.5) for v in exact]
        assert_inside(torch.tensor(values), bounds, owners)


# Exact arithmetic


def test_silu_minimum_constant_is_a_lower_bound():
    """min x·σ(x) = −W(1/e); W(1/e) solved by Newton in 60-digit decimal arithmetic."""
    getcontext().prec = 60
    target = Decimal(-1).exp()
    w = Decimal("0.28")
    for _ in range(60):
        w = w - (w * w.exp() - target) / (w.exp() * (w + 1))
    assert Decimal(repr(SILU_MINIMUM_LOWER)) <= -w
    assert -w - Decimal(repr(SILU_MINIMUM_LOWER)) < Decimal("1e-15")


def test_rms_norm_encloses_the_exact_real_result():
    """The real-valued LlamaRMSNorm of grid inputs (60-digit Decimal) lies in the faithful output enclosure."""
    getcontext().prec = 60
    generator = torch.Generator().manual_seed(6)
    eps32 = Fraction(float(torch.tensor(1e-5, dtype=torch.float32)))
    for width, scale in ((9, 0.3), (576, 12.0)):
        inputs = grid_enclosure(generator, width, scale, max_steps=1)
        weight = (torch.randn(width, generator=generator)).to(BF16)
        bounds = rms_norm(inputs, weight, 1e-5, RoundingModel.FAITHFUL, FP32_ACCUMULATION_UNIT_ROUNDOFF)
        for row in sample_rows(inputs, generator, 5):
            x = [Fraction(v) for v in row.tolist()]
            variance = sum(v * v for v in x) / len(x) + eps32
            q = 1 / Decimal(variance.numerator / Decimal(variance.denominator)).sqrt()
            assert bounds.scale_lower <= float(q) <= bounds.scale_upper
            exact = [float(Decimal(float(w)) * Decimal(float(v)) * q) for w, v in zip(weight.double(), x)]
            assert_inside(torch.tensor(exact, dtype=torch.float64), bounds.output)


# Guards


def test_unbounded_inputs_are_refused():
    huge = Enclosure(torch.tensor([2.0**60, 1.0]).double(), torch.tensor([2.0**60, 1.0]).double())
    with pytest.raises(UnboundedValue):
        rms_norm(huge, torch.ones(2, dtype=BF16), 1e-5, RoundingModel.FAITHFUL, FP32_ACCUMULATION_UNIT_ROUNDOFF)
    infinite = Enclosure(torch.tensor([-math.inf]).double(), torch.tensor([1.0]).double())
    with pytest.raises(UnboundedValue):
        silu(infinite, BF16, RoundingModel.FAITHFUL)
    with pytest.raises(ValueError):
        Enclosure(torch.tensor([1.0]).double(), torch.tensor([0.0]).double())


def test_nearest_even_in_place_of_faithful_is_caught(monkeypatch):
    """Guard check: an RN-even enclosure labelled faithful misses faithful realizations that round the other way."""
    import random

    from awpmi.bounds import rounding

    nearest = rounding.round_enclosure
    monkeypatch.setattr(
        operators, "round_enclosure", lambda lower, upper, dtype, model: nearest(lower, upper, dtype, RoundingModel.NEAREST_EVEN)
    )
    rng = random.Random(8)
    generator = torch.Generator().manual_seed(8)
    inputs = grid_enclosure(generator, 512, 4.0, max_steps=0)
    bounds = silu(inputs, BF16, RoundingModel.FAITHFUL)
    (x,), owners = elementwise_cases(inputs)
    exact = (x / (1 + torch.exp(-x))).tolist()
    values = torch.tensor([faithful_round(v, BF16, rng.random() < 0.5) for v in exact])
    outside = (values < bounds.lower[owners]) | (values > bounds.upper[owners])
    assert bool(outside.any())


def test_disabling_the_fp32_error_term_is_caught(monkeypatch):
    """Guard check: without the fp32 operation error, an adversarial realization escapes the RMSNorm bound."""
    import random

    monkeypatch.setattr(operators, "FP32_OP_RELATIVE_ERROR", 0.0)
    rng = random.Random(7)
    generator = torch.Generator().manual_seed(7)
    escaped = 0
    for _ in range(20):
        inputs = grid_enclosure(generator, 16, 1.0, max_steps=0)
        weight = torch.ones(16, dtype=BF16)
        bounds = rms_norm(inputs, weight, 1e-5, RoundingModel.FAITHFUL, FP32_ACCUMULATION_UNIT_ROUNDOFF)
        q, _, _ = faithful_rms_norm(inputs.lower.tolist(), weight.double().tolist(), 1e-5, rng)
        escaped += int(not bounds.scale_lower <= q <= bounds.scale_upper)
    assert escaped > 0
