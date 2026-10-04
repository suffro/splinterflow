"""Phase 5A (decision 0009): the expert-AWPMI oracle on a DeepSeek-V3 model at test size.

The oracle's reference recomputation against the model's own forward; enclosures of every intermediate in every tier
and state; the mixture's decomposed bound against the reference's Δ·y; certificates (only ever the reference's token,
monotone in knowledge); byte accounting; the orderings (the realistic one sees no weight value); refinement steps in
order; the binary32 projection's error bound; the relaxation; and sabotage: narrowed enclosures are caught.
"""

from __future__ import annotations

import math

import pytest
import torch

from awpmi.bounds import pairwise
from awpmi.bounds.enclosure import Enclosure
from awpmi.bounds.rounding import CERTIFIED
from awpmi.oracle import experts as oracle
from tests.conftest import DEVICES
from tests.test_moonlight import tiny_moonlight

BF16 = torch.bfloat16


def decode_capture(model, ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """One prefill, then one decode step; the last MoE layer's tensors at the decode token, as the model computed them."""
    layer = model.model.layers[-1]
    values: dict[str, torch.Tensor] = {}

    def last(tensor):
        return tensor.detach().reshape(-1, tensor.shape[-1])[-1].clone()

    hooks = [
        layer.mlp.experts.register_forward_pre_hook(lambda m, args: values.update(x=last(args[0]), index=args[1][-1].clone(), weights=args[2][-1].clone())),
        layer.mlp.experts.register_forward_hook(lambda m, args, out: values.update(R=last(out))),
        layer.post_attention_layernorm.register_forward_pre_hook(lambda m, args: values.update(r=last(args[0]))),
        layer.mlp.shared_experts.register_forward_hook(lambda m, args, out: values.update(S=last(out))),
        layer.mlp.register_forward_hook(lambda m, args, out: values.update(m=last(out))),
        layer.register_forward_hook(lambda m, args, out: values.update(y=last(out[0] if isinstance(out, tuple) else out))),
        model.model.norm.register_forward_hook(lambda m, args, out: values.update(h=last(out))),
    ]
    try:
        with torch.inference_mode():
            output = model(input_ids=ids, use_cache=True)
            following = output.logits[0, -1].argmax().view(1, 1)
            values.clear()
            output = model(input_ids=following, past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
    finally:
        for hook in hooks:
            hook.remove()
    values["logits"] = output.logits[0, -1]
    return values


def build(device: str, seed: int = 0, margin: float | None = None):
    """A tiny Moonlight, one decode sample at its last MoE layer; `margin` plants a top-1 lead of about that many logits."""
    model = tiny_moonlight(seed=seed).to(device)
    model.config._experts_implementation = "grouped_mm"
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.config.vocab_size, (1, 12), generator=generator).to(device)
    captured = decode_capture(model, ids)
    block = model.model.layers[-1].mlp
    layer = oracle.ExpertLayer(block.experts.gate_up_proj.detach(), block.experts.down_proj.detach(), block.experts.act_fn)
    lm_weight = model.lm_head.weight.detach().clone()
    if margin is not None:
        h = captured["h"].to(torch.float64)
        top = int(captured["logits"].argmax())
        lm_weight[top] = (lm_weight[top].to(torch.float64) + margin * h / float(h @ h)).to(BF16)
    norm = model.model.norm
    sample = oracle.decode_sample(layer, captured["x"], captured["r"], captured["S"], captured["index"].tolist(), captured["weights"], norm, lm_weight)
    NORMS[id(sample)] = norm.weight.detach()
    EPS[id(sample)] = norm.variance_epsilon
    return model, layer, sample, captured, lm_weight


NORMS: dict[int, torch.Tensor] = {}  # each built sample's final-norm weight and epsilon (what `propagate` needs)
EPS: dict[int, float] = {}


def offsets_for(layer: oracle.ExpertLayer) -> dict[int, oracle.ExpertOffsets]:
    """A packed row-major layout, as a checkpoint file would hold the experts."""
    element, result, position = layer.element_size, {}, 0
    for expert in range(layer.experts):
        sizes = [layer.rows(name) * layer.width(name) * element for name in oracle.MATRICES]
        result[expert] = oracle.ExpertOffsets("experts", position, position + sizes[0], position + sizes[0] + sizes[1])
        position += sum(sizes)
    return result


STRATEGIES = oracle.strategies([1, 4], ["q8", "q6+q4"])


def random_states(strategy: oracle.Strategy, layer, slots, device, generator) -> oracle.ExpertStates:
    """A random state reachable by the strategy: a random prefix of a random order of its units."""
    states = strategy.initial(layer, slots, device)
    units = strategy.units(layer, states)
    gains = torch.rand(len(units), generator=generator, dtype=torch.float64).to(device)
    ordered = oracle.order_units(units, gains, max(1, len(strategy.level_bits)))
    count = int(torch.randint(0, len(units) + 1, (1,), generator=generator))
    strategy.apply(states, ordered.take(torch.arange(count, device=device)))
    return states


# The reference


@pytest.mark.parametrize("device", DEVICES)
def test_the_reference_recomputation_is_the_models_own_forward(device):
    model, layer, sample, captured, _ = build(device, seed=1)
    for key in ("R", "m", "y", "h"):
        assert torch.equal(sample.reference[key], captured[key]), key
    assert torch.equal(sample.reference["logits"], captured["logits"])
    assert sample.token == int(torch.argmax(captured["logits"]))
    # The intermediates are the call's own: their combine gives its output, and the real forward is close to them.
    combined = sample.reference["z"].view(1, len(sample.experts), -1).sum(dim=1).to(BF16)[0]
    assert torch.equal(combined, sample.reference["R"])
    assert torch.allclose(sample.real["y"], sample.reference["y"].to(torch.float64), rtol=0.05, atol=0.05)


# Enclosures


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tier", ["certified", "rn_even", "real"])
def test_enclosures_contain_the_reference_at_every_state(device, tier):
    _, layer, sample, _, lm_weight = build(device, seed=2)
    weights = oracle.SampleWeights(layer, sample.experts)
    generator = torch.Generator().manual_seed(2)
    arithmetic = oracle.ARITHMETIC[tier]
    for strategy in STRATEGIES:
        for _ in range(3):
            states = random_states(strategy, layer, len(sample.experts), device, generator)
            bounds = propagate(sample, weights, strategy, states, tier)
            assert sum(oracle.enclosure_violations(bounds, sample, arithmetic.real).values()) == 0, strategy.name


def propagate(sample, weights, strategy, states, tier="certified", bound=oracle.BoundTier.REALISTIC):
    return oracle.propagate(sample, strategy.knowledge(weights, states), oracle.ARITHMETIC[tier], bound, NORMS[id(sample)], EPS[id(sample)], weights.column_norms)


@pytest.mark.parametrize("device", DEVICES)
def test_the_mixture_bound_never_exceeds_the_reference_difference(device):
    """The decomposed lower bound on Δ·y (every term, every state) is at most the reference's own Δ·y."""
    _, layer, sample, _, lm_weight = build(device, seed=3)
    weights = oracle.SampleWeights(layer, sample.experts)
    generator = torch.Generator().manual_seed(3)
    y = sample.reference["y"].to(torch.float64)
    gain = NORMS[id(sample)].to(torch.float64)
    rows = torch.arange(lm_weight.shape[0], device=device)
    for strategy in STRATEGIES:
        for _ in range(2):
            bounds = propagate(sample, weights, strategy, random_states(strategy, layer, len(sample.experts), device, generator))
            candidate = sample.token
            result = pairwise.pairwise_certificate(
                candidate, rows, lm_weight, torch.full((rows.numel(),), -64.0, dtype=torch.float64, device=device),
                torch.full((rows.numel(),), 64.0, dtype=torch.float64, device=device), NORMS[id(sample)], bounds.y,
                bounds.norm.scale_lower, 0.0, CERTIFIED, mixture=bounds.mixture,
            )
            delta = (lm_weight[candidate].to(torch.float64)[None, :] - lm_weight[result.contenders].to(torch.float64)) * gain[None, :]
            truth = delta @ y
            slack = 1e-12 * (delta.abs() @ y.abs())
            assert bool((result.decomposed_bound <= truth + slack).all()), strategy.name
            assert bool((result.box_bound <= truth + slack).all())


# Certificates


@pytest.mark.parametrize("device", DEVICES)
def test_a_certificate_names_the_reference_token_and_needs_its_margin(device):
    for margin, expected in ((200.0, True), (None, False)):
        model, layer, sample, _, lm_weight = build(device, seed=4, margin=margin)
        weights = oracle.SampleWeights(layer, sample.experts)
        certifier = oracle.Certifier(lm_weight, NORMS[id(sample)], sample.reference["logits"], oracle.CERTIFIED_ARITHMETIC)
        outcome, _, violations = oracle.ceiling(sample, weights, oracle.CERTIFIED_ARITHMETIC, certifier, NORMS[id(sample)], EPS[id(sample)])
        assert sum(violations.values()) == 0
        assert outcome.certified is expected
        if outcome.certified:
            assert outcome.candidate == sample.token
            for strategy in STRATEGIES:
                result = oracle.run_cell(sample, weights, strategy, oracle.CERTIFIED_ARITHMETIC, oracle.BoundTier.REALISTIC,
                                         oracle.Ordering.REALISTIC, certifier, 1 / 16, NORMS[id(sample)], EPS[id(sample)], offsets_for(layer))
                assert result.certified and result.winner == sample.token, strategy.name
                assert sum(result.violations.values()) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_partial_knowledge_never_certifies_more_than_full_knowledge(device):
    """With realistic bounds, more rows read only narrow the enclosures: the lower bounds on Δ·y (box and decomposed)
    only grow, and so does the certificate (a margin is (bound − slack)·q⁻ − ..., so a negative one may shrink as q⁻
    grows, but a positive one never turns negative). This is what lets a cell search budgets by bisection."""
    for margin in (40.0, 400.0):
        _, layer, sample, _, lm_weight = build(device, seed=5, margin=margin)
        weights = oracle.SampleWeights(layer, sample.experts)
        generator = torch.Generator().manual_seed(5)
        rows = torch.arange(lm_weight.shape[0], device=device)
        bounds_of = lambda b: pairwise.pairwise_certificate(  # noqa: E731
            sample.token, rows, lm_weight, torch.full((rows.numel(),), -64.0, dtype=torch.float64, device=device),
            torch.full((rows.numel(),), 64.0, dtype=torch.float64, device=device), NORMS[id(sample)], b.y, b.norm.scale_lower,
            0.0, CERTIFIED, mixture=b.mixture,
        )
        for strategy in STRATEGIES:
            states = strategy.initial(layer, len(sample.experts), device)
            units = strategy.units(layer, states)
            ordered = oracle.order_units(units, torch.rand(len(units), generator=generator, dtype=torch.float64).to(device), max(1, len(strategy.level_bits)))
            previous = None
            for count in (0, len(units) // 3, 2 * len(units) // 3, len(units)):
                current = strategy.initial(layer, len(sample.experts), device)
                strategy.apply(current, ordered.take(torch.arange(count, device=device)))
                result = bounds_of(propagate(sample, weights, strategy, current))
                if previous is not None:
                    for name in ("box_bound", "decomposed_bound"):
                        now, before = getattr(result, name), getattr(previous, name)
                        assert bool((now >= before - 1e-9 * before.abs().clamp_min(1.0)).all()), (strategy.name, name)
                    assert bool((result.margin > 0)[previous.margin > 0].all()), strategy.name
                previous = result


# Bytes


def test_byte_accounting_of_every_strategy():
    _, layer, sample, _, _ = build("cpu", seed=6)
    slots = len(sample.experts)
    element = layer.element_size
    expert_bf16 = layer.expert_bytes
    for strategy in STRATEGIES:
        states = strategy.initial(layer, slots, "cpu")
        before = strategy.bytes_read(layer, states)
        units = strategy.units(layer, states)
        strategy.apply(states, units)
        after = strategy.bytes_read(layer, states)
        read = lambda totals: int(sum(totals[k].sum() for k in ("gate", "up", "down", "levels")))  # noqa: E731
        assert read(after) - read(before) == int(units.nbytes.sum()), strategy.name  # each unit's bytes, once
        assert int(after["unread"].sum()) == 0
        assert int((after["gate"] + after["up"] + after["down"]).sum()) == slots * expert_bf16
        levels = sum(layer.rows(name) * sum(strategy.level_row_bytes(layer.width(name))) for name in oracle.MATRICES)
        assert int(after["levels"].sum()) == slots * levels
        assert strategy.fallback_bytes(layer, slots) == read(after) + strategy.metadata_bytes(layer, slots)
        # The modelled I/O of everything read: every block of the experts' tensors (neuron-major: also the copy's).
        physical = oracle.physical_estimate(strategy, layer, sample.experts, states, offsets_for(layer), fallback=False)
        assert physical["checkpoint"]["logical_bytes"] == slots * expert_bf16
        assert physical["checkpoint"]["physical_bytes"] >= slots * expert_bf16
    d = next(s for s in STRATEGIES if s.spec == "q6+q4")
    assert d.level_row_bytes(layer.hidden) == [math.ceil(layer.hidden * 6 / 8) + 4, math.ceil(layer.hidden * 4 / 8) + 4]
    assert d.storage_multiplier(layer) > 1.6  # the levels are stored next to the BF16 rows
    assert next(s for s in STRATEGIES if s.family == "neuron_major").storage_multiplier(layer) == pytest.approx(4 / 3)
    del element


def test_blocks_and_extents_count_touched_4k_blocks():
    assert oracle.blocks_and_extents([("f", 0, 4096)]) == (1, 1)
    assert oracle.blocks_and_extents([("f", 100, 4200)]) == (2, 1)
    assert oracle.blocks_and_extents([("f", 0, 10), ("f", 8192, 8200)]) == (2, 2)
    assert oracle.blocks_and_extents([("f", 0, 10), ("g", 0, 10)]) == (2, 2)


# Orderings


def test_the_realistic_ordering_sees_no_weight_value():
    """Neuron-major pages, nothing read: the realistic ordering may use the down columns' norms and the activations'
    bounds only. Permuting the rows of every down projection keeps both and moves the values: the realistic gains stay,
    the ideal ones (true contributions to the reference's top-2 difference) move."""
    _, layer, sample, _, lm_weight = build("cpu", seed=7)
    strategy = next(s for s in STRATEGIES if s.family == "neuron_major")
    states = strategy.initial(layer, len(sample.experts), "cpu")
    units = strategy.units(layer, states)
    delta = oracle.pair_delta(lm_weight, NORMS[id(sample)], sample.token, sample.runner_up)

    def gains(weights):
        bounds = propagate(sample, weights, strategy, states)
        return oracle.realistic_gains(units, strategy, weights, bounds, sample.weights, delta), oracle.ideal_gains(units, strategy, weights, sample, delta, False)

    weights = oracle.SampleWeights(layer, sample.experts)
    realistic, ideal = gains(weights)
    permuted = oracle.SampleWeights(layer, sample.experts)
    permuted.truth["down"] = permuted.truth["down"][:, torch.randperm(layer.hidden, generator=torch.Generator().manual_seed(7)), :]
    permuted_realistic, permuted_ideal = gains(permuted)
    assert torch.equal(realistic, permuted_realistic)
    assert not torch.equal(ideal, permuted_ideal)
    assert torch.equal(oracle.order_units(units, realistic).index, oracle.order_units(units, realistic.clone()).index)  # deterministic


def test_refinement_steps_stay_in_order_in_every_prefix():
    _, layer, sample, _, _ = build("cpu", seed=8)
    strategy = next(s for s in STRATEGIES if s.spec == "q6+q4")
    states = strategy.initial(layer, len(sample.experts), "cpu")
    units = strategy.units(layer, states)
    generator = torch.Generator().manual_seed(8)
    ordered = oracle.order_units(units, torch.rand(len(units), generator=generator, dtype=torch.float64), 2)
    seen: dict[tuple[int, int, int], int] = {}
    for kind, slot, index, step in zip(ordered.kind.tolist(), ordered.slot.tolist(), ordered.index.tolist(), ordered.step.tolist()):
        rank = 1 if step == oracle.EXACT else 0
        assert seen.get((kind, slot, index), -1) < rank  # the level step comes before the exact step
        seen[(kind, slot, index)] = rank
    count = len(ordered) // 2
    strategy.apply(states, ordered.take(torch.arange(count)))
    assert bool(((states.gate == 1) | (states.gate == 0) | (states.gate == oracle.EXACT)).all())


# Arithmetic of the full check


@pytest.mark.parametrize("device", DEVICES)
def test_the_binary32_projection_error_bound_holds(device):
    generator = torch.Generator().manual_seed(9)
    for scale in (1e-3, 1.0, 50.0):
        delta = (torch.randn(64, 256, generator=generator, dtype=torch.float64) * scale).to(device)
        known = (torch.randn(256, 48, generator=generator, dtype=torch.float64) * 0.05).to(BF16).to(torch.float64).to(device)
        term = pairwise.MixtureTerm(1.0, known, Enclosure.exact(torch.zeros(48, dtype=torch.float64, device=device)), torch.zeros(256, dtype=torch.float64, device=device))
        mixed, error = oracle.Float32Projection()(delta, term)
        exact = delta @ known
        bound = error.rows[:, None] * error.columns[None, :]
        assert bool(((mixed - exact).abs() <= bound).all())


@pytest.mark.parametrize("device", DEVICES)
def test_relaxing_the_mixture_never_strengthens_the_certificate(device):
    _, layer, sample, _, lm_weight = build(device, seed=10, margin=30.0)
    weights = oracle.SampleWeights(layer, sample.experts)
    strategy = next(s for s in STRATEGIES if s.spec == "q8")
    states = strategy.initial(layer, len(sample.experts), device)
    bounds = propagate(sample, weights, strategy, states)
    certifier = oracle.Certifier(lm_weight, NORMS[id(sample)], sample.reference["logits"], oracle.CERTIFIED_ARITHMETIC)
    certifier.candidate = sample.token
    rows = torch.arange(lm_weight.shape[0], device=device)
    rows = rows[rows != sample.token]
    full, _ = certifier.margins(bounds, bounds.mixture, rows, oracle.exact_projection)
    for keep in (0, 8, layer.intermediate):
        masks = []
        for term in bounds.mixture.terms:
            mask = torch.zeros(term.activation.lower.numel(), dtype=torch.bool, device=device)
            mask[:keep] = True
            masks.append(mask)
        relaxed, _ = certifier.margins(bounds, pairwise.relax_mixture(bounds.mixture, masks), rows, oracle.Float32Projection())
        assert bool((relaxed <= full + 1e-9 * full.abs().clamp_min(1.0)).all())


# Sabotage


@pytest.mark.parametrize("device", DEVICES)
def test_dropping_the_unread_part_is_caught(device, monkeypatch):
    """Guard check: an enclosure without the unknown part's bound misses reference values."""
    _, layer, sample, _, _ = build(device, seed=11)
    weights = oracle.SampleWeights(layer, sample.experts)
    strategy = next(s for s in STRATEGIES if s.family == "down_rows")
    states = strategy.initial(layer, len(sample.experts), device)
    monkeypatch.setattr(oracle, "remainder_mass", lambda knowledge, magnitude, bound, reference, gamma_acc: torch.zeros(knowledge.known.shape[:2], dtype=torch.float64, device=knowledge.known.device))
    bounds = propagate(sample, weights, strategy, states)
    assert sum(oracle.enclosure_violations(bounds, sample, False).values()) > 0


@pytest.mark.parametrize("device", DEVICES)
def test_dropping_the_experts_rounding_from_the_mixture_is_caught(device, monkeypatch):
    """Guard check: without the experts' output roundings and accumulation, the decomposed bound on Δ·y exceeds the
    reference's own value for some contender."""
    _, layer, sample, _, lm_weight = build(device, seed=12)
    weights = oracle.SampleWeights(layer, sample.experts)
    strategy = next(s for s in STRATEGIES if s.family == "neuron_pages")
    states = oracle.ExpertStates.filled(layer, len(sample.experts), device, oracle.EXACT, columns=True)
    bounds = propagate(sample, weights, strategy, states)
    errors = {name: torch.zeros_like(vector) for name, vector in bounds.mixture.errors.items()}
    narrowed = pairwise.ExpertMixture(bounds.mixture.base, bounds.mixture.terms, errors)
    zero_activation = [pairwise.MixtureTerm(t.weight, t.known, Enclosure.exact(t.activation.center), t.row_remainder) for t in narrowed.terms]
    narrowed = pairwise.ExpertMixture(narrowed.base, tuple(zero_activation), errors)
    rows = torch.arange(lm_weight.shape[0], device=device)
    gain = NORMS[id(sample)].to(torch.float64)
    delta = (lm_weight[sample.token].to(torch.float64)[None, :] - lm_weight[rows].to(torch.float64)) * gain[None, :]
    part = pairwise.mixture_bound(delta, narrowed)
    truth = delta @ sample.reference["y"].to(torch.float64)
    assert bool((part.exact_and_read - part.unread > truth).any())
