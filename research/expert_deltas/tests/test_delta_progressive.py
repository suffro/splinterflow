"""Bit-plane sets and the exact optimum of the decision over them: brute force on toys, soundness, witnesses, poisoning."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from expert_deltas import bits
from expert_deltas.progressive import (
    SILU_ARGMIN,
    STEP_PREFIX,
    Box,
    ExpertSet,
    activation_range,
    box,
    exact_minimum,
    linf_bounds,
    read_view,
    silu,
    witness,
)


def bf16_patterns(rng, shape, scale):
    values = torch.from_numpy(rng.normal(0, scale, shape).astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(bits.patterns(values).astype(np.int64))


def toy(seed, hidden=3, inner=2, experts=1, prefix=10):
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.normal(0, 1, hidden)).to(torch.bfloat16).to(torch.float64)
    sets, truths = [], []
    for e in range(experts):
        mats = {"gate": bf16_patterns(rng, (inner, hidden), 0.5), "up": bf16_patterns(rng, (inner, hidden), 0.5),
                "down": bf16_patterns(rng, (hidden, inner), 0.5)}
        boxes = {m: box(read_view(p, torch.full((p.shape[0],), prefix)), torch.full((p.shape[0],), prefix)) for m, p in mats.items()}
        sets.append(ExpertSet(boxes["gate"], boxes["up"], boxes["down"], float(rng.uniform(0.2, 1.0))))
        truths.append({m: bits.pattern_values(p) for m, p in mats.items()})
    base = torch.from_numpy(rng.normal(0, 1, hidden))
    delta = torch.from_numpy(rng.normal(0, 1, (4, hidden)))
    return sets, truths, x, base, delta


def forward(truths, sets, x, base):
    y = base.clone()
    for t, s in zip(truths, sets):
        y = y + s.weight * (t["down"] @ (silu(t["gate"] @ x) * (t["up"] @ x)))
    return y


def brute_force(sets, x, base, delta):
    """min of Δ·y over the set by enumeration: every down vertex, and per neuron g ∈ {g⁻, g⁺, g* if inside}, u ∈ {u⁻, u⁺}
    (for a fixed down and u, the minimum over g of c·silu(g) is at an end or at silu's minimum)."""
    out = delta @ base
    for s in sets:
        act = activation_range(s.gate, s.up, x)
        options = []
        for i in range(act.g[0].numel()):
            gs = [float(act.g[0][i]), float(act.g[1][i])]
            if gs[0] <= SILU_ARGMIN <= gs[1]:
                gs.append(SILU_ARGMIN)
            options.append([float(silu(torch.tensor(g, dtype=torch.float64))) * u for g in gs for u in (float(act.u[0][i]), float(act.u[1][i]))])
        lo, hi = s.down.lower, s.down.upper
        best = torch.full((delta.shape[0],), float("inf"), dtype=torch.float64)
        flat = lo.numel()
        for vertex in itertools.product((0, 1), repeat=flat):
            D = torch.where(torch.tensor(vertex).view(lo.shape).bool(), hi, lo)
            for a in itertools.product(*options):
                a = torch.tensor(a, dtype=torch.float64)
                best = torch.minimum(best, delta @ (D @ a))
        out = out + s.weight * best
    return out


@pytest.mark.parametrize("seed", range(6))
def test_exact_minimum_equals_brute_force(seed):
    sets, _, x, base, delta = toy(seed, prefix=10 + seed % 4)
    found = exact_minimum(sets, x, base, delta)
    expected = brute_force(sets, x, base, delta)
    assert torch.allclose(found.value, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_random_points_of_the_set_never_go_below_the_minimum_and_the_witness_attains_it(seed):
    sets, truths, x, base, delta = toy(seed, hidden=6, inner=4, experts=2, prefix=9 + seed)
    found = exact_minimum(sets, x, base, delta)
    truth = delta @ forward(truths, sets, x, base)
    assert bool((found.value <= truth + 1e-12).all())  # the true weights lie in the set
    generator = torch.Generator().manual_seed(seed)
    for _ in range(300):
        sample = [{m: getattr(s, m).lower + torch.rand(getattr(s, m).centre.shape, generator=generator, dtype=torch.float64) * 2 * getattr(s, m).half
                   for m in ("gate", "up", "down")} for s in sets]
        assert bool((found.value <= delta @ forward(sample, sets, x, base) + 1e-12).all())
    for j in range(delta.shape[0]):
        w = witness(sets, x, base, delta[j], found, j)
        assert w.inside
        assert abs(w.value - float(found.value[j])) <= 1e-9 * (1 + float(found.magnitude[j]))


def test_the_set_never_depends_on_an_unread_bit():
    rng = np.random.default_rng(7)
    p = bf16_patterns(rng, (16, 64), 0.02)
    for t in STEP_PREFIX:
        prefix = torch.full((16,), t)
        clean = box(read_view(p, prefix), prefix)
        for seed in range(3):
            poisoned = box(read_view(p, prefix, seed=seed), prefix)
            assert torch.equal(clean.centre, poisoned.centre) and torch.equal(clean.half, poisoned.half)
        assert clean.contains(bits.pattern_values(p))


def test_more_planes_only_shrink_the_set_and_raise_the_minimum():
    rng = np.random.default_rng(8)
    sets_by_t, minima = [], []
    x = torch.from_numpy(rng.normal(0, 1, 8)).to(torch.bfloat16).to(torch.float64)
    mats = {"gate": bf16_patterns(rng, (5, 8), 0.3), "up": bf16_patterns(rng, (5, 8), 0.3), "down": bf16_patterns(rng, (8, 5), 0.3)}
    base = torch.from_numpy(rng.normal(0, 1, 8))
    delta = torch.from_numpy(rng.normal(0, 1, (3, 8)))
    linf = {m: linf_bounds(p) for m, p in mats.items()}
    for t in STEP_PREFIX:
        boxes = {m: box(read_view(p, torch.full((p.shape[0],), t)), torch.full((p.shape[0],), t), linf[m]) for m, p in mats.items()}
        sets_by_t.append(boxes)
        minima.append(exact_minimum([ExpertSet(boxes["gate"], boxes["up"], boxes["down"], 0.7)], x, base, delta).value)
    for before, after in zip(sets_by_t, sets_by_t[1:]):
        for m in mats:
            assert bool((before[m].lower <= after[m].lower).all() and (after[m].upper <= before[m].upper).all())
    for before, after in zip(minima, minima[1:]):
        assert bool((after >= before - 1e-12).all())
    truth = delta @ (base + 0.7 * bits.pattern_values(mats["down"]) @ (silu(bits.pattern_values(mats["gate"]) @ x) * (bits.pattern_values(mats["up"]) @ x)))
    assert torch.allclose(minima[-1], truth, rtol=1e-12, atol=1e-12)  # every bit read: the minimum is the truth


def test_linf_metadata_holds_the_true_rows():
    rng = np.random.default_rng(9)
    p = bf16_patterns(rng, (32, 128), 0.05)
    n = linf_bounds(p)
    assert bool((bits.pattern_values(p).abs().amax(dim=1) <= n).all())
    unread = box(read_view(p, torch.zeros(32, dtype=torch.int64)), torch.zeros(32, dtype=torch.int64), n)
    assert torch.equal(unread.half, n[:, None].expand(32, 128)) and unread.contains(bits.pattern_values(p))


def test_a_flipping_witness_is_found_when_the_minimum_is_negative():
    """With nothing but sign and exponent read the set is wide: the minimum is negative and its witness flips the pair."""
    sets, truths, x, base, delta = toy(11, hidden=6, inner=4, experts=2, prefix=9)
    base = base * 0.0
    found = exact_minimum(sets, x, base, delta)
    j = int(torch.argmin(found.value))
    assert float(found.value[j]) < 0
    w = witness(sets, x, base, delta[j], found, j)
    assert w.inside and w.value < 0
    assert isinstance(sets[0].gate, Box)


@pytest.mark.parametrize("seed", range(3))
def test_the_sketch_bound_is_sound_and_no_weaker_than_ignoring_it(seed):
    """Sketches through any bases: the relaxation never exceeds the truth or any point of the set satisfying the sketch,
    and with an exact basis of the whole space it recovers the truth's input part exactly."""
    from expert_deltas.progressive import Sketch, sketched_minimum

    sets, truths, x, base, delta = toy(seed, hidden=6, inner=4, experts=2, prefix=10)
    rng = np.random.default_rng(seed)
    U = torch.linalg.qr(torch.from_numpy(rng.normal(0, 1, (6, 2))))[0]
    V = torch.linalg.qr(torch.from_numpy(rng.normal(0, 1, (6, 3))))[0]
    sketch = Sketch.build(U, V, truths)
    bound = sketched_minimum(sets, x, base, delta, sketch)
    truth = delta @ forward(truths, sets, x, base)
    assert bool((bound.value <= truth + 1e-12).all())
    plain = exact_minimum(sets, x, base, delta)
    # Both enclosures hold, so the sketch can only raise the bound (up to float64 rounding).
    assert bool((bound.value >= plain.value - 1e-9 * (1 + plain.magnitude)).all())
    # A full basis of the input space: g and u are known up to the sketch's float32 rounding.
    full = Sketch.build(torch.eye(6, dtype=torch.float64), torch.eye(6, dtype=torch.float64), truths)
    exact = sketched_minimum(sets, x, base, delta, full)
    assert torch.allclose(exact.value, truth, rtol=1e-5, atol=1e-5)
