"""The progressive oracle's schedules see only what is read and resident (no hidden reads); states follow the schedule."""

from __future__ import annotations

import numpy as np
import torch

from expert_deltas import bits
from expert_deltas.oracle import ExpertPages, Sample, Schedule, greedy, prefixes, sequential
from expert_deltas.progressive import exact_minimum


def synthetic(seed: int, mantissa_noise: int | None = None):
    """Six small experts (32 neurons, 32 inputs), an LM head of 50 rows; optionally every weight's low 7 bits replaced
    (bits a runtime has not read at prefix 9)."""
    rng = np.random.default_rng(seed)

    def matrix(shape, scale):
        p = bits.patterns(torch.from_numpy(rng.normal(0, scale, shape).astype(np.float32)).to(torch.bfloat16))
        if mantissa_noise is not None:
            noise = np.random.default_rng(mantissa_noise).integers(0, 128, shape).astype(np.uint16)
            p = (p & 0xFF80) | noise
        return p

    experts = [{"gate": matrix((32, 32), 0.3), "up": matrix((32, 32), 0.3), "down": matrix((32, 32), 0.3)} for _ in range(6)]
    tensors = np.random.default_rng(seed + 100)
    x = torch.from_numpy(tensors.normal(0, 1, 32).astype(np.float32)).to(torch.bfloat16)
    r = torch.from_numpy(tensors.normal(0, 1, 32).astype(np.float32)).to(torch.bfloat16)
    s = torch.from_numpy(tensors.normal(0, 1, 32).astype(np.float32)).to(torch.bfloat16)
    lm = torch.from_numpy(tensors.normal(0, 1, (80, 32)).astype(np.float32)).to(torch.bfloat16)
    gain = torch.from_numpy(tensors.uniform(0.5, 1.5, 32).astype(np.float32)).to(torch.bfloat16)
    weights = tensors.uniform(0.2, 1.0, 6).tolist()
    return experts, x, r, s, lm, gain, weights


def test_the_greedy_order_never_depends_on_an_unread_bit():
    experts, x, r, s, lm, gain, weights = synthetic(0)
    poisoned, *_ = synthetic(0, mantissa_noise=7)
    pages = [ExpertPages(e, threads=2) for e in experts]  # the page index (bytes per page and plane) is resident
    one = Sample(x, r, s, list(range(6)), weights, pages, lm, gain, "cpu")
    other = Sample(x, r, s, list(range(6)), weights, [ExpertPages(e, threads=2) for e in poisoned], lm, gain, "cpu")
    other.linf = one.linf  # resident metadata is what it is; only the hidden source's unread bits are poisoned
    a, b = greedy(pages, one), greedy(pages, other)
    assert a.units == b.units and np.array_equal(a.nbytes, b.nbytes)
    # The sets at a state where every page holds sign and exponent only are equal too (the low bits were never read).
    steps = np.ones((6, 2), dtype=np.int64)
    sa, sb = one.sets(steps, steps, seed=None), other.sets(steps, steps, seed=None)
    for left, right in zip(sa, sb):
        for m in ("gate", "up", "down"):
            assert torch.equal(getattr(left, m).centre, getattr(right, m).centre) and torch.equal(getattr(left, m).half, getattr(right, m).half)


def test_states_follow_the_schedule_and_every_unit_is_charged():
    experts, x, r, s, lm, gain, weights = synthetic(1)
    pages = [ExpertPages(e, threads=2) for e in experts]
    schedule = sequential(pages)
    total = int(schedule.spent[-1])
    assert total == sum(p.total for p in pages)  # reading every step of every page reads every frame once
    neuron, down, spent = schedule.state(total, 6, 2, 2)
    assert (neuron == 8).all() and (down == 8).all() and spent == total
    neuron, down, spent = schedule.state(schedule.spent[10], 6, 2, 2)
    assert spent == schedule.spent[10] and int((neuron > 0).sum() + (down > 0).sum()) == 11
    assert torch.equal(prefixes(np.array([[0, 1, 8]]), 40), torch.tensor([[0] * 16 + [9] * 16 + [16] * 8]))


def test_full_read_gives_the_truth_and_the_reference_token_wins():
    experts, x, r, s, lm, gain, weights = synthetic(2)
    pages = [ExpertPages(e, threads=2) for e in experts]
    sample = Sample(x, r, s, list(range(6)), weights, pages, lm, gain, "cpu")
    full = np.full((6, 2), 8, dtype=np.int64)
    sets = sample.sets(full, full, seed=3)
    logits = sample.centre_logits(sets)
    winner = int(torch.argmax(logits))
    rows = torch.tensor([j for j in range(lm.shape[0]) if j != winner])
    found = exact_minimum(sets, sample.x, sample.base, sample.delta(winner, rows))
    assert torch.allclose(found.value, sample.truth(rows, winner), rtol=1e-12, atol=1e-12)
    assert sample.certify(sets, near=8, full=True, chunk=16)["certified"] == bool((sample.truth(rows, winner) > 0).all())


def test_physical_reads_count_whole_blocks_and_skip_unread_planes():
    from expert_deltas.oracle import physical

    experts, *_ = synthetic(3)
    pages = [ExpertPages(e, threads=2) for e in experts]
    full = np.full((6, 2), 8, dtype=np.int64)
    found = physical(pages, full, full)
    total = sum(p.total for p in pages)
    for layout in ("page_major", "plane_major"):
        assert found[layout]["logical_bytes"] == total
        assert found[layout]["extents"] == 6  # one contiguous run per expert file when everything is read
        assert total <= found[layout]["physical_bytes"] < total + 6 * 4096
    seven = np.full((6, 2), 7, dtype=np.int64)  # the last mantissa plane unread
    partial = physical(pages, seven, seven)
    assert partial["plane_major"]["extents"] == 6  # the skipped plane is one region at each file's end
    assert partial["page_major"]["extents"] >= 6
    assert partial["plane_major"]["logical_bytes"] == partial["page_major"]["logical_bytes"] < total
    nothing = physical(pages, np.zeros((6, 2), dtype=np.int64), np.zeros((6, 2), dtype=np.int64))
    assert nothing["page_major"]["blocks_4k"] == 0 and nothing["plane_major"]["logical_bytes"] == 0


def test_attribution_separates_the_matrices_and_shares_are_fractions():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("progressive_run", Path(__file__).resolve().parents[1] / "progressive_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    experts, x, r, s, lm, gain, weights = synthetic(4)
    pages = [ExpertPages(e, threads=2) for e in experts]
    sample = Sample(x, r, s, list(range(6)), weights, pages, lm, gain, "cpu")
    full = np.full((6, 2), 8, dtype=np.int64)
    winner = int(torch.argmax(sample.centre_logits(sample.sets(full, full, seed=None))))
    rows = torch.tensor([j for j in range(lm.shape[0]) if j != winner][:16])
    truth = sample.truth(rows, winner)
    found = module.attribution(sample, winner, rows, truth, 2, 2)
    for short in ("planes_unread_1", "planes_unread_2"):
        entry = found[short]
        assert entry["all"]["min_gap"] >= entry["gate_up_only"]["min_gap"] - 1e-9 and entry["all"]["min_gap"] >= entry["down_only"]["min_gap"] - 1e-9
        assert all(0.0 < v <= 1.0 for v in entry["top10_share"].values())
    assert found["planes_unread_2"]["all"]["min_gap"] >= found["planes_unread_1"]["all"]["min_gap"] - 1e-9
