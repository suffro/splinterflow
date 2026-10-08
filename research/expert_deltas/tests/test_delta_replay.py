"""The replay charges every byte a cache miss reads: experts, deltas and bases, cold and warm."""

from __future__ import annotations

from expert_deltas.replay import Representation, replay

MB = 1 << 20


def steps(routes, phase="decode"):
    return [{"prompt_id": p, "step": s, "phase": phase, "routed": {1: r}} for p, s, r in routes]


def independent(sizes):
    return Representation("independent", sizes)


def delta(sizes, base, base_stored, base_resident):
    """Every expert but `base` is a delta against one base object ("b"); the base expert itself is stored alone."""
    return Representation("delta", sizes, [() if e == base else ("b",) for e in range(len(sizes))], {"b": base_stored}, {"b": base_resident})


def test_no_cache_reads_every_routed_expert_every_step():
    trace = steps([(0, 1, [0, 1]), (0, 2, [1, 2]), (1, 1, [0, 2])])
    out = replay(trace, {1: independent([3 * MB, 5 * MB, 7 * MB])}, 0, "host", 0)
    assert out["decode_bytes"] == ((3 + 5) + (5 + 7) + (3 + 7)) * MB / 3
    assert out["cold_start_bytes"] == 8 * MB and out["hits"] == 0


def test_warm_cache_reads_each_expert_once():
    trace = steps([(0, 1, [0, 1]), (0, 2, [1, 2]), (1, 1, [0, 2])])
    out = replay(trace, {1: independent([3 * MB, 5 * MB, 7 * MB])}, 100 * MB, "host", 1)
    assert [out["decode_bytes"] * 3] == [15 * MB] and out["steady_decode_bytes"] == 0 and out["hits"] == 3


def test_pinned_base_is_charged_once_at_cold_start():
    trace = steps([(0, 1, [1, 2]), (0, 2, [2, 3])])
    rep = delta([10 * MB, 2 * MB, 2 * MB, 2 * MB], base=0, base_stored=10 * MB, base_resident=16 * MB)
    out = replay(trace, {1: rep}, 16 * MB, "host", 0)  # the base fits exactly: pinned, nothing else cached
    assert out["bases_pinned_layers"] == 1
    assert out["cold_start_bytes"] == 10 * MB + 4 * MB  # the base, then the first step's two deltas
    assert out["decode_bytes"] == 4 * MB and out["base_misses"] == 0


def test_bounded_base_is_reread_after_eviction():
    """Base (16 MB decoded) and one delta (6 MB) do not fit in 20 MB together: every step evicts and rereads (thrashing)."""
    trace = steps([(0, 1, [1]), (0, 2, [2]), (0, 3, [1])])
    rep = delta([10 * MB, 6 * MB, 6 * MB], base=0, base_stored=10 * MB, base_resident=16 * MB)
    out = replay(trace, {1: rep}, 20 * MB, "bounded", 0)
    assert out["decode_bytes"] == 16 * MB  # each step: its delta (6) and the base again (10 stored)
    assert out["base_misses"] == 3 and out["base_hits"] == 0 and out["evictions"] == 5
    roomy = replay(trace, {1: rep}, 40 * MB, "bounded", 0)  # everything fits: the base is read once
    assert roomy["base_misses"] == 1 and roomy["base_hits"] == 2 and roomy["decode_bytes"] * 3 == (6 + 10 + 6) * MB


def test_an_expert_needing_several_bases_charges_each():
    trace = steps([(0, 1, [1]), (0, 2, [2])])
    rep = Representation("kinds", [9 * MB, 2 * MB, 2 * MB], [(), (("gate", 0), ("down", 0)), (("gate", 0), ("down", 5))],
                         {("gate", 0): 3 * MB, ("down", 0): 3 * MB, ("down", 5): 4 * MB},
                         {("gate", 0): 5 * MB, ("down", 0): 5 * MB, ("down", 5): 5 * MB})
    out = replay(trace, {1: rep}, 100 * MB, "bounded", 0)
    assert out["base_misses"] == 3 and out["base_hits"] == 1
    assert out["decode_bytes"] * 2 == (2 + 3 + 3) * MB + (2 + 4) * MB


def test_gpu_bases_stay_outside_the_host_budget():
    trace = steps([(0, 1, [1, 2])])
    rep = delta([10 * MB, 2 * MB, 2 * MB], base=0, base_stored=10 * MB, base_resident=16 * MB)
    out = replay(trace, {1: rep}, 0, "gpu", 0)
    assert out["gpu_base_bytes"] == 16 * MB and out["base_misses"] == 0
    assert out["cold_start_bytes"] == 10 * MB + 4 * MB
