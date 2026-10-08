"""Exact BF16 transforms and prefix intervals, exhaustively over the 65,536 patterns."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from expert_deltas import bits

ALL = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
FINITE = ALL[(ALL & 0x7F80) != 0x7F80]
SPECIAL = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0x7F81,
                    0xFFFF, 0x3F80, 0xBF80], dtype=np.uint16)  # ±0, subnormals, smallest normal, ±max, ±inf, NaNs, ±1


def test_views_round_trip_every_pattern():
    tensor = bits.to_bfloat16(ALL)
    assert tensor.dtype == torch.bfloat16
    assert np.array_equal(bits.patterns(tensor), ALL)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_deltas_restore_every_pattern(seed):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 1 << 16, ALL.size, dtype=np.uint32).astype(np.uint16)
    for w, b in ((ALL, base), (ALL, ALL[::-1].copy()), (ALL, np.zeros_like(ALL)), (np.tile(SPECIAL, 16), np.repeat(SPECIAL, 16))):
        assert np.array_equal(bits.xor_restore(bits.xor_delta(w, b), b), w)
        assert np.array_equal(bits.modular_restore(bits.modular_delta(w, b), b), w)


def test_byte_split_and_planes_restore_every_pattern():
    rng = np.random.default_rng(3)
    for p in (ALL, rng.permutation(ALL), SPECIAL, ALL[:13], ALL[:1]):  # counts not divisible by 8 included
        assert np.array_equal(bits.byte_merge(bits.byte_split(p), p.size), p)
        planes = bits.split_planes(p)
        assert {k: len(v) for k, v in planes.items()} == bits.plane_bytes(p.size)
        assert np.array_equal(bits.merge_planes(planes, p.size), p)


def test_byte_split_layout():
    p = np.array([0x1234, 0xABCD], dtype=np.uint16)
    assert bits.byte_split(p) == bytes([0x12, 0xAB, 0x34, 0xCD])


def test_prefix_order():
    assert bits.PREFIX_AFTER == {"sign": 1, "exponent": 9, "m6": 10, "m5": 11, "m4": 12, "m3": 13, "m2": 14, "m1": 15, "m0": 16}


def brute_force_interval(t: int) -> tuple[np.ndarray, np.ndarray]:
    """[min, max] over the finite completions of every pattern's top-t prefix (per pattern of ALL)."""
    values = bits.pattern_values(torch.from_numpy(ALL.astype(np.int64))).numpy()
    finite = (ALL & 0x7F80) != 0x7F80
    groups = (ALL.astype(np.int64) >> (16 - t)) if t < 16 else ALL.astype(np.int64)
    lo = np.full(groups.max() + 1, np.inf)
    hi = np.full(groups.max() + 1, -np.inf)
    np.minimum.at(lo, groups[finite], values[finite])
    np.maximum.at(hi, groups[finite], values[finite])
    return lo[groups], hi[groups]


@pytest.mark.parametrize("t", range(17))
def test_prefix_interval_equals_brute_force(t):
    lo, hi = bits.prefix_interval(torch.from_numpy(FINITE.astype(np.int64)), t)
    expected_lo, expected_hi = brute_force_interval(t)
    finite = (ALL & 0x7F80) != 0x7F80
    assert np.array_equal(lo.numpy(), expected_lo[finite])
    assert np.array_equal(hi.numpy(), expected_hi[finite])


def test_prefix_intervals_contain_the_value_and_shrink():
    p = torch.from_numpy(FINITE.astype(np.int64))
    value = bits.pattern_values(p)
    previous = None
    for t in range(17):
        lo, hi = bits.prefix_interval(p, t)
        assert bool(((lo <= value) & (value <= hi)).all())
        if previous is not None:
            assert bool(((previous[0] <= lo) & (hi <= previous[1])).all())  # more bits: a subset
        previous = (lo, hi)
    assert torch.equal(previous[0], value) and torch.equal(previous[1], value)


def test_prefix_interval_per_weight_lengths_and_xor_equivalence():
    rng = np.random.default_rng(4)
    w = rng.choice(FINITE, 4096)
    base = rng.choice(FINITE, 4096)
    t = torch.from_numpy(rng.integers(0, 17, 4096))
    lo, hi = bits.prefix_interval(torch.from_numpy(w.astype(np.int64)), t)
    for k in range(0, 4096, 97):
        one = bits.prefix_interval(torch.tensor([int(w[k])]), int(t[k]))
        assert float(one[0]) == float(lo[k]) and float(one[1]) == float(hi[k])
    # The top t bits of an XOR delta and of the base give the weight's top t bits.
    delta = bits.xor_delta(w, base)
    for k in range(0, 4096, 101):
        mask = (0xFFFF << (16 - int(t[k]))) & 0xFFFF
        assert (int(delta[k]) & mask) ^ (int(base[k]) & mask) == int(w[k]) & mask


def test_non_finite_refused():
    with pytest.raises(ValueError):
        bits.require_finite(np.array([0x3F80, 0x7F80], dtype=np.uint16))
    bits.require_finite(FINITE)
    with pytest.raises(ValueError):
        bits.prefix_interval(torch.tensor([0x7FC0]), 9)  # a NaN's prefix has no finite completion
