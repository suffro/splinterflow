"""Weight and activation sets (Phase 5A2): they contain the true weights, shrink as rows are refined, and never read an
unread byte (the anti-cheating rule)."""

from __future__ import annotations

import math

import pytest
import torch

from crown_oracle.artifact import EXACT, UNKNOWN, MatrixData
from crown_oracle.sets import Box, activation_box, matrix_box, read_view


def round_up_f32(x: torch.Tensor) -> torch.Tensor:
    candidate = x.to(torch.float32)
    low = candidate.to(torch.float64) < x
    return torch.where(low, torch.nextafter(candidate, torch.full_like(candidate, math.inf)), candidate).to(torch.float64)


def quantize(values: torch.Tensor, bits: int):
    """Decision 0003's per-row symmetric int-b level (awpmi.decomposition.quantization), for synthetic matrices."""
    limit = 2 ** (bits - 1) - 1
    row_max = values.abs().amax(dim=1)
    scales = round_up_f32(torch.nextafter(row_max / limit, torch.full_like(row_max, math.inf)))
    codes = torch.clamp(torch.round(values / scales[:, None]), -limit, limit)
    return codes.to(torch.int8), scales.to(torch.float32)


def synthetic_matrix(seed: int, rows: int = 12, columns: int = 9) -> MatrixData:
    gen = torch.Generator().manual_seed(seed)
    truth = (torch.randn(rows, columns, generator=gen) * 0.05).to(torch.bfloat16)
    remainder = truth.to(torch.float64)
    codes, scales, linf, l2 = [], [], [], []
    for bits in (6, 4):
        c, s = quantize(remainder, bits)
        remainder = remainder - c.to(torch.float64) * s.to(torch.float64)[:, None]
        codes.append(c)
        scales.append(s)
        linf.append(round_up_f32(remainder.abs().amax(dim=1)).to(torch.float32))
        l2.append(round_up_f32(remainder.norm(dim=1) * (1 + 1e-12)).to(torch.float32))
    own = truth.to(torch.float64)
    return MatrixData(truth, tuple(codes), tuple(scales), tuple(linf), tuple(l2), round_up_f32(own.abs().amax(dim=1)).to(torch.float32),
                      round_up_f32(own.norm(dim=1) * (1 + 1e-12)).to(torch.float32))


def box_at(data: MatrixData, states: torch.Tensor) -> Box:
    return matrix_box(data, states, read_view(data.truth, states))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_every_state_contains_the_truth_and_refinement_only_shrinks(seed):
    data = synthetic_matrix(seed)
    rows = data.truth.shape[0]
    previous = None
    for state in (UNKNOWN, 0, 1, EXACT):
        box = box_at(data, torch.full((rows,), state))
        assert box.contains(data.truth), state
        assert bool(((box.lower <= box.centre) & (box.centre <= box.upper)).all())
        if previous is not None:
            assert previous.includes(box), state
        previous = box
    assert torch.equal(previous.lower, previous.upper) and torch.equal(previous.lower, data.truth.to(torch.float64))


def test_mixed_states_follow_each_row():
    data = synthetic_matrix(4)
    states = torch.tensor([UNKNOWN, 0, 1, EXACT] * 3)
    mixed = box_at(data, states)
    for state in (UNKNOWN, 0, 1, EXACT):
        alone = box_at(data, torch.full_like(states, state))
        at = states == state
        assert torch.equal(mixed.lower[at], alone.lower[at]) and torch.equal(mixed.upper[at], alone.upper[at])


@pytest.mark.parametrize("poison", ["nan", "random", "huge"])
def test_a_set_never_reads_an_unread_byte(poison):
    """Anti-cheating: replacing the BF16 values of every row not in state EXACT changes no set."""
    data = synthetic_matrix(5)
    states = torch.tensor([UNKNOWN, 0, 1, EXACT] * 3)
    clean = box_at(data, states)
    unread = (states != EXACT).unsqueeze(-1)
    junk = {"nan": torch.full_like(data.truth, math.nan), "random": torch.randn(data.truth.shape).to(torch.bfloat16),
            "huge": torch.full_like(data.truth, 1e30)}[poison]
    dirty = MatrixData(torch.where(unread, junk, data.truth), *[getattr(data, f) for f in ("codes", "scales", "remainder_linf", "remainder_l2", "own_linf", "own_l2")])
    other = box_at(dirty, states)
    for field in ("lower", "upper", "centre"):
        assert torch.equal(getattr(clean, field), getattr(other, field)), field


def test_the_read_view_hides_unread_rows():
    data = synthetic_matrix(6)
    states = torch.tensor([UNKNOWN, 0, 1, EXACT] * 3)
    view = read_view(data.truth, states)
    assert bool(torch.isnan(view[states != EXACT]).all())
    assert torch.equal(view[states == EXACT], data.truth[states == EXACT].to(torch.float64))


def test_activation_boxes_intersect_every_level_reached():
    levels = torch.tensor([0, 1, EXACT])
    lower = torch.tensor([[[-1.0, -1.0]], [[-0.5, -2.0]], [[-0.1, -0.2]]])
    upper = torch.tensor([[[1.0, 1.0]], [[0.4, 2.0]], [[0.1, 0.3]]])
    box = activation_box(levels, lower, upper, torch.tensor([[1, EXACT]]))
    assert box.lower.tolist() == [[-0.5, -0.2]] and box.upper.tolist() == [[0.4, 0.3]]
    with pytest.raises(ValueError):
        activation_box(levels, lower, upper, torch.tensor([[UNKNOWN, 0]]))
