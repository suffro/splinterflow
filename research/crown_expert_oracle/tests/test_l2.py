"""Stage 1.5 (Phase 5A2): the L2 sets from Phase 5A's remainder norms, auto_LiRPA's bounds on them, and the sets' exact
optima. The sets contain the true weights and never read an unread byte; split per-row balls are exact and independent
with an exact input; on the reduced graph the box-hull mode stays below the set's exact minimum while upstream's
default mode does not (pinned); the closed forms and witnesses of `rowsets` against brute force."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from crown_oracle import rowsets
from crown_oracle.artifact import EXACT, UNKNOWN, MatrixData
from crown_oracle.graph import Bounder, boxed_input
from crown_oracle.l2 import LinearReduced, SplitLinear, StackedLinear, l2_interval_mode
from crown_oracle.sets import matrix_balls, matrix_box, read_view
from test_sets import synthetic_matrix

CPU = torch.device("cpu")


def balls_at(data: MatrixData, states: torch.Tensor):
    return matrix_balls(data, states, read_view(data.truth, states))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_state_s_balls_contain_the_truth_and_only_add_constraints(seed):
    data = synthetic_matrix(seed)
    rows = data.truth.shape[0]
    previous = None
    for state in (UNKNOWN, 0, 1, EXACT):
        balls = balls_at(data, torch.full((rows,), state))
        assert balls.contains(data.truth), state
        if previous is not None:  # every earlier ball is still there, about the same centre, no wider
            for (c0, r0), (c1, r1) in zip(previous.extra, balls.extra):
                kept = torch.isfinite(r0) & (r0 > 0)
                assert bool((r1[kept] <= r0[kept]).all())
                assert torch.equal(c0[kept & (r1 > 0)], c1[kept & (r1 > 0)])
        previous = balls
    assert torch.equal(previous.centre, data.truth.to(torch.float64)) and bool((previous.radius == 0).all())


@pytest.mark.parametrize("poison", ["nan", "random", "huge"])
def test_the_balls_never_read_an_unread_byte(poison):
    """Anti-cheating: replacing the BF16 values of every row not in state EXACT changes no ball."""
    data = synthetic_matrix(5)
    states = torch.tensor([UNKNOWN, 0, 1, EXACT] * 3)
    clean = balls_at(data, states)
    unread = (states != EXACT).unsqueeze(-1)
    junk = {"nan": torch.full_like(data.truth, math.nan), "random": torch.randn(data.truth.shape).to(torch.bfloat16),
            "huge": torch.full_like(data.truth, 1e30)}[poison]
    dirty = MatrixData(torch.where(unread, junk, data.truth), *[getattr(data, f) for f in ("codes", "scales", "remainder_linf", "remainder_l2", "own_linf", "own_l2")])
    other = balls_at(dirty, states)
    assert torch.equal(clean.centre, other.centre) and torch.equal(clean.radius, other.radius)
    for (a, b), (c, d) in zip(clean.extra, other.extra):
        assert torch.equal(a, c) and torch.equal(b, d)


def test_the_l2_crown_bound_does_not_depend_on_unread_bytes():
    """The reduced graph's bound is a function of the balls only: poisoned unread rows give the same bound."""
    data = synthetic_matrix(7, rows=6, columns=4)
    states = torch.tensor([0, 1, EXACT, 0, UNKNOWN, 1])
    unread = (states != EXACT).unsqueeze(-1)
    dirty = MatrixData(torch.where(unread, torch.full_like(data.truth, 1e30), data.truth), *[getattr(data, f) for f in ("codes", "scales", "remainder_linf", "remainder_l2", "own_linf", "own_l2")])
    a_lo, a_hi = torch.tensor([[-0.2, 0.1, -1.0, 0.3]]), torch.tensor([[0.4, 0.5, -0.2, 0.9]])
    C = torch.randn(3, 6, generator=torch.Generator().manual_seed(1))
    bounds = []
    for matrix in (data, dirty):
        balls = balls_at(matrix, states)
        with l2_interval_mode(True):
            model = LinearReduced([SplitLinear(balls.centre, balls.radius, 1)], [0.7], torch.zeros(6))
            bounds.append(Bounder(model, (boxed_input(a_lo, a_hi),), CPU).lower(C, "crown", chunk=3))
    assert torch.equal(bounds[0], bounds[1])


@pytest.mark.parametrize("box_hull", [False, True])
def test_split_balls_with_an_exact_input_are_exact_and_independent(box_hull):
    """Every row its own ball, exact input: CROWN returns Σ_r (c_r·W₀_r·x − ρ_r·|c_r|·‖x‖₂), not a joint ball's bound."""
    gen = torch.Generator().manual_seed(3)
    W0, x, C = torch.randn(5, 4, generator=gen), torch.randn(1, 4, generator=gen), torch.randn(2, 5, generator=gen)
    rho = 0.05 + 0.2 * torch.rand(5, generator=gen)
    v = x.reshape(-1)
    with l2_interval_mode(box_hull):
        lower = Bounder(SplitLinear(W0, rho, 1), (x,), CPU).lower(C, "crown", chunk=2)
    expected = C @ W0 @ v - (C.abs() @ rho) * torch.linalg.vector_norm(v)
    assert torch.allclose(lower, expected, rtol=1e-9, atol=1e-12)


def test_stacking_balls_into_one_weight_is_not_supported_upstream():
    """Pinned upstream behaviour: a weight concatenated from L2-perturbed parameters fails in auto_LiRPA's backward pass,
    so the oracle splits the linear map by group instead."""
    gen = torch.Generator().manual_seed(4)
    W0, x, C = torch.randn(4, 3, generator=gen), torch.randn(1, 3, generator=gen), torch.randn(2, 4, generator=gen)
    with pytest.raises(Exception):
        Bounder(StackedLinear(W0, torch.full((4,), 0.1), 1), (x,), CPU).lower(C, "crown", chunk=2)


def reduced_problem(seed: int, hidden: int = 5, intermediate: int = 4):
    gen = torch.Generator().manual_seed(seed)
    centre = torch.randn(hidden, intermediate, generator=gen)
    rho = 0.1 + 0.4 * torch.rand(hidden, generator=gen)
    a_lo = torch.randn(intermediate, generator=gen)
    a_hi = a_lo + torch.rand(intermediate, generator=gen)
    delta = torch.randn(3, hidden, generator=gen)
    return centre, rho, a_lo, a_hi, delta


def exact_reduced_minimum(centre, rho, a_lo, a_hi, delta):
    M, c = delta @ centre, delta.abs() @ rho
    return rowsets.vertex_enumeration(lambda vertices: (vertices @ M.T) - torch.linalg.vector_norm(vertices, dim=1, keepdim=True) * c, a_lo, a_hi)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_box_hull_crown_on_the_reduced_graph_stays_below_the_l2_minimum(seed):
    centre, rho, a_lo, a_hi, delta = reduced_problem(seed)
    exact = exact_reduced_minimum(centre, rho, a_lo, a_hi, delta)
    with l2_interval_mode(True):
        model = LinearReduced([SplitLinear(centre, rho, 1)], [1.0], torch.zeros(centre.shape[0]))
        bounder = Bounder(model, (boxed_input(a_lo[None], a_hi[None]),), CPU, alpha_iterations=30)
        for method in ("crown", "alpha-crown"):
            assert bool((bounder.lower(delta, method, chunk=3) <= exact + 1e-9).all()), method


def test_upstream_default_mode_is_unsound_for_an_l2_weight_with_an_uncertain_input():
    """Pinned upstream behaviour (why the oracle uses the box-hull mode): with an L2 root's interval bounds at its centre,
    CROWN's product relaxation treats the weight as fixed, and its bound exceeds the set's exact minimum."""
    found = False
    for seed in range(8):
        centre, rho, a_lo, a_hi, delta = reduced_problem(seed)
        exact = exact_reduced_minimum(centre, rho, a_lo, a_hi, delta)
        with l2_interval_mode(False):
            model = LinearReduced([SplitLinear(centre, rho, 1)], [1.0], torch.zeros(centre.shape[0]))
            lower = Bounder(model, (boxed_input(a_lo[None], a_hi[None]),), CPU).lower(delta, "crown", chunk=3)
        found |= bool((lower > exact + 1e-6).any())
    assert found


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_l2_closed_forms_bracket_the_exact_minimum(seed):
    centre, rho, a_lo, a_hi, delta = reduced_problem(seed, intermediate=8)
    exact = exact_reduced_minimum(centre, rho, a_lo, a_hi, delta)
    M, c = delta @ centre, delta.abs() @ rho
    lower = rowsets.l2_decoupled(M, c, a_lo, a_hi)
    attained, vertex = rowsets.l2_vertex_search(M, c, a_lo, a_hi)
    assert bool((lower <= exact + 1e-12).all())
    assert bool((attained >= exact - 1e-12).all())
    assert torch.allclose(attained, rowsets.l2_value(M, c, vertex))
    assert bool(((vertex == a_lo) | (vertex == a_hi)).all())


def test_the_extreme_point_is_the_maximizer_of_its_set():
    """Water-filling against random feasible points of a ball ∩ box (none does better), and the closed forms of a ball
    alone and a box alone."""
    gen = torch.Generator().manual_seed(9)
    centre = torch.randn(4, 6, generator=gen)
    lower, upper = centre - torch.rand(4, 6, generator=gen) * 0.3, centre + torch.rand(4, 6, generator=gen) * 0.3
    radius = 0.25 + 0.1 * torch.rand(4, generator=gen)
    v = torch.randn(6, generator=gen)
    both = rowsets.RowSets(centre, radius, lower, upper)
    best = both.support(v)
    assert bool((both.excess(centre + both.extreme(v)) <= 1e-12).all())
    points = centre[None] + (torch.rand(200_000, 4, 6, generator=gen) * 2 - 1) * 0.3
    inside = (torch.linalg.vector_norm(points - centre[None], dim=2) <= radius[None]) & ((points >= lower[None]) & (points <= upper[None])).all(dim=2)
    values = torch.where(inside, points @ v, torch.full((200_000, 4), -math.inf))
    assert bool((values.max(dim=0).values <= best + 1e-12).all())
    ball = rowsets.RowSets(centre, radius)
    assert torch.allclose(ball.support(v), centre @ v + radius * torch.linalg.vector_norm(v))
    box = rowsets.RowSets((lower + upper) * 0.5, torch.full((4,), math.inf), lower, upper)
    assert torch.allclose(box.support(v), torch.where(v > 0, upper, lower) @ v)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_two_ball_max_is_the_maximizer_of_the_intersection(seed):
    """Against random points of the first ball that lie in the second: none does better, and the point is in both."""
    gen = torch.Generator().manual_seed(seed)
    rows, dims = 4, 5
    c1, r1 = torch.randn(rows, dims, generator=gen), 0.5 + torch.rand(rows, generator=gen)
    c2 = c1 + 0.7 * torch.randn(rows, dims, generator=gen)
    r2 = torch.maximum(0.4 + torch.rand(rows, generator=gen), torch.linalg.vector_norm(c2 - c1, dim=1) - r1 + 0.2)  # overlapping
    v = torch.randn(dims, generator=gen)
    best = rowsets.two_ball_max(c1, r1, c2, r2, v)
    assert bool((torch.linalg.vector_norm(best - c1, dim=1) <= r1 * (1 + 1e-12)).all())
    assert bool((torch.linalg.vector_norm(best - c2, dim=1) <= r2 * (1 + 1e-12)).all())
    direction = torch.randn(200_000, rows, dims, generator=gen)
    direction = direction / torch.linalg.vector_norm(direction, dim=2, keepdim=True)
    points = c1[None] + direction * (torch.rand(200_000, rows, 1, generator=gen) ** (1 / dims)) * r1[None, :, None]
    inside = torch.linalg.vector_norm(points - c2[None], dim=2) <= r2[None]
    values = torch.where(inside, points @ v, torch.full((200_000, rows), -math.inf))
    assert bool((values.max(dim=0).values <= best @ v + 1e-12).all())


def test_a_point_satisfies_every_ball_and_keeps_most_of_the_reach():
    """`RowSets.point` with the row's own-norm ball (the true row on its sphere, as the metadata makes it): every
    constraint holds, and the point reaches far more of the current ball's extreme than the anchor does."""
    gen = torch.Generator().manual_seed(13)
    truth = torch.randn(6, 50, generator=gen)
    centre = truth + 0.05 * torch.randn(6, 50, generator=gen)  # a coarse level's approximation
    radius = torch.linalg.vector_norm(truth - centre, dim=1) * (1 + 1e-9)
    own = torch.linalg.vector_norm(truth, dim=1) * (1 + 1e-12)
    rows = rowsets.RowSets(centre, radius, extra=((torch.zeros_like(truth), own),))
    v = torch.randn(50, generator=gen)
    anchor = rows.anchor(truth)
    point, _ = rows.point(v, anchor)
    assert bool((rows.excess(point) <= 1e-12).all())
    reach = (point - centre) @ v
    assert bool((reach >= 0.9 * radius * torch.linalg.vector_norm(v)).all())  # the own ball only bends the direction
    assert bool((reach > (anchor - centre) @ v).all())


def test_the_activations_range_is_exact_and_witnessed():
    """Every activation end is reached by weights of the rows' sets (the witness's rows), and random points never leave
    the range."""
    gen = torch.Generator().manual_seed(11)
    gate_c, up_c = torch.randn(3, 5, generator=gen), torch.randn(3, 5, generator=gen)
    gate = rowsets.RowSets(gate_c, 0.2 + 0.3 * torch.rand(3, generator=gen))
    up = rowsets.RowSets(up_c, 0.2 + 0.3 * torch.rand(3, generator=gen))
    x = torch.randn(5, generator=gen)
    ranges = rowsets.activation_range(gate, up, x)
    a_lo, a_hi = ranges.a
    down = rowsets.RowSets(torch.randn(2, 3, generator=gen), torch.zeros(2))
    for side in (0.0, 1.0):
        w = rowsets.realize(gate, up, down, x, torch.ones(2), torch.full((3,), side), ranges, {"gate": gate_c, "up": up_c, "down": down.centre})
        assert w.largest_excess <= 1e-12
        assert torch.allclose(w.a, a_hi if side else a_lo, rtol=1e-12, atol=1e-12)
    for _ in range(2000):
        g_rows = gate_c + torch.randn(3, 5, generator=gen) * 0.1
        g_rows = gate_c + (g_rows - gate_c) * torch.clamp(gate.radius / torch.linalg.vector_norm(g_rows - gate_c, dim=1), max=1.0)[:, None]
        u_rows = up_c + torch.randn(3, 5, generator=gen) * 0.1
        u_rows = up_c + (u_rows - up_c) * torch.clamp(up.radius / torch.linalg.vector_norm(u_rows - up_c, dim=1), max=1.0)[:, None]
        a = rowsets.silu(g_rows @ x) * (u_rows @ x)
        assert bool(((a >= a_lo - 1e-12) & (a <= a_hi + 1e-12)).all())


def test_vertex_enumeration_matches_every_point():
    lo, hi = torch.tensor([-1.0, 0.5]), torch.tensor([2.0, 1.5])
    value = rowsets.vertex_enumeration(lambda v: -(v**2).sum(dim=1, keepdim=True), lo, hi)
    corners = torch.tensor(list(itertools.product((-1.0, 2.0), (0.5, 1.5))))
    assert float(value[0]) == float(-(corners**2).sum(dim=1).max())


def test_box_and_balls_from_the_same_metadata_both_hold_the_truth():
    data = synthetic_matrix(12)
    states = torch.tensor([0, 1, EXACT, 0] * 3)
    view = read_view(data.truth, states)
    box, balls = matrix_box(data, states, view), matrix_balls(data, states, view)
    resident = rowsets.RowSets(balls.centre, balls.radius, box.lower, box.upper, balls.extra)
    assert bool((resident.excess(data.truth.to(torch.float64)) <= 0).all())
