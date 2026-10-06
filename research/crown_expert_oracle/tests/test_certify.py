"""The certificate's assembly helpers (Phase 5A2), repeated from awpmi because the verifier imports none of it: the BF16
grid functions against brute force over every finite BF16 value. (The assembly as a whole is checked against Phase 5A's
own margins on real data by `run.py --part validate`.)"""

from __future__ import annotations

import math

import torch

from crown_oracle import certify


def all_finite_bfloat16() -> torch.Tensor:
    bits = torch.arange(0, 2**16, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    return values[torch.isfinite(values)]


def test_round_up_to_the_bfloat16_grid():
    grid = all_finite_bfloat16().to(torch.float64).unique()
    gen = torch.Generator().manual_seed(0)
    x = torch.cat([grid, torch.randn(50_000, generator=gen) * torch.logspace(-40, 38, 50_000), grid * (1 + 2.0**-20)])
    x = x[torch.isfinite(x) & (x.abs() < 3.3e38)]
    rounded = certify.round_up_to_grid(x, torch.bfloat16)
    position = torch.searchsorted(grid, x)  # the first grid value ≥ x
    assert torch.equal(rounded, grid[position])


def test_spacing_upper_bounds_the_bfloat16_spacing():
    grid = all_finite_bfloat16().to(torch.float64).unique()
    positive = grid[grid > 0]
    gaps = torch.diff(grid)  # spacing between consecutive grid values
    upper = certify.spacing_upper_bf16(positive)
    # Every value of magnitude ≤ m lies between grid neighbours at most spacing_upper(m) apart.
    for m, bound in zip(positive.tolist()[::97], upper.tolist()[::97]):
        inside = (grid[:-1].abs() <= m) & (grid[1:].abs() <= m)
        assert float(gaps[inside].max()) <= bound
    assert float(certify.spacing_upper_bf16(torch.tensor([0.0]))[0]) == 2.0**-133
    assert math.isinf(float(certify.spacing_upper_bf16(torch.tensor([math.inf]))[0]))


def test_structural_margins_take_the_larger_bound():
    lm = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.bfloat16)
    gain = torch.ones(2, dtype=torch.bfloat16)
    rows = torch.tensor([1, 2])
    y_lower, y_upper = torch.tensor([1.0, -0.1]), torch.tensor([1.2, 0.1])
    structural = torch.tensor([0.5, -10.0])
    out = certify.structural_margins(lm, gain, 0, rows, y_lower, y_upper, structural, torch.zeros(2), 4, 0.0)
    box = out["box"]
    # Δ = W_0 − W_j: row 1 → (1, −1): box = 1·1 − 1·0.1 = 0.9 (y₀ ≥ 1, y₁ ≤ 0.1); row 2 → (0.5, −0.5): 0.45.
    assert torch.allclose(box, torch.tensor([0.9, 0.45]), atol=1e-12)
    assert torch.allclose(out["margin"], torch.maximum(box, structural), atol=1e-9)
