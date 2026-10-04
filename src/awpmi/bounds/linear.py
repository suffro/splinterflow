"""Cauchy–Schwarz bounds for a column-paged linear map z = W h.

For a column block p of W and the matching block h_p of the input:

    |(W_p h_p)[j]| ≤ ‖W[j, p]‖₂ · ‖h_p‖₂

`block_l2_norm_upper` returns float64 upper bounds on these norms that remain
valid despite the rounding of the norm computation itself.
"""

from __future__ import annotations

import torch

from awpmi.bounds.floating import FLOAT64_UNIT_ROUNDOFF, gamma, next_up, round_up_to_grid


def block_l2_norm_upper(
    values: torch.Tensor,
    column_slices: list[slice],
    storage_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Upper bounds on ‖values[..., s]‖₂ for every slice s, stacked on the last axis.

    `values` is [..., K]; the result is [..., len(column_slices)] in float64 holding
    values from the `storage_dtype` grid. Each entry is ≥ the exact real norm.
    """
    blocks = []
    for column_slice in column_slices:
        block = values[..., column_slice].to(torch.float64)
        width = block.shape[-1]
        norm = torch.sqrt((block * block).sum(dim=-1))
        # Sum of `width` squares (γ_width) and the sqrt (one rounding) can each
        # under-estimate; inflate by γ_{width+2} and round the product upward.
        inflated = next_up(norm * (1.0 + gamma(width + 2, FLOAT64_UNIT_ROUNDOFF)))
        blocks.append(round_up_to_grid(inflated, storage_dtype))
    return torch.stack(blocks, dim=-1)


def page_contribution(weight_page: torch.Tensor, hidden_block: torch.Tensor) -> torch.Tensor:
    """W_p h_p in float64. Error ≤ γ_width(2⁻⁵³)·Σ|W_jk h_k|, accounted for in residual bounds."""
    return weight_page.to(torch.float64) @ hidden_block.to(torch.float64)


def l1_norm_upper(values: torch.Tensor) -> torch.Tensor:
    """Upper bounds on ‖values[..., :]‖₁ along the last axis, in float64."""
    block = values.to(torch.float64).abs()
    # A sum of K non-negative terms errs by at most γ_K relative; one more ulp for the product.
    return next_up(block.sum(dim=-1) * (1.0 + gamma(block.shape[-1] + 1, FLOAT64_UNIT_ROUNDOFF)))


def matvec(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """matrix @ vector, also for a batch: [B, N, K] matrices with one [K] vector or a [B, K] vector each (Phase 5A)."""
    if matrix.dim() == 3 and vector.dim() == 2:
        return torch.bmm(matrix, vector.unsqueeze(-1)).squeeze(-1)
    return matrix @ vector


def absolute_mass_upper(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Upper bound on |left| @ |right| (entrywise absolute values), e.g. Σ_k |W_jk|·|h_k| per row.

    Both operands must be float64. Each product rounds once and the K-term sum of
    non-negative terms errs by at most γ_K relative, whatever the evaluation order.
    A batch of matrices [B, N, K] takes one [K] or one [B, K] vector per matrix (`matvec`).
    """
    if left.dtype != torch.float64 or right.dtype != torch.float64:
        raise TypeError("absolute_mass_upper expects float64 operands")
    products = matvec(left.abs(), right.abs())
    return next_up(products * (1.0 + gamma(left.shape[-1] + 2, FLOAT64_UNIT_ROUNDOFF)))
