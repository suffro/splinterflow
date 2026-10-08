"""Empirical entropies of BF16 bit fields (Phase 5C).

Order-0 empirical entropies (bits per symbol, from counts) bound what an order-0 entropy coder can reach on a stream;
zstd's Huffman literals come close to them on large blocks. They are measurements, not codecs (`structure` adds the
conditional entropies of shared contexts).
"""

from __future__ import annotations

import numpy as np
import torch


def entropy(counts: np.ndarray | torch.Tensor) -> float:
    """−Σ p·log2 p of a histogram (bits per symbol)."""
    c = torch.as_tensor(counts, dtype=torch.float64).reshape(-1)
    c = c[c > 0]
    p = c / c.sum()
    return float(-(p * torch.log2(p)).sum())


def histogram(symbols: torch.Tensor, size: int) -> torch.Tensor:
    return torch.bincount(symbols.reshape(-1).to(torch.int64), minlength=size)


def fields(p: torch.Tensor) -> dict[str, tuple[torch.Tensor, int]]:
    """Bit fields of uint16 patterns held in an int32 tensor: (values, alphabet size)."""
    p = p.to(torch.int32)
    return {
        "pattern": (p, 1 << 16),
        "high_byte": (p >> 8, 256),
        "low_byte": (p & 0xFF, 256),
        "sign": (p >> 15, 2),
        "exponent": ((p >> 7) & 0xFF, 256),
        "mantissa": (p & 0x7F, 128),
        **{f"m{b}": ((p >> b) & 1, 2) for b in range(6, -1, -1)},
    }


def field_entropies(p: torch.Tensor) -> dict[str, float]:
    """Bits per weight of each field (order 0), and their sum for the byte split and the bit planes."""
    out = {name: entropy(histogram(values, size)) for name, (values, size) in fields(p).items()}
    out["byte_split_total"] = out["high_byte"] + out["low_byte"]
    out["planes_total"] = out["sign"] + out["exponent"] + sum(out[f"m{b}"] for b in range(7))
    return out


def as_int32(patterns: np.ndarray, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.from_numpy(patterns.view(np.int16)).to(device).to(torch.int32) & 0xFFFF
