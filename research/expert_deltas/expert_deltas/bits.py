"""Exact transforms of BF16 bit patterns (Phase 5C).

Everything here works on the 16-bit patterns (numpy uint16), never on values, so every transform is a bijection whose
inverse restores the original bits, for every one of the 65,536 patterns (±0, subnormals, ±inf, NaN payloads):

    xor         d = w ^ b                       w = d ^ b
    modular     d = (w − b) mod 2¹⁶             w = (d + b) mod 2¹⁶   (integer arithmetic on the patterns)
    byte split  every high byte, then every low byte (blosc's "shuffle" filter, ZipNN's byte grouping)
    bit planes  sign (1 bit per weight, packed), exponent (8 bits, a byte stream), mantissa bits 6..0 (1 bit each,
                packed), most significant first

Floating-point `BF16(base + delta)` is never used: it does not reproduce the original bits in general.

Prefix intervals. After reading the top t bits of a weight's pattern (MSB order: sign, exponent bits 7..0, mantissa
bits 6..0), a runtime knows the weight lies among the finite completions of that prefix; `prefix_interval` returns the
closed interval of their values. The magnitude of a BF16 pattern is non-decreasing in its 15 low bits over the finite
range (0x0000..0x7F7F), so the extremes are the completions with every unknown bit 0 and every unknown bit 1 (capped at
the largest finite pattern). That the weights are finite is checked when they are read (`require_finite`) and is part
of what a runtime may assume. Since an XOR delta's top t bits and the base's give the weight's top t bits, the same
intervals hold for XOR planes; a modular delta's top bits do not localize the weight's bits (carries), so its planes are
not used progressively.
"""

from __future__ import annotations

import numpy as np
import torch

PLANES = ("sign", "exponent", "m6", "m5", "m4", "m3", "m2", "m1", "m0")
PLANE_BITS = {"sign": 1, "exponent": 8, **{f"m{b}": 1 for b in range(7)}}
# The prefix length t after each plane (cumulative bits known, MSB first).
PREFIX_AFTER = dict(zip(PLANES, np.cumsum([PLANE_BITS[p] for p in PLANES]).tolist()))
LARGEST_FINITE = 0x7F7F  # the largest finite BF16 magnitude pattern


# Views


def patterns(tensor: torch.Tensor) -> np.ndarray:
    """A BF16 tensor's bit patterns as a contiguous numpy uint16 array (a copy on the host)."""
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16, got {tensor.dtype}")
    return tensor.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16).copy()


def to_bfloat16(p: np.ndarray) -> torch.Tensor:
    """uint16 patterns back to a BF16 tensor (same shape), bit for bit."""
    return torch.from_numpy(np.ascontiguousarray(p, dtype=np.uint16).view(np.int16)).view(torch.bfloat16)


def require_finite(p: np.ndarray) -> None:
    """Raise unless every pattern is a finite value (exponent field below 255)."""
    if int(((p & 0x7F80) == 0x7F80).sum()):
        raise ValueError("non-finite BF16 patterns (inf or NaN)")


# Deltas


def xor_delta(w: np.ndarray, base: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(w, base)


def xor_restore(delta: np.ndarray, base: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(delta, base)


def modular_delta(w: np.ndarray, base: np.ndarray) -> np.ndarray:
    return (w.astype(np.int32) - base.astype(np.int32)).astype(np.uint16)  # two's complement wrap = mod 2¹⁶


def modular_restore(delta: np.ndarray, base: np.ndarray) -> np.ndarray:
    return (delta.astype(np.int32) + base.astype(np.int32)).astype(np.uint16)


# Byte split


def byte_split(p: np.ndarray) -> bytes:
    """The high bytes of every pattern (in order), then the low bytes."""
    raw = np.ascontiguousarray(p, dtype=np.uint16).reshape(-1).view(np.uint8)  # little endian: low, high
    return raw[1::2].tobytes() + raw[0::2].tobytes()


def byte_merge(data: bytes, count: int) -> np.ndarray:
    if len(data) != 2 * count:
        raise ValueError("byte split data must hold two bytes per pattern")
    raw = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(2 * count, dtype=np.uint8)
    out[1::2], out[0::2] = raw[:count], raw[count:]
    return out.view(np.uint16)


# Bit planes


def split_planes(p: np.ndarray) -> dict[str, bytes]:
    """`PLANES` of the patterns (flattened): sign and mantissa planes packed 8 per byte (MSB first), the exponent a
    byte stream."""
    flat = np.ascontiguousarray(p, dtype=np.uint16).reshape(-1)
    planes = {"sign": np.packbits((flat >> 15).astype(np.uint8)).tobytes()}
    planes["exponent"] = ((flat >> 7) & 0xFF).astype(np.uint8).tobytes()
    for b in range(6, -1, -1):
        planes[f"m{b}"] = np.packbits(((flat >> b) & 1).astype(np.uint8)).tobytes()
    return planes


def merge_planes(planes: dict[str, bytes], count: int) -> np.ndarray:
    """Inverse of `split_planes` for `count` patterns (every plane needed)."""
    def bits(name: str) -> np.ndarray:
        return np.unpackbits(np.frombuffer(planes[name], dtype=np.uint8), count=count).astype(np.uint16)

    out = bits("sign") << 15
    out |= np.frombuffer(planes["exponent"], dtype=np.uint8).astype(np.uint16)[:count] << 7
    for b in range(6, -1, -1):
        out |= bits(f"m{b}") << b
    return out


def plane_bytes(count: int) -> dict[str, int]:
    """Raw bytes of each plane for `count` patterns."""
    packed = (count + 7) // 8
    return {name: count if name == "exponent" else packed for name in PLANES}


# Prefix intervals (torch, any device)


def pattern_values(p: torch.Tensor) -> torch.Tensor:
    """Values (float64) of 16-bit patterns held in an integer tensor with entries in [0, 65536), exactly."""
    signed = torch.where(p >= 0x8000, p - 0x10000, p).to(torch.int16)
    return signed.view(torch.bfloat16).to(torch.float64)


def prefix_interval(p: torch.Tensor, t: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[lo, hi] (float64) of the finite values whose patterns share the top `t` bits of `p` (int32/int64, [0, 65536)).

    `t` is an int or an integer tensor broadcastable to `p` (a prefix length per weight, 0..16).
    """
    p = p.to(torch.int64)
    t = torch.as_tensor(t, dtype=torch.int64, device=p.device).expand_as(p)
    free = (torch.ones_like(p) << (16 - t)) - 1  # the unknown low bits
    known = p & (~free & 0xFFFF)
    negative = (known & 0x8000) != 0
    low = known & 0x7FFF  # unknown bits 0 (when t ≥ 1 the sign is known)
    high = torch.clamp((known & 0x7FFF) | (free & 0x7FFF), max=LARGEST_FINITE)
    if bool((low > LARGEST_FINITE).any()):
        raise ValueError("a prefix with no finite completion")
    low_value, high_value = pattern_values(low), pattern_values(high)
    lo = torch.where(negative, -high_value, low_value)
    hi = torch.where(negative, -low_value, high_value)
    unsigned = t == 0  # nothing known: every finite value
    lo = torch.where(unsigned, -pattern_values(torch.full_like(p, LARGEST_FINITE)), lo)
    exact = t >= 16
    value = pattern_values(p)
    return torch.where(exact, value, lo), torch.where(exact, value, hi)
