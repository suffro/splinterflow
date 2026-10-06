"""The verification graphs and their auto_LiRPA bounds (Phase 5A2).

Two graphs of the target MoE layer's suffix at one decode position, in real arithmetic, as ordinary PyTorch modules:

  ExpertsSuffix   y = b + Σ_e w_e · D_e (silu(G_e·x) ⊙ U_e·x)        x exact, G_e, U_e, D_e weight sets
  ReducedSuffix   y = b + Σ_e w_e · D_e·a_e                           a_e an activation set, D_e weight sets

b = r + S is exact (the residual and the shared experts' output), w_e are the router's weights (constants: the router is
exact). ExpertsSuffix is transformers' grouped_mm experts call, DeepseekV3MoE's shared addition and the decoder's residual
addition in real arithmetic: the routing products and the float32 combine become a weighted sum, every rounding is gone.
SiLU is written g·σ(g): auto_LiRPA has no SiLU operator, and this composition of two it supports (`BoundSigmoid`,
`BoundMul`) is the same function. The final RMSNorm is not in the graph: for a candidate w and a contender j,

    ℓ_w − ℓ_j = (W_w − W_j)·h = q·Δ·y,     Δ = (W_w − W_j) ⊙ g,  q = rsqrt(mean(y²) + eps) > 0,

so the sign of the decision difference is the sign of Δ·y (the scale-free property of Phase 2's pairwise certificate),
and auto_LiRPA bounds Δ·y directly through the output specification matrix C (one row per contender). `ThroughNorm`
keeps the norm in the graph instead, as a diagnostic of what relaxing it costs.

A weight set is auto_LiRPA's own: `BoundedParameter` with an L∞ perturbation given by explicit elementwise bounds
(`PerturbationLpNorm(norm=inf, x_L, x_U)`); an activation set is a `BoundedTensor` of the same kind. No bound is computed
here: `Bounder` calls `BoundedModule.compute_bounds` and records its cost.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import BoundedModule, BoundedParameter, BoundedTensor, PerturbationLpNorm

# auto_LiRPA's names (BoundedModule.compute_bounds) of the methods compared.
METHODS = {"ibp": "IBP", "crown-ibp": "CROWN-IBP", "crown": "CROWN", "alpha-crown": "alpha-CROWN"}
BOUND_OPTIONS = {
    # The upstream weight-perturbation test turns these off; sparse intermediate bounds assume a batch of inputs.
    "sparse_intermediate_bounds": False,
    "sparse_conv_intermediate_bounds": False,
    "sparse_intermediate_bounds_with_ibp": False,
}


def box_perturbation(lower: torch.Tensor, upper: torch.Tensor) -> PerturbationLpNorm:
    """auto_LiRPA's elementwise interval set [lower, upper]."""
    if not bool((lower <= upper).all()):
        raise ValueError("empty box")
    return PerturbationLpNorm(norm=np.inf, x_L=lower, x_U=upper)


def boxed_parameter(center: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> BoundedParameter:
    """A weight known to lie in [lower, upper] elementwise.

    For a parameter, auto_LiRPA's interval pass uses the box without a batch dimension and its concretization with one,
    so explicit x_L/x_U cannot serve both (stage 1 probe); its own weight-perturbation example gives the box as `eps`
    about the parameter's value, which works in both. The parameter is therefore the box's midpoint and `eps` its half
    width, rounded up so that midpoint ± eps ⊇ [lower, upper]. `center` (the runtime's centre execution) is kept only to
    check that it lies in the box.
    """
    if not bool(((lower <= center) & (center <= upper)).all()):
        raise ValueError("the centre must lie in its box")
    midpoint = lower * 0.5 + upper * 0.5
    half = torch.maximum(upper - midpoint, midpoint - lower)
    # One ulp for the subtraction, and 2⁻⁴⁰ relative (plus the smallest normal) for auto_LiRPA's re-derivation of the box
    # (x ± eps, then midpoint and half width): a few more roundings of the same size.
    eps = torch.nextafter(half, torch.full_like(half, math.inf)) * (1.0 + 2.0**-40) + 2.0**-1022
    eps = torch.where(upper > lower, eps, torch.zeros_like(eps))  # a point stays a point (lower = upper = midpoint)
    return BoundedParameter(midpoint, PerturbationLpNorm(norm=np.inf, eps=eps), requires_grad=False)


def boxed_input(lower: torch.Tensor, upper: torch.Tensor) -> BoundedTensor:
    """An input known to lie in [lower, upper] elementwise (batch dimension first)."""
    return BoundedTensor(lower * 0.5 + upper * 0.5, box_perturbation(lower, upper))


def _zero_bias(features: int, like: torch.Tensor) -> nn.Parameter:
    """An exact zero bias. With a bias, F.linear traces to ONNX Gemm, auto_LiRPA's `BoundLinear`: the weight-perturbation
    path its own test exercises (`mlp_3layer_weight_perturb`). Without one it traces to Transpose + MatMul, whose backward
    pass fails on a perturbed parameter (stage 1 probe). Adding 0 changes no value."""
    return nn.Parameter(torch.zeros(features, dtype=like.dtype, device=like.device), requires_grad=False)


class ExpertsSuffix(nn.Module):
    """y = b + Σ_e w_e · D_e (silu(G_e·x) ⊙ U_e·x) for x of shape [1, H] (module docstring)."""

    def __init__(self, gates, ups, downs, routing, base: torch.Tensor) -> None:
        super().__init__()
        if not len(gates) == len(ups) == len(downs) == len(routing):
            raise ValueError("one gate, up, down and routing weight per expert")
        self.gates, self.ups, self.downs = nn.ParameterList(gates), nn.ParameterList(ups), nn.ParameterList(downs)
        self.inner_bias = _zero_bias(gates[0].shape[0], base)
        self.outer_bias = _zero_bias(downs[0].shape[0], base)
        self.routing = [float(w) for w in routing]
        self.register_buffer("base", base.reshape(-1))  # no batch dimension: a constant operand (auto_LiRPA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base
        for gate, up, down, weight in zip(self.gates, self.ups, self.downs, self.routing):
            g = F.linear(x, gate, self.inner_bias)
            a = g * torch.sigmoid(g) * F.linear(x, up, self.inner_bias)
            y = y + weight * F.linear(a, down, self.outer_bias)
        return y


class ReducedSuffix(nn.Module):
    """y = b + Σ_e w_e · D_e·a_e, one input a_e of shape [1, I] per expert (module docstring)."""

    def __init__(self, downs, routing, base: torch.Tensor) -> None:
        super().__init__()
        if len(downs) != len(routing):
            raise ValueError("one down projection per routing weight")
        self.downs = nn.ParameterList(downs)
        self.outer_bias = _zero_bias(downs[0].shape[0], base)
        self.routing = [float(w) for w in routing]
        self.register_buffer("base", base.reshape(-1))  # no batch dimension: a constant operand (auto_LiRPA)

    def forward(self, *activations: torch.Tensor) -> torch.Tensor:
        y = self.base
        for a, down, weight in zip(activations, self.downs, self.routing):
            y = y + weight * F.linear(a, down, self.outer_bias)
        return y


class ThroughNorm(nn.Module):
    """Diagnostic: the suffix followed by the final RMSNorm (LlamaRMSNorm's formula in real arithmetic, its gain folded
    into C): its output is h / g = y·rsqrt(mean(y²) + eps), so C·output = (W_w − W_j)·h."""

    def __init__(self, suffix: nn.Module, eps: float) -> None:
        super().__init__()
        self.suffix, self.eps = suffix, float(eps)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        y = self.suffix(*inputs)
        return y / torch.sqrt((y * y).mean(dim=-1, keepdim=True) + self.eps)


def gpu_reset(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def gpu_peak(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def process_peak_rss() -> int:
    """The process's peak working set (Windows) or resident size (elsewhere), bytes."""
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", info.rss))


@dataclass
class BoundCost:
    """What computing bounds cost: the BoundedModule's construction and every compute_bounds call."""

    setup_ms: float = 0.0
    bound_ms: float = 0.0
    calls: int = 0
    specifications: int = 0
    peak_device_bytes: int = 0
    peak_rss_bytes: int = 0
    by_method: dict = field(default_factory=dict)

    def add(self, method: str, ms: float, specs: int, device_bytes: int) -> None:
        self.bound_ms += ms
        self.calls += 1
        self.specifications += specs
        self.peak_device_bytes = max(self.peak_device_bytes, device_bytes)
        self.peak_rss_bytes = max(self.peak_rss_bytes, process_peak_rss())
        entry = self.by_method.setdefault(method, {"ms": 0.0, "calls": 0, "specifications": 0})
        entry["ms"] += ms
        entry["calls"] += 1
        entry["specifications"] += specs

    def merge(self, other: BoundCost) -> None:
        self.setup_ms += other.setup_ms
        self.bound_ms += other.bound_ms
        self.calls += other.calls
        self.specifications += other.specifications
        self.peak_device_bytes = max(self.peak_device_bytes, other.peak_device_bytes)
        self.peak_rss_bytes = max(self.peak_rss_bytes, other.peak_rss_bytes)
        for method, entry in other.by_method.items():
            mine = self.by_method.setdefault(method, {"ms": 0.0, "calls": 0, "specifications": 0})
            for key in mine:
                mine[key] += entry[key]

    def to_json(self) -> dict:
        return {
            "setup_ms": self.setup_ms, "bound_ms": self.bound_ms, "calls": self.calls, "specifications": self.specifications,
            "peak_device_bytes": self.peak_device_bytes, "peak_rss_bytes": self.peak_rss_bytes, "by_method": self.by_method,
        }


class Bounder:
    """auto_LiRPA's bounds on C·output of one graph with its sets fixed (one BoundedModule)."""

    def __init__(self, model: nn.Module, inputs: tuple[torch.Tensor, ...], device: torch.device,
                 alpha_iterations: int = 20, cost: BoundCost | None = None) -> None:
        self.device = device
        self.inputs = inputs
        self.cost = cost if cost is not None else BoundCost()
        options = dict(BOUND_OPTIONS)
        options["optimize_bound_args"] = {"iteration": int(alpha_iterations)}
        gpu_reset(device)
        started = time.perf_counter()
        model.eval()
        self.module = BoundedModule(model, inputs, bound_opts=options, device=device, verbose=0)
        self.module.eval()
        self.cost.setup_ms += (time.perf_counter() - started) * 1e3
        self.cost.peak_device_bytes = max(self.cost.peak_device_bytes, gpu_peak(device))

    def lower(self, C: torch.Tensor, method: str, chunk: int) -> torch.Tensor:
        """Lower bounds of C·output ([J, out] → [J]), `chunk` specifications per compute_bounds call."""
        results = []
        for start in range(0, C.shape[0], chunk):
            part = C[start : start + chunk].unsqueeze(0)
            gpu_reset(self.device)
            started = time.perf_counter()
            lower, _ = self.module.compute_bounds(x=self.inputs, C=part, method=METHODS[method], bound_upper=False)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.cost.add(method, (time.perf_counter() - started) * 1e3, part.shape[1], gpu_peak(self.device))
            results.append(lower.detach().reshape(-1))
        return torch.cat(results) if results else torch.zeros(0, dtype=C.dtype, device=C.device)

    def bounds(self, C: torch.Tensor, method: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Lower and upper bounds (probe only: the certificate needs lower bounds)."""
        gpu_reset(self.device)
        started = time.perf_counter()
        lower, upper = self.module.compute_bounds(x=self.inputs, C=C.unsqueeze(0), method=METHODS[method])
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.cost.add(method, (time.perf_counter() - started) * 1e3, C.shape[0], gpu_peak(self.device))
        return lower.detach().reshape(-1), upper.detach().reshape(-1)


def silu(g: torch.Tensor) -> torch.Tensor:
    """The forward SiLU of the graphs (g·σ(g)), for evaluating realizations outside auto_LiRPA."""
    return g * torch.sigmoid(g)


def suffix_value(x: torch.Tensor, gates, ups, downs, routing, base: torch.Tensor) -> torch.Tensor:
    """y of ExpertsSuffix for batches of weight realizations: gates [N, K, I, H] (or [K, I, H]) etc.; returns [N, H]."""
    single = gates.dim() == 3
    if single:
        gates, ups, downs = gates.unsqueeze(0), ups.unsqueeze(0), downs.unsqueeze(0)
    v = x.reshape(-1)
    g = torch.einsum("nkih,h->nki", gates, v)
    u = torch.einsum("nkih,h->nki", ups, v)
    a = silu(g) * u
    o = torch.einsum("nkhi,nki->nkh", downs, a)
    weights = torch.as_tensor(routing, dtype=o.dtype, device=o.device)
    y = base.reshape(1, -1) + torch.einsum("nkh,k->nh", o, weights)
    return y[0] if single else y


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None
