"""Phase 6A, §7: what the BF16 reference's kernels do, measured on this machine (decision 0012; no new kernel).

    uv run python benchmarks/reference_numerics.py --output experiments/phase6a/bf16/numerics.json

Moonlight's `BF16_REFERENCE` is transformers' DeepSeek-V3 code run by PyTorch's kernels with
`awpmi.runtime.configure_reproducible_numerics` (deterministic algorithms, no TF32, no reduced-precision reductions).
Phase 6C may give some of its operations explicit native semantics. This probe records, for each kind of operation the
reference uses, how its result relates to the correctly rounded one and whether a row's result depends on the shape it
is computed in. It changes nothing and certifies nothing; inputs are random (fixed seeds) in normal ranges, with
Moonlight's shapes where shapes matter.

  conversion      float32 -> BF16 (`.to`): against round-to-nearest-even, ties included
  elementwise     BF16 add and multiply (residual adds, RMSNorm's weight, SiLU · up, RoPE): computed in float32, rounded
                  once to BF16; against the correctly rounded BF16 result (exact in float64, or innocuously double
                  rounded: 53 >= 2 * 8 + 2)
  transcendental  float32 exp, sigmoid (router scores) and rsqrt (RMSNorm) against float64 rounded to float32, in ulps;
                  BF16 SiLU (experts and MLPs) on every finite BF16 input against the correctly rounded BF16 SiLU
  reductions      RMSNorm's float32 mean of 2,048 squares and the experts' float32 combine of 6 weighted outputs, against
                  their exact sums rounded once to float32: the summation order is the kernel's
  shapes          one row through each of Moonlight's GEMMs (BF16, cuBLAS) alone and inside batches of up to 1,024 rows;
                  through grouped_mm in groups of 1 to 300 rows; attention's last query alone (a decode step) and inside a
                  causal block (a prefill): does a row's result depend on the rows computed with it?

The attention and GEMM kernels each call dispatches to are recorded with the profiler.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from awpmi.runtime import configure_reproducible_numerics

NUMERICS = configure_reproducible_numerics()

import torch.nn.functional as F  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from transformers.integrations.moe import _grouped_linear  # noqa: E402

from awpmi.tracing import environment_metadata  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# Moonlight-16B-A3B's linear layers (in features, out features); every one is a BF16 nn.Linear without bias.
LINEARS = {
    "q_proj": (2048, 3072),
    "kv_a_proj_with_mqa": (2048, 576),
    "kv_b_proj": (512, 4096),
    "o_proj": (2048, 2048),
    "shared_experts.gate_proj": (2048, 2816),
    "shared_experts.down_proj": (2816, 2048),
    "dense_mlp.gate_proj": (2048, 11264),
    "dense_mlp.down_proj": (11264, 2048),
    "lm_head": (2048, 163840),
}
# The routed experts' grouped GEMMs (in, out): gate and up fused, then down.
GROUPED = {"experts.gate_up_proj": (2048, 2816), "experts.down_proj": (1408, 2048)}
BATCHES = [1, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024]
GROUP_ROWS = [1, 2, 3, 6, 8, 16, 64, 128, 300]
ATTENTION_LENGTHS = [16, 128, 512, 1024]
HEADS, QK_DIM, V_DIM = 16, 192, 128


def round_bf16(values: np.ndarray) -> np.ndarray:
    """float64 -> the nearest BF16 value, ties to even (as float64; normal range)."""
    mantissa, exponent = np.frexp(values)
    return np.ldexp(np.rint(np.ldexp(mantissa, 8)), exponent - 8)


def ordered(bits: np.ndarray) -> np.ndarray:
    """float32 bit patterns -> integers in the floats' order (ulp distances are differences)."""
    bits = bits.astype(np.int64)
    return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)


def ulps32(actual: torch.Tensor, exact: np.ndarray) -> np.ndarray:
    """|actual - round32(exact)| in float32 ulps."""
    reference = exact.astype(np.float32)
    got = actual.detach().cpu().numpy().astype(np.float32)
    return np.abs(ordered(got.view(np.int32)) - ordered(reference.view(np.int32)))


def ulp_summary(distances: np.ndarray) -> dict:
    return {
        "values": int(distances.size),
        "correctly_rounded": float((distances == 0).mean()),
        "max_ulps": int(distances.max()),
        "histogram": {str(k): int((distances == k).sum()) for k in range(int(min(distances.max(), 4)) + 1)},
    }


def random_bf16(count: int, generator: torch.Generator, low: int = -20, high: int = 20) -> torch.Tensor:
    """BF16 values with random signs, mantissas and exponents in [2**low, 2**high)."""
    mantissa = torch.randint(0, 128, (count,), generator=generator, dtype=torch.int32)
    exponent = torch.randint(low + 127, high + 127, (count,), generator=generator, dtype=torch.int32)
    sign = torch.randint(0, 2, (count,), generator=generator, dtype=torch.int32)
    bits = (sign << 15) | (exponent << 7) | mantissa
    return bits.to(torch.int16).view(torch.bfloat16)


def probe_conversion(device: str, generator: torch.Generator) -> dict:
    """float32 -> BF16: random float32 patterns, then exact midpoints (ties) between adjacent BF16 values."""
    count = 1 << 22
    bits = torch.randint(-(1 << 31), 1 << 31, (count,), generator=generator, dtype=torch.int64)
    values = bits.to(torch.int32).view(torch.float32)
    values = values[torch.isfinite(values) & (values.abs() > 2.0**-126) & (values.abs() < 2.0**127)]
    lower = random_bf16(count, generator)
    upper = (lower.view(torch.int16) + 1).view(torch.bfloat16)  # the next BF16 value away from zero
    ties = (lower.float() + upper.float()) / 2  # exact in float32: nine significant bits
    results = {}
    for name, sample in (("random", values), ("ties", ties)):
        got = sample.to(device).to(torch.bfloat16).float().cpu().numpy().astype(np.float64)
        expected = round_bf16(sample.numpy().astype(np.float64))
        results[name] = {"values": int(sample.numel()), "round_to_nearest_even": float((got == expected).mean())}
    return results


def probe_elementwise(device: str, generator: torch.Generator) -> dict:
    count = 1 << 22
    a = random_bf16(count, generator)
    b = random_bf16(count, generator)
    close = random_bf16(count, generator, -2, 2)  # sums with cancellation and every alignment
    out = {}
    for name, x, y, exact in (
        ("add", a, b, lambda u, v: u + v),
        ("add_close", close, random_bf16(count, generator, -2, 2), lambda u, v: u + v),
        ("multiply", a, b, lambda u, v: u * v),
    ):
        op = torch.add if name.startswith("add") else torch.mul
        got = op(x.to(device), y.to(device)).float().cpu().numpy().astype(np.float64)
        expected = round_bf16(exact(x.float().numpy().astype(np.float64), y.float().numpy().astype(np.float64)))
        out[name] = {"values": count, "correctly_rounded": float((got == expected).mean())}
    return out


def probe_transcendental(device: str, generator: torch.Generator) -> dict:
    count = 1 << 22
    out = {}
    uniform = (torch.rand(count, generator=generator, dtype=torch.float64) * 40 - 20).float()
    out["exp_float32"] = ulp_summary(ulps32(torch.exp(uniform.to(device)), np.exp(uniform.numpy().astype(np.float64))))
    router = (torch.randn(count, generator=generator, dtype=torch.float64) * 4).float()
    exact = 1.0 / (1.0 + np.exp(-router.numpy().astype(np.float64)))
    out["sigmoid_float32"] = ulp_summary(ulps32(torch.sigmoid(router.to(device)), exact))
    positive = torch.exp2(torch.rand(count, generator=generator, dtype=torch.float64) * 40 - 20).float()
    exact = 1.0 / np.sqrt(positive.numpy().astype(np.float64))
    out["rsqrt_float32"] = ulp_summary(ulps32(torch.rsqrt(positive.to(device)), exact))
    # SiLU on every finite BF16 value: the experts' and MLPs' activation, BF16 in and out.
    every = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    every = every[torch.isfinite(every.float())]
    x = every.float().numpy().astype(np.float64)
    with np.errstate(over="ignore"):
        exact = x / (1.0 + np.exp(-x))
    got = F.silu(every.to(device)).float().cpu().numpy().astype(np.float64)
    expected = round_bf16(exact)
    finite = np.isfinite(expected) & ((np.abs(expected) >= 2.0**-126) | (expected == 0))  # BF16 subnormals left out
    wrong = finite & (got != expected)
    out["silu_bfloat16_exhaustive"] = {
        "values": int(finite.sum()),
        "correctly_rounded": float(1 - wrong.sum() / finite.sum()),
        "not_correctly_rounded": int(wrong.sum()),
        "examples": [[float(v), float(g), float(e)] for v, g, e in zip(x[wrong][:8], got[wrong][:8], expected[wrong][:8])],
    }
    return out


def probe_reductions(device: str, generator: torch.Generator) -> dict:
    out = {}
    rows = 4096
    hidden = torch.randn(rows, 2048, generator=generator).to(torch.bfloat16)
    squares = hidden.float().pow(2)
    got = squares.to(device).mean(-1)
    exact = squares.numpy().astype(np.float64).sum(-1) / 2048.0  # each square is exact in float32; the sum in float64
    out["rmsnorm_mean_of_squares"] = ulp_summary(ulps32(got, exact))
    tokens = 4096
    z = (torch.randn(tokens, 6, 2048, generator=generator) * torch.rand(tokens, 6, 1, generator=generator)).float()
    got = z.to(device).sum(dim=1)
    exact = z.numpy().astype(np.float64).sum(1)
    out["experts_combine_sum_of_6"] = ulp_summary(ulps32(got, exact).reshape(-1))
    # Where the terms cancel, ulps of the result mean little: the error in units of 2**-24 * sum |z| (the scale a
    # summation bound such as gamma_n uses).
    magnitude = np.abs(z.numpy().astype(np.float64)).sum(1)
    error = np.abs(got.cpu().numpy().astype(np.float64) - exact) / (2.0**-24 * magnitude)
    out["experts_combine_sum_of_6"]["max_error_in_u_sum_abs"] = float(error.max())
    out["experts_combine_sum_of_6"]["mean_error_in_u_sum_abs"] = float(error.mean())
    return out


def kernels_of(run) -> list[str]:
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    names = []
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA and event.name not in names:
            names.append(event.name)
    return names


def probe_linears(device: str, generator: torch.Generator) -> dict:
    out = {}
    for name, (k, n) in LINEARS.items():
        weight = (torch.randn(n, k, generator=generator) * 0.02).to(torch.bfloat16).to(device)
        inputs = torch.randn(max(BATCHES), k, generator=generator).to(torch.bfloat16).to(device)
        alone = F.linear(inputs[:1], weight)
        again = F.linear(inputs[:1], weight)
        differs = {}
        for m in BATCHES[1:]:
            row = F.linear(inputs[:m], weight)[:1]
            changed = (row != alone).float().mean().item()
            if changed:
                differs[str(m)] = changed
        out[name] = {
            "shape": [k, n],
            "repeatable": bool(torch.equal(alone, again)),
            "row_differs_from_alone_in_batches": differs,
            "kernels": {str(m): kernels_of(lambda m=m: F.linear(inputs[:m], weight)) for m in (1, 16, 1024)},
        }
        del weight, inputs
        torch.cuda.empty_cache()
    return out


def probe_grouped(device: str, generator: torch.Generator) -> dict:
    out = {}
    experts = 8
    for name, (k, n) in GROUPED.items():
        weight = (torch.randn(experts, n, k, generator=generator) * 0.02).to(torch.bfloat16).to(device)
        inputs = torch.randn(max(GROUP_ROWS) * experts, k, generator=generator).to(torch.bfloat16).to(device)

        def first_row(rows: int, others: int) -> torch.Tensor:
            # group 0 holds `rows` rows; each of the other experts `others` rows
            counts = torch.tensor([rows] + [others] * (experts - 1), dtype=torch.int32)
            offsets = torch.cumsum(counts, 0, dtype=torch.int32).to(device)
            return _grouped_linear(inputs[: int(counts.sum())], weight, offsets)[:1]

        alone = first_row(1, 0)
        differs = {str(rows): (first_row(rows, 0) != alone).float().mean().item() for rows in GROUP_ROWS[1:]}
        neighbours = {str(others): (first_row(6, others) != first_row(6, 0)).float().mean().item() for others in (1, 6, 300)}
        out[name] = {
            "shape": [k, n],
            "row_differs_with_group_rows": {key: value for key, value in differs.items() if value},
            "row_differs_with_other_groups": {key: value for key, value in neighbours.items() if value},
            "kernels": {str(rows): kernels_of(lambda rows=rows: first_row(rows, 6)) for rows in (1, 300)},
        }
        del weight, inputs
        torch.cuda.empty_cache()
    return out


def probe_attention(device: str, generator: torch.Generator) -> dict:
    out = {}
    scale = QK_DIM**-0.5
    for length in ATTENTION_LENGTHS:
        # The reference's layouts (DeepseekV3Attention): queries and keys contiguous [1, heads, T, 192] (built by cat and
        # copies), values a view of kv_b_proj's output [1, T, heads, 256] transposed, its last 128 features.
        q = torch.randn(1, HEADS, length, QK_DIM, generator=generator).to(torch.bfloat16).to(device)
        k = torch.randn(1, HEADS, length, QK_DIM, generator=generator).to(torch.bfloat16).to(device)
        kv = torch.randn(1, length, HEADS, 128 + V_DIM, generator=generator).to(torch.bfloat16).to(device)
        v = kv.transpose(1, 2)[..., 128:]
        last_query = q[:, :, -1:].contiguous()

        def prefill() -> torch.Tensor:
            return F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)

        def decode() -> torch.Tensor:
            return F.scaled_dot_product_attention(last_query, k, v, scale=scale, is_causal=False)

        last = prefill()[:, :, -1:]
        alone = decode()
        out[str(length)] = {
            "repeatable": bool(torch.equal(alone, decode()) and torch.equal(last, prefill()[:, :, -1:])),
            "last_query_differs_decode_vs_prefill": (alone != last).float().mean().item(),
            "max_abs_difference": (alone.float() - last.float()).abs().max().item(),
            "kernels": {"prefill": kernels_of(prefill), "decode": kernels_of(decode)},
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=6)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        print("needs CUDA: the reference's kernels are CUDA kernels", file=sys.stderr)
        return 2
    device = "cuda"
    generator = torch.Generator().manual_seed(args.seed)
    results = {
        "environment": environment_metadata(REPO_ROOT, {"probe": "the BF16 reference's kernels"}, NUMERICS),
        "capability": list(torch.cuda.get_device_capability()),
        "seed": args.seed,
    }
    for name, probe in (
        ("conversion", probe_conversion),
        ("elementwise", probe_elementwise),
        ("transcendental", probe_transcendental),
        ("reductions", probe_reductions),
        ("linears", probe_linears),
        ("grouped_mm", probe_grouped),
        ("attention", probe_attention),
    ):
        results[name] = probe(device, generator)
        print(name, json.dumps(results[name], default=str)[:2000], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
