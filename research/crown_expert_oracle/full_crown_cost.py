"""Phase 5A2: the cost of auto_LiRPA's CROWN with backward intermediate bounds on the full expert graph (stage 1 probe).

    uv run --project research/crown_expert_oracle python research/crown_expert_oracle/full_crown_cost.py --output <dir>

auto_LiRPA computes CROWN's intermediate bounds with an identity specification over each nonlinear node's inputs: for one
expert at Moonlight's shape that is 1,408 specifications, each a backward pass to the 1,408 × 2,048 gate and up weights
(30 GiB at once in float64, the probe's failure). Its `crown_batch_size` option bounds the memory by batching them. This
measures one expert (synthetic weights, q6-like boxes, as the probe's part B) for the time per bound and its tightness
against CROWN-IBP; the routed layer has six.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe  # noqa: E402
from auto_LiRPA import BoundedModule  # noqa: E402
from crown_oracle.graph import BOUND_OPTIONS, gpu_peak, gpu_reset, leave  # noqa: E402

torch.set_default_dtype(torch.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    device = torch.device("cuda")
    experts, hidden, intermediate = 1, 2048, 1408
    gen = torch.Generator(device=device).manual_seed(7)
    truth = {n: torch.randn(experts, *shape, device=device, generator=gen) * 0.02
             for n, shape in (("gate", (intermediate, hidden)), ("up", (intermediate, hidden)), ("down", (hidden, intermediate)))}
    boxes = {n: probe.q6_box(v) for n, v in truth.items()}
    x = torch.randn(1, hidden, device=device, generator=gen)
    base = torch.randn(hidden, device=device, generator=gen) * 0.5
    C = torch.randn(1, 1, hidden, device=device, generator=gen)
    results = {"experts": experts, "batch": args.batch}
    for label, options, method in (("crown-ibp", {}, "CROWN-IBP"),
                                   ("crown", {"crown_batch_size": args.batch, "batched_crown_max_vram_ratio": 0.5}, "CROWN")):
        model = probe.build_full(boxes, [0.42], base)
        module = BoundedModule(model, (x,), bound_opts={**BOUND_OPTIONS, **options}, device=device, verbose=0)
        module.eval()
        gpu_reset(device)
        started = time.perf_counter()
        lower, _ = module.compute_bounds(x=(x,), C=C, method=method, bound_upper=False)
        torch.cuda.synchronize(device)
        results[label] = {"lower": float(lower), "seconds": time.perf_counter() - started, "peak_device_bytes": gpu_peak(device)}
        print(label, results[label], flush=True)
        del module, model
        torch.cuda.empty_cache()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "full_crown_cost.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    leave(main())
