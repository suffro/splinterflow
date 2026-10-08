"""Shared structure between the routed experts of one layer, and the bases it would support (Phase 5C, stage 5C1-B).

Measurements on one matrix kind of one layer (patterns [E, R·C] held as int32 on the device; every pass works on a few
experts at a time, so device memory stays near twice the layer's patterns):

  census        each bit field's order-0 entropy; what shared contexts would save (H(exponent | column) across every
                expert: a per-input-dimension scale shared by the experts; H(exponent | expert, row): a per-row scale);
                the cross-expert spectrum of the values (the E × E Gram matrix: a shared component shows as a dominant
                eigenvalue) and the experts' pairwise correlations; sign and exponent agreement against independence
  costs         the pairwise proxy cost: bits per weight of an order-0 coder of each bit plane of the delta of expert j
                against base m (XOR or modular), for every (j, m); the diagonal holds the expert's own cost (stored
                alone). zstd-19 on bit planes comes within about 0.5% of these entropies on whole tensors (codec probe)
  permutation   experts are trained independently, so similar neurons may sit at different indices: the best |cosine| of
                each neuron (its gate row, up row and down column) of one expert against every neuron of another, against
                the same statistic for independent Gaussian matrices of the same shape

Base strategies (`assignments`), a base per expert (−1: stored alone; a base is always stored alone): A1 the first
expert; A2 the medoid (smallest total proxy cost); A3 each expert's best among the `candidates` most central experts;
C-k average-linkage clusters of the symmetrized proxy cost (SciPy), each cluster's medoid its base. Every non-base expert
is stored as a delta, so a poor base shows its cost. `best_pair_bound` is the oracle that gives each expert its best
actual-expert base, or keeps it alone when no delta is cheaper: no assignment to actual experts saves more.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from expert_deltas.census import entropy, field_entropies

CHUNK = 8  # experts per pass


def to_device(patterns: np.ndarray, device) -> torch.Tensor:
    """uint16 patterns [E, R, C] → int32 [E, R·C] on the device (unsigned values)."""
    flat = torch.from_numpy(patterns.reshape(patterns.shape[0], -1).view(np.int16))
    return (flat.to(device).to(torch.int32) & 0xFFFF).contiguous()


def values(p: torch.Tensor) -> torch.Tensor:
    """float32 values of int32-held patterns (exact)."""
    signed = torch.where(p >= 0x8000, p - 0x10000, p).to(torch.int16)
    return signed.view(torch.bfloat16).to(torch.float32)


def _binary_entropy(ones: torch.Tensor, n: int) -> torch.Tensor:
    p = (ones.to(torch.float64) / n).clamp(0.0, 1.0)
    q = 1.0 - p
    return -(torch.where(p > 0, p * torch.log2(p), 0.0) + torch.where(q > 0, q * torch.log2(q), 0.0))


def _row_entropy(counts: torch.Tensor) -> torch.Tensor:
    p = counts.to(torch.float64) / counts.sum(dim=1, keepdim=True)
    return -(torch.where(p > 0, p * torch.log2(p), 0.0)).sum(dim=1)


def plane_costs(d: torch.Tensor) -> torch.Tensor:
    """Bits per weight of an order-0 coder of each bit plane, per row of d ([rows, N] int32 patterns) → [rows] (float64)."""
    out = []
    for start in range(0, d.shape[0], CHUNK):
        part = d[start : start + CHUNK]
        rows, n = part.shape
        total = _binary_entropy((part >> 15).sum(dim=1), n)
        for b in range(7):
            total = total + _binary_entropy(((part >> b) & 1).sum(dim=1), n)
        exponent = ((part >> 7) & 0xFF) + torch.arange(rows, device=d.device, dtype=torch.int32)[:, None] * 256
        counts = torch.bincount(exponent.reshape(-1), minlength=rows * 256).view(rows, 256)
        out.append(total + _row_entropy(counts))
    return torch.cat(out)


def pairwise_costs(p: torch.Tensor, encoding: str) -> torch.Tensor:
    """[E, E] bits per weight (CPU): entry (j, m) is expert j's delta against base m; (j, j) is expert j stored alone."""
    experts = p.shape[0]
    out = torch.empty(experts, experts, dtype=torch.float64)
    for m in range(experts):
        if encoding == "xor":
            column = torch.cat([plane_costs(p[s : s + CHUNK] ^ p[m]) for s in range(0, experts, CHUNK)])
        elif encoding == "modular":
            column = torch.cat([plane_costs((p[s : s + CHUNK] - p[m]) & 0xFFFF) for s in range(0, experts, CHUNK)])
        else:
            raise ValueError(encoding)
        out[:, m] = column.cpu()
    out[torch.arange(experts), torch.arange(experts)] = plane_costs(p).cpu()
    return out


def _joint_entropy(symbols, size: int, contexts, context_size: int, experts: int) -> tuple[float, float]:
    """H(symbol, context) and H(context) from histograms accumulated a few experts at a time (callables of an expert range)."""
    joint = torch.zeros(size * context_size, dtype=torch.int64)
    marginal = torch.zeros(context_size, dtype=torch.int64)
    for start in range(0, experts, CHUNK):
        stop = min(start + CHUNK, experts)
        c = contexts(start, stop).to(torch.int64)
        s = symbols(start, stop).to(torch.int64)
        joint += torch.bincount((s * context_size + c).reshape(-1), minlength=size * context_size).cpu()
        marginal += torch.bincount(c.reshape(-1), minlength=context_size).cpu()
    return entropy(joint), entropy(marginal)


def census(p: torch.Tensor, rows: int, cols: int) -> dict:
    """Entropies, shared contexts, the cross-expert spectrum and agreement statistics of one matrix kind ([E, R·C])."""
    experts, n = p.shape
    device = p.device
    fields = [field_entropies(p[e : e + 1]) for e in range(experts)]
    result = {"fields_per_expert_mean": {k: float(np.mean([f[k] for f in fields])) for k in fields[0]}}
    position = torch.arange(n, device=device)

    def exponent(start, stop):
        return (p[start:stop] >> 7) & 0xFF

    def context(kind):
        def make(start, stop):
            span = torch.arange(start, stop, device=device)[:, None]
            if kind == "none":
                return torch.zeros(stop - start, n, dtype=torch.int64, device=device)
            if kind == "column":
                return (position % cols).expand(stop - start, -1)
            if kind == "row":
                return (position // cols).expand(stop - start, -1)
            if kind == "expert":
                return span.expand(-1, n)
            if kind == "expert_row":
                return span * rows + (position // cols)[None, :]
            if kind == "expert_column":
                return span * cols + (position % cols)[None, :]
            raise ValueError(kind)
        return make

    sizes = {"none": 1, "column": cols, "row": rows, "expert": experts, "expert_row": experts * rows, "expert_column": experts * cols}
    given = {}
    for kind, size in sizes.items():
        joint, marginal = _joint_entropy(exponent, 256, context(kind), size, experts)
        given[kind] = joint - marginal
    h = given.pop("none")
    result["exponent_entropy_pooled"] = h
    result["exponent_given"] = given
    result["exponent_context_saving_bits"] = {k: h - v for k, v in given.items()}
    # What describing a context costs (a one-byte offset per context value), in bits per weight.
    result["context_description_bits"] = {k: 8.0 * size / (experts * n) for k, size in sizes.items() if k != "none"}
    # Cross-expert spectrum and correlations of the values.
    gram = torch.zeros(experts, experts, dtype=torch.float64, device=device)
    sums = torch.zeros(experts, dtype=torch.float64, device=device)
    for start in range(0, n, 1 << 18):
        v = values(p[:, start : start + (1 << 18)]).to(torch.float64)
        gram += v @ v.t()
        sums += v.sum(dim=1)
    eig = torch.linalg.eigvalsh(gram).flip(0)
    mean = sums / n
    covariance = gram / n - mean[:, None] * mean[None, :]
    std = covariance.diagonal().sqrt()
    correlation = covariance / (std[:, None] * std[None, :])
    off_diagonal = ~torch.eye(experts, dtype=torch.bool, device=device)
    off = correlation[off_diagonal]
    result["spectrum"] = {
        "top_eigenvalue_share": float(eig[0] / eig.sum()), "top4_share": float(eig[:4].sum() / eig.sum()),
        "independent_share": 1.0 / experts,
        "mean_expert_energy_share": float(gram.sum() / experts / gram.trace()),  # 1/E if independent, 1 if all equal
    }
    result["correlation"] = {"mean_abs": float(off.abs().mean()), "max_abs": float(off.abs().max()), "mean": float(off.mean()),
                             "null_scale": 1.0 / math.sqrt(n)}
    # Sign agreement from the Gram matrix of ±1 signs; exponent agreement pair by pair; against independence.
    signs = torch.zeros(experts, experts, dtype=torch.float64, device=device)
    ones = torch.zeros(experts, dtype=torch.float64, device=device)
    for start in range(0, n, 1 << 18):
        s = 1.0 - 2.0 * (p[:, start : start + (1 << 18)] >> 15).to(torch.float32)
        signs += (s @ s.t()).to(torch.float64)
        ones += (s < 0).sum(dim=1).to(torch.float64)
    sign_agree = (1.0 + signs / n) * 0.5
    q = ones / n
    sign_expected = q[:, None] * q[None, :] + (1 - q[:, None]) * (1 - q[None, :])
    exponent_agree = torch.zeros(experts, experts, dtype=torch.float64)
    histograms = torch.zeros(experts, 256, dtype=torch.float64)
    for m in range(experts):
        e_m = (p[m] >> 7) & 0xFF
        histograms[m] = torch.bincount(e_m, minlength=256).to(torch.float64).cpu() / n
        for start in range(0, experts, CHUNK):
            exponent_agree[start : start + CHUNK, m] = (((p[start : start + CHUNK] >> 7) & 0xFF) == e_m).sum(dim=1).to(torch.float64).cpu() / n
    exponent_expected = histograms @ histograms.t()
    mask = ~torch.eye(experts, dtype=torch.bool)
    result["agreement"] = {
        "sign": float(sign_agree.cpu()[mask].mean()), "sign_expected_if_independent": float(sign_expected.cpu()[mask].mean()),
        "exponent": float(exponent_agree[mask].mean()), "exponent_expected_if_independent": float(exponent_expected[mask].mean()),
        "exponent_excess_max_pair": float((exponent_agree - exponent_expected)[mask].max()),
    }
    return result


def permutation_similarity(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, shape: tuple[int, int], pairs: list[tuple[int, int]],
                           seed: int) -> dict:
    """Best |cosine| of each neuron of expert a against every neuron of expert b (neuron = gate row, up row, down column),
    per pair, and the same statistic for independent Gaussian matrices of the same shape (the null). gate, up: [E, I·H]
    patterns; down [E, H·I]; shape (I, H)."""
    neurons_count, hidden = shape

    def neurons(e: int) -> torch.Tensor:
        g = values(gate[e]).view(neurons_count, hidden)
        u = values(up[e]).view(neurons_count, hidden)
        d = values(down[e]).view(hidden, neurons_count).t()
        v = torch.cat([g, u, d], dim=1).to(torch.float64)
        return v / torch.linalg.vector_norm(v, dim=1, keepdim=True)

    out = []
    for a, b in pairs:
        best = (neurons(a) @ neurons(b).t()).abs().max(dim=1).values
        out.append({"pair": [a, b], "mean_best": float(best.mean()), "max_best": float(best.max()),
                    "share_above_0.3": float((best > 0.3).to(torch.float64).mean())})
    generator = torch.Generator(device="cpu").manual_seed(seed)
    width = 3 * hidden
    null = [torch.randn(neurons_count, width, generator=generator, dtype=torch.float64) for _ in range(2)]
    null = [(x / torch.linalg.vector_norm(x, dim=1, keepdim=True)).to(gate.device) for x in null]
    null_best = (null[0] @ null[1].t()).abs().max(dim=1).values
    return {"pairs": out, "mean_best": float(np.mean([r["mean_best"] for r in out])), "max_best": float(max(r["max_best"] for r in out)),
            "null_mean_best": float(null_best.mean()), "null_max_best": float(null_best.max())}


# Base strategies


def assignments(costs: torch.Tensor, candidates: int, clusters: list[int]) -> dict[str, dict]:
    """Base per expert (−1: stored alone) for each strategy, from the pairwise proxy costs ([E, E], diagonal: alone);
    each strategy's predicted bits per weight (mean over the experts, bases stored alone)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    experts = costs.shape[0]
    alone = costs.diagonal().clone()
    delta = costs.clone()
    delta[torch.arange(experts), torch.arange(experts)] = math.inf
    finite_delta = torch.where(torch.isfinite(delta), delta, 0.0)
    out: dict[str, dict] = {}

    def strategy(name: str, base: list[int], extra: dict | None = None) -> None:
        base_t = torch.tensor(base)
        predicted = torch.where(base_t < 0, alone, costs[torch.arange(experts), base_t.clamp_min(0)])
        out[name] = {"base": base, "bases": sorted({b for b in base if b >= 0}), "predicted_bits": float(predicted.mean()),
                     "predicted_alone_bits": float(alone.mean()), **(extra or {})}

    def with_bases(bases: list[int], choose) -> list[int]:
        return [-1 if j in bases else choose(j) for j in range(experts)]

    strategy("A1-first", with_bases([0], lambda j: 0))
    centrality = finite_delta.sum(dim=0) + alone  # Σ_{j≠m} cost(j | m) + m stored alone
    medoid = int(torch.argmin(centrality))
    strategy("A2-medoid", with_bases([medoid], lambda j: medoid))
    central = torch.argsort(centrality, stable=True)[:candidates].tolist()
    strategy(f"A3-best-of-{candidates}", with_bases(central, lambda j: min(central, key=lambda m: float(costs[j, m]))), {"candidates": central})
    symmetric = ((finite_delta + finite_delta.t()) * 0.5).numpy()
    np.fill_diagonal(symmetric, 0.0)
    tree = linkage(squareform(symmetric, checks=False), method="average")
    for k in clusters:
        labels = fcluster(tree, t=k, criterion="maxclust")
        bases, members = {}, {}
        for label in sorted(set(labels.tolist())):
            group = [j for j in range(experts) if labels[j] == label]
            score = costs[group][:, group].sum(dim=0)  # Σ_j cost(j | m) within the group (diagonal: m alone)
            bases[label] = group[int(torch.argmin(score))]
            members[label] = group
        base = [-1 if bases[labels[j]] == j else bases[labels[j]] for j in range(experts)]
        strategy(f"C{k}-clusters", base, {"clusters": [members[label] for label in sorted(members)]})
    best = torch.minimum(alone, delta.min(dim=1).values)
    out["best_pair_bound"] = {"predicted_bits": float(best.mean()), "predicted_alone_bits": float(alone.mean()),
                              "experts_with_a_cheaper_delta": int((delta.min(dim=1).values < alone).sum()),
                              "largest_saving_bits": float((alone - delta.min(dim=1).values).max())}
    return out


def elementwise_median(p: torch.Tensor) -> torch.Tensor:
    """The elementwise median value across the experts (torch's lower median: one of the experts' own patterns)."""
    out = torch.empty(p.shape[1], dtype=p.dtype, device=p.device)
    for start in range(0, p.shape[1], 1 << 18):
        part = p[:, start : start + (1 << 18)]
        index = values(part).median(dim=0).indices
        out[start : start + part.shape[1]] = part.gather(0, index[None, :]).squeeze(0)
    return out


def synthetic_costs(p: torch.Tensor, base: torch.Tensor) -> dict[str, torch.Tensor]:
    """Bits per weight of each expert's XOR and modular deltas against a synthetic base [N], and of the base alone."""
    xor = torch.cat([plane_costs(p[s : s + CHUNK] ^ base[None, :]) for s in range(0, p.shape[0], CHUNK)]).cpu()
    modular = torch.cat([plane_costs((p[s : s + CHUNK] - base[None, :]) & 0xFFFF) for s in range(0, p.shape[0], CHUNK)]).cpu()
    return {"xor": xor, "modular": modular, "base_alone": plane_costs(base[None, :]).cpu()}
