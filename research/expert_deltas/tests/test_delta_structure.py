"""The structure census sees shared structure when it exists and none when experts are independent; base assignments."""

from __future__ import annotations

import numpy as np
import torch

from expert_deltas import bits, structure


def experts_from(values: np.ndarray) -> torch.Tensor:
    p = bits.patterns(torch.from_numpy(values.astype(np.float32)).to(torch.bfloat16))
    return structure.to_device(p, "cpu")


def test_independent_experts_show_no_shared_structure():
    rng = np.random.default_rng(0)
    p = experts_from(rng.normal(0, 0.02, (16, 64, 128)))
    c = structure.census(p, 64, 128)
    assert abs(c["spectrum"]["top_eigenvalue_share"] - 1 / 16) < 0.03
    assert c["correlation"]["mean_abs"] < 5 * c["correlation"]["null_scale"]
    assert abs(c["agreement"]["sign"] - c["agreement"]["sign_expected_if_independent"]) < 0.01
    costs = structure.pairwise_costs(p, "xor")
    alone = costs.diagonal()
    off = costs[~torch.eye(16, dtype=torch.bool)].view(16, 15)
    assert bool((off > alone[:, None]).all())  # an independent expert's delta always costs more than the expert alone


def test_near_copies_are_cheap_deltas_and_dominate_the_spectrum():
    rng = np.random.default_rng(1)
    shared = rng.normal(0, 0.02, (64, 128))
    values = shared[None] + rng.normal(0, 0.0005, (16, 64, 128))  # experts = a shared matrix + a small private part
    p = experts_from(values)
    c = structure.census(p, 64, 128)
    assert c["spectrum"]["top_eigenvalue_share"] > 0.9
    assert c["agreement"]["sign"] > 0.9 and c["agreement"]["exponent"] > c["agreement"]["exponent_expected_if_independent"]
    costs = structure.pairwise_costs(p, "xor")
    found = structure.assignments(costs, 3, [2, 4])
    assert found["A2-medoid"]["predicted_bits"] < found["A2-medoid"]["predicted_alone_bits"]
    assert found["best_pair_bound"]["experts_with_a_cheaper_delta"] == 16


def test_column_scales_shared_across_experts_are_seen_as_context():
    rng = np.random.default_rng(2)
    scale = np.exp(rng.normal(0, 1.5, 128))  # a per-input-column scale every expert shares
    p = experts_from(rng.normal(0, 1, (16, 64, 128)) * scale[None, None, :] * 0.01)
    c = structure.census(p, 64, 128)
    assert c["exponent_context_saving_bits"]["column"] > 0.5
    assert c["exponent_context_saving_bits"]["column"] > c["context_description_bits"]["column"]


def test_assignments_cover_every_strategy_and_bases_are_alone():
    rng = np.random.default_rng(3)
    costs = torch.from_numpy(rng.uniform(10.0, 11.0, (12, 12)))
    found = structure.assignments(costs, 3, [2, 3])
    for name in ("A1-first", "A2-medoid", "A3-best-of-3", "C2-clusters", "C3-clusters"):
        base = found[name]["base"]
        for b in found[name]["bases"]:
            assert base[b] == -1  # a base is stored alone
        assert all(b == -1 or base[b] == -1 for b in base)
    assert found["A1-first"]["base"][0] == -1 and set(found["A1-first"]["base"][1:]) == {0}


def test_elementwise_median_is_one_of_the_experts_patterns():
    rng = np.random.default_rng(4)
    p = experts_from(rng.normal(0, 0.02, (7, 8, 16)))
    median = structure.elementwise_median(p)
    assert bool((p == median[None, :]).any(dim=0).all())
    v = structure.values(p)
    m = structure.values(median[None, :])[0]
    assert bool(((v <= m[None, :]).sum(dim=0) >= 4).all())  # the lower median of 7 values
