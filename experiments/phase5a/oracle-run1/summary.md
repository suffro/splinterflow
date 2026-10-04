# Phase 5A expert oracle — oracle-run1

**Gate: FAIL.** Best realistic strategy A: mean 1.003 of the routed bytes, coverage 8.1%. Best ideal (certified tier): C, mean 0.993.

## Correctness

| Check | Result |
| --- | --- |
| Capture equals Phase 4B's reference (shared steps) | 144 steps, all equal: True |
| Target layer's weights equal the reference's rows | True |
| Reference recomputation: R | 768/768 |
| Reference recomputation: h | 768/768 |
| Reference recomputation: logits | 768/768 |
| Reference recomputation: m | 768/768 |
| Reference recomputation: shared | 768/768 |
| Reference recomputation: token | 768/768 |
| Reference recomputation: y | 768/768 |
| certified: enclosure violations (ceilings, realistic cells) | 0, 0 |
| certified: wrong (would-)certified tokens | 0 |
| certified_u24: enclosure violations (ceilings, realistic cells) | 0, 0 |
| certified_u24: wrong (would-)certified tokens | 0 |
| real: enclosure violations (ceilings, realistic cells) | 0, 0 |
| real: wrong (would-)certified tokens | 42 |
| rn_elementwise: enclosure violations (ceilings, realistic cells) | 0, 0 |
| rn_elementwise: wrong (would-)certified tokens | 0 |
| rn_even: enclosure violations (ceilings, realistic cells) | 0, 0 |
| rn_even: wrong (would-)certified tokens | 0 |
| Reproducible (digests; same source tree) | True; True |

## Ceilings: every routed weight read

| Tier | Coverage | Margin median (logits) |
| --- | --- | --- |
| certified | 8.1% | -3.434 |
| certified_u24 | 16.5% | -2.339 |
| real | 100.0% | 2.421 |
| rn_elementwise | 17.8% | -2.099 |
| rn_even | 24.7% | -1.296 |

Coverage by the reference's top-2 gap:

| Gap (logits) | Samples | certified | certified_u24 | real | rn_elementwise | rn_even |
| --- | --- | --- | --- | --- | --- | --- |
| [0, 0.5) | 207 | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% |
| [0.5, 1) | 137 | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% |
| [1, 2) | 166 | 0.0% | 0.0% | 100.0% | 0.6% | 3.6% |
| [2, 4) | 133 | 4.5% | 18.8% | 100.0% | 26.3% | 51.9% |
| [4, 8) | 99 | 44.4% | 76.8% | 100.0% | 76.8% | 89.9% |
| [8, inf) | 26 | 46.2% | 100.0% | 100.0% | 96.2% | 100.0% |

Floor of the tightest pair (certified tier, mean, logits):

norm_rounding 1.373, y_rounding 0.493, m_rounding 0.362, R_rounding 0.275, o_rounding 0.348, down_accumulation 0.258, z_rounding 0.000, combine 0.000, base_rounding 0.000, lm_accumulation 0.067, unread_neurons 0.000

## certified/realistic/realistic

| Strategy | Coverage | Mean | Median | p90 | p95 | When certified | Physical (layout: fraction, 4 KiB amplification) | Storage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 8.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000 | 1.00x |
| B1 | 8.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000 | 1.00x |
| B16 | 8.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000 | 1.00x |
| B128 | 8.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000 | 1.00x |
| C | 8.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000; neuron_major_copy: 1.000, 1.000 | 1.33x |
| D-q8 | 8.1% | 1.501 | 1.506 | 1.506 | 1.506 | 1.448 | levels_and_checkpoint: 1.501, 1.003 | 1.50x |
| D-q6+q4 | 8.1% | 1.624 | 1.634 | 1.634 | 1.634 | 1.509 | levels_and_checkpoint: 1.624, 1.004 | 1.63x |
| D-q4+q4 | 8.1% | 1.509 | 1.509 | 1.509 | 1.509 | 1.501 | levels_and_checkpoint: 1.503, 1.001 | 1.51x |

## certified/realistic/ideal

| Strategy | Coverage | Mean | Median | p90 | p95 | When certified | Physical (layout: fraction, 4 KiB amplification) | Storage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 8.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000 | 1.00x |
| B1 | 8.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000 | 1.00x |
| B16 | 8.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000 | 1.00x |
| B128 | 8.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000 | 1.00x |
| C | 8.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000; neuron_major_copy: 1.000, 1.000 | 1.33x |
| D-q8 | 8.1% | 1.502 | 1.506 | 1.506 | 1.506 | 1.453 | levels_and_checkpoint: 1.501, 1.003 | 1.50x |
| D-q6+q4 | 8.1% | 1.633 | 1.634 | 1.634 | 1.634 | 1.615 | levels_and_checkpoint: 1.628, 1.001 | 1.63x |
| D-q4+q4 | 8.1% | 1.509 | 1.509 | 1.509 | 1.509 | 1.504 | levels_and_checkpoint: 1.503, 1.001 | 1.51x |

## certified/ideal/ideal

| Strategy | Coverage | Mean | Median | p90 | p95 | When certified | Physical (layout: fraction, 4 KiB amplification) | Storage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 8.1% | 0.994 | 1.003 | 1.003 | 1.003 | 0.897 | checkpoint: 0.998, 1.006 | 1.00x |
| B1 | 8.1% | 0.993 | 1.003 | 1.003 | 1.003 | 0.887 | checkpoint: 0.998, 1.007 | 1.00x |
| B16 | 8.1% | 0.994 | 1.003 | 1.003 | 1.003 | 0.897 | checkpoint: 0.998, 1.006 | 1.00x |
| B128 | 8.1% | 0.994 | 1.003 | 1.003 | 1.003 | 0.897 | checkpoint: 0.998, 1.006 | 1.00x |
| C | 8.1% | 0.993 | 1.003 | 1.003 | 1.003 | 0.879 | checkpoint: 0.998, 1.005; neuron_major_copy: 0.990, 1.000 | 1.33x |
| D-q8 | 8.1% | 1.439 | 1.506 | 1.506 | 1.506 | 0.678 | levels_and_checkpoint: 1.440, 1.004 | 1.50x |
| D-q6+q4 | 8.1% | 1.555 | 1.634 | 1.634 | 1.634 | 0.648 | levels_and_checkpoint: 1.556, 1.006 | 1.63x |
| D-q4+q4 | 8.1% | 1.458 | 1.509 | 1.509 | 1.509 | 0.879 | levels_and_checkpoint: 1.463, 1.008 | 1.51x |

## rn_even/realistic/realistic

| Strategy | Coverage | Mean | Median | p90 | p95 | When certified | Physical (layout: fraction, 4 KiB amplification) | Storage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 27.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000 | 1.00x |
| B1 | 27.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000 | 1.00x |
| B16 | 27.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000 | 1.00x |
| B128 | 27.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000 | 1.00x |
| C | 27.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000; neuron_major_copy: 1.000, 1.000 | 1.33x |
| D-q8 | 27.1% | 1.482 | 1.506 | 1.506 | 1.506 | 1.416 | levels_and_checkpoint: 1.495, 1.013 | 1.50x |
| D-q6+q4 | 27.1% | 1.584 | 1.634 | 1.634 | 1.634 | 1.448 | levels_and_checkpoint: 1.606, 1.018 | 1.63x |
| D-q4+q4 | 27.1% | 1.506 | 1.509 | 1.509 | 1.509 | 1.498 | levels_and_checkpoint: 1.503, 1.002 | 1.51x |

## real/realistic/realistic

| Strategy | Coverage | Mean | Median | p90 | p95 | When certified | Physical (layout: fraction, 4 KiB amplification) | Storage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 100.0% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000 | 1.00x |
| B1 | 100.0% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000 | 1.00x |
| B16 | 100.0% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000 | 1.00x |
| B128 | 100.0% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000 | 1.00x |
| C | 100.0% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | checkpoint: 1.000, 1.000; neuron_major_copy: 1.000, 1.000 | 1.33x |
| D-q8 | 100.0% | 1.289 | 1.350 | 1.490 | 1.506 | 1.289 | levels_and_checkpoint: 1.393, 1.085 | 1.50x |
| D-q6+q4 | 100.0% | 1.266 | 1.274 | 1.602 | 1.633 | 1.266 | levels_and_checkpoint: 1.381, 1.096 | 1.63x |
| D-q4+q4 | 100.0% | 1.464 | 1.477 | 1.509 | 1.509 | 1.464 | levels_and_checkpoint: 1.493, 1.025 | 1.51x |

## real/ideal/ideal

| Strategy | Coverage | Mean | Median | p90 | p95 | When certified | Physical (layout: fraction, 4 KiB amplification) | Storage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 100.0% | 0.708 | 0.695 | 0.898 | 0.930 | 0.708 | checkpoint: 0.830, 1.176 | 1.00x |
| B1 | 100.0% | 0.707 | 0.706 | 0.893 | 0.924 | 0.707 | checkpoint: 0.857, 1.217 | 1.00x |
| B16 | 100.0% | 0.706 | 0.706 | 0.893 | 0.924 | 0.706 | checkpoint: 0.828, 1.177 | 1.00x |
| B128 | 100.0% | 0.706 | 0.706 | 0.893 | 0.924 | 0.706 | checkpoint: 0.828, 1.177 | 1.00x |
| C | 100.0% | 0.853 | 0.940 | 1.003 | 1.003 | 0.853 | checkpoint: 0.964, 1.070; neuron_major_copy: 0.851, 1.000 | 1.33x |
| D-q8 | 100.0% | 0.559 | 0.506 | 0.678 | 0.912 | 0.559 | levels_and_checkpoint: 0.581, 1.050 | 1.50x |
| D-q6+q4 | 100.0% | 0.471 | 0.399 | 0.633 | 0.883 | 0.471 | levels_and_checkpoint: 0.521, 1.122 | 1.63x |
| D-q4+q4 | 100.0% | 0.534 | 0.399 | 1.118 | 1.461 | 0.534 | levels_and_checkpoint: 0.636, 1.207 | 1.51x |

## Projection (Q6)

- certified_realistic: A, fraction 1.003: last layer only 2.70 GB, every layer alike 2.71 GB (baseline 2.7 GB)
- certified_ideal: C, fraction 0.993: last layer only 2.70 GB, every layer alike 2.68 GB (baseline 2.7 GB)
- real_ideal: D-q6+q4, fraction 0.471: last layer only 2.65 GB, every layer alike 1.27 GB (baseline 2.7 GB)
