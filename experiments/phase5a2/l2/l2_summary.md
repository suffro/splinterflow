# Phase 5A2 stage 1.5 (L2 probe) summary: experiments\phase5a2\l2

- verifier: Python 3.11.16, torch 2.11.0+cu130, auto_LiRPA 0.7.2 @ `5a098e8f9fb5`; GPU NVIDIA GeForce RTX 4060 Ti; PYTHONHASHSEED 1

## Correctness

| Check | Result |
| --- | --- |
| samples validated | 3 |
| wrapper max relative error | 1.93e-16 |
| sets without the true weights (boxes and L2 balls, every state) | 0 |
| sets unchanged by poisoned unread bytes (boxes and L2 balls) | yes |
| assembly vs Phase 5A margins (max relative difference) | 7.87e-11 |
| stage 1.5 states recorded (sets; CROWN points) | [246, 18] |
| stage 1.5 failures (a bound above an attained value or the truth, a witness outside its set or below its set's lower bound) | {"l2": [], "l2crown": []} |
| largest witness constraint excess | 2.22e-16 |
| reference reproduced bitwise at export | yes |

Units: lower bounds and attained values of Δ·y in real arithmetic (Δ = (W_w − W_j)⊙g; positive decides the pair). Per state, each quantity's minimum over the focus contenders (the runner-up, the tightest by Phase 5A's realistic bound, the tightest truly). Reduced sets: Phase 5A's activation enclosure × down's rows in their sets; weight sets: gate, up and down rows in their sets. `[a, b]`: a lower bound and an attained value of the set's minimum.

## Sample 0: prompt 12, step 7, gap 0.000 (stage 2 position 0)

| Bytes | Truth | 5A realistic | Box CROWN | Box set (reduced) | Box set (weights) | L2 CROWN | L2 set (reduced) | L2 set (weights) | Resident witness | 5A ideal | Gap closed by L2 CROWN | L2 CROWN s | peak VRAM GB | peak RSS GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.383 | 0.108 | -950.268 | -2178.512 | -1097.430 | -1862.597 | -3.22e+04 | [-950.268, -803.052] | [-950.264, -799.972] | -576.722 | -3.474 | -35.879 | 105.0 | 0.76 | 1.53 |
| 0.696 | 0.108 | -30.588 | -33.682 | -27.748 | -51.893 | -258.071 | [-30.711, -28.509] | [-30.711, -20.952] | -14.040 | -0.157 | -8.127 | 86.9 | 0.75 | 1.53 |
| 1.008 | 0.108 | -5.576 | -5.689 | -5.265 | -7.512 | -19.969 | [-5.647, -5.446] | [-5.647, -3.968] | -3.127 | 0.023 | -2.361 | 25.2 | 0.70 | 1.53 |
| 1.336 | 0.108 | -1.804 | -1.806 | -1.760 | -2.450 | -3.051 | [-1.843, -1.820] | [-1.843, -1.306] | -1.035 | 0.076 | -0.594 | 6.9 | 0.71 | 1.53 |
| 1.524 | 0.108 | -0.634 | -0.635 | -0.625 | -0.914 | -0.887 | [-0.664, -0.660] | [-0.664, -0.480] | -0.338 | 0.095 | -0.283 | 3.3 | 0.71 | 1.53 |
| 1.634 | 0.108 | 0.107 | 0.107 | 0.107 | 0.108 | 0.107 | [0.107, 0.107] | [0.108, 0.108] | 0.108 | 0.107 | – | 1.3 | 0.71 | 1.53 |

Routed-byte fraction from which each decides every comparison pair (every state of the schedule; CROWN at the comparison points), and the last fraction at which a set holds weights that flip a pair:

| Quantity | Decides from | | Set | Flipping weights until |
| --- | --- | --- | --- | --- |
| phase5a_realistic | 1.634 | | box | 1.633 |
| phase5a_ideal | 0.946 | | l2 | 1.633 |
| box_set_optimum_reduced | 1.634 | | resident | 1.617 |
| box_set_optimum_weights | 1.634 | | l2_set_reduced_attained | 1.633 |
| l2_set_reduced_lower | 1.634 | | box_set_reduced | 1.633 |
| l2_set_reduced_attained | 1.634 | |  | – |
| l2_set_weights_lower | 1.634 | |  | – |
| box_crown (points) | 1.634 | |  | – |
| l2_crown (points) | 1.634 | |  | – |
| phase5a_realistic (points) | 1.634 | |  | – |

Phase 5A's own cells on this sample: {"realistic/realistic": {"fraction": 1.6343217329545454, "would_certify": true}, "ideal/ideal": {"fraction": 0.9925446558480311, "would_certify": true}}; gap closed by L2 CROWN (median, before full read): -3.999, by the L2 set's attained optimum at most 0.036; witness rows pulled toward the truth: 1309183; down balls not nested in earlier ones: 312023.

## Sample 1: prompt 12, step 9, gap 3.875 (stage 2 position 6)

| Bytes | Truth | 5A realistic | Box CROWN | Box set (reduced) | Box set (weights) | L2 CROWN | L2 set (reduced) | L2 set (weights) | Resident witness | 5A ideal | Gap closed by L2 CROWN | L2 CROWN s | peak VRAM GB | peak RSS GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.383 | 8.944 | -738.560 | -1710.882 | -860.962 | -1478.584 | -2.53e+04 | [-738.560, -628.956] | [-738.557, -626.459] | -451.831 | 6.538 | -34.806 | 104.5 | 0.73 | 1.98 |
| 0.696 | 8.944 | -22.349 | -24.890 | -19.960 | -39.014 | -208.803 | [-22.770, -20.883] | [-22.770, -14.687] | -9.012 | 8.730 | -7.288 | 88.6 | 0.73 | 1.98 |
| 1.008 | 8.944 | 2.379 | 2.282 | 2.671 | -1.770 | -9.109 | [2.244, 2.391] | [2.245, 4.843] | 6.081 | 8.873 | -1.882 | 25.4 | 0.70 | 1.98 |
| 1.336 | 8.944 | 7.529 | 7.526 | 7.564 | 7.028 | 6.506 | [7.491, 7.507] | [7.491, 7.888] | 8.101 | 8.919 | -0.736 | 7.1 | 0.71 | 1.98 |
| 1.524 | 8.944 | 8.445 | 8.446 | 8.451 | 8.301 | 8.276 | [8.423, 8.425] | [8.423, 8.562] | 8.656 | 8.933 | -0.346 | 3.2 | 0.71 | 1.98 |
| 1.634 | 8.944 | 8.944 | 8.944 | 8.944 | 8.944 | 8.944 | [8.944, 8.944] | [8.944, 8.944] | 8.944 | 8.944 | – | 1.3 | 0.71 | 1.98 |

Routed-byte fraction from which each decides every comparison pair (every state of the schedule; CROWN at the comparison points), and the last fraction at which a set holds weights that flip a pair:

| Quantity | Decides from | | Set | Flipping weights until |
| --- | --- | --- | --- | --- |
| phase5a_realistic | 0.961 | | box | 1.039 |
| phase5a_ideal | 0.383 | | l2 | 0.868 |
| box_set_optimum_reduced | 0.946 | | resident | 0.821 |
| box_set_optimum_weights | 1.055 | | l2_set_reduced_attained | 0.946 |
| l2_set_reduced_lower | 0.961 | | box_set_reduced | 0.930 |
| l2_set_reduced_attained | 0.961 | |  | – |
| l2_set_weights_lower | 0.961 | |  | – |
| box_crown (points) | 1.008 | |  | – |
| l2_crown (points) | 1.336 | |  | – |
| phase5a_realistic (points) | 1.008 | |  | – |

Phase 5A's own cells on this sample: {"realistic/realistic": {"fraction": 0.9612546593251855, "would_certify": true}, "ideal/ideal": {"fraction": 0.3831972064393939, "would_certify": true}}; gap closed by L2 CROWN (median, before full read): -4.022, by the L2 set's attained optimum at most 0.027; witness rows pulled toward the truth: 914242; down balls not nested in earlier ones: 315966.

## Sample 2: prompt 23, step 4, gap 11.062 (stage 2 position 11)

| Bytes | Truth | 5A realistic | Box CROWN | Box set (reduced) | Box set (weights) | L2 CROWN | L2 set (reduced) | L2 set (weights) | Resident witness | 5A ideal | Gap closed by L2 CROWN | L2 CROWN s | peak VRAM GB | peak RSS GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.383 | 20.908 | -2010.561 | -4735.940 | -2391.932 | -3951.565 | -6.99e+04 | [-2010.561, -1702.669] | [-2010.554, -1694.608] | -1240.079 | 11.155 | -32.531 | 104.5 | 0.73 | 2.03 |
| 0.696 | 20.908 | -66.384 | -74.096 | -60.774 | -111.272 | -561.983 | [-66.989, -62.114] | [-66.989, -46.609] | -31.622 | 19.968 | -5.412 | 84.2 | 0.73 | 2.03 |
| 1.008 | 20.908 | 0.834 | 0.569 | 1.383 | -8.572 | -24.519 | [0.558, 0.941] | [0.559, 6.383] | 8.966 | 20.598 | -1.202 | 28.8 | 0.71 | 2.03 |
| 1.336 | 20.908 | 15.113 | 15.107 | 15.167 | 12.407 | 13.468 | [15.042, 15.071] | [15.043, 16.652] | 17.276 | 20.818 | -0.288 | 8.2 | 0.71 | 2.03 |
| 1.524 | 20.908 | 19.291 | 19.294 | 19.298 | 18.514 | 19.175 | [19.269, 19.270] | [19.269, 19.687] | 19.855 | 20.882 | -0.073 | 3.2 | 0.71 | 2.03 |
| 1.634 | 20.908 | 20.908 | 20.908 | 20.908 | 20.908 | 20.908 | [20.908, 20.908] | [20.908, 20.908] | 20.908 | 20.908 | – | 1.3 | 0.71 | 2.03 |

Routed-byte fraction from which each decides every comparison pair (every state of the schedule; CROWN at the comparison points), and the last fraction at which a set holds weights that flip a pair:

| Quantity | Decides from | | Set | Flipping weights until |
| --- | --- | --- | --- | --- |
| phase5a_realistic | 1.008 | | box | 1.086 |
| phase5a_ideal | 0.383 | | l2 | 0.914 |
| box_set_optimum_reduced | 0.993 | | resident | 0.852 |
| box_set_optimum_weights | 1.102 | | l2_set_reduced_attained | 0.993 |
| l2_set_reduced_lower | 1.008 | | box_set_reduced | 0.977 |
| l2_set_reduced_attained | 1.008 | |  | – |
| l2_set_weights_lower | 1.008 | |  | – |
| box_crown (points) | 1.008 | |  | – |
| l2_crown (points) | 1.336 | |  | – |
| phase5a_realistic (points) | 1.008 | |  | – |

Phase 5A's own cells on this sample: {"realistic/realistic": {"fraction": 1.0081953568892046, "would_certify": true}, "ideal/ideal": {"fraction": 0.3831972064393939, "would_certify": true}}; gap closed by L2 CROWN (median, before full read): -3.274, by the L2 set's attained optimum at most 0.026; witness rows pulled toward the truth: 960765; down balls not nested in earlier ones: 311065.

## Stop conditions (config `l2`, the brief's §17–18)

- **B_uncertainty_set**: not triggered — {"met_on_samples": [0, 2], "l2_set_weights_flip_until": {"0": 1.633195125695431, "1": 0.8675705880829783, "2": 0.9144102539679017}, "l2_reduced_set_flips_until": {"0": 1.633195125695431, "1": 0.9456387144146543, "2": 0.992505507035689}, "resident_set_flips_until": {"0": 1.6174941400084832, "1": 0.8206878430915602, "2": 0.8519370339133523}, "near_full_fraction": 0.9}
- **C_weak_improvement**: TRIGGERED — {"gap_closed_median": -3.998571807885431, "threshold": 0.2, "l2_crown_decides_from": {"0": 1.6343217329545454, "1": 1.3362638685438368, "2": 1.3362466060754024}, "states_where_l2_crown_beats_phase5a_realistic": 0}
- **D_verifier_cost**: not triggered — {"l2_crown_seconds_per_state_max": 104.98006829991937, "threshold": 600}
- STOP A (representation): the probe's finding, `probe_l2_A.json`.
- **Proceed to stage 2: no**

