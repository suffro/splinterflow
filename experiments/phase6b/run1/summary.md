# Phase 6B run experiments/phase6b/run1

## Gates

| Gate | Result |
| --- | --- |
| correctness | PASS |
| encoded | PASS |
| io_parity | PASS |
| caches | PASS |
| layering | PASS |
| 6B1: engine submit ms per decode token (max over the engine configurations) | 1.34 (PASS) |
| 6B1: BF16 host-cache decode against the baseline tree | 0.964 (PASS) |
| 6B2: encoded against BF16 rows, same host budget | 0.776 (PASS) |
| 6B3-A: device cache against the same host budget without it | 0.808, peak reserved 5.80 GB (PASS) |
| 6B4: decode graphs and the fills off, against the same configuration without them | 0.828 (PASS) |
| Phase: best warm decode (encoded-host-12g-freeze-dev1.9g-fast) against Phase 6A's | 0.507 (PASS; aspirational met) |

## Correctness

| Configuration | Steps | Equal | Audit failures | Poisoned | Chunked calls |
| --- | --- | --- | --- | --- | --- |
| python-stream | 144 | 144 | 0 | 27 | 416 |
| native-stream | 144 | 144 | 0 | 27 | 416 |
| native-host-12g-freeze | 144 | 144 | 0 | 0 | 416 |
| encoded-stream | 144 | 144 | 0 | 27 | 416 |
| encoded-chunk-1 | 36 | 36 | 0 | 0 | 936 |
| encoded-host-tiny | 36 | 36 | 0 | 0 | 104 |
| encoded-host-12g-freeze | 144 | 144 | 0 | 0 | 416 |
| encoded-dev-tiny | 36 | 36 | 0 | 0 | 104 |
| encoded-host-12g-freeze-dev1.9g | 144 | 144 | 0 | 0 | 416 |
| encoded-host-12g-freeze-dev1.9g-fast | 144 | 144 | 0 | 0 | 416 |

Encoded pack: 3328 rows of 52 segments decoded on the GPU before inference (19.42 GB stored, 28.79 GB restored, ratio 0.6747): differing from the reference 0, from the pack's record 0, missing 0.

I/O parity (native-stream against python-stream): records True over [144, 144] steps; raw ranges True; OS counters True.

## Caches against a replay of the reference's routing

| Configuration | Host (GB) | Device slots | Host decode hits | Device decode byte hits | Evictions (host / device) | Within budget | Replay = measured |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native-host-12g-freeze | 12.00 | – | 0.724 | – | 9612 / – | True | True |
| encoded-host-tiny | 0.03 | – | 0.000 | – | 7232 / – | True | None |
| encoded-host-12g-freeze | 12.00 | – | 0.856 | – | 3678 / – | True | True |
| encoded-dev-tiny | 0.00 | {3891200: 4, 7782400: 4} | – | 0.000 | – / 18982 | True | True |
| encoded-host-12g-freeze-dev1.9g | 12.00 | {3891200: 162, 7782400: 162} | 0.753 | 0.380 | 4048 / 24446 | True | True |
| encoded-host-12g-freeze-dev1.9g-fast | 12.00 | {3891200: 162, 7782400: 162} | 0.753 | 0.380 | 4048 / 24446 | True | True |

## Performance (profiles; decode means over 4 prompts x 8 steps)

| Profile | Decode ms/token (median, p95) | Tokens/s | Drive GB/token | H2D GB/token | Copy ms | Decode host ms | Submit ms | Host / device hits | Launches/token (kernels + graphs; kernels run) | Peak RAM / VRAM alloc / reserved (GB) | Prefill ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| encoded-host-12g-freeze-cold | 620.8 (604.2, 696.7) | 1.611 | 0.463 | 1.821 | 292.1 | 17.8 | 1.50 | 0.746 / – | – + –; – | 15.32 / 3.82 / 3.90 | 4583 |
| encoded-host-12g-freeze-dev1.9g-cold | 520.7 (501.9, 660.8) | 1.920 | 0.464 | 1.241 | 199.7 | 17.6 | 1.48 | 0.626 / 0.319 | – + –; – | 15.32 / 5.71 / 5.80 | 4455 |
| encoded-host-12g-freeze-dev1.9g-fast-cold | 436.6 (410.4, 574.1) | 2.290 | 0.464 | 1.241 | 199.8 | 15.6 | 1.43 | 0.626 / 0.319 | – + –; – | 15.34 / 5.78 / 5.90 | 4410 |
| encoded-host-12g-freeze-dev1.9g-fast-warm-seed2 | 408.7 (395.4, 499.7) | 2.447 | 0.270 | 1.237 | 198.3 | 15.1 | 0.79 | 0.782 / 0.321 | – + –; – | 15.40 / 5.78 / 5.90 | 3593 |
| encoded-host-12g-freeze-dev1.9g-fast-warm-trace | 420.8 (409.9, 497.9) | 2.377 | 0.266 | 1.238 | 198.8 | 16.8 | 0.88 | 0.785 / 0.320 | 2344 + 54; 4161 | 16.27 / 5.73 / 5.90 | 3387 |
| encoded-host-12g-freeze-dev1.9g-fast-warm | 409.0 (393.4, 492.6) | 2.445 | 0.270 | 1.237 | 198.4 | 15.2 | 0.78 | 0.782 / 0.321 | – + –; – | 15.40 / 5.78 / 5.90 | 3593 |
| encoded-host-12g-freeze-dev1.9g-graphs-warm | 431.9 (416.1, 525.2) | 2.316 | 0.270 | 1.237 | 198.4 | 17.1 | 0.80 | 0.782 / 0.321 | – + –; – | 15.40 / 5.78 / 5.90 | 3629 |
| encoded-host-12g-freeze-dev1.9g-nofill-warm | 462.9 (447.4, 547.8) | 2.160 | 0.270 | 1.237 | 198.4 | 15.3 | 0.83 | 0.782 / 0.321 | – + –; – | 15.38 / 5.71 / 5.79 | 3593 |
| encoded-host-12g-freeze-dev1.9g-warm-seed2 | 494.6 (491.2, 582.4) | 2.022 | 0.270 | 1.237 | 198.5 | 17.2 | 0.84 | 0.782 / 0.321 | – + –; – | 15.38 / 5.71 / 5.79 | 3636 |
| encoded-host-12g-freeze-dev1.9g-warm | 492.7 (478.5, 578.8) | 2.029 | 0.270 | 1.237 | 198.4 | 17.3 | 0.70 | 0.782 / 0.321 | – + –; – | 15.38 / 5.71 / 5.79 | 3631 |
| encoded-host-12g-freeze-warm-seed2 | 609.7 (607.2, 623.0) | 1.640 | 0.241 | 1.821 | 291.6 | 17.6 | 0.90 | 0.869 / – | – + –; – | 15.39 / 3.82 / 3.90 | 3944 |
| encoded-host-12g-freeze-warm | 612.0 (608.8, 637.3) | 1.634 | 0.241 | 1.821 | 291.6 | 17.5 | 0.90 | 0.869 / – | – + –; – | 15.39 / 3.82 / 3.90 | 3955 |
| encoded-host-4g-warm | 670.0 (619.7, 849.9) | 1.493 | 0.947 | 1.821 | 298.3 | 18.1 | 0.93 | 0.481 / – | – + –; – | 7.38 / 3.82 / 3.90 | 5896 |
| encoded-stream-warm | 862.4 (862.6, 869.7) | 1.160 | 1.821 | 1.821 | 295.9 | 17.6 | 0.71 | – / – | – + –; – | 3.38 / 3.82 / 3.90 | 6050 |
| native-host-12g-freeze-cold | 797.1 (778.7, 883.6) | 1.255 | 0.906 | 2.699 | 436.3 | – | 1.48 | 0.665 / – | – + –; – | 15.29 / 3.70 / 3.84 | 6262 |
| native-host-12g-freeze-warm-seed2 | 787.0 (777.6, 867.8) | 1.271 | 0.758 | 2.699 | 435.7 | – | 1.31 | 0.720 / – | – + –; – | 15.37 / 3.70 / 3.84 | 5655 |
| native-host-12g-freeze-warm-trace | 820.4 (809.2, 891.6) | 1.219 | 0.753 | 2.699 | 439.6 | – | 1.34 | 0.722 / – | 5195 + 0; 5195 | 16.25 / 3.65 / 3.84 | 5289 |
| native-host-12g-freeze-warm | 787.4 (780.6, 872.4) | 1.270 | 0.758 | 2.699 | 435.9 | – | 1.03 | 0.720 / – | – + –; – | 15.37 / 3.70 / 3.84 | 5647 |
| native-host-4g-warm-seed2 | 904.6 (894.7, 1074.8) | 1.105 | 1.823 | 2.699 | 446.6 | – | 1.14 | 0.325 / – | – + –; – | 7.37 / 3.70 / 3.84 | 7879 |
| native-host-4g-warm | 903.0 (868.1, 1063.1) | 1.107 | 1.823 | 2.699 | 446.6 | – | 1.26 | 0.325 / – | – + –; – | 7.37 / 3.70 / 3.84 | 7836 |
| native-stream-warm | 1122.3 (1123.8, 1136.2) | 0.891 | 2.700 | 2.699 | 438.9 | – | 0.95 | – / – | – + –; – | 3.38 / 3.70 / 3.84 | 7869 |

Warm decode per configuration (mean of its runs; each run; spread):

- encoded-host-12g-freeze-dev1.9g-fast: 408.9 ms (408.7, 409.0; spread 0.1%)
- encoded-host-12g-freeze-dev1.9g-graphs: 431.9 ms (431.9; spread –%)
- encoded-host-12g-freeze-dev1.9g-nofill: 462.9 ms (462.9; spread –%)
- encoded-host-12g-freeze-dev1.9g: 493.7 ms (494.6, 492.7; spread 0.4%)
- encoded-host-12g-freeze: 610.8 ms (609.7, 612.0; spread 0.4%)
- encoded-host-4g: 670.0 ms (670.0; spread –%)
- native-host-12g-freeze: 787.2 ms (787.0, 787.4; spread 0.1%)
- encoded-stream: 862.4 ms (862.4; spread –%)
- native-host-4g: 903.8 ms (904.6, 903.0; spread 0.2%)
- native-stream: 1122.3 ms (1122.3; spread –%)

Compared with experiments/phase6b/run2: digests equal True, same source tree True, same native tree True.
