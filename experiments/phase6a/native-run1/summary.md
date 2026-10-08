# Phase 6A run experiments/phase6a/native-run1

## Gates

| Gate | Result |
| --- | --- |
| correctness | PASS |
| io_parity | PASS |
| host_cache | PASS |
| performance | PASS |
| best native (native-host-12g-freeze) / best Python (python-hotness-80) decode, warm | 0.697 |
| best native / python-stream decode, warm (aspirational ≤ 0.80) | 0.654 (met) |
| layering | PASS |

## Correctness

| Configuration | Steps | Equal | Audit failures | Poisoned | Chunked calls | Expert outputs checked (layers) |
| --- | --- | --- | --- | --- | --- | --- |
| python-stream | 144 | 144 | 0 | 27 | 416 | 3744 |
| native-stream | 144 | 144 | 0 | 27 | 416 | 3744 |
| native-chunk-1 | 36 | 36 | 0 | 0 | 936 | 936 |
| native-host-tiny | 36 | 36 | 0 | 0 | 104 | 884 |
| native-host-4g | 144 | 144 | 0 | 0 | 416 | 3380 |
| native-host-12g | 144 | 144 | 0 | 0 | 416 | 3380 |
| native-host-12g-prefetch | 144 | 144 | 0 | 0 | 416 | 3380 |
| native-host-12g-freeze | 144 | 144 | 0 | 0 | 416 | 3380 |

I/O parity (native-stream against python-stream): records True over [144, 144] steps (differing fields: none); raw ranges True over [27, 27] steps; OS counters True.

## Host cache

| Configuration | Budget (GB) | Decode hits | Prefill hits | Drive GB / decode token | Evictions | Bypassed | Max resident (GB) | Max working set (GB) | Replay = measured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native-host-tiny | 0.03 | 0.000 | 0.000 | 2.700 | 5495 | 13492 | 0.03 | 3.41 | None |
| native-host-4g | 4.00 | 0.383 | 0.006 | 1.668 | 66108 | 0 | 4.00 | 7.38 | True |
| native-host-12g | 12.00 | 0.700 | 0.131 | 0.814 | 47239 | 0 | 12.00 | 15.39 | True |
| native-host-12g-prefetch | 12.00 | 0.700 | 1.000 | 0.814 | 46904 | 0 | 12.00 | 15.45 | True |
| native-host-12g-freeze | 12.00 | 0.724 | 0.447 | 0.746 | 9612 | 23337 | 12.00 | 15.40 | True |

## Performance (profiles, un-instrumented; decode means over 4 prompts x 8 steps)

| Profile | Decode ms/token (median, p95) | Tokens/s | Drive GB/token | Drive GB/s | Host-cache hits | I/O wait ms | Main-thread CPU ms | Requests / read calls / extents per token | Read amplification | Peak RAM (GB) | Peak VRAM (GB) | GPU idle | H2D device ms | Prefill ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native-host-12g-cold | 866.3 (831.9, 1013.2) | 1.154 | 0.825 | 3.28 | 0.696 | 221.7 | 563.0 | 52 / 858 / 95 | 1.0005 | 16.45 | 3.40 | 0.47 | 439.3 | 7146 |
| native-host-12g-freeze-cold | 808.8 (789.0, 911.9) | 1.236 | 0.906 | 3.30 | 0.665 | 228.6 | 504.4 | 52 / 942 / 105 | 1.0005 | 15.28 | 3.40 | – | 435.6 | 6232 |
| native-host-12g-freeze-warm | 807.0 (797.1, 897.1) | 1.239 | 0.758 | 3.29 | 0.720 | 188.8 | 519.5 | 52 / 788 / 87 | 1.0005 | 15.36 | 3.40 | – | 435.4 | 5584 |
| native-host-12g-prefetch-cold | 838.9 (800.8, 987.8) | 1.192 | 0.847 | 3.28 | 0.688 | 231.7 | 517.6 | 52 / 881 / 97 | 1.0005 | 15.34 | 3.40 | – | 436.1 | 9366 |
| native-host-12g-prefetch-warm | 834.4 (796.9, 980.3) | 1.198 | 0.847 | 3.30 | 0.688 | 231.5 | 500.5 | 52 / 881 / 97 | 1.0005 | 15.40 | 3.40 | – | 436.1 | 9269 |
| native-host-12g-warm-repeat | 837.3 (799.6, 986.1) | 1.194 | 0.848 | 3.31 | 0.688 | 227.4 | 505.9 | 52 / 881 / 97 | 1.0005 | 15.37 | 3.40 | – | 436.5 | 7629 |
| native-host-12g-warm | 863.0 (826.7, 1006.5) | 1.159 | 0.825 | 3.29 | 0.696 | 220.9 | 547.4 | 52 / 858 / 95 | 1.0005 | 16.48 | 3.40 | 0.47 | 438.1 | 7077 |
| native-host-4g-cold | 939.7 (918.6, 1110.7) | 1.064 | 1.823 | 3.32 | 0.325 | 547.7 | 379.9 | 52 / 1896 / 210 | 1.0005 | 7.28 | 3.40 | – | 444.4 | 8293 |
| native-host-4g-warm | 944.5 (911.7, 1110.3) | 1.059 | 1.823 | 3.31 | 0.325 | 549.0 | 366.7 | 52 / 1896 / 210 | 1.0005 | 7.37 | 3.40 | – | 445.1 | 8218 |
| native-host-8g-cold | 870.0 (814.8, 1067.3) | 1.149 | 1.233 | 3.27 | 0.544 | 348.7 | 456.5 | 52 / 1282 / 142 | 1.0005 | 11.29 | 3.40 | – | 438.5 | 8136 |
| native-host-8g-warm | 861.8 (813.4, 1056.7) | 1.160 | 1.233 | 3.33 | 0.544 | 344.7 | 448.7 | 52 / 1282 / 142 | 1.0005 | 11.37 | 3.40 | – | 438.8 | 8080 |
| native-stream-cold | 1156.6 (1154.9, 1174.2) | 0.865 | 2.700 | 3.38 | – | 747.1 | 401.6 | 52 / 2808 / 312 | 1.0005 | 4.45 | 3.40 | 0.56 | 442.0 | 7389 |
| native-stream-warm | 1119.6 (1118.9, 1141.5) | 0.893 | 2.700 | 3.38 | – | 754.3 | 347.2 | 52 / 2808 / 312 | 1.0005 | 3.37 | 3.40 | – | 439.0 | 7845 |
| python-hotness-80-cold | 1162.9 (1147.1, 1279.5) | 0.860 | 2.340 | 3.19 | – | 734.5 | 433.6 | 52 / 2433 / 270 | 1.0005 | 2.75 | 4.90 | – | 380.2 | 8996 |
| python-hotness-80-warm | 1157.4 (1144.3, 1275.8) | 0.864 | 2.335 | 3.18 | – | 734.0 | 452.6 | 52 / 2428 / 270 | 1.0005 | 2.76 | 4.91 | – | 379.4 | 8947 |
| python-stream-cold | 1274.0 (1269.1, 1300.9) | 0.785 | 2.700 | 3.18 | – | 849.9 | 412.0 | 52 / 2808 / 312 | 1.0005 | 4.71 | 3.40 | 0.56 | 444.4 | 8461 |
| python-stream-warm-repeat | 1238.3 (1238.3, 1255.1) | 0.808 | 2.700 | 3.18 | – | 847.9 | 403.8 | 52 / 2808 / 312 | 1.0005 | 2.76 | 3.40 | – | 440.0 | 8804 |
| python-stream-warm | 1228.5 (1228.3, 1249.7) | 0.814 | 2.700 | 3.18 | – | 848.4 | 396.5 | 52 / 2808 / 312 | 1.0005 | 2.76 | 3.40 | – | 440.9 | 8830 |

## I/O microbenchmark `io-cpu` (delivered to cpu)

| Pattern | Path | GB/s delivered | Wall ms (mean, median, p95) | Main-thread CPU ms | Physical MB / request | Read calls / request | Drive GB/s busy | OS = store |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode | python | 2.30 | 45.0 (44.5, 48.1) | 14.5 | 103.9 | 108 | – | True |
| decode | native | 3.17 | 32.8 (31.9, 35.1) | 9.8 | 103.9 | 108 | 3.51 | True |
| decode | native-hit | 5.60 | 18.6 (18.1, 20.5) | 9.4 | 0.0 | 0 | 0.00 | True |
| decode | planning only (ms / request) | Python 1.823, native 0.128 | | | | | | |
| prefill | python | 2.41 | 53.1 (37.6, 73.7) | 14.7 | 128.1 | 133 | – | True |
| prefill | native | 3.27 | 39.2 (36.2, 51.6) | 13.1 | 128.1 | 133 | 3.52 | True |
| prefill | native-hit | 5.61 | 22.8 (19.8, 32.6) | 13.1 | 0.0 | 0 | 0.00 | True |
| layer | python | 2.48 | 446.2 (445.3, 450.7) | 117.2 | 1107.8 | 1152 | – | True |
| layer | native | 3.39 | 326.9 (314.0, 344.5) | 142.6 | 1107.8 | 1152 | 3.52 | True |
| layer | native-hit | 5.44 | 203.5 (194.0, 224.4) | 117.2 | 0.0 | 0 | 0.00 | True |

## I/O microbenchmark `io-cuda` (delivered to cuda:0)

| Pattern | Path | GB/s delivered | Wall ms (mean, median, p95) | Main-thread CPU ms | Physical MB / request | Read calls / request | Drive GB/s busy | OS = store |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode | python | 2.51 | 41.4 (40.6, 45.0) | 10.2 | 103.9 | 108 | – | True |
| decode | native | 2.88 | 36.0 (35.0, 38.8) | 7.7 | 103.9 | 108 | 3.38 | True |
| decode | native-hit | 4.34 | 23.9 (24.0, 24.9) | 17.2 | 0.0 | 0 | 0.00 | True |
| decode | planning only (ms / request) | Python 1.719, native 0.126 | | | | | | |
| prefill | python | 2.56 | 50.0 (39.8, 70.1) | 9.1 | 128.1 | 133 | – | True |
| prefill | native | 3.01 | 42.5 (35.2, 57.0) | 6.1 | 128.1 | 133 | 3.38 | True |
| prefill | native-hit | 4.55 | 28.2 (21.0, 37.4) | 20.2 | 0.0 | 0 | 0.00 | True |
| layer | python | 2.84 | 389.6 (378.1, 390.6) | 41.0 | 1107.8 | 1152 | – | True |
| layer | native | 3.27 | 338.6 (328.8, 331.3) | 56.6 | 1107.8 | 1152 | 3.41 | True |
| layer | native-hit | 5.90 | 187.6 (187.7, 188.6) | 121.1 | 0.0 | 0 | 0.00 | True |

| Sweep pattern | Readers | Read call (MiB) | GB/s delivered | Wall ms (mean) |
| --- | --- | --- | --- | --- |
| decode | 4 | 1 | 2.87 | 36.2 |
| decode | 4 | 4 | 2.81 | 36.9 |
| decode | 8 | 1 | 2.88 | 36.0 |
| decode | 8 | 4 | 2.67 | 38.8 |
| decode | 16 | 1 | 2.35 | 44.2 |
| decode | 16 | 4 | 2.62 | 39.6 |
| layer | 4 | 1 | 3.09 | 358.6 |
| layer | 4 | 4 | 3.30 | 335.7 |
| layer | 8 | 1 | 3.28 | 337.6 |
| layer | 8 | 4 | 3.07 | 360.6 |
| layer | 16 | 1 | 3.23 | 342.9 |
| layer | 16 | 4 | 3.25 | 341.0 |

Compared with experiments/phase6a/native-run2: digests equal True, same source tree True, same native tree True.

Python baseline experiments/phase6a/baseline-run1: digests against Phase 4B run1: {'index_sha256': True, 'reference_rows_sha256': True, 'prompts_sha256': True, 'reference_sha256': True, 'records_sha256': True, 'ranges_sha256': True}.
