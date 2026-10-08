# Phase 6A run experiments/phase6a/native-run2

## Gates

| Gate | Result |
| --- | --- |
| correctness | PASS |
| io_parity | PASS |
| host_cache | PASS |
| performance | n/a |
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

Compared with experiments/phase6a/native-run1: digests equal True, same source tree True, same native tree True.
