# Phase 6A: repeated expert bytes and LRU capacity (recorded routing)

## phase4b: 16 prompts, 144 steps (128 decode), 82,118 row requests

- Decode: 2.699 GB requested per token; 0.995 of decode bytes were requested before (prefill: 0.926).
- Where a decode step's bytes were last requested: previous step same layer 0.482, earlier in prompt 0.499, earlier prompt 0.015, never 0.005.
- Previous-step predictor (decode): precision 0.249, recall 0.482.
- Share of an LRU cache's decode misses (steady) a previous-step prefetch would cover: 0.0 GB 0.488, 2.77 GB 0.191, 11.07 GB 0.208, 16.61 GB 0.197.
- Stack distances equal a PageCache replay at 2.77, 11.07, 16.61 GB on every step: True.

| LRU capacity (GB) | of all experts | decode hits (cold) | decode hits (steady) | decode byte hits (steady) | prefill hits (steady) | drive GB per decode token |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 | 2.699 |
| 1 | 0.035 | 0.000 | 0.000 | 0.000 | 0.000 | 2.699 |
| 2 | 0.069 | 0.000 | 0.000 | 0.000 | 0.000 | 2.699 |
| 2.77 | 0.096 | 0.361 | 0.367 | 0.367 | 0.000 | 1.724 |
| 4 | 0.139 | 0.382 | 0.388 | 0.388 | 0.006 | 1.669 |
| 5.54 | 0.192 | 0.480 | 0.474 | 0.473 | 0.019 | 1.406 |
| 6 | 0.208 | 0.503 | 0.497 | 0.496 | 0.023 | 1.343 |
| 8 | 0.278 | 0.578 | 0.571 | 0.570 | 0.052 | 1.143 |
| 10 | 0.347 | 0.642 | 0.634 | 0.633 | 0.090 | 0.969 |
| 11.07 | 0.385 | 0.674 | 0.665 | 0.663 | 0.114 | 0.883 |
| 12 | 0.417 | 0.699 | 0.689 | 0.688 | 0.135 | 0.816 |
| 14 | 0.486 | 0.754 | 0.742 | 0.741 | 0.179 | 0.666 |
| 16 | 0.556 | 0.805 | 0.795 | 0.793 | 0.224 | 0.531 |
| 16.61 | 0.577 | 0.820 | 0.811 | 0.809 | 0.237 | 0.490 |
| 20 | 0.695 | 0.898 | 0.895 | 0.894 | 0.345 | 0.277 |
| 21.6 | 0.750 | 0.926 | 0.925 | 0.924 | 0.412 | 0.200 |
| 24 | 0.834 | 0.960 | 0.961 | 0.960 | 0.564 | 0.109 |
| 28.79 | 1.000 | 0.995 | 0.998 | 0.998 | 0.963 | 0.013 |

## phase5a: 48 prompts, 816 steps (768 decode), 366,306 row requests

- Decode: 2.699 GB requested per token; 0.999 of decode bytes were requested before (prefill: 0.976).
- Where a decode step's bytes were last requested: previous step same layer 0.463, earlier in prompt 0.522, earlier prompt 0.014, never 0.001.
- Previous-step predictor (decode): precision 0.316, recall 0.463.
- Share of an LRU cache's decode misses (steady) a previous-step prefetch would cover: 0.0 GB 0.465, 2.77 GB 0.102, 11.07 GB 0.127, 16.61 GB 0.124.
- Stack distances equal a PageCache replay at 2.77, 11.07, 16.61 GB on every step: True.

| LRU capacity (GB) | of all experts | decode hits (cold) | decode hits (steady) | decode byte hits (steady) | prefill hits (steady) | drive GB per decode token |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 | 2.699 |
| 1 | 0.035 | 0.000 | 0.000 | 0.000 | 0.000 | 2.699 |
| 2 | 0.069 | 0.000 | 0.000 | 0.000 | 0.000 | 2.699 |
| 2.77 | 0.096 | 0.403 | 0.405 | 0.404 | 0.000 | 1.612 |
| 4 | 0.139 | 0.424 | 0.426 | 0.426 | 0.006 | 1.555 |
| 5.54 | 0.192 | 0.520 | 0.517 | 0.517 | 0.019 | 1.296 |
| 6 | 0.208 | 0.548 | 0.546 | 0.546 | 0.024 | 1.220 |
| 8 | 0.278 | 0.639 | 0.637 | 0.636 | 0.053 | 0.976 |
| 10 | 0.347 | 0.701 | 0.698 | 0.697 | 0.091 | 0.808 |
| 11.07 | 0.385 | 0.733 | 0.730 | 0.730 | 0.115 | 0.721 |
| 12 | 0.417 | 0.760 | 0.757 | 0.756 | 0.139 | 0.651 |
| 14 | 0.486 | 0.807 | 0.804 | 0.803 | 0.198 | 0.522 |
| 16 | 0.556 | 0.848 | 0.846 | 0.845 | 0.264 | 0.412 |
| 16.61 | 0.577 | 0.860 | 0.858 | 0.857 | 0.285 | 0.379 |
| 20 | 0.695 | 0.920 | 0.919 | 0.918 | 0.407 | 0.218 |
| 21.6 | 0.750 | 0.942 | 0.941 | 0.941 | 0.486 | 0.159 |
| 24 | 0.834 | 0.969 | 0.969 | 0.968 | 0.623 | 0.087 |
| 28.79 | 1.000 | 0.999 | 1.000 | 1.000 | 0.989 | 0.003 |
