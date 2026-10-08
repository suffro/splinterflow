# Phase 5C1-A codec probe

Layer 26: 64 experts in model-27-of-27.safetensors, ['BF16'], shapes {'gate': [1408, 2048], 'up': [1408, 2048], 'down': [2048, 1408]}, 17301504 bytes per expert; down, gate, up adjacent: True.
Read paths equal: True; finite: True; file verification: {'model-27-of-27.safetensors': True}.

## Entropies (bits per weight, order 0)

| Matrix | sign | exponent | mantissa | high byte | low byte | byte split | planes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate | 1.000 | 2.561 | 6.972 | 2.719 | 7.970 | 10.689 | 10.534 |
| up | 1.000 | 2.551 | 6.972 | 2.719 | 7.970 | 10.690 | 10.524 |
| down | 1.000 | 2.549 | 6.972 | 2.714 | 7.970 | 10.684 | 10.522 |

## Independent compression (stored / BF16 bytes; all exact)

| Block | Transform | zstd-1 | zstd-3 | zstd-9 | zstd-19 | lz4-0 | lz4hc-9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| row | raw | 0.8009 | 0.8010 | 0.8011 | 0.8041 | 1.0043 | 1.0039 |
| row | byte_split | 0.8004 | 0.7865 | 0.7822 | 0.8012 | 0.9459 | 0.8948 |
| row | planes | 0.7203 | 0.7347 | 0.7307 | 0.6954 | 0.9187 | 0.8744 |
| rows16 | raw | 0.7836 | 0.7836 | 0.7838 | 0.7851 | 1.0039 | 0.9986 |
| rows16 | byte_split | 0.7710 | 0.7496 | 0.7345 | 0.7833 | 0.8628 | 0.8058 |
| rows16 | planes | 0.6860 | 0.6988 | 0.6970 | 0.6648 | 0.8416 | 0.7879 |
| rows64 | raw | 0.7831 | 0.7833 | 0.7823 | 0.7820 | 1.0039 | 0.9941 |
| rows64 | byte_split | 0.7112 | 0.7197 | 0.7089 | 0.6971 | 0.8497 | 0.7758 |
| rows64 | planes | 0.6872 | 0.7007 | 0.6949 | 0.6622 | 0.8295 | 0.7601 |
| tensor | raw | 0.7830 | 0.7831 | 0.7771 | 0.7754 | 1.0039 | 0.9925 |
| tensor | byte_split | 0.6844 | 0.7130 | 0.6999 | 0.6725 | 0.8453 | 0.7631 |
| tensor | planes | 0.6796 | 0.7023 | 0.6902 | 0.6615 | 0.8253 | 0.7484 |
| expert | raw | 0.7830 | 0.7831 | 0.7745 | 0.7748 | 1.0039 | 0.9925 |
| expert | byte_split | 0.6844 | 0.7135 | 0.7004 | 0.6725 | 0.8452 | 0.7627 |
| expert | planes | 0.6797 | 0.7031 | 0.6907 | 0.6615 | 0.8252 | 0.7481 |

Decompression throughput (6 threads, GB/s of BF16 out), tensor blocks:

- raw zstd-1: 1.70 (1 thread 0.58); compression 2060 MB/s
- raw zstd-3: 1.43 (1 thread 0.54); compression 1263 MB/s
- raw zstd-9: 1.00 (1 thread 0.41); compression 119 MB/s
- raw zstd-19: 1.22 (1 thread 0.38); compression 19 MB/s
- raw lz4-0: 3.25 (1 thread 1.82); compression 3129 MB/s
- raw lz4hc-9: 2.95 (1 thread 1.55); compression 213 MB/s
- byte_split zstd-1: 1.89 (1 thread 0.86); compression 1310 MB/s
- byte_split zstd-3: 1.39 (1 thread 0.56); compression 516 MB/s
- byte_split zstd-9: 1.07 (1 thread 0.56); compression 151 MB/s
- byte_split zstd-19: 2.46 (1 thread 1.17); compression 13 MB/s
- byte_split lz4-0: 3.26 (1 thread 1.58); compression 2492 MB/s
- byte_split lz4hc-9: 3.18 (1 thread 1.48); compression 61 MB/s
- planes zstd-1: 1.78 (1 thread 0.73); compression 1155 MB/s
- planes zstd-3: 1.47 (1 thread 0.55); compression 631 MB/s
- planes zstd-9: 1.60 (1 thread 0.55); compression 214 MB/s
- planes zstd-19: 1.80 (1 thread 0.91); compression 14 MB/s
- planes lz4-0: 2.79 (1 thread 1.57); compression 2276 MB/s
- planes lz4hc-9: 2.98 (1 thread 1.56); compression 45 MB/s

## Dictionaries (evaluation experts; dictionary bytes not included in the ratio)

- row raw zstd-3 dict 16384: 0.7894 (without 0.8009)
- row raw zstd-19 dict 16384: 0.7901 (without 0.8042)
- row raw zstd-3 dict 65536: 0.7896 (without 0.8009)
- row raw zstd-19 dict 65536: 0.7902 (without 0.8042)
- row byte_split zstd-3 dict 16384: 0.7608 (without 0.7865)
- row byte_split zstd-19 dict 16384: 0.7292 (without 0.8012)
- row byte_split zstd-3 dict 65536: 0.7539 (without 0.7865)
- row byte_split zstd-19 dict 65536: 0.7056 (without 0.8012)
- rows16 raw zstd-3 dict 16384: 0.7836 (without 0.7836)
- rows16 raw zstd-19 dict 16384: 0.7845 (without 0.7852)
- rows16 raw zstd-3 dict 65536: 0.7833 (without 0.7836)
- rows16 raw zstd-19 dict 65536: 0.7831 (without 0.7852)
- rows16 byte_split zstd-3 dict 16384: 0.7413 (without 0.7496)
- rows16 byte_split zstd-19 dict 16384: 0.6996 (without 0.7833)
- rows16 byte_split zstd-3 dict 65536: 0.7388 (without 0.7496)
- rows16 byte_split zstd-19 dict 65536: 0.6957 (without 0.7833)

## Deltas against the first expert (tensor blocks)

- xor byte_split zstd-3: 0.7319 vs independent 0.7130; restored exactly: True
- xor byte_split zstd-19: 0.6969 vs independent 0.6725; restored exactly: True
- xor planes zstd-3: 0.7312 vs independent 0.7023; restored exactly: True
- xor planes zstd-19: 0.6944 vs independent 0.6615; restored exactly: True
- modular byte_split zstd-3: 0.7425 vs independent 0.7130; restored exactly: True
- modular byte_split zstd-19: 0.7076 vs independent 0.6725; restored exactly: True
- modular planes zstd-3: 0.7403 vs independent 0.7023; restored exactly: True
- modular planes zstd-19: 0.7028 vs independent 0.6615; restored exactly: True

## Floating-point deltas

- gate: BF16(base + BF16 delta) differs from the weight on 1096459 of 2883584 (38.02%); with a float32 delta on 19
- up: BF16(base + BF16 delta) differs from the weight on 1080948 of 2883584 (37.49%); with a float32 delta on 9
- down: BF16(base + BF16 delta) differs from the weight on 1026652 of 2883584 (35.60%); with a float32 delta on 14

Digest: `c663d86d0b5ecd5e21ea750c58ef35c16b1f5e9d43ce9d63e0df7666ca2f5da7`
