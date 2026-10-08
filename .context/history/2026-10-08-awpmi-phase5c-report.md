# AWPMI Phase 5C report — exact shared expert bases and progressive expert deltas

Date: 2026-10-08 · Follows: `history/2026-10-07-awpmi-phase5a2-report.md` (Phase 5A2) ·
Decision: `decisions/0011-phase5c-expert-deltas.md` ·
Status: **complete. 5C1 (exact structural reuse): gate FAIL — Moonlight's routed experts share no exact structure; a
shared base costs more bytes than compressing each expert alone. 5C2 (progressive materialization): STRUCTURAL FAIL —
exact bit-plane sets are tighter than Phase 5A's but still need 82–100% of an expert's exact bytes to decide a token in
real arithmetic; the faithful BF16 stage was not run (gated), and Phase 5A's rounding floor remains.**

## Outcome in brief

Phase 5C asks two questions about Moonlight-16B-A3B's routed experts, kept apart: can a shared resident base plus exact,
losslessly encoded per-expert deltas move fewer bytes (exact structural reuse), and can an exact representation read
progressively certify the reference's token before every byte is read (AWPMI's central question)?

| Question | Answer |
| --- | --- |
| Exact reconstruction (XOR and modular deltas, byte split, bit planes, zstd and LZ4 frames) | bit for bit, every tensor of every configuration, against safetensors' own read; floating-point deltas fail (36–38% of weights) |
| Shared structure between a layer's 64 experts (4 layers, 3 matrices) | none measurable: independent to every test; **0 of 1,536** (expert, matrix, encoding) have a delta cheaper than the expert alone |
| Best shared base vs best independent compression (stored bytes / BF16) | 0.702 vs **0.661** (bit planes, zstd-19): +6.1% |
| Drive bytes per decode token on Phase 5A's routing trace, same host budget | BF16 2.70 GB, independent 1.78 GB (−34%), best shared base 1.87 GB at best (projected to 26 layers); shared bases lose at every budget (−4.5% to −61%). **Gate 5C1: FAIL** |
| Progressive representation | bit planes in 16-row pages, each plane its own frame: exact, page-addressable, 0.664 of BF16 read whole |
| Its uncertainty vs Phase 5A's at the same raw bytes | about 3× closer to the truth (median distance share 0.33) — mostly because it is exact and not redundant |
| Do weights consistent with everything read still flip the decision? | yes, until 0.80–0.99 of the representation (witnesses; the sets' minimum is exact, so this is information, not verifier looseness) |
| Bytes to certify in real arithmetic (12 samples, greedy schedule) | 0.83 (gap > 6 logits) to 1.00 (gap < 1) of the best independent exact bytes; mean **0.920** (gate ≤ 0.90), 0.943 on the real gap distribution. **STRUCTURAL FAIL** |
| Structured metadata (sketches through shared bases, 1% and 10%) | does not pay: a 1% sketch moves a certificate by at most one checkpoint and costs as much on average (0.927 against 0.920); a 10% sketch adds 13–17% |
| Faithful BF16 certification | not run (gated); bounded by Phase 5A's ceilings: 8.1% of tokens certify even with every byte read |
| Correctness | 0 soundness violations; every page decodes to the checkpoint's bytes; Phase 5A's reference recomputed bitwise on 15 samples, also from decoded pages |
| Reproducible | run1 and run2 (`PYTHONHASHSEED` 1 and 2): identical digests for all four stages |

## 1. Questions and plan

> 1. Can MoE experts be represented with shared resident bases and exact, losslessly encoded per-expert deltas, reducing
>    the bytes transferred during inference?
> 2. Can those deltas be materialized progressively, with residual metadata informative enough to certify the same
>    discrete decision as the fully materialized reference before every delta byte has been read?

The user's brief (2026-10-08) staged the work as small gates: 5C1-A (layout, exact transforms, independent
baselines), 5C1-B (shared bases, routing replay, cache cost), 5C2-A (a cheap progressive feasibility probe, allowed even if
5C1 disappoints), 5C2-B (structural certification on a small sample set, only if 5C2-A is promising), 5C2-C (the faithful
BF16 certificate, only if 5C2-B is genuinely useful). Thresholds were fixed in `configs/phase5c-expert-deltas.yaml`
before any measurement (the brief's: 15% fewer steady-state drive bytes than the best independent compression for 5C1;
20% structural coverage, mean bytes ≤ 0.90 of the best independent exact baseline, sets at least twice as close to the
truth as Phase 5A's, zero violations for 5C2). The codec probe changed operational settings only (which codec settings
5C1-B measures), recorded in the config with its reasons.

## 2. Setup

| Item | Value |
| --- | --- |
| Model, profile | Moonlight-16B-A3B @ `476b36a4…`, `BF16_REFERENCE` (the Phase 4B/5A reference), read in place from the local Hugging Face cache; every file used checked against the publisher's sha256 by a full direct read |
| Tensors | routed experts: gate and up [1408, 2048], down [2048, 1408], BF16, 17,301,504 bytes per expert (down, gate, up adjacent in the layer's file) |
| 5C1 sample | layers 1, 9, 17, 26 (26 is Phase 5A's), all 64 experts, all three matrices (4.4 GB, one layer in memory at a time); the codec probe: 8 experts of layer 26 |
| Routing trace | Phase 5A's capture records: 48 prompts × (1 prefill + 16 decode steps), every MoE layer's routed experts (`experiments/phase5a/oracle-run1/capture.jsonl.gz`) |
| 5C2 samples | Moonlight's last MoE layer at decode, upstream exact (Phase 5A's setting): Phase 5A2's three samples (probe), then 12 samples, two per top-2 gap bin, from prompts 0..23 (5C2-B) |
| Codecs | zstandard 0.25.0 (libzstd 1.5.7, C backend), lz4 4.4.5 (liblz4 1.9.4), as published; SciPy 1.18.1 (clustering) |
| Environment | the Weightsift environment (Python 3.13, torch 2.14.1+cu130, transformers 5.18.0); the dependency group `research` adds the codecs and SciPy; nothing in `src/` changed |
| Hardware | i7-8700K (6 cores), 32 GB RAM, RTX 4060 Ti 8 GB, Samsung 990 PRO, Windows 11 |

## 3. Exact representations (stage 5C1-A)

### 3.1 Transforms

All work on the 16-bit patterns, never on values (`expert_deltas/bits.py`):

| Transform | Encode | Decode |
| --- | --- | --- |
| XOR delta | d = w ⊕ b | w = d ⊕ b |
| modular delta | d = (w − b) mod 2¹⁶ | w = (d + b) mod 2¹⁶ |
| byte split | every high byte, then every low byte | interleave |
| bit planes | sign (packed bits), exponent (a byte stream), mantissa bits 6..0 (packed bits) | merge |

Each restores all 65,536 BF16 patterns bit for bit (±0, subnormals, ±inf, NaN payloads; tests). Floating-point deltas do
not: with expert 1 against expert 0 of layer 26, BF16(base + delta) with a BF16 delta differs from the weight on 36–38% of
the weights of every matrix, and even with a float32 delta on 9–19 weights per matrix (`probe_codec.json`,
`float_delta`). They are not used anywhere.

Every measurement compresses each block alone, decodes every frame, restores the patterns, undoes any delta against its
base, and compares the result with the same tensor read by safetensors' own loader, bit for bit; a mismatch stops the
stage. The positioned reads and the safetensors reads of every tensor were equal, and the layer files equal the
publisher's sha256.

### 3.2 What the bits hold

Order-0 entropies, bits per weight (8 experts of layer 26; 5C1-B's 256 expert-matrices agree to ±0.01):

| Field | sign | exponent | mantissa (7 bits) | m6 | m5 | m4..m0 | total, bit planes | total, byte split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate, up, down | 1.000 | 2.55–2.56 | 6.97 | 0.98 | 0.995 | 1.000 | 10.52–10.53 | 10.68–10.69 |

Only the exponent compresses. The order-0 limit of independent coding is about 10.52 bits per weight, 0.658 of BF16.

### 3.3 Independent baselines

Stored bytes over BF16 bytes, 8 experts of layer 26, every reconstruction exact (`experiments/phase5c/probe-codec`):

| Block | Transform | zstd-1 | zstd-3 | zstd-9 | zstd-19 | lz4 | lz4hc-9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| row (4 KiB, 2.75 KiB) | raw | 0.801 | 0.801 | 0.801 | 0.804 | 1.004 | 1.004 |
| row | byte split | 0.800 | 0.787 | 0.782 | 0.801 | 0.946 | 0.895 |
| row | bit planes | 0.720 | 0.735 | 0.731 | **0.695** | 0.919 | 0.874 |
| 16 rows (64 KiB, 44 KiB) | byte split | 0.771 | 0.750 | 0.735 | 0.783 | 0.863 | 0.806 |
| 16 rows | bit planes | 0.686 | 0.699 | 0.697 | **0.665** | 0.842 | 0.788 |
| tensor (5.5 MiB) | byte split | 0.684 | 0.713 | 0.700 | 0.673 | 0.845 | 0.763 |
| tensor | bit planes | 0.680 | 0.702 | 0.690 | **0.662** | 0.825 | 0.748 |
| expert (16.5 MiB) | raw | 0.783 | 0.783 | 0.775 | 0.775 | 1.004 | 0.993 |
| expert | bit planes | 0.680 | 0.703 | 0.691 | **0.662** | 0.825 | 0.748 |

- Generic zstd on the raw bytes saves 22%; LZ4 saves nothing. Splitting the streams is what makes the exponent visible:
  bit planes with zstd-19 reach 0.6615, within 0.5% of the order-0 limit (zstd's Huffman literals), per tensor or per
  16-row page alike. That is the competent independent baseline every shared-base strategy is compared with.
- Small blocks: a single row costs 0.695 with bit planes, 0.80 with a byte split. Trained dictionaries (zstd's own
  trainer, 64 KiB, trained on experts 0–3, evaluated on 4–7) bring byte-split rows from 0.801 to 0.706 and 16-row pages
  from 0.783 to 0.696, about what bit planes reach without one.
- Oddities, measured: zstd-1 beats zstd-3 on split streams; zstd-19 on a 16-row byte split is worse (0.783) than zstd-9
  (0.735).
- A first look at deltas (experts 1–7 against expert 0, tensor blocks): XOR planes cost 0.694 and modular planes 0.703,
  against 0.662 for the same experts alone.

## 4. Exact structural reuse (stage 5C1-B)

Layers 1, 9, 17 and 26, all 64 experts, gate, up and down (`experiments/phase5c/structure-run1`, reproduced by
`structure-run2`). Every reconstruction of every configuration equal to the safetensors copy; the four files equal to the
publisher's sha256.

### 4.1 Is there shared structure?

| Measure (per layer and matrix kind, 12 groups of 64 experts) | Moonlight | If the experts were independent |
| --- | --- | --- |
| Largest eigenvalue's share of the 64 × 64 Gram matrix of the experts' values | 0.017–0.022 | 1/64 = 0.0156 (all equal) |
| Mean absolute correlation between two experts (largest) | 0.0005 (0.0025) for up and down; gate 0.0005–0.004 (0.018) | ~0.0006 (1/√N) |
| Same sign at the same position | 0.5000–0.5012 | 0.5000 |
| Same exponent at the same position | within 0.0003 of the product of the marginals | the product of the marginals |
| Exponent entropy saved by a shared context (column, row, or per-expert row or column), net of describing it | ≤ 0.020 bits per weight (of 10.52) | 0 |
| Best absolute cosine of a neuron (gate row, up row, down column) against every neuron of another expert, mean | 0.047–0.052 (a few outliers per pair, at most 2 of 1,408 neurons above 0.3) | 0.045 (Gaussian null) |

The routed experts are, to every test here, independent draws: no shared component, no shared positional scale, no
permuted copies of each other's neurons. Gate matrices show a faint common component (gate correlations up to 0.018 in
layer 26), far too small to pay for a base. The census sees structure where it exists: on synthetic experts made of a
shared matrix plus small private parts it reports a top eigenvalue share above 0.9, cheap deltas and agreeing signs, and
a shared column scale as context entropy (tests).

### 4.2 What a base costs

The pairwise proxy (order-0 entropy of each bit plane of the delta, which zstd-19 matches within about 0.5%) for all
64 × 64 (expert, base) pairs, both encodings, every kind and layer: **no expert has a delta cheaper than itself alone,
against any of the 63 others** (0 of 1,536 (expert, kind, encoding) combinations); the best base of any expert still adds
0.47 bits per weight. The proxy's extra bits per weight over storing every expert alone, mean over layers and kinds:

| Strategy | XOR | modular |
| --- | --- | --- |
| A1: the first expert as base | +0.551 | +0.665 |
| A2: the medoid | +0.543 | +0.664 |
| A3: each expert's best of the 3 most central (2 bits of mapping) | +0.526 | +0.642 |
| C2 / C4 / C8: average-linkage clusters (SciPy), each with its medoid | +0.535 / +0.517 / +0.482 | +0.653 / +0.632 / +0.590 |
| synthetic: the elementwise median expert | +0.789 | +0.632 |

Measured (zstd-19 on bit planes, every expert restored from its delta and compared bitwise; each base matrix stored
again as its own object; frame index 16 bytes per frame), stored bytes over BF16, mean over the four layers:

| Representation | Stored / BF16 |
| --- | --- |
| **each expert alone, bit planes, zstd-19 (best independent)** | **0.6611** |
| 16-row pages alone, bit planes, zstd-19 (baseline C) | 0.6669 |
| each expert alone, byte split, zstd-19 / zstd-1 | 0.6724 / 0.6851 |
| byte-split pages with a trained 64 KiB dictionary, 16 rows / one row (baseline D) | 0.6924 / 0.7031 |
| best shared base: C-best clusters, XOR | 0.7017 (+6.1%) |
| A2 medoid, XOR / modular; per 16-row page | 0.7052 / 0.7124; 0.7100 |
| A1 first, A3 best of three, synthetic median (XOR) | 0.7055, 0.7146, 0.7213 |
| raw bytes, zstd-3; bit planes with LZ4-HC; byte split with LZ4 | 0.7829; 0.7476; 0.8465 |

Strategy D (a shared low-rank predictor plus its exact correction) was not run: its pre-registered trigger (some strategy
saving 5%, a shared component explaining 10% of the variance, or a shared context worth 0.1 bit per weight) was not met
(−6.2%, 2.2%, 0.02 bits). With a shared component at the level of independence there is nothing for a predictor to predict.

### 4.3 Drive bytes on the routing trace

Phase 5A's routing (48 prompts, 816 steps) through per-layer LRU caches in host RAM, the host budget split evenly over
the 26 layers, the bases (decoded, 5.77 MB per matrix) pinned in the same budget when they fit, else bounded (evicted and
read again), or on the GPU (outside the host budget, reported apart). Steady state: decode steps after the first prompt.
Bytes per decode token on the four sampled layers, and projected to all 26 (× 26/4, a projection):

| Host budget | BF16 | best independent (expert alone) | best shared base, bases in host RAM | change | bases on the GPU | change |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 415 MB (2.70 GB) | 275 MB (1.78 GB) | 525 MB (3.41 GB) | −91% | 287 MB (1.87 GB) | −4.5% |
| 0.5 GB | 415 MB | 275 MB | 287 MB | −4.5% | 287 MB | −4.5% |
| 2 GB | 406 MB | 194 MB | 274 MB | −42% | 203 MB | −4.5% |
| 8 GB | 161 MB (1.05 GB) | 73 MB (0.47 GB) | 84 MB | −15% | 80 MB | −9.6% |
| 16 GB | 71 MB (0.46 GB) | 9.8 MB (0.06 GB) | 15.8 MB | −61% | 13.1 MB | −34% |

- Without room for the bases, every delta reads its bases again: the cold-base case doubles the drive bytes (−91%).
- With the bases pinned, the deltas are 6% larger than the experts alone and the bases take capacity a plain expert
  cache would turn into hits: the shared base loses at every budget, most where caching works (−61% at 16 GB).
- Bases on the GPU (69 MB for the four layers, 0.45 GB for 26, of device memory the comparison does not grant the
  independent representation) still lose (−4.5% to −34%).
- Independent compression is the measured gain: −34% drive bytes per decode token without a cache, and it compounds
  with caching (more experts fit): −55% at 8 GB, −86% at 16 GB, against a BF16 cache of the same size.
- Cold start (the first prompt's prefill, four layers): BF16 2.15 GB, independent 1.42 GB, shared base 1.53 GB.
- Host-to-device bytes are the same for every representation, 2 bytes per routed weight (a decoded expert, or its decoded
  planes merged on the device); only a GPU-side decompressor (not in scope) would move compressed bytes instead.

### 4.4 Decode cost

One decode step's routed experts of a layer (6), decoded and restored on this machine, quiet (structure-run2), GB/s of
BF16 weights out, mean over the four layers:

| Setting (6 experts, one layer) | decompression (zstd's threads / a thread pool) | CPU restoration (NumPy, 6 threads) | GPU restoration (copies + PyTorch) |
| --- | --- | --- | --- |
| independent/expert/planes/zstd-19 | 3.31 | 0.32 | 1.14 |
| A2-xor/expert/planes/zstd-19 | 3.68 | 0.32 | 1.12 |
| independent/rows16/planes/zstd-19 | 4.82 | 0.07 | 1.03 |
| A2-xor/rows16/planes/zstd-19 | 4.91 | 0.07 | 1.03 |
| independent/expert/byte_split/zstd-1 | 2.54 | 4.93 | 2.51 |
| A2-xor/expert/byte_split/zstd-1 | 3.03 | 2.87 | 2.49 |
| independent/expert/planes/lz4hc-9 | 2.53 | 0.32 | 1.16 |
| A2-xor/expert/planes/lz4hc-9 | 2.52 | 0.32 | 1.11 |

zstd's decompression keeps up with the drive (Phase 4B: 3.19 GB/s during a decode step) for bit planes, not for
fast-level byte splits. Restoring bit planes is the bottleneck: NumPy on the CPU (0.07–0.3 GB/s) and this harness's GPU
path (about 1 GB/s, dominated by Python-level copies of the decoded streams before a few PyTorch elementwise kernels) are
both below the drive rate; a byte split restores at 4.9 GB/s on the CPU but decompresses at 2.5 GB/s. A base's XOR adds
nothing measurable. The decode-cost criterion is therefore not met by the best-compressing settings as implemented here;
it is a property of the codec and its restoration, the same for independent and shared-base representations.

Memory of the measurements (not of a runtime): a 5C1-B layer stage peaks at 9.9 GB of host working set (two copies of a
layer's experts, its frames and decoded streams) and 1.6 GB of device memory; the codec probe at 3.4 GB of host memory;
the 5C2 stages at 4.9–5.6 GB of host memory and 3.6–5.5 GB of device memory (the LM head, the layer, the sets in
float64). A runtime holds what the replay charges: its cache and, for shared bases, the decoded bases.

### 4.5 Gate 5C1

| Criterion (config `gate_5c1`) | Result |
| --- | --- |
| Exact: every reconstruction equal to the checkpoint's bytes | PASS |
| ≥ 15% fewer steady-state drive bytes than the best independent compression, at some host budget | **FAIL**: −4.5% at best (0.5–1 GB), −5% to −61% elsewhere |
| Base residency ≤ 4 GB for every layer | 0.45 GB (one base per kind per layer) |
| Decode throughput ≥ 3.19 GB/s for decompression and restoration (the best setting) | FAIL (decompression 3.68 GB/s, restoration 1.12 GB/s) |
| **Verdict** | **FAIL** (run1 and run2) |

Exact shared bases do not pay on Moonlight: the routed experts share no exploitable structure, so a delta against any
base, actual or synthetic, carries more information than the expert itself.

## 5. Progressive materialization: the representation and its sets (stage 5C2)

### 5.1 Which exact representation can be read progressively

| Representation | Exact restore | One unit decodable alone | A partial read bounds the rest |
| --- | --- | --- | --- |
| an expert compressed whole (5C1's best) | yes | no: one frame per expert (or per matrix) | no |
| byte split | yes | per block | only by halves (the high byte: sign and 7 exponent bits) |
| XOR delta, bit planes, per page | yes | yes: each plane of each page is its own frame | yes: prefix intervals |
| modular delta | yes | yes | no: its top bits do not localize the weight's (carries) |

Bit planes of a 16-row page are the representation used: a page is read step by step, its sign and exponent first
(prefix 9: the weight's binade and sign), then one mantissa plane per step (prefixes 10..16). XOR planes against a
resident base give exactly the same prefixes, so a base changes only what each plane costs (5C1: more), never what a
partial read proves. The representation is not redundant: read whole it is the expert itself (0.664 of BF16 with zstd-19
frames per page and plane, against 0.6615 for whole tensors: 0.4% of framing), so a token that never certifies costs no
more than independent compression. Phase 5A's precision levels (D-q6+q4) cost 1.63× the BF16 bytes in that case.

### 5.2 The sets and their exact optimum

What a runtime knows of a weight with prefix t is the interval of the finite completions of its top t bits
(`bits.prefix_interval`, equal to brute force for every pattern and every t). An unread row is bounded by its resident
L∞ norm (4 bytes per row, 0.11% of the expert; charged). The set of weights consistent with every byte read and every
resident byte is the product of these intervals: a box, element by element, built from a read view whose unread bits
are poisoned (it never changes; tested, and checked on every sample).

Over a box the real-arithmetic decision has a closed-form exact minimum (`progressive.exact_minimum`, module docstring):
with x exact, each g_i and u_i ranges over an interval independently of every other neuron, so the activations form a
box (silu's minimum handled); each element of D takes the end of its interval minimizing Δ_k·D_ki·a_i, so
min_D Δ·D·a = M·a − H·|a|, and M_i·a_i − H_i·|a_i| is concave in a_i, minimized at an end of a_i's interval, neuron by
neuron. The minimum equals enumeration on toys (tests, and a toy cut from the real weights in the probe), no random point
of the set goes below it, and its minimizer is a **witness**: weights inside the set whose real forward attains it. A
negative minimum therefore proves that weights consistent with everything read flip the pair: no sound verifier certifies
that state, whatever its method. On box sets there is no verifier looseness to separate from the information.

### 5.3 The feasibility probe (stage 5C2-A)

Phase 5A2's three samples, real arithmetic, two schedules (sequential: every page's sign and exponent, then every page's
next plane, plane after plane; greedy: every page's sign and exponent, then the remaining steps by estimated bound
reduction per byte for the tightest pair at that state, which needs no unread bit: after the exponent every later
half-width is known), checkpoints every 1/64 of the routed BF16 bytes (`experiments/phase5c/progressive-probe`).

| Sample (gap) | Certified at (sequential; greedy), of the BF16 bytes | of the best independent exact bytes | of the representation | Last witnessed flip (BF16 bytes) | Phase 5A realistic certifies at | Phase 5A2: resident set flips until |
| --- | --- | --- | --- | --- | --- | --- |
| prompt 12 step 7 (tie) | 0.665; 0.665 | 1.005; 1.005 | 1.000 (all) | 0.657 | 1.634 | 1.617 |
| prompt 12 step 9 (3.88) | 0.595; 0.595 | **0.899**; 0.899 | 0.894 | 0.579 | 0.961 | 0.821 |
| prompt 23 step 4 (11.06) | 0.610; 0.595 | 0.923; **0.899** | 0.917; 0.894 | 0.595; 0.579 | 1.008 | 0.852 |

At matched byte fractions (Phase 5A's state with at least as many bytes) the box set's exact minimum against the truth,
as a share of Phase 5A's realistic bound's distance to the truth (median over the 64 comparison pairs, then over the
checkpoints from 0.38 to the full read): 0.33–0.38. The share exceeds 1 below about 0.42 of the BF16 bytes (the planes
are looser with only sign, exponent and a bit or two), drops below 1 from about 0.42, to about 0.5 by 0.49, and toward 0
as the planes complete while Phase 5A's levels are still being refined. So the gain over Phase 5A is real at matched raw
bytes, and most of it is the representation being exact and not redundant.

The probe's rule (config `gate_5c2`, fixed before the probe) asked for no violation and one sample with a margin
certified at ≤ 0.90 of the best independent exact bytes with a distance share ≤ 0.5: met, by 0.0007 (0.8993 on the gap
3.88 sample; 0.8989 with the greedy schedule on the gap 11.06 sample). Stage 5C2-B followed.

Correctness of the probe: the capture equals Phase 5A run1's (sha256); Phase 5A's reference recomputed from it equals
the capture bitwise on all three samples (R, m, y, h, the token); every page decoded and merged equals the checkpoint's
bytes, and the reference recomputed with the decoded experts equals the capture bitwise (the full fallback); the real
forward's Δ·y equals Phase 5A2's truth to 10⁻¹⁵; no set minimum above the truth, no witness outside its set or away from
its minimum, and the poisoning check held in every cell.

Structured metadata's precondition, measured on the probe samples with bases fitted on prompts 24–47 only: a k-dimensional
subspace of the experts' input x leaves 74% (k = 102, the 10% budget) to 86% (k = 5, 0.5%) of x's box-relevant mass
Σ_c h_c·|x_c| outside; a subspace of the decision directions Δ leaves 65–92% of their ℓ1 mass outside. Projections onto a
shared basis can therefore tighten the box terms by a quarter to a third at most, while costing up to 10% of the expert's
bytes: 5C2-B measured two budgets.

## 6. Structural certification on 12 samples (stage 5C2-B)

Twelve samples, two per top-2 gap bin (0–0.5, 0.5–1, 1–2, 2–4, 4–8, ≥ 8 logits), evenly spaced in (prompt, step) order
among Phase 5A run1's samples of prompts 0..23 (`experiments/phase5c/progressive-run1`, reproduced by
`progressive-run2`). Per sample, Phase 5A's realistic D-q6+q4 schedule and bounds were recomputed with Phase 5A's own code
(Phase 5A2's `export.schedule` and `real_components`) on the same comparison pairs. Four cells: the two schedules with the
L∞ metadata, and the greedy schedule with sketches of 1% and 10% of the experts' bytes (bases fitted on prompts 24–47; a
sound relaxation, no witnesses).

Bytes at the first certifying checkpoint (every vocabulary row decided in real arithmetic), over the best independent
exact bytes of the same six experts (bit planes, zstd-19, each expert alone); a token never certified would cost the full
read (1.004):

| Sample (gap) | sequential | greedy | greedy + 1% sketch | greedy + 10% sketch | greedy: of the representation; last witnessed flip | Phase 5A realistic decides at (of BF16) |
| --- | --- | --- | --- | --- | --- | --- |
| 17/1 (0.13) | 1.005 | 1.005 | 1.022 | 1.180 | 1.000; 0.989 | 1.618 |
| 5/2 (0.25) | 0.994 | 0.994 | 1.011 | 1.169 | 0.989; 0.965 | 1.493 |
| 16/1 (0.56) | 0.994 | 0.994 | 1.011 | 1.169 | 0.989; 0.965 | 1.555 |
| 7/8 (0.63) | 0.994 | 0.994 | 1.011 | 1.169 | 0.989; 0.965 | 1.524 |
| 17/2 (1.63) | 0.970 | 0.946 | 0.963 | 1.121 | 0.941; 0.918 | 1.368 |
| 6/2 (1.75) | 0.946 | 0.923 | 0.916 | 1.027 | 0.918; 0.894 | 1.055 |
| 5/7 (3.50) | 0.946 | 0.876 | 0.869 | 1.027 | 0.871; 0.847 | 1.024 |
| 18/9 (3.81) | 0.875 | 0.875 | 0.869 | 1.003 | 0.871; 0.847 | 0.868 |
| 18/1 (4.25) | 0.970 | 0.946 | 0.963 | 1.121 | 0.941; 0.918 | 1.196 |
| 5/15 (6.38) | 0.828 | 0.828 | 0.822 | 0.956 | 0.824; 0.800 | 0.758 |
| 8/15 (8.25) | 0.852 | 0.828 | 0.845 | 0.980 | 0.824; 0.800 | 0.758 |
| 21/16 (8.56) | 0.852 | 0.828 | 0.822 | 0.956 | 0.824; 0.800 | 0.743 |
| **mean** | 0.936 | **0.920** | 0.927 | 1.073 | | |
| mean, weighted by the gap distribution of Phase 5A's 768 tokens | | **0.943** | | | | |

In BF16 bytes the greedy cell averages 0.609; Phase 5A's realistic D-q6+q4 decides the same pairs at 0.743–1.618 (above
1.0 on eight of the twelve: its levels are read in addition to the BF16 rows). Every certified token is the reference's
(no exact BF16 tie among these samples).

- **What decides is the margin.** Close calls (gap below 1 logit, 45% of real tokens) certify only once (almost) every
  plane is read; gaps of 1.6–4.3 leave 5–13% of the compressed bytes unread; gaps above 6 leave 17%. The sets hold
  decision-flipping weights (witnesses) until 0.80–0.99 of the representation: one checkpoint (1/64 of the BF16 bytes)
  before each certificate.
- **Schedules.** The greedy order (estimated bound reduction per byte, from the state after every sign and exponent)
  beats the plane-major order on six samples by 0.02–0.07 and ties on the other six.
- **Structured metadata does not pay.** A 1% sketch moves the certificate by at most one checkpoint, and its cost
  cancels the gain on average (0.927 against 0.920); a 10% sketch costs more than it can save on every sample (1.073). The
  probe's precondition predicted this: a calibration subspace leaves three quarters of the box-relevant mass outside.
- **Layout.** Plane-major files (every page's sign frames, then every exponent frame, …) keep the certifying reads
  contiguous: 4 KiB amplification 1.000–1.007, 6–69 extents per token (one file per routed expert). Page-major files read
  the same bytes in 341–414 extents at 1.04.
- **Where the last uncertainty sits** (the comparison pairs, every page one plane short of exact): 8 of 12 samples still
  flip a pair; gate and up alone flip 7, down alone 5; the median gap (truth − minimum) splits evenly (5.1 for gate and
  up, 4.6 for down, 9.8 together). The terms are spread out: the largest 10% of the input columns hold 26% of the gate
  and up box mass, the largest 10% of down rows 43%, of neurons 53%. With two planes short every sample flips (median gap
  32). No block shape (rows, columns, neurons) concentrates the remaining uncertainty into a few blocks to skip.

| Gate 5C2 (config `gate_5c2`, best cell: greedy) | Result |
| --- | --- |
| Soundness violations (no set minimum above the truth, witnesses in their sets and attaining them, poisoning, toys, Phase 5A's reference bitwise, pages exact) | **0** |
| Coverage: certified with bytes unread ≥ 0.20 | 0.917 (11 of 12; one only at the full read) |
| Mean charged bytes ≤ 0.90 of the best independent exact bytes | **0.920: FAIL** (0.943 on the token distribution) |
| Distance share vs Phase 5A ≤ 0.5 (median) | 0.328 |
| Strong: mean ≤ 0.70 of BF16 and below the best independent | not applicable (no structural pass) |
| **Verdict** | **STRUCTURAL FAIL** (run1 and run2) |

## 7. The faithful BF16 certificate (stage 5C2-C): not run

The brief runs 5C2-C only if structural certification is genuinely useful; the gate says it is not. Phase 5A's ceilings
also bound what it could do. Under the certified (faithful) rounding model, with every routed byte read, 4 of these 12
samples certify (margins 0.32–3.19 logits: 18/9, 5/15, 8/15, 21/16) and 2 of the probe's 3 (0.19, 0.59); over Phase 5A's
768 tokens, 8.1%. Phase 5A's propagation is inclusion-monotone, so no partial state certifies a token its ceiling does not:
on the other 92% of tokens a faithful certificate saves nothing, whatever the representation. On the certifiable 8% it
must absorb the same weight uncertainty plus the roundings, so the real-arithmetic savings above are optimistic for it;
even granting every certifiable token the best case measured (17%), the saving on the token distribution is about
8% × 17% ≈ 1.4% of the compressed bytes **[projection]**. The rounding floor remains.

## 8. Where the difficulty is

| Source | Evidence | Weight |
| --- | --- | --- |
| Shared structure (5C1) | experts independent to every test; no delta cheaper than the expert alone | decisive for bases |
| Insufficient information (5C2) | box sets are optimized exactly, with witnesses: weights consistent with everything read flip the decision until 0.80–0.99 of the representation | decisive for early certification |
| Verifier looseness (5C2) | none: the box set's minimum is exact (closed form, equal to enumeration, attained by witnesses) | 0 |
| Structured metadata | 74–86% of x's box mass and 65–92% of Δ's ℓ1 mass outside any calibration subspace of 0.5–10% budget; measured sketches gain ≤ one checkpoint | does not pay |
| Finite-precision floor (Phase 5A) | 92% of tokens undecidable under the faithful model even with every byte read | decisive for BF16 |
| Decode cost (5C1) | zstd decompression keeps up with the drive; bit-plane restoration needs a device kernel faster than this harness | engineering |

## 9. Correctness and reproducibility

| Check | Result |
| --- | --- |
| Files | the four checkpoint files read (layers 1, 9, 17 and 26; the last also holds the LM head and the final norm) equal the publisher's sha256 (direct-read hash) |
| Two read paths | every expert tensor read by positioned reads equals safetensors' own `get_tensor` |
| Transforms | XOR, modular, byte split and bit planes restore all 65,536 BF16 patterns (±0, subnormals, ±inf, NaN payloads); prefix intervals equal brute force for every pattern and every prefix length |
| Every measured configuration | each block compressed alone, decoded, restored (and a delta undone against its base) and compared bitwise with the safetensors copy: 5C1-A 90 baselines, 16 dictionary and 8 delta configurations; 5C1-B 11 independent, 2 dictionary and 7 base configurations × 4 layers, and every base object; 5C2 every routed expert's pages |
| Independent pages | a page decodes from its own byte range with every other byte poisoned; batch frames equal single frames |
| Replay | charges every miss (experts, deltas, bases), checked on hand-computed traces (cold, pinned, thrashing, several bases) |
| Phase 5A's reference | recomputed with Phase 5A's code from the capture (sha256 equal to Phase 5A run1's) on all 15 samples: R, m, y, h and the token bitwise; also with the experts decoded from their bit-plane pages (the full fallback) on the probe's 3 |
| Sets and optima | the box set never depends on an unread bit (poisoned views, every cell); its minimum equals enumeration on toys (including one cut from the real weights); no set minimum above the truth; every witness inside its set and attaining its minimum; more planes never lower the minimum; every plane read gives the truth; the real forward's Δ·y equals Phase 5A2's truth to 10⁻¹⁵ |
| Orderings | the greedy order is unchanged when unread bits change (resident metadata fixed) |
| Certificates | every certified token is the real tier's winner and the reference's (no BF16 tie among the 15 samples) |
| Tests | 79 in `research/expert_deltas/tests` (new); Weightsift's 555 pass unchanged |
| Reproducibility | run1 (`PYTHONHASHSEED` 1) and run2 (2): codec probe (5C1-A) equal; structural reuse (5C1-B: layers and replay) equal; progressive probe (5C2-A) equal; progressive samples (5C2-B) equal |

Not charged in 5C2's figures: the page index (an offset and a length per frame, 11,664 frames per token, 0.09–0.18% of
the routed BF16 bytes): at most +0.003 on every fraction of the best independent bytes. 5C1's figures charge it (16 bytes
per frame).

## 10. Answers

Labels: **[measured]** a measurement on Moonlight's checkpoint or trace; **[structural]** a real-arithmetic diagnostic,
never a certified result; **[certified]** a result under the faithful BF16 model; **[hypothesis]**; **[projection]**.

1. **Do Moonlight's routed experts share enough structure for exact base/delta compression?** No **[measured]**. On four
   layers × 64 experts × three matrices, the experts behave as independent draws: the cross-expert spectrum is flat
   (largest eigenvalue 0.017–0.022 of the energy against 1/64 = 0.016 for independence), correlations are at the noise
   level, signs and exponents agree exactly as often as independence predicts, shared contexts save at most 0.02 bits per
   weight, and neurons are not permuted copies of each other's. No expert has a delta cheaper than itself against any of
   the other 63.
2. **Which bases work best?** None pays **[measured]**. The least bad is a few clusters with their medoids (C-best, XOR:
   +6.1% bytes over the best independent compression), then the medoid and the first expert (+6.7%), the best of three
   (+8.1%), the synthetic median (+9.1%). XOR beats modular deltas by about 0.12 bits per weight. A low-rank predictor
   was not built: its pre-registered trigger was not met, and with no shared component there is nothing to predict.
3. **Against independently compressed experts?** Every shared-base strategy is larger: 0.70–0.72 of BF16 against 0.661
   for bit planes with zstd-19 per expert (0.667 in 16-row pages; 0.692 for byte-split pages with a trained dictionary)
   **[measured]**.
4. **Actual drive bytes on representative routing traces?** On Phase 5A's trace (four layers), steady state per decode
   token **[measured]**: BF16 415 MB, independent compression 275 MB (−34%), the best shared base 287 MB with its bases on
   the GPU or pinned in host RAM, 525 MB when its bases are not resident. Projected to 26 layers **[projection]**: 2.70 GB
   (Phase 4B's figure), 1.78 GB, 1.87 GB. With a host cache, compression compounds (8 GB: −55% against a BF16 cache of
   the same size; 16 GB: −86%), and the shared base loses more (−15%, −61%).
5. **Amortization cost of resident bases?** One base per matrix kind per layer is 17.3 MB decoded (0.45 GB for 26 layers;
   clusters multiply it) **[measured]**. It is paid in cache capacity a plain expert cache turns into hits, and it buys
   nothing: the deltas are larger than the experts. Without residency the bases are read again with every delta (cold:
   −91%).
6. **Which exact representation supports independently addressable progressive refinement?** Bit planes in pages, each
   plane of each page its own frame **[measured]**: exact, decodable page by page and plane by plane, and not redundant
   (0.664 of BF16 in 16-row pages, 0.4% over per-tensor compression). XOR planes against any base give the same
   information (the prefixes), so the base only changes their size (larger). Modular deltas and byte splits do not refine
   progressively in a useful way; a whole-expert frame does not at all.
7. **Does progressive materialization give stronger uncertainty than Phase 5A?** At the same raw bytes, yes
   **[structural]**: the bit-plane box set's exact minimum is about three times closer to the truth than Phase 5A's
   realistic bound (median distance share 0.33 on 12 samples, 0.33–0.38 on the probe's 3), comparable around 0.42–0.45 of
   the BF16 bytes and looser below. Most of the gain is the representation being exact and not redundant: it is complete
   at 0.664 of the BF16 bytes where Phase 5A's levels are complete at 1.63.
8. **Can alternative weights still flip the decision under the new metadata?** Yes **[structural]**: witnesses (weights
   inside the set, by the real forward) flip a pair until 0.80–0.99 of the representation's bytes, one checkpoint before
   every certificate; with every page one plane short of exact, 8 of 12 samples still flip, with two planes short all 12.
   This is insufficient information, not a loose verifier: the minimum over the set is exact.
9. **How much must be materialized before structural certification?** 0.82–1.00 of the representation **[structural]**:
   about 82% for gaps above 6 logits, 87–94% for gaps of 1.6–4.3, all but about 1% for close calls (gap below 1, 45% of
   real tokens). Mean 0.920 of the best independent exact bytes on the 12 samples (gate: ≤ 0.90; FAIL), 0.943 weighted by
   the real gap distribution.
10. **Does faithful BF16 certification also become possible?** Not run (5C2-C is gated on a structural pass). The floor
    remains **[certified, Phase 5A]**: 8.1% of tokens certify even with every byte read; partial states can only do worse.
    At most about 1.4% of the compressed bytes could be saved on the token distribution **[projection]**.
11. **Do the costs justify a future runtime?** Not for shared bases, not for progressive certification of expert deltas
    **[measured, structural]**. Independent exact compression does reduce the drive bytes by a third, at a decode cost
    this harness meets for decompression (3.3–4.9 GB/s) but not for bit-plane restoration (needs a device kernel), or with
    a byte split at 2.5 GB/s of decompression for 0.672–0.685 of the bytes.
12. **Next milestone?** See §11.

## 11. Recommendation

- **Close Phase 5C.** Do not build a shared-base format or runtime: Moonlight's experts share no exact structure. Do not
  build a progressive expert-delta runtime: certification saves 6–8% of the compressed bytes in real arithmetic and about
  1% under the faithful model, at the cost of a per-page, per-plane read pattern.
- **Stop expert AWPMI on BF16 Moonlight** unless the rounding-model decision (decisions 0001, 0005) changes: three
  representations (Phase 5A's spatial and precision decompositions, Phase 5A2's verifier on them, and here exact bit
  planes optimized exactly) agree that the information needed to decide a token is almost all of the expert, and the
  reference's own roundings decide 92% of tokens anyway.
- **Next milestone (the user's decision): the engineering path, with exact compression.** Phase 4B's native runtime and
  host-RAM expert tier, storing the routed experts as exact bit-plane (or byte-split) zstd pages: a third fewer drive
  bytes per decode token without a cache, and about 1.5 times as many experts per byte of host cache (the replay: −55%
  at 8 GB, −86% at 16 GB against BF16 caches of the same size). Its open engineering question is decode cost: a GPU-side
  plane merge (or nvCOMP-style GPU decompression) so restoration keeps up with the drive. This is established practice
  (ZipNN, DFloat11-style entropy coding of BF16 exponents), not Weightsift novelty, and it composes with every exact
  reference profile (bit for bit).

## 12. Limitations

- **One model, four layers.** The census and the bases cover layers 1, 9, 17 and 26 (all 64 experts, every matrix);
  the 26-layer drive bytes are a projection (× 26/4) of the sampled layers' replay. The layers agree closely (stored
  ratios 0.6606–0.6616 for the best independent setting), so the projection is unlikely to move the comparison. A model
  whose experts were upcycled from one dense model (Mixtral-style) could share structure Moonlight's do not; the census
  would show it (it does on synthetic shared structure).
- **Codecs and settings.** zstd and LZ4 at a few levels, bit planes and byte splits; no entropy coder conditioned on a
  base (that would be a custom compressor, outside the brief). The order-0 census bounds what any such coder could gain
  from the measured contexts: at most 0.02 bits per weight.
- **The replay** models per-layer LRU caches of whole stored objects, an even split of the host budget over layers, and
  bases as their own stored objects (one extra copy of each base matrix on disk). Other policies (hotness, uneven splits)
  shift every representation alike; Phase 4B found LRU and hotness similar on this trace.
- **Decode throughput** is this machine's (6 cores) and this harness's: the bit-plane restoration on the GPU passes the
  decoded streams through Python bytes before the PyTorch kernels, so a runtime decoding into pinned buffers should do
  better than 1 GB/s (not measured). The comparison between representations is unaffected (the XOR is free).
- **Progressive sets are boxes.** The bit-plane sets keep exactly what is read (per-weight intervals) plus the rows' L∞
  bounds. Any additional resident information (norms, sketches) can only shrink them; the sketch measured here is a sound
  relaxation (its equality constraints dropped on the complement), so its failure to help is "not shown to help", with
  its cost alone exceeding the room it could save.
- **The real tier.** Every 5C2 certificate is the structural diagnostic of decision 0009's real tier (no rounding): it
  overstates what the faithful BF16 certificate could do. The finite-precision tier with bit-plane sets was not run (the
  gate stopped before 5C2-C); Phase 5A's ceilings bound it.
- **Comparison rows.** The distance-share comparison with Phase 5A uses the reference token against its 64 nearest rows
  (the pairs Phase 5A2 compared); every certificate checks the whole vocabulary.
- **Physical I/O is modelled**, not timed: 4 KiB blocks and extents from the frames' sizes in two layouts.

## 13. Reproduction

```bash
python -m uv sync                                                       # the default groups include `research`
python -m uv run python -m pytest research/expert_deltas/tests          # 79 tests
export PYTHONIOENCODING=utf-8 PYTHONHASHSEED=1                          # run2: PYTHONHASHSEED=2
python -m uv run python research/expert_deltas/probe_codec.py --output experiments/phase5c/probe-codec-run1          # ~10 min
R=experiments/phase5c/structure-run1
python -m uv run python research/expert_deltas/structure_run.py --output $R --stage prepare
for L in 26 1 9 17; do python -m uv run python research/expert_deltas/structure_run.py --output $R --stage layer --layer $L; done   # ~55 min each
python -m uv run python research/expert_deltas/structure_run.py --output $R --stage replay
python -m uv run python research/expert_deltas/structure_run.py --output $R --stage report
# 5C2 needs experiments/phase5a2/capture/capture.safetensors (benchmarks/expert_oracle.py --stage capture; sha256 = Phase 5A run1's)
python -m uv run python research/expert_deltas/progressive_probe.py --output experiments/phase5c/progressive-probe-run1   # ~15 min
for S in 0 1; do python -m uv run python research/expert_deltas/progressive_run.py --output experiments/phase5c/progressive-run1 --shard $S --shards 2; done  # ~50 min each
python -m uv run python research/expert_deltas/progressive_run.py --output experiments/phase5c/progressive-run1 --report
python -m uv run python research/expert_deltas/structure_report.py experiments/phase5c/structure-run2 --compare experiments/phase5c/structure-run1
```

The decode throughputs reported come from structure-run2, run alone on an otherwise idle machine. In structure-run1,
layers 26 and 1 overlapped other stages (the probe; a 5C2-B start that was stopped because the GPU's 8 GB do not hold both:
under WDDM, oversubscription slows both instead of failing), which changes only their timings. Digests exclude timings,
throughputs and process memory.

| Stage | run1 | run2 |
| --- | --- | --- |
| codec probe (5C1-A) | `41ec8052e50ad38bcc02dac4472ef002fa8eb742cdef58bfb46225e13bafc6d4` | `41ec8052e50ad38bcc02dac4472ef002fa8eb742cdef58bfb46225e13bafc6d4` |
| structural reuse (5C1-B: layers and replay) | `9b961f30077516b57ec6463e3697da37191e6e4a3bd249f51125513bfe233b24` | `9b961f30077516b57ec6463e3697da37191e6e4a3bd249f51125513bfe233b24` |
| progressive probe (5C2-A) | `6a6599cfb11a2480058d174d5657cfa657956ef827ecf41f7d531aceb696ee16` | `6a6599cfb11a2480058d174d5657cfa657956ef827ecf41f7d531aceb696ee16` |
| progressive samples (5C2-B) | `2ccf4204fe915a378a36c4585476c01d7f907826ddf83c8430fa20e4e84eb90c` | `2ccf4204fe915a378a36c4585476c01d7f907826ddf83c8430fa20e4e84eb90c` |

The research trees: the codec probes, progressive probe run2, structure run2 and progressive run2 ran on one tree;
structure run1, progressive run1 and progressive probe run1 on earlier trees that differ from it only in fields outside
the digests (process memory), report formatting and removed unused functions. After the runs, `structure_report.py`'s
summary table header was corrected (a cell that broke the Markdown table) and both structure summaries were regenerated
from the same records (digests unchanged): the committed tree differs from the runs' only there. `probe-codec` and
`progressive-probe` (no suffix) are the development-tree runs whose results set the codec settings and the probe's
PROCEED; the run1/run2 pairs reproduce them value for value.
