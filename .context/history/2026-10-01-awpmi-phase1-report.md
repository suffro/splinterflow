# AWPMI Phase 1 report — Minimal certified materialization

Date: 2026-10-01 · Roadmap: the AWPMI implementation roadmap (Phase 1; since replaced by `state/Weightsift_Implementation_Guide.md`) ·
Status: **complete. The correctness gate passes; the scientific hypothesis is not
supported by this bound and decomposition. Phase 2 and Phase 3 are blocked.**

## Outcome in brief

AWPMI's Phase 1 is a progressively materialized LM head with a sound certificate and an
exact fallback. It is **correct**: 0 certified mismatches and 0 fallback mismatches in
12,000 runs. The fallback reproduces the reference bitwise, and the results are
bit-for-bit reproducible.

It does **not save materialization**. With Cauchy–Schwarz (CS) bounds on input-column
pages, the certificate fires almost only at full materialization. The best
configuration reads 98.5% of the LM head on average. A scheduler-independent oracle
shows why. The schedulers are already near the best any ordering can achieve. Even a
perfect per-page magnitude bound would save at most about 17%, because every column
page carries far more logit mass than the top-1/top-2 gap. The bottleneck is the
**page decomposition**, not the ordering, and only secondarily the bound.

Per the roadmap ("If certification occurs almost exclusively after ~100%
materialization: DO NOT implement streaming yet; improve bounds, page decomposition,
page ordering"), the project stays in Phase 1. The next step is a new decomposition
(§8).

## 1. Question

> Can AWPMI certify the full-model next-token decision before all weight pages of the
> progressively refined region have been materialized?

The refined region is the LM head, z = W h, with W ∈ BF16^{49152×576}. W is split into
input-column pages W = [W₁ … W_P], so z = Σ_p W_p h_p exactly. The transformer body runs
exactly (roadmap Mode A), and all weights stay resident, so materialization is logical.

## 2. Setup

| Item | Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f77366fa6b3b4b768ec3050bf629380bac` (LlamaForCausalLM, tied embeddings, bias-free LM head) |
| Reference | unmodified HF forward, `logits_to_keep=1`, eval mode, greedy argmax; BF16 logits (no upcast) |
| dtype / device | BF16 / NVIDIA GeForce RTX 4060 Ti (8 GB), CUDA 13.0 |
| Software | Python 3.13.3, torch 2.14.1+cu130, transformers 5.18.0, safetensors 0.8.0, numpy 2.5.3 (locked in `uv.lock`) |
| Numerics | deterministic algorithms, no TF32, **no BF16/FP16 reduced-precision reductions**, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Prompts | 1000 random-length prefixes (8–128 tokens) of distinct non-heading lines of wikitext-2-raw test (`Salesforce/wikitext` @ `b08601e0…`), seed 0, no special tokens |
| Page widths | 8, 16, 32, 64, giving 72, 36, 18, 9 pages of 49152 × width |
| Schedulers | `sequential` (lowest id), `largest_residual` (‖h_p‖·max_j N[j,p]), `bound_per_byte` (‖h_p‖·(N[w,p] + max over contenders N[j,p]) / bytes) |
| Runs | 1000 prompts × 4 widths × 3 schedulers = 12,000 AWPMI runs, executed twice |

## 3. Method

- **Metadata:** N[j,p] ≥ ‖W[j,p]‖₂, computed in float64, inflated by γ, and rounded up to
  float32.
- **Bound:** a missing page contributes |W[j,p]·h_p| ≤ N[j,p]·‖h_p‖₂ (Cauchy–Schwarz).
- **Residual state:** the partial logits ẑ (exact float64 page contributions) plus an
  interval on every *reference* logit.
- **Certificate:** w = argmax ẑ. The result is CERTIFIED iff lower[w] > max_{j≠w}
  upper[j]; otherwise it is UNKNOWN. There is no confidence-based outcome.
- **Fallback:** once all pages are materialized without a certificate, AWPMI computes
  `F.linear(hidden, cat(pages))`, the reference operation itself.

Making the certificate sound against the *floating-point* reference required two
mechanisms (decision 0001), and both were observed to be necessary:

1. **Accumulation-error term.** The reference accumulator satisfies
   |acc − exact| ≤ γ_{K+2}(2⁻²²)·S_j, where S_j = Σ_p N[j,p]‖h_p‖ and 2⁻²² is 4× the
   binary32 unit roundoff. With this term set to 0, real reference logits fell outside
   the envelope (1 of 8192 on a test GEMM).
2. **Directed rounding onto the BF16 output grid.** Two different exact logits can round
   to the same BF16 value, and argmax then returns the lower index. A constructed case
   has exact argmax 1 and reference argmax 0. Without grid rounding the certificate
   **certified the wrong token**; with it, the case falls back correctly.

Two backend facts mattered. PyTorch defaults
`allow_bf16_reduced_precision_reduction=True`, which permits BF16 split-K reductions
that cannot be bounded usefully, so it is disabled for the reference and AWPMI alike.
The fallback reproduces the reference **bitwise** (decision 0002).

For every prompt and page width, the benchmark checks the following, and any failure
stops the run:

- the exact prefix is bitwise-equal to the reference's LM-head input;
- the fallback logits are bitwise-equal to the reference logits;
- every reference logit lies inside the full-materialization envelope.

## 4. Results

Source: `experiments/phase1/cs-column-baseline-run1/summary.md` (machine-readable:
`experiments/phase1/cs-column-baseline-run1/summary.json`).

| width / scheduler | coverage | early certified | certified at full | fallback | mean materialized fraction | earliest certification |
| --- | --- | --- | --- | --- | --- | --- |
| 8 / bound_per_byte | 86.6 % | **49.6 %** | 370 | 13.4 % | **0.985** | 64 / 72 |
| 8 / largest_residual | 86.6 % | 47.0 % | 396 | 13.4 % | 0.986 | 64 / 72 |
| 8 / sequential | 86.6 % | 21.7 % | 649 | 13.4 % | 0.995 | 65 / 72 |
| 16 / bound_per_byte | 86.5 % | 17.0 % | 695 | 13.5 % | 0.994 | 32 / 36 |
| 16 / largest_residual | 86.5 % | 15.6 % | 709 | 13.5 % | 0.995 | 33 / 36 |
| 16 / sequential | 86.5 % | 9.0 % | 775 | 13.5 % | 0.997 | 33 / 36 |
| 32 / bound_per_byte | 86.4 % | 2.3 % | 841 | 13.6 % | 0.999 | 17 / 18 |
| 32 / largest_residual | 86.4 % | 2.1 % | 843 | 13.6 % | 0.999 | 17 / 18 |
| 32 / sequential | 86.4 % | 1.6 % | 848 | 13.6 % | 0.999 | 17 / 18 |
| 64 / any scheduler | 86.3 % | 0 % | 863 | 13.7 % | 1.000 | — |

Observations:

- **Early certification happens late.** Among early-certified inputs, the mean
  materialized fraction is 0.94–0.98. Median, p90 and p95 of the materialized fraction
  are 1.0 in every configuration.
- **Difficulty is governed by the reference top-2 gap.** The median gap is 0.875, p90
  is 3.5, and there are 38 exact BF16 ties in 1000 prompts. For width 8 with
  `bound_per_byte`:

  | top-2 gap | [0, 0.5) | [0.5, 1) | [1, 2) | [2, 4) | [4, 8) | ≥ 8 |
  | --- | --- | --- | --- | --- | --- | --- |
  | inputs | 317 | 221 | 245 | 133 | 76 | 8 |
  | early certified | 0.3 % | 29.9 % | 86.9 % | 99.2 % | 100 % | 100 % |
  | mean materialized fraction | 1.000 | 0.996 | 0.983 | 0.963 | 0.939 | 0.911 |

  Even the most confident inputs still materialize about 91% of the head.
- **Fallbacks are expected and correct.** The 134–137 fallback prompts per width have
  top-2 gaps ≤ 0.5 (median 0.125, one BF16 ulp for logits in [16, 32)). Even exact
  float64 logits cannot separate them once accumulation error and output rounding are
  accounted for.
- **Metadata overhead cancels the savings.** The resident row-page norms take
  1.8 / 3.5 / 7.1 / 14.2 MB for widths 64 / 32 / 16 / 8, against a 56.6 MB LM head. At
  width 8 that is 25% of the weight bytes, versus 1.5% saved.
- **Full-materialization validation** (4 × 1000 checks): prefix mismatches 0,
  non-bitwise fallbacks 0, forced-fallback token mismatches 0, envelope violations 0.
  The largest |float64 page-accumulated − reference| was 0.1249, which is within one
  BF16 ulp, against a maximum envelope half-width of 0.25.
- Timings were recorded (`elapsed_ms`) but carry no meaning in Phase 1, because
  materialization is logical. They are excluded from the reproducibility digest.

## 5. Diagnosis: ordering or bound or decomposition?

Source: `experiments/phase1/cs-column-baseline-oracle` (`benchmarks/oracle.py`, run on
run1's prompts).

Intervals are nested as pages are added: a page's true contribution lies inside the
radius it removes. Certification is therefore monotone in the materialized set, and an
input can certify before the last page under *any* ordering iff it certifies with
exactly one page missing. Checking every singleton with the real certificate gives the
exact early-certification ceiling for any scheduler. The *ideal* ceiling replaces each
missing page's CS bound with its true magnitude |W[j,p]·h_p|, the tightest
sign-agnostic per-page bound. It is unattainable, and serves only as a diagnostic.

| width | early ceiling, any scheduler (CS) | achieved (bound_per_byte) | greedy-oracle mean fraction (CS) | early ceiling, ideal bound | greedy mean fraction, ideal bound |
| --- | --- | --- | --- | --- | --- |
| 8 | 56.2 % | 49.6 % | 0.982 | 100 % | 0.830 |
| 16 | 20.6 % | 17.0 % | 0.993 | 98.8 % | 0.842 |
| 32 | 2.4 % | 2.3 % | 0.999 | 88.1 % | 0.875 |
| 64 | 0 % | 0 % | 1.000 | 52.1 % | 0.920 |

Medians over the 1000 inputs (width 32):

| Quantity | Median |
| --- | --- |
| Winner's CS radius / top-2 gap | 121× |
| Winner's absolute page mass Σ_p \|W[w,p]·h_p\| / top-2 gap | 30× |
| CS looseness (CS radius / absolute page mass) | 4.0× (2.6× at width 8, 4.8× at width 64) |

Conclusions:

1. **Ordering is not the bottleneck.** `bound_per_byte` is within 0–6.6 points of the
   best possible ordering.
2. **The CS bound is loose by 2.6–4.8×**, but fixing it completely would not be enough.
3. **The column decomposition is the binding constraint.** Each column page moves the
   winner's logit by 26–40× the top-2 gap in absolute terms. Even perfect per-page
   bounds would leave 83–92% of the head materialized.

## 6. Phase 1 gate

| Criterion | Result | Evidence |
| --- | --- | --- |
| Certified mismatches = 0 | PASS | 0 of 10,374 certified runs |
| Fallback mismatches = 0 | PASS | 0 of 1,626 adaptive fallbacks; 0 of 4,000 forced full materializations |
| Full-materialization path reproduces the reference | PASS | bitwise-equal logits 4,000 of 4,000; 0 envelope violations; exact prefix bitwise-equal |
| Certificate triggers before the final page on some real inputs | PASS (weak) | widths 8/16/32 only; earliest at 64 of 72 pages (89%) |
| Results reproducible | PASS | run1 and run2 have the same source-tree hash `06041dea…`; records digest `60d1d8bd…` and validation digest `2ca50515…` are identical |

**Roadmap condition not met:** certification occurs almost exclusively near 100%
materialization (best mean fraction 0.985).

**Verdict:** Phase 1's correctness machinery is validated and can be reused. The
efficiency hypothesis fails for CS bounds on column pages of the LM head. **Do not
start Phase 2 (adaptive transformer suffix) or Phase 3 (streaming).** Iterate on the
Phase 1 decomposition first.

## 7. What is reusable

These components do not depend on the column decomposition and carry over to any new
one:

- the declared numerics;
- the accumulation model and output-grid directed rounding;
- the strict certificate;
- the bitwise fallback;
- the benchmark, report and oracle tooling, with their hard-failure checks and
  reproducibility digests.

## 8. Recommended next step (requires a decision: it changes roadmap §1.4)

Replace input-column pages with a decomposition whose *missing mass* is small relative
to the top-2 gap:

1. **Precision-refinement pages.** Use a coarse page, for example a per-row-scaled int8
   or int4 copy of W, plus residual pages that refine it, so that W = coarse + Σ
   residuals exactly. Bound the missing residual with per-row residual-norm metadata.
   The missing mass then shrinks geometrically with each refinement, instead of by
   about 1/P per page as with column pages.
2. **Vocabulary-row pages on top of (1).** After the coarse pass, only rows whose upper
   bound still reaches the winner's lower bound need refinement. Most of the 49,152 rows
   can be ruled out without ever reading their exact values.
3. **Measure the ceiling before building.** Extend `benchmarks/oracle.py` to the new
   decomposition, and count the bytes of every resident page and every piece of
   metadata.
4. Optional, small: a tie-aware certificate (`lower[w] ≥ upper[j]` for j > w) could
   recover part of the 13.5% near-tie fallbacks.

## 9. Limitations

- One model (135M), one GPU, BF16 only. The FP32 path is supported (`model.dtype:
  float32`) but was not benchmarked. Prompts are raw wikitext prefixes, not
  chat-formatted.
- The accumulation model (u_acc = 2⁻²²) is a conservative assumption about cuBLAS. It
  is validated empirically: 0 violations over 4,000 × 49,152 logits, plus synthetic
  cancellation-heavy GEMMs. It is not derived from vendor documentation.
- The oracle ceilings rely on interval nesting, which holds in exact arithmetic. The
  float64 slack involved is about 10⁻¹⁰ relative.
- `benchmarks/oracle.py` and the roadmap-check line in `benchmarks/report.py` were
  added after the two runs. They read raw data and do not change it, so the oracle
  run's source-tree hash differs from the runs' hash.

## 10. Reproduction

```bash
uv sync
uv run pytest                      # 61 tests
uv run python benchmarks/run.py --output experiments/phase1/<name>          # about 25 min on an RTX 4060 Ti
uv run python benchmarks/report.py experiments/phase1/<name> --compare experiments/phase1/cs-column-baseline-run1
uv run python benchmarks/oracle.py experiments/phase1/<name> --output experiments/phase1/<name>-oracle
```

The run is reproduced if `digest.json` matches `records_sha256 = 60d1d8bd…` and
`validation_sha256 = 2ca50515…`. This is expected on the same hardware and software
stack; a different GPU or library version may legitimately change the low-order bits.
