# 0011 — Phase 5C: exact shared expert bases and progressive expert deltas; neither pays

Status: accepted (Phase 5C, 2026-10-08)

## Context

Phases 5A and 5A2 (decisions 0009, 0010) found AWPMI inside Moonlight's routed experts not economical: the faithful BF16
reference's rounding floor leaves 92% of tokens undecidable even with every byte read, and the resident metadata (row
norms, precision levels' remainder norms) defines uncertainty sets that hold decision-flipping weights until 0.82–0.91 of
the routed bytes even in real arithmetic. No verifier can fix insufficient information. The guide's §7 (exact structural
reuse) and §8 (its composition with certification) were the open direction.

The user's Phase 5C brief (2026-10-08):

- Two questions, kept apart: (1) can shared resident bases plus exact, losslessly encoded per-expert deltas reduce the
  bytes moved during inference? (2) can those deltas be materialized progressively, with residual metadata informative
  enough to certify the reference's decision before every delta byte is read? The second is AWPMI's central question;
  neither may be assumed.
- Exactness is bitwise: `BF16(base + delta)` is not a valid reconstruction; every encoding verified against independently
  read checkpoint bytes, edge-case bit patterns included.
- Reuse open source (PyTorch, NumPy, safetensors, zstd and LZ4 bindings, Weightsift's checkpoint, oracle, certificate and
  cache code); no custom compression, SVD, GEMM, CUDA, loader, verifier or runtime.
- Moonlight-16B-A3B at its pinned revision, `BF16_REFERENCE`; the final MoE layer plus a few others, all 64 experts,
  gate, up and down; streamed tensor by tensor, never the whole checkpoint in memory.
- Baselines first: BF16; each expert compressed alone (zstd, LZ4); independently compressed pages; trained zstd
  dictionaries on disjoint data, their bytes counted. Shared bases: an actual expert (first, medoid, a few candidates), per
  layer and tensor kind, clusters, and a low-rank predictor plus exact correction only if the simpler ones show structure.
  Encodings: XOR, modular, byte-oriented, bit planes, each losslessly compressed, in independently addressable blocks.
- Cache and I/O accounting on Phase 4B's routing traces under identical memory budgets: cold, warm host, warm GPU,
  bounded with evictions, clusters competing; every byte (bases, dictionaries, indexes) charged; decompression and
  reconstruction times; peak memory.
- Gate 5C1 (fixed before results): ≥ ~15% fewer steady-state SSD bytes than the best independent compression, exact, a
  plausible base footprint, reasonable decode cost. A negative 5C1 does not close AWPMI: one cheap progressive feasibility
  probe is allowed.
- Progressive part: not norm metadata again on a renamed delta; structured residual information, bit-plane refinement,
  block refinement, correlated uncertainty; compression compatible with partial reads; metadata budgets of 0.5–10%;
  anti-cheating (unread bytes poisoned, the reference token never privileged); toys with exact optima first, then a few
  real samples across margins; three measurements kept apart (lossless reconstruction, structural certification in real
  arithmetic, certified reference execution); the existing certifier first; whether the uncertainty set itself admits
  decision flips (witnesses; insufficient metadata, loose verifier and finite-precision floor told apart); simple
  deterministic schedules; every byte charged, against both BF16 and the best independent compression.
- Gate 5C2 (fixed before the final samples): ≥ 20% of samples structurally certifiable with bytes unread, meaningful
  savings in charged bytes, materially tighter uncertainty than Phase 5A, zero soundness violations; strong: ≥ 30% fewer
  bytes than BF16 and an advantage over the best independent compression. "STRUCTURAL PASS — BF16 CERTIFICATION BLOCKED" if
  only the rounding floor stops it.
- Staged: 5C1-A, 5C1-B, 5C2-A, 5C2-B only on a promising probe, 5C2-C only on useful structural certification; stop
  conditions (§28); two reproducible final executions for any accepted result; Q1–Q12 answered with measured, structural,
  certified, hypothesis and projection kept apart; no physical runtime.

## Decision

1. **Research code in the Weightsift environment, no new environment.** `research/expert_deltas` imports `awpmi`
   (safetensors layout, publisher digests, `PageCache`, Phase 5A's oracle) and adds a root dependency group `research`
   (default): zstandard 0.25 (libzstd 1.5.7), lz4 4.4.5 (liblz4 1.9.4), SciPy 1.18. Nothing in `src/`, `benchmarks/` or
   `tests/` changed; Weightsift's 555 tests pass unchanged.
2. **Exactness is bitwise on the patterns.** XOR and modular deltas, byte splits and bit planes are integer operations on
   the BF16 patterns, each exhaustively tested on the 65,536 patterns. Every configuration measured is decoded, restored and
   compared bit for bit with the same tensor read by safetensors' own loader (a second read path), from files checked
   against the publisher's sha256. Floating-point deltas are rejected: wrong on 36–38% of weights with a BF16 delta.
3. **Baselines first, competent ones.** Raw zstd saves 22%, LZ4 nothing. Bit planes (sign, exponent byte, mantissa bits,
   each its own frame) with zstd-19 reach 0.661 of BF16, within 0.5% of the order-0 entropy (10.52 bits per weight);
   16-row pages 0.667; trained dictionaries (64 KiB, disjoint training experts) help small byte-split blocks only. The best
   independent exact representation is the reference for every comparison.
4. **Shared bases: measured, rejected.** Census of 4 layers × 64 experts × 3 matrices: a flat cross-expert spectrum
   (0.017–0.022 against 1/64), correlations, sign and exponent agreement at independence, shared exponent contexts worth
   ≤ 0.02 bits per weight, no permuted neuron copies. No expert has a cheaper delta than itself against any other (0 of
   1,536). Every strategy (first expert, medoid, best of three, 2/4/8 clusters by SciPy's average linkage, a synthetic
   median; XOR and modular) costs 6–9% more bytes than independent compression; Strategy D (a low-rank predictor) was not
   built: its pre-registered trigger failed. On Phase 5A's routing trace (per-layer LRU host caches, bases as their own
   objects, pinned, bounded or on the GPU) the best shared base loses at every host budget (−4.5% to −61%). **Gate 5C1:
   FAIL.**
5. **Independent exact compression pays and is recorded as such.** −34% drive bytes per decode token without a cache
   (2.70 → 1.78 GB projected to 26 layers), −55% and −86% against BF16 caches of 8 and 16 GB. zstd decompression keeps up
   with the drive (3.3–4.9 GB/s on 6 threads); bit-plane restoration does not on this harness (NumPy 0.07–0.3 GB/s, a
   PyTorch device path about 1 GB/s behind Python copies): a runtime needs a device-side merge. Not a Weightsift novelty
   (ZipNN- and DFloat11-style exponent coding).
6. **Progressive representation: bit planes in pages.** 16-row pages (16 neurons' gate and up rows; 16 down rows), each
   plane its own zstd-19 frame; read step by step: sign and exponent, then one mantissa plane per step. Exact,
   page-addressable, not redundant (0.664 of BF16 read whole). XOR planes against any base carry the same prefixes, so a
   base would change only their (larger) size; modular deltas do not localize; a whole-expert frame is not progressive.
7. **Sets and their exact optimum, no verifier.** The weights consistent with what is read are a box (the finite
   completions of each prefix, the rows' resident L∞ bound), built from read views with poisoned unread bits. Over a box the
   real-arithmetic decision Δ·y has a closed-form minimum (separable activations; concave per neuron); it equals enumeration
   on toys and is attained by witnesses. Insufficient information and verifier looseness are therefore told apart
   exactly; the finite-precision floor is Phase 5A's.
8. **Probe (5C2-A) and samples (5C2-B).** The probe's pre-registered rule passed by 0.0007 (one sample at 0.8993 of the
   best independent bytes); 5C2-B ran 12 samples (two per gap bin, prompts 0..23), two schedules (plane-major; greedy by
   estimated bound reduction per byte, which needs no unread bit), and sound sketch metadata at 1% and 10% (bases fitted
   on prompts 24..47). At the same raw bytes the sets are about 3× closer to the truth than Phase 5A's realistic bound;
   but witnesses flip the decision until 0.80–0.99 of the representation, and the certificate needs 0.83 (gaps above 6
   logits) to 1.00 (gaps below 1) of the best independent exact bytes: mean 0.920 (gate ≤ 0.90), 0.943 on the real gap
   distribution. Sketches do not pay (1%: 0.927; 10%: 1.073). Plane-major files keep the reads contiguous (amplification
   ≤ 1.007). **Gate 5C2: STRUCTURAL FAIL.**
9. **5C2-C not run.** Gated on a structural pass. Phase 5A's ceilings bound it: 8.1% of tokens certify under the faithful
   model with every byte read; at most about 1.4% of the compressed bytes could be saved on the token distribution.
10. **Records.** `configs/phase5c-expert-deltas.yaml` (thresholds fixed before any measurement; operational settings
    changed after the codec probe, with reasons), `experiments/phase5c/` (codec probe, structure runs, progressive probe
    and runs; JSON and gzip JSONL with digests; nothing regenerable of size committed), two final runs of every stage with
    `PYTHONHASHSEED` 1 and 2.

## Results

Full report: `history/2026-10-08-awpmi-phase5c-report.md`. Raw data: `experiments/phase5c/`.

- **5C1.** Exact everywhere; no shared structure; best shared base +6.1% stored bytes and −4.5% to −61% drive bytes
  against independent compression on the routing trace; gate FAIL. Independent bit-plane zstd: 0.661 of BF16.
- **5C2.** Bit-plane boxes are the tightest exact partial states measured so far, but carry too little information: the
  decision stays open until the last mantissa plane or two. STRUCTURAL FAIL (mean 0.920 of the best independent exact
  bytes; 0.943 weighted); BF16 blocked by Phase 5A's floor.
- **Verdict.** Neither exact shared bases nor progressive exact deltas make expert AWPMI worth a runtime on Moonlight.
  Exact independent compression is the one measured gain (a third of the drive bytes), an engineering matter.

## Rejected

- *`BF16(base + delta)`*: not exact (above).
- *Shared bases of any kind tried, and a low-rank predictor*: the experts share nothing a base or predictor could carry
  (measured; trigger not met).
- *A context-modelling coder conditioned on a base*: a custom compressor (forbidden by the brief); the order-0 census
  bounds its gain from the measured contexts at 0.02 bits per weight.
- *Modular deltas for progressive reads*: their top bits do not localize the weight.
- *Norm metadata again, renamed* (the brief's §12): the sets here are per-weight intervals from the bits themselves, and
  their optimum is exact.
- *Structured sketches as resident metadata*: measured at 1% and 10%; their bytes cancel or exceed what they let the
  certificate skip.
- *A learned or reuse-aware scheduler*: with every page one plane short of exact, 8 of the 12 samples still admit
  flipping weights, so any order must read part of the last plane there; the remaining uncertainty is spread over input
  columns, output rows and neurons (their largest 10% hold 26–53% of it), leaving an order little to skip. The greedy
  order already gains at most 0.07 over the plane-major one.
- *auto_LiRPA or any verifier*: the box set's optimum is computed exactly; decision 0010's environment is not needed.
- *5C2-C and a physical runtime*: gated (above).
- *Committing reconstructed tensors or compressed artifacts*: regenerable; only records and digests are committed.
