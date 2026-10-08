# Current State

**Project name: Weightsift.** Use this name consistently in documentation, code comments, and filenames.

**CLI:** `weightsift` is the command; `wsift` is its equivalent shorthand. Both expose
the `pack` command (`lm-head`, `experts`, `expert-index`). The Python package is `awpmi`.

## Latest: Phase 5C (exact shared bases and progressive expert deltas) is complete (2026-10-08)

**Answer: neither pays on Moonlight.** The report is `history/2026-10-08-awpmi-phase5c-report.md`; the decision is 0011.
The brief was the user's of 2026-10-08.

- **Exact structural reuse (5C1): gate FAIL.** A layer's 64 routed experts are, to every test, independent (4 layers × 3
  matrices: flat cross-expert spectrum, correlations and sign/exponent agreement at independence, no permuted neuron
  copies). No expert has a delta (XOR or modular, on the BF16 patterns, reconstructed bit for bit) cheaper than itself
  against any other expert. Every base strategy (first expert, medoid, best of three, clusters, a synthetic median) costs
  6–9% more stored bytes than compressing each expert alone, and loses on Phase 5A's routing trace at every host budget
  (−4.5% to −61%), bases pinned, bounded or on the GPU.
- **Independent exact compression is the measured gain**: bit planes (sign, exponent byte, mantissa bits) with zstd-19,
  0.661 of BF16 (the order-0 entropy is 0.658): −34% drive bytes per decode token without a cache (2.70 → 1.78 GB,
  projected to 26 layers), −55%/−86% against BF16 host caches of 8/16 GB. zstd decompression keeps up with the drive;
  restoring bit planes needs a device-side merge (this harness: ~1 GB/s).
- **Progressive materialization (5C2): STRUCTURAL FAIL.** Bit planes in 16-row pages (each plane its own frame) are an
  exact, non-redundant, page-addressable representation; the set of weights consistent with what is read is a box whose
  real-arithmetic decision minimum is exact (closed form, witnesses). At the same raw bytes it is about 3× closer to the
  truth than Phase 5A's realistic bound, but weights consistent with everything read flip the decision until 0.80–0.99 of
  the representation. On 12 samples the certificate needs 0.83 (gap > 6 logits) to 1.00 (gap < 1) of the best independent
  exact bytes: mean 0.920 (gate ≤ 0.90), 0.943 on the real gap distribution. Sketch metadata (1%, 10%) does not pay.
- **Faithful BF16 (5C2-C): not run** (gated). Phase 5A's floor bounds it: 8.1% of tokens certify even with every byte
  read; at most ~1.4% of the compressed bytes could be saved.
- **Correctness**: every reconstruction bit for bit against safetensors' own read; files equal the publisher's sha256;
  Phase 5A's reference recomputed bitwise on 15 samples (also from decoded pages); 0 soundness violations (sets never
  depend on unread bits, minima equal enumeration on toys, witnesses inside their sets). Tests: 79 new in
  `research/expert_deltas/tests`; Weightsift's 555 unchanged.

## Previous focus

**Phase 5A2 (CROWN / auto_LiRPA expert oracle) is complete (2026-10-07), stopped after stage 1.5** (report
`history/2026-10-07-awpmi-phase5a2-report.md`, decision 0010). Given Phase 5A's L2 remainder norms, auto_LiRPA's CROWN
does not make expert AWPMI materially more viable: it is sound there only in an experimental mode and about 34× below
Phase 5A's bound, and the uncertainty sets themselves hold decision-flipping weights until about 0.82–0.91 of the routed
bytes in real arithmetic.

Phases 1A, 1B, 1C, 2, 3, 4A, 4B, 5A, 5A2 and 5C are complete; their reports are in `history/`. Phase 4B (decision 0008)
runs Moonlight out of VRAM and host RAM, bit for bit equal to an independent reference.

## Recent relevant changes

- `research/expert_deltas` (new; decision 0011): `expert_deltas/` (`bits`, `codecs`, `source`, `compression`, `census`,
  `structure`, `replay`, `progressive`, `oracle`, `records`), drivers `probe_codec.py`, `structure_run.py`,
  `structure_report.py`, `progressive_probe.py`, `progressive_run.py`, tests, README.
- `pyproject.toml` / `uv.lock`: dependency group `research` (default): zstandard, lz4, SciPy. The package's own
  dependencies are unchanged.
- `configs/phase5c-expert-deltas.yaml`; raw results in `experiments/phase5c/`.
- No change under `src/`, `benchmarks/` or `tests/`: Phase 4B's runtime and Phase 5A's oracle are unchanged.
- Decision 0011 is new.

## Next

The next phase is **not started**. It needs the user's go-ahead. The candidates:

1. **The engineering path with exact compression** (recommended): Phase 4B's native runtime and host-RAM expert tier,
   with the routed experts stored as exact bit-plane (or byte-split) zstd pages: a third fewer drive bytes per decode
   token, about 1.5 times as many experts per byte of cache. Its open question is decode cost (a device-side plane merge, or
   GPU decompression).
2. **No more expert AWPMI on BF16 Moonlight** unless the rounding model for certificates changes: three representations
   (Phase 5A's decompositions, Phase 5A2's verifier, Phase 5C's exact bit planes optimized exactly) agree that deciding a
   token needs almost all of an expert's information, and the faithful floor decides 92% of tokens anyway.
3. **Exact structural reuse on another model** only if its experts were upcycled from a shared dense model (the census
   would show shared structure; Moonlight's show none).

Open decisions for the user:

- the next milestone;
- the rounding model for certificates (RN-even, still open from Phase 2);
- the reference for FP8 experts;
- Phase 2 on a larger model;
- whether the Phase 5C brief (left untracked in the working tree's `.context/state`) is added to the repository.

## Blockers

None technical. The next phase needs the user's decision to proceed.
