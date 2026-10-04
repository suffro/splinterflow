# 0005 — Phase 2: masked fallback by default, faithful rounding kept, an adaptive suffix in the last MLP

Status: accepted (Phase 2, 2026-10-02)

## Context

Phase 1C left two decisions to the user (decision 0004): make the masked LM-head fallback
the default, and adopt round-to-nearest-even (RN-even) as the output-rounding model. Before
Phase 2 the user decided:

- MASKED becomes the default runtime fallback, keeping its mandatory start-up self-test and
  its per-call guard. If either fails, FULL runs. FULL stays the canonical correctness
  fallback and is never removed.
- RN-even is **not** adopted as the certified rounding model. The faithful model of decision
  0001 is kept. RN-even stays an experimental, probed mode whose potential coverage is
  recorded, outside the certified path.

Phase 2 of the roadmap extends certification beyond the LM head, in the order LM head →
final projection → last MLP → last block → …, with bounds for only the operators the suffix
needs, each derived, documented, isolated, unit-tested and independently checked (§2.1–§2.5),
and a stated KV-cache mode (§2.8).

A feasibility probe came first (200 prompts, every suffix weight exact, bounds only through
the reference's roundings; preliminary, superseded by the benchmark below). With per-logit
intervals, the faithful model certified 17% of tokens through the final projection and 0%
through the last MLP; RN-even, 64% and 6%. Under the faithful model every BF16 rounding of a
value that is not known exactly may move it by up to one ulp, about 2⁻⁷ of its size, and the
LM head turns the roundings of h into Σ_k |W_jk|·|h_k|·2⁻⁷ per logit, comparable to the
median top-2 gap (0.875 logits). The obstacle is the reference's own rounding of activations,
not weight bytes.

## Decision

1. **Fallback (user decision).** `RefinementLMHead` defaults to `FallbackMode.MASKED`. The
   self-test is mandatory (`self_test_trials` > 0). A failed self-test no longer refuses to
   build: the head sets `masked_enabled = False` and runs FULL for every fallback. A tripped
   guard runs FULL for that call. FULL is unchanged.
2. **Rounding (user decision).** `awpmi.bounds.rounding` defines `RoundingModel.FAITHFUL`
   and `NEAREST_EVEN`, and `RoundingAssumptions` pairs one model for GEMM epilogues with one
   for elementwise kernels. Only `CERTIFIED` (faithful for both) certifies.
   `AdaptiveSuffixRuntime` refuses any other pair unless `experimental=True`, and then
   reports `would_certify` and the would-be token, never `certified`, and runs no fallback.
   Two experimental pairs are recorded: RN-even in elementwise kernels only (they convert
   their fp32 result with PyTorch's own BF16 conversion, round-to-nearest-even in its
   source; the operator tests confirm it on this platform's CPU and CUDA kernels) and
   RN-even everywhere (adding decision 0004's probe of the GEMM epilogue). In experimental runs the LM head's own logit rounding stays
   faithful, so their coverage is a lower bound of the what-if.
3. **Stages.** `SuffixStage`: `lm_head` (depth 0, Phase 1C), `down` (depth 1, the *final
   projection*: for a Llama model, the last projection before the final norm is the last
   layer's `down_proj`) and `mlp` (depth 2, the last MLP). The last block's attention is
   not implemented. The depth-2 results (below) show that a deeper suffix cannot certify
   at a useful cost under the faithful model, and the roadmap asks for bounds only where
   the current suffix needs them.
4. **Operators** (`awpmi.bounds.operators`): `linear`, `residual_add`, `multiply`, `silu`,
   `rms_norm`, each modelling the reference operation as `transformers` 5.18 runs it on
   BF16 tensors. Documented fp32 assumptions: at most 2⁻¹⁸ relative error per elementwise
   fp32 operation (32 ulps, against ≤ 2 ulps documented for `expf`/`rsqrtf` and ≤ 4 for
   `powf`), possible flushing below the normal range, γ_{n+2}(2⁻²²) per reduction (decision
   0001). Each output is rounded by the operation's rounding model. `Enclosure` (lower,
   upper, provenance) is the residual state at internal tensors (roadmap §2.2).
5. **Pairwise certificate** (`awpmi.bounds.pairwise`). Behind the final RMSNorm every logit
   shares the positive scale q and the same rounding errors of h, so separate intervals
   double-count them. For a winner w and each contender j the certificate lower-bounds
   (acc_w − acc_j)/q, scale-free. It takes the larger of two bounds: a box bound over the
   enclosure of y, and a decomposed bound y = r + W_d·a + errors that keeps the down
   projection's cancellation (M = W_dᵀΔ), bounding unread columns by Cauchy–Schwarz on
   resident column norms. A pair is certified when the faithful logits are at least two
   grid spacings apart. This settles ties whatever the indices, and needs no assumption
   that elements are rounded by the same function. In the probe it raised the
   exact-weight ceiling from 17% to 30.5% (final projection) and from 0% to 11% (last MLP).
6. **Suffix pages** (`awpmi.stores.suffix.MLPStore`). The page is one role of one neuron:
   a gate row, an up row or a down column, stored neuron-major so that each is
   contiguous. `WeightPage` gains `layer` and `tensor_role` (roadmap §2.7). Resident
   metadata: down column norms, down row norms (L2, L∞) and up row norms. The budgeted
   share of pages is read in a fixed order of decreasing bound contribution: |a_i|·C_i for
   the final projection; the unread-activation bound times C_i for the MLP, after every
   gate row.
7. **Runtime** (`awpmi.suffix_runtime.AdaptiveSuffixRuntime`):
   - exact prefix (`suffix_prefix`: the model's own forward, stopped by a pre-hook);
   - enclosures through the suffix;
   - the Phase 1C states on the enclosure of h (`run_enclosure`: input spread added, no
     fallback);
   - then the pairwise certificate;
   - else read the rest of the suffix, recompute it with the reference's own operations
     on all positions (bitwise), and run Phase 1C on the exact h.

   One token's passes share their reads (`HeldReads`), so no row or page is read or
   counted twice. Row limits on the enclosure pass (32,768 int4, 256 exact rows; the
   "limited" policy) are an efficiency rule. The benchmark also runs without them
   ("unlimited") at budget 1.0, to show the certificates' ceiling whatever it costs.
8. **KV cache (§2.8).** Every stage starts after the last layer's attention, so all keys
   and values are written by the exact prefix: Mode A holds for MLP suffixes. A test runs
   cached greedy generations and checks the cache bitwise against the reference's at every
   step.
9. **Independent validation (§2.4, §2.5).** `auto_LiRPA` cannot be installed here: every
   release pins torch < 1.13, against torch 2.14 and Python 3.13. Bounds are checked
   instead in three independent ways:
   - against the reference's own kernels, on every grid input of small enclosures, on CPU
     and CUDA;
   - against adversarial faithful realizations, with each rounding up or down at random
     and each fp32 operation at its assumed error;
   - against exact `Decimal`/`Fraction` arithmetic.

   auto_LiRPA would in any case bound real arithmetic only, not floating-point rounding.
   Every benchmark prompt then checks every intermediate's enclosure, the pairwise bounds
   and the margins against the reference's values.
10. **Gate and decision** (`configs/phase2-suffix.yaml`, fixed before the full run). The
    gate is the roadmap's:
    - zero certified and fallback mismatches;
    - at least one certified decision with some suffix page unread;
    - conservative propagation (zero violations);
    - curves recorded;
    - plus bitwise exact paths, audited bytes and reproducible digests.

    Separately, a stage is BENEFICIAL if its best budget saves at least 1% of the adaptive
    region against reading the whole suffix and running Phase 1C.

## Results

Full report: `history/2026-10-02-awpmi-phase2-report.md` (1000 prompts, two identical runs).

- **Gate: PASS.**
  - 0 mismatches in 37,000 runs; 0 violations at any intermediate, pairwise bound or logit.
  - Exact paths bitwise.
  - 232 certified decisions with suffix pages unread.
- **Coverage of the suffix certificate under the faithful model.**
  - Depth 1: 27.3% with every page read, 18.8% at 95%, 4.4% at 90%, 0% at ≤ 80%.
  - Depth 2: 0% with row limits; 14.9% without them, at 1.49× the region's bytes.
  - Under RN-even (what-if) depth 1 reaches 63.7%. The RN-even model held on every
    intermediate of every prompt.
- **Decision: NOT BENEFICIAL at either depth.** The best mean saving is −0.0002 of the
  region against reading the suffix in full and running Phase 1C. The reference's own
  roundings are 88% of the uncertainty with every page read.
- *Later (Phase 5A, decision 0009):* the same floor on Moonlight's last MoE layer. With every
  routed weight read the faithful certificate holds on 8.1% of tokens; RN-even in elementwise
  kernels would give 17.8%, RN-even everywhere 24.7%.

## Rejected

- *RN-even in the certified path.* The user's decision. It is recorded as a what-if.
- *Precision levels (int-b) for suffix weights.* An int8 level's remainder bound on one
  down-projection output is 5–20× the faithful rounding floor of that output, so it would
  never certify. Only whole pages help.
- *The masked GEMM's row independence in the certified path* (recomputing a subset of rows
  of a suffix GEMM "exactly"). It stays in the fallback path, behind its self-test and
  guard.
- *Per-logit intervals alone through the final norm.* They lose the common scale q and the
  down projection's cancellation (17% against 30.5% at depth 1 in the probe).
- *The last block's attention now.* It would deepen a suffix that already cannot certify at
  depth 2 under the faithful model.
- *auto_LiRPA as the independent validator.* It cannot be installed with the pinned
  dependencies.
