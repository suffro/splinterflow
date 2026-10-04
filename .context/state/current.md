# Current State

## Current focus

**Phase 5A (AWPMI inside routed experts: an oracle study) is complete (2026-10-04). Correctness passes;
the gate FAILs.** The full report is `history/2026-10-04-awpmi-phase5a-report.md`; the decisions are in
decision 0009.

- **Answer to the phase's question: no, not economically.** On Moonlight-16B-A3B's last MoE layer
  (decode, last position, upstream exact), under the certified rounding model, no token certifies
  with any routed-expert byte unread, in any decomposition. Even with every routed byte read, the
  certificate holds on only 8.1% of tokens. A physical expert-AWPMI runtime (Phase 5B) is not
  justified.
- **What was built** (decision 0009):
  - the oracle (`awpmi.oracle.experts`): transformers' experts call recomputed step by step
    (bitwise equal to the capture on every token); enclosures of every intermediate through the
    experts, the routing weights, the combine, the residual and the final norm; Phase 2's pairwise
    certificate generalized to a mixture of experts and checked against all 163,840 vocabulary
    rows; eight decompositions (A, B1, B16, B128, C, D-q8, D-q6+q4, D-q4+q4); realistic and ideal
    bounds and orderings; arithmetic tiers (certified, and labelled what-ifs and diagnostics);
    byte accounting and a physical model of 4 KiB blocks and extents;
  - new bound operators: `reduce_sum` (the combine), `multiply` into float32 (routing weights),
    `linear` on a batch of weights; each validated against the reference's kernels, adversarial
    realizations and exact arithmetic;
  - a capture of the target layer's tensors on the Phase 4B streamed path, equal to Phase 4B's
    reference on every shared step.
- **Results** (768 decode tokens: 48 prompts × 16 steps):

  | Configuration | Coverage | Routed bytes needed (mean) |
  | --- | --- | --- |
  | Certified tier, realistic bounds and ordering (the gate's) | 8.1% | A, B, C 1.003; D 1.50–1.62 |
  | Certified tier, ideal bounds (diagnostic) | 8.1% | 0.65–0.90 when certified; 0.99 at best overall |
  | Real arithmetic, realistic bounds (diagnostic, 96 tokens) | 100% | A, B, C 1.003; D 1.27–1.46 |
  | Real arithmetic, ideal bounds (diagnostic, 96 tokens) | 100% | D-q6+q4 0.47, A and B 0.71, C 0.85 |

  - Ceilings with every byte read: 8.1% certified; what-ifs 16.5% (binary32 accumulator), 17.8%
    (RN-even in elementwise kernels), 24.7% (RN-even everywhere).
  - Correctness: 0 enclosure violations; 0 wrong certified tokens; the reference recomputed bitwise on
    every token.
  - Reproducible: two runs (`PYTHONHASHSEED` 1 and 2, same source tree) with identical digests.
- **Why** (two obstacles, each decisive alone):
  - the rounding floor: with every weight read, 3.2 logits of named error terms on the tightest
    pair (the final norm 1.37, y 0.49, m 0.36, o 0.35, R 0.28, the down accumulation 0.26), plus
    the experts' internal roundings, against a median top-2 gap of 1.1 logits;
  - the bounds on unread parts: norm-based bounds need every byte even in real arithmetic.
- **Findings:**
  - the per-logit interval pass eliminates no vocabulary row on an MoE layer's enclosures; the
    pairwise certificate is needed;
  - reading all of down (A) and paging down by output rows (B) end at the same fraction: nearly
    every down row matters;
  - a neuron-major down (C) reads cleanly (12 KiB records) but needs more neurons;
  - heavier routing weights need more of their expert.

Phases 1A, 1B, 1C, 2, 3, 4A and 4B are complete. Their reports are in `history/`. Phase 4B (decision
0008) runs Moonlight out of VRAM and host RAM, bit for bit equal to an independent reference.

## Recent relevant changes

- New module: `src/awpmi/oracle/experts.py` (imported by nothing outside `awpmi.oracle`; it imports no
  storage, transfer or materialization module: `tests/test_layering.py`).
- Changed modules:
  - `bounds/operators.py` (`reduce_sum`, `linear` on batched weights);
  - `bounds/linear.py` (`matvec`);
  - `bounds/pairwise.py` (`ExpertMixture`, `MixtureTerm`, `mixture_bound`, `relax_mixture`; the
    certificate takes `mixture=`; the Phase 2 `DownProjection` path is unchanged and its benchmark
    reproduces `suffix-run1`);
  - `models/moonlight.py` (names of the residual norm, final norm and LM head).
- The Phase 4B runtime, `StreamedExperts` and the checkpoint formats are unchanged.
- Benchmarks:
  - `benchmarks/expert_oracle.py` (stages prepare, capture, oracle, oracle-real, digest);
  - `benchmarks/expert_oracle_report.py`;
  - `configs/phase5a-expert-oracle.yaml`, with the gate fixed before the full runs;
  - raw results in `experiments/phase5a/oracle-run{1,2}` (the captured tensors are regenerated,
    not committed).
- Decision 0009 is new.
- 551 tests, 41 of them new: the new operators, the experts oracle on a test-size DeepSeek-V3 model
  (recomputation, enclosures in every tier, the mixture bound, the certificate, monotonicity, byte
  accounting, orderings that see no values, sabotage), layering.

## Next

The next phase is **not started**. It needs the user's go-ahead. The report (§9) recommends:

1. **Native runtime and a host-RAM expert tier** (the Phase 4B report's engineering path): it attacks
   the measured bottleneck (drive reads, two thirds of decode) with no change to exactness:
   - an 11–17 GB host tier cuts 67–83% of decode reads (replay);
   - Rust for planning, submission and cache bookkeeping (11% of decode, 18% of prefill);
   - CUDA graphs for the launch-bound decode path (22%).
2. **No expert-AWPMI runtime (Phase 5B).** It would need both obstacles lifted, not one:
   - the rounding model (RN-even would raise the ceiling to 17.8–24.7%, which still caps savings);
   - sharper sound bounds on unread parts (richer resident metadata).
3. **DeepSeek-V3-class scaling last**: it needs the FP8 reference decision, a host for 11 GB reference
   layers, about 700 GB of storage and fewer bytes per token.

Open decisions for the user:

- the next milestone;
- the rounding model for certificates (RN-even, still open from Phase 2; Phase 5A adds its MoE what-ifs);
- the reference for FP8 experts;
- Phase 2 on a larger model.

## Blockers

None technical. The next phase needs the user's decision to proceed.
