# 0009 — Phase 5A: an oracle of AWPMI inside routed experts; not economical under the certified rounding model

Status: accepted (Phase 5A, 2026-10-04)

## Context

Phase 4B (decision 0008) runs Moonlight-16B-A3B out of VRAM and host RAM. A decode token reads 2.70 GB of routed
experts (6 of 64 in each of 26 layers), and the drive is two thirds of a decode step. The Phase 4B report recommended
measuring first, as an oracle, whether AWPMI could avoid reading most of the experts the router selects.

The user's Phase 5A brief:

- The question: once the router has selected an expert, must Weightsift materialize all of it to keep the final
  discrete decision? What fraction of the routed bytes must be read before AWPMI can certify the reference's token?
- Scope: Moonlight, the last MoE layer, the last position, decode only, upstream exact (Mode A). The target stays the
  Phase 4B BF16 reference; the certificate stays conservative and is never set by a heuristic.
- Inspect the experts' computation exactly as transformers runs it. Bound one expert, the routing weights and the
  combine, then propagate to the token through the exact suffix with the strongest validated certificate.
- Strategies: A (neuron pages, whole down projection), B (neuron pages and down output-row pages at several
  granularities), C (a neuron-major down, its storage counted), D (precision refinement, storage counted), and E (a
  hybrid) only if justified.
- Routed experts only: the AWPMI fraction is relative to the routed bytes, not to all 64 experts. Several experts are
  combined exactly.
- Realistic deterministic schedulers and an idealized diagnostic oracle, kept apart.
- Per sample and strategy: bytes, fractions, pages, neurons, levels, margins, residual bounds, refinement steps; modelled
  4 KiB amplification, extents, physical bytes and storage expansion. Coverage, mean, median, p90, p95, fallback share,
  and breakdowns by margin, prompt length and expert.
- A gate fixed before the final runs (strong pass ≤ 0.50, pass 0.50–0.70, weak 0.70–0.90, fail > 0.90).
- Keep the faithful rounding model; an RN-even what-if only if nearly free and labelled; no physical runtime; the Phase
  4B path unchanged; regressions; two runs with different `PYTHONHASHSEED` and identical digests; Q1–Q9 answered.

A feasibility probe came first (16 Phase 4B prompts × 8 decode steps, every routed weight exact, the pairwise
certificate against the 64 nearest rows): the faithful certificate held for 7.8% of tokens even then. The uncertainty
of the tightest pair was about 3.6 logits, mostly the reference's own roundings after the experts. The probe fixed the
design below (whole-vocabulary pairwise certificate; arithmetic tiers to attribute the floor) and the gate's coverage
band; it did not change the bands themselves, which are the brief's.

## Decision

1. **Samples.** 48 wikitext-2 prompts (16–1,024 tokens; the first 16 are Phase 4B's), 16 greedy decode steps each:
   768 decode samples. A capture stage runs Moonlight on Weightsift's Phase 4B streamed path (no cache, the 256 MiB call
   budget, the 6 GB cap; the files verified against the Hub's sha256) and records, at the last MoE layer's last
   position, the experts' input x, the router's choice and weights, the residual r, the shared experts' output S, and
   the reference's R, m, y, h and logits. Every step of Phase 4B's prompts is compared with Phase 4B's reference
   records in every digest. The tensors are not committed (regenerated; sha256 in the digest).
2. **The reference, recomputed.** `experts_call_reference` runs transformers' `grouped_mm_experts_forward` step by step
   with its own functions on the layer's whole weights (the call's shapes), giving every intermediate: g, u (BF16
   GEMM epilogues), s = rnd(silu(g)), a = rnd(s·u), o (BF16), z = fl32(o·w) (BF16 × float32 routing weight → float32),
   R = rnd(Σ_k z_k) (float32 sum, then BF16); then m = rnd(R + S), y = rnd(r + m), n, h = RMSNorm(y) and the logits by
   the reference's operations. On every sample it must equal the capture bitwise (R, m, y, h, the logits' sha256, the
   token) and the shared experts recomputed from x must equal S: this is the exact full-materialization fallback. The
   target layer's weights equal the Phase 4B reference's row digests.
3. **Bounds.** The existing operators (`awpmi.bounds.operators`), with three additions, each tested against the
   reference's kernels, adversarial realizations and exact arithmetic: `reduce_sum` (the combine), `multiply` into
   float32 (the routing weights), `linear` on a batch of weights (one call's routed experts). `DeepseekV3RMSNorm` runs
   LlamaRMSNorm's operations (tested). Every intermediate's enclosure is checked against the reference's values on every
   evaluation; in the certified tier a violation is a hard failure.
4. **Certificate.** The pairwise certificate of decision 0005, its decomposed bound generalized to a mixture of routed
   experts (`ExpertMixture`): y = r + S + Σ_e w_e·(K_e + X_e)·a_e + η, each expert's known down part through its exact
   projection M_e = K_eᵀ·Δ, its unknown part by row or column norms, η the named roundings and accumulations (down
   accumulation, o, z, the combine, R, m; y's own rounding and the norm's as in decision 0005). It is checked against
   every one of the 163,840 vocabulary rows: the 64 nearest by the reference's logits first (a filter that can only
   reject), then all in blocks, M in binary32 with a rank-one error bound. The candidate is the row with the largest
   centre logit, as a runtime would choose it. The per-logit interval pass of Phases 1–2 is not used: on the MoE's
   enclosures it eliminates no row.
5. **Tiers.** Arithmetic: `certified` (faithful, u = 2⁻²², decisions 0001 and 0005) is the only one that certifies;
   `certified_u24` (an IEEE binary32 accumulator), `rn_elementwise`, `rn_even` (round-to-nearest-even in elementwise
   kernels, or everywhere) are labelled what-ifs, ceilings only except rn_even; `real` (exact real arithmetic, no
   rounding floor) is a diagnostic of the decompositions themselves. Bounds: realistic (resident metadata only) or
   ideal (the missing part's true contribution). Orderings: realistic (bound contribution per byte to the tightest
   pair, from the first state) or ideal (true contribution to the reference's top-2 difference).
6. **Strategies.** A: the whole down projection, then neuron pages (gate and up rows). B1, B16, B128: neuron pages and
   down output-row pages of 1, 16 or 128 rows. C: neuron-major pages (gate row, up row, down column). D-q8, D-q6+q4,
   D-q4+q4: a coarse per-row level of all three matrices (decision 0003's decompositions), then row by row up to the
   BF16 rows. E (a hybrid) was not built: under realistic bounds no strategy certified before reading everything, so no
   evidence pointed to a combination.
7. **Search.** A cell reads its units in a fixed order and looks for the first budget (multiples of 1/64 of the routed
   bytes) at which the certificate holds. With realistic bounds more knowledge only narrows every enclosure (each
   operator is inclusion-monotone), so certification is monotone and a bisection on the nearest pairs finds the first
   budget exactly; the full check runs there. A sample whose ceiling fails in a tier falls back in every cell of that
   tier without a search (for realistic bounds this is a theorem, for ideal ones the ceiling caps the diagnostic).
8. **Bytes.** A unit's bytes are what the checkpoint, a level or a neuron-major copy stores for it; the routed experts'
   metadata is charged to every token; a token a strategy never certifies costs its whole schedule (levels, then every
   BF16 byte; decision 0003). The physical model counts 4 KiB blocks and extents from the experts' real offsets in the
   published file (C also on a neuron-major copy, D on level files), and storage multipliers.
9. **Gate** (`configs/phase5a-expert-oracle.yaml`, fixed before the final runs): the brief's bands on the primary
   configuration (certified tier, realistic bounds and ordering, best strategy), with coverage thresholds (strong pass
   ≥ 0.80, pass ≥ 0.60, fail below 0.10), and pass also if an ideal certified-tier oracle reaches 0.70.
10. **Code.** The oracle lives in `awpmi.oracle.experts`. Nothing outside `awpmi.oracle` imports it, and it imports no
    storage, transfer or materialization module (`tests/test_layering.py`). The Phase 4B runtime, `StreamedExperts` and
    the checkpoint formats are unchanged; Phase 2's `DownProjection` path of the pairwise certificate is unchanged
    (its benchmark reproduces `suffix-run1`).

## Results

Full report: `history/2026-10-04-awpmi-phase5a-report.md`. Raw data: `experiments/phase5a/oracle-run{1,2}`: two runs
with `PYTHONHASHSEED` 1 and 2 on the same source tree, all five digests identical.

- **Correctness.** The capture equals Phase 4B's reference on all 144 shared steps; the target layer's weights equal
  the reference's digests; the recomputed reference equals the capture on 768 of 768 tokens (R, m, y, h, logits,
  token, shared experts). No enclosure was violated in any tier, and no certified or would-certify token differs from
  the reference's, except in the real-arithmetic diagnostic on 10 exact BF16 ties (gap 0), where the real computation
  has a strict winner the reference's lowest-index rule does not pick.
- **Ceilings** (every routed weight read): the certificate holds on 8.1% of tokens under the certified model; 16.5%
  with a binary32 accumulator, 17.8% with RN-even in elementwise kernels, 24.7% with RN-even everywhere, 100% in real
  arithmetic. The tightest pair's named error terms sum to 3.18 logits on average (the final norm 1.37, y 0.49, m 0.36,
  o 0.35, R 0.28, the down accumulation 0.26, the LM head 0.07), and the experts' internal roundings widen a inside
  the main term. The median top-2 gap is 1.13 logits; nothing below 2.75 certifies; 44–46% certify above 4.
- **Gate configuration** (certified tier, realistic bounds and ordering): A, B1, B16, B128 and C 1.003 of the routed
  bytes (every byte plus metadata) at mean, median, p90 and p95; D-q8 1.501, D-q6+q4 1.624, D-q4+q4 1.509. Coverage
  8.1% for all. No token was certified with any routed byte unread. **Gate: FAIL.**
- **Diagnostics.** The ideal ordering with realistic bounds changes nothing. Ideal bounds certify the 62 certifiable
  tokens at 0.65 (D-q6+q4) to 0.90 (A) of the routed bytes; 0.99 at best over all tokens. In real arithmetic (96
  tokens), realistic bounds still need every byte for A, B and C (D 1.27–1.46); ideal bounds need 0.47 (D-q6+q4), 0.53
  (D-q4+q4), 0.56 (D-q8), 0.71 (A, B), 0.85 (C). Heavier routing weights need more of their expert.
- **Layouts and storage** (modelled): reading all of down (A) and down-row pages (B) both end at 0.71 in the best
  diagnostic; B1's 2.75 KiB rows amplify 4 KiB reads 1.22×. A neuron-major copy (1.33× storage) reads cleanly but needs
  0.85. D's levels cost 1.50–1.63× storage. Where the diagnostics save bytes, a token's reads become 420–1,450 extents
  instead of 6.
- **Projection** (decode, 2.70 GB of routed experts per token): no saving under the gate's configuration; 1.27 GB only
  if every layer behaved like the last with both obstacles removed (real arithmetic and ideal bounds).
- **Verdict (Q7): NO, expert AWPMI is currently not economical.** Two obstacles, each decisive alone: the rounding
  floor (92% of tokens undecidable even with every byte read) and the looseness of sound bounds on unread parts (no
  early certificate even without the floor). No Phase 5B runtime is built; the next milestone is the user's choice
  (the Phase 4B report's engineering path is recommended).

## Rejected

- *Per-group exactness in the certified path* (computing a fully read expert's output with the reference's grouped GEMM
  run on that expert's group alone). It is the empirical kernel property the chunked call relies on for execution
  (decision 0008), but as decision 0005 rejected the masked GEMM's row independence in a certificate, it stays out of
  the certified path. It would also not help: the roundings after the experts remain.
- *RN-even, or a binary32 accumulator model, in the certified path.* The user's decisions 0001 and 0005; reported as
  what-ifs.
- *The per-logit interval pass on the MoE's enclosure of h*: it eliminated no vocabulary row on Moonlight.
- *A certificate against the nearest rows only*: not a certificate; the nearest rows only filter.
- *Statistical or confidence bounds on unread parts*: not conservative.
- *Learned or LLM schedulers*: out of scope; deterministic orders only.
- *A physical runtime, a new checkpoint format in the runtime, changes to `StreamedExperts`*: out of scope (the brief).
- *Strategy E*: not justified by the evidence (above).
