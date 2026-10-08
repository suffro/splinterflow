# Current State

**Project name: Weightsift.** Use this name consistently in documentation, code comments, and filenames.

**CLI:** `weightsift` is the command; `wsift` is its equivalent shorthand. Both expose
the `pack` command (`lm-head`, `experts`, `expert-index`). The Python package is `awpmi`.

## Latest: Phase 5A2 (CROWN / auto_LiRPA expert oracle) is complete (2026-10-07), stopped after stage 1.5

**Answer: no.** Given Phase 5A's L2 remainder norms, auto_LiRPA's CROWN does not make expert AWPMI materially more
viable. Stage 2 (12 samples) was not run. The report is `history/2026-10-07-awpmi-phase5a2-report.md`; the decision is
0010.

- **What stage 1.5 asked** (the user's brief of 2026-10-07): give auto_LiRPA the L2 information Phase 5A already holds
  (‖residual row‖₂ ≤ R instead of independent boxes), upstream functionality only, and tell verifier looseness apart from
  the uncertainty set's own looseness. Toys first, then three real samples (prompt 12 step 7, the exact tie; prompt 12
  step 9, gap 3.88; prompt 23 step 4, gap 11.06), D-q6+q4, real arithmetic.
- **Upstream (toys, exact optima).** Independent L2 balls go in as one root per group of rows with one `F.linear` per
  group: with an exact input CROWN is exact and the groups independent. Stacking the groups into one weight fails
  upstream. For down after uncertain activations, upstream's default interval mode is **unsound** (an L2 root's interval
  bounds are its centre); only the experimental `AUTOLIRPA_L2_DEBUG=1` mode, on the reduced graph, is sound.
- **Verifier.** L2 CROWN is 15× looser than box CROWN at the q6 state, 34–35× below Phase 5A's realistic bound and about
  40× below its own set's attained value (its product relaxation works on each ball's box hull, ±ρ2 per entry, ρ2 ≈
  22·ρ∞). It decides the samples with a margin only from 1.336 of the routed bytes (Phase 5A: 0.961, 1.008). Box CROWN is
  about 2× below its set's minimum at q6, close later.
- **Uncertainty sets (decisive).** Phase 5A's realistic bound is already the L2 set's decoupled lower bound; the set's
  minimum is within 16% of it at q6. Weights consistent with the L2 metadata flip a pair until 1.633, 0.868 and 0.914 of
  the routed bytes; with every resident byte (L2 and L∞) until 1.617, 0.821 and 0.852. So no verifier given this
  metadata saves more than ~0.12–0.14 of the routed bytes over Phase 5A on these samples, in real arithmetic, before the
  rounding floor (Phase 5A: 8.1% certified ceiling).
- **Stop conditions** (config `l2`): C met; B met on two of three samples by the pre-registered rule; A in part; D not
  met (L2 CROWN ~100 s per state, about 30× the phase's runtime bar).
- **The checkpoint's unexplained `exit 127`, resolved.** A process that ran auto_LiRPA on CUDA fails fast while its
  native libraries unload at exit (0xC0000409; Git Bash shows 127), after every record is written. The verifier's scripts
  now end with `TerminateProcess` (`crown_oracle.graph.leave`): exit status 0 after the same work.
- **Correctness.** Every box and L2 ball of every state holds the true weights; poisoning unread rows changes no set; no
  bound above the truth or an attained value of its set; every witness inside its set. Tests: 54 in the verifier
  environment (26 new), 555 in Weightsift (unchanged behaviour; the root cannot import auto_LiRPA).

## Previous focus

**Phase 5A (AWPMI inside routed experts: an oracle study) is complete (2026-10-04). Correctness passes; the gate FAILs**
(report `history/2026-10-04-awpmi-phase5a-report.md`, decision 0009). On Moonlight-16B-A3B's last MoE layer, under the
certified rounding model, no token certifies with any routed-expert byte unread; with every byte read the certificate
holds on 8.1% of tokens. Two obstacles, each decisive alone: the rounding floor (3.2 logits of named error terms on the
tightest pair against a median top-2 gap of 1.1) and the bounds on unread parts (norm-based bounds need every byte even
in real arithmetic; with ideal bounds and no floor, D-q6+q4 would need 0.47).

Phases 1A, 1B, 1C, 2, 3, 4A, 4B and 5A are complete; their reports are in `history/`. Phase 4B (decision 0008) runs
Moonlight out of VRAM and host RAM, bit for bit equal to an independent reference.

## Recent relevant changes

- `research/crown_expert_oracle` (the isolated verifier environment; decision 0010):
  - new: `crown_oracle/l2.py` (L2 groups as auto_LiRPA graphs), `crown_oracle/rowsets.py` (the sets' optima and
    witnesses; diagnostics), `probe_l2.py` (the stage 1.5 probe), `tests/test_l2.py`;
  - changed: `sets.py` (`matrix_balls`), `attack.py` (`reduced_vertex`), `graph.py` (`leave`), `run.py` (parts `l2` and
    `l2crown`; L2 balls in `validate`), `report.py` (`--l2`), `export.py` (`--select`), the README.
- `configs/phase5a2-crown-oracle.yaml`: the `l2` section (stage 1.5's settings and stop conditions).
- Raw results: `experiments/phase5a2/probe_l2`, `experiments/phase5a2/l2` (the artifact's tensors are regenerated, not
  committed).
- No change under `src/`, `benchmarks/` or `tests/`: the Weightsift environment, Phase 4B's runtime and Phase 5A's
  oracle are unchanged.
- Decision 0010 is new.

## Next

The next phase is **not started**. It needs the user's go-ahead. The candidates:

1. **Exact structural reuse** (the guide's §7): a shared exact expert base plus exact per-expert deltas, which reduces
   bytes without certification. The likely next investigation; outside Phase 5A2's scope.
2. **Native runtime and a host-RAM expert tier** (the Phase 4B report's engineering path): it attacks the measured
   bottleneck (drive reads, two thirds of decode) with no change to exactness.
3. **No expert-AWPMI runtime (Phase 5B)**, and no more verifier work on norm metadata: Phase 5A2 shows that the
   uncertainty sets themselves, not the verifier, keep decisions open until ~0.82–0.91 of the routed bytes.

Open decisions for the user:

- the next milestone;
- the rounding model for certificates (RN-even, still open from Phase 2);
- the reference for FP8 experts;
- Phase 2 on a larger model.

## Blockers

None technical. The next phase needs the user's decision to proceed.
