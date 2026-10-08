# 0010 — Phase 5A2: auto_LiRPA's CROWN on the expert oracle; stopped after stage 1.5

Status: accepted (Phase 5A2, 2026-10-07)

## Context

Phase 5A (decision 0009) found expert AWPMI not economical on Moonlight's last MoE layer, for two reasons each decisive
alone: the rounding floor of the certified model, and the looseness of sound bounds on unread parts (norm-based bounds
needed every byte even in real arithmetic).

Phase 5A2 (started 2026-10-04) asks whether a mature bound-propagation verifier, auto_LiRPA's CROWN family, tightens
Phase 5A's bounds on unread routed-expert weights enough for expert AWPMI to pay. Its plan was staged, with gates fixed
in `configs/phase5a2-crown-oracle.yaml` before any CROWN run on real data: a feasibility probe (stage 1), 12 samples
across top-2 gaps (stage 2), more only under an escalation rule. The development checkpoint (`c128712`, 2026-10-06)
completed stage 1 and one development sample (prompt 12 step 7, an exact BF16 tie): in real arithmetic at the q6 state
the exact optimum of auto_LiRPA's box sets (median −865 over the comparison pairs) lay below Phase 5A's realistic bound
(median −694), because boxes cannot express the L2 remainder norms Phase 5A uses.

The user's brief of 2026-10-07 inserted stage 1.5 before stage 2:

- give auto_LiRPA the L2 information Phase 5A already holds (an unread residual row with ‖r‖₂ ≤ R instead of independent
  boxes), with upstream functionality only: no custom CROWN, LiRPA, concretizer, operator, fork or patch; graph
  construction from existing `BoundedParameter`s, `torch.stack`/`cat`, reshapes and additions is allowed;
- find which structures upstream can express (one ball per tensor, per matrix, per page, per row) and their cost;
- reuse the q6+q4 hierarchy and its remainder norms (D-q6+q4); never use unread values (a poisoning test);
- toys first (soundness against realizations, zero radius, shrinking radii, independence, CROWN against the exact
  optimum, box against L2); then 1–3 real samples (the tie, a moderate and a large margin), real arithmetic first;
- per state: bytes, the reference margin, Phase 5A's realistic and ideal bounds, box and L2 CROWN, the box and L2 sets'
  tight optima, cost; the share of Phase 5A's realistic→ideal distance closed by L2 CROWN; Q1–Q7;
- stop without stage 2 on A (upstream cannot express the useful L2 structure), B (even the L2 set's exact optimum stays
  negative until about full materialization), C (L2 CROWN closes under ~20% of the distance, stays near full
  materialization, does not beat Phase 5A's realistic bound) or D (prohibitive cost); stage 2 only on clear positive
  evidence (L2 CROWN materially above Phase 5A's realistic bound, and fewer bytes or ≥ 30–40% of the distance closed).

## Decision

1. **Environments** (unchanged from the checkpoint). The verifier lives in `research/crown_expert_oracle`: Python
   3.11.16, torch 2.11.0+cu130, auto_LiRPA 0.7.2 @ `5a098e8f` (BSD-3-Clause, unmodified), its own `uv.lock`. The
   Weightsift environment is unchanged and cannot import auto_LiRPA (`tests/test_crown_boundary.py`); the verifier imports
   no `awpmi`. They exchange only `export.py`'s artifact (safetensors and a manifest with every file's sha256).
2. **Graphs** (stage 1): the verification graphs are PyTorch modules: the full experts graph and the reduced graph (an
   activation box × the down projections); SiLU written g·σ(g); exact zero biases (auto_LiRPA's Gemm path); Δ·y bounded
   directly through the specification matrix, expert by expert.
3. **Sets from resident metadata only**, built from a read view that hides every unread row (tested by poisoning it):
   boxes (each row's own L∞ norm, each read level's L∞ remainder; stage 1) and L2 balls (each row's own L2 norm, each read
   level's L2 remainder; `sets.matrix_balls`, stage 1.5). "Resident" names their intersection: every resident byte.
4. **L2 for auto_LiRPA** (`crown_oracle/l2.py`): one `BoundedParameter` with `PerturbationLpNorm(norm=2)` per group of
   rows, one `F.linear` per group ("split"); a group's radius is ‖ρ_G‖₂, the smallest ball about its centre holding every
   product of its rows' balls. On real samples, L2 CROWN is the reduced graph with each row of down its own ball, in
   auto_LiRPA's box-hull interval mode (`AUTOLIRPA_L2_DEBUG=1`): the only configuration stage 1.5's probe found sound.
5. **The sets' own optima** (`crown_oracle/rowsets.py`; diagnostics, never a certificate). Every set is a product over
   rows and x is exact, so the activations' set is exactly a box and Δ·y attains its minimum at one of its vertices, each
   down row then at its own extreme. On toys every vertex is enumerated (exact). On Moonlight: the L2 set's decoupled
   lower bound and an attained vertex value; the box set's exact minimum (separable); and per set a witness, weights
   inside the set (each row at its set's maximizer: closed forms for a ball, a ball ∩ a further ball, a ball ∩ a box)
   with their Δ·y. A witness below zero proves that the set itself holds weights that flip the pair, whatever the
   verifier. Phase 5A's realistic bound is a lower bound for the resident set.
6. **Samples and settings** (config `l2`, fixed before any L2 run on real data): positions 0, 6 and 11 of the stage 2
   selection; D-q6+q4; real arithmetic; every state of Phase 5A's realistic schedule for the sets; six comparison points
   for CROWN (box CROWN on the 64 comparison pairs, L2 CROWN on three focus pairs: the runner-up, the tightest by Phase
   5A's realistic bound, the tightest truly); the brief's stop conditions as numbers (gap closed 0.20, near-full fraction
   0.90, 600 s per state).
7. **No verification code.** Every bound on a graph is auto_LiRPA's. Weightsift's code is graph construction, sets,
   closed forms of one row at a time (diagnostics), the certificate's assembly (stage 1) and the driver.
8. **Exit status.** A process that ran auto_LiRPA on CUDA fails fast while its native libraries unload at exit (Windows
   status 0xC0000409, which Git Bash shows as 127), after every record is written, and even after `os._exit`: the
   checkpoint's unexplained `exit 127`. The verifier's scripts now end with `TerminateProcess` once everything is written
   and flushed (`crown_oracle.graph.leave`), so that their exit status is the run's (checked: 0 after the same work).

## Results

Full report: `history/2026-10-07-awpmi-phase5a2-report.md`. Raw data: `experiments/phase5a2/probe_l2` (toys, cost),
`experiments/phase5a2/l2` (three samples: validation, every state's sets, CROWN at the comparison points, summary).

- **What upstream can express** (toys, against exact optima). Independent L2 balls are accepted as one root per group
  with one linear map per group: with an exact input, CROWN returns exactly the per-group closed form (groups, rows
  included, stay independent). Concatenating the groups into one weight fails in auto_LiRPA's backward pass. An
  L2-perturbed weight multiplied by an uncertain input (down, whenever a gate or up row is unread) gets unsound bounds in
  upstream's default mode: CROWN, CROWN-IBP and α-CROWN above attained values, because an L2 root's interval bounds are
  its centre. In the box-hull mode IBP with L2 weights is unsound, and so is CROWN on a full graph whose first layer
  takes IBP bounds; the reduced graph is sound there, its product relaxation working on each ball's box hull (±ρ per
  entry): the L2 coupling survives only in the final concretization.
- **Toys.** At toy sizes the sets differ little (the first contender's exact minima: box −2.01, L2 −1.90, both −1.79;
  truth −1.05), CROWN lies within 0.06 of each set's optimum and α-CROWN attains it.
- **Cost at Moonlight's shape** (one expert, synthetic weights, q6 balls): the L2 set's optimum lies in [−3,000, −2,500];
  L2 CROWN gives −95k with a ball per row (12 s to build, 6 s per 8 contenders), −0.38M per 16 rows, −1.07M per 128,
  −4.3M with one ball.
- **Three real samples** (real arithmetic, the reference token against its 64 nearest rows):

  | Sample (gap) | Phase 5A realistic decides from | L2 set (reduced) | Box set (reduced, exact) | L2 set holds flipping weights until | Resident set holds flipping weights until | Phase 5A ideal |
  | --- | --- | --- | --- | --- | --- | --- |
  | prompt 12 step 7 (tie) | 1.634 | 1.634 | 1.634 | 1.633 | 1.617 | 0.946 |
  | prompt 12 step 9 (3.88) | 0.961 | 0.961 | 0.946 | 0.868 | 0.821 | 0.383 |
  | prompt 23 step 4 (11.06) | 1.008 | 1.008 | 0.993 | 0.914 | 0.852 | 0.383 |

  (Fractions of the routed experts' BF16 bytes, metadata included; D-q6+q4 runs to 1.634.)
  - Phase 5A's realistic bound is the L2 set's decoupled lower bound (equal to 10⁻³ at the q6 state; its Hölder half
    never binds there). The L2 set's optimum lies within 16% of it at q6 (sample 0: [−950, −800], against an ideal of
    −3.5 and a truth of +0.11): no verifier on the L2 set can close more than 12–16% of Phase 5A's realistic→ideal
    distance there; with every resident byte, at most 36–38% (45% at any comparison point).
  - With Phase 5A's activation enclosure (the reduced sets), the L2 set decides exactly where Phase 5A does, and the box
    set at most 0.015 of the bytes earlier. Weights consistent with every resident byte flip a pair until 0.82–0.85 of
    the routed bytes on the samples with a margin: no verifier given this metadata certifies them below that.
  - CROWN at the comparison points: box CROWN is about twice its set's optimum at the q6 state (−2,178 against −1,097 on
    sample 0), 20–25% below it at 0.70 and within 3% from 1.336; it never beats Phase 5A's realistic bound before full
    materialization. L2
    CROWN is 34–35× below Phase 5A's realistic bound and about 40× below its own set's attained value at q6 (−32,171;
    −25,291; −69,948), 8–9× below Phase 5A at 0.70, and still negative at 1.008 where Phase 5A and box CROWN decide
    samples 1 and 2; it decides them from 1.336. Its share of the realistic→ideal distance is negative at every point
    before full materialization.
  - Cost: L2 CROWN 84–105 s per state while most rows are unread (six experts, 2,048 roots each, three contenders;
    57–68 s of it building the graphs), under 10 s late in the schedule; box CROWN 0.8 s per state for 64 contenders
    (peak device memory 6.3 GB, against 0.76 GB for L2 CROWN).
- **Correctness.** The wrapper equals Phase 5A's real forward (2·10⁻¹⁶); every box and every L2 ball of every state holds
  the true weights; the boxes only shrink along a schedule (the L2 sets by construction); poisoning the unread rows
  changes no box and no ball; the
  certificate's assembly reproduces Phase 5A's margins (8·10⁻¹¹). No bound exceeded the truth or an attained value of its
  set; every witness lies in its set (largest excess 2·10⁻¹⁶) and at or above its set's lower bound.
- **Stop conditions.** C is met: L2 CROWN never beats Phase 5A's realistic bound, closes a negative share of the
  distance, and decides only from 1.336 of the bytes; no verifier on the L2 set could close more than 12–16% at q6. B is
  met by the pre-registered rule on two samples (the L2 set flips a pair at ≥ 0.90: the tie until 1.633, the
  large-margin sample until 0.914); on the moderate one the L2 set flips until 0.868 and its optimum turns positive by
  0.961. A holds in part: the structure is expressible (split groups), but upstream bounds it soundly where it matters
  only in an interval mode its source marks experimental. D is not met (≤ 105 s per state against 600 s), though that
  is about 30× the phase's 3.25 s-per-token runtime bar. **Stage 2 is not justified; Phase 5A2 closes.**
- **Verdict.** Given Phase 5A's L2 remainder information, auto_LiRPA's CROWN does not make expert AWPMI materially more
  viable. Two findings, distinct:
  1. *Verifier:* auto_LiRPA cannot use the L2 information soundly where it matters without an experimental mode, and
     there its product relaxation on the balls' box hulls makes it 40× looser than the L2 set's own value at q6 and
     9–10× at 0.70 of the routed bytes.
  2. *Uncertainty set:* the sets this metadata defines hold decision-flipping weights until about 0.82–0.91 of the routed
     bytes on the samples with a margin, in real arithmetic. A perfect verifier would save at most 0.12–0.14 of the routed
     bytes over Phase 5A's realistic bound, before the rounding floor (Phase 5A: 8.1% certified ceiling) is considered.

## Rejected

- *Patching auto_LiRPA* (box-hull interval bounds for L2 roots in IBP, axis 0 in `BoundConcat`), and *a custom
  relaxation of D·a* that keeps the L2 coupling (v = X·a with |v_k| ≤ ρ_k·‖a‖₂): forbidden by the brief; the second is
  Phase 5A's realistic bound itself, nothing a verifier adds.
- *One ball per matrix or per page*: sound on the reduced graph in box-hull mode, but a group's radius must hold every
  row's ball at once (‖ρ_G‖₂): 4–45× looser than a ball per row at Moonlight's shape.
- *α-CROWN and the certified (finite-precision) tier with L2 sets*: not run. The brief allows them only after L2 CROWN
  shows substantial improvement in the structural diagnostic; it did not.
- *Stage 2*: not run (stop conditions C and B).
- *The next investigation* (a shared exact expert base and exact per-expert deltas, guide §7): out of this phase's
  scope; the user's decision.
