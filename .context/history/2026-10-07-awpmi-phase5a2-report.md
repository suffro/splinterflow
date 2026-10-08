# AWPMI Phase 5A2 report — auto_LiRPA's CROWN on the expert oracle

Date: 2026-10-07 · Follows: `history/2026-10-04-awpmi-phase5a-report.md` (Phase 5A) ·
Decision: `decisions/0010-phase5a2-crown-oracle.md` ·
Status: **complete; stopped after stage 1.5 (the L2 probe), stage 2 not run. Given Phase 5A's L2 remainder norms,
auto_LiRPA's CROWN does not make expert AWPMI materially more viable.**

## Outcome in brief

Phase 5A2 asks whether a mature bound-propagation verifier, auto_LiRPA's CROWN family, can tighten Phase 5A's bounds on
the routed-expert weights a runtime has not read, enough for AWPMI inside Moonlight's routed experts to pay. The
development checkpoint showed that auto_LiRPA's box (L∞) sets are weaker than the L2 information Phase 5A already uses.
Stage 1.5 gave auto_LiRPA that L2 information, upstream functionality only, and measured both the verifier and the
uncertainty sets themselves.

| Question | Answer |
| --- | --- |
| Can independent L2 remainder balls be given to upstream auto_LiRPA? | As one root per group of rows with one linear map per group, yes; stacked into one weight, no. **Soundly only where the weight's input is exact**: for the down projection after uncertain activations upstream's default mode is unsound, and only its experimental box-hull mode, on the reduced graph, is sound |
| Is L2 CROWN tighter than box CROWN? | **No: 15× looser** at the q6 state (−32,171 against −2,178 on the tie sample) and looser at every point before full materialization |
| Does it beat Phase 5A's realistic bound? | **No: 34–35× below it at q6** (and 40× below the L2 set's own attained value), 8–9× at 0.70 of the routed bytes; it decides the moderate- and large-margin samples only from 1.336 (Phase 5A: 0.961 and 1.008) |
| Does the L2 set itself hold weights that flip the decision? | **Yes**: until 1.633, 0.868 and 0.914 of the routed bytes; with every resident byte (L2 and L∞), until 1.617, 0.821 and 0.852 |
| How close can any verifier get to Phase 5A's ideal? | On the L2 set at most 12–16% of Phase 5A's realistic→ideal distance at q6; with every resident byte at most 36–38% there (45% at any comparison point). L2 CROWN: a negative share (median −3.3 to −4.0) |
| Bytes a perfect verifier could certify from (real arithmetic) | above 0.82–0.85 (all metadata) and 0.87–0.91 (L2 metadata) on the samples with a margin; Phase 5A realistic 0.961–1.008; Phase 5A ideal 0.383 |
| What blocks expert AWPMI? | **The uncertainty sets** the metadata defines (decisive), then the verifier's looseness with L2 sets, the rounding floor (Phase 5A), the cost (~100 s per state for L2 CROWN) |
| Stop conditions | C met; B met on two of three samples by the pre-registered rule; A in part; D not met. **Stage 2 not justified** |

## 1. Question and plan

> Can a mature bound-propagation verifier, auto_LiRPA's CROWN family, tighten Phase 5A's bounds on routed-expert weights
> that have not been read, enough for AWPMI inside Moonlight's routed experts to pay?

Phase 5A (decision 0009) found two obstacles, each decisive alone: the rounding floor of the certified model, and the
bounds on unread parts (Cauchy–Schwarz and Hölder on resident norms), which needed every byte even in real arithmetic.
Phase 5A2 attacks the second with auto_LiRPA, used as published. Its plan (`configs/phase5a2-crown-oracle.yaml`): a
feasibility probe (stage 1), 12 samples across top-2 gaps (stage 2), more only under an escalation rule; gates fixed
before any CROWN run on real data. The development checkpoint (`c128712`) completed stage 1 and one development sample.
The brief of 2026-10-07 inserted stage 1.5 before stage 2: give auto_LiRPA the L2 information Phase 5A holds, with
upstream functionality only, measure the verifier against the sets' own optima, and stop unless it helps materially
(the brief's §17–18, config `l2`).

## 2. Setup

| Item | Value |
| --- | --- |
| Verifier environment | `research/crown_expert_oracle`: Python 3.11.16, torch 2.11.0+cu130, numpy 2.4.6, auto_LiRPA 0.7.2 @ `5a098e8f` (BSD-3-Clause), unmodified, its own `uv.lock` |
| Weightsift environment | unchanged (Python 3.13, torch 2.14.1+cu130, transformers 5.18); cannot import auto_LiRPA (tested); runs `export.py` only |
| Exchange | `export.py`'s artifact: safetensors and a manifest with every file's sha256; the verifier imports no `awpmi` (tested) |
| Target | Moonlight-16B-A3B's last MoE layer (model.layers.26), decode, last position, upstream exact (Phase 5A's setting) |
| Capture | Phase 5A's capture stage, regenerated into `experiments/phase5a2/capture`, equal to Phase 5A run1's digests |
| Strategy | D-q6+q4 (every row's int6 level, then int4 refinements, then the BF16 rows; decision 0003's levels) on Phase 5A's realistic schedule: 82 states from 0.383 to 1.634 of the routed BF16 bytes, metadata included |
| Arithmetic | real (the structural diagnostic: no rounding floor; the brief's §13) |
| Samples (stage 1.5) | positions 0, 6, 11 of the stage 2 selection: prompt 12 step 7 (exact BF16 tie, gap 0), prompt 12 step 9 (gap 3.88), prompt 23 step 4 (gap 11.06) |
| Contenders | the reference token against its 64 nearest rows by the reference's logits. A certificate needs every vocabulary row: these give necessary conditions; a flip on one of them is final |
| Hardware | RTX 4060 Ti 8 GB, Windows 11, 32 GB RAM |

## 3. Stage 1 and the development sample (checkpoint `c128712`)

- **Stage 1 probe: PASS.** Weight perturbations propagate through Linear → SiLU (written g·σ(g)) → multiply → Linear;
  the bounds held against 2¹⁸ box vertices, 200,000 random points and 3⁹ discrete realizations; zero-width boxes give
  the exact value. Upstream notes: a perturbed `F.linear` needs a (zero) bias to take the Gemm path; a parameter's box is
  given as `eps` about its midpoint. At Moonlight's shape full-graph CROWN (backward intermediate bounds) and α-CROWN do
  not fit the card; CROWN-IBP takes 0.4 s per call; the reduced graph (an activation box × the down projections) 0.1–0.2 s.
- **Development sample** (prompt 12 step 7). Certified tier: CROWN ≈ Phase 5A; α-CROWN reaches its box set's exact
  optimum, 0.1–0.2 logits better than Phase 5A. Real tier at the q6 state: the exact optimum of auto_LiRPA's box sets
  (median −865 over the comparison pairs) lies below Phase 5A's realistic bound (median −694): boxes cannot express the
  L2 remainder norms Phase 5A uses, so no verifier on them can do better.

## 4. Method of stage 1.5

### 4.1 The sets

Every row of a routed expert's matrices is in one of Phase 5A's states: a level of its precision refinement, or exact.
What a runtime holds about a row at level l (decision 0003; Phase 5A charges these bytes):

```text
box        |W_rc| ≤ n∞_r,  |W_rc − A_l',rc| ≤ ρ∞_l',r  (l' ≤ l)              stage 1's sets (L∞)
L2 balls   ‖W_r‖₂ ≤ n2_r,  ‖W_r − A_l',r‖₂ ≤ ρ2_l',r   (l' ≤ l)              stage 1.5 (sets.matrix_balls)
resident   both                                                            every resident byte
```

All three are built from a read view that hides every unread row (anti-cheating: poisoning the unread rows changes no
box and no ball, tested and checked on the real samples). On the real metadata ρ2/ρ∞ is about 26 for gate and up rows and
22 for down rows (√n: 45 and 37.5); the true rows lie strictly inside every ball.

Phase 5A's realistic bound uses both halves row by row: the unread part of a row contributes at most
min(ρ2·‖v‖₂, ρ∞·‖v‖₁) for an input bounded by v.

### 4.2 auto_LiRPA with L2 balls

A group of rows becomes one `BoundedParameter` with `PerturbationLpNorm(norm=2, eps=‖ρ_G‖₂)` (the smallest ball about the
group's centre holding every product of its rows' balls). Two assemblies, both graph construction: `stacked` (one weight
from `torch.cat` of the groups) and `split` (one `F.linear` per group, outputs concatenated). auto_LiRPA has two interval
modes for L2 roots: by default a root's interval bounds are its centre (its source: "FIXME This causes confusing lower
bound and upper bound"); with `AUTOLIRPA_L2_DEBUG=1` they are the box hull centre ± ε ("FIXME Experimental code. Need to
change the IBP code also").

### 4.3 The sets' own optima

Every set is a product over rows and x is exact. So g_i = G_i·x ranges over an interval (one row, a convex set), u_i
likewise, independently per neuron, and the activations' set is exactly a box. For a fixed a each down row contributes
its own minimum, a concave function of a. The minimum of Δ·y over the whole set is therefore attained at a vertex of the
activation box (`crown_oracle/rowsets.py`):

```text
min Δ·y = Δ·b + Σ_e w_e · min over vertices a of φ_e(a),    φ_e(a) = Σ_k min_{D_k ∈ S_k} Δ_k·D_k·a
L2 balls:  φ_e(a) = M_e·a − c_e·‖a‖₂,   M_e = C_eᵀΔ,  c_e = Σ_k |Δ_k|·ρ_k
```

- toys: every vertex enumerated (exact);
- Moonlight: the box set's exact minimum (separable); for the L2 set a decoupled lower bound (both terms minimized apart)
  and an attained vertex value (monotone local search);
- witnesses: weights inside a set (each row at its set's maximizer in the needed direction; closed forms for a ball, a
  ball ∩ a further ball, a ball ∩ a box), evaluated by the real forward. A witness below zero proves that the set holds
  weights that flip the pair: no sound verifier given that metadata certifies it. Phase 5A's realistic bound is a lower
  bound for the resident set.

Two variants of each L2 and box set: *reduced* (Phase 5A's activation enclosure × down's sets: the set auto_LiRPA's
reduced graph bounds) and *weights* (gate, up and down rows in their sets).

## 5. What upstream auto_LiRPA can express (toys)

Source: `experiments/phase5a2/probe_l2/probe_l2_A.json`. Every lower bound against the exact minimum of its set and
random points of it; "unsound" means a bound above an attained value.

| Graph | Default (centre) mode | Box-hull mode |
| --- | --- | --- |
| One ball on a whole weight, exact input | CROWN, CROWN-IBP, α-CROWN exact | the same; IBP unsound |
| One ball per matrix: down only (a exact) | exact | exact; IBP unsound |
| One ball per matrix: gate and up | sound | **all unsound** (IBP feeds the first layer's bounds) |
| One ball per matrix: gate, up and down | CROWN, CROWN-IBP, α-CROWN **unsound** | CROWN sound; α-CROWN unsound |
| Pages or rows, stacked into one weight | CROWN fails (backward pass) | CROWN fails |
| Pages or rows, split, exact input | CROWN exact: each group concretized over its own ball | the same; IBP unsound |
| Rows split: gate and up (down exact) | sound; α-CROWN exact, CROWN within 0.04 | the same; IBP unsound |
| Rows split: gate, up and down | **unsound** | sound; α-CROWN exact, CROWN within 0.06 |
| Reduced graph (activation box × split rows of down) | **unsound** | sound; α-CROWN exact, CROWN within 0.013 |

The cause is upstream's: in its default mode an L2 root's interval bounds are its centre, and the product relaxation of
a perturbed weight with an uncertain input (`BoundLinear.bound_backward_with_weight`, McCormick planes) reads them,
treating the weight as fixed. In the box-hull mode that relaxation is valid, but IBP of an L2-perturbed linear layer
takes the hull's lower end as its centre, so every graph whose bounds pass through IBP there is unsound. Concatenated
parameters fail in `BoundConcat`, which reads axis 0 as the batch axis, and get no interval bounds of their own.

Box against L2 on one q6-quantized toy expert (exact minima per set, first contender): box −2.01, L2 −1.90, both −1.79,
truth −1.05. CROWN lies within 0.06 of each set's minimum and α-CROWN attains it. At toy sizes the two halves of the
metadata differ little, and the verifier is nearly exact.

## 6. Cost at Moonlight's shape

Source: `experiments/phase5a2/probe_l2/probe_l2_B.json`. One expert, synthetic weights, q6 balls on every row, the
reduced graph on the L2 set's exact activation box, box-hull mode, eight contenders.

| Groups of down's rows | Roots | Build (s) | Bound (s, 8 contenders) | Lower bound (min) |
| --- | --- | --- | --- | --- |
| one ball | 1 | 0.02 | 0.04 | −4.26M |
| 128-row pages | 16 | 0.07 | 0.04 | −1.07M |
| 16-row pages | 128 | 0.46 | 0.38 | −0.38M |
| rows | 2,048 | 11.9 | 5.8 | −95k |

For every contender the L2 set's own minimum lies between −2,489 and −3,111 (each bracketed between its decoupled lower
bound and an attained value, 15–20% apart), against true values within ±40. With a ball per row CROWN is about 35×
below it: the product relaxation works on each ball's box hull (±ρ2 per entry, ρ2 ≈ 22·ρ∞), so the L2 coupling survives
only in the final concretization. Every bound was sound; peak device memory 0.71 GB.

## 7. Three real samples

Source: `experiments/phase5a2/l2/l2_summary.md` (its JSON beside it, and the records in the same directory).

### 7.1 Where each bound and each set decides

Routed-byte fraction (of the BF16 bytes, metadata included) from which each lower bound decides all 64 pairs at every
later state; the last fraction at which a set holds weights that flip a pair (a witness below zero).

| | Tie (prompt 12 step 7) | Gap 3.88 (prompt 12 step 9) | Gap 11.06 (prompt 23 step 4) |
| --- | --- | --- | --- |
| Phase 5A realistic decides | 1.634 | 0.961 | 1.008 |
| Phase 5A's own cell, whole vocabulary (realistic; ideal) | 1.634; 0.993 | 0.961; 0.383 | 1.008; 0.383 |
| Box set, reduced, exact minimum decides | 1.634 | 0.946 | 0.993 |
| L2 set, reduced: lower bound; attained value positive | 1.634; 1.634 | 0.961; 0.961 | 1.008; 1.008 |
| L2 set, weights: lower bound decides | 1.634 | 0.961 | 1.008 |
| **Box set, weights: flipping weights until** | 1.633 | 1.039 | 1.086 |
| **L2 set, weights: flipping weights until** | 1.633 | **0.868** | **0.914** |
| **Resident set: flipping weights until** | 1.617 | **0.821** | **0.852** |
| L2 CROWN decides (comparison points) | 1.634 | 1.336 | 1.336 |
| Box CROWN decides (comparison points) | 1.634 | 1.008 | 1.008 |

So with Phase 5A's activation enclosure the L2 set decides exactly where Phase 5A does, and the box set at most 0.015 of
the bytes earlier. In weight space, no verifier given the L2 metadata certifies the samples with a margin before 0.87 or
0.91 of the routed bytes, and none given every resident byte before 0.82 or 0.85. Between a set's last witnessed flip
and the state from which its lower bound decides, the true crossing is not located (the L2 set: 0.883–0.961 on sample 1,
0.930–1.008 on sample 2).

### 7.2 At the comparison points

Δ·y lower bounds and set values, each the minimum over the focus contenders (the runner-up, the tightest by Phase 5A's
realistic bound, the tightest truly); `[a, b]`: a lower bound and an attained value of the set's minimum; the last column
is L2 CROWN's share of Phase 5A's realistic→ideal distance (median over the focus contenders).

| Sample | Bytes | Truth | 5A realistic | Box CROWN | Box set, reduced (exact) | Box set, weights (exact) | L2 CROWN | L2 set, reduced | L2 set, weights | Resident witness | 5A ideal | Share closed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tie | 0.383 | 0.108 | −950.268 | −2178.512 | −1097.430 | −1862.597 | −32,171 | [−950.268, −803.052] | [−950.264, −799.972] | −576.722 | −3.474 | −35.9 |
| tie | 0.696 | 0.108 | −30.588 | −33.682 | −27.748 | −51.893 | −258.071 | [−30.711, −28.509] | [−30.711, −20.952] | −14.040 | −0.157 | −8.1 |
| tie | 1.008 | 0.108 | −5.576 | −5.689 | −5.265 | −7.512 | −19.969 | [−5.647, −5.446] | [−5.647, −3.968] | −3.127 | 0.023 | −2.4 |
| tie | 1.336 | 0.108 | −1.804 | −1.806 | −1.760 | −2.450 | −3.051 | [−1.843, −1.820] | [−1.843, −1.306] | −1.035 | 0.076 | −0.59 |
| tie | 1.524 | 0.108 | −0.634 | −0.635 | −0.625 | −0.914 | −0.887 | [−0.664, −0.660] | [−0.664, −0.480] | −0.338 | 0.095 | −0.28 |
| tie | 1.634 | 0.108 | 0.107 | 0.107 | 0.107 | 0.108 | 0.107 | [0.107, 0.107] | [0.108, 0.108] | 0.108 | 0.107 | – |
| 3.88 | 0.383 | 8.944 | −738.560 | −1710.882 | −860.962 | −1478.584 | −25,291 | [−738.560, −628.956] | [−738.557, −626.459] | −451.831 | 6.538 | −34.8 |
| 3.88 | 0.696 | 8.944 | −22.349 | −24.890 | −19.960 | −39.014 | −208.803 | [−22.770, −20.883] | [−22.770, −14.687] | −9.012 | 8.730 | −7.3 |
| 3.88 | 1.008 | 8.944 | 2.379 | 2.282 | 2.671 | −1.770 | −9.109 | [2.244, 2.391] | [2.245, 4.843] | 6.081 | 8.873 | −1.9 |
| 3.88 | 1.336 | 8.944 | 7.529 | 7.526 | 7.564 | 7.028 | 6.506 | [7.491, 7.507] | [7.491, 7.888] | 8.101 | 8.919 | −0.74 |
| 3.88 | 1.524 | 8.944 | 8.445 | 8.446 | 8.451 | 8.301 | 8.276 | [8.423, 8.425] | [8.423, 8.562] | 8.656 | 8.933 | −0.35 |
| 3.88 | 1.634 | 8.944 | 8.944 | 8.944 | 8.944 | 8.944 | 8.944 | [8.944, 8.944] | [8.944, 8.944] | 8.944 | 8.944 | – |
| 11.06 | 0.383 | 20.908 | −2010.561 | −4735.940 | −2391.932 | −3951.565 | −69,948 | [−2010.561, −1702.669] | [−2010.554, −1694.608] | −1240.079 | 11.155 | −32.5 |
| 11.06 | 0.696 | 20.908 | −66.384 | −74.096 | −60.774 | −111.272 | −561.983 | [−66.989, −62.114] | [−66.989, −46.609] | −31.622 | 19.968 | −5.4 |
| 11.06 | 1.008 | 20.908 | 0.834 | 0.569 | 1.383 | −8.572 | −24.519 | [0.558, 0.941] | [0.559, 6.383] | 8.966 | 20.598 | −1.2 |
| 11.06 | 1.336 | 20.908 | 15.113 | 15.107 | 15.167 | 12.407 | 13.468 | [15.042, 15.071] | [15.043, 16.652] | 17.276 | 20.818 | −0.29 |
| 11.06 | 1.524 | 20.908 | 19.291 | 19.294 | 19.298 | 18.514 | 19.175 | [19.269, 19.270] | [19.269, 19.687] | 19.855 | 20.882 | −0.07 |
| 11.06 | 1.634 | 20.908 | 20.908 | 20.908 | 20.908 | 20.908 | 20.908 | [20.908, 20.908] | [20.908, 20.908] | 20.908 | 20.908 | – |

What the table separates:

- **Verifier against its own set.** Box CROWN lies about 2× below its set's exact minimum at q6 and within 3% of it from
  1.336 on. L2 CROWN lies 40–41× below its set's attained value at q6 and 9–10× at 0.70; at 1.008 it is still −20, −9
  and −25 where its set's attained values are −5.4, +2.4 and +0.9.
- **Set against Phase 5A.** With Phase 5A's activation enclosure, the box set's exact minimum is a little above Phase 5A's
  bound from 0.70 on (below it at q6), and the L2 set's minimum lies in its bracket, from just below Phase 5A's bound to
  at most 16% above it. Neither changes where the pairs are decided by more than one schedule step.
- **What every resident byte allows.** The resident witnesses lie between Phase 5A's bound and the ideal, about 60% of
  the way from the ideal to Phase 5A's bound at q6: real headroom for a perfect verifier, but the set still flips the
  pairs at q6, at 0.70, and (tie) until 1.617.

## 8. Correctness

| Check | Result |
| --- | --- |
| The verification graph at the true weights against Phase 5A's real forward | 1.9·10⁻¹⁶ (relative) |
| Boxes and L2 balls of every state hold the true weights | all, three samples, both strategies, both tiers |
| Sets only shrink along a schedule | boxes: checked, all; L2 sets: by construction (each level read adds a ball), though a refined row's current ball alone is not always inside its earlier one (the single ball L2 CROWN receives; `down_balls_not_nested`) |
| Poisoned unread rows (random, NaN) change a box or a ball | never |
| The certificate's assembly against Phase 5A's margins | 7.9·10⁻¹¹ (relative) |
| A lower bound above the truth or above an attained value of its set | none (246 states; 18 CROWN points) |
| A witness outside its set, or below its set's lower bound | none (largest constraint excess 2.2·10⁻¹⁶) |
| The reference reproduced bitwise at export | all three samples |
| Determinism | The final runs (probe, validation, CROWN points, every state's sets) ran on research tree `49f842c5…`; the committed tree (`a651447c…`) differs from it only in `report.py`, which reads the records and writes the summary (checked: the committed tree with the run-time `report.py` hashes to `49f842c5…`). Their records equal the first runs' (`experiments/phase5a2/l2/run1`, earlier trees): validation and the 246 states' sets have identical digests; the CROWN part's 18 values equal the first run's log line for line (§12); the probe equals its first final run apart from timings |
| Tests | verifier environment: 54 (26 new: the L2 sets, their poisoning, the split balls' exactness and independence, upstream's unsound default mode and failing concatenation pinned, the closed forms and witnesses against brute force); Weightsift: 555, unchanged behaviour |

## 9. Answers

1. **Can independent Phase 5A-style L2 remainder constraints be represented using only upstream auto_LiRPA?** They can
   be written down (one root per group, one linear map per group), and with an exact input auto_LiRPA bounds them exactly
   and independently. Where they matter, on the down projection after uncertain activations, upstream's default mode
   gives unsound bounds; only its experimental box-hull mode, on the reduced graph, is sound. Not cleanly.
2. **Does L2-aware CROWN produce materially tighter bounds than box CROWN?** No. It is 15× looser at the q6 state and
   looser at every comparison point before full materialization: its product relaxation works on the balls' box hulls,
   whose half-widths ρ2 are about 22 times the boxes' ρ∞.
3. **Does it beat Phase 5A's realistic bound?** No, nowhere before full materialization: 34–35× below it at q6, 8–9× at
   0.70, still negative at 1.008 where Phase 5A decides samples 1 and 2.
4. **Does the L2 uncertainty set itself still contain weights that can flip the decision?** Yes: until 1.633, 0.868 and
   0.914 of the routed bytes; with the L∞ metadata too, until 1.617, 0.821 and 0.852. Phase 5A's realistic bound is
   already the L2 set's decoupled lower bound, and the set's minimum lies within 16% of it at q6.
5. **How close does L2 CROWN get to the Phase 5A ideal diagnostic?** It closes a negative share of the realistic→ideal
   distance: −33 to −36 at q6, −5 to −8 at 0.70, −1.2 to −2.4 at 1.008 (median over the points before full
   materialization −3.3 to −4.0 per sample). The sets bound what any verifier could close: 12–16% at q6 on the L2 set,
   36–38% with every resident byte (at most 45% at any comparison point).
6. **At what routed-byte fraction does the structural certificate become positive?** L2 CROWN: 1.336 on the samples with
   a margin, 1.634 on the tie (comparison points). Box CROWN and Phase 5A realistic: about 0.96–1.01 and 1.634. A perfect
   verifier: at best from 0.84–0.87 (all metadata) or 0.88–0.93 (L2 metadata) on the samples with a margin.
7. **Is the remaining problem verifier looseness, uncertainty metadata, rounding floor or verification cost?** All four
   exist; the metadata decides. See §10.

## 10. Where the difficulty is

| Aspect | Finding |
| --- | --- |
| Box uncertainty (L∞ metadata) | The box set (weights) holds flipping weights until 1.04–1.09 on the samples with a margin: weaker than Phase 5A's bound. With Phase 5A's activation enclosure (reduced) it decides 0.015 of the bytes earlier than Phase 5A |
| L2 uncertainty (L2 metadata) | Phase 5A's realistic bound is the L2 set's decoupled bound; the set's minimum is within 16% of it at q6, and it holds flipping weights until 0.87–0.91 on the samples with a margin |
| Verifier tightness | Box CROWN is about 2× below its set's minimum at q6, 20–25% at 0.70 and within 3% from 1.336; α-CROWN reached its box set's minimum on the development sample. L2 CROWN is 40–41× below its set's attained value at q6 and 9–10× at 0.70 (the box hull); still negative at 1.008 where its set's values are already positive on two samples |
| Uncertainty-set tightness | Decisive: weights consistent with every resident byte flip a pair until 0.82–0.85 of the routed bytes on the samples with a margin, in real arithmetic. No verifier can certify there |
| Finite-precision floor | Not revisited (stage 1.5 is the structural diagnostic): Phase 5A's 3.2 logits of named error terms on the tightest pair and its 8.1% certified ceiling remain, on top of everything above |
| Verification cost | L2 CROWN 84–105 s per state while most rows are unread (six experts, 2,048 roots each, three contenders: 57–68 s building the graphs, 24–33 s bounding; peak device memory 0.76 GB, host 2.0 GB), 25–29 s at 1.008, under 10 s later. Box CROWN 0.8 s per state for 64 contenders, peak device memory 6.3 GB. The phase's runtime bar was 3.25 s per token |

Even a perfect verifier given every resident byte would save at most 0.12–0.14 of the routed bytes over Phase 5A's
realistic bound on these samples, in real arithmetic. Under the certified rounding model Phase 5A certifies 8.1% of
tokens even with every byte read.

## 11. Stop conditions and recommendation

Config `l2`, fixed before any L2 run on real data:

- **STOP C (weak improvement): met.** L2 CROWN never beats Phase 5A's realistic bound, closes a negative share of the
  distance, and decides only from 1.336; no verifier on the L2 set could close more than 12–16% at q6 (threshold 20%).
- **STOP B (uncertainty set): met on two of three samples** by the pre-registered rule (the L2 set flips a pair at ≥ 0.90
  of the routed bytes): the tie until 1.633, the large-margin sample until 0.914. On the moderate one the L2 set flips
  until 0.868 and its minimum is positive from 0.961.
- **STOP A (representation): in part.** Expressible as split groups; sound where it matters only in an interval mode
  upstream marks experimental.
- **STOP D (cost): not met** (at most 105 s per state against 600 s), though about 30× the phase's runtime bar.
- **Proceed to stage 2: no** (§18 requires L2 CROWN materially above Phase 5A's realistic bound).

**Recommendation.** Close Phase 5A2. Do not run stage 2, α-CROWN at scale, or the certified tier with L2 sets: no
verifier can certify what the metadata's own sets do not decide, and these sets stay undecided until ~0.82–0.91 of the
routed bytes in real arithmetic. Making expert AWPMI pay would need resident information that pins the unread weights'
contributions more tightly than norms, and a rounding model that lifts Phase 5A's floor.

The next milestone is the user's decision; this phase starts none. The guide's exact structural reuse (§7: a shared
exact expert base and exact per-expert deltas) is the likely separate investigation; the Phase 4B and 5A reports'
engineering path (a native runtime and a host-RAM expert tier) remains open.

## 12. Limitations

- **Three samples, one strategy, one tier.** D-q6+q4 (the best decomposition under ideal bounds in Phase 5A), real
  arithmetic only, the 64 nearest contenders. Spatial strategies (A, B, C) bound unread rows by their own norms alone, a
  weaker summary than a level's remainder; nothing here suggests they would fare better.
- **The sets' optima are bracketed on Moonlight.** A negative witness is exact evidence; a positive one is not. Where the
  bracket straddles zero the crossing lies between two states.
- **The resident set's lower bound is Phase 5A's realistic bound.** A tighter one was not computed: the resident witness
  is the binding evidence.
- **auto_LiRPA 0.7.2 at one commit.** A later upstream that gave L2 roots their box hull in IBP would make more
  configurations sound, but not remove the box-hull relaxation's looseness.
- **The finite-precision tier** was not revisited with L2 sets; it can only be harder than the real tier.
- **Witness construction** moves a row toward a point of its set near the true row when its closed form leaves a further
  constraint: valid because a witness only has to lie in its set (checked for every row); rows moved by rounding alone
  are counted too (`witness_pulled_rows`).
- **One development slip.** A test launch of the CROWN part with an invalid sample index truncated the first run's CROWN
  records; that part was rerun on the final tree, and its values equal the first run's log line for line (§8).

## 13. Reproduction

```bash
V="python -m uv run --project research/crown_expert_oracle python research/crown_expert_oracle"
export PYTHONHASHSEED=1 PYTHONIOENCODING=utf-8
$V/probe_l2.py --output experiments/phase5a2/probe_l2                                  # toys and cost (~2 min)
python -m uv run python research/crown_expert_oracle/export.py --output experiments/phase5a2/l2 --select 0,6,11   # Weightsift environment (~4 min)
$V/run.py --run experiments/phase5a2/l2 --part validate                               # ~4 min
$V/run.py --run experiments/phase5a2/l2 --part l2crown                                # ~13 min
$V/run.py --run experiments/phase5a2/l2 --part l2                                     # ~78 min
$V/report.py experiments/phase5a2/l2 --l2
```

The capture (`experiments/phase5a2/capture/capture.safetensors`) and the artifact's tensors are regenerated, not
committed (their sha256 are in the records and the manifest). A run is reproduced if `l2_summary.json`'s digests match
(timings and costs excluded):

| Digest | Value |
| --- | --- |
| validation | `fa667094905ce47fac18fb13d77486d40d852680c79d5f6f270b0eb00bac3dcd` |
| every state's sets (`l2`) | `b983d43a0e45a3b4cb7dd411413effb7f23c843d2e293c3a28d994ebcf66f6c3` |
| CROWN points (`l2crown`) | `5b5cf35d0ac8dcf673a8b67ae169ec1f1475bfd3bc737ab79ebc045127f153b3` |
| artifact manifest | `db79108440b15982204dbaea6e230f7043f156baf306090e5799a3fa6e110759` |
