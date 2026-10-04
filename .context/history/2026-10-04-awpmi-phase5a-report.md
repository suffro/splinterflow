# AWPMI Phase 5A report — AWPMI inside routed experts: an oracle study

Date: 2026-10-04 · Follows: `history/2026-10-03-awpmi-phase4b-report.md` (Phase 4B) ·
Decision: `decisions/0009-phase5a-expert-oracle.md` ·
Status: **complete. Gate: FAIL. Under the certified rounding model a token of Moonlight's last MoE layer cannot be
certified with any routed-expert byte unread; a physical expert-AWPMI runtime (Phase 5B) is not justified.**

## Outcome in brief

Phase 5A asks whether, once the router has chosen Moonlight's experts, Shardraw must read all of them to keep the final
token. It answers with an oracle: for the last MoE layer at decode, upstream exact, it reads the six routed experts
progressively in eight decompositions, bounds every intermediate of the experts call, the MoE block, the residual and
the final norm through the reference's own operations, and certifies the token against all 163,840 vocabulary rows
with Phase 2's pairwise certificate, generalized to a mixture of experts.

| Over 768 decode tokens (48 prompts × 16 steps) | Result |
| --- | --- |
| Tokens certifiable with every routed byte read (certified model) | **8.1%** (62) |
| Tokens certified with any routed byte unread (realistic bounds, any strategy) | **0** |
| Best strategy, gate configuration (certified tier, realistic bounds and ordering) | **1.003** of the routed bytes on average (A, B, C: every byte plus metadata); D 1.50–1.62. **FAIL** |
| Ideal bounds and ordering (diagnostic), certified tier | 0.65–0.90 on the 62 certifiable tokens; at best 0.99 over all tokens (C) |
| Real arithmetic (no rounding floor), realistic bounds (diagnostic, 96 tokens) | A, B, C 1.003; D 1.27–1.46 |
| Real arithmetic, ideal bounds (diagnostic, 96 tokens) | D-q6+q4 **0.47**, D-q8 0.56, A and B 0.71, C 0.85 |
| Correctness | 0 enclosure violations; 0 wrong certified tokens; the reference recomputed bitwise on 768 of 768 tokens; the capture equal to Phase 4B's reference on 144 of 144 steps |
| Reproducible | run1 and run2 (`PYTHONHASHSEED` 1 and 2, same source tree `a414afff…`): identical digests |

Two obstacles block expert AWPMI, and removing either alone is not enough:

1. **The rounding floor.** With every routed weight read, the certificate's named error terms on the tightest pair
   still sum to 3.2 logits on average: the final norm 1.37, y's rounding 0.49, m's 0.36, the experts' outputs' 0.35,
   R's 0.28, the down projection's accumulation 0.26, the LM head's 0.07. On top of them, the experts' internal
   roundings (g, s and a, each faithful) widen a inside the certificate's main term. The median top-2 gap is 1.1 logits;
   no token with a gap below 2.75 logits certifies.
2. **The bounds on unread parts.** A neuron never read is bounded by Cauchy–Schwarz on its rows' norms, and a coarse
   level's remainder the same way: both are far wider than the true contributions. With realistic bounds every
   strategy certifies only once everything is read, even in real arithmetic. With the true contributions
   as bounds (ideal), precision refinement would certify at 0.47 of the bytes, but only if the floor were also gone.

## 1. Question

> Once the router has selected an expert, does Shardraw need to materialize the whole expert to keep the final discrete
> decision? What fraction of the routed expert bytes must be read before AWPMI can certify the token of the fully
> materialized reference?

Phase 5A is an oracle: it holds every weight and simulates what a runtime would have read. It claims no physical I/O
saving; it predicts what a runtime could save, and whether one is worth building.

## 2. Setup

| Item | Value |
| --- | --- |
| Model, reference | Moonlight-16B-A3B @ `476b36a4…`, the Phase 4B BF16 reference (grouped_mm experts, SDPA), torch 2.14.1+cu130, transformers 5.18.0, deterministic numerics |
| Adaptive region | the routed experts of the last MoE layer (model.layers.26) at the last position, decode only: 6 experts × 17.3 MB = 103.8 MB per token |
| Upstream | exact (Mode A): every earlier layer, this layer's attention, its router and its shared experts are the reference's |
| Downstream | the reference's operations, bounded: the combine, the shared experts' addition, the residual, the final RMSNorm, the LM head (163,840 × 2,048, exact weights) |
| Samples | 48 wikitext-2 prompts (16–1,024 tokens; the first 16 are Phase 4B's), 16 greedy decode steps each: 768 decode tokens |
| Capture | Moonlight on the Phase 4B streamed path (no cache, the 256 MiB call budget, the 6 GB cap); files verified against the Hub's sha256 |
| Certificate | Phase 2's pairwise certificate with a decomposed bound for a mixture of experts, against all 163,840 rows |
| Arithmetic tiers | certified (faithful, u = 2⁻²²); what-ifs, not certified: binary32 accumulator (u = 2⁻²⁴), RN-even in elementwise kernels, RN-even everywhere; diagnostic: real arithmetic |
| Strategies | A, B1, B16, B128, C, D-q8, D-q6+q4, D-q4+q4 |
| Cells | certified tier, every token whose ceiling holds: realistic bounds and ordering (the gate's), realistic bounds with ideal ordering, ideal bounds and ordering; rn_even: realistic (steps 1, 5, 9, 13); real: realistic and ideal (steps 1 and 9: 96 tokens) |
| Budgets | multiples of 1/64 of the routed bytes (1.6 MB) |
| Hardware | RTX 4060 Ti 8 GB, Windows 11, 32 GB RAM |
| Device memory | peak 5.6 GB. The first development runs reached 8.03 GB and ran several times slower: under WDDM the driver pages device memory to the host instead of failing. The LM head is therefore never copied to float32 (centres in chunks of 4,096 rows), and no float64 copy of the layer is cached |

## 3. Method

### 3.1 The experts as transformers runs them

transformers 5.18's `grouped_mm_experts_forward`, for one decode token and its six experts e (top-k order):

| Step | Operation | Rounding |
| --- | --- | --- |
| gate, up | `_grouped_linear(x, gate_up_proj)`: `torch._grouped_mm`, on this GPU one cuBLAS GEMM per expert, float32 accumulation over 2,048 | BF16 epilogue |
| s | `act_fn(gate)`: SiLU in float32 | BF16 |
| a | `s * up` | BF16 |
| o | `_grouped_linear(a, down_proj)`: accumulation over 1,408 | BF16 epilogue |
| z | `o * routing_weight`: BF16 × float32 → float32 | float32 |
| R | `view(T, K, H).sum(dim=1)` in float32, then `.to(bfloat16)` | float32 sum, then BF16 |
| m | `R + shared_experts(x)` (DeepseekV3MoE) | BF16 |
| y | `residual + m` (the decoder layer) | BF16 |
| n, h | DeepseekV3RMSNorm (LlamaRMSNorm's operations): n = rnd(y·q), h = rnd(g·n) | BF16, twice |
| ℓ | the LM head: accumulation over 2,048 | BF16 epilogue |

The routing weights are the router's float32 values (sigmoid scores, renormalized, × 2.446); they multiply the
experts' BF16 outputs before the float32 sum. The oracle recomputes the call with transformers' own functions on the
layer's whole weights (`experts_call_reference`), and every later step with the reference's operations; on every
token this equals the capture bitwise. It is the exact fallback: with every routed byte read, the reference is
reproduced.

### 3.2 Bounds through the MoE block

Every intermediate is enclosed by Phase 2's operators (decision 0005), each modelling the reference operation:
`linear` (a known part, and an unknown part bounded by its rows' norms times the input's norms), `silu`, `multiply`,
`residual_add`, `rms_norm`; and three new ones: `reduce_sum` (the combine), `multiply` into float32 (the routing
weights), `linear` on a batch of weights. In the certified tier every rounding is faithful (either grid neighbour) and
every accumulation errs by at most γ_{n+2}(2⁻²²) of its absolute mass (decision 0001). Each enclosure is checked against
the reference's value on every evaluation.

### 3.3 The certificate

The per-logit interval pass of Phases 1–2 eliminates no vocabulary row on these enclosures: h is known only to about an
ulp per element under the faithful model, and each logit's interval adds that up without sign over 2,048 coordinates,
separately for every row. The pairwise certificate (decision 0005) bounds the difference of two logits directly and is
scale-free through the final norm. Its decomposed bound is generalized to the mixture:

```text
y = r + S + Σ_e w_e·(K_e + X_e)·a_e + η
```

K_e is the known part of expert e's down projection, X_e the rest, η the named errors (the down projection's
accumulation and o's rounding, each × w_e; the routing products' float32 rounding; the combine's float32 sum; R's and
m's roundings). For a candidate w and contender j, with Δ = (W_w − W_j)⊙g and M_e = K_eᵀ·Δ:

```text
Δ·y ≥ Δ·(r + S) + Σ_e w_e·[Σ_i min(M_e,i·a⁻, M_e,i·a⁺) − min(|Δ|·ρ_e, ‖Δ‖₂·Γ_e)] − |Δ|·N − |Δ|·(y's rounding)
```

The norm's roundings and the LM head's accumulation are then subtracted as in decision 0005, and a pair is certified
when the faithful logits are two grid spacings apart. The certificate checks every one of the 163,840 rows: the 64
nearest by the reference's logits first (a filter: one failure rejects), then all of them, with M in binary32 and a
rank-one error bound. The candidate is the row with the largest centre logit, as a runtime would choose it.

### 3.4 Strategies, orderings, search

| Strategy | Unit | First reads | Unread parts bounded by |
| --- | --- | --- | --- |
| A | neuron page: gate row and up row (8 KiB) | the whole down projection (a third of the expert) | gate and up row norms |
| B1, B16, B128 | neuron page; down page of 1, 16 or 128 output rows (2.75 KiB to 352 KiB) | — | row norms of gate, up and down |
| C | neuron-major page: gate row, up row, down column (12 KiB) | — | row norms; down column norms |
| D-q8, D-q6+q4, D-q4+q4 | a row's next level (a neuron's gate and up rows together; a down row) | every row's coarse level | each level's remainder norms |

D's levels are decision 0003's per-row int-b decompositions; a row ends at its original BF16 bytes. Orderings are fixed
per cell from its first state. Realistic: the largest bound contribution per byte to the tightest pair, from what has
been read. Ideal (diagnostic): the largest true contribution per byte to the reference's top-2 difference. A cell looks
for the first budget at which the certificate holds; with realistic bounds, reading more only narrows every enclosure,
so certification is monotone and a bisection finds that budget exactly.

### 3.5 Bytes and the physical model

A unit's bytes are what the checkpoint (or a level, or a neuron-major copy) stores for it. The routed experts' metadata
(row and column norms, the levels' remainder norms) is charged to every token: 0.003 of the routed bytes (0.005–0.007
for D). A token a strategy never certifies costs its whole schedule: every BF16 byte for A, B and C (1.003 with the
metadata), the levels and every BF16 byte for D (1.51 for D-q8, 1.63 for D-q6+q4, 1.51 for D-q4+q4). The physical model
counts the 4 KiB blocks and extents of the reads at the experts' real offsets in the published file; for C also on a
neuron-major copy (down transposed, one 12 KiB record per neuron), for D also its level files.

## 4. Correctness

Source: `experiments/phase5a/oracle-run1/summary.md` (and `oracle-run2`). Raw data: `samples.{0,1}.jsonl.gz`,
`real_cells.0.jsonl.gz`, `capture.jsonl.gz`.

| Criterion | Result |
| --- | --- |
| Capture equal to Phase 4B's reference records (all digests, steps of Phase 4B's prompts) | PASS: 144 of 144 |
| Target layer's weights equal the Phase 4B reference's digests (every expert's gate_up and down) | PASS: 128 of 128 |
| The reference recomputed from the captured inputs: R, m, y, h, logits (sha256), token, shared experts | PASS: 768 of 768 each |
| Enclosure violations, certified tier (every intermediate, ceilings and realistic cells) | PASS: 0 |
| Certified tokens differing from the reference's | PASS: 0 |
| What-if tiers (binary32 accumulator, RN-even): enclosure violations, wrong would-certify tokens | 0 and 0: both models held on every token |
| Real tier: its winner differs from the BF16 token | on 10 of 768 tokens, and in the 32 cells of the two of them in the cells' subset (the summary's "42"): all exact BF16 ties (gap 0), which the reference breaks by the lowest index while the real computation has a strict winner. A property of the diagnostic, not an error |
| Reproducible: run1 and run2 (`PYTHONHASHSEED` 1 and 2, the same source tree `a414afff…`) | PASS: all five digests identical (§11); the two summaries are identical apart from the runs' names |

**Tests.** 551 pass on CPU and CUDA, 41 of them new:

- `reduce_sum` and the routing-weight `multiply` against the reference's kernels on every grid point of small
  enclosures, against adversarial realizations in any summation order and against exact `Fraction` sums; the guard
  (no reduction error) caught;
- `linear` on a batch equal to each weight's own; `DeepseekV3RMSNorm` enclosed like `LlamaRMSNorm` (Moonlight's 2,048
  dimensions included);
- on a test-size DeepSeek-V3 model: the recomputation equal to the model's own forward; every enclosure containing the
  reference in random states of every strategy and tier; the mixture's decomposed bound never above the reference's
  Δ·y; a certificate only for the reference's token, and only with its margin; monotonicity in what is read; byte
  accounting of every strategy; the realistic ordering unchanged by permuting values that keep the metadata (the ideal
  one changes); refinement steps in order; the binary32 projection's error bound; a relaxed mixture never stronger;
- sabotage: dropping the unread parts' bound, or the experts' roundings from the mixture, is caught;
- layering: nothing outside `awpmi.oracle` imports it, and it imports no storage or runtime module.

**Phase 2 unchanged.** After the operators and the pairwise certificate were extended, the Phase 2 benchmark on its
first 20 prompts gives records and validation identical to `suffix-run1` (740 records each).

## 5. Results

### 5.1 The ceilings: what rounding alone leaves

Every routed weight read; coverage is the share of tokens whose certificate holds.

| Arithmetic tier | Coverage | Tightest pair's named error terms, mean (logits) | Margin, median |
| --- | --- | --- | --- |
| **certified** (faithful, u = 2⁻²²) | **8.1%** | 3.18 | −3.43 |
| binary32 accumulator (u = 2⁻²⁴), what-if | 16.5% | 2.55 | −2.34 |
| RN-even in elementwise kernels, what-if | 17.8% | 1.84 | −2.10 |
| RN-even everywhere, what-if | 24.7% | 1.62 | −1.30 |
| real arithmetic, diagnostic | 100% | 0 | +2.42 |

Coverage by the reference's top-2 gap:

| Gap (logits) | Tokens | certified | u = 2⁻²⁴ | RN-even elementwise | RN-even | real |
| --- | --- | --- | --- | --- | --- | --- |
| < 0.5 | 207 | 0% | 0% | 0% | 0% | 100% |
| 0.5–1 | 137 | 0% | 0% | 0% | 0% | 100% |
| 1–2 | 166 | 0% | 0% | 0.6% | 3.6% | 100% |
| 2–4 | 133 | 4.5% | 18.8% | 26.3% | 51.9% | 100% |
| 4–8 | 99 | 44.4% | 76.8% | 76.8% | 89.9% | 100% |
| ≥ 8 | 26 | 46.2% | 100% | 96.2% | 100% | 100% |

The certificate's named error terms on the tightest pair, certified tier, every weight read, mean over tokens (logits,
converted with the norm's lower scale bound):

| Term | Logits |
| --- | --- |
| the final norm (its two BF16 roundings and its scale's error) | 1.37 |
| y's rounding | 0.49 |
| m's rounding (routed + shared) | 0.36 |
| the experts' outputs' rounding (o, × routing weights) | 0.35 |
| R's rounding (the combine's BF16 conversion) | 0.28 |
| the down projection's accumulation | 0.26 |
| the LM head's accumulation | 0.07 |
| the routing products, the combine's float32 sum, r + S | < 0.001 |
| **Total** (mean; median 3.13, p90 4.23, max 6.01) | **3.18** |

Two losses come on top of these terms: the experts' internal roundings (g, s and a, each faithful) widen a, which the
certificate meets through M = Kᵀ·Δ inside its main term; and the faithful logits must be two grid spacings apart.
Together they put the median certified margin at −3.43 logits, against a median top-2 gap of 1.13 (p90 5.4; 27% of
tokens have a gap below 0.5). Of the 26 tokens with a gap of 8 logits or more, 14 fail, all on the 64 nearest rows:
there the main term's lower bound is 4.4–6.0 logits where the true difference is at least 8, and the named terms take
another 3.9–5.2.

### 5.2 Strategies under the gate's configuration

Certified tier, realistic bounds and ordering. A token whose ceiling fails costs the strategy's whole schedule.

| Strategy | Coverage | Mean | Median | p90 | p95 | When certified | Storage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 8.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | 1.00× |
| B1, B16, B128 | 8.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | 1.00× |
| C | 8.1% | 1.003 | 1.003 | 1.003 | 1.003 | 1.003 | 1.33× |
| D-q8 | 8.1% | 1.501 | 1.506 | 1.506 | 1.506 | 1.448 | 1.50× |
| D-q6+q4 | 8.1% | 1.624 | 1.634 | 1.634 | 1.634 | 1.509 | 1.63× |
| D-q4+q4 | 8.1% | 1.509 | 1.509 | 1.509 | 1.509 | 1.501 | 1.51× |

No strategy certifies any token with fewer bytes than the BF16 experts: A, B and C certify the 62 certifiable tokens
only with every byte read; D certifies them only after its levels and 92–98% of the BF16 rows (1.24–1.63). With the
ideal ordering and realistic bounds the results are the same (A, B, C 1.003; D 1.50–1.63): the order of reads does not
matter while the bounds on what is unread are this wide. The same holds under RN-even (192 tokens, steps 1, 5, 9 and
13): coverage 27%, A, B and C at 1.003, D at 1.42–1.50 when certified.

### 5.3 What the diagnostics show

**Ideal bounds (the unread parts' true contributions) and ideal ordering, certified tier** — on the 62 certifiable
tokens:

| Strategy | When certified: mean (min) | Neurons read | Down read |
| --- | --- | --- | --- |
| A | 0.897 (0.695) | 84% | 100% (first) |
| B1 / B16 / B128 | 0.887 / 0.897 / 0.897 | 84% | 97–100% of the rows |
| C | 0.879 (0.518) | 88% (with their columns) | 88% of the columns |
| D-q8 | 0.678 (0.506) | 14% exact | every row at int8, 25% exact |
| D-q6+q4 | 0.648 (0.383) | 12% exact | 19% exact |
| D-q4+q4 | 0.879 (0.368) | 36% exact | 53% exact |

Over all 768 tokens this is still 0.99 for A, B and C and above 1.4 for D, because 92% of the tokens fall back.

**Real arithmetic (no rounding floor)**, 96 tokens:

| Strategy | Realistic bounds: mean (min) | Ideal bounds: mean | median | p90 | p95 |
| --- | --- | --- | --- | --- | --- |
| A | 1.003 | 0.708 | 0.695 | 0.898 | 0.930 |
| B1 / B16 / B128 | 1.003 | 0.707 / 0.706 / 0.706 | 0.706 | 0.893 | 0.924 |
| C | 1.003 | 0.853 | 0.940 | 1.003 | 1.003 |
| D-q8 | 1.289 (0.850) | 0.559 | 0.506 | 0.678 | 0.912 |
| D-q6+q4 | 1.266 (0.774) | **0.471** | 0.399 | 0.633 | 0.883 |
| D-q4+q4 | 1.464 (1.211) | 0.534 | 0.399 | 1.118 | 1.461 |

Without the floor, realistic bounds still need every byte for A, B and C; D certifies below the BF16 bytes on 21 of 96
tokens (D-q6+q4) and 11 (D-q8), but averages above it. Only with both obstacles removed does precision refinement
reach 0.47. D-q6+q4 then reads every row's 6-bit level, the 4-bit refinement of 14% of the gate and up rows and 25% of
the down rows, and the BF16 bytes of 3% and 6% of them.

### 5.4 Bytes against margin, prompt length and expert

- **Margin:** coverage is zero below a 2-logit gap and reaches 44–46% above 4 logits (§5.1): every certified token has
  a large gap. Where the floor is gone, bytes still fall with the gap: in the real tier with ideal bounds D-q6+q4 needs
  0.69 below a 0.5-logit gap and 0.38 above 4, A 0.84 and 0.57–0.66. With realistic bounds in real arithmetic only D-q6+q4
  averages below 1.0, and only on gaps above 4 (0.96 and 0.98).
- **Prompt length:** no trend; in the real tier with ideal bounds D-q6+q4 needs 0.41–0.52 at every length from 16 to
  1,024 tokens.
- **Expert:** heavier routing weights need more bytes. Real tier, ideal bounds: the lightest third of the routed experts
  (mean weight 0.18) against the heaviest third (0.66): A 0.62 against 0.80 of the expert, C 0.79 against 0.92, D-q6+q4
  0.43 against 0.50.

### 5.5 The down projection's layout

| Option | Strategy | What it needs (real tier, ideal bounds) | Physical reads (modelled) |
| --- | --- | --- | --- |
| 1. Read the whole down projection | A | down 0.333 + 56% of the gate and up rows: 0.708 | about 1,450 extents per token; 4 KiB amplification 1.18 |
| 2. Down output-row pages | B1, B16, B128 | 90–99% of the down rows anyway: 0.706–0.707 | B1's 2.75 KiB rows: amplification 1.22; B16, B128 1.18 |
| 3. Neuron-major copy (gate row, up row, down column per neuron) | C | 85% of the neurons: 0.853 | on the copy: amplification 1.00 (12 KiB records), 800 extents; on the checkpoint 0.96 of the routed bytes physically, since a down column is strided across all of down |

Down-row pages save almost nothing: the logit difference Δ has weight on every output coordinate, so nearly every row
of the down projection matters. A neuron-major copy (storage 1.33×; preprocessing: transposing 9.6 GB of down
projections, one pass of reads and writes) gives clean 12 KiB reads but needs more neurons than A: without its down
column, an unread neuron's contribution cannot be projected onto Δ, and is bounded coordinate by coordinate (or by
norms). Changing the layout does not lift either obstacle: under the certified model every option needs every byte.

### 5.6 Physical I/O and storage

All figures are modelled, not measured.

- Under the gate's configuration every strategy reads its whole schedule: A, B and C 1.003 of the routed bytes in about
  6 extents per token (one per expert: its three tensors are adjacent in the file), amplification 1.00, the Phase 4B
  pattern; D 1.50–1.63 in 39–85 extents.
- Where the diagnostics save bytes, the reads become small and scattered: about 1,450 extents per token for A and B,
  542 for D-q6+q4, against 6 today. The model counts them; the oracle does not time them, and small scattered reads
  cost more per byte than the large sequential ones Phase 4B measured.
- Persistent storage: A and B 1.00× (the checkpoint as published), C 1.33× (a transposed copy of down), D 1.50× (q8),
  1.63× (q6+q4), 1.51× (q4+q4) next to the BF16 checkpoint, which the exact fallback needs.

## 6. Where the difficulty comes from (Q4)

| Source | Evidence | Weight |
| --- | --- | --- |
| Floating-point rounding (faithful model) | ceiling 8.1% certified, 24.7% RN-even, 100% real; with every weight read, 3.2 logits of named error terms on the tightest pair (2.6 of them after the experts' outputs), plus the experts' internal roundings inside the main term | decisive |
| Final-logit margin | median gap 1.1 logits; nothing below 2.75 certifies | decisive, given the floor |
| Bound looseness (realistic vs true contributions) | realistic bounds need every byte even in real arithmetic; ideal bounds 0.47–0.85 there | decisive, even without the floor |
| down_proj | nearly every down row matters (B); a neuron-major copy needs more neurons (C) | secondary |
| gate/up | 56–84% of the neurons in spatial strategies even with ideal bounds | secondary |
| Scheduler ordering | realistic bounds: ideal ordering changes nothing | negligible |

## 7. Gate

Thresholds fixed in `configs/phase5a-expert-oracle.yaml` before the full runs, on the certified tier with realistic
bounds and ordering, best strategy.

| Criterion | Result |
| --- | --- |
| Correctness (every row of §4) | PASS |
| STRONG PASS: mean ≤ 0.50, coverage ≥ 0.80 | no (1.003, 8.1%) |
| PASS: mean ≤ 0.70, coverage ≥ 0.60; or an ideal certified-tier mean ≤ 0.70 | no (1.003; ideal 0.993) |
| WEAK: mean ≤ 0.90, coverage ≥ 0.10 | no |
| **Verdict** | **FAIL** (run1 and run2) |

The feasibility probe (16 prompts, 8 steps; before the design) had already shown a 7.8% ceiling; the bands are the
brief's, the coverage thresholds were set with that knowledge.

## 8. Answers

1. **Can the final token often be certified without fully materializing all routed experts of the last MoE layer?**
   No. Under the certified model no token was certified with any routed byte unread, in any strategy; even with every
   byte read, 8.1% of tokens certify.
2. **What fraction of routed-expert bytes is needed?** With the realistic schedulers: mean 1.003, median 1.003, p90
   1.003, p95 1.003 (A, B, C: every byte plus 0.3% of metadata); precision refinement 1.50–1.62. The diagnostics: with
   ideal bounds 0.65–0.90 on the certifiable 8%; with real arithmetic and ideal bounds 0.47 (D-q6+q4; median 0.40, p90
   0.63, p95 0.88).
3. **Which decomposition is best?** Under realistic bounds, none saves anything: the spatial ones tie at 1.003 (the
   least metadata), precision refinement costs more than the BF16 experts. Under ideal bounds precision refinement is
   best: q6+q4 (0.47 in real arithmetic, 0.65 on the certifiable tokens), then q4+q4 and q8 (0.53 and 0.56 in real
   arithmetic). It beats neuron pages (0.71), down-row pages (0.71) and the neuron-major layout (0.85). No hybrid (E)
   was built, since nothing pointed to one.
4. **How much of the difficulty comes from each source?** §6: rounding and bound looseness are each decisive on their
   own; the margin decides which tokens survive the floor; down_proj and gate/up are secondary; ordering does not matter.
5. **Does the idealized oracle show substantially more headroom?** Only where the bounds are ideal: the ordering alone
   (ideal order, realistic bounds) gives nothing. Ideal bounds give 0.65–0.90 on the 8% of certifiable tokens and 0.47
   without the rounding floor. The headroom is in the bounds and the arithmetic model, not in the scheduler.
6. **Projected decode I/O** (a projection, not a measurement). Baseline 2.70 GB of routed experts per token (Phase 4B):
   - gate configuration (f = 1.003): 2.70 GB from the last layer alone, 2.71 GB if every layer behaved like it (the
     metadata adds 0.3%); no saving;
   - ideal bounds, certified tier (f = 0.99): 2.70 GB from the last layer alone, 2.68 GB if every layer behaved like it;
   - real arithmetic and ideal bounds (f = 0.47, both obstacles removed): 2.65 GB from the last layer alone, 1.27 GB if
     every layer behaved like it. This is a best case of these decompositions, not a reachable target.
7. **Is physical Phase 5B justified?** **NO — expert AWPMI is currently not economical.** Not because of the layout or
   the scheduler: under the certified rounding model the certificate cannot decide 92% of tokens even with everything
   read, and the sound bounds available cannot certify the rest with anything unread.
8. **Physical layout for Phase 5B:** none is selected. If the two obstacles were lifted, the diagnostics favour a
   per-row precision hierarchy (q6+q4 levels of every expert matrix, stored next to the BF16 checkpoint: 1.63×) over
   any spatial layout; a neuron-major copy is not worth its 1.33×.
9. **Where should AWPMI expand after the last MoE layer?** Not deeper into the model. Anything before the last layer's
   routed experts meets at least the same floor after it, and with this certificate an earlier layer's output passes
   through more operations (its own roundings, every later layer) before the token, so its floor can only be wider. In
   order:
   1. **LM head integration**, where certificates already pay (Phase 1C reads 0.40 of SmolLM2's head, with h exact).
      For Moonlight the head (0.67 GB) is resident, so this would save device memory, not drive reads: worth it only if
      the head is moved off the device to make room for an expert cache.
   2. More positions (prefill), the shared experts, dense MLPs, more MoE layers: not justified by this evidence (the
      same or a wider floor; the shared experts are resident).

## 9. Recommendation

- **Do not build an expert-AWPMI runtime (Phase 5B).** The oracle shows that the bytes of the routed experts are not
  what limits the certificate.
- **Next milestone: the engineering path of the Phase 4B report (§19, item 2).** A native runtime and a host-RAM expert
  tier attack the measured bottleneck directly: the replay predicts 67–83% fewer decode reads with an 11–17 GB host
  tier, with no change to exactness.
- **Decisions for the user, if expert AWPMI is to be revisited:**
  - the rounding model (decisions 0001 and 0005): RN-even in elementwise kernels raises this ceiling from 8.1% to
    17.8%, RN-even everywhere to 24.7%, a binary32 accumulator model to 16.5%. Even at 24.7% coverage the fallbacks
    alone keep the mean above 0.75;
  - sharper sound bounds on unread parts. Today an unread row's contribution is bounded through norms (Cauchy–Schwarz,
    Hölder), which for vectors without alignment overestimates it by a factor of order √2048 ≈ 45; a bound that keeps
    more of the structure would need richer resident metadata. Without one, no decomposition certifies early even in
    real arithmetic.

## 10. Limitations

- **One layer, one position.** With this certificate earlier layers can only be harder: their outputs pass through
  more operations before the token.
- **The certificate.** The pairwise certificate is the strongest validated one, not the strongest possible. One that
  followed a rounding error to every place it reaches, instead of bounding each occurrence separately, could narrow the
  floor somewhat. Under the faithful model each rounding may fall either way, so no sound certificate can treat the
  roundings as random.
- **Oracle orders are static.** A realistic order is fixed from the first state; with realistic bounds the ideal order
  changes nothing, so an adaptive one would not either.
- **Diagnostic subsets.** The real tier's cells run on 96 tokens and the RN-even cells on 192, to keep each stage under
  two hours; the ceilings run on all 768.
- **Physical I/O is modelled.** Blocks and extents from the published offsets; no read was timed.
- **One GPU and one software stack** (sm_89, torch 2.14.1, cuBLAS 13); the accumulation and rounding models are those
  of decisions 0001 and 0005, validated against the reference on every token.

## 11. Reproduction

```bash
uv sync
uv run pytest                                                                                   # 551 tests
uv run awpmi pack expert-index --config configs/phase4b-moonlight.yaml                          # the Phase 4B index (headers only)
PYTHONHASHSEED=1 uv run python benchmarks/expert_oracle.py --output experiments/phase5a/<name> --stage prepare
PYTHONHASHSEED=1 uv run python benchmarks/expert_oracle.py --output experiments/phase5a/<name> --stage capture       # ~25 min
PYTHONHASHSEED=1 uv run python benchmarks/expert_oracle.py --output experiments/phase5a/<name> --stage oracle --shard 0   # ~1 h
PYTHONHASHSEED=1 uv run python benchmarks/expert_oracle.py --output experiments/phase5a/<name> --stage oracle --shard 1   # ~35 min
PYTHONHASHSEED=1 uv run python benchmarks/expert_oracle.py --output experiments/phase5a/<name> --stage oracle-real        # ~50 min
PYTHONHASHSEED=1 uv run python benchmarks/expert_oracle.py --output experiments/phase5a/<name> --stage digest
uv run python benchmarks/expert_oracle_report.py experiments/phase5a/<name> --compare experiments/phase5a/oracle-run1
```

A run is reproduced if its `digest.json` matches run1's (timings and system fields excluded). Run2 (`PYTHONHASHSEED`
2) matches it in all five:

| Digest | run1 = run2 |
| --- | --- |
| prompts | `c3b0c0f38eacba4fbc4211c10af9d70bbc4b769b10257f739673d21ee9ec8cc4` |
| capture records | `7de8897612f8014ae3e351647009f3e8db1109bfaa9941661fe36cc35986ddaa` |
| capture tensors | `c20a83917363b480c6cb672e61b641aadce17249666558a675e9367334c29aa0` |
| samples | `faf7653de53c1edabcd363a74a7c7d9452bdc868ac3e333c6300d8f34841e4f3` |
| real cells | `1686fa68beafac9e10e0230283db3707e64d2d0114202fac84642ef3c811cbf0` |
