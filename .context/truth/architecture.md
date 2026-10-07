# Architecture

## Overview

This repository hosts **Weightsift**, implementing **AWPMI — Adaptive Weight-Page Materialization for Inference**, a
research prototype (Python package `awpmi`, at the repository root). AWPMI materializes
independently fetchable weight pages one at a time. It stops only when a conservative,
deterministic certificate proves that the next-token argmax equals that of the fully
materialized reference model. The guide is `state/Weightsift_Implementation_Guide.md` (it replaced the AWPMI roadmap).

Implemented so far:

- **Phase 1A**: a runtime that materializes input-column pages. Only the LM head is
  adaptive. The transformer body runs exactly ("exact prefix", roadmap Mode A), and all
  weights stay resident, so materialization is logical rather than physical.
- **Phase 1B**: an *oracle* for precision-refinement decompositions of the LM head, with
  row-selective refinement. It is a simulator, not a runtime: nothing in `awpmi.oracle`
  is called by a runtime.
- **Phase 1C**: the runtime of the decomposition Phase 1B chose (q6+q4, row-selective,
  tie-aware), `RefinementLMHead`. It reads a bit-packed store whose every read is
  counted. The store is still resident in memory: materialization is logical, but the
  bytes are those of real packed buffers.
- **Phase 2**: certification extended into the transformer, as an adaptive suffix inside
  the last layer's MLP (`AdaptiveSuffixRuntime`): the final projection (`down_proj`,
  depth 1) or the whole last MLP (depth 2), ahead of the Phase 1C LM head. Enclosures
  are propagated through the reference's own operations under the faithful rounding
  model; a scale-free pairwise certificate works through the final RMSNorm. The masked
  LM-head fallback is the default (decision 0005).
- **Phase 3**: physical selective materialization (decision 0006). A storage core that knows no
  model (`awpmi.storage`, `awpmi.streaming`, `awpmi.materialization`) reads only the requested
  rows of segments from files (direct I/O) or host memory, moves only their bytes to the
  device, caches pages under a budget, and counts every byte, cross-checked against the OS.
  The Phase 1C LM head runs on it unchanged (its segments in a pack, its exact rows read from
  the published checkpoint), bitwise equal to the resident runtime. A model-agnostic
  mixture-of-experts adapter serves transformers' experts modules from the same backend,
  bitwise equal to the resident model (7 architectures in tests, Granite 3.1 1B-A400M in the
  benchmark).
- **Phase 4A**: a real MoE whose experts exceed the GPU (decision 0007). OLMoE-1B-7B (12.9 GB of
  experts, an 8 GB GPU, a 6 GB allocator cap) runs with its experts read from the published,
  split checkpoint through composed segments and executed by a compact experts call (buffers for
  the routed experts only), bit for bit equal to the fully materialized reference executed one
  layer at a time. Reference profiles declare what "exact" is relative to.
- **Phase 4B**: a DeepSeek-V3-architecture MoE out of both VRAM and host RAM (decision 0008).
  Moonlight-16B-A3B (28.8 GB of routed experts, 31.9 GB in all, against 32 GB of RAM and an 8 GB
  GPU under a 6 GB cap) runs with bounded expert buffers: an experts call whose routed experts
  exceed a byte budget runs in chunks of experts, with the experts implementation's own combine.
  It is compared with an independent streaming reference: transformers' model and loader, each
  experts layer materialized whole from the checkpoint when it runs.
- **Phase 5A**: an *oracle* of AWPMI inside routed experts (decision 0009), like Phase 1B a
  simulator, not a runtime (`awpmi.oracle.experts`). For Moonlight's last MoE layer at decode, upstream
  exact, it reads the routed experts' weights progressively in several decompositions, bounds every
  intermediate of the experts call, the MoE block, the residual and the final norm with the reference's
  own operations, and asks a pairwise certificate (generalized to a mixture of experts) to decide the
  token against the whole vocabulary. Under the certified rounding model, certification needs every
  routed byte, and then holds for 8.1% of tokens; the Phase 4B path is unchanged.

## Major components (`src/awpmi/`)

| Module | Role |
| --- | --- |
| `runtime.py` | Declared numerical environment (determinism, FP32 accumulation, no TF32 or reduced-precision reductions). Call it before any CUDA matmul. |
| `models/smollm2.py` | Loads the pinned model, gives access to the LM-head weight, runs the exact prefix (`final_hidden_state`). Phase 2: `suffix_prefix`, the model's own forward interrupted by a pre-hook where the adaptive suffix begins (`down_proj` or the last `mlp`), optionally with a KV cache. |
| `reference.py` | `ReferenceRunner.next_token`: the unmodified HF forward (`logits_to_keep=1`), with the LM-head input captured by a hook; with `intermediates=True`, also every intermediate of the last MLP and the final norm (hooks only observe). |
| `paging/` | `WeightPage` (column block of a weight; Phase 2 adds `layer` and `tensor_role`), `PageIndex` (pages plus bound metadata), `PageBoundMetadata` (row-page L2 norms, rounded up to float32), `InMemoryPageSource` (fetch-counting views). |
| `bounds/floating.py` | γ_n error bounds; directed rounding of float64 values onto a dtype's grid. |
| `bounds/linear.py` | Cauchy–Schwarz block-norm upper bounds; exact float64 page contributions. Phase 5A: `matvec` and `absolute_mass_upper` also take a batch of matrices [B, N, K]. |
| `bounds/residual.py` | `ReferenceNumerics` (accumulation model of the reference GEMM); `ResidualBounder` (partial logits and remaining pages → intervals on the reference logits); `reference_logit_interval` (centre and radius → output-grid interval, shared by Phase 1A and 1B). |
| `bounds/remainder.py` | Phase 1B: per-row remainder norm metadata (`RowNormBounds`); bound min(‖x‖₂‖h‖₂, ‖x‖_∞‖h‖₁) on a missing remainder row. |
| `bounds/coarse.py` | Phase 1C: error model of the runtime's binary32 coarse pass (`CoarseArithmetic`, γ_{K+2}(2⁻²²) plus a flush-to-zero term) and the free Hölder bound s·limit·‖h‖₁ on a level's absolute mass. |
| `bounds/rounding.py` | Phase 2: rounding models of the reference's outputs. `FAITHFUL` (certified, decision 0001) and `NEAREST_EVEN` (experimental, decision 0005); `RoundingAssumptions` pairs a model for GEMM epilogues and one for elementwise kernels (`CERTIFIED` is faithful for both); `round_enclosure`, `spacing_upper`, rounding-error bounds. |
| `bounds/enclosure.py` | Phase 2: `Enclosure` (lower, upper, provenance), the residual state generalized to any internal tensor (roadmap §2.2); `UnboundedValue` makes the caller fall back. |
| `bounds/operators.py` | Phase 2: verified propagation through the reference operations of the last MLP and the final norm: `linear` (exact or partly unread weights, interval input), `residual_add`, `multiply`, `silu`, `rms_norm` (returns q, n and h). Documented fp32 assumptions: 2⁻¹⁸ relative error per elementwise fp32 operation, flushing below the normal range, γ_{n+2}(2⁻²²) per reduction. Phase 5A: `reduce_sum` (an experts call's combine: the float32 sum over the top-k, then the BF16 conversion), `multiply` into float32 (the routing weights), `linear` on a batch of weights (one call's routed experts); `DeepseekV3RMSNorm` runs LlamaRMSNorm's operations (tested). |
| `bounds/pairwise.py` | Phase 2: scale-free pairwise certificate through the final RMSNorm. For a winner w and each contender j it lower-bounds (acc_w − acc_j)/q with a box bound and a decomposed bound that keeps the down projection's cancellation (M = W_dᵀΔ, unread columns by Cauchy–Schwarz), then requires a 2-ulp separation of the faithful logits. Phase 5A: the decomposed bound for a mixture of routed experts (`ExpertMixture`, `MixtureTerm`, `mixture_bound`): y = b + Σ_e w_e·(K_e + X_e)·a_e + η, each expert's known down part K_e by its exact projection M_e = K_eᵀΔ (or a binary32 one with a rank-one error bound, `RankOneError`), its unknown part by row norms or column norms, η the named roundings and accumulations; `relax_mixture` moves columns into the base (a weaker, cheaper, still valid certificate). Phase 2's `DownProjection` is unchanged. |
| `state.py` | `ResidualState`: partial logits, lower/upper bounds, materialized and remaining pages. |
| `certificate.py` | `Certificate.check`: CERTIFIED iff lower[w] > max_{j≠w} upper[j] with w = argmax(partial), otherwise UNKNOWN. `TieBreak.LOWEST_INDEX` (decision 0003) also accepts equality against higher-index rows. `certify_columns` and `contenders` are batched forms used by the oracle (row elimination). |
| `decomposition/` | Phase 1B: `RefinementDecomposition` (W = base + refinement levels + exact remainder, per-row int-b levels with float32 scales, exactness checked with TwoSum, byte accounting). Phase 1C adds `packing.py` (per-row little-endian bitstream of biased codes, `CodeLayout.unpack`) and `accounting.py` (4 KiB block accounting, shared with the oracle). |
| `oracle/refinement.py` | Phase 1B diagnostic simulator: `RefinementBatch` (per-state intervals for the `realistic`, `abs_mass` and `ideal` bound tiers), `simulate` (global or row-selective refinement). |
| `oracle/experts.py` | Phase 5A diagnostic oracle of AWPMI inside routed experts (decision 0009). `experts_call_reference` (transformers' `grouped_mm_experts_forward` step by step with its own functions: every intermediate, bitwise), `decode_sample` (the reference's values from a decode token's captured inputs, and the same forward in real arithmetic), `ExpertLayer` / `SampleWeights` (the layer's experts, metadata, per-row refinement decompositions), `ExpertStates` and `MatrixKnowledge` (what is read: per row UNKNOWN, a level, or EXACT; down columns), `propagate` (enclosures of g, u, s, a, o, z, R, m, y, n, h and the `ExpertMixture`, batched over the routed experts), `Certifier` (pairwise certificate against every vocabulary row: the nearest rows first, then all in binary32 blocks; candidate = largest centre logit), arithmetic tiers (`certified` only certifies; what-ifs `certified_u24`, `rn_elementwise`, `rn_even`; diagnostic `real`), bound tiers (realistic metadata or ideal true remainders), orderings (realistic or ideal), `Strategy` (A neuron pages + whole down, B neuron pages + down row pages, C neuron-major pages, D per-row precision refinement), `run_cell` (the first byte budget that certifies, by bisection over a fixed order), `physical_estimate` (4 KiB blocks and extents per layout), `ceiling`. No runtime imports it (layering test). |
| `stores/refinement.py` | Phase 1C: `PackedRefinementStore`, packed levels, scales, resident remainder norms and the original rows. `read_level` / `read_exact` / `read_fallback` are the only way to weight values; each read is logged with its rows and bytes (`bytes_read`, `reads`). Phase 3: it reads through a `MaterializationBackend`: `from_decomposition` (segments resident on the weight's device, Phase 1C) or `from_pack` (a refinement pack on storage); a level row is one record, the packed codes then the float32 scale (`level_records`); `write_refinement_pack` writes the pack, its exact rows referring to the checkpoint tensor. |
| `storage/` | Phase 3 storage core (no model, no certificate). `layout`: `Segment` (fixed-size rows at an offset of a file; a safetensors tensor is one as it is), Phase 4A `ComposedSegment` (each row the concatenation of byte spans of one or more files: a logical tensor a checkpoint stores as several tensors), `byte_runs`, safetensors headers, typed row views. `fileio`: `PositionedFile` (thread-safe positioned reads; direct = `FILE_FLAG_NO_BUFFERING` / `O_DIRECT`), the OS's per-process read counters, process memory, aligned (pinned) host buffers. `store`: `plan_reads` (runs of requested bytes, sorted by file and offset → 4 KiB-aligned extents per file, merged when their blocks touch; each run keeps its output offset; `positions` writes rows into chosen rows of an output), `IOStats`, `PageStore` with `InMemoryPageStore` and `FileBackedPageStore` (reads the planned extents only, of any of its files, with 8 worker threads). `cache`: `PageCache` (device pages under a byte budget, pinned entries, admission freeze) with `LRUPolicy` and `HotnessPolicy`. `pack`: `PackWriter` (v2 adds composed segments and publisher-declared file sha256), `open_pack` (manifest with files, segments and sha256; segments or files re-hashed with direct reads), `Pack.store` / `Pack.load` (the host-memory tier), `SourceFile` (a published checkpoint file in the Hugging Face cache; `published_sha256`), `sha256_file_direct`. |
| `streaming/streamer.py` | Phase 3 transfer layer: `PageStreamer`. Pinned, page-aligned staging slots (two: double buffering), a CUDA copy stream, per-slot events, and the compute stream waiting on a ready event. A file-backed fetch reads its plan in slot-sized pieces and copies only the requested rows' bytes to the device; a host-memory fetch gathers into a slot. Phase 4A: `fetch(…, out, positions)` writes into a caller's buffer (the copy stream first waits for the caller's stream); runs of at least 256 KiB go straight from staging, shorter ones are gathered; scattered destinations get one copy per contiguous range. `prefetch` returns a `Ticket` (consumed and wasted bytes counted). |
| `materialization/` | Phase 3: `MaterializationBackend` (`materialize(segment, rows, out=None)` → device rows: a store in memory on the compute device answers directly; otherwise the cache, then the streamer; with `out`, written into a caller's buffer, cached rows copied first ("hits first"); cache entries in their own CUDA memory pool; every request counted; `report` gathers the storage, transfer and cache counters), `WeightStore` (typed rows of named weights), `ExpertStore` / `ExpertGroup` (a layer's expert-sliced segments; `load`, `fill` into full-shape buffers, Phase 4A `assemble` into compact buffers; Phase 4B: `assemble` of some parameters only, and the backend's `largest_request_bytes`). |
| `models/moe.py` | MoE adapter, model-agnostic over the experts convention of transformers 5: `find_expert_modules` (a module with `num_experts` and a ≥3-D parameter whose first dimension is that count), `write_expert_pack` (refers to checkpoint tensors with the same bytes, copies the rest), `StreamedExperts` (Phase 3 full-shape slot buffers; Phase 4A `compact=True`: parameters `None` between calls, buffers for the routed experts in ascending order, `num_experts` and `top_k_index` remapped for the call; poison mode; `all_experts` for dense layer streaming; `on_call` with an `ExpertCall`, whose `per_assignment_outputs` re-runs the call per (token, expert) on the same weights). Phase 4B `max_call_bytes`: a call whose buffers would exceed it runs in chunks of consecutive slots; each expert matrix is a `ChunkedExpertWeight` for the call, a stand-in (not a tensor) reachable only by `weight[slot]` (eager) or `torch._grouped_mm` (computed chunk by chunk, each chunk's groups by one call on its rows); every chunk of every matrix materialized exactly once, the stand-ins closed after the call; `ExpertCall.chunks`, and `per_assignment_outputs(weights=…)` for chunked calls. `FullLayerOffload` (the reference for a model whose experts do not fit: host parameters copied whole to the device before each experts call), `move_except_experts`, `routed_experts`, `record_routing`. |
| `models/checkpoint.py` | Phase 4A: a published checkpoint served in place. `checkpoint_sources` (the files of a pinned revision, with their Hub sha256), `expert_sources` (each expert-sliced parameter's checkpoint tensors, derived from the transformers conversion mapping: stacking per expert, optionally concatenation along each expert's first dimension; literal renamings reversed; anything else refused), `expert_index_segments` (composed segments from headers, shapes and dtypes checked), `write_expert_index` (a pack v2 that copies nothing), `load_model_without_experts` (meta skeleton, every non-expert tensor read with direct I/O and cast by transformers' dtype plan, non-persistent buffers from `_init_weights`; experts stay on `meta`). Phase 4B: `checked_expert_sources` and `neighbours` (an adapter's layout check and its per-layer modules, shared by the OLMoE and Moonlight adapters), `parameter_segments` (named parameters' checkpoint tensors as segments, headers only). |
| `models/olmoe.py` | Phase 4A OLMoE adapter: `EXPERT_LAYOUT` checked against transformers' mapping, `routers`, `REFERENCE_PROFILE` (BF16, `grouped_mm`, SDPA). |
| `models/moonlight.py` | Phase 4B Moonlight adapter (DeepSeek-V3 architecture): `EXPERT_LAYOUT` (checked), `routers` (with their float32 correction bias), `shared_experts`, `moe_blocks`, `ROUTING` and `check_config` (sigmoid `noaux_tc`, one group, top 6, renormalized × 2.446: what transformers' router computes), `REFERENCE_PROFILE` (BF16, `grouped_mm`, SDPA). Phase 5A: the names around the last MoE block (`RESIDUAL_NORM_SUFFIX`, `FINAL_NORM`, `LM_HEAD`). |
| `models/streamed.py` | Phase 4B `StreamedParameters`: chosen dense parameters (e.g. shared experts) served whole from a `WeightStore` at every call of their module, `None` between calls; `remove(restore=True)` makes them resident again. |
| `streaming_reference.py` | Phase 4B independent reference, `StreamingReference`: transformers' model built on `meta` with the declared kernels; every non-expert weight loaded by transformers' own `convert_and_load_state_dict_in_model` (safetensors slices opened as `from_pretrained` opens them, its dtype plan and mapping, its finalization); each experts layer loaded by the same function in a pre-hook and released after the module ran (`None` between calls); `resident` and `set_resident` for the residency check; `ReferenceCall.per_assignment_outputs`. Imports nothing of Weightsift's path (layering test). |
| `profiles.py` | Phase 4A reference profiles: `ReferenceProfile` (kind, stored weight dtype, compute dtype, kernels, numerics, quantization), `BF16_REFERENCE`, `FP16_REFERENCE`, `native_quantized_reference` (declared, refused by `check_model`); `check_model`, `check_weights`. |
| `cli.py` | `awpmi pack lm-head`, `awpmi pack experts` (roadmap §3.2) and `awpmi pack expert-index` (Phase 4A: the composed-segment index of a split checkpoint, from headers only). |
| `stores/suffix.py` | Phase 2: `MLPStore`, neuron-major pages of the last MLP (gate row, up row, down column of each neuron, one contiguous run each), served only for the stage's adaptive roles, every read logged; resident metadata: down column norms, down row norms (L2, L∞), up row norms. |
| `refinement_head.py` | Phase 1C runtime `RefinementLMHead.run`: binary32 coarse pass over every row → certificate → elimination → float64 refinement of the contenders (the oracle's realistic arithmetic) → exact rows → certificate or fallback (`FallbackMode.MASKED`, the default since decision 0005, self-tested and guarded, else `FULL`). Phase 2: `run_enclosure` runs the same states on an `Enclosure` of h (input spread \|L\|·ρ added to every radius, no fallback, optional row limits), and `HeldReads` lets one token's passes share their reads. Optional `RunTrace` and `StageTimer` (Phase 3 adds `*:read` stages around every store read; timing only). |
| `suffix_runtime.py` | Phase 2 runtime `AdaptiveSuffixRuntime.run` for a `SuffixStage` (lm_head, down, mlp): read a budgeted share of the suffix's neuron pages (largest bound contribution first) → enclosures through the suffix → LM-head pass on the enclosure → pairwise certificate → else exact recomputation of the suffix (the reference's operations and shapes, bitwise) and the Phase 1C LM head on the exact h, reusing the reads. Experimental rounding models report `would_certify` only. |
| `schedulers/` | Deterministic page ordering: `sequential`, `largest_residual`, `bound_per_byte`. They see metadata and state, never page values. |
| `executor.py` | `AdaptiveLMHead.run`: the certify → schedule → materialize → refine loop, plus the exact fallback. |
| `tracing.py`, `config.py` | Environment metadata, source-tree hash, JSONL, digests, `StageTimer`; YAML config. |

Outside the package: `benchmarks/run.py` (experiment driver), `benchmarks/report.py`
(aggregation and gate evaluation), `benchmarks/prompts.py` (deterministic wikitext-2
prompts), `benchmarks/oracle.py` (scheduler-independent ceilings: real-certificate
singleton check plus an ideal magnitude-bound ceiling; diagnostic only),
`configs/smollm2-135m.yaml`, `experiments/phase1/<run>/` (raw results), and `tests/`.
Phase 1B adds `benchmarks/refinement_oracle.py` (driver: reference pass, every decomposition ×
mode × bound tier × tie rule, hard-failure checks), `benchmarks/refinement_report.py`
(aggregation, Phase 1A comparison, decision-gate classification),
`configs/phase1b-refinement.yaml` and `experiments/phase1b/<run>/`.
Phase 1C adds `benchmarks/refinement_runtime.py` (driver: both fallback modes per prompt,
envelope and coarse-arithmetic validation, byte audit, comparison with the Phase 1B records,
stage timing, kernel profile), `benchmarks/refinement_runtime_report.py` (aggregation, oracle
comparison, round-to-nearest what-if, gate), `benchmarks/fallback_study.py` (masked-GEMM
row independence across shapes, dtypes and devices; output-rounding probe),
`configs/phase1c-runtime.yaml` and `experiments/phase1c/<run>/`.
Phase 2 adds `benchmarks/suffix_runtime.py` (driver: reference with every intermediate, the
exact prefix of each stage, the depth-0 LM head, then every stage × budget × rounding model
× row-limit policy, with enclosure, pairwise-bound, exact-suffix, byte and single-read
checks), `benchmarks/suffix_report.py` (curves, bound provenance, gate, per-stage decision),
`configs/phase2-suffix.yaml` and `experiments/phase2/<run>/`.
Phase 3 adds `benchmarks/storage_runtime.py` (the Phase 1C LM head on a tier: the full BF16
head from the drive or from host memory, the resident runtime, the runtime on the drive, on
host memory and with a cached base level; parity with the resident runtime and the Phase 1C
records, storage audits, OS cross-check), `benchmarks/storage_report.py` (gates A and B),
`benchmarks/moe_runtime.py` (Granite MoE: resident reference, then experts from the drive
under several cache budgets and policies, every step compared bit for bit),
`benchmarks/moe_report.py` (gate C, routing statistics), `configs/phase3-storage.yaml`,
`configs/phase3-moe.yaml`, `experiments/phase3/<run>/`, and packs under `packs/` (gitignored,
rebuilt by `awpmi pack`).
Phase 4A adds `benchmarks/olmoe_runtime.py` (two processes: the reference, verified sources,
index audit, residency check and decoding with every layer materialized in turn; then the
streamed model under a device-memory cap, every configuration compared with the reference in
every digest and audited), `benchmarks/olmoe_report.py` (correctness and gates A–D),
`benchmarks/olmoe_profile.py` (un-instrumented timing per stage, kernel trace),
`configs/phase4a-olmoe.yaml` and `experiments/phase4a/<run>/`.
Phase 4B adds `benchmarks/moonlight_runtime.py` (two processes: the streaming reference with its
residency check and the sha256 of every expert row it loads; then, under the device cap, the
files verified, every index row audited against the reference's digests, and every
configuration compared with the reference in every digest and audited; chunked calls' expert
outputs checked by rereading their experts through a separate store),
`benchmarks/moonlight_reference_check.py` (the streaming reference against `from_pretrained` on
Moonlight truncated to its first layers), `benchmarks/moonlight_report.py` (correctness, gates
A–F, prefill and memory tables, a replay of the routing through the cache),
`benchmarks/moonlight_profile.py` (un-instrumented timing per stage, one configuration per
process, kernel trace by kind and module scope), `configs/phase4b-moonlight.yaml` and
`experiments/phase4b/<run>/`.
Phase 5A adds `benchmarks/expert_oracle.py` (stages: capture, Moonlight on the Phase 4B streamed path
with the target layer's tensors at every decode step, compared with Phase 4B's reference records;
oracle, in shards of prompts: the reference recomputed bitwise from the captured inputs, the target
weights against the reference's row digests, every arithmetic tier's ceiling, the cells; oracle-real:
the real tier's cells), `benchmarks/expert_oracle_report.py` (correctness, ceilings and the floor's
terms, every strategy's bytes with breakdowns, the modelled physical reads, the gate, the decode-I/O
projection), `configs/phase5a-expert-oracle.yaml` and `experiments/phase5a/<run>/` (the captured
tensors are gitignored: regenerated by the capture stage, their sha256 in the digest).

## Data flow (one input)

1. `ReferenceRunner.next_token(ids)` returns the reference token, its BF16 logits, and the
   LM-head input.
2. `final_hidden_state(model, ids)` is the exact prefix. It must be bitwise-equal to the
   captured LM-head input.
3. `AdaptiveLMHead.run(hidden, scheduler)`:
   - computes per-page norms of the hidden state and builds a `ResidualBounder`;
   - loops: `Certificate.check(state)` → if CERTIFIED, stop; else `scheduler.select` →
     `PageSource.get(page)` → `page_contribution` (float64) → `refine_state`;
   - if every page is materialized and the result is still UNKNOWN, the fallback runs
     `F.linear(hidden, cat(pages))`, which is bitwise-equal to the reference.

Phase 1C runtime (`RefinementLMHead.run(h)`), for the same exact LM-head input:

1. `store.read_level(0)` (every row) → decode in chunks → binary32 `torch.mv` → centre
   s·fl(S) and the coarse interval (`bounds/coarse.py`).
2. Loop: `certify_columns` (tie-aware) → if CERTIFIED, stop; else `contenders` → read only
   those rows of the next level (or their original rows in the exact state) → float64
   intervals → intersect with the previous ones.
3. Still UNKNOWN after the exact state: MASKED fallback (the default: full-shape `F.linear`
   on the survivors only, self-tested and guarded) or FULL fallback (read every unread row,
   `F.linear`, bitwise).

Phase 2 runtime (`AdaptiveSuffixRuntime.run`), stages down and mlp:

1. `suffix_prefix(model, ids, boundary)`: the reference forward up to the boundary, all
   positions (residual, MLP input, and for down the MLP activation).
2. Read the budgeted neuron pages from the `MLPStore` (mlp: every gate row first, then up
   rows and down columns; down: down columns), largest bound contribution first.
3. Enclosures at the last position: [gate → SiLU, up → product →] down projection
   (unread columns bounded by row norms) → residual addition → final RMSNorm → h.
4. `RefinementLMHead.run_enclosure(h)`: the Phase 1C states with the input's spread;
   CERTIFIED stops. Else, if the exact state was reached: `pairwise_certificate` on the
   remaining contenders.
5. Still UNKNOWN: read the rest of the suffix, recompute it with the reference's own
   operations on all positions (bitwise the reference's o, y, h), then
   `RefinementLMHead.run(h, held=…)`: Phase 1C on the exact h, sharing the first pass's
   reads.

Phase 3, the Phase 1C runtime on storage (`PackedRefinementStore.from_pack`): every
`read_level` / `read_exact` / `read_fallback` becomes `MaterializationBackend.materialize`:

1. the page cache, if any, serves what it holds (e.g. a pinned base level);
2. the store plans the read: runs of consecutive rows → 4 KiB-aligned extents;
3. the streamer reads the extents into a pinned staging slot (8 threads of positioned reads,
   direct I/O), gathers the requested rows and copies only them to the device on the copy
   stream; the next piece's reads overlap the previous piece's copy;
4. the compute stream waits for the copy; the runtime computes exactly as before.

Phase 3, a MoE model (`StreamedExperts`): the router runs (resident); the experts module's
pre-hook takes `top_k_index`, materializes those experts of that layer (cache, then drive),
writes them into the full-shape slot buffers, and the module's own forward runs.

Phase 4A, compact (`StreamedExperts(compact=True)`), for each experts call:

1. the router runs (resident); the pre-hook takes the routed experts R from `top_k_index`
   (unique, ascending);
2. buffers [|R|, …] are allocated; `ExpertStore.assemble` → `materialize(segment, R, out=…)`
   per parameter: cached experts are copied first, the rest planned as composed spans of the
   checkpoint files, read (direct I/O, 32 MiB pieces) and copied into their rows;
3. the parameters become those buffers, `num_experts` = |R|, and `top_k_index` is replaced by
   each expert's slot; the module's own forward runs;
4. a forward hook releases the buffers: the parameters are `None` again.

Phase 4B, chunked (`StreamedExperts(compact=True, max_call_bytes=B)`), when |R| × expert bytes > B:

1. R (ascending) is split into chunks of ⌊B / expert bytes⌋ consecutive slots; each expert matrix
   becomes a `ChunkedExpertWeight` (small expert-sliced parameters, e.g. biases, are assembled
   whole); `num_experts` and `top_k_index` are remapped as in compact mode;
2. the module's own forward runs. `grouped_mm`: its sort, offsets, gating, weighting and combine
   run as written; each `torch._grouped_mm` on a stand-in materializes chunk after chunk (cache
   first, then the drive, into a buffer of that chunk only), multiplies the chunk's groups by one
   call on their rows into one output, and releases the chunk. eager: `weight[slot]` materializes
   the chunk holding the slot (ascending, so each chunk once);
3. the release hook checks that every chunk of every matrix was materialized exactly once, and
   closes the stand-ins.

Phase 4B, the reference (`StreamingReference`), in its own process: transformers' model with
every non-expert weight on the device (loaded by transformers' loader); before each experts
module runs, a pre-hook loads that layer's checkpoint tensors through the same loader (stacked,
gate/up fused, on the device), the module runs on the full layer, and a hook releases it.

Phase 5A, the expert oracle (`awpmi.oracle.experts`), for one decode token:

1. capture (benchmark, Phase 4B's streamed path): the target layer's experts input x, the router's
   choice and weights, the residual r, the shared experts' output S, and the reference's R, m, y, h,
   logits;
2. `decode_sample`: the experts call recomputed with transformers' own functions on the layer's
   weights (every intermediate g, u, s, a, o, z), then m, y, n, h and the logits by the reference's
   operations: bitwise the capture (the exact fallback);
3. per arithmetic tier, the ceiling: every routed weight known → `propagate` → `Certifier.evaluate`;
4. per strategy and cell: a fixed order of units (realistic: bound contribution per byte from the
   first state; ideal: true contribution), then a bisection over byte budgets for the first state whose
   nearest pairs all certify (monotone with realistic bounds), then the full check there;
5. bytes from the state reached (metadata charged; a token never certified costs the whole schedule),
   and the modelled 4 KiB reads per layout.

## Important constraints

- The certificate bounds the **floating-point reference logits** (after accumulation
  error and output rounding), not just the exact product. See
  `decisions/0001-certificate-targets-floating-point-reference.md`.
- The fallback recomputes the reference operation. See
  `decisions/0002-fallback-recomputes-reference-operation.md`.
- Storage, scheduling, partial execution and certification stay in separate modules.
  Schedulers affect efficiency only.
- Bound metadata is resident and counts against savings. For the LM head
  (49152 × 576, BF16, 56.6 MB) it is 1.8 / 3.5 / 7.1 / 14.2 MB for page widths
  64 / 32 / 16 / 8. Phase 1B remainder norms cost 0.39 MB per non-exact state.
- Decomposition, bounds, certification and simulation are separate modules. A
  decomposition only builds levels and counts bytes. The oracle's `abs_mass` and `ideal`
  tiers use information a runtime cannot have and must never reach a runtime path.
- The Phase 1C runtime touches weights only through its store. Its refined and exact
  states share their radius formulas with the oracle (`remainder_radius`,
  `exact_radius`), so only the coarse state differs from the oracle's arithmetic
  (decision 0004).
- Only the faithful rounding model certifies (decisions 0001 and 0005), for every rounding
  of the suffix: GEMM epilogues and elementwise kernels alike. Round-to-nearest-even is an
  experimental what-if: `AdaptiveSuffixRuntime` refuses it unless `experimental=True`, and
  then never sets `certified`.
- The masked LM-head fallback assumes row independence of the reference GEMM. It is used
  only in the fallback path, behind its self-test and guard, never in a certificate.
- The adaptive suffix starts after the last layer's attention, so the KV cache is written by
  the exact prefix and stays the reference's (roadmap §2.8, Mode A; tested over cached
  greedy generations).
- Layering (decision 0006, enforced by `tests/test_layering.py`): the storage core
  (`storage`, `streaming`, `materialization`) names no model, tensor or router and imports no
  layer above it; the certification layer (`bounds`, `certificate`, `refinement_head`) imports
  no storage. Where bytes come from never changes what is computed: storage-backed runs are
  checked bit for bit against resident ones.
- A file-backed store reads the planned extents and nothing else: a block is read only if it
  holds a requested byte, and only requested rows reach the device. Physical bytes are what
  the reads returned, and they must equal the OS's own per-process counters.
- On NTFS, direct reads of a file that has an active OS cache map are several times slower
  (decision 0006). Packs are verified with direct reads; benchmarks never mix buffered and
  direct access to one file, and wait for cached handles to close after setup.
- The MoE adapter's Phase 3 slot buffers have the experts' full shape (E × expert bytes per
  layer and buffer set). The compact mode (decision 0007) holds only the routed experts, in
  ascending expert order (required for the eager implementation to stay bitwise). A call budget
  (decision 0008) bounds a call's expert buffers, a prefill that routes every expert included:
  the call runs in chunks of consecutive experts, and the combine stays the implementation's own,
  once per call. The `grouped_mm` path is exact where each group's product does not depend on the
  call's other groups (here: one cuBLAS GEMM per group); accumulating per chunk would not be
  (tested adversarially).
- "Exact" is relative to a declared reference profile (`awpmi.profiles`, decision 0007):
  streamed weights are used in their stored dtype and never converted on the fly.
- A model whose experts do not fit on the device is compared with `FullLayerOffload`, the same
  model with each experts layer materialized whole in turn, in a separate process; residency is
  checked not to change any result. A model that does not fit in host memory either is compared
  with `StreamingReference` (decision 0008): transformers' own loader, one experts layer at a
  time from the checkpoint, sharing no code with Weightsift's expert path (layering test), and
  itself checked against `from_pretrained` where that fits.
- The oracles (`awpmi.oracle`: Phase 1B, Phase 5A) are diagnostic simulators: no other module
  imports them, and they import no storage, streaming, materialization, store or runtime module
  (layering test). In the expert oracle only the `certified` arithmetic tier (faithful, u = 2⁻²²) may
  set `certified`; the what-if and diagnostic tiers record `would_certify` only. Its ideal bound tier and
  ideal ordering use true weight values and never inform a realistic bound or ordering (tested by
  permuting values that keep the metadata).
