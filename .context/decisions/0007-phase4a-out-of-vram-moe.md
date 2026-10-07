# 0007 — Phase 4A: an out-of-VRAM MoE through compact expert calls and split-checkpoint indexes

Status: accepted (Phase 4A, 2026-10-03)

## Context

Phase 3 (decision 0006) served Granite's experts from the drive bit for bit, but with two limits
that ruled out larger models:

- the adapter's slot buffers had the experts' full shape, E × expert bytes per layer (11 GB for a
  DeepSeek-V3 layer);
- an expert pack could refer to a checkpoint tensor only if it held exactly a parameter's bytes,
  so checkpoints that store each expert as separate tensors needed a converted copy.

The user's Phase 4A brief:

- Run a real MoE whose expert weights do not fit in GPU memory, through the existing generic
  streaming and cache backend, reproducing the fully materialized reference's discrete decisions
  exactly. First target: OLMoE-1B-7B. Not DeepSeek-V3 yet.
- Blocker A: a compact experts call. Device memory must follow the routed or cached experts, with
  no dummy slots, unloaded experts never dereferenced, exact routing semantics, repeated experts
  handled once, prefill and decode with the KV cache, and peak VRAM measured.
- Blocker B: split gate/up checkpoints read in place, with no conversion, no duplicated files and
  no preprocessing proportional to the model; one logical expert from several file ranges;
  auditable reads with 4 KiB accounting.
- A reference-profile abstraction (BF16, FP16, native quantized). "Exact" means exact relative to
  the declared fully materialized reference. Decision 0001 must not be replaced globally, and FP8
  certification is not in scope.
- An OLMoE adapter on the generic backend; validation of router output, selected experts, expert
  weights, dispatch, expert outputs and aggregation against the Hugging Face reference.
- If the reference does not fit on the GPU, a CPU or offload reference that still compares
  exactly.
- Gates: correctness (0 mismatches of tokens, routing, expert outputs, KV cache), A (out of VRAM),
  B (selective physical I/O), C (compact residency), D (a model-general core). Two reproducible
  runs.
- Profile, name the top three bottlenecks, and do not over-optimize. Record what was reused from
  DwarfStar, ds4 and Soup.

## Decision

1. **Model: `allenai/OLMoE-1B-7B-0924` at `6d84c485`.**
   - 16 layers of 64 experts, 8 routed per token; 6.9 B parameters; Apache 2.0.
   - Its experts are 12,884,901,888 bytes (12.9 GB) in BF16, against an 8 GB GPU (8.59 GB).
   - The checkpoint is stored in BF16, the reference dtype, so expert bytes are streamed as
     published. The newer `0125` base checkpoint is stored in FP32 (27.7 GB): its bytes are not
     the BF16 reference's, so streaming it would need conversion.
   - Each expert is three tensors (`down_proj`, `gate_proj`, `up_proj`, 4 MiB each, adjacent in
     the file). Data offsets are not 4 KiB aligned, and some experts straddle two shards.
2. **Reference profiles (`awpmi.profiles`).**
   - `ReferenceProfile`: kind (`BF16_REFERENCE`, `FP16_REFERENCE`, `NATIVE_QUANTIZED_REFERENCE`),
     the stored weights' dtype, the compute dtype, the experts and attention kernels, and the
     numerical environment.
   - "Exact" means the same tokens and routed experts, and bit-for-bit the same tensors, as the
     declared reference executed with every weight materialized. Where bytes live is not part of
     a profile and must not change any result.
   - `check_model` raises if the model's dtype or kernels differ from the declaration.
     `check_weights` raises if a streamed weight's stored dtype differs: streamed bytes are used
     as stored, never converted on the fly.
   - `NATIVE_QUANTIZED_REFERENCE` is declared, not implemented: `check_model` refuses it. A
     DeepSeek-V3-class model would be compared against its native FP8 execution, not a BF16
     dequantization.
   - Decision 0001 is unchanged: its certificate bounds the `BF16_REFERENCE` logits.
   - OLMoE's profile is BF16 with `grouped_mm` experts and SDPA attention, transformers' defaults.
3. **The reference that does not fit: `FullLayerOffload`, in its own process.**
   - transformers loads the model as published, every weight in host memory (14.8 GB).
   - The non-expert weights move to the device. Each experts module's own parameters stay in
     host memory (pinned) and are copied whole to the device before the module runs; its forward
     runs unchanged. That is the fully materialized model, one layer at a time (Soup's layer
     streaming). The experts of 5 layers stay resident to save time.
   - Residency must not change results. The first prompts also run with every layer offloaded
     and must agree on every step, in every digest.
   - The reference is independent of Weightsift's I/O path:
     - its weights come from transformers' loader;
     - the source files are re-hashed with direct reads and compared with the sha256 the Hub
       declares;
     - every row of Weightsift's expert index, read from the drive, is compared byte for byte with
       the loader's fused expert tensors.
   - `device_map` offloading was not used: it needs `accelerate`, and transformers' disk offload
     re-saves converted experts, which is a full copy.
4. **The compact experts call (Blocker A), `StreamedExperts(compact=True)`.**
   - Between calls, the expert-sliced parameters are `None`: the full tensors exist nowhere, and
     any use raises. A `meta` placeholder was rejected: given a meta weight, the CUDA grouped GEMM
     returned NaN garbage instead of failing.
   - The pre-hook works in five steps:
     1. takes the routed experts R from `top_k_index` (unique and ascending, so an expert chosen
        by many tokens is loaded once);
     2. allocates buffers [|R|, …] and assembles R into them (`ExpertStore.assemble`; row i holds
        expert R[i]);
     3. sets the parameters to those buffers and `num_experts` to |R|;
     4. replaces `top_k_index` by each expert's slot (sentinels outside [0, E) map to |R|);
     5. runs the module's own forward. A forward hook (`always_call`) then restores `None` and
        the expert count.
   - **Slots keep ascending expert order.** That keeps every experts implementation bit for bit:
     - eager accumulates experts in loop order;
     - `grouped_mm` sorts by id and sums each token's top-k in their own order;
     - `batched_mm` gathers per token.

     A probe at OLMoE's shapes on this GPU (54 trials each, 1–128 tokens) found ascending slots
     exact for all three. Shuffled slots differed for eager on 54 of 54 trials.
   - Device memory follows the served experts: |R| × expert bytes per call, allocated per call.
     Poison mode adds 2 spare slots and fills every slot with NaN before assembly. The spares
     must never be read, and assembly must write every routed byte. Sabotage check: NaN in a
     served slot changes the output.
   - `all_experts` (dense streaming, every expert every step) uses the same path.
   - `ExpertCall` exposes each call before its weights are released. `per_assignment_outputs`
     re-runs the module on the same weights, with each (token, k) assignment as a token routed
     alone with weight 1. That groups the same rows per expert as the call, and multiplying by 1
     and summing one term are exact. It gives a direct check of every expert's output.
   - Limitation: a prefill that routes every expert needs one layer of compact buffers (805 MB
     here; 11 GB for a DeepSeek-V3 layer). Bounding it means splitting a call over experts and
     recombining exactly. The combine is the experts implementation's own (for `grouped_mm`:
     weight, unpermute, then a sum over top-k as [T, K, H]), so this is left for the
     DeepSeek-class step. The same piece enables ds4's compute-level hits-first.
5. **Split checkpoints in place (Blocker B): composed segments.**
   - Storage core (`awpmi.storage.layout.ComposedSegment`): a row is the concatenation of its
     parts, part p being `part_bytes[p]` bytes at (file, offset) `spans[r][p]`. Rows may lie in
     different files.
   - Read plans (`plan_reads`):
     - one run per span, merged where file and output are both contiguous;
     - sorted by file and offset; extents never cross files;
     - every run carries its output offset, and `positions` can scatter rows into a caller's
       buffer.
   - A plain segment's plan is unchanged.
   - The 4 KiB accounting, the OS cross-check and the no-hidden-reads property hold and are
     tested on three files with random spans.
   - Packs: manifest version 2 adds composed segments; a pack without them is still written as
     version 1, byte for byte.
     - An index is built from safetensors headers alone (`awpmi pack expert-index`: a 372 KB
       manifest for 12.9 GB of experts; nothing copied).
     - A source file may carry the sha256 its publisher declares (the Hub's LFS digest) instead
       of one computed by reading it.
     - Composed segments carry no digest. `open_pack(verify="files")` re-hashes files with
       direct reads (0.42 GB/s on this CPU).
   - Layout (`awpmi.models.checkpoint.expert_sources`): derived from the model's transformers
     conversion mapping, the loader's own declaration.
     - Only per-expert stacking (`MergeModulelist(dim=0)`), optionally followed by concatenation
       along each expert's first dimension (`Concatenate(dim=1)`), is understood. For row-major
       tensors that concatenates bytes.
     - Literal key renamings are reversed when a checkpoint uses older names (Mixtral's
       `block_sparse_moe`).
     - Anything else (transposes, interleaving, dequantization) is refused.
     - Every shape and dtype is checked against the model.
     - Tests: save_pretrained checkpoints of Mixtral, Qwen2-MoE, Qwen3-MoE, OLMoE and
       DeepSeek-V3 are served in place, bit for bit.
6. **Loading everything but the experts (`load_model_without_experts`).**
   - The skeleton is built on `meta` from the config, with the kernels `from_pretrained` would
     pick.
   - Every non-expert parameter and persistent buffer is the checkpoint tensor of the same name,
     read with direct I/O and cast by transformers' own rule: its dtype plan (e.g.
     `_keep_in_fp32_modules_strict`, which DeepSeek-V3's router bias needs), then the declared
     dtype.
   - Non-persistent buffers (rotary frequencies) are recomputed by the model's `_init_weights`.
   - An expert parameter the dtype plan would cast is refused.
   - The streaming process never maps the checkpoint through the OS cache (decision 0006's NTFS
     finding), and never holds an expert tensor: 0.95 GB loaded in 1.2 s, a 2.7 GB peak of
     host memory, against 14.8 GB for transformers' loader.
7. **Device memory under a budget.**
   - The streaming process runs under an allocator cap (`torch.cuda.set_per_process_memory_fraction`)
     of 6.0 GB, below both the experts (12.9 GB) and the card.
   - Finding: long-lived cache pages scattered among short-lived compact buffers of varying sizes
     fragmented the caching allocator. The first capped run failed with 3.92 GiB allocated and
     1.61 GiB reserved but unusable.
   - Cache entries are therefore allocated from a CUDA memory pool of their own
     (`MaterializationBackend`). Pages reuse each other's memory within the cache's budget, and
     the rest stays contiguous for working buffers.
8. **Materialization and transfer.**
   - `materialize(segment, rows, out=buffer)` writes into a caller's device buffer: compact
     buffers are filled in place.
   - Cache hits are copied first on the compute stream, so the copies run while the misses are
     read (ds4's "hits first", at the copy level). Misses are streamed straight into their rows.
   - `PageStreamer.fetch(…, out, positions)`: the copy stream first waits for the caller's
     stream, which may still be reading that memory. A piece whose runs scatter is copied once per
     contiguous destination range.
   - Profile-driven:
     - runs of at least 256 KiB are copied straight from pinned staging, without a host gather;
     - the OLMoE configuration uses 32 MiB staging slots (8 MiB in Phase 3).

     A decode step without a cache fell from 969 to 740 ms in the profile, with the drive at
     94% of its sequential rate.
   - Phase 3 regression check, rerun on its first prompts against the committed run1 records:
     - Granite: 249 of 249 records identical;
     - LM head: 159 of 160. The other one differs only in the transfer counters
       `gathered_bytes` and `h2d_copies`, the effect of direct copies. Bytes read and moved are
       unchanged.
9. **The OLMoE adapter (`awpmi.models.olmoe`).**
   - `EXPERT_LAYOUT` (gate rows first, then up) is checked against the layout transformers
     declares, so a change on either side is caught.
   - `routers` gives each layer's router, so router logits are recorded where routing is
     decided.
   - `REFERENCE_PROFILE`.
   - Nothing else is OLMoE-specific.
10. **Benchmark (`benchmarks/olmoe_runtime.py`), two processes.**
    - Reference stage: verify the files, load, audit the index, run the residency check, then
      greedy decoding with the KV cache.
    - Stream stage: a fresh process under the cap; every configuration.
    - Per step and layer, both stages record digests of:
      - the router logits;
      - the routed experts;
      - the top-k indices and weights;
      - every (token, expert) output before weighting;
      - the experts module's output.

      Per step, also the logits, the token and the whole KV cache.
    - The stream stage compares everything and audits every step:
      - requested = served experts;
      - cache + fetched = requested;
      - storage = fetched;
      - device received = storage;
      - every block read holds a requested byte;
      - OS counters = store;
      - compact buffers = served experts.
    - Raw extents are traced for the first prompts of each configuration.
    - The two runs use different `PYTHONHASHSEED` values. Step times in records include the
      digests; `benchmarks/olmoe_profile.py` measures without them.
11. **Gates, fixed in `configs/phase4a-olmoe.yaml` before the full runs.**
    - Correctness: every step of every configuration equal in every digest; every audit clean;
      sources, index and residency checks hold.
    - A: experts greater than both the 6.0 GB cap and the device; every configuration completes.
    - B: decode drive bytes without a cache ≤ 0.15 of all expert bytes, and ≤ 0.20 of what
      streaming every expert reads.
    - C: compact buffers audited on every step; ≤ 0.20 of a layer on decode steps; working set ≤
      cache capacity plus one call's buffers.
    - D: `tests/test_layering.py`, now also forbidding `gate_proj`, `up_proj`, `num_experts` and
      index-file names in the core. It caught two docstrings that named the model during
      development.
12. **External systems: reused conceptually, implemented independently.** No runtime dependency.
    - Soup (layer streaming):
      - pinned staging, copy streams and events, double buffering (Phase 3);
      - layer-at-a-time materialization of a resident-in-RAM model, which is the reference
        executor here.
      - Not reused: next-layer prefetch. A layer's experts are unknown until its router runs.
    - ds4:
      - a pool of positioned reads (8 threads, Phase 3);
      - hits first, adapted at the copy level only. ds4 also computes hits before misses and sums
        partials in slot order, which it can because it owns its combine;
      - its prefill finding (LRU thrash, issue #1119, fixed by an admission freeze) is measured
        here: prefill routes 57 of 64 experts per layer.
    - DwarfStar:
      - an expert cache with a budget in whole experts, hotness and an admission freeze
        (Phase 3);
      - the separate memory pool here plays the role of its fixed expert arena.
    - Implemented independently: the compact call, composed segments and their plans, layouts
      derived from transformers' mapping, the expert-free loader, the offload reference, the
      reference profiles, and every check.

## Results

Full report: `history/2026-10-03-awpmi-phase4a-report.md`.

- **Correctness** (OLMoE-1B-7B, 40 prompts × 13 steps, two runs with identical digests):
  - all 2,089 streamed steps per run are equal to the fully materialized reference in every
    recorded digest: token, logits, KV cache, routed experts, and per layer router logits,
    top-k indices and weights, every (token, expert) output and the experts output;
  - 39 of them ran with NaN-poisoned slots;
  - every audit and OS cross-check holds;
  - all 2,048 index rows equal the loader's tensors; the residency check agreed on 26 of 26
    steps.
- **A:** experts 12.88 GB, 2.15× the 6.0 GB cap and 1.50× the GPU. Peak 5.04 GB allocated;
  every configuration completed.
- **B:** a decode token without a cache reads 0.1251 of the expert bytes (1.61 GB), 0.125× of
  streaming every expert (1.0006). With caches of 1/8 and 1/4: LRU 0.0819 and 0.0672, hotness
  1/4 0.0658. Amplification is 1.00065.
- **C:** compact buffers are |R| × 12 MiB, audited on every step: 101 MB per decode call (126 MB
  with poison spares, 0.156 of a layer), 805 MB in a prefill that routes every expert.
- **D:** the layering test passes.
- **Time** (not gated; profile without digests): a decode step takes 759 ms without a cache and
  525 ms with LRU at a quarter of the experts. The top three bottlenecks:
  - drive reads, 67%, at 94% of the drive's rate;
  - the transformer's Python and launches, 20%: 2,847 launches for 20 ms of GPU work;
  - per-request transfer overhead, 13%.

## Rejected

- *Full-shape slot buffers* (Phase 3): E × expert bytes per layer.
- *Compact slots in cache order* (a slab indexed by cache slot): not exact for eager experts.
- *`meta` placeholders between calls*: silent garbage on CUDA instead of an error.
- *A converted or repacked checkpoint*: one more copy of every expert (12.9 GB here; about
  690 GB in FP8 for DeepSeek-V3), against the brief.
- *Hashing every segment when indexing*: proportional to the model. Replaced by the publisher's
  file digests, direct-read verification on demand, and the benchmark's row-by-row audit
  against the loader.
- *`device_map` / accelerate offloading*: a new dependency, and its disk offload re-saves
  converted experts.
- *Splitting calls over experts (bounded prefill buffers) and compute-level hits-first now*: both
  need an exact replica of each experts implementation's combine. Next, for DeepSeek-class
  layers. **Bounded calls adopted in decision 0008 (Phase 4B)**, without a replica: the experts
  matrices become stand-ins that compute the grouped GEMM chunk by chunk, and the
  implementation's own combine runs once per call.
- *Speculative next-layer or previous-step prefetch now*: a decode step reuses 37.6% of the
  previous step's experts in a layer, so most of a prefetch would be wasted.
- *An allocator-wide setting (expandable segments) against fragmentation*: environment-wide,
  unverified on this platform. A pool for the cache fixes the cause.
- *FP8 / native quantized certification*: declared as a profile only.
