# 0008 — Phase 4B: Moonlight out of VRAM and host RAM, bounded chunked expert calls, an independent streaming reference

Status: accepted (Phase 4B, 2026-10-03)

## Context

Phase 4A (decision 0007) ran OLMoE-1B-7B out of VRAM, bit for bit, with two limits that rule out
DeepSeek-class models:

- a prefill that routes every expert of a layer needs one layer of compact buffers (805 MB for
  OLMoE; 11 GB for DeepSeek-V3);
- the reference (`FullLayerOffload`) holds the whole model in host memory (15.1 GB for OLMoE).

The user's Phase 4B brief:

- Run `moonshotai/Moonlight-16B-A3B`, DeepSeek-V3's architecture at a checkable size, although it
  fits neither the GPU nor, safely, host RAM, reproducing the declared fully materialized reference
  exactly. Not DeepSeek-V3 itself.
- Blocker A: an independent streaming reference. It reuses transformers' model implementation (no
  second transformer), materializes each layer's full operation and releases it, keeps host RAM
  bounded, and shares nothing with Shardraw's selective path. It must itself be validated against
  `from_pretrained` where that fits: router, expert, attention outputs, KV cache, logits, bitwise.
- Blocker B: bounded chunked expert execution under an explicit byte budget, with the reference's
  exact accumulation order (inspect transformers' combine first), shared experts, repeated
  experts, prefill and decode with the KV cache; adversarial tests showing that reordering the
  accumulation changes results.
- Shared experts measured separately and placed (resident, streamed or cached) on measured cost.
- DeepSeek-V3 routing validated per layer (logits/scores, selected experts, weights, shared
  contribution). Reference profiles kept; FP8 not implemented.
- Hard guards against hidden full materialization; host-RAM measurements for both processes,
  separating OS file-cache effects from process allocations; the Phase 3/4A I/O accounting.
- Cache experiments (none, LRU small and medium, hotness medium), prefill gated separately,
  correctness on every step, two runs with different `PYTHONHASHSEED`.
- A serious profile with percentages and the top three bottlenecks; a native-runtime
  classification (Python, Rust, C++/CUDA, already native); no custom GEMM without evidence; I/O
  overlap only if measured to help. ds4, DwarfStar and Soup reviewed again.
- No AWPMI inside experts yet, but the exact hook where it would go.
- Gates: correctness, A (out of VRAM), B (out of RAM), C (bounded prefill), D (selective I/O), E
  (exact chunking), F (generic core).

## Decision

1. **Model: `moonshotai/Moonlight-16B-A3B` at `476b36a4` (MIT).**
   - transformers' native `DeepseekV3ForCausalLM` (no remote modeling code): 27 layers, the first
     dense, then 26 MoE layers of 64 routed experts (6 per token) and 2 shared experts; MLA
     attention (KV LoRA rank 512, no query LoRA); a 163,840-token vocabulary.
   - Stored in BF16, router bias included: 31.92 GB in 27 files, one per layer.
     - Routed experts: 28,789,702,656 bytes (28.8 GB). An expert is three 5.5 MiB tensors (down,
       gate, up, adjacent); a layer of experts is 1.107 GB.
     - Shared experts: 899,678,208 bytes; everything else: 2,230,843,008 bytes.
   - Against this machine: 32 GB of RAM (the checkpoint does not fit next to the OS), an 8 GB GPU,
     the streamed process capped at 6.0 GB.
   - The official tokenizer is remote code (`tokenization_moonshot.py`, tiktoken): read before use,
     run at the pinned revision with `trust_remote_code`. `tiktoken` 0.14.0 was added as a
     dependency for it.
2. **Reference profile: `BF16_REFERENCE` with `grouped_mm` experts and SDPA attention**, the
   defaults transformers 5.18 picks.
   - On this GPU (sm_89), `torch._grouped_mm` runs one cuBLAS GEMM per group (traced: one kernel per
     non-empty group, chosen by its row count; a zero-filled output first).
   - The router bias is stored in BF16 and kept in float32 by transformers' dtype plan
     (`_keep_in_fp32_modules_strict`); both loaders apply it.
   - `moonlight.ROUTING` states the routing configuration this was validated for (sigmoid,
     `noaux_tc`, one group, top 6, renormalized, × 2.446, two shared experts, one dense layer), and
     `check_config` refuses any other: transformers hard-codes that routing.
3. **The independent streaming reference (Blocker A): `awpmi.streaming_reference.StreamingReference`.**
   - transformers' model, built on `meta` from the configuration with the declared kernels.
   - Every non-expert weight is loaded once by `convert_and_load_state_dict_in_model`, the function
     `from_pretrained` calls, from safetensors slices opened as `from_pretrained` opens them
     (`pread` on Windows), with the model's dtype plan and conversion mapping; then the model's own
     `_finalize_model_loading` (non-persistent buffers, initialization of what checkpoints never
     hold).
   - Each experts module gets a pre-hook that loads its layer's checkpoint tensors through the same
     function: transformers' own per-expert stacking and gate/up concatenation, on the device.
     After the module ran, a hook releases them; between calls the parameters are `None`. Which
     checkpoint tensor belongs to which experts module is transformers' own renaming of its key.
   - Independence: no awpmi storage, index, cache, compact or chunked call, routed-only
     materialization or Shardraw-derived layout; `tests/test_layering.py` forbids the imports. The
     reference process records the sha256 of every expert row it loaded; Shardraw's index is
     audited against those digests in the other process, so neither reads through the other's
     path.
   - Validation where `from_pretrained` fits:
     - tests: save_pretrained checkpoints of DeepSeek-V3, Qwen3-MoE and Mixtral (Mixtral's keys
       renamed), CPU and CUDA: every weight and buffer, and every attention output, experts call
       (inputs, output, per-assignment outputs), MoE block output, KV cache and logits, equal;
     - Moonlight itself, truncated to its first 4 layers (the dense layer and 3 MoE layers),
       `from_pretrained` resident against the streaming reference
       (`benchmarks/moonlight_reference_check.py`): 57 weights (experts included) and 20 steps of 4
       prompts (16 to 1,024 tokens), every recorded quantity, equal.
   - Residency check: the first prompts also run with an experts layer kept resident; equal.
   - Cost: transformers' loader reads the checkpoint at 1.0–1.1 GB/s here (safetensors' buffered
     `pread`), so a step that loads 26 layers takes about 30 s. Accepted: the reference is for
     correctness.
4. **Bounded experts calls (Blocker B): chunks, with the implementation's own combine.**
   - `StreamedExperts(compact=True, max_call_bytes=B)`. A call whose buffers would exceed B splits
     its served experts (ascending) into chunks of ⌊B / expert bytes⌋ consecutive slots. Small
     expert-sliced parameters (biases) are assembled whole; each expert matrix becomes a
     `ChunkedExpertWeight` for the call.
   - The stand-in is not a tensor. The module's own forward reaches the weights through exactly two
     operations:
     - `weight[slot]` (eager, one expert at a time, ascending): the chunk holding the slot is
       materialized, the previous one released;
     - `torch._grouped_mm(x, weight or weight.transpose(-2, -1), offs=…)`, through PyTorch's
       `__torch_function__` protocol: chunk after chunk, each chunk's groups are computed by one
       call on their rows, into one output; rows after the last group stay zero, as in the
       per-group fallback.
   - Anything else raises: other operations, other indices, spare slots, use after the call. The
     release hook requires every chunk of every matrix to have been materialized exactly once.
   - Everything else in the forward is transformers' code, run once on the whole call: the sort by
     expert, offsets, gating, routing weights and the combine.
   - Chunks hold at most B (plus the small parameters); `grouped_mm` holds one matrix's chunk at a
     time. B = 256 MiB here: 15 experts; decode calls (6 experts, 8 with poison spares) are never
     chunked.
5. **Exact accumulation.**
   - transformers 5.18's combines:
     - `grouped_mm`: each assignment's output (BF16) is multiplied by its float32 weight (float32
       rows), unpermuted, viewed as [T, K, H] and summed over K in float32, then cast to BF16 once;
     - eager: `index_add_` into a BF16 accumulator, experts in ascending order.
   - The chunked path keeps both, because the combine runs once on the whole call.
   - Exactness then rests on one kernel property: a group's product must not depend on the other
     groups of the call. It holds where `_grouped_mm` runs one GEMM per group. Here a probe found
     chunked equal to full in 160 of 160 trials (64 experts, 1–300 rows per group, chunks of 1–33);
     the tests check it on 7 architectures × CPU/CUDA × both implementations × chunks of 1 and 3.
     The eager path changes nothing but when weights exist.
   - Adversarial test: combining per chunk (summing chunk partials) differs from the reference, for
     both implementations: `grouped_mm` rounds twice, eager reassociates.
   - The ds4 parallel: its hits-first sums partials in slot order to stay byte-identical. Here the
     order chunks are computed in is free for `grouped_mm`, since nothing is summed until the end
     (a lever for compute-level hits-first; eager must stay ascending).
6. **Shared experts: resident.**
   - They are a dense MLP, not an experts module; they stay with the non-expert weights (0.90 GB on
     the device) and transformers adds their output to the routed experts' in BF16.
   - Measured alternative, `awpmi.models.streamed.StreamedParameters` (generic: any parameters
     served whole from a `WeightStore` at every call of their module, `None` between calls), in
     configuration `shared-streamed`. Every step equal to the reference; each step reads the
     0.90 GB of shared experts (936 reads), which costs 437–462 ms per step against the resident
     configuration on the same prompts (both runs): about 290–300 ms of drive time plus 78 more
     requests. A decode step goes from 1.30 to 1.74 s (instrumented).
   - Every token uses them, so a cache would hold them at a hit rate of 1: residency is the same
     bytes at no reads.
7. **Hard guards against hidden full materialization.**
   - Expert parameters `None` between calls (compact, chunked, and in the reference);
     `ChunkedExpertWeight` closed after its call; a budget guard (`_hold`) raises if a chunked
     call's buffers exceed B.
   - Tracked per step: compact/chunk buffer peak, the largest single materialization request
     (`MaterializationStats.largest_request_bytes`), expert bytes on the device (cache + buffers),
     host staging.
   - A CUDA test runs a prefill that routes every expert: chunked, the allocator's peak stays below
     half a layer; unbounded, it holds the layer (the guard sees it).
8. **Host memory: working set and host commit, not private bytes.**
   - Finding: under Windows' WDDM driver model, every byte of device memory a process allocates is
     charged to its private commit (measured: +2 GiB on the device, +2.15 GB of private bytes, the
     working set unchanged). Private bytes are therefore not a host-RAM measure.
   - Gate B uses the working set (physical memory) and the host commit (private bytes less the
     device memory torch's allocator holds); private bytes are reported.
   - The reference reads through the OS file cache (`pread`); the streamed process uses direct I/O.
     The OS's file cache is reported system-wide, outside both processes.
9. **Device budget and the caches.**
   - Non-expert weights take 3.21 GB of the 6.0 GB cap. A 1,024-token prefill's working buffers
     take 0.6 GiB with the call budget, and the cache's memory pool fragments: rows of 11 and
     5.5 MiB leave about 0.4 GiB reserved beyond the cache's bytes.
   - Measured in development: a cache of 120 experts ran out of the cap; 96 peaked 10 MiB below it;
     80 peaks at 5.27 of 5.59 GiB reserved.
   - Configurations: LRU 40 and 80 experts, hotness 80 (2.4% and 4.8% of the 1,664 experts),
     against 156 expert loads per decode token.
10. **Benchmark (`benchmarks/moonlight_runtime.py`), stage by stage.**
    - Reference stage: the streaming reference; the residency check; per step and layer, digests of
      the attention output, the router's logits, scores, indices and weights, the experts call's
      inputs and output, every (token, expert) output (re-run on the full layer), the shared
      experts' output, the MoE block's output, the dense MLP; logits, token, KV cache; and the
      sha256 of every expert row.
    - Stream stage, under the cap:
      - the 27 files re-hashed with direct reads against the Hub's sha256;
      - all 3,328 index rows read through Shardraw and compared with the reference's digests;
      - every configuration, every step compared and audited (requested = served experts; cache +
        fetched = requested; storage = fetched; device = storage; every block read holds a
        requested byte; OS counters = store; chunk buffers within the budget).
    - Chunked calls' per-(token, expert) outputs are checked by rereading their experts through a
      separate uncached store after the step's accounting: every step of the configurations
      `stream`, `compact` and `chunk-1`, and the first prompts of the others. Unchunked calls are
      checked on every step, from their live buffers.
    - Prompts: wikitext-2 paragraphs cut to 16–1,024 tokens (16 prompts), 8 greedy decode steps.
11. **Gates, fixed in `configs/phase4b-moonlight.yaml` before the full runs.** B and the cache sizes
    were set from the development runs (the WDDM finding, the cap), before any full run.
12. **I/O overlap: investigated, not adopted.**
    - Prefill: the experts' GEMMs take 142 ms of a 1,024-token prefill's 11.45 s (traced). Reading
      chunk N+1 while chunk N computes can hide at most that, about 1%. The drive is busy 78.6% of
      a prefill step; the rest is host work (chunk handling, copies, planning) that a native path,
      not overlap, would shrink.
    - Decode: a layer's experts are unknown until its router runs. 41.2% of a decode step's
      experts were routed at the previous step in the same layer, so a speculative prefetch would
      waste most of its bytes on a drive already busy 66.5% of the step.
    - Within one fetch, the streamer already overlaps the read of piece k+1 with the copy of piece
      k: the 0.53–0.57 s of host-to-device copies per decode step run while the drive reads.
13. **External systems** (no dependency; reviewed again):
    - **ds4 / DwarfStar 4** (antirez; `stefandsl/DwarfStar`, cited in decision 0007, is a fork of
      it):
      - hits-first (PR #1082): cached experts computed while misses are read, partials summed in
        slot order from +0.0f. Here the copy-level hits-first of 4A is kept; the deferred combine
        of the chunked `grouped_mm` path makes compute-level reordering exact without a slot-order
        sum. Not built yet.
      - LRU thrash in prefill (issue #1119): reproduced at Moonlight's scale (below).
      - a budget in whole experts: as Phase 3's cache.
    - **Soup**: layer-at-a-time execution with copy streams: the reference executor's pattern
      (now one layer from the checkpoint, not from RAM).
    - **Implemented independently**: the chunked call and its stand-ins, the streaming reference
      through transformers' own loader, `StreamedParameters`, the cache replay, every check.
14. **Where AWPMI goes inside experts (design note).**
    - The one place an expert's weights meet its tokens is the chunked path:
      `_ChunkedCall.materialize` (which rows) and `_chunked_grouped_mm` (the product, per chunk of
      groups). An AWPMI expert executor replaces "materialize every row of the chunk's experts, then
      one grouped GEMM" with "materialize neuron pages progressively, propagate enclosures through
      the expert MLP (Phase 2's operators), stop when the certificate holds".
    - The combine stays the reference's: an enclosure of each (token, expert) output, then the
      float32 [T, K, H] sum and the BF16 cast bounded like any other reduction.
    - Addressability: a composed segment's row is gate rows then up rows (each H × 2 bytes,
      contiguous: a neuron's gate and up rows are 4 KiB runs at H = 2,048), and down as [H, I]
      row-major. A neuron's down column is therefore strided. Either all of down is read (a third
      of the expert), or down is paged by output rows, or a neuron-major down is written (a copy,
      against the zero-copy rule). That layout choice comes first.
    - First target: the last MoE layer's routed experts on the last position, upstream exact (Mode A,
      as Phase 2's suffix), where the existing pairwise certificate applies directly.
    - *Measured in Phase 5A (decision 0009), as an oracle:* under the certified rounding model no
      token certifies with any routed byte unread, in any of these layouts; the executor is not built.
15. **Native runtime: classification from the profile** (decode 1,273 ms and prefill 8,364 ms
    without a cache, un-instrumented). No native code was written in this phase.
    - *Already native through PyTorch*:
      - the GEMMs (one cuBLAS GEMM per expert group; the experts' GEMMs are 23–37 ms of a decode
        step, about 2%);
      - SDPA attention, the elementwise kernels;
      - the host-to-device copies (overlapped with the reads).
    - *Move to Rust*: the host I/O path, 11% of decode and 18% of prefill:
      - read planning (43 / 180 ms);
      - issuing copies (46 / 414 ms) and assembly (37 / 246 ms);
      - the chunked call's per-chunk host work (694 ms per prefill: per-chunk syncs and launches);
      - direct-read submission (8 Python threads calling `ReadFile`);
      - cache bookkeeping: admission 30–53 ms per decode step and up to 564 ms per prefill;
        hotness eviction is a Python scan, which a host-RAM tier of a thousand experts would make
        prohibitive.
    - *Move to C++/CUDA*:
      - the launch-bound transformer in decode, 22%: 5,195 kernel launches for about 150 ms of
        kernels, the GPU idle 54–59% of the step. CUDA graphs fit attention, norms, routers and
        shared experts, whose shapes are static in decode;
      - a grouped GEMM with fewer launches (today one GEMM per group, plus a zero-fill), whose
        per-group results must stay those of the reference (re-verified);
      - on-device routing to compact slots (it removes the routing sync).
    - *Keep in Python*: adapters, configuration, the reference, audits and benchmarks, and the
      per-call chunk planning (one decision per layer).

## Results

Full report: `history/2026-10-03-awpmi-phase4b-report.md`.

- **Correctness** (16 prompts of 16–1,024 tokens × 9 steps, two runs with identical digests):
  - all 726 streamed steps per run, in 8 configurations, are equal to the streaming reference in
    every recorded digest: token, logits, whole KV cache, routed experts, and per layer attention
    output, router logits, scores, indices and weights, the experts call's inputs and output, the
    shared experts' output, the MoE block's output, the dense MLP;
  - every (token, expert) output was checked in 17,732 of 18,876 experts calls: every unchunked
    call, and every chunked one in the configurations `stream`, `compact`, `chunk-1` and
    `all-experts` (the first 2 prompts of the others);
  - 27 steps ran with NaN-poisoned spares and chunks; every audit and OS cross-check holds;
  - the 27 files equal the Hub's sha256; all 3,328 index rows equal the reference's;
  - the residency check agreed on 18 of 18 steps.
- **A:** routed experts 28.79 GB, 4.80× the 6.0 GB cap and 3.35× the GPU; every configuration
  completed under the cap (peak 5.43 GB allocated).
- **B:** peak working set 3.25 GB streamed, 2.84 GB reference (0.113 and 0.098 of the experts);
  host commit 4.36 and 3.07 GB. Private bytes, device memory included (WDDM): 9.7 and 10.7 GB.
- **C:** 654 steps under the 256 MiB budget, none over it; at most 173 MB of expert buffers (0.156
  of a layer), in prefills routing up to 64 experts per layer. One buffer per call held
  1,107 MB, the whole layer.
- **D:** a decode token reads 0.0938 of the expert bytes (2.70 GB), 0.094× streaming every
  expert.
- **E:** `compact` (72 steps, no chunk) and `chunk-1` (36 steps, every call in chunks of one
  expert, decode included) equal the reference, as `stream` (416 chunked prefill calls).
- **F:** the layering test passes.
- **Caches:**
  - LRU with 40 or 80 experts never hits: a decode token loads 156 experts, more than either
    holds.
  - Hotness with 80 hits 15.1% of decode lookups (drive 0.0796 per token).
  - A replay of the routing through the cache reproduces the measured hits on every step, and
    then predicts decode hit rates of 36% / 48% / 67% / 82% / 93% for 160 / 320 / 640 / 960 /
    1,248 experts (LRU; hotness similar). That is 2.8–21.6 GB: host-RAM sizes.
- **Time** (profile, no digests):
  - a decode token takes 1.27 s without a cache and 1.19 s with hotness 80;
  - a prefill takes 5.7–10.4 s for 16–512 tokens, 11.45 s for 1,024 tokens;
  - drive reads are 66.5% of decode and 78.6% of prefill;
  - chunks of 15 experts add 8.5% to a prefill's host time against one buffer per call, chunks of
    one expert 2.4×;
  - the reference takes 29 s per step (1.06 GB/s through transformers' loader).

## Rejected

- *Combining per chunk*: not the reference's association; the adversarial test shows the
  difference for both implementations.
- *A Shardraw replica of the experts forward*: a second implementation of transformers' sort,
  gating, weighting and combine. The stand-in keeps transformers' code and changes only when
  weights exist.
- *Monkeypatching transformers' `_grouped_linear`*: the `__torch_function__` protocol is PyTorch's
  supported way for a non-tensor to receive an operation, and it raises on anything else.
- *A reference through accelerate or transformers' disk offload*: a new dependency, and disk
  offload re-saves converted experts (a 28.8 GB copy).
- *A reference with its own safetensors reader and fusion*: a second conversion to get right.
  transformers' loader is the definition.
- *Speeding the reference with host-resident layers*: bounded, but it weakens the out-of-RAM claim
  for 30 s per step that only costs wall time.
- *Private bytes as the host-memory gate*: WDDM charges device memory to them.
- *Cache budgets near the cap* (120, 96 experts): out of memory, or 10 MiB from it, once the pool
  fragments.
- *Speculative next-layer prefetch*: a layer's experts are unknown until its router runs.
- *A host-RAM expert tier now*: the replay quantifies it (below); it is a new mechanism, for the
  next phase.
- *Custom GEMM or attention kernels*: the GPU computes for a few percent of a step.
- *DeepSeek-V3 and FP8*: out of scope; `NATIVE_QUANTIZED_REFERENCE` stays declared only.
