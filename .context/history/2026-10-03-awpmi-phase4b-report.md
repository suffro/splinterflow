# AWPMI Phase 4B report — Moonlight-16B-A3B out of VRAM and out of host RAM, exactly

Date: 2026-10-03 · Follows: `history/2026-10-03-awpmi-phase4a-report.md` (Phase 4A) ·
Decision: `decisions/0008-phase4b-moonlight-out-of-ram.md` ·
Status: **complete. Correctness and gates A–F pass.**

## Outcome in brief

**Yes: Weightsift runs Moonlight-16B-A3B, DeepSeek-V3's architecture at 16 B parameters, although
the model fits neither this GPU nor this machine's host memory, and reproduces an independent
fully materialized reference exactly.**

- **The setup.**
  - The checkpoint is 31.9 GB: 28.8 GB of routed experts, 0.9 GB of shared experts, 2.2 GB of
    the rest. The machine has 32 GB of RAM and an 8 GB GPU; the streamed process runs under a
    6.0 GB device cap.
  - The routed experts are 4.8× the cap and 3.4× the GPU.
  - They are read in place from the published files (an index built from headers, nothing
    copied).
- **The reference.** transformers' own model, loaded by transformers' own loader, one experts layer
  at a time from the checkpoint: a 2.1 GB working set. It shares no code with Weightsift's expert
  path. It was itself checked against `from_pretrained` on Moonlight truncated to 4 layers: every
  weight and every recorded quantity equal.
- **Bounded prefill.** An experts call that would need more than 256 MiB of expert buffers runs in
  chunks of 15 experts, with transformers' own combine once per call. A prefill holds at most
  173 MB of experts, where one buffer per call holds the whole layer (1,107 MB).
- **Correctness.** Every streamed step of every configuration equals the reference in every
  recorded digest:
  - per step: token, logits, whole KV cache;
  - per layer: attention output, router logits/scores/indices/weights, the experts call's inputs
    and output, every (token, expert) output where checked, shared experts' output, MoE block
    output.

  That holds for 726 of 726 streamed steps per run, in two runs with different hash seeds and
  identical digests.

Per decode token (16 prompts × 8 decode steps, run1). Fractions are of all routed expert bytes
(28.8 GB).

| | No cache | LRU, 40 experts | LRU, 80 | Hotness, 80 | Shared experts streamed | Every expert every step |
| --- | --- | --- | --- | --- | --- | --- |
| Read from the drive | **0.0938** (2.70 GB) | 0.0938 | 0.0938 | **0.0796** | 0.0938 (+0.90 GB of shared) | 1.0005 |
| Cache hit rate (lookups) | – | 0.000 | 0.000 | 0.151 | – | – |
| Expert bytes on the GPU (max) | 138 MB (6 + 2 spares) | 796 MB | 1,488 MB | 1,488 MB | 104 MB | 173 MB (chunks) |
| Decode step, no digests (profile) | **1,273 ms** | – | 1,302 ms | **1,191 ms** | +437–462 ms (instrumented) | – |
| Steps equal to the reference | 144 of 144 | 144 of 144 | 144 of 144 | 144 of 144 | 36 of 36 | 6 of 6 |

**Gates.**

- Correctness: PASS.
- A (out of VRAM): PASS, 4.8× the cap.
- B (out of RAM): PASS, working sets 0.11 (streamed) and 0.10 (reference) of the expert bytes.
- C (bounded prefill): PASS, at most 0.156 of a layer.
- D (selective I/O): PASS, 0.0938 per decode token, 0.094× streaming every expert.
- E (exact chunking): PASS.
- F (generic core): PASS.

## 1. Questions

The brief's engineering questions, answered in section 18:

1. Can Weightsift execute a DeepSeek-V3-like architecture without keeping all expert weights in RAM
   or VRAM?
2. Can the reference be evaluated without loading the whole model into RAM?
3. Can expert execution be split into bounded chunks without changing the exact result?
4. Can prefill memory stay bounded when many experts are routed?
5. Can the generic backend support this without becoming Moonlight-specific?
6. Are the results deterministic and reproducible?
7. What dominates after scaling from 12.9 GB to 28.8 GB of experts?

## 2. Setup

| Item | Value |
| --- | --- |
| Machine | RTX 4060 Ti 8 GB (8.59 GB; sm_89; WDDM), i7-8700K, 32 GB RAM, Windows 11 |
| Drive | Samsung 990 PRO (PCIe 3.0 x4 here): 3.2 GB/s measured with direct I/O |
| Software | torch 2.14.1+cu130, transformers 5.18.0, safetensors 0.8.0, tiktoken 0.14.0 |
| Model | `moonshotai/Moonlight-16B-A3B` @ `476b36a4`, BF16, 27 files (one per layer): 27 layers (layer 0 dense; 26 MoE layers of 64 routed experts, 6 per token, plus 2 shared experts), MLA attention, 163,840-token vocabulary |
| Experts | 28,789,702,656 bytes routed (an expert: down, gate, up, 5.5 MiB each, adjacent; 16.5 MiB); 899,678,208 bytes shared |
| Profile | `BF16_REFERENCE`, `grouped_mm` experts, SDPA attention (transformers' defaults); `_grouped_mm` runs one cuBLAS GEMM per group on this GPU |
| Expert index | `packs/moonlight-16b-a3b-expert-index`: 52 composed segments (26 layers × gate_up, down), all bytes in place; a 577 KB manifest built in 11 s; files identified by the Hub's sha256 |
| Prompts | 16 wikitext-2 test paragraphs cut to 16, 32, 64, 128, 256, 512, 768 and 1,024 tokens (twice each), Moonlight's official tokenizer; 8 greedy decode steps with the KV cache |
| Streamed process | GPU cap 6.0 GB; non-expert weights (3.21 GB on the device, shared experts included) read with direct I/O; experts from the drive (direct I/O, 8 threads, 32 MiB pinned staging); call budget 256 MiB (15 experts) |
| Configurations | no cache (`stream`); one buffer per call (`compact`, 8 prompts); one expert per chunk (`chunk-1`, 4 prompts); LRU 40 and 80 experts; hotness 80; shared experts streamed (4 prompts); every expert every step (2 prompts × 3 steps) |
| Reference | its own process, no device cap: `StreamingReference`, every experts layer loaded from the checkpoint by transformers' loader when it runs |

## 3. What was built

Decision 0008 has the details.

- **Bounded experts calls** (`StreamedExperts(max_call_bytes=…)`, Blocker B).
  - A call whose routed experts need more than the budget runs in chunks of consecutive experts.
    Each expert matrix is a `ChunkedExpertWeight` for the call, a stand-in that is not a tensor.
  - transformers' forward reaches the weights only through `weight[slot]` (eager) or
    `torch._grouped_mm` (the `__torch_function__` protocol). The grouped GEMM is computed chunk
    by chunk into one output; everything else, combine included, is transformers' code, run once.
  - Every chunk of every matrix is materialized exactly once; anything else raises.
- **The independent streaming reference** (`awpmi.streaming_reference`, Blocker A).
  - transformers' model on `meta`; non-expert weights through transformers' own loading function;
    each experts layer loaded by the same function in a pre-hook and released after.
  - No Weightsift import (layering test).
- **The Moonlight adapter** (`awpmi.models.moonlight`): the checked layout, routers (float32 bias),
  shared experts, MoE blocks, the routing configuration it was validated for, the profile.
- **`StreamedParameters`** (`awpmi.models.streamed`): any dense parameters served from storage at
  every call. It measures the cost of not keeping the shared experts resident.
- **Smaller pieces**:
  - `ExpertStore.assemble` of some parameters only;
  - the backend's largest-request counter;
  - `checkpoint.parameter_segments`;
  - the adapters' shared layout check (`checked_expert_sources`, `neighbours`).
- **Benchmark** (`moonlight_runtime.py`, `moonlight_reference_check.py`, `moonlight_profile.py`,
  `moonlight_report.py`).

## 4. The streaming reference and its validation

- **What it is.** `from_pretrained`'s own steps, one experts layer at a time:
  - the skeleton on `meta`;
  - `convert_and_load_state_dict_in_model` on safetensors slices opened as `from_pretrained`
    opens them (`pread` on Windows), with the model's dtype plan and conversion mapping;
  - the model's `_finalize_model_loading`.

  For each experts module, a pre-hook loads its layer's checkpoint tensors through the same
  function: transformers' own per-expert stacking and gate/up concatenation. A hook releases
  them after the module ran.
- **Validated where `from_pretrained` fits:**

  | Check | Result |
  | --- | --- |
  | Tests: save_pretrained checkpoints of DeepSeek-V3, Qwen3-MoE, Mixtral (renamed keys), CPU and CUDA | every weight and buffer, every attention output, experts call, per-assignment output, MoE block output, KV cache and logits: equal |
  | Moonlight truncated to 4 layers (dense layer + 3 MoE layers): `from_pretrained` resident vs streaming (`experiments/phase4b/reference-check`) | 57 weights (6 expert tensors) equal; 20 steps (4 prompts of 16–1,024 tokens, prefill + 4 decode) equal in 16 quantities; 60 per-layer expert-output checks |
  | Residency (benchmark, both runs) | 18 of 18 steps equal with an experts layer kept resident |
  | Index audit (benchmark) | the reference's sha256 of every expert row it loaded = Weightsift's index rows read from the drive: 3,328 of 3,328 |

- **Memory and time.**
  - The reference process peaks at a 2.1 GB working set during steps (2.84 GB lifetime, at load)
    and 5.15 GB on the device (one experts layer and its conversion).
  - It reads the checkpoint through the OS file cache at 1.06 GB/s (transformers' loader,
    `pread`): 4,169 s for 144 steps, 29 s per step.

## 5. Correctness

Sources: `experiments/phase4b/moonlight-run1/summary.md` and `moonlight-run2`. Raw records:
`reference.jsonl.gz`, `records.jsonl.gz`, `index.json`, `reference_rows.json`, `ranges.jsonl.gz`.

| Criterion | Result (per run) |
| --- | --- |
| Source files | all 27 re-hashed with direct reads: their sha256 equal the Hub's |
| Expert index | all 3,328 rows equal the reference's digests |
| Residency check | 18 of 18 steps equal |
| Streamed steps equal to the reference | 726 of 726: `stream` 144, `compact` 72, `chunk-1` 36, `lru-40` 144, `lru-80` 144, `hotness-80` 144, `shared-streamed` 36, `all-experts` 6 |
| …in each recorded quantity | token, logits, KV cache, routed experts; per layer attention, router logits, scores, indices, weights, the experts call's top-k index and weights, its output, shared experts' output, MoE block output; the dense MLP |
| Per-(token, expert) outputs | 17,732 of 18,876 experts calls checked: every unchunked call (from its live buffers), every chunked call of `stream`, `compact`, `chunk-1` and `all-experts`, the first 2 prompts of the others (their experts reread through a separate uncached store after the step's accounting) |
| Poison | 27 steps (first 3 prompts of `stream`) with NaN spare slots and NaN-filled chunks: equal |
| Audits | 0 problems: requested = served experts; cache + storage = requested; storage = fetched; device = storage; every block read holds a requested byte; chunk buffers within the budget; the shared experts' own audit in `shared-streamed` |
| OS counters | read calls and bytes equal the store's on every step |
| Reproducible | run1 (`PYTHONHASHSEED=1`) and run2 (`PYTHONHASHSEED=2`): identical index, reference-row, prompt, reference, record and range digests, same source tree |

**Tests:** 510, 58 of them new:

- chunked calls bit for bit against the resident model: 7 architectures × CPU/CUDA × `grouped_mm`
  and eager, chunks of 1 and 3 experts, with poison;
- the adversarial accumulation test;
- the stand-in's misuses and use after the call, a budget below one expert, the allocator never
  holding a layer;
- evictions during chunked calls, per-assignment outputs of chunked calls, dense streaming in
  chunks, a split DeepSeek-V3 checkpoint in chunks, hash seeds;
- the streaming reference against `from_pretrained` (3 architectures × CPU/CUDA), one layer at a
  time, residency;
- the Moonlight adapter: layout check, routers, shared experts, the routing formula, the routing
  configuration, the profile;
- streamed shared experts;
- layering.

**Guards confirmed to fail when sabotaged:**

| Guard | What failed |
| --- | --- |
| Combining per chunk (adversarial) | `grouped_mm` and eager both differ from the reference |
| A chunk materialized twice | the release hook raises (every chunk exactly once) |
| A chunk never assembled (poison on) | the output differs |
| A plan over the budget | the budget guard raises |
| Unbounded buffers in a prefill routing every expert | the allocator test sees a layer-sized allocation |
| A chunked weight used outside its two operations, or after its call | it raises |

## 6. Out of VRAM (gate A)

| | Bytes |
| --- | --- |
| Routed experts | 28.79 GB (28,789,702,656) |
| GPU total | 8.59 GB |
| Cap of the streamed process | 6.00 GB: experts 4.80× the cap, 3.35× the GPU |
| Non-expert weights on the GPU | 3.21 GB (shared experts 0.90 GB), loaded with direct reads in 5.6 s |
| Peak GPU memory, any step | 5.43 GB allocated; the allocator's reserve reached the cap in the 80-expert caches, and every step completed |

**Gate A: PASS.**

## 7. Out of host RAM (gate B)

| Process | Before setup | After the non-expert load | Peak, prefill | Peak, decode | Host commit (peak) | Private bytes (incl. device, WDDM) |
| --- | --- | --- | --- | --- | --- | --- |
| Streamed (all configurations) | 0.63 GB | 1.50 GB | 3.25 GB | 3.25 GB | 4.36 GB | 9.7 GB |
| Reference | – | – | 2.11 GB | 2.10 GB | 3.07 GB | 10.7 GB |

- **Gate B: PASS.** Worst working sets are 0.113 (streamed) and 0.098 (reference, its load
  included) of the routed expert bytes, against a limit of 0.25; host commits 0.152 and 0.107.
- **Pinned host memory**: 128 MiB of staging per backend; at most two backends live (the
  configuration's and the check's).
- **Finding: private bytes are not host memory under WDDM.** Windows charges every byte of device
  memory a process allocates to its commit: +2 GiB on the device was +2.15 GB of private bytes,
  the working set unchanged. The gate uses the working set and the private bytes less the device
  memory torch holds (fixed in the configuration before the full runs).
- **OS file cache** (system-wide, outside both processes): after the reference stage the cache held
  19.9 GB, mostly checkpoint pages read through `pread`. They are opportunistic standby memory,
  not needed by the process, and less than the checkpoint. The streamed process reads with direct
  I/O and does not use it.

## 8. Bounded prefill (gate C)

Prefill, configuration `stream` (chunks of 15 experts) against `compact` (one buffer per call), by
prompt length (means of run1; time instrumented):

| Tokens | Experts / layer | Drive | Expert buffers, chunked | Expert buffers, one per call | Step, chunked | Step, one per call |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 32.0 | 14.4 GB | 165 MiB | 660 MiB | 6.0 s | 5.0 s |
| 64 | 47.9 | 21.6 GB | 165 MiB | 924 MiB | 8.1 s | 7.7 s |
| 256 | 56.2 | 25.3 GB | 165 MiB | 1,040 MiB | 9.9 s | 9.6 s |
| 1,024 | 60.9 | 27.4 GB | 165 MiB | 1,056 MiB | 11.5 s | 12.0 s |

- **Gate C: PASS.**
  - 654 steps ran under the budget, none over it; at most 173 MB of expert buffers, 0.156 of a
    layer.
  - 74 prefills completed, with up to 64 experts routed in a layer. The expert working set
    stayed within the cache's capacity plus the budget.
- One buffer per call reached 1,107 MB, the whole layer: the allocation the budget removes. On
  DeepSeek-V3 that would be 11 GB.
- Peak device memory in prefill: 3.65 GB chunked (no cache), 4.26 GB with one buffer per call;
  5.05 GB with an 80-expert cache.
- The cost, un-instrumented: chunks of 15 experts add 8.5% to a prefill's host time (8.36 against
  7.71 s for the same prompts), mostly one host sync per chunk and per matrix. Chunks of one expert
  cost 2.4× (20.1 s).

## 9. Selective physical I/O (gate D)

Per step, as fractions of all routed expert bytes (mean / median / p95); prefill reads for all of a
prompt's tokens.

| Configuration | Phase | Experts per layer | Drive | Host → GPU | Read calls | Chunked calls |
| --- | --- | --- | --- | --- | --- | --- |
| no cache | prefill | 50.7 | 0.793 / 0.823 / 0.957 | 0.792 | 23,727 | 416 of 416 |
| no cache | decode | 6.0 | **0.0938** / 0.0938 / 0.0938 | 0.0938 | 2,808 | 0 |
| hotness 80 | decode | 6.0 | **0.0796** / 0.0788 / 0.0938 | 0.0796 | 2,383 | 0 |
| one expert per chunk | decode | 6.0 | 0.0938 | 0.0938 | 2,808 | 832 of 832 |
| every expert | decode | 64.0 | 1.0005 | 1.0000 | 29,952 | 104 of 104 |

- **Gate D: PASS.** A decode token reads 0.0938 of the expert bytes (limit 0.11), 0.094× what
  streaming every expert reads (limit 0.12). The excess over 6/64 = 0.09375 is the 4 KiB blocks at
  the ends of unaligned tensors (amplification 1.0005).
- What reaches the GPU is exactly what was requested and not cached.
- The largest single request is 69 MB in decode (a layer's 6 gate/up rows) and 173 MB in a
  chunked prefill.

## 10. Exact chunking (gate E)

- transformers 5.18's combines:
  - `grouped_mm`: weight each row in float32, unpermute, sum the top-k as [T, K, H] in float32,
    cast once;
  - eager: BF16 `index_add_`, experts in ascending order.
- The chunked path leaves both as they are: they run once per call. Exactness then rests on one
  kernel property: a group's product must not depend on the call's other groups.
  - On this GPU `_grouped_mm` runs one cuBLAS GEMM per non-empty group, chosen by its row count
    (traced).
  - A probe found chunked equal to full in 160 of 160 trials (64 experts, 0–300 rows per group,
    chunks of 1–33).
- **Gate E: PASS.**
  - `compact` (no call chunked) and `chunk-1` (every call chunked by single experts, decode
    included) equal the reference on every step, as do all chunked prefills of every other
    configuration.
  - The adversarial test shows that combining per chunk differs, for both implementations.

## 11. Shared experts

| | Resident (all configurations but one) | Streamed (`shared-streamed`) |
| --- | --- | --- |
| Bytes | 0.90 GB on the GPU (26 layers × 3 × 2,048 × 2,816 × 2) | 0.90 GB read per step (936 read calls) |
| Host → GPU per step | 0 (loaded once, 5.6 s for all non-expert weights) | 0.90 GB |
| Step time | – | +437–462 ms per step on the same prompts (both runs; drive 292–301 ms of it) |
| Correct | yes | 36 of 36 steps equal |

**Decision: resident.**

- Every token uses the shared experts in every layer, so any cache would hold them at a hit rate
  of 1, and residency costs no reads.
- Streaming them adds a third to a decode step.
- The 0.9 GB they hold would buy 52 routed experts of cache. The measurements below show that
  such a small cache does not pay.

## 12. Routing and caches

- **DeepSeek-V3 routing.**
  - `moonlight.ROUTING` matches the checkpoint's configuration: sigmoid, `noaux_tc`, one group,
    top 6, renormalized, × 2.446.
  - A unit test recomputes transformers' router from that formula: the same chosen sets and
    weights, and the bias changes the choice, not the weights.
  - Both stages record the router's logits, scores, indices and weights per layer, all equal on
    every step.
- **Routing statistics** (reference, run1):
  - decode routes 6 experts per layer;
  - prefill routes 32 of 64 at 16 tokens, 47.9 at 64, 56.2 at 256, 60.9 at 1,024;
  - the most used 10% of (layer, expert) slots receive 30.3% of decode routings, the top 25%
    receive 55.1%;
  - of a decode step's experts, 41.2% were routed at the previous step in the same layer.
- **Measured caches** (decode / prefill hit rates of lookups):
  - LRU 40: 0 / 0. LRU 80: 0 / 0.
  - Hotness 80: 0.151 / 0.014.

  A decode token loads 156 experts (6 × 26 layers), more than either cache holds. LRU evicts each
  expert before its next use, and prefill floods the cache as in ds4's issue #1119.
- **The device cache cannot grow.**
  - After 3.21 GB of non-expert weights, the cap leaves 2.6 GiB.
  - A 1,024-token prefill needs 0.6 GiB of working buffers with the budget, and the cache's
    pool fragments by about 0.4 GiB (rows of 11 and 5.5 MiB).
  - Development runs: 120 experts ran out of memory; 96 peaked 10 MiB below the cap; 80 peaks at
    5.27 GiB.
- **Replay at larger capacities.** The report replays the recorded routing through `PageCache`,
  with the backend's request order and the real row sizes. On all 144 steps of LRU 40, LRU 80
  and hotness 80 it reproduces the measured hits exactly. Decode hit rates:

  | Experts (GB) | 160 (2.8) | 320 (5.5) | 640 (11.1) | 960 (16.6) | 1,248 (21.6) |
  | --- | --- | --- | --- | --- | --- |
  | LRU | 0.362 | 0.480 | 0.675 | 0.820 | 0.926 |
  | Hotness | 0.302 | 0.500 | 0.688 | 0.830 | 0.934 |

  Caching starts to pay above one token's working set (156 experts): from 160 experts LRU
  catches the previous step's reuse. Those are host-RAM sizes, not GPU sizes on this machine: a
  host-RAM tier of 11–17 GB would cut decode drive reads by 67–83%.

## 13. Time and the top three bottlenecks

From `benchmarks/moonlight_profile.py`: each configuration in its own process; 4 prompts (1,024,
16, 128 and 512 tokens) × 9 steps; no digests. The 1,024-token prompt's prefill and first two
decode steps are traced, and kept out of the means.

| Configuration | Phase | Step | Drive I/O | Other (transformer) | Chunk handling | H2D issue | Plan | Assemble | Admit | Route |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no cache | decode | **1,273 ms** | 847 (66.5%) | 283 (22.2%) | – | 46 | 43 | 37 | – | 16 |
| LRU 80 | decode | 1,302 ms | 848 | 271 | – | 47 | 46 | 44 | 30 | 16 |
| hotness 80 | decode | **1,191 ms** | 725 | 271 | – | 39 | 44 | 43 | 53 | 15 |
| one expert per chunk | decode | 2,785 ms | 914 | 202 | 1,215 | 54 | 216 | 118 | – | 66 |
| no cache | prefill (16–512 tokens) | 8,364 ms | 6,570 (78.6%) | 239 | 694 (8.3%) | 414 | 180 | 246 | – | 20 |
| one buffer per call | prefill | 7,708 ms | 6,555 | 480 | – | 400 | 53 | 203 | – | 18 |

- **Drive.** It reads at 3.19 GB/s, 95% of its direct-I/O rate on this PCIe 3.0 x4 link.
- **GPU** (traced decode step): busy 637–724 ms of 1.56 s, idle 54–59%.
  - Host-to-device copies: 529–575 ms, running while the drive reads.
  - GEMM: 67–89 ms in all: routed experts 23–37 ms, attention 19–23, shared experts 7–11,
    LM head 2.4.
  - Other kernels: 38–57 ms.
  - 5,195 kernel launches, with 92–109 ms of CPU time in launch calls.
- **The 1,024-token prefill** (traced): 11.45 s.
  - 4.29 s of copies; 197 ms of GEMM, 142 ms of it the experts'.
  - 9,377 launches; the GPU idle 59% of the time.
- **Decode rate:** 0.79 tokens per second without a cache, 0.84 with hotness 80.

**Top three bottlenecks** (decode without a cache, 1,273 ms):

1. **Drive reads: 847 ms (66.5%).**
   - A decode token reads its 156 routed experts, 2.70 GB, at 95% of the drive's rate; a prefill
     reads 14–27 GB.
   - Only fewer bytes help: a host-RAM expert tier (the replay: 67–83% fewer reads with 11–17 GB),
     partial experts (AWPMI), smaller stored experts, or a faster link.
2. **Python and kernel launches of the transformer: 283 ms (22.2%).**
   - Attention (MLA), norms, routers, shared experts, the experts calls' own operations and the
     LM head: 5,195 launches for about 150 ms of kernels.
   - The remedy: CUDA graphs or fused kernels for the decode path, whose shapes are static outside
     the experts.
3. **Per-request transfer overhead: 142 ms (11.2%).**
   - Planning 43, issuing 312 copies 46, assembly 37, the routing sync 16: 52 requests, 130 pieces
     and 2,808 read calls per step.
   - In prefill, chunk handling adds 694 ms (8.3%).
   - The remedies: native planning and submission, one request per expert for all its tensors
     (down, gate, up are adjacent in the file), offsets computed once per call instead of per
     matrix.

GEMM is about 2% of a decode step and 1.7% of a prefill: no custom expert kernel is justified.

**Instrumented step times** (records, with digests and checks) are slower: 1.30 s per decode
step without a cache.

## 14. I/O overlap

Investigated, not adopted (decision 0008, point 12).

- Prefill: chunk N+1's reads could overlap chunk N's GEMMs, but those are about 1% of a prefill
  (142 ms of 11.45 s).
- Decode: a layer's experts exist only after its router; a previous-step prefetch would waste 59%
  of its bytes on a drive that is already busy 66.5% of the step.
- Inside one fetch, the streamer already overlaps reads with copies (Phase 3).

## 15. Native runtime

Classification from the profile (decision 0008, point 15).

| Work | Share | Where |
| --- | --- | --- |
| Drive reads | 66.5% decode, 78.6% prefill | already native (8 threads of `ReadFile`); fewer bytes, not code |
| Transformer Python and kernel launches | 22.2% decode | **C++/CUDA**: CUDA graphs for attention, norms, routers, shared experts; on-device routing to slots |
| Planning, copy issuing, assembly, chunk handling, cache bookkeeping | 11% decode, 18% prefill | **Rust**: read planning, direct-I/O submission and completion, transfer orchestration, cache bookkeeping (hotness eviction is a Python scan: prohibitive for a host tier of a thousand experts) |
| GEMMs, attention kernels, copies | about 2% compute; copies overlapped | **already native through PyTorch** (a grouped GEMM with fewer launches only if its per-group results stay the reference's) |
| Adapters, configuration, reference, audits, per-call chunk planning | negligible per step | **keep in Python** |

## 16. Where AWPMI goes inside experts

Design note (decision 0008, point 14).

- **The hook.** The chunked call is the single place where an expert's weights meet its tokens:
  - `_ChunkedCall.materialize` decides which rows;
  - `_chunked_grouped_mm` computes the product per chunk of groups.

  An AWPMI expert executor replaces "materialize every row of the chunk's experts, then one
  grouped GEMM" with "materialize neuron pages progressively, propagate enclosures through the
  expert MLP (Phase 2's operators), stop when the certificate holds".
- **The combine stays the reference's.** An enclosure of each (token, expert) output, then the
  float32 [T, K, H] sum and the BF16 cast are bounded like any other reduction.
- **Addressability.** A row of the composed gate/up segment is gate rows then up rows, each H × 2
  bytes contiguous: a neuron's gate and up rows are 4 KiB runs. down is [H, I] row-major, so a
  neuron's down column is strided. The milestone first has to choose one of:
  - read all of down (a third of the expert);
  - page down by output rows;
  - write a neuron-major down (a copy, against the zero-copy rule).
- **First target.** The last MoE layer's routed experts at the last position, upstream exact
  (Mode A, as Phase 2's suffix), where the existing pairwise certificate applies directly.

## 17. ds4 / DwarfStar 4 and Soup

No runtime dependency; reviewed again for this phase (sources below).

| System | Pattern | In Weightsift |
| --- | --- | --- |
| ds4 (antirez's DwarfStar 4; `stefandsl/DwarfStar`, cited in decision 0007, is a fork of it) | Hits-first (PR #1082): cached experts computed while misses are read, partials summed in slot order from +0.0f to stay byte-identical; a pool of 8 pread workers with pinned staging | The read pool since Phase 3; hits first at the copy level since 4A. New here: in the chunked `grouped_mm` path nothing is summed until the call's end, so chunks can be computed in any order (compute-level hits-first) without a slot-order sum. Not built |
| ds4 (issue #1119) | LRU thrash in prefill; admission freeze after the first batch | Reproduced at Moonlight's scale: LRU 40 and 80 never hit. The freeze exists (Phase 3) and was not a measured configuration |
| ds4 / DwarfStar | A budget in whole experts, hot-expert preload, budget from free memory | Phase 3's cache budget in bytes of whole rows; hotness. The replay sizes a host tier |
| Soup | Layer-at-a-time streaming with copy streams and pinned memory | The reference executor's pattern, now one layer from the checkpoint rather than from RAM |
| Implemented independently | – | Chunked calls and their stand-ins, the streaming reference through transformers' own loader, `StreamedParameters`, the cache replay, every check |

Sources: [ds4 (DwarfStar 4)](https://github.com/antirez/ds4),
[ds4 PR #1082](https://github.com/antirez/ds4/pull/1082),
[ds4 issue #1119](https://github.com/antirez/ds4/issues/1119),
[stefandsl/DwarfStar](https://github.com/stefandsl/DwarfStar),
[Soup layer streaming](https://trysoup.dev/docs/layer-streaming).

## 18. Answers

1. **Can Weightsift execute Moonlight-16B-A3B despite the model exceeding both practical GPU
   residency and full-reference host-RAM residency? Yes.**
   - The routed experts are 28.8 GB: 4.8× the 6 GB cap and 3.4× the GPU.
   - The checkpoint is 31.9 GB, against 32 GB of RAM.
   - The streamed process peaks at a 3.25 GB working set, the reference at 2.1 GB per step.
2. **Can bounded chunked expert execution reproduce the fully materialized reference exactly?
   Yes.**
   - 726 of 726 steps per run are equal in every recorded quantity, including prefills in chunks
     of 15 experts and every call in chunks of one.
   - transformers' combine runs once per call; per-chunk combining would not be exact
     (adversarial test).
3. **What fraction of expert weights is physically read?**
   - Per decode token: 0.0938 (2.70 GB) without a cache, 0.0796 with hotness 80.
   - Per prefill: 0.79 on average, from 0.50 at 16 tokens to 0.95 at 1,024.
4. **Peak VRAM and RAM.**
   - VRAM: 5.43 GB allocated under the 6.0 GB cap (80-expert caches); 3.65 GB without a cache
     (prefill, 1,024 tokens).
   - RAM: 3.25 GB working set for the streamed process, 2.1 GB (2.84 GB lifetime) for the
     reference.
   - Host commit 4.36 and 3.07 GB. Private bytes 9.7 and 10.7 GB, device memory included (WDDM).
5. **Decode latency.**
   - 1.27 s per token without a cache (0.79 tokens per second); 1.19 s with hotness 80 (0.84).
   - Prefill: 5.7 s (16 tokens) to 11.5 s (1,024 tokens).
6. **Where time is spent.** In decode: drive 66.5%, the transformer's Python and launches 22.2%,
   per-request transfer overhead 11.2%; GEMM about 2%. In prefill: drive 78.6%, chunk handling
   8.3%.
7. **What moves where.**
   - To Rust: read planning, I/O submission, transfer orchestration, cache bookkeeping.
   - To C++/CUDA: CUDA graphs or fused kernels for the launch-bound decode path, and on-device
     routing to slots.
   - Already native through PyTorch: GEMMs, attention, copies.
   - In Python: everything per call or per run.
8. **Is the substrate ready for DeepSeek-V3-class models?** In architecture, yes:
   - bounded expert buffers (a DeepSeek-V3 layer's 11 GB is exactly what the budget removes);
   - an out-of-RAM reference pattern;
   - DeepSeek-V3's routing and shared experts validated;
   - split checkpoints indexed in place;
   - a model-free core.

   Not yet in four specifics:
   - its FP8 experts need the native quantized reference, which is still a declaration only, and
     an index of block-scaled FP8 weights (the layout derivation refuses dequantization today);
   - the reference's single experts layer (11 GB in FP8) exceeds this GPU, so it would run on the
     CPU or a larger GPU;
   - about 700 GB of storage;
   - speed: some 20 GB of FP8 experts per token, about 6 s per token from this drive before any
     cache.
9. **Where AWPMI goes next.**
   - The chunked call's materialization and product (`_ChunkedCall.materialize`,
     `_chunked_grouped_mm`): neuron pages of the routed experts, enclosures through each expert's
     MLP, the reference's own combine bounded.
   - First the last MoE layer at the last position, upstream exact.
   - The down projection's layout must be decided first (section 16).

## 19. Recommendation for the next milestone

The measured bottleneck is bytes from the drive: two thirds of decode and four fifths of prefill.
Compute is about 2%.

1. **AWPMI inside routed experts, first as a measurement.**
   - It is the only lever in the list that lowers the bytes a decision *needs*, rather than
     reusing them or moving them faster.
   - Its payoff is unknown: Phase 2 found that BF16 activation roundings cap certificates in a
     dense last MLP.
   - Start as Phase 1B did: an oracle of how many neuron pages of the last MoE layer's routed
     experts a certificate needs. Build the runtime only if that ceiling shows savings.
   - The substrate is in place: bounded chunks, addressable rows, the reference's combine, an
     independent reference.
2. **Native runtime and a host-RAM expert tier.** An engineering milestone with predictable
   gains:
   - the replay shows 67–83% fewer decode reads with 11–17 GB of this machine's RAM;
   - Rust planning, submission and cache bookkeeping remove most of the 11% decode / 18% prefill
     host overhead;
   - CUDA graphs attack the launch-bound 22%.

   If usable speed on this machine is the priority, this comes first.
3. **DeepSeek-V3-class scaling last.** It is ready in architecture, but needs:
   - the FP8 reference decision (and FP8 layouts);
   - a reference host for 11 GB layers;
   - 700 GB of storage.

   Until (1) or (2) reduces bytes per token, it would mostly measure the drive.

## 20. Findings

1. **Private bytes include device memory under WDDM.** Windows charges a process's device
   allocations to its commit: +2 GiB on the device was +2.15 GB of private bytes, the working set
   unchanged. Host-memory gates use the working set and the host commit.
2. **The device cache is bounded by fragmentation too.** Long-lived rows of 11 and 5.5 MiB in the
   cache's pool leave about 0.4 GiB reserved beyond the cache's bytes. 120 experts did not fit
   under the cap, 96 fit with 10 MiB to spare.
3. **LRU never hits below one token's working set.** With 156 expert loads per decode token, LRU
   caches of 40 and 80 experts hit nothing. The replay shows the knee above 156 experts: 36% at
   160, 93% at 1,248.
4. **Per-request overhead sets the chunk size.**
   - Chunks of 15 experts add 8.5% to a prefill; chunks of one expert multiply it by 2.4.
   - A host sync per chunk and per matrix (`offs.tolist()`, as the per-group fallback does) is
     most of it.
5. **transformers' loader is the reference's speed limit.** 1.06 GB/s through safetensors'
   buffered `pread` on Windows, against 3.2 GB/s with direct I/O: 29 s per reference step.
6. **A loader of the non-expert weights needs every file.** The first full development run failed
   because the expert index lists only the files that hold experts. Moonlight's first file holds
   the embeddings and the dense layer. The benchmark loads from all the checkpoint's files and
   verifies them all.
7. **A profiler's GPU annotation ranges are not activities.** With `record_function` scopes,
   Kineto records each scope's GPU range (gaps included) as a device event. Counted as kernels,
   they inflated "other kernels" to 16 s in an 8.5 s step. The profile attributes activities to
   scopes instead.

## 21. Limitations

- **One machine**: PCIe 3.0, Windows, WDDM. The per-group independence of `_grouped_mm` was
  verified on sm_89, where it is a loop of GEMMs. On sm_90+ (CUTLASS grouped kernels) the chunked
  path must be re-verified.
- **Speed is not a result yet**: 1.27 s per decode token, bound by the drive.
- **Per-assignment outputs** were checked on 94% of experts calls. The chunked prefills of the
  cache configurations beyond their first 2 prompts are checked through their outputs only.
- **The shared-experts profile** was not taken: the profile script does not stream them. Their
  cost comes from the instrumented records, on the same prompts, same instrumentation.
- **The traces of one expert per chunk** show slower kernels: GPU clocks during long I/O waits,
  probably. They are not used for the bottlenecks.
- **The reference's process reads through the OS file cache**: about 20 GB of standby pages
  afterwards, system-wide. Not process memory, and not required.

## 22. Reproduction

```bash
uv sync
uv run pytest                                                              # 510 tests
uv run weightsift pack expert-index --config configs/phase4b-moonlight.yaml     # headers only; packs/ (gitignored)
uv run python benchmarks/moonlight_reference_check.py --output experiments/phase4b/reference-check
export PYTHONHASHSEED=1
for stage in prepare reference stream digest; do                           # reference ~80 min, stream ~35 min
  uv run python benchmarks/moonlight_runtime.py --output experiments/phase4b/<name> --stage $stage
done
uv run python benchmarks/moonlight_profile.py --run experiments/phase4b/<name> --configuration stream \
    --prompts 7 0 3 5 --trace --output experiments/phase4b/<name>/profile-stream.json   # one process per configuration
uv run python benchmarks/moonlight_report.py experiments/phase4b/<name> --compare experiments/phase4b/moonlight-run1
```

A run is reproduced if its `digest.json` matches run1's. Both runs were made on source tree
`46dae54c…`, run1 with `PYTHONHASHSEED=1` and run2 with `PYTHONHASHSEED=2`.

| Digest | Value |
| --- | --- |
| index (manifest, verification, audit) | `26c1d89c…` |
| reference rows (sha256 of every expert row) | `7c101ec2…` |
| prompts | `9966d0c1…` |
| reference | `2e4d4a26…` |
| records | `7266d6c8…` |
| ranges (raw I/O traces) | `d99aabdf…` |

Timing fields (`timings_ms`, `system`), the stage reports and the profiles are excluded from the
digests. The profiles were written after both runs, on run1's prompts, one configuration per
process.
