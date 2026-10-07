# AWPMI Phase 4A report — The first real out-of-VRAM mixture of experts

Date: 2026-10-03 · Follows: `history/2026-10-02-awpmi-phase3-report.md` (Phase 3) ·
Decision: `decisions/0007-phase4a-out-of-vram-moe.md` ·
Status: **complete. Correctness and gates A, B, C and D pass.**

## Outcome in brief

**Yes: Weightsift now runs a real MoE whose expert weights exceed the available GPU memory, and
reproduces the reference model exactly.**

- **The setup.** OLMoE-1B-7B has 12.9 GB of BF16 experts: 1.50× this GPU (8.59 GB) and 2.15×
  the 6.0 GB memory cap the streamed process runs under. Its experts are read in place from the
  published checkpoint, which stores every expert as three separate tensors. The index is built
  from the file headers alone, a 372 KB manifest; nothing is converted or copied. A compact
  experts call holds only the routed experts on the GPU.
- **Correctness.** Every one of 2,089 streamed steps per run is equal to the fully materialized
  reference:
  - in tokens and logits, prefill and decode with the KV cache;
  - in the whole KV cache;
  - per layer, in the router logits, the routed experts, their weights, every (token, expert)
    output and the experts module's output.

  The reference is transformers' own model, loaded in host memory and executed one experts
  layer at a time on the GPU.

Per decode token (40 prompts × 12 decode steps, run1). Fractions are of all expert bytes
(12.9 GB).

| | No cache | LRU, 1/8 of the experts | LRU, 1/4 | Hotness, 1/4 | Every expert every step |
| --- | --- | --- | --- | --- | --- |
| Read from the drive | **0.1251** (1.61 GB) | 0.0819 | 0.0672 | **0.0658** | 1.0006 |
| Moved to the GPU | 0.1250 | 0.0818 | 0.0672 | 0.0657 | 1.0000 |
| Cache hit rate (lookups) | – | 0.345 | 0.463 | 0.475 | – |
| Expert bytes on the GPU (max) | 126 MB (8 experts + 2 spares) | 1.71 GB | 3.32 GB | 3.32 GB | 805 MB |
| Peak GPU memory, everything | 1.14 GB | 2.74 GB | 4.35 GB | 4.35 GB | 1.82 GB |
| Decode step, no digests (profile) | 759 ms | – | 525 ms | – | – |
| Steps equal to the reference | 520 of 520 | 520 of 520 | 520 of 520 | 520 of 520 | 9 of 9 |

- **Selective reads.** A decode token reads its 8 routed experts per layer and nothing else:
  0.1251 of the expert bytes, 8.0× less than streaming every expert. The excess over 0.125 is
  the 4 KiB blocks at the ends of unaligned tensors (amplification 1.00065).
- **Device memory** follows the routed and cached experts, never the expert count. Without a
  cache the GPU holds 0.95 GB of non-expert weights plus at most 126 MB of experts in decode
  (805 MB in a prefill that routes every expert).
- **Caches.** Half of a decode token's experts come from a cache of a quarter of the experts.
  Hotness also helps prefill (0.706 instead of 0.806).
- **Time.** No win is required yet. The drive is the first bottleneck: it reads 1.6 GB per token
  at 94% of its sequential rate.

**Gates.**

- Correctness: PASS.
- A (out of VRAM): PASS.
- B (selective physical I/O): PASS, 0.1251 against limits of 0.15 and 0.20 × all-experts.
- C (compact residency): PASS, at most 0.156 of a layer in decode.
- D (model-general core): PASS.
- Two runs with different Python hash seeds: identical digests (index, prompts, reference, records and I/O traces) on the same source tree, with `PYTHONHASHSEED` 1 and 2.

## 1. Question

> Can Weightsift run a real MoE model whose expert weights exceed the available GPU memory, with
> the generic streaming and cache backend, while reproducing the fully materialized reference
> model's discrete decisions exactly?

## 2. Setup

| Item | Value |
| --- | --- |
| Machine | RTX 4060 Ti 8 GB (8.59 GB; sm_89; PCIe 3.0 x8, 6.5 GB/s pinned host-to-device), i7-8700K, 32 GB RAM, Windows 11 |
| Drive | Samsung 990 PRO (PCIe 3.0 x4 here): 3.37 GB/s sequential with direct I/O |
| Software | torch 2.14.1+cu130, transformers 5.18.0 (`grouped_mm` experts, SDPA attention) |
| Model | `allenai/OLMoE-1B-7B-0924` @ `6d84c485`, BF16, 3 shards (13.8 GB): 16 layers × 64 experts, 8 routed per token; an expert is 12 MiB (gate, up, down; 4 MiB each, adjacent in the file) |
| Expert index | `packs/olmoe-1b-7b-0924-expert-index`: 32 composed segments (16 layers × gate_up, down), 12,884,901,888 bytes in place; 372 KB manifest; files identified by the Hub's sha256 |
| Prompts | 40 of the Phase 1 wikitext prompts, re-tokenized (mean 50.9 tokens, 7 to 127); greedy, 12 decode steps with the KV cache |
| Streamed process | GPU cap 6.0 GB (allocator limit); only non-expert weights loaded (0.95 GB, direct reads); experts from the drive (direct I/O, 8 threads, 32 MiB pinned staging slots) |
| Configurations | no cache; LRU caches of 1/8 and 1/4 of the expert bytes; hotness 1/4; every expert every step (3 prompts, 2 decode steps) |
| Reference | its own process: transformers' model in host memory (15.1 GB), experts of 5 layers resident on the GPU, the other 11 copied whole before they run |

## 3. What was built

Decision 0007 has the details.

- **Compact experts call** (`StreamedExperts(compact=True)`, Blocker A).
  - Between calls the expert parameters are `None`.
  - For a call, the pre-hook assembles the routed experts in ascending order into buffers
    [|R|, …], then swaps them in with `num_experts` = |R| and `top_k_index` remapped to slots.
    The module's own forward runs, and a hook releases the buffers.
  - Ascending order keeps eager, `grouped_mm` and `batched_mm` bit for bit; shuffled slots
    break eager.
  - Every expert chosen by several tokens is read once.
- **Composed segments** (Blocker B).
  - The storage core describes a row as byte spans of one or more files.
  - Plans read only those spans, sorted by file, aligned to 4 KiB, with explicit output
    offsets, and can scatter rows into a caller's buffer.
  - Pack v2 manifests hold the index. An index is written from headers (`weightsift pack
    expert-index`), with the layout derived from transformers' own conversion mapping and
    checked by the OLMoE adapter.
- **Loading without experts.**
  - The skeleton is built on `meta`.
  - Non-expert tensors are read with direct I/O and cast as transformers casts them.
  - Non-persistent buffers come from the model's initializer.
  - The streamed process never holds an expert tensor: its host memory peaks at 2.7 GB.
- **Reference profiles** (`awpmi.profiles`): BF16, FP16 and native-quantized (declared only),
  checked against the model and the stored dtypes.
- **`FullLayerOffload`.** The reference executor: transformers' own model, one experts layer
  materialized at a time.
- **Backend.**
  - `materialize(…, out=)` fills compact buffers in place, cached experts first.
  - Cache entries have a CUDA memory pool of their own.
  - The streamer copies long runs straight from pinned staging and uses 32 MiB slots for
    experts.
- **OLMoE adapter** (`awpmi.models.olmoe`): the checked layout, the routers, the reference
  profile. Nothing else is model-specific.

## 4. Correctness

Sources: `experiments/phase4a/olmoe-run1/summary.md` and `olmoe-run2`. Raw records:
`reference.jsonl.gz`, `records.jsonl.gz`, `index.json`, `ranges.jsonl.gz`.

| Criterion | Result (per run) |
| --- | --- |
| Source files | the three shards re-hashed with direct reads: their sha256 equal the Hub's |
| Expert index against the loader | all 2,048 expert rows (32 segments × 64), read through the index from the drive, are byte for byte transformers' own fused `gate_up_proj` and `down_proj` |
| Residency check | 26 of 26 reference steps equal (prefill and decode) with every layer offloaded and with 5 layers resident |
| Streamed steps equal to the reference | 2,089 of 2,089: no cache 520, LRU 1/8 520, LRU 1/4 520, hotness 1/4 520, every expert 9 |
| …in each recorded quantity | token, logits, whole KV cache, routed experts; per layer router logits, top-k indices, top-k weights, every (token, expert) output before weighting, experts output: all 2,089 |
| Poison | 39 steps (first 3 prompts, no cache) with 2 NaN spare slots and every slot NaN-filled before assembly: equal |
| Audits | 0 problems on 2,089 steps: requested = served experts; cache + storage = requested; storage = fetched; device received = storage; every block read holds a requested byte; compact buffers = served experts |
| OS counters | read calls and bytes equal the store's on every step |
| Reproducible | run1 (`PYTHONHASHSEED=1`) and run2 (`PYTHONHASHSEED=2`): identical index, prompts, reference, records and range digests, same source tree |

The per-(token, expert) check is direct. Inside each call, both stages run the experts module
again on the call's own weights, with every (token, k) assignment as a token routed to its
expert alone with weight 1. That groups the same rows per expert as the call, so it yields
exactly the expert outputs the call combined.

**Tests.** The suite has 452 tests, 53 of them new:

- composed segments:
  - plans against brute force over three files;
  - exact reads, direct and buffered;
  - OS counters, no hidden reads, bytes outside the spans never reaching the output;
  - scattered output rows, direct copies of long runs;
  - index packs and their tampering detection;
- the compact call, bit for bit against the resident model with the KV cache:
  - 7 architectures, on CPU and CUDA;
  - top-1 routing and dense streaming;
  - LRU, hotness and random-eviction caches;
  - hash-seed independence in subprocesses;
- the offload reference and per-assignment outputs;
- split checkpoints written by `save_pretrained` and served in place, bit for bit: Mixtral,
  Qwen2-MoE, Qwen3-MoE, OLMoE and DeepSeek-V3. The expert-free loader equals `from_pretrained`
  on every non-expert tensor and buffer;
- the OLMoE adapter's layout check;
- layering.

**Guards confirmed to fail when disabled or sabotaged:**

| Guard | What failed |
| --- | --- |
| NaN written into a served slot (compact poison mode) | the output changes: poison checks something |
| OLMoE's layout swapped (up before gate) | the adapter's check against transformers' mapping raises |
| The model named in two storage-core docstrings | gate D (`tests/test_layering.py`) failed during development, and the docstrings were fixed |
| `meta` placeholders between calls | not an error but CUDA garbage (section 11). The guard was changed to `None`, and the test requires that use raises |
| Every plan reads one 4 KiB block more than its runs need | the composed no-hidden-reads test (data tests alone still pass) |

## 5. Out of VRAM (gate A)

| | Bytes |
| --- | --- |
| Expert weights | 12.88 GB (12,884,901,888) |
| GPU total | 8.59 GB |
| Cap of the streamed process | 6.00 GB: experts are 2.15× the cap, 1.50× the GPU |
| Non-expert weights on the GPU | 0.95 GB (147 tensors, loaded with direct reads in 1.2 s) |
| Peak GPU memory, any step | 5.04 GB allocated (LRU or hotness 1/4, prefill), 6.00 GB reserved |
| Host memory of the streamed process | 2.7 GB peak (1.25 GB after loading; 128 MiB of pinned staging) |
| Reference process | 15.1 GB of host memory; 5.0 GB on the GPU (5 resident layers), 5.9 GB peak |

Every configuration completed all its steps under the cap. The first capped run did not: it ran
out of memory with 3.92 GiB allocated and 1.61 GiB reserved but unusable. Cache pages and the
compact buffers had fragmented the allocator. Cache entries now have their own memory pool
(section 11).

`stream_stage.json` reports a configuration's device memory at start as 0.99, 2.60 and 4.20 GB.
That is a measurement artifact of the benchmark. A default argument of its step callback still
held the previous configuration's cache, until the first prompt of the next configuration
replaced the callback. The per-step peaks show the cache was gone before any step: under the
cap, two caches of a quarter would need 7.4 GB.

## 6. Selective physical I/O (gate B)

Per step, as fractions of all expert bytes; each cell is the mean, then the median, then the p95.
Prefill steps read for all of a prompt's tokens.

| Configuration | Phase | Experts per layer | Drive | Host → GPU | Read calls | Extents | Drive I/O ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no cache | prefill | 51.5 | 0.806 / 0.852 / 0.932 | 0.805 | 11,539 | 1,646 | 3,042 |
| no cache | decode | 8.0 | **0.1251** / 0.1251 / 0.1251 | 0.1250 | 1,792 | 256 | 496 |
| LRU 1/8 | decode | 8.0 | 0.0819 / 0.0821 / 0.1244 | 0.0818 | 1,173 | 168 | 324 |
| LRU 1/4 | prefill | 51.5 | 0.776 / 0.825 / 0.905 | 0.776 | 11,119 | 1,589 | 2,930 |
| LRU 1/4 | decode | 8.0 | 0.0672 / 0.0664 / 0.1042 | 0.0672 | 963 | 137 | 266 |
| hotness 1/4 | prefill | 51.5 | **0.706** / 0.767 / 0.843 | 0.706 | 10,121 | 1,446 | 2,677 |
| hotness 1/4 | decode | 8.0 | **0.0658** / 0.0674 / 0.0987 | 0.0657 | 942 | 134 | 260 |
| every expert | decode | 64.0 | 1.0006 | 1.0000 | 14,332 | 2,044 | 3,970 |

- **Gate B: PASS.**
  - Without a cache a decode token reads 0.1251 of the expert bytes (limit 0.15), 0.125× of
    what streaming every expert reads (limit 0.20).
  - Physical bytes are logical bytes × 1.00065 in every configuration. A 4 MiB tensor starting
    off the 4 KiB grid costs one extra block; the gate and up tensors of an expert are one run.
- What reaches the GPU is exactly what was requested and not cached: host-to-device bytes equal
  the storage's logical bytes on every step.
- Raw physical-I/O traces: `ranges.jsonl.gz` lists every extent read (file, offset, length) for
  the first 5 prompts of every configuration.

## 7. Compact device residency (gate C)

| Configuration | Phase | Compact buffers (max) | Experts on the GPU (max) | Peak GPU (max) |
| --- | --- | --- | --- | --- |
| no cache | decode | 126 MB (10 slots: 8 routed + 2 poison spares; 101 MB without) | 126 MB | 1,137 MB |
| no cache | prefill | 830 MB (64 + 2 spares) | 830 MB | 1,834 MB |
| LRU 1/8 | decode | 101 MB | 1,711 MB | 2,736 MB |
| LRU 1/4, hotness 1/4 | decode | 101 MB | 3,322 MB | 4,347 MB |
| LRU 1/4, hotness 1/4 | prefill | 805 MB | 4,027 MB | 5,040 MB |
| every expert | any | 805 MB | 805 MB | 1,829 MB |

- **Gate C: PASS.**
  - On every step the compact buffers held exactly the served experts (audited): |R| × 12 MiB.
  - In decode they never exceeded 0.156 of a layer (limit 0.20; 0.125 without poison spares).
  - Experts on the GPU never exceeded the cache's capacity plus one call's buffers.
- For comparison, Phase 3's full-shape buffers would have held 805 MB per layer in every call.
  A decode call now holds 101 MB.
- The limit that remains: a prefill that routes every expert of a layer needs that layer's
  805 MB. A prompt of 51 tokens already routes 51.5 of 64 experts per layer. A DeepSeek-V3 layer
  would need 11 GB (section 13).

## 8. Caches and routing

- Decode routes 8.0 experts per layer, prefill 51.5 of 64.
- Routing is moderately skewed: the most used 25% of (layer, expert) slots receive 47.6% of
  decode routings; the top 10% receive 25%.
- Of a decode step's experts, 37.6% were routed at the previous step in the same layer (Granite:
  50%).
- Hit rates (lookups):

  | | Decode | Prefill |
  | --- | --- | --- |
  | LRU 1/8 | 0.345 | 0.004 |
  | LRU 1/4 | 0.463 | 0.035 |
  | hotness 1/4 | 0.475 | 0.122 |

  Miss rates are the complements.
- LRU and hotness are close in decode. Hotness keeps experts that prompts share, so it also
  helps prefill (drive 0.706 against 0.776 for LRU and 0.806 without a cache), as in Phase 3.
- Prefill floods an LRU cache: a prompt brings in 51.5 experts per layer and hits almost none.
  This is the behaviour behind ds4's admission freeze (ds4 issue #1119). The freeze exists
  (`PageCache.admit`) but was not one of the measured configurations.

## 9. Time and the top three bottlenecks

The records' step times include the benchmark's digests: hashing every intermediate on the
host and re-running every experts call for the direct check. These figures come from
`benchmarks/olmoe_profile.py`, which runs the same configurations recording nothing but time
(`profile.json`).

Per step, 4 prompts per configuration, each configuration in a fresh process (its cache
starts empty: hit rates are below the 40-prompt runs'). Host milliseconds by stage, with the
device time of the host-to-device copies and of the experts calls.

| Configuration | Phase | Step | Drive I/O | Other (transformer) | Issue copies | Plan | Assemble | Admit to cache | Route | H2D (device) | Experts (device) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no cache | decode | **759** | 508 | 154 | 36 | 26 | 25 | – | 10 | 265 | 46 |
| no cache | prefill | 4,508 | 3,639 | 402 | 291 | 35 | 121 | – | 20 | 1,921 | 124 |
| LRU 1/8 | decode | 579 | 335 | 143 | 23 | 27 | 29 | 12 | 9 | 174 | 44 |
| LRU 1/4 | decode | **525** | 294 | 134 | 20 | 25 | 31 | 11 | 9 | 152 | 42 |
| hotness 1/4 | decode | 549 | 286 | 125 | 20 | 26 | 71 | 12 | 9 | 148 | 42 |

- The drive reads at 3.16–3.22 GB/s, 94–96% of its sequential rate. The copies to the GPU
  (6.5 GB/s) overlap the reads.
- A traced decode step without a cache runs about 20 ms of kernels: 9.9 ms of GEMM and 8 ms of
  everything else. It launches 2,847 kernels, with 55 ms of CPU time in launch calls. The GPU is
  idle for over 97% of the step.
- Hotness's eviction is a Python scan: 71 ms of assembly against 31 ms for LRU.

**Top three bottlenecks** (decode without a cache, 759 ms):

1. **Drive reads: 508 ms (67%).** 1.6 GB per token at 94% of the drive's sequential rate. Only
   fewer bytes help: caches (294 ms at a quarter of the experts), smaller experts (quantized
   references), or a faster drive. This machine's PCIe 3.0 x4 link caps the drive at
   3.4 GB/s.
2. **Python and kernel launches of the transformer: 154 ms (20%).** Attention, norms, routers,
   the experts calls' own operations and the LM head: 2,847 launches for about 20 ms of GPU
   work. The remedy is fused kernels or CUDA graphs, which the compact call's data-dependent
   shapes complicate.
3. **Per-request transfer overhead: about 97 ms (13%).** It consists of:
   - 32 plans and 80 pieces per step: planning 26 ms, issuing 256 copies 36 ms, assembly 25 ms;
   - the routing sync, 10 ms.

   The remedies are one plan per expert for all its tensors (its down, gate and up tensors are
   adjacent in the file), native read submission, and overlapping a layer's reads with the
   previous layer's compute.

Before this phase's two transfer changes, a no-cache decode step took 969 ms in the same
profile: 384 pieces of 8 MiB, and a host gather of every byte. GEMM is about 1% of a step, so
custom kernels for the experts are not justified by this profile.

## 10. Model-general architecture (gate D)

- **Gate D: PASS.** The storage, transfer and materialization core (`awpmi.storage`,
  `awpmi.streaming`, `awpmi.materialization`) names no model, tensor layout or router, and
  imports nothing above it. `tests/test_layering.py` now also forbids `gate_proj`, `up_proj`,
  `num_experts` and index-file names in the core. It caught two docstrings that named the
  model during this phase.
- What is OLMoE-specific is the adapter (`awpmi.models.olmoe`, 60 lines):
  - the layout, checked against transformers;
  - where the routers are;
  - the reference profile.
- The composed index, the expert-free loader and the compact call are generic. They derive the
  layout from transformers' conversion mapping, the same mapping DeepSeek-V2/V3, Qwen3-MoE and
  GLM4-MoE use, and the tests run them on five architectures' split checkpoints.

## 11. Findings

1. **A `meta` weight does not make a CUDA grouped GEMM fail.** It returned NaN garbage. Unloaded
   experts are therefore `None` between calls, so that any use raises.
2. **Cache pages fragment the allocator under a cap.** Long-lived 4 and 8 MiB pages among
   short-lived compact buffers of 10–830 MB left 1.6 GiB reserved but unusable, and the first
   capped run failed. A separate CUDA memory pool for the cache fixed it.
3. **transformers casts some weights at load time.** DeepSeek-V3's router bias is kept in
   float32 (`_keep_in_fp32_modules_strict`) although stored in BF16. A loader that bypasses
   `from_pretrained` must apply the same dtype plan; the expert-free loader does, and refuses
   it for streamed experts.
4. **Slot size decided where the time went.** With Phase 3's 8 MiB slots, a decode step moved
   384 slot-sized pieces, each paying Python overhead. With 32 MiB slots it moves 80, and
   copying long runs straight from pinned staging removes a host copy of every byte.
5. **The newer OLMoE base checkpoint (`0125`) is stored in FP32.** Its bytes are not the BF16
   reference's, so zero-copy streaming needs a checkpoint stored in the reference dtype, or a
   declared FP32 reference.
6. **transformers' own disk offload re-saves converted experts** (`accelerate_disk_offload`):
   it would copy every fused expert. Composed segments read the published tensors in place.
7. **Prefill is the working set that matters.** Decode is selective (8 of 64 experts per
   layer), but a 51-token prompt routes 80% of each layer's experts. Both device memory and
   drive bytes per prompt are dominated by prefill.

## 12. Reuse of DwarfStar, ds4 and Soup

No runtime dependency. Their patterns were reviewed again for this phase (sources below).

| System | Pattern | In Weightsift |
| --- | --- | --- |
| Soup | Model in RAM or NVMe, copied into pre-allocated VRAM buffers one layer at a time on a dedicated stream, page-locked memory | Reused conceptually: pinned staging, copy stream and events (Phase 3), and the layer-at-a-time reference executor here (`FullLayerOffload`). Not applicable: next-layer prefetch, since a MoE layer's experts are unknown until its router runs |
| ds4 (PR #1082) | Hits first: run cached experts while miss reads are in flight, sum partials in slot order; a pool of 8 pread workers with pinned staging | The read pool (8 threads, positioned direct reads) since Phase 3. Hits first adopted at the copy level: cached experts are copied before the misses are read. Not adopted: computing hits before misses arrive, which needs an exact combine (section 13) |
| ds4 (issue #1119) | LRU thrash in prefill (0% hits); fix: freeze admission after the first batch | The freeze exists since Phase 3; this phase measures the thrash: 0.4–3.5% prefill hits with LRU |
| DwarfStar | Routed experts as "a first-class disk citizen"; a memory budget for complete experts; hot-expert preload | A byte-budget cache of whole expert rows (Phase 3), hotness; its own memory pool here plays the role of a fixed expert arena |
| Implemented independently | – | The compact call, composed segments, layouts from transformers' mapping, the expert-free loader, reference profiles, the offload reference, every check |

Sources: [Soup layer streaming](https://trysoup.dev/docs/layer-streaming),
[ds4 PR #1082](https://github.com/antirez/ds4/pull/1082),
[ds4 issue #1119](https://github.com/antirez/ds4/issues/1119),
[DwarfStar](https://github.com/stefandsl/DwarfStar).

## 13. Answer, and the next target

**Can Weightsift now run a real MoE model whose expert weights exceed available GPU memory while
reproducing the reference model exactly? Yes.**

- OLMoE-1B-7B's 12.9 GB of experts run on an 8 GB GPU, under a 6 GB cap.
- 2,089 of 2,089 streamed steps per run are equal to the fully materialized reference in every
  recorded quantity, including every expert output and the whole KV cache. Two runs with
  different hash seeds produced identical digests.

**Recommended next model: `moonshotai/Moonlight-16B-A3B`, before DeepSeek-V3.**

- **It is DeepSeek-V3's architecture at 1/40 of the size** (`DeepseekV3ForCausalLM`, MIT
  license):
  - MLA attention (KV LoRA rank 512);
  - one dense layer, then 26 MoE layers of 64 routed and 2 shared experts, 6 routed per token;
  - V3's router: sigmoid scores, `noaux_tc` with a float32 correction bias;
  - a 163,840-token vocabulary.

  Every component DeepSeek-V3 adds over OLMoE is exercised, the router dtype plan included,
  which the loader already handles.
- **Its experts are 28.8 GB in BF16**, 3.4× this GPU. An expert is 16.5 MiB; a layer is 1.1 GB.
  The checkpoint stores experts split per tensor, under the mapping the composed index already
  derives (a tiny DeepSeek-V3 checkpoint passes the split-checkpoint test).
- **It forces the two capabilities DeepSeek-V3 will need, at a size where they can be checked:**
  1. *A reference that does not fit in host memory.* The model (31.9 GB) does not fit this
     machine's 32 GB of RAM next to the OS. The reference executor must stream each layer's
     full experts from the checkpoint through an independent reader, not Weightsift's path. The
     residency check and the index audit carry over.
  2. *Bounded prefill buffers.* A layer is 1.1 GB, which still fits, but it is the first model
     where the prefill working set crowds the cap. Splitting an experts call over groups of
     experts, with an exact replica of the experts implementation's combine, can be developed
     and checked bit for bit there before DeepSeek-V3 needs it (11 GB layers). The same combine
     gives ds4's compute-level hits first.
- **Why not DeepSeek-V2-Lite.** It has the same sizes (31.4 GB, 64 + 2 experts, top-6), but V2's
  softmax router and attention variant: it would test less of what DeepSeek-V3 needs, under a
  more restrictive license. Moonlight subsumes it.
- **Before DeepSeek-V3 itself:**
  - the open decision on the reference for FP8 experts (`NATIVE_QUANTIZED_REFERENCE`, declared
    here);
  - about 700 GB of storage;
  - and, for useful speed, native read submission and overlap across layers (section 9).

## 14. Limitations

- **Prefill holds one layer of experts.** Fine at 805 MB; not at DeepSeek-V3's 11 GB (section
  13).
- **The reference must fit in host memory** (15.1 GB here). Moonlight's would not.
- **Speed is not a result yet.** The drive is near its peak, and Python overhead and serialization
  remain.
- **One machine** (PCIe 3.0 platform), Windows only for direct I/O. The POSIX path is untested
  here, as in Phase 3.
- **Experts implementations.** The compact call is validated for transformers' eager,
  `grouped_mm` and `batched_mm` kernels, not for the optional `deepgemm` or `sonicmoe` ones.
- **A benchmark artifact** inflates `device_bytes_at_start` of later configurations (section 5).
  No gate or digest uses it.

## 15. Reproduction

```bash
uv sync
uv run pytest                                                              # 452 tests
uv run weightsift pack expert-index                                             # headers only; packs/ (gitignored)
PYTHONHASHSEED=1 uv run python benchmarks/olmoe_runtime.py --output experiments/phase4a/<name>   # about 46 min
uv run python benchmarks/olmoe_profile.py --configurations stream lru-12 lru-25 hotness-25 --output experiments/phase4a/<name>/profile.json
uv run python benchmarks/olmoe_report.py experiments/phase4a/<name> --compare experiments/phase4a/olmoe-run1
```

A run is reproduced if its `digest.json` matches run1's. Both runs were made on source tree
`ccffba4e…`, run1 with `PYTHONHASHSEED=1` and run2 with `PYTHONHASHSEED=2`.

| Digest | Value |
| --- | --- |
| index (manifest and audit) | `54d999f9…` |
| prompts | `81e7e6a9…` |
| reference | `3e77f600…` |
| records | `da5c0a54…` |
| ranges (raw I/O traces) | `93f53db7…` |

Timing fields (`timings_ms`, `system`), the stage reports and `profile.json` are excluded from
the digests. The profile was written by `olmoe_profile.py`, run once per cache configuration:
in one process, the script's timing wrapper keeps a reference cycle to the previous
configuration's cache until a garbage collection, and the second configuration ran out of the
capped memory.
