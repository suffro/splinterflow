# AWPMI Phase 3 report — Real selective materialization and a general storage substrate

Date: 2026-10-02 · Follows: `history/2026-10-02-awpmi-phase2-report.md` (Phase 2) ·
Decision: `decisions/0006-phase3-storage-substrate-and-moe-backend.md` ·
Status: **complete. Gates A, B and C pass.**

## Outcome in brief

Phase 3 moves the weights out of memory. A storage core that knows no model now does five
things:

- reads only the requested rows of a weight from files, with direct I/O, or from host memory;
- moves only those bytes to the GPU, through a pinned, double-buffered streamer;
- keeps pages in a budgeted cache;
- counts every byte;
- checks its reads against the operating system's own counters.

The Phase 1C LM head runs on it with its mathematics unchanged. A model-agnostic adapter serves
the experts of a mixture-of-experts model through the same API.

**LM head** (SmolLM2-135M, 1000 prompts, two runs with identical digests). Fractions are of the
BF16 LM head per token (56.6 MB); C-direct is the Phase 1C runtime on the drive, masked
fallback.

| | Full BF16 head from the drive (A-direct) | Resident Phase 1C (B) | **On the drive (C-direct)** | C with the int6 base cached on the GPU |
| --- | --- | --- | --- | --- |
| Logical bytes (Phase 1C's log) | 1.000 | 0.404 | **0.404** | 0.404 |
| Bytes read from the drive | 1.000 | – | **0.468** (median 0.449) | **0.090** |
| Bytes moved host → device | 1.000 | – | **0.390** | **0.011** |
| Device memory held between tokens | 0 | 93.2 MB (1.65× the head) | 0.8 MB | 22.2 MB |
| Time per token | 22.2 ms | 20.1 ms | 43.3 ms | 31.3 ms |
| Certified / fallback mismatches | 0 / 0 (GEMV bitwise 2000 of 2000) | 0 / 0 | 0 / 0 | 0 / 0 |

- The ~0.40 logical fraction of Phase 1C becomes a real reduction:
  - 2.1× fewer bytes read from the drive and 2.6× fewer moved to the device than the full head;
  - with the int6 base level cached, 11× fewer read and 90× fewer moved.
- Physical bytes exceed logical bytes only by the 4 KiB blocks around the int4 refinement rows:
  for those rows, 0.090 of the head is read for 0.011 needed. The base level is one sequential
  read. The bytes are not invalidated; they are amplified 1.20× overall.
- Correctness is unchanged:
  - the drive-backed runtime is bit for bit the resident one: decision, reads, every state's
    bounds and the fallback logits, on every prompt;
  - the resident one reproduces the Phase 1C records field by field.
- Time is not a win yet, as the brief allowed. The eager PyTorch runtime itself takes about
  20 ms. To that the drive adds about 23 ms of waited reads:
  - 6.4 ms of sequential base level;
  - about 440 random 4 KiB extents at about 14 µs each, issued from Python;
  - planning, gathering and copying.

**MoE** (Granite 3.1 1B-A400M Instruct, 24 layers × 32 experts, 8 routed per token, 2.42 GB of
experts; 50 prompts × 13 steps, two runs): the same backend serves the experts through a model-agnostic adapter.

| | Resident model | Experts from the drive, no cache | LRU cache, half the experts | Hotness cache, half | Every expert every step (dense streaming) |
| --- | --- | --- | --- | --- | --- |
| Device memory between tokens | 2.70 GB | **0.39 GB** | 0.39 + 1.21 GB | 0.39 + 1.21 GB | 0.39 GB |
| Expert bytes read per decode token | 0 | **0.251** (605 MB) | 0.096 | **0.080** | 1.000 |
| Decode step | – | 451 ms | 314 ms | 345 ms | 1,074 ms |
| Steps bit for bit equal to the resident model | – | 650 of 650 | 650 of 650 | 650 of 650 | 25 of 25 |

- Every one of 3,275 streamed steps reproduces the resident logits bit for bit, prefill and
  decode with the KV cache. On 39 of them every unrouted slot was poisoned with NaN.
- Only the routed experts are read: 8 of 32 per layer in decode.
- The experts are read straight from the published safetensors file; nothing is copied into a
  pack.

**Gates.**

- A (physical selective materialization): PASS.
- B (real byte reduction): PASS, 0.468 drive and 0.390 host-to-device, against thresholds of 0.75.
- C (general backend): PASS. The same `PageStore`, `PageStreamer`, `PageCache` and
  `MaterializationBackend` serve the dense LM head and the MoE experts, and the storage core
  names no model, tensor or router (enforced by a test).

## 1. Question

> Does the ~0.40–0.50 logical materialization fraction of Phase 1C become a meaningful reduction
> in real bytes moved, once weights live on a drive and only the selected pages are read? And
> can one storage and materialization API serve both the certified LM head and a
> mixture-of-experts model, without model-specific logic in its core?

## 2. Setup

| Item | Value |
| --- | --- |
| Machine | RTX 4060 Ti 8 GB (PCIe, pinned host-to-device 6.5 GB/s), i7-8700K, 32 GB RAM, Windows 11 (NTFS, 4 KiB clusters) |
| Drive | Samsung 990 PRO 1 TB NVMe (512 B logical, 4 KiB physical sectors). Measured with direct I/O: 3.37 GB/s sequential, 14.4 µs per random 4 KiB extent with 8 threads |
| LM head | SmolLM2-135M-Instruct @ `12fd25f7…`, q6+q4 refinement pack (`packs/smollm2-135m-instruct-lm-head-q6+q4`): level records and remainder norms in a 36.6 MB safetensors file; exact rows = the checkpoint's own `model.embed_tokens.weight` (the LM head is tied), read from the published file |
| Prompts | The 1000 wikitext-2 prompts of Phase 1 |
| Configurations | A-direct, A-host (the full head from the drive / from pinned host memory, then the reference GEMV); B (resident Phase 1C; masked and full fallback); C-direct (masked and full); C-host; C-cached-base (base level pinned in the device cache) |
| Storage options | 4 KiB alignment, no gap merging, 8 worker threads, ≤ 1 MiB per read call, two 8 MiB pinned staging slots |
| MoE | `ibm-granite/granite-3.1-1b-a400m-instruct` @ `0da7a48b`; expert pack = the published `model.safetensors` (48 segments referenced, none copied); 50 prompts re-tokenized from the same wikitext prompts (≤ 128 tokens), greedy, 12 decode steps with the KV cache |

## 3. What was built

Decision 0006 has the details. In short:

- **Storage** (`awpmi.storage`):
  - segments of fixed-size rows (a safetensors tensor is one as it is);
  - read plans: runs of requested rows, widened to 4 KiB and merged where blocks touch;
  - `FileBackedPageStore`, positioned reads with 8 threads and direct I/O;
  - `InMemoryPageStore`;
  - `IOStats`, cross-checked against the OS;
  - `PageCache` (LRU or hotness, byte budget, pinning, admission freeze);
  - packs with a hashed manifest; `weightsift pack`.
- **Transfer** (`awpmi.streaming.PageStreamer`):
  - pinned, aligned, double-buffered staging;
  - a copy stream and events;
  - only requested rows cross to the device;
  - prefetch tickets.
- **Materialization** (`awpmi.materialization`): `MaterializationBackend`, `WeightStore`,
  `ExpertStore`.
- **Model adapters:**
  - the Phase 1C store reads through a backend (`PackedRefinementStore.from_pack`);
  - `awpmi.models.moe` serves the experts modules of transformers 5.

## 4. Correctness (gate A)

Sources: `experiments/phase3/storage-run1/summary.md` and `storage-run2`. Raw records are in
`records.jsonl.gz`, `validation.jsonl`, `reference.jsonl`, `pack.json` and `ranges.jsonl.gz`.

| Criterion | Result |
| --- | --- |
| Certified / fallback mismatches | 0 / 0 in 6,000 runtime runs per run (B and C-direct in both fallback modes; C-host; C-cached-base) |
| Full head from storage, GEMV bitwise | 2,000 of 2,000 (drive and host) |
| Envelope / coarse-arithmetic violations | 0 / 0 (largest binary32 error 2.1·10⁻⁴ of its bound, as in Phase 1C) |
| B against the Phase 1C run, 14 recorded fields | identical on 2,000 of 2,000 runs |
| C against B, bit for bit (decision, reads, every state's bounds, fallback logits) | identical on 4,000 of 4,000 runs |
| Storage audit, every C and A run | 6,000 of 6,000 clean. The backend was asked for exactly the rows the store logged; storage served what the cache did not hold; the device received exactly those bytes; every block read holds a requested byte |
| OS counters (direct I/O) | the process's read calls and bytes equal the store's on every one of the 4,000 direct-I/O runs |
| Pack | levels digest = Phase 1B/1C's; content digest = the Phase 1C store's; every segment re-hashed on open |
| Reproducible | run1 and run2: identical pack, reference, validation, records and ranges digests, same source tree |

**Tests.** The suite has 399 tests, 76 of them new:

- storage: plans checked against brute force; exact reads with direct and buffered I/O; OS
  counters; no hidden reads; bytes outside the requested rows never reaching the output;
- streamer, caches and backend;
- packs, including tampering detection;
- the Phase 1C runtime on a pack, bit for bit equal to the resident runtime, on the drive, in
  host memory and with a cached base, with rows the run did not read overwritten;
- the real LM head;
- MoE on 7 architectures;
- layering.

**Guards confirmed to fail when disabled:**

| Guard sabotaged | What failed |
| --- | --- |
| The store reads whole segments (more than planned) | The no-hidden-reads test (data tests alone still passed) |
| A routed expert's slot poisoned instead of an unrouted one | The MoE output changes (the poison mode is meaningful) |
| A split extent shares a block with its neighbour | The plan test, before the fix (blocks would be read twice) |
| Hotness ties broken by set iteration order | Eviction sequences differ across `PYTHONHASHSEED` values (the regression test) |

**One defect found by the reproducibility check.**

- What happened: in the first pair of MoE runs, the records of the two hotness configurations
  differed between run1 and run2. Only their cache counters differed; every step was bit for
  bit the resident model's in both runs.
- Cause: pages inserted by one request tie on hotness. `min` over a set of string keys then
  picked the victim in a per-process hash order.
- Fix: every touch now gets a unique sequence number, and ties go to the least recently
  touched.
- Coverage: a test runs the cache in processes with different hash seeds. It was confirmed to
  fail with the old rule.
- Cleanup: all four benchmarks were run again on the corrected tree. The storage digests were
  unchanged; the storage benchmark uses no hotness cache.

## 5. Bytes (gate B)

Bytes per token as fractions of the BF16 LM head. Each cell gives the mean, then the median,
then the p95.

| Configuration | Logical | Drive | Host memory | Host → device | Amplification | Read calls |
| --- | --- | --- | --- | --- | --- | --- |
| A-direct | 1.000 | 1.000 | – | 1.000 | 1.00× | 55 |
| A-host | 1.000 | – | 1.000 | 1.000 | – | 1 |
| B / masked | 0.404 / 0.398 / 0.433 | – | – | – | – | – |
| B / full | 0.500 / 0.400 / 1.397 | – | – | – | – | – |
| **C-direct / masked** | 0.404 / 0.398 / 0.433 | **0.468 / 0.449 / 0.611** | – | **0.390 / 0.384 / 0.419** | 1.20× | 461 |
| C-direct / full | 0.500 / 0.400 / 1.397 | 0.564 / 0.465 / 1.443 | – | 0.486 / 0.386 / 1.383 | 1.16× | 466 |
| C-host / masked | 0.404 | – | 0.390 / 0.384 / 0.419 | 0.390 | – | – |
| C-cached-base / masked | 0.404 | **0.090 / 0.070 / 0.232** | – | **0.011 / 0.006 / 0.041** | 7.98× | 440 |

Where the drive bytes of C-direct (masked) go, per token:

| Segment | Rows | Logical | Drive | |
| --- | --- | --- | --- | --- |
| int6 base, every row | 49,152 | 0.3785 | 0.3785 | one sequential extent (21 calls of 1 MiB) |
| int4 refinement, contenders | 2,174 (mean) | 0.0112 | **0.0896** | about 440 extents; 292-byte rows in 4 KiB blocks, so 8× amplification |
| exact BF16 rows | 1.4 | 0.0000 | 0.0001 | from the published checkpoint |
| resident metadata (remainder norms) | – | 0.014 | 0 | held on the device (0.8 MB); counted as logical, as in Phase 1C |

- The masked fallback reads nothing more. The full fallback reads the rest of the head on the
  96 prompts it serves (1.47 of the head on those).
- Selecting rows costs nothing physical beyond the 4 KiB blocks around the refinement rows: the
  base level is one sequential read.
- That overhead (0.078 of the head) has three remedies, all left for later:
  - 512-byte reads, which the drive's logical sector allows. Estimated, not measured: an
    isolated 292-byte row spans about 800 bytes of 512-byte sectors, against 2,330 bytes read
    per row now.
  - row orders that cluster contenders;
  - caching the int4 level.
- **Gate B: PASS.** On the primary configuration the drive bytes are 0.468 and the
  host-to-device bytes 0.390, against limits of 0.75 and 0.75. The baseline reads and moves 1.0.
- **Invalidation check:** 0.468 is far below 0.90. The Phase 1C assumption holds physically.

## 6. Time and memory

Per token, run1. The device is synchronized at stage boundaries, as in Phase 1C. Each cell is
the mean (median, p95).

| Configuration | Total | Reads, waited (I/O + copy) | Drive I/O (host) | Copy (device) | Decode and matvec | Bounds | Certificate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A-direct | 22.2 (21.8, 23.9) | 21.8 | 18.2 | 9.0 | 0.5 (GEMV) | – | – |
| A-host | 9.6 | 9.1 | – | 8.8 | 0.4 | – | – |
| B / masked | 20.1 (20.0, 25.8) | 2.1 | – | – | 4.3 | 8.9 | 3.7 |
| **C-direct / masked** | **43.3 (43.9, 56.3)** | 24.6 | 15.6 | 3.6 | 4.5 | 9.3 | 3.7 |
| C-host / masked | 23.9 | 6.1 | – | 3.5 | 4.3 | 8.8 | 3.7 |
| C-cached-base / masked | 31.3 | 13.4 | 6.8 | 0.2 | 4.2 | 8.9 | 3.7 |

- Overlap already works. In A-direct, drive I/O (18.2 ms) and copies (9.0 ms) overlap within
  21.8 ms. The streamer reads piece k+1 while piece k is copied.
- The drive costs about 23 ms per token on top of the resident runtime:
  - the base level: 6.4 ms of sequential reading and 3.5 ms of copying;
  - about 440 random extents: about 7 ms;
  - planning, gathering, the Python thread hand-offs and stream synchronization (the rest).
- The resident runtime remains launch-bound eager PyTorch, as Phase 1C found. Turning bytes
  into time needs fused kernels (Phase 4) and native I/O submission, not more bytes saved.
- Memory:
  - device memory held between tokens: 93.2 MB resident (B), 0.8 MB on the drive (C), 22.2 MB
    with the cached base;
  - the transient device peak of a token is the runtime's own (98 MB mean, 355 MB on the
    prompt with the most contenders), the same in B and C;
  - host: 33.6 MB of pinned staging, plus 93 MB for the host tier.

## 7. Mixture of experts (gate C)

Sources: `experiments/phase3/moe-run1/summary.md` and `moe-run2`. Raw data: `reference.jsonl`
(the resident model's steps and routing) and `records.jsonl.gz` (every streamed step).

**Setup.**

- The model: `ibm-granite/granite-3.1-1b-a400m-instruct`. It has 24 experts modules of 32
  experts, 8 routed per token. Each expert is 3 MiB (gate and up, then down); 2.42 GB in all.
  The default experts implementation on CUDA is `grouped_mm`.
- The adapter replaces each module's stacked parameters by full-shape slot buffers (101 MB).
  The pre-hook fills the routed experts from the drive through `ExpertStore` →
  `MaterializationBackend` → `PageCache` → `FileBackedPageStore` → `PageStreamer`.
- The resident model runs first and is the reference of every step. Then its experts are
  released, and each configuration runs the same 50 prompts: on average 64 tokens (8 to 128),
  then 12 greedy decode steps.
- A cache persists across the prompts of its configuration, as in a serving process.

**Correctness.**

- Every streamed step's logits are bit for bit the resident step's, its token is the same, and
  its routing is the same: 3,275 of 3,275 steps per run.
- On the first three prompts, 39 steps, every unrouted slot was filled with NaN first, and the
  logits were still bit for bit equal: the experts implementation reads only routed experts.
- Every step's audit is clean:
  - the backend was asked for exactly the routed experts' rows;
  - the cache plus storage served them;
  - the device received exactly the fetched bytes;
  - every block read holds a requested byte;
  - the OS's read counters equal the store's.
- In tests, the same adapter is bit for bit on CPU and CUDA, in prefill and in decode, for 7
  architectures: Mixtral, Qwen2-MoE, Qwen3-MoE, OLMoE, GraniteMoE, DeepSeek-V3 and GPT-OSS
  (with biases and a transposed layout). It needs no change for any of them.

**Bytes, time and memory.** Fractions are of all expert bytes per step. Each cell is the mean,
then the median, then the p95.

| Configuration | Phase | Drive | Cache hit rate | Host → device MB | Step ms | Drive I/O ms | Copy ms | Device peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no cache | prefill | 0.952 / 0.977 / 0.992 | – | 2,300 | 1,093 | 749 | 379 | 491 |
| no cache | decode | **0.251** / 0.251 / 0.251 | – | 604 | 451 (p95 501) | 201 | 102 | 417 |
| LRU 25% | decode | 0.135 / 0.121 / 0.249 | 0.46 | 326 | 369 | 111 | 52 | 1,030 |
| hotness 25% | decode | 0.156 / 0.158 / 0.217 | 0.38 | 376 | 437 | 126 | 62 | 1,030 |
| LRU 50% | decode | 0.096 / 0.086 / 0.191 | 0.62 | 231 | 314 | 80 | 37 | 1,634 |
| hotness 50% | decode | **0.080** / 0.076 / 0.163 | 0.68 | 193 | 345 | 68 | 31 | 1,634 |
| hotness 50% | prefill | 0.699 / 0.710 / 0.761 | 0.27 | 1,686 | 1,473 | 547 | 269 | 1,732 |
| dense streaming | decode | 1.000 | – | 2,416 | 1,074 | 787 | 397 | 496 |

- **Selective reads work as designed.**
  - Without a cache, a decode token reads 0.251 of the expert bytes: 605 MB, at 3.0 GB/s, in
    307 extents. That is 4.0× less than dense streaming, and 2.4× faster.
  - Prefill routes 30.5 of 32 experts per layer, so it reads nearly everything. Caches matter
    for decoding, as in DwarfStar.
- **Caches.**
  - With half the experts cached, the drive bytes per decode token fall to 0.096 (LRU) or
    0.080 (hotness).
  - Hotness keeps the experts that prompts share, so it also helps prefill (0.70 instead of
    0.83 for LRU).
  - At a quarter of the experts, LRU does better in decode (0.135 against 0.156): recent
    reuse dominates there.
  - Hotness's eviction is a Python scan, which costs time (345 against 314 ms at 50%).
- **Memory.** Device memory falls from 2.70 GB to 0.39 GB: the non-expert weights (0.29 GB)
  plus one set of slot buffers (0.10 GB). A cache adds its budget.
- **Time** is dominated by the drive and by Python. A decode step without a cache takes 451 ms.
  - 201 ms are the drive's reads.
  - The other 250 ms, about 10 ms per layer, cover:
    - the copies (102 ms of device time, partly overlapping the reads);
    - the hook's Python work: deduplicating the routed experts (which waits for the device),
      planning and assembling;
    - the forward itself.

**Routing statistics** (resident reference):

- Decode routes 8.0 experts per layer; prefill 30.5.
- Routing is skewed: the most used 25% of (layer, expert) slots receive 45% of decode routings.
- **Of a decode step's experts, 50% were routed at the previous step in the same layer.** A
  prefetch of the previous step's experts would therefore fetch twice what it uses. Better
  predictors are a policy question for Phase 4, and the streamer's prefetch tickets are ready
  for them.

## 8. Findings

1. **Direct reads on NTFS are slowed by the OS cache of the same file.** While a file has an
   active cache map, each random direct read costs 59–90 µs instead of 14 µs. The cache map
   exists while a buffered handle or a memory mapping is open, and for about a second after it
   closes. Any of these causes it:
   - a buffered verification of a pack;
   - transformers loading a model from the same checkpoint;
   - a buffered configuration in the same process.

   The bytes do not change, so the OS counters still agree; only the time does. Packs are
   therefore verified with direct reads, and benchmarks keep tiers apart and settle after setup
   (decision 0006).
2. **Random-read cost is in the issuing, not in the drive.** From one Python thread, overlapped
   I/O is limited by issuing (21 µs per read); 8 threads with positioned reads reach 14 µs per
   extent (71 k reads/s). The drive itself can do far more.
3. **Freshly written files read slowly for a while.** This affected only setup in this phase:
   packs are built by `weightsift pack` in their own process.

## 9. Gates

| Gate | Criterion | Result |
| --- | --- | --- |
| A | Unselected LM-head data is not read; physical bytes measured; 0 correctness regressions | **PASS** (section 4) |
| B | Real byte reduction against the full BF16 head, logical and physical reported | **PASS**: drive 0.468, host → device 0.390 (≤ 0.75); amplification 1.20× |
| C | The same storage and materialization API serves the dense LM head and MoE experts, with no model-specific logic in the core | **PASS**: 3,275 of 3,275 streamed steps bit for bit; decode reads 0.251 of the experts without a cache (≤ 0.30); the core names no model (`tests/test_layering.py`) |
| Stop condition | Physical bytes invalidating Phase 1C (> 0.90) | not met (0.468) |

## 10. Recommendation: toward a DeepSeek-class demonstration

The architecture held. Nothing in the storage core or the adapter had to know Granite. The next
steps scale it one constraint at a time, and each keeps the bit-for-bit check against a
reference.

1. **Step 1: a MoE whose experts do not fit on the GPU, with the same code.**
   `allenai/OLMoE-1B-7B-0125-Instruct`: 64 experts, top-8, 16 layers, 12.9 GB of experts, more
   than this 8 GB GPU.
   - Load only the non-expert weights (experts on the meta device).
   - The reference becomes dense streaming (every expert every step), shown here bit for bit
     equal to the resident model.
   - Slot buffers are 768 MiB per set, which still fits.
   - Its checkpoint stores gate, up and down per expert, and the loader concatenates them. That
     needs item 2b, or a one-time repack.
2. **What a DeepSeek-class model needs that Granite did not:**
   1. *A compact expert call.* Full-shape slot buffers are E × expert bytes per layer: 11 GB for
      a DeepSeek-V3 layer (256 experts of 44 MB). Instead, call the module on a buffer of the k
      routed experts, with `top_k_index` remapped.
      - Exactness then depends on the experts implementation: `grouped_mm` sums in top-k order,
        eager accumulates in loop order. The remap must keep expert order, and the bit-for-bit
        and poison tests carry over directly.
      - It also enables DwarfStar's "hits first" overlap: computing cached experts while misses
        are read.
   2. *Zero-copy packs for split checkpoints.* A segment row composed of ranges of several
      source tensors (gate rows, then up rows). Otherwise a one-time repack doubles the disk
      (about 690 GB in FP8 for DeepSeek-V3).
   3. *A streaming packer and loader* that index and hash tensors without loading the model,
      and place only non-expert weights on the device.
   4. *Native read submission and per-layer batching.* Python issues about 14 µs per read, and a
      decode step spends about 10 ms per layer beyond the drive's reads. A small native extension, or `io_uring`/`IoRing`,
      is justified now: section 6 and this section are the profile.
   5. *A decision on the reference.* DeepSeek-V3 ships FP8 experts, and DwarfStar runs 2-bit
      ones. Certifying against a quantized reference changes decision 0001's reference. That is
      the user's call.
3. **The DeepSeek-class demonstration on this machine.** DeepSeek-V2-Lite (15.7 B; 64 routed
   plus 2 shared experts, top-6; 31 GB in BF16) or Moonlight-16B-A3B (DeepSeek-V3 architecture).
   - Both exceed the GPU, approach the 32 GB of RAM, and fit on the drive.
   - transformers already gives them the experts convention this adapter uses, and a tiny
     DeepSeek-V3 passes the tests.
   - Full DeepSeek-V3/R1 (671 B) needs about 700 GB of storage and item 2.5.
4. **Where certification adds value at that scale.** The LM head: DeepSeek-V3's is 129,280 ×
   7,168, 1.85 GB in BF16. The Phase 1C refinement, now physical, reads about 0.47 of it per
   token from the drive, or 0.09 with its base cached. Certifying through MoE layers is not
   recommended: Phase 2 found activation rounding, not weight bytes, to be the limit.
5. **Policies (Phase 4).** These questions are now measurable on real routing:
   - cache budget and policy: LRU at small budgets, hotness at larger ones;
   - an admission freeze during long prefills;
   - prefetch predictors: the previous step's experts waste half.

**Recommended next step:** item 1 (OLMoE beyond GPU memory) with items 2.1 and 2.2, then
DeepSeek-V2-Lite. Before that, the open decision on quantized references (item 2.5) belongs
to the user.

## 11. Limitations

- **One machine, one drive, one OS.** The direct-I/O behaviour and the NTFS cache effect are
  this platform's. The POSIX path (`os.preadv`, `O_DIRECT`, `/proc/self/io`) follows the same
  contract but is not exercised by the tests here.
- **Timing is eager Python and PyTorch.** Read issuing, planning and gathering run in Python, and
  the Phase 1C runtime is launch-bound. The byte results do not depend on this; the times do.
- **Start-up reads.** The masked fallback's self-test reads every exact row once when a head is
  built (56.6 MB). Pack verification re-hashes every segment.
- **Prefetch is implemented but no policy uses it yet.** The MoE routing statistics measure its
  potential (section 7).
- **MoE slot buffers have the full expert shape.** That is 96 MiB per layer for Granite; for
  DeepSeek-V3 it would not fit (section 10).

## 12. Reproduction

```bash
uv sync
uv run pytest                                                              # 399 tests
uv run weightsift pack lm-head                                                  # packs/ (deterministic)
uv run weightsift pack experts
uv run python benchmarks/storage_runtime.py --output experiments/phase3/<name>   # about 6 min (loop)
uv run python benchmarks/storage_report.py experiments/phase3/<name> --compare experiments/phase3/storage-run1
uv run python benchmarks/moe_runtime.py --output experiments/phase3/<moe name>   # about 30 min
uv run python benchmarks/moe_report.py experiments/phase3/<moe name> --compare experiments/phase3/moe-run1
```

A run is reproduced if its `digest.json` matches run1's. Both pairs of runs were made on the
same source tree, `a7c92b05…`.

| Benchmark | Digest | Value |
| --- | --- | --- |
| storage | pack | `62343ed5…` |
| storage | reference | `fc3a8706…` |
| storage | validation | `6e5ba26d…` |
| storage | records | `01e2c1c3…` |
| storage | ranges | `80ce1b79…` |
| MoE | pack | `6c18f449…` |
| MoE | prompts | `83c79399…` |
| MoE | reference | `9243f8c9…` |
| MoE | records | `a5c33c8a…` |

Timing fields (`timings_ms`, `system`) and `profile.json` are excluded from the digests.
