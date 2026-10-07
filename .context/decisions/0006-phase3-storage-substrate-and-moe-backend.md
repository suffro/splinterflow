# 0006 — Phase 3: physical selective materialization, a general storage backend, MoE experts

Status: accepted (Phase 3, 2026-10-02)

## Context

Through Phase 2 every store was resident: materialization was logical. The Phase 1C LM head
needs 0.40–0.50 of the BF16 head's bytes per token on paper (decision 0004), but nothing was
read from storage.

The user's Phase 3 brief:

- Move Weightsift/AWPMI from a resident-memory research runtime to a real selective-materialization
  system. The long-term target is running models larger than accelerator memory, materializing
  only the weight data needed to certify the same discrete decision as the full model.
- Build a generic storage layer (`PageStore`: in memory and file-backed) and a separate
  streaming layer (`PageStreamer`), reusing Soup's streaming patterns without depending on Soup.
  Do the synchronous path first, then async and prefetch.
- Run the validated Phase 1C LM head on that storage without changing its mathematics. Measure
  real bytes and time against the full BF16 head and the resident runtime.
- Then a general backend for models whose weights exceed RAM or VRAM (`WeightStore`,
  `PageCache`, `ExpertStore`, `MaterializationBackend`), with MoE support inspired by DwarfStar
  (SSD expert storage, expert cache, eviction, router-driven requests, prefetch). It must not
  become a DwarfStar wrapper or DeepSeek-specific. The first target is the smallest practical
  open MoE.
- Keep the layers separate: certification, materialization policy, model adapter, storage
  backend, transfer layer.
- Gates: A (physical selective materialization), B (real byte reduction), C (general backend).
- Stop and report if the real physical bytes invalidate the Phase 1C assumptions.

## Decision

1. **Layers, as packages.**
   - Storage: `awpmi.storage` (layout, file I/O, page stores, page cache, packs).
   - Transfer: `awpmi.streaming` (`PageStreamer`).
   - Materialization: `awpmi.materialization` (`MaterializationBackend`, `WeightStore`,
     `ExpertStore`).
   - Model adapters: `awpmi.models.smollm2` and `awpmi.models.moe`.
   - Policies: the Phase 1C certificate's contenders, the experts a router selects, the cache's
     replacement policy.
   - Certification: unchanged.

   `tests/test_layering.py` enforces two rules. The storage core names no model, tensor or router
   and imports no layer above it. Certification imports no storage.
2. **Pages are rows of segments; segments are safetensors tensors as they are** (roadmap §3.3).
   - A segment is fixed-size rows at an offset of a file. A tensor [R, ...] is R rows.
   - Read plan:
     - consecutive requested rows form runs;
     - each run is widened to the 4 KiB I/O unit;
     - runs merge into one extent when their blocks touch (`max_gap` 0);
     - an extent holds at most 8 MiB, but runs that share a block are never split.
   - A file-backed store reads only the planned extents. Every block read holds a requested byte,
     and bytes outside the requested rows never reach the output. Both properties are tested:
     every read is recorded and must tile a planned extent, and the file is overwritten outside
     the requested rows. The first property was confirmed to fail when the store reads whole
     segments.
3. **Positioned reads, a thread pool, direct I/O.**
   - Windows: `ReadFile` with an `OVERLAPPED` offset, on one synchronous handle per thread (a
     synchronous handle serializes its operations). POSIX: `os.preadv`.
   - Direct mode (`FILE_FLAG_NO_BUFFERING` / `O_DIRECT`) sends every read to the drive.
   - 8 worker threads, one task each; at most 1 MiB per call.
   - Prototype on this machine (Samsung 990 PRO, NTFS):
     - one Python thread with overlapped I/O is limited by issuing, at 21 µs per read;
     - 8 threads reach 14 µs per random 4 KiB extent, 2.6–3.3 GB/s sequential.
4. **Every byte counted and cross-checked.**
   - Logical: the store's log, unchanged from Phase 1C.
   - Requested from the backend, served by the cache, read by storage.
   - Physical: the bytes the reads returned; plus read calls, extents and distinct 4 KiB blocks.
   - Host-to-device: bytes and copies.
   - The physical reads and bytes must equal the OS's own per-process counters
     (`GetProcessIoCounters`; `/proc/self/io` on Linux). They did, exactly, from the first
     prototype on.
5. **OS page cache and direct reads (finding).**
   - On NTFS, a direct read of a file with an active cache map costs about 4–6 times more: 59–90
     µs per random extent instead of 14 µs. The cache map exists while a buffered handle or a
     mapping is open, and for about a second after it closes.
   - A pack verified with buffered reads, a model loaded from the same checkpoint, or a buffered
     configuration in the same benchmark therefore slows every direct read.
   - Consequences:
     - packs are verified with direct reads;
     - benchmarks never mix buffered and direct access to one file, and settle 3 s after setup;
     - the RAM tier is benchmarked as an explicit pinned host-memory store (`Pack.load`), not
       through the OS cache;
     - buffered mode stays supported and tested.
6. **`PageStreamer` (Soup's patterns, adapted).**
   - Two pinned, page-aligned staging slots of 8 MiB, with gather buffers: double buffering.
   - A CUDA copy stream, and per-slot events: a slot is refilled only after its copy completed.
   - The compute stream waits on a ready event; `record_stream` protects the allocator.
   - A file-backed fetch reads slot-sized pieces. The storage read of piece k+1 overlaps the
     copy of piece k.
   - Only the requested rows cross to the device: they are gathered on the host. Copying whole
     blocks and gathering on the GPU would move the 4 KiB amplification across PCIe.
   - Prefetch: a background thread, with tickets; consumed and wasted bytes are counted. It is
     implemented and tested; no policy uses it yet (point 12).
7. **`PageCache`** (the device tier).
   - A byte budget, with pinned entries.
   - LRU, or hotness: exponentially decayed access counts, DwarfStar's "route hotness".
   - An admission freeze: DwarfStar's fix for long prefills.
   - A cache decides what stays resident, never what bytes are. It affects efficiency only.
   - Eviction is deterministic. Hotness ties go to the least recently touched page, by a unique
     sequence number, never by set iteration order: string keys iterate in a per-process hash
     order. The first pair of MoE runs showed that defect.
8. **`MaterializationBackend`.**
   - `materialize(segment, rows)` returns device rows. A store in memory on the compute device
     answers directly; otherwise the cache, then the streamer.
   - `WeightStore` gives typed rows. `ExpertStore` serves expert groups: `load`, and `fill` into
     full-shape buffers.
9. **Packs** (roadmap §3.2).
   - `manifest.json` records:
     - the format version;
     - files, with size and sha256;
     - segments, with offset, rows, row bytes, dtype and the sha256 of their bytes;
     - the source model and revision, the decomposition, the levels digest;
     - the packing configuration.
   - A segment whose bytes are a published checkpoint tensor refers to it instead of copying it.
     Two cases do so:
     - the LM head's exact rows are SmolLM2's tied embedding;
     - Granite's stacked expert tensors are only renamed by the transformers loader.
   - `open_pack` re-hashes every segment, with direct reads.
   - `weightsift pack lm-head` and `weightsift pack experts` write packs. Packs live in `packs/`: they are
     deterministic and gitignored.
10. **The Phase 1C LM head on storage, mathematics unchanged.**
    - `PackedRefinementStore` reads through a backend.
    - A level row is one record, the packed codes followed by the float32 scale (decision 0004's
      unit), so one row is one contiguous read.
    - The resident configuration reproduces the Phase 1C run's records field by field. The
      file-backed runtime reproduces the resident one bit for bit: decision, reads, every
      state's bounds and the fallback logits.
    - The masked fallback's start-up self-test reads every exact row once. That is start-up I/O,
      not per-token I/O.
    - The runtime gains `*:read` timer stages, for timing only.
11. **MoE adapter** (`awpmi.models.moe`), over transformers 5's experts convention only.
    - It finds modules with `num_experts` and a ≥3-D parameter whose first dimension is that
      count, so routers are excluded.
    - It replaces their expert-sliced parameters by full-shape slot buffers.
    - A pre-hook takes `top_k_index`, materializes those experts and writes them into their rows.
      Then the module's own forward runs on the same bytes as the resident model.
    - Rows of unrouted experts hold stale data. A poison mode fills them with NaN to check that
      no experts implementation reads them. Validated:
      - bit for bit on 7 architectures (Mixtral, Qwen2-MoE, Qwen3-MoE, OLMoE, GraniteMoE,
        DeepSeek-V3, GPT-OSS), on CPU and CUDA, in prefill and in decode with the KV cache;
      - on the real Granite model in the benchmark.
    - Poisoning a routed row instead changes the output: the sabotage check.
12. **First MoE: `ibm-granite/granite-3.1-1b-a400m-instruct`** at `0da7a48b`.
    - Why: 1.3 B parameters; 24 layers of 32 experts, 8 routed per token; one 2.67 GB file;
      Apache 2.0.
    - The experts are stacked [32, …] tensors, so each expert is a contiguous 3 MiB in the
      published file, and its pack copies nothing.
    - Its resident reference fits on this 8 GB GPU, so every streamed step can be compared bit
      for bit.
    - Speculative prefetch is measured first, not built. The benchmark computes how many of a
      decode step's experts the previous step had routed in the same layer.
13. **Gates, fixed in config before the full runs.**
    - A: every Phase 1C check, every storage run equal to the resident one, the resident one
      equal to Phase 1C, and every storage audit and OS cross-check holding.
    - B: on the primary configuration (drive, direct I/O, masked fallback), at most 0.75 of the
      BF16 head read from the drive and moved to the device per token. Invalidation if the drive
      bytes exceed 0.90.
    - C: the same API serves Granite with every step equal bit for bit to the resident model,
      every audit holding, and a decode step without a cache reading at most 0.30 of the expert
      bytes. The layering test also holds.

## Results

Full report: `history/2026-10-02-awpmi-phase3-report.md`.

- **LM head on the drive** (SmolLM2, 1000 prompts, two runs with identical digests). Per token,
  as fractions of the BF16 head:
  - masked fallback: logical bytes 0.404 (as Phase 1C); drive 0.468 (median 0.449, p95 0.611);
    host-to-device 0.390;
  - amplification 1.20×: the int6 base is one sequential read, and 4 KiB blocks around the
    int4 rows cost 0.078;
  - with the base level cached on the device: drive 0.090, host-to-device 0.011.

  The full head from the drive reads and moves 1.0. There are 0 mismatches. The drive-backed
  runtime is bit for bit the resident one, and the resident one reproduces Phase 1C field by
  field. Every storage audit and OS cross-check holds.
- **Time** (not gated): 43.3 ms per token on the drive, against 22.2 ms for the full head from
  the drive and 20.1 ms resident.
  - The eager PyTorch runtime is still about 20 ms.
  - The drive adds about 23 ms of waited reads: a 6.4 ms sequential base, about 440 random
    extents at 14 µs each, and Python planning and gathering.
- **MoE** (Granite 3.1 1B-A400M, 50 prompts × 13 steps, two runs).
  - All 3,275 streamed steps are bit for bit the resident model's, including 39 with every
    unrouted slot poisoned.
  - Device memory falls from 2.70 GB to 0.39 GB.
  - A decode token reads 0.251 of the expert bytes without a cache (605 MB, 451 ms), 0.096 with
    an LRU cache of half the experts, and 0.080 with a hotness cache; dense streaming reads 1.0
    in 1,074 ms.
  - The previous step had routed 50% of a decode step's experts in the same layer.
- **Gates:** A, B and C pass. The Phase 1C assumption holds physically: 0.468 against an
  invalidation threshold of 0.90.

## Rejected

- *A custom storage format or a database.* Safetensors plus a manifest suffice. The one layout
  choice, the level record, is an ordinary U8 tensor.
- *Depending on Soup's or DwarfStar's code.* Their patterns were adapted: staging slots, a copy
  stream, events and prefetch; an expert cache with a byte budget, hotness and an admission
  freeze.
- *The OS page cache as the RAM tier in the benchmark.* It is shared state, and it slows direct
  reads of the same file (point 5).
- *mmap for reads.* Page-fault granularity and read-ahead are the OS's choice, so the bytes read
  could not be planned or counted.
- *Overlapped I/O from one Python thread, or `IoRing`/`io_uring`.* The first is limited by
  issuing; the others are platform-specific and more machinery than the 14 µs per extent that
  threads already reach.
- *Copying whole blocks to the GPU and gathering there.* It would move the 4 KiB amplification
  across PCIe.
- *A compact, remapped expert call* (slot ids in place of expert ids). Its exactness would depend
  on how each experts implementation orders its work: eager accumulates expert outputs in loop
  order. It is the next step for DeepSeek-class layers, where full-shape buffers do not fit
  (E × expert bytes = 11 GB per DeepSeek-V3 layer). **Adopted in decision 0007 (Phase 4A)**, with
  slots in ascending expert order, which keeps eager, `grouped_mm` and `batched_mm` bit for bit.
- *Speculative next-layer prefetch now.* It is measured instead (point 12).
- *DeepSeek-class directly.* The user ruled it out until the generic storage and MoE
  abstractions work.
