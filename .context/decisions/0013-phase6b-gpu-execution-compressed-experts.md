# 0013 — Phase 6B: experts encoded for the GPU, a device tier in fixed slots, a churn-free host cache, decode graphs

Status: accepted (Phase 6B, 2026-10-09)

## Context

Phase 6A (decision 0012) runs Moonlight-16B-A3B on a native (Rust) storage core at 807 ms per decode token with 12 GB of
host cache. What bounded a token then: copying its 2.70 GB of routed experts to the GPU (about 435 ms over this machine's
PCIe 3.0 x8), the transformer's Python and 5,195 kernel launches (about 260 ms), and the host cache's submit cost (56–105
ms, attributed to freeing and allocating entries of different sizes inline). Phase 5C (decision 0011) had found exact
independent compression at 0.661 of BF16 (bit planes with zstd) but no fast reconstruction.

The user's Phase 6B brief (2026-10-09, `state/phase_6B.md`):

- move fewer bytes to the GPU, remove runtime overhead and speed up decode, bit for bit equal to `BF16_REFERENCE` (no
  change to roundings, GEMM algorithms or batching, routing, the combine, normalization, attention and the KV cache,
  logits or token choice); a faster but numerically different path is experimental and never the reference;
- read and profile first; reproduce Phase 6A's baseline before changing performance-critical code;
- reuse mature native code (nvCOMP, the CUDA runtime, cuBLAS and graphs, PyTorch's extension facilities, DS4, Colibrì);
  test nvCOMP on Phase 5C's zstd frames as recorded (level 19), evaluate other codecs on the 5C corpus otherwise; record
  licenses; a custom kernel only for a measured, missing operation;
- stages in order, each measured alone and end to end, with stop conditions: 6B1 the cache's submit overhead; 6B2
  compressed experts to the GPU (versioned, indexed, independently addressable; the Rust cache holding compressed
  blocks; GPU decompression and BF16 restoration; validated against the checkpoint before inference; the uncompressed
  path kept); 6B3 a GPU expert cache (strict VRAM budget, deterministic eviction baseline, separate accounting) and native
  copy issuing if Python overhead remains; 6B4 CUDA Graphs on an isolated, graph-safe decode component first;
- Phase 6C preparation as documentation only; gates frozen before the final runs; two runs with different hash seeds;
  the report, this decision, `state/current.md`; commit and push; no Phase 6C.

## Decision

1. **Baseline first.** Phase 6A's best configuration measured again on its own tree: 810.8 and 806.9 ms per decode token
   (6A: 807.0), its trace kept as the reference point (5,195 launches, 524 ms of copies, the GPU idle 39–41%).
2. **The host cache holds rows in blocks of one size (6B1).** An evicted row's blocks go to a pool any later row takes
   from, whatever its size; the pool, the rows and the reservations of loads in flight never exceed the budget; the block
   is the greatest common divisor of the segments' rows of at least 1 MiB (at least 1 MiB). Tasks wake readers one at a
   time (a reader that takes a task wakes the next while tasks remain), not all of them per task: the second cause,
   found by the engine's own timers, was 76 ms per token. A request never waits for another request's load of a row: it
   reads the row itself without admitting it (a miss and a bypass), which removes a deadlock latent since Phase 6A (two
   jobs waiting on each other's staging slots); prefetched loads are still waited for. LRU order, leases, load-once and
   the hit and miss sequences are unchanged and independent of the block size. Submit: about 100 → 1 ms per decode token.
3. **Encoded rows: nvCOMP's rANS over raw BF16 rows (6B2), not Phase 5C's bit planes.** nvCOMP decodes 5C's zstd frames
   exactly, but zstd decompression plus the plane merge cost more than the copy they save (−54 ms per decode call net; a
   fused merge kernel could not recover the decompression). rANS in float16 mode over 1 MiB chunks of the rows as stored
   keeps 0.6747 of the bytes and decodes a call in 0.7 ms. The encoded pack (`weightsift-encoded-rows` v1,
   `awpmi.storage.encoded`; `weightsift pack encoded-experts`) holds every index row as independent chunks with their
   table, padded to its segment's stored size (a multiple of the pack's block), with each row's sha256 before and after
   encoding; it refers to nothing and is an optional second form of the experts. The native core reads and caches stored
   rows like any rows (it stays codec- and CUDA-free); `MaterializationBackend(decoder=RowDecoder(...))` fetches a call's
   stored rows into device staging and decodes them with one batched nvCOMP launch per codec into the experts' buffers,
   on the compute stream after the copies. nvCOMP is used through its C API (ctypes, `awpmi.streaming.nvcomp`), from the
   optional dependency group `gpu`; it is NVIDIA's proprietary SDK, never vendored or redistributed. Its rANS does not
   flag corrupt input: the decoder checks every chunk's status and size, and integrity rests on the pack's file hashes,
   an audit of every row decoded on the GPU against the reference before inference, and the step audits.
4. **A device tier of encoded rows in fixed slots (6B3-A).** `SlotCache`: one allocation per stored row size, made with
   the cache and carved into slots, each size with its own replacement policy (LRU: the deterministic baseline, and the
   best measured); `put` copies a fetched row into its slot on the compute stream, so a freed slot is overwritten only
   after every earlier use of it on that stream; hits are decoded from their slots and cross no bus. Its budget is the
   memory it holds (no allocator blocks, no fragmentation). It pays only from one token's working set (an LRU cliff at
   1.82 GB of encoded rows on the routing traces): 1.9 GB (162 slots per size), admission frozen during prefills, under
   the 6 GB process cap. A `PageCache` of copies in a memory pool remains for BF16 rows and the tests (`device_cache:
   pages`).
5. **No native copy issuing (6B3-B).** Issuing the streamer's copies through the CUDA runtime saves 11 ms of host time per
   token but 1.3% end to end, and nothing measurable once the fills are off: the copy engine paces the transfer.
6. **Decode graphs for DeepSeek-V3 (6B4).** `awpmi.models.decode_graphs.DecodeGraphs`: per decoder layer, the static decode
   pieces (before the KV cache: input norm, q and kv_a projections, kv_a norm, RoPE, the query's concatenation; after the
   attention: o_proj, the residual add, the post-attention norm, the router and the shared experts, or the dense MLP) as
   two CUDA Graphs; the KV cache, the attention over it, the routed experts and the block's sums stay eager between them,
   in transformers' order. The first decode step runs the pieces eagerly and keeps their inputs; then all pieces are
   captured back to back into one pool (captured one at a time, each graph kept a 32 MiB cuBLAS workspace). Only one
   token of one sequence with a non-empty cache is graphed. Observers' forward hooks on the stood-in modules are called
   with the replayed outputs; anything a replay would skip raises. The code between the graphs repeats transformers
   5.18's forwards and refuses other sources. Configuration option `decode_graphs: true`.
7. **PyTorch's NaN fill of uninitialized memory is a per-configuration option** (`fill_uninitialized_memory: false`). With
   deterministic algorithms on, PyTorch fills every `torch.empty` with NaN to expose reads of uninitialized memory; it
   changes no value that is read, and the reference process keeps it. The streamed runtime may turn it off; the digests
   check that nothing depended on it.
8. **Exactness and audits.** Every configuration is compared with the independent reference in every Phase 4B digest;
   encoded rows add an audit of every row before inference and per-step identities (decoded = requested bytes; device
   tier + store stored bytes = the requested rows' stored bytes); both cache tiers' hits and misses must equal a replay of
   the reference's routing (device tier first, the host cache behind it); budgets are checked from the caches' own
   counters. The runtime's recorder keeps a chunked call only when the step checks it, and every configuration's memory
   is collected before the next starts (both needed under the cap with the device tier; neither changes a record).
9. **Phase 6C preparation** is documentation (the report's § 12): shape-dependent kernel choices (kept by graph capture),
   workspace-dependent cuBLAS choices (left alone), the reductions' orders, where PyTorch changes execution without
   changing values, and the opportunities for an explicitly reproducible native reference. Nothing of 6C was built.

## Results

Full report: `history/2026-10-09-weightsift-phase6b-report.md`. Raw data: `experiments/phase6b/` (`baseline`,
`baseline-repeat`; `stage*`, `variants`, `codec`, `io`, `gpucache`, `copyissue`, `graphs`, `submit`, `pack`; the
correctness runs `run1` and `run2`, with the final profiles and `experiments/phase6b/run1/summary.md`).

- **Correctness** (gate): 1,116 steps per run over 10 configurations (BF16 and encoded rows; host caches; device tiers with
  hits, misses and evictions; graphs with the fills off; chunked calls; poisoned spares) equal to the reference in every
  digest; two runs (`PYTHONHASHSEED` 1 and 2) with identical digests on the same source and native trees; I/O parity of
  the backends; every cache tier equal to its replay; the encoded pack's 3,328 rows equal to the reference's.
- **6B1**: engine submit 0.7–1.5 ms per decode token (gate ≤ 10; 6A 56–105); BF16 host-cache decode 0.964 of the Phase 6A
  tree's (gate ≤ 0.99): 787.2 against 808.9 ms (12 GB, freeze), 903.8 against 944.6 (4 GB).
- **6B2**: encoded rows 610.8 against 787.2 ms per decode token at the same 12 GB host cache (0.776, gate ≤ 0.90); drive
  0.758 → 0.241 GB and copies 2.70 → 1.82 GB per token; decoding about 33 ms of GPU and 17.5 ms of host time per token.
- **6B3-A**: the 1.9 GB device tier 493.7 against 610.8 ms (0.808, gate ≤ 0.95), 0.38 of decode bytes served on the
  device; peak 5.80 GB reserved under the 6.0 GB cap.
- **6B4 with the fills off**: 408.9 against 493.7 ms (0.828, gate ≤ 0.97); graphs alone −12.5%, the fills alone −6.2%;
  2,344 kernel and 54 graph launches per token instead of 5,195; 54 graphs, 65–68 MB.
- **The phase** (gate ≤ 0.90 of 807 ms): **408.9 ms per decode token, 0.507** (2.45 tokens/s against 1.24; the brief's
  aspirational 0.80 met); seeds 1 and 2 within 0.1–0.4%; prefills 3.6 s against 5.6 s.

## Rejected

- *Phase 5C's bit planes decoded on the GPU* (nvCOMP's zstd: exact; the merge alone 50 ms per call): net −54 ms per call;
  byte-split zstd −6.8 ms; *CPU decompression then BF16 copies*: 71–362 ms per call and no fewer bus bytes.
- *A custom CUDA kernel for the plane merge*: zstd's decompression alone exceeds the copy time it would save.
- *GDeflate, LZ4, Bitcomp, rANS over split bytes*: lower net gain than rANS over raw rows (§ 6 of the report).
- *A device tier of copies in a CUDA memory pool*: fragments and exceeds the 6 GB cap at the size that pays; *hotness*:
  fewer hits at 2 GB and 61 ms per token in its victim search; *a device tier below one token's working set*: the LRU
  serves nothing there.
- *Native copy issuing (C++/CUDA or the runtime through ctypes)*: −1.3% alone, unmeasurable with the fills off.
- *Graph capture one layer at a time inside the step*: 1 GiB of cuBLAS workspaces; *graphs of the attention over the
  cache*: dynamic shapes, and a static, padded cache changes the attention's kernels (6C first); *torch.compile*: fuses
  and reorders operations, changing roundings.
- *Turning off deterministic algorithms* (instead of only the fills): it changes kernel choices.
- *A smaller cuBLAS workspace to save graph memory*: it can change cuBLAS's algorithms, so the reference.
