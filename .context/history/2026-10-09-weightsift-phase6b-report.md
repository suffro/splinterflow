# Weightsift Phase 6B report — fewer bytes to the GPU, less host work per token

Date: 2026-10-09 · Follows: `history/2026-10-08-weightsift-phase6a-report.md` (Phase 6A, the runtime it extends) and
`history/2026-10-08-awpmi-phase5c-report.md` (Phase 5C's exact expert codecs) ·
Decision: `decisions/0013-phase6b-gpu-execution-compressed-experts.md` · Brief: `state/phase_6B.md` ·
Status: **complete. Every gate passes: every step of every configuration equals the independent reference in every
digest, in two runs with identical digests; the best configuration decodes a token in 408.9 ms, 0.507 of Phase 6A's 807
ms (2.45 tokens/s against 1.24).**

## Outcome in brief

Phase 6B cuts what a Moonlight-16B-A3B decode token costs on this machine without changing a bit of what it computes. The
host cache stops freeing and allocating on the critical path; the routed experts travel to the GPU compressed and are
restored there exactly; a small device tier keeps the previous token's experts; and the transformer's static decode
pieces are replayed from CUDA Graphs.

| Question | Answer |
| --- | --- |
| Phase 6A reproduced first | yes: 810.8 and 806.9 ms per decode token on the Phase 6A tree (6A: 807.0); its trace: 5,195 kernel launches per token, 524 ms of copies to the device, the GPU idle 39–41% of a step |
| 6B1: the host cache's submit cost | from 99 ms to 0.7–1.5 ms per decode token. Two causes, both found by measurement: inline frees and allocations of entries of different sizes (rows are now held in blocks of one size, reused across row sizes, the pool inside the budget), and waking all 8 readers on every task (`notify_all`: 76 ms per token; now one wake-up at a time). Decode with BF16 host caches 0.964 of the Phase 6A tree's. A latent Phase 6A deadlock (two jobs waiting on each other's loads) found and fixed |
| 6B2: Phase 5C's zstd frames on the GPU | nvCOMP 5.3 decodes them exactly (level 19 included), but restoring BF16 from bit planes costs more than the copy it saves (−54 ms per decode call net); byte-split zstd −6.8 ms. **nvCOMP's rANS over raw BF16 rows** (float16 mode, 1 MiB chunks) stores 0.6747 of the bytes and decodes a call in 0.7 ms: +4.5 ms per call. No custom kernel was justified |
| 6B2: encoded rows end to end | a versioned, indexed pack (19.42 GB for 28.79 GB of experts), read by the native core like BF16 rows, decoded on the GPU into the experts' buffers: 610.8 against 787.2 ms per decode token with the same 12 GB host cache (**0.776**); drive bytes 0.758 → 0.241 GB per token, bytes to the GPU 2.70 → 1.82 GB; every row checked against the reference before inference |
| 6B3-A: a device expert cache | pays only from one token's working set (1.82 GB of encoded rows: an LRU cliff, measured on the routing traces); a pool of copies fragments and runs out of the 6 GB cap; **fixed slots** (162 per stored row size, 1.89 GB, allocated once) fit and serve 0.38 of decode bytes: 493.7 against 610.8 ms (**0.808**) at the same host budget |
| 6B3-B: native copy issuing | measured, not adopted: issuing through the CUDA runtime saves 11 ms of host time per token but 6.5 ms end to end (1.3%), about 2 ms once the next change is in; the copy engine paces the transfer |
| 6B4: CUDA Graphs | the static decode pieces of every layer (norms, projections, RoPE, o_proj, router, shared experts) as 54 graphs replayed bit for bit (tests; Moonlight's every digest), the KV cache, attention and routed experts eager between them: −12.5% alone. With PyTorch's NaN fill of uninitialized memory turned off (a debugging aid of deterministic mode: 270 fill kernels per token; no arithmetic changes), **408.9 against 493.7 ms (0.828)** |
| Correctness | 1,116 steps per run (10 configurations: BF16 and encoded rows, host caches, device caches with hits and evictions, graphs, chunked calls, poisoned spares) equal to the reference in every Phase 4B digest; two runs (`PYTHONHASHSEED` 1 and 2) with identical digests; I/O parity of the native and Python backends; every cache tier's hits and misses equal a replay of the reference's routing |
| Performance (phase gate ≤ 0.90 of 6A) | **PASS: 408.9 ms per decode token, 0.507 of Phase 6A's 807** (the brief's aspirational 0.80 met); repeats with another hash seed within 0.1–0.4%; prefills 36% faster than 6A |
| Phase 6C preparation | the reference's shape-dependent kernels, reduction orders and the places PyTorch changes execution are listed (§ 12); graph capture kept every kernel choice; nothing of 6C was started |

## 1. Questions and plan

The user's brief (2026-10-09, `state/phase_6B.md`) asks Phase 6B to reduce GPU transfer costs, remove runtime overhead and
speed up decode while staying bit for bit equal to `BF16_REFERENCE`, in stages measured one at a time:

- **read and profile first**, and reproduce Phase 6A's baseline before changing performance-critical code;
- **reuse native libraries** (nvCOMP, the CUDA runtime, cuBLAS and graphs, PyTorch's extension facilities, DS4 and
  Colibrì): test whether nvCOMP decodes Phase 5C's zstd frames (recorded version and settings, level 19), evaluate other
  codecs on the 5C corpus if not, record licenses; a custom CUDA kernel only for a measured, missing operation;
- **6B1**: eliminate the host cache's submit overhead (LRU, hit/miss sequences, the strict budget with pool memory,
  leases and in-flight deduplication preserved; benchmarked alone and in Moonlight; end-to-end benefit required);
- **6B2**: compressed experts directly to the GPU (a versioned, indexed, independently addressable representation; the Rust
  cache holding compressed blocks; compressed bytes over PCIe; GPU decompression and BF16 restoration; validation against
  the checkpoint before inference; compared with uncompressed streaming, CPU decompression, and an alternative codec;
  gated on useful end-to-end savings);
- **6B3**: (A) a GPU expert cache (strict VRAM budget, deterministic eviction baseline, separate accounting of compressed
  and restored tensors, marginal value at equal host budgets); (B) native CUDA copy submission if profiling still shows
  meaningful Python overhead in `PageStreamer._transfer_native`;
- **6B4**: fewer kernel launches with CUDA Graphs on an isolated, graph-safe decode component first, larger regions only
  if it pays; replay bit for bit or eager execution kept;
- **6C preparation** (documentation only), and the finalization (tests, two runs with different hash seeds, the report,
  the decision, `state/current.md`, commit and push; no Phase 6C).

The gates were frozen in `configs/phase6b-gpu.yaml` (`gates`) after the exploratory profiles and before the final runs:
correctness, the encoded pack's audit, I/O parity, the caches' replays and budgets, and per stage a separately measured
gain (6B1: engine submit ≤ 10 ms per decode token and BF16 host-cache decode ≤ 0.99 of the Phase 6A tree's; 6B2: encoded
≤ 0.90 of BF16 rows at the same host budget; 6B3-A: device cache ≤ 0.95 of the same configuration without it, within the
GPU cap; 6B4 with the fills: ≤ 0.97; the phase: best ≤ 0.90 of 807 ms, the brief's 0.80 reported).
`benchmarks/gpu_report.py` evaluates them.

## 2. Setup

Phase 6A's machine, software, model, prompts and budgets (its report's § 2), plus:

| Item | Value |
| --- | --- |
| GPU decompressor | NVIDIA nvCOMP 5.3.0 (`nvidia-libnvcomp-cu13` 5.3.0.16, CUDA 13 runtime), through its batched C API (ctypes, `awpmi.streaming.nvcomp`) |
| PCIe | the RTX 4060 Ti on PCIe 3.0 x8: 6.1–6.2 GB/s per direction measured; an idle GPU drops its link to Gen 1 (a timing trap the I/O benchmarks avoid by running calls back to back) |
| Process caps | Phase 4B's: GPU cap 6.0 GB (5.59 GiB) for the whole process, device tier included; host cache budgets 4 and 12 GB |
| Reference | Phase 6A's `baseline-run1` (Phase 4B's streaming reference, computed once); every run of this phase is compared with it (`--reference-from`) |

## 3. Reuse and licenses

- **nvCOMP** (NVIDIA, proprietary: the NVIDIA SDK license; installed from PyPI as a separate, optional dependency group,
  `gpu`, never vendored or redistributed by Weightsift; Linux and Windows wheels). Used through its documented batched C
  API with device-side arrays of chunk pointers and sizes. Tested here: its zstd decoder on Phase 5C's frames (exact,
  level 19 included), and its rANS, GDeflate, LZ4 and Bitcomp codecs on the 5C corpus. Chosen: rANS (`ans`, float16 data
  type). Not used: its high-level C++ manager API (a C++ build for no gain), its CPU library.
- **The CUDA runtime and cuBLAS**: through PyTorch. CUDA Graphs through PyTorch's `torch.cuda.CUDAGraph` and graph pools
  (no C++). The CUDA runtime was also called directly (`cudaMemcpyAsync`, ctypes) for the 6B3-B measurement only.
- **PyTorch C++/CUDA extensions**: not needed. Neither GPU decoding (nvCOMP's kernels), nor the device tier (copies),
  nor graphs (PyTorch's API) needed code of Weightsift's own in C++ or CUDA; native copy issuing, the one candidate, does
  not pay (§ 8).
- **DS4 / DwarfStar and Colibrì** (Phase 6A's review): DS4's device-side expert cache with recency stamps, hits protected
  before victims, and its admission freeze during prefills are the device tier's pattern (§ 7); Colibrì's per-layer LRU
  with a learned pinned set was considered: a per-layer partition of a device budget below one token's working set falls
  off the same LRU cliff (§ 7).
- **No custom kernel**: the only candidate was a fused bit-plane merge for Phase 5C's format, and zstd's decompression
  alone exceeds the copy time it saves (§ 6).

## 4. Baseline: Phase 6A reproduced before any change

`experiments/phase6b/baseline` and `baseline-repeat` (Phase 6A's tree, the 6B config's prompts and profile method):

| Profile (warm unless noted) | Decode ms/token | Note |
| --- | --- | --- |
| `native-host-12g-freeze` | 810.8, 806.9 (cold 812.9) | Phase 6A: 807.0 |
| `native-host-12g` | 841.7, 842.8 | |
| `native-host-4g` | 942.8, 946.4 | |
| `native-stream` | 1,135.0 | |

The traced decode step (`native-host-12g-freeze`): 5,195 kernel launches (85 ms of launch CPU), 524.6 ms of host-to-device
copies (traced), 66 ms of GEMMs, 40 ms of other kernels, 2.4 ms of attention, the GPU idle 39–41% of the step. Among the
other kernels, 270 BF16 fill launches took 27.7 ms per token, outside every module scope: a lead § 9 resolves.

## 5. Stage 6B1: the host cache's submit cost

**Causes, measured.** The submit microbenchmark (`experiments/phase6b/submit`, two row sizes, evicting: 1.30 ms per
transfer) pointed at evictions whose memory could not be reused (an 11.5 MB gate/up entry evicted for a 5.8 MB down row
is freed, and a new buffer allocated, inline). Recycling across sizes alone took Moonlight's submit from 98.9 to 78 ms per
token: the profile's engine timers then showed `start_pieces` at 76 ms per token: waking the 8 readers with
`notify_all` on every pushed task cost about 0.58 ms per push in the model's process.

**Fix** (`native/core/src/cache.rs`, `engine.rs`):

- rows are held in blocks of one size (`block_for`: the greatest common divisor of the segments' rows of at least 1 MiB,
  at least 1 MiB: 5.5 MiB for Moonlight's BF16 rows, 3.7 MiB for its encoded rows); an evicted row's blocks go to a pool
  any later row takes from, whatever its size; the pool, the rows and the reservations of loads in flight together never
  exceed the budget (surplus blocks are freed); 512 KiB blocks were measured and rejected (the heap zeroes small
  allocations: 2.2 ms per row);
- a pushed task wakes one reader; a reader that takes a task wakes the next while tasks remain;
- LRU order, leases, load-once and the hit and miss sequences do not depend on the block size (tests); an aborted fill
  keeps its reservation until its memory comes back; clearing gives every block back.

**A latent Phase 6A deadlock**: `concurrent_jobs_load_each_row_once` hung in 4 of 30 serial runs (about 10 of 12 on the
6A tree): two jobs could each wait for a row the other was loading while holding the staging slots the other needed. Now
a request never waits for another request's load: it reads the row itself, without admitting it (a miss and a bypass),
and only prefetched loads are waited for; 0 hangs in 60 runs, and the test fails again when the rule is removed.

**Results.** Submit (the microbenchmark): 1.30 → 0.23 ms per transfer. A replay of the routing through the engine without
the model (12 GB): 1.75 ms of engine submit per token, all 9,612 evictions recycled, 12.00 GB allocated, none released.
In Moonlight (final profiles): **0.7–1.5 ms of engine submit per decode token** (Phase 6A: 56–105 ms); warm decode
787.2 ms (12 GB, freeze) against 808.9 on the Phase 6A tree, 903.8 against 944.6 (4 GB): **0.964** over both (gate ≤
0.99); prefills that admit 5–6% faster; streaming unchanged; host memory unchanged.

## 6. Stage 6B2: experts encoded on the drive, restored on the GPU

**Phase 5C's format on the GPU** (`benchmarks/gpu_codec_probe.py`, `experiments/phase6b/codec/probe.json`; per decode
call of 6 experts, 104 MB of BF16, whose copy takes 16.0 ms):

| Representation | Stored / BF16 | Restore on the device (ms per call) | Net per call (ms) |
| --- | --- | --- | --- |
| 5C bit planes per tensor, zstd-19 (nvCOMP zstd: exact) | 0.661 | 8.45 decompression + 50.2 plane merge | −53.6 |
| 5C byte split (high and low bytes), zstd-19 | 0.6725 | 10.8 + 1.2 | −6.8 |
| nvCOMP rANS, float16 mode, raw BF16 rows, 64 KiB chunks | 0.693 | 0.73 | +4.2 |
| nvCOMP rANS, float16 mode, 1 MiB chunks | 0.674 | 0.69 | +4.5 |
| nvCOMP rANS, 2 / 4 / 12 MiB chunks | 0.6725 / 0.6718 / 0.6713 | 0.71 / 0.73 / 0.81 | – |
| nvCOMP rANS over the high bytes only, 256 KiB | 0.684 | 0.27 + 1.29 merge | +3.5 |
| nvCOMP GDeflate (entropy only), high bytes split | 0.6735 | 3.26 | +2.0 |
| CPU: 5C planes / byte split at zstd-1 | – | 362 / 71 (then the BF16 copy) | – |

nvCOMP reads 5C's zstd frames (the recorded zstd, level 19) bit for bit, but zstd decompression on this GPU already costs
more than half the copy it saves, before any plane merge; a fused merge kernel could not change that, so none was
written. rANS over raw BF16 rows needs no layout change at all and costs 0.7 ms per call. **Chosen: rANS, float16 mode,
1 MiB chunks** (smaller chunks cost ratio, larger ones decode time).

**The encoded pack** (`weightsift pack encoded-experts`, `awpmi.storage.encoded`; format `weightsift-encoded-rows` v1):
every expert row of the index in independent 1 MiB chunks, each compressed with nvCOMP and checked by a GPU round trip
when written; chunks 16-byte aligned in the row; rows padded to their segment's stored size (a multiple of the pack's
block: gate/up 2 blocks of 3,891,200 bytes, down 1); one file per expert group; the manifest records the codec, the
library versions, each row's chunk table, and each row's sha256 before and after encoding. It is a second copy of the
experts in another form (19.42 GB for 28.79 GB; written in 178 s), not a copy of the checkpoint: the index still refers
to the checkpoint for BF16 rows. Stored 0.6747 of BF16 (0.6733 compressed, the rest alignment).

**The decoder** (`awpmi.streaming.codec.RowDecoder`, `awpmi.streaming.nvcomp`): a call's encoded requests are fetched
together into device staging (the native core reads and caches stored rows exactly like BF16 rows: it knows bytes, not
codecs), then one batched nvCOMP launch per codec decodes every chunk into the experts' compact buffers, on the compute
stream after the copies. Counted: logical bytes requested and decoded, stored bytes served and fetched. nvCOMP's rANS
does not detect corruption (a zeroed header decodes as zero bytes with status 0, a flipped payload decodes wrong): the
decoder checks every chunk's status and size, and integrity rests on the pack's sha256 (`verify="files"` before a run),
the audit below, and the step audits.

**Validation before inference** (every run): all 3,328 rows read from the drive and decoded on the GPU by the runtime's
decoder, each sha256 compared with the reference's row digest and the pack's record: 0 differing.

**I/O per decode call** (`benchmarks/encoded_io.py`, `experiments/phase6b/io/encoded-io.json`; ms, mean and median, calls
back to back):

| Path | Cold host cache | Warm (rows in host RAM) |
| --- | --- | --- |
| BF16 rows, native, uncompressed H2D (Phase 6A) | 39.5 (35.3) | 29.3 (24.7) |
| encoded rows, compressed H2D, decoded on the GPU | 31.2 (26.2) | 21.6 (17.0) |
| 5C planes, nvCOMP zstd + device merge | – | 93.9 (73.8) |
| 5C planes, CPU zstd + merge, then BF16 H2D | – | 400 |

**End to end** (final warm profiles, same engine, same 12 GB host cache): **610.8 against 787.2 ms per decode token
(0.776)**; drive bytes 0.758 → 0.241 GB per token (the cache holds 1.5 times as many experts), bytes to the GPU 2.70 →
1.82 GB, copy time 436 → 292 ms, the decoder 17.5 ms of host time and about 33 ms of GPU time per token; prefills 3.9 s
against 5.6 s. Without a host cache, 862.4 against 1,122.3 ms; with 4 GB, 670.0 against 903.8.

## 7. Stage 6B3-A: a device tier for encoded rows

**Replay first** (`experiments/phase6b/gpucache/replay.py`, Phase 4B's and 5A's routing): a device LRU of BF16 rows serves
nothing below 3 GB (a decode token touches 156 experts, 2.70 GB); of encoded rows, nothing below 1.80 GB, then 0.368–0.385
of decode bytes from 1.83 GB on (prefill admission frozen; 0.40–0.41 on 5A's trace), a plateau to 2 GB. Below the cliff,
a hotness policy keeps 0.16–0.25 (1–1.5 GB). The device's room: the 6 GB cap less the prefill's 3.56 GiB peak.

**A page cache of copies does not fit.** The 2 GB LRU as Phase 4B's `PageCache` (copies in their own CUDA memory pool)
ran out of memory under the cap: two row sizes in the allocator's 20 MiB segments strand blocks. With the cap raised to 7
GB (exploratory, `experiments/phase6b/stage3-cap7g`) it showed the tier's value: 506.6 against 619.0 ms per decode token
(LRU; hotness 532.0, its victim search 61 ms per token).

**Fixed slots** (`awpmi.storage.cache.SlotCache`): one allocation per stored row size, made when the cache is, carved into
slots; each size has its slots and its own LRU (a row is evicted only for a row of its size; with an expert's rows always
used together this is one LRU over all of them); `put` copies the staging row into its slot on the compute stream, so a
freed slot is overwritten only after every earlier use of it on that stream. Budget = memory held: 162 slots of each
size, 1.89 GB. Hits are decoded straight from their slots: nothing crosses the bus. Admission is frozen during prefills,
as the host cache's.

**Result** (final warm profiles, same 12 GB host cache): **493.7 against 610.8 ms per decode token (0.808)**; 0.38 of
decode bytes served on the device, copies 292 → 198 ms per token; peak device memory 5.38 GiB allocated, 5.49 GiB
reserved, within the 5.59 GiB cap. Accounting: the device tier's stored bytes (its hits) and the store's fetched stored
bytes add up to the requested rows' stored bytes; restored BF16 lives only in the compact call buffers.

## 8. Stage 6B3-B: native copy issuing (measured, not adopted)

The streamer issues about 150 copies per decode token, 143 µs each in host time (`experiments/phase6b/copyissue`): 92 µs
without the per-copy timing events, 31 µs through the CUDA runtime directly (ctypes `cudaMemcpyAsync`), the same 6.2 GB/s
either way. End to end, on the best configuration of the time: 502.8 / 502.5 ms (PyTorch), 496.5 / 495.8 ms (the runtime):
−1.3%, with the issuing host time down from 18 to 7 ms per token; together with § 9's fill change, 472.6 against 474.9: no
longer measurable. The copy engine paces the transfer; a C++ or CUDA component would not change that. Not built (the
brief's stop condition: the isolated benefit disappears end to end).

## 9. Stage 6B4: CUDA Graphs, and PyTorch's NaN fills

**Isolated first** (`experiments/phase6b/graphs/static.py`): per layer, the pieces of a decode step whose shapes never
change and that touch neither the KV cache nor the routed experts: (pre) input RMSNorm, q_proj, kv_a_proj_with_mqa,
kv_a_layernorm, the RoPE of the position, the query's concatenation; (post) o_proj, the residual add, the post-attention
RMSNorm, the router and the shared experts (layer 0: the dense MLP). Recomputed from a real decode step's inputs with the
model's own modules they equal the step's values bit for bit, and so do their CUDA Graph replays. Host time to issue them
for one token: 59.9 ms eagerly, 4.1 ms as graphs; 2,146 launches against 245; 68 MB of graph memory.

**The fills.** Tracing the 270 BF16 fill kernels per token (`experiments/phase6b/graphs/fills.py`) found their source in
PyTorch, not in Weightsift or transformers: with deterministic algorithms on, `torch.utils.deterministic.
fill_uninitialized_memory` (default True) fills every `torch.empty` with NaN, a debugging aid that makes reads of
uninitialized memory visible. It fills the compact expert buffers (104 MB per layer) and the staging rows before the
copies into them. It changes no arithmetic: a computation that read such memory would differ between the two settings,
and the digests would show it. Turned off per configuration (`fill_uninitialized_memory: false`): −6.2% alone.

**Integrated** (`awpmi.models.decode_graphs.DecodeGraphs`, `decode_graphs: true`): two graphs per DeepSeek-V3 decoder
layer, replayed on every decode step but the first; between them, eagerly as transformers runs it, the KV cache's update,
the keys and values expanded from it and the attention, then the routed experts' call (streamed, compact), the MoE block's
sum and the residual add. The first decode step runs the pieces eagerly as functions and keeps their inputs in static
buffers; after its last layer all 54 pieces are captured back to back, on one side stream, into one pool. (Captured one at
a time between the step's eager work, each graph kept a 32 MiB cuBLAS workspace of its own: 1,025 MiB for 54 graphs,
against 65 MiB, `experiments/phase6b/graphs/memory.json`.) A step runs eagerly unless it decodes one token of one sequence
after the layer's cache holds a position. Observers see what eager calls show them: the forward hooks of the attention,
the router, the shared experts, the MoE block and the dense MLP are called with the replayed outputs in eager order;
whatever a replay would skip (a pre-hook on those modules, any hook inside a graphed piece, a hook that returns a value, a
global hook) makes the step raise. The code between the graphs repeats transformers 5.18's DeepseekV3Attention, MoE and
decoder-layer forwards, and installation refuses any other source (sha256 of each method).

**Result** (final warm profiles): graphs alone 431.9 ms (−12.5% from 493.7), the fills off alone 462.9 (−6.2%), both
**408.9 ms (0.828)**. Per decode token: 2,344 kernel launches and 54 graph launches instead of 5,195 launches (4,161
kernels run), 44 ms of launch CPU instead of 84. Capture: 54 graphs, 65–68 MB, 200–233 ms once per process. Every graphed
step of the final runs equals the reference (§ 10).

## 10. Correctness

Sources: `experiments/phase6b/run1/summary.md` and `run2` (each run's `records.jsonl.gz`, `ranges.jsonl.gz`,
`index.json`, `stream_stage.json`; the reference files are Phase 6A's `baseline-run1`, sha256 in `environment.json`).

| Criterion | Result (per run) |
| --- | --- |
| Source files, index, encoded pack | the checkpoint's 27 files re-hashed (the Hub's sha256); 3,328 index rows equal the reference's digests; the encoded pack's files against its manifest, and all 3,328 rows decoded on the GPU equal the reference's digests and the pack's records |
| Steps equal to the reference | 1,116 of 1,116: `python-stream` 144, `native-stream` 144, `native-host-12g-freeze` 144, `encoded-stream` 144, `encoded-chunk-1` 36 (one expert per chunk: 936 chunked calls), `encoded-host-tiny` 36 (a host cache smaller than a transfer), `encoded-host-12g-freeze` 144, `encoded-dev-tiny` 36 (4 slots per size: 18,982 evictions, slots reused within a call), `encoded-host-12g-freeze-dev1.9g` 144, `-fast` 144 (graphs, fills off) |
| …in each recorded quantity | Phase 4B's: token, logits, KV cache, routed experts; per layer attention, router logits, scores, indices, weights, the experts call's inputs and output, shared experts' output, MoE block output; the dense MLP; per-(token, expert) outputs where checked |
| Audits | 0 problems on every step: Phase 4B's and 6A's (requested = served experts; cache + storage = requested; storage = fetched; device = storage; physical bytes exactly the rows' 4 KiB blocks; OS counters = the store's; the host cache's lookups, copies and budget), and for encoded rows: decoded bytes = requested, the device tier's and the store's stored bytes = the requested rows' stored bytes |
| I/O parity | `native-stream`'s records, raw physical-I/O ranges and OS counters equal `python-stream`'s on all 144 steps |
| Caches against a replay | on every step, the device tier's hits and misses (LRU per stored row size, the run's slots) and the host cache's (LRU over its rows' bytes, behind the device tier) equal a replay of the reference's routing in the runtime's order, admission frozen on prefills where configured: `native-host-12g-freeze`, `encoded-host-12g-freeze`, `encoded-dev-tiny`, `-dev1.9g`, `-dev1.9g-fast`; every budget held |
| Reproducible | `run1` (`PYTHONHASHSEED=1`) and `run2` (`=2`): identical index, reference-row, prompt, reference, record and range digests, on the same source tree (`01465c71…`) and native tree (`759df48b…`) |

`benchmarks/gpu_report.py` was fixed after the runs (its replay skipped the first prefill's admission freeze: an empty
`PageCache` is falsy). It is the only file that differs from the runs' recorded source tree: the tree with the version
the runs hashed reproduces `01465c71…`, and Phase 6A's replay (`native_report.replay_native`) agrees with the corrected
one and with the measured counters.

**Tests** (all pass): the root suite's 883 tests (87 skipped: CPU-only or CUDA-only variants), among them the encoded
pack and decoder (11), the slot cache (3), decode graphs (7 on CUDA: four model variants, observers, streamed experts,
refused hooks), the MoE parity tests through encoded rows (no cache, a host cache, a device tier), the native store (39); the
research suite's 79 (Phase 5C's exact codecs); the Rust core's 23 unit and 15 integration tests; `cargo fmt --check` and
`cargo clippy -D warnings` clean.

**Guards confirmed to fail when broken:** a slot returned wrongly to the free list (the slot cache's content test), stale
static inputs in the graphs (the bitwise test), the busy rule removed (the concurrent-jobs test hangs again), the block
pool's trim and reuse and an aborted fill's reservation (the cache tests).

## 11. End-to-end performance

`benchmarks/moonlight_profile.py`, one configuration per process, nothing else running, Phase 4B's prompts 7, 0, 3 and 5
with 8 greedy decode steps (32 decode steps per profile); warm = the prompts 8–15 first. Profiles in
`experiments/phase6b/run1`; repeats with `PYTHONHASHSEED=2` (`-seed2`).

| Metric | Phase 6A baseline | Phase 6B (`encoded-host-12g-freeze-dev1.9g-fast`, warm) |
| --- | --- | --- |
| Decode latency | 807 ms/token | **408.9 ms/token** (408.7 and 409.0 in two runs; median 393–395) |
| Tokens/s | 1.24 | **2.45** |
| SSD bytes/token | 0.758 GB (warm, 12 GB cache) | 0.270 GB |
| H2D bytes/token | 2.70 GB | 1.24 GB (stored bytes; 0.38 of decode bytes served on the device) |
| GPU copy time | ~435 ms/token | 198.4 ms/token |
| Decode/reconstruction time | — | 15.2 ms host (batched launches), ~33 ms of GPU decode kernels per token |
| Cache submit overhead | ~100 ms/token | 0.8 ms/token (the engine's) |
| Kernel launches | 5,195/token | 2,344 kernel + 54 graph launches/token (4,161 kernels run) |
| Peak RAM and VRAM | 15.4 GB / 3.4 GB | 15.4 GB / 5.78 GB allocated, 5.90 GB reserved (the cap: 6.0 GB) |
| Prefill (mean of 4) | 5,584 ms | 3,593 ms |
| Reference mismatches | 0 | **0** |

Warm decode per configuration (ms per token; each run; the spread between seeds):

| Configuration | Decode | Runs | Copy ms | Drive GB | Host / device hits |
| --- | --- | --- | --- | --- | --- |
| `native-stream` | 1,122.3 | 1 | 438.9 | 2.700 | – |
| `native-host-4g` | 903.8 | 904.6, 903.0 | 446.6 | 1.823 | 0.325 |
| `native-host-12g-freeze` | 787.2 | 787.0, 787.4 | 435.9 | 0.758 | 0.720 |
| `encoded-stream` | 862.4 | 1 | 295.9 | 1.821 | – |
| `encoded-host-4g` | 670.0 | 1 | 298.3 | 0.947 | 0.481 |
| `encoded-host-12g-freeze` | 610.8 | 609.7, 612.0 | 291.6 | 0.241 | 0.869 |
| `encoded-host-12g-freeze-dev1.9g` | 493.7 | 494.6, 492.7 | 198.4 | 0.270 | 0.782 / 0.321 |
| `…-dev1.9g-nofill` | 462.9 | 1 | 198.4 | 0.270 | 0.782 / 0.321 |
| `…-dev1.9g-graphs` | 431.9 | 1 | 198.4 | 0.270 | 0.782 / 0.321 |
| **`…-dev1.9g-fast`** | **408.9** | 408.7, 409.0 | 198.4 | 0.270 | 0.782 / 0.321 |

Cold (caches empty at the first measured prompt): `native-host-12g-freeze` 797.1, `encoded-host-12g-freeze` 620.8,
`-dev1.9g` 520.7, `-fast` 436.6. Seeds 1 and 2 differ by 0.1–0.4% on every repeated configuration.

- **The gates**: 6B1 submit 1.34 ms (≤ 10) and 0.964 (≤ 0.99); 6B2 0.776 (≤ 0.90); 6B3-A 0.808 (≤ 0.95), peak reserved
  5.80 GB (≤ 6.0); 6B4 with the fills 0.828 (≤ 0.97); the phase **0.507** (≤ 0.90; the aspirational 0.80 met).
- **Where a decode token's 409 ms go now** (regions of the main thread, warm): waiting for pieces (the host cache's
  copies and the drive's reads) 124 ms, assembly including waits for slots still being copied 55 ms, copy issuing 17 ms,
  decoding launches 15 ms, the routing sync 13 ms, the device tier's admissions 11 ms, planning 2 ms, everything else
  (the attention over the cache, the experts' compute and their eager launches, the LM head, the GPU work the step waits
  for) 172 ms. The copies (198 ms) overlap most of it; the GPU is idle about 35% of a traced step.
- **Stage by stage** (warm, same tree): 808.9 (Phase 6A tree) → 787.2 (6B1) → 610.8 (encoded rows) → 493.7 (device tier)
  → 408.9 (graphs and fills): each change measured alone against the configuration before it, each kept because it paid
  end to end.

## 12. Phase 6C preparation (documentation; nothing changed)

What prevents a clean, explicit BF16 reference, as observed while profiling (Phase 6A's § 11 holds and is not repeated):

1. **Kernel selection by shape.** cuBLAS chooses GEMV for one row and tensor-core GEMMs for more (6A); `grouped_mm`
   chooses per expert row count; the router's float32 GEMM and the LM head likewise. CUDA Graph capture kept every choice
   (replays are bit for bit), because capture records the kernels the eager calls launch at the same shapes; a capture at
   another batch size would record other kernels. The reference therefore remains "this code at these shapes".
2. **Workspace-dependent choices.** cuBLAS's algorithm can depend on the workspace it is given; `CUBLAS_WORKSPACE_CONFIG`
   fixes its size (`:4096:8`) and capture allocates the same size inside the graph pool. Changing it (to save memory, as
   the per-capture workspaces tempted) could change kernels, so it was not touched.
3. **Accumulation and reduction order.** RMSNorm's float32 mean, the router's float32 GEMM, `grouped_mm`'s per-group
   accumulation and the combine's float32 sum over 6 experts (transformers' reshape + sum) run in orders the kernels
   choose; none is documented, all are repeatable here, none changed.
4. **Where PyTorch changes execution, not arithmetic.** Deterministic mode's NaN fill of uninitialized memory (§ 9) adds
   kernels but never changes a value that is read; PyTorch's caching allocator, streams and graph pools change addresses
   and timing only; TF32 and reduced-precision reductions stay off (`configure_reproducible_numerics`), and
   `allow_bf16_reduced_precision_reduction_split_k` is True but inert while its parent flag is False.
5. **Lossless steps.** nvCOMP's rANS restores the stored bytes exactly (every row audited); the slot cache and graphs move
   or replay bytes and kernels, never round.
6. **Opportunities for an explicitly reproducible native reference** (6C's to decide): batch-invariant GEMMs (one
   accumulation order for any row count, which batching, speculative verification and a fused decoder need), a fixed
   pairwise order for the norms' and combine's reductions, owned elementwise kernels with round to nearest even as their
   contract (RMSNorm's cast and product, SiLU, RoPE, the residual adds), and, given those, a graph of owned kernels as the
   decode step whose numerics are documented rather than observed.

Nothing of 6C was started: no kernel was written, no rounding claimed, the reference unchanged.

## 13. Answers

1. **Did fewer bytes to the GPU pay?** Yes, by a codec that needs no reconstruction: rANS over raw BF16 rows restores a
   call in 0.7 ms while saving a third of its copy; Phase 5C's bit planes, exact on the GPU too, cost more to merge than
   they save. Encoded rows: −22% per decode token at the same host budget, and the host cache holds 1.5 times as many
   experts.
2. **Does a device tier pay?** Only from one token's working set (an LRU cliff at 1.82 GB of encoded rows), and only in
   fixed slots under this GPU's cap: −19% at the same host budget.
3. **Was the cache's submit overhead what Phase 6A thought?** Partly: the frees were a third of it; waking every reader
   for every task was the rest. Both are gone (100 → 1 ms per token), and a latent deadlock with them.
4. **Did fewer launches pay?** Yes, once the step was found to be host-bound: graphs of the static pieces −12.5%, and the
   NaN fills PyTorch adds in deterministic mode −6.2%; together −17%. Native copy issuing did not: the copy engine, not
   the host, paces the transfers.
5. **Is it exact?** Yes: every step of every configuration, including encoded rows, device-tier hits, misses and
   evictions, graph replays and the fills off, equals the independent reference in every digest, in two runs with
   identical digests.
6. **What bounds a token now?** The bytes still copied (1.24 GB, 198 ms), the host cache's copies and the drive behind
   them (124 ms of waits), and the eager remainder of the transformer (attention over the cache, the experts' launches,
   about 170 ms of the main thread).

## 14. Recommendation

The next phase is the user's decision; nothing of 6C was started. By measured return:

1. **Phase 6C, the reference's semantics**, which the fastest remaining options need: batch-invariant kernels would allow
   batching several sequences and graphs over the experts and the attention (§ 12).
2. **More of the transformer in graphs**: the attention over a growing cache (a static, bucketed cache would change its
   kernels and so the reference today: 6C first), and the experts' compute after their bytes arrive.
3. **A larger device tier per byte**: a codec with a better ratio decoded as fast (rANS leaves 0.67), or a hotness-aware
   slot policy below the LRU cliff for GPUs with less room.

## 15. Limitations

- One machine (Windows 11, RTX 4060 Ti 8 GB on PCIe 3.0 x8, Samsung 990 PRO, 32 GB RAM): the copy floor, the cliff's
  position relative to the cap, and the host-bound step are this machine's. Linux was not run.
- The profiles are 4 prompts × 8 decode steps, one process each; repeats with another seed agree within 0.4%; the
  single-run variants (nofill, graphs alone, 4 GB, streams) have no repeat.
- nvCOMP is proprietary and CUDA-only: encoded packs need an NVIDIA GPU and the `gpu` dependency group; the BF16 paths
  need neither. Its rANS does not detect corruption; integrity rests on the pack's hashes and the audits.
- `DecodeGraphs` is tied to transformers 5.18's DeepSeek-V3 code (it refuses other sources) and to one sequence; other
  models need their own pieces.
- The device tier's 1.89 GB fits next to Phase 4B's 256 MiB call budget and the 1,024-token prefill with little room:
  the correctness runs' recorder had to stop keeping unchecked chunked calls (§ 16), and the fast configuration peaks at
  5.49 of the cap's 5.59 GiB reserved.
- The traced profiles' launch counts come from PyTorch's profiler (kernel launches and graph launches counted apart).

## 16. Reproduction

Raw results are in `experiments/phase6b/` (packs and model files stay out of Git; the encoded pack is rebuilt by
`weightsift pack encoded-experts`). Two benchmark fixes were made during the phase and are in the runs: the runtime's
recorder keeps a chunked call only when the step checks it, and every configuration's memory is collected before the
next one starts (both needed for the device tier under the cap; neither changes a record). Nothing else may run on the
machine during the timed stages (one GPU process at a time).

```bash
python -m uv sync                                     # native core (Rust) and nvCOMP (the default groups native, gpu)
cd native && cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace && cd ..
python -m uv run python -m pytest && python -m uv run python -m pytest research/expert_deltas
python -m uv run weightsift pack encoded-experts       # the encoded pack (configs/phase6b-gpu.yaml), about 3 minutes
# Exploration: the codec probe, the I/O per call, the device tier's replay
python -m uv run python benchmarks/gpu_codec_probe.py --output experiments/phase6b/codec/probe.json
python -m uv run python benchmarks/encoded_io.py --output experiments/phase6b/io/encoded-io.json
python -m uv run python experiments/phase6b/gpucache/replay.py > experiments/phase6b/gpucache/replay.txt
# Correctness: two runs with different hash seeds against Phase 6A's baseline reference
for seed in 1 2; do
  PYTHONHASHSEED=$seed python -m uv run python benchmarks/moonlight_runtime.py --config configs/phase6b-gpu.yaml \
      --output experiments/phase6b/run$seed --reference-from experiments/phase6a/baseline-run1
done
# Performance: one configuration per process (warm, cold, traced, seed 2); the list is in the report's § 11
PYTHONHASHSEED=1 python -m uv run python benchmarks/moonlight_profile.py --run experiments/phase6b/run1 --config configs/phase6b-gpu.yaml \
    --configuration encoded-host-12g-freeze-dev1.9g-fast --prompts 7 0 3 5 --warm \
    --output experiments/phase6b/run1/profile-encoded-host-12g-freeze-dev1.9g-fast-warm.json
python -m uv run python benchmarks/gpu_report.py experiments/phase6b/run1 --compare experiments/phase6b/run2
```
