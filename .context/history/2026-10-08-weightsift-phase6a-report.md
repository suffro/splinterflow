# Weightsift Phase 6A report — a native (Rust) I/O core and a host-RAM expert cache

Date: 2026-10-08 · Follows: `history/2026-10-08-awpmi-phase5c-report.md` (Phase 5C) and
`history/2026-10-03-awpmi-phase4b-report.md` (the runtime it extends) ·
Decision: `decisions/0012-phase6a-native-runtime-foundation.md` ·
Status: **complete. Correctness, I/O parity, host-cache and performance gates pass: every step of every configuration
equals the independent reference in two runs; the best native configuration decodes a token in 0.697 of the best Python
configuration's time (807 against 1,157 ms) and 0.654 of Phase 4B's streaming.**

## Outcome in brief

Phase 6A moves Weightsift's storage path (planning, reads, the host-RAM expert tier, transfer scheduling) into a small,
model-agnostic Rust core behind the existing Python storage contract, and measures it end to end on Phase 4B's
Moonlight-16B-A3B benchmark. The PyTorch execution, and every number it computes, is unchanged.

| Question | Answer |
| --- | --- |
| Phase 4B reproduced before any Rust (stage A) | yes, bit for bit: `baseline-run1` equals Phase 4B's run1 in all six digests (index, reference rows, prompts, reference, records, raw I/O ranges); its profile 1,274 ms per decode token (Phase 4B: 1,273) |
| Where a decode token's time went | drive reads 66.5% (2.70 GB of routed experts), the transformer's Python and launches 22%, per-request transfer overhead 11% |
| Repeated expert bytes | 99.5–99.9% of a decode token's bytes were read before; an LRU host tier hits nothing below one token's working set (2.6 GB), then 0.39–0.43 of decode lookups at 4 GB and 0.69–0.76 at 12 GB (exact, from byte stack distances) |
| Reuse | patterns from ds4 (hits protected first, staged direct reads, cancellation) and colibri (a bounded read pool, one transfer per call); PyO3, maturin, `lru`, `libc`; no engine vendored |
| The native core | `native/`: plans equal to Python's, direct positioned reads on 8 threads, a host-RAM cache (strict budget, deterministic LRU, leases, load-once, admission freeze, memory recycling), transfer jobs into the streamer's pinned slots, exact prefetch, full accounting; no model knowledge, no CUDA, one `unsafe` type; the Python backend stays the fallback |
| Native I/O alone (stage B) | the same reads (bytes, read calls, OS counters equal), 13% faster per decode call to the GPU and 27% to host memory; planning 0.13 ms against 1.8 ms per call; 16 threads or 4 MiB reads do not help |
| The host cache (stage C) | on every step of every host-cache configuration that holds a transfer, measured hits and misses equal an LRU replay of the reference's routing; budget never exceeded; drive bytes per decode token 2.70 → 1.67 (4 GB) → 0.81 GB (12 GB) |
| Prefetch (stage D) | exact chunk prefetch is correct (36,318 rows prefetched, all used) but makes prefills slower: off by default; speculative prefetch not built (10–21% of misses covered at 25–32% precision, on the traces) |
| Correctness | every step of 8 configurations equal to the independent reference in every Phase 4B digest, in two runs with different hash seeds and identical digests; I/O parity of the native and Python backends; every audit clean |
| Performance (gate: ≤ 0.95 of the best Python decode) | **PASS**: 807 ms per decode token (12 GB host cache, prefill admission frozen) against 1,157 ms (Phase 4B's device cache): **0.697**; against Phase 4B's streaming 1,233 ms: 0.654 (the aspirational 0.80 met); 1.24 tokens/s against 0.81; prefills 37% faster |
| What bounds a decode step now | the copies to the GPU: 2.70 GB per token over PCIe 3.0 x8, 435–445 ms whatever tier serves the bytes; then the transformer's Python and 5,195 kernel launches (≈260 ms) and the cache's submit cost (≈100 ms, a known fix) |
| §7, the BF16 reference (for 6C) | BF16 roundings are round to nearest even and correctly rounded (probed); float32 transcendentals within 2–3 ulps; a row's GEMM result depends on its batch (cuBLAS GEMV for one row, tensor-core kernels for more): `BF16_REFERENCE` is "this code at these shapes"; candidates for explicit native semantics listed, nothing claimed |
| §8, Phase 6B | the integration point is the transfer job's byte operations and the streamer's single issuing loop; nothing built |
| A defect the gates caught | the first full run's working set grew with each configuration: a closed store's cache outlived it; fixed (close releases the memory, a reference cycle removed, two tests), run repeated |

## 1. Questions and plan

The user's brief (2026-10-08) starts moving Weightsift's performance-critical runtime infrastructure from Python to
Rust (I/O, caching, prefetching, scheduling) while the Python research layer and the PyTorch/CUDA execution stay. It
asks for a small, model-agnostic core behind the existing storage contract, built in stages that each had to be
measured before the next:

- **A — baseline and profiling** before any Rust: reproduce Phase 4B's Moonlight benchmark; where a decode token's time
  goes (I/O, cache, scheduling, transfer, Python); how many expert bytes are read again.
- **B — native I/O**, benchmarked alone before inference: the same bytes as the Python backend, through the existing
  index.
- **C — a native host-RAM cache**: a strict budget, a deterministic LRU baseline, thread safety, in-flight
  deduplication, metrics, several budgets measured on Phase 4B's routing.
- **D — conservative exact prefetch** (speculative prefetch only as a separate, counted evaluation) and an explicit
  `python` / `native` backend choice that changes no arithmetic.

Gates (`configs/phase6a-native.yaml`, fixed before the full runs): correctness on every step of every configuration
(Phase 4B's digests, plus audits of the cache); I/O parity of the native backend without a cache against the Python
one; the measured host-cache hits equal to an LRU replay of the reference's routing; and a measurable end-to-end
benefit: the best native configuration's warm decode step at most 0.95 of the best Python configuration's (the brief's
20% is reported as aspirational, not as a gate). §7 (the BF16 reference's kernels, for Phase 6C) and §8 (an
integration point for Phase 6B's CUDA decoder) are documentation; neither builds kernels.

## 2. Setup

| Item | Value |
| --- | --- |
| Machine | RTX 4060 Ti 8 GB (sm_89, WDDM), i7-8700K (6 cores), 32 GB RAM, Windows 11 |
| Drive | Samsung 990 PRO 1 TB on PCIe 3.0 x4 (about 3.2–3.5 GB/s with direct I/O) |
| Software | Python 3.13.3, torch 2.14.1+cu130, transformers 5.18.0, safetensors 0.8.0; Rust 1.95.0 (pinned), PyO3 0.29, maturin 1.15, `lru` 0.18 |
| Model | `moonshotai/Moonlight-16B-A3B` @ `476b36a4`, BF16 (Phase 4B's): 26 MoE layers × 64 routed experts (6 per token) + 2 shared; 28.8 GB of routed experts, an expert 16.5 MiB (down, gate, up adjacent) |
| Profile | `BF16_REFERENCE`, `grouped_mm` experts, SDPA attention; `configure_reproducible_numerics` (deterministic algorithms, no TF32, no reduced-precision reductions) |
| Index, prompts | Phase 4B's: `packs/moonlight-16b-a3b-expert-index` (52 composed segments, bytes in place); 16 wikitext-2 prompts of 16–1,024 tokens, 8 greedy decode steps |
| Streamed process | Phase 4B's: GPU cap 6.0 GB, call budget 256 MiB (15 experts), direct I/O, 8 reader threads, 1 MiB read calls, 32 MiB pinned slots (2 for the Python streamer, 4 for native transfers) |
| Reference | Phase 4B's streaming reference, computed once in its own process (`baseline-run1`) and shared by the native runs |

## 3. Open-source reuse

Investigated before any code (licenses and activity checked on 2026-10-08; decision 0012, point 2, has the details).

| Project | License, state | What it does that matters here | Reused | Not reused, and why |
| --- | --- | --- | --- | --- |
| antirez/ds4 | MIT, active | C/CUDA/Metal engine; CUDA SSD streaming of experts into a device-side cache with LRU stamps; protects every hit of a request before choosing victims; look-ahead loads published at the lowest recency; `O_DIRECT` reads into a pinned staging ring with events; a cancellable reader thread joined on shutdown | the patterns: hits protected first (two-pass lookup), staged direct reads, cancellation and join | the code: a monolithic engine tied to its own format, graph and kernels |
| JustVugg/colibri (`jenovauh/colibri-LLM` is a fork) | Apache-2.0, active | C engine streaming experts with one read per expert (its matrices adjacent, as Moonlight's are), an 8-worker read pool with per-slot readiness, batch-union, a per-layer LRU with a learned pinned set, router look-ahead | a bounded worker pool; one transfer for all of an experts call's rows | router look-ahead (model-specific routing in the core, which the brief excludes); the engine |
| ggml-org/llama.cpp | MIT, active | experts resident, offloaded to the CPU, or mapped and paged by the OS; its loader reads with direct I/O through four pinned staging buffers with events; no runtime SSD expert cache (a two-tier expert cache request, issue #20757, was closed) | nothing new | `mmap` (decision 0006: reads that cannot be planned or counted) |
| PyO3 0.29, maturin 1.15 | Apache-2.0/MIT, active | Rust ↔ Python bindings; PEP 517 builds | both (abi3 module built by uv) | – |
| `lru` 0.18 | MIT, active | an O(1) LRU order | the recency order of the host cache | – |
| `libc` 0.2 | MIT/Apache-2.0, active | `O_DIRECT` on Linux | yes (Linux only) | – |
| `moka`, `quick_cache`, `foyer` | MIT/Apache-2.0, active | concurrent and hybrid caches | – | approximate (TinyLFU, CLOCK-Pro) or asynchronous eviction; the brief asks for a deterministic LRU with an exact budget, leases and load-once semantics |
| `tokio`, `rayon` | MIT, active | async runtime; data parallelism | – | tokio's file I/O is a blocking pool anyway; rayon is not for blocking I/O |
| `io-uring`, Windows `IoRing` | MIT/Apache-2.0; OS API | asynchronous submission | – | platform-specific; decision 0006 measured that threads of positioned reads already reach the drive's rate (stage B confirms it, § 6) |

Written in Weightsift (a few thousand lines of Rust): the planner (it must equal Python's `plan_reads` and
`PageStreamer._pieces`, byte for byte), transfer jobs into caller-owned staging slots, the cache's budget, leases and
load-once semantics, and the byte accounting the audits need.

## 4. Stage A: where the time goes, before any Rust

**Phase 4B, reproduced.** `experiments/phase6a/baseline-run1` is Phase 4B's benchmark (`configs/phase4b-moonlight.yaml`,
`PYTHONHASHSEED=1`) rerun on the Phase 6A source tree, Python backend: every digest equals Phase 4B run1's (index,
reference rows, prompts, reference, records and raw I/O ranges), so Phase 4B is reproduced bit for bit, every streamed
step's records included. Its reference stage is the one every native run is compared with (`--reference-from`: the
files copied, their sha256 recorded in each run's environment).

**Phase 4B's profile** (its report, § 13) already split a decode step without a cache (1,273 ms): drive reads 847 ms
(66.5%; 2.70 GB of routed experts per token at 3.19 GB/s, 95% of the drive's direct-I/O rate), the transformer's Python
and kernel launches 283 ms (22.2%), and the per-request transfer overhead 142 ms (11.2%: planning 43, issuing 312 device
copies 46, assembly 37, the routing sync 16; 52 requests, 130 pieces and 2,808 read calls per step). GEMMs are about 2%
of a step; the device copies run while the drive reads. The Phase 6A profiles (§ 10) measure the same split again,
with the main thread's CPU time and the GPU's idle time added.

**Repeated bytes** (`benchmarks/native_trace.py`, `experiments/phase6a/trace`: the routing of Phase 4B's 16 prompts and
of Phase 5A's 48, replayed in the order the streamed model requests rows; byte-weighted LRU stack distances from a
Fenwick tree give the exact LRU hit rate at every capacity, checked against a `PageCache` replay on every step):

- a decode token requests 2.70 GB of routed experts, and 99.5–99.9% of those bytes were requested before: 46–48% by the
  previous step of the same layer, 50–52% earlier in the same prompt (mostly by its prefill), 1.5% by an earlier
  prompt;
- an LRU tier hits nothing below one token's working set (156 experts, 2.6 GB), then (steady state, decode lookups):

| LRU capacity | of all experts | Phase 4B prompts | Phase 5A prompts | drive GB per decode token (4B / 5A) |
| --- | --- | --- | --- | --- |
| 2 GB | 7% | 0 | 0 | 2.70 / 2.70 |
| 2.77 GB | 10% | 0.37 | 0.41 | 1.72 / 1.61 |
| 4 GB | 14% | 0.39 | 0.43 | 1.67 / 1.56 |
| 8 GB | 28% | 0.57 | 0.64 | 1.14 / 0.98 |
| 12 GB | 42% | 0.69 | 0.76 | 0.82 / 0.65 |
| 16 GB | 56% | 0.80 | 0.85 | 0.53 / 0.41 |
| 20 GB | 70% | 0.90 | 0.92 | 0.28 / 0.22 |

- a speculative prefetch of a layer's previous-step experts is right 25–32% of the time (precision) and would cover
  10–21% of a 2.8–16.6 GB cache's decode misses.

**What this decided.** The drive is two thirds of a decode step and nearly every byte it reads was read before, so the
native core is first a host-RAM expert tier with a native read path; the Python overhead it removes (planning, read
submission, gathering) comes with it. The trace sized the budgets measured end to end (4, 8 and 12 GB: the machine has
32 GB of RAM and the streamed process held 3.25 GB in Phase 4B), and showed that a speculative prefetch could not pay on
a drive already busy two thirds of the step, so it is evaluated on the traces only (§ 8). The brief's order (I/O, then
cache, then prefetch) stood: the cache is worth only as much as the read path that fills it.

## 5. What was built

Decision 0012 has the details; `native/README.md` the build, toolchain, platforms and fallback.

- **`native/`, a Cargo workspace** (Rust 1.95.0 pinned; 3,600 lines of Rust in the core, 650 of Rust tests, 530 in the
  binding):
  - `weightsift-io` (`native/core`): segments described by the caller (the checkpoint index's plain and composed
    segments: no new parser), read plans equal to Python's `plan_reads` and `PageStreamer._pieces`, positioned reads on
    a pool of threads (direct I/O: `FILE_FLAG_NO_BUFFERING` with one handle per thread and file on Windows, `O_DIRECT`
    on Linux, at most `max_read_bytes` per call), the host-RAM cache, transfer jobs, prefetch, statistics. No Python, no
    CUDA, no model knowledge: `tests/test_layering.py` scans the Rust sources for model, tensor and routing names.
  - `weightsift-native` (`native/python`): the PyO3 module `weightsift_native` (abi3, Python 3.11+), built by maturin
    through uv (a default dependency group `native`). Coarse calls only: one `Engine.submit` per experts call, one
    `Job.next` per slot-sized piece; every blocking call releases the GIL (tested: Python threads keep running during a
    native read). Errors arrive as Python exceptions (`OSError` with path, offset and length for I/O, `ValueError` and
    `IndexError` for bad requests), before any read for bad requests.
- **`awpmi.storage.native.NativePageStore`**: the `FileBackedPageStore` contract (the same `ReadPlan`, the same
  counters, the OS's read counters measured around each native call) plus the host cache, a transfer API and prefetch.
  `Pack.store(backend="python" | "native")` chooses; configurations say `backend:`.
- **The streamer and the experts call**: `ExpertStore.assemble` asks `MaterializationBackend.materialize_many` for all
  of a call's parameters at once, and `PageStreamer.fetch_many` submits one native transfer for them. Pieces are copied
  to the device on the copy stream as soon as they are ready and released once copied; the readers keep filling the
  other slots (four, `native_slots`). `StreamedExperts`, the compact and chunked calls, the combine and every kernel are
  unchanged.
- **The host cache** (`native/core/src/cache.rs`): rows of segments under a strict byte budget (every byte held counts,
  rows being loaded included); least recently used first (`lru`); leased rows (being copied) and rows being loaded never
  evicted; a row loaded once however many requests want it (`waits`), and read again by its waiters if its load fails
  or is cancelled; a transfer first looks up and leases all its rows, then admits its misses, so a miss never evicts a
  hit of the same transfer (ds4's rule; a Rust test fails if the two passes are merged); an evicted entry of the new
  row's size lends it its memory (`recycled`: a first write into fresh memory ran at about 3 GB/s against 10–17 GB/s
  into touched memory); an admission freeze (`set_admit(False)`: hits served, nothing admitted, DwarfStar's prefill
  fix); counters: lookups, hits, waits, misses, inserts, evictions, bypassed, aborted fills, recycled, prefetch fills,
  used and wasted, resident and peak bytes.
- **Prefetch** (stage D, `Engine::prefetch` → `NativePageStore.prefetch` → `ExpertStore.prefetch`): rows loaded into the
  host cache at the lowest priority (a background queue the readers serve only when no transfer task waits), through
  staging of its own, without leasing or counting lookups; a transfer that wants a row being prefetched waits for it.
  Used by `StreamedExperts(prefetch_chunks=True)`: a chunked call announces all its experts when it starts, so later
  chunks' rows load while earlier chunks are computed. A prefetched row counts as used when a transfer takes it,
  wasted if evicted first; prefetching never changes bytes delivered or results.
- **Accounting and audits**: every byte counted where Python counts it (requests, rows, logical and physical bytes,
  read calls, extents, 4 KiB blocks of the rows read, raw ranges, per segment), plus what the cache served and the
  drive's busy time. `moonlight_runtime.py`'s step audit gains the cache's identities (lookups = rows = hits + waits +
  misses; bytes copied from the cache = its hits' and waits' bytes; resident ≤ budget; no failed load; no prefetched row
  wasted).
- **Tests**: 16 Rust unit tests and 14 Rust integration tests (`native/core/tests/engine.rs`: every byte against the
  file with and without caches of several sizes, reads equal to the plan's extents, budgets under eviction, the
  hits-first rule, concurrent jobs loading each row once, cancellation and shutdown (a closed engine holds no cache
  memory), read errors, refusals, prefetch used, raced, cancelled and wasted); 38 Python tests of the store against the
  Python one (`tests/test_native_storage.py`: plans, pieces, reads and counters equal on plain and composed segments, no
  hidden reads, the streamer's bytes and transfer counters equal, the host cache under budgets 0 to everything,
  prefetch, freezes, threads sharing a cache, errors, the GIL released, cancellation, a closed store freed without the
  garbage collector); and Phase 4B's MoE parity tests (exact compact and chunked calls on seven
  architectures, CPU and CUDA) run through every backend: Python, native, native with a cache that evicts, and native
  with chunk prefetch.
- **The `unsafe` boundary**: one type, `RawBuffer` (`native/core/src/buffer.rs`), for writing into staging memory the
  caller owns (PyTorch's pinned buffers), so that reads land where the device copies from: the zero-copy property
  decision 0007 bought. Seven `unsafe` blocks use it (six in the engine, one in the binding), each stating the invariant
  it relies on (memory alive while any task may touch it; disjoint ranges per task; a slot handed over only after its
  writes and refilled only after its release).

## 6. Stage B: the native I/O core alone

`benchmarks/native_io.py` (`experiments/phase6a/io/io-cpu.json` and `io-cuda.json`, 100 requests per pattern and path,
on the final build): Moonlight's expert index on the drive, three request patterns (a decode experts call: one layer's 6
experts, both segments, 104 MB; a prefill chunk: 15 experts of one segment, 128 MB; a whole layer: 64 experts, 1.1 GB),
each served three ways (Phase 4B's Python store and streamer; the native store through the same streamer; the native
store with every row in its host cache), into host memory or onto the GPU (copies included). The bytes delivered are
checksummed against the Python path's (equal on every pattern), the OS's read counters against the store's (equal), and
the drive is read with direct I/O (the OS page cache plays no part); nothing else ran.

| Pattern | Delivered to | Python | Native | Native, all hits |
| --- | --- | --- | --- | --- |
| decode call (104 MB, 108 read calls) | host | 45.0 ms (2.30 GB/s) | 32.8 ms (3.17 GB/s) | 18.6 ms (5.60 GB/s) |
| | GPU | 41.4 ms (2.51 GB/s) | 36.0 ms (2.88 GB/s) | 23.9 ms (4.34 GB/s) |
| prefill chunk (128 MB, 133 read calls) | host | 53.1 ms (2.41 GB/s) | 39.2 ms (3.27 GB/s) | 22.8 ms (5.61 GB/s) |
| | GPU | 50.0 ms (2.56 GB/s) | 42.5 ms (3.01 GB/s) | 28.2 ms (4.55 GB/s) |
| whole layer (1.1 GB, 1,152 read calls) | host | 446.2 ms (2.48 GB/s) | 326.9 ms (3.39 GB/s) | 203.5 ms (5.44 GB/s) |
| | GPU | 389.6 ms (2.84 GB/s) | 338.6 ms (3.27 GB/s) | 187.6 ms (5.90 GB/s) |

- **The same reads, less time around them.** Native and Python issue exactly the same reads (physical bytes and read
  calls per request equal: the plans are equal) and the native readers keep the drive busy at 3.38–3.52 GB/s. A decode
  call is 13% faster to the GPU and 27% faster to host memory; the native path removes Python's planning (1.72–1.82 ms
  per decode request against 0.13 ms) and the per-piece Python work, and keeps four slots in flight instead of two.
- **The knobs**: 8 reader threads with 1 MiB read calls (Phase 4B's settings) deliver 2.88 GB/s on decode calls and 3.28
  GB/s on whole layers; 4 or 16 threads and 4 MiB calls do no better (2.35–2.87 and 3.07–3.30 GB/s). The settings stay;
  `io_uring` or Windows `IoRing` have nothing to gain at this drive's rate.
- **The hit path** (rows copied from the cache into the pinned slots on the core's threads, then to the device)
  delivers 4.3 GB/s on a decode call, where the first piece's host copy precedes its device copy, and 5.9 GB/s on a
  whole layer, close to the device link's rate (PCIe 3.0 x8 here, about 6.1 GB/s for pinned copies).
- **Main-thread CPU per decode request**: 10.2 ms (Python) and 7.7 ms (native) to the GPU, 14.5 and 9.8 ms to host
  memory (Windows' thread clock ticks in 15.6 ms: means over 100 requests; host delivery includes the main thread's
  own copies out of the slots).

## 7. Stage C: the host-RAM cache

Budgets from stage A's trace analysis, on Phase 4B's 16 prompts (the cache persists across a configuration's prompts and
starts empty; `experiments/phase6a/native-run1`, instrumented; per step, its hits and misses equal an LRU replay of the
reference's routing in the native order, on every step of every configuration whose budget holds a transfer):

| Configuration | Budget | Decode lookups served | Prefill lookups served | Drive GB per decode token | Drive GB per prefill | Read calls / extents per decode token | Evictions | Not admitted (bypassed) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `native-stream` (no cache) | – | – | – | 2.700 | 22.8 | 2,808 / 312 | – | – |
| `native-host-tiny` | 2 experts | 0 | 0 | 2.700 | 19.5 (4 prompts) | 2,808 / 312 | 5,495 | 13,492 |
| `native-host-4g` | 4 GB | 0.383 | 0.006 | 1.668 | 22.7 | 1,734 / 193 | 66,108 | 0 |
| `native-host-12g` | 12 GB | 0.700 | 0.131 | 0.814 | 19.8 | 846 / 94 | 47,239 | 0 |
| `native-host-12g-prefetch` | 12 GB | 0.700 | 1.000 (prefetched) | 0.814 | 19.6 | 846 / 94 | 46,904 | 0 |
| `native-host-12g-freeze` | 12 GB | 0.724 | 0.447 | 0.746 | 12.6 | 776 / 86 | 9,612 | 23,337 (prefill rows) |

- **Exact LRU, measured.** On every step of `native-host-4g`, `-12g`, `-12g-prefetch` and `-12g-freeze`, the cache's
  measured hits and misses equal a replay of the reference's routing through an LRU of the same budget in the backend's
  order (per transfer: every row looked up, then the misses admitted; with the freeze, nothing admitted on prefill steps;
  with prefetch, a chunked call's rows promoted or admitted when it starts). The decode hit rates match stage A's
  prediction (0.39 at 4 GB, 0.69 at 12 GB, for these prompts).
- **The budget holds**: resident and peak resident bytes never above the budget (rows being loaded counted), on every
  step. The process's working set stayed within the budget plus 5 GB: 7.4 GB at 4 GB, 15.4 GB at 12 GB (the base process 3.4 GB).
- **A tiny cache degrades to streaming**, never below it: with room for two experts, a decode call's rows cannot all
  be admitted (the transfer's own hits and loads are leased), so most are bypassed and read as without a cache; the
  bytes delivered and the physical reads are the same.
- **The admission freeze pays twice**: decode hits rise from 0.700 to 0.724 (a prefill no longer flushes the experts
  the decode steps keep using), and the prefills themselves read 36% fewer bytes (12.6 against 19.8 GB), because the
  experts the previous prompts' decode steps used stay cached for the next prefill.
- **Memory recycling**: 5–37 thousand evicted entries per configuration (about half of all evictions) lent their memory to the row that replaced
  them (no allocation, no page faults).
- **No load failed**: no aborted fill, no fallback read, in any configuration.

**A defect the gate caught.** The first `native-run1` failed the host-cache gate on memory, not on hits: with a 12 GB
budget the process's working set reached 19.4–24.9 GB. Each configuration's store, and its cache, outlived the
configuration: closing a store stopped its threads but left the cache's memory to the engine's destructor, and a
reference cycle (the store and its statistics object) kept the store alive until a full garbage collection, which
Python seldom runs. `Engine::close` now releases the cache at once, the cycle is gone, and two tests (Rust: a closed
engine's cache is empty; Python: a closed store holds nothing and is freed with the collector disabled) fail without
the fix. The run was repeated; hits, misses, bytes and every digest are the same, only the working set changed.

## 8. Stage D: prefetch and the backend choice

**Exact prefetch.** The only rows Weightsift knows for certain before it needs them are a chunked call's later chunks:
once the router has chosen, a prefill's experts call that exceeds the 256 MiB budget runs in chunks of 15 experts, and
every chunk's experts are known when the call starts. With `prefetch_chunks`, the call announces all of them
(`ExpertStore.prefetch`), and the native host cache loads them in the background (lowest priority, staging of its own)
while the first chunks are read and computed; each chunk's transfer then finds its rows cached or being loaded. Decode
calls (6 experts) are never chunked and already move all their rows in one transfer as soon as the router has chosen,
so there is nothing exact to prefetch there: the next layer's experts are not known until its router runs.

Measured (`native-host-12g-prefetch`): exact on every step; 36,318 rows prefetched over the 16 prompts, every one used
by its call, none wasted or aborted; a chunk's transfer finds all its rows cached or loading (prefill lookups served:
1.000, against 0.131 without prefetch). It does not pay: prefills take 9.3 s against 7.1 s without it (warm profiles).
The loads go through two staging slots of their own at the lowest priority and are copied twice in host memory
(staging to entry, then entry to slot), while the GEMMs they could overlap are about 1% of a prefill (Phase 4B). Decode
is unaffected (834 ms against 837–863 ms). The option stays, off by default; a device-side decoder of compressed pages
(Phase 6B) would give the chunks more work to overlap.

**Speculative prefetch** (evaluated on the traces only, as the brief asks; § 4): the best simple predictor, a layer's
previous-step experts, is right for 25–32% of what it would load and would cover 10–21% of an LRU host cache's decode
misses. Each wrong guess costs drive time on a drive that is busy most of the step, and the host tier already hits
most of what the predictor would guess right (the previous step's experts are the most recently used rows). It was not
built.

**The backend choice** is explicit and changes no arithmetic: `backend: python | native` per configuration
(`Pack.store(backend=…)`), the host cache's budget (`host_cache_bytes` or `host_cache_experts`), `freeze_prefill` and
`prefetch_chunks`. The router, the combine, the KV cache, the accumulation order, the logits and the tokens are
untouched (the correctness runs, § 9, compare every one of them with the reference).

## 9. Correctness

Sources: `experiments/phase6a/native-run1/summary.md` and `native-run2` (and `baseline-run1`). Raw records:
`records.jsonl.gz`, `ranges.jsonl.gz`, `index.json`; the reference files are the baseline's (sha256 in each run's
`environment.json`, `reference_from`).

| Criterion | Result (per run) |
| --- | --- |
| The Python baseline | `baseline-run1` (Phase 4B's configuration, Python backend, on the Phase 6A tree) equals Phase 4B's run1 in every digest: index, reference rows, prompts, reference, records and raw I/O ranges; Phase 4B's report passes its correctness and gates A–F (`experiments/phase6a/baseline-run1/summary.md`; its comparison flags only the source tree, which is meant to differ) |
| Source files, index | all 27 re-hashed with direct reads (the Hub's sha256); all 3,328 index rows equal the reference's digests |
| Streamed steps equal to the reference | 1,008 of 1,008 per run: `python-stream` 144, `native-stream` 144, `native-chunk-1` 36, `native-host-tiny` 36, `native-host-4g` 144, `-12g` 144, `-12g-prefetch` 144, `-12g-freeze` 144 |
| …in each recorded quantity | Phase 4B's: token, logits, KV cache, routed experts; per layer attention, router logits, scores, indices, weights, the experts call's inputs and output, shared experts' output, MoE block output; the dense MLP; per-(token, expert) outputs where checked |
| I/O parity | `native-stream`'s records equal `python-stream`'s in every field but the backend's own, on all 144 steps (requests, rows, logical and physical bytes, read calls, extents, 4 KiB blocks, device copies and their bytes, buffers); its raw physical-I/O ranges equal on the 27 recorded steps; the OS's read counters equal |
| Host cache against an LRU replay | measured hits (with waits) and misses equal an LRU replay of the reference's routing in the native order on every step of `native-host-4g`, `-12g`, `-12g-prefetch` and `-12g-freeze` (the tiny cache, smaller than one transfer, bypasses by design and is excluded) |
| Audits | 0 problems on every step: Phase 4B's (requested = served experts; cache + storage = requested; storage = fetched; device = storage; physical bytes exactly the 4 KiB blocks of the rows read, a prefetch's included; chunk buffers within the budget; OS counters = the store's) and the host cache's (lookups = rows = hits + waits + misses; bytes copied from the cache = its served bytes; resident and peak ≤ budget; no failed load or fallback read; no prefetched row wasted) |
| Poison | the first 3 prompts of `python-stream` and `native-stream` with NaN spare slots and NaN-filled chunks: equal |
| Reproducible | `native-run1` (`PYTHONHASHSEED=1`) and `native-run2` (`PYTHONHASHSEED=2`): identical index, reference-row, prompt, reference, record and range digests, on the same source tree and the same native tree (`environment.json`); every host-cache counter and replay identical |

**Tests** (all pass): the root suite's 753 Python tests (198 new: the native store's 38, two layering tests, and 158
runs of Phase 4B's MoE parity tests through the native backends: native, native with an evicting cache, native with
chunk prefetch, on CPU and CUDA), the research suite's 79, and 16 Rust unit tests and 14 Rust integration tests;
`cargo fmt --check` and `cargo clippy -D warnings` are clean.

**Guards confirmed to fail when broken:**

| Guard | What failed |
| --- | --- |
| A miss may never evict a hit of its own transfer | merging the lookup and admission passes (one pass, as a naive cache would) fails `a_miss_never_evicts_a_hit_of_its_own_job` |
| No model, tensor or routing names in the Rust core | the layering test failed on a doc comment that said "router" (rephrased) |
| Only `awpmi.storage.native` names the extension | the layering test failed on an environment key spelled like the module (renamed) |

## 10. End-to-end performance

`benchmarks/moonlight_profile.py`, one configuration per process, nothing else running (`experiments/phase6a/native-run1/
profile-*.json`; the commands are `native-run1/profiles.sh`): Phase 4B's prompts 7, 0, 3 and 5 (1,024, 16, 128 and 512
tokens) with 8 greedy decode steps each, un-instrumented (no digests); means over the 32 decode steps (the traced
profiles leave out their three traced steps). **Cold**: every cache starts empty. **Warm**: the prompts 8–15 run first in
the same process, as a serving process's caches would be. All expert reads are direct I/O in both backends, so the OS
page cache holds none of them, cold or warm; "cold" and "warm" are Weightsift's own caches.

**Warm** (the gate's comparison):

| Configuration | Decode ms/token (median, p95) | Tokens/s | Drive GB/token | Drive GB/s | Host-cache hits | I/O wait ms | Requests / read calls / extents per token | Peak RAM / VRAM (GB) | H2D device ms | Prefill ms (mean of 4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `python-stream` (Phase 4B) | 1,228.5 (1,228.3, 1,249.7); repeat 1,238.3 | 0.81 | 2.700 | 3.18 | – | 848 | 52 / 2,808 / 312 | 2.8 / 3.4 | 441 | 8,830 |
| `python-hotness-80` (device cache) | 1,157.4 (1,144.3, 1,275.8) | 0.86 | 2.335 | 3.18 | – | 734 | 52 / 2,428 / 270 | 2.8 / 4.9 | 379 | 8,947 |
| `native-stream` | 1,119.6 (1,118.9, 1,141.5) | 0.89 | 2.700 | 3.38 | – | 754 | 52 / 2,808 / 312 | 3.4 / 3.4 | 439 | 7,845 |
| `native-host-4g` | 944.5 (911.7, 1,110.3) | 1.06 | 1.823 | 3.31 | 0.325 | 549 | 52 / 1,896 / 210 | 7.4 / 3.4 | 445 | 8,218 |
| `native-host-8g` | 861.8 (813.4, 1,056.7) | 1.16 | 1.233 | 3.33 | 0.544 | 345 | 52 / 1,282 / 142 | 11.4 / 3.4 | 439 | 8,080 |
| `native-host-12g` | 863.0 (826.7, 1,006.5); repeat 837.3 | 1.16 | 0.825 | 3.29 | 0.696 | 221 | 52 / 858 / 95 | 16.5 / 3.4 | 438 | 7,077 |
| `native-host-12g-prefetch` | 834.4 (796.9, 980.3) | 1.20 | 0.847 | 3.30 | 0.688 | 232 | 52 / 881 / 97 | 15.4 / 3.4 | 436 | 9,269 |
| **`native-host-12g-freeze`** | **807.0** (797.1, 897.1) | **1.24** | 0.758 | 3.29 | 0.720 | 189 | 52 / 788 / 87 | 15.4 / 3.4 | 435 | **5,584** |

**Cold**: `python-stream` 1,274.0 ms (Phase 4B: 1,273), `python-hotness-80` 1,162.9, `native-stream` 1,156.6,
`native-host-4g` 939.7, `-8g` 870.0, `-12g` 866.3, `-12g-prefetch` 838.9, `-12g-freeze` 808.8 (decode hits 0.665: a frozen
prefill admits nothing, so a cold cache fills from decode steps only). Cold and warm differ little for the host caches:
the first measured prompt is a 1,024-token prefill that routes nearly every expert and fills the cache.

- **Gate: PASS.** The best native configuration (`native-host-12g-freeze`, 807.0 ms) is **0.697** of the best Python
  configuration (`python-hotness-80`, 1,157.4 ms; the gate asked ≤ 0.95), and **0.654** of Phase 4B's configuration
  (`python-stream`, 1,233.4 ms over its two warm runs): the brief's aspirational 20% is met with room, at 1.24 tokens per
  second against 0.81. Run-to-run noise (the repeats): 0.8% and 3%.
- **The native read path alone** (`native-stream`, the same reads, no cache) saves 109 ms per decode token (−8.9%)
  and 11% of a prefill: the I/O wait drops from 848 to 754 ms (the drive's effective rate 3.38 GB/s against 3.18), the
  planning from 40 ms to 3 ms (in Rust, 26 transfers per token instead of 52 Python requests), assembly from 35 to 16 ms.
- **The host cache** removes what stage A predicted: 1.82, 1.23 and 0.83 GB of drive reads per decode token at 4, 8 and
  12 GB (from 2.70), and decode time falls to 945, 862 and 863/837 ms.
- **Freezing admission during prefills pays twice**: decode hits rise (0.720 against 0.696), and prefills read 36% fewer
  bytes and take 37% less time than Phase 4B's (5,584 against 8,830 ms), because the experts the decode steps keep are
  also most of what the next prompt's prefill needs.
- **Exact chunk prefetch does not pay**: decode is unchanged (it never chunks) and prefill is slower (9,269 against
  7,077 ms). Its loads go through two staging slots of their own at the lowest priority and are copied twice in host
  memory (staging to entry, entry to slot), while the GEMMs it could overlap are about 1% of a prefill (Phase 4B). It
  stays an option, off by default.

**Where a decode token's time goes now** (`native-host-12g-freeze`, warm; regions of the main thread):

| Region (ms per decode token) | Python stream | Native, 12 GB, freeze |
| --- | --- | --- |
| Waiting for pieces: drive reads, or the cache's copies (`io`) | 848 | 189 |
| Assembly, including waits for slots still being copied to the device (`assemble`) | 35 | 217 |
| Planning, or submitting transfers (`plan`) | 40 | 98 |
| Issuing device copies (`h2d`) | 40 | 31 |
| Routing sync (`route`) | 12 | 11 |
| Everything else: Python and kernel launches of the transformer, GPU waits (`other`) | 252 | 261 |
| **Step** | **1,228** | **807** |

- **The copies to the device are now the floor.** Every configuration without a device cache copies the 2.70 GB of a
  token's routed experts to the GPU: 435–445 ms of device time per token over this machine's PCIe 3.0 x8 link (6.1–6.2
  GB/s), whatever tier the bytes come from. With 12 GB of host cache the main thread's waits (406 ms) are those copies.
  The traced decode steps show it from the GPU's side: 434 ms of copies, 47 ms of kernels (attention 11, experts' GEMMs
  13, shared experts 4, LM head 2.4, the rest small) and 49% idle, against 59% idle and 510–584 ms of copies for
  Python streaming. A decode step of this design cannot go below about 440 ms of copies plus the transformer's 250–290
  ms of Python and 5,195 kernel launches (88 ms of launch CPU) without moving fewer bytes to the device (a device-side
  cache, compressed experts decoded on the GPU) or fusing the transformer (CUDA Graphs): Phase 6B's subjects.
- **Submitting a transfer with a cache costs 56–105 ms per token** (3 ms without one), all of it on the critical path.
  A microbenchmark (`experiments/phase6a/io/submit.txt`) puts it on evictions whose memory cannot be recycled: a
  transfer's submit takes 0.2 ms when every eviction hands its buffer to the new row (rows of one size), and 1.2 ms
  (about 4 ms in the model's process) when the victim and the new row differ in size, as Moonlight's 11.5 MB gate/up
  and 5.8 MB down rows do half the time; the buffer is then freed and a new one allocated inline. Recycling across row
  sizes, or freeing and allocating on the readers, would take most of it off the critical path: the first follow-up,
  not done here (it changes the cache's memory handling, and the gate does not need it).
- **Main-thread CPU** (Windows' thread clock): 397 ms per decode token for Python streaming, 347 for native streaming,
  367–563 ms with host caches. It is not a clean measure of Python overhead here: waiting on a CUDA event spins, and
  the more a step waits for device copies instead of drive reads (which block on a condition variable), the more CPU
  the wait burns. The FFI boundary is coarse: about 300 crossings per decode token (26 submits, about 120 pieces and
  their releases), against 52 Python requests and 130 Python pieces before.
- **Memory**: peak host working set 3.4 GB plus the cache's budget (15.4 GB at 12 GB, within the gate's budget plus 5
  GB); device peak 3.40 GB in every native configuration (4.9 GB with the device cache), under the 6 GB cap; read
  amplification 1.0005 everywhere (the 4 KiB blocks of the rows read, nothing else).

## 11. The BF16 reference (§7, for Phase 6C; nothing changed)

**What defines `BF16_REFERENCE` on Moonlight.** transformers 5.18's DeepSeek-V3 code, run by PyTorch 2.14.1's CUDA
kernels under `configure_reproducible_numerics` (deterministic algorithms; no TF32; no reduced-precision reductions in
BF16 GEMMs; a fixed cuBLAS workspace). Per decoder layer, in order:

| Operation | Inputs → output | Arithmetic | Roundings |
| --- | --- | --- | --- |
| RMSNorm (input, post-attention, `kv_a_layernorm`, final) | BF16 → BF16 | to float32 (exact); squares, mean over 2,048 (512) values, + ε, `rsqrt`, product, all float32; cast to BF16; times the BF16 weight | float32 reduction and elementwise; **2 BF16** (the cast, the weight product) |
| Projections (`q_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`, shared experts, layer 0's dense MLP, LM head) | BF16 → BF16 | cuBLAS GEMM, FP32 accumulation in the kernel's order (split-K partials reduced in FP32), epilogue to BF16 | **1 BF16** (round to nearest even: decision 0004's probe) |
| RoPE (interleaved) | BF16 → BF16 | cos and sin computed in float32 and cast to BF16; `q1·cos − q2·sin`, `q2·cos + q1·sin` as BF16 tensor ops | 2 casts; **3 BF16** per output (two products, one sum) |
| Attention (SDPA; MLA: query and key 192 wide, value 128; causal prefill, one query at decode) | BF16 → BF16 | one fused kernel (PyTorch's memory-efficient CUTLASS attention): float32 scores and online softmax, float32 accumulation | **1 BF16** at the output (and the kernel's internal ones) |
| Router | BF16 → float32 | input and weight cast to float32 (exact); float32 GEMM (no TF32), sigmoid, + bias, top-6, gather, sum + 1e-20, division, × 2.446 | float32 only; top-6 ties broken by the kernel |
| Experts, gate and up (`grouped_mm`) | BF16 → BF16 | one GEMM per group (expert) of the call, FP32 accumulation | **1 BF16** |
| SiLU(gate) · up | BF16 → BF16 | SiLU in float32, cast; the product in float32, cast | **2 BF16** |
| Experts, down (`grouped_mm`) | BF16 → BF16 o | as gate and up | **1 BF16** |
| Routing weight | BF16 × float32 → float32 z | float32 product | 1 float32 |
| Combine | float32 [tokens, 6, 2,048] → BF16 R | sum over the 6 in float32, in the kernel's order; one cast | **1 BF16** |
| MoE output m = R + S (shared experts) | BF16 | float32 sum, cast | **1 BF16** |
| Residual adds (after attention and after the MLP) | BF16 | float32 sum, cast | **1 BF16** each |
| Greedy choice | BF16 logits | `argmax`: the first maximal index | ties possible (Phase 5A: 10 of 768 decode tokens had an exact BF16 tie) |

The KV cache holds the compressed latents (after `kv_a_layernorm`, and the rotated key part), and every step recomputes
the keys and values of all cached positions with `kv_b_proj` (a GEMM over all positions).

**Measured on this machine** (`benchmarks/reference_numerics.py`, `experiments/phase6a/bf16/numerics.json`: random
inputs with fixed seeds in normal ranges, Moonlight's shapes, the reference's numerics flags):

| Behaviour | Result |
| --- | --- |
| float32 → BF16 conversion | round to nearest even on 4.1 M random float32 patterns and on 4.2 M exact ties: 100% |
| BF16 add (with cancellation) and multiply | correctly rounded on 3 × 4.2 M pairs: 100% (computing in float32 and rounding once to BF16 is innocuous double rounding: 24 ≥ 2·8 + 2) |
| float32 `exp`, `sigmoid`, `rsqrt` (CUDA) | within 2, 3 and 2 ulps; correctly rounded on 70%, 61% and 77% of 4.2 M inputs (the certified model allows 32 ulps) |
| BF16 SiLU, on every finite BF16 input | the correctly rounded BF16 SiLU on 64,386 of 64,392; the other 6 (x from −91.5 to −89) return −0 because exp(−x) overflows in float32: the case the certified operator's documented absolute allowance (2⁻¹¹⁰) covers |
| RMSNorm's float32 mean of 2,048 squares | correctly rounded on 66% of rows, within 2 ulps |
| the experts' float32 combine of 6 | error within 3.2·u·Σ\|z\| (γ₅ allows about 5), 0.32 on average |
| one row through each of Moonlight's GEMMs alone (M = 1, a decode step) and inside batches of 2–1,024 rows | M = 1 runs cuBLAS's GEMV kernel, M ≥ 2 tensor-core kernels (CUTLASS WMMA at small M, `ampere_bf16_s1688gemm`/`s16816gemm` at large M). The row's result differs between M = 1 and M ≥ 2 in 0.04–0.7% of its elements for `q_proj`, `o_proj`, the shared experts, layer 0's MLP and the LM head, from M ≥ 32 for `kv_a_proj_with_mqa`, and never for `kv_b_proj` (K = 512). Every shape is repeatable run to run |
| `grouped_mm` (the routed experts) | a group's result never depends on the other groups (Phase 4B's chunking property, again); but a token's gate/up output depends on how many tokens its expert has in the call (one row: GEMV; more: tensor-core kernels; 0.07–0.25% of elements differ); the down projection did not differ |
| attention | PyTorch's memory-efficient CUTLASS kernel (`fmha_cutlassF_bf16_aligned_32x128_gmem_sm80`) for prefill and decode alike; the last query alone (a decode step) equals its row in a causal prefill at 16–1,024 positions |

So `BF16_REFERENCE` is "this code at these shapes". A decode step and the same position inside a prefill do not
compute the same projections bit for bit, and an expert's output for a token depends on how many tokens chose that
expert in the call. Weightsift's streamed runtime equals the reference because it runs the same shapes (Phase 4B, and
every native configuration here). Batching several sequences, verifying several draft tokens at once, or a fused decoder
with other tile shapes would change BF16 outputs unless its GEMMs were batch-invariant. The reference's per-step
recomputation of all cached keys and values through `kv_b_proj` is benign on this GPU (no shape dependence at K = 512),
and attention is the same at decode and prefill.

**Where the certificate's rounding uncertainty comes from** (decisions 0001, 0004, 0005, 0009). A certificate bounds the
reference's floating-point computation, not real arithmetic, and must hold whatever the kernels do within their
documented behaviour. Its certified model is *faithful* rounding (each BF16 rounding of an unknown value may go to either
neighbour: up to 2⁻⁷ of the value), an accumulator error of γ_{K+2}(2⁻²²) (4× binary32's unit roundoff, which covers
tensor-core accumulation whose internal order and precision are not documented), and up to 2⁻¹⁸ relative error per
float32 elementwise operation. Round to nearest even is an observation (decision 0004's probe, and the one above), not a
contract of cuBLAS epilogues or PyTorch's kernels, so it stays a labelled what-if (the user's decisions 0001 and 0005).
On Moonlight's last MoE layer with every routed byte read (Phase 5A), the tightest pair's named error terms averaged 3.18
logits (the final norm 1.37, y 0.49, m 0.36, o 0.35, R 0.28, the down accumulation 0.26, the LM head 0.07), against a
median top-2 gap of 1.13: the faithful certificate held on 8.1% of tokens, 16.5% with a binary32 accumulator, 17.8%
with round to nearest even in elementwise kernels, 24.7% with it everywhere, 100% in real arithmetic. The obstacle is the
reference's own roundings of activations after the experts, amplified by the final norm and the LM head
(Σ_k |W_jk|·|h_k|·2⁻⁷ per logit), not the bytes read. 8.1% is that observation, not a target.

**Operations that might gain explicit native semantics in Phase 6C** (candidates, each to be decided there; nothing is
claimed or built here):

1. **The elementwise BF16 roundings** (RMSNorm's cast and weight product, SiLU and its product, RoPE, m, the residual
   adds). Today's kernels already round them to nearest even (the probe), but as a property observed of PyTorch's
   kernels. A kernel Weightsift owned, with round to nearest even as its documented contract and tested bit for bit
   against the reference, would let the certified model use it for those operations (Phase 5A's what-if: 8.1% → 17.8%)
   without changing a single output.
2. **The float32 transcendentals** (`exp` in SiLU and sigmoid, `rsqrt` in RMSNorm): not correctly rounded (within 2–3
   ulps here), and SiLU's formula overflows for x < −88.7. The certified model allows 32 ulps and an absolute term for
   the overflow. A documented bound per function, or correctly rounded implementations, would replace the allowance
   with the real error; their effect on BF16 outputs is rare (SiLU: 6 of 64,392 BF16 inputs, all in the overflow case).
3. **The reductions** (RMSNorm's mean of squares, the experts' combine): float32 sums in an order the kernel chooses
   (correctly rounded on 55–66% of outputs here). A fixed, documented order (a pairwise tree) would bound them by
   γ_⌈log₂ n⌉ instead of γ_n.
4. **GEMM accumulation and shape dependence**: cuBLAS chooses a kernel per shape (GEMV for one row, tensor-core GEMMs
   for more), tensor-core accumulation is not documented (hence u = 2⁻²²), and a row's result depends on its batch.
   A GEMM with a documented accumulation (FP32 fused multiply-adds in a fixed order, the same for every M) would
   justify 2⁻²⁴ (Phase 5A's what-if: 8.1% → 16.5%) and make the reference independent of batch composition, which
   batched serving and a fused decoder (6B) need; otherwise the reference must state its shapes, as it implicitly does
   today.
5. **The combine order**: fixed by transformers (`grouped_mm`: weighted outputs summed over the 6 in float32, one cast;
   Phase 4B's chunked calls rely on it). A fused experts kernel must keep it, or the reference changes.

None of these changes what the reference computes today; each would change what Weightsift can prove about it.

## 12. Where Phase 6B plugs in (§8; nothing built)

Phase 6B (C++/CUDA execution: GPU bit-plane reconstruction, pinned staging with asynchronous copies, CUDA Graphs, fused
decode) needs three things from this phase, and gets them without changes to the core's model:

- **A transfer is a list of byte operations, not a tensor.** A job hands Python, piece by piece, a slot and its ops
  (`(source buffer, offset in the slot, request, offset in the destination, length)`), and Python issues them. An op is
  a copy of bytes as stored. A stored representation that is not the reference's bytes (Phase 5C's exact bit planes,
  zstd frames per 16-row page: a third fewer drive bytes per decode token) would add an op that names a decoder: the
  core reads the frames into the slot and decompresses them on its threads (Phase 5C: zstd keeps up with the drive),
  and the caller launches a device kernel that merges the planes into BF16 rows in the destination instead of a copy.
  Plans, slots, leases, the cache (then holding compressed rows: about 1.5 times as many experts per byte) and the
  accounting stay as they are; exactness stays checkable the same way (every row against the reference's bytes).
- **The copy issuer is one loop.** `PageStreamer._transfer_native` is the only place that turns ops into device work
  and releases slots (after the copy stream's event). A native issuer (CUDA in a C++ extension: `cudaMemcpyAsync` or a
  decode kernel per op on a stream, slots released from a host callback or an event poll) would replace that loop
  without changing the job or its Python caller; the slots are already page-aligned pinned memory, and the core never
  touches CUDA.
- **Static decode shapes, outside the experts.** CUDA Graphs for attention, norms, routers, shared experts and the LM
  head (Phase 4B: 22% of a decode step is Python and kernel launches) need the experts' buffers at fixed addresses: the
  compact call's buffers are already allocated per call from a fixed budget, and a graph-captured decode step would
  replay with the experts' transfer as its input. The reference's per-shape kernel choices are the constraint (§ 11:
  a row's result can depend on its batch), so a fused decoder must reproduce them or the reference must be redefined
  first (Phase 6C's question).

## 13. Answers

1. **Does native code pay end to end?** Yes, measurably and not only in microbenchmarks: the native read path alone
   saves 9% of a decode step and 11% of a prefill (the same reads, less Python around them), and the host-RAM tier it
   makes affordable saves 30% of a decode step against the best Python configuration (807 against 1,157 ms), 35%
   against Phase 4B's streaming, and 37% of a prefill. The gate (≤ 0.95) and the aspirational 20% are both met.
2. **Where did it pay, and where not?** Planning (14× faster), read submission and piece handling (the drive at 3.38
   GB/s instead of 3.18), and above all the host-RAM tier, whose lookups, leases, admissions and copies run on the
   core's threads without the GIL, where the device cache cannot grow (the GPU has no memory left). Not: more reader
   threads or larger read calls (the drive's rate is reached), exact chunk prefetch (slower prefills), speculative
   prefetch (not built). The simpler paths were kept.
3. **Is it exact?** Yes: bytes, routing, expert outputs, logits, tokens and the KV cache equal the independent
   reference on every step of every configuration, through cache hits, misses, evictions, bypasses, waits on
   prefetches and chunked calls, in two runs with different hash seeds; the native backend without a cache reads
   exactly what the Python backend reads.
4. **What bounds it now?** The bytes copied to the GPU (2.70 GB per token at 6.1 GB/s here), then the transformer's
   Python and launches, then the cache's submit cost. More host cache cannot help much beyond 12 GB on this machine
   (the copies stay); the next gains are fewer bytes to the device and a fused transformer: Phase 6B.
5. **What does Phase 6C inherit?** A probed description of the reference's rounding behaviour, the finding that the
   reference is shape-dependent, and a list of candidate operations for explicit native semantics; no claim.

## 14. Recommendation

The next phase is the user's decision; nothing of 6B or 6C was started. In order of measured return:

1. **Take the cache's memory churn off the critical path** (a small change in `cache.rs`/`engine.rs`: recycle evicted
   buffers across row sizes, or free and allocate on the readers): up to about 100 ms of an 807 ms decode step, by the
   profiles and the microbenchmark.
2. **Move fewer bytes to the GPU** (Phase 6B): Phase 5C's exact bit planes decoded on the device would cut the copies by
   a third (2.70 → 1.78 GB per token, about 150 ms here) and let the host cache hold 1.5 times as many experts; a device
   cache in front of the host tier saves the copies of its hits (Phase 4B's 80-expert hotness cache: 65 ms).
3. **Fuse the transformer's decode** (CUDA Graphs or a fused decoder): about 260 ms of Python and 5,195 launches per
   token, but only with batch-invariant kernels or a reference redefined first (§ 11).

## 15. Limitations

- One machine: Windows 11, an 8 GB RTX 4060 Ti on PCIe 3.0 x8, a Samsung 990 PRO on PCIe 3.0 x4, 32 GB of RAM. The
  copy floor, the drive's rate and the cache's best budget are this machine's; Linux builds (`O_DIRECT`) but was not run.
- The profiles are 4 prompts × 8 decode steps per configuration, one process each; the repeats put run-to-run noise at
  1–3%. Cold and warm refer to Weightsift's caches; the drive's own caching cannot be controlled (direct I/O bypasses the
  OS's).
- Main-thread CPU time is Windows' coarse thread clock and includes spin-waits on CUDA events; it bounds the Python and
  FFI overhead from above, it does not isolate it.
- The host cache is LRU only (the brief's deterministic baseline); a frequency-aware policy (Phase 4B's hotness) runs only
  in the Python device cache, and the two tiers together were not measured.
- The submit-cost attribution comes from a synthetic microbenchmark (1.2 ms per transfer there, about 4 ms in the model's
  process); the fix is not measured because it was not built.
- §7's probe uses random inputs in normal ranges and Moonlight's shapes on this GPU, driver and cuBLAS; kernel choices,
  and hence shape dependence, can differ elsewhere.

## 16. Reproduction

Raw results are in `experiments/phase6a/` (packs and model files stay out of Git; `packs/moonlight-16b-a3b-expert-index`
is rebuilt by `weightsift pack expert-index --config configs/phase4b-moonlight.yaml`). Nothing else may run on the
machine during the timed stages (one GPU process at a time: 8 GB under WDDM).

```bash
python -m uv sync                                    # builds native/ (Rust 1.95.0 via rustup; maturin through uv)
cd native && cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace && cd ..
python -m uv run python -m pytest                    # the whole suite, every MoE parity test through every backend
# Stage A: the Phase 4B benchmark again (the reference every native run shares), and the routing traces
PYTHONHASHSEED=1 python -m uv run python benchmarks/moonlight_runtime.py --output experiments/phase6a/baseline-run1
python -m uv run python benchmarks/native_trace.py --output experiments/phase6a/trace
# Stage B: the I/O core alone
python -m uv run python benchmarks/native_io.py --output experiments/phase6a/io/io-cpu.json --requests 100
python -m uv run python benchmarks/native_io.py --output experiments/phase6a/io/io-cuda.json --requests 100 --cuda --sweep
python -m uv run python experiments/phase6a/io/submit_bench.py > experiments/phase6a/io/submit.txt   # the submit cost (§ 10)
# Stages C and D, correctness: two runs with different hash seeds against the baseline's reference
for seed in 1 2; do
  PYTHONHASHSEED=$seed python -m uv run python benchmarks/moonlight_runtime.py --config configs/phase6a-native.yaml \
      --output experiments/phase6a/native-run$seed --stage prepare --reference-from experiments/phase6a/baseline-run1
  PYTHONHASHSEED=$seed python -m uv run python benchmarks/moonlight_runtime.py --config configs/phase6a-native.yaml \
      --output experiments/phase6a/native-run$seed --stage stream
  PYTHONHASHSEED=$seed python -m uv run python benchmarks/moonlight_runtime.py --config configs/phase6a-native.yaml \
      --output experiments/phase6a/native-run$seed --stage digest
done
# Performance: one configuration per process, cold and warm (the commands are in experiments/phase6a/native-run1/profiles.sh)
python -m uv run python benchmarks/moonlight_profile.py --run experiments/phase6a/native-run1 --config configs/phase6a-native.yaml \
    --configuration native-host-12g --prompts 7 0 3 5 --warm --output experiments/phase6a/native-run1/profile-native-host-12g-warm.json
python -m uv run python benchmarks/native_report.py experiments/phase6a/native-run1 --compare experiments/phase6a/native-run2 \
    --baseline experiments/phase6a/baseline-run1 --io experiments/phase6a/io/io-cpu.json experiments/phase6a/io/io-cuda.json
# §7: the reference's kernels
python -m uv run python benchmarks/reference_numerics.py --output experiments/phase6a/bf16/numerics.json
```
