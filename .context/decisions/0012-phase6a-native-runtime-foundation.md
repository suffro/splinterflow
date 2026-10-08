# 0012 — Phase 6A: a native (Rust) I/O core with a host-RAM expert cache, behind the Python storage contract

Status: accepted (Phase 6A, 2026-10-08)

## Context

Phase 4B (decision 0008) runs Moonlight-16B-A3B out of VRAM and host RAM, bit for bit, at 1.27 s per decode token: the
drive is 66.5% of a decode step (2.70 GB of routed experts per token at 95% of its rate), the transformer's Python and
kernel launches 22%, and the per-request transfer overhead (planning, copy issuing, assembly, the routing sync) 11%. Its
replay of the routing through a cache found that caching pays only above one decode token's working set (156 experts,
2.6 GB): a host-RAM tier, not the GPU, which has no room. Phases 5A, 5A2 and 5C closed expert AWPMI on BF16 Moonlight
and pointed to this engineering path (decisions 0009–0011).

The user's Phase 6A brief (2026-10-08, `state/phase_6A.md`):

- Begin moving Weightsift's performance-critical runtime infrastructure from Python to Rust (I/O, caching, prefetching,
  scheduling), keeping the Python research layer and the PyTorch/CUDA execution. 6B (C++/CUDA execution, fused
  reconstruction, CUDA Graphs), 6C (BF16 numerical fidelity) and Phase 7 come later: clean interfaces for them, nothing
  more.
- Inspect first; a mandatory open-source investigation (ds4, colibri-LLM, llama.cpp, Rust crates for file I/O, bounded
  caching, async execution, bindings) with licenses and maintenance checked; do not vendor an engine for its cache.
- A small, model-agnostic Rust core through PyO3 and maturin (or an equally justified binding): no model names, tensor
  names, expert counts or routing logic in Rust; no GIL during blocking work; bounded host memory with eviction;
  in-flight deduplication; coalesced reads where measurable; cancellation and safe shutdown; errors into Python; a coarse
  FFI boundary; platform-specific I/O only when measured, with a portable fallback; no unsafe memory handling without a
  demonstrated requirement; the Python backend kept as the fallback.
- Stages: A baseline and profiling (reproduce Phase 4B; token latency; I/O, cache, scheduling, transfer and Python
  overhead; repeated expert bytes) before any Rust; B native I/O (benchmarked alone before inference; bytes identical to
  the Python backend; the existing index, no new parser); C a native host-RAM cache (strict budget, deterministic LRU
  baseline, thread-safe, in-flight deduplication, metrics, several budgets on Phase 4B's routing traces); D conservative
  exact prefetch, speculative prefetch only as a separate evaluation, and an explicit `python` / `native` backend choice
  that changes no BF16 arithmetic, routing, combine, KV cache, accumulation order, logits or token.
- Gates: a measurable end-to-end benefit, not microbenchmarks (20% lower decode latency is aspirational, not to be
  manufactured); every native change measured separately and end to end, cold and warm, the OS page cache controlled;
  simpler implementations kept where native code does not pay.
- Correctness: `BF16_REFERENCE`; bytes, routing, expert outputs, logits, tokens and the KV cache identical; identical
  across cache hits, misses and evictions; no hidden reads; memory limits respected; no races or double loads; the
  Phase 4B parity tests through both backends; Rust unit tests and Python integration tests.
- §7: collect, for 6C, the kernels that define the reference, their BF16 and accumulation behaviour, where the
  certificate's rounding uncertainty comes from, and which operations might gain explicit native semantics; no new
  kernels, no rounding claim. §8: an integration point for a future CUDA decoder of Phase 5C's bit planes.
- A technical report, reproducible results, this decision, `state/current.md` and the architecture updated; Rust format,
  lint and tests; Python tests; Moonlight parity; benchmarks; `syngraphe check`; commit and push.

## Decision

1. **Where the time goes decides what moves (stage A).** Phase 4B's benchmark was reproduced in full on the same
   machine and software (§ Results); its profile and a routing-trace analysis (`benchmarks/native_trace.py`) answer
   the brief's questions:
   - decode reads every byte it uses again and again: 99.5% of a decode token's 2.70 GB was requested before (48% by
     the previous step of the same layer, 50% earlier in the same prompt, mostly by its prefill);
   - an LRU tier in host RAM hits nothing below one token's working set (2.6 GB), then 0.37–0.40 of decode lookups at
     2.77 GB, 0.57–0.64 at 8 GB, 0.69–0.76 at 12 GB, 0.80–0.85 at 16 GB (exact, from byte stack distances over Phase
     4B's and Phase 5A's traces, checked against a `PageCache` replay);
   - a speculative prefetch of a layer's previous-step experts would cover only 10–21% of such a cache's misses, at
     25–32% precision.

   So the native core is first a host-RAM expert tier with a native read path, and the Python overhead it removes
   (planning, read submission, gathering) comes with it. Speculative prefetch is evaluated on the traces only.
2. **Reuse, not a new engine.** Reviewed (licenses, activity, fit):
   - *ds4* (antirez, MIT, active): a C/CUDA/Metal engine whose CUDA SSD streaming keeps a device-side expert cache with
     LRU stamps, protects every hit of a request before choosing victims, publishes look-ahead loads at the lowest
     recency, reads with `O_DIRECT` into a pinned staging ring with events, and joins a cancellable reader thread on
     shutdown. Patterns reused: hits protected first, staged direct reads, cancellation and join.
   - *colibri* (JustVugg/colibri, Apache-2.0, active; `jenovauh/colibri-LLM` is a fork): a C engine streaming experts
     with one read per expert (its three matrices adjacent, as Moonlight's are), an 8-worker read pool with per-slot
     readiness, batch-union, a per-layer LRU with a learned pinned set, `O_DIRECT`, and router look-ahead. Patterns
     reused: a bounded worker pool, one transfer for all of an experts call's rows. Not reused: router look-ahead
     (model-specific routing logic, which the brief keeps out of the core).
   - *llama.cpp* (MIT, active): experts resident, offloaded to the CPU, or mapped and paged by the OS; its loader reads
     with direct I/O (POSIX) through four pinned staging buffers with events. No runtime SSD expert cache (a request for
     a two-tier expert cache, issue #20757, was closed). Nothing new for Weightsift; `mmap` stays rejected (decision
     0006: its reads cannot be planned or counted).
   - Rust crates: PyO3 0.29 and maturin 1.15 (bindings and builds, active), `lru` 0.18 (the recency order), `libc` 0.2
     (Linux `O_DIRECT`). Positioned reads and direct-I/O flags come from the standard library. Not used: `moka`,
     `quick_cache`, `foyer` (concurrent or hybrid caches whose eviction is approximate or asynchronous: the brief asks
     for a deterministic LRU with exact budget and lease semantics), `tokio` (file I/O there is a blocking pool anyway),
     `io-uring` and Windows `IoRing` (platform-specific; decision 0006 measured that threads of positioned reads already
     reach the drive's rate), `rayon` (not for blocking I/O).

   Weightsift-specific, written here: the planner (it must equal Python's), the transfer jobs into caller-owned pinned
   slots, the cache's budget, leases and load-once semantics, and the byte accounting the audits need.
3. **Architecture.** `native/` is a Cargo workspace (toolchain pinned to 1.95.0):
   - `weightsift-io` (`native/core`): segments described by the caller (the checkpoint index: plain and composed
     segments of Phase 3 and 4A), read plans equal to Python's `plan_reads` and `PageStreamer._pieces`, positioned reads
     on a pool of threads (direct I/O; one handle per thread and file on Windows), a host-RAM cache, transfer jobs,
     statistics. No Python, no CUDA, no model knowledge (`tests/test_layering.py` scans the Rust sources too).
   - `weightsift-native` (`native/python`): the PyO3 module `weightsift_native` (abi3, Python 3.11+), built by maturin
     through uv (a default dependency group `native`; `uv sync --no-group native` leaves it out).
   - `awpmi.storage.native.NativePageStore`: the `FileBackedPageStore` contract (plans, reads, counters, OS
     cross-check) on the core, plus the host cache; `Pack.store(backend="python" | "native")` selects it. Without the
     extension `NATIVE_AVAILABLE` is False and nothing else changes.
   - The streamer drives a native store through one transfer per call (`PageStreamer.fetch_many`,
     `MaterializationBackend.materialize_many`; `ExpertStore.assemble` asks for all of a call's parameters at once):
     pieces are delivered as soon as they are ready, copied to the device on the copy stream, and released once copied;
     the readers keep filling the other slots (four by default). The PyTorch execution, the compact and chunked calls,
     the combine and every kernel are unchanged.
4. **Transfer jobs, piece by piece.** One call (`Engine.submit`, the GIL released) plans a whole transfer: every row is
   looked up in the host cache first (hits leased, so that no miss of the same job evicts them), then the misses are
   admitted and planned exactly as Python plans them. Cached rows are copied into the first pieces (hits first); the
   misses' extents are read into the next ones (at most `max_read_bytes` per call, eight threads); short parts are
   gathered as Python's streamer gathers them. A piece is handed to Python as soon as its reads and copies are done,
   with the list of device copies to issue (the same copies the Python path issues for the same plan). Missed rows the
   cache admitted are copied from the slot into their entries meanwhile; a slot is refilled only after those copies and
   Python's release. Python never holds more than all but one slot, so the readers always have one to fill.
5. **The host cache.** Rows of segments under a strict byte budget: every byte held counts, rows being loaded included
   (reserved when their load starts), so resident bytes never exceed the budget. Least recently used first, by the
   `lru` crate's order. Leased rows (being copied) and rows being loaded are never evicted; a row that would need them
   is not admitted (`bypassed`). A row is loaded once however many requests want it (`waits` follow the load; a failed
   or cancelled load sends its waiters to read the row themselves). An evicted entry of the new row's size gives it its
   memory (no allocation and no page faults: a first write into fresh memory ran at 3 GB/s against 10–17 GB/s). An
   admission freeze serves hits and admits nothing (DwarfStar's prefill fix). Counters: lookups, hits, waits, misses,
   inserts, evictions, bypassed, aborted fills, recycled, prefetch fills, used and wasted, resident and peak bytes.
   Closing the engine releases the cache's memory at once (a closed store serves nothing again): the first full run
   showed that leaving it to the engine's destructor let each configuration's cache outlive it, because a reference
   cycle (the store and its statistics) kept the store alive until a full garbage collection; the cycle is gone too.
6. **Exactness and audits.** The native path changes where bytes move, never what is computed:
   - plans, pieces and copies equal the Python ones (tests on random plain and composed requests); without a cache,
     a native configuration's records equal the Python one's in every field but timings and system, and its raw
     physical-I/O traces are the same extents;
   - every step of every native configuration (with and without caches, under eviction, chunked calls, poisoned
     spares) is compared with the independent streaming reference in every Phase 4B digest;
   - the step audit gains the cache's identities (lookups = rows = hits + waits + misses; bytes copied from the cache =
     its hits' and waits' bytes; resident within the budget); physical bytes are still exactly the 4 KiB blocks of the
     rows read from storage, and the OS's counters still equal the store's;
   - the measured hits and misses of every step equal an LRU replay of the reference's routing in the native order;
   - whether a row the cache serves was ready (a hit) or still being loaded by a prefetch (a wait) is a matter of
     timing, so records digest their sum (`served`) and keep the split with the timings; everything else a native step
     records (bytes, reads, copies, evictions, admissions, resident bytes) is deterministic and digested.
7. **`unsafe`, where required.** One type (`RawBuffer`, `native/core/src/buffer.rs`): writing into staging memory the
   caller owns (PyTorch's pinned buffers), so that reads land where the device copies from (the zero-copy property
   decision 0007 bought). Its invariants (memory alive while any task may touch it; disjoint ranges per task; a slot
   handed over only after its writes, refilled only after its release) are upheld by the engine and the binding (`Job`
   keeps the buffers referenced until none of its tasks runs). Seven `unsafe` blocks use it (the engine's reads, cache
   copies, fallback reads, gathers, admissions and `read_rows`' copy out; the binding's wrapping of a Python buffer),
   each with the invariant it relies on; everything else is safe Rust.
8. **Prefetch: exact only (stage D).** The rows Weightsift knows before it needs them are a chunked call's later
   chunks (a prefill's experts call over the 256 MiB budget): `StreamedExperts(prefetch_chunks=True)` announces them
   when the call starts, and the native cache loads them at the lowest priority (a background queue the readers serve
   only when no transfer task waits; staging of its own; no leases, no lookups counted) while the first chunks are
   read and computed. A decode call is one transfer issued as soon as its router has chosen; the next layer's experts
   are not known before its router runs. Speculative prefetch (a layer's previous-step experts) is not built: on the
   traces it is right for 25–32% of what it loads and covers 10–21% of a host cache's misses, on a drive already busy
   most of a step.
9. **The BF16 reference (§7): documented, unchanged.** `benchmarks/reference_numerics.py` records how the reference's
   kernels behave on this machine: BF16 conversions and BF16 add and multiply are round to nearest even and correctly
   rounded; float32 `exp`, `sigmoid` and `rsqrt` are within 2–3 ulps (the certified model allows 32); SiLU's float32
   formula overflows below −88.7 and returns −0 (the certified operator's documented allowance); and a row's GEMM result
   depends on its batch (cuBLAS's GEMV kernel for one row, tensor-core kernels for more; `grouped_mm` likewise with an
   expert's rows in the call), while attention is the same at decode and prefill. `BF16_REFERENCE` is therefore "this
   code at these shapes". The report lists the operations that might gain explicit native semantics in Phase 6C
   (elementwise roundings, transcendentals, reduction order, batch-invariant GEMMs, the combine order); nothing is
   claimed, and Phase 5A's 8.1% faithful ceiling is an observation, not a target.
10. **Where Phase 6B plugs in (§8; not built).** A transfer is a list of byte operations into caller-owned pinned slots,
    and the streamer's `_transfer_native` is the only loop that turns them into device work and releases slots. A
    stored representation that is not the reference's bytes (Phase 5C's bit planes in zstd frames) would add an op
    naming a device decoder; a native copy issuer (CUDA in a C++ extension) would replace the loop. Plans, slots, the
    cache (then holding compressed rows) and the accounting stay; the core stays CUDA-free.

## Results

Full report: `history/2026-10-08-weightsift-phase6a-report.md`. Raw data: `experiments/phase6a/` (`baseline-run1`;
`native-run1` with its profiles and `profiles.sh`; `native-run2`; `io`; `trace`; `bf16`).

- **Stage A.** Phase 4B reproduced bit for bit: `baseline-run1` (Phase 4B's configuration on the Phase 6A tree) equals
  Phase 4B's run1 in all six digests; its profile 1,274 ms per decode token (Phase 4B: 1,273).
- **Correctness.** Every step of 8 configurations (1,008 per run) equal to the reference in every Phase 4B digest;
  `native-stream`'s records and raw physical-I/O ranges equal `python-stream`'s; on every step of every host-cache
  configuration the measured hits and misses equal an LRU replay of the reference's routing; every audit clean.
  Two runs (`native-run1`, `native-run2`; `PYTHONHASHSEED` 1 and 2) have identical digests on the same source and
  native trees.
- **Stage B.** The same reads as Python's; a decode call 13% faster to the GPU and 27% to host memory; planning 0.13 ms
  against 1.8 ms; 16 threads or 4 MiB read calls do not help.
- **Stage C.** Drive bytes per decode token 2.70 → 1.67 GB (4 GB of cache) → 0.81 GB (12 GB); budget never exceeded
  (peak working set 15.4 GB with 12 GB).
- **Stage D.** Exact chunk prefetch is correct (36,318 rows prefetched, all used) but makes prefills slower (9.3 s
  against 7.1 s): off by default. Speculative prefetch not built.
- **Performance (gate: PASS).** Warm, the best native configuration (12 GB host cache, prefill admission frozen) takes
  807 ms per decode token against 1,157 ms for the best Python configuration (Phase 4B's hotness-80 device cache):
  0.697 (gate ≤ 0.95); against Phase 4B's streaming (1,233 ms) 0.654, the brief's aspirational 20% met; 1.24 tokens/s
  against 0.81; prefills 37% faster. The native read path alone: −9% per decode token, −11% per prefill.
- **What bounds a decode token now**: copying its 2.70 GB of routed experts to the GPU (435–445 ms over this machine's
  PCIe 3.0 x8, whatever tier serves the bytes), the transformer's Python and 5,195 kernel launches (about 260 ms), and
  the host cache's submit cost (56–105 ms: freeing and allocating entries whose sizes differ, inline; the first
  follow-up).
- **A defect the gates caught**: in the first full run the process's working set grew with each configuration (a closed
  store's cache outlived it); fixed (point 5), tested, and the run repeated.

## Rejected

- *Vendoring or binding an inference engine (ds4, colibri, llama.cpp) for its cache or reader*: each is a monolithic C
  or C++ engine tied to its own model format, graph and kernels; their patterns are reused (decision point 2).
- *`moka`, `quick_cache` or `foyer` as the cache*: approximate or asynchronous eviction and admission; the baseline the
  brief asks for is a deterministic LRU with an exact budget, leases and load-once semantics, a few hundred lines on
  `lru`.
- *`tokio`, `io-uring`, Windows `IoRing`*: no measured gain over a pool of positioned reads at this drive's rate, and
  the latter two are platform-specific (decision 0006's measurement stands).
- *`mmap`*: the OS decides what is read; bytes could not be planned, counted or audited (decision 0006).
- *Model-specific prefetch (router look-ahead) in the core*: routing logic does not belong in Rust; the trace shows the
  previous-step predictor adds little over an LRU tier of a token's working set or more.
- *Pinning the host cache (registering it with CUDA)*: a pinned arena of many gigabytes, or a registration per entry.
  Copying hits into the pinned slots runs on the core's threads, and on long transfers the hit path already reaches
  5.9 GB/s to the GPU against the link's 6.1 GB/s; a pinned cache would save a host copy, not the link's time. A pinned
  arena is a candidate for 6B, together with native copy issuing.
- *Exact chunk prefetch by default*: correct, but prefills were 31% slower with it (two copies in host memory, two
  staging slots, little compute to overlap); kept as an option.
- *CUDA in the Rust core*: 6B's subject; the core stays CUDA-free and PyTorch keeps streams, events and copies.
