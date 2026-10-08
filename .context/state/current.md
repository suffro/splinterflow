# Current State

**Project name: Weightsift.** Use this name consistently in documentation, code comments, and filenames.

**CLI:** `weightsift` is the command; `wsift` is its equivalent shorthand. Both expose
the `pack` command (`lm-head`, `experts`, `expert-index`). The Python package is `awpmi`.

## Latest: Phase 6A (the native runtime foundation) is complete (2026-10-08)

**Answer: native code pays end to end.** The report is `history/2026-10-08-weightsift-phase6a-report.md`; the decision
is 0012. The brief was the user's of 2026-10-08 (`state/phase_6A.md`, left untracked like the earlier briefs).

- **Stage A.** Phase 4B's benchmark reproduced bit for bit on the Phase 6A tree (`experiments/phase6a/baseline-run1`:
  every digest equal to Phase 4B run1's). A decode token reads 2.70 GB of routed experts, 99.5% of them read before; an
  LRU host tier hits nothing below one token's working set (2.6 GB), then 0.39–0.43 of decode lookups at 4 GB and
  0.69–0.76 at 12 GB (exact stack distances, `benchmarks/native_trace.py`).
- **The native core** (`native/`: Rust 1.95.0, PyO3 0.29, built by maturin through uv's default `native` group): read
  plans equal to Python's, direct positioned reads on 8 threads, a host-RAM expert cache (strict byte budget,
  deterministic LRU, leases, load-once, hits protected before admissions, admission freeze, memory recycling), transfer
  jobs into the streamer's pinned slots, exact chunk prefetch, full byte accounting. No model knowledge, no CUDA, one
  `unsafe` type. `awpmi.storage.native.NativePageStore` puts it behind the storage contract; configurations choose
  `backend: python | native`; the Python backend stays the reference implementation and the fallback.
- **Stage B.** The same reads as Python (bytes, read calls, OS counters equal); a decode call 13% faster to the GPU and
  27% to host memory; planning 0.13 ms against 1.8 ms; more threads or larger reads do not help.
- **Stages C and D, correctness.** Every step of 8 configurations equal to the independent reference in every Phase 4B
  digest, in two runs (`native-run1`, `native-run2`; `PYTHONHASHSEED` 1 and 2) with identical digests; `native-stream`'s
  records and raw I/O equal `python-stream`'s; the cache's hits and misses equal an LRU replay on every step; budget
  never exceeded. Exact chunk prefetch is correct but slows prefills (off by default); speculative prefetch was evaluated
  on traces only (10–21% of misses covered) and not built.
- **Performance (gate PASS).** Warm, 807 ms per decode token with 12 GB of host cache and prefill admission frozen,
  against 1,157 ms for the best Python configuration (0.697; gate ≤ 0.95) and 1,233 ms for Phase 4B's streaming (0.654;
  the aspirational 20% met); prefills 37% faster. The native read path alone: −9% per decode token.
- **What bounds a decode token now**: copying its 2.70 GB of experts to the GPU (about 440 ms over this machine's PCIe
  3.0 x8, whatever tier serves them), the transformer's Python and 5,195 kernel launches (about 260 ms), and the cache's
  submit cost (about 100 ms: freeing and allocating entries of different sizes inline; the first follow-up).
- **§7 (for 6C).** `BF16_REFERENCE` is "this code at these shapes": BF16 roundings are round to nearest even and
  correctly rounded (probed, `benchmarks/reference_numerics.py`), float32 transcendentals within 2–3 ulps, and a row's
  GEMM result depends on its batch (cuBLAS GEMV for one row, tensor-core kernels for more). Candidate operations for
  explicit native semantics are listed in the report; nothing is claimed, and the 8.1% faithful ceiling is not a target.
- **§8 (for 6B).** The integration point is the transfer job's byte operations and `PageStreamer._transfer_native`, the
  only loop that turns them into device work; nothing built.
- **A defect the gates caught**: a closed native store's cache outlived it (the first full run's working set grew per
  configuration). `close` now releases the cache, a reference cycle is gone, two tests guard it, and the run was repeated.

## Previous focus

**Phase 5C (exact shared bases and progressive expert deltas) is complete (2026-10-08)** (report
`history/2026-10-08-awpmi-phase5c-report.md`, decision 0011): neither pays on Moonlight. A layer's routed experts share
no exact structure; independent exact compression (bit planes with zstd, 0.661 of BF16) is the measured gain (a third
fewer drive bytes per decode token); progressive bit-plane materialization needs 0.83–1.00 of the exact bytes to decide a
token in real arithmetic, and Phase 5A's BF16 rounding floor remains.

Phases 1A, 1B, 1C, 2, 3, 4A, 4B, 5A, 5A2, 5C and 6A are complete; their reports are in `history/`. Phase 4B (decision
0008) runs Moonlight out of VRAM and host RAM, bit for bit equal to an independent reference; Phase 6A (decision 0012)
runs it on the native core.

## Recent relevant changes

- `native/` (new; decision 0012): the Cargo workspace (`core/`: `weightsift-io`; `python/`: the PyO3 module
  `weightsift_native`), `native/README.md` (build, toolchain, platforms, fallback, the `unsafe` boundary).
- `src/awpmi/storage/native.py` (new): `NativePageStore`, `NativeTransfer`, `NativePrefetch`, `NativeIOStats`.
- `src/awpmi/storage/pack.py` (`Pack.store(backend=…)`), `streaming/streamer.py` (native transfers through
  `native_slots`; `fetch_many`), `materialization/backend.py` (`materialize_many`, `prefetch_rows`),
  `materialization/weights.py` (`ExpertStore.assemble` in one request list; `prefetch`), `models/moe.py`
  (`prefetch_chunks`), `tracing.py` (`native_tree_sha256` and the extension's version in every run's environment).
- `benchmarks/native_trace.py`, `native_io.py`, `native_report.py`, `reference_numerics.py` (new);
  `moonlight_runtime.py` and `moonlight_profile.py` gain backends, host caches, freeze, prefetch, `--reference-from`,
  `--warm` and the cache's audits; `configs/phase6a-native.yaml`; raw results in `experiments/phase6a/`.
- Tests: `tests/test_native_storage.py` (new), the MoE parity tests through every backend (`tests/conftest.py`), the
  layering test over the Rust sources; Rust tests under `native/core`.
- `pyproject.toml` / `uv.lock`: the default dependency group `native` (the local `weightsift-native` package).
- Decision 0012 is new.

## Next

The next phase is **not started**. It needs the user's go-ahead. The candidates, by measured return:

1. **The cache's memory churn off the critical path** (small, in the native core): up to about 100 ms of an 807 ms
   decode token.
2. **Phase 6B, fewer bytes to the GPU and a fused decode**: Phase 5C's exact bit planes decoded on the device (copies
   2.70 → 1.78 GB per token, about 150 ms here; 1.5× the experts per byte of host cache), a device cache in front of the
   host tier, CUDA Graphs for the transformer's 5,195 launches per token, native copy issuing; the integration point is
   ready (report § 12).
3. **Phase 6C, the reference's semantics**: whether `BF16_REFERENCE` should be shape-independent (batch-invariant GEMMs)
   and which roundings Weightsift should own (report § 11).

Open decisions for the user:

- the next milestone;
- the rounding model for certificates (RN-even, still open from Phase 2);
- the reference for FP8 experts;
- Phase 2 on a larger model;
- whether the Phase 6A brief (left untracked in the working tree's `.context/state`) is added to the repository.

## Blockers

None technical. The next phase needs the user's decision to proceed.
