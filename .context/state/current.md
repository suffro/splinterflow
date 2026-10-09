# Current State

**Project name: Weightsift.** Use this name consistently in documentation, code comments, and filenames.

**CLI:** `weightsift` is the command; `wsift` is its equivalent shorthand. Both expose
the `pack` command (`lm-head`, `experts`, `expert-index`, `encoded-experts`). The Python package is `awpmi`.

## Latest: Phase 6B (GPU transfers and decode execution) is complete (2026-10-09)

**Answer: 408.9 ms per decode token, 0.507 of Phase 6A's 807, bit for bit equal to the reference.** The report is
`history/2026-10-09-weightsift-phase6b-report.md`; the decision is 0013. The brief is `state/phase_6B.md` (the user's of
2026-10-09, committed with the roadmap).

- **Baseline first**: Phase 6A's best configuration reproduced on its own tree (810.8 and 806.9 ms; 5,195 launches per
  token, 524 ms of copies, the GPU idle 39–41%).
- **6B1, the host cache's submit cost** (native core): rows held in blocks of one size reused across row sizes (the pool
  inside the budget), readers woken one at a time instead of all per task; 100 → about 1 ms per decode token; BF16
  host-cache decode 0.964 of the 6A tree's. A latent 6A deadlock (two jobs waiting on each other's loads) fixed: a request
  reads a row another request is loading itself, unadmitted.
- **6B2, experts encoded for the GPU**: nvCOMP decodes Phase 5C's zstd frames exactly but too slowly with the plane merge;
  nvCOMP's rANS over raw BF16 rows (1 MiB chunks) stores 0.6747 and decodes a call in 0.7 ms. The encoded pack
  (`weightsift pack encoded-experts`, 19.42 GB) is read and cached by the native core as bytes and decoded on the GPU into
  the experts' buffers: 610.8 against 787.2 ms at the same 12 GB host cache (0.776); every row audited before inference.
- **6B3-A, a device tier**: `SlotCache`, fixed slots per stored row size (1.9 GB, LRU, prefill admission frozen) under the
  6 GB cap; it pays only from one token's working set (an LRU cliff at 1.82 GB): 493.7 against 610.8 ms (0.808).
- **6B3-B, native copy issuing**: measured (−1.3%, then nothing with the fills off), not built.
- **6B4, decode graphs**: `DecodeGraphs` replays each DeepSeek-V3 layer's static decode pieces from two CUDA Graphs (54 in
  all, 65–68 MB), the KV cache, attention and routed experts eager between them; and PyTorch's NaN fill of uninitialized
  memory (deterministic mode's debugging aid: 270 fill kernels per token) is off per configuration: 408.9 against 493.7
  ms (0.828); 2,344 kernel + 54 graph launches per token.
- **Correctness**: 1,116 steps per run (10 configurations) equal to the reference in every digest, two runs with
  identical digests, I/O parity, both cache tiers equal to their replays, the encoded pack's 3,328 rows equal.
- **What bounds a token now**: the bytes still copied (1.24 GB, 198 ms), the host cache's copies and the drive behind
  them (124 ms of waits), and the eager remainder of the transformer (about 170 ms of the main thread).
- **§ 6C (prepared, not started)**: shape-dependent kernels (kept by graph capture), workspace-dependent cuBLAS choices,
  reduction orders, where PyTorch changes execution without changing values, and the candidates for an explicitly
  reproducible native reference are in the report's § 12.

## Previous focus

**Phase 6A (the native runtime foundation) is complete (2026-10-08)** (report
`history/2026-10-08-weightsift-phase6a-report.md`, decision 0012): a small, model-agnostic Rust I/O core (`native/`)
behind the Python storage contract, with a host-RAM expert cache; 807 ms per decode token against 1,157 ms for the best
Python configuration, bit for bit equal to the independent reference.

Phases 1A, 1B, 1C, 2, 3, 4A, 4B, 5A, 5A2, 5C, 6A and 6B are complete; their reports are in `history/`. Phase 4B (decision
0008) runs Moonlight out of VRAM and host RAM, bit for bit equal to an independent reference; Phase 6A (decision 0012)
runs it on the native core; Phase 6B (decision 0013) moves encoded experts to the GPU and graphs the decode step.

## Recent relevant changes

- `native/core/src/cache.rs`, `engine.rs` (decision 0013): block-held rows with a pool inside the budget, one-at-a-time
  reader wake-ups, the busy rule; `EngineConfig.cache_block_bytes`; submit timers in the I/O statistics;
  `native/README.md` (the cache's memory, the `unsafe` boundary's new use).
- `src/awpmi/storage/encoded.py`, `streaming/codec.py`, `streaming/nvcomp.py` (new): the encoded pack, its writer and the
  GPU decoder (nvCOMP's C API through ctypes); `cli.py` (`pack encoded-experts`).
- `src/awpmi/storage/cache.py`: `SlotCache` (fixed device slots per page size); `materialization/backend.py`: encoded
  segments through a decoder, with or without a device cache.
- `src/awpmi/models/decode_graphs.py` (new): `DecodeGraphs` for DeepSeek-V3 decoder layers.
- `benchmarks/gpu_codec_probe.py`, `encoded_io.py`, `gpu_report.py` (new); `moonlight_runtime.py` and
  `moonlight_profile.py`: encoded experts, the encoded audit, device slot caches, `decode_graphs` and
  `fill_uninitialized_memory` per configuration, graph launches in traces, the recorder's chunked-call fix;
  `configs/phase6b-gpu.yaml` (configurations and frozen gates); raw results in `experiments/phase6b/`.
- Tests: `tests/test_encoded_storage.py`, `tests/test_decode_graphs.py` (new), slot-cache tests in `test_streaming.py`,
  the MoE parity tests through encoded rows (`conftest.py`), a block-size test in `test_native_storage.py`; Rust tests for
  the block pool, the busy rule and recycled evictions.
- `pyproject.toml` / `uv.lock`: the default dependency group `gpu` (`nvidia-libnvcomp-cu13`, NVIDIA's SDK license).
- Decision 0013 is new.

## Next

The next phase is **not started**. It needs the user's go-ahead. The candidates, by measured return:

1. **Phase 6C, the reference's semantics**: whether `BF16_REFERENCE` should be shape-independent (batch-invariant GEMMs,
   fixed reduction orders, owned elementwise kernels), which the fastest remaining options need (report § 12, § 14).
2. **More of the decode step in graphs**: the attention over a growing cache and the experts' compute (needs 6C's answer
   for a static, padded cache).
3. **More experts per device byte**: a better-ratio codec as fast to decode, or a hotness-aware slot policy for GPUs with
   less room than one token's working set.

Open decisions for the user:

- the next milestone;
- the rounding model for certificates (RN-even, still open from Phase 2);
- the reference for FP8 experts;
- Phase 2 on a larger model.

## Blockers

None technical. The next phase needs the user's decision to proceed.
