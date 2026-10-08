# Weightsift — Phase 6B: Native GPU Execution & Compressed Expert Streaming

**Repository:** https://github.com/suffro/weightsift  
**Branch:** `weightsift-and-AWPMI-implementation`

Work from the current HEAD.

## Mission

Continue Weightsift's native-runtime development following the successful Phase 6A.

**Phase 6B must reduce GPU transfer costs, eliminate unnecessary runtime overhead, and optimize decode execution while preserving bit-for-bit equivalence to the existing `BF16_REFERENCE`.**

Phase 6A established a Rust I/O and host-cache backend achieving **807 ms/token**, against **1,157 ms/token** for the best Python configuration.

The measured remaining bottlenecks are:

- ~435 ms/token transferring 2.70 GB of expert weights over PCIe 3.0 x8.
- ~56–105 ms/token in native cache submission, largely caused by inline allocation and eviction.
- ~260 ms/token in transformer/Python execution overhead, with approximately 5,195 GPU kernel launches.

Phase 5C also established that exact independent expert compression achieves approximately **0.661× BF16 storage**, but the experimental bit-plane reconstruction is too slow.

Our goal is to turn these findings into a faster, practical runtime.

**Do not begin Phase 6C or resume AWPMI certification research yet.**

---

## 1. Read and profile first

Read:

- `.context/state/Weightsift_Implementation_Guide.md`
- `.context/state/current.md`
- `.context/truth/architecture.md`
- `.context/truth/conventions.md`
- Phase 4B, 5C and 6A reports and decisions.

Inspect:

- `native/core/`
- `native/python/`
- `src/awpmi/storage/native.py`
- `src/awpmi/streaming/streamer.py`
- `src/awpmi/materialization/`
- `src/awpmi/models/moe.py`
- `research/expert_deltas/`
- Phase 6A benchmark/profiling tools.

Reuse the existing Moonlight benchmark, pinned model revision, hardware and memory budgets.

Establish a reproducible Phase 6A baseline before modifying performance-critical components.

---

## 2. Reuse existing native libraries

Before developing GPU code, investigate reusable implementations from:

- NVIDIA nvCOMP — https://docs.nvidia.com/cuda/nvcomp/
- CUDA Runtime, cuBLAS and CUDA Graph APIs
- PyTorch C++/CUDA extension facilities
- existing CUDA-compatible bit-packing/unpacking utilities
- DS4/DwarfStar and Colibrì where applicable.

Prioritize mature existing code.

Do not implement a custom decompression algorithm if an existing library can handle the format efficiently.

In particular, investigate whether nvCOMP can directly decompress Phase 5C's Zstd frames on the actual RTX 4060 Ti. **Do not assume format or compression-level compatibility.** Test frames produced with the recorded Zstd version and settings, including level 19.

If incompatible or too slow, evaluate supported alternative codecs and framing using the existing Phase 5C corpus. Compare their compression ratios and actual end-to-end decode costs.

Record licenses, platform support and integration decisions.

Write a small custom CUDA kernel only for genuinely missing operations, such as a fused BF16 bit-plane merge, and only after measuring the alternatives.

---

# Stage 6B1 — Eliminate Rust cache submission overhead

The first objective is to eliminate the measured allocation/eviction overhead without changing caching semantics.

Investigate:

- different-sized cache entries preventing buffer recycling;
- inline frees and allocations;
- size-classed reusable buffer pools;
- delayed reclamation;
- off-critical-path memory management.

Implement the simplest solution justified by profiling.

Requirements:

- preserve deterministic LRU behavior;
- preserve existing cache hit/miss sequences;
- preserve the strict memory budget, including retained pool memory;
- preserve leases and in-flight deduplication;
- avoid races and use-after-free;
- keep cancellation and cleanup correct.

Benchmark independently and in Moonlight.

Do not accept a microbenchmark improvement if end-to-end decode does not benefit.

**Gate:** correctness unchanged, no additional memory leaks, and measurable reduction in submission overhead.

---

# Stage 6B2 — Compressed experts directly to GPU

This is the main performance objective.

We want to reduce traffic over PCIe by sending compressed expert data rather than reconstructing full BF16 weights on the CPU.

Target pipeline:

```text
SSD / NVMe
    ↓
Rust host-RAM cache
    (compressed expert blocks)
    ↓
Pinned staging buffers
    ↓
PCIe transfer of compressed bytes
    ↓
GPU decompression
    ↓
GPU bit-plane restoration
    ↓
Original BF16 expert tensor
    ↓
Existing PyTorch expert execution
```

The reconstructed BF16 tensor must be identical, bit for bit, to the published checkpoint.

### Implementation requirements

1. Reuse Phase 5C's exact compression formats, benchmark data and reconstruction tests.
2. Introduce a versioned, indexed, independently addressable compressed representation, without duplicating the original checkpoint unnecessarily.
3. Allow the Rust cache to store compressed blocks with accurate byte accounting.
4. Transfer compressed data using existing pinned staging and CUDA streams.
5. Use an existing GPU decompressor where possible.
6. Restore BF16 values directly on the GPU.
7. Preserve proper stream/event synchronization and buffer ownership.
8. Keep the original uncompressed path available.

**Critical distinction:** decompressing on the CPU and then transferring the entire BF16 tensor does not reduce PCIe traffic. It is a valid SSD-I/O baseline, but not a successful compressed-H2D implementation.

Compare at least:

- existing uncompressed native streaming;
- CPU decompression + full BF16 H2D;
- GPU decompression + GPU reconstruction;
- alternative codec/layout if the Phase 5C Zstd format is unsuitable.

Account for compression-frame overhead, cache residency, decompression buffers and temporary GPU allocations.

Validate GPU reconstruction against the original checkpoint bytes before running inference.

**Gate:** compressed GPU streaming must produce useful end-to-end savings after all decompression, reconstruction and memory overhead. Reject it if the decoder costs more than the transferred bytes save.

---

# Stage 6B3 — GPU cache and native transfer issuing

Once the compressed pipeline is measured, evaluate two independent optimizations.

### A. GPU expert cache

Consider a small VRAM tier ahead of the host cache.

Reuse the established routing trace and caching research.

Requirements:

- strict VRAM budget;
- no unnecessary duplication;
- safe handling of in-flight expert execution;
- deterministic eviction baseline;
- separate accounting for compressed and reconstructed tensors.

Do not assume that caching on the GPU pays. Earlier hotness-based caching had limited benefits.

Benchmark its marginal value with identical host-cache budgets.

### B. Native CUDA copy submission

If profiling still shows meaningful Python overhead in `PageStreamer._transfer_native`, replace its high-frequency copy submission loop through a narrowly scoped C++/CUDA component.

Do not move the whole execution engine to C++.

Preserve asynchronous H2D transfers, stream dependencies, pinned staging lifetimes and safe slot reuse.

Measure the actual reduction in host overhead.

---

# Stage 6B4 — Reduce transformer launch overhead

Phase 6A measured approximately 5,195 kernel launches per decode token.

Investigate opportunities to reduce them using:

1. Existing PyTorch CUDA Graph facilities.
2. Static-shape graph capture of suitable subgraphs.
3. Graph bucketing where necessary.
4. Existing fused PyTorch/CUDA operations.
5. Small, targeted CUDA kernels only when justified.

**Do not attempt to capture the entire dynamic MoE execution blindly.**

Routing, streaming, dynamic expert selection, memory allocation and host synchronization complicate CUDA Graph capture.

Start with an isolated, graph-safe decode component.

Ensure:

- fixed buffer addresses during replay;
- correct input updates between tokens;
- no stale activations or KV-cache state;
- no hidden CPU-side behavior skipped by capture;
- bounded graph memory;
- correct stream synchronization.

Graph replay must reproduce the existing reference bit for bit.

If graph capture changes BF16 results, document the incompatibility and retain eager execution. Do not weaken reference parity.

Advance to larger graph regions only if the smaller experiment is beneficial.

---

## 3. Preserve exactness

This phase is an optimization of the current runtime, **not a numerical reinterpretation of the model**.

Do not change:

- BF16 rounding rules;
- GEMM algorithms or batching where they affect outputs;
- expert routing;
- expert-combination order;
- final normalization;
- attention/KV-cache semantics;
- reference logits or token selection.

Required validations:

- original versus GPU-reconstructed BF16 bytes;
- independent reference token parity;
- bitwise logits parity;
- expert and router parity;
- KV-cache parity;
- identical outputs across cache hits, misses and evictions;
- repeated decode and prefill correctness;
- no uncounted disk reads or transfers;
- no stale asynchronous buffers;
- safe teardown/reinitialization.

Use existing Phase 4B/6A reference-validation infrastructure.

A faster but numerically different implementation must be labelled experimental and cannot replace the default `BF16_REFERENCE`.

---

## 4. Performance evaluation

Use the exact Phase 6A baseline conditions:

- Moonlight-16B-A3B, pinned reference revision.
- RTX 4060 Ti 8 GB.
- 6 GB GPU process cap.
- Existing 32 GB host-RAM machine.
- Identical prompt sets and decode steps.
- Identical cache-budget comparisons.
- Cold and warm scenarios.

Record:

| Metric | Phase 6A baseline | Phase 6B |
| --- | --- | --- |
| Decode latency | 807 ms/token | measured |
| Tokens/sec | 1.24 | measured |
| SSD bytes/token | ~0.758 GB (warm, 12 GB cache) | measured |
| H2D bytes/token | ~2.70 GB | measured |
| GPU copy time | ~435 ms/token | measured |
| Decode/reconstruction time | — | measured |
| Cache submit overhead | ~100 ms/token | measured |
| Kernel launches | ~5,195/token | measured |
| Peak RAM and VRAM | recorded baseline | measured |
| Reference mismatches | 0 | must remain 0 |

Measure each stage independently and the combined configuration.

Keep comparisons fair: equal memory budgets, same warm-up, same source, same hardware and same reference.

Report physical rather than theoretical I/O savings.

Aspirational end-to-end target: **at least 20% less decode latency than Phase 6A**, while preserving exactness.

This is not a forced PASS criterion. Freeze practical acceptance gates before final benchmark runs.

If the best measured result is smaller, report it accurately.

---

## 5. Efficiency and stop conditions

Do not build everything simultaneously.

Follow:

```text
Fix cache overhead
        ↓
Benchmark
        ↓
Compressed GPU pipeline feasibility
        ↓
Benchmark and correctness gate
        ↓
GPU cache / native transfer issuing
        ↓
Benchmark
        ↓
CUDA Graph feasibility
        ↓
Final end-to-end comparison
```

Stop or skip an individual optimization if:

- it cannot preserve exact reference behavior;
- an existing library is unsuitable and replacing it would require a major new engine;
- GPU memory overhead exceeds the available budget;
- its isolated benefit disappears end to end;
- complexity substantially exceeds the likely performance return.

A failed compressed-streaming experiment should not prevent an independently useful cache or CUDA Graph optimization.

No complete transformer rewrite.

No new training, quantization or approximation scheme.

---

## 6. Phase 6C preparation

While profiling, identify which operations prevent a clean, explicit BF16 reference.

Document:

- kernel selection dependent on tensor shape;
- accumulation and reduction ordering;
- places where PyTorch changes numerical execution;
- opportunities for an explicitly reproducible native reference.

**Do not implement Phase 6C.**

A custom kernel that changes numerical results must not silently become the reference.

We will investigate numerical fidelity separately before resuming AWPMI research.

---

## 7. Finalization

Deliver:

- tested Rust cache improvements;
- any successful C++/CUDA integration;
- optional compressed streaming backend;
- reproducible benchmarks and profiling;
- documentation of reused OSS;
- Phase 6B report and next architecture decision;
- updated `.context/state/current.md`.

Run:

- Rust tests, formatting and Clippy;
- CUDA/C++ tests where applicable;
- existing Python root tests;
- Phase 5C exact-codec tests;
- full Phase 4B/6A reference parity;
- cross-configuration determinism checks;
- `syngraphe check`.

Repeat accepted benchmark results on the same source and native build with different hash seeds. Preserve digests and report measurement variance.

Do not commit checkpoint weights or large generated artifacts.

Commit and push the completed phase to:

`weightsift-and-AWPMI-implementation`

Do not begin Phase 6C automatically.

---

## Final principle

**The objective of Phase 6B is to move fewer bytes, execute fewer unnecessary operations and preserve the exact reference.**

Rust handles the storage hierarchy and orchestration.

C++/CUDA handles justified GPU-critical operations.

Python remains the research and integration layer.

Reuse mature implementations wherever possible, prove correctness at each stage, and make every additional optimization earn its complexity through measured end-to-end performance.

The result should be a faster and more mature native runtime that we can later use to investigate BF16 numerical fidelity and resume certified adaptive weight materialization.