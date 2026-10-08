# Weightsift — Phase 6A: Native Runtime Foundation

Repository: https://github.com/suffro/weightsift

Branch: `weightsift-and-AWPMI-implementation`

Start from the current HEAD.

## Mission

Begin migrating Weightsift's performance-critical runtime infrastructure from Python to **Rust**, while preserving its existing Python research layer and PyTorch/CUDA execution backend.

This is the first milestone of a broader native-runtime transition:

- **6A:** Rust I/O, caching, prefetching and scheduling.
- **6B:** C++/CUDA execution optimizations, fused weight reconstruction, CUDA Graphs.
- **6C:** BF16 numerical fidelity and tighter reference-aware certification.
- **Phase 7:** Resume AWPMI research using the optimized runtime.

**Implement Phase 6A only.** Design clean interfaces for 6B/6C, but do not prematurely implement them.

The objectives are measurable performance improvements, exact reference parity, minimal duplicated code and a reusable native foundation.

---

## 1. Inspect the repository first

Read and follow:

- `.context/state/Weightsift_Implementation_Guide.md`
- `.context/state/current.md`
- `.context/truth/architecture.md`
- `.context/truth/conventions.md`
- Phase 3, 4A, 4B and 5C reports and decisions.

Inspect the existing:

- `src/awpmi/storage/`
- `src/awpmi/streaming/`
- `src/awpmi/materialization/`
- `src/awpmi/models/moe.py`
- `src/awpmi/runtime.py`
- Phase 4B benchmarks and reference-parity tests.

Identify the actual performance-critical Python paths before changing the architecture.

Preserve existing behavior and public interfaces wherever possible.

## 2. Mandatory open-source reuse investigation

Before implementing Rust functionality, examine:

- https://github.com/antirez/ds4
- https://github.com/jenovauh/colibri-LLM
- https://github.com/ggml-org/llama.cpp
- relevant mature Rust crates for file I/O, bounded caching, async execution and Python bindings.

Focus specifically on:

- SSD/NVMe expert streaming;
- host-RAM expert caching;
- prefetch and routing-aware scheduling;
- request batching and coalescing;
- asynchronous read pipelines;
- memory placement;
- buffer reuse;
- CPU/GPU transfer coordination.

**Do not reinvent mature implementations.**

Reuse compatible upstream components where practical, and otherwise reuse proven architectural patterns.

Check licenses, actual compatibility and maintenance status before incorporating source code.

Produce a brief reuse-vs-build decision explaining what was reused and what must remain Weightsift-specific.

Do not vendor an entire inference engine merely to obtain its cache.

---

## 3. Native architecture

Implement a small Rust core, preferably as an isolated workspace/crate integrated through **PyO3 + maturin** or an equally justified maintained binding solution.

Conceptual architecture:

```text
Python / PyTorch
    │
    ▼
Weightsift Native Backend (Rust)
    ├── I/O planner
    ├── Async/batched reader
    ├── Host-RAM expert cache
    ├── Prefetch coordinator
    ├── Storage metrics
    └── Buffer management
    │
    ▼
Existing PyTorch/CUDA execution
```

Requirements:

- Model-agnostic native core.
- No hardcoded Moonlight tensor names or expert counts.
- No model-specific routing logic in Rust.
- No GIL held during blocking I/O or long native operations.
- Bounded host memory and cache eviction.
- Concurrent request deduplication where beneficial.
- Batch/coalesced disk reads where measurable.
- Cancellation and safe shutdown.
- Robust error propagation into Python.
- Existing Python backend remains available as fallback.

Keep the FFI boundary coarse-grained. Avoid one Python↔Rust call per small page if requests can be batched.

Use appropriate platform-specific I/O only when measured and provide a portable fallback.

Do not introduce unsafe memory handling without a demonstrated requirement.

---

## 4. Implement incrementally

### Stage A — Baseline and profiling

Before writing Rust code:

1. Reproduce the Phase 4B Moonlight benchmark.
2. Record token-by-token latency and profiling.
3. Identify I/O, cache, scheduling, transfer and Python overhead.
4. Identify how many expert bytes are repeatedly read.
5. Save reproducible baseline measurements.

Use the existing pinned model revision, prompts, hardware and memory restrictions.

If the historical environment is unavailable, document that and establish a clearly labelled comparable baseline.

### Stage B — Native I/O

Replace the highest-overhead Python storage path with Rust.

Implement only what profiling justifies:

- positioned file reads;
- batched asynchronous requests;
- read coalescing;
- bounded worker execution;
- reusable host buffers;
- correct file/page offsets;
- native byte accounting.

Integrate with existing safetensors/checkpoint indexes.

Do not build another checkpoint parser if the current index already supplies the required offsets.

Verify that the loaded bytes are identical to the Python backend.

Benchmark Rust I/O independently before integrating it into inference.

### Stage C — Native host-RAM cache

Introduce a configurable expert/page cache in Rust.

Requirements:

- strict memory budget;
- deterministic replacement baseline, such as LRU;
- thread-safe lookups;
- deduplicated in-flight loads;
- correct eviction;
- hit/miss and eviction metrics;
- support for different expert sizes;
- no duplicate unaccounted memory residency.

Test multiple cache budgets using Phase 4B routing traces.

Avoid complex predictive policies until the simple cache is measured.

### Stage D — Prefetch and integration

Add conservative asynchronous prefetching.

Prefetch exact selected experts once routing makes their identities available. Speculative prefetch may be evaluated separately, but every speculative read must be counted and must not affect correctness.

Integrate the native backend into the existing Moonlight inference path through an explicit backend choice, for example:

`python` / `native`

Retain existing PyTorch kernels, expert execution and reference semantics.

Do not silently alter:

- BF16 arithmetic;
- router behavior;
- expert combination;
- KV cache;
- accumulation order;
- logits or token selection.

---

## 5. Performance gates

Measure each native change separately and end-to-end.

Required comparisons:

| Metric | Python baseline | Native Rust |
| --- | --- | --- |
| Decode latency (ms/token) | measured | measured |
| Tokens/sec | measured | measured |
| Physical SSD bytes/token | measured | measured |
| I/O throughput | measured | measured |
| Host-cache hit rate | measured | measured |
| I/O requests and read amplification | measured | measured |
| Python/FFI overhead | measured | measured |
| Peak host RAM | measured | measured |
| Peak VRAM | measured | measured |
| GPU idle/transfer time | measured | measured |

Use cold and warm cache scenarios. Control OS page-cache effects as far as practical and document what was actually controlled.

The native backend must deliver a **measurable end-to-end benefit**, not merely faster isolated microbenchmarks.

Target at least a 20% reduction in decode latency as an aspirational milestone, not a requirement to manufacture a positive result.

If a native component adds complexity without useful improvement, retain the simpler implementation.

---

## 6. Correctness is non-negotiable

Use the existing `BF16_REFERENCE`.

The native backend changes **where and how bytes move**, not what the model computes.

Required tests:

- loaded weight bytes identical;
- router outputs identical;
- expert outputs identical;
- logits and generated tokens bit-identical;
- KV cache identical;
- identical results across cache hit/miss and eviction scenarios;
- no hidden or unauthorized weight reads;
- memory limits respected;
- no races, double loads or corrupted buffers.

Run the existing Phase 4B reference-parity tests through both backends.

Use Rust unit tests plus Python integration tests.

Do not weaken previous tests or tolerances.

The native backend must remain optional so the Python reference continues to work independently.

---

## 7. BF16 investigation: prepare, do not solve prematurely

Phase 6A must collect useful information for Phase 6C.

Identify and document:

- which PyTorch/CUDA kernels define the current reference;
- their relevant BF16 and accumulation behavior;
- where the certificate's rounding uncertainty originates;
- which operations might benefit from explicit native numerical semantics.

Do not introduce new BF16 kernels or claim a rounding fix in Phase 6A.

Changing execution semantics creates a different reference profile unless exact equivalence is proven.

The existing 8.1% certificate coverage is not a target for this phase.

---

## 8. Prepare for Phase 6B

Keep the architecture compatible with later:

- C++/CUDA kernels;
- GPU-side bit-plane reconstruction;
- pinned staging and asynchronous H2D transfers;
- CUDA Graphs;
- fused decoding and materialization.

Phase 5C showed that independent lossless expert compression can reduce SSD traffic significantly, but bit-plane restoration is currently slow.

Document a clean integration point for a future CUDA decoder.

Do not implement it now unless a tiny isolated feasibility experiment is essential to an architectural decision.

---

## 9. Validation and completion

Run:

- Rust formatting, linting and tests;
- native Python-binding integration tests;
- existing root Python tests;
- relevant Moonlight parity tests;
- end-to-end benchmarks;
- `syngraphe check`.

Keep generated artifacts and model files out of Git.

Document build instructions, Rust toolchain, dependencies, supported platforms and fallback behavior.

Create:

- a Phase 6A technical report;
- reproducible benchmark results;
- a new architecture decision;
- updates to `.context/state/current.md` and architectural documentation where appropriate.

Commit and push to:

`weightsift-and-AWPMI-implementation`

Do not start Phase 6B or 6C automatically.

---

## Final priorities

1. Reuse existing code and mature OSS.
2. Preserve bitwise correctness.
3. Reduce physical I/O and Python runtime overhead.
4. Introduce effective host-RAM caching.
5. Integrate and benchmark the native backend.
6. Keep the architecture simple, generic and extensible.

**The objective is not to rewrite Weightsift in Rust. It is to move the proven performance-critical components into native code, measure the real benefits, and create a solid foundation for CUDA optimizations and future AWPMI research.**

If profiling contradicts the proposed implementation order, follow the evidence and explain why.

Stop at a clean, tested Phase 6A checkpoint.