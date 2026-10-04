# AWPMI — Implementation Roadmap

## Purpose

This document defines the implementation plan for a first research-grade version of **AWPMI — Adaptive Weight-Page Materialization for Inference**.

The objective is to validate the central research hypothesis as efficiently and rigorously as possible:

> Progressively materialize independently fetchable weight pages and stop only when a conservative certificate proves that the discrete decision is identical to the fully materialized reference model.

The implementation must be **research-first, failure-driven, modular, reproducible, and correctness-preserving**.

The plan is divided into four phases. Each phase has explicit acceptance criteria and must be completed before advancing.

---

# 0. Core principles

These principles are non-negotiable.

1. **Correctness before performance.**
2. **Confidence is never a certificate.**
3. **Every certified output must be compared against the full reference model during development and benchmarking.**
4. **Fallback must always be possible:** if certification fails, materialize everything required and reproduce the reference computation.
5. **Learned scheduling must not be introduced before deterministic scheduling works.**
6. **Quantization must not be introduced before the BF16/FP32 path is validated.**
7. **Triton/custom CUDA kernels must not be written before profiling proves they are necessary.**
8. **Mathematical bounds must be isolated, documented, unit-tested, and independently validated.**
9. **Storage, streaming, scheduling, partial execution, and certification must remain separate modules.**
10. **Model revision and dependency versions must be pinned.**
11. **Every benchmark must save raw results, configuration, and environment metadata.**
12. **The reference model must never be modified to make AWPMI appear correct.**
13. **The learned scheduler, if introduced later, may affect efficiency only, never correctness.**
14. **Do not optimize a subsystem before the scientific hypothesis it supports has passed its gate.**

---

# 1. Initial target

Use:

- **Model:** `HuggingFaceTB/SmolLM2-135M-Instruct`
- **Framework:** PyTorch
- **Model integration:** Hugging Face Transformers
- **Weights:** safetensors
- **Environment:** Python 3.11+
- **Dependency management:** `uv`
- **Initial dtype:** BF16 if supported, otherwise FP32
- **Batch size:** 1
- **Decoding:** greedy
- **Sampling:** disabled
- **Temperature:** 0
- **Quantization:** disabled
- **Learned scheduling:** disabled
- **Custom kernels:** disabled

The first milestone is not production inference.

The first milestone is:

> Demonstrate that the reference next-token argmax can be certified before all relevant weight pages are materialized on at least a meaningful subset of inputs.

---

# 2. Target architecture

```text
                         SmolLM2
                            │
                    exact reference
                            │
             ┌──────────────┴──────────────┐
             │                             │
       AWPMI packer                  ReferenceRunner
             │                             │
      weight-page index              full-model logits
             │                             │
             ▼                             │
        PageStore                         │
      RAM / NVMe                          │
             │                             │
             ▼                             │
       PageStreamer                       │
    RAM/NVMe → GPU                        │
             │                             │
             ▼                             │
       PartialExecutor                    │
             │                             │
             ▼                             │
       ResidualState                      │
             │                             │
             ▼                             │
        Certificate ──────────────────────┘
          │       │
        YES       NO
          │       │
        STOP      ▼
               Scheduler
                  │
               next page
```

The architecture must preserve one key separation:

```text
Scheduler
    │
    │ decides which page is likely most useful
    ▼
Materialization
    │
    ▼
Partial computation
    │
    ▼
Certificate
    │
    ├── PROVEN → stop
    └── UNKNOWN → continue
```

The scheduler may be heuristic or learned.

The certificate must be conservative and deterministic.

---

# PHASE 1 — Minimal certified materialization

## Objective

Build the smallest possible implementation that can answer:

> Can AWPMI certify the full-model next-token decision before all weight pages of the progressively refined region have been materialized?

Do not implement real disk streaming yet.

All model weights may remain physically resident in memory during this phase.

The goal is to validate the algorithmic core.

---

## 1.1 Repository bootstrap

Recommended structure:

```text
awpmi/
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   └── smollm2-135m.yaml
├── src/awpmi/
│   ├── reference.py
│   ├── state.py
│   ├── certificate.py
│   ├── tracing.py
│   │
│   ├── paging/
│   │   ├── page.py
│   │   ├── index.py
│   │   └── metadata.py
│   │
│   ├── bounds/
│   │   ├── linear.py
│   │   └── residual.py
│   │
│   ├── schedulers/
│   │   ├── sequential.py
│   │   └── bound.py
│   │
│   └── models/
│       └── smollm2.py
│
├── tests/
│   ├── test_reference.py
│   ├── test_pages.py
│   ├── test_bounds.py
│   ├── test_certificate.py
│   └── test_fallback.py
│
├── benchmarks/
│   ├── run.py
│   └── report.py
│
└── experiments/
    └── phase1/
```

---

## 1.2 Reproducible environment

Create:

```text
pyproject.toml
uv.lock
```

Pin at minimum:

```text
torch
transformers
safetensors
numpy
pytest
```

Optional:

```text
typer
rich
pydantic
```

Every benchmark must record:

```text
model repository
model revision / commit hash
torch version
transformers version
safetensors version
Python version
CUDA version
GPU model
dtype
OS
```

---

## 1.3 Reference runner

Implement:

```python
ReferenceRunner.next_token(input_ids)
```

Return:

```python
ReferenceResult(
    token_id,
    logits,
    hidden_state,
)
```

The reference definition is:

```python
token_id = torch.argmax(reference_logits, dim=-1)
```

Requirements:

- model in `eval()` mode;
- deterministic execution where possible;
- no sampling;
- no quantization;
- no altered architecture;
- reference output must be reusable by the test harness.

---

## 1.4 Start with the LM head

Do not begin by progressively materializing arbitrary pages across the full transformer.

Start from the final language-model projection.

For:

```text
z = W h
```

partition `W` by input columns:

```text
W = [W1, W2, ..., WP]
```

and the corresponding hidden vector:

```text
h = [h1, h2, ..., hP]
```

Then:

```text
z = Σp Wp hp
```

This gives an exact additive decomposition.

Recommended initial page widths:

```text
32
64
```

Make page width configurable.

---

## 1.5 Page representation

Implement:

```python
WeightPage(
    page_id,
    parameter_name,
    offset,
    shape,
    dtype,
    storage_bytes,
    metadata,
)
```

For Phase 1, the page may reference an in-memory tensor slice.

The rest of the code must not depend on that fact.

---

## 1.6 Conservative metadata

Precompute enough metadata to bound the contribution of a page without using its full values during refinement.

For each output row `j` and page `p`, compute:

```text
||W[j,p]||₂
```

Using Cauchy-Schwarz:

```text
|(Wp hp)[j]| ≤ ||W[j,p]||₂ · ||hp||₂
```

For the set of unmaterialized pages `U`:

```text
r[j] = Σp∈U ||W[j,p]||₂ · ||hp||₂
```

If the current partial logit is:

```text
z_hat[j]
```

then:

```text
z[j] ∈ [z_hat[j] - r[j], z_hat[j] + r[j]]
```

The implementation must not silently substitute a heuristic estimate for this bound.

---

## 1.7 Residual state

Introduce a central representation such as:

```python
ResidualState(
    partial_logits,
    lower_bounds,
    upper_bounds,
    materialized_pages,
    remaining_pages,
)
```

This object represents:

```text
known contribution
+
conservative uncertainty from missing pages
```

---

## 1.8 Certificate

Implement:

```python
Certificate.check(state)
```

Return:

```python
CertificateResult(
    certified,
    winner,
    certificate_margin,
    winner_lower_bound,
    competitor_upper_bound,
)
```

Let:

```text
w = argmax(partial_logits)
```

Certification succeeds only when:

```text
lower[w] > max(upper[j] for j != w)
```

If not provable, return:

```text
UNKNOWN
```

Never return:

```text
likely
probably
high confidence
```

as a substitute for certification.

---

## 1.9 Initial scheduler

Do not use a learned scheduler.

Implement deterministic baselines:

### Sequential

```text
lowest page_id first
```

### Largest residual contribution first

Example score:

```text
score(page) =
    ||h_page||₂ × max_row_norm(page)
```

### Bound reduction per byte

Example:

```text
score(page) =
    estimated_bound_reduction / storage_bytes
```

Use deterministic tie-breaking.

---

## 1.10 Fallback path

If every page has been materialized:

```text
AWPMI logits ≈ reference logits
```

within a defined numerical tolerance.

And:

```text
argmax(AWPMI logits) == argmax(reference logits)
```

must hold.

Failure here blocks the project.

---

## 1.11 Phase 1 benchmark

Run at minimum:

```text
hundreds of prompts
preferably ≥ 1,000 once stable
```

Record per input:

```json
{
  "reference_token": 0,
  "awpmi_token": 0,
  "certified": true,
  "pages_total": 0,
  "pages_materialized": 0,
  "materialized_fraction": 0.0,
  "certificate_margin": 0.0
}
```

Aggregate:

```text
certified mismatch count
fallback mismatch count
certificate coverage
mean materialized fraction
median materialized fraction
p90/p95 materialized fraction
distribution of certification points
```

---

## Phase 1 gate

Do not proceed until all are true:

```text
certified mismatches = 0
fallback mismatches = 0
full-materialization path reproduces reference
certificate triggers before the final page on at least some real inputs
results are reproducible
```

If certification occurs almost exclusively after ~100% materialization:

```text
DO NOT implement streaming yet.
```

Instead improve:

```text
bounds
page decomposition
page ordering
```

The first question is scientific, not systems-related.

---

# PHASE 2 — Extend certification into the transformer

## Objective

Move from:

```text
exact transformer
+
adaptive LM head
```

toward:

```text
exact prefix
+
progressively materialized suffix
+
certified final decision
```

This is the main research phase.

---

## 2.1 Incremental expansion only

Do not make the entire transformer adaptive immediately.

Recommended order:

```text
LM head
↓
final projection
↓
last MLP
↓
last transformer block
↓
last 2 blocks
↓
last N blocks
```

Each extension must independently pass all correctness tests.

---

## 2.2 Generalized residual representation

Extend `ResidualState` so it can track uncertainty at internal points.

Possible structure:

```python
ResidualState(
    nominal,
    lower,
    upper,
    provenance,
)
```

`provenance` should identify which missing pages contribute to the current uncertainty.

Avoid coupling this object to storage or streaming.

---

## 2.3 Supported operators

Implement verified propagation only for operators actually encountered in SmolLM2:

```text
Linear
Residual addition
RMSNorm
SiLU
Elementwise multiplication
RoPE
Attention
Softmax
```

Do not implement all operators in advance.

Implement only what the current adaptive suffix requires.

---

## 2.4 Bound engineering rule

For every new operator:

1. derive or source a conservative bound;
2. document the formula;
3. implement it separately;
4. add unit tests;
5. test on tiny random tensors;
6. compare against brute-force enumeration where feasible;
7. optionally compare against an independent verifier.

No unproven approximation may participate in `certified=True`.

---

## 2.5 Independent validation with auto_LiRPA

Use `auto_LiRPA` as an external research validator, not necessarily as the runtime engine.

Compare:

```text
AWPMI custom bound
vs
auto_LiRPA
vs
brute-force exact enumeration on tiny cases
```

Focus on:

```text
tiny linear blocks
small MLPs
single transformer components
```

The goal is to validate soundness and measure bound looseness.

---

## 2.6 Exact-prefix strategy

Introduce:

```text
layers [0, N)      = exact
layers [N, end)    = adaptive
```

Move `N` progressively earlier.

For every `N`, measure:

```text
certificate rate
materialized fraction
bound width
latency
fallback rate
```

This produces an important research curve:

```text
adaptive suffix depth
vs
certification efficiency
```

---

## 2.7 Page indexing across layers

Generalize:

```python
WeightPage(
    page_id,
    parameter_name,
    layer,
    tensor_role,
    offset,
    shape,
    dtype,
    storage_bytes,
    bound_metadata,
)
```

Possible tensor roles:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
lm_head
```

The index must expose:

```text
location
size
layer
dependency
cost
bound metadata
materialization state
```

---

## 2.8 KV-cache correctness

Do not assume:

```text
same generated token
=
same KV cache
```

If hidden states were only bounded/approximated, the KV cache may differ even when the token is certified identical.

For initial multi-token support choose one of:

### Mode A — Exact transformer state

Only make the LM head adaptive.

Result:

```text
KV cache remains reference-exact
```

### Mode B — Exact recomputation

After certifying a token:

```text
recompute required internal state exactly
```

before storing reusable KV.

Do not claim exactness under KV reuse until this is explicitly validated.

---

## Phase 2 gate

Proceed only when:

```text
certified mismatches = 0
fallback mismatches = 0
adaptive region extends beyond LM head
all participating bounds are independently tested
bound propagation remains conservative
materialization curves are recorded
```

---

# PHASE 3 — Real selective storage and streaming

## Objective

Convert logical non-materialization into physical non-materialization.

A page that has not been selected should not be read from its backing storage.

---

## 3.1 Storage abstraction

Implement:

```python
class PageStore:
    def get(self, page_id):
        ...
```

Backends:

```text
InMemoryPageStore
SafetensorsPageStore
PreadPageStore
```

The certificate, scheduler, and executor must not know where pages are stored.

---

## 3.2 AWPMI packer

Create a command:

```bash
awpmi pack ...
```

Output:

```text
packed weights
page index
bound metadata
manifest
source model hash
packing configuration
```

Manifest should contain enough information to verify:

```text
packed model == expected reference model
```

Suggested fields:

```text
source repository
source revision
tensor names
tensor hashes
page size
dtype
packing strategy
metadata version
AWPMI format version
```

---

## 3.3 Safetensors first

Prefer existing safetensors slicing/range-access mechanisms where practical.

Do not invent a new storage format unless profiling or correctness requirements justify it.

The packer may initially preserve the original safetensors organization plus an AWPMI page index.

Only introduce a custom layout after measuring the cost of the simpler design.

---

## 3.4 Page streamer

Implement:

```python
PageStreamer
```

Responsibilities:

```text
request page
load into host buffer
optionally use pinned memory
copy asynchronously to GPU
reuse preallocated device buffers
track transfer completion
```

Keep this separate from the page store.

---

## 3.5 Reuse Soup selectively

Study and reuse/adapt architectural patterns from Soup for:

```text
pinned host buffers
preallocated GPU buffers
double buffering
CUDA copy stream
async H2D
event synchronization
prefetch
```

Do not make AWPMI dependent on Soup's training architecture.

Preferred approach:

```text
extract/adapt proven streaming primitives
```

instead of:

```text
implement AWPMI as a Soup-specific feature
```

AWPMI must retain its own clean runtime abstractions.

---

## 3.6 Runtime pipeline

Target:

```text
Certificate = UNKNOWN
        │
        ▼
Scheduler chooses next page
        │
        ▼
PageStore.get(page)
        │
        ▼
host buffer
        │
        ▼
async H2D
        │
        ▼
GPU page buffer
        │
        ▼
partial computation
        │
        ▼
ResidualState update
        │
        ▼
Certificate.check()
```

---

## 3.7 Prefetching

Only implement after the synchronous version is correct.

Initial strategy:

```text
while computing page N
prefetch likely page N+1
```

But:

```text
certificate may succeed before prefetched page is consumed
```

Therefore prefetch must not force logical materialization.

Track separately:

```text
requested bytes
prefetched bytes
consumed bytes
wasted prefetch bytes
```

---

## 3.8 Telemetry

Every inference should emit raw structured traces.

Example:

```json
{
  "reference_token": 421,
  "awpmi_token": 421,
  "certified": true,
  "pages_total": 18,
  "pages_materialized": 7,
  "bytes_total": 58000000,
  "bytes_read": 23000000,
  "materialized_fraction": 0.397,
  "certificate_margin": 0.017,
  "storage_io_ms": 3.1,
  "h2d_ms": 1.2,
  "compute_ms": 5.8,
  "certificate_ms": 0.4,
  "scheduler_ms": 0.1
}
```

Separate timing for:

```text
storage I/O
host-to-device transfer
compute
certificate
scheduler
```

Do not report only end-to-end latency.

---

## 3.9 Baselines

At minimum compare:

### Baseline A

```text
fully resident Hugging Face reference
```

### Baseline B

```text
sequential full streaming
```

### AWPMI

```text
adaptive certified materialization
```

Optional later baselines:

```text
DeepSpeed ZeRO-Inference
Soup-style layer streaming
```

---

## Phase 3 gate

Proceed only when:

```text
non-selected pages are not physically read
bytes read can be lower than full adaptive-region size
certified mismatches = 0
fallback mismatches = 0
peak VRAM is measured
I/O is measured
H2D is measured
certificate overhead is measured
```

---

# PHASE 4 — Scheduling, optimization, evaluation, and production hardening

## Objective

Once correctness and real selective loading work, optimize how quickly the certificate is reached.

Correctness must remain independent of optimization.

---

## 4.1 Deterministic scheduler baselines

Benchmark:

```text
sequential
random seeded
largest residual first
bound-reduction-per-byte
activation-aware
```

Track:

```text
pages to certificate
bytes to certificate
latency to certificate
certificate success rate
```

---

## 4.2 Learned scheduler

Only after sufficient execution traces exist.

A Jev-like / DecisionCore-style model may rank remaining pages.

Potential state features:

```text
partial top-1/top-2 margin
certificate margin
activation norms
remaining page bounds
page IDs
layer IDs
tensor roles
page sizes
previous refinement history
estimated I/O cost
```

Actions:

```text
remaining candidate pages
```

Possible training target:

```text
page that maximally reduces uncertainty per byte
```

or:

```text
page that minimizes remaining pages until certification
```

Architecture:

```text
current AWPMI state
        │
        ▼
learned scheduler
        │
        ▼
page ranking
        │
        ▼
materialize selected page
        │
        ▼
deterministic certificate
```

Important:

```text
the learned model must never emit certified=True
```

Its failure mode should be:

```text
load more pages
```

not:

```text
wrong decision
```

---

## 4.3 DecGuard integration

Use DecGuard as a reliability and regression layer.

AWPMI run output should expose enough information for:

```text
decision parity
certificate coverage
materialized fraction
fallback rate
latency
errors
```

Hard gates:

```text
certified mismatch count = 0
fallback mismatch count = 0
decision parity = 1.0
```

Optional warning gates:

```text
mean materialized fraction
p95 materialized fraction
p95 latency
minimum certificate coverage
maximum fallback rate
```

DecGuard is not the mathematical certificate.

It validates the implementation and its behavior across datasets and versions.

---

## 4.4 lm-evaluation-harness

Add standard downstream evaluation only after next-token correctness is stable.

Use it to verify that:

```text
AWPMI outputs remain decision-equivalent
```

across representative tasks.

Do not use downstream score similarity as a substitute for exact decision parity.

---

## 4.5 Quantization

Only after the BF16/FP32 implementation is validated.

Possible tools:

```text
TorchAO
bitsandbytes
```

Recommended order:

```text
full BF16 reference
vs
AWPMI BF16
```

then:

```text
full quantized reference
vs
AWPMI using exactly the same quantized representation
```

Do not compare:

```text
AWPMI INT4
vs
BF16 reference
```

and call the result decision-exact.

If quantization changes the reference model, it defines a different reference.

---

## 4.6 Triton / custom kernels

Do not implement custom kernels until profiling shows a real bottleneck.

Correct order:

```text
correctness
↓
benchmarking
↓
profiling
↓
identify bottleneck
↓
optimize
```

Candidate optimization areas:

```text
partial GEMM
page accumulation
bound updates
page metadata reductions
H2D overlap
```

Use Triton only where measurable benefit exists.

---

## 4.7 DwarfStar

Do not integrate DwarfStar into the initial dense SmolLM2 runtime.

Study it later for:

```text
SSD cache design
expert caching
eviction policy
prefetch
MoE models
```

Potential later architecture:

```text
MoE router
    │
    ▼
DwarfStar-like expert materialization
    │
    ▼
AWPMI refinement inside selected experts
    │
    ▼
certificate
```

This is a later research direction, not an initial dependency.

**Status (2026-10-03).** Two milestones have realized the "expert materialization" box of this
diagram, without the AWPMI refinement inside experts:

- Phase 3 (decision 0006): an expert cache and router-driven reads.
- Phase 4A (decision 0007), a milestone the user added outside the four phases: a model whose
  experts exceed the GPU (OLMoE-1B-7B), with compact expert calls and split checkpoints read in
  place, bit for bit equal to the fully materialized reference.
- Phase 4B (decision 0008), also added by the user: DeepSeek-V3's architecture (Moonlight-16B-A3B)
  out of both VRAM and host RAM, with expert calls bounded by a byte budget (chunks of experts,
  the implementation's own combine) and an independent streaming reference. Decision 0008 names
  the hook where AWPMI refinement inside selected experts would go.

**Status (2026-10-04).** Phase 5A (decision 0009), added by the user, measured the "AWPMI
refinement inside selected experts" box as an oracle, on Moonlight's last MoE layer: under the
certified rounding model no token certifies with any routed byte unread (8.1% certify even with
every byte read), so the box is not built. See `history/2026-10-04-awpmi-phase5a-report.md`.

The scheduling, optimization and evaluation items of this Phase 4 are not started.

---

# 5. Final repository structure

Recommended mature layout:

```text
awpmi/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE
│
├── configs/
│   ├── smollm2-135m.yaml
│   └── benchmarks/
│
├── src/awpmi/
│   ├── reference.py
│   ├── state.py
│   ├── certificate.py
│   ├── executor.py
│   ├── tracing.py
│   │
│   ├── paging/
│   │   ├── page.py
│   │   ├── index.py
│   │   ├── packer.py
│   │   └── metadata.py
│   │
│   ├── stores/
│   │   ├── base.py
│   │   ├── memory.py
│   │   ├── safetensors.py
│   │   └── pread.py
│   │
│   ├── streaming/
│   │   ├── streamer.py
│   │   ├── buffers.py
│   │   └── cuda.py
│   │
│   ├── bounds/
│   │   ├── linear.py
│   │   ├── residual.py
│   │   ├── norm.py
│   │   ├── nonlinear.py
│   │   └── attention.py
│   │
│   ├── schedulers/
│   │   ├── base.py
│   │   ├── sequential.py
│   │   ├── bound.py
│   │   ├── activation.py
│   │   └── learned.py
│   │
│   ├── models/
│   │   ├── base.py
│   │   └── smollm2.py
│   │
│   └── cli/
│       ├── pack.py
│       ├── infer.py
│       └── benchmark.py
│
├── tests/
│   ├── unit/
│   │   ├── test_pages.py
│   │   ├── test_bounds.py
│   │   ├── test_certificate.py
│   │   └── test_stores.py
│   │
│   ├── integration/
│   │   ├── test_reference_parity.py
│   │   ├── test_fallback.py
│   │   ├── test_streaming.py
│   │   └── test_multitoken.py
│   │
│   └── property/
│       └── test_bound_soundness.py
│
├── benchmarks/
│   ├── run.py
│   ├── datasets.py
│   ├── metrics.py
│   └── report.py
│
├── experiments/
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   └── phase4/
│
└── docs/
    ├── architecture.md
    ├── certificate.md
    ├── paging.md
    └── experiments.md
```

---

# 6. Required tests

The following tests are mandatory.

## Reference parity

```text
full-materialization AWPMI logits ≈ reference logits
```

## Decision parity

```text
certified AWPMI token == reference token
```

## Fallback parity

```text
fallback AWPMI token == reference token
```

## Bound soundness

For tiny systems:

```text
true value ∈ [lower, upper]
```

for every tested sample.

## Storage correctness

```text
loaded page bytes == expected tensor slice
```

## No hidden reads

During physical selective loading:

```text
unselected pages must not be fetched by the PageStore
```

## Determinism

With identical:

```text
model
revision
input
configuration
seed
```

results must be reproducible.

---

# 7. Research metrics

Do not optimize only for latency.

Primary metrics:

```text
certified mismatch count
fallback mismatch count
certificate coverage
materialized weight fraction
bytes read fraction
pages materialized
certificate margin
```

Systems metrics:

```text
peak VRAM
peak RAM
storage bytes read
H2D bytes
TTFT
per-token latency
I/O time
compute time
certificate time
scheduler time
wasted prefetch
```

Research curves:

```text
page size
vs
materialized fraction

adaptive suffix depth
vs
certificate rate

bound tightness
vs
certificate rate

scheduler strategy
vs
bytes-to-certificate

model/input difficulty
vs
materialization fraction
```

---

# 8. Failure criteria

The implementation should fail explicitly rather than hide bad results.

Important failure modes:

## Certificate never fires early

Interpretation:

```text
bounds are too loose
or
page decomposition is ineffective
or
the hypothesis is weak for this region/model
```

Action:

```text
improve bound/page design
DO NOT optimize streaming
```

## Certified mismatch occurs

Interpretation:

```text
certificate is unsound
or
implementation is incorrect
```

Action:

```text
stop immediately
minimize failing case
fix before continuing
```

## Full materialization differs from reference

Interpretation:

```text
partial executor is not semantically equivalent
```

Action:

```text
stop
fix executor
```

## Real streaming is slower than full loading

Interpretation may be:

```text
pages too small
I/O overhead dominates
certificate overhead too high
poor prefetch
insufficient compute/I/O overlap
```

This is a systems optimization problem only if the algorithmic materialization savings are already real.

---

# 9. Development sequence

Codex should follow this order strictly.

```text
PHASE 1
LM-head progressive materialization
+
sound certificate
+
deterministic scheduler
        │
        ▼
Does certification happen early?
        │
   ┌────┴────┐
   │         │
  NO        YES
   │         │
Improve      ▼
bounds      PHASE 2
/page       adaptive suffix
design      verified bounds
              │
              ▼
            PHASE 3
        real selective I/O
        Soup-inspired streaming
              │
              ▼
            PHASE 4
        learned scheduler
        quantization
        profiling
        Triton
        DecGuard
        broader benchmarks
```

---

# 10. Codex execution policy

Codex should receive the full roadmap, but implement **one phase at a time**.

For each phase:

1. inspect the existing repository;
2. state the implementation plan;
3. implement the smallest correct version;
4. add tests;
5. run tests;
6. run a small experiment;
7. save raw results;
8. summarize results;
9. explicitly evaluate the phase gate;
10. stop if the gate fails.

Do not continue automatically into later phases after a failed gate.

The project should prefer:

```text
small validated increments
```

over:

```text
large speculative implementation
```

---

# 11. Open-source components to reuse

Use existing tooling wherever it reduces implementation effort without obscuring correctness.

## Core model stack

```text
PyTorch
Hugging Face Transformers
safetensors
```

## Streaming / low-memory patterns

```text
Soup
```

Use primarily for:

```text
pinned host buffers
double buffering
CUDA transfer streams
preallocated buffers
prefetch
```

## Formal-bound validation

```text
auto_LiRPA
```

Use as independent research validation where helpful.

## Evaluation

```text
lm-evaluation-harness
DecGuard
```

## Quantization

Later only:

```text
TorchAO
bitsandbytes
```

## Learned scheduling

Later only:

```text
Jev-like / DecisionCore-style controller
```

## MoE / SSD research reference

Later only:

```text
DwarfStar
```

---

# 12. Definition of the first successful AWPMI prototype

The first version is considered scientifically successful when all of the following hold:

```text
1. A real pretrained transformer is used.

2. The reference model is fixed and reproducible.

3. The adaptive region is split into independently materializable pages.

4. Missing pages are represented by conservative residual bounds.

5. AWPMI can return UNKNOWN and continue refinement.

6. AWPMI can return CERTIFIED only from a sound certificate.

7. Every certified token matches the fully materialized reference.

8. Full fallback reproduces the reference.

9. At least some real inputs certify before all adaptive-region pages are materialized.

10. The materialized fraction is measurable.

11. In the physical-I/O implementation, skipped pages are genuinely not read.

12. Results and configurations are reproducible.
```

The first prototype does **not** need:

```text
production serving
multi-GPU
arbitrary model support
quantization
custom CUDA
learned scheduling
MoE
high throughput
```

Those are later engineering layers.

---

# 13. Central research question

Every engineering decision should remain subordinate to this question:

> Can a real transformer produce the exact same discrete next-token decision as its fully materialized reference while progressively materializing only the weight pages needed to conservatively certify that decision?

If the answer is yes, optimize.

If the answer is no with the current bounds, improve the mathematical representation.

Do not let infrastructure obscure the research question.
