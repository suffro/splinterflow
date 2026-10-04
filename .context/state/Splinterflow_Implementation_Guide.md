# Splinterflow — Development Guide

## Purpose

Splinterflow is an open source research and systems project for running large language models on constrained hardware by treating model weights as a selectively materializable external-memory object.

The project should support three complementary ideas:

1. **External-memory execution**  
   Move weights efficiently across storage tiers such as SSD, host RAM, and GPU memory.

2. **Certified adaptive materialization**  
   Materialize only enough weight information to prove that the declared reference model would make the same discrete decision.

3. **Exact structural reuse**  
   Reduce transferred bytes by reusing exact shared structure, for example shared expert bases plus exact per-expert deltas.

These are separate capabilities. They may be combined later, but each must be independently useful and validated.

---

# 1. Core principle

The central question is:

> How little model information can be moved or materialized while still reproducing the declared reference execution exactly, or conservatively proving that the same discrete decision will be produced?

Every major task should improve at least one of:

```text
move fewer exact bytes
reuse more exact bytes
prove that remaining bytes are unnecessary
```

---

# 2. Reference semantics

Every correctness claim must be relative to an explicit **ReferenceProfile**.

A reference profile defines the execution being reproduced, including:

```text
weight representation
dtype
operator implementation
accumulation rules
rounding semantics
routing semantics
attention implementation
expert combination
KV-cache semantics
tie-breaking
```

Possible profiles include:

```text
BF16_REFERENCE
FP16_REFERENCE
NATIVE_QUANTIZED_REFERENCE
```

“Exact” always means exact relative to the selected profile.

Do not silently compare one representation against another and call it exact.

---

# 3. Architectural separation

Keep the project layered.

```text
ReferenceProfile
      ↓
Model Adapter
      ↓
Materialization Policy
      ↓
Materialization Backend
      ↓
Storage / Cache / Transfer
      ↓
Execution Backend
```

Certification is orthogonal:

```text
partial / coarse execution
        ↓
certificate
   ┌────┴────┐
 proven    unknown
   ↓          ↓
 stop       refine
```

A scheduler may decide what to load next.

A scheduler must never decide whether a result is correct.

---

# 4. External-memory runtime

The runtime should support a hierarchy such as:

```text
SSD / NVMe
    ↓
host-RAM cache
    ↓
pinned staging
    ↓
GPU cache
    ↓
active execution buffers
```

The runtime should be generic with respect to the model.

Model-specific concerns belong in adapters.

The runtime should expose:

```text
logical bytes requested
physical bytes read
cache hits/misses
host-to-device bytes
resident bytes by tier
peak RAM
peak VRAM
request count
extent count
timings
```

Skipped data must genuinely remain unread.

---

# 5. Certified adaptive materialization

AWPMI is the research track for progressive materialization with conservative termination.

The generic loop is:

```text
coarse / partial state
        ↓
certificate
   ┌────┴────┐
 proven    unknown
   ↓          ↓
 stop      select next refinement
              ↓
         materialize more
              ↓
          recompute/update
```

Requirements:

- certification must be deterministic and sound;
- confidence or probability is not certification;
- fallback to the complete reference execution must always remain possible;
- a failed certificate should only cost more work, never correctness.

---

# 6. Formal verification

Prefer mature formal-bound tooling over reimplementing verification machinery.

Primary reusable tool:

```text
auto_LiRPA
```

Relevant methods may include:

```text
CROWN
CROWN-IBP
optimized backward LiRPA
```

Use these where they can represent the uncertainty of partially materialized weights.

The property of interest should preferably be expressed directly, for example:

```text
logit(candidate) - logit(contender) > 0
```

rather than bounding every intermediate independently when that loses too much information.

Custom verification code should be limited to what existing tools cannot express cleanly.

Any verifier used in a certified path must still be checked against the chosen ReferenceProfile, especially for finite-precision rounding.

---

# 7. Exact structural reuse

A separate research direction is to reduce I/O without relying on certification.

For MoE experts, investigate representations such as:

```text
Expert_i = SharedBase + ExactDelta_i
```

or:

```text
Expert_i = ClusterBase_c + ExactDelta_i
```

The key requirement is exact reconstruction.

```text
reconstructed expert == reference expert
```

Possible representations include:

```text
arithmetic deltas
XOR deltas
bit-plane deltas
shared low-rank component + exact residual
cluster-specific bases
losslessly compressed residuals
```

Approximate expert merging is a different problem and should not be mixed into the exact path.

---

# 8. Interaction between structural reuse and certification

The two ideas can eventually compose:

```text
resident shared base
        +
progressively materialized delta
        ↓
formal certificate
        ↓
stop before the complete delta if proven
```

Do not combine them until each has independently shown value.

---

# 9. Model adapters

A model adapter is responsible for:

```text
checkpoint tensor mapping
expert discovery
routing semantics
shared experts
tensor layouts
dtype rules
reference kernel choices
KV-cache behavior
model-specific validation
```

The generic core should not contain:

```text
model names
hard-coded tensor names
hard-coded expert counts
router formulas
architecture-specific assumptions
```

---

# 10. Storage design

Prefer original published checkpoint data when practical.

Use:

```text
original checkpoint
+
small index / manifest
```

before introducing a custom format.

Derived representations are justified only when they provide measurable value, for example:

```text
precision-refinement levels
exact expert deltas
alternate layouts
native quantized representations
```

Every derived artifact should record enough metadata to reconstruct and verify its origin.

---

# 11. Native code policy

Use Python for:

```text
research
model integration
experiments
orchestration
benchmarking
```

Move code native only when profiling justifies it.

Typical Rust candidates:

```text
I/O planning
direct-I/O submission
cache bookkeeping
request coalescing
transfer orchestration
```

Typical C++/CUDA candidates:

```text
launch-heavy GPU paths
CUDA Graph integration
small fused operations
GPU-side routing
```

Do not replace PyTorch GEMM or attention kernels without evidence that they are the bottleneck.

---

# 12. Scheduling

Schedulers are optimization components.

Possible signals:

```text
cache state
routing history
bound contribution
bytes
transfer cost
refinement level
expert hotness
certificate margin
```

Possible policies:

```text
deterministic
cost-aware
reuse-aware
bound-reduction-per-byte
learned ranking
```

A learned scheduler may affect efficiency only.

Correctness must remain entirely outside the scheduler.

---

# 13. Research workflow

Every speculative idea should follow the same process:

```text
idea
 ↓
small feasibility probe
 ↓
oracle / simulation
 ↓
fixed acceptance gate
 ↓
PASS?
 ├── no → document and stop
 └── yes
      ↓
physical runtime
      ↓
profiling
      ↓
optimization
```

Do not build a production path before the oracle demonstrates that the idea is worth implementing.

Negative results are valid results.

---

# 14. Testing policy

Permanent test classes should include:

## Reference parity

```text
fallback/full execution == declared reference
```

## Certificate soundness

```text
CERTIFIED ⇒ same discrete reference decision
```

## Bound soundness

```text
reference value ∈ certified enclosure
```

## Storage correctness

```text
loaded bytes == expected checkpoint bytes
```

## No hidden reads

```text
unselected data is not physically fetched
```

## Exact reconstruction

For structural factoring:

```text
reconstruct(base, delta) == original weights
```

## Determinism

Equivalent configurations must remain reproducible across process runs and hash seeds.

## Layering

Prevent architectural shortcuts between:

```text
model adapters
storage
runtime
certification
reference execution
research oracles
```

---

# 15. Benchmarking

Every benchmark should record:

```text
model and revision
tokenizer revision
reference profile
dependency versions
source-tree hash
hardware
OS
storage device
configuration
seed / hash seed
```

Measure correctness and systems behavior separately.

## Correctness

```text
reference mismatches
certified mismatches
fallback mismatches
KV mismatches
router/expert mismatches where relevant
```

## Materialization

```text
logical bytes
physical bytes
fraction materialized
read amplification
derived-storage overhead
```

## Memory

```text
peak VRAM
peak RAM
cache bytes by tier
active-buffer bytes
pinned memory
```

## Performance

```text
TTFT
decode latency
tokens/s
I/O time
H2D time
compute time
certificate time
planning / scheduling time
GPU idle time
```

---

# 16. Optimization order

Optimize based on measured bottlenecks.

Default priority:

```text
reduce physical bytes
↓
increase reuse
↓
improve request granularity
↓
overlap I/O and transfer
↓
reduce runtime overhead
↓
optimize compute kernels only if necessary
```

Do not optimize FLOPs while the workload is dominated by storage.

---

# 17. Open-source reuse

Use existing systems wherever they solve the problem well.

Relevant categories include:

```text
PyTorch / Transformers
safetensors
auto_LiRPA
DwarfStar / ds4
Soup
llama.cpp / GGML
TorchAO
lossless compression libraries
```

Reuse engineering patterns aggressively.

Do not claim established SSD streaming, caching, quantization, or transfer techniques as Splinterflow novelty.

---

# 18. Distinction from conventional SSD inference

Splinterflow may share infrastructure with other external-memory inference systems.

Its broader design space is:

```text
external-memory execution
+
exact structural reuse
+
progressive materialization
+
formal certification
```

The project should remain capable of operating at granularities smaller than a full layer or full expert when evidence supports it.

Streaming whole experts is infrastructure, not the endpoint.

---

# 19. Quantization

Quantization defines a different reference when it changes the model representation or execution.

Always compare:

```text
quantized Splinterflow
vs
the same quantized reference
```

not:

```text
quantized execution
vs
BF16 reference
```

unless the latter equivalence is separately proven.

---

# 20. Fallback discipline

Every optimization must have a complete reference-equivalent fallback.

Examples:

```text
partial expert → full expert
coarse precision → exact precision
base + delta → original/reconstructed exact expert
fast path → complete reference path
```

An optimization that prevents exact fallback does not belong in the exact execution path.

---

# 21. Repository guidance

Keep stable architectural documentation separate from evolving project state.

Use the roadmap/guide for:

```text
principles
architecture
research methodology
module boundaries
validation rules
```

Use state/history/decision documents for:

```text
completed work
specific experiments
accepted decisions
current priorities
benchmark results
```

Do not put volatile project state into this guide.

---

# 22. Task policy

Concrete implementation work should be specified in separate milestone prompts.

A milestone should define:

```text
goal
scope
starting assumptions
deliverables
tests
benchmark
acceptance gate
stop conditions
```

Do not implement the entire roadmap in one task.

---

# 23. Long-term target

The strongest form of Splinterflow would combine:

```text
external-memory runtime
        +
exact reusable weight structure
        +
adaptive refinement
        +
formal certification
```

so that a large model can execute on constrained hardware while moving only the information actually required by the reference decision.

That is the direction.

The exact mechanisms should remain evidence-driven.
