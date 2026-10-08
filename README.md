# Weightsift — Adaptive Weight-Page Materialization for Inference

Research prototype. AWPMI progressively materializes independently fetchable weight
pages. It stops only when a conservative certificate proves that the discrete
next-token decision is identical to that of the fully materialized reference model.
Confidence is never a certificate. If certification fails, AWPMI materializes
everything and reproduces the reference computation exactly.

The full plan is in [`.context/state/Weightsift_Implementation_Guide.md`](.context/state/Weightsift_Implementation_Guide.md).
The current status and results are in [`.context/state/current.md`](.context/state/current.md).

## Status

**Phase 1A — minimal certified materialization** is implemented. The adaptive region
is the LM head of `HuggingFaceTB/SmolLM2-135M-Instruct`, split into input-column
pages with Cauchy–Schwarz residual bounds. The transformer body runs exactly, and
all weights stay resident in memory.

**Phase 1B — precision-refinement decomposition search** is an oracle, not a
runtime. It decomposes the LM head exactly as W = coarse + refinements + exact
remainder, adds row-selective refinement and a tie-aware certificate, and measures
how many bytes a certified decision would need.

**Phase 1C — refinement runtime** implements the decomposition Phase 1B chose: an int6
coarse pass over every row, int4 refinement of the rows that can still win, then the
original BF16 rows of the last few. It reads a bit-packed store that counts every byte,
and certifies with on average 0.50 of the BF16 LM-head bytes, or 0.40 with the masked
fallback (the default since Phase 2).

**Phase 2 — certification into the transformer** extends the adaptive region into the
last layer's MLP: the final projection (`down_proj`) or the whole MLP is read only in
part, its output is bounded through the reference's own BF16 operations and the final
RMSNorm, and a scale-free pairwise certificate decides through the norm. Only the faithful
rounding model certifies; round-to-nearest-even is recorded as a what-if. It is sound
(zero mismatches and zero bound violations over 1000 prompts), but on this model it does
not save bytes: the reference's own BF16 roundings of activations cap the final
projection's certificate at 27% of tokens even with every weight read.

**Phase 3 — real selective storage** moves the weights out of memory. A storage core that
knows no model reads only the requested rows of a weight from files (direct I/O) or host
memory, moves only those bytes to the GPU through a pinned double-buffered streamer, keeps
pages in a budgeted cache, and counts every byte against the OS's own counters. The Phase 1C
LM head runs on it unchanged and bit for bit equal to the resident runtime, its exact rows read
straight from the published checkpoint. The same backend serves the experts of a
mixture-of-experts model (Granite 3.1 1B-A400M; seven architectures in tests) bit for bit,
with only the routed experts read from the drive.

**Phase 4A — a mixture of experts larger than the GPU** runs OLMoE-1B-7B (12.9 GB of experts)
on an 8 GB GPU under a 6 GB memory cap. Its experts are read in place from the published
checkpoint, which stores each expert as separate tensors, through an index built from file
headers alone. A compact experts call holds only the routed experts on the GPU. Every step
reproduces the fully materialized reference bit for bit: tokens, logits, routing, every expert
output and the KV cache.

**Phase 4B — out of VRAM and out of host RAM** runs Moonlight-16B-A3B, DeepSeek-V3's
architecture at 16 B parameters (28.8 GB of routed experts, a 31.9 GB checkpoint, against
32 GB of RAM and an 8 GB GPU under a 6 GB cap). An experts call that would need more than a byte
budget runs in chunks of experts, with the experts implementation's own combine once per call,
so a prefill that routes every expert holds 165 MiB of experts instead of a whole layer. It is
compared with an independent streaming reference, transformers' own model and loader with one
experts layer materialized at a time, itself checked against `from_pretrained` where that fits.
Every step equals the reference bit for bit, in two runs with identical digests.

**Phase 5A — AWPMI inside routed experts, an oracle study** asks whether, once the router has chosen
Moonlight's experts, Weightsift must read all of them to keep the final token. For the last MoE layer at
decode, upstream exact, an oracle reads the routed experts progressively in several decompositions
(neuron pages, down-projection row pages, neuron-major pages, precision levels), bounds every
intermediate of the experts call, the MoE block, the residual and the final norm through the reference's
own operations, and certifies the token against the whole vocabulary with a pairwise certificate
generalized to a mixture of experts. Under the certified (faithful) rounding model the answer is no
saving (gate: FAIL): the reference's own BF16 roundings leave about three logits of uncertainty on the
closest pair, so even with every routed byte read only 8.1% of 768 tokens certify, and the realistic
bounds on unread parts are too loose to certify any token earlier, even in real arithmetic. Labelled
round-to-nearest-even what-ifs and a real-arithmetic diagnostic are reported beside it; no expert-AWPMI
runtime is built.

**Phase 5A2 — a formal verifier on the same question** asks whether auto_LiRPA's CROWN family, used as published
in an isolated environment (`research/crown_expert_oracle`), bounds the unread expert weights more tightly than
Phase 5A. It does not make expert AWPMI pay. Given the L2 remainder norms Phase 5A already uses, upstream
auto_LiRPA bounds them soundly only in an experimental mode, and there about 34 times below Phase 5A's own bound;
and the uncertainty sets this metadata defines hold weights that flip the decision until about 0.82–0.91 of the
routed bytes even in real arithmetic, so no verifier could do much better. The phase stopped before its 12-sample
stage.

**Phase 5C — exact shared bases and progressive expert deltas** (`research/expert_deltas`) asks two separate questions
about Moonlight's routed experts. First, can a shared resident base plus exact, losslessly compressed per-expert deltas
(XOR or modular deltas of the BF16 bit patterns, reconstructed bit for bit) move fewer bytes than competent independent
compression? No: across four layers the 64 experts of a layer are, to every test, independent (no shared component, no
shared positional scale, no permuted copies of neurons), no expert has a delta cheaper than itself against any base, and
every base strategy costs 4.5% or more extra drive bytes on Phase 5A's routing trace. Independent exact compression
(bit planes with zstd) does pay: 0.66 of the BF16 bytes, a third fewer drive bytes per decode token. Second, can a
partially read exact representation certify the reference's token while some bytes stay unread? Bit planes read page by
page give, at the same raw bytes, uncertainty sets about three times closer to the truth than Phase 5A's, but their exact
optimum still admits decision-flipping weights until the last mantissa plane or two: in real arithmetic the certificate
leaves 5–17% of the compressed bytes unread on tokens with a margin of more than a logit and about none on close calls (8%
on a gap-stratified sample, 6% over the real gap distribution; the gate asked for 10%), and Phase 5A's BF16 rounding floor
remains. No runtime is built.

**Phase 6A — the native runtime foundation** moves the storage path into a small Rust core (`native/`, built by maturin
through uv, Python bindings with PyO3) behind the existing storage contract: read plans equal to Python's, direct reads
on a pool of threads, a host-RAM expert cache with a strict byte budget and an exact LRU, and transfer jobs into the
streamer's pinned buffers. It knows no model and no CUDA; the Python backend stays the fallback, and a configuration
chooses `backend: python` or `native`. On Phase 4B's Moonlight benchmark every step of every configuration equals the
independent reference bit for bit, in two runs; the native backend without a cache reads exactly what the Python one
reads, 9% faster per decode token; with 12 GB of host cache (prefills not admitted) a decode token takes 807 ms instead of
1,157 ms for the best Python configuration and 1,233 ms for Phase 4B's streaming, and prefills 37% less. What bounds a
token now is copying its 2.7 GB of experts to the GPU (PCIe 3.0 x8 here), then the transformer's Python and kernel
launches.

## Setup

```bash
python -m pip install uv     # if uv is not installed
uv sync                      # Python 3.11+, torch 2.14.1 (CUDA 13.0 wheels), transformers 5.18.0
```

`uv sync` also builds the native I/O core (`native/`, Phase 6A) with maturin, which needs a Rust toolchain
([rustup](https://rustup.rs); `native/rust-toolchain.toml` pins 1.95.0, fetched on first build). Without Rust,
`uv sync --no-group native` leaves it out: everything else runs on the Python storage backend as before, the native
tests are skipped, and configurations with `backend: native` refuse to start. Build details, platforms and the fallback
are in [`native/README.md`](native/README.md).

## Usage

```bash
uv run pytest                                                   # unit + model tests
uv run python benchmarks/run.py --output experiments/phase1/my-run [--num-prompts 50]
uv run python benchmarks/report.py experiments/phase1/my-run [--compare experiments/phase1/other-run]
uv run python benchmarks/oracle.py experiments/phase1/my-run --output experiments/phase1/my-run-oracle
uv run python benchmarks/refinement_oracle.py --output experiments/phase1b/my-run [--num-prompts 50]
uv run python benchmarks/refinement_report.py experiments/phase1b/my-run [--compare experiments/phase1b/other-run]
uv run python benchmarks/refinement_runtime.py --output experiments/phase1c/my-run [--num-prompts 50]
uv run python benchmarks/refinement_runtime_report.py experiments/phase1c/my-run [--compare experiments/phase1c/other-run]
uv run python benchmarks/fallback_study.py --output experiments/phase1c/my-study
uv run python benchmarks/suffix_runtime.py --output experiments/phase2/my-run [--num-prompts 50]
uv run python benchmarks/suffix_report.py experiments/phase2/my-run [--compare experiments/phase2/other-run]
uv run weightsift pack lm-head                                       # Phase 3 packs, under packs/
uv run weightsift pack experts
uv run python benchmarks/storage_runtime.py --output experiments/phase3/my-run [--num-prompts 50]
uv run python benchmarks/storage_report.py experiments/phase3/my-run [--compare experiments/phase3/other-run]
uv run python benchmarks/moe_runtime.py --output experiments/phase3/my-moe-run [--num-prompts 5]
uv run python benchmarks/moe_report.py experiments/phase3/my-moe-run [--compare experiments/phase3/other-moe-run]
uv run weightsift pack expert-index                                  # Phase 4A: OLMoE's expert index (headers only)
uv run python benchmarks/olmoe_runtime.py --output experiments/phase4a/my-run [--num-prompts 2]
uv run python benchmarks/olmoe_profile.py --output experiments/phase4a/my-run/profile.json
uv run python benchmarks/olmoe_report.py experiments/phase4a/my-run [--compare experiments/phase4a/other-run]
uv run weightsift pack expert-index --config configs/phase4b-moonlight.yaml  # Phase 4B: Moonlight's expert index
uv run python benchmarks/moonlight_reference_check.py --output experiments/phase4b/reference-check
uv run python benchmarks/moonlight_runtime.py --output experiments/phase4b/my-run [--stage prepare|reference|stream|digest]
uv run python benchmarks/moonlight_profile.py --run experiments/phase4b/my-run --configuration stream --trace \
    --output experiments/phase4b/my-run/profile-stream.json
uv run python benchmarks/moonlight_report.py experiments/phase4b/my-run [--compare experiments/phase4b/other-run]
uv run python benchmarks/expert_oracle.py --output experiments/phase5a/my-run --stage prepare   # Phase 5A, then each stage:
uv run python benchmarks/expert_oracle.py --output experiments/phase5a/my-run --stage capture   # capture | oracle --shard i | oracle-real | digest
uv run python benchmarks/expert_oracle_report.py experiments/phase5a/my-run [--compare experiments/phase5a/other-run]
# Phase 5A2 runs in its own environment: see research/crown_expert_oracle/README.md
# Phase 5C (the Weightsift environment; codecs from the default dependency group `research`): see research/expert_deltas/README.md
uv run python research/expert_deltas/probe_codec.py --output experiments/phase5c/my-probe
uv run python research/expert_deltas/structure_run.py --output experiments/phase5c/my-run --stage prepare   # then layer --layer L, replay, report
uv run python research/expert_deltas/progressive_run.py --output experiments/phase5c/my-progressive --shard 0 --shards 2   # then --report
# Phase 6A: the native (Rust) I/O core and host-RAM expert cache (configs/phase6a-native.yaml)
uv run python benchmarks/native_trace.py --output experiments/phase6a/my-trace          # stage A: repeated bytes, LRU capacity
uv run python benchmarks/native_io.py --output experiments/phase6a/my-io.json [--cuda] [--sweep]   # stage B: I/O alone
uv run python benchmarks/moonlight_runtime.py --config configs/phase6a-native.yaml --output experiments/phase6a/my-run \
    --stage prepare --reference-from experiments/phase6a/baseline-run1   # then --stage stream, --stage digest
uv run python benchmarks/moonlight_profile.py --run experiments/phase6a/my-run --config configs/phase6a-native.yaml \
    --configuration native-host-12g --prompts 7 0 3 5 --warm --output experiments/phase6a/my-run/profile-native-host-12g-warm.json
uv run python benchmarks/native_report.py experiments/phase6a/my-run [--compare experiments/phase6a/other-run] \
    [--baseline experiments/phase6a/baseline-run1] [--io experiments/phase6a/io/io-cuda.json]
uv run python benchmarks/reference_numerics.py --output experiments/phase6a/bf16/numerics.json   # §7: the reference's kernels
```

`run.py` writes raw per-input records, validation records, the prompts, the
config and the environment metadata. It stops at the first hard failure.
`report.py` aggregates the results and evaluates the Phase 1 gate. `oracle.py`
computes ceilings that do not depend on the scheduler, to tell whether the
ordering or the bound is the bottleneck. `refinement_oracle.py` and
`refinement_report.py` do the same for Phase 1B decompositions
(`configs/phase1b-refinement.yaml`), and classify each one against Phase 1A.
`refinement_runtime.py` runs the Phase 1C runtime on every prompt with both fallbacks,
validates it against the reference, and compares its bytes with the Phase 1B oracle;
`refinement_runtime_report.py` evaluates the gate (`configs/phase1c-runtime.yaml`).
`fallback_study.py` gathers the evidence behind the fallback decisions.
`suffix_runtime.py` runs the Phase 2 adaptive suffix for every stage, budget and rounding
model, checks every intermediate against the reference, and `suffix_report.py` draws the
materialization curves and evaluates the gate (`configs/phase2-suffix.yaml`).
The CLI is `weightsift`; `wsift` is an alias with the same commands and options
(for example, `uv run wsift pack expert-index`).

`weightsift pack` writes Phase 3 packs: safetensors files and a manifest with every segment's
location and hash, referring to the published checkpoint wherever it holds the bytes.
`storage_runtime.py` runs the Phase 1C LM head against the full BF16 head on the drive and in
host memory, resident, on the drive and with a cached base level, audits every byte, and
`storage_report.py` evaluates gates A and B (`configs/phase3-storage.yaml`). `moe_runtime.py`
serves a MoE model's experts from the drive under several cache budgets and policies, compares
every decoding step with the resident model bit for bit, and `moe_report.py` evaluates gate C
(`configs/phase3-moe.yaml`). `weightsift pack expert-index` indexes a checkpoint whose experts are
split into several tensors without copying them. `olmoe_runtime.py` runs the out-of-VRAM
benchmark in two processes: the fully materialized reference, then the streamed model under a
device-memory cap, compared with it in every recorded digest. `olmoe_profile.py` times the
streamed steps without those digests, and `olmoe_report.py` evaluates correctness and gates A–D
(`configs/phase4a-olmoe.yaml`). `moonlight_reference_check.py` checks the streaming reference
against `from_pretrained` on Moonlight truncated to its first layers. `moonlight_runtime.py` runs
the Phase 4B benchmark: the streaming reference, then the streamed model under a device cap with
bounded expert calls, every configuration compared with the reference and audited;
`moonlight_profile.py` times one configuration per process, and `moonlight_report.py` evaluates
correctness and gates A–F (`configs/phase4b-moonlight.yaml`). The Moonlight tokenizer is the
official remote code (tiktoken), run at the pinned revision. `expert_oracle.py` runs the Phase 5A
oracle: a capture of the last MoE layer's tensors at every decode step on the Phase 4B streamed path,
then, per sample, the reference recomputed bitwise, every arithmetic tier's ceiling and every strategy's
cells; `expert_oracle_report.py` evaluates correctness and the decision gate
(`configs/phase5a-expert-oracle.yaml`). For Phase 6A, `native_trace.py` measures on recorded routing how often expert
bytes repeat and exactly what an LRU host tier of each size would hit; `native_io.py` measures the native I/O core alone
against the Python path; `moonlight_runtime.py` and `moonlight_profile.py` take `configs/phase6a-native.yaml`, whose
configurations choose the storage backend (`python` or `native`) and the native host cache, and `native_report.py`
evaluates its gates (correctness, I/O parity, host-cache replay, end-to-end decode time). `reference_numerics.py`
records how the reference's kernels round and whether a row's result depends on its batch.

## Layout

```text
src/awpmi/      reference, paging, bounds, state, certificate, schedulers, executor, tracing,
                decomposition (Phase 1B, packing), oracle (Phase 1B simulator),
                stores (Phase 1C packed store, Phase 2 MLP store), refinement_head (Phase 1C runtime),
                bounds/{rounding,enclosure,operators,pairwise} and suffix_runtime (Phase 2),
                storage, streaming, materialization, models/moe and cli (Phase 3),
                profiles, models/checkpoint and models/olmoe (Phase 4A),
                streaming_reference, models/moonlight and models/streamed (Phase 4B),
                oracle/experts (Phase 5A expert oracle), storage/native (Phase 6A: the native store)
native/         the Rust I/O core (Phase 6A): core (weightsift-io: plans, direct reads, host-RAM cache,
                transfer jobs) and python (the PyO3 module weightsift_native); see native/README.md
tests/          bound soundness, pages, certificate and ties, reference parity, fallback parity,
                decomposition exactness, refinement oracle, packing, coarse bounds, runtime,
                rounding models, operator bounds, enclosure LM head, adaptive suffix,
                storage (no hidden reads), streaming and caches, runtime on storage, MoE experts, layering,
                composed segments and compact expert calls (Phase 4A), chunked expert calls,
                the streaming reference, the Moonlight adapter and streamed parameters (Phase 4B),
                the expert oracle (Phase 5A), the native store against the Python one (Phase 6A; the MoE
                tests run through both backends)
benchmarks/     run.py, report.py, prompts.py, oracle.py, refinement_oracle.py, refinement_report.py,
                refinement_runtime.py, refinement_runtime_report.py, fallback_study.py,
                suffix_runtime.py, suffix_report.py, storage_runtime.py, storage_report.py,
                moe_runtime.py, moe_report.py, olmoe_runtime.py, olmoe_profile.py, olmoe_report.py,
                moonlight_runtime.py, moonlight_reference_check.py, moonlight_profile.py, moonlight_report.py,
                expert_oracle.py, expert_oracle_report.py, native_trace.py, native_io.py, native_report.py,
                reference_numerics.py
configs/        smollm2-135m.yaml (pinned model and dataset revisions), phase1b-refinement.yaml,
                phase1c-runtime.yaml, phase2-suffix.yaml, phase3-storage.yaml, phase3-moe.yaml,
                phase4a-olmoe.yaml, phase4b-moonlight.yaml, phase5a-expert-oracle.yaml,
                phase5a2-crown-oracle.yaml, phase5c-expert-deltas.yaml, phase6a-native.yaml
research/       crown_expert_oracle (Phase 5A2: the auto_LiRPA verifier, its own environment and lockfile),
                expert_deltas (Phase 5C: exact shared bases, bit-plane deltas, their codecs, replay and progressive oracle)
experiments/    raw results per phase and run
packs/          packs and expert indexes (gitignored; rebuilt by `weightsift pack`)
```
