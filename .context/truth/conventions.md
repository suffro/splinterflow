# Conventions

## Repository conventions

- Python ≥ 3.11 (the environment uses 3.13, see `.python-version`), managed with `uv`.
  Dependencies are pinned in `pyproject.toml` and `uv.lock`. Torch comes from the
  PyTorch cu130 index (`[tool.uv.sources]`). If `uv` is not on PATH, use `python -m uv`.
- The model revision and the dataset revision are pinned by commit hash in
  `configs/smollm2-135m.yaml`.
- Every benchmark writes to its own `experiments/phase<N>/<run-name>/`: config,
  environment metadata (versions, GPU, numerics flags, git state, source-tree hash,
  dataset sha256), prompts, raw per-run records, validation records, and a digest. Raw
  results are kept in the repository.

## Development workflow

- Tests: `python -m uv run pytest`. Tests marked `model` load the pinned SmolLM2
  (downloaded on first use).
- Benchmark: `python -m uv run python benchmarks/run.py --output experiments/phase1/<name>`.
  Add `--num-prompts N` for a quick run.
- Report and gate: `python -m uv run python benchmarks/report.py <run> --compare <second run>`.
  Reproducibility is judged by comparing the digests of two identical runs.
- Ordering vs. bound diagnosis:
  `python -m uv run python benchmarks/oracle.py <run> --output <run>-oracle`.
- Phase 1B decomposition oracle:
  `python -m uv run python benchmarks/refinement_oracle.py --output experiments/phase1b/<name>`.
  It reuses the prompts of `source_run` in `configs/phase1b-refinement.yaml`. Then
  `python -m uv run python benchmarks/refinement_report.py <run> --compare <second run>`.
- Phase 1C runtime benchmark:
  `python -m uv run python benchmarks/refinement_runtime.py --output experiments/phase1c/<name>`
  (config `configs/phase1c-runtime.yaml`, prompts of its `source_run`, comparison with its
  `oracle_run`). Then
  `python -m uv run python benchmarks/refinement_runtime_report.py <run> --compare <second run>`.
  Fallback evidence: `python -m uv run python benchmarks/fallback_study.py --output <dir>`.
- Phase 2 adaptive-suffix benchmark:
  `python -m uv run python benchmarks/suffix_runtime.py --output experiments/phase2/<name>`
  (config `configs/phase2-suffix.yaml`, prompts of its `source_run`). Then
  `python -m uv run python benchmarks/suffix_report.py <run> --compare <second run>`.
- Phase 3 packs: `python -m uv run awpmi pack lm-head` and `python -m uv run awpmi pack experts`
  write them under `packs/` (gitignored; deterministic, so they are rebuilt rather than
  committed). Build them in their own process before a benchmark that times direct reads.
- Phase 3 storage benchmark (the Phase 1C LM head on a tier):
  `python -m uv run python benchmarks/storage_runtime.py --output experiments/phase3/<name>`
  (config `configs/phase3-storage.yaml`), then
  `python -m uv run python benchmarks/storage_report.py <run> --compare <second run>`.
- Phase 3 MoE benchmark: `python -m uv run python benchmarks/moe_runtime.py --output experiments/phase3/<name>`
  (config `configs/phase3-moe.yaml`), then
  `python -m uv run python benchmarks/moe_report.py <run> --compare <second run>`.
- Phase 4A out-of-VRAM MoE: `python -m uv run awpmi pack expert-index` (the OLMoE expert index,
  from headers only; needs the checkpoint in the Hugging Face cache and the Hub's file
  digests), then `PYTHONHASHSEED=<n> python -m uv run python benchmarks/olmoe_runtime.py --output experiments/phase4a/<name>`
  (config `configs/phase4a-olmoe.yaml`; it starts the reference and the stream stage in
  processes of their own), `python -m uv run python benchmarks/olmoe_profile.py --output <run>/profile.json`
  (un-instrumented timing), and
  `python -m uv run python benchmarks/olmoe_report.py <run> --compare <second run>`. The two runs
  of a pair use different `PYTHONHASHSEED` values.
- Phase 4B Moonlight (out of VRAM and host RAM): `python -m uv run awpmi pack expert-index --config configs/phase4b-moonlight.yaml`
  (headers only), `python -m uv run python benchmarks/moonlight_reference_check.py --output experiments/phase4b/reference-check`
  (the streaming reference against `from_pretrained` on the truncated model), then
  `PYTHONHASHSEED=<n> python -m uv run python benchmarks/moonlight_runtime.py --output experiments/phase4b/<name>`
  (reference and stream stages in processes of their own; `--stage prepare|reference|stream|digest` runs them one at a
  time, with the same `PYTHONHASHSEED`, which keeps each command under two hours: the reference stage takes about 85
  minutes; `--num-prompts`, `--decode-steps`, `--configurations` and `--skip-audit` are for development runs only), `python -m uv run python benchmarks/moonlight_profile.py --run <run> --configuration <name> --output <run>/profile-<name>.json [--trace]`
  (one configuration per process), and `python -m uv run python benchmarks/moonlight_report.py <run> --compare <second run>`.
  The tokenizer is the official one (`trust_remote_code`, tiktoken) at the pinned revision; its code was read before use.
- Phase 5A expert oracle: `PYTHONHASHSEED=<n> python -m uv run python benchmarks/expert_oracle.py --output experiments/phase5a/<name> --stage prepare`,
  then `--stage capture` (about 25 minutes: Moonlight streamed, the target layer's tensors at every decode step), `--stage oracle --shard 0`
  and `--shard 1` (about 60 and 35 minutes), `--stage oracle-real --shard 0` (about 50 minutes), `--stage digest` (config `configs/phase5a-expert-oracle.yaml`;
  each stage under two hours, launched one by one), and `python -m uv run python benchmarks/expert_oracle_report.py <run> --compare <second run>`.
  The captured tensors (`capture.safetensors`) are not committed: the capture stage regenerates them (their sha256 is in the digest).
- Timing fields (`timings_ms`, `reference_ms`, `wall_ms`, `system`, `profile.json`) are recorded but
  excluded from digests. Runs that are compared for reproducibility must use the same source
  tree (`src`, `benchmarks`, `configs`; tests are not part of it).
- Large raw record files are written as reproducible gzip (`*.jsonl.gz`, mtime 0);
  `read_jsonl` reads both forms. Digests are computed over the decoded records.

## Important rules

- No heuristic or estimate may participate in `certified=True`. Bounds are upper bounds
  on true values, including floating-point error, and every new bound needs a
  soundness test (exact `Fraction` arithmetic where feasible).
- When a guard is added, confirm once that it fails when disabled (see decision 0001).
- Never change the reference model to make AWPMI agree with it. Numerical-environment
  flags apply to both sides and are recorded.
- Benchmarks stop at the first hard failure (certified or fallback mismatch, envelope
  violation, non-bitwise fallback, prefix mismatch) and save it to `failure.json`.
- Byte savings are reported as *effective* bytes: every resident metadata byte, every
  scale, and the fallback's reads count (decision 0003). Never report page counts
  alone. A runtime's bytes are what its store's read log says, and must equal the
  decomposition's accounting for the same rows (decision 0004).
- A runtime reads weight values only through a store. Bounds for the runtime's own
  arithmetic live in `awpmi.bounds` and are validated twice: against exact rational
  arithmetic in tests, and against float64 on every benchmark prompt.
- Only the faithful rounding model may produce `certified=True` (decisions 0001 and 0005).
  Experimental rounding models are run with `experimental=True` and recorded as
  `would_certify`. A bound on a reference operation is validated three ways: the
  reference's own kernels on every grid input of small enclosures (CPU and CUDA),
  adversarial faithful realizations, and exact `Decimal`/`Fraction` arithmetic; then every
  benchmark checks every intermediate's enclosure against the reference's values.
- An exact path (prefix, suffix recomputation, fallback) runs the reference's own operations
  with the reference's shapes (all positions), never a subset, so that it is bitwise equal
  by determinism, and is checked bitwise on every prompt.
- Decision-gate thresholds are fixed in config before a full run, not after seeing it.
- Storage, transfer and materialization code (`awpmi.storage`, `awpmi.streaming`,
  `awpmi.materialization`) never names a model, a tensor or a router, and never imports a layer
  above it; model knowledge lives in `awpmi.models.*` (checked by `tests/test_layering.py`).
- A storage-backed run is checked bit for bit against the same run on resident weights, and its
  bytes are audited: the store's log (logical), what the backend was asked, what the cache
  served, what storage read (physical: the bytes the reads returned), what crossed to the
  device. With direct I/O the physical reads must equal the OS's own per-process counters.
- Do not mix buffered and direct access to one file in a measurement: on NTFS a file with an
  active cache map makes direct reads several times slower. Verify packs with direct reads; let
  the OS close cached handles (a few seconds) after loading a model from the same file.
- A new storage or transfer path needs the no-hidden-reads property tested: every positioned
  read lies in a planned extent, and bytes outside the requested rows never reach the output.
- "Exact" is relative to a declared reference profile (`awpmi.profiles`, decision 0007): the
  stored weight dtype, compute dtype and kernels are checked against the model, and streamed
  weights are never converted on the fly.
- A reference too large for the device runs in its own process, loaded by transformers as
  published, each experts layer materialized whole in turn (`FullLayerOffload`); some prompts
  run with different residency and must agree. Its weights must not come through Shardraw's
  own I/O path; the index Shardraw streams from is audited against them row by row.
- Compact expert buffers keep ascending expert order (the eager implementation accumulates in
  loop order), and expert parameters are `None` between calls, never `meta` (a CUDA grouped
  GEMM given a meta weight returned garbage instead of failing).
- A checkpoint index records the publisher's file digests and is built from headers only;
  nothing proportional to the model is read or written to make it.
- A model too large for host memory is compared with `StreamingReference` (decision 0008), which
  imports nothing of Shardraw's storage, transfer, materialization or expert adapters
  (`tests/test_layering.py`), and is itself checked against `from_pretrained` where that fits.
- A bounded experts call keeps the experts implementation's own combine, once per call; never
  combine per chunk. A new experts implementation or GPU needs the chunked-equals-unchunked tests
  (per-group GEMM independence is a property of the kernel, not of the algorithm).
- An oracle may hold every weight, but keeps realistic and ideal apart: a realistic bound or ordering
  uses only what a runtime could have read (values read, resident metadata); an ideal one is labelled
  and reported beside it, never instead. A what-if rounding or accumulation model is a separate tier
  whose results are `would_certify`, never `certified`.
- A certificate against an LM head checks every vocabulary row. A filter on the nearest rows may
  only reject early (a necessary condition), never accept.
