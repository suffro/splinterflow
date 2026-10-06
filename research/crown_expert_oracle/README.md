# Phase 5A2: CROWN / auto_LiRPA expert oracle

A research oracle, not a runtime. It asks whether a mature bound-propagation verifier, auto_LiRPA's CROWN family, can
tighten Phase 5A's bounds on routed-expert weights that have not been read, enough to make AWPMI inside Moonlight's
routed experts pay. This is an in-progress development checkpoint: the Stage 1 probe and one development sample are
complete. Stage 2, the final decision and the phase report are pending; current status is recorded in
`.context/state/current.md`.

## Two environments

| | Splinterflow (repository root) | Verifier (this directory) |
| --- | --- | --- |
| Python, torch | 3.13, 2.14.1+cu130 | 3.11.16, 2.11.0+cu130 |
| Packages | awpmi, transformers 5.18 | auto_LiRPA 0.7.2 @ `5a098e8f9fb5786a428a024981d833d303921f2d` (BSD-3-Clause), numpy 2.4.6, safetensors, pyyaml, psutil |
| Lockfile | `uv.lock` (root) | `uv.lock` (here) |
| Runs | `export.py` | `probe.py`, `run.py`, `report.py`, `full_crown_cost.py` |

auto_LiRPA declares Python 3.11 and torch < 2.12, so it gets its own interpreter and lockfile; the Splinterflow
environment is not touched (it cannot import auto_LiRPA: `tests/test_crown_boundary.py`). The two exchange only the
artifact `export.py` writes (safetensors and a JSON manifest with the sha256 of every file); the verifier imports no
`awpmi` (`tests/test_boundary.py` here).

```bash
python -m uv sync --project research/crown_expert_oracle          # creates research/crown_expert_oracle/.venv
python -m uv run --project research/crown_expert_oracle pytest research/crown_expert_oracle/tests
```

## Pipeline

```bash
# Splinterflow environment: the Phase 5A capture (benchmarks/expert_oracle.py, ~25 min), checked against Phase 5A run1's digests
PYTHONHASHSEED=1 python -m uv run python benchmarks/expert_oracle.py --output experiments/phase5a2/capture --stage prepare
PYTHONHASHSEED=1 python -m uv run python benchmarks/expert_oracle.py --output experiments/phase5a2/capture --stage capture
# Splinterflow environment: the artifact (the selected samples, Phase 5A's schedules, states, enclosures, bounds, cells)
PYTHONHASHSEED=1 python -m uv run python research/crown_expert_oracle/export.py --output experiments/phase5a2/<run>
# Verifier environment
V="python -m uv run --project research/crown_expert_oracle python research/crown_expert_oracle"
PYTHONHASHSEED=1 $V/probe.py --output experiments/phase5a2/probe                       # stage 1 (toy + cost calibration)
PYTHONHASHSEED=1 $V/run.py --run experiments/phase5a2/<run> --part validate
PYTHONHASHSEED=1 $V/run.py --run experiments/phase5a2/<run> --part compare [--shard i --shards n]
PYTHONHASHSEED=1 $V/run.py --run experiments/phase5a2/<run> --part search  [--shard i --shards n]
$V/report.py experiments/phase5a2/<run> [--compare experiments/phase5a2/<run2>]
```

The captured tensors and the artifact are regenerated, not committed (`.gitignore`); their sha256 are in the records.

## What auto_LiRPA provides, what Splinterflow adds

auto_LiRPA (used as published, unmodified):

- `BoundedModule` (graph tracing, bound propagation), `BoundedParameter` and `BoundedTensor` with `PerturbationLpNorm`
  (L∞ boxes on weights and activations);
- CROWN / backward LiRPA, CROWN-IBP, α-CROWN (optimized relaxations), IBP; output specifications `C` (one row per
  contender: the bound is on Δ·y directly);
- the relaxations: weight-perturbed linear layers and products (McCormick, `BoundLinear`, `BoundMul`), sigmoid
  (`BoundSigmoid`); batched intermediate bounds (`crown_batch_size`).

Splinterflow (this directory, `export.py`):

- the verification graphs as forward PyTorch modules (`crown_oracle/graph.py`): the experts, the routing-weighted
  combine, the shared experts and residual as exact constants; SiLU written g·σ(g) (auto_LiRPA has no SiLU operator;
  ONNX tracing decomposes `F.silu` the same way); exact zero biases (auto_LiRPA's tested Gemm path for perturbed weights);
- the weight sets from what a runtime could have read (`crown_oracle/sets.py`): the q6 and q4 levels and their resident
  remainder norms, intersected; unread values hidden by a read view (anti-cheating);
- the per-expert decomposition of the separable property (Δ·b plus each expert's auto_LiRPA bound);
- the finite-precision combination (`crown_oracle/certify.py`): Phase 5A's pairwise certificate (decision 0009), its
  named rounding terms and assembly, around auto_LiRPA's structural bound;
- byte accounting and schedules (Phase 5A's, exported), sample selection, the search, validation, diagnostics
  (`crown_oracle/attack.py`: adversarial realizations, the reduced set's exact optimum), the report.

No CROWN, LiRPA, α-CROWN or branch-and-bound code is written here.
