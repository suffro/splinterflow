"""Stage 5C1-B: exact structural reuse on a representative subset of Moonlight's routed experts (Phase 5C).

    python -m uv run python research/expert_deltas/structure_run.py --output experiments/phase5c/<run> --stage prepare
    python -m uv run python research/expert_deltas/structure_run.py --output experiments/phase5c/<run> --stage layer --layer 26
    python -m uv run python research/expert_deltas/structure_run.py --output experiments/phase5c/<run> --stage replay
    python -m uv run python research/expert_deltas/structure_run.py --output experiments/phase5c/<run> --stage report

Stages (one process each; `all` runs them in turn):

  prepare  config and environment
  layer    one sampled layer (config `structure.layers`), all its experts and matrix kinds, one layer in memory at a time:
           the file checked against the publisher's sha256; every tensor read twice (positioned reads, safetensors) and
           compared; per matrix kind on the GPU the census, the pairwise proxy costs (XOR and modular deltas), the base
           strategies' assignments and a synthetic median base; the neuron-permutation diagnostic; then actual
           compression, every reconstruction compared bit for bit: the independent baselines (B, C), trained dictionaries
           on disjoint halves (D), and the base strategies' deltas. Writes layer_<L>.json
  replay   Phase 5A's routing trace through host-RAM caches of each representation under the configured budgets, bases
           pinned, bounded or on the GPU. Writes replay.json
  report   the gate (config `gate_5c1`), summary.json, summary.md, digest.json

A stage stops at its first inexact reconstruction and writes failure.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from expert_deltas import bits, records, structure  # noqa: E402
from expert_deltas.codecs import Codec, Dictionary  # noqa: E402
from expert_deltas.compression import blocks, decode_throughput, measure, streams  # noqa: E402
from expert_deltas.replay import Representation, replay, trace_steps  # noqa: E402
from expert_deltas.source import MATRICES, Checkpoint, verify_files  # noqa: E402

ENCODE = {"xor": bits.xor_delta, "modular": bits.modular_delta}
DECODE = {"xor": bits.xor_restore, "modular": bits.modular_restore}


class Inexact(Exception):
    pass


def codec_of(pair) -> Codec:
    return Codec(pair[0], int(pair[1]))


def label(entry: dict) -> str:
    return f"{entry['block']}/{entry['transform']}/{entry['codec']['name']}-{entry['codec']['level']}"


# Stage: one layer


def run_layer(config: dict, output: Path, layer: int, skip_verify: bool, limit: int | None = None) -> int:
    settings, model = config["structure"], config["model"]
    threads = int(settings["threads"])
    device = torch.device("cuda", torch.cuda.current_device())
    checkpoint = Checkpoint(model["repository"], model["revision"])
    experts = checkpoint.experts(layer) if limit is None else int(limit)
    result: dict = {"layer": layer, "experts": experts, "timings": {}, "development_expert_limit": limit}
    started = time.perf_counter()
    file = checkpoint.tensor(layer, 0, "gate").file
    result["verification"] = "skipped (development)" if skip_verify else verify_files(checkpoint, [file])
    result["timings"]["verify_s"] = time.perf_counter() - started

    started = time.perf_counter()
    weights = {e: {m: checkpoint.read_patterns(checkpoint.tensor(layer, e, m)) for m in MATRICES} for e in range(experts)}
    reference = {e: {m: bits.patterns(checkpoint.read_reference(checkpoint.tensor(layer, e, m))) for m in MATRICES} for e in range(experts)}
    result["read_paths_equal"] = all(np.array_equal(weights[e][m], reference[e][m]) for e in weights for m in MATRICES)
    for e in weights:
        for m in MATRICES:
            bits.require_finite(weights[e][m])
    result["shapes"] = {m: list(weights[0][m].shape) for m in MATRICES}
    result["expert_bytes"] = sum(weights[0][m].nbytes for m in MATRICES)
    result["timings"]["read_s"] = time.perf_counter() - started
    if not result["read_paths_equal"]:
        records.write_json(output / "failure.json", {"kind": "read_paths", "layer": layer})
        return 1

    # Census, proxy costs, assignments, synthetic base (GPU, one matrix kind at a time).
    started = time.perf_counter()
    result["kinds"], assignments, medians = {}, {}, {}
    for m in MATRICES:
        rows, cols = weights[0][m].shape
        p = structure.to_device(np.stack([weights[e][m] for e in range(experts)]), device)
        entry = {"census": structure.census(p, rows, cols)}
        entry["costs"] = {}
        assignments[m] = {}
        for encoding in settings["bases"]["encodings"]:
            costs = structure.pairwise_costs(p, encoding)
            entry["costs"][encoding] = {
                "alone_bits": costs.diagonal().tolist(), "matrix_bits": [[round(float(v), 6) for v in row] for row in costs],
            }
            assignments[m][encoding] = structure.assignments(costs, int(settings["bases"]["candidates"]), list(settings["bases"]["clusters"]))
        median = structure.elementwise_median(p)
        medians[m] = median.cpu().numpy().astype(np.uint16).reshape(rows, cols)
        entry["synthetic"] = {k: v.tolist() for k, v in structure.synthetic_costs(p, median).items()}
        entry["assignments"] = assignments[m]
        result["kinds"][m] = entry
        del p
        torch.cuda.empty_cache()
    rng = np.random.default_rng(int(settings["seed"]) + layer)
    pairs = [tuple(int(x) for x in rng.choice(experts, 2, replace=False)) for _ in range(int(settings["permutation_pairs"]))]
    tensors = {m: structure.to_device(np.stack([weights[e][m] for e in sorted({x for pair in pairs for x in pair})]), device) for m in MATRICES}
    position = {e: k for k, e in enumerate(sorted({x for pair in pairs for x in pair}))}
    result["permutation"] = structure.permutation_similarity(
        tensors["gate"], tensors["up"], tensors["down"], tuple(weights[0]["gate"].shape), [(position[a], position[b]) for a, b in pairs],
        int(settings["seed"]),
    )
    for entry, (a, b) in zip(result["permutation"]["pairs"], pairs):
        entry["pair"] = [a, b]
    del tensors
    torch.cuda.empty_cache()
    result["peak_device_bytes"] = torch.cuda.max_memory_allocated(device)
    result["timings"]["census_s"] = time.perf_counter() - started

    try:
        # Independent baselines B and C.
        started = time.perf_counter()
        independent = settings["independent"]
        result["independent"] = []
        for kind, transform, name, level in independent["configurations"]:
            entry = measure(weights, reference, kind, transform, Codec(name, int(level)), threads)
            result["independent"].append(entry)
            print(f"layer {layer} independent {label(entry)}: {entry['ratio']:.4f} exact {entry['exact']}", flush=True)
            if not entry["exact"]:
                raise Inexact(entry)
        result["timings"]["independent_s"] = time.perf_counter() - started

        # Baseline D: dictionaries, two-fold (train on one half of the experts, compress the other).
        started = time.perf_counter()
        spec = independent["dictionary"]
        half = int(spec["train_experts"])
        folds = [(list(range(half)), list(range(half, experts))), (list(range(half, experts)), list(range(half)))]
        result["dictionary"] = []
        for kind in spec["blocks"]:
            for transform in spec["transforms"]:
                per_object, total, dict_bytes, exact = {}, 0, [], True
                timings = []
                for train, evaluate in folds:
                    pool = [s for e in train for _, p in blocks(weights[e], kind) for s in streams(p, transform)]
                    chosen = sorted(rng.choice(len(pool), size=min(int(spec["samples"]), len(pool)), replace=False).tolist())
                    dictionary = Dictionary.train(int(spec["size"]), [pool[i] for i in chosen], level=int(spec["codec"][1]))
                    entry = measure({e: weights[e] for e in evaluate}, {e: reference[e] for e in evaluate}, kind, transform,
                                    codec_of(spec["codec"]), threads, dictionary)
                    exact &= entry["exact"]
                    per_object.update(zip(evaluate, entry["per_object_stored"]))
                    dict_bytes.append(dictionary.nbytes)
                    timings.append(entry["timings"])
                summary = {"block": kind, "transform": transform, "codec": codec_of(spec["codec"]).to_json(), "dictionary_bytes": dict_bytes,
                           "per_object_stored": [per_object[e] for e in range(experts)], "stored_bytes": sum(per_object.values()),
                           "bf16_bytes": experts * result["expert_bytes"], "exact": bool(exact), "timings": timings}
                summary["ratio"] = summary["stored_bytes"] / summary["bf16_bytes"]
                result["dictionary"].append(summary)
                print(f"layer {layer} dictionary {kind}/{transform}: {summary['ratio']:.4f} exact {exact}", flush=True)
                if not exact:
                    raise Inexact(summary)
        result["timings"]["dictionary_s"] = time.perf_counter() - started

        # Base strategies: actual compression of their deltas (the configured subset), every expert restored exactly, and
        # each base matrix the strategy uses as its own compressed object (derived storage).
        started = time.perf_counter()
        bases = settings["bases"]
        result["bases"] = []
        for name in bases["measure"]:
            for encoding in bases["encodings"] if name in bases.get("all_encodings", []) else ["xor"]:
                plan = {}  # per kind: each expert's base (−1 alone; "median": the synthetic base)
                for m in MATRICES:
                    if name == "synthetic-median":
                        plan[m] = ["median"] * experts
                    else:
                        plan[m] = assignments[m][encoding][strategy_name(assignments[m][encoding], name)]["base"]

                def base_patterns(m, b):
                    return medians[m] if b == "median" else weights[b][m]

                objects = {e: {m: weights[e][m] if plan[m][e] == -1 else ENCODE[encoding](weights[e][m], base_patterns(m, plan[m][e]))
                               for m in MATRICES} for e in range(experts)}

                def finish(e, m, p, plan=plan, encoding=encoding):
                    b = plan[m][e]
                    return p if b == -1 else DECODE[encoding](p, base_patterns(m, b))

                keys = sorted({(m, b) for m in MATRICES for b in plan[m] if b != -1}, key=repr)
                kinds = list(bases["blocks"]) + (["rows16"] if name in bases.get("page_blocks", []) and encoding == "xor" else [])
                for kind in kinds:
                    for transform in bases["transforms"]:
                        for pair in bases["codecs"]:
                            entry = measure(objects, reference, kind, transform, codec_of(pair), threads, finish=finish)
                            stored_bases, exact_bases = {}, True
                            for m, b in keys:
                                size, exact = matrix_stored(base_patterns(m, b), transform, codec_of(pair), threads)
                                stored_bases[f"{m}:{b}"] = size
                                exact_bases &= exact
                            entry.update(strategy=name, encoding=encoding, base_of={m: plan[m] for m in MATRICES},
                                         base_objects=stored_bases, bases_exact=bool(exact_bases))
                            result["bases"].append(entry)
                            print(f"layer {layer} base {name}/{encoding} {label(entry)}: {entry['ratio']:.4f} "
                                  f"(+{len(keys)} base objects, {sum(stored_bases.values()) / (experts * result['expert_bytes']):.4f}) "
                                  f"exact {entry['exact'] and exact_bases}", flush=True)
                            if not (entry["exact"] and exact_bases):
                                raise Inexact(entry)
        result["timings"]["bases_s"] = time.perf_counter() - started

        # Decode throughput: one decode step's routed experts of this layer (6), decoded and restored on this CPU's threads.
        started = time.perf_counter()
        sample = list(range(int(settings.get("throughput_experts", 6))))
        plan = {m: assignments[m]["xor"][strategy_name(assignments[m]["xor"], "A2")]["base"] for m in MATRICES}
        alone = {e: weights[e] for e in sample}
        deltas = {e: {m: weights[e][m] if plan[m][e] == -1 else bits.xor_delta(weights[e][m], weights[plan[m][e]][m]) for m in MATRICES} for e in sample}
        base_of = lambda e, m: None if plan[m][e] == -1 else weights[plan[m][e]][m]  # noqa: E731
        result["throughput"] = []
        for kind, transform, name, level in settings["throughput_configurations"]:
            codec = Codec(name, int(level))
            result["throughput"].append({**decode_throughput(alone, kind, transform, codec, threads, device=device), "representation": "independent"})
            result["throughput"].append({**decode_throughput(deltas, kind, transform, codec, threads, base_of, device=device), "representation": "A2-xor"})
        for entry in result["throughput"]:
            rates = ", ".join(f"{k} {v:.2f}" for k, v in entry["throughput"].items())
            print(f"layer {layer} throughput {entry['representation']} {entry['block']}/{entry['transform']}/{entry['codec']['name']}-"
                  f"{entry['codec']['level']} (GB/s): {rates}", flush=True)
        result["timings"]["throughput_s"] = time.perf_counter() - started
    except Inexact as failure:
        records.write_json(output / "failure.json", {"kind": "inexact", "layer": layer, "entry": records.strip(failure.args[0])})
        records.write_json(output / f"layer_{layer}.json", result)
        return 1
    from awpmi.storage.fileio import process_memory

    # Peak host memory (working set) and device memory: measured, excluded from digests ("system").
    result["system"] = {"process_memory": process_memory(), "peak_device_bytes": torch.cuda.max_memory_allocated(device)}
    records.write_json(output / f"layer_{layer}.json", result)
    return 0


def matrix_stored(patterns: np.ndarray, transform: str, codec: Codec, threads: int) -> tuple[int, bool]:
    """One matrix compressed as one block (its streams), decoded and compared: (stored bytes, exact)."""
    from expert_deltas.codecs import compress_many, decompress_many
    from expert_deltas.compression import restore

    parts = streams(patterns.reshape(-1), transform)
    frames = compress_many(parts, codec, threads=threads)
    back = restore(decompress_many(frames, [len(x) for x in parts], codec, threads=threads), patterns.size, transform)
    return sum(len(f) for f in frames), bool(np.array_equal(back.reshape(patterns.shape), patterns))


def strategy_name(found: dict, name: str) -> str:
    """The assignment key for a configured strategy name ("A1", "A2", "A3", "C-best": the cluster count with the smallest
    predicted bits)."""
    if name == "C-best":
        clusters = [k for k in found if k.startswith("C")]
        return min(clusters, key=lambda k: found[k]["predicted_bits"])
    matches = [k for k in found if k.split("-")[0] == name]
    if len(matches) != 1:
        raise KeyError(name)
    return matches[0]


# Stage: replay


def representations(layer_result: dict) -> dict[str, Representation]:
    """Per representation, one layer's stored bytes per expert and its base objects (the replay's input)."""
    experts, expert_bytes = layer_result["experts"], layer_result["expert_bytes"]
    matrix_bytes = expert_bytes // len(MATRICES)
    out = {"bf16": Representation("bf16", [expert_bytes] * experts)}
    for entry in layer_result["independent"]:
        name = f"independent/{label(entry)}"
        out[name] = Representation(name, list(entry["per_object_stored"]))
    for entry in layer_result["dictionary"]:
        name = f"dictionary/{entry['block']}/{entry['transform']}"
        out[name] = Representation(name, list(entry["per_object_stored"]))
    for entry in layer_result["bases"]:
        name = f"base/{entry['strategy']}/{entry['encoding']}/{label(entry)}"
        needs = [tuple((m, entry["base_of"][m][e]) for m in MATRICES if entry["base_of"][m][e] != -1) for e in range(experts)]
        stored = {tuple(key.split(":", 1)[0:1]) + (_base_id(key.split(":", 1)[1]),): size for key, size in entry["base_objects"].items()}
        resident = {key: matrix_bytes for key in stored}
        out[name] = Representation(name, list(entry["per_object_stored"]), needs, stored, resident)
    return out


def _base_id(text: str):
    return text if text == "median" else int(text)


def run_replay(config: dict, output: Path) -> int:
    from awpmi.tracing import read_jsonl

    settings = config["structure"]
    layers = [int(x) for x in settings["layers"]]
    results = {layer: json.loads((output / f"layer_{layer}.json").read_text(encoding="utf-8")) for layer in layers}
    trace = trace_steps(read_jsonl(records.REPO_ROOT / settings["traces"]), layers)
    total_layers = 26
    reps = {layer: representations(results[layer]) for layer in layers}
    names = sorted(set.intersection(*(set(r) for r in reps.values())))
    out = {"layers": layers, "steps": len(trace), "budgets": {}}
    for budget in settings["host_budgets_gb"]:
        per_layer = int(float(budget) * 1e9 / total_layers)
        out["budgets"][str(budget)] = {}
        for name in names:
            chosen = {layer: reps[layer][name] for layer in layers}
            modes = ["none"] if not chosen[layers[0]].base_ids else ["host", "bounded", "gpu"]
            for mode in modes:
                found = replay(trace, chosen, per_layer, "host" if mode == "none" else mode, int(settings.get("warmup_prompts", 1)))
                found["projected_steady_decode_bytes_all_layers"] = found["steady_decode_bytes"] * total_layers / len(layers)
                out["budgets"][str(budget)][f"{name}@{mode}"] = found
        print(f"replay budget {budget} GB done", flush=True)
    records.write_json(output / "replay.json", out)
    return 0


# Driver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(records.REPO_ROOT / "configs" / "phase5c-expert-deltas.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["all", "prepare", "layer", "replay", "report"], default="all")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--skip-verify", action="store_true", help="skip the file sha256 (development; recorded)")
    parser.add_argument("--experts", type=int, default=None, help="only the first N experts (development; recorded)")
    args = parser.parse_args()
    output = Path(args.output)
    if args.stage == "prepare" or args.stage == "all":
        if output.exists() and any(output.iterdir()):
            parser.error(f"{output} already exists and is not empty")
        output.mkdir(parents=True, exist_ok=True)
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        records.write_json(output / "environment.json", records.environment(config["model"]))
        if args.stage == "prepare":
            return 0
    config = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
    if args.stage == "layer":
        return run_layer(config, output, int(args.layer), args.skip_verify, args.experts)
    if args.stage == "replay":
        return run_replay(config, output)
    if args.stage == "report":
        from structure_report import report

        return report(config, output)
    command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output)]
    for layer in config["structure"]["layers"]:
        code = subprocess.run([*command, "--stage", "layer", "--layer", str(layer)] + (["--skip-verify"] if args.skip_verify else []), check=False).returncode
        if code:
            return code
    for stage in ("replay", "report"):
        code = subprocess.run([*command, "--stage", stage], check=False).returncode
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
