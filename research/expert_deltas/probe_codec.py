"""Stage 5C1-A: the codec probe (Phase 5C).

    python -m uv run python research/expert_deltas/probe_codec.py --output experiments/phase5c/probe-codec

On a few experts of one layer (config `probe`): the checkpoint's layout; the file verified against the publisher's
sha256; every tensor read by two independent paths (positioned reads, safetensors) and compared; the bit fields'
entropies; then every block size × exact transform × codec, each block compressed alone, decoded and reassembled, and
compared bit for bit with the safetensors copy (baselines B and C); trained Zstandard dictionaries on disjoint experts
(baseline D); a preview of XOR and modular deltas against the layer's first expert; and the floating-point delta
counterexample (BF16(base + delta) with a BF16 delta does not restore the weights). Stops at the first inexact
reconstruction (failure.json).

Writes probe_codec.json (everything; timings under "timings"/"throughput", excluded from its digest) and
probe_codec.md (a summary).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from expert_deltas import bits, census, records  # noqa: E402
from expert_deltas.codecs import Codec, Dictionary  # noqa: E402
from expert_deltas.compression import FILE_ORDER, blocks, measure, streams  # noqa: E402
from expert_deltas.source import MATRICES, Checkpoint, verify_files  # noqa: E402

class Inexact(Exception):
    pass


def float_delta_counterexample(target: np.ndarray, base: np.ndarray) -> dict:
    """BF16(base + delta): with the delta rounded to BF16 it does not restore the weights; with an exact float32 delta it
    happens to here, but needs twice the bytes (the brief's §4)."""
    w = bits.to_bfloat16(target).to(torch.float32)
    b = bits.to_bfloat16(base).to(torch.float32)
    delta32 = w - b  # exact in float32 for these magnitudes? counted below
    delta16 = delta32.to(torch.bfloat16)
    back16 = (b + delta16.to(torch.float32)).to(torch.bfloat16)
    back32 = (b + delta32).to(torch.bfloat16)
    orig = bits.to_bfloat16(target)
    differs16 = (back16.view(torch.int16) != orig.view(torch.int16))
    differs32 = (back32.view(torch.int16) != orig.view(torch.int16))
    return {"weights": int(orig.numel()), "bf16_delta_mismatches": int(differs16.sum()), "fp32_delta_mismatches": int(differs32.sum()),
            "bf16_delta_mismatch_share": float(differs16.float().mean())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(records.REPO_ROOT / "configs" / "phase5c-expert-deltas.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-verify", action="store_true", help="skip the file sha256 (development; recorded)")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    settings, model = config["probe"], config["model"]
    threads = int(settings["threads"])
    result: dict = {"stage": "5C1-A", "config": settings, "environment": records.environment(model), "timings": {}}
    checkpoint = Checkpoint(model["repository"], model["revision"])
    layer = int(settings["layer"])
    started = time.perf_counter()

    # Layout: every expert tensor of the layer, from the headers; adjacency and order as stored.
    count = checkpoint.experts(layer)
    tensors = [checkpoint.tensor(layer, e, m) for e in range(count) for m in MATRICES]
    by_expert = {e: {t.matrix: t for t in tensors if t.expert == e} for e in range(count)}
    adjacency = all(
        by_expert[e]["down"].offset + by_expert[e]["down"].nbytes == by_expert[e]["gate"].offset
        and by_expert[e]["gate"].offset + by_expert[e]["gate"].nbytes == by_expert[e]["up"].offset
        for e in range(count)
    )
    result["layout"] = {
        "layer": layer, "experts": count, "file": tensors[0].file, "dtype": sorted({t.dtype for t in tensors}),
        "shapes": {m: list(by_expert[0][m].shape) for m in MATRICES}, "expert_bytes": sum(t.nbytes for t in by_expert[0].values()),
        "file_order_down_gate_up_adjacent": adjacency, "first_expert": [by_expert[0][m].to_json() for m in FILE_ORDER],
    }
    if not args.skip_verify:
        result["verification"] = verify_files(checkpoint, [tensors[0].file])
    else:
        result["verification"] = "skipped (development)"
    result["timings"]["layout_and_verify_s"] = time.perf_counter() - started

    # The sample, by two independent read paths.
    started = time.perf_counter()
    experts, reference = {}, {}
    for e in settings["experts"]:
        experts[e] = {m: checkpoint.read_patterns(by_expert[e][m]) for m in MATRICES}
        reference[e] = {m: bits.patterns(checkpoint.read_reference(by_expert[e][m])) for m in MATRICES}
    result["read_paths_equal"] = all(np.array_equal(experts[e][m], reference[e][m]) for e in experts for m in MATRICES)
    for e in experts:
        for m in MATRICES:
            bits.require_finite(experts[e][m])
    result["finite"] = True
    result["sample_sha256"] = {f"{e}.{m}": hashlib.sha256(experts[e][m].tobytes()).hexdigest() for e in experts for m in MATRICES}
    result["timings"]["read_s"] = time.perf_counter() - started
    if not result["read_paths_equal"]:
        records.write_json(output / "failure.json", {"kind": "read_paths"})
        return 1

    # Bit fields' entropies (order 0), per matrix kind over the sample.
    result["census"] = {
        m: census.field_entropies(census.as_int32(np.stack([experts[e][m] for e in experts]))) for m in MATRICES
    }

    # Baselines B and C: every block kind × transform × codec.
    started = time.perf_counter()
    result["baselines"] = []
    try:
        for kind in settings["blocks"]:
            for transform in settings["transforms"]:
                for name, level in settings["codecs"]:
                    entry = measure(experts, reference, kind, transform, Codec(name, int(level)), threads)
                    result["baselines"].append(entry)
                    print(f"{kind:7s} {transform:10s} {name}-{level:<3d} ratio {entry['ratio']:.4f} exact {entry['exact']} "
                          f"{entry['timings']['compress_s']:.1f}s", flush=True)
                    if not entry["exact"]:
                        raise Inexact(entry)
        result["timings"]["baselines_s"] = time.perf_counter() - started

        # Baseline D: dictionaries trained on disjoint experts.
        started = time.perf_counter()
        rng = np.random.default_rng(int(settings["seed"]))
        train = {e: experts[e] for e in settings["dictionary_train_experts"]}
        evaluate = {e: experts[e] for e in settings["dictionary_eval_experts"]}
        evaluate_reference = {e: reference[e] for e in evaluate}
        result["dictionary"] = []
        for kind in ("row", "rows16"):
            for transform in ("raw", "byte_split"):
                pool = [s for mats in train.values() for _, p in blocks(mats, kind) for s in streams(p, transform)]
                chosen = rng.choice(len(pool), size=min(int(settings["dictionary_samples"]), len(pool)), replace=False)
                samples = [pool[i] for i in sorted(chosen.tolist())]
                for size in settings["dictionary_sizes"]:
                    dictionary = Dictionary.train(int(size), samples, level=3)
                    for level in (3, 19):
                        codec = Codec("zstd", level)
                        with_dict = measure(evaluate, evaluate_reference, kind, transform, codec, threads, dictionary)
                        without = measure(evaluate, evaluate_reference, kind, transform, codec, threads)
                        with_dict["without_dictionary_ratio"] = without["ratio"]
                        with_dict["dictionary_bytes"] = dictionary.nbytes
                        result["dictionary"].append(with_dict)
                        print(f"dict {size} {kind} {transform} zstd-{level}: {with_dict['ratio']:.4f} vs {without['ratio']:.4f}", flush=True)
                        if not with_dict["exact"]:
                            raise Inexact(with_dict)
        result["timings"]["dictionary_s"] = time.perf_counter() - started

        # Preview of base/delta: XOR and modular deltas of the sample's experts against its first expert (tensor blocks).
        started = time.perf_counter()
        first, others = settings["experts"][0], settings["experts"][1:]
        result["deltas"] = []
        for encoding in ("xor", "modular"):
            delta = bits.xor_delta if encoding == "xor" else bits.modular_delta
            undo = bits.xor_restore if encoding == "xor" else bits.modular_restore
            deltas = {e: {m: delta(experts[e][m], experts[first][m]) for m in MATRICES} for e in others}
            for transform in ("byte_split", "planes"):
                for level in (3, 19):
                    codec = Codec("zstd", level)
                    # The deltas' own compressed size (exactness of the delta stream), then the weights restored from it.
                    entry = measure(deltas, deltas, "tensor", transform, codec, threads)
                    restored = all(np.array_equal(undo(deltas[e][m], experts[first][m]), reference[e][m]) for e in others for m in MATRICES)
                    independent = measure({e: experts[e] for e in others}, {e: reference[e] for e in others}, "tensor", transform, codec, threads)
                    entry.update(encoding=encoding, base=first, restored_exact=bool(restored), independent_ratio=independent["ratio"])
                    result["deltas"].append(entry)
                    print(f"delta {encoding} {transform} zstd-{level}: {entry['ratio']:.4f} vs independent {independent['ratio']:.4f}", flush=True)
                    if not (entry["exact"] and restored):
                        raise Inexact(entry)
        result["timings"]["deltas_s"] = time.perf_counter() - started
    except Inexact as failure:
        records.write_json(output / "failure.json", {"kind": "inexact", "entry": records.strip(failure.args[0])})
        records.write_json(output / "probe_codec.json", result)
        return 1

    # The floating-point delta counterexample (expert 1 against expert 0).
    result["float_delta"] = {m: float_delta_counterexample(experts[settings["experts"][1]][m], experts[settings["experts"][0]][m]) for m in MATRICES}
    from awpmi.storage.fileio import process_memory

    result["system"] = {"process_memory": process_memory()}
    result["digest"] = records.digest({k: v for k, v in result.items() if k != "environment"})
    records.write_json(output / "probe_codec.json", result)
    (output / "probe_codec.md").write_text(summary(result), encoding="utf-8")
    print(f"wrote {output / 'probe_codec.json'}")
    return 0


def summary(result: dict) -> str:
    lines = ["# Phase 5C1-A codec probe", ""]
    layout = result["layout"]
    lines.append(f"Layer {layout['layer']}: {layout['experts']} experts in {layout['file']}, {layout['dtype']}, shapes {layout['shapes']}, "
                 f"{layout['expert_bytes']} bytes per expert; down, gate, up adjacent: {layout['file_order_down_gate_up_adjacent']}.")
    lines.append(f"Read paths equal: {result['read_paths_equal']}; finite: {result['finite']}; file verification: "
                 f"{json.dumps(result['verification']) if isinstance(result['verification'], str) else {k: v['equal'] for k, v in result['verification'].items()}}.")
    lines += ["", "## Entropies (bits per weight, order 0)", "", "| Matrix | sign | exponent | mantissa | high byte | low byte | byte split | planes |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m, h in result["census"].items():
        lines.append(f"| {m} | {h['sign']:.3f} | {h['exponent']:.3f} | {h['mantissa']:.3f} | {h['high_byte']:.3f} | {h['low_byte']:.3f} | "
                     f"{h['byte_split_total']:.3f} | {h['planes_total']:.3f} |")
    lines += ["", "## Independent compression (stored / BF16 bytes; all exact)", "", "| Block | Transform | " +
              " | ".join(f"{c[0]}-{c[1]}" for c in result["config"]["codecs"]) + " |", "| --- | --- |" + " --- |" * len(result["config"]["codecs"])]
    table: dict = {}
    for entry in result["baselines"]:
        table.setdefault((entry["block"], entry["transform"]), {})[f"{entry['codec']['name']}-{entry['codec']['level']}"] = entry
    for (kind, transform), row in table.items():
        lines.append(f"| {kind} | {transform} | " + " | ".join(f"{row[f'{c[0]}-{c[1]}']['ratio']:.4f}" for c in result["config"]["codecs"]) + " |")
    lines += ["", "Decompression throughput (6 threads, GB/s of BF16 out), tensor blocks:", ""]
    for entry in result["baselines"]:
        if entry["block"] == "tensor":
            lines.append(f"- {entry['transform']} {entry['codec']['name']}-{entry['codec']['level']}: {entry['throughput']['decompress_gb_s']:.2f} "
                         f"(1 thread {entry['throughput']['decompress_1t_gb_s']:.2f}); compression {entry['throughput']['compress_mb_s']:.0f} MB/s")
    lines += ["", "## Dictionaries (evaluation experts; dictionary bytes not included in the ratio)", ""]
    for entry in result["dictionary"]:
        lines.append(f"- {entry['block']} {entry['transform']} zstd-{entry['codec']['level']} dict {entry['dictionary_bytes']}: {entry['ratio']:.4f} "
                     f"(without {entry['without_dictionary_ratio']:.4f})")
    lines += ["", "## Deltas against the first expert (tensor blocks)", ""]
    for entry in result["deltas"]:
        lines.append(f"- {entry['encoding']} {entry['transform']} zstd-{entry['codec']['level']}: {entry['ratio']:.4f} vs independent "
                     f"{entry['independent_ratio']:.4f}; restored exactly: {entry['restored_exact']}")
    lines += ["", "## Floating-point deltas", ""]
    for m, f in result["float_delta"].items():
        lines.append(f"- {m}: BF16(base + BF16 delta) differs from the weight on {f['bf16_delta_mismatches']} of {f['weights']} "
                     f"({100 * f['bf16_delta_mismatch_share']:.2f}%); with a float32 delta on {f['fp32_delta_mismatches']}")
    lines += ["", f"Digest: `{result['digest']}`", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
