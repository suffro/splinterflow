"""Stage 5C1-B report: the census, the bases, the replay and the gate (config `gate_5c1`).

    python -m uv run python research/expert_deltas/structure_run.py --output experiments/phase5c/<run> --stage report
    python -m uv run python research/expert_deltas/structure_report.py experiments/phase5c/<run> [--compare <run2>]

Reads layer_<L>.json and replay.json; writes summary.json, summary.md and digest.json (timings and throughputs excluded).
The gate compares, on the same routing trace and the same host-RAM budget, the steady-state drive bytes per decode token
of the best shared-base representation (bases resident in host RAM, pinned or bounded, counted against the budget) with
the best independently compressed exact representation (each expert alone or in independent pages, with or without a
trained dictionary). Bases held on the GPU are reported beside it: they use device memory the comparison does not grant
the independent representations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # summaries print Unicode on any console

from expert_deltas import records  # noqa: E402

TOTAL_LAYERS = 26


def label(entry: dict) -> str:
    return f"{entry['block']}/{entry['transform']}/{entry['codec']['name']}-{entry['codec']['level']}"


def load(run: Path) -> tuple[dict, dict[int, dict], dict | None]:
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    layers = {int(L): json.loads((run / f"layer_{L}.json").read_text(encoding="utf-8")) for L in config["structure"]["layers"]}
    replay_path = run / "replay.json"
    return config, layers, json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else None


def representation_bytes(layer: dict) -> dict[str, dict]:
    """Total derived on-disk bytes per representation of one layer: objects, base objects, an index entry per frame."""
    out = {"bf16": {"objects": layer["experts"] * layer["expert_bytes"], "bases": 0, "index": 0}}
    for entry in layer["independent"]:
        out[f"independent/{label(entry)}"] = {"objects": entry["stored_bytes"], "bases": 0, "index": 16 * entry["frames"]}
    for entry in layer["dictionary"]:
        out[f"dictionary/{entry['block']}/{entry['transform']}"] = {
            "objects": entry["stored_bytes"], "bases": 0, "index": 0, "dictionary": max(entry["dictionary_bytes"]),
        }
    for entry in layer["bases"]:
        name = f"base/{entry['strategy']}/{entry['encoding']}/{label(entry)}"
        out[name] = {"objects": entry["stored_bytes"], "bases": sum(entry["base_objects"].values()), "index": 16 * entry["frames"],
                     "base_count": len(entry["base_objects"])}
    for value in out.values():
        value["total"] = value["objects"] + value["bases"] + value["index"] + value.get("dictionary", 0)
        value["ratio"] = value["total"] / (layer["experts"] * layer["expert_bytes"])
    return out


def summarize(config: dict, layers: dict[int, dict], replay: dict | None) -> dict:
    gate = config["gate_5c1"]
    structure = config["structure"]
    summary: dict = {"layers": sorted(layers), "experts": {L: r["experts"] for L, r in layers.items()}}
    exact = all(
        r["read_paths_equal"]
        and all(e["exact"] for e in r["independent"])
        and all(e["exact"] for e in r["dictionary"])
        and all(e["exact"] and e["bases_exact"] for e in r["bases"])
        for r in layers.values()
    )
    verified = all(isinstance(r["verification"], dict) and all(v["equal"] for v in r["verification"].values()) for r in layers.values())
    summary["correctness"] = {"every_reconstruction_exact": exact, "files_equal_publisher_sha256": verified,
                              "development_limits": [r.get("development_expert_limit") for r in layers.values()]}
    # Census
    census = {}
    for L, r in layers.items():
        for m, kind in r["kinds"].items():
            c = kind["census"]
            census[f"{L}/{m}"] = {
                "bits_per_weight_alone": float(np.mean(kind["costs"]["xor"]["alone_bits"])),
                "exponent_bits": c["fields_per_expert_mean"]["exponent"],
                "exponent_context_saving_bits": c["exponent_context_saving_bits"],
                "context_description_bits": c["context_description_bits"],
                "top_eigenvalue_share": c["spectrum"]["top_eigenvalue_share"], "independent_share": c["spectrum"]["independent_share"],
                "mean_expert_energy_share": c["spectrum"]["mean_expert_energy_share"],
                "correlation_mean_abs": c["correlation"]["mean_abs"], "correlation_max_abs": c["correlation"]["max_abs"],
                "correlation_null_scale": c["correlation"]["null_scale"],
                "sign_agreement": c["agreement"]["sign"], "sign_agreement_expected": c["agreement"]["sign_expected_if_independent"],
                "exponent_agreement": c["agreement"]["exponent"], "exponent_agreement_expected": c["agreement"]["exponent_expected_if_independent"],
            }
    summary["census"] = census
    summary["permutation"] = {L: {k: r["permutation"][k] for k in ("mean_best", "max_best", "null_mean_best", "null_max_best")} for L, r in layers.items()}
    # Proxy predictions per strategy (bits per weight, averaged over the sampled layers and kinds)
    proxy: dict[str, list[float]] = {}
    pairs = []
    for L, r in layers.items():
        for m, kind in r["kinds"].items():
            for encoding, found in kind["assignments"].items():
                for name, entry in found.items():
                    proxy.setdefault(f"{name}/{encoding}", []).append(entry["predicted_bits"] - entry["predicted_alone_bits"])
                pairs.append({"layer": L, "kind": m, "encoding": encoding, **found["best_pair_bound"]})
            for encoding in ("xor", "modular"):
                synthetic = np.mean(kind["synthetic"][encoding]) - np.mean(kind["costs"]["xor"]["alone_bits"])
                proxy.setdefault(f"synthetic-median/{encoding}", []).append(float(synthetic))
    summary["proxy_extra_bits_vs_alone"] = {k: float(np.mean(v)) for k, v in proxy.items()}
    summary["best_pair_bound"] = {
        "experts_with_a_cheaper_delta": int(sum(p["experts_with_a_cheaper_delta"] for p in pairs)),
        "expert_kinds_examined": int(sum(r["experts"] for r in layers.values()) * 3 * 2),
        "largest_saving_bits": float(max(p["largest_saving_bits"] for p in pairs)),
    }
    # Actual bytes
    reps = {L: representation_bytes(r) for L, r in layers.items()}
    names = sorted(set.intersection(*(set(v) for v in reps.values())))
    summary["stored_ratio"] = {n: float(np.mean([reps[L][n]["ratio"] for L in layers])) for n in names}
    independent = [n for n in names if n.startswith(("independent/", "dictionary/"))]
    bases = [n for n in names if n.startswith("base/")]
    best_independent = min(independent, key=lambda n: summary["stored_ratio"][n])
    best_independent_expert = min((n for n in independent if "/expert/" in n), key=lambda n: summary["stored_ratio"][n])
    best_base = min(bases, key=lambda n: summary["stored_ratio"][n])
    summary["best"] = {
        "independent": best_independent, "independent_ratio": summary["stored_ratio"][best_independent],
        "independent_expert_block": best_independent_expert, "independent_expert_block_ratio": summary["stored_ratio"][best_independent_expert],
        "base": best_base, "base_ratio": summary["stored_ratio"][best_base],
        "base_extra_vs_best_independent": summary["stored_ratio"][best_base] / summary["stored_ratio"][best_independent] - 1.0,
    }
    # Throughput (decode of one step's routed experts of a layer)
    throughput = {}
    for L, r in layers.items():
        for entry in r.get("throughput", []):
            key = f"{entry['representation']}/{entry['block']}/{entry['transform']}/{entry['codec']['name']}-{entry['codec']['level']}"
            throughput.setdefault(key, []).append(entry["throughput"])
    summary["throughput"] = {k: {m: float(np.mean([v[m] for v in values])) for m in values[0]} for k, values in throughput.items()}
    # Replay and the gate
    verdict = {"exact": exact}
    if replay is not None:
        table = {}
        for budget, found in replay["budgets"].items():
            row = {}
            for name, entry in found.items():
                row[name] = {k: entry[k] for k in ("steady_decode_bytes", "projected_steady_decode_bytes_all_layers", "decode_bytes",
                                                   "prefill_bytes", "cold_start_bytes", "base_misses", "evictions", "bases_pinned_layers",
                                                   "host_resident_peak_bytes", "gpu_base_bytes", "hits", "lookups")}
            table[budget] = row
        summary["replay"] = table
        comparisons = {}
        for budget, row in table.items():
            ind = {n: v for n, v in row.items() if n.split("@")[0] in independent}
            host = {n: v for n, v in row.items() if n.startswith("base/") and not n.endswith("@gpu")}
            gpu = {n: v for n, v in row.items() if n.startswith("base/") and n.endswith("@gpu")}
            best_i = min(ind, key=lambda n: ind[n]["steady_decode_bytes"])
            best_h = min(host, key=lambda n: host[n]["steady_decode_bytes"])
            best_g = min(gpu, key=lambda n: gpu[n]["steady_decode_bytes"])
            reference = ind[best_i]["steady_decode_bytes"]
            comparisons[budget] = {
                "best_independent": best_i, "best_independent_bytes": reference,
                "best_base_host": best_h, "best_base_host_bytes": host[best_h]["steady_decode_bytes"],
                "reduction_host": 1.0 - host[best_h]["steady_decode_bytes"] / reference if reference else 0.0,
                "best_base_gpu": best_g, "best_base_gpu_bytes": gpu[best_g]["steady_decode_bytes"],
                "reduction_gpu": 1.0 - gpu[best_g]["steady_decode_bytes"] / reference if reference else 0.0,
                "bf16_bytes": row["bf16@none"]["steady_decode_bytes"],
                "reduction_best_independent_vs_bf16": 1.0 - reference / row["bf16@none"]["steady_decode_bytes"],
            }
        summary["comparisons"] = comparisons
        best_budget = max(comparisons, key=lambda b: comparisons[b]["reduction_host"])
        verdict["best_budget_gb"] = float(best_budget)
        verdict["reduction_vs_best_independent"] = comparisons[best_budget]["reduction_host"]
        verdict["reduction_met"] = verdict["reduction_vs_best_independent"] >= float(gate["min_steady_ssd_reduction_vs_best_independent"])
    # Base residency, projected to every layer: the best base strategy's decoded bases.
    best_base_layer = [next(e for e in r["bases"] if f"base/{e['strategy']}/{e['encoding']}/{label(e)}" == best_base) for r in layers.values()]
    resident = float(np.mean([len(e["base_objects"]) for e in best_base_layer])) * (layers[next(iter(layers))]["expert_bytes"] / 3) * TOTAL_LAYERS
    verdict["base_residency_gb_projected"] = resident / 1e9
    verdict["residency_met"] = verdict["base_residency_gb_projected"] <= float(gate["max_base_residency_gb"])
    best_codec = best_base.split("/", 3)[3]
    rates = [v for k, v in summary["throughput"].items() if k.startswith("A2-xor/") and k.endswith(best_codec)]
    if rates:
        decompress = rates[0]["decompress_gb_s"]
        restore = max(rates[0].get("cpu_restore_gb_s", 0.0), rates[0].get("gpu_restore_gb_s", 0.0))
        verdict["decode_throughput_gb_s"] = {"setting": best_codec, "decompress": decompress, "restore_best": restore}
        verdict["throughput_met"] = min(decompress, restore) >= float(gate["min_decode_throughput_gb_s"])
    else:
        verdict["decode_throughput_gb_s"] = f"not measured for {best_codec}"
        verdict["throughput_met"] = False
    verdict["pass"] = bool(verdict["exact"] and verdict.get("reduction_met") and verdict["residency_met"] and verdict.get("throughput_met", False))
    summary["gate"] = verdict
    # Strategy D's trigger (a shared predictor), from the census and the measured strategies.
    trigger = structure["predictor_trigger"]
    shared_variance = max(c["top_eigenvalue_share"] for c in census.values())
    context = max(max(c["exponent_context_saving_bits"][k] - c["context_description_bits"][k] for k in c["exponent_context_saving_bits"]) for c in census.values())
    saving = -summary["best"]["base_extra_vs_best_independent"]
    summary["predictor_trigger"] = {
        "best_strategy_saving": saving, "max_top_eigenvalue_share": shared_variance, "max_net_context_saving_bits": context,
        "triggered": bool(saving >= trigger["min_saving"] or shared_variance >= trigger["min_shared_variance"] or context >= trigger["min_context_bits"]),
    }
    return summary


def markdown(summary: dict) -> str:
    lines = ["# Phase 5C1-B: exact structural reuse (summary)", ""]
    c = summary["correctness"]
    lines.append(f"Layers {summary['layers']}; every reconstruction exact: {c['every_reconstruction_exact']}; files equal to the "
                 f"publisher's sha256: {c['files_equal_publisher_sha256']}.")
    lines += ["", "## Shared structure (census)", "",
              "| Layer/kind | bits/weight alone | top eigenvalue share (independent 1/E) | mean absolute correlation (max) | sign agreement (expected) | exponent agreement (expected) | best exponent context saving − its cost (bits) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for key, v in summary["census"].items():
        net = max(v["exponent_context_saving_bits"][k] - v["context_description_bits"][k] for k in v["exponent_context_saving_bits"])
        lines.append(f"| {key} | {v['bits_per_weight_alone']:.3f} | {v['top_eigenvalue_share']:.4f} ({v['independent_share']:.4f}) | "
                     f"{v['correlation_mean_abs']:.5f} ({v['correlation_max_abs']:.4f}) | {v['sign_agreement']:.5f} ({v['sign_agreement_expected']:.5f}) | "
                     f"{v['exponent_agreement']:.5f} ({v['exponent_agreement_expected']:.5f}) | {net:+.4f} |")
    lines += ["", "Neuron-permutation diagnostic (best absolute cosine of a neuron against another expert's, mean; Gaussian null):", ""]
    for L, v in summary["permutation"].items():
        lines.append(f"- layer {L}: mean {v['mean_best']:.4f}, max {v['max_best']:.4f}; null mean {v['null_mean_best']:.4f}, max {v['null_max_best']:.4f}")
    lines += ["", "## Proxy cost of base strategies (extra bits per weight against storing every expert alone; > 0 is worse)", ""]
    for k, v in sorted(summary["proxy_extra_bits_vs_alone"].items()):
        lines.append(f"- {k}: {v:+.4f}")
    b = summary["best_pair_bound"]
    lines.append(f"- every expert at its best actual-expert base: {b['experts_with_a_cheaper_delta']} of {b['expert_kinds_examined']} "
                 f"(expert, kind, encoding) have a cheaper delta than alone; largest saving {b['largest_saving_bits']:+.4f} bits")
    lines += ["", "## Stored bytes (derived on disk / BF16, mean over layers; bases and index included)", ""]
    for k, v in sorted(summary["stored_ratio"].items(), key=lambda kv: kv[1]):
        lines.append(f"- {k}: {v:.4f}")
    best = summary["best"]
    lines += ["", f"Best independent: {best['independent']} {best['independent_ratio']:.4f}; best shared base: {best['base']} "
              f"{best['base_ratio']:.4f} ({100 * best['base_extra_vs_best_independent']:+.2f}%).", ""]
    if "comparisons" in summary:
        lines += ["## Replay (steady-state drive bytes per decode token, sampled layers; projection to 26 layers)", "",
                  "| Host budget (GB) | BF16 | best independent | best shared base (host) | reduction | best shared base (GPU) | reduction |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for budget, v in summary["comparisons"].items():
            lines.append(f"| {budget} | {v['bf16_bytes'] / 1e6:.1f} MB | {v['best_independent_bytes'] / 1e6:.1f} MB | {v['best_base_host_bytes'] / 1e6:.1f} MB | "
                         f"{100 * v['reduction_host']:+.1f}% | {v['best_base_gpu_bytes'] / 1e6:.1f} MB | {100 * v['reduction_gpu']:+.1f}% |")
    lines += ["", "## Decode throughput (one step's routed experts of a layer, GB/s of BF16 out)", ""]
    for k, v in summary["throughput"].items():
        lines.append(f"- {k}: " + ", ".join(f"{m} {x:.2f}" for m, x in v.items()))
    g = summary["gate"]
    lines += ["", "## Gate 5C1", "", f"```\n{json.dumps(g, indent=1)}\n```", "",
              f"Strategy D trigger: {json.dumps(summary['predictor_trigger'])}", ""]
    return "\n".join(lines)


def report(config: dict, run: Path) -> int:
    _, layers, replay = load(run)
    summary = summarize(config, layers, replay)
    summary["digest"] = records.digest({"layers": {L: r for L, r in layers.items()}, "replay": replay})
    records.write_json(run / "summary.json", summary)
    (run / "summary.md").write_text(markdown(summary), encoding="utf-8")
    records.write_json(run / "digest.json", {"layers_and_replay": summary["digest"], "excluded_fields": list(records.EXCLUDED_FIELDS)})
    print(markdown(summary))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--compare", default=None)
    args = parser.parse_args()
    run = Path(args.run)
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    code = report(config, run)
    if args.compare:
        mine = json.loads((run / "digest.json").read_text(encoding="utf-8"))
        theirs = json.loads((Path(args.compare) / "digest.json").read_text(encoding="utf-8"))
        print(f"digests equal: {mine['layers_and_replay'] == theirs['layers_and_replay']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
