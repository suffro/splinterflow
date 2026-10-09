"""Phase 6B4 diagnostic: the device memory decode graphs take in Moonlight's streamed decode (each capture's allocated
bytes, before and after, and the step's peak), with the experts streamed natively from the index (no cache).

    .venv/Scripts/python.exe experiments/phase6b/graphs/memory.py > experiments/phase6b/graphs/memory.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from moonlight_runtime import DTYPES  # noqa: E402,F401  (configures the numerics first)

import torch  # noqa: E402

import awpmi.models.decode_graphs as graphs_module  # noqa: E402
from awpmi.materialization.backend import MaterializationBackend  # noqa: E402
from awpmi.materialization.weights import ExpertStore, WeightStore  # noqa: E402
from awpmi.models.checkpoint import checkpoint_sources, load_model_without_experts  # noqa: E402
from awpmi.models.moe import StreamedExperts, groups_from_pack  # noqa: E402
from awpmi.storage.pack import open_pack  # noqa: E402
from awpmi.streaming.streamer import PageStreamer  # noqa: E402
from awpmi.tracing import read_jsonl  # noqa: E402


def main() -> None:
    run = REPO_ROOT / "experiments/phase6a/native-run1"
    raw = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    model_config, storage = raw["model"], raw["storage"]
    device = torch.device("cuda", 0)
    torch.cuda.set_per_process_memory_fraction(int(raw["gpu_budget_bytes"]) / torch.cuda.get_device_properties(device).total_memory, device)
    prompt = read_jsonl(run / "prompts.jsonl")[0]["token_ids"][:16]
    pack = open_pack(REPO_ROOT / raw["index"]["directory"], verify="size")
    files = {key: entry.path for key, entry in checkpoint_sources(model_config["repository"], model_config["revision"], declared_sha256=False).items()}
    model, _ = load_model_without_experts(model_config["repository"], model_config["revision"], DTYPES[model_config["dtype"]], device, files)
    store = pack.store(backend="native", direct=True, alignment=int(storage["alignment"]), max_gap=int(storage["max_gap"]),
                       workers=int(storage["workers"]), max_read_bytes=int(storage["max_read_bytes"]), max_extent_bytes=int(storage["max_extent_bytes"]))
    backend = MaterializationBackend(store, device, PageStreamer(device, int(storage["slot_bytes"]), int(storage["slots"])))
    streamed = StreamedExperts(model, ExpertStore(WeightStore(backend), groups_from_pack(pack)), compact=True,
                               max_call_bytes=int(raw["call_budget_bytes"])).install()
    captures = []
    original = graphs_module._Graph.__init__

    def logged(self, function, inputs, pool, device, side):
        before = torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
        original(self, function, inputs, pool, device, side)
        after = torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
        captures.append({"allocated": after[0] - before[0], "reserved": after[1] - before[1]})

    graphs_module._Graph.__init__ = logged
    graphs = graphs_module.DecodeGraphs(model).install()
    out = {"steps": []}
    with torch.inference_mode():
        input_ids, cache_kv = torch.tensor([prompt], device=device), None
        for step in range(4):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.memory_allocated(device)
            output = model(input_ids=input_ids, past_key_values=cache_kv, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize()
            out["steps"].append({"step": step, "allocated_at_start": start, "allocated_after": torch.cuda.memory_allocated(device),
                                 "peak": torch.cuda.max_memory_allocated(device), "reserved": torch.cuda.memory_reserved(device)})
            input_ids, cache_kv = output.logits[0, -1].argmax().view(1, 1), output.past_key_values
            del output
    out["captures"] = captures
    out["capture_allocated_total"] = sum(c["allocated"] for c in captures)
    out["graphs_memory_bytes"] = graphs.memory_bytes
    out["capture_ms"] = graphs.capture_ms
    graphs.remove()
    streamed.remove()
    store.close()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
