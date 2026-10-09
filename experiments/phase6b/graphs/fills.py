"""Phase 6B4 lead: which operations launch a Moonlight decode step's BF16 fill kernels (Phase 6A's trace: ~270 per token,
~28 ms of device time, outside every module scope). One 16-token prefill and two decode steps of Phase 4B's prompt 0, the
experts streamed from the index by the native core (no cache); the second decode step is traced with shapes and Python
stacks, and every operation whose kernels include a fill is listed with its shapes and innermost repository frames.

    .venv/Scripts/python.exe experiments/phase6b/graphs/fills.py > experiments/phase6b/graphs/fills.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from moonlight_runtime import DTYPES  # noqa: E402,F401  (configures the numerics first)

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

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
    out = {"model": model_config["repository"], "prompt_tokens": len(prompt), "fills": [], "by_operation": {}}
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([prompt], device=device), use_cache=True, logits_to_keep=1)
        token, cache_kv = output.logits[0, -1].argmax().view(1, 1), output.past_key_values
        output = model(input_ids=token, past_key_values=cache_kv, use_cache=True, logits_to_keep=1)
        token, cache_kv = output.logits[0, -1].argmax().view(1, 1), output.past_key_values
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
            model(input_ids=token, past_key_values=cache_kv, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize()
    totals: dict = defaultdict(lambda: {"launches": 0, "device_us": 0.0})
    kernels = Counter()
    for event in prof.events():
        fills = [k for k in event.kernels if "FillFunctor" in k.name]
        if not fills or any("FillFunctor" in k.name for c in event.cpu_children for k in c.kernels):
            continue  # the innermost operation that launched them
        frames = [f for f in (event.stack or []) if "weightsift" in f.replace("\\", "/") and ".venv" not in f][:3]
        library = [f for f in (event.stack or []) if "transformers" in f][:2]
        key = f"{event.name} {event.input_shapes} @ {(frames or library or ['?'])[0]}"
        totals[key]["launches"] += len(fills)
        totals[key]["device_us"] += sum(k.duration for k in fills)
        for k in fills:
            kernels[k.name.split("FillFunctor<")[-1][:30]] += 1
    out["by_operation"] = dict(sorted(totals.items(), key=lambda kv: -kv[1]["device_us"]))
    out["fill_types"] = dict(kernels)
    streamed.remove()
    store.close()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
