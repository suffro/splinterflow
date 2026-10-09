"""Phase 4B profile: where a streamed Moonlight step spends its time, prefill and decode (decision 0008).

    uv run python benchmarks/moonlight_profile.py --output experiments/phase4b/<run>/profile-<configuration>.json
        --configuration stream [--prompts 0 7] [--decode-steps 8] [--trace]

One configuration per process (a cache must start empty, and a process ends with all its memory returned). The
correctness benchmark (`moonlight_runtime.py`) hashes every intermediate inside the timed steps; this script runs the
same configuration recording nothing but time. Host wall time per step is split with timers around the generic path
(no model code changes; nested regions are not counted twice):

  route    the experts pre-hook waiting for the router's choice (the GPU finishing attention and router, then the
           device-to-host sync of the routed experts)
  cache    cache lookups, and the device copies of cached rows ("hits first")
  admit    cache admission: copies of fetched rows into the cache's pool, and evictions
  plan     read planning (runs, alignment, extents, pieces)
  io       positioned reads in flight (8 threads, direct I/O; the store's own clock): the drive
  gather   host gathers in pinned staging
  h2d      host time issuing the copies (their device time is reported separately)
  assemble the rest of expert assembly (buffers, bookkeeping)
  chunks   the chunked grouped GEMM's own host work (decision 0008), its reads excluded
  other    everything else: Python and kernel launches of attention, norms, routers, the experts' and shared experts'
           compute and the LM head, and the GPU work the host waits for at the end of the step

Phase 6A (decision 0012): `--config` names the file the configuration comes from (default: the run's config.yaml;
configs/phase6a-native.yaml lists its profile configurations), whose `backend` is python or native and whose
`host_cache_bytes` sizes the native host cache. A native transfer's planning and cache lookups (in the core, called from
Python) count as `plan`, and the time Python waits for its pieces as `io`. `--warm` first runs the config's
`profile.warm_prompts` (not measured), so caches start as a serving process's would. Every step also records the main
thread's CPU time (the Python and FFI overhead: waits excluded), the host cache's counters, and the process's memory.

With --trace, a torch.profiler trace of the prefill and two decode steps of the first prompt gives device time by
kernel kind (GEMM, attention, other kernels, host-to-device and device-to-device copies) and by module scope
(attention, router, experts, shared experts, LM head), the number of kernel launches and their CPU time, and the GPU's
busy time (union of its activities) against the step's wall time. Timings only: nothing here is digested.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from moonlight_runtime import DTYPES, NUMERICS_FLAGS, device_cache, load_adapter  # noqa: E402  (configures the numerics first)

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

import awpmi.materialization.backend as backend_module  # noqa: E402
import awpmi.models.moe as moe_module  # noqa: E402
import awpmi.storage.cache as cache_module  # noqa: E402
import awpmi.storage.native as native_module  # noqa: E402
import awpmi.storage.store as store_module  # noqa: E402
import awpmi.streaming.codec as codec_module  # noqa: E402
import awpmi.streaming.streamer as streamer_module  # noqa: E402
from awpmi.materialization.backend import MaterializationBackend  # noqa: E402
from awpmi.materialization.weights import ExpertStore, WeightStore  # noqa: E402
from awpmi.models.checkpoint import checkpoint_sources, load_model_without_experts  # noqa: E402
from awpmi.models.decode_graphs import DecodeGraphs  # noqa: E402
from awpmi.models.moe import StreamedExperts, groups_from_pack  # noqa: E402
from awpmi.storage.fileio import process_memory  # noqa: E402
from awpmi.storage.pack import open_pack  # noqa: E402
from awpmi.streaming.streamer import PageStreamer  # noqa: E402
from awpmi.tracing import environment_metadata, read_jsonl  # noqa: E402


class Clock:
    """Accumulated host seconds per region, with nesting: an inner region's time is not its outer one's."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self._stack: list[list] = []

    def wrap(self, region: str, function):
        def timed(*args, **kwargs):
            started = time.perf_counter()
            self._stack.append([region, 0.0])
            try:
                return function(*args, **kwargs)
            finally:
                _, inner = self._stack.pop()
                elapsed = time.perf_counter() - started
                self.totals[region] += elapsed - inner
                if self._stack:
                    self._stack[-1][1] += elapsed

        return timed

    def reset(self) -> dict[str, float]:
        totals = dict(self.totals)
        self.totals.clear()
        return totals

    def wrap_generator(self, region: str, function):
        """A generator function whose every step (one next) is timed as `region`."""
        clock = self

        def timed(*args, **kwargs):
            iterator = function(*args, **kwargs)
            while True:
                try:
                    value = clock.wrap(region, next)(iterator)
                except StopIteration:
                    return
                yield value

        return timed


def instrument(clock: Clock) -> None:
    """Timers around the generic path's stages (module functions and methods, wrapped in place)."""
    moe_module.routed_experts = clock.wrap("route", moe_module.routed_experts)
    moe_module._chunked_grouped_mm = clock.wrap("chunks", moe_module._chunked_grouped_mm)
    store_module.plan_reads = clock.wrap("plan", store_module.plan_reads)
    streamer_module.PageStreamer._pieces = clock.wrap("plan", streamer_module.PageStreamer._pieces)
    store_module.FileBackedPageStore.read_extents = clock.wrap("io", store_module.FileBackedPageStore.read_extents)
    streamer_module.gather_runs = clock.wrap("gather", streamer_module.gather_runs)
    streamer_module.PageStreamer._copy = clock.wrap("h2d", streamer_module.PageStreamer._copy)
    backend_module.PageCache.get_many = clock.wrap("cache", backend_module.PageCache.get_many)
    backend_module.MaterializationBackend._cache_copy = clock.wrap("admit", backend_module.MaterializationBackend._cache_copy)
    cache_module.PageCache.put = clock.wrap("admit", cache_module.PageCache.put)
    cache_module.SlotCache.get_many = clock.wrap("cache", cache_module.SlotCache.get_many)  # Phase 6B: a device cache in slots
    cache_module.SlotCache.put = clock.wrap("admit", cache_module.SlotCache.put)
    ExpertStore.assemble = clock.wrap("assemble", ExpertStore.assemble)
    # The native path (Phase 6A): submit plans and looks the cache up in the core; Python then waits for pieces.
    native_module.NativeTransfer.__init__ = clock.wrap("plan", native_module.NativeTransfer.__init__)
    native_module.NativeTransfer.__iter__ = clock.wrap_generator("io", native_module.NativeTransfer.__iter__)
    native_module.NativeTransfer.close = clock.wrap("io", native_module.NativeTransfer.close)
    # Phase 6B (decision 0013): building and launching the GPU decoding of encoded rows (host time; the kernels' device
    # time is in the trace).
    codec_module.RowDecoder.decode = clock.wrap("decode", codec_module.RowDecoder.decode)


def scopes(model, adapter, experts: list[str], graphed: bool = False) -> list:
    """record_function scopes around the modules whose device time the trace attributes (with decode graphs, only the
    modules that still run eagerly: the routed experts and the LM head; the graphed ones replay outside any scope)."""
    targets = {"lm_head": model.lm_head}
    for name, module in model.named_modules():
        if name.endswith(".self_attn") and not graphed:
            targets[name] = module
    for name in experts:
        head = name[: -len(adapter.EXPERTS_SUFFIX)]
        targets[name] = model.get_submodule(name)
        if not graphed:
            targets[head + adapter.ROUTER_SUFFIX] = model.get_submodule(head + adapter.ROUTER_SUFFIX)
            targets[head + adapter.SHARED_SUFFIX] = model.get_submodule(head + adapter.SHARED_SUFFIX)

    def kind(name: str) -> str:
        if name == "lm_head":
            return "scope:lm_head"
        if name.endswith(".self_attn"):
            return "scope:attention"
        if name.endswith(adapter.EXPERTS_SUFFIX):
            return "scope:experts"
        if name.endswith(adapter.ROUTER_SUFFIX):
            return "scope:router"
        return "scope:shared_experts"

    handles, stack = [], []
    for name, module in targets.items():
        label = kind(name)

        def enter(module_, args, label=label):
            scope = record_function(label)
            scope.__enter__()
            stack.append(scope)

        def leave(module_, args, output):
            stack.pop().__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(enter))
        handles.append(module.register_forward_hook(leave, always_call=True))
    return handles


def kernel_kind(name: str) -> str:
    lowered = name.lower()
    if "memcpy htod" in lowered or "memcpy h2d" in lowered:
        return "memcpy_h2d"
    if "memcpy dtod" in lowered or "memcpy d2d" in lowered:
        return "memcpy_d2d"
    if "memcpy" in lowered or "memset" in lowered:
        return "memcpy_other"
    if any(k in lowered for k in ("flash", "fmha", "attention", "sdpa")):
        return "attention_kernels"
    if any(k in lowered for k in ("gemm", "cutlass", "gemv", "sm80_xmma", "sm89", "ampere", "cublas", "s16816")):
        return "gemm"
    return "other_kernels"


def summarize_trace(prof, wall_ms: float) -> dict:
    """Device time by kind and by module scope, launches, and the GPU's busy time.

    The scopes' own GPU-side annotations (ranges from a scope's first to last kernel, gaps included) are not activities:
    they only attribute each kernel or copy, by its start time, to the scope whose range holds it.
    """
    import bisect

    kinds: dict[str, float] = defaultdict(float)
    scopes_ms: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_name: dict[str, list] = defaultdict(lambda: [0, 0.0])  # kernel name: launches, device ms (Phase 6A's kernel inventory)
    annotations, activities = [], []
    launches = graph_launches = device_kernels = 0
    launch_cpu_ms = 0.0
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            if event.name in ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cudaLaunchKernelEx"):
                launches += 1
                launch_cpu_ms += event.self_cpu_time_total / 1e3
            elif event.name in ("cudaGraphLaunch", "cuGraphLaunch"):  # Phase 6B4: one host call, many kernels
                graph_launches += 1
                launch_cpu_ms += event.self_cpu_time_total / 1e3
            continue
        if not event.name.startswith("scope:") and "memcpy" not in event.name.lower() and "memset" not in event.name.lower():
            device_kernels += 1
        span = (event.time_range.start, event.time_range.end)
        if event.name.startswith("scope:"):
            annotations.append((*span, event.name[6:]))
        else:
            activities.append((*span, kernel_kind(event.name)))
            by_name[event.name][0] += 1
            by_name[event.name][1] += (span[1] - span[0]) / 1e3
    annotations.sort()
    starts = [a[0] for a in annotations]
    intervals = []
    for start, end, kind in activities:
        kinds[kind] += (end - start) / 1e3
        intervals.append((start, end))
        index = bisect.bisect_right(starts, start) - 1
        label = annotations[index][2] if index >= 0 and start < annotations[index][1] else "outside_scopes"
        scopes_ms[label][kind] += (end - start) / 1e3
    intervals.sort()
    busy, end = 0.0, None
    for start, stop in intervals:
        if end is None or start > end:
            busy += stop - start
            end = stop
        elif stop > end:
            busy += stop - end
            end = stop
    busy_ms = busy / 1e3
    return {
        "device_ms_by_kind": dict(kinds),
        "device_ms_by_scope": {label: dict(values) for label, values in scopes_ms.items()},
        "kernel_launches": launches,
        "graph_launches": graph_launches,
        "device_kernels": device_kernels,  # kernels the GPU ran, launched one by one or by a graph
        "launch_cpu_ms": launch_cpu_ms,
        "gpu_busy_ms": busy_ms,
        "kernels": [
            {"name": name, "launches": count, "device_ms": ms}
            for name, (count, ms) in sorted(by_name.items(), key=lambda item: -item[1][1])[:60]
        ],
        "wall_ms": wall_ms,
        "gpu_idle_fraction": max(0.0, 1.0 - busy_ms / wall_ms) if wall_ms else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="a moonlight_runtime run directory (its config and prompts are used)")
    parser.add_argument("--output", required=True, help="the profile JSON file to write")
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--prompts", type=int, nargs="+", default=[0, 3, 5, 7], help="indices into the run's prompts")
    parser.add_argument("--decode-steps", type=int, default=None)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--config", default=None, help="the file of the configuration (default: the run's config.yaml)")
    parser.add_argument("--warm", action="store_true", help="first run the config's profile.warm_prompts, not measured")
    args = parser.parse_args()
    run = Path(args.run)
    raw = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    chosen = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if args.config else raw
    model_config, storage = raw["model"], raw["storage"]
    adapter = load_adapter(model_config)
    device = torch.device("cuda", torch.cuda.current_device())
    time.sleep(float(raw["timing"]["settle_seconds"]))
    torch.cuda.set_per_process_memory_fraction(int(raw["gpu_budget_bytes"]) / torch.cuda.get_device_properties(device).total_memory, device)
    all_prompts = read_jsonl(run / "prompts.jsonl")
    prompts = [all_prompts[i] for i in args.prompts]
    steps = int(args.decode_steps if args.decode_steps is not None else raw["prompts"]["decode_steps"])
    pack = open_pack(REPO_ROOT / raw["index"]["directory"], verify="size")
    groups = groups_from_pack(pack)
    files = {key: entry.path for key, entry in checkpoint_sources(model_config["repository"], model_config["revision"], declared_sha256=False).items()}
    model, _ = load_model_without_experts(model_config["repository"], model_config["revision"], DTYPES[model_config["dtype"]], device, files)
    expert_row = max(sum(pack.segments[s].row_bytes for s in group.segments.values()) for group in groups.values())
    entries = {e["name"]: e for e in [*chosen.get("configurations", []), *chosen.get("profile", {}).get("configurations", [])]}
    entry = entries[args.configuration]
    capacity = int(entry.get("cache_experts", 0)) * expert_row
    if "device_cache_bytes" in entry:  # Phase 6B: a device cache of encoded rows, in stored bytes
        capacity = int(float(entry["device_cache_bytes"]))
    call_budget = int(entry["call_budget_experts"]) * expert_row if "call_budget_experts" in entry else entry.get("call_budget_bytes", raw.get("call_budget_bytes"))
    store_options = dict(
        direct=True, alignment=int(storage["alignment"]), max_gap=int(storage["max_gap"]), workers=int(storage["workers"]),
        max_read_bytes=int(storage["max_read_bytes"]), max_extent_bytes=int(storage["max_extent_bytes"]),
    )
    native = entry.get("backend", "python") == "native"
    # Phase 6B (decision 0013): PyTorch's NaN fill of uninitialized memory (on with deterministic algorithms) can be turned
    # off per configuration; it changes no arithmetic, only what memory nothing reads holds.
    fill = bool(entry.get("fill_uninitialized_memory", True))
    torch.utils.deterministic.fill_uninitialized_memory = fill
    encoded = None
    if entry.get("experts") == "encoded":  # Phase 6B: the encoded pack's rows, decoded on the GPU (decision 0013)
        from awpmi.storage.encoded import open_encoded

        encoded = open_encoded(REPO_ROOT / chosen["encoded"]["directory"], verify="size")
    source = pack if encoded is None else encoded.pack
    cache = device_cache(entry, capacity, float(raw["hotness_half_life"]), encoded, device) if capacity else None
    if native:
        host_cache = int(entry["host_cache_experts"]) * expert_row if "host_cache_experts" in entry else int(float(entry.get("host_cache_bytes", 0)))
        store = source.store(backend="native", host_cache_bytes=host_cache, **store_options)
    else:
        store = source.store(**store_options)
    freeze_prefill = bool(entry.get("freeze_prefill", False))
    native_slots = int(chosen.get("storage", storage).get("native_slots", 4))
    streamer = PageStreamer(device, int(storage["slot_bytes"]), int(storage["slots"]), native_slots=native_slots)
    decoder = None if encoded is None else codec_module.RowDecoder(encoded.encodings, device)
    backend = MaterializationBackend(store, device, streamer, cache, decoder=decoder)
    clock = Clock()
    instrument(clock)
    experts = ExpertStore(WeightStore(backend), groups)
    streamed = StreamedExperts(
        model, experts, compact=True, all_experts=bool(entry.get("all_experts", False)), max_call_bytes=call_budget,
        prefetch_chunks=bool(entry.get("prefetch_chunks", False)),
    ).install()
    names = [m.name for m in streamed.modules]
    graphs = DecodeGraphs(model).install() if entry.get("decode_graphs") else None  # Phase 6B4 (decision 0013)
    result = {"gpu": torch.cuda.get_device_name(device), "configuration": args.configuration, "prompts": [p["prompt_id"] for p in prompts],
              "lengths": [p["length"] for p in prompts], "decode_steps": steps, "call_budget_bytes": call_budget, "cache_capacity_bytes": capacity,
              "backend": "native" if native else "python", "host_cache_bytes": store.host_cache_bytes if native else 0,
              "native_slots": native_slots, "freeze_prefill": freeze_prefill, "warm": args.warm, "memory_after_load": process_memory(),
              "experts": "encoded" if encoded is not None else "bf16",
              "fill_uninitialized_memory": fill, "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
              "device_cache_slots": {str(size): count for size, count in cache.slots.items()} if cache is not None and cache.copies else None,
              "encoding": None if encoded is None else {**encoded.metadata["encoding"], "stored_ratio": encoded.stored_ratio},
              "environment": environment_metadata(REPO_ROOT, {"repository": model_config["repository"], "revision": model_config["revision"]}, NUMERICS_FLAGS)}
    records: list[dict] = []
    traces: list[dict] = []
    # Warm-up: one short prefill and a decode step (kernels, allocator), not recorded.
    with torch.inference_mode():
        warm = model(input_ids=torch.tensor([all_prompts[0]["token_ids"][:16]], device=device), use_cache=True, logits_to_keep=1)
        model(input_ids=warm.logits[0, -1].argmax().view(1, 1), past_key_values=warm.past_key_values, use_cache=True, logits_to_keep=1)
    del warm
    if cache is not None:  # the warm-up must not leave anything in the measured cache
        cache.clear()
        cache.stats.reset()
    if native:
        store.clear_cache()
    if args.warm:
        # A serving process's caches: the warm prompts, in full, before the measured ones (nothing recorded).
        with torch.inference_mode():
            for index in chosen["profile"]["warm_prompts"]:
                input_ids, cache_kv = torch.tensor([all_prompts[index]["token_ids"]], device=device), None
                for step in range(steps + 1):
                    if native and freeze_prefill:
                        store.set_admit(step > 0)
                        if cache is not None and encoded is not None:
                            cache.admit = step > 0
                    output = model(input_ids=input_ids, past_key_values=cache_kv, use_cache=True, logits_to_keep=1)
                    input_ids, cache_kv = output.logits[0, -1].argmax().view(1, 1), output.past_key_values
                del output, cache_kv
        result["after_warm"] = {"host_cache": store.cache_stats() if native else None, "memory": process_memory()}
    gc.collect()
    torch.cuda.synchronize(device)
    with torch.inference_mode():
        for index, prompt in enumerate(prompts):
            input_ids = torch.tensor([prompt["token_ids"]], device=device)
            cache_kv = None
            for step in range(steps + 1):
                trace = args.trace and index == 0 and step <= 2
                handles = scopes(model, adapter, names, graphed=graphs is not None) if trace else []
                if native and freeze_prefill:
                    store.set_admit(step > 0)
                    if cache is not None and encoded is not None:
                        cache.admit = step > 0
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                backend.reset_stats()
                clock.reset()
                calls = (streamed.calls, streamed.chunked_calls)
                profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) if trace else None
                if profiler is not None:
                    profiler.__enter__()
                started, cpu_started = time.perf_counter(), time.thread_time()
                output = model(input_ids=input_ids, past_key_values=cache_kv, use_cache=True, logits_to_keep=1)
                token = output.logits[0, -1].argmax().view(1, 1)
                torch.cuda.synchronize(device)
                wall = (time.perf_counter() - started) * 1e3
                main_cpu = (time.thread_time() - cpu_started) * 1e3
                if profiler is not None:
                    profiler.__exit__(None, None, None)
                    traces.append({"step": step, "phase": "prefill" if step == 0 else "decode", **summarize_trace(profiler, wall)})
                for handle in handles:
                    handle.remove()
                regions = {k: v * 1e3 for k, v in clock.reset().items()}
                report = backend.report()
                records.append({
                    "prompt": prompt["prompt_id"], "length": prompt["length"], "step": step, "phase": "prefill" if step == 0 else "decode",
                    "wall_ms": wall, "regions_ms": regions, "store_io_ms": report["storage"]["io_ms"], "h2d_device_ms": report["transfer"]["copy_ms"],
                    "physical_bytes": report["storage"]["physical_bytes"], "h2d_bytes": report["transfer"]["h2d_bytes"],
                    "read_calls": report["storage"]["read_calls"], "pieces": report["transfer"]["pieces"], "h2d_copies": report["transfer"]["h2d_copies"],
                    "requests": report["materialization"]["requests"], "chunked_calls": streamed.chunked_calls - calls[1],
                    "cache_hits": None if cache is None else report["cache"]["hits"], "traced": trace,
                    # Phase 6B: the device cache (encoded rows: stored bytes) and the allocator's reservation (its pool too).
                    "cache": None if cache is None else {
                        **{k: report["cache"][k] for k in ("lookups", "hits", "misses", "hit_bytes", "miss_bytes", "inserts", "evictions", "bypassed")},
                        "resident_bytes": cache.resident_bytes,
                    },
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "main_thread_cpu_ms": main_cpu, "requested_bytes": report["materialization"]["requested_bytes"],
                    "fetched_bytes": report["materialization"]["fetched_bytes"],
                    "extents": report["storage"]["extents"], "peak_device_bytes": torch.cuda.max_memory_allocated(device),
                    "memory": process_memory(),
                    "host_cache": None if "host_cache" not in report["storage"] else {
                        k: report["storage"]["host_cache"][k]
                        for k in ("lookups", "hits", "waits", "misses", "hit_bytes", "wait_bytes", "miss_bytes", "evictions", "resident_bytes",
                                  "prefetch_fills", "prefetch_used", "prefetch_wasted", "held_bytes", "peak_held_bytes", "recycled_bytes",
                                  "allocated_bytes", "released_bytes")
                        if k in report["storage"]["host_cache"]
                    },
                    "native": report["storage"].get("native"),
                    "decoder": report.get("decoder"),
                    "stored_bytes": report["materialization"].get("stored_requested_bytes", 0),
                })
                cache_kv = output.past_key_values
                input_ids = token
    if graphs is not None:
        result["decode_graphs"] = {"captures": graphs.captures, "replays": graphs.replays, "eager_steps": graphs.eager_steps, "memory_bytes": graphs.memory_bytes, "capture_ms": graphs.capture_ms}
        graphs.remove()
    streamed.remove()
    store.close()
    backend.streamer.close()
    result["summary"] = summarize(records)
    result["traces"] = traces
    result["steps"] = records
    Path(args.output).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "traces": traces}, indent=1), flush=True)
    return 0


def summarize(steps: list[dict]) -> dict:
    out = {}
    for phase in ("prefill", "decode"):
        chosen = [s for s in steps if s["phase"] == phase and not s["traced"]]
        if not chosen:
            continue
        regions = sorted({k for s in chosen for k in s["regions_ms"]})
        mean = lambda key: statistics.mean(s[key] for s in chosen)  # noqa: E731
        entry = {
            "steps": len(chosen),
            "wall_ms": mean("wall_ms"),
            "regions_ms": {k: statistics.mean(s["regions_ms"].get(k, 0.0) for s in chosen) for k in regions},
            "h2d_device_ms": mean("h2d_device_ms"),
            "physical_mb": mean("physical_bytes") / 1e6,
            "read_calls": mean("read_calls"),
            "pieces": mean("pieces"),
            "h2d_copies": mean("h2d_copies"),
            "requests": mean("requests"),
            "chunked_calls": mean("chunked_calls"),
        }
        accounted = sum(entry["regions_ms"].values())
        entry["regions_ms"]["other"] = entry["wall_ms"] - accounted
        entry["share"] = {k: v / entry["wall_ms"] for k, v in entry["regions_ms"].items()}
        entry["drive_gb_per_s"] = entry["physical_mb"] / 1e3 / max(1e-9, entry["regions_ms"].get("io", 0.0) / 1e3)
        # Phase 6A: per-step distribution, the main thread's CPU time, the host cache, the drive's busy time, memory.
        walls = sorted(s["wall_ms"] for s in chosen)
        entry["wall_ms_median"] = statistics.median(walls)
        entry["wall_ms_p95"] = walls[int(0.95 * (len(walls) - 1))]
        entry["tokens_per_s"] = 1e3 / entry["wall_ms"] if phase == "decode" else None
        if all("main_thread_cpu_ms" in s for s in chosen):
            entry["main_thread_cpu_ms"] = mean("main_thread_cpu_ms")
            entry["requested_mb"] = mean("requested_bytes") / 1e6
            entry["extents"] = mean("extents")
            entry["peak_device_bytes"] = max(s["peak_device_bytes"] for s in chosen)
            entry["peak_resident_bytes"] = max((s.get("memory") or {}).get("peak_resident_bytes", 0) for s in chosen)
        devices = [s["cache"] for s in chosen if s.get("cache")]
        if devices:  # Phase 6B: a device cache
            hit = sum(c["hit_bytes"] for c in devices)
            entry["device_cache_byte_hit_rate"] = hit / max(1, hit + sum(c["miss_bytes"] for c in devices))
            entry["device_cache_resident_bytes"] = max(c["resident_bytes"] for c in devices)
            entry["device_cache_evictions"] = statistics.mean(c["evictions"] for c in devices)
        if all("peak_reserved_bytes" in s for s in chosen):
            entry["peak_reserved_bytes"] = max(s["peak_reserved_bytes"] for s in chosen)
        caches = [s["host_cache"] for s in chosen if s.get("host_cache")]
        if caches:
            lookups = sum(c["lookups"] for c in caches)
            entry["host_cache_hit_rate"] = sum(c["hits"] + c["waits"] for c in caches) / max(1, lookups)
            served = sum(c["hit_bytes"] + c["wait_bytes"] for c in caches)
            entry["host_cache_byte_hit_rate"] = served / max(1, served + sum(c["miss_bytes"] for c in caches))
            entry["host_cache_resident_bytes"] = max(c["resident_bytes"] for c in caches)
        natives = [s["native"] for s in chosen if s.get("native")]
        if natives:
            entry["drive_busy_ms"] = statistics.mean(n["busy_ms"] for n in natives)
            entry["drive_gb_per_s_busy"] = entry["physical_mb"] / 1e3 / max(1e-9, entry["drive_busy_ms"] / 1e3)
            entry["cache_copied_mb"] = statistics.mean(n["cache_copied_bytes"] for n in natives) / 1e6
            if all("submit_ms" in n for n in natives):  # Phase 6B: the engine's own time in submit (decision 0013)
                entry["engine_submit_ms"] = {
                    k: statistics.mean(n[f"submit{k}_ms"] for n in natives) for k in ("", "_cache", "_plan", "_start")
                }
        if phase == "prefill":
            entry["by_length"] = {
                int(length): statistics.mean(s["wall_ms"] for s in chosen if s["length"] == length) for length in sorted({s["length"] for s in chosen})
            }
        out[phase] = entry
    return out


if __name__ == "__main__":
    raise SystemExit(main())
