"""Phase 4B: Moonlight-16B-A3B out of VRAM and out of host RAM, reproduced bit for bit (decision 0008).

    uv run python benchmarks/moonlight_runtime.py --output experiments/phase4b/<name> [--num-prompts N]

The model, prompts, storage options, budgets and configurations are in configs/phase4b-moonlight.yaml. The run has two
stages, each in its own process, started by this script:

  reference  the independent streaming reference (`awpmi.streaming_reference`): transformers' model, loaded by
             transformers' own loader, each experts layer materialized whole from the published checkpoint when it runs
             and released after, so neither host nor device memory holds the experts. The first prompts also run with
             experts layers kept resident after their first load (residency check). It records the sha256 of every expert
             row it loaded, for the index audit. Nothing of Weightsift's own expert path runs in this process.
  stream     a fresh process under the configured device-memory cap: the published files re-hashed with direct reads
             against the Hub's sha256; every row of Weightsift's expert index read from the drive and compared with the
             reference's row digests; every weight but the routed experts read from the checkpoint (direct I/O); then every
             configuration (cache, call budget), each step compared with the reference and audited.

Per step and layer both stages record digests of the attention output; the router's logits, scores, chosen experts and
weights; the routed experts; every (token, expert) output before weighting (the direct expert check: every unchunked call,
and chunked calls on the configured prompts, whose experts are then read again through a separate uncached store after the
step's accounting); the experts call's output; the shared experts' output; the MoE block's output; the dense MLP of the
first layer; and per step the logits, the token and the whole KV cache.

Writes, into the output directory:
  config.yaml, environment.json   as every benchmark (with the prompts' provenance)
  prompts.jsonl        the prompts as Moonlight token ids
  reference_rows.json  the reference's sha256 of every expert row (layer, parameter, expert)
  index.json           the expert index's manifest, the file verification and the row audit
  reference.jsonl.gz   per prompt and step: the reference's digests (and the residency check)
  records.jsonl.gz     per configuration, prompt and step: comparisons, bytes, cache, buffers, memory, audit
  ranges.jsonl.gz      raw physical-I/O traces: every extent read, for the first prompts of each configuration
  reference_stage.json, stream_stage.json   load reports, memory, timings
  digest.json          sha256 of index, prompts, reference rows, reference, records and ranges (timings, system excluded)
A stage stops at its first hard failure (unless --keep-going) and saves it to failure.json.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

from awpmi.runtime import configure_reproducible_numerics  # noqa: E402

NUMERICS_FLAGS = configure_reproducible_numerics()

import importlib  # noqa: E402

import torch  # noqa: E402

from awpmi.storage.fileio import process_memory  # noqa: E402  (a measurement utility; it reads no checkpoint)
from awpmi.streaming_reference import ReferenceCall, StreamingReference, experts_modules  # noqa: E402
from awpmi.tracing import JsonlWriter, canonical_digest, environment_metadata, read_jsonl, sha256_file, tensor_digest  # noqa: E402

EXCLUDED_FIELDS = ("timings_ms", "system")
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
LAYER_KINDS = (
    "router_logits", "router_scores", "router_indices", "router_weights", "top_k_index", "top_k_weights",
    "experts_output", "shared_output", "moe_output",
)


class HardFailure(Exception):
    pass


def short_digest(*tensors: torch.Tensor) -> str:
    return tensor_digest(*tensors)[:32]


def logits_sha256(logits: torch.Tensor) -> str:
    return hashlib.sha256(logits.detach().contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()


def kv_digest(cache) -> str:
    return tensor_digest(*[t for layer in cache.layers for t in (layer.keys, layer.values)])


def load_adapter(model_config: dict):
    return importlib.import_module(f"awpmi.models.{model_config['adapter']}")


def device_cache(entry: dict, capacity: int, half_life: float, encoded=None, device=None):
    """A configuration's device cache: Phase 4B's page cache of BF16 rows, or for encoded rows (Phase 6B, decision
    0013) fixed slots of the encoded pack's stored row sizes, each size's share of `capacity` in proportion to its
    rows' bytes (no allocator blocks, so the budget is the memory held); `device_cache: pages` keeps encoded rows in a
    page cache of copies instead (the exploratory runs)."""
    from collections import Counter

    from awpmi.storage.cache import POLICIES, PageCache, SlotCache

    policy_class = POLICIES[entry["policy"]]
    policy = (lambda: policy_class(half_life)) if entry["policy"] == "hotness" else policy_class
    if entry.get("experts") == "encoded" and entry.get("device_cache", "slots") == "slots":
        sizes = Counter()
        for name in encoded.encodings:
            segment = encoded.pack.segments[name]
            sizes[segment.row_bytes] += segment.nbytes
        return SlotCache.sized(capacity, sizes, device, policy)
    return PageCache(capacity, policy())


class _PerformanceInformation(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t if name not in ("cb", "HandleCount", "ProcessCount", "ThreadCount") else ctypes.c_uint32)
                for name in ("cb", "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable", "SystemCache",
                             "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize", "HandleCount", "ProcessCount", "ThreadCount")]


def system_memory() -> dict | None:
    """The OS's view (Windows): physical memory available and the file cache, outside any process's own accounting."""
    if sys.platform != "win32":
        return None
    info = _PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
        return None
    page = info.PageSize
    return {"physical_available_bytes": info.PhysicalAvailable * page, "system_cache_bytes": info.SystemCache * page,
            "commit_total_bytes": info.CommitTotal * page}


class MemorySampler:
    """Process memory sampled at every experts call (the peak of a step, beyond what the OS's lifetime peak says).

    resident_bytes   the working set: physical host memory in use by the process
    private_bytes    its commit charge. Under Windows' WDDM driver model this includes every byte of device memory the
                     process allocates (measured: +2 GiB on the device is +2.15 GB of commit, the working set unchanged)
    host_commit_bytes  private bytes minus the device memory torch's allocator holds: the host's own commitments
    """

    KEYS = ("resident_bytes", "private_bytes", "host_commit_bytes")

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.step = {key: 0 for key in self.KEYS}

    def sample(self) -> None:
        memory = process_memory() or {}
        reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        values = {
            "resident_bytes": int(memory.get("resident_bytes") or 0),
            "private_bytes": int(memory.get("private_bytes") or 0),
            "host_commit_bytes": int(memory.get("private_bytes") or 0) - reserved,
        }
        for key in self.KEYS:
            self.step[key] = max(self.step[key], values[key])


# Prompts


def build_prompts(config: dict, tokenizer) -> tuple[list[dict], dict]:
    """wikitext-2 test paragraphs: from drawn starting lines, consecutive lines joined until each target length, cut to it."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    dataset = config["dataset"]
    path = hf_hub_download(dataset["repository"], dataset["file"], repo_type="dataset", revision=dataset["revision"])
    lines = pq.read_table(path).column(dataset["text_column"]).to_pylist()
    starts = [i for i, text in enumerate(lines) if text.strip() and not text.strip().startswith("=")]
    lengths = [int(n) for n in config["lengths"]]
    rng = random.Random(int(config["seed"]))
    chosen = rng.sample(starts, k=len(lengths))
    prompts = []
    for prompt_id, (start, length) in enumerate(zip(chosen, lengths, strict=True)):
        text, line, ids = "", start, []
        while len(ids) < length and line < len(lines):
            text += lines[line]
            line += 1
            ids = tokenizer(text)["input_ids"]
        if len(ids) < length:
            raise ValueError(f"not enough text after line {start} for {length} tokens")
        prompts.append({"prompt_id": prompt_id, "source_line": start, "length": length, "token_ids": ids[:length]})
    provenance = {**dataset, "file_sha256": sha256_file(path), "seed": int(config["seed"]), "lengths": lengths}
    return prompts, provenance


# Recording (both stages)


class StepRecorder:
    """One forward's observations: per layer its attention output, and per MoE layer its router, experts call, shared experts
    and block. Hook placement uses transformers' experts convention and the adapter's suffixes only, so both stages observe
    the same modules."""

    def __init__(self, model, adapter) -> None:
        experts = [name for name in experts_modules(model)]
        self.experts = experts
        self.num_experts = None
        self.attention = [name for name, _ in model.named_modules() if name.endswith(".self_attn")]
        head = {name: name[: -len(adapter.EXPERTS_SUFFIX)] for name in experts}
        moe_blocks = {head[name] + adapter.BLOCK_SUFFIX for name in experts}
        self.dense = [name for name, _ in model.named_modules() if name.endswith(".mlp") and name not in moe_blocks]
        self.handles = []
        self.on_experts_call = None  # an extra observer (memory sampling)
        self.pending: list = []  # chunked calls whose per-assignment outputs are checked after the step, or not at all
        # Whether this step checks its chunked calls (`resolve_pending`): a call the step will not check is not kept, so
        # that a long prefill does not hold every layer's inputs and outputs (Phase 6B: the device cache leaves no room).
        self.keep_chunked = True
        for name in self.attention:
            self.handles.append(model.get_submodule(name).register_forward_hook(self._capture("attention", name)))
        for name in self.dense:
            self.handles.append(model.get_submodule(name).register_forward_hook(self._capture("dense_mlp", name)))
        for name in experts:
            self.handles.append(model.get_submodule(head[name] + adapter.ROUTER_SUFFIX).register_forward_hook(self._router(name)))
            self.handles.append(model.get_submodule(head[name] + adapter.SHARED_SUFFIX).register_forward_hook(self._capture("shared_output", name)))
            self.handles.append(model.get_submodule(head[name] + adapter.BLOCK_SUFFIX).register_forward_hook(self._capture("moe_output", name)))
        self.clear()

    def clear(self) -> None:
        self.values: dict[str, dict] = {"attention": {}, "dense_mlp": {}}
        self.layers: dict[str, dict] = {name: {} for name in self.experts}
        self.pending = []

    def _capture(self, kind: str, name: str):
        def hook(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if kind in ("attention", "dense_mlp"):
                self.values[kind][name] = short_digest(tensor)
            else:
                self.layers[name][kind] = short_digest(tensor)

        return hook

    def _router(self, name: str):
        def hook(module, args, output):
            logits, weights, indices = output
            entry = self.layers[name]
            entry["router_logits"] = short_digest(logits)
            entry["router_scores"] = short_digest(logits.sigmoid())
            entry["router_indices"] = short_digest(indices)
            entry["router_weights"] = short_digest(weights)

        return hook

    def on_call(self, call) -> None:
        entry = self.layers[call.module]
        self.num_experts = self.num_experts or getattr(call.experts_module, "num_experts", None)
        experts = torch.unique(call.top_k_index.detach())
        entry["routed"] = experts.tolist()
        entry["top_k_index"] = short_digest(call.top_k_index)
        entry["top_k_weights"] = short_digest(call.top_k_weights)
        entry["experts_output"] = short_digest(call.output)
        if isinstance(call, ReferenceCall) or call.chunks is None:
            entry["expert_outputs"] = short_digest(call.per_assignment_outputs())
            entry["chunks"] = None
        else:
            entry["expert_outputs"] = None
            entry["chunks"] = len(call.chunks)
            if self.keep_chunked:
                self.pending.append(call)
        if self.on_experts_call is not None:
            self.on_experts_call(call)

    def resolve_pending(self, weights_for) -> None:
        """The deferred direct check of chunked calls: their experts' outputs, with weights from `weights_for(call)`."""
        for call in self.pending:
            self.layers[call.module]["expert_outputs"] = short_digest(call.per_assignment_outputs(weights=weights_for(call)))
        self.pending = []

    def summary(self) -> dict:
        missing = [name for name in self.experts if not {"routed", "expert_outputs", *LAYER_KINDS} <= set(self.layers[name])]
        if missing or len(self.values["attention"]) != len(self.attention):
            raise RuntimeError(f"incomplete observations for {missing[:3]}")
        return {
            "routed": [self.layers[name]["routed"] for name in self.experts],
            "layers": {kind: [self.layers[name][kind] for name in self.experts] for kind in LAYER_KINDS},
            "expert_outputs": [self.layers[name]["expert_outputs"] for name in self.experts],
            "attention": [self.values["attention"][name] for name in self.attention],
            "dense_mlp": [self.values["dense_mlp"][name] for name in self.dense],
        }


@torch.inference_mode()
def decode(model, token_ids: list[int], steps: int, device: torch.device, before, after) -> None:
    """Greedy decoding with the KV cache (logits of the last position only); `before(step)` and `after(...)` frame every forward."""
    input_ids = torch.tensor([token_ids], device=device)
    cache = None
    for step in range(steps + 1):
        before(step)
        output = model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits = output.logits[0, -1]
        cache = output.past_key_values
        after(step, logits, input_ids.shape[1], cache)
        input_ids = logits.argmax().view(1, 1)


def step_record(prompt: dict, step: int, positions: int, logits: torch.Tensor, cache, recorder: StepRecorder) -> dict:
    top2 = torch.topk(logits.float(), 2).values.tolist()
    return {
        "prompt_id": prompt["prompt_id"],
        "step": step,
        "phase": "prefill" if step == 0 else "decode",
        "positions": positions,
        "token": int(logits.argmax()),
        "logits_sha256": logits_sha256(logits),
        "top2_gap": top2[0] - top2[1],
        "kv_sha256": kv_digest(cache),
        **recorder.summary(),
    }


def compare(record: dict, expected: dict) -> dict:
    """Which recorded quantities equal the reference's; per-layer kinds also name the differing layers."""
    matches = {
        "token": record["token"] == expected["token"],
        "logits": record["logits_sha256"] == expected["logits_sha256"],
        "kv_cache": record["kv_sha256"] == expected["kv_sha256"],
        "routed": record["routed"] == expected["routed"],
        "attention": record["attention"] == expected["attention"],
        "dense_mlp": record["dense_mlp"] == expected["dense_mlp"],
    }
    differing = {}
    for kind in LAYER_KINDS:
        layers = [k for k, (a, b) in enumerate(zip(record["layers"][kind], expected["layers"][kind], strict=True)) if a != b]
        matches[kind] = not layers
        if layers:
            differing[kind] = layers
    checked = [k for k, value in enumerate(record["expert_outputs"]) if value is not None]
    layers = [k for k in checked if record["expert_outputs"][k] != expected["expert_outputs"][k]]
    matches["expert_outputs"] = not layers
    if layers:
        differing["expert_outputs"] = layers
    return {"matches": matches, "differing_layers": differing, "expert_outputs_checked": len(checked)}


# Reference stage


def run_reference(raw_config: dict, output: Path, prompts: list[dict], keep_going: bool) -> int:
    model_config, reference_config = raw_config["model"], raw_config["reference"]
    adapter = load_adapter(model_config)
    profile = adapter.REFERENCE_PROFILE
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = DTYPES[model_config["dtype"]]
    steps = int(raw_config["prompts"]["decode_steps"])
    report: dict = {"timings_ms": {}, "memory": {"start": process_memory()}, "system": {"start": system_memory()}}
    failures: list[dict] = []

    def fail(kind: str, entry: dict) -> None:
        failures.append({"kind": kind, **entry})
        if not keep_going:
            raise HardFailure

    rows: dict[str, list[str]] = {}

    def digest_rows(name: str, module) -> None:
        """sha256 of every expert row of a layer the first time it is loaded (the index audit's expected values)."""
        for parameter, tensor in module.named_parameters(recurse=False):
            key = f"{name}.{parameter}"
            if key in rows:
                continue
            data = tensor.detach().contiguous().view(torch.uint8).reshape(tensor.shape[0], -1).cpu().numpy()
            rows[key] = [hashlib.sha256(data[expert].tobytes()).hexdigest() for expert in range(data.shape[0])]

    sampler = MemorySampler()
    started = time.perf_counter()
    reference = StreamingReference(
        model_config["repository"], model_config["revision"], dtype, device,
        experts_implementation=profile.experts_implementation, attn_implementation=profile.attention_implementation,
        on_load=digest_rows,
    ).load()
    report["timings_ms"]["load"] = (time.perf_counter() - started) * 1e3
    model = reference.model
    profile.check_model(model)
    adapter.check_config(model.config)
    report["load"] = {k: v for k, v in reference.report.items() if k != "files"}
    report["memory"]["after_load"] = process_memory()
    report["non_expert_device_bytes"] = torch.cuda.memory_allocated(device)
    recorder = StepRecorder(model, adapter)
    recorder.on_experts_call = lambda call: sampler.sample()
    reference.on_call = recorder.on_call
    experts_names = recorder.experts
    check_layers = tuple(experts_names[i - int(model.config.first_k_dense_replace)] for i in reference_config["residency_check_layers"])
    check_count = int(reference_config["residency_check_prompts"])
    check: dict = {}
    phases = {phase: {key: 0 for key in MemorySampler.KEYS} for phase in ("prefill", "decode")}
    try:
        with JsonlWriter(output / "reference.jsonl.gz") as log:
            for label, resident, chosen in (("resident_check", check_layers, prompts[:check_count]), ("streamed", (), prompts)):
                reference.set_resident(resident)
                torch.cuda.reset_peak_memory_stats(device)
                loads_before, bytes_before, seconds_before = reference.loads, reference.loaded_bytes, reference.load_seconds
                started = time.perf_counter()
                for prompt in chosen:
                    timing = {}

                    def before(step):
                        recorder.clear()
                        sampler.clear()
                        torch.cuda.synchronize(device)
                        timing["start"] = time.perf_counter()
                        timing["load"] = reference.load_seconds

                    def after(step, logits, positions, cache, prompt=prompt, label=label):
                        record = step_record(prompt, step, positions, logits, cache, recorder)
                        torch.cuda.synchronize(device)
                        key = (prompt["prompt_id"], step)
                        if label == "resident_check":
                            check[key] = record
                            return
                        phase = phases[record["phase"]]
                        for k in phase:
                            phase[k] = max(phase[k], sampler.step[k])
                        record["timings_ms"] = {
                            "step": (time.perf_counter() - timing["start"]) * 1e3, "expert_loads": (reference.load_seconds - timing["load"]) * 1e3,
                        }
                        record["system"] = {"process": dict(sampler.step), "peak_device_bytes": torch.cuda.max_memory_allocated(device)}
                        if key in check:
                            verdict = compare(record, check[key])
                            record["residency_check"] = all(verdict["matches"].values())
                            if not record["residency_check"]:
                                fail("residency", {"prompt_id": prompt["prompt_id"], "step": step, **verdict})
                        log.write(record)

                    decode(model, prompt["token_ids"], steps, device, before, after)
                report["timings_ms"][label] = (time.perf_counter() - started) * 1e3
                report[label] = {
                    "prompts": len(chosen),
                    "peak_device_bytes": torch.cuda.max_memory_allocated(device),
                    "layer_loads": reference.loads - loads_before,
                    "loaded_bytes": reference.loaded_bytes - bytes_before,
                    "load_seconds": reference.load_seconds - seconds_before,
                    "peak_expert_device_bytes": reference.peak_expert_device_bytes,
                }
                print(f"reference {label}: {len(chosen)} prompts, {report['timings_ms'][label] / 1e3:.0f}s", flush=True)
        report["residency_checked_steps"] = len(check)
    except HardFailure:
        pass
    (output / "reference_rows.json").write_text(json.dumps(rows, indent=1, sort_keys=True), encoding="utf-8")
    report["memory"]["peak"] = process_memory()
    report["memory"]["phases"] = phases
    report["system"]["end"] = system_memory()
    report["profile"] = profile.to_json()
    report["experts_implementation"] = model.config._experts_implementation
    report["attention_implementation"] = model.config._attn_implementation
    report["expert_layers"] = len(experts_names)
    (output / "reference_stage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failures:
        (output / "failure.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return 1 if failures else 0


# Stream stage


def audit_step(report: dict, requested: int) -> list[str]:
    from awpmi.storage.store import IO_BLOCK_BYTES

    problems = []
    served, storage, transfer = report["materialization"], report["storage"], report.get("transfer", {})
    if served["requested_bytes"] != requested:
        problems.append("requested bytes != served experts")
    if served.get("decoded_rows"):
        # Phase 6B (decision 0013): encoded rows. Every requested row was decoded on the device; the store served their
        # stored bytes (what crossed the bus); the decoder wrote the requested bytes.
        if served["decoded_bytes"] != served["requested_bytes"] or report["decoder"]["decoded_bytes"] != served["decoded_bytes"]:
            problems.append("decoded bytes != requested")
        if served["cache_hit_bytes"] + served["fetched_bytes"] != served["stored_requested_bytes"]:
            problems.append("cache hits + fetched != the requested rows' stored bytes")
    elif served["cache_hit_bytes"] + served["fetched_bytes"] != served["requested_bytes"]:
        problems.append("cache hits + fetched != requested")
    if storage["logical_bytes"] != served["fetched_bytes"]:
        problems.append("storage logical != fetched")
    if transfer.get("h2d_bytes") != storage["logical_bytes"]:
        problems.append("h2d != storage logical")
    native = storage.get("native") or {}
    # Physical bytes are the 4 KiB blocks of the rows read (by a request, or by a prefetch of the native host cache).
    blocks = storage["blocks_4k"] + native.get("prefetch_blocks_4k", 0)
    reads = max(1, storage["requests"] + native.get("prefetches", 0))
    if not 0 <= blocks * IO_BLOCK_BYTES - storage["physical_bytes"] < IO_BLOCK_BYTES * reads:
        problems.append("physical bytes are not the requested rows' 4 KiB blocks")
    if storage["os_read_bytes"] is not None and (
        storage["os_read_bytes"] != storage["physical_bytes"] or storage["os_read_calls"] != storage["read_calls"]
    ):
        problems.append("OS counters differ from the store's reads")
    cache = storage.get("host_cache")
    if native and native["fallback_rows"]:
        problems.append("rows read again after a failed cache load")
    if cache is not None:
        # Phase 6A (decision 0012): every row was looked up once; what the cache served was copied; the budget held.
        if cache["lookups"] != storage["rows"] or cache["lookups"] != cache["hits"] + cache["waits"] + cache["misses"]:
            problems.append("host cache lookups != rows requested = hits + waits + misses")
        if native["cache_copied_bytes"] != cache["hit_bytes"] + cache["wait_bytes"]:
            problems.append("bytes copied from the host cache != its hits and waits")
        held = (cache.get("held_bytes", 0), cache.get("peak_held_bytes", 0))  # rows and pooled blocks (decision 0013)
        if max(cache["resident_bytes"], cache["peak_resident_bytes"], *held) > cache["capacity_bytes"]:
            problems.append("host cache above its budget")
        if cache["prefetch_wasted"] or cache["prefetch_used"] != cache["prefetch_fills"]:
            problems.append("prefetched rows unused or evicted before use")
    return problems


def index_audit(pack, rows: dict[str, list[str]]) -> dict:
    """Every row of every index segment, read from the drive through Weightsift's store, against the reference's row digests."""
    store = pack.store(direct=True)
    audit = {"segments": 0, "rows": 0, "bytes": 0, "differing": [], "missing": sorted(set(pack.segments) ^ set(rows))}
    try:
        for name in sorted(pack.segments):
            segment = pack.segments[name]
            expected = rows.get(name, [])
            for first in range(0, segment.rows, 8):
                data = store.read_rows(name, torch.arange(first, min(first + 8, segment.rows))).numpy()
                for offset in range(data.shape[0]):
                    if expected[first + offset] != hashlib.sha256(data[offset].tobytes()).hexdigest():
                        audit["differing"].append([name, first + offset])
            audit["segments"] += 1
            audit["rows"] += segment.rows
            audit["bytes"] += segment.nbytes
    finally:
        store.close()
    return audit


def encoded_audit(encoded, rows: dict[str, list[str]], device: torch.device) -> dict:
    """Phase 6B (decision 0013): every row of the encoded pack read from the drive and decoded on the GPU by the runtime's
    decoder, its sha256 against the reference's row digests (and the pack's own record), before any inference."""
    import numpy as np

    from awpmi.storage.native import NATIVE_AVAILABLE
    from awpmi.streaming.codec import RowDecoder

    decoder = RowDecoder(encoded.encodings, device)
    store = encoded.pack.store(backend="native" if NATIVE_AVAILABLE else "python", direct=True)
    audit = {"segments": 0, "rows": 0, "stored_bytes": 0, "logical_bytes": 0, "differing": [], "differing_from_pack": [],
             "missing": sorted(set(encoded.encodings) ^ set(rows))}
    try:
        for name in sorted(encoded.encodings):
            item = encoded.encodings[name]
            expected = rows.get(name, [])
            for first in range(0, item.logical.rows, 8):
                chosen = torch.arange(first, min(first + 8, item.logical.rows))
                stored = store.read_rows(name, chosen).to(device)
                out = torch.empty(chosen.numel(), item.logical.row_bytes, dtype=torch.uint8, device=device)
                sources = stored.data_ptr() + np.arange(chosen.numel(), dtype=np.int64) * item.stored_row_bytes
                decoder.decode([(name, chosen, sources, out)])
                decoded = out.cpu().numpy()
                for offset in range(chosen.numel()):
                    digest = hashlib.sha256(decoded[offset].tobytes()).hexdigest()
                    if digest != expected[first + offset]:
                        audit["differing"].append([name, first + offset])
                    if digest != item.row_sha256[first + offset]:
                        audit["differing_from_pack"].append([name, first + offset])
            audit["segments"] += 1
            audit["rows"] += item.logical.rows
            audit["stored_bytes"] += item.logical.rows * item.stored_row_bytes
            audit["logical_bytes"] += item.logical.nbytes
        decoder.check()
    finally:
        store.close()
    return audit


def run_stream(raw_config: dict, output: Path, prompts: list[dict], keep_going: bool) -> int:
    from awpmi.materialization.backend import MaterializationBackend
    from awpmi.materialization.weights import ExpertStore, WeightStore
    from awpmi.models.checkpoint import checkpoint_sources, load_model_without_experts, parameter_segments
    from awpmi.storage.pack import sha256_file_direct
    from awpmi.models.decode_graphs import DecodeGraphs
    from awpmi.models.moe import POISON_SPARE_SLOTS, ExpertCall, StreamedExperts, find_expert_modules, groups_from_pack
    from awpmi.models.streamed import StreamedParameters
    from awpmi.storage.cache import PageCache
    from awpmi.storage.pack import open_pack
    from awpmi.storage.store import FileBackedPageStore
    from awpmi.streaming.streamer import PageStreamer

    model_config, storage = raw_config["model"], raw_config["storage"]
    adapter = load_adapter(model_config)
    profile = adapter.REFERENCE_PROFILE
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = DTYPES[model_config["dtype"]]
    steps = int(raw_config["prompts"]["decode_steps"])
    time.sleep(float(raw_config["timing"]["settle_seconds"]))
    report: dict = {"timings_ms": {}, "configurations": {}, "memory": {"start": process_memory()}, "system": {"start": system_memory()}}
    total = torch.cuda.get_device_properties(device).total_memory
    budget = int(raw_config["gpu_budget_bytes"])
    torch.cuda.set_per_process_memory_fraction(budget / total, device)
    report.update(gpu_total_bytes=total, gpu_budget_bytes=budget)
    failures: list[dict] = []

    def fail(kind: str, entry: dict) -> None:
        failures.append({"kind": kind, **entry})
        if not keep_going:
            raise HardFailure

    # The published files and every expert row, before anything else reads them. The index refers to the files that hold
    # experts; the others (embeddings, the dense first layer) are verified against the Hub's sha256 here too.
    skip = bool(raw_config.get("development", {}).get("skip_audit", False))
    started = time.perf_counter()
    pack = open_pack(REPO_ROOT / raw_config["index"]["directory"], verify="size" if skip else "files")
    sources = checkpoint_sources(model_config["repository"], model_config["revision"], declared_sha256=not skip)
    checkpoint_files = {key: entry.path for key, entry in sources.items()}
    others = {key: entry for key, entry in sources.items() if key not in pack.files}
    other_files = {key: {"sha256_declared": entry.sha256} for key, entry in others.items()}
    if not skip:
        for key, entry in others.items():
            other_files[key]["sha256_direct_read"] = sha256_file_direct(entry.path)
    report["timings_ms"]["verify_files"] = (time.perf_counter() - started) * 1e3
    rows = json.loads((output / "reference_rows.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    audit = {"skipped": True, "differing": [], "missing": []} if skip else index_audit(pack, rows)
    report["timings_ms"]["index_audit"] = (time.perf_counter() - started) * 1e3
    verification = "skipped (development run)" if skip else "files against the Hub's sha256"
    (output / "index.json").write_text(
        json.dumps({"manifest": pack.manifest, "verification": verification, "other_files": other_files, "audit": audit}, indent=2),
        encoding="utf-8",
    )
    encoded = None
    if any(entry.get("experts") == "encoded" for entry in raw_config["configurations"]):
        # Phase 6B (decision 0013): the encoded pack, its files re-hashed (direct reads), every row decoded on the GPU and
        # compared with the reference's digests before any configuration runs.
        from awpmi.storage.encoded import open_encoded

        started = time.perf_counter()
        encoded = open_encoded(REPO_ROOT / raw_config["encoded"]["directory"], verify="size" if skip else "files")
        encoded_check = {"skipped": True, "differing": [], "differing_from_pack": [], "missing": []} if skip else encoded_audit(encoded, rows, device)
        report["timings_ms"]["encoded_audit"] = (time.perf_counter() - started) * 1e3
        index_info = json.loads((output / "index.json").read_text(encoding="utf-8"))
        index_info["encoded"] = {
            "manifest": {k: encoded.pack.manifest[k] for k in ("format", "format_version", "kind", "files", "packing")},
            "encoding": encoded.metadata["encoding"],
            "stored_ratio": encoded.stored_ratio,
            "verification": "skipped (development run)" if skip else "files against the manifest's sha256",
            "audit": encoded_check,
        }
        (output / "index.json").write_text(json.dumps(index_info, indent=2), encoding="utf-8")
    try:
        if audit["differing"] or audit["missing"]:
            fail("index_audit", {"differing": audit["differing"][:10], "missing": audit["missing"][:10]})
        unverified = [key for key, entry in other_files.items() if not skip and entry["sha256_direct_read"] != entry["sha256_declared"]]
        if unverified:
            fail("source_files", {"differing": unverified})
        if encoded is not None and (encoded_check["differing"] or encoded_check["differing_from_pack"] or encoded_check["missing"]):
            fail("encoded_audit", {k: encoded_check[k][:10] for k in ("differing", "differing_from_pack", "missing")})
    except HardFailure:
        (output / "failure.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        return 1

    groups = groups_from_pack(pack)
    profile.check_weights({name: segment.dtype for name, segment in pack.segments.items()})
    started = time.perf_counter()
    model, load_report = load_model_without_experts(model_config["repository"], model_config["revision"], dtype, device, checkpoint_files)
    report["timings_ms"]["load"] = (time.perf_counter() - started) * 1e3
    profile.check_model(model)
    adapter.check_config(model.config)
    report["load"] = {k: v for k, v in load_report.items() if k != "expert_parameters"}
    report["non_expert_device_bytes"] = torch.cuda.memory_allocated(device)
    shared_modules = adapter.shared_experts(model)
    shared_owners = sorted(name for name, module in model.named_modules() if any(module is m for m in shared_modules.values()))
    shared_names = [f"{owner}.{p}" for owner in shared_owners for p, _ in model.get_submodule(owner).named_parameters()]
    report["shared_experts_device_bytes"] = sum(p.numel() * p.element_size() for module in shared_modules.values() for p in module.parameters())
    report["memory"]["after_load"] = process_memory()
    modules = find_expert_modules(model)
    num_experts = modules[0].num_experts
    expert_bytes = sum(segment.nbytes for segment in pack.segments.values())
    row_bytes = {key: sum(pack.segments[s].row_bytes for s in group.segments.values()) for key, group in groups.items()}
    expert_row = max(row_bytes.values())
    layer_bytes = max(row_bytes[key] * groups[key].experts for key in groups)
    recorder = StepRecorder(model, adapter)
    reference = {(r["prompt_id"], r["step"]): r for r in read_jsonl(output / "reference.jsonl.gz")}
    default_call_budget = raw_config.get("call_budget_bytes")

    store_options = dict(
        direct=True, alignment=int(storage["alignment"]), max_gap=int(storage["max_gap"]), workers=int(storage["workers"]),
        max_read_bytes=int(storage["max_read_bytes"]), max_extent_bytes=int(storage["max_extent_bytes"]),
    )

    def host_cache_bytes(entry: dict) -> int:
        if "host_cache_experts" in entry:
            return int(entry["host_cache_experts"]) * expert_row
        return int(float(entry.get("host_cache_bytes", 0)))

    def build(cache: PageCache | None, entry: dict | None = None) -> MaterializationBackend:
        """The configuration's backend: Phase 4B's Python store, or the native one with its host cache (decision 0012); with
        `experts: encoded`, the encoded pack's rows, decoded on the GPU (decision 0013)."""
        entry = entry or {}
        source = encoded.pack if entry.get("experts") == "encoded" else pack
        if entry.get("backend", "python") == "native":
            store = source.store(backend="native", host_cache_bytes=host_cache_bytes(entry), **store_options)
        else:
            store = source.store(**store_options)
        streamer = PageStreamer(device, int(storage["slot_bytes"]), int(storage["slots"]), native_slots=int(storage.get("native_slots", 4)))
        if entry.get("experts") == "encoded":
            from awpmi.streaming.codec import RowDecoder

            return MaterializationBackend(store, device, streamer, cache, decoder=RowDecoder(encoded.encodings, device))
        return MaterializationBackend(store, device, streamer, cache)

    def stream_shared() -> StreamedParameters:
        """The shared experts served from the checkpoint at every call (their own store and streamer, so counted apart)."""
        store = FileBackedPageStore(checkpoint_files, parameter_segments(model, shared_names, checkpoint_files), **store_options)
        backend = MaterializationBackend(store, device, PageStreamer(device, int(storage["slot_bytes"]), int(storage["slots"])))
        return StreamedParameters(model, WeightStore(backend), {name: name for name in shared_names}).install()

    # The direct check of chunked calls rereads their experts through a store of its own (no cache), after the step's accounting.
    check_backend = build(None)
    check_experts = ExpertStore(WeightStore(check_backend), groups)
    report["host_staging_bytes"] = check_backend.host_resident_bytes  # per backend; the streamed one has the same

    def chunk_weights(call: ExpertCall):
        return lambda experts: check_experts.load(call.module, experts)

    # Warm-up (kernels, allocator) with a backend of its own, so that no configuration's cache sees it.
    backend = build(None)
    streamed = StreamedExperts(model, ExpertStore(WeightStore(backend), groups), compact=True, on_call=recorder.on_call,
                               max_call_bytes=default_call_budget).install()
    for prompt in prompts[: int(raw_config["timing"]["warmup_prompts"])]:
        decode(model, prompt["token_ids"], 1, device, lambda step: recorder.clear(), lambda *args: recorder.clear())
    streamed.remove()
    backend.store.close()
    backend.streamer.close()

    sampler = MemorySampler()
    phases: dict = {}
    started_all = time.perf_counter()
    with JsonlWriter(output / "records.jsonl.gz") as log, JsonlWriter(output / "ranges.jsonl.gz") as range_log:
        try:
            for index, entry in enumerate(raw_config["configurations"]):
                name = entry["name"]
                capacity = int(entry.get("cache_experts", 0)) * expert_row
                if "device_cache_bytes" in entry:  # Phase 6B: a device cache of encoded rows, in stored bytes
                    capacity = int(float(entry["device_cache_bytes"]))
                cache = None
                if capacity:
                    cache = device_cache(entry, capacity, float(raw_config["hotness_half_life"]), encoded, device)
                if "call_budget_experts" in entry:
                    call_budget = int(entry["call_budget_experts"]) * expert_row
                else:
                    call_budget = entry.get("call_budget_bytes", default_call_budget)
                backend = build(cache, entry)
                native = entry.get("backend", "python") == "native"
                # Phase 6B (decision 0013): PyTorch's NaN fill of uninitialized memory off for this configuration (no
                # arithmetic changes: the digests say whether anything read such memory), and its decode graphs.
                fill = bool(entry.get("fill_uninitialized_memory", True))
                torch.utils.deterministic.fill_uninitialized_memory = fill
                freeze_prefill = bool(entry.get("freeze_prefill", False))
                poisoned_prompts = int(entry.get("poison_prompts", raw_config["poison_prompts"] if index == 0 else 0))
                all_experts = bool(entry.get("all_experts", False))
                streamed = StreamedExperts(
                    model, ExpertStore(WeightStore(backend), groups), compact=True, all_experts=all_experts, on_call=recorder.on_call,
                    max_call_bytes=call_budget, prefetch_chunks=bool(entry.get("prefetch_chunks", False)),
                ).install()
                shared = stream_shared() if entry.get("stream_shared") else None
                graphs = DecodeGraphs(model).install() if entry.get("decode_graphs") else None
                peaks = {"expert": 0}
                calls_seen: list = []

                def account(call: ExpertCall, cache=cache, streamed=streamed, peaks=peaks, calls_seen=calls_seen) -> None:
                    peaks["expert"] = max(peaks["expert"], streamed.peak_compact_bytes + (0 if cache is None else cache.resident_bytes))
                    calls_seen.append((call.module, len(call.served), call.chunks))
                    sampler.sample()

                recorder.on_experts_call = account
                gc.collect()
                torch.cuda.empty_cache()
                count = int(entry.get("num_prompts", len(prompts)))
                checked_prompts = entry.get("per_assignment_prompts", raw_config["per_assignment_prompts"])
                checked_prompts = len(prompts) if checked_prompts == "all" else int(checked_prompts)
                config_steps = min(int(entry.get("decode_steps", steps)), steps)  # the reference has `steps` decode steps
                report["configurations"][name] = {
                    "cache_capacity_bytes": capacity, "cache_experts": int(entry.get("cache_experts", 0)), "call_budget_bytes": call_budget,
                    "device_bytes_at_start": torch.cuda.memory_allocated(device), "host_staging_bytes": backend.host_resident_bytes,
                }
                if native:
                    report["configurations"][name].update(
                        backend="native", host_cache_bytes=backend.store.host_cache_bytes, freeze_prefill=freeze_prefill,
                        native_slots=backend.streamer.native_slots,
                    )
                if cache is not None and cache.copies:  # Phase 6B: a device cache in fixed slots (all allocated up front)
                    report["configurations"][name]["device_cache_slots"] = {str(size): count for size, count in cache.slots.items()}
                report["configurations"][name]["fill_uninitialized_memory"] = fill
                started = time.perf_counter()
                for position, prompt in enumerate(prompts[:count]):
                    streamed.poison = position < poisoned_prompts
                    record_ranges = position < int(raw_config["record_ranges_prompts"])
                    check_chunked = position < checked_prompts
                    timing = {}

                    def before(step, record_ranges=record_ranges, shared=shared, freeze_prefill=freeze_prefill, check_chunked=check_chunked):
                        if freeze_prefill:
                            backend.store.set_admit(step > 0)  # no admission into the host cache during a prefill
                            if cache is not None and entry.get("experts") == "encoded":
                                cache.admit = step > 0  # nor into a device cache of encoded rows (Phase 6B)
                        recorder.clear()
                        recorder.keep_chunked = check_chunked
                        sampler.clear()
                        calls_seen.clear()
                        backend.reset_stats(record_ranges=record_ranges)
                        if shared is not None:
                            shared.weights.backend.reset_stats()
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        streamed.reset_peak()
                        peaks["expert"] = 0 if cache is None else cache.resident_bytes
                        timing["start"] = time.perf_counter()
                        timing["counters"] = (streamed.calls, streamed.chunked_calls, streamed.chunk_loads)

                    def after(step, logits, positions, cache_kv, prompt=prompt, name=name, cache=cache, streamed=streamed,
                              check_chunked=check_chunked, call_budget=call_budget, record_ranges=record_ranges, shared=shared):
                        torch.cuda.synchronize(device)
                        elapsed = (time.perf_counter() - timing["start"]) * 1e3
                        peak_device = torch.cuda.max_memory_allocated(device)
                        peak_reserved = torch.cuda.max_memory_reserved(device)
                        report_ = backend.report()  # the step's accounting, before any check reads anything
                        shared_experts = {"resident_bytes": report["shared_experts_device_bytes"], "h2d_bytes": 0}
                        shared_problems = []
                        if shared is not None:
                            shared_report = shared.weights.backend.report()
                            shared_problems = [f"shared experts: {p}" for p in audit_step(shared_report, report["shared_experts_device_bytes"])]
                            shared_experts = {
                                "resident_bytes": 0,
                                "served_bytes": shared_report["materialization"]["requested_bytes"],
                                "physical_bytes": shared_report["storage"]["physical_bytes"],
                                "read_calls": shared_report["storage"]["read_calls"],
                                "h2d_bytes": shared_report["transfer"]["h2d_bytes"],
                            }
                            timing["shared_io"] = shared_report["storage"]["io_ms"]
                        chunked_steps = streamed.chunked_calls - timing["counters"][1]
                        if check_chunked:
                            recorder.resolve_pending(chunk_weights)
                        record = step_record(prompt, step, positions, logits, cache_kv, recorder)
                        expected = reference[(prompt["prompt_id"], step)]
                        verdict = compare(record, expected)
                        served_counts = [num_experts if all_experts else len(r) for r in record["routed"]]
                        requested = sum(n * row_bytes[module] for n, module in zip(served_counts, recorder.experts))
                        problems = audit_step(report_, requested) + shared_problems
                        spare = POISON_SPARE_SLOTS if streamed.poison else 0
                        whole = [(n + spare) * row_bytes[module] for module, n, chunks in calls_seen if chunks is None]
                        if any(chunks is not None for _, _, chunks in calls_seen):
                            if call_budget is None or streamed.peak_compact_bytes > call_budget or (whole and streamed.peak_compact_bytes < max(whole)):
                                problems.append("chunked buffers exceed the call budget")
                        elif streamed.peak_compact_bytes != max(whole):
                            problems.append("compact buffers != the served experts")
                        storage_stats, served = report_["storage"], report_["materialization"]
                        result = {
                            "configuration": name,
                            "prompt_id": prompt["prompt_id"],
                            "step": step,
                            "phase": record["phase"],
                            "positions": positions,
                            "poisoned": streamed.poison,
                            "token": record["token"],
                            "logits_sha256": record["logits_sha256"],
                            **verdict,
                            "routed_per_layer": [len(r) for r in record["routed"]],
                            "served_per_layer": served_counts,
                            "chunked_calls": chunked_steps,
                            "chunks_per_layer": [chunks if chunks is None else len(chunks) for _, _, chunks in calls_seen],
                            "requested_bytes": served["requested_bytes"],
                            "cache_hit_bytes": served["cache_hit_bytes"],
                            "fetched_bytes": served["fetched_bytes"],
                            "device_copy_bytes": served["device_copy_bytes"],
                            "largest_request_bytes": served["largest_request_bytes"],
                            "storage": {k: storage_stats[k] for k in ("requests", "rows", "logical_bytes", "physical_bytes", "read_calls", "extents", "blocks_4k")},
                            "h2d_bytes": report_["transfer"]["h2d_bytes"],
                            "h2d_copies": report_["transfer"]["h2d_copies"],
                            "cache": None if cache is None else {k: report_["cache"][k] for k in ("lookups", "hits", "misses", "hit_bytes", "miss_bytes", "inserts", "evictions", "bypassed")},
                            "cache_resident_bytes": None if cache is None else cache.resident_bytes,
                            "compact_peak_bytes": streamed.peak_compact_bytes,
                            "expert_peak_bytes": peaks["expert"],
                            "requested_fraction": served["requested_bytes"] / expert_bytes,
                            "drive_fraction": storage_stats["physical_bytes"] / expert_bytes,
                            "h2d_fraction": report_["transfer"]["h2d_bytes"] / expert_bytes,
                            "compact_fraction_of_layer": streamed.peak_compact_bytes / layer_bytes,
                            "shared_experts": shared_experts,
                            "audit": problems,
                            "timings_ms": {
                                "step": elapsed, "io": storage_stats["io_ms"], "copy": report_["transfer"]["copy_ms"],
                                **({"shared_io": timing["shared_io"]} if "shared_io" in timing else {}),
                            },
                            "system": {
                                "os_read_calls": storage_stats["os_read_calls"],
                                "os_read_bytes": storage_stats["os_read_bytes"],
                                "peak_device_bytes": peak_device,
                                "peak_reserved_bytes": peak_reserved,
                                "process": dict(sampler.step),
                            },
                        }
                        if "decoder" in report_:
                            # Phase 6B: encoded rows decoded on the GPU (counts only: deterministic).
                            result["decoded"] = {
                                "rows": served["decoded_rows"], "bytes": served["decoded_bytes"], "stored_bytes": served["stored_requested_bytes"],
                                "chunks": report_["decoder"]["chunks"], "launches": report_["decoder"]["launches"],
                            }
                        if native:
                            # Native steps only, so Python configurations' records stay as Phase 4B wrote them.
                            result["backend"] = "native"
                            result["native"] = {
                                k: storage_stats["native"][k]
                                for k in ("cache_copied_bytes", "gathered_bytes", "admitted_bytes", "fallback_rows", "prefetches", "prefetch_rows", "prefetch_bytes")
                            }
                            if "submits" in storage_stats["native"]:
                                result["native"]["submits"] = storage_stats["native"]["submits"]
                            result["timings_ms"]["drive_busy"] = storage_stats["native"]["busy_ms"]
                            if "submit_ms" in storage_stats["native"]:
                                result["timings_ms"]["submit"] = storage_stats["native"]["submit_ms"]
                            if "host_cache" in storage_stats:
                                cache_stats = storage_stats["host_cache"]
                                result["host_cache"] = {
                                    k: cache_stats[k]
                                    for k in ("lookups", "misses", "miss_bytes", "inserts", "evictions", "bypassed", "aborted_fills", "recycled",
                                              "prefetch_fills", "prefetch_used", "prefetch_wasted", "resident_bytes", "peak_resident_bytes",
                                              "capacity_bytes", "admit")
                                }
                                # Phase 6B (decision 0013): the cache's memory, rows in blocks reused across row sizes; what it
                                # holds (rows and pooled blocks) and what it allocated, reused and gave back (deterministic).
                                if "held_bytes" in cache_stats:
                                    result["host_cache"].update({
                                        k: cache_stats[k]
                                        for k in ("block_bytes", "pool_bytes", "held_bytes", "peak_held_bytes", "recycled_bytes", "allocated_bytes",
                                                  "released_bytes")
                                    })
                                # A row served from the cache was either there (a hit) or being loaded by a prefetch (a wait):
                                # which, is a matter of timing, so the records hold their sum and the split goes with the timings.
                                result["host_cache"]["served"] = cache_stats["hits"] + cache_stats["waits"]
                                result["host_cache"]["served_bytes"] = cache_stats["hit_bytes"] + cache_stats["wait_bytes"]
                                result["system"]["host_cache"] = {k: cache_stats[k] for k in ("hits", "waits", "hit_bytes", "wait_bytes")}
                        phase = phases.setdefault(name, {}).setdefault(record["phase"], {**{k: 0 for k in MemorySampler.KEYS}, "device_bytes": 0})
                        for key in MemorySampler.KEYS:
                            phase[key] = max(phase[key], sampler.step[key])
                        phase["device_bytes"] = max(phase["device_bytes"], peak_device)
                        log.write(result)
                        if record_ranges:
                            range_log.write({"configuration": name, "prompt_id": prompt["prompt_id"], "step": step, "ranges": storage_stats.get("ranges", [])})
                        if not all(verdict["matches"].values()):
                            fail("mismatch", {k: result[k] for k in ("configuration", "prompt_id", "step", "matches", "differing_layers")})
                        if problems:
                            fail("audit", {"configuration": name, "prompt_id": prompt["prompt_id"], "step": step, "problems": problems})

                    decode(model, prompt["token_ids"], config_steps, device, before, after)
                elapsed = time.perf_counter() - started
                report["timings_ms"][name] = elapsed * 1e3
                report["configurations"][name]["peak_cache_resident_bytes"] = None if cache is None else cache.peak_resident_bytes
                report["configurations"][name]["calls"] = streamed.calls
                report["configurations"][name]["chunked_calls"] = streamed.chunked_calls
                print(f"stream {name}: {min(count, len(prompts))} prompts, {elapsed:.0f}s", flush=True)
                if graphs is not None:
                    report["configurations"][name]["decode_graphs"] = {
                        "captures": graphs.captures, "replays": graphs.replays, "eager_steps": graphs.eager_steps, "memory_bytes": graphs.memory_bytes, "capture_ms": graphs.capture_ms,
                    }
                    graphs.remove()
                torch.utils.deterministic.fill_uninitialized_memory = True
                streamed.remove()
                backend.store.close()
                backend.streamer.close()
                if shared is not None:  # resident again for the configurations after it
                    shared.remove(restore=True)
                    shared.weights.backend.store.close()
                    shared.weights.backend.streamer.close()
                recorder.on_experts_call = None
                # Nothing of this configuration may outlive it (a device cache's slots are allocated when it is made).
                before = after = account = None
                del streamed, backend, cache, shared, graphs, before, after, account
                gc.collect()  # reference cycles too: the next configuration's device cache needs this one's memory
                torch.cuda.empty_cache()
        except HardFailure:
            pass
    check_backend.store.close()
    check_backend.streamer.close()
    report["timings_ms"]["configurations"] = (time.perf_counter() - started_all) * 1e3
    report["memory"]["peak"] = process_memory()
    report["memory"]["phases"] = phases
    report["system"]["end"] = system_memory()
    report["expert_bytes_total"] = expert_bytes
    report["layer_bytes"] = layer_bytes
    report["expert_row_bytes"] = sorted(set(row_bytes.values()))
    report["experts_implementation"] = model.config._experts_implementation
    report["attention_implementation"] = model.config._attn_implementation
    if hasattr(torch.cuda, "host_memory_stats"):
        try:
            report["pinned_host_memory"] = {k: v for k, v in torch.cuda.host_memory_stats().items() if "bytes" in k}
        except RuntimeError:
            pass
    (output / "stream_stage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failures:
        path = output / "failure.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        path.write_text(json.dumps(existing + failures, indent=2), encoding="utf-8")
    return 1 if failures else 0


# Driver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase4b-moonlight.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-prompts", type=int, default=None, help="use only the first N prompts")
    parser.add_argument("--decode-steps", type=int, default=None, help="override the configured decode steps (development)")
    parser.add_argument("--configurations", nargs="+", default=None, help="run only these configurations (development)")
    parser.add_argument("--skip-audit", action="store_true", help="skip the file verification and the index audit (development; recorded)")
    parser.add_argument("--keep-going", action="store_true", help="record hard failures instead of stopping")
    # A run is prepare (config, prompts, environment), reference, stream and digest; "all" runs them in turn, each stage
    # in a process of its own. They can also be run one by one, with the same PYTHONHASHSEED.
    parser.add_argument("--stage", choices=["all", "prepare", "reference", "stream", "digest"], default="all")
    parser.add_argument(
        "--reference-from", default=None,
        help="prepare: take the reference stage's records from this run (same model, prompts and steps; checked, sha256 recorded)",
    )
    args = parser.parse_args()
    raw_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    if args.stage in ("reference", "stream"):
        raw_config = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
        prompts = read_jsonl(output / "prompts.jsonl")
        runner = run_reference if args.stage == "reference" else run_stream
        return runner(raw_config, output, prompts, args.keep_going)
    if args.stage == "digest":
        print(json.dumps(write_digest(output), indent=1))
        return 0

    if output.exists() and any(output.iterdir()):
        parser.error(f"{output} already exists and is not empty")
    output.mkdir(parents=True, exist_ok=True)
    model_config, prompt_config = raw_config["model"], dict(raw_config["prompts"])
    if args.num_prompts is not None:
        prompt_config["lengths"] = prompt_config["lengths"][: args.num_prompts]
    if args.decode_steps is not None:
        raw_config["prompts"]["decode_steps"] = args.decode_steps
    if args.configurations is not None:
        raw_config["configurations"] = [entry for entry in raw_config["configurations"] if entry["name"] in args.configurations]
    if args.skip_audit:
        raw_config["development"] = {"skip_audit": True}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["repository"], revision=model_config["tokenizer_revision"], trust_remote_code=bool(model_config["trust_remote_code_tokenizer"])
    )
    prompts, provenance = build_prompts(prompt_config, tokenizer)
    with open(output / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_config, handle, sort_keys=False)
    with JsonlWriter(output / "prompts.jsonl") as log:
        for prompt in prompts:
            log.write(prompt)
    environment = environment_metadata(
        REPO_ROOT,
        {"repository": model_config["repository"], "revision": model_config["revision"], "dtype": model_config["dtype"], "device": "cuda"},
        NUMERICS_FLAGS,
    )
    environment["tokenizer"] = {"class": type(tokenizer).__name__, "revision": model_config["tokenizer_revision"], "trust_remote_code": True}
    environment["prompts"] = provenance
    environment["num_prompts"] = len(prompts)
    environment["python_hash_seed"] = os.environ.get("PYTHONHASHSEED")
    environment["profile"] = load_adapter(model_config).REFERENCE_PROFILE.to_json()
    environment["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    environment["system_memory"] = system_memory()
    from storage_runtime import describe_disk  # noqa: E402  (benchmarks/)

    from awpmi.storage.pack import open_pack

    files = open_pack(REPO_ROOT / raw_config["index"]["directory"], verify="size").files
    environment["storage"] = {"disk": describe_disk(next(iter(files.values())))}
    if args.reference_from is not None:
        # Phase 6A (decision 0012): several stream stages against one reference run, its records copied as they are.
        source = Path(args.reference_from)
        source_config = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
        same = {key: source_config[key] for key in ("model", "prompts", "reference")} == {key: raw_config[key] for key in ("model", "prompts", "reference")}
        if not same or read_jsonl(source / "prompts.jsonl") != prompts:
            parser.error(f"{source} ran another model, prompts or reference configuration")
        copied = {}
        for name in ("reference.jsonl.gz", "reference_rows.json", "reference_stage.json"):
            (output / name).write_bytes((source / name).read_bytes())
            copied[name] = sha256_file(output / name)
        environment["reference_from"] = {"run": source.as_posix(), "files_sha256": copied}
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    if args.stage == "prepare":
        print(f"prepared {output}: {len(prompts)} prompts, source tree {environment['source_tree_sha256'][:12]}")
        return 0

    command = [sys.executable, str(Path(__file__).resolve()), "--config", args.config, "--output", str(output)]
    if args.keep_going:
        command.append("--keep-going")
    codes = {}
    for stage in ("reference", "stream") if args.reference_from is None else ("stream",):
        started = time.perf_counter()
        codes[stage] = subprocess.run([*command, "--stage", stage], check=False).returncode
        print(f"stage {stage}: exit {codes[stage]} after {time.perf_counter() - started:.0f}s", flush=True)
        if codes[stage] and not args.keep_going:
            break
    write_digest(output)
    print(f"done: {codes}; output: {output}")
    return 1 if any(codes.values()) or len(codes) < (1 if args.reference_from is not None else 2) else 0


def write_digest(output: Path) -> dict:
    def jsonl(name: str):
        path = output / name
        return canonical_digest(read_jsonl(path), EXCLUDED_FIELDS) if path.exists() else None

    index_info = json.loads((output / "index.json").read_text(encoding="utf-8")) if (output / "index.json").exists() else {}
    rows = json.loads((output / "reference_rows.json").read_text(encoding="utf-8")) if (output / "reference_rows.json").exists() else {}
    digest = {
        "index_sha256": canonical_digest([index_info]),
        "reference_rows_sha256": canonical_digest([rows]),
        "prompts_sha256": canonical_digest(read_jsonl(output / "prompts.jsonl")),
        "reference_sha256": jsonl("reference.jsonl.gz"),
        "records_sha256": jsonl("records.jsonl.gz"),
        "ranges_sha256": canonical_digest(read_jsonl(output / "ranges.jsonl.gz")) if (output / "ranges.jsonl.gz").exists() else None,
        "excluded_fields": list(EXCLUDED_FIELDS),
    }
    (output / "digest.json").write_text(json.dumps(digest, indent=2), encoding="utf-8")
    return digest


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
    raise SystemExit(main())
