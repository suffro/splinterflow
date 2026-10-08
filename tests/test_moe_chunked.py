"""Phase 4B (decision 0008): experts calls split into chunks under a byte budget reproduce the reference bit for bit."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch
import transformers

from awpmi.models import checkpoint, moonlight
from awpmi.models.moe import ChunkedExpertWeight, ExpertCall, RoutingRecord, StreamedExperts, find_expert_modules, write_expert_pack
from awpmi.storage.cache import LRUPolicy, PageCache
from awpmi.storage.pack import SourceFile, open_pack
from tests.conftest import BACKENDS, CHUNKED_BACKENDS, DEVICES
from tests.test_moe import ARCHITECTURES, COMMON, tiny_model
from tests.test_moe_compact import RandomPolicy, assert_same_run, expert_store, greedy, prompts

IMPLEMENTATIONS = ("grouped_mm", "eager")


def matrix_bytes(model) -> int:
    """Bytes of one expert's matrices (the chunked parameters) in the first experts module."""
    entry = find_expert_modules(model)[0]
    return sum(getattr(entry.module, p)[0].numel() * getattr(entry.module, p).element_size() for p in entry.parameters if getattr(entry.module, p).dim() >= 3)


def small_bytes(model, slots: int) -> int:
    entry = find_expert_modules(model)[0]
    return slots * sum(getattr(entry.module, p)[0].numel() * getattr(entry.module, p).element_size() for p in entry.parameters if getattr(entry.module, p).dim() < 3)


def budget(model, experts: int) -> int:
    """A call budget that holds `experts` experts' matrices plus every small parameter of a call (8 experts + 2 spares)."""
    return experts * matrix_bytes(model) + small_bytes(model, 10)


@pytest.mark.parametrize("backend", CHUNKED_BACKENDS)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_chunked_calls_reproduce_the_resident_model_bitwise(tmp_path, device, architecture, implementation, backend):
    model = tiny_model(architecture, device)
    model.config._experts_implementation = implementation
    inputs = prompts(device)
    reference = [greedy(model, ids, steps=3) for ids in inputs]
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    modules = find_expert_modules(model)
    matrices = {m.name: sum(getattr(m.module, p).dim() >= 3 for p in m.parameters) for m in modules}
    limits = {experts: budget(model, experts) for experts in (1, 3)}
    for experts in (1, 3):
        store = expert_store(pack, device, backend=backend)
        routes: list[RoutingRecord] = []
        calls: list[ExpertCall] = []
        streamed = StreamedExperts(
            model, store, compact=True, poison=True, max_call_bytes=limits[experts], on_route=routes.append, on_call=calls.append,
            prefetch_chunks=backend == "native-prefetch",
        ).install()
        try:
            for ids, expected in zip(inputs, reference):
                assert_same_run(greedy(model, ids, steps=3), expected)
            if backend == "native-prefetch":
                # Every chunked call's rows were announced and used: none wasted, none read twice.
                cache = store.weights.backend.store.cache_stats()
                assert cache["prefetch_fills"] > 0 and cache["prefetch_wasted"] == 0
                assert cache["prefetch_used"] == cache["prefetch_fills"] and cache["aborted_fills"] == 0
            # Every call ran in chunks of consecutive slots covering exactly its routed experts, in order.
            assert streamed.chunked_calls == streamed.calls == len(calls) > 0
            for call, route in zip(calls, routes):
                assert call.served == route.experts
                assert [first for first, _ in call.chunks] == [0, *(end for _, end in call.chunks[:-1])] and call.chunks[-1][1] == len(call.served)
                assert len({end - first for first, end in call.chunks[:-1]}) <= 1 and all(end > first for first, end in call.chunks)
            assert streamed.chunk_loads == sum(len(call.chunks) * matrices[call.module] for call in calls)  # each chunk of each matrix once
            assert streamed.peak_compact_bytes <= limits[experts] and streamed.compact_bytes == 0
            for entry in modules:  # between calls nothing is reachable
                assert all(getattr(entry.module, p) is None for p in entry.parameters)
        finally:
            streamed.remove()
            store.weights.backend.store.close()


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_reordering_the_accumulation_across_chunks_changes_results(implementation):
    """Adversarial: summing each chunk's partial output (a different association) is not the reference's result.

    The chunked path keeps the experts implementation's own combine (one per call), which is exact;
    combining per chunk is what it must not do.
    """
    device = DEVICES[-1]
    model = tiny_model("deepseek_v3", device, seed=12)
    model.config._experts_implementation = implementation
    entry = find_expert_modules(model)[0]
    generator = torch.Generator().manual_seed(12)
    differing = 0
    for trial in range(10):
        hidden = (torch.randn(64, 64, generator=generator) * 4).to(torch.bfloat16).to(device)
        index = torch.stack([torch.randperm(8, generator=generator)[:4] for _ in range(64)]).to(device)
        weights = torch.rand(64, 4, generator=generator).to(device)
        full = entry.module(hidden, index, weights)
        partials = []
        for chunk in ([0, 1, 2, 3], [4, 5, 6, 7]):  # zero weights drop the other chunk's assignments exactly
            mask = torch.isin(index, torch.tensor(chunk, device=device))
            partials.append(entry.module(hidden, index, weights * mask))
        # grouped_mm sums each token's terms in float32 and rounds once: per-chunk partials round twice.
        # eager accumulates in BF16 in expert order: the second chunk first is another association.
        reordered = partials[1] + partials[0]
        differing += int(not torch.equal(reordered, full))
    assert differing > 0


def test_chunked_weights_fail_loudly_outside_their_two_uses(tmp_path):
    device = DEVICES[-1]
    model = tiny_model("olmoe", device, seed=13)
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    store = expert_store(pack, device)
    streamed = StreamedExperts(model, store, compact=True, poison=True, max_call_bytes=budget(model, 1)).install()
    entry = streamed.modules[0]
    seen: list[ChunkedExpertWeight] = []

    def probe(module, args, kwargs):  # runs after StreamedExperts' own pre-hook
        weight = module.gate_up_proj
        assert isinstance(weight, ChunkedExpertWeight) and not isinstance(weight, torch.Tensor)
        for misuse in (
            lambda: torch.zeros_like(weight),
            lambda: weight + 1,
            lambda: weight[torch.tensor([0, 1])],
            lambda: weight.transpose(0, 1),
            lambda: weight.data,
            lambda: weight[weight.shape[0] - 1],  # a poison spare: serves no expert
            lambda: torch.matmul(args[0], weight),
        ):
            with pytest.raises((TypeError, AttributeError, IndexError)):
                misuse()
        seen.append(weight)

    handle = entry.module.register_forward_pre_hook(probe, with_kwargs=True)
    try:
        ids = torch.randint(0, 128, (1, 6), generator=torch.Generator().manual_seed(13)).to(device)
        greedy(model, ids, steps=1)
        with pytest.raises(RuntimeError):
            seen[0][0]  # after its call a stand-in reaches no weight
        assert all(getattr(entry.module, p) is None for p in entry.parameters)
    finally:
        handle.remove()
        streamed.remove()
        store.weights.backend.store.close()


def test_a_budget_below_one_expert_is_refused(tmp_path):
    device = DEVICES[-1]
    model = tiny_model("qwen3_moe", device, seed=14)
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    store = expert_store(pack, device)
    streamed = StreamedExperts(model, store, compact=True, max_call_bytes=matrix_bytes(model) - 1).install()
    try:
        with pytest.raises(ValueError):
            greedy(model, torch.arange(5).view(1, -1).to(device), steps=0)
    finally:
        streamed.remove()
        store.weights.backend.store.close()


@pytest.mark.skipif("cuda" not in DEVICES, reason="device memory is measured on CUDA")
def test_a_chunked_prefill_never_allocates_a_layer_of_experts(tmp_path):
    """Gate C in miniature: a prefill routing every expert stays under the budget; unbounded, it holds the layer."""
    config_name, model_name, extra = ARCHITECTURES["deepseek_v3"]
    config = getattr(transformers, config_name)(
        **{**COMMON, "hidden_size": 256}, **{**extra, "moe_intermediate_size": 512, "n_routed_experts": 16, "n_group": 1, "num_experts_per_tok": 4}
    )
    torch.manual_seed(15)
    model = getattr(transformers, model_name)(config).to(torch.bfloat16).to("cuda").eval()
    entry = find_expert_modules(model)[0]
    layer_bytes = sum(getattr(entry.module, p).numel() * getattr(entry.module, p).element_size() for p in entry.parameters)
    ids = torch.randint(0, 128, (1, 48), generator=torch.Generator().manual_seed(15)).cuda()
    expected = greedy(model, ids, steps=0)
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    peaks = {}
    for label, limit in (("chunked", 3 * layer_bytes // 16), ("unbounded", None)):
        store = expert_store(pack, "cuda")
        routes: list[RoutingRecord] = []
        streamed = StreamedExperts(model, store, compact=True, max_call_bytes=limit, on_route=routes.append).install()
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            assert_same_run(greedy(model, ids, steps=0), expected)
            peaks[label] = (torch.cuda.max_memory_allocated() - baseline, streamed.peak_compact_bytes)
            assert all(len(route.experts) == 16 for route in routes)  # every expert routed: a full layer's working set
        finally:
            streamed.remove()
            store.weights.backend.store.close()
    assert peaks["chunked"][1] <= 3 * layer_bytes // 16 < layer_bytes
    assert peaks["chunked"][0] < layer_bytes // 2  # the allocator never held anything near a layer
    assert peaks["unbounded"][1] == layer_bytes and peaks["unbounded"][0] >= layer_bytes  # the guard sees a layer when unbounded


@pytest.mark.parametrize("policy", [LRUPolicy, lambda: RandomPolicy(2)])
def test_cache_evictions_during_chunked_calls_change_nothing(tmp_path, policy):
    device = DEVICES[-1]
    model = tiny_model("deepseek_v3", device, seed=16)
    ids = torch.randint(0, 128, (1, 14), generator=torch.Generator().manual_seed(16)).to(device)
    expected = greedy(model, ids, steps=8)
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    cache = PageCache(6 * matrix_bytes(model), policy())  # 6 of 16 experts: smaller than a prefill's working set
    store = expert_store(pack, device, cache)
    streamed = StreamedExperts(model, store, compact=True, max_call_bytes=budget(model, 2)).install()
    try:
        assert_same_run(greedy(model, ids, steps=8), expected)
        assert cache.stats.evictions > 0 and cache.stats.hits > 0 and streamed.chunked_calls > 0
        served = store.weights.backend.report()["materialization"]
        assert served["cache_hit_rows"] + served["fetched_rows"] == served["rows"]
        assert served["largest_request_bytes"] <= 2 * max(pack.segments[s].row_bytes for s in pack.segments)
    finally:
        streamed.remove()
        store.weights.backend.store.close()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", DEVICES)
def test_per_assignment_outputs_of_chunked_calls_equal_the_full_layers(tmp_path, device, backend):
    model = tiny_model("deepseek_v3", device, seed=17)
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    ids = torch.randint(0, 128, (1, 12), generator=torch.Generator().manual_seed(17)).to(device)
    seen: dict[str, list] = {"full": [], "chunked": []}
    full_store = expert_store(pack, device)
    streamed = StreamedExperts(model, full_store, on_call=lambda call: seen["full"].append(call.per_assignment_outputs().clone())).install()
    greedy(model, ids, steps=2)
    streamed.remove()
    full_store.weights.backend.store.close()
    store = expert_store(pack, device, backend=backend)
    check = expert_store(pack, device)  # the caller's own source of a chunked call's experts (no cache)
    calls: list[ExpertCall] = []
    streamed = StreamedExperts(model, store, compact=True, max_call_bytes=budget(model, 1), on_call=calls.append).install()
    try:
        greedy(model, ids, steps=2)
        assert all(call.chunks is not None for call in calls)
        with pytest.raises(RuntimeError):
            calls[0].per_assignment_outputs()  # its weights are gone
        got = [call.per_assignment_outputs(weights=lambda experts, call=call: check.load(call.module, experts)) for call in calls]
    finally:
        streamed.remove()
        store.weights.backend.store.close()
        check.weights.backend.store.close()
    assert len(got) == len(seen["full"]) == 2 * 3
    assert all(torch.equal(a, b) for a, b in zip(got, seen["full"]))


def test_dense_streaming_in_chunks(tmp_path):
    device = DEVICES[-1]
    model = tiny_model("mixtral", device, seed=18)
    ids = torch.randint(0, 128, (1, 5), generator=torch.Generator().manual_seed(18)).to(device)
    expected = greedy(model, ids, steps=3)
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    store = expert_store(pack, device)
    streamed = StreamedExperts(model, store, compact=True, all_experts=True, max_call_bytes=budget(model, 3)).install()
    try:
        assert_same_run(greedy(model, ids, steps=3), expected)
        report = store.weights.backend.report()["materialization"]
        entry = streamed.modules[0]
        assert report["rows"] == streamed.calls * entry.num_experts * len(entry.parameters)  # every expert, once per call
    finally:
        streamed.remove()
        store.weights.backend.store.close()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_split_deepseek_checkpoint_is_served_in_chunks(tmp_path, implementation, backend):
    """A DeepSeek-V3 checkpoint as published (one tensor per expert projection), read in place, in chunks."""
    device = DEVICES[-1]
    source = tiny_model("deepseek_v3", "cpu", seed=19)
    source.save_pretrained(tmp_path / "checkpoint")
    files = {p.name: p for p in sorted((tmp_path / "checkpoint").glob("*.safetensors"))}
    reference_model = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "checkpoint", dtype=torch.bfloat16).to(device).eval()
    reference_model.config._experts_implementation = implementation
    inputs = prompts(device)
    reference = [greedy(reference_model, ids, steps=3) for ids in inputs]
    model, _ = checkpoint.load_model_without_experts(str(tmp_path / "checkpoint"), None, torch.bfloat16, device, files)
    model.config._experts_implementation = implementation
    files_info = {k: checkpoint.CheckpointFile(k, SourceFile("example/split", "0" * 40, k), p, None) for k, p in files.items()}
    checkpoint.write_expert_index(model, tmp_path / "index", files_info, sources=moonlight.expert_sources(model))
    pack = open_pack(tmp_path / "index", verify="size", resolve=lambda requested: files[requested.filename])
    store = expert_store(pack, device, backend=backend)
    streamed = StreamedExperts(model, store, compact=True, poison=True, max_call_bytes=budget(model, 1)).install()
    try:
        for ids, expected in zip(inputs, reference):
            assert_same_run(greedy(model, ids, steps=3), expected)
        assert streamed.chunked_calls == streamed.calls
    finally:
        streamed.remove()
        store.weights.backend.store.close()


CHUNKED_RUN = """
import json, sys, torch, transformers
from awpmi.runtime import configure_reproducible_numerics
configure_reproducible_numerics()
from awpmi.materialization.backend import MaterializationBackend
from awpmi.materialization.weights import ExpertStore, WeightStore
from awpmi.models.moe import StreamedExperts, groups_from_pack, write_expert_pack
from awpmi.storage.cache import POLICIES, PageCache
from awpmi.storage.pack import open_pack
from awpmi.streaming.streamer import PageStreamer
from awpmi.tracing import tensor_digest
directory, policy = sys.argv[1], sys.argv[2]
config = transformers.DeepseekV3Config(vocab_size=128, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=64, intermediate_size=48, moe_intermediate_size=32, n_routed_experts=8, num_experts_per_tok=2,
    n_shared_experts=1, n_group=2, topk_group=1, first_k_dense_replace=0, q_lora_rank=None, kv_lora_rank=16, qk_rope_head_dim=8,
    qk_nope_head_dim=8, v_head_dim=16, rope_scaling=None)
torch.manual_seed(0)
model = transformers.DeepseekV3ForCausalLM(config).to(torch.bfloat16).eval()
pack = open_pack(write_expert_pack(model, directory).directory)
row = sum(pack.segments[s].row_bytes for s in groups_from_pack(pack)["model.layers.0.mlp.experts"].segments.values())
cache = PageCache(4 * row, POLICIES[policy]())
backend = MaterializationBackend(pack.store(direct=True), "cpu", PageStreamer("cpu"), cache)
streamed = StreamedExperts(model, ExpertStore(WeightStore(backend), groups_from_pack(pack)), compact=True, max_call_bytes=2 * row).install()
evicted = []
original = cache._evict
cache._evict = lambda key: (evicted.append(list(key)), original(key))
digests = []
with torch.inference_mode():
    for length in (5, 11):
        output = model(input_ids=torch.arange(length).view(1, -1) % 128, use_cache=True)
        for step in range(4):
            digests.append(tensor_digest(output.logits[:, -1]))
            output = model(input_ids=output.logits[:, -1].argmax(-1, keepdim=True), past_key_values=output.past_key_values, use_cache=True)
print(json.dumps({"digests": digests, "evicted": evicted, "chunked": streamed.chunked_calls, "stats": cache.stats.as_dict()}))
"""


@pytest.mark.parametrize("policy", ["lru", "hotness"])
def test_chunked_cached_runs_do_not_depend_on_the_process_hash_seed(tmp_path, policy):
    outputs = []
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", CHUNKED_RUN, str(tmp_path / f"pack-{seed}"), policy], capture_output=True, text=True, env=env, check=True
        )
        outputs.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert outputs[0]["evicted"] and outputs[0]["chunked"] > 0 and outputs[0]["stats"]["hits"] > 0
    assert outputs[0] == outputs[1] == outputs[2]
