"""Phase 4A MoE (decision 0007): compact experts calls, split checkpoints served in place, the offload reference."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys

import pytest
import torch
import transformers

from awpmi.materialization.weights import ExpertStore, WeightStore
from awpmi.models import checkpoint, olmoe
from awpmi.models.moe import ExpertCall, FullLayerOffload, RoutingRecord, StreamedExperts, find_expert_modules, groups_from_pack, write_expert_pack
from awpmi.storage.cache import HotnessPolicy, LRUPolicy, PageCache, ReplacementPolicy
from awpmi.storage.pack import SourceFile, open_pack
from tests.conftest import BACKENDS, DEVICES, materialization_backend
from tests.test_moe import ARCHITECTURES, COMMON, tiny_model

# Architectures whose checkpoints split each expert into separate tensors (save_pretrained reverses the fusion).
SPLIT = ("mixtral", "qwen2_moe", "qwen3_moe", "olmoe", "deepseek_v3")


def kv_tensors(cache) -> list[torch.Tensor]:
    return [t for layer in cache.layers for t in (layer.keys, layer.values) if isinstance(t, torch.Tensor)]


@torch.inference_mode()
def greedy(model, input_ids: torch.Tensor, steps: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Logits of the prompt's last position and of `steps` greedy tokens (KV cache), and the final cache."""
    output = model(input_ids=input_ids, use_cache=True)
    logits = [output.logits[:, -1]]
    cache = output.past_key_values
    for _ in range(steps):
        token = logits[-1].argmax(dim=-1, keepdim=True)
        output = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        logits.append(output.logits[:, -1])
    return logits, [t.clone() for t in kv_tensors(cache)]


def assert_same_run(got, expected) -> None:
    for a, b in zip(got[0], expected[0], strict=True):
        assert torch.equal(a, b)
    for a, b in zip(got[1], expected[1], strict=True):
        assert torch.equal(a, b)


def expert_store(pack, device, cache: PageCache | None = None, backend: str = "python") -> ExpertStore:
    backend = materialization_backend(pack, backend, device, cache, direct=True)
    return ExpertStore(WeightStore(backend), groups_from_pack(pack))


def prompts(device: str) -> list[torch.Tensor]:
    return [torch.randint(0, 128, (1, n), generator=torch.Generator().manual_seed(n)).to(device) for n in (1, 9, 23)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_compact_experts_reproduce_the_resident_model_bitwise(tmp_path, device, architecture, backend):
    model = tiny_model(architecture, device)
    inputs = prompts(device)
    reference = [greedy(model, ids, steps=3) for ids in inputs]
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    store = expert_store(pack, device, backend=backend)
    modules = find_expert_modules(model)
    routes: list[RoutingRecord] = []
    calls: list[tuple[int, list[int], int]] = []

    def observe(call: ExpertCall) -> None:
        slots = {getattr(call.experts_module, p).shape[0] for p in modules[0].parameters}
        calls.append((slots.pop(), call.served, int(call.call_index.max())))

    streamed = StreamedExperts(model, store, compact=True, poison=True, on_route=routes.append, on_call=observe).install()
    try:
        for ids, expected in zip(inputs, reference):
            routes.clear()
            calls.clear()
            assert_same_run(greedy(model, ids, steps=3), expected)
            # Each call held exactly its routed experts (ascending), plus the NaN spares, and used only them.
            assert [served for _, served, _ in calls] == [route.experts for route in routes]
            for slots, served, highest in calls:
                assert slots == len(served) + 2 and highest < len(served) and served == sorted(set(served))
        for entry in modules:  # between calls the full expert tensors exist nowhere
            assert all(getattr(entry.module, p) is None for p in entry.parameters)
        assert streamed.compact_bytes == 0
    finally:
        streamed.remove()
        store.weights.backend.store.close()


def test_poisoning_a_served_slot_changes_the_output(tmp_path):
    """Sabotage check of the compact poison mode: NaN in a slot that the call uses must show."""
    device = DEVICES[-1]
    model = tiny_model("olmoe", device, seed=11)
    ids = torch.randint(0, 128, (1, 5), generator=torch.Generator().manual_seed(11)).to(device)
    expected = greedy(model, ids, steps=1)
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    store = expert_store(pack, device)
    original = store.assemble

    def assemble_then_poison_first_slot(key, experts, buffers):
        original(key, experts, buffers)
        for buffer in buffers.values():
            buffer[0].fill_(float("nan"))

    store.assemble = assemble_then_poison_first_slot
    streamed = StreamedExperts(model, store, compact=True, poison=True).install()
    try:
        got = greedy(model, ids, steps=1)
        assert not all(torch.equal(a, b) for a, b in zip(got[0], expected[0]))
    finally:
        streamed.remove()
        store.weights.backend.store.close()


def test_compact_buffers_follow_the_routed_experts_not_the_expert_count(tmp_path):
    device = DEVICES[-1]
    model = tiny_model("olmoe", device, seed=4)
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    store = expert_store(pack, device)
    entry = find_expert_modules(model)[0]
    expert_bytes = sum(getattr(entry.module, p)[0].numel() * getattr(entry.module, p).element_size() for p in entry.parameters)
    streamed = StreamedExperts(model, store, compact=True).install()
    routes: list[RoutingRecord] = []
    streamed.on_route = routes.append
    try:
        ids = torch.randint(0, 128, (1, 7), generator=torch.Generator().manual_seed(3)).to(device)
        with torch.inference_mode():
            output = model(input_ids=ids, use_cache=True)
            streamed.reset_peak()
            routes.clear()
            model(input_ids=output.logits[:, -1].argmax(-1, keepdim=True), past_key_values=output.past_key_values, use_cache=True)
        assert all(len(route.experts) == 2 for route in routes)  # one token, top-2
        assert streamed.peak_compact_bytes == 2 * expert_bytes < entry.num_experts * expert_bytes
    finally:
        streamed.remove()
        store.weights.backend.store.close()


class RandomPolicy(ReplacementPolicy):
    """Evicts a random page (seeded): residency must never change a result."""

    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed)

    def touch(self, key) -> None:
        pass

    def forget(self, key) -> None:
        pass

    def victim(self, candidates: set):
        return self._random.choice(sorted(candidates, key=repr))


@pytest.mark.parametrize("policy", [LRUPolicy, HotnessPolicy, lambda: RandomPolicy(0), lambda: RandomPolicy(1)])
def test_cached_compact_experts_reproduce_the_resident_model(tmp_path, policy):
    device = DEVICES[-1]
    model = tiny_model("qwen3_moe", device, seed=1)
    ids = torch.randint(0, 128, (1, 6), generator=torch.Generator().manual_seed(1)).to(device)
    expected = greedy(model, ids, steps=8)
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    row = sum(pack.segments[s].row_bytes for s in groups_from_pack(pack)["model.layers.0.mlp.experts"].segments.values())
    cache = PageCache(5 * row, policy())
    store = expert_store(pack, device, cache)
    streamed = StreamedExperts(model, store, compact=True).install()
    try:
        assert_same_run(greedy(model, ids, steps=8), expected)
        assert cache.stats.hits > 0 and cache.stats.misses > 0 and cache.peak_resident_bytes <= 5 * row
        served = store.weights.backend.report()["materialization"]
        assert served["cache_hit_rows"] + served["fetched_rows"] == served["rows"]
    finally:
        streamed.remove()
        store.weights.backend.store.close()


def test_top1_routing_and_dense_streaming_in_compact_mode(tmp_path):
    device = DEVICES[-1]
    config_name, model_name, extra = ARCHITECTURES["mixtral"]
    config = getattr(transformers, config_name)(**COMMON, **{**extra, "num_experts_per_tok": 1})
    torch.manual_seed(5)
    model = getattr(transformers, model_name)(config).to(torch.bfloat16).to(device).eval()
    ids = torch.randint(0, 128, (1, 5), generator=torch.Generator().manual_seed(5)).to(device)
    expected = greedy(model, ids, steps=4)
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    for all_experts in (False, True):
        store = expert_store(pack, device)
        streamed = StreamedExperts(model, store, compact=True, all_experts=all_experts).install()
        try:
            assert_same_run(greedy(model, ids, steps=4), expected)
        finally:
            streamed.remove()
            store.weights.backend.store.close()


def test_placeholders_fail_when_used_outside_a_call(tmp_path):
    device = DEVICES[-1]
    model = tiny_model("olmoe", device, seed=6)
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    store = expert_store(pack, device)
    streamed = StreamedExperts(model, store, compact=True).install()
    entry = streamed.modules[0]
    try:
        hidden = torch.randn(3, 64, device=device).to(torch.bfloat16)
        index = torch.tensor([[0, 1], [2, 3], [1, 5]], device=device)
        weights = torch.full((3, 2), 0.5, device=device).to(torch.bfloat16)
        with pytest.raises((TypeError, AttributeError, RuntimeError)):
            entry.module.forward(hidden, index, weights)  # no hook: there are no expert tensors to use
        served = entry.module(hidden, index, weights)  # through the hook: served from storage
        assert served.device.type == hidden.device.type and torch.isfinite(served).all()
        assert all(getattr(entry.module, p) is None for p in entry.parameters)
    finally:
        streamed.remove()
        store.weights.backend.store.close()
    # After removal the experts are meta again (shapes kept), so another server can be installed.
    assert {m.name for m in find_expert_modules(model)} == {f"model.layers.{i}.mlp.experts" for i in range(2)}
    store = expert_store(pack, device)
    streamed = StreamedExperts(model, store, compact=True).install()
    try:
        assert torch.isfinite(entry.module(hidden, index, weights)).all()
    finally:
        streamed.remove()
        store.weights.backend.store.close()


@pytest.mark.parametrize("resident", [(), ("model.layers.1.mlp.experts",)])
def test_full_layer_offload_reproduces_the_resident_model(resident):
    device = DEVICES[-1]
    if device != "cuda":
        pytest.skip("offloading needs a device away from host memory")
    model = tiny_model("deepseek_v3", device, seed=7)
    inputs = prompts(device)
    reference = [greedy(model, ids, steps=3) for ids in inputs]
    offload = FullLayerOffload(model, device, resident=resident).install()
    try:
        for entry in find_expert_modules(model):
            for parameter in entry.parameters:
                assert getattr(entry.module, parameter).device.type == ("cuda" if entry.name in resident else "cpu")
        for ids, expected in zip(inputs, reference):
            assert_same_run(greedy(model, ids, steps=3), expected)
        assert offload.copied_bytes > 0
    finally:
        offload.remove()


def test_per_assignment_outputs_agree_between_the_offload_reference_and_compact_calls(tmp_path):
    device = DEVICES[-1]
    model = tiny_model("olmoe", device, seed=8)
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    ids = torch.randint(0, 128, (1, 11), generator=torch.Generator().manual_seed(8)).to(device)
    seen: dict[str, list] = {"reference": [], "compact": []}

    def recorder(kind):
        return lambda call: seen[kind].append((call.output.clone(), call.per_assignment_outputs().clone()))

    if device == "cuda":
        offload = FullLayerOffload(model, device, on_call=recorder("reference")).install()
        greedy(model, ids, steps=2)
        offload.remove()
        for entry in find_expert_modules(model):  # back on the device for the compact run's source
            for parameter in entry.parameters:
                tensor = getattr(entry.module, parameter)
                entry.module._parameters[parameter] = torch.nn.Parameter(tensor.detach().to(device), requires_grad=False)
    else:
        streamed = StreamedExperts(model, expert_store(pack, device), on_call=recorder("reference")).install()
        greedy(model, ids, steps=2)
        streamed.remove()
    store = expert_store(pack, device)
    streamed = StreamedExperts(model, store, compact=True, on_call=recorder("compact")).install()
    try:
        greedy(model, ids, steps=2)
    finally:
        streamed.remove()
        store.weights.backend.store.close()
    assert len(seen["reference"]) == len(seen["compact"]) == 2 * 3
    for (out_a, each_a), (out_b, each_b) in zip(seen["reference"], seen["compact"]):
        assert torch.equal(out_a, out_b) and torch.equal(each_a, each_b)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("architecture", SPLIT)
def test_split_checkpoints_are_served_in_place(tmp_path, architecture, backend):
    """save_pretrained splits experts again; the index refers to those tensors, and streaming them is exact."""
    device = DEVICES[-1]
    source = tiny_model(architecture, "cpu", seed=9)
    source.save_pretrained(tmp_path / "checkpoint")
    files = {p.name: p for p in sorted((tmp_path / "checkpoint").glob("*.safetensors"))}
    reference_model = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "checkpoint", dtype=torch.bfloat16).to(device).eval()
    inputs = prompts(device)
    reference = [greedy(reference_model, ids, steps=3) for ids in inputs]

    model, report = checkpoint.load_model_without_experts(str(tmp_path / "checkpoint"), None, torch.bfloat16, device, files)
    modules = find_expert_modules(model)
    assert all(getattr(m.module, p).is_meta for m in modules for p in m.parameters)
    expected_state = reference_model.state_dict()
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        if not tensor.is_meta:
            assert tensor.dtype == expected_state.get(name, tensor).dtype
            assert torch.equal(tensor, dict(reference_model.named_buffers()).get(name, expected_state.get(name))), name
    files_info = {k: checkpoint.CheckpointFile(k, SourceFile("example/split", "0" * 40, k), p, None) for k, p in files.items()}
    index = checkpoint.write_expert_index(model, tmp_path / "index", files_info)
    assert not any(p.suffix == ".safetensors" for p in (tmp_path / "index").iterdir())
    resolve = lambda requested: files[requested.filename]  # noqa: E731
    pack = open_pack(tmp_path / "index", verify="size", resolve=resolve)
    assert pack.segments == index.segments and pack.manifest["format_version"] == 2
    store = expert_store(pack, device, backend=backend)
    streamed = StreamedExperts(model, store, compact=True, poison=True).install()
    try:
        for ids, expected in zip(inputs, reference):
            assert_same_run(greedy(model, ids, steps=3), expected)
    finally:
        streamed.remove()
        store.weights.backend.store.close()


def test_olmoe_adapter_checks_its_layout_against_transformers(monkeypatch):
    model = tiny_model("olmoe", "cpu", seed=10)
    sources = olmoe.expert_sources(model)
    assert sources["model.layers.1.mlp.experts"]["gate_up_proj"] == (
        "model.layers.1.mlp.experts.{expert}.gate_proj.weight", "model.layers.1.mlp.experts.{expert}.up_proj.weight",
    )
    assert {name: type(router).__name__ for name, router in olmoe.routers(model).items()} == {
        f"model.layers.{i}.mlp.experts": "OlmoeTopKRouter" for i in range(2)
    }
    swapped = {**olmoe.EXPERT_LAYOUT, "gate_up_proj": olmoe.EXPERT_LAYOUT["gate_up_proj"][::-1]}
    monkeypatch.setattr(olmoe, "EXPERT_LAYOUT", swapped)
    with pytest.raises(ValueError):
        olmoe.expert_sources(model)
    olmoe.REFERENCE_PROFILE.check_model(transformers.OlmoeForCausalLM(model.config).to(torch.bfloat16))


CACHED_RUN = """
import json, torch, transformers
from awpmi.runtime import configure_reproducible_numerics
configure_reproducible_numerics()
from awpmi.materialization.backend import MaterializationBackend
from awpmi.materialization.weights import ExpertStore, WeightStore
from awpmi.models.moe import StreamedExperts, groups_from_pack, write_expert_pack
from awpmi.storage.cache import POLICIES, PageCache
from awpmi.storage.pack import open_pack
from awpmi.streaming.streamer import PageStreamer
from awpmi.tracing import tensor_digest
import sys
directory, policy = sys.argv[1], sys.argv[2]
config = transformers.OlmoeConfig(vocab_size=128, hidden_size=64, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, max_position_embeddings=64, num_experts=8, num_experts_per_tok=2, eos_token_id=None, pad_token_id=None)
torch.manual_seed(0)
model = transformers.OlmoeForCausalLM(config).to(torch.bfloat16).eval()
pack = open_pack(write_expert_pack(model, directory).directory)
cache = PageCache(10 * sum(pack.segments[s].row_bytes for s in groups_from_pack(pack)["model.layers.0.mlp.experts"].segments.values()), POLICIES[policy]())
backend = MaterializationBackend(pack.store(direct=True), "cpu", PageStreamer("cpu"), cache)
StreamedExperts(model, ExpertStore(WeightStore(backend), groups_from_pack(pack)), compact=True).install()
evicted = []
original = cache._evict
cache._evict = lambda key: (evicted.append(list(key)), original(key))
digests = []
with torch.inference_mode():
    for length in (4, 9):
        output = model(input_ids=torch.arange(length).view(1, -1) % 128, use_cache=True)
        for step in range(5):
            digests.append(tensor_digest(output.logits[:, -1]))
            output = model(input_ids=output.logits[:, -1].argmax(-1, keepdim=True), past_key_values=output.past_key_values, use_cache=True)
print(json.dumps({"digests": digests, "evicted": evicted, "stats": cache.stats.as_dict()}))
"""


@pytest.mark.parametrize("policy", ["lru", "hotness"])
def test_compact_cached_runs_do_not_depend_on_the_process_hash_seed(tmp_path, policy):
    outputs = []
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", CACHED_RUN, str(tmp_path / f"pack-{seed}"), policy], capture_output=True, text=True, env=env, check=True
        )
        outputs.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert outputs[0]["evicted"] and outputs[0]["stats"]["hits"] > 0
    assert outputs[0] == outputs[1] == outputs[2]
