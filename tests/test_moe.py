"""Phase 3 MoE backend: experts served from storage reproduce the resident model bitwise, on several architectures."""

from __future__ import annotations

import pytest
import torch
import transformers

from awpmi.materialization.backend import MaterializationBackend
from awpmi.materialization.weights import ExpertStore, WeightStore
from awpmi.models.moe import RoutingRecord, StreamedExperts, find_expert_modules, groups_from_pack, write_expert_pack
from awpmi.storage.cache import HotnessPolicy, LRUPolicy, PageCache
from awpmi.storage.pack import SourceFile, open_pack
from awpmi.streaming.streamer import PageStreamer
from tests.conftest import BACKENDS, DEVICES, page_store

COMMON = dict(vocab_size=128, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
# Architecture fixtures only: the adapter itself knows none of these names.
ARCHITECTURES = {
    "mixtral": ("MixtralConfig", "MixtralForCausalLM", dict(intermediate_size=32, num_local_experts=8, num_experts_per_tok=2)),
    "qwen2_moe": (
        "Qwen2MoeConfig", "Qwen2MoeForCausalLM",
        dict(intermediate_size=48, moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=8, num_experts_per_tok=2),
    ),
    "qwen3_moe": ("Qwen3MoeConfig", "Qwen3MoeForCausalLM", dict(intermediate_size=48, moe_intermediate_size=32, num_experts=8, num_experts_per_tok=2, head_dim=16)),
    "olmoe": ("OlmoeConfig", "OlmoeForCausalLM", dict(intermediate_size=32, num_experts=8, num_experts_per_tok=2, eos_token_id=None, pad_token_id=None)),
    "granitemoe": ("GraniteMoeConfig", "GraniteMoeForCausalLM", dict(intermediate_size=32, num_local_experts=8, num_experts_per_tok=2)),
    "deepseek_v3": (
        "DeepseekV3Config", "DeepseekV3ForCausalLM",
        dict(
            intermediate_size=48, moe_intermediate_size=32, n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
            n_group=2, topk_group=1, first_k_dense_replace=0, q_lora_rank=None, kv_lora_rank=16, qk_rope_head_dim=8,
            qk_nope_head_dim=8, v_head_dim=16, rope_scaling=None,
        ),
    ),
    "gpt_oss": ("GptOssConfig", "GptOssForCausalLM", dict(intermediate_size=32, num_local_experts=8, num_experts_per_tok=2, head_dim=16, sliding_window=16)),
}


def tiny_model(architecture: str, device: str, seed: int = 0) -> transformers.PreTrainedModel:
    config_name, model_name, extra = ARCHITECTURES[architecture]
    config = getattr(transformers, config_name)(**COMMON, **extra)
    torch.manual_seed(seed)
    return getattr(transformers, model_name)(config).to(torch.bfloat16).to(device).eval()


@torch.inference_mode()
def greedy_logits(model, input_ids: torch.Tensor, steps: int) -> list[torch.Tensor]:
    """Logits of the prompt's last position, then of `steps` greedy tokens decoded with the KV cache."""
    output = model(input_ids=input_ids, use_cache=True)
    logits = [output.logits[:, -1]]
    cache = output.past_key_values
    for _ in range(steps):
        token = logits[-1].argmax(dim=-1, keepdim=True)
        output = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        logits.append(output.logits[:, -1])
    return logits


def expert_store(pack, device: str, cache: PageCache | None = None, backend: str = "python") -> ExpertStore:
    backend = MaterializationBackend(page_store(pack, backend, direct=True), device, PageStreamer(device), cache)
    return ExpertStore(WeightStore(backend), groups_from_pack(pack))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_streamed_experts_reproduce_the_resident_model_bitwise(tmp_path, device, architecture, backend):
    model = tiny_model(architecture, device)
    prompts = [torch.randint(0, 128, (1, length), generator=torch.Generator().manual_seed(length)).to(device) for length in (1, 9, 23)]
    reference = [greedy_logits(model, ids, steps=3) for ids in prompts]
    modules = find_expert_modules(model)
    assert len(modules) == 2 and all(m.num_experts == 8 for m in modules)
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    store = expert_store(pack, device, backend=backend)
    routes: list[RoutingRecord] = []
    streamed = StreamedExperts(model, store, poison=True, on_route=routes.append).install()
    backend = store.weights.backend
    try:
        for ids, expected in zip(prompts, reference):
            backend.reset_stats()
            routes.clear()
            got = greedy_logits(model, ids, steps=3)
            for a, b in zip(got, expected, strict=True):
                assert torch.equal(a, b)
            # Storage served exactly the routed experts of every call, each once per call.
            rows = sum(len(route.experts) for route in routes)
            report = backend.report()
            assert report["materialization"]["rows"] == rows * len(modules[0].parameters)
            assert report["storage"]["rows"] == report["materialization"]["rows"]
            assert all(route.experts == sorted(set(route.experts)) for route in routes)
            if ids.shape[1] == 1:
                assert all(len(route.experts) <= 2 for route in routes)  # one token: top-2 experts only
    finally:
        streamed.remove()
        backend.store.close()


@pytest.mark.parametrize("policy", [LRUPolicy, HotnessPolicy])
def test_cached_experts_reproduce_the_resident_model(tmp_path, policy):
    device = DEVICES[-1]
    model = tiny_model("mixtral", device, seed=1)
    ids = torch.randint(0, 128, (1, 6), generator=torch.Generator().manual_seed(1)).to(device)
    expected = greedy_logits(model, ids, steps=8)
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    row = pack.segments["model.layers.0.mlp.experts.gate_up_proj"].row_bytes + pack.segments["model.layers.0.mlp.experts.down_proj"].row_bytes
    cache = PageCache(5 * row, policy())  # five experts of sixteen
    store = expert_store(pack, device, cache)
    streamed = StreamedExperts(model, store).install()
    try:
        got = greedy_logits(model, ids, steps=8)
        for a, b in zip(got, expected, strict=True):
            assert torch.equal(a, b)
        stats = cache.stats
        assert stats.hits > 0 and stats.misses > 0 and cache.resident_bytes <= 5 * row
        report = store.weights.backend.report()
        served = report["materialization"]
        assert served["cache_hit_rows"] + served["fetched_rows"] == served["rows"]
    finally:
        streamed.remove()
        store.weights.backend.store.close()


def test_unrouted_slots_are_never_read_by_the_reference_kernels(tmp_path):
    """Sabotage check of the poison mode: poisoning a routed slot instead must change the output."""
    device = DEVICES[-1]
    model = tiny_model("olmoe", device, seed=2)
    ids = torch.randint(0, 128, (1, 5), generator=torch.Generator().manual_seed(2)).to(device)
    expected = greedy_logits(model, ids, steps=1)
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    store = expert_store(pack, device)
    original_fill = store.fill

    def fill_then_poison_one_routed(key, experts, buffers):
        original_fill(key, experts, buffers)
        for buffer in buffers.values():
            buffer[int(experts[0])].fill_(float("nan"))

    store.fill = fill_then_poison_one_routed
    streamed = StreamedExperts(model, store).install()
    try:
        got = greedy_logits(model, ids, steps=1)
        assert not all(torch.equal(a, b) for a, b in zip(got, expected))
    finally:
        streamed.remove()
        store.weights.backend.store.close()


def test_pack_refers_to_checkpoint_tensors_that_hold_the_experts_bytes(tmp_path):
    """A loader that only renames stacked expert tensors (GraniteMoE): the pack copies nothing."""
    model = tiny_model("granitemoe", "cpu", seed=3)
    model.save_pretrained(tmp_path / "checkpoint")
    checkpoint = tmp_path / "checkpoint" / "model.safetensors"
    source = SourceFile("example/tiny-moe", "0" * 40, "model.safetensors")
    pack = write_expert_pack(model, tmp_path / "pack", sources={"checkpoint": (source, checkpoint)})
    segments = [name for m in find_expert_modules(model) for name in (m.segment(p) for p in m.parameters)]
    assert all(pack.segments[name].file == "checkpoint" for name in segments)
    assert set(pack.metadata["referenced"]) == set(segments)
    assert "pack" not in pack.files
    reopened = open_pack(tmp_path / "pack", resolve=lambda requested: checkpoint)
    store = expert_store(reopened, "cpu")
    try:
        module = find_expert_modules(model)[1]
        loaded = store.load(module.name, torch.tensor([0, 5]))
        for parameter in module.parameters:
            assert torch.equal(loaded[parameter], getattr(module.module, parameter)[[0, 5]])
    finally:
        store.weights.backend.store.close()


def test_routers_and_dense_mlps_are_not_expert_modules():
    model = tiny_model("qwen2_moe", "cpu")
    names = {m.name for m in find_expert_modules(model)}
    assert names == {f"model.layers.{i}.mlp.experts" for i in range(2)}
    for name, module in model.named_modules():
        if name.endswith("gate") or "shared_expert" in name:
            assert name not in names
