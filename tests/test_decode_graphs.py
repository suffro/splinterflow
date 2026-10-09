"""Phase 6B4: a DeepSeek-V3 model's decode steps through CUDA Graphs equal eager decoding bit for bit (decision 0013)."""

from __future__ import annotations

import pytest
import torch
import transformers

from awpmi.models.decode_graphs import DecodeGraphs
from tests.test_moe import ARCHITECTURES, COMMON

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graphs need CUDA")

VARIANTS = {
    "moe": {},
    "dense-first": {"first_k_dense_replace": 1, "num_hidden_layers": 3},
    "q-lora": {"q_lora_rank": 24},
    "rope-halves": {"rope_interleave": False},
}


def tiny_deepseek(variant: str, seed: int = 0) -> transformers.PreTrainedModel:
    _, model_name, extra = ARCHITECTURES["deepseek_v3"]
    config = transformers.DeepseekV3Config(**{**COMMON, **extra, **VARIANTS[variant]})
    torch.manual_seed(seed)
    return getattr(transformers, model_name)(config).to(torch.bfloat16).to("cuda").eval()


@torch.inference_mode()
def decode(model, prompt: torch.Tensor, steps: int) -> tuple[list[torch.Tensor], object]:
    output = model(input_ids=prompt, use_cache=True)
    logits, cache = [output.logits[:, -1].clone()], output.past_key_values
    for _ in range(steps):
        output = model(input_ids=logits[-1].argmax(dim=-1, keepdim=True), past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        logits.append(output.logits[:, -1].clone())
    return logits, cache


def same_cache(a, b) -> bool:
    return all(torch.equal(x.keys, y.keys) and torch.equal(x.values, y.values) for x, y in zip(a.layers, b.layers, strict=True))


@needs_gpu
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_graphed_decode_steps_equal_eager_ones_bit_for_bit(variant):
    model = tiny_deepseek(variant)
    prompts = [torch.randint(0, 128, (1, length), generator=torch.Generator().manual_seed(length)).cuda() for length in (1, 7, 19)]
    eager = [decode(model, prompt, steps=5) for prompt in prompts]
    graphs = DecodeGraphs(model).install()
    try:
        graphed = [decode(model, prompt, steps=5) for prompt in prompts]
    finally:
        graphs.remove()
    for (logits, cache), (expected, expected_cache) in zip(graphed, eager):
        assert all(torch.equal(a, b) for a, b in zip(logits, expected, strict=True))
        assert same_cache(cache, expected_cache)
    layers = model.config.num_hidden_layers
    # The first decode step runs the pieces eagerly; then they are captured once and replayed for every later decode step.
    assert graphs.recorded_steps == 1 and graphs.captures == 2 * layers
    assert graphs.replays == 2 * layers * (5 * len(prompts) - 1)
    assert graphs.eager_steps == len(prompts)  # the prefills (a one-token prompt's too)
    assert graphs.memory_bytes > 0
    # Removed: the layers are eager again.
    assert all("forward" not in layer.__dict__ for layer in graphs.layers)
    again = decode(model, prompts[1], steps=5)[0]
    assert all(torch.equal(a, b) for a, b in zip(again, eager[1][0], strict=True))


@needs_gpu
def test_observers_see_the_replayed_outputs_as_eager_calls_would_show_them():
    model = tiny_deepseek("dense-first")
    prompt = torch.randint(0, 128, (1, 9), generator=torch.Generator().manual_seed(3)).cuda()

    def observe(model):
        seen, handles = [], []
        for name, module in model.named_modules():
            if name.endswith(("self_attn", "mlp.gate", "mlp.shared_experts")) or name.endswith(".mlp"):
                def hook(module, args, output, name=name):
                    output = output if isinstance(output, tuple) else (output,)
                    seen.append((name, [o.clone() for o in output if isinstance(o, torch.Tensor)]))

                handles.append(module.register_forward_hook(hook))
        return seen, handles

    seen_eager, handles = observe(model)
    decode(model, prompt, steps=3)
    for handle in handles:
        handle.remove()
    graphs = DecodeGraphs(model).install()
    seen_graphed, handles = observe(model)
    try:
        decode(model, prompt, steps=3)
    finally:
        for handle in handles:
            handle.remove()
        graphs.remove()
    assert [name for name, _ in seen_graphed] == [name for name, _ in seen_eager]
    for (_, a), (_, b) in zip(seen_graphed, seen_eager):
        assert len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


@needs_gpu
def test_streamed_experts_run_eagerly_between_the_graphs(tmp_path):
    from awpmi.materialization.weights import ExpertStore, WeightStore
    from awpmi.models.moe import StreamedExperts, groups_from_pack, write_expert_pack
    from awpmi.storage.pack import open_pack
    from tests.conftest import materialization_backend

    model = tiny_deepseek("dense-first")
    prompt = torch.randint(0, 128, (1, 11), generator=torch.Generator().manual_seed(5)).cuda()
    expected, expected_cache = decode(model, prompt, steps=4)
    pack = open_pack(write_expert_pack(model, tmp_path / "pack").directory)
    backend = materialization_backend(pack, "python", "cuda", None, direct=True)
    streamed = StreamedExperts(model, ExpertStore(WeightStore(backend), groups_from_pack(pack)), compact=True).install()
    graphs = DecodeGraphs(model).install()
    try:
        logits, cache = decode(model, prompt, steps=4)
    finally:
        graphs.remove()
        streamed.remove()
    assert all(torch.equal(a, b) for a, b in zip(logits, expected, strict=True)) and same_cache(cache, expected_cache)
    assert streamed.calls == 2 * 5  # every step's experts calls ran (eagerly), graphed or not


@needs_gpu
def test_hooks_a_replay_would_skip_are_refused():
    model = tiny_deepseek("moe")
    prompt = torch.randint(0, 128, (1, 5), generator=torch.Generator().manual_seed(1)).cuda()
    layer = model.model.layers[0]
    handle = layer.self_attn.q_proj.register_forward_hook(lambda *args: None)  # inside a graphed piece
    with pytest.raises(RuntimeError):
        DecodeGraphs(model).install()
    handle.remove()
    handle = layer.mlp.gate.register_forward_pre_hook(lambda *args: None)  # a pre-hook on an observed module
    with pytest.raises(RuntimeError):
        DecodeGraphs(model).install()
    handle.remove()
    graphs = DecodeGraphs(model).install()
    try:
        decode(model, prompt, steps=1)
        handle = layer.post_attention_layernorm.register_forward_hook(lambda *args: None)  # appears after install
        with pytest.raises(RuntimeError):
            decode(model, prompt, steps=1)
        handle.remove()
        handle = layer.mlp.gate.register_forward_hook(lambda module, args, output: output)  # changes the output
        with pytest.raises(RuntimeError):
            decode(model, prompt, steps=1)
        handle.remove()
    finally:
        graphs.remove()
