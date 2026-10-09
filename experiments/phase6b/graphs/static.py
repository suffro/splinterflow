"""Phase 6B4: CUDA Graphs on Moonlight's static decode pieces, in isolation, before anything is built into the runtime.

A decode step (one token) has, per decoder layer, three pieces whose shapes never change and that touch neither the KV
cache, the routed experts nor the host:

  attn-pre   input RMSNorm, q_proj, kv_a_proj_with_mqa, kv_a_layernorm, the interleaved RoPE of this position, and the
             query's concatenation (everything of the attention before the cache update and the attention over the cache)
  attn-post  o_proj and the residual add
  moe-static the post-attention RMSNorm, the router (float32 GEMM, sigmoid, grouped top-k, weights) and the shared
             experts (layer 0: its dense MLP)
  final      the final RMSNorm and the LM head (once per token)

The script runs prompt 0's 16-token prefill and two decode steps with the experts streamed (native, no cache), records
every piece's inputs and outputs inside the second decode step (hooks on the model's own modules), then:

  1. recomputes each piece eagerly from the recorded inputs with the model's modules (the same operations, in the same
     order, as transformers' code) and checks it against the step's own outputs: bit for bit;
  2. captures each piece in a CUDA Graph (static input and output buffers, a warm-up on a side stream first) and
     replays it on the recorded inputs: bit for bit against the eager outputs;
  3. times, per token, issuing all pieces of all layers eagerly against copying their inputs into the graphs' buffers
     and replaying them (host time to issue, and wall time with the device's).

    .venv/Scripts/python.exe experiments/phase6b/graphs/static.py > experiments/phase6b/graphs/static.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from moonlight_runtime import DTYPES, NUMERICS_FLAGS  # noqa: E402  (configures the numerics first)

import torch  # noqa: E402

from awpmi.materialization.backend import MaterializationBackend  # noqa: E402
from awpmi.materialization.weights import ExpertStore, WeightStore  # noqa: E402
from awpmi.models.checkpoint import checkpoint_sources, load_model_without_experts  # noqa: E402
from awpmi.models.moe import StreamedExperts, groups_from_pack  # noqa: E402
from awpmi.storage.pack import open_pack  # noqa: E402
from awpmi.streaming.streamer import PageStreamer  # noqa: E402
from awpmi.tracing import read_jsonl  # noqa: E402

REPEATS = 20


def pieces(model):
    """The static pieces of a decode step, as functions of their inputs over the model's own modules."""
    from transformers.models.deepseek_v3 import modeling_deepseek_v3 as ds

    config = model.config
    out = []
    for index, layer in enumerate(model.model.layers):
        attn = layer.self_attn

        def attn_pre(hidden, cos, sin, layer=layer, attn=attn):
            batch, length = hidden.shape[:-1]
            h = layer.input_layernorm(hidden)
            if attn.q_lora_rank is None:
                q_states = attn.q_proj(h)
            else:
                q_states = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(h)))
            q_states = q_states.view(batch, length, -1, attn.qk_head_dim).transpose(1, 2)
            q_pass, q_rot = torch.split(q_states, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)
            compressed_kv = attn.kv_a_proj_with_mqa(h)
            kv_nope, k_rot = torch.split(compressed_kv, [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
            kv_nope = attn.kv_a_layernorm(kv_nope).view(batch, 1, length, attn.kv_lora_rank)
            k_rot = k_rot.view(batch, 1, length, attn.qk_rope_head_dim)
            if config.rope_interleave:
                q_rot, k_rot = ds.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
            else:
                q_rot, k_rot = ds.apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
            return torch.cat((q_pass, q_rot), dim=-1), kv_nope, k_rot

        def attn_post(attended, residual, attn=attn):
            return residual + attn.o_proj(attended)

        if hasattr(layer.mlp, "gate"):
            def moe_static(hidden, layer=layer):
                h = layer.post_attention_layernorm(hidden)
                logits, weights, indices = layer.mlp.gate(h)
                return h, logits, weights, indices, layer.mlp.shared_experts(h)
        else:
            def moe_static(hidden, layer=layer):  # the dense layer: its whole MLP and the residual
                return (hidden + layer.mlp(layer.post_attention_layernorm(hidden)),)

        out.append({"attn-pre": attn_pre, "attn-post": attn_post, "moe-static": moe_static})

    def final(hidden):
        return (model.lm_head(model.model.norm(hidden)),)

    return out, final


def record(model, step):
    """Hooks recording each piece's inputs and the step's own outputs for it, during one call of `step`."""
    seen: dict = {}
    handles = []
    for index, layer in enumerate(model.model.layers):
        attn = layer.self_attn

        def layer_in(module, args, kwargs, index=index):
            seen[(index, "hidden")] = kwargs.get("hidden_states", args[0] if args else None).clone()
            cos, sin = kwargs["position_embeddings"]
            seen[(index, "cos")], seen[(index, "sin")] = cos.clone(), sin.clone()

        def o_in(module, args, index=index):
            seen[(index, "attended")] = args[0].clone()

        def o_out(module, args, output, index=index):
            seen[(index, "o_proj")] = output.clone()

        def post_in(module, args, index=index):
            seen[(index, "post_hidden")] = args[0].clone()

        def post_out(module, args, output, index=index):
            seen[(index, "post_norm")] = output.clone()

        handles.append(layer.register_forward_pre_hook(layer_in, with_kwargs=True))
        handles.append(attn.o_proj.register_forward_pre_hook(o_in))
        handles.append(attn.o_proj.register_forward_hook(o_out))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(post_in))
        handles.append(layer.post_attention_layernorm.register_forward_hook(post_out))
        if hasattr(layer.mlp, "gate"):
            def gate_out(module, args, output, index=index):
                seen[(index, "router")] = tuple(t.clone() for t in output)

            def shared_out(module, args, output, index=index):
                seen[(index, "shared")] = output.clone()

            handles.append(layer.mlp.gate.register_forward_hook(gate_out))
            handles.append(layer.mlp.shared_experts.register_forward_hook(shared_out))
        else:
            def layer_out(module, args, kwargs, output, index=index):
                seen[(index, "layer_out")] = output.clone() if isinstance(output, torch.Tensor) else output[0].clone()

            handles.append(layer.register_forward_hook(layer_out, with_kwargs=True))

    def norm_in(module, args):
        seen["final_hidden"] = args[0].clone()

    handles.append(model.model.norm.register_forward_pre_hook(norm_in))
    try:
        output = step()
    finally:
        for handle in handles:
            handle.remove()
    seen["logits"] = output.logits.clone()
    return seen, output


def same(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


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
    result = {"numerics": NUMERICS_FLAGS, "attention": model.config._attn_implementation, "experts": model.config._experts_implementation,
              "q_lora_rank": model.config.q_lora_rank, "rope_interleave": model.config.rope_interleave, "layers": len(model.model.layers)}
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([prompt], device=device), use_cache=True, logits_to_keep=1)
        token, cache_kv = output.logits[0, -1].argmax().view(1, 1), output.past_key_values
        output = model(input_ids=token, past_key_values=cache_kv, use_cache=True, logits_to_keep=1)
        token, cache_kv = output.logits[0, -1].argmax().view(1, 1), output.past_key_values
        seen, _ = record(model, lambda: model(input_ids=token, past_key_values=cache_kv, use_cache=True, logits_to_keep=1))
        torch.cuda.synchronize()
        streamed.remove()
        store.close()
        layers, final = pieces(model)
        # Each piece's inputs, from the recorded step.
        inputs = []
        for index in range(len(layers)):
            inputs.append({
                "attn-pre": (seen[(index, "hidden")], seen[(index, "cos")], seen[(index, "sin")]),
                "attn-post": (seen[(index, "attended")], seen[(index, "hidden")]),
                "moe-static": (seen[(index, "post_hidden")],),
            })
        final_inputs = (seen["final_hidden"],)

        # 1. Eager recomputation against the step's own values.
        mismatches = []
        eager = []
        for index, functions in enumerate(layers):
            outputs = {name: functions[name](*inputs[index][name]) for name in functions}
            eager.append(outputs)
            post = outputs["attn-post"]
            if not torch.equal(post, seen[(index, "post_hidden")]):
                mismatches.append([index, "attn-post: residual + o_proj != the post-attention input"])
            moe = outputs["moe-static"]
            if (index, "router") in seen:
                if not torch.equal(moe[0], seen[(index, "post_norm")]):
                    mismatches.append([index, "post-attention norm"])
                if not same(moe[1:4], seen[(index, "router")]):
                    mismatches.append([index, "router"])
                if not torch.equal(moe[4], seen[(index, "shared")]):
                    mismatches.append([index, "shared experts"])
            elif not torch.equal(moe[0], seen[(index, "layer_out")]):
                mismatches.append([index, "dense layer"])
        # attn-pre's latents against what the step appended to the KV cache (its last position).
        for index in range(len(layers)):
            _, kv_nope, k_rot = eager[index]["attn-pre"]
            cached = cache_kv.layers[index]
            if not (torch.equal(kv_nope, cached.keys[:, :, -1:, :]) and torch.equal(k_rot, cached.values[:, :, -1:, :])):
                mismatches.append([index, "attn-pre: latents != the KV cache's last position"])
        logits = final(*final_inputs)[0]
        if not torch.equal(logits[:, -1:], seen["logits"]):
            mismatches.append(["final", "logits"])
        result["eager_vs_step_mismatches"] = mismatches

        # 2. Graphs: static buffers, warm-up on a side stream, capture, replay on the recorded inputs.
        graphs, statics = [], []
        side = torch.cuda.Stream(device)
        memory_before = torch.cuda.memory_allocated(device)
        pool = torch.cuda.graph_pool_handle()

        def capture(function, args):
            static_in = tuple(a.clone() for a in args)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                for _ in range(2):
                    function(*static_in)
            torch.cuda.current_stream(device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                static_out = function(*static_in)
            return graph, static_in, static_out if isinstance(static_out, tuple) else (static_out,)

        for index, functions in enumerate(layers):
            entry = {}
            for name, function in functions.items():
                entry[name] = capture(function, inputs[index][name])
            graphs.append(entry)
        final_graph = capture(final, final_inputs)
        torch.cuda.synchronize()
        result["graph_memory_bytes"] = torch.cuda.memory_allocated(device) - memory_before

        def replay(entry, args):
            graph, static_in, static_out = entry
            for buffer, value in zip(static_in, args):
                buffer.copy_(value)
            graph.replay()
            return static_out

        replay_mismatches = []
        for index, entry in enumerate(graphs):
            for name in entry:
                expected = eager[index][name]
                expected = expected if isinstance(expected, tuple) else (expected,)
                if not same(replay(entry[name], inputs[index][name]), expected):
                    replay_mismatches.append([index, name])
        if not torch.equal(replay(final_graph, final_inputs)[0], logits):
            replay_mismatches.append(["final", "logits"])
        result["replay_vs_eager_mismatches"] = replay_mismatches
        result["kernels_per_token"] = {}

        # 3. Time per token: every piece of every layer, eager against replay.
        def eager_token():
            for index, functions in enumerate(layers):
                for name, function in functions.items():
                    function(*inputs[index][name])
            final(*final_inputs)

        def graph_token():
            for index, entry in enumerate(graphs):
                for name in entry:
                    replay(entry[name], inputs[index][name])
            replay(final_graph, final_inputs)

        timings = {}
        for name, token_function in (("eager", eager_token), ("graphs", graph_token), ("eager-again", eager_token), ("graphs-again", graph_token)):
            issue, wall = [], []
            for _ in range(REPEATS + 2):
                torch.cuda.synchronize()
                started = time.perf_counter()
                token_function()
                issued = time.perf_counter()
                torch.cuda.synchronize()
                done = time.perf_counter()
                issue.append((issued - started) * 1e3)
                wall.append((done - started) * 1e3)
            issue, wall = issue[2:], wall[2:]
            timings[name] = {"issue_ms": statistics.mean(issue), "issue_ms_median": statistics.median(issue),
                             "wall_ms": statistics.mean(wall), "wall_ms_median": statistics.median(wall)}
        result["per_token"] = timings
        # Kernel launches per token, eager and replayed (the profiler's count).
        from torch.profiler import ProfilerActivity, profile

        for name, token_function in (("eager", eager_token), ("graphs", graph_token)):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                token_function()
                torch.cuda.synchronize()
            launches = sum(1 for e in prof.events() if e.name in ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cudaGraphLaunch", "cudaMemcpyAsync"))
            result["kernels_per_token"][name] = launches
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
