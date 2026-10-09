"""Decode steps of DeepSeek-V3 decoder layers through CUDA Graphs (Phase 6B4, decision 0013).

transformers' DeepSeek-V3 decoder layer (Moonlight's architecture) does three kinds of work for one token of one
sequence: pieces whose shapes never change and that touch neither the KV cache nor the routed experts; the KV cache's
update and the attention over the cache (their shapes grow with every token); and the routed experts (streamed: the
router's choice synchronized to the host, a transfer, a compact call). `DecodeGraphs` runs the first kind as two CUDA
Graphs per layer, replayed on every decode step but the first; the rest runs eagerly between them, as transformers runs
it:

  graph "pre"   the input RMSNorm, q_proj (or q_a_proj, q_a_layernorm, q_b_proj), kv_a_proj_with_mqa, kv_a_layernorm, the
                RoPE of this position and the query's concatenation: the layer's input and the position's cos and sin in,
                the query and the new latents out
  eager         the latents appended to the KV cache, the keys and values expanded from it, the attention
  graph "post"  o_proj, the residual add and the post-attention RMSNorm, then the router and the shared experts (a MoE
                layer) or the dense MLP and its residual add (a dense layer)
  eager         the routed experts' call, the MoE block's sum (experts, then shared experts) and the residual add

The first decode step runs the same pieces eagerly, as functions, keeping their inputs in static buffers; after its last
layer every piece is captured, back to back on one side stream (a warm-up first), into one graph pool. (Captured one at a
time between the step's eager work, each graph would keep a cuBLAS workspace of its own: 32 MiB each, 1 GiB for
Moonlight's 54, against about 70 MB.) A replay launches the kernels the eager modules launch, with the same shapes, on the
same values (copied into the static buffers first): bit for bit the same outputs, which the tests check against eager
decoding and Moonlight's runs against the reference in every digest. A step runs eagerly unless it decodes one token of one sequence on CUDA,
without autograd, after the layer's KV cache holds a position (a prefill, even of one token, is eager: the cache may keep
its first latents as they are, and the graphs' outputs are overwritten by the next replay).

The code between the graphs repeats transformers' DeepseekV3Attention.forward, DeepseekV3MoE.forward and
DeepseekV3DecoderLayer.forward, so it is tied to their source: `install` refuses any other (`SOURCE_SHA256`).

Observers: the forward hooks of the modules the graphs stand in for (the attention, the router, the shared experts, the
MoE block, the dense MLP) are called with the replayed outputs, in eager order, with the arguments eager calls pass
(the attention's are keywords). What a replay would skip is refused: a forward pre-hook on those modules, any hook on a
module inside a graphed piece (projections, norms, activations), a hook that returns a value, or a global module hook
make `install` raise, and are checked again before every graphed step.

Memory: one graph pool for every layer's graphs (`memory_bytes`; Moonlight: about 70 MB).
"""

from __future__ import annotations

import hashlib
import inspect
import time

import torch
from torch import nn

LAYER_CLASS = "DeepseekV3DecoderLayer"
# The transformers code `_forward` repeats, as validated (transformers 5.18.0): sha256 of each method's source.
SOURCE_SHA256 = {
    "DeepseekV3Attention.forward": "d51be5973c10f50356175dd8822ef97d1c09d17a5da620923d8761098287c66b",
    "DeepseekV3MoE.forward": "626736877659398958dfb2f78f67394cac2b01546a0d9bf17ff0cab741ad49ca",
    "DeepseekV3DecoderLayer.forward": "75d1d62e173e8c574b8a61338d1fa5052774d13b848eecd282cde88cd6690b8e",
}


def _sources(module) -> dict[str, str]:
    out = {}
    for name in SOURCE_SHA256:
        owner, method = name.split(".")
        source = inspect.getsource(getattr(getattr(module, owner), method)).replace(chr(13), "")  # line endings as LF
        out[name] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return out


class _Graph:
    """One captured piece: static inputs, the graph, static outputs."""

    def __init__(self, function, inputs: tuple[torch.Tensor, ...], pool, device: torch.device, side: torch.cuda.Stream) -> None:
        self.inputs = inputs
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):  # lazy initializations (cuBLAS handles and workspaces) before the capture
            for _ in range(2):
                function(*self.inputs)
        torch.cuda.current_stream(device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            outputs = function(*self.inputs)
        self.outputs = outputs if isinstance(outputs, tuple) else (outputs,)


class DecodeGraphs:
    """CUDA Graphs for the static pieces of a DeepSeek-V3 model's decode steps; see the module docstring."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.layers = [module for module in model.modules() if type(module).__name__ == LAYER_CLASS]
        if not self.layers:
            raise ValueError("no DeepSeek-V3 decoder layer in the model")
        self._graphs: dict[tuple[int, str], _Graph] = {}
        self._pieces: dict[tuple[int, str], tuple] = {}  # (function, static inputs) of each piece, kept by the first step
        # Per layer, computed once: the modules whose forward hooks the graphs call (observed), and those they hide.
        self._observed: list[list[nn.Module]] = []
        self._hidden_modules: list[list[nn.Module]] = []
        # Static inputs: each layer's input (also its "post" graph's residual) and attention output, and the position's
        # cos and sin (every "pre" graph reads the same two, copied once per step).
        self._hidden: dict[int, torch.Tensor] = {}
        self._attended: dict[int, torch.Tensor] = {}
        self._position: tuple[torch.Tensor, torch.Tensor] | None = None
        self._position_of = None  # the step's position embeddings the static ones hold
        self._pool = None
        self._ds = None
        self._installed = False
        self.captures = 0
        self.replays = 0
        self.eager_steps = 0  # steps (counted at the first layer) that ran transformers' forward: prefills
        self.recorded_steps = 0  # decode steps that ran the pieces eagerly, before the graphs were captured
        self.memory_bytes = 0
        self.capture_ms = 0.0  # host time of the captures (warm-ups included), once per installation

    # Installation

    def install(self) -> DecodeGraphs:
        from transformers.models.deepseek_v3 import modeling_deepseek_v3 as ds

        mismatched = {name for name, digest in _sources(ds).items() if digest != SOURCE_SHA256[name]}
        if mismatched:
            raise ValueError(f"transformers' DeepSeek-V3 code differs from the validated one ({sorted(mismatched)}): no decode graphs")
        self._ds = ds
        for index, layer in enumerate(self.layers):
            moe = hasattr(layer.mlp, "experts")
            observed = [layer.self_attn, layer.mlp] + ([layer.mlp.gate, layer.mlp.shared_experts] if moe else [])
            routed = set(layer.mlp.experts.modules()) if moe else set()
            self._observed.append(observed)
            self._hidden_modules.append([m for m in layer.modules() if m is not layer and m not in observed and m not in routed])
            self._check(index)

            def forward(*args, index=index, layer=layer, **kwargs):
                return self._forward(index, layer, *args, **kwargs)

            layer.forward = forward
        self._installed = True
        return self

    def remove(self) -> None:
        for layer in self.layers:
            layer.__dict__.pop("forward", None)
        self._graphs.clear()
        self._pieces.clear()
        self._observed.clear()
        self._hidden_modules.clear()
        self._hidden.clear()
        self._attended.clear()
        self._position = self._position_of = None
        self._pool = None
        self._installed = False

    # The pieces (transformers' operations, in its order)

    def _pre(self, layer, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        attn = layer.self_attn
        batch_size, seq_length = hidden.shape[:-1]
        query_shape = (batch_size, seq_length, -1, attn.qk_head_dim)
        h = layer.input_layernorm.forward(hidden)
        if attn.q_lora_rank is None:
            q_states = attn.q_proj(h)
        else:
            q_states = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(h)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)
        compressed_kv = attn.kv_a_proj_with_mqa(h)
        kv_nope, k_rot = torch.split(compressed_kv, [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
        kv_nope = attn.kv_a_layernorm(kv_nope)
        kv_nope = kv_nope.view(batch_size, 1, seq_length, attn.kv_lora_rank)
        k_rot = k_rot.view(batch_size, 1, seq_length, attn.qk_rope_head_dim)
        if attn.config.rope_interleave:
            q_rot, k_rot = self._ds.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = self._ds.apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        query_states = torch.cat((q_pass, q_rot), dim=-1)
        return query_states, kv_nope, k_rot

    def _post(self, layer, attended: torch.Tensor, residual: torch.Tensor):
        attn_output = layer.self_attn.o_proj(attended)
        hidden = residual + attn_output
        normed = layer.post_attention_layernorm.forward(hidden)
        if hasattr(layer.mlp, "experts"):
            logits, weights, indices = layer.mlp.gate.forward(normed)
            return attn_output, hidden, normed, logits, weights, indices, layer.mlp.shared_experts.forward(normed)
        mlp_output = layer.mlp.forward(normed)
        return attn_output, hidden, normed, mlp_output, hidden + mlp_output

    # A layer's forward

    def _forward(self, index: int, layer, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False,
                 position_embeddings=None, **kwargs):
        attn = layer.self_attn
        graphed = (
            self._installed and hidden_states.is_cuda and tuple(hidden_states.shape[:2]) == (1, 1) and not torch.is_grad_enabled()
            and not layer.training and past_key_values is not None and position_embeddings is not None
            and past_key_values.get_seq_length(attn.layer_idx) > 0
        )
        if not graphed:
            self.eager_steps += index == 0
            return type(layer).forward(layer, hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                                       past_key_values=past_key_values, use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)
        self._check(index)
        cos, sin = position_embeddings
        if self._position is None:
            self._position = (cos.clone(), sin.clone())
        elif self._position_of is not cos:
            self._position[0].copy_(cos)
            self._position[1].copy_(sin)
        self._position_of = cos
        static = _static(self._hidden, index, hidden_states)
        query_states, kv_nope, k_rot = self._run(index, "pre", lambda h, c, s: self._pre(layer, h, c, s), (static, *self._position))
        # Eager, as DeepseekV3Attention.forward: the cache, the keys and values, the attention.
        kv_nope, k_rot = past_key_values.update(kv_nope, k_rot, attn.layer_idx)
        key_states, value_states = attn.expand_kv(kv_nope, k_rot)
        interface = self._ds.ALL_ATTENTION_FUNCTIONS.get_interface(attn.config._attn_implementation, self._ds.eager_attention_forward)
        attended, _ = interface(
            attn, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout, scaling=attn.scaling, position_ids=position_ids, use_cache=use_cache, **kwargs,
        )
        attended = _static(self._attended, index, attended.reshape(1, 1, -1).contiguous())
        outputs = self._run(index, "post", lambda a, r: self._post(layer, a, r), (attended, static))
        if index == len(self.layers) - 1 and not self._graphs:
            self._capture()  # after the first decode step's last layer: every piece, back to back
        _observe(attn, (), (outputs[0], None))
        if hasattr(layer.mlp, "experts"):
            attn_output, hidden, normed, logits, weights, indices, shared = outputs
            _observe(layer.mlp.gate, (normed,), (logits, weights, indices))
            # Eager, as DeepseekV3MoE.forward and the layer: the routed experts, the block's sum, the residual add.
            routed = layer.mlp.experts(normed.view(-1, normed.shape[-1]), indices, weights).view(*normed.shape)
            _observe(layer.mlp.shared_experts, (normed,), shared)
            block = routed + shared
            _observe(layer.mlp, (normed,), block)
            return hidden + block
        attn_output, hidden, normed, mlp_output, layer_output = outputs
        _observe(layer.mlp, (normed,), mlp_output)
        return layer_output

    def _run(self, index: int, piece: str, function, inputs: tuple[torch.Tensor, ...]) -> tuple:
        """The piece's outputs for `inputs` (static buffers already holding this step's values): its graph replayed, or,
        before the graphs exist, the piece itself (kept for the capture)."""
        graph = self._graphs.get((index, piece))
        if graph is None:
            self._pieces[(index, piece)] = (function, inputs)
            self.recorded_steps += index == 0 and piece == "pre"
            outputs = function(*inputs)
            return outputs if isinstance(outputs, tuple) else (outputs,)
        graph.graph.replay()
        self.replays += 1
        return graph.outputs

    def _capture(self) -> None:
        """Every piece the step ran, captured back to back into one pool (module docstring)."""
        expected = {(index, piece) for index in range(len(self.layers)) for piece in ("pre", "post")}
        if set(self._pieces) != expected:
            return  # an incomplete step: the next decode step runs the pieces again
        device = next(iter(self._pieces.values()))[1][0].device
        self._pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream(device)
        before, started = torch.cuda.memory_allocated(device), time.perf_counter()
        for key, (function, inputs) in self._pieces.items():
            self._graphs[key] = _Graph(function, inputs, self._pool, device, side)
            self.captures += 1
        torch.cuda.synchronize(device)
        self.capture_ms += (time.perf_counter() - started) * 1e3
        self.memory_bytes += torch.cuda.memory_allocated(device) - before
        self._pieces.clear()

    # Observers

    def _check(self, index: int) -> None:
        """Refuse what a replay would skip (module docstring)."""
        if nn.modules.module._global_forward_hooks or nn.modules.module._global_forward_pre_hooks:
            raise RuntimeError("decode graphs: a global module hook would not see the graphed modules")
        for module in self._observed[index]:
            if module._forward_pre_hooks or module._forward_hooks_with_kwargs:
                raise RuntimeError(f"decode graphs: a forward pre-hook, or a hook taking keywords, on {type(module).__name__} would not run as eagerly")
        for module in self._hidden_modules[index]:
            if module._forward_hooks or module._forward_pre_hooks:
                raise RuntimeError(f"decode graphs: a hook on {type(module).__name__} inside a graphed piece would not run")


def _static(buffers: dict[int, torch.Tensor], index: int, value: torch.Tensor) -> torch.Tensor:
    """`buffers[index]` holding `value` (made on first use: a graph reads its inputs from where it was captured)."""
    buffer = buffers.get(index)
    if buffer is None:
        buffer = buffers[index] = value.clone()
    else:
        buffer.copy_(value)
    return buffer


def _observe(module: nn.Module, args: tuple, output) -> None:
    """What calling `module` would show its forward hooks, with `output` as its result."""
    for hook in tuple(module._forward_hooks.values()):
        if hook(module, args, output) is not None:
            raise RuntimeError(f"decode graphs: a forward hook of {type(module).__name__} changes its output")
