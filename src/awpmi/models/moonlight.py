"""Moonlight adapter (Phase 4B, decision 0008): moonshotai/Moonlight-16B-A3B, DeepSeek-V3's architecture at 16 B parameters.

Only what is Moonlight-specific lives here; storage, caching, streaming, the compact and chunked
experts calls are the generic MoE path (`awpmi.models.moe`, `awpmi.models.checkpoint`).

  * Checkpoint layout. Each expert is three BF16 tensors, model.layers.L.mlp.experts.E.
    {gate,up,down}_proj.weight, adjacent in the layer's file (down, gate, up). transformers
    stacks them and fuses gate with up: gate_up_proj [64, 2·1408, 2048] (gate rows first) and
    down_proj [64, 2048, 1408]. `EXPERT_LAYOUT` states that layout; `expert_sources` requires the
    transformers mapping to agree.
  * Routers. model.layers.L.mlp.gate, DeepSeek-V3's: float32 logits of the float32-cast hidden
    state and weight, sigmoid scores; the experts are chosen on scores plus a correction bias
    (stored in BF16, kept in float32 by transformers' dtype plan) within the best groups (here
    one group, so all 64), top 6; their weights are the unbiased scores, renormalized (+1e-20)
    and scaled by 2.446. It returns (logits, weights, indices). `ROUTING` is the configuration
    this adapter was validated for, and `check_config` refuses any other: transformers'
    implementation hard-codes sigmoid scoring and noaux_tc selection.
  * Shared experts. model.layers.L.mlp.shared_experts, a dense MLP of 2 × 1408 neurons whose
    output is added to the routed experts' (in BF16) by the MoE block. It is not an experts
    module, so it stays with the non-expert weights: resident, since every token uses it.
  * Layer 0 is a dense MLP (first_k_dense_replace = 1); layers 1-26 are MoE layers.
  * Reference profile. BF16 (the checkpoint is stored in BF16, router bias included),
    grouped_mm experts and SDPA attention, transformers' defaults.
  * Phase 5A's oracle names the modules around the last MoE block: the post-attention norm
    (its input is the residual), the final norm and the LM head.
"""

from __future__ import annotations

from torch import nn

from awpmi.models import checkpoint
from awpmi.models.moe import ExpertModule, find_expert_modules
from awpmi.profiles import BF16_REFERENCE

EXPERT_LAYOUT = {
    "gate_up_proj": ("mlp.experts.{expert}.gate_proj.weight", "mlp.experts.{expert}.up_proj.weight"),
    "down_proj": ("mlp.experts.{expert}.down_proj.weight",),
}
EXPERTS_SUFFIX = "mlp.experts"
ROUTER_SUFFIX = "mlp.gate"
SHARED_SUFFIX = "mlp.shared_experts"
BLOCK_SUFFIX = "mlp"
# Phase 5A (the expert oracle): the norm whose input is the residual the MoE block's output is added to, the final
# norm and the LM head (DeepseekV3DecoderLayer, DeepseekV3Model, DeepseekV3ForCausalLM).
RESIDUAL_NORM_SUFFIX = "post_attention_layernorm"
FINAL_NORM = "model.norm"
LM_HEAD = "lm_head"
REFERENCE_PROFILE = BF16_REFERENCE.with_kernels(experts="grouped_mm", attention="sdpa")
ROUTING = {
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "n_routed_experts": 64,
    "num_experts_per_tok": 6,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.446,
    "n_shared_experts": 2,
    "first_k_dense_replace": 1,
}


def check_config(config) -> None:
    """Raise unless the configuration routes as `ROUTING` declares (what transformers' implementation computes)."""
    actual = {key: getattr(config, key, None) for key in ROUTING}
    if actual != ROUTING:
        raise ValueError(f"routing configuration {actual} differs from the validated {ROUTING}")


def expert_sources(model: nn.Module, modules: list[ExpertModule] | None = None) -> dict[str, dict[str, tuple[str, ...]]]:
    """The checkpoint tensors of every expert-sliced parameter, as transformers declares them, checked against Moonlight's layout."""
    return checkpoint.checked_expert_sources(model, EXPERT_LAYOUT, EXPERTS_SUFFIX, modules)


def routers(model: nn.Module) -> dict[str, nn.Module]:
    """Experts module name → its router (DeepSeek-V3's: a float32 correction bias next to the weight)."""
    modules = find_expert_modules(model)
    found = checkpoint.neighbours(model, EXPERTS_SUFFIX, ROUTER_SUFFIX, modules)
    for entry in modules:
        router = found[entry.name]
        bias = getattr(router, "e_score_correction_bias", None)
        if getattr(router, "num_experts", None) != entry.num_experts or getattr(router, "top_k", None) != ROUTING["num_experts_per_tok"]:
            raise ValueError(f"{entry.name}: no router next to it")
        if bias is None or bias.shape != (entry.num_experts,):
            raise ValueError(f"{entry.name}: the router has no correction bias")
    return found


def shared_experts(model: nn.Module) -> dict[str, nn.Module]:
    """Experts module name → the shared experts of its layer (a dense MLP, not an experts module)."""
    found = checkpoint.neighbours(model, EXPERTS_SUFFIX, SHARED_SUFFIX)
    for name, module in found.items():
        if hasattr(module, "num_experts"):
            raise ValueError(f"{name}: the shared experts look like routed experts")
    return found


def moe_blocks(model: nn.Module) -> dict[str, nn.Module]:
    """Experts module name → its MoE block (routed experts + shared experts)."""
    return checkpoint.neighbours(model, EXPERTS_SUFFIX, BLOCK_SUFFIX)
