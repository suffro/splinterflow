"""An independent reference for a MoE model that fits in neither device nor host memory (Phase 4B, decision 0008).

The reference is transformers' own model, loaded by transformers' own loader, with every expert
weight materialized when its layer runs. Only *when* an experts layer exists differs from
`from_pretrained`:

  * The skeleton is built on `meta` from the configuration, with the declared kernels.
  * Every weight but the experts is loaded once by `convert_and_load_state_dict_in_model`, the
    function `from_pretrained` calls, from safetensors slices opened as `from_pretrained` opens
    them, with the model's dtype plan and conversion mapping; the model's own finalization step
    then initializes what checkpoints never hold (non-persistent buffers).
  * Each experts module gets a pre-hook that loads that layer's checkpoint tensors through the
    same function (the mapping's per-expert stacking and gate/up concatenation, on the device),
    and a hook that releases them once the module has run: one experts layer exists at a time.
    Between calls the experts' parameters are None, so any use outside a call raises.
  * Nothing of Weightsift's own path is used: no awpmi storage, index, cache, compact or chunked
    call, routed-only materialization, or expert layout derived by Weightsift
    (`tests/test_layering.py` checks the imports). Which checkpoint tensor belongs to which
    experts module is transformers' own renaming of its key.

Host memory holds at most one layer's checkpoint slices while they are converted. The files are
read through the operating system's file cache, which the OS accounts for, not the process.
`resident` keeps some experts layers on the device after their first load: results must not
depend on it (the residency check).
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


@dataclass
class ReferenceCall:
    """One experts call of the reference, seen while its whole layer is materialized."""

    module: str
    experts_module: nn.Module
    hidden_states: torch.Tensor
    top_k_index: torch.Tensor
    top_k_weights: torch.Tensor
    output: torch.Tensor

    def per_assignment_outputs(self) -> torch.Tensor:
        """Every (token, k) assignment's expert output before routing weights, [tokens·k, hidden].

        The module runs again on its full layer (no hooks), with each assignment as a token routed
        to its expert alone with weight 1: multiplying by 1 and summing one term are exact.
        """
        top_k = self.top_k_index.shape[-1]
        hidden = self.hidden_states.repeat_interleave(top_k, dim=0)
        index = self.top_k_index.reshape(-1, 1)
        ones = torch.ones(index.shape[0], 1, dtype=self.top_k_weights.dtype, device=index.device)
        return self.experts_module.forward(hidden, index, ones)


def checkpoint_files(repository: str, revision: str | None) -> list[Path]:
    """The local safetensors files of a checkpoint (a directory, or a revision in the Hugging Face cache)."""
    local = Path(repository)
    if local.is_dir():
        found = sorted(local.glob("*.safetensors"))
        if not found:
            raise FileNotFoundError(f"no safetensors files in {local}")
        return found
    from huggingface_hub import hf_hub_download

    try:
        index = Path(hf_hub_download(repository, "model.safetensors.index.json", revision=revision, local_files_only=True))
        names = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
    except Exception:  # no index: a single file
        names = ["model.safetensors"]
    return [Path(hf_hub_download(repository, name, revision=revision, local_files_only=True)) for name in names]


def experts_modules(model: nn.Module) -> dict[str, nn.Module]:
    """The modules transformers' experts convention marks (`use_experts_implementation`), by name."""
    return {
        name: module
        for name, module in model.named_modules()
        if hasattr(module, "_is_expert_parallel") and hasattr(module, "is_concatenated")
    }


def _open(path: Path):
    """A safetensors file opened as `from_pretrained` opens it (positioned reads on Windows, a mapping elsewhere)."""
    from safetensors import safe_open

    return safe_open(str(path), framework="pt", device="cpu", backend="pread" if sys.platform == "win32" else "mmap")


def _close(handles) -> None:
    for handle in handles:
        handle.__exit__(None, None, None)


def _slice_bytes(slice_) -> int:
    return math.prod(slice_.get_shape()) * _DTYPE_BYTES[slice_.get_dtype()]


@dataclass
class StreamingReference:
    """transformers' model with its experts materialized one layer at a time; see the module docstring."""

    repository: str
    revision: str | None
    dtype: torch.dtype
    device: torch.device | str
    experts_implementation: str | None = None
    attn_implementation: str | None = None
    config_overrides: dict = field(default_factory=dict)
    resident: tuple[str, ...] = ()
    on_call: Callable[[ReferenceCall], None] | None = None
    on_load: Callable[[str, nn.Module], None] | None = None  # after a layer's experts are loaded, before it runs
    model: nn.Module | None = field(default=None, init=False)
    report: dict = field(default_factory=dict, init=False)
    loads: int = field(default=0, init=False)
    loaded_bytes: int = field(default=0, init=False)  # checkpoint bytes converted for experts layers
    load_seconds: float = field(default=0.0, init=False)
    expert_device_bytes: int = field(default=0, init=False)  # expert parameters on the device now
    peak_expert_device_bytes: int = field(default=0, init=False)
    _keys: dict = field(default_factory=dict, init=False)  # experts module → [(file, checkpoint key)]
    _shapes: dict = field(default_factory=dict, init=False)
    _hooks: list = field(default_factory=list, init=False)
    _resident_loaded: set = field(default_factory=set, init=False)

    def load(self) -> StreamingReference:
        """Build the model, load every weight but the experts, and install the per-layer hooks."""
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        started = time.perf_counter()
        config = AutoConfig.from_pretrained(self.repository, revision=self.revision)
        for key, value in self.config_overrides.items():
            setattr(config, key, value)
        kernels = {}
        if self.experts_implementation is not None:
            kernels["experts_implementation"] = self.experts_implementation
        if self.attn_implementation is not None:
            kernels["attn_implementation"] = self.attn_implementation
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, dtype=self.dtype, **kernels)
        model.eval()
        experts = experts_modules(model)
        if not experts:
            raise ValueError("the model has no experts module (transformers' experts convention)")
        unknown = set(self.resident) - set(experts)
        if unknown:
            raise KeyError(f"no experts modules {sorted(unknown)}")
        self._shapes = {
            name: {p: (tuple(t.shape), t.dtype) for p, t in module.named_parameters(recurse=False)} for name, module in experts.items()
        }
        files = checkpoint_files(self.repository, self.revision)
        handles = [_open(path) for path in files]
        try:
            rest, self._keys = self._assign_keys(model, files, handles, experts)
            info = self._convert(model, rest)
        finally:
            _close(handles)
        expert_parameters = {f"{name}.{p}" for name, shapes in self._shapes.items() for p in shapes}
        problems = {
            "missing": sorted(set(info.missing_keys) - expert_parameters),
            "mismatched": sorted(map(str, info.mismatched_keys)),
            "conversion_errors": sorted(map(str, info.conversion_errors)),
        }
        if any(problems.values()):
            raise RuntimeError(f"loading the non-expert weights failed: {problems}")
        unexpected = sorted(info.unexpected_keys)
        info.missing_keys = set(info.missing_keys) - expert_parameters
        type(model)._finalize_model_loading(model, self._load_config(model), info)
        model.eval()
        for name, module in experts.items():
            for parameter in self._shapes[name]:
                module._parameters[parameter] = None
        leftovers = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
        if leftovers:
            raise RuntimeError(f"weights left on meta: {leftovers[:5]}")
        self.model = model
        for name, module in experts.items():
            self._hooks.append(module.register_forward_pre_hook(self._materialize(name), with_kwargs=True))
            self._hooks.append(module.register_forward_hook(self._release(name), with_kwargs=True, always_call=True))
        self.report = {
            "load_seconds": time.perf_counter() - started,
            "files": [str(path) for path in files],
            "non_expert_tensors": len(rest),
            "non_expert_checkpoint_bytes": sum(_slice_bytes(s) for s in rest.values()),
            "unexpected_keys": unexpected,
            "experts_modules": len(experts),
            "expert_checkpoint_tensors": sum(len(keys) for keys in self._keys.values()),
            "experts_implementation": model.config._experts_implementation,
            "attention_implementation": model.config._attn_implementation,
        }
        return self

    @staticmethod
    def _assign_keys(model, files: Sequence[Path], handles: Sequence, experts: dict[str, nn.Module]):
        """Checkpoint keys of every experts module (transformers' renaming of each key), and the slices of all the others."""
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

        mapping = get_model_conversion_mapping(model, None, None)
        renamings = [m for m in mapping if isinstance(m, WeightRenaming)]
        converters = [m for m in mapping if isinstance(m, WeightConverter)]
        meta_state = model.state_dict()
        keys: dict[str, list[tuple[Path, str]]] = {name: [] for name in experts}
        rest = {}
        for path, handle in zip(files, handles, strict=True):
            for key in handle.keys():
                renamed, _ = rename_source_key(key, renamings, converters, model.base_model_prefix, meta_state)
                owner = renamed.rpartition(".")[0]
                if owner in keys:
                    keys[owner].append((path, key))
                else:
                    rest[key] = handle.get_slice(key)
        empty = [name for name, found in keys.items() if not found]
        if empty:
            raise KeyError(f"the checkpoint holds no tensors for {empty[:3]}")
        return rest, keys

    def _load_config(self, model):
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.modeling_utils import LoadStateDictConfig

        return LoadStateDictConfig(
            pretrained_model_name_or_path=self.repository,
            device_map={"": torch.device(self.device)},
            dtype=self.dtype,
            dtype_plan=model._get_dtype_plan(self.dtype),
            weight_mapping=get_model_conversion_mapping(model, None, None),
        )

    def _convert(self, model, slices: dict):
        """transformers' loading function on these checkpoint slices (with fresh mapping objects)."""
        from transformers.core_model_loading import convert_and_load_state_dict_in_model

        info, _ = convert_and_load_state_dict_in_model(model, slices, self._load_config(model))
        return info

    def _materialize(self, name: str):
        def hook(module, args, kwargs):
            if name in self._resident_loaded:
                return None
            started = time.perf_counter()
            for parameter, (shape, dtype) in self._shapes[name].items():
                module._parameters[parameter] = nn.Parameter(torch.empty(shape, dtype=dtype, device="meta"), requires_grad=False)
            paths = sorted({path for path, _ in self._keys[name]})
            handles = dict(zip(paths, [_open(path) for path in paths], strict=True))
            try:
                slices = {key: handles[path].get_slice(key) for path, key in self._keys[name]}
                info = self._convert(self.model, slices)
            finally:
                _close(handles.values())
            for parameter, (shape, dtype) in self._shapes[name].items():
                tensor = module._parameters[parameter]
                if tensor is None or tensor.is_meta or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                    raise RuntimeError(f"{name}.{parameter} was not loaded ({info.unexpected_keys or info.conversion_errors})")
            nbytes = sum(t.numel() * t.element_size() for t in module.parameters(recurse=False))
            self.expert_device_bytes += nbytes
            self.peak_expert_device_bytes = max(self.peak_expert_device_bytes, self.expert_device_bytes)
            self.loads += 1
            self.loaded_bytes += sum(_slice_bytes(s) for s in slices.values())
            self.load_seconds += time.perf_counter() - started
            if name in self.resident:
                self._resident_loaded.add(name)
            if self.on_load is not None:
                self.on_load(name, module)
            return None

        return hook

    def _release(self, name: str):
        def hook(module, args, kwargs, output):
            try:
                if self.on_call is not None:
                    hidden = args[0] if args else kwargs["hidden_states"]
                    index = args[1] if len(args) > 1 else kwargs["top_k_index"]
                    weights = args[2] if len(args) > 2 else kwargs["top_k_weights"]
                    self.on_call(ReferenceCall(name, module, hidden, index, weights, output))
            finally:
                if name not in self._resident_loaded:
                    self.expert_device_bytes -= sum(
                        t.numel() * t.element_size() for t in module.parameters(recurse=False) if t is not None
                    )
                    for parameter in self._shapes[name]:
                        module._parameters[parameter] = None

        return hook

    def set_resident(self, names: Sequence[str]) -> None:
        """Keep these experts layers on the device after their next load; any other resident layer is released now."""
        unknown = set(names) - set(self._shapes)
        if unknown:
            raise KeyError(f"no experts modules {sorted(unknown)}")
        for name in sorted(self._resident_loaded - set(names)):
            module = self.model.get_submodule(name)
            self.expert_device_bytes -= sum(t.numel() * t.element_size() for t in module.parameters(recurse=False) if t is not None)
            for parameter in self._shapes[name]:
                module._parameters[parameter] = None
            self._resident_loaded.discard(name)
        self.resident = tuple(names)

    def close(self) -> None:
        """Remove the hooks (the experts modules keep None parameters; resident layers stay on the device)."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
