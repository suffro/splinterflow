"""Reference profiles: what "exact" is exact relative to (decision 0007).

Weightsift reproduces the discrete decisions of a *declared, fully materialized reference
execution*. Until Phase 4A that declaration was implicit: the model in BF16, transformers'
default kernels, the declared numerical environment (decision 0001). A profile states it:

  kind           BF16_REFERENCE, FP16_REFERENCE or NATIVE_QUANTIZED_REFERENCE
  weight dtype   the dtype of the stored weight bytes the reference computes with. Weightsift
                 streams exactly those bytes, so a checkpoint stored in another dtype is
                 refused, not converted on the fly
  compute dtype  the dtype of the model's activations
  kernels        the experts and attention implementations (None: the model's default)
  numerics       the declared numerical environment (`awpmi.runtime`)
  quantization   NATIVE_QUANTIZED_REFERENCE only: the native format, e.g. FP8 e4m3 with block
                 scales

"Exact" always means: the same discrete outputs (tokens, routed experts) and bit for bit the
same tensors as that reference, executed with every weight materialized. Where the bytes live
(device, host memory, drive) is not part of a profile, and must not change any result.

NATIVE_QUANTIZED_REFERENCE is declared, not implemented. A model whose native execution is
quantized (DeepSeek-V3 ships FP8 experts) would be compared against that native execution,
not against a dequantized BF16 copy; `native_quantized_reference` builds such a profile so it
can be recorded, and `check_model` refuses it until the native kernels are part of the
reference. Decision 0001's certificate is unchanged: it bounds the BF16_REFERENCE logits.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import torch


class ReferenceKind(str, Enum):
    BF16 = "bf16_reference"
    FP16 = "fp16_reference"
    NATIVE_QUANTIZED = "native_quantized_reference"


_DTYPE_NAMES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}


@dataclass(frozen=True)
class ReferenceProfile:
    kind: ReferenceKind
    weight_dtype: str  # safetensors name of the stored weights' dtype (e.g. "BF16")
    compute_dtype: torch.dtype
    experts_implementation: str | None = None
    attention_implementation: str | None = None
    numerics: str = "awpmi.runtime.configure_reproducible_numerics"
    quantization: dict[str, Any] = field(default_factory=dict)

    def with_kernels(self, experts: str | None = None, attention: str | None = None) -> ReferenceProfile:
        """The same profile with its kernels declared."""
        return replace(self, experts_implementation=experts, attention_implementation=attention)

    def check_model(self, model: torch.nn.Module) -> None:
        """Raise unless `model` executes as this profile declares (dtype, kernels)."""
        if self.kind is ReferenceKind.NATIVE_QUANTIZED:
            raise NotImplementedError("a native quantized reference is declared, not implemented (decision 0007)")
        dtype = getattr(model, "dtype", None)
        if dtype != self.compute_dtype:
            raise ValueError(f"the model computes in {dtype}, the profile declares {self.compute_dtype}")
        config = getattr(model, "config", None)
        for attribute, declared in (
            ("_experts_implementation", self.experts_implementation),
            ("_attn_implementation", self.attention_implementation),
        ):
            actual = getattr(config, attribute, None)
            if declared is not None and actual != declared:
                raise ValueError(f"the model's {attribute} is {actual!r}, the profile declares {declared!r}")

    def check_weights(self, dtypes: dict[str, str]) -> None:
        """Raise unless every streamed weight (name → safetensors dtype name) is stored in the declared dtype."""
        wrong = {name: dtype for name, dtype in dtypes.items() if dtype != self.weight_dtype}
        if wrong:
            name, dtype = next(iter(sorted(wrong.items())))
            raise ValueError(f"{len(wrong)} weights are stored as {dtype} (e.g. {name}), the profile declares {self.weight_dtype}")

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "weight_dtype": self.weight_dtype,
            "compute_dtype": str(self.compute_dtype).removeprefix("torch."),
            "experts_implementation": self.experts_implementation,
            "attention_implementation": self.attention_implementation,
            "numerics": self.numerics,
            "quantization": dict(self.quantization),
        }


BF16_REFERENCE = ReferenceProfile(ReferenceKind.BF16, "BF16", torch.bfloat16)
FP16_REFERENCE = ReferenceProfile(ReferenceKind.FP16, "F16", torch.float16)


def native_quantized_reference(weight_dtype: str, compute_dtype: torch.dtype, **quantization: Any) -> ReferenceProfile:
    """A declared native quantized reference (e.g. weight_dtype "F8_E4M3", block scales): recorded, not yet executable."""
    return ReferenceProfile(ReferenceKind.NATIVE_QUANTIZED, weight_dtype, compute_dtype, quantization=dict(quantization))


PROFILES = {"bf16": BF16_REFERENCE, "fp16": FP16_REFERENCE}


def profile_for_dtype(dtype: torch.dtype) -> ReferenceProfile:
    """The plain (non-quantized) profile of a model dtype."""
    for profile in PROFILES.values():
        if profile.compute_dtype == dtype:
            return profile
    raise ValueError(f"no reference profile for {dtype} ({_DTYPE_NAMES.get(dtype, dtype)})")
