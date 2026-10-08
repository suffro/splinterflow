"""Moonlight's routed experts, read tensor by tensor from the published checkpoint (Phase 5C).

Names, shapes, dtypes and byte offsets come from the safetensors headers (`awpmi.storage.layout`) and the checkpoint
index; expert tensor names follow the Moonlight adapter's `EXPERT_LAYOUT` (gate and up rows are neurons, down rows are
outputs). Two read paths that share no code:

  `read_patterns`   plain positioned reads of the file at the header's offset (the bytes as stored)
  `read_reference`  safetensors' own `safe_open(...).get_tensor` (the independent copy reconstructions are compared to)

Nothing proportional to the model is held: at most one matrix kind of one layer (64 × 5.5 MiB) at a time. The files'
sha256 are the publisher's (`awpmi.models.checkpoint.checkpoint_sources`), checked by a full direct read
(`awpmi.storage.pack.sha256_file_direct`) when asked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from awpmi.models.moonlight import EXPERT_LAYOUT
from awpmi.storage.layout import safetensors_segments

MATRICES = ("gate", "up", "down")
_TEMPLATES = {"gate": EXPERT_LAYOUT["gate_up_proj"][0], "up": EXPERT_LAYOUT["gate_up_proj"][1], "down": EXPERT_LAYOUT["down_proj"][0]}


@dataclass(frozen=True)
class ExpertTensor:
    layer: int
    expert: int
    matrix: str  # gate, up, down
    name: str
    file: str
    offset: int  # absolute byte offset in the file
    nbytes: int
    shape: tuple[int, int]
    dtype: str

    def to_json(self) -> dict:
        return {"layer": self.layer, "expert": self.expert, "matrix": self.matrix, "name": self.name, "file": self.file,
                "offset": self.offset, "nbytes": self.nbytes, "shape": list(self.shape), "dtype": self.dtype}


class Checkpoint:
    """The pinned Moonlight checkpoint in the local Hugging Face cache (no download)."""

    def __init__(self, repository: str, revision: str) -> None:
        from huggingface_hub import hf_hub_download

        self.repository, self.revision = repository, revision
        index = Path(hf_hub_download(repository, "model.safetensors.index.json", revision=revision, local_files_only=True))
        self.weight_map: dict[str, str] = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        self._paths: dict[str, Path] = {}
        self._segments: dict[str, dict] = {}

    def path(self, file: str) -> Path:
        if file not in self._paths:
            from huggingface_hub import hf_hub_download

            self._paths[file] = Path(hf_hub_download(self.repository, file, revision=self.revision, local_files_only=True))
        return self._paths[file]

    def _segment(self, name: str):
        file = self.weight_map[name]
        if file not in self._segments:
            self._segments[file] = safetensors_segments(self.path(file), file)
        return self._segments[file][name]

    def experts(self, layer: int) -> int:
        count = 0
        while f"model.layers.{layer}." + _TEMPLATES["gate"].format(expert=count) in self.weight_map:
            count += 1
        return count

    def tensor(self, layer: int, expert: int, matrix: str) -> ExpertTensor:
        name = f"model.layers.{layer}." + _TEMPLATES[matrix].format(expert=expert)
        segment = self._segment(name)
        shape = (segment.rows, *segment.row_shape)
        if len(shape) != 2:
            raise ValueError(f"{name}: expected a matrix, got {shape}")
        return ExpertTensor(layer, expert, matrix, name, segment.file, segment.offset, segment.nbytes, shape, segment.dtype)

    # Two independent read paths

    def read_patterns(self, tensor: ExpertTensor) -> np.ndarray:
        """The tensor's bytes as stored (positioned read), as uint16 patterns [rows, cols]."""
        if tensor.dtype != "BF16":
            raise ValueError(f"{tensor.name}: {tensor.dtype}, expected BF16")
        out = np.empty(tensor.nbytes // 2, dtype=np.uint16)
        with open(self.path(tensor.file), "rb", buffering=0) as handle:
            handle.seek(tensor.offset)
            view = memoryview(out.view(np.uint8))
            done = 0
            while done < tensor.nbytes:
                count = handle.readinto(view[done:])
                if not count:
                    raise EOFError(f"{tensor.name}: short read")
                done += count
        return out.reshape(tensor.shape)

    def read_reference(self, tensor: ExpertTensor) -> torch.Tensor:
        """The same tensor through safetensors' own loader (BF16, CPU)."""
        from safetensors import safe_open

        with safe_open(str(self.path(tensor.file)), framework="pt", device="cpu") as handle:
            return handle.get_tensor(tensor.name)

    def matrices(self, layer: int, matrix: str, experts: list[int] | None = None) -> np.ndarray:
        """One matrix kind of one layer: uint16 patterns [E, rows, cols] (at most 64 × 5.5 MiB)."""
        experts = list(range(self.experts(layer))) if experts is None else experts
        return np.stack([self.read_patterns(self.tensor(layer, e, matrix)) for e in experts])


def verify_files(checkpoint: Checkpoint, files: list[str]) -> dict[str, dict]:
    """Each file's sha256 by a full direct read, against the publisher's (the Hub's LFS id; None when offline)."""
    from awpmi.models.checkpoint import checkpoint_sources
    from awpmi.storage.pack import sha256_file_direct

    sources = checkpoint_sources(checkpoint.repository, checkpoint.revision, declared_sha256=True)
    result = {}
    for file in files:
        declared = sources[file].sha256
        actual = sha256_file_direct(checkpoint.path(file))
        result[file] = {"declared": declared, "actual": actual, "equal": declared is not None and declared == actual}
    return result
