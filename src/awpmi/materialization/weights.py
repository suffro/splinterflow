"""Typed weights and expert groups over a `MaterializationBackend`.

`WeightStore.rows(name, rows)` is a weight's rows as tensors of the weight's dtype and row
shape. `ExpertStore` serves mixture-of-experts layers: a group is one layer's expert-sliced
parameters (each a segment with one row per expert); `load` returns the requested experts'
slices, `fill` writes them into caller-owned full-shape buffers at their expert indices, and
`assemble` writes them into caller-owned compact buffers, row i holding the i-th requested
expert (decision 0007). Group keys and parameter names are whatever the model adapter
registers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from awpmi.materialization.backend import MaterializationBackend
from awpmi.storage.layout import AnySegment, row_bytes_of, typed_rows


class WeightStore:
    def __init__(self, backend: MaterializationBackend) -> None:
        self.backend = backend

    def segment(self, name: str) -> AnySegment:
        return self.backend.segment(name)

    def rows(self, name: str, rows: torch.Tensor | None = None) -> torch.Tensor:
        """Rows of weight `name` as [n, *row_shape] tensors of its dtype, on the compute device."""
        return typed_rows(self.backend.materialize(name, rows), self.segment(name))


@dataclass(frozen=True)
class ExpertGroup:
    """The expert-sliced parameters of one MoE layer: parameter name → segment (one row per expert)."""

    key: str
    experts: int
    segments: Mapping[str, str]


class ExpertStore:
    def __init__(self, weights: WeightStore, groups: Mapping[str, ExpertGroup]) -> None:
        for group in groups.values():
            for segment in group.segments.values():
                if weights.segment(segment).rows != group.experts:
                    raise ValueError(f"{group.key}: segment {segment} does not have one row per expert")
        self.weights = weights
        self.groups = dict(groups)

    def load(self, key: str, experts: torch.Tensor) -> dict[str, torch.Tensor]:
        """The slices of `experts` (ascending, unique) of every parameter of group `key`."""
        group = self.groups[key]
        return {name: self.weights.rows(segment, experts) for name, segment in group.segments.items()}

    def fill(self, key: str, experts: torch.Tensor, buffers: Mapping[str, torch.Tensor]) -> None:
        """Write the slices of `experts` into `buffers[name][experts]` (full-shape device tensors)."""
        if experts.numel() == 0:
            return
        index = experts.to(next(iter(buffers.values())).device)
        for name, data in self.load(key, experts).items():
            buffers[name].index_copy_(0, index, data)

    def prefetch(self, key: str, experts: torch.Tensor, parameters: list[str] | None = None):
        """A hint that `experts` (ascending) of group `key` will be assembled soon: every parameter, or `parameters`, in the
        group's order. Returns the backend's handle (close it once they were used) or None (`prefetch_rows`)."""
        group = self.groups[key]
        names = [name for name in group.segments if parameters is None or name in parameters]
        return self.weights.backend.prefetch_rows([(group.segments[name], experts) for name in names])

    def assemble(self, key: str, experts: torch.Tensor, buffers: Mapping[str, torch.Tensor]) -> None:
        """Write the slices of `experts` (ascending, unique) into the first rows of `buffers[name]`, in that order.

        Each buffer is a contiguous device tensor [≥ n, *row_shape] of the parameter's dtype;
        rows beyond n are left as they are. `buffers` may name only some of the group's
        parameters (a chunked call materializes one parameter at a time); they are filled in
        the group's order.
        """
        group = self.groups[key]
        unknown = set(buffers) - set(group.segments)
        if unknown:
            raise KeyError(f"{key} has no parameters {sorted(unknown)}")
        count = experts.numel()
        requests = []
        for name, segment in group.segments.items():
            if name not in buffers:
                continue
            info = self.weights.segment(segment)
            buffer = buffers[name]
            if buffer.dtype != info.torch_dtype or tuple(buffer.shape[1:]) != info.row_shape or buffer.shape[0] < count:
                raise ValueError(f"{key}.{name}: expected a buffer [>= {count}, {info.row_shape}] of {info.torch_dtype}")
            requests.append((segment, experts, row_bytes_of(buffer[:count])))
        if count:
            # Every parameter of the experts at once: a native store moves them in one transfer (decision 0012).
            self.weights.backend.materialize_many(requests)
