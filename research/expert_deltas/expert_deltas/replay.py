"""Routing traces replayed through a host-RAM cache of stored experts, deltas and bases (Phase 5C, stage 5C1-B).

The trace is Phase 5A's capture records: every prompt's prefill and decode steps, each with the routed experts of every
MoE layer. A representation stores, per (layer, expert), one object on the drive (the checkpoint's BF16 bytes, an
independently compressed expert, or its compressed deltas), and, for base/delta representations, base objects: one per
(matrix kind, base) that some expert's delta refers to. A base is stored as its own compressed object (a copy of an
actual expert's matrix, or a synthetic matrix: derived storage, charged), so reading it never needs another expert.

Per layer, a cache of `capacity` bytes (Weightsift's `PageCache` with `LRUPolicy`, holding size-only entries as the Phase
4B report's replay did) serves the step's routed experts in ascending order: the expert's object is looked up, and on a
miss read from the drive (its stored bytes) and inserted; then each base it needs. Base residency:

  host     bases pinned in the cache, decoded (XOR reconstruction reads every base pattern: their BF16 bytes count
           against the capacity), read once before the first step; when they do not fit, as "bounded"
  bounded  bases are ordinary entries (decoded size), evicted under pressure and read again on a miss
  gpu      bases held in device memory, outside the host budget (their bytes reported as device memory), read once

Reported: drive bytes per step (prefill and decode), the steady state (decode steps after the warm-up prompts), the cold
start (the first step, with any up-front base reads), hits and misses, base misses and evictions, peak host bytes.
Bytes are logical object sizes (a 4 KiB-aligned layout adds at most one block per object read).
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

from awpmi.storage.cache import LRUPolicy, PageCache


class _Sized:
    """A cache entry of a given size, without its bytes."""

    def __init__(self, nbytes: int) -> None:
        self.nbytes = int(nbytes)

    def numel(self) -> int:
        return self.nbytes

    def element_size(self) -> int:
        return 1


@dataclass(frozen=True)
class Representation:
    """One layer's stored objects: bytes per expert, the bases each expert needs, and each base's stored and decoded bytes."""

    name: str
    expert_bytes: list[int]
    bases_of: list[tuple[Hashable, ...]] = field(default_factory=list)  # per expert (empty: independent)
    base_stored_bytes: dict[Hashable, int] = field(default_factory=dict)
    base_resident_bytes: dict[Hashable, int] = field(default_factory=dict)

    @property
    def base_ids(self) -> list[Hashable]:
        return sorted(self.base_stored_bytes, key=repr)

    def needs(self, expert: int) -> tuple[Hashable, ...]:
        return self.bases_of[expert] if self.bases_of else ()


def trace_steps(records: list[dict], layers: list[int]) -> list[dict]:
    """(prompt, step, phase, routed experts per sampled layer) in the capture's order; `layers` are model layer numbers
    (the records list the MoE layers 1..26 in order)."""
    return [{"prompt_id": r["prompt_id"], "step": r["step"], "phase": r["phase"],
             "routed": {layer: sorted(r["routed"][layer - 1]) for layer in layers}} for r in records]


def replay(steps: list[dict], layers: dict[int, Representation], capacity_per_layer: int, base_residency: str,
           warmup_prompts: int) -> dict:
    """`base_residency`: "host", "bounded" or "gpu" (module docstring); irrelevant for independent representations."""
    caches, pinned, gpu_bytes, up_front = {}, {}, 0, 0
    for layer, rep in layers.items():
        cache = PageCache(capacity_per_layer, LRUPolicy())
        resident = sum(rep.base_resident_bytes.values())
        pinned[layer] = bool(rep.base_ids) and base_residency == "host" and resident <= capacity_per_layer
        if pinned[layer]:
            for b in rep.base_ids:
                cache.pin(("base", b), _Sized(rep.base_resident_bytes[b]))
        if rep.base_ids and (pinned[layer] or base_residency == "gpu"):
            up_front += sum(rep.base_stored_bytes.values())
        if rep.base_ids and base_residency == "gpu":
            gpu_bytes += resident
        caches[layer] = cache
    per_step, base_misses, base_hits = [], 0, 0
    for step in steps:
        drive = 0
        for layer, rep in layers.items():
            cache = caches[layer]
            for e in step["routed"][layer]:
                if cache.get(("expert", e), rep.expert_bytes[e]) is None:
                    drive += rep.expert_bytes[e]
                    cache.put(("expert", e), _Sized(rep.expert_bytes[e]))
                if pinned[layer] or base_residency == "gpu":
                    continue
                for b in rep.needs(e):
                    if cache.get(("base", b), rep.base_resident_bytes[b]) is None:
                        base_misses += 1
                        drive += rep.base_stored_bytes[b]
                        cache.put(("base", b), _Sized(rep.base_resident_bytes[b]))
                    else:
                        base_hits += 1
        per_step.append({"prompt_id": step["prompt_id"], "step": step["step"], "phase": step["phase"], "drive_bytes": drive})
    order = list(dict.fromkeys(s["prompt_id"] for s in steps))
    steady = [s["drive_bytes"] for s in per_step if s["phase"] == "decode" and s["prompt_id"] in order[warmup_prompts:]]
    decode = [s["drive_bytes"] for s in per_step if s["phase"] == "decode"]
    prefill = [s["drive_bytes"] for s in per_step if s["phase"] == "prefill"]
    stats = [c.stats for c in caches.values()]
    return {
        "capacity_per_layer": capacity_per_layer, "base_residency": base_residency, "bases_pinned_layers": sum(pinned.values()),
        "steady_decode_bytes": sum(steady) / max(1, len(steady)), "decode_bytes": sum(decode) / max(1, len(decode)),
        "prefill_bytes": sum(prefill) / max(1, len(prefill)),
        "cold_start_bytes": up_front + (per_step[0]["drive_bytes"] if per_step else 0), "up_front_base_bytes": up_front,
        "steady_steps": len(steady),
        # Cache statistics count expert and unpinned base lookups together; base_hits and base_misses separate the latter.
        "lookups": sum(s.lookups for s in stats), "hits": sum(s.hits for s in stats), "misses": sum(s.misses for s in stats),
        "evictions": sum(s.evictions for s in stats), "base_hits": base_hits, "base_misses": base_misses,
        "host_resident_peak_bytes": sum(c.peak_resident_bytes for c in caches.values()), "gpu_base_bytes": gpu_bytes,
    }
