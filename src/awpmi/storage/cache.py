"""PageCache: materialized pages kept on the device between requests, within a byte budget.

A cache entry is the device copy of one page (a segment row) or of a whole segment. The
budget is in bytes (DwarfStar's expert cache is likewise a memory budget for complete
pages, not a byte cache). Pinned entries are never evicted. On a miss the caller fetches
the page and offers it with `put`; the cache admits it if it fits after evicting
unpinned entries chosen by its replacement policy:

  LRUPolicy      least recently used first
  HotnessPolicy  lowest exponentially decayed access count first ("route hotness"): every
                 access adds 1, and a score halves every `half_life` accesses to the cache

`admit = False` freezes the contents: lookups still hit, misses are served and dropped.
DwarfStar does this during long prefills, where a request's working set is larger than
the cache and LRU would evict every page between its insertion and its next use.

The policy decides only *which* pages stay resident. It never changes a page's bytes,
so it affects efficiency only.

SlotCache (Phase 6B, decision 0013) keeps pages in fixed slots of one device allocation per page
size, made when the cache is: its budget is the memory it holds from the start, and nothing is
allocated, freed or fragmented afterwards. Each page size has its own slots and policy (a page is
evicted only for a page of its size), and `put` copies the page into its slot.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass

import torch


class ReplacementPolicy(ABC):
    @abstractmethod
    def touch(self, key: Hashable) -> None:
        """An access to a resident entry (or the insertion of a new one)."""

    @abstractmethod
    def forget(self, key: Hashable) -> None: ...

    @abstractmethod
    def victim(self, candidates: set) -> Hashable:
        """The entry to evict among `candidates` (resident, unpinned)."""

    def tick(self) -> None:
        """One access to the cache, hit or miss."""


class LRUPolicy(ReplacementPolicy):
    def __init__(self) -> None:
        self._order: OrderedDict = OrderedDict()

    def touch(self, key) -> None:
        self._order[key] = None
        self._order.move_to_end(key)

    def forget(self, key) -> None:
        self._order.pop(key, None)

    def victim(self, candidates: set):
        for key in self._order:
            if key in candidates:
                return key
        raise LookupError("no evictable entry")


class HotnessPolicy(ReplacementPolicy):
    """Exponentially decayed access counts; ties go to the least recently touched.

    Every touch gets a unique sequence number, so the victim never depends on the order in
    which candidates are iterated (a set of string keys iterates in a per-process hash order).
    """

    def __init__(self, half_life: float = 1024.0) -> None:
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        self.half_life = half_life
        self._now = 0
        self._sequence = 0
        self._score: dict = {}
        self._last: dict = {}
        self._touched: dict = {}

    def tick(self) -> None:
        self._now += 1

    def _decayed(self, key) -> float:
        return self._score.get(key, 0.0) * math.exp2(-(self._now - self._last.get(key, self._now)) / self.half_life)

    def touch(self, key) -> None:
        self._score[key] = self._decayed(key) + 1.0
        self._last[key] = self._now
        self._sequence += 1
        self._touched[key] = self._sequence

    def forget(self, key) -> None:
        # Keep the history: a page that comes back keeps its hotness.
        pass

    def victim(self, candidates: set):
        return min(candidates, key=lambda key: (self._decayed(key), self._touched.get(key, -1)))


POLICIES = {"lru": LRUPolicy, "hotness": HotnessPolicy}


@dataclass
class CacheStats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    hit_bytes: int = 0
    miss_bytes: int = 0
    inserts: int = 0
    insert_bytes: int = 0
    evictions: int = 0
    evicted_bytes: int = 0
    bypassed: int = 0
    bypassed_bytes: int = 0

    def reset(self) -> None:
        self.__init__()

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class PageCache:
    """Device-resident pages under a byte budget; see the module docstring."""

    copies = False  # `put` keeps the caller's tensor (SlotCache copies it)

    def __init__(self, capacity_bytes: int, policy: ReplacementPolicy | None = None) -> None:
        if capacity_bytes < 0:
            raise ValueError("capacity must be non-negative")
        self.capacity_bytes = capacity_bytes
        self.policy = policy or LRUPolicy()
        self.admit = True
        self._entries: dict = {}
        self._pinned: set = set()
        self._namespaces: Counter = Counter()  # entries per key[0], for (namespace, item) keys
        self.pinned_bytes = 0
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.stats = CacheStats()

    def __contains__(self, key) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def peek(self, key) -> torch.Tensor | None:
        """The entry without counting an access."""
        return self._entries.get(key)

    def get(self, key, nbytes: int) -> torch.Tensor | None:
        """Look `key` up, counting a hit or a miss of `nbytes`."""
        self.stats.lookups += 1
        self.policy.tick()
        entry = self._entries.get(key)
        if entry is None:
            self.stats.misses += 1
            self.stats.miss_bytes += nbytes
            return None
        self.stats.hits += 1
        self.stats.hit_bytes += nbytes
        self.policy.touch(key)
        return entry

    def get_many(self, keys: list, nbytes: int) -> list[torch.Tensor | None]:
        """`get` for several keys of `nbytes` each."""
        return [self.get(key, nbytes) for key in keys]

    def holds(self, namespace) -> bool:
        """Whether any entry has a key (namespace, ...)."""
        return self._namespaces[namespace] > 0

    def count_misses(self, pages: int, nbytes: int) -> None:
        """Count `pages` lookups known to miss (the caller checked `holds`), without looking each up."""
        self.stats.lookups += pages
        self.stats.misses += pages
        self.stats.miss_bytes += nbytes
        for _ in range(pages):
            self.policy.tick()

    def can_admit(self, nbytes: int) -> bool:
        """Whether a page of `nbytes` would be admitted now (possibly after evictions)."""
        return self.admit and nbytes <= self.capacity_bytes - self.pinned_bytes

    def bypass(self, pages: int, nbytes: int) -> None:
        """Count pages that were served without being offered (the caller knew they would not be admitted)."""
        self.stats.bypassed += pages
        self.stats.bypassed_bytes += nbytes

    def put(self, key, tensor: torch.Tensor) -> bool:
        """Offer a fetched page; returns whether it was admitted. The cache keeps `tensor` itself."""
        nbytes = tensor.numel() * tensor.element_size()
        if key in self._entries:
            return True
        if not self.can_admit(nbytes):
            self.bypass(1, nbytes)
            return False
        while self.resident_bytes + nbytes > self.capacity_bytes:
            self._evict(self.policy.victim(set(self._entries) - self._pinned))
        self._insert(key, tensor, nbytes)
        return True

    def pin(self, key, tensor: torch.Tensor) -> None:
        """Make `tensor` resident and never evict it (it counts against the budget)."""
        nbytes = tensor.numel() * tensor.element_size()
        if key in self._pinned:
            return
        if key not in self._entries:
            if nbytes > self.capacity_bytes - self.pinned_bytes:
                raise ValueError("pinned pages exceed the cache budget")
            while self.resident_bytes + nbytes > self.capacity_bytes:
                self._evict(self.policy.victim(set(self._entries) - self._pinned))
            self._insert(key, tensor, nbytes)
        self._pinned.add(key)
        self.pinned_bytes += nbytes

    def clear(self) -> None:
        self._entries.clear()
        self._pinned.clear()
        self._namespaces.clear()
        self.pinned_bytes = 0
        self.resident_bytes = 0

    def _insert(self, key, tensor: torch.Tensor, nbytes: int) -> None:
        self._entries[key] = tensor
        if isinstance(key, tuple) and key:
            self._namespaces[key[0]] += 1
        self.resident_bytes += nbytes
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)
        self.stats.inserts += 1
        self.stats.insert_bytes += nbytes
        self.policy.touch(key)

    def _evict(self, key) -> None:
        tensor = self._entries.pop(key)
        if isinstance(key, tuple) and key:
            self._namespaces[key[0]] -= 1
        nbytes = tensor.numel() * tensor.element_size()
        self.resident_bytes -= nbytes
        self.stats.evictions += 1
        self.stats.evicted_bytes += nbytes
        self.policy.forget(key)


class _Slots(PageCache):
    """The slots of one page size (SlotCache): a PageCache whose entries are views of one allocation."""

    def __init__(self, page_bytes: int, count: int, device, policy: ReplacementPolicy, stats: CacheStats) -> None:
        super().__init__(page_bytes * count, policy)
        self.stats = stats  # one CacheStats for every size
        self.page_bytes = page_bytes
        self.memory = torch.empty((count, page_bytes), dtype=torch.uint8, device=device)
        self._free = list(range(count - 1, -1, -1))  # slot 0 first
        self._slot_of: dict = {}

    def place(self, key) -> int | None:
        """The slot a new page `key` goes to (after evicting by the policy if none is free), or None if `key` is
        resident or the cache admits nothing."""
        if key in self._entries:
            return None
        if not self.can_admit(self.page_bytes):
            self.bypass(1, self.page_bytes)
            return None
        while not self._free:
            self._evict(self.policy.victim(set(self._entries) - self._pinned))
        slot = self._free.pop()
        self._slot_of[key] = slot
        self._insert(key, self.memory[slot], self.page_bytes)
        return slot

    def clear(self) -> None:
        super().clear()
        self._free = list(range(self.memory.shape[0] - 1, -1, -1))
        self._slot_of.clear()

    def _evict(self, key) -> None:
        super()._evict(key)
        self._free.append(self._slot_of.pop(key))


class SlotCache:
    """Pages in fixed slots (module docstring): `PageCache`'s lookups, admission freeze and counters, for pages whose sizes
    are known in advance.

    `put` copies a page into its slot on the current stream; a slot an eviction frees is overwritten by a later `put`,
    after every earlier use of the old page on that stream, so entries must be used only on the stream that puts them.
    Pages always used together (an expert's rows) behave as under one policy when each size has as many slots.
    """

    copies = True

    def __init__(self, slots: Mapping[int, int], device, policy: Callable[[], ReplacementPolicy] = LRUPolicy) -> None:
        if not slots or any(int(size) <= 0 or int(count) <= 0 for size, count in slots.items()):
            raise ValueError("a slot cache needs at least one slot of a positive size per page size")
        self.stats = CacheStats()
        self._sizes = {int(size): _Slots(int(size), int(count), device, policy(), self.stats) for size, count in sorted(slots.items())}
        self.capacity_bytes = sum(pages.capacity_bytes for pages in self._sizes.values())
        self.peak_resident_bytes = 0
        self.pinned_bytes = 0
        self._admit = True

    @classmethod
    def sized(cls, capacity_bytes: int, page_bytes: Mapping[int, int], device, policy: Callable[[], ReplacementPolicy] = LRUPolicy) -> SlotCache:
        """Slots within `capacity_bytes` for pages of the sizes in `page_bytes` (a size → the bytes of all the pages of
        that size there are): each size gets a share of the budget in proportion to its pages' bytes, in whole slots."""
        total = sum(page_bytes.values())
        slots = {size: (capacity_bytes * share // total) // size for size, share in page_bytes.items()}
        if total <= 0 or not all(slots.values()):
            raise ValueError(f"{capacity_bytes} bytes hold no slot of some page size: {slots}")
        return cls(slots, device, policy)

    @property
    def slots(self) -> dict[int, int]:
        return {size: pages.memory.shape[0] for size, pages in self._sizes.items()}

    @property
    def admit(self) -> bool:
        return self._admit

    @admit.setter
    def admit(self, value: bool) -> None:
        self._admit = bool(value)
        for pages in self._sizes.values():
            pages.admit = self._admit

    @property
    def resident_bytes(self) -> int:
        return sum(pages.resident_bytes for pages in self._sizes.values())

    def __contains__(self, key) -> bool:
        return any(key in pages for pages in self._sizes.values())

    def __len__(self) -> int:
        return sum(len(pages) for pages in self._sizes.values())

    def get(self, key, nbytes: int) -> torch.Tensor | None:
        """Look `key` up, counting a hit or a miss of `nbytes` (a page size without slots always misses)."""
        pages = self._sizes.get(nbytes)
        if pages is None:
            self.stats.lookups += 1
            self.stats.misses += 1
            self.stats.miss_bytes += nbytes
            return None
        return pages.get(key, nbytes)

    def get_many(self, keys: list, nbytes: int) -> list[torch.Tensor | None]:
        return [self.get(key, nbytes) for key in keys]

    def can_admit(self, nbytes: int) -> bool:
        return self._admit and nbytes in self._sizes

    def bypass(self, pages: int, nbytes: int) -> None:
        self.stats.bypassed += pages
        self.stats.bypassed_bytes += nbytes

    def put(self, key, tensor: torch.Tensor) -> bool:
        """Offer a fetched page (a contiguous tensor of a slot's size); copies it into its slot if admitted. Returns
        whether the page is resident."""
        nbytes = tensor.numel() * tensor.element_size()
        pages = self._sizes.get(nbytes)
        if pages is None or not self._admit:
            if key not in self:
                self.bypass(1, nbytes)
            return key in self
        slot = pages.place(key)
        if slot is None:
            return key in pages
        pages.memory[slot].copy_(tensor.reshape(-1).view(torch.uint8))
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)
        return True

    def clear(self) -> None:
        for pages in self._sizes.values():
            pages.clear()
