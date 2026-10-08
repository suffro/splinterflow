"""The progressive oracle of Phase 5C (stage 5C2): routed experts as bit-plane pages, read along a schedule, and the
decision over what is read at each checkpoint (`expert_deltas.progressive` for the sets and their exact optimum).

Shared by the probe (`progressive_probe.py`, stage 5C2-A) and the sample run (`progressive_run.py`, stage 5C2-B). A
diagnostic simulator in real arithmetic: it holds every weight, builds every set from a read view whose unread bits are
poisoned, and keeps the truth for validation only (the comparison rows' Δ·y, never a bound or an ordering).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections.abc import Callable

import numpy as np
import torch

from expert_deltas import bits, records
from expert_deltas.codecs import Codec, compress_many, decompress_many
from expert_deltas.progressive import (
    PAGE_ROWS,
    STEP_PREFIX,
    ExpertSet,
    Minimum,
    box,
    exact_minimum,
    linf_bounds,
    read_view,
    silu,
    witness,
)

CODEC = Codec("zstd", 19)
STEPS = len(STEP_PREFIX) - 1  # 8 steps per page
MinimumFn = Callable[[list, torch.Tensor, torch.Tensor, torch.Tensor], Minimum]


# Loading


def load_capture(settings: dict) -> tuple[dict, dict]:
    from safetensors.torch import load_file

    path = records.REPO_ROOT / settings["capture"] / "capture.safetensors"
    digest = json.loads((records.REPO_ROOT / settings["phase5a_run"] / "digest.json").read_text(encoding="utf-8"))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    check = {"capture_sha256": actual, "phase5a_digest": digest["capture_tensors_sha256"], "equal": actual == digest["capture_tensors_sha256"]}
    return load_file(str(path)), check


def comparison_records(settings: dict) -> dict[tuple[int, int], list[dict]]:
    path = records.REPO_ROOT / settings["phase5a2_run"] / "l2.0.jsonl.gz"
    found: dict[tuple[int, int], list[dict]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            found.setdefault((r["prompt_id"], r["step"]), []).append(
                {"fraction": r["fraction"], "rows": r["rows"], "truth": r["truth"], "realistic": r["phase5a"]["realistic"]["decomposed"],
                 "ideal": r["phase5a"]["ideal"]["decomposed"], "witness": r["witness"]})
    for value in found.values():
        value.sort(key=lambda s: s["fraction"])
    return found


def capture_tokens(settings: dict) -> dict[tuple[int, int], int]:
    from awpmi.tracing import read_jsonl

    return {(r["prompt_id"], r["step"]): r["token"] for r in read_jsonl(records.REPO_ROOT / settings["phase5a_run"] / "capture.jsonl.gz")}


# The representation: pages of bit planes


class ExpertPages:
    """One routed expert stored as bit-plane pages: frames, sizes, and its patterns (the oracle holds them)."""

    def __init__(self, patterns: dict[str, np.ndarray], threads: int) -> None:
        self.patterns = patterns
        self.neurons, self.hidden = patterns["gate"].shape
        self.pages = []  # (kind, index, flat patterns)
        for p in range(0, self.neurons, PAGE_ROWS):
            self.pages.append(("neuron", p // PAGE_ROWS, np.concatenate([patterns["gate"][p : p + PAGE_ROWS].reshape(-1),
                                                                         patterns["up"][p : p + PAGE_ROWS].reshape(-1)])))
        for p in range(0, self.hidden, PAGE_ROWS):
            self.pages.append(("down", p // PAGE_ROWS, patterns["down"][p : p + PAGE_ROWS].reshape(-1)))
        streams = [plane for _, _, flat in self.pages for plane in bits.split_planes(flat).values()]
        self.frames = compress_many(streams, CODEC, None, threads)
        self.raw_lengths = [len(s) for s in streams]
        sizes = np.array([len(f) for f in self.frames], dtype=np.int64).reshape(len(self.pages), len(bits.PLANES))
        self.neuron_planes = sizes[: self.neurons // PAGE_ROWS]
        self.down_planes = sizes[self.neurons // PAGE_ROWS :]

    @staticmethod
    def steps(planes: np.ndarray) -> np.ndarray:
        """[pages, 8]: step 1 = sign + exponent frames, steps 2..8 = one mantissa frame each."""
        return np.concatenate([planes[:, :2].sum(axis=1, keepdims=True), planes[:, 2:]], axis=1)

    @property
    def total(self) -> int:
        return int(self.neuron_planes.sum() + self.down_planes.sum())

    def decode(self, threads: int) -> dict[str, np.ndarray]:
        """Every page decoded and merged (the full fallback's reconstruction)."""
        out = decompress_many(self.frames, self.raw_lengths, CODEC, None, threads)
        planes = len(bits.PLANES)
        gate, up, down = [], [], []
        for k, (kind, _, flat) in enumerate(self.pages):
            merged = bits.merge_planes(dict(zip(bits.PLANES, out[k * planes : (k + 1) * planes])), flat.size)
            if kind == "neuron":
                half = merged.size // 2
                gate.append(merged[:half].reshape(-1, self.hidden))
                up.append(merged[half:].reshape(-1, self.hidden))
            else:
                down.append(merged.reshape(-1, self.neurons))
        return {"gate": np.concatenate(gate), "up": np.concatenate(up), "down": np.concatenate(down)}


# States


class Schedule:
    """Units (slot, kind, page, step) in reading order with their bytes; a state is a prefix of the order."""

    def __init__(self, units: list[tuple[int, str, int, int]], nbytes: np.ndarray) -> None:
        self.units, self.nbytes = units, nbytes
        self.spent = np.cumsum(nbytes)

    def state(self, budget: float, slots: int, neuron_pages: int, down_pages: int) -> tuple[np.ndarray, np.ndarray, int]:
        """Steps done per page ([slots, pages] for neurons and down) after reading every unit within `budget` bytes."""
        count = int(np.searchsorted(self.spent, budget + 0.5, side="right"))
        neuron = np.zeros((slots, neuron_pages), dtype=np.int64)
        down = np.zeros((slots, down_pages), dtype=np.int64)
        for slot, kind, page, step in self.units[:count]:
            target = neuron if kind == "neuron" else down
            target[slot, page] = max(target[slot, page], step)
        return neuron, down, int(self.spent[count - 1]) if count else 0


def prefixes(steps: np.ndarray, rows: int) -> torch.Tensor:
    """Per row the prefix length of its page's steps ([slots, pages] → [slots, rows])."""
    table = torch.tensor(STEP_PREFIX, dtype=torch.int64)
    per_page = table[torch.from_numpy(steps)]
    return per_page.repeat_interleave(PAGE_ROWS, dim=1)[:, :rows]


# Decision


class Sample:
    """One decode sample: the routed experts' patterns on the device, metadata, inputs, the comparison rows."""

    def __init__(self, x, residual, shared, experts, weights, pages: list[ExpertPages], lm_weight, norm_weight, device) -> None:
        self.device = device
        self.x = x.to(device, torch.float64)
        self.base = residual.to(device, torch.float64) + shared.to(device, torch.float64)
        self.experts, self.weights = list(experts), [float(w) for w in weights]
        self.lm, self.gain = lm_weight, norm_weight.to(torch.float64)
        self.patterns = [{m: torch.from_numpy(p.patterns[m].astype(np.int64)).to(device) for m in ("gate", "up", "down")} for p in pages]
        self.linf = [{m: linf_bounds(mats[m]) for m in mats} for mats in self.patterns]
        self.neurons, self.hidden = pages[0].neurons, pages[0].hidden

    def sets(self, neuron_steps: np.ndarray, down_steps: np.ndarray, seed: int | None) -> list[ExpertSet]:
        """The box of every routed expert from a read view whose unread bits are poisoned with `seed`."""
        gate_prefix = prefixes(neuron_steps, self.neurons).to(self.device)
        down_prefix = prefixes(down_steps, self.hidden).to(self.device)
        out = []
        for slot, mats in enumerate(self.patterns):
            boxes = {}
            for m, prefix in (("gate", gate_prefix[slot]), ("up", gate_prefix[slot]), ("down", down_prefix[slot])):
                boxes[m] = box(read_view(mats[m], prefix, seed), prefix, self.linf[slot][m])
            out.append(ExpertSet(boxes["gate"], boxes["up"], boxes["down"], self.weights[slot]))
        return out

    def delta(self, winner: int, rows: torch.Tensor) -> torch.Tensor:
        w = self.lm[winner].to(torch.float64)
        return (w[None, :] - self.lm.index_select(0, rows).to(torch.float64)) * self.gain[None, :]

    def centre_logits(self, sets: list[ExpertSet]) -> torch.Tensor:
        """W·(g⊙y) at the boxes' centres, real arithmetic (q > 0 is common): what a runtime ranks its candidate by."""
        y = self.base.clone()
        for s in sets:
            y = y + s.weight * (s.down.centre @ (silu(s.gate.centre @ self.x) * (s.up.centre @ self.x)))
        v = (y * self.gain).to(torch.float32)
        return torch.cat([self.lm[i : i + 16384].to(torch.float32) @ v for i in range(0, self.lm.shape[0], 16384)])

    def truth(self, rows: torch.Tensor, winner: int) -> torch.Tensor:
        y = self.base.clone()
        for slot, mats in enumerate(self.patterns):
            v = {m: bits.pattern_values(mats[m]) for m in mats}
            y = y + self.weights[slot] * (v["down"] @ (silu(v["gate"] @ self.x) * (v["up"] @ self.x)))
        return self.delta(winner, rows) @ y

    def certify(self, sets: list[ExpertSet], near: int, full: bool, chunk: int, minimum: MinimumFn = exact_minimum) -> dict:
        logits = self.centre_logits(sets)
        candidate = int(torch.argmax(logits))
        order = torch.argsort(logits, descending=True, stable=True)
        rows = order[order != candidate][:near]
        found = minimum(sets, self.x, self.base, self.delta(candidate, rows))
        near_ok = bool(found.decided.all())
        result = {"candidate": candidate, "near_decided": near_ok, "near_min": float(found.value.min()), "full_checked": False, "certified": False}
        if near_ok and full:
            undecided = 0
            others = torch.arange(self.lm.shape[0], device=self.device)
            others = others[others != candidate]
            for start in range(0, others.numel(), chunk):
                part = others[start : start + chunk]
                undecided += int((~minimum(sets, self.x, self.base, self.delta(candidate, part)).decided).sum())
            result.update(full_checked=True, undecided=undecided, certified=undecided == 0)
        return result


# Schedules


def sequential(pages: list[ExpertPages]) -> Schedule:
    """Plane-major: every page's step 1 (sign and exponent), then every page's step 2, ..."""
    units, nbytes = [], []
    for step in range(1, STEPS + 1):
        for slot, p in enumerate(pages):
            for kind, table in (("neuron", ExpertPages.steps(p.neuron_planes)), ("down", ExpertPages.steps(p.down_planes))):
                for page in range(table.shape[0]):
                    units.append((slot, kind, page, step))
                    nbytes.append(int(table[page, step - 1]))
    return Schedule(units, np.array(nbytes, dtype=np.int64))


def greedy(pages: list[ExpertPages], sample: Sample) -> Schedule:
    """Every page's step 1, then the remaining steps by estimated bound reduction per byte for the tightest pair at that
    first state (resident information and what was read only: after the exponent, every later half-width is known), each
    page's steps in order (a step's score is capped by its earlier steps')."""
    first = sequential(pages)
    count = sum(1 for u in first.units if u[3] == 1)
    units, nbytes = first.units[:count], list(first.nbytes[:count])
    slots = len(pages)
    neuron_steps = np.ones((slots, pages[0].neuron_planes.shape[0]), dtype=np.int64)
    down_steps = np.ones((slots, pages[0].down_planes.shape[0]), dtype=np.int64)
    sets = sample.sets(neuron_steps, down_steps, seed=None)
    logits = sample.centre_logits(sets)
    candidate = int(torch.argmax(logits))
    order = torch.argsort(logits, descending=True, stable=True)
    rows = order[order != candidate][:64]
    found = exact_minimum(sets, sample.x, sample.base, sample.delta(candidate, rows))
    tight = int(torch.argmin(found.value))
    delta = sample.delta(candidate, rows[tight : tight + 1])[0]
    # Half-width of a weight after k more mantissa bits, from its half-width at t = 9: h·(2^(7−k) − 1)/(2^7 − 1).
    factors = torch.tensor([(2.0 ** (7 - k) - 1) / 127.0 for k in range(8)], dtype=torch.float64, device=sample.device)
    candidates, scores = [], []
    for slot, (p, s) in enumerate(zip(pages, sets)):
        act = found.activations[slot]
        reach = torch.maximum(act.a[0].abs(), act.a[1].abs())  # [I]
        M = delta @ s.down.centre
        H = delta.abs() @ s.down.half
        # down page steps: the reduction of H·reach
        per_row = (delta.abs()[:, None] * s.down.half) @ reach  # [H] rows' share at t = 9
        down_pages = per_row.view(-1, PAGE_ROWS).sum(dim=1) if per_row.numel() % PAGE_ROWS == 0 else per_row
        # neuron page steps: the reduction of a's radius times (|M| + H)
        g_half = s.gate.half @ sample.x.abs()
        u_half = s.up.half @ sample.x.abs()
        s_reach = torch.maximum(silu(act.g[0]).abs(), silu(act.g[1]).abs())
        u_reach = torch.maximum(act.u[0].abs(), act.u[1].abs())
        neuron_share = (M.abs() + H) * (1.1 * u_reach * g_half + s_reach * u_half)
        neuron_pages = neuron_share.view(-1, PAGE_ROWS).sum(dim=1)
        steps_neuron = ExpertPages.steps(p.neuron_planes)
        steps_down = ExpertPages.steps(p.down_planes)
        for kind, share, table in (("neuron", neuron_pages, steps_neuron), ("down", down_pages, steps_down)):
            for page in range(table.shape[0]):
                previous = math.inf
                for step in range(2, STEPS + 1):
                    k = step - 2  # mantissa bits read before this step
                    gain = float(share[page]) * float(factors[k] - factors[k + 1] if k + 1 < 8 else factors[k])
                    score = min(previous, gain * sample.weights[slot] / max(1, int(table[page, step - 1])))
                    previous = score
                    candidates.append((slot, kind, page, step))
                    scores.append(score)
    ranked = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    for i in ranked:
        slot, kind, page, step = candidates[i]
        units.append(candidates[i])
        table = ExpertPages.steps(pages[slot].neuron_planes if kind == "neuron" else pages[slot].down_planes)
        nbytes.append(int(table[page, step - 1]))
    return Schedule(units, np.array(nbytes, dtype=np.int64))


# Physical reads (modelled)


def physical(pages: list[ExpertPages], neuron_steps: np.ndarray, down_steps: np.ndarray) -> dict[str, dict[str, int]]:
    """4 KiB blocks and extents of the frames read at a state, per routed expert's file, for two layouts:

      page_major   each page's frames back to back in plane order (sign, exponent, m6..m0), pages in order: a page
                   read to step s is one run of its first s + 1 frames
      plane_major  every page's sign frame, then every page's exponent frame, ...: a plane's region is read where its
                   pages reached it, and an unread last plane is one region skipped

    Phase 5A's `blocks_and_extents` counts the blocks (a block is read if it holds a requested byte)."""
    from awpmi.oracle.experts import blocks_and_extents

    out = {}
    for layout in ("page_major", "plane_major"):
        ranges = []
        logical = 0
        for slot, p in enumerate(pages):
            frames = np.concatenate([p.neuron_planes, p.down_planes])  # [pages, 9]
            steps = np.concatenate([neuron_steps[slot], down_steps[slot]])
            # Step 1 reads frames 0 and 1 (sign, exponent); step k ≥ 2 reads frame k (mantissa bit 8 − k).
            read = np.zeros(frames.shape, dtype=np.int64)
            read[:, 0] = read[:, 1] = steps >= 1
            read[:, 2:] = steps[:, None] >= np.arange(2, 9)[None, :]
            logical += int((frames * read).sum())
            if layout == "page_major":
                offsets = np.concatenate([[0], np.cumsum(frames.reshape(-1))[:-1]]).reshape(frames.shape)
            else:
                offsets = np.concatenate([[0], np.cumsum(frames.T.reshape(-1))[:-1]]).reshape(frames.T.shape).T
            for page, plane in zip(*np.nonzero(read)):
                start = int(offsets[page, plane])
                ranges.append((f"expert{slot}", start, start + int(frames[page, plane])))
        blocks, extents = blocks_and_extents(ranges)
        out[layout] = {"logical_bytes": logical, "blocks_4k": blocks, "physical_bytes": blocks * 4096, "extents": extents}
    return out


# The probe


def run_sample(sample: Sample, pages: list[ExpertPages], schedule: Schedule, name: str, comparison: list[dict], token: int,
               settings: dict, routed: int, metadata: int, best_independent: int, minimum: MinimumFn = exact_minimum,
               exact_sets: bool = True) -> dict:
    """A schedule's checkpoints on one sample: the set's minimum on the comparison rows against the truth and Phase 5A's
    realistic bound at a matched (no smaller) byte fraction, witnesses where negative (`exact_sets`: the minimum is the
    set's own, so its minimizer lies in the set; a relaxation has no witness), the first certifying checkpoint, and a
    poisoning check. `metadata` is charged at every checkpoint."""
    device = sample.device
    rows = torch.tensor(comparison[0]["rows"], device=device)
    truth = sample.truth(rows, token)
    phase5a_truth = torch.tensor(comparison[0]["truth"], dtype=torch.float64, device=device)
    step = float(settings["budget_step"]) * routed
    total = int(schedule.spent[-1])
    slots, neuron_pages, down_pages = len(pages), pages[0].neuron_planes.shape[0], pages[0].down_planes.shape[0]
    checkpoints = []
    budgets = sorted(set([min(total, step * k) for k in range(1, math.ceil(total / step) + 1)] + [total]))
    last_flip = last_flip_representation = None
    for index, budget in enumerate(budgets):
        neuron_steps, down_steps, spent = schedule.state(budget, slots, neuron_pages, down_pages)
        sets = sample.sets(neuron_steps, down_steps, seed=index)
        found = minimum(sets, sample.x, sample.base, sample.delta(token, rows))
        charged = spent + metadata
        f_raw = charged / routed
        matched = next((s for s in comparison if s["fraction"] >= f_raw - 1e-12), comparison[-1])
        realistic = torch.tensor(matched["realistic"], dtype=torch.float64, device=device)
        new = found.value
        share = ((truth - new) / (truth - realistic).clamp_min(1e-300)).median()
        entry = {
            "budget": budget, "charged": charged, "f_raw": f_raw, "f_best_independent": charged / best_independent,
            "f_representation": spent / total,
            "steps": {"neuron": np.bincount(neuron_steps.reshape(-1), minlength=STEPS + 1).tolist(), "down": np.bincount(down_steps.reshape(-1), minlength=STEPS + 1).tolist()},
            "min_over_rows": float(new.min()), "truth_min": float(truth.min()), "decided_rows": int(found.decided.sum()),
            "phase5a_fraction": matched["fraction"], "phase5a_realistic_min": float(realistic.min()),
            "distance_share_median": float(share), "rows_negative": int((new < 0).sum()),
            "rows_above_truth": int((new > truth + 1e-9 * (1 + truth.abs())).sum()),  # must be 0: the true weights lie in the set
        }
        if exact_sets and bool((new < 0).any()):
            j = int(torch.argmin(new))
            w = witness(sets, sample.x, sample.base, sample.delta(token, rows[j : j + 1])[0], found, j)
            entry["witness"] = {"row": int(rows[j]), "value": w.value, "inside": w.inside, "minimum": float(new[j])}
            last_flip, last_flip_representation = f_raw, spent / total
        checkpoints.append(entry)
    # Consistency: the real forward's Δ·y equals Phase 5A2's truth for the same rows.
    consistency = float(((truth - phase5a_truth).abs() / (1 + phase5a_truth.abs())).max())
    # The first certifying checkpoint: bisection on the nearest rows (monotone: sets only shrink), then every row.
    def certify_at(index: int, full: bool) -> dict:
        neuron_steps, down_steps, _ = schedule.state(budgets[index], slots, neuron_pages, down_pages)
        return sample.certify(sample.sets(neuron_steps, down_steps, seed=None), int(settings["near_rows"]), full, int(settings["chunk_rows"]), minimum)

    low, high = 0, len(budgets) - 1
    while low < high:
        middle = (low + high) // 2
        if certify_at(middle, full=False)["near_decided"]:
            high = middle
        else:
            low = middle + 1
    certificate = None
    for index in range(low, len(budgets)):
        result = certify_at(index, full=True)
        if result.get("certified"):
            neuron_steps, down_steps, _ = schedule.state(budgets[index], slots, neuron_pages, down_pages)
            certificate = {"index": index, "f_raw": checkpoints[index]["f_raw"], "f_best_independent": checkpoints[index]["f_best_independent"],
                           "f_representation": checkpoints[index]["f_representation"], **result,
                           "physical": physical(pages, neuron_steps, down_steps)}
            break
    full_state = schedule.state(total, slots, neuron_pages, down_pages)
    full_physical = physical(pages, full_state[0], full_state[1])
    # Poisoning: the same state from views with different unread bits gives the same sets and minimum.
    probe_index = len(budgets) // 3
    neuron_steps, down_steps, _ = schedule.state(budgets[probe_index], slots, neuron_pages, down_pages)
    a = minimum(sample.sets(neuron_steps, down_steps, seed=101), sample.x, sample.base, sample.delta(token, rows)).value
    b = minimum(sample.sets(neuron_steps, down_steps, seed=None), sample.x, sample.base, sample.delta(token, rows)).value
    return {
        "schedule": name, "exact_sets": exact_sets, "metadata_bytes": metadata, "total_representation_bytes": total,
        "checkpoints": checkpoints, "certificate": certificate, "full_read_physical": full_physical,
        "last_flip_f_raw": last_flip, "last_flip_f_representation": last_flip_representation, "truth_consistency_vs_phase5a2": consistency, "poisoning_invariant": bool(torch.equal(a, b)),
        "violations": {"minimum_above_truth": sum(c["rows_above_truth"] for c in checkpoints),
                       "witness_outside": sum(1 for c in checkpoints if "witness" in c and not c["witness"]["inside"]),
                       "witness_mismatch": sum(1 for c in checkpoints if "witness" in c and abs(c["witness"]["value"] - c["witness"]["minimum"]) > 1e-6 * (1 + abs(c["witness"]["minimum"])))},
    }


def toy_check(patterns: dict[str, np.ndarray], x: torch.Tensor, seed: int) -> dict:
    """A toy cut from real weights: 2 neurons, 3 hidden inputs, prefix 10: the closed form against enumeration."""
    import itertools

    from expert_deltas.progressive import SILU_ARGMIN, activation_range

    rng = np.random.default_rng(seed)
    neurons = sorted(rng.choice(patterns["gate"].shape[0], 2, replace=False).tolist())
    inputs = sorted(rng.choice(patterns["gate"].shape[1], 3, replace=False).tolist())
    gate = torch.from_numpy(patterns["gate"][np.ix_(neurons, inputs)].astype(np.int64))
    up = torch.from_numpy(patterns["up"][np.ix_(neurons, inputs)].astype(np.int64))
    down = torch.from_numpy(patterns["down"][np.ix_(inputs, neurons)].astype(np.int64))
    xs = x.cpu()[inputs].to(torch.float64)
    base = torch.zeros(3, dtype=torch.float64)
    delta = torch.from_numpy(rng.normal(0, 1, (3, 3)))
    out = []
    for t in (9, 10, 12):
        prefix = torch.full((2,), t)
        dprefix = torch.full((3,), t)
        s = ExpertSet(box(read_view(gate, prefix), prefix), box(read_view(up, prefix), prefix), box(read_view(down, dprefix), dprefix), 1.0)
        found = exact_minimum([s], xs, base, delta).value
        act = activation_range(s.gate, s.up, xs)
        options = []
        for i in range(2):
            gs = [float(act.g[0][i]), float(act.g[1][i])] + ([SILU_ARGMIN] if float(act.g[0][i]) <= SILU_ARGMIN <= float(act.g[1][i]) else [])
            options.append([float(silu(torch.tensor(g, dtype=torch.float64))) * u for g in gs for u in (float(act.u[0][i]), float(act.u[1][i]))])
        best = torch.full((3,), math.inf, dtype=torch.float64)
        for vertex in itertools.product((0, 1), repeat=6):
            D = torch.where(torch.tensor(vertex).view(3, 2).bool(), s.down.upper, s.down.lower)
            for a in itertools.product(*options):
                best = torch.minimum(best, delta @ (D @ torch.tensor(a, dtype=torch.float64)))
        out.append({"prefix": t, "closed_form": found.tolist(), "enumeration": best.tolist(),
                    "equal": bool(torch.allclose(found, best, rtol=1e-12, atol=1e-12))})
    return {"neurons": neurons, "inputs": inputs, "cases": out, "all_equal": all(c["equal"] for c in out)}


def anisotropy(capture: dict, lm: torch.Tensor, gain: torch.Tensor, calibration: list[int], evaluation: list[int], budgets: list[float],
               widths: torch.Tensor, device) -> dict:
    """How much of x (and of Δ) lies outside a calibration subspace of k directions, k from the metadata budgets (4k bytes
    per gate or up row of 4,096 bytes: k = budget × 1024): energy, and the box-relevant mass Σ_c h_c·|x_c| (h: mean
    half-width of a column's weights at t = 9)."""
    prompts = capture["prompt_id"].tolist()
    cal = [n for n, p in enumerate(prompts) if p in calibration]
    x_cal = capture["x"][cal].to(device, torch.float64)
    x_eval = capture["x"][evaluation].to(device, torch.float64)
    _, _, vh = torch.linalg.svd(x_cal, full_matrices=False)
    out = {"calibration_samples": len(cal), "x": {}, "delta": {}}
    for budget in budgets:
        k = max(1, int(round(budget * 1024)))
        U = vh[:k].t()
        resid = x_eval - (x_eval @ U) @ U.t()
        energy = float(((resid**2).sum(dim=1) / (x_eval**2).sum(dim=1)).mean())
        mass = float(((resid.abs() @ widths) / (x_eval.abs() @ widths)).mean())
        out["x"][str(budget)] = {"k": k, "energy_outside": energy, "box_mass_outside": mass}
    # Δ directions: each calibration sample's reference token against its 64 nearest rows by the captured h's logits.
    deltas = []
    for n in cal[:: max(1, len(cal) // 64)]:
        h = capture["h"][n].to(device)
        logits = torch.cat([lm[i : i + 16384] @ h for i in range(0, lm.shape[0], 16384)]).float()
        order = torch.argsort(logits, descending=True)
        w, rows = order[0], order[1:65]
        deltas.append((lm[w].to(torch.float64)[None, :] - lm[rows].to(torch.float64)) * gain[None, :])
    d_cal = torch.cat(deltas)
    _, _, dh = torch.linalg.svd(d_cal, full_matrices=False)
    d_eval = []
    for n in evaluation:
        h = capture["h"][n].to(device)
        logits = torch.cat([lm[i : i + 16384] @ h for i in range(0, lm.shape[0], 16384)]).float()
        order = torch.argsort(logits, descending=True)
        d_eval.append((lm[order[0]].to(torch.float64)[None, :] - lm[order[1:65]].to(torch.float64)) * gain[None, :])
    d_eval = torch.cat(d_eval)
    for budget in budgets:
        k = max(1, int(round(budget * 704)))  # a down row's 2,816 bytes: 4k bytes per row
        V = dh[:k].t()
        resid = d_eval - (d_eval @ V) @ V.t()
        out["delta"][str(budget)] = {"k": k, "energy_outside": float(((resid**2).sum(dim=1) / (d_eval**2).sum(dim=1)).mean()),
                                     "l1_outside": float((resid.abs().sum(dim=1) / d_eval.abs().sum(dim=1)).mean())}
    return out
