"""Phase 6B3-B probe: the host cost of issuing a pinned host-to-device copy, PyTorch's way (as PageStreamer._copy does
it: two timing events, a stream switch, the copy, the slot's event) against the CUDA runtime called directly (ctypes
cudaMemcpyAsync on the same stream), for the copies of a decode token (sizes like the profiles': ~150 copies of a few
MB to 32 MiB, out of 4 pinned slots). What a native copy-issuing component could save is the difference.

    .venv/Scripts/python.exe experiments/phase6b/copyissue/probe.py > experiments/phase6b/copyissue/probe.json
"""

from __future__ import annotations

import ctypes
import json
import statistics
import time
from pathlib import Path

import torch

SLOT_BYTES = 32 << 20
SLOTS = 4
COPIES = 148
REPEATS = 12


def cudart():
    lib = ctypes.CDLL(str(Path(torch.__file__).parent / "lib" / "cudart64_13.dll"))
    lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    lib.cudaMemcpyAsync.restype = ctypes.c_int
    return lib


def main() -> None:
    device = torch.device("cuda", 0)
    generator = torch.Generator().manual_seed(0)
    # Sizes: the profiles' mean copy is ~12 MB (1.8 GB of encoded rows in 148 copies); spread from 2 MB to a full slot.
    sizes = [int(s) for s in torch.randint(2 << 20, SLOT_BYTES, (COPIES,), generator=generator)]
    slots = [torch.empty(SLOT_BYTES, dtype=torch.uint8).pin_memory() for _ in range(SLOTS)]
    for slot in slots:
        slot.fill_(7)
    dest = torch.empty(sum(sizes), dtype=torch.uint8, device=device)
    copy_stream = torch.cuda.Stream(device)
    free = [torch.cuda.Event() for _ in range(SLOTS)]
    lib = cudart()

    def torch_streamer(offset, size, k):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(copy_stream):
            start.record()
            dest[offset : offset + size].copy_(slots[k][:size], non_blocking=True)
            end.record()
            free[k].record()

    def torch_plain(offset, size, k):
        with torch.cuda.stream(copy_stream):
            dest[offset : offset + size].copy_(slots[k][:size], non_blocking=True)
            free[k].record()

    handle = ctypes.c_void_p(copy_stream.cuda_stream)
    base = dest.data_ptr()
    sources = [s.data_ptr() for s in slots]

    def runtime(offset, size, k):
        status = lib.cudaMemcpyAsync(ctypes.c_void_p(base + offset), ctypes.c_void_p(sources[k]), size, 1, handle)
        if status:
            raise RuntimeError(f"cudaMemcpyAsync: {status}")
        free[k].record(copy_stream)

    out = {"copies": COPIES, "bytes": sum(sizes), "slots": SLOTS, "slot_bytes": SLOT_BYTES, "repeats": REPEATS, "paths": {}}
    for name, issue in (("torch-streamer", torch_streamer), ("torch-plain", torch_plain), ("cudart", runtime)):
        issue_ms, wait_ms, wall_ms = [], [], []
        for repeat in range(REPEATS + 1):
            torch.cuda.synchronize()
            issued = waited = 0.0
            started = time.perf_counter()
            offset = 0
            for index, size in enumerate(sizes):
                k = index % SLOTS
                if index >= SLOTS:  # the slot's previous copy must be done before it is refilled (the streamer's rule)
                    t = time.perf_counter()
                    free[k].synchronize()
                    waited += time.perf_counter() - t
                t = time.perf_counter()
                issue(offset, size, k)
                issued += time.perf_counter() - t
                offset += size
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
            if repeat:  # the first is a warm-up
                issue_ms.append(issued * 1e3)
                wait_ms.append(waited * 1e3)
                wall_ms.append(wall * 1e3)
        out["paths"][name] = {
            "issue_ms": statistics.mean(issue_ms), "issue_us_per_copy": statistics.mean(issue_ms) * 1e3 / COPIES,
            "slot_wait_ms": statistics.mean(wait_ms), "wall_ms": statistics.mean(wall_ms), "wall_ms_stdev": statistics.stdev(wall_ms),
            "gb_s": sum(sizes) / statistics.mean(wall_ms) / 1e6,
        }
    if not torch.equal(dest[: sizes[0]].cpu(), torch.full((sizes[0],), 7, dtype=torch.uint8)):
        raise RuntimeError("copies wrong")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
