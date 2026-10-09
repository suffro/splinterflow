"""Phase 6B3-B, end to end: `moonlight_profile.py` with the streamer's copies issued by the CUDA runtime directly (ctypes
`cudaMemcpyAsync` on the copy stream, the slot's event recorded after it, no per-copy timing events) instead of
PyTorch's `copy_` between two timing events. Same arguments as the profile; the device copy time is then not measured
(`h2d_device_ms` reads 0), everything else is.

    .venv/Scripts/python.exe experiments/phase6b/copyissue/profile_cudart.py --run <run> --config <config> \
        --configuration <name> --prompts 7 0 3 5 --warm --output <profile.json>
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import moonlight_profile  # noqa: E402  (configures the numerics first)
import torch  # noqa: E402

import awpmi.streaming.streamer as streamer_module  # noqa: E402

_lib = ctypes.CDLL(str(Path(torch.__file__).parent / "lib" / "cudart64_13.dll"))
_lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
_lib.cudaMemcpyAsync.restype = ctypes.c_int
_original = streamer_module.PageStreamer._copy


def _copy(self, dest, source, slot, asynchronous):
    if not (asynchronous and dest.is_cuda and not source.is_cuda and source.is_pinned() and dest.is_contiguous() and source.is_contiguous()):
        return _original(self, dest, source, slot, asynchronous)
    nbytes = source.numel() * source.element_size()
    status = _lib.cudaMemcpyAsync(ctypes.c_void_p(dest.data_ptr()), ctypes.c_void_p(source.data_ptr()), nbytes, 1, ctypes.c_void_p(self.copy_stream.cuda_stream))
    if status:
        raise RuntimeError(f"cudaMemcpyAsync failed: {status}")
    if slot is not None:
        slot.free.record(self.copy_stream)
        slot.used = True
    self.stats.h2d_bytes += source.numel()
    self.stats.h2d_copies += 1


streamer_module.PageStreamer._copy = _copy

if __name__ == "__main__":
    raise SystemExit(moonlight_profile.main())
