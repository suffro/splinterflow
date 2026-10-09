"""Phase 6B, end-to-end probes of two cheap changes before either is built: `moonlight_profile.py` with

  WS_NO_FILL=1   PyTorch's `torch.utils.deterministic.fill_uninitialized_memory` off (with deterministic algorithms on,
                 PyTorch fills every `torch.empty` with NaN: a debugging aid that changes no arithmetic, ~270 BF16 fill
                 kernels per decode token, the compact expert buffers and the staging rows among them)
  WS_CUDART=1    the streamer's copies issued by the CUDA runtime directly (copyissue/profile_cudart.py's `_copy`)

Same arguments as the profile.

    WS_NO_FILL=1 .venv/Scripts/python.exe experiments/phase6b/profile_variant.py --run <run> --config <config> ...
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "benchmarks"))

import moonlight_profile  # noqa: E402  (configures the numerics first)
import torch  # noqa: E402

if os.environ.get("WS_NO_FILL") == "1":
    torch.utils.deterministic.fill_uninitialized_memory = False
if os.environ.get("WS_CUDART") == "1":
    runpy.run_path(str(HERE / "copyissue" / "profile_cudart.py"), run_name="patch")  # patches PageStreamer._copy only

if __name__ == "__main__":
    raise SystemExit(moonlight_profile.main())
