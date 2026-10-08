"""The verifier environment's side of the boundary (decision 0010): it has no Weightsift package, and reads the
artifacts Weightsift writes bit for bit (the root's tests/test_crown_boundary.py writes the same fixture)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_the_verifier_environment_has_no_weightsift():
    assert importlib.util.find_spec("awpmi") is None
    assert importlib.util.find_spec("auto_LiRPA") is not None


VERIFIER_SCRIPTS = ("probe.py", "probe_l2.py", "run.py", "report.py", "full_crown_cost.py")  # export.py runs in the Weightsift environment, by design


def test_the_verifier_imports_no_weightsift_module():
    root = Path(__file__).resolve().parents[1]
    paths = [*(root / name for name in VERIFIER_SCRIPTS if (root / name).exists()), *(root / "crown_oracle").glob("*.py")]
    assert len(paths) >= 7
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "import awpmi" not in text and "from awpmi" not in text, path.name


def roundtrip_tensors() -> dict[str, torch.Tensor]:
    """The root's tests/test_crown_boundary.roundtrip_tensors, repeated (exact arithmetic only)."""
    steps = torch.arange(64, dtype=torch.float64)
    values = torch.ldexp((steps * 0.6180339887498949) % 1.0 - 0.5, (steps - 32).to(torch.int64) * 30)
    values[:6] = torch.tensor([0.0, -0.0, 2.0**-1074, -(2.0**-1022), 1.0 / 3.0, -(2.0**1000)], dtype=torch.float64)
    return {
        "float64": values,
        "float32": values.clamp(-3e38, 3e38).to(torch.float32),
        "bfloat16": values.clamp(-3e38, 3e38).to(torch.bfloat16),
        "int8": torch.arange(-128, 128, dtype=torch.int8),
        "int64": (torch.arange(-5, 5, dtype=torch.int64) * 2**40),
        "matrix_bfloat16": values.clamp(-1e5, 1e5).to(torch.bfloat16).reshape(8, 8).t().contiguous(),
    }


def test_artifacts_from_the_weightsift_environment_read_back_exactly():
    loaded = load_file(str(FIXTURES / "roundtrip.safetensors"))
    expected = json.loads((FIXTURES / "roundtrip.json").read_text(encoding="utf-8"))
    mine = roundtrip_tensors()
    assert set(loaded) == set(expected) == set(mine)
    for name, tensor in loaded.items():
        digest = hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        assert digest == expected[name], name
        assert tensor.dtype == mine[name].dtype and torch.equal(tensor.contiguous().view(torch.uint8), mine[name].contiguous().view(torch.uint8)), name
