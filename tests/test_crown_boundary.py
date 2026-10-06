"""Phase 5A2 (decision 0010): auto_LiRPA lives in the isolated verifier environment (`research/crown_expert_oracle`) only.
Splinterflow neither depends on it nor imports the verifier; the two environments exchange serialized artifacts, which
survive the trip bit for bit (the verifier environment checks the same fixture: its tests/test_boundary.py)."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "awpmi"
FIXTURES = ROOT / "research" / "crown_expert_oracle" / "tests" / "fixtures"
VERIFIER_IMPORT = re.compile(r"^\s*(from|import)\s+(auto_LiRPA|crown_oracle)\b", re.MULTILINE)


def test_the_runtime_environment_has_no_auto_lirpa():
    assert importlib.util.find_spec("auto_LiRPA") is None
    assert "auto-lirpa" not in (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "auto-lirpa" not in (ROOT / "uv.lock").read_text(encoding="utf-8").lower()


def test_no_splinterflow_module_imports_the_verifier():
    paths = [*sorted(SRC.rglob("*.py")), *sorted((ROOT / "benchmarks").glob("*.py"))]
    assert len(paths) > 40
    for path in paths:
        assert not VERIFIER_IMPORT.search(path.read_text(encoding="utf-8")), path.name


def test_importing_every_awpmi_module_loads_no_verifier_module():
    code = (
        "import importlib, pkgutil, sys, awpmi\n"
        "for module in pkgutil.walk_packages(awpmi.__path__, 'awpmi.'):\n"
        "    importlib.import_module(module.name)\n"
        "print(sorted(n for n in sys.modules if n.split('.')[0] in ('auto_LiRPA', 'crown_oracle')))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def roundtrip_tensors() -> dict[str, torch.Tensor]:
    """Every dtype the artifact carries, with edge values, from exact arithmetic only (no generator: the verifier
    environment's torch builds the same tensors). The same function is in the verifier's tests/test_boundary.py."""
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


def tensor_sha256(tensor: torch.Tensor) -> str:
    import hashlib

    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def test_artifact_tensors_survive_serialization(tmp_path):
    tensors = roundtrip_tensors()
    save_file(tensors, str(tmp_path / "roundtrip.safetensors"))
    loaded = load_file(str(tmp_path / "roundtrip.safetensors"))
    assert set(loaded) == set(tensors)
    for name, tensor in tensors.items():
        assert loaded[name].dtype == tensor.dtype and loaded[name].shape == tensor.shape
        assert torch.equal(loaded[name].contiguous().view(torch.uint8), tensor.contiguous().view(torch.uint8)), name
    # The committed fixture is this writer's output, and its digests are what the verifier environment reads back.
    assert (FIXTURES / "roundtrip.safetensors").read_bytes() == (tmp_path / "roundtrip.safetensors").read_bytes()
    expected = json.loads((FIXTURES / "roundtrip.json").read_text(encoding="utf-8"))
    assert expected == {name: tensor_sha256(tensor) for name, tensor in tensors.items()}
