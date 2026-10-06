"""The artifact `export.py` writes, as the verifier reads it (Phase 5A2).

The verifier sees the experiment only through this artifact: safetensors files and a JSON manifest whose sha256 are
checked on opening. Tensors are loaded by key, never as a whole file, so a sample costs only its own rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

FORMAT = "phase5a2-crown-artifact/1"
UNKNOWN, EXACT = -1, 99  # row states, as Phase 5A's oracle writes them (a level index otherwise)
SOURCE_TREES = ("src", "benchmarks", "configs")  # awpmi.tracing's source tree
RESEARCH_TREE = Path("research") / "crown_expert_oracle"
RESEARCH_EXCLUDED = (".venv", "__pycache__", ".pytest_cache")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(repo_root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(repo_root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def source_tree_sha256(repo_root: Path) -> str:
    """awpmi.tracing.source_tree_sha256's rule (src, benchmarks, configs), repeated here: the verifier imports no awpmi."""
    files = [p for tree in SOURCE_TREES for p in (repo_root / tree).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    return _tree_sha256(repo_root, files)


def research_tree_sha256(repo_root: Path) -> str:
    """export.py's rule: research/crown_expert_oracle without its environment and caches."""
    root = repo_root / RESEARCH_TREE
    files = [p for p in root.rglob("*") if p.is_file() and not any(part in RESEARCH_EXCLUDED for part in p.relative_to(root).parts)]
    return _tree_sha256(repo_root, files)


@dataclass(frozen=True)
class MatrixData:
    """One routed expert matrix as exported: the BF16 rows, the q6 and q4 levels, their remainder norms, its own norms."""

    truth: torch.Tensor  # BF16 [R, C]
    codes: tuple[torch.Tensor, ...]  # int8 [R, C] per level
    scales: tuple[torch.Tensor, ...]  # float32 [R] per level
    remainder_linf: tuple[torch.Tensor, ...]  # float32 [R] per level: ≥ max |W − Σ_{l' ≤ l} L_l'| per row (resident)
    remainder_l2: tuple[torch.Tensor, ...]
    own_linf: torch.Tensor  # float32 [R]: ≥ max |W| per row (resident)
    own_l2: torch.Tensor


class Artifact:
    def __init__(self, directory: Path, verify: bool = True) -> None:
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"unexpected artifact format {self.manifest.get('format')!r}")
        self.verified = {}
        if verify:
            for name, expected in self.manifest["files"].items():
                actual = sha256_file(self.directory / name)
                if actual != expected:
                    raise ValueError(f"{name}: sha256 {actual} differs from the manifest's {expected}")
                self.verified[name] = actual
        self.constants = self.manifest["constants"]
        self.shapes = self.manifest["shapes"]

    @property
    def samples(self) -> list[dict]:
        return self.manifest["samples"]

    def _weights(self):
        return safe_open(str(self.directory / "weights.safetensors"), framework="pt", device="cpu")

    def tensor(self, key: str, device="cpu") -> torch.Tensor:
        with self._weights() as handle:
            return handle.get_tensor(key).to(device)

    def matrix(self, expert: int, name: str, device="cpu") -> MatrixData:
        prefix = f"expert.{expert}.{name}"
        with self._weights() as handle:
            keys = set(handle.keys())
            levels = sorted({int(k.split(".")[3][1:]) for k in keys if k.startswith(prefix + ".q") and k.endswith(".codes")})

            def get(key):
                return handle.get_tensor(key).to(device)

            return MatrixData(
                truth=get(f"{prefix}.truth"),
                codes=tuple(get(f"{prefix}.q{level}.codes") for level in levels),
                scales=tuple(get(f"{prefix}.q{level}.scales") for level in levels),
                remainder_linf=tuple(get(f"{prefix}.r{level}.linf") for level in levels),
                remainder_l2=tuple(get(f"{prefix}.r{level}.l2") for level in levels),
                own_linf=get(f"{prefix}.norm.linf"),
                own_l2=get(f"{prefix}.norm.l2"),
            )

    def sample_tensors(self, index: int, device="cpu", prefix: str | None = None) -> dict[str, torch.Tensor]:
        """A sample's tensors (or those under `prefix`)."""
        with safe_open(str(self.directory / f"sample.{index}.safetensors"), framework="pt", device="cpu") as handle:
            return {k: handle.get_tensor(k).to(device) for k in handle.keys() if prefix is None or k.startswith(prefix)}
