"""Raw structured traces and the environment metadata every benchmark must save."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import torch

TRACKED_PACKAGES = ("torch", "transformers", "safetensors", "numpy", "huggingface-hub", "tokenizers", "pyarrow")


def tensor_digest(*tensors: torch.Tensor | None) -> str:
    """sha256 over the raw bytes of the given tensors, in order (None entries are skipped)."""
    digest = hashlib.sha256()
    for tensor in tensors:
        if tensor is not None:
            data = tensor.detach().cpu().contiguous()
            if data.dtype == torch.bfloat16:  # numpy has no bfloat16: hash the same bytes as int16
                data = data.view(torch.int16)
            digest.update(data.numpy().tobytes())
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


SOURCE_TREES = ("src", "benchmarks", "configs")
NATIVE_TREE = "native"


def _files_sha256(repo_root: Path, files: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(repo_root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def source_tree_sha256(repo_root: Path) -> str:
    """Hash of the code and configs that produced a run, independent of git state."""
    return _files_sha256(
        repo_root,
        (
            path
            for tree in SOURCE_TREES
            for path in (repo_root / tree).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ),
    )


def native_tree_sha256(repo_root: Path) -> str | None:
    """Hash of the native core's sources (`native/` without its build directory `target`), by the same rule.

    Kept apart from `source_tree_sha256`, whose rule other tools repeat; None without a `native/` directory.
    """
    root = repo_root / NATIVE_TREE
    if not root.is_dir():
        return None
    return _files_sha256(
        repo_root, (path for path in root.rglob("*") if path.is_file() and path.relative_to(root).parts[0] != "target")
    )


def _version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata(repo_root: Path, model: Mapping[str, Any], numerics_flags: Mapping[str, Any]) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    lock = repo_root / "uv.lock"
    status = _git(repo_root, "status", "--porcelain")
    return {
        "model": dict(model),
        "python": sys.version,
        "os": platform.platform(),
        "packages": {name: importlib_metadata.version(name) for name in TRACKED_PACKAGES},
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "numerics": dict(numerics_flags),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "source_tree_sha256": source_tree_sha256(repo_root),
        "native_tree_sha256": native_tree_sha256(repo_root),
        "native_extension": _version("weightsift-native"),  # None: the extension is not installed
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
    }


def _is_gzip(path: str | Path) -> bool:
    return str(path).endswith(".gz")


class JsonlWriter:
    """One JSON object per line; a `.gz` path is gzip-compressed reproducibly (no name, mtime 0)."""

    def __init__(self, path: str | Path) -> None:
        if _is_gzip(path):
            self._raw = open(path, "wb")
            compressed = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0)
            self._handle = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
        else:
            self._raw = None
            self._handle = open(path, "w", encoding="utf-8", newline="\n")

    def write(self, record: Mapping[str, Any]) -> None:
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        if self._raw is None:
            # Per-line gzip flushes would bloat the stream; plain files stay readable mid-run.
            self._handle.flush()

    def close(self) -> None:
        self._handle.close()
        if self._raw is not None:
            # GzipFile does not close a file object it was given.
            self._raw.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    opener = gzip.open if _is_gzip(path) else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class StageTimer:
    """Wall-clock milliseconds per named stage of one run.

    With `synchronize` on a CUDA device, every mark first waits for the queued
    device work, so a stage is charged for its own kernels. Repeated stage names
    accumulate.
    """

    def __init__(self, device: torch.device, synchronize: bool = True) -> None:
        self._synchronize = synchronize and device.type == "cuda"
        self.stages: dict[str, float] = {}
        self._last = 0.0

    def _now(self) -> float:
        if self._synchronize:
            torch.cuda.synchronize()
        return time.perf_counter()

    def start(self) -> None:
        self.stages = {}
        self._last = self._now()

    def mark(self, stage: str) -> None:
        now = self._now()
        self.stages[stage] = self.stages.get(stage, 0.0) + (now - self._last) * 1e3
        self._last = now


class NullTimer:
    """A StageTimer that measures nothing and never synchronizes."""

    stages: dict[str, float] = {}

    def start(self) -> None:
        pass

    def mark(self, stage: str) -> None:
        pass


def canonical_digest(records: Iterable[Mapping[str, Any]], exclude_keys: Iterable[str] = ()) -> str:
    """sha256 over records with non-deterministic fields (e.g. timings) removed."""
    excluded = set(exclude_keys)
    digest = hashlib.sha256()
    for record in records:
        kept = {key: value for key, value in record.items() if key not in excluded}
        digest.update(json.dumps(kept, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
