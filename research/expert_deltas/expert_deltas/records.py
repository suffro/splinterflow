"""Run metadata for Phase 5C's experiments: environment, source trees, codec versions, digests."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import lz4
import zstandard

from awpmi.tracing import canonical_digest, environment_metadata, source_tree_sha256

REPO_ROOT = Path(__file__).resolve().parents[3]
RESEARCH = REPO_ROOT / "research" / "expert_deltas"
EXCLUDED_FIELDS = ("timings", "timings_ms", "system", "throughput", "peak_device_bytes")


def research_tree_sha256() -> str:
    """Weightsift's source tree (src, benchmarks, configs) plus this research directory (its code, not its tests)."""
    digest = hashlib.sha256(source_tree_sha256(REPO_ROOT).encode("ascii"))
    for path in sorted(RESEARCH.rglob("*.py")):
        if "__pycache__" in path.parts or "tests" in path.relative_to(RESEARCH).parts:
            continue
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def environment(model: dict, numerics: dict | None = None) -> dict:
    env = environment_metadata(REPO_ROOT, model, numerics or {})
    env["codecs"] = {"zstandard": zstandard.__version__, "libzstd": ".".join(map(str, zstandard.ZSTD_VERSION)),
                     "zstandard_backend": zstandard.backend, "lz4": lz4.__version__, "liblz4": lz4.library_version_string()}
    env["research_tree_sha256"] = research_tree_sha256()
    env["python_hash_seed"] = os.environ.get("PYTHONHASHSEED")
    env["cpu_count"] = os.cpu_count()
    env["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return env


def strip(record, excluded=EXCLUDED_FIELDS):
    """A record without its non-deterministic fields, recursively."""
    if isinstance(record, dict):
        return {k: strip(v, excluded) for k, v in record.items() if k not in excluded}
    if isinstance(record, list):
        return [strip(v, excluded) for v in record]
    return record


def digest(record) -> str:
    return canonical_digest([strip(record)])


def write_json(path: Path, record) -> None:
    path.write_text(json.dumps(record, indent=1, sort_keys=False), encoding="utf-8")
