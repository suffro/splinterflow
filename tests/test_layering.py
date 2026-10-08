"""Layering (decisions 0006-0008): the storage core knows no model and no certificate; certification knows no storage;
the streaming reference shares nothing with Weightsift's own expert path."""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "awpmi"
CORE = ("storage", "streaming", "materialization")
# Names that would tie the core to one model, one tensor layout or one routing implementation.
MODEL_WORDS = re.compile(
    r"smollm|granite|llama|mixtral|qwen|deepseek|olmoe|moonlight|kimi|gpt.?oss|lm_head|embed_tokens|gate_up_proj|down_proj|"
    r"gate_proj|up_proj|block_sparse_moe|router|top_k_index|num_experts|shared_expert|e_score|safetensors\.index",
    re.IGNORECASE,
)
CORE_FORBIDDEN_IMPORTS = re.compile(
    r"^\s*(from|import)\s+(transformers|awpmi\.models|awpmi\.bounds|awpmi\.certificate|awpmi\.refinement_head|"
    r"awpmi\.suffix_runtime|awpmi\.stores|awpmi\.oracle|awpmi\.decomposition)\b",
    re.MULTILINE,
)
CERTIFICATION_FORBIDDEN_IMPORTS = re.compile(
    r"^\s*(from|import)\s+awpmi\.(storage|streaming|materialization)\b", re.MULTILINE
)


def core_files():
    for package in CORE:
        yield from sorted((SRC / package).glob("*.py"))


def test_the_storage_core_knows_no_model():
    files = list(core_files())
    assert len(files) >= 8
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not MODEL_WORDS.search(text), f"{path.name}: {MODEL_WORDS.search(text).group(0)!r}"
        assert not CORE_FORBIDDEN_IMPORTS.search(text), f"{path.name} imports a layer above it"


NATIVE = Path(__file__).resolve().parents[1] / "native"


def test_the_native_core_knows_no_model():
    """Phase 6A (decision 0012): the Rust core and its binding are model-agnostic like the Python storage core."""
    files = sorted(path for path in NATIVE.rglob("*.rs") if "target" not in path.relative_to(NATIVE).parts)
    assert len(files) >= 8
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not MODEL_WORDS.search(text), f"{path.name}: {MODEL_WORDS.search(text).group(0)!r}"


def test_only_the_storage_core_imports_the_native_extension():
    """The extension is reached through `awpmi.storage.native` only, so the Python backend never depends on it."""
    for path in sorted(SRC.rglob("*.py")):
        if path == SRC / "storage" / "native.py":
            continue
        assert "weightsift_native" not in path.read_text(encoding="utf-8"), path.name


def test_certification_does_not_depend_on_storage():
    for path in [*sorted((SRC / "bounds").glob("*.py")), SRC / "certificate.py", SRC / "refinement_head.py"]:
        assert not CERTIFICATION_FORBIDDEN_IMPORTS.search(path.read_text(encoding="utf-8")), path.name


# The independent reference (decision 0008) imports nothing of Weightsift's path: no storage, transfer, materialization,
# expert adapter (compact or chunked calls) or checkpoint index of Weightsift's.
REFERENCE_FORBIDDEN_IMPORTS = re.compile(
    r"^\s*(from|import)\s+awpmi\.(storage|streaming|materialization|models|stores)\b|^\s*from\s+awpmi\s+import", re.MULTILINE
)


ORACLE_IMPORT = re.compile(r"^\s*(from|import)\s+awpmi\.oracle\b|^\s*from\s+awpmi\s+import\s+.*\boracle\b", re.MULTILINE)
ORACLE_FORBIDDEN_IMPORTS = re.compile(
    r"^\s*(from|import)\s+awpmi\.(storage|streaming|materialization|stores|models|refinement_head|suffix_runtime|streaming_reference)\b",
    re.MULTILINE,
)


def test_no_runtime_path_uses_the_oracles():
    """Phases 1B and 5A (decisions 0003, 0009): the oracles are diagnostic simulators; nothing outside `awpmi.oracle`
    imports them, and they read no storage and run no runtime path (the expert oracle holds every weight itself)."""
    for path in sorted(SRC.rglob("*.py")):
        if "oracle" in path.relative_to(SRC).parts:
            assert not ORACLE_FORBIDDEN_IMPORTS.search(path.read_text(encoding="utf-8")), path.name
        else:
            assert not ORACLE_IMPORT.search(path.read_text(encoding="utf-8")), path.name


def test_the_streaming_reference_shares_nothing_with_weightsifts_path():
    text = (SRC / "streaming_reference.py").read_text(encoding="utf-8")
    assert not REFERENCE_FORBIDDEN_IMPORTS.search(text)
    assert not re.search(r"\bawpmi\b", text.split('"""', 2)[2])  # beyond its docstring it names no awpmi module at all
