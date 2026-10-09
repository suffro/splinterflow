"""Command line (roadmap §3.2): `weightsift pack`.

    uv run weightsift pack lm-head [--config configs/phase3-storage.yaml] [--output DIR]
    uv run weightsift pack experts [--config configs/phase3-moe.yaml] [--output DIR]
    uv run weightsift pack expert-index [--config configs/phase4a-olmoe.yaml] [--output DIR]
    uv run weightsift pack encoded-experts [--config configs/phase6b-gpu.yaml] [--output DIR]

`wsift` is an alias for `weightsift` and accepts the same commands and options.

lm-head       the refinement pack of the configured decomposition of a model's LM head: level
              records and remainder norms in an AWPMI safetensors file, exact rows referring to
              the published checkpoint tensor (checked byte for byte against the model's LM-head
              weight).
experts       the expert pack of a mixture-of-experts model: every expert-sliced parameter,
              referring to the published checkpoint wherever it holds the same bytes.
expert-index  the expert index of a checkpoint that stores each expert as separate tensors
              (decision 0007): one composed segment per expert-sliced parameter, its rows the byte
              ranges of the published tensors. Built from the safetensors headers and the model's
              skeleton on `meta` (through the configured adapter's layout); no weight byte is read
              or written. Source files carry the sha256 the Hub declares.
encoded-experts  an encoded pack of an expert index (Phase 6B, decision 0013): every row compressed on the GPU by
              nvCOMP in independent chunks, checked to decode to the source bytes, padded to its segment's slot; the
              source files are first re-hashed against the publisher's sha256 (direct reads).

Each writes `manifest.json` (files with sizes and sha256, segments with their location and,
when computed, sha256; source model and revision; packing configuration) and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def pack_lm_head(config_path: Path, output: Path | None) -> dict:
    import torch

    from awpmi.config import load_config
    from awpmi.decomposition import RefinementDecomposition
    from awpmi.models.smollm2 import ModelSpec, lm_head_weight, load_model, resolve_dtype
    from awpmi.storage.pack import SourceFile
    from awpmi.stores.refinement import write_refinement_pack

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config, _ = load_config(REPO_ROOT / config["source_run"] / "config.yaml")
    device = torch.device(model_config.model.device if torch.cuda.is_available() else "cpu")
    spec = ModelSpec(model_config.model.repository, model_config.model.revision, resolve_dtype(model_config.model.dtype, device), device)
    model, _ = load_model(spec)
    decomposition = RefinementDecomposition.build(lm_head_weight(model), config["decomposition"])
    source = SourceFile(spec.repository, spec.revision, config["pack"]["source_file"])
    pack = write_refinement_pack(
        decomposition,
        output or REPO_ROOT / config["pack"]["directory"],
        source=(source, config["pack"]["source_tensor"]),
        packing={"tool": "weightsift pack lm-head", "config": str(config_path)},
    )
    return pack.manifest


def pack_experts(config_path: Path, output: Path | None) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    from awpmi.models.moe import write_expert_pack
    from awpmi.storage.pack import SourceFile

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(config["repository"], revision=config["revision"], dtype=dtype)
    source = SourceFile(config["repository"], config["revision"], config["source_file"])
    directory = output or REPO_ROOT / yaml.safe_load(config_path.read_text(encoding="utf-8"))["pack"]["directory"]
    pack = write_expert_pack(model, directory, sources={"checkpoint": (source, None)}, packing={"tool": "weightsift pack experts"})
    return pack.manifest


def model_skeleton(model_config: dict):
    """The configured model on `meta` (no weights) and its adapter module, if any."""
    import importlib

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[model_config["dtype"]]
    config = AutoConfig.from_pretrained(model_config["repository"], revision=model_config["revision"])
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    adapter = importlib.import_module(f"awpmi.models.{model_config['adapter']}") if model_config.get("adapter") else None
    return model, adapter


def pack_expert_index(config_path: Path, output: Path | None) -> dict:
    from awpmi.models import checkpoint

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = raw["model"]
    model, adapter = model_skeleton(model_config)
    sources = adapter.expert_sources(model) if adapter is not None and hasattr(adapter, "expert_sources") else None
    files = checkpoint.checkpoint_sources(model_config["repository"], model_config["revision"])
    missing = [key for key, entry in files.items() if entry.sha256 is None]
    if missing:
        raise RuntimeError(f"the Hub declares no sha256 for {missing}: cannot record their identity")
    directory = output or REPO_ROOT / raw["index"]["directory"]
    pack = checkpoint.write_expert_index(
        model, directory, files, packing={"tool": "weightsift pack expert-index", "config": config_path.name},
        metadata={"model": {k: model_config[k] for k in ("repository", "revision", "dtype")}, "adapter": model_config.get("adapter")},
        sources=sources,
    )
    return pack.manifest


def pack_encoded_experts(config_path: Path, output: Path | None) -> dict:
    from awpmi.storage.native import NATIVE_AVAILABLE
    from awpmi.storage.pack import open_pack
    from awpmi.streaming.codec import write_encoded_pack

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = raw["encoded"]
    source = open_pack(REPO_ROOT / raw["index"]["directory"], verify="files")
    encoded = write_encoded_pack(
        source, output or REPO_ROOT / settings["directory"], codec=settings["codec"], options=settings["options"],
        chunk_bytes=int(settings["chunk_bytes"]), store_options={"backend": "native" if NATIVE_AVAILABLE else "python", "direct": True},
        progress=lambda message: print(message, flush=True),
    )
    return encoded.pack.manifest


def main(argv: list[str] | None = None) -> int:
    prog = Path(sys.argv[0]).stem
    parser = argparse.ArgumentParser(
        prog=prog if prog in {"weightsift", "wsift"} else "weightsift",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="write a pack and its manifest")
    pack.add_argument("kind", choices=["lm-head", "experts", "expert-index", "encoded-experts"])
    pack.add_argument("--config", default=None)
    pack.add_argument("--output", default=None, help="pack directory (default: the config's)")
    args = parser.parse_args(argv)
    default = {"lm-head": "phase3-storage.yaml", "experts": "phase3-moe.yaml", "expert-index": "phase4a-olmoe.yaml",
               "encoded-experts": "phase6b-gpu.yaml"}[args.kind]
    config = Path(args.config) if args.config else REPO_ROOT / "configs" / default
    output = Path(args.output) if args.output else None
    writer = {"lm-head": pack_lm_head, "experts": pack_experts, "expert-index": pack_expert_index, "encoded-experts": pack_encoded_experts}[args.kind]
    manifest = writer(config, output)
    summary = {
        "kind": manifest["kind"],
        "files": {key: {k: v for k, v in entry.items() if k != "sha256"} for key, entry in manifest["files"].items()},
        "segments": len(manifest["segments"]),
        "bytes": sum(entry["rows"] * entry["row_bytes"] for entry in manifest["segments"].values()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
