"""Phase 5C research code: exact shared expert bases and progressive expert deltas (research, not a runtime).

Runs in the Weightsift environment (it reads the checkpoint through `awpmi`'s safetensors layout and reuses Phase 5A's
oracle inputs); the codecs are zstandard and lz4 as published (`pyproject.toml`, dependency group `research`).
"""
