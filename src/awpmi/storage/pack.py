"""Packs: weight files plus a manifest that says where every segment is and what it holds.

A pack is a directory with `manifest.json` and the safetensors files AWPMI wrote (roadmap
§3.2, §3.3). A segment either lives in a pack file or is a tensor of the published
checkpoint itself, located through the Hugging Face cache: when a weight's bytes are the
checkpoint's own (an LM head tied to the embedding, stacked expert tensors that the loader
only renames), the pack refers to them instead of copying them.

The manifest records, for every file, its size and sha256, and for every segment its
location and the sha256 of its bytes, plus what the pack was made from (`metadata`) and
how (`packing`). `open_pack` checks sizes, and by default re-hashes every segment, so a
pack that opens is the pack that was written: packed model = expected reference model.

Version 2 (decision 0007) adds composed segments: rows assembled from spans of the source
files, e.g. one expert from several published tensors. Such an index is written from
safetensors headers alone; nothing proportional to the model is read or written. A source
file's sha256 may then be the one its publisher declares (the Hugging Face LFS digest)
rather than one computed here, and segments may carry no digest: `verify="files"` re-hashes
the files themselves (with direct reads), `verify="size"` only checks sizes. A pack without
composed segments is still written as version 1, byte for byte as before.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from awpmi.storage.fileio import DIRECT_ALIGNMENT, PositionedFile, aligned_host_buffer
from awpmi.storage.layout import AnySegment, ComposedSegment, Segment, row_bytes_of, safetensors_segments, segment_from_json
from awpmi.storage.native import NativePageStore
from awpmi.storage.store import FileBackedPageStore, InMemoryPageStore
from awpmi.tracing import sha256_file

FORMAT = "awpmi-pack"
FORMAT_VERSION = 2
MANIFEST = "manifest.json"
_HASH_CHUNK = 64 << 20


@dataclass(frozen=True)
class SourceFile:
    """A file of a published checkpoint at a pinned revision, found in the local Hugging Face cache."""

    repository: str
    revision: str
    filename: str

    def resolve(self) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(self.repository, self.filename, revision=self.revision, local_files_only=True))

    def published_sha256(self) -> str | None:
        """The sha256 the Hub declares for this file at this revision (its LFS object id), or None."""
        from huggingface_hub import HfApi

        try:
            (info,) = HfApi().get_paths_info(self.repository, [self.filename], revision=self.revision)
        except Exception:  # offline, or not an LFS file
            return None
        lfs = getattr(info, "lfs", None)
        return None if lfs is None else lfs.sha256

    def to_json(self) -> dict[str, str]:
        return {"repository": self.repository, "revision": self.revision, "filename": self.filename}


def tensor_bytes_sha256(tensor: torch.Tensor) -> str:
    """sha256 of a tensor's raw bytes in row-major order (any dtype, CPU or device)."""
    digest = hashlib.sha256()
    data = row_bytes_of(tensor.contiguous()).reshape(-1)
    for start in range(0, data.numel(), _HASH_CHUNK):
        digest.update(data[start : start + _HASH_CHUNK].cpu().numpy().tobytes())
    return digest.hexdigest()


def segment_sha256(store: FileBackedPageStore, segment: str) -> str:
    """sha256 of a segment's bytes as stored, read in chunks of whole rows."""
    info = store.segment(segment)
    digest = hashlib.sha256()
    step = max(1, _HASH_CHUNK // info.row_bytes)
    for first in range(0, info.rows, step):
        rows = torch.arange(first, min(first + step, info.rows))
        digest.update(store.read_rows(segment, rows).numpy().tobytes())
    return digest.hexdigest()


def sha256_file_direct(path: str | Path) -> str:
    """sha256 of a whole file, read with direct I/O (so hashing leaves the OS page cache alone, decision 0006)."""
    digest = hashlib.sha256()
    buffer = aligned_host_buffer(_HASH_CHUNK, DIRECT_ALIGNMENT, pin=False)
    view = buffer.numpy()
    with PositionedFile(path, direct=True) as file:
        for offset in range(0, file.size, _HASH_CHUNK):
            count = file.read_into(offset, _HASH_CHUNK, buffer.data_ptr())
            digest.update(view[: min(count, file.size - offset)].tobytes())
    return digest.hexdigest()


@dataclass
class Pack:
    directory: Path
    manifest: dict[str, Any]
    files: dict[str, Path]
    segments: dict[str, AnySegment]

    @property
    def kind(self) -> str:
        return self.manifest["kind"]

    @property
    def metadata(self) -> dict[str, Any]:
        return self.manifest["metadata"]

    def store(self, backend: str = "python", **options) -> FileBackedPageStore | NativePageStore:
        """A file-backed page store over this pack's segments (options: `FileBackedPageStore`'s).

        `backend="native"` gives the native core's store (`NativePageStore`, decision 0012), which also takes
        `host_cache_bytes`; "python" (the default) the Python store.
        """
        if backend == "native":
            return NativePageStore(self.files, self.segments, **options)
        if backend != "python":
            raise ValueError(f"unknown storage backend {backend!r} (python or native)")
        return FileBackedPageStore(self.files, self.segments, **options)

    def load(self, segments: list[str] | None = None, pin: bool | None = None) -> InMemoryPageStore:
        """Every segment (or `segments`) read once into host memory (pinned when CUDA exists): a RAM tier."""
        reader = self.store(direct=True)
        tensors = {}
        try:
            for name in segments or list(self.segments):
                info = self.segments[name]
                data = aligned_host_buffer(info.nbytes, pin=pin).view(info.rows, info.row_bytes)
                data.copy_(reader.read_rows(name))
                tensors[name] = data
        finally:
            reader.close()
        return InMemoryPageStore(tensors, {name: self.segments[name] for name in tensors})

    def segment_bytes(self) -> dict[str, int]:
        return {name: segment.nbytes for name, segment in self.segments.items()}


class PackWriter:
    """Collects segments (new tensors, or tensors of source checkpoint files) and writes a pack."""

    def __init__(self, directory: str | Path, kind: str) -> None:
        self.directory = Path(directory)
        self.kind = kind
        self._tensors: dict[str, dict[str, torch.Tensor]] = {}
        self._sources: dict[str, SourceFile] = {}
        self._source_paths: dict[str, Path] = {}
        self._source_digests: dict[str, str] = {}
        self._source_segments: dict[str, tuple[str, str, torch.Tensor | None]] = {}
        self._composed: dict[str, ComposedSegment] = {}

    def add_tensor(self, segment: str, tensor: torch.Tensor, file: str = "pack") -> None:
        """A new tensor, written to the pack file `file`.safetensors."""
        self._check_new(segment)
        self._tensors.setdefault(file, {})[segment] = tensor.detach().contiguous().cpu()

    def add_source(self, key: str, source: SourceFile, path: str | Path | None = None, sha256: str | None = None) -> None:
        """A checkpoint file; `path` overrides its resolution through the Hugging Face cache.

        `sha256` is the digest its publisher declares; given, it is recorded instead of hashing
        the file now (which reads all of it).
        """
        self._sources[key] = source
        self._source_paths[key] = Path(path) if path is not None else source.resolve()
        if sha256 is not None:
            self._source_digests[key] = sha256

    def add_composed_segment(self, segment: ComposedSegment) -> None:
        """A segment whose rows are spans of source files (registered with `add_source`). Nothing is read."""
        self._check_new(segment.name)
        unknown = set(segment.files) - set(self._sources)
        if unknown:
            raise KeyError(f"{segment.name}: unknown sources {sorted(unknown)}")
        self._composed[segment.name] = segment

    def add_source_segment(self, segment: str, source: str, tensor_name: str, expected: torch.Tensor | None = None) -> None:
        """Refer to tensor `tensor_name` of source file `source`; with `expected`, its bytes must equal it."""
        self._check_new(segment)
        if source not in self._sources:
            raise KeyError(f"unknown source {source!r}")
        self._source_segments[segment] = (source, tensor_name, expected)

    def _check_new(self, segment: str) -> None:
        if (
            any(segment in tensors for tensors in self._tensors.values())
            or segment in self._source_segments
            or segment in self._composed
        ):
            raise ValueError(f"segment {segment!r} added twice")

    def write(self, metadata: Mapping[str, Any], packing: Mapping[str, Any] | None = None) -> Pack:
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / MANIFEST).exists():
            raise FileExistsError(f"{self.directory} already holds a pack")
        files: dict[str, dict[str, Any]] = {}
        paths: dict[str, Path] = {}
        segments: dict[str, AnySegment] = {}
        digests: dict[str, str | None] = {}
        for key, tensors in self._tensors.items():
            path = self.directory / f"{key}.safetensors"
            save_file(tensors, str(path), metadata={"format": "pt", "awpmi_pack": self.kind})
            found = safetensors_segments(path, key)
            for name, tensor in tensors.items():
                segments[name] = found[name]
                digests[name] = tensor_bytes_sha256(tensor)
            files[key] = {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            paths[key] = path
        for key, source in self._sources.items():
            path = self._source_paths[key]
            files[key] = {"source": source.to_json(), "bytes": path.stat().st_size}
            if key in self._source_digests:
                files[key]["sha256"] = self._source_digests[key]
                files[key]["sha256_from"] = "publisher"
            else:
                files[key]["sha256"] = sha256_file(path)
            paths[key] = path
        sizes = {key: path.stat().st_size for key, path in paths.items()}
        for name, segment in self._composed.items():
            if not segment.spans_within(sizes):
                raise ValueError(f"{name} has a span beyond the end of its file")
            segments[name] = segment
            digests[name] = None
        if self._source_segments:
            by_source = {key: safetensors_segments(paths[key], key) for key in self._sources}
            wanted = {
                f"{source}/{tensor_name}": by_source[source][tensor_name]
                for source, tensor_name, _ in self._source_segments.values()
            }
            reader = FileBackedPageStore(
                {key: paths[key] for key in self._sources},
                {key: Segment(key, s.file, s.offset, s.rows, s.row_bytes, s.dtype, s.row_shape) for key, s in wanted.items()},
                direct=False,
            )
            try:
                for name, (source, tensor_name, expected) in self._source_segments.items():
                    found = by_source[source][tensor_name]
                    digest = segment_sha256(reader, f"{source}/{tensor_name}")
                    if expected is not None and tensor_bytes_sha256(expected) != digest:
                        raise ValueError(f"{source}:{tensor_name} does not hold the bytes of segment {name}")
                    segments[name] = Segment(name, source, found.offset, found.rows, found.row_bytes, found.dtype, found.row_shape)
                    digests[name] = digest
            finally:
                reader.close()
        manifest = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION if self._composed else 1,
            "kind": self.kind,
            "files": files,
            "segments": {name: {**segment.to_json(), "sha256": digests[name]} for name, segment in sorted(segments.items())},
            "metadata": dict(metadata),
            "packing": dict(packing or {}),
        }
        (self.directory / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return Pack(self.directory, manifest, paths, segments)


def open_pack(
    directory: str | Path, verify: str = "segments", resolve: Callable[[SourceFile], Path] | None = None
) -> Pack:
    """Open a pack. `verify`: "segments" re-hashes every segment, "files" every file, "size" checks sizes only.

    Hashing reads with direct I/O. A segment without a recorded digest (a composed index) can
    only be verified through its files. `resolve` locates source checkpoint files (default:
    the local Hugging Face cache).
    """
    resolve = resolve or SourceFile.resolve
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("format_version") not in (1, FORMAT_VERSION):
        raise ValueError(f"{directory}: not an {FORMAT} v1 or v{FORMAT_VERSION} pack")
    if verify not in ("segments", "files", "size"):
        raise ValueError(f"unknown verification {verify!r}")
    paths: dict[str, Path] = {}
    for key, entry in manifest["files"].items():
        path = directory / entry["path"] if "path" in entry else Path(resolve(SourceFile(**entry["source"])))
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{path}: size {path.stat().st_size} != {entry['bytes']} in the manifest")
        if verify == "files" and sha256_file_direct(path) != entry["sha256"]:
            raise ValueError(f"{path}: sha256 differs from the manifest")
        paths[key] = path
    segments = {name: segment_from_json(name, entry) for name, entry in manifest["segments"].items()}
    pack = Pack(directory, manifest, paths, segments)
    if verify == "segments":
        unverifiable = sorted(name for name, entry in manifest["segments"].items() if entry.get("sha256") is None)
        if unverifiable:
            raise ValueError(f"{directory}: {len(unverifiable)} segments have no digest (e.g. {unverifiable[0]}); verify the files")
        # Direct reads: verifying must not pull the pack into the OS page cache (decision 0006: on
        # NTFS, direct reads of a file that has cached pages cost several times more).
        reader = pack.store(direct=True)
        try:
            for name, entry in manifest["segments"].items():
                if segment_sha256(reader, name) != entry["sha256"]:
                    raise ValueError(f"{directory}: segment {name} differs from the manifest")
        finally:
            reader.close()
    return pack
