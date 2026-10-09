from __future__ import annotations

from pathlib import Path

import pytest
import torch

from awpmi.runtime import configure_reproducible_numerics

# Before any CUDA matmul: the certificate's accumulation model assumes these flags.
configure_reproducible_numerics()

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "smollm2-135m.yaml"

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _native_available() -> bool:
    from awpmi.storage.native import NATIVE_AVAILABLE

    return NATIVE_AVAILABLE


def _encoded_available() -> bool:
    from awpmi.streaming import nvcomp

    return torch.cuda.is_available() and nvcomp.AVAILABLE and _native_available()


_NATIVE = pytest.mark.skipif(not _native_available(), reason="the native extension weightsift_native is not built")
_ENCODED = pytest.mark.skipif(not _encoded_available(), reason="encoded rows need CUDA, nvCOMP and the native extension")
# Storage backends of the parity tests (decision 0012): the Python store, the native store, and the native store with a
# host cache so small that rows are evicted between calls (hits, misses and evictions in one run). Phase 6B (decision
# 0013): the experts encoded by nvCOMP and decoded on the GPU, from the native store, without and with an evicting host
# cache (CUDA only).
BACKENDS = [
    "python", pytest.param("native", marks=_NATIVE), pytest.param("native-cache", marks=_NATIVE),
    pytest.param("encoded", marks=_ENCODED), pytest.param("encoded-cache", marks=_ENCODED), pytest.param("encoded-device-cache", marks=_ENCODED),
]
# Chunked calls also run with their experts prefetched into a host cache that holds the whole pack (decision 0012).
CHUNKED_BACKENDS = [*BACKENDS, pytest.param("native-prefetch", marks=_NATIVE)]
# Small chunks, so that the tiny models' rows hold several (Moonlight's runs use 1 MiB).
TEST_CHUNK_BYTES = 1024


def page_store(pack, backend: str = "python", **options):
    """A file-backed store of `pack` for a test backend; "native-cache" holds about three of its largest rows,
    "native-prefetch" every row."""
    if backend == "native-cache":
        largest = max(segment.row_bytes for segment in pack.segments.values())
        return pack.store(backend="native", host_cache_bytes=3 * largest, **options)
    if backend == "native-prefetch":
        return pack.store(backend="native", host_cache_bytes=sum(s.nbytes for s in pack.segments.values()), **options)
    return pack.store(backend=backend, **options)


def materialization_backend(pack, backend: str, device, cache=None, **options):
    """The materialization backend of a parity test: `page_store`'s store, a streamer and `cache`; for the encoded backends,
    an encoded pack written next to `pack` (once) on the native store and a GPU decoder (on CUDA only: elsewhere the test
    is skipped)."""
    from awpmi.materialization.backend import MaterializationBackend
    from awpmi.streaming.streamer import PageStreamer

    if not backend.startswith("encoded"):
        return MaterializationBackend(page_store(pack, backend, **options), device, PageStreamer(device), cache)
    if torch.device(device).type != "cuda":
        pytest.skip("encoded rows are decoded on CUDA")
    from awpmi.storage.encoded import is_encoded, open_encoded
    from awpmi.streaming.codec import RowDecoder, write_encoded_pack

    directory = Path(pack.directory).with_name(Path(pack.directory).name + "-encoded")
    if (directory / "manifest.json").exists() and is_encoded(directory):
        encoded = open_encoded(directory, verify="files")
    else:
        encoded = write_encoded_pack(pack, directory, chunk_bytes=TEST_CHUNK_BYTES)
    store = page_store(encoded.pack, "native-cache" if backend == "encoded-cache" else "native", **options)
    if backend == "encoded-device-cache" and cache is None:
        # Encoded rows cached on the device in fixed slots (Phase 6B): two slots of each stored row size, so that calls
        # hit, miss and evict.
        from awpmi.storage.cache import SlotCache

        cache = SlotCache({size: 2 for size in {encoded.pack.segments[name].row_bytes for name in encoded.encodings}}, device)
    return MaterializationBackend(store, device, PageStreamer(device), cache, decoder=RowDecoder(encoded.encodings, device))

PROMPTS = (
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return",
    "Once upon a time, there was a",
    " The game 's battle system , the BliTZ system , is carried over directly from Valkyira Chronicles",
    "1, 2, 3, 4, 5, 6, 7,",
    "Water boils at a temperature of 100 degrees",
)


@pytest.fixture(scope="session")
def experiment_config():
    from awpmi.config import load_config

    return load_config(CONFIG_PATH)[0]


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def loaded_model(experiment_config, device):
    from awpmi.models.smollm2 import ModelSpec, load_model, resolve_dtype

    spec = ModelSpec(
        repository=experiment_config.model.repository,
        revision=experiment_config.model.revision,
        dtype=resolve_dtype(experiment_config.model.dtype, device),
        device=device,
    )
    return load_model(spec)


@pytest.fixture(scope="session")
def model(loaded_model):
    return loaded_model[0]


@pytest.fixture(scope="session")
def tokenizer(loaded_model):
    return loaded_model[1]


@pytest.fixture(scope="session")
def prompt_ids(tokenizer, device):
    return [tokenizer(text, return_tensors="pt").input_ids.to(device) for text in PROMPTS]


@pytest.fixture(scope="session")
def reference_numerics(model, experiment_config):
    from awpmi.bounds.residual import ReferenceNumerics
    from awpmi.models.smollm2 import lm_head_weight

    weight = lm_head_weight(model)
    return ReferenceNumerics(
        output_dtype=weight.dtype,
        accumulation_unit_roundoff=experiment_config.numerics.accumulation_unit_roundoff,
        reduction_length=weight.shape[1],
    )
