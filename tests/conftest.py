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


_NATIVE = pytest.mark.skipif(not _native_available(), reason="the native extension weightsift_native is not built")
# Storage backends of the parity tests (decision 0012): the Python store, the native store, and the native store with a
# host cache so small that rows are evicted between calls (hits, misses and evictions in one run).
BACKENDS = ["python", pytest.param("native", marks=_NATIVE), pytest.param("native-cache", marks=_NATIVE)]
# Chunked calls also run with their experts prefetched into a host cache that holds the whole pack (decision 0012).
CHUNKED_BACKENDS = [*BACKENDS, pytest.param("native-prefetch", marks=_NATIVE)]


def page_store(pack, backend: str = "python", **options):
    """A file-backed store of `pack` for a test backend; "native-cache" holds about three of its largest rows,
    "native-prefetch" every row."""
    if backend == "native-cache":
        largest = max(segment.row_bytes for segment in pack.segments.values())
        return pack.store(backend="native", host_cache_bytes=3 * largest, **options)
    if backend == "native-prefetch":
        return pack.store(backend="native", host_cache_bytes=sum(s.nbytes for s in pack.segments.values()), **options)
    return pack.store(backend=backend, **options)

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
