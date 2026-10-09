"""nvCOMP's batched low-level C API, through ctypes (Phase 6B, decision 0013).

NVIDIA's nvCOMP (the `nvidia-libnvcomp-cu13` wheel: `nvcomp64_5.dll` / `libnvcomp.so.5`, under NVIDIA's SDK license; the
optional dependency group `gpu`) compresses and decompresses independent chunks on the GPU, many per launch: device arrays
of the chunks' addresses and sizes, a temporary buffer, a CUDA stream. A call enqueues work and returns; nothing here
synchronizes but `compress` (which returns the compressed chunks) and `Statuses.check`. This module binds that API and
nothing else: it defines no format and decodes nothing on the CPU. Without the wheel `AVAILABLE` is False.

The option structs are nvCOMP 5's (`nvcomp/<codec>.h`), 64 bytes each, unused bytes zero.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import torch

_SIZE, _INT, _U8, _PTR = ctypes.c_size_t, ctypes.c_int, ctypes.c_uint8, ctypes.c_void_p


def _options(name: str, fields: list) -> type:
    used = sum(ctypes.sizeof(kind) for _, kind in fields)
    return type(name, (ctypes.Structure,), {"_fields_": [*fields, ("reserved", ctypes.c_char * (64 - used))]})


# codec: (nvCOMP's name, compression options, decompression options)
OPTIONS = {
    "ans": (
        "ANS",
        _options("ANSCompressOpts", [("type", _INT), ("data_type", _INT), ("max_sub_chunk_count", _U8)]),
        _options("ANSDecompressOpts", [("backend", _INT), ("data_type", _INT), ("max_sub_chunk_count", _U8)]),
    ),
    "zstd": ("Zstd", _options("ZstdCompressOpts", []), _options("ZstdDecompressOpts", [("backend", _INT)])),
    "gdeflate": ("Gdeflate", _options("GdeflateCompressOpts", [("algorithm", _INT)]), _options("GdeflateDecompressOpts", [("backend", _INT)])),
    "lz4": (
        "LZ4",
        _options("LZ4CompressOpts", [("data_type", _INT), ("bitshuffle_mode", _INT)]),
        _options("LZ4DecompressOpts", [("backend", _INT), ("sort_before_hw_decompress", _INT), ("data_type", _INT), ("bitshuffle_mode", _INT)]),
    ),
    "bitcomp": ("Bitcomp", _options("BitcompCompressOpts", [("algorithm", _INT), ("data_type", _INT)]), _options("BitcompDecompressOpts", [("backend", _INT)])),
}
# nvcompType_t
DATA_TYPES = {"char": 0, "uchar": 1, "ushort": 3, "float16": 9}


class _Alignments(ctypes.Structure):
    _fields_ = [("input", _SIZE), ("output", _SIZE), ("temp", _SIZE)]


class _Properties(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("cudart_version", ctypes.c_uint32)]


def _library_path() -> Path | None:
    try:
        import nvidia.libnvcomp as package
    except ImportError:
        return None
    base = Path(package.__file__).parent
    path = base / ("bin/nvcomp64_5.dll" if sys.platform == "win32" else "lib/libnvcomp.so.5")
    return path if path.exists() else None


AVAILABLE = _library_path() is not None
_LIBRARY = None


def library() -> ctypes.CDLL:
    """The loaded nvCOMP library (CUDA initialized first; raises without the wheel)."""
    global _LIBRARY
    if _LIBRARY is None:
        path = _library_path()
        if path is None:
            raise RuntimeError("nvCOMP is not installed: `uv sync` installs it with the dependency group gpu (nvidia-libnvcomp-cu13)")
        torch.cuda.init()  # nvCOMP uses the CUDA runtime torch loaded
        if sys.platform == "win32":
            os.add_dll_directory(str(path.parent))
        _LIBRARY = ctypes.CDLL(str(path))
    return _LIBRARY


def _check(status: int, what: str) -> None:
    if status != 0:
        raise RuntimeError(f"nvCOMP {what}: status {status}")


def version() -> dict[str, int]:
    """nvCOMP's version and the CUDA runtime it was built with (e.g. 5300 and 13030)."""
    properties = _Properties()
    _check(library().nvcompGetProperties(ctypes.byref(properties)), "nvcompGetProperties")
    return {"version": properties.version, "cudart_version": properties.cudart_version}


def _function(codec: str, suffix: str):
    return getattr(library(), f"nvcompBatched{OPTIONS[codec][0]}{suffix}")


def compress_options(codec: str, **values):
    return OPTIONS[codec][1](**values)


def decompress_options(codec: str, **values):
    return OPTIONS[codec][2](**values)


def alignments(codec: str, decompress: bool, options) -> dict[str, int]:
    """The minimum alignments nvCOMP requires of the input, output and temporary buffers."""
    out = _Alignments()
    kind = "Decompress" if decompress else "Compress"
    _check(_function(codec, f"{kind}GetRequiredAlignments")(options, ctypes.byref(out)), "alignments")
    return {"input": out.input, "output": out.output, "temp": out.temp}


def compress(codec: str, options, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
    """Each device chunk (uint8, contiguous) compressed alone, on the current stream; synchronizes, checks every chunk's
    status and returns the compressed chunks (new device tensors)."""
    count, sizes = len(chunks), [c.numel() for c in chunks]
    largest, total = max(sizes), sum(sizes)
    temp, max_out = _SIZE(), _SIZE()
    _check(_function(codec, "CompressGetTempSizeAsync")(_SIZE(count), _SIZE(largest), options, ctypes.byref(temp), _SIZE(total)), "temp size")
    _check(_function(codec, "CompressGetMaxOutputChunkSize")(_SIZE(largest), options, ctypes.byref(max_out)), "output size")
    stride = -(-max_out.value // 256) * 256
    out = torch.empty(count * stride, dtype=torch.uint8, device="cuda")
    arrays = torch.tensor(
        [[c.data_ptr() for c in chunks], sizes, [out.data_ptr() + k * stride for k in range(count)]], dtype=torch.int64, device="cuda"
    )
    out_bytes = torch.zeros(count, dtype=torch.int64, device="cuda")
    statuses = torch.zeros(count, dtype=torch.int32, device="cuda")
    scratch = torch.empty(max(temp.value, 1), dtype=torch.uint8, device="cuda")
    stream = _PTR(torch.cuda.current_stream().cuda_stream)
    _check(_function(codec, "CompressAsync")(
        _PTR(arrays[0].data_ptr()), _PTR(arrays[1].data_ptr()), _SIZE(largest), _SIZE(count), _PTR(scratch.data_ptr()), _SIZE(temp.value),
        _PTR(arrays[2].data_ptr()), _PTR(out_bytes.data_ptr()), options, _PTR(statuses.data_ptr()), stream), "compress")
    torch.cuda.current_stream().synchronize()
    if bool((statuses != 0).any()):
        raise RuntimeError(f"nvCOMP compression statuses {statuses.unique().tolist()}")
    lengths = out_bytes.tolist()
    return [out[k * stride : k * stride + lengths[k]].clone() for k in range(count)]


def decompress_temp_bytes(codec: str, options, chunks: int, largest: int, total: int) -> int:
    """Bytes of temporary device memory a decompression of `chunks` chunks needs."""
    temp = _SIZE()
    _check(_function(codec, "DecompressGetTempSizeAsync")(_SIZE(chunks), _SIZE(largest), options, ctypes.byref(temp), _SIZE(total)), "temp size")
    return temp.value


def decompress(codec: str, options, arrays: torch.Tensor, scratch: torch.Tensor, actual: torch.Tensor, statuses: torch.Tensor) -> None:
    """Enqueue, on the current stream, the decompression of the chunks of `arrays`: a device int64 [4, n] holding each
    chunk's address and compressed size, then its output's address and size. `actual` (int64 [n]) receives the sizes
    decompressed, `statuses` (int32 [n]) each chunk's status; `scratch` is the temporary buffer. Nothing synchronizes."""
    count = arrays.shape[1]
    stream = _PTR(torch.cuda.current_stream().cuda_stream)
    _check(_function(codec, "DecompressAsync")(
        _PTR(arrays[0].data_ptr()), _PTR(arrays[1].data_ptr()), _PTR(arrays[3].data_ptr()), _PTR(actual.data_ptr()), _SIZE(count),
        _PTR(scratch.data_ptr()), _SIZE(scratch.numel()), _PTR(arrays[2].data_ptr()), options, _PTR(statuses.data_ptr()), stream), "decompress")
