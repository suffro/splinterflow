# Weightsift native core

Phase 6A (decision `.context/decisions/0012-phase6a-native-runtime-foundation.md`): the performance-critical part of
Weightsift's storage path, in Rust, behind the Python storage contract. Python's `awpmi.storage.native.NativePageStore`
is the only user; the Python backend (`FileBackedPageStore`) remains the reference implementation and the fallback.

| Crate | What it is |
| --- | --- |
| `core/` (`weightsift-io`) | Read plans (the same as Python's `plan_reads` and `PageStreamer._pieces`), positioned reads on a pool of threads (direct I/O), a host-RAM row cache under a strict byte budget, transfer jobs into caller-owned staging slots, byte accounting. No Python, no CUDA, no model knowledge. |
| `python/` (`weightsift-native`) | The PyO3 module `weightsift_native` (abi3, Python 3.11+): `Engine` and `Job`. |

## Building

The root project builds it: `python -m uv sync` (the default dependency groups include `native`). uv calls maturin as a
PEP 517 backend on `native/pyproject.toml` and installs the module into `.venv`; it rebuilds when any `*.rs`,
`Cargo.toml` or `Cargo.lock` changes.

| Requirement | Version |
| --- | --- |
| Rust toolchain | 1.95.0, pinned by `rust-toolchain.toml` (rustup installs it; `rustfmt` and `clippy` components) |
| C linker | MSVC build tools on Windows (`x86_64-pc-windows-msvc`); the system `cc` on Linux |
| maturin | ≥ 1.15, < 2 (fetched by uv for the build) |
| Python | ≥ 3.11 (abi3 wheel) |

Crates: `pyo3` 0.29 (bindings), `lru` 0.18 (the LRU order), `libc` 0.2 (Linux only: `O_DIRECT`). Positioned reads and
direct-I/O flags are the standard library's (`FileExt`, `OpenOptionsExt`).

Development:

```bash
cd native
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings      # PYO3_PYTHON=<repo>/.venv/Scripts/python.exe if needed
cargo test --workspace                                       # unit tests and core/tests/engine.rs (temporary files)
cd .. && python -m uv run python -m pytest tests/test_native_storage.py tests/test_moe.py tests/test_moe_compact.py tests/test_moe_chunked.py
```

## Platforms

| Platform | State |
| --- | --- |
| Windows 11, x86-64 (MSVC) | built and tested here; direct I/O with `FILE_FLAG_NO_BUFFERING`, one handle per reader thread |
| Linux, x86-64 | builds; direct I/O with `O_DIRECT`; not exercised on this machine (as the Python store's POSIX path) |
| Others | buffered reads only (`direct=False`); direct I/O is refused |

## Without the extension

`python -m uv sync --no-group native` installs everything but the extension. `awpmi.storage.native.NATIVE_AVAILABLE` is
then False, `Pack.store(backend="native")` raises, the native tests are skipped (reported as such), and every Python
path works as before Phase 6A: the native backend changes where bytes move, never what is computed.

## The `unsafe` boundary

`core/src/buffer.rs` holds the only `unsafe` type, `RawBuffer`: writing into staging memory the caller owns (PyTorch's
pinned buffers), so that reads land where the device copies from. Its invariants (the memory outlives every task; tasks
write disjoint ranges; a slot is handed to the caller only after its writes finished and refilled only after the
caller released it) are documented there and upheld by the engine and the binding (`Job` keeps its buffers referenced
until no task can touch them). Seven `unsafe` blocks use it, six in `core/src/engine.rs` (the tasks that read into,
copy into or out of, gather in and admit from a slot, and `read_rows`' copy out) and one in the binding (wrapping a
Python buffer), each stating the invariant it relies on.
