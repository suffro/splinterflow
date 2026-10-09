//! `weightsift_native`: the Python module of Weightsift's native I/O core (Phase 6A, decision 0012).
//!
//! Python's `awpmi.storage.native` wraps it; nothing else should import it. The boundary is coarse: one call plans
//! and starts a whole transfer (every request of an experts call), one call returns each ready piece with all its
//! copies, and the GIL is released while the core plans, reads or waits. Buffers cross the boundary through the
//! buffer protocol (NumPy views of PyTorch tensors): staging slots the core writes, row and position arrays it reads.
//! A `Job` keeps every buffer it was given referenced until none of its tasks can touch them.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyIndexError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use weightsift_io::engine::{self, EngineConfig, Request, SlotBuffers};
use weightsift_io::{CacheStats, Error, FileTable, IoStats, PlanConfig, RawBuffer, Segment, Source};

fn to_py(error: Error) -> PyErr {
    match error {
        Error::Invalid(message) => PyValueError::new_err(message),
        Error::Index(message) => PyIndexError::new_err(message),
        error @ (Error::Io { .. } | Error::ShortRead { .. }) => PyOSError::new_err(error.to_string()),
        error => PyRuntimeError::new_err(error.to_string()),
    }
}

/// (name, files, rows, row_bytes, offset, part_bytes, spans): a plain segment has an offset (one file), a composed one
/// its part sizes and its spans (row-major, (index into `files`, offset)).
#[derive(FromPyObject)]
struct SegmentSpec(
    String,
    Vec<u32>,
    u64,
    u64,
    Option<u64>,
    Option<Vec<u64>>,
    Option<Vec<(u32, u64)>>,
);

impl SegmentSpec {
    fn build(self) -> weightsift_io::Result<Segment> {
        let SegmentSpec(name, files, rows, row_bytes, offset, part_bytes, spans) = self;
        match (offset, part_bytes, spans) {
            (Some(offset), None, None) => {
                let [file] = files[..] else {
                    return Err(Error::invalid(format!("{name}: a plain segment has one file")));
                };
                Segment::plain(&name, file, offset, rows, row_bytes)
            }
            (None, Some(part_bytes), Some(spans)) => {
                let segment = Segment::composed(&name, files, rows, part_bytes, spans)?;
                if segment.row_bytes != row_bytes {
                    return Err(Error::invalid(format!(
                        "{name}: parts make {} bytes, not {row_bytes}",
                        segment.row_bytes
                    )));
                }
                Ok(segment)
            }
            _ => Err(Error::invalid(format!(
                "{name}: give an offset, or part sizes and spans"
            ))),
        }
    }
}

fn indices(py: Python<'_>, buffer: Option<PyBuffer<i64>>) -> PyResult<Option<Vec<i64>>> {
    buffer.map(|b| b.to_vec(py)).transpose()
}

fn writable(buffer: &PyBuffer<u8>, what: &str) -> PyResult<RawBuffer> {
    if buffer.readonly() || !buffer.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{what} must be a writable, contiguous byte buffer"
        )));
    }
    // SAFETY: the buffer view keeps its exporter alive and its memory in place; the Job (or the call) that uses the
    // RawBuffer holds the view until no task can touch the memory (see `Job`).
    Ok(unsafe { RawBuffer::new(buffer.buf_ptr() as *mut u8, buffer.len_bytes()) })
}

fn io_stats<'py>(py: Python<'py>, stats: &IoStats) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("requests", stats.requests)?;
    dict.set_item("rows", stats.rows)?;
    dict.set_item("logical_bytes", stats.logical_bytes)?;
    dict.set_item("physical_bytes", stats.physical_bytes)?;
    dict.set_item("read_calls", stats.read_calls)?;
    dict.set_item("extents", stats.extents)?;
    dict.set_item("blocks_4k", stats.blocks_4k)?;
    dict.set_item("busy_ms", stats.busy_ns as f64 / 1e6)?;
    dict.set_item("read_ms", stats.read_ns as f64 / 1e6)?;
    dict.set_item("cache_copied_bytes", stats.cache_copied_bytes)?;
    dict.set_item("gathered_bytes", stats.gathered_bytes)?;
    dict.set_item("admitted_bytes", stats.admitted_bytes)?;
    dict.set_item("fallback_rows", stats.fallback_rows)?;
    dict.set_item("prefetches", stats.prefetches)?;
    dict.set_item("prefetch_rows", stats.prefetch_rows)?;
    dict.set_item("prefetch_bytes", stats.prefetch_bytes)?;
    dict.set_item("prefetch_blocks_4k", stats.prefetch_blocks_4k)?;
    dict.set_item("submits", stats.submits)?;
    dict.set_item("submit_ms", stats.submit_ns as f64 / 1e6)?;
    dict.set_item("submit_cache_ms", stats.submit_cache_ns as f64 / 1e6)?;
    dict.set_item("submit_plan_ms", stats.submit_plan_ns as f64 / 1e6)?;
    dict.set_item("submit_start_ms", stats.submit_start_ns as f64 / 1e6)?;
    let by_segment = PyDict::new(py);
    for (segment, entry) in &stats.by_segment {
        let item = PyDict::new(py);
        item.set_item("requests", entry.requests)?;
        item.set_item("rows", entry.rows)?;
        item.set_item("logical_bytes", entry.logical_bytes)?;
        item.set_item("physical_bytes", entry.physical_bytes)?;
        by_segment.set_item(segment, item)?;
    }
    dict.set_item("by_segment", by_segment)?;
    dict.set_item("ranges", stats.ranges.clone())?;
    Ok(dict)
}

fn cache_stats<'py>(py: Python<'py>, stats: &CacheStats) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in [
        ("lookups", stats.lookups),
        ("hits", stats.hits),
        ("hit_bytes", stats.hit_bytes),
        ("waits", stats.waits),
        ("wait_bytes", stats.wait_bytes),
        ("misses", stats.misses),
        ("miss_bytes", stats.miss_bytes),
        ("inserts", stats.inserts),
        ("insert_bytes", stats.insert_bytes),
        ("evictions", stats.evictions),
        ("evicted_bytes", stats.evicted_bytes),
        ("bypassed", stats.bypassed),
        ("bypassed_bytes", stats.bypassed_bytes),
        ("aborted_fills", stats.aborted_fills),
        ("recycled", stats.recycled),
        ("recycled_bytes", stats.recycled_bytes),
        ("allocated_bytes", stats.allocated_bytes),
        ("released_bytes", stats.released_bytes),
        ("prefetch_fills", stats.prefetch_fills),
        ("prefetch_fill_bytes", stats.prefetch_fill_bytes),
        ("prefetch_skipped", stats.prefetch_skipped),
        ("prefetch_bypassed", stats.prefetch_bypassed),
        ("prefetch_used", stats.prefetch_used),
        ("prefetch_wasted", stats.prefetch_wasted),
    ] {
        dict.set_item(key, value)?;
    }
    Ok(dict)
}

type RunTuple = (u32, u64, u64, u64, u32);
type ExtentTuple = (u32, u64, u64);
type PlanTuple = (Vec<RunTuple>, Vec<ExtentTuple>, u64, u64, u64);
type OpTuple = (u8, u64, u32, u64, u64);
type PieceTuple = (Vec<ExtentTuple>, Vec<(u64, u64, u64)>);
/// (segment, rows or None, positions or None) of `Engine.submit`.
type RequestArg = (u32, Option<PyBuffer<i64>>, Option<PyBuffer<i64>>);

/// The native engine: files, segments, reader threads and an optional host-RAM cache.
#[pyclass(frozen, module = "weightsift_native")]
struct Engine {
    inner: engine::Engine,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (files, segments, *, direct=true, alignment=4096, max_gap=0, max_read_bytes=1<<20,
                        max_extent_bytes=8<<20, workers=8, host_cache_bytes=0, cache_block_bytes=0,
                        direct_copy_bytes=256<<10, copy_chunk_bytes=4<<20))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        files: Vec<(String, String)>,
        segments: Vec<SegmentSpec>,
        direct: bool,
        alignment: u64,
        max_gap: u64,
        max_read_bytes: u64,
        max_extent_bytes: u64,
        workers: usize,
        host_cache_bytes: u64,
        cache_block_bytes: u64,
        direct_copy_bytes: u64,
        copy_chunk_bytes: u64,
    ) -> PyResult<Self> {
        let segments: Vec<Segment> = segments
            .into_iter()
            .map(SegmentSpec::build)
            .collect::<weightsift_io::Result<_>>()
            .map_err(to_py)?;
        let config = EngineConfig {
            direct,
            plan: PlanConfig {
                alignment,
                max_gap,
                max_extent_bytes,
            },
            max_read_bytes,
            workers,
            host_cache_bytes,
            cache_block_bytes,
            direct_copy_bytes,
            copy_chunk_bytes,
        };
        let inner = py
            .detach(|| {
                let table = FileTable::open(
                    files.into_iter().map(|(key, path)| (key, path.into())).collect(),
                    direct,
                )?;
                engine::Engine::new(table, segments, config)
            })
            .map_err(to_py)?;
        Ok(Engine { inner })
    }

    /// The plan of a request: (runs (file, offset, length, output, extent), extents (file, offset, length),
    /// rows, output rows, distinct 4 KiB blocks). File indices are the segment's.
    #[pyo3(signature = (segment, rows=None, positions=None))]
    fn plan(
        &self,
        py: Python<'_>,
        segment: u32,
        rows: Option<PyBuffer<i64>>,
        positions: Option<PyBuffer<i64>>,
    ) -> PyResult<PlanTuple> {
        let request = Request {
            segment,
            rows: indices(py, rows)?,
            positions: indices(py, positions)?,
        };
        let plan = self.inner.plan(&request).map_err(to_py)?;
        let runs = plan
            .runs
            .iter()
            .map(|r| (r.file, r.offset, r.length, r.output, r.extent))
            .collect();
        let extents = plan.extents.iter().map(|e| (e.file, e.offset, e.length)).collect();
        Ok((runs, extents, plan.row_count, plan.output_rows, plan.blocks_4k()))
    }

    /// The staging pieces of a request's plan for slots of `slot_bytes`: [(extents, parts (staging, length, output))].
    #[pyo3(signature = (segment, slot_bytes, rows=None, positions=None))]
    fn pieces(
        &self,
        py: Python<'_>,
        segment: u32,
        slot_bytes: u64,
        rows: Option<PyBuffer<i64>>,
        positions: Option<PyBuffer<i64>>,
    ) -> PyResult<Vec<PieceTuple>> {
        let request = Request {
            segment,
            rows: indices(py, rows)?,
            positions: indices(py, positions)?,
        };
        let plan = self.inner.plan(&request).map_err(to_py)?;
        Ok(weightsift_io::plan::pieces(&plan, slot_bytes)
            .into_iter()
            .map(|piece| {
                let extents = piece.extents.iter().map(|e| (e.file, e.offset, e.length)).collect();
                let parts = piece.parts.iter().map(|p| (p.staging, p.length, p.output)).collect();
                (extents, parts)
            })
            .collect())
    }

    /// Start a transfer of `requests` ((segment, rows or None, positions or None)) into `slots` ((staging, gather
    /// buffer or None)); with `use_cache`, through the host-RAM cache.
    #[pyo3(signature = (requests, slots, use_cache=true))]
    fn submit(
        &self,
        py: Python<'_>,
        requests: Vec<RequestArg>,
        slots: Vec<(PyBuffer<u8>, Option<PyBuffer<u8>>)>,
        use_cache: bool,
    ) -> PyResult<Job> {
        let requests = requests
            .into_iter()
            .map(|(segment, rows, positions)| {
                Ok(Request {
                    segment,
                    rows: indices(py, rows)?,
                    positions: indices(py, positions)?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut views = Vec::with_capacity(slots.len());
        let mut buffers = Vec::with_capacity(slots.len() * 2);
        for (staging, compact) in slots {
            let staging_view = writable(&staging, "a staging slot")?;
            let compact_view = compact.as_ref().map(|c| writable(c, "a gather buffer")).transpose()?;
            views.push(SlotBuffers {
                staging: staging_view,
                compact: compact_view,
            });
            buffers.push(staging);
            buffers.extend(compact);
        }
        let job = py
            .detach(|| self.inner.submit(&requests, views, use_cache))
            .map_err(to_py)?;
        Ok(Job {
            inner: Some(job),
            buffers,
        })
    }

    /// Load the rows of `requests` ((segment, rows or None)) into the host cache in the background: rows soon asked for.
    fn prefetch(&self, py: Python<'_>, requests: Vec<(u32, Option<PyBuffer<i64>>)>) -> PyResult<Prefetch> {
        let requests = requests
            .into_iter()
            .map(|(segment, rows)| {
                Ok(Request {
                    segment,
                    rows: indices(py, rows)?,
                    positions: None,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let prefetch = py.detach(|| self.inner.prefetch(&requests)).map_err(to_py)?;
        Ok(Prefetch { inner: Some(prefetch) })
    }

    /// Read rows from storage into `out` (bytes of the request's output rows), bypassing the cache.
    #[pyo3(signature = (segment, out, rows=None, positions=None))]
    fn read_rows(
        &self,
        py: Python<'_>,
        segment: u32,
        out: PyBuffer<u8>,
        rows: Option<PyBuffer<i64>>,
        positions: Option<PyBuffer<i64>>,
    ) -> PyResult<()> {
        let request = Request {
            segment,
            rows: indices(py, rows)?,
            positions: indices(py, positions)?,
        };
        let view = writable(&out, "the output")?;
        // `out` lives until this call returns, and `read_rows` returns only after its job is closed.
        py.detach(|| self.inner.read_rows(&request, view)).map_err(to_py)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        io_stats(py, &self.inner.stats())
    }

    /// The host cache's counters, budget and contents (None without a cache): resident bytes are its rows' (ready and
    /// loading), held bytes everything it holds (its rows and its pool of evicted rows' blocks).
    fn cache_stats<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(cache) = self.inner.cache() else {
            return Ok(None);
        };
        let dict = cache_stats(py, &cache.stats())?;
        dict.set_item("capacity_bytes", cache.capacity())?;
        dict.set_item("block_bytes", cache.block_bytes())?;
        dict.set_item("resident_bytes", cache.resident_bytes())?;
        dict.set_item("peak_resident_bytes", cache.peak_resident_bytes())?;
        dict.set_item("pool_bytes", cache.pool_bytes())?;
        dict.set_item("held_bytes", cache.held_bytes())?;
        dict.set_item("peak_held_bytes", cache.peak_held_bytes())?;
        dict.set_item("entries", cache.len())?;
        dict.set_item("admit", cache.admit())?;
        Ok(Some(dict))
    }

    #[pyo3(signature = (record_ranges=false))]
    fn reset_stats(&self, record_ranges: bool) {
        self.inner.reset_stats(record_ranges);
    }

    /// Freeze (False) or resume (True) admission into the host cache.
    fn set_admit(&self, admit: bool) -> PyResult<()> {
        let cache = self
            .inner
            .cache()
            .ok_or_else(|| PyValueError::new_err("the engine has no host cache"))?;
        cache.set_admit(admit);
        Ok(())
    }

    /// Drop every cached row nobody uses.
    fn clear_cache(&self) {
        if let Some(cache) = self.inner.cache() {
            cache.clear();
        }
    }

    /// The cached rows (segment, row), least recently used first.
    fn cached_rows(&self) -> Vec<(u32, u64)> {
        self.inner.cache().map(|c| c.keys_lru_first()).unwrap_or_default()
    }

    /// Stop the reader threads (in-flight reads finish; unfinished jobs fail) and release the host cache's memory.
    fn close(&self, py: Python<'_>) {
        py.detach(|| self.inner.close());
    }
}

/// A transfer started by `Engine.submit`: pieces in order, each a slot and its copies.
#[pyclass(module = "weightsift_native")]
struct Job {
    // Declared before `buffers`: dropped (closed: no task in flight) before the buffers it writes are released.
    inner: Option<engine::Job>,
    buffers: Vec<PyBuffer<u8>>,
}

impl Job {
    fn job(&self) -> PyResult<&engine::Job> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("the transfer is closed"))
    }
}

#[pymethods]
impl Job {
    /// The next ready piece (blocks, without the GIL): (slot, copies, gathered bytes), each copy (source (0 staging,
    /// 1 gather buffer), source offset, request, destination offset, length); None when every piece was delivered.
    fn next(&self, py: Python<'_>) -> PyResult<Option<(usize, Vec<OpTuple>, u64)>> {
        let job = self.job()?;
        let piece = py.detach(|| job.next()).map_err(to_py)?;
        Ok(piece.map(|piece| {
            let ops = piece
                .ops
                .iter()
                .map(|op| {
                    (
                        matches!(op.source, Source::Compact) as u8,
                        op.src,
                        op.request,
                        op.dst,
                        op.length,
                    )
                })
                .collect();
            (piece.slot, ops, piece.gathered_bytes)
        }))
    }

    /// The caller's device copies out of `slot` finished: the engine may refill it.
    fn release(&self, slot: usize) -> PyResult<()> {
        self.job()?.release(slot).map_err(to_py)
    }

    fn cancel(&self) -> PyResult<()> {
        self.job()?.cancel();
        Ok(())
    }

    /// The number of pieces of the transfer.
    #[getter]
    fn pieces(&self) -> PyResult<usize> {
        Ok(self.job()?.pieces())
    }

    /// Cancel if unfinished, wait until no task touches the slots, and release the buffers.
    fn close(&mut self, py: Python<'_>) {
        if let Some(job) = self.inner.take() {
            py.detach(move || drop(job));
        }
        self.buffers.clear();
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&mut self, py: Python<'_>, _args: &Bound<'_, pyo3::types::PyTuple>) {
        self.close(py);
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        if let Some(job) = self.inner.take() {
            // Waiting for in-flight tasks must not hold the GIL.
            Python::attach(|py| py.detach(move || drop(job)));
        }
    }
}

/// Rows being loaded into the host cache by `Engine.prefetch`.
#[pyclass(module = "weightsift_native")]
struct Prefetch {
    inner: Option<engine::Prefetch>,
}

#[pymethods]
impl Prefetch {
    /// The rows this prefetch loads.
    #[getter]
    fn rows(&self) -> u64 {
        self.inner.as_ref().map_or(0, |p| p.rows())
    }

    /// Wait (without the GIL) until every row is in the cache.
    fn wait(&self, py: Python<'_>) -> PyResult<()> {
        match &self.inner {
            Some(prefetch) => py.detach(|| prefetch.wait()).map_err(to_py),
            None => Ok(()),
        }
    }

    fn cancel(&self) {
        if let Some(prefetch) = &self.inner {
            prefetch.cancel();
        }
    }

    /// Cancel if unfinished, and wait until none of its tasks runs.
    fn close(&mut self, py: Python<'_>) {
        if let Some(prefetch) = self.inner.take() {
            py.detach(move || drop(prefetch));
        }
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&mut self, py: Python<'_>, _args: &Bound<'_, pyo3::types::PyTuple>) {
        self.close(py);
    }
}

impl Drop for Prefetch {
    fn drop(&mut self) {
        if let Some(prefetch) = self.inner.take() {
            Python::attach(|py| py.detach(move || drop(prefetch)));
        }
    }
}

#[pymodule]
fn weightsift_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Engine>()?;
    module.add_class::<Job>()?;
    module.add_class::<Prefetch>()?;
    module.add("DIRECT_ALIGNMENT", weightsift_io::DIRECT_ALIGNMENT)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
