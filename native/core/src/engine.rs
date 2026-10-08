//! The engine: a file table, segments, a pool of reader threads, an optional host-RAM cache, and transfer jobs.
//!
//! A job moves the rows of one or more requests (segment, rows, output positions) into staging slots the caller owns
//! (pinned host memory), one slot-sized piece at a time, and tells the caller, piece by piece, which bytes of the slot
//! go where in each request's output. The caller copies them to the device and releases the slot; the engine fills
//! released slots with the next pieces, so the drive keeps reading while earlier pieces are copied (the read-ahead is
//! the number of slots). Within a job:
//!
//!   1. every row is looked up in the host cache (when the job uses it): hits, and rows another job is loading, go to
//!      the first pieces (copied from the cache into the slot: hits first); the misses are planned exactly as Python's
//!      `plan_reads` plans them and split into pieces as `PageStreamer._pieces` splits them;
//!   2. a piece's reads (positioned, direct, at most `max_read_bytes` each) and copies run on the pool; short parts are
//!      then gathered as Python's streamer gathers them;
//!   3. the piece is ready for the caller; meanwhile the missed rows the cache admitted are copied from the slot into
//!      their cache entries (the slot is reused only after these copies and the caller's release).
//!
//! A prefetch (`Engine::prefetch`) loads rows a request will ask for soon into the host cache, at the lowest priority,
//! through staging of its own that it releases itself; the request then finds them cached, or waits for their load.
//!
//! Where Phase 6B plugs in (not built): an `Op` is a copy of bytes as stored. A stored representation that is not the
//! reference's bytes (Phase 5C's bit planes in zstd frames per page) would add an op that names a decoder: the core
//! reads the frames into the slot and decompresses them on its threads, and the caller launches a device decoder
//! (planes merged into BF16 rows, written into the destination) instead of a copy. Plans, slots, the cache (then
//! holding compressed rows: about 1.5 times as many) and the accounting stay as they are; a native copy issuer (CUDA
//! in a C++ extension, releasing slots from a stream callback) would replace the caller's loop without changing the job.
//!
//! Every byte is counted where Python's store counts it (requests, rows, logical bytes, physical bytes the reads
//! returned, read calls, extents, 4 KiB blocks of the rows read, optional raw ranges, per segment), plus what the
//! cache served and the time the drive was busy. Nothing here knows a model, what a tensor means, or how rows are
//! chosen.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::buffer::RawBuffer;
use crate::cache::{CacheStats, Entry, Fill, HostCache, Lookup, Pending, Probe};
use crate::error::{Error, Result};
use crate::file::{AlignedBuffer, FileTable, Handles, DIRECT_ALIGNMENT};
use crate::layout::Segment;
use crate::plan::{check_rows, copy_ops, pieces, plan_reads, Gather, Part, Plan, PlanConfig, Source};

/// A prefetch reads into staging of its own: this many slots of this size.
const PREFETCH_SLOTS: usize = 2;
const PREFETCH_SLOT_BYTES: u64 = 32 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineConfig {
    pub direct: bool,
    pub plan: PlanConfig,
    /// One read call reads at most this much (a multiple of the alignment).
    pub max_read_bytes: u64,
    pub workers: usize,
    /// The host-RAM cache's budget; 0: no cache.
    pub host_cache_bytes: u64,
    /// Parts at least this long are copied to the device straight from staging (Python's `DIRECT_COPY_BYTES`).
    pub direct_copy_bytes: u64,
    /// Host copies (cache to slot, slot to cache entry) are split into tasks of at most this much.
    pub copy_chunk_bytes: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            direct: true,
            plan: PlanConfig {
                alignment: DIRECT_ALIGNMENT,
                max_gap: 0,
                max_extent_bytes: 8 << 20,
            },
            max_read_bytes: 1 << 20,
            workers: 8,
            host_cache_bytes: 0,
            direct_copy_bytes: 256 << 10,
            copy_chunk_bytes: 4 << 20,
        }
    }
}

/// Counters of one segment (Python's `IOStats.by_segment`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegmentStats {
    pub requests: u64,
    pub rows: u64,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
}

/// The engine's counters since the last reset (Python's `IOStats`, plus the native path's own).
#[derive(Clone, Debug, Default)]
pub struct IoStats {
    pub requests: u64,
    pub rows: u64,
    pub logical_bytes: u64,
    /// Bytes the reads returned.
    pub physical_bytes: u64,
    pub read_calls: u64,
    pub extents: u64,
    /// Distinct 4 KiB blocks of the rows read from storage (not of the rows the cache served).
    pub blocks_4k: u64,
    /// Time with at least one read in flight.
    pub busy_ns: u64,
    /// Sum of the read calls' durations.
    pub read_ns: u64,
    /// Bytes copied from the host cache into slots.
    pub cache_copied_bytes: u64,
    /// Bytes gathered into gather buffers.
    pub gathered_bytes: u64,
    /// Bytes copied from slots into new cache entries.
    pub admitted_bytes: u64,
    /// Rows read again because the load they waited for failed or was cancelled.
    pub fallback_rows: u64,
    /// Prefetches started; the rows they loaded into the host cache, their bytes, and the 4 KiB blocks they read.
    pub prefetches: u64,
    pub prefetch_rows: u64,
    pub prefetch_bytes: u64,
    pub prefetch_blocks_4k: u64,
    pub ranges: Option<Vec<(u32, u64, u64)>>,
    pub by_segment: BTreeMap<u32, SegmentStats>,
}

struct Busy {
    in_flight: u32,
    since: Option<Instant>,
}

struct Shared {
    table: Arc<FileTable>,
    segments: Vec<Arc<Segment>>,
    config: EngineConfig,
    cache: Option<Arc<HostCache>>,
    stats: Mutex<IoStats>,
    busy: Mutex<Busy>,
    queue: Mutex<Queue>,
    available: Condvar,
}

impl Shared {
    fn begin_read(&self) -> Instant {
        let now = Instant::now();
        let mut busy = self.busy.lock().unwrap();
        if busy.in_flight == 0 {
            busy.since = Some(now);
        }
        busy.in_flight += 1;
        now
    }

    fn end_read(&self, started: Instant) {
        let now = Instant::now();
        let mut busy = self.busy.lock().unwrap();
        busy.in_flight -= 1;
        let mut busy_ns = 0;
        if busy.in_flight == 0 {
            if let Some(since) = busy.since.take() {
                busy_ns = now.duration_since(since).as_nanos() as u64;
            }
        }
        drop(busy);
        let mut stats = self.stats.lock().unwrap();
        stats.busy_ns += busy_ns;
        stats.read_ns += now.duration_since(started).as_nanos() as u64;
    }
}

struct Queue {
    tasks: VecDeque<Task>,
    /// Prefetch reads: run only when no foreground task waits.
    background: VecDeque<Task>,
    closed: bool,
}

pub struct Engine {
    shared: Arc<Shared>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

/// A request of a job: rows of a segment (`None`: every row), request i written to output row `positions[i]`.
#[derive(Clone, Debug)]
pub struct Request {
    pub segment: u32,
    pub rows: Option<Vec<i64>>,
    pub positions: Option<Vec<i64>>,
}

/// A staging slot: the buffer reads and copies land in, and an optional gather buffer for short parts.
#[derive(Clone, Copy, Debug)]
pub struct SlotBuffers {
    pub staging: RawBuffer,
    pub compact: Option<RawBuffer>,
}

/// What the caller copies to the device for one piece: `length` bytes at `src` of the slot's `source` buffer to byte
/// `dst` of request `request`'s output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Op {
    pub source: Source,
    pub src: u64,
    pub request: u32,
    pub dst: u64,
    pub length: u64,
}

/// A piece ready for the caller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Delivered {
    pub slot: usize,
    pub ops: Vec<Op>,
    /// Bytes of the piece gathered into the gather buffer (Python's `gathered_bytes`).
    pub gathered_bytes: u64,
}

impl Engine {
    pub fn new(table: FileTable, segments: Vec<Segment>, config: EngineConfig) -> Result<Self> {
        config.plan.validate()?;
        if config.workers == 0 {
            return Err(Error::invalid("at least one worker"));
        }
        if config.direct && config.plan.alignment % DIRECT_ALIGNMENT != 0 {
            return Err(Error::invalid(format!(
                "direct I/O needs a multiple of {DIRECT_ALIGNMENT}-byte alignment"
            )));
        }
        if config.max_read_bytes == 0
            || config.max_read_bytes % config.plan.alignment != 0
            || config.plan.max_extent_bytes % config.plan.alignment != 0
        {
            return Err(Error::invalid(
                "read and extent limits must be multiples of the alignment",
            ));
        }
        if config.copy_chunk_bytes == 0 {
            return Err(Error::invalid("copy chunks must hold at least one byte"));
        }
        if table.direct != config.direct {
            return Err(Error::invalid("the file table's direct mode differs from the engine's"));
        }
        let sizes = table.sizes();
        for segment in &segments {
            if segment.files.iter().any(|&f| f as usize >= sizes.len()) {
                return Err(Error::invalid(format!("{} refers to an unknown file", segment.name)));
            }
            if !segment.within(&sizes) {
                return Err(Error::invalid(format!("{} extends beyond its file", segment.name)));
            }
        }
        let cache = (config.host_cache_bytes > 0).then(|| HostCache::new(config.host_cache_bytes));
        let shared = Arc::new(Shared {
            table: Arc::new(table),
            segments: segments.into_iter().map(Arc::new).collect(),
            config,
            cache,
            stats: Mutex::new(IoStats::default()),
            busy: Mutex::new(Busy {
                in_flight: 0,
                since: None,
            }),
            queue: Mutex::new(Queue {
                tasks: VecDeque::new(),
                background: VecDeque::new(),
                closed: false,
            }),
            available: Condvar::new(),
        });
        let mut threads = Vec::with_capacity(config.workers);
        for k in 0..config.workers {
            let shared = Arc::clone(&shared);
            let thread = std::thread::Builder::new()
                .name(format!("weightsift-io-{k}"))
                .spawn(move || worker(shared))
                .map_err(|e| Error::Internal(format!("cannot start a worker: {e}")))?;
            threads.push(thread);
        }
        Ok(Engine {
            shared,
            threads: Mutex::new(threads),
        })
    }

    pub fn config(&self) -> EngineConfig {
        self.shared.config
    }

    pub fn segment(&self, id: u32) -> Result<&Segment> {
        self.shared
            .segments
            .get(id as usize)
            .map(|s| s.as_ref())
            .ok_or_else(|| Error::invalid(format!("no segment {id}")))
    }

    pub fn files(&self) -> &FileTable {
        &self.shared.table
    }

    pub fn cache(&self) -> Option<&Arc<HostCache>> {
        self.shared.cache.as_ref()
    }

    pub fn stats(&self) -> IoStats {
        let mut stats = self.shared.stats.lock().unwrap().clone();
        let busy = self.shared.busy.lock().unwrap();
        if let Some(since) = busy.since {
            stats.busy_ns += since.elapsed().as_nanos() as u64;
        }
        stats
    }

    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.shared.cache.as_ref().map(|c| c.stats())
    }

    /// Start a new window of counters; with `record_ranges`, every extent read is recorded (file id, offset, length).
    pub fn reset_stats(&self, record_ranges: bool) {
        *self.shared.stats.lock().unwrap() = IoStats {
            ranges: record_ranges.then(Vec::new),
            ..IoStats::default()
        };
        let mut busy = self.shared.busy.lock().unwrap();
        if busy.since.is_some() {
            busy.since = Some(Instant::now());
        }
        drop(busy);
        if let Some(cache) = &self.shared.cache {
            cache.reset_stats();
        }
    }

    /// The plan of a request, as Python's `plan_reads` makes it (for tests and audits).
    pub fn plan(&self, request: &Request) -> Result<Plan> {
        let segment = self.segment(request.segment)?;
        let rows = check_rows(segment, request.rows.as_deref())?;
        plan_reads(
            segment,
            rows.as_deref(),
            request.positions.as_deref(),
            &self.shared.config.plan,
        )
    }

    /// Start moving `requests` into `slots`. Slots must all be the same size, a multiple of the alignment (and
    /// aligned with direct I/O); gather buffers, when every slot has one, at least as large. Pieces are delivered in
    /// order by `Job::next`; each slot must be released before the engine reuses it.
    pub fn submit(&self, requests: &[Request], slots: Vec<SlotBuffers>, use_cache: bool) -> Result<Job> {
        let shared = &self.shared;
        if shared.queue.lock().unwrap().closed {
            return Err(Error::Closed);
        }
        let slot_bytes = check_slots(&slots, &shared.config)?;
        let gather = slots.iter().all(|s| s.compact.is_some());
        // Check every request before any cache lookup (a lookup may reserve cache bytes).
        let mut checked = Vec::with_capacity(requests.len());
        for request in requests {
            let segment = Arc::clone(self.segment_arc(request.segment)?);
            let rows = check_rows(&segment, request.rows.as_deref())?.unwrap_or_else(|| (0..segment.rows).collect());
            let positions: Vec<i64> = match &request.positions {
                Some(positions) => {
                    check_positions(rows.len(), positions)?;
                    positions.clone()
                }
                None => (0..rows.len() as i64).collect(),
            };
            checked.push((segment, rows, positions));
        }
        let cache = if use_cache { shared.cache.clone() } else { None };
        // First every row of every request is looked up (hits leased, so that the misses admitted next never evict a
        // row this job is about to copy), then the misses are admitted in order (ds4: protect every hit first).
        let mut cached_rows = Vec::new();
        let mut missing: Vec<Vec<(u64, i64, bool)>> = Vec::with_capacity(requests.len()); // row, position, probed
        let mut seen: HashSet<(u32, u64)> = HashSet::new();
        for (index, (segment, rows, positions)) in checked.iter().enumerate() {
            let id = requests[index].segment;
            let mut missed = Vec::new();
            for (&row, &position) in rows.iter().zip(positions) {
                // A row asked twice in one job is read twice: waiting on the job's own load could deadlock its slots.
                let probed = match &cache {
                    Some(cache) if seen.insert((id, row)) => Some(cache.probe((id, row), segment.row_bytes)),
                    _ => None,
                };
                let request = index as u32;
                match probed {
                    Some(Probe::Hit(entry)) => cached_rows.push(CachedRow {
                        request,
                        position: position as u64,
                        row_bytes: segment.row_bytes,
                        source: CachedSource::Entry(entry),
                    }),
                    Some(Probe::Wait(pending)) => cached_rows.push(CachedRow {
                        request,
                        position: position as u64,
                        row_bytes: segment.row_bytes,
                        source: CachedSource::Pending(pending, id, row),
                    }),
                    Some(Probe::Miss) => missed.push((row, position, true)),
                    None => missed.push((row, position, false)),
                }
            }
            missing.push(missed);
        }
        let mut misses = Vec::with_capacity(requests.len());
        for (index, missed) in missing.into_iter().enumerate() {
            let id = requests[index].segment;
            let row_bytes = checked[index].0.row_bytes;
            let mut fills: HashMap<u64, Arc<FillBuffer>> = HashMap::new();
            let mut missed_rows = Vec::with_capacity(missed.len());
            let mut missed_positions = Vec::with_capacity(missed.len());
            for (row, position, probed) in missed {
                let admitted = match (probed, &cache) {
                    (true, Some(cache)) => cache.fill((id, row), row_bytes),
                    _ => Lookup::Bypass,
                };
                let request = index as u32;
                let source = match admitted {
                    Lookup::Fill(fill) => {
                        fills.insert(position as u64, FillBuffer::new(fill));
                        None
                    }
                    Lookup::Bypass => None,
                    // Another job admitted the row since the probe: it is served from the cache after all.
                    Lookup::Hit(entry) => Some(CachedSource::Entry(entry)),
                    Lookup::Wait(pending) => Some(CachedSource::Pending(pending, id, row)),
                };
                match source {
                    Some(source) => cached_rows.push(CachedRow {
                        request,
                        position: position as u64,
                        row_bytes,
                        source,
                    }),
                    None => {
                        missed_rows.push(row);
                        missed_positions.push(position);
                    }
                }
            }
            misses.push((missed_rows, missed_positions, fills));
        }
        let mut specs: Vec<PieceSpec> = Vec::new();
        let mut works: Vec<PieceWork> = Vec::new();
        // Hits first: the cached rows, back to back in slot-sized pieces (a row may continue in the next piece).
        for (ops, work) in pack_cached(cached_rows, slot_bytes, shared.config.copy_chunk_bytes) {
            specs.push(PieceSpec {
                ops,
                gathered_bytes: 0,
                extents: Vec::new(),
            });
            works.push(work);
        }
        // Then each request's misses, planned and pieced as Python plans them (all rows missed: the same plan).
        let mut counted = Vec::with_capacity(requests.len());
        for (index, (missed_rows, missed_positions, fills)) in misses.into_iter().enumerate() {
            let (segment, rows, _) = &checked[index];
            let plan = plan_reads(
                segment,
                Some(&missed_rows),
                Some(&missed_positions),
                &shared.config.plan,
            )?;
            counted.push((
                requests[index].segment,
                rows.len() as u64,
                rows.len() as u64 * segment.row_bytes,
                plan.blocks_4k(),
            ));
            for piece in pieces(&plan, slot_bytes) {
                let (ops, gathers) = copy_ops(&piece.parts, shared.config.direct_copy_bytes, gather);
                let gathered_bytes = gathers.iter().map(|g| g.length).sum();
                let mut reads = Vec::new();
                let mut extents = Vec::new();
                let mut at = 0;
                for extent in &piece.extents {
                    let file = segment.files[extent.file as usize];
                    extents.push((file, extent.offset, extent.length));
                    chunked(extent.length, shared.config.max_read_bytes, |start, length| {
                        reads.push(ReadCall {
                            file,
                            offset: extent.offset + start,
                            length,
                            slot_offset: at + start,
                            segment: requests[index].segment,
                        });
                    });
                    at += extent.length;
                }
                let admits = admissions(&piece.parts, segment.row_bytes, &fills, shared.config.copy_chunk_bytes);
                let ops = ops
                    .into_iter()
                    .map(|op| Op {
                        source: op.source,
                        src: op.src,
                        request: index as u32,
                        dst: op.dst,
                        length: op.length,
                    })
                    .collect();
                specs.push(PieceSpec {
                    ops,
                    gathered_bytes,
                    extents,
                });
                works.push(PieceWork {
                    reads,
                    admits,
                    gathers,
                    ..PieceWork::default()
                });
            }
        }
        {
            let mut stats = shared.stats.lock().unwrap();
            for &(segment, rows, logical, blocks) in &counted {
                stats.requests += 1;
                stats.rows += rows;
                stats.logical_bytes += logical;
                stats.blocks_4k += blocks;
                let entry = stats.by_segment.entry(segment).or_default();
                entry.requests += 1;
                entry.rows += rows;
                entry.logical_bytes += logical;
            }
        }
        let core = Arc::new(JobCore {
            shared: Arc::clone(shared),
            slots: slots.clone(),
            pieces: specs,
            state: Mutex::new(JobState {
                next_start: 0,
                // Slot 0 first: a stack popped from the end.
                free_slots: (0..slots.len()).rev().collect(),
                pieces: works
                    .into_iter()
                    .map(|work| PieceState {
                        work: Some(work),
                        ..PieceState::default()
                    })
                    .collect(),
                next_deliver: 0,
                error: None,
                outstanding: 0,
            }),
            changed: Condvar::new(),
            prefetch: false,
        });
        core.start_pieces();
        Ok(Job { core })
    }

    /// Load the rows of `requests` (positions ignored) into the host cache in the background: rows a request will ask
    /// for soon. Cached rows are promoted (and leased meanwhile), rows being loaded are left to their load, the others
    /// are admitted now (their bytes reserved) and read at the lowest priority into the prefetch's own staging, then
    /// copied into their entries. Nothing here counts as a lookup; the cache counts every prefetched row as used or
    /// wasted. Needs a host cache.
    pub fn prefetch(&self, requests: &[Request]) -> Result<Prefetch> {
        let shared = &self.shared;
        if shared.queue.lock().unwrap().closed {
            return Err(Error::Closed);
        }
        let cache = shared
            .cache
            .clone()
            .ok_or_else(|| Error::invalid("prefetching needs a host cache"))?;
        let mut checked = Vec::with_capacity(requests.len());
        for request in requests {
            let segment = Arc::clone(self.segment_arc(request.segment)?);
            let rows = check_rows(&segment, request.rows.as_deref())?.unwrap_or_else(|| (0..segment.rows).collect());
            checked.push((request.segment, segment, rows));
        }
        let mut leases = Vec::new();
        let mut absent: Vec<Vec<u64>> = Vec::with_capacity(checked.len());
        for (id, _, rows) in &checked {
            let mut missing = Vec::new();
            for &row in rows {
                let mut is_absent = false;
                if let Some(entry) = cache.prefetch_probe((*id, row), &mut is_absent) {
                    leases.push(entry);
                }
                if is_absent {
                    missing.push(row);
                }
            }
            absent.push(missing);
        }
        let alignment = shared.config.plan.alignment;
        let slot_bytes = PREFETCH_SLOT_BYTES.div_ceil(alignment) * alignment;
        let (mut specs, mut works) = (Vec::new(), Vec::new());
        let (mut loaded, mut loaded_bytes, mut blocks) = (0u64, 0u64, 0u64);
        for ((id, segment, _), missing) in checked.iter().zip(absent) {
            let mut fills: HashMap<u64, Arc<FillBuffer>> = HashMap::new();
            let mut rows = Vec::new();
            for row in missing {
                if let Some(fill) = cache.prefetch_fill((*id, row), segment.row_bytes) {
                    fills.insert(rows.len() as u64, FillBuffer::new(fill));
                    rows.push(row);
                }
            }
            if rows.is_empty() {
                continue;
            }
            loaded += rows.len() as u64;
            loaded_bytes += rows.len() as u64 * segment.row_bytes;
            let plan = plan_reads(segment, Some(&rows), None, &shared.config.plan)?;
            blocks += plan.blocks_4k();
            for piece in pieces(&plan, slot_bytes) {
                let mut reads = Vec::new();
                let mut extents = Vec::new();
                let mut at = 0;
                for extent in &piece.extents {
                    let file = segment.files[extent.file as usize];
                    extents.push((file, extent.offset, extent.length));
                    chunked(extent.length, shared.config.max_read_bytes, |start, length| {
                        reads.push(ReadCall {
                            file,
                            offset: extent.offset + start,
                            length,
                            slot_offset: at + start,
                            segment: *id,
                        });
                    });
                    at += extent.length;
                }
                let admits = admissions(&piece.parts, segment.row_bytes, &fills, shared.config.copy_chunk_bytes);
                specs.push(PieceSpec {
                    ops: Vec::new(),
                    gathered_bytes: 0,
                    extents,
                });
                works.push(PieceWork {
                    reads,
                    admits,
                    ..PieceWork::default()
                });
            }
        }
        drop(leases);
        {
            let mut stats = shared.stats.lock().unwrap();
            stats.prefetches += 1;
            stats.prefetch_rows += loaded;
            stats.prefetch_bytes += loaded_bytes;
            stats.prefetch_blocks_4k += blocks;
        }
        if specs.is_empty() {
            return Ok(Prefetch {
                core: None,
                buffers: Vec::new(),
                rows: 0,
            });
        }
        let mut buffers: Vec<AlignedBuffer> = (0..PREFETCH_SLOTS)
            .map(|_| AlignedBuffer::new(slot_bytes as usize, DIRECT_ALIGNMENT as usize))
            .collect();
        let slots: Vec<SlotBuffers> = buffers
            .iter_mut()
            .map(|b| SlotBuffers {
                staging: RawBuffer::from_slice(b.as_mut_slice()),
                compact: None,
            })
            .collect();
        let core = Arc::new(JobCore {
            shared: Arc::clone(shared),
            slots: slots.clone(),
            pieces: specs,
            state: Mutex::new(JobState {
                next_start: 0,
                free_slots: (0..slots.len()).rev().collect(),
                pieces: works
                    .into_iter()
                    .map(|work| PieceState {
                        work: Some(work),
                        ..PieceState::default()
                    })
                    .collect(),
                next_deliver: 0,
                error: None,
                outstanding: 0,
            }),
            changed: Condvar::new(),
            prefetch: true,
        });
        core.start_pieces();
        Ok(Prefetch {
            core: Some(core),
            buffers,
            rows: loaded,
        })
    }

    fn segment_arc(&self, id: u32) -> Result<&Arc<Segment>> {
        self.shared
            .segments
            .get(id as usize)
            .ok_or_else(|| Error::invalid(format!("no segment {id}")))
    }

    /// Read rows of a segment from storage (the cache is neither used nor filled) into `out`, the request's output
    /// (`output_rows × row_bytes`), on the pool, through two temporary slots.
    pub fn read_rows(&self, request: &Request, out: RawBuffer) -> Result<()> {
        let plan = self.plan(request)?;
        if out.len() as u64 != plan.output_rows * plan.row_bytes {
            return Err(Error::invalid(format!(
                "the output holds {} bytes, the request {}",
                out.len(),
                plan.output_rows * plan.row_bytes
            )));
        }
        let alignment = self.shared.config.plan.alignment;
        let slot_bytes = (plan.physical_bytes().clamp(alignment, 32 << 20)).div_ceil(alignment) * alignment;
        let mut buffers: Vec<(AlignedBuffer, AlignedBuffer)> = (0..2)
            .map(|_| {
                (
                    AlignedBuffer::new(slot_bytes as usize, DIRECT_ALIGNMENT as usize),
                    AlignedBuffer::new(slot_bytes as usize, 64),
                )
            })
            .collect();
        let slots = buffers
            .iter_mut()
            .map(|(s, c)| SlotBuffers {
                staging: RawBuffer::from_slice(s.as_mut_slice()),
                compact: Some(RawBuffer::from_slice(c.as_mut_slice())),
            })
            .collect();
        let job = self.submit(std::slice::from_ref(request), slots, false)?;
        let outcome = (|| -> Result<()> {
            while let Some(piece) = job.next()? {
                let slot = job.core.slots[piece.slot];
                for op in &piece.ops {
                    let source = match op.source {
                        Source::Staging => slot.staging,
                        Source::Compact => slot.compact.expect("gathered parts have a gather buffer"),
                    };
                    // SAFETY: the piece is delivered (its writes finished) and not yet released; `out` is the caller's
                    // and written here only, at the disjoint ranges the plan gives.
                    unsafe {
                        out.slice_mut(op.dst as usize, op.length as usize)
                            .copy_from_slice(source.slice(op.src as usize, op.length as usize));
                    }
                }
                job.release(piece.slot)?;
            }
            Ok(())
        })();
        job.close();
        drop(buffers);
        outcome
    }

    /// Stop the workers (in-flight reads finish, queued tasks fail their jobs with `Closed`) and release the host cache.
    pub fn close(&self) {
        let pending: Vec<Task> = {
            let mut queue = self.shared.queue.lock().unwrap();
            queue.closed = true;
            let mut pending: Vec<Task> = queue.tasks.drain(..).collect();
            pending.extend(queue.background.drain(..));
            pending
        };
        self.shared.available.notify_all();
        for task in pending {
            let job = Arc::clone(&task.job);
            job.fail(Error::Closed);
            job.task_finished(task.piece, task.phase, Ok(()));
        }
        let threads: Vec<JoinHandle<()>> = self.threads.lock().unwrap().drain(..).collect();
        for thread in threads {
            let _ = thread.join();
        }
        // A closed engine serves nothing again: its cache's memory goes back now, not when the last reference to the
        // engine goes (a row still leased by an unclosed job is freed with that job).
        if let Some(cache) = &self.shared.cache {
            cache.clear();
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.close();
    }
}

fn check_slots(slots: &[SlotBuffers], config: &EngineConfig) -> Result<u64> {
    let first = slots
        .first()
        .ok_or_else(|| Error::invalid("a job needs at least one staging slot"))?;
    let slot_bytes = first.staging.len() as u64;
    let alignment = config.plan.alignment;
    if slot_bytes == 0 || slot_bytes % alignment != 0 {
        return Err(Error::invalid(
            "staging slots must be a positive multiple of the alignment",
        ));
    }
    for slot in slots {
        if slot.staging.len() as u64 != slot_bytes {
            return Err(Error::invalid("staging slots must all have the same size"));
        }
        if config.direct && slot.staging.addr() as u64 % DIRECT_ALIGNMENT != 0 {
            return Err(Error::invalid("direct reads need aligned staging slots"));
        }
        if slot.compact.is_some_and(|c| (c.len() as u64) < slot_bytes) {
            return Err(Error::invalid("a gather buffer must be at least as large as its slot"));
        }
    }
    Ok(slot_bytes)
}

/// Python's checks of output positions: one non-negative position per row, all distinct.
fn check_positions(count: usize, positions: &[i64]) -> Result<()> {
    if positions.len() != count || positions.iter().any(|&p| p < 0) {
        return Err(Error::invalid("one non-negative position per requested row"));
    }
    let mut sorted = positions.to_vec();
    sorted.sort_unstable();
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(Error::invalid("positions must be distinct"));
    }
    Ok(())
}

/// The copies of a piece's parts into the cache entries of the rows `fills` admitted (by output row), in chunks.
fn admissions(parts: &[Part], row_bytes: u64, fills: &HashMap<u64, Arc<FillBuffer>>, chunk: u64) -> Vec<AdmitCall> {
    let mut admits = Vec::new();
    if fills.is_empty() {
        return admits;
    }
    for part in parts {
        let (begin, end) = (part.output, part.output + part.length);
        for position in begin / row_bytes..=(end - 1) / row_bytes {
            let Some(buffer) = fills.get(&position) else {
                continue;
            };
            let (row_begin, row_end) = (position * row_bytes, (position + 1) * row_bytes);
            let (b, e) = (begin.max(row_begin), end.min(row_end));
            chunked(e.saturating_sub(b), chunk, |offset, length| {
                admits.push(AdmitCall {
                    buffer: Arc::clone(buffer),
                    from: part.staging + (b - begin) + offset,
                    to: b - row_begin + offset,
                    length,
                });
            });
        }
    }
    admits
}

fn chunked(length: u64, chunk: u64, mut each: impl FnMut(u64, u64)) {
    let mut offset = 0;
    while offset < length {
        let n = chunk.min(length - offset);
        each(offset, n);
        offset += n;
    }
}

// Pieces and their work.

struct PieceSpec {
    ops: Vec<Op>,
    gathered_bytes: u64,
    /// The extents the piece reads (file id, offset, length): counted and recorded when it starts.
    extents: Vec<(u32, u64, u64)>,
}

#[derive(Default)]
struct PieceWork {
    reads: Vec<ReadCall>,
    copies: Vec<CopyCall>,
    waits: Vec<WaitCall>,
    gathers: Vec<Gather>,
    admits: Vec<AdmitCall>,
}

#[derive(Clone, Copy)]
struct ReadCall {
    file: u32,
    offset: u64,
    length: u64,
    slot_offset: u64,
    segment: u32,
}

struct CopyCall {
    entry: Arc<Entry>,
    from: u64,
    length: u64,
    slot_offset: u64,
}

struct WaitCall {
    pending: Arc<Pending>,
    segment: u32,
    row: u64,
    from: u64,
    length: u64,
    slot_offset: u64,
}

struct AdmitCall {
    buffer: Arc<FillBuffer>,
    from: u64,
    to: u64,
    length: u64,
}

enum CachedSource {
    Entry(Arc<Entry>),
    Pending(Arc<Pending>, u32, u64),
}

struct CachedRow {
    request: u32,
    position: u64,
    row_bytes: u64,
    source: CachedSource,
}

/// Cached rows back to back in slot-sized pieces: each piece's host copies (in chunks), and its device copies, merged
/// where both the slot and the request's output are contiguous.
fn pack_cached(rows: Vec<CachedRow>, slot_bytes: u64, chunk: u64) -> Vec<(Vec<Op>, PieceWork)> {
    let mut out: Vec<(Vec<Op>, PieceWork)> = Vec::new();
    let mut used = slot_bytes; // the first row opens a piece
    for row in rows {
        let mut from = 0;
        while from < row.row_bytes {
            if used == slot_bytes {
                out.push((Vec::new(), PieceWork::default()));
                used = 0;
            }
            let length = (row.row_bytes - from).min(slot_bytes - used);
            let (ops, work) = out.last_mut().unwrap();
            let dst = row.position * row.row_bytes + from;
            match ops.last_mut() {
                Some(op) if op.request == row.request && op.src + op.length == used && op.dst + op.length == dst => {
                    op.length += length
                }
                _ => ops.push(Op {
                    source: Source::Staging,
                    src: used,
                    request: row.request,
                    dst,
                    length,
                }),
            }
            match &row.source {
                CachedSource::Entry(entry) => chunked(length, chunk, |offset, n| {
                    work.copies.push(CopyCall {
                        entry: Arc::clone(entry),
                        from: from + offset,
                        length: n,
                        slot_offset: used + offset,
                    })
                }),
                CachedSource::Pending(pending, segment, index) => work.waits.push(WaitCall {
                    pending: Arc::clone(pending),
                    segment: *segment,
                    row: *index,
                    from,
                    length,
                    slot_offset: used,
                }),
            }
            from += length;
            used += length;
        }
    }
    out
}

/// A row being loaded into the cache: written by admission copies (disjoint ranges), completed by the last one.
struct FillBuffer {
    data: Mutex<Option<Box<[u8]>>>,
    view: RawBuffer,
    remaining: AtomicU64,
    fill: Mutex<Option<Fill>>,
}

impl FillBuffer {
    fn new(mut fill: Fill) -> Arc<Self> {
        // An evicted entry's memory when there is one (already touched: no page faults); new memory otherwise.
        let mut data = fill
            .take_buffer()
            .unwrap_or_else(|| vec![0u8; fill.nbytes() as usize].into_boxed_slice());
        let view = RawBuffer::from_slice(&mut data);
        let remaining = AtomicU64::new(fill.nbytes());
        Arc::new(FillBuffer {
            data: Mutex::new(Some(data)),
            view,
            remaining,
            fill: Mutex::new(Some(fill)),
        })
    }

    /// Account `length` bytes written; the last write completes the fill.
    fn wrote(&self, length: u64) -> Result<()> {
        if self.remaining.fetch_sub(length, Ordering::AcqRel) == length {
            let data = self.data.lock().unwrap().take();
            let fill = self.fill.lock().unwrap().take();
            if let (Some(data), Some(fill)) = (data, fill) {
                fill.complete(data)?;
            }
        }
        Ok(())
    }

    fn abort(&self, error: Error) {
        if let Some(fill) = self.fill.lock().unwrap().take() {
            fill.abort(error);
        }
    }
}

// Jobs.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Waiting,
    Main,
    Gathering,
    Ready,
    Delivered,
}

#[derive(Default)]
struct PieceState {
    work: Option<PieceWork>,
    slot: Option<usize>,
    phase: Phase,
    main_left: usize,
    gathers_left: usize,
    admits_left: usize,
    released: bool,
    gathers: Vec<Gather>,
    admits: Vec<AdmitCall>,
}

struct JobState {
    next_start: usize,
    free_slots: Vec<usize>,
    pieces: Vec<PieceState>,
    next_deliver: usize,
    error: Option<Error>,
    /// Tasks queued, running, or waiting for another job's load.
    outstanding: usize,
}

struct JobCore {
    shared: Arc<Shared>,
    slots: Vec<SlotBuffers>,
    pieces: Vec<PieceSpec>,
    state: Mutex<JobState>,
    changed: Condvar,
    /// A prefetch: its reads wait behind every foreground task, and its pieces release their slots themselves.
    prefetch: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TaskPhase {
    Main,
    Gather,
    Admit,
}

enum Work {
    Read(ReadCall),
    Copy(CopyCall),
    Fallback(WaitCall),
    Gather(Gather),
    Admit(AdmitCall),
}

struct Task {
    job: Arc<JobCore>,
    piece: usize,
    slot: usize,
    phase: TaskPhase,
    work: Work,
}

impl JobCore {
    fn lock(&self) -> MutexGuard<'_, JobState> {
        self.state.lock().unwrap()
    }

    fn push(self: &Arc<Self>, tasks: Vec<Task>, urgent: bool) {
        if tasks.is_empty() {
            return;
        }
        let mut queue = self.shared.queue.lock().unwrap();
        if queue.closed {
            drop(queue);
            self.fail(Error::Closed);
            for task in tasks {
                self.task_finished(task.piece, task.phase, Ok(()));
            }
            return;
        }
        let count = tasks.len();
        if urgent {
            for task in tasks.into_iter().rev() {
                queue.tasks.push_front(task);
            }
        } else if self.prefetch {
            queue.background.extend(tasks);
        } else {
            queue.tasks.extend(tasks);
        }
        drop(queue);
        if count == 1 {
            self.shared.available.notify_one();
        } else {
            self.shared.available.notify_all();
        }
    }

    /// Give free slots to the next pieces and queue their work.
    fn start_pieces(self: &Arc<Self>) {
        let mut tasks = Vec::new();
        let mut waits: Vec<(usize, usize, WaitCall)> = Vec::new();
        let mut started = Vec::new();
        {
            let mut state = self.lock();
            while state.error.is_none() && state.next_start < state.pieces.len() {
                let Some(slot) = state.free_slots.pop() else { break };
                let index = state.next_start;
                state.next_start += 1;
                let piece = &mut state.pieces[index];
                let work = piece.work.take().unwrap_or_default();
                piece.slot = Some(slot);
                piece.phase = Phase::Main;
                piece.main_left = work.reads.len() + work.copies.len() + work.waits.len();
                piece.gathers = work.gathers;
                piece.admits = work.admits;
                let count = piece.main_left;
                state.outstanding += count;
                let task = |phase, work| Task {
                    job: Arc::clone(self),
                    piece: index,
                    slot,
                    phase,
                    work,
                };
                tasks.extend(work.reads.into_iter().map(|r| task(TaskPhase::Main, Work::Read(r))));
                tasks.extend(work.copies.into_iter().map(|c| task(TaskPhase::Main, Work::Copy(c))));
                waits.extend(work.waits.into_iter().map(|w| (index, slot, w)));
                started.push((index, count == 0));
            }
        }
        if started.iter().any(|&(index, _)| !self.pieces[index].extents.is_empty()) {
            // Extents are counted (and recorded) when their piece is read, as Python counts them.
            let mut stats = self.shared.stats.lock().unwrap();
            for &(index, _) in &started {
                let extents = &self.pieces[index].extents;
                stats.extents += extents.len() as u64;
                if let Some(ranges) = stats.ranges.as_mut() {
                    ranges.extend(extents.iter().copied());
                }
            }
        }
        self.push(tasks, false);
        for (piece, slot, wait) in waits {
            let job = Arc::clone(self);
            let pending = Arc::clone(&wait.pending);
            pending.on_done(move |outcome| {
                let work = match outcome {
                    Ok(entry) => Work::Copy(CopyCall {
                        entry,
                        from: wait.from,
                        length: wait.length,
                        slot_offset: wait.slot_offset,
                    }),
                    Err(_) => Work::Fallback(wait),
                };
                let task = Task {
                    job: Arc::clone(&job),
                    piece,
                    slot,
                    phase: TaskPhase::Main,
                    work,
                };
                job.push(vec![task], false);
            });
        }
        for (index, empty) in started {
            if empty {
                self.main_done(index);
            }
        }
    }

    fn fail(self: &Arc<Self>, error: Error) {
        self.abandon(error);
        // Its queued tasks are dropped now rather than skipped when their turn comes (a prefetch's may wait long).
        let removed: Vec<Task> = {
            let mut guard = self.shared.queue.lock().unwrap();
            let queue = &mut *guard;
            let mut removed = Vec::new();
            for tasks in [&mut queue.tasks, &mut queue.background] {
                let (mine, others): (VecDeque<Task>, VecDeque<Task>) =
                    tasks.drain(..).partition(|t| Arc::ptr_eq(&t.job, self));
                *tasks = others;
                removed.extend(mine);
            }
            removed
        };
        for task in removed {
            self.task_finished(task.piece, task.phase, Ok(()));
        }
    }

    /// Record the error and abort the job's unfinished cache loads.
    fn abandon(&self, error: Error) {
        let buffers: Vec<Arc<FillBuffer>> = {
            let mut state = self.lock();
            if state.error.is_none() {
                state.error = Some(error.clone());
            }
            state
                .pieces
                .iter_mut()
                .flat_map(|p| {
                    let mut admits: Vec<AdmitCall> = p.admits.drain(..).collect();
                    if let Some(work) = p.work.take() {
                        admits.extend(work.admits);
                    }
                    admits
                })
                .map(|a| a.buffer)
                .collect()
        };
        for buffer in buffers {
            buffer.abort(error.clone());
        }
        self.changed.notify_all();
    }

    fn failed(&self) -> bool {
        self.lock().error.is_some()
    }

    /// A task ended (or was skipped).
    fn task_finished(self: &Arc<Self>, piece: usize, phase: TaskPhase, outcome: Result<()>) {
        if let Err(error) = outcome {
            self.fail(error);
        }
        let mut next = None;
        let idle = {
            let mut state = self.lock();
            state.outstanding -= 1;
            let failed = state.error.is_some();
            let p = &mut state.pieces[piece];
            match phase {
                TaskPhase::Main => {
                    p.main_left -= 1;
                    if p.main_left == 0 && !failed {
                        next = Some(TaskPhase::Main);
                    }
                }
                TaskPhase::Gather => {
                    p.gathers_left -= 1;
                    if p.gathers_left == 0 && !failed {
                        next = Some(TaskPhase::Gather);
                    }
                }
                TaskPhase::Admit => {
                    p.admits_left -= 1;
                    if p.admits_left == 0 && p.released {
                        if let Some(slot) = p.slot {
                            state.free_slots.push(slot);
                            next = Some(TaskPhase::Admit);
                        }
                    }
                }
            }
            state.outstanding == 0
        };
        match next {
            Some(TaskPhase::Main) => self.main_done(piece),
            Some(TaskPhase::Gather) => self.ready(piece),
            Some(TaskPhase::Admit) => self.start_pieces(),
            None => {}
        }
        if idle {
            self.changed.notify_all(); // `close` waits for this
        }
    }

    fn main_done(self: &Arc<Self>, piece: usize) {
        let (slot, gathers) = {
            let mut state = self.lock();
            let p = &mut state.pieces[piece];
            p.phase = Phase::Gathering;
            let gathers: Vec<Gather> = std::mem::take(&mut p.gathers);
            p.gathers_left = gathers.len();
            let slot = p.slot.unwrap();
            state.outstanding += gathers.len();
            (slot, gathers)
        };
        if gathers.is_empty() {
            self.ready(piece);
            return;
        }
        let tasks = gathers
            .into_iter()
            .map(|g| Task {
                job: Arc::clone(self),
                piece,
                slot,
                phase: TaskPhase::Gather,
                work: Work::Gather(g),
            })
            .collect();
        self.push(tasks, true);
    }

    fn ready(self: &Arc<Self>, piece: usize) {
        let (slot, admits, free) = {
            let mut state = self.lock();
            let p = &mut state.pieces[piece];
            // A prefetch has no caller to deliver to: its piece is released at once, its slot freed after its admissions.
            p.phase = if self.prefetch { Phase::Delivered } else { Phase::Ready };
            p.released = self.prefetch;
            let admits: Vec<AdmitCall> = std::mem::take(&mut p.admits);
            p.admits_left = admits.len();
            let slot = p.slot.unwrap();
            let free = self.prefetch && admits.is_empty();
            state.outstanding += admits.len();
            if free {
                state.free_slots.push(slot);
            }
            (slot, admits, free)
        };
        self.changed.notify_all();
        if free {
            self.start_pieces();
        }
        let tasks = admits
            .into_iter()
            .map(|a| Task {
                job: Arc::clone(self),
                piece,
                slot,
                phase: TaskPhase::Admit,
                work: Work::Admit(a),
            })
            .collect();
        self.push(tasks, true);
    }
}

fn worker(shared: Arc<Shared>) {
    let mut handles = Handles::new(Arc::clone(&shared.table));
    loop {
        let task = {
            let mut queue = shared.queue.lock().unwrap();
            loop {
                if let Some(task) = queue.tasks.pop_front().or_else(|| queue.background.pop_front()) {
                    break task;
                }
                if queue.closed {
                    return;
                }
                queue = shared.available.wait(queue).unwrap();
            }
        };
        let Task {
            job,
            piece,
            slot,
            phase,
            work,
        } = task;
        let outcome = if job.failed() {
            Ok(()) // skipped: the job failed or was cancelled
        } else {
            match catch_unwind(AssertUnwindSafe(|| run(&shared, &mut handles, &job, slot, work))) {
                Ok(outcome) => outcome,
                Err(panic) => {
                    let message = panic
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| panic.downcast_ref::<String>().cloned());
                    Err(Error::Internal(message.unwrap_or_else(|| "a worker panicked".into())))
                }
            }
        };
        job.task_finished(piece, phase, outcome);
    }
}

fn run(shared: &Shared, handles: &mut Handles, job: &JobCore, slot: usize, work: Work) -> Result<()> {
    let buffers = job.slots[slot];
    match work {
        Work::Read(read) => {
            // SAFETY: a piece's extents occupy disjoint ranges of its slot (back to back), and the slot is handed to
            // the caller only after every read of the piece finished.
            let target = unsafe {
                buffers
                    .staging
                    .slice_mut(read.slot_offset as usize, read.length as usize)
            };
            let count = read_checked(shared, handles, read.file, read.offset, target)?;
            let mut stats = shared.stats.lock().unwrap();
            stats.physical_bytes += count;
            stats.read_calls += 1;
            stats.by_segment.entry(read.segment).or_default().physical_bytes += count;
        }
        Work::Copy(copy) => {
            // SAFETY: as for reads: a piece's copies have disjoint ranges of its slot.
            let target = unsafe {
                buffers
                    .staging
                    .slice_mut(copy.slot_offset as usize, copy.length as usize)
            };
            target.copy_from_slice(&copy.entry.bytes()[copy.from as usize..(copy.from + copy.length) as usize]);
            shared.stats.lock().unwrap().cache_copied_bytes += copy.length;
        }
        Work::Fallback(wait) => {
            // SAFETY: the waited row's range of the slot is this task's alone.
            let target = unsafe {
                buffers
                    .staging
                    .slice_mut(wait.slot_offset as usize, wait.length as usize)
            };
            fallback_read(shared, handles, wait.segment, wait.row, wait.from, target)?;
        }
        Work::Gather(gather) => {
            let compact = buffers
                .compact
                .ok_or_else(|| Error::Internal("a gather without a gather buffer".into()))?;
            // SAFETY: gathers read staging the piece's reads finished writing, and write disjoint ranges of the gather
            // buffer.
            unsafe {
                compact
                    .slice_mut(gather.dst as usize, gather.length as usize)
                    .copy_from_slice(buffers.staging.slice(gather.src as usize, gather.length as usize));
            }
            shared.stats.lock().unwrap().gathered_bytes += gather.length;
        }
        Work::Admit(admit) => {
            // SAFETY: admissions read staging the piece's reads finished writing (nothing writes the slot until they
            // end), and write disjoint ranges of a cache entry that nothing reads before the last of them completes it.
            unsafe {
                admit
                    .buffer
                    .view
                    .slice_mut(admit.to as usize, admit.length as usize)
                    .copy_from_slice(buffers.staging.slice(admit.from as usize, admit.length as usize));
            }
            shared.stats.lock().unwrap().admitted_bytes += admit.length;
            admit.buffer.wrote(admit.length)?;
        }
    }
    Ok(())
}

/// A positioned read with Python's short-read rule: fewer bytes are an error unless the file ends.
fn read_checked(shared: &Shared, handles: &mut Handles, file: u32, offset: u64, target: &mut [u8]) -> Result<u64> {
    let length = target.len() as u64;
    let started = shared.begin_read();
    let outcome = handles.read_at(file, offset, target);
    shared.end_read(started);
    let count = outcome? as u64;
    let size = shared.table.files[file as usize].size;
    if count < length && offset + count < size.min(offset + length) {
        let path = shared.table.files[file as usize].path.display().to_string();
        return Err(Error::ShortRead {
            path,
            offset,
            wanted: length,
            got: count,
        });
    }
    Ok(count)
}

/// Bytes `[from, from + target.len())` of a row read from storage, for a request whose wait on another job's load
/// failed: the row's own plan, read into a temporary buffer.
fn fallback_read(
    shared: &Shared,
    handles: &mut Handles,
    segment: u32,
    row: u64,
    from: u64,
    target: &mut [u8],
) -> Result<()> {
    let segment = &shared.segments[segment as usize];
    let plan = plan_reads(segment, Some(&[row]), None, &shared.config.plan)?;
    let mut buffer = AlignedBuffer::new(plan.physical_bytes() as usize, DIRECT_ALIGNMENT as usize);
    let mut starts = Vec::with_capacity(plan.extents.len());
    let mut at = 0u64;
    let mut physical = 0;
    let mut calls = 0;
    for extent in &plan.extents {
        starts.push(at);
        let file = segment.files[extent.file as usize];
        let mut start = 0;
        while start < extent.length {
            let length = shared.config.max_read_bytes.min(extent.length - start);
            let slice = &mut buffer.as_mut_slice()[(at + start) as usize..(at + start + length) as usize];
            physical += read_checked(shared, handles, file, extent.offset + start, slice)?;
            calls += 1;
            start += length;
        }
        at += extent.length;
    }
    let (want_begin, want_end) = (from, from + target.len() as u64);
    for run in &plan.runs {
        let (b, e) = (run.output.max(want_begin), (run.output + run.length).min(want_end));
        if b < e {
            let source =
                starts[run.extent as usize] + run.offset - plan.extents[run.extent as usize].offset + (b - run.output);
            target[(b - want_begin) as usize..(e - want_begin) as usize]
                .copy_from_slice(&buffer.as_slice()[source as usize..(source + e - b) as usize]);
        }
    }
    let mut stats = shared.stats.lock().unwrap();
    stats.physical_bytes += physical;
    stats.read_calls += calls;
    stats.extents += plan.extents.len() as u64;
    stats.fallback_rows += 1;
    if let Some(ranges) = stats.ranges.as_mut() {
        ranges.extend(
            plan.extents
                .iter()
                .map(|e| (segment.files[e.file as usize], e.offset, e.length)),
        );
    }
    Ok(())
}

/// A transfer started by `Engine::submit`.
pub struct Job {
    core: Arc<JobCore>,
}

impl Job {
    /// The next piece once it is ready (blocks); `None` when every piece was delivered.
    pub fn next(&self) -> Result<Option<Delivered>> {
        let core = &self.core;
        let mut state = core.lock();
        loop {
            if let Some(error) = &state.error {
                return Err(error.clone());
            }
            let index = state.next_deliver;
            if index >= state.pieces.len() {
                return Ok(None);
            }
            if state.pieces[index].phase == Phase::Ready {
                state.pieces[index].phase = Phase::Delivered;
                state.next_deliver += 1;
                let slot = state.pieces[index].slot.expect("a ready piece has a slot");
                let spec = &core.pieces[index];
                return Ok(Some(Delivered {
                    slot,
                    ops: spec.ops.clone(),
                    gathered_bytes: spec.gathered_bytes,
                }));
            }
            state = core.changed.wait(state).unwrap();
        }
    }

    /// The pieces of this job, delivered or not.
    pub fn pieces(&self) -> usize {
        self.core.pieces.len()
    }

    /// The caller is done with `slot` (its device copies finished): the engine may refill it.
    pub fn release(&self, slot: usize) -> Result<()> {
        let free = {
            let mut state = self.core.lock();
            let found = state
                .pieces
                .iter()
                .position(|p| p.slot == Some(slot) && p.phase == Phase::Delivered && !p.released);
            let Some(index) = found else {
                return Err(Error::invalid(format!("slot {slot} is not held by the caller")));
            };
            let piece = &mut state.pieces[index];
            piece.released = true;
            let free = piece.admits_left == 0;
            if free {
                state.free_slots.push(slot);
            }
            free
        };
        if free {
            self.core.start_pieces();
        }
        Ok(())
    }

    /// Stop the job: queued tasks are skipped, `next` returns `Cancelled`, unfinished cache loads are aborted.
    pub fn cancel(&self) {
        self.core.fail(Error::Cancelled);
    }

    /// Cancel an unfinished job, then wait until none of its tasks runs or waits: its slots are no longer touched.
    pub fn close(&self) {
        let unfinished = {
            let state = self.core.lock();
            state.error.is_none() && state.next_deliver < state.pieces.len()
        };
        if unfinished {
            self.cancel();
        }
        let mut state = self.core.lock();
        while state.outstanding > 0 {
            state = self.core.changed.wait(state).unwrap();
        }
    }

    /// The slots the job writes (the caller's buffers).
    pub fn slots(&self) -> &[SlotBuffers] {
        &self.core.slots
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        self.close();
    }
}

/// Rows being loaded into the host cache by `Engine::prefetch`.
pub struct Prefetch {
    core: Option<Arc<JobCore>>,
    /// The prefetch's staging: kept until none of its tasks can touch it (`close`).
    buffers: Vec<AlignedBuffer>,
    rows: u64,
}

impl Prefetch {
    /// The rows this prefetch loads (the others were cached or being loaded already, or not admitted).
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// Wait until every row is in the cache (or the prefetch failed).
    pub fn wait(&self) -> Result<()> {
        let Some(core) = &self.core else { return Ok(()) };
        let mut state = core.lock();
        loop {
            if let Some(error) = &state.error {
                return Err(error.clone());
            }
            if state.next_start == state.pieces.len() && state.outstanding == 0 {
                return Ok(());
            }
            state = core.changed.wait(state).unwrap();
        }
    }

    /// Stop: unread rows are not loaded (their reservations are released; a request waiting for one reads it).
    pub fn cancel(&self) {
        if let Some(core) = &self.core {
            core.fail(Error::Cancelled);
        }
    }

    /// Cancel if unfinished, then wait until none of its tasks runs; its staging is then released.
    pub fn close(&mut self) {
        if let Some(core) = self.core.take() {
            let unfinished = {
                let state = core.lock();
                state.error.is_none() && (state.next_start < state.pieces.len() || state.outstanding > 0)
            };
            if unfinished {
                core.fail(Error::Cancelled);
            }
            let mut state = core.lock();
            while state.outstanding > 0 {
                state = core.changed.wait(state).unwrap();
            }
        }
        self.buffers.clear();
    }
}

impl Drop for Prefetch {
    fn drop(&mut self) {
        self.close();
    }
}
