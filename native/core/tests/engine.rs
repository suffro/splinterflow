//! The engine against files on disk: every byte delivered equals the file's, with and without the host cache, under
//! eviction, concurrent jobs, cancellation, shutdown and read errors.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use weightsift_io::engine::{Engine, EngineConfig, Request, SlotBuffers};
use weightsift_io::{AlignedBuffer, Error, FileTable, PlanConfig, RawBuffer, Segment, Source};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

struct Fixture {
    dir: PathBuf,
    data: Vec<Vec<u8>>,
    segments: Vec<Segment>,
}

impl Fixture {
    /// Three files; a plain segment of short rows (not on the 4 KiB grid), a plain segment of long rows, and a
    /// composed segment whose rows are three spans scattered over the files.
    fn new(name: &str, seed: u64) -> Self {
        let mut rng = Lcg(seed);
        let dir = std::env::temp_dir().join(format!("weightsift-io-engine-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let sizes = [3_000_000u64, 2_500_000, 1_200_000];
        let data: Vec<Vec<u8>> = sizes
            .iter()
            .map(|&n| (0..n).map(|_| rng.next() as u8).collect())
            .collect();
        for (k, bytes) in data.iter().enumerate() {
            std::fs::File::create(dir.join(format!("f{k}.bin")))
                .unwrap()
                .write_all(bytes)
                .unwrap();
        }
        let short = Segment::plain("short", 0, 1_003, 2_000, 292).unwrap();
        let long = Segment::plain("long", 1, 7, 6, 300_001).unwrap();
        let part_bytes = vec![8_192u64, 4_100, 100];
        let rows = 20u64;
        let mut spans = Vec::new();
        let mut cursor = [50_000u64, 30_000, 10_000];
        for _ in 0..rows {
            for &n in &part_bytes {
                let file = rng.below(3) as usize;
                let gap = [0, 0, 17, 5_000][rng.below(4) as usize];
                spans.push((file as u32, cursor[file] + gap));
                cursor[file] += gap + n;
            }
        }
        let composed = Segment::composed("composed", vec![0, 1, 2], rows, part_bytes, spans).unwrap();
        Fixture {
            dir,
            data,
            segments: vec![short, long, composed],
        }
    }

    fn table(&self, direct: bool) -> FileTable {
        let paths = (0..3)
            .map(|k| (format!("f{k}"), self.dir.join(format!("f{k}.bin"))))
            .collect();
        FileTable::open(paths, direct).unwrap()
    }

    fn engine(&self, direct: bool, cache: u64, workers: usize) -> Engine {
        let config = EngineConfig {
            direct,
            plan: PlanConfig {
                alignment: 4096,
                max_gap: 0,
                max_extent_bytes: 64 << 10,
            },
            max_read_bytes: 16 << 10,
            workers,
            host_cache_bytes: cache,
            direct_copy_bytes: 256 << 10,
            copy_chunk_bytes: 50_000,
        };
        Engine::new(self.table(direct), self.segments.clone(), config).unwrap()
    }

    fn row(&self, segment: usize, row: u64) -> Vec<u8> {
        let s = &self.segments[segment];
        let mut out = Vec::with_capacity(s.row_bytes as usize);
        match &s.layout {
            weightsift_io::layout::Layout::Plain { offset } => {
                let file = s.files[0] as usize;
                let start = (offset + row * s.row_bytes) as usize;
                out.extend_from_slice(&self.data[file][start..start + s.row_bytes as usize]);
            }
            weightsift_io::layout::Layout::Composed { part_bytes, spans } => {
                for (p, &n) in part_bytes.iter().enumerate() {
                    let (file, offset) = spans[row as usize * part_bytes.len() + p];
                    let file = s.files[file as usize] as usize;
                    out.extend_from_slice(&self.data[file][offset as usize..(offset + n) as usize]);
                }
            }
        }
        out
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

struct Slots {
    buffers: Vec<(AlignedBuffer, AlignedBuffer)>,
}

impl Slots {
    fn new(count: usize, bytes: usize) -> Self {
        Slots {
            buffers: (0..count)
                .map(|_| (AlignedBuffer::new(bytes, 4096), AlignedBuffer::new(bytes, 64)))
                .collect(),
        }
    }

    fn views(&mut self) -> Vec<SlotBuffers> {
        self.buffers
            .iter_mut()
            .map(|(s, c)| SlotBuffers {
                staging: RawBuffer::from_slice(s.as_mut_slice()),
                compact: Some(RawBuffer::from_slice(c.as_mut_slice())),
            })
            .collect()
    }
}

/// Run a job to the end, copying every op into per-request outputs, releasing each slot at once.
fn transfer(engine: &Engine, requests: &[Request], slots: &mut Slots, use_cache: bool) -> Result<Vec<Vec<u8>>, Error> {
    let views = slots.views();
    let mut outputs: Vec<Vec<u8>> = requests
        .iter()
        .map(|r| {
            let s = engine.segment(r.segment).unwrap();
            let count = r.rows.as_ref().map_or(s.rows as usize, |rows| rows.len());
            let rows = r
                .positions
                .as_ref()
                .map_or(count, |p| p.iter().max().map_or(0, |&m| m as usize + 1));
            vec![0xEEu8; rows * s.row_bytes as usize]
        })
        .collect();
    let job = engine.submit(requests, views.clone(), use_cache)?;
    while let Some(piece) = job.next()? {
        let slot = views[piece.slot];
        for op in &piece.ops {
            let source = match op.source {
                Source::Staging => slot.staging,
                Source::Compact => slot.compact.unwrap(),
            };
            let bytes = unsafe { source.slice(op.src as usize, op.length as usize) };
            outputs[op.request as usize][op.dst as usize..(op.dst + op.length) as usize].copy_from_slice(bytes);
        }
        job.release(piece.slot)?;
    }
    Ok(outputs)
}

fn expected(fixture: &Fixture, request: &Request, len: usize) -> Vec<u8> {
    let s = &fixture.segments[request.segment as usize];
    let mut out = vec![0xEEu8; len];
    let rows: Vec<i64> = request.rows.clone().unwrap_or_else(|| (0..s.rows as i64).collect());
    for (k, &row) in rows.iter().enumerate() {
        let position = request.positions.as_ref().map_or(k as u64, |p| p[k] as u64);
        let start = (position * s.row_bytes) as usize;
        out[start..start + s.row_bytes as usize].copy_from_slice(&fixture.row(request.segment as usize, row as u64));
    }
    out
}

fn random_request(rng: &mut Lcg, fixture: &Fixture, segment: u32) -> Request {
    let s = &fixture.segments[segment as usize];
    let fraction = [5, 30, 70, 100][rng.below(4) as usize];
    let rows: Vec<i64> = (0..s.rows as i64).filter(|_| rng.below(100) < fraction).collect();
    let positions = if rng.below(2) == 0 {
        None
    } else {
        // A random injective placement into a slightly larger output.
        let mut slots: Vec<i64> = (0..rows.len() as i64 + 3).collect();
        for k in (1..slots.len()).rev() {
            slots.swap(k, rng.below(k as u64 + 1) as usize);
        }
        Some(slots[..rows.len()].to_vec())
    };
    Request {
        segment,
        rows: Some(rows),
        positions,
    }
}

#[test]
fn every_byte_equals_the_file_without_and_with_a_cache() {
    let fixture = Fixture::new("exact", 1);
    for direct in [true, false] {
        for cache in [0u64, 200_000, 50 << 20] {
            let engine = fixture.engine(direct, cache, 4);
            let mut slots = Slots::new(3, 64 << 10);
            let mut rng = Lcg(direct as u64 * 10 + cache);
            for _ in 0..40 {
                let count = 1 + rng.below(3);
                let requests: Vec<Request> = (0..count)
                    .map(|_| {
                        let segment = rng.below(3) as u32;
                        random_request(&mut rng, &fixture, segment)
                    })
                    .collect();
                let outputs = transfer(&engine, &requests, &mut slots, true).unwrap();
                for (request, output) in requests.iter().zip(&outputs) {
                    assert_eq!(
                        output,
                        &expected(&fixture, request, output.len()),
                        "direct={direct} cache={cache}"
                    );
                }
                if let Some(cache) = engine.cache() {
                    assert!(cache.resident_bytes() <= cache.capacity());
                }
            }
            let stats = engine.stats();
            if let Some(cache) = engine.cache_stats() {
                assert_eq!(cache.lookups, cache.hits + cache.waits + cache.misses);
                if engine.config().host_cache_bytes == 50 << 20 {
                    assert!(cache.hits > 0 && cache.evictions == 0);
                }
            } else {
                assert_eq!(stats.cache_copied_bytes, 0);
            }
            assert!(stats.physical_bytes >= stats.blocks_4k.min(1));
        }
    }
}

#[test]
fn without_a_cache_the_reads_are_the_plans_extents() {
    let fixture = Fixture::new("plans", 2);
    let engine = fixture.engine(true, 0, 4);
    let mut slots = Slots::new(2, 64 << 10);
    let mut rng = Lcg(5);
    for _ in 0..30 {
        let segment = rng.below(3) as u32;
        let request = random_request(&mut rng, &fixture, segment);
        engine.reset_stats(true);
        transfer(&engine, std::slice::from_ref(&request), &mut slots, false).unwrap();
        let plan = engine.plan(&request).unwrap();
        let stats = engine.stats();
        let segment = engine.segment(request.segment).unwrap();
        // Every extent read is a planned extent (chunked at the slot size), each once, in plan order.
        let mut planned = Vec::new();
        for extent in &plan.extents {
            let mut start = 0;
            while start < extent.length {
                let n = (extent.length - start).min(64 << 10);
                planned.push((segment.files[extent.file as usize], extent.offset + start, n));
                start += n;
            }
        }
        let ranges = stats.ranges.clone().unwrap();
        if plan.extents.iter().all(|e| e.length <= 64 << 10) {
            assert_eq!(
                ranges,
                plan.extents
                    .iter()
                    .map(|e| (segment.files[e.file as usize], e.offset, e.length))
                    .collect::<Vec<_>>()
            );
        } else {
            assert_eq!(ranges, planned);
        }
        assert_eq!(stats.blocks_4k, plan.blocks_4k());
        assert_eq!(stats.logical_bytes, plan.logical_bytes());
        // Physical bytes are the extents' (less what lies beyond a file's end).
        assert!(stats.physical_bytes <= plan.physical_bytes());
        let reads: u64 = planned.iter().map(|&(_, _, n)| n.div_ceil(16 << 10)).sum();
        assert_eq!(stats.read_calls, reads);
    }
}

#[test]
fn small_caches_evict_and_never_exceed_their_budget() {
    let fixture = Fixture::new("evict", 3);
    let engine = fixture.engine(true, 700_000, 6); // two long rows fit, not three
    let mut slots = Slots::new(2, 64 << 10);
    let mut rng = Lcg(9);
    for _ in 0..60 {
        let segment = [1u32, 2, 0][rng.below(3) as usize];
        let request = random_request(&mut rng, &fixture, segment);
        let output = transfer(&engine, std::slice::from_ref(&request), &mut slots, true)
            .unwrap()
            .remove(0);
        assert_eq!(output, expected(&fixture, &request, output.len()));
        let cache = engine.cache().unwrap();
        assert!(cache.resident_bytes() <= cache.capacity());
        assert!(cache.peak_resident_bytes() <= cache.capacity());
    }
    let stats = engine.cache_stats().unwrap();
    assert!(stats.evictions > 0 && stats.hits > 0, "{stats:?}");
    assert_eq!(stats.lookups, stats.hits + stats.waits + stats.misses);
}

#[test]
fn a_miss_never_evicts_a_hit_of_its_own_job() {
    // A cache of exactly two long rows holding 1 and 2, row 1 the least recently used. A job asking for rows 0 (a miss,
    // looked up first) and 1 (a hit) must keep 1: the miss evicts 2, the only row the job does not use (ds4: protect
    // every hit first). Admitting each miss as it is looked up would evict row 1 before its own lookup.
    let fixture = Fixture::new("protect", 9);
    let row_bytes = fixture.segments[1].row_bytes;
    let engine = fixture.engine(true, 2 * row_bytes, 2);
    let mut slots = Slots::new(2, 64 << 10);
    for rows in [vec![1], vec![2]] {
        let request = Request {
            segment: 1,
            rows: Some(rows),
            positions: None,
        };
        transfer(&engine, std::slice::from_ref(&request), &mut slots, true).unwrap();
    }
    assert_eq!(engine.cache().unwrap().keys_lru_first(), vec![(1, 1), (1, 2)]);
    let request = Request {
        segment: 1,
        rows: Some(vec![0, 1]),
        positions: None,
    };
    let output = transfer(&engine, std::slice::from_ref(&request), &mut slots, true)
        .unwrap()
        .remove(0);
    assert_eq!(output, expected(&fixture, &request, output.len()));
    let cache = engine.cache().unwrap();
    assert_eq!(cache.keys_lru_first(), vec![(1, 1), (1, 0)]);
    let stats = cache.stats();
    assert_eq!(
        (stats.hits, stats.misses, stats.evictions, stats.bypassed),
        (1, 3, 1, 0)
    );
}

#[test]
fn concurrent_jobs_load_each_row_once() {
    let fixture = Arc::new(Fixture::new("concurrent", 4));
    let engine = Arc::new(fixture.engine(true, 64 << 20, 8));
    let rows = fixture.segments[2].rows;
    std::thread::scope(|scope| {
        for t in 0..6u64 {
            let (engine, fixture) = (Arc::clone(&engine), Arc::clone(&fixture));
            scope.spawn(move || {
                let mut slots = Slots::new(3, 64 << 10);
                let mut rng = Lcg(100 + t);
                for _ in 0..20 {
                    let request = random_request(&mut rng, &fixture, 2);
                    let output = transfer(&engine, std::slice::from_ref(&request), &mut slots, true)
                        .unwrap()
                        .remove(0);
                    assert_eq!(output, expected(&fixture, &request, output.len()));
                }
            });
        }
    });
    let stats = engine.cache_stats().unwrap();
    // Room for every row: each row was loaded once at most, every other lookup was a hit or a wait.
    assert!(stats.inserts <= rows, "{stats:?}");
    assert_eq!(stats.evictions, 0);
    assert_eq!(stats.aborted_fills, 0);
    assert_eq!(engine.stats().fallback_rows, 0);
}

#[test]
fn cancelled_and_closed_jobs_stop_cleanly() {
    let fixture = Fixture::new("cancel", 5);
    let engine = fixture.engine(true, 32 << 20, 4);
    let mut slots = Slots::new(2, 64 << 10);
    let views = slots.views();
    let request = Request {
        segment: 1,
        rows: None,
        positions: None,
    };
    // Cancel after the first piece: next() reports it, close() returns, the cache holds no half-loaded row.
    let job = engine
        .submit(std::slice::from_ref(&request), views.clone(), true)
        .unwrap();
    let first = job.next().unwrap().unwrap();
    job.release(first.slot).unwrap();
    job.cancel();
    assert_eq!(job.next().unwrap_err(), Error::Cancelled);
    job.close();
    let cache = engine.cache().unwrap();
    let held = cache.resident_bytes();
    assert!(held <= cache.capacity());
    assert_eq!(
        held % fixture.segments[1].row_bytes,
        0,
        "only whole, completed rows stay"
    );
    // A job dropped unconsumed is cancelled by its drop; then the same rows transfer exactly.
    drop(
        engine
            .submit(std::slice::from_ref(&request), views.clone(), true)
            .unwrap(),
    );
    drop(views);
    let output = transfer(&engine, std::slice::from_ref(&request), &mut slots, true)
        .unwrap()
        .remove(0);
    assert_eq!(output, expected(&fixture, &request, output.len()));
    assert!(engine.cache().unwrap().resident_bytes() > 0);
    // After close, submit fails; the engine's threads are gone, and its cache's memory is released at once (not when the
    // last reference to the engine goes).
    engine.close();
    assert_eq!(
        engine.submit(std::slice::from_ref(&request), slots.views(), true).err(),
        Some(Error::Closed)
    );
    let cache = engine.cache().unwrap();
    assert_eq!((cache.resident_bytes(), cache.len()), (0, 0));
}

#[test]
fn read_errors_reach_the_caller() {
    let fixture = Fixture::new("errors", 6);
    let engine = fixture.engine(false, 0, 2);
    // Truncate the long rows' file after the engine checked it: their reads come back short.
    std::fs::OpenOptions::new()
        .write(true)
        .open(fixture.dir.join("f1.bin"))
        .unwrap()
        .set_len(1000)
        .unwrap();
    let mut slots = Slots::new(2, 64 << 10);
    let request = Request {
        segment: 1,
        rows: Some(vec![2, 3]),
        positions: None,
    };
    let error = transfer(&engine, std::slice::from_ref(&request), &mut slots, true).unwrap_err();
    assert!(matches!(error, Error::ShortRead { .. }), "{error:?}");
    // The engine still serves other files.
    let request = Request {
        segment: 0,
        rows: Some(vec![1, 5, 9]),
        positions: None,
    };
    let output = transfer(&engine, std::slice::from_ref(&request), &mut slots, true)
        .unwrap()
        .remove(0);
    assert_eq!(output, expected(&fixture, &request, output.len()));
}

#[test]
fn bad_requests_are_refused_before_any_read() {
    let fixture = Fixture::new("refuse", 7);
    let engine = fixture.engine(true, 1 << 20, 2);
    let mut slots = Slots::new(2, 64 << 10);
    let views = slots.views();
    let cases = [
        Request {
            segment: 9,
            rows: None,
            positions: None,
        },
        Request {
            segment: 0,
            rows: Some(vec![3, 2]),
            positions: None,
        },
        Request {
            segment: 0,
            rows: Some(vec![5000]),
            positions: None,
        },
        Request {
            segment: 0,
            rows: Some(vec![1, 2]),
            positions: Some(vec![0, 0]),
        },
        Request {
            segment: 0,
            rows: Some(vec![1, 2]),
            positions: Some(vec![0]),
        },
    ];
    for request in &cases {
        assert!(engine
            .submit(std::slice::from_ref(request), views.clone(), true)
            .is_err());
    }
    assert!(matches!(
        engine.submit(&cases[2..3], views.clone(), true).err(),
        Some(Error::Index(_))
    ));
    assert_eq!(engine.stats().physical_bytes, 0);
    assert_eq!(engine.cache().unwrap().resident_bytes(), 0);
    // Misaligned or unequal slots are refused.
    let mut odd = AlignedBuffer::new((64 << 10) + 4096, 4096);
    let shifted = RawBuffer::from_slice(&mut odd.as_mut_slice()[1..(64 << 10) + 1]);
    let bad = vec![SlotBuffers {
        staging: shifted,
        compact: None,
    }];
    assert!(engine
        .submit(
            &[Request {
                segment: 0,
                rows: None,
                positions: None
            }],
            bad,
            true
        )
        .is_err());
}

#[test]
fn read_rows_returns_exact_rows() {
    let fixture = Fixture::new("read-rows", 8);
    let engine = fixture.engine(true, 0, 4);
    let mut rng = Lcg(3);
    for segment in 0..3u32 {
        for _ in 0..5 {
            let request = random_request(&mut rng, &fixture, segment);
            let plan = engine.plan(&request).unwrap();
            let mut out = vec![0xEEu8; (plan.output_rows * plan.row_bytes) as usize];
            engine.read_rows(&request, RawBuffer::from_slice(&mut out)).unwrap();
            assert_eq!(out, expected(&fixture, &request, out.len()));
        }
    }
}

fn all_rows(segment: u32, rows: u64) -> Request {
    Request {
        segment,
        rows: Some((0..rows as i64).collect()),
        positions: None,
    }
}

#[test]
fn prefetched_rows_are_served_from_the_cache_and_counted_as_used() {
    let fixture = Fixture::new("prefetch-used", 10);
    let engine = fixture.engine(true, 64 << 20, 4);
    let rows = fixture.segments[2].rows;
    let request = all_rows(2, rows);
    let prefetch = engine.prefetch(std::slice::from_ref(&request)).unwrap();
    assert_eq!(prefetch.rows(), rows);
    prefetch.wait().unwrap();
    let physical = engine.stats().physical_bytes;
    assert!(physical > 0 && engine.stats().prefetch_rows == rows);
    let mut slots = Slots::new(2, 64 << 10);
    let output = transfer(&engine, std::slice::from_ref(&request), &mut slots, true)
        .unwrap()
        .remove(0);
    assert_eq!(output, expected(&fixture, &request, output.len()));
    // The transfer read nothing: every row came from the prefetch's loads, each counted as used once.
    assert_eq!(engine.stats().physical_bytes, physical);
    let stats = engine.cache_stats().unwrap();
    assert_eq!((stats.lookups, stats.hits, stats.misses), (rows, rows, 0));
    assert_eq!(
        (stats.prefetch_fills, stats.prefetch_used, stats.prefetch_wasted),
        (rows, rows, 0)
    );
    // A second prefetch of the same rows has nothing to load.
    let again = engine.prefetch(std::slice::from_ref(&request)).unwrap();
    assert_eq!(again.rows(), 0);
}

#[test]
fn a_transfer_racing_a_prefetch_reads_every_row_once() {
    let fixture = Fixture::new("prefetch-race", 11);
    for _ in 0..10 {
        let engine = fixture.engine(true, 64 << 20, 4);
        let rows = fixture.segments[1].rows;
        let request = all_rows(1, rows);
        let mut prefetch = engine.prefetch(std::slice::from_ref(&request)).unwrap();
        let mut slots = Slots::new(2, 64 << 10);
        let output = transfer(&engine, std::slice::from_ref(&request), &mut slots, true)
            .unwrap()
            .remove(0);
        assert_eq!(output, expected(&fixture, &request, output.len()));
        prefetch.close();
        let stats = engine.cache_stats().unwrap();
        // Rows the prefetch loaded were waited for or hit; none was loaded twice.
        assert_eq!(stats.inserts, rows);
        assert_eq!(stats.hits + stats.waits + stats.misses, rows);
        assert_eq!(stats.prefetch_fills + stats.misses, rows);
        assert_eq!(stats.prefetch_used, stats.prefetch_fills);
        assert_eq!(engine.stats().fallback_rows, 0);
    }
}

#[test]
fn a_cancelled_prefetch_releases_what_it_did_not_load() {
    let fixture = Fixture::new("prefetch-cancel", 12);
    let engine = fixture.engine(true, 64 << 20, 1);
    let rows = fixture.segments[0].rows;
    let request = all_rows(0, rows);
    let mut prefetch = engine.prefetch(std::slice::from_ref(&request)).unwrap();
    prefetch.cancel();
    prefetch.close();
    let cache = engine.cache().unwrap();
    let held = cache.resident_bytes();
    assert_eq!(held % fixture.segments[0].row_bytes, 0, "only completed rows stay");
    assert!(cache.stats().aborted_fills > 0 || held == rows * fixture.segments[0].row_bytes);
    let mut slots = Slots::new(2, 64 << 10);
    let output = transfer(&engine, std::slice::from_ref(&request), &mut slots, true)
        .unwrap()
        .remove(0);
    assert_eq!(output, expected(&fixture, &request, output.len()));
    assert!(cache.resident_bytes() <= cache.capacity());
}

#[test]
fn prefetched_rows_evicted_unused_are_counted_as_wasted() {
    let fixture = Fixture::new("prefetch-wasted", 13);
    let row_bytes = fixture.segments[1].row_bytes;
    let engine = fixture.engine(true, 2 * row_bytes, 2);
    let first = Request {
        segment: 1,
        rows: Some(vec![0, 1]),
        positions: None,
    };
    engine.prefetch(std::slice::from_ref(&first)).unwrap().wait().unwrap();
    let mut slots = Slots::new(2, 64 << 10);
    let other = Request {
        segment: 1,
        rows: Some(vec![2, 3]),
        positions: None,
    };
    transfer(&engine, std::slice::from_ref(&other), &mut slots, true).unwrap();
    let stats = engine.cache_stats().unwrap();
    assert_eq!(
        (stats.prefetch_fills, stats.prefetch_used, stats.prefetch_wasted),
        (2, 0, 2)
    );
}

#[test]
fn prefetching_needs_a_host_cache() {
    let fixture = Fixture::new("prefetch-none", 14);
    let engine = fixture.engine(true, 0, 2);
    assert!(matches!(engine.prefetch(&[all_rows(0, 3)]), Err(Error::Invalid(_))));
}
