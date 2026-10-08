//! Read plans and their staging pieces: Python's `awpmi.storage.store.plan_reads` and `PageStreamer._pieces`
//! (decisions 0006 and 0007), mirrored exactly so that the native backend issues the same reads of the same extents.
//!
//! A plan groups the requested rows' bytes into runs (one contiguous byte range of a file each, sorted by file and
//! offset), widens every run to the I/O alignment and merges runs whose aligned ranges touch or lie within `max_gap`
//! of each other into extents, which hold at most `max_extent_bytes` unless runs share a block. Pieces split a plan
//! into staging-slot-sized pieces: extents read back to back into a slot, and the parts of runs they hold. Every
//! function here is pure; the tests compare them with the Python planner on random requests.

use crate::error::{Error, Result};
use crate::layout::Segment;

/// The 4 KiB block of the block accounting (decisions 0003, 0004), independent of a plan's alignment.
pub const IO_BLOCK_BYTES: u64 = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanConfig {
    pub alignment: u64,
    pub max_gap: u64,
    pub max_extent_bytes: u64,
}

impl PlanConfig {
    pub fn validate(&self) -> Result<()> {
        if self.alignment == 0 || self.max_gap % self.alignment != 0 || self.max_extent_bytes < self.alignment {
            return Err(Error::invalid("bad alignment, gap or extent limit"));
        }
        Ok(())
    }
}

/// A run of a plan: `length` bytes at `offset` of the segment's file `file` (index into `Segment::files`), going to
/// byte `output` of the request's output, read as part of extent `extent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanRun {
    pub file: u32,
    pub offset: u64,
    pub length: u64,
    pub output: u64,
    pub extent: u32,
}

/// An aligned byte range of the segment's file `file` that a plan reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent {
    pub file: u32,
    pub offset: u64,
    pub length: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub runs: Vec<PlanRun>,
    pub extents: Vec<Extent>,
    pub row_count: u64,
    pub output_rows: u64,
    pub row_bytes: u64,
}

impl Plan {
    pub fn logical_bytes(&self) -> u64 {
        self.row_count * self.row_bytes
    }

    pub fn physical_bytes(&self) -> u64 {
        self.extents.iter().map(|e| e.length).sum()
    }

    /// Distinct 4 KiB blocks that hold a requested byte (consecutive runs of one file share at most a boundary block).
    pub fn blocks_4k(&self) -> u64 {
        let mut blocks = 0;
        let mut previous: Option<(u32, u64)> = None;
        for run in &self.runs {
            let first = run.offset / IO_BLOCK_BYTES;
            let last = (run.offset + run.length - 1) / IO_BLOCK_BYTES;
            blocks += last - first + 1;
            if previous == Some((run.file, first)) {
                blocks -= 1;
            }
            previous = Some((run.file, last));
        }
        blocks
    }
}

/// Checked rows of a request: `None` is every row. As Python's `check_rows`: in range, ascending, unique.
pub fn check_rows(segment: &Segment, rows: Option<&[i64]>) -> Result<Option<Vec<u64>>> {
    let Some(rows) = rows else { return Ok(None) };
    if let (Some(&first), Some(&last)) = (rows.first(), rows.last()) {
        if first < 0 || last as u64 >= segment.rows {
            return Err(Error::Index(format!(
                "{}: rows out of range [0, {})",
                segment.name, segment.rows
            )));
        }
        if rows.windows(2).any(|pair| pair[1] <= pair[0]) {
            return Err(Error::invalid(format!(
                "{}: rows must be ascending and unique",
                segment.name
            )));
        }
    }
    Ok(Some(rows.iter().map(|&r| r as u64).collect()))
}

/// The plan of a read of `rows` (checked; `None`: every row) of `segment`, request i written to output row
/// `positions[i]` (default i). Mirrors Python's `plan_reads`.
pub fn plan_reads(
    segment: &Segment,
    rows: Option<&[u64]>,
    positions: Option<&[i64]>,
    config: &PlanConfig,
) -> Result<Plan> {
    config.validate()?;
    let all: Vec<u64>;
    let rows = match rows {
        Some(rows) => rows,
        None => {
            if positions.is_some() {
                return Err(Error::invalid("positions need explicit rows"));
            }
            all = (0..segment.rows).collect();
            &all
        }
    };
    let count = rows.len() as u64;
    let (positions, output_rows) = match positions {
        None => ((0..count).collect::<Vec<u64>>(), count),
        Some(positions) => {
            if positions.len() as u64 != count || positions.iter().any(|&p| p < 0) {
                return Err(Error::invalid("one non-negative position per requested row"));
            }
            let positions: Vec<u64> = positions.iter().map(|&p| p as u64).collect();
            let mut sorted = positions.clone();
            sorted.sort_unstable();
            if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(Error::invalid("positions must be distinct"));
            }
            let output_rows = sorted.last().map_or(0, |&p| p + 1);
            (positions, output_rows)
        }
    };
    let mut found = segment.byte_runs(rows, &positions);
    let empty = Plan {
        runs: Vec::new(),
        extents: Vec::new(),
        row_count: count,
        output_rows,
        row_bytes: segment.row_bytes,
    };
    if found.is_empty() {
        return Ok(empty);
    }
    // Sort by (file, offset), stably: a plain segment's runs (already ascending) keep their order.
    found.sort_by_key(|run| (run.file, run.offset));
    for pair in found.windows(2) {
        if pair[0].file == pair[1].file && pair[1].offset < pair[0].offset + pair[0].length {
            return Err(Error::invalid(format!(
                "{}: the requested rows overlap in their file",
                segment.name
            )));
        }
    }
    let alignment = config.alignment;
    let begin: Vec<u64> = found.iter().map(|r| r.offset / alignment * alignment).collect();
    let end: Vec<u64> = found
        .iter()
        .map(|r| (r.offset + r.length).div_ceil(alignment) * alignment)
        .collect();
    let mut new = vec![true; found.len()];
    for k in 1..found.len() {
        new[k] = found[k].file != found[k - 1].file || begin[k] > end[k - 1] + config.max_gap;
    }
    // Extents: each group's first begin and last end; then, if one is too long, the greedy re-grouping.
    let mut extent_of_run = Vec::with_capacity(found.len());
    let mut bounds: Vec<(u64, u64)> = Vec::new();
    for k in 0..found.len() {
        if new[k] {
            bounds.push((begin[k], end[k]));
        } else {
            bounds.last_mut().unwrap().1 = end[k];
        }
        extent_of_run.push(bounds.len() - 1);
    }
    if bounds.iter().any(|&(b, e)| e - b > config.max_extent_bytes) {
        // Two runs that share an aligned block always stay together: extents never overlap, no block is read twice.
        extent_of_run.clear();
        bounds.clear();
        for k in 0..found.len() {
            let split = match bounds.last() {
                None => true,
                Some(&(extent_begin, extent_end)) => {
                    new[k] || (end[k] - extent_begin > config.max_extent_bytes && begin[k] >= extent_end)
                }
            };
            if split {
                bounds.push((begin[k], end[k]));
            } else {
                let last = bounds.last_mut().unwrap();
                last.1 = last.1.max(end[k]);
            }
            extent_of_run.push(bounds.len() - 1);
        }
    }
    let mut extents: Vec<Extent> = bounds
        .iter()
        .map(|&(b, e)| Extent {
            file: 0,
            offset: b,
            length: e - b,
        })
        .collect();
    for (run, &extent) in found.iter().zip(&extent_of_run) {
        extents[extent].file = run.file;
    }
    let runs = found
        .iter()
        .zip(&extent_of_run)
        .map(|(r, &extent)| PlanRun {
            file: r.file,
            offset: r.offset,
            length: r.length,
            output: r.output,
            extent: extent as u32,
        })
        .collect();
    Ok(Plan { runs, extents, ..empty })
}

/// Bytes `length` at `staging` of a slot that go to byte `output` of the request's output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Part {
    pub staging: u64,
    pub length: u64,
    pub output: u64,
}

/// A slot-sized piece of a plan: extents read back to back from the start of a slot, and the parts of runs they hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    pub extents: Vec<Extent>,
    pub parts: Vec<Part>,
}

/// Python's `PageStreamer._pieces`: extents longer than a slot are chunked (each chunk a piece of its own, its runs
/// clipped at chunk boundaries); the others are packed in order while they fit.
pub fn pieces(plan: &Plan, slot_bytes: u64) -> Vec<Piece> {
    if plan.extents.is_empty() {
        return Vec::new();
    }
    // (extent, alone) and the runs re-indexed to the possibly chunked extents.
    let mut extents: Vec<(Extent, bool)> = Vec::with_capacity(plan.extents.len());
    let mut runs: Vec<(u64, u64, u64, usize)> = Vec::with_capacity(plan.runs.len()); // offset, length, output, extent
    let mut index = 0;
    for (id, extent) in plan.extents.iter().enumerate() {
        let first = index;
        while index < plan.runs.len() && plan.runs[index].extent as usize == id {
            index += 1;
        }
        let mine = &plan.runs[first..index];
        if extent.length <= slot_bytes {
            extents.push((*extent, false));
            runs.extend(mine.iter().map(|r| (r.offset, r.length, r.output, extents.len() - 1)));
            continue;
        }
        let stop = extent.offset + extent.length;
        let mut start = extent.offset;
        while start < stop {
            let chunk_end = (start + slot_bytes).min(stop);
            extents.push((
                Extent {
                    file: extent.file,
                    offset: start,
                    length: chunk_end - start,
                },
                true,
            ));
            for run in mine {
                let (b, e) = (run.offset.max(start), (run.offset + run.length).min(chunk_end));
                if b < e {
                    runs.push((b, e - b, run.output + b - run.offset, extents.len() - 1));
                }
            }
            start = chunk_end;
        }
    }
    let mut piece_of_extent = Vec::with_capacity(extents.len());
    let mut position = Vec::with_capacity(extents.len());
    let (mut piece, mut used, mut closed) = (0usize, 0u64, false);
    for &(extent, alone) in &extents {
        if used > 0 && (closed || alone || used + extent.length > slot_bytes) {
            piece += 1;
            used = 0;
        }
        piece_of_extent.push(piece);
        position.push(used);
        used += extent.length;
        closed = alone;
    }
    let mut out: Vec<Piece> = (0..=piece)
        .map(|_| Piece {
            extents: Vec::new(),
            parts: Vec::new(),
        })
        .collect();
    for (k, &(extent, _)) in extents.iter().enumerate() {
        out[piece_of_extent[k]].extents.push(extent);
    }
    for &(offset, length, output, extent) in &runs {
        let staging = position[extent] + offset - extents[extent].0.offset;
        out[piece_of_extent[extent]].parts.push(Part {
            staging,
            length,
            output,
        });
    }
    out
}

/// Where a copy to the device reads from: the slot's staging buffer, or its gather buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Staging,
    Compact,
}

/// One host-to-device copy of a piece: `length` bytes from `src` of `source` to byte `dst` of the request's output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyOp {
    pub source: Source,
    pub src: u64,
    pub dst: u64,
    pub length: u64,
}

/// A host gather of a short part: `length` bytes from `src` of staging to `dst` of the gather buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gather {
    pub src: u64,
    pub dst: u64,
    pub length: u64,
}

/// Python's `PageStreamer._move_piece`: with several parts, those of at least `direct_copy_bytes` are copied straight
/// from staging, one copy each; the rest, if more than one, are gathered back to back into the gather buffer and
/// copied once per range that is contiguous in the destination; a single part is copied from staging. Without a
/// gather buffer every part is copied from staging.
pub fn copy_ops(parts: &[Part], direct_copy_bytes: u64, gather: bool) -> (Vec<CopyOp>, Vec<Gather>) {
    let mut ops = Vec::new();
    let mut gathers = Vec::new();
    if parts.is_empty() {
        return (ops, gathers);
    }
    let rest: Vec<Part> = if parts.len() > 1 {
        for part in parts.iter().filter(|p| p.length >= direct_copy_bytes) {
            ops.push(CopyOp {
                source: Source::Staging,
                src: part.staging,
                dst: part.output,
                length: part.length,
            });
        }
        parts.iter().copied().filter(|p| p.length < direct_copy_bytes).collect()
    } else {
        parts.to_vec()
    };
    if rest.is_empty() {
        return (ops, gathers);
    }
    if rest.len() == 1 || !gather {
        // One part: copied from staging (Python takes the same path, with one transfer per destination range).
        ops.extend(rest.iter().map(|p| CopyOp {
            source: Source::Staging,
            src: p.staging,
            dst: p.output,
            length: p.length,
        }));
        return (ops, gathers);
    }
    let mut within = 0;
    for part in &rest {
        gathers.push(Gather {
            src: part.staging,
            dst: within,
            length: part.length,
        });
        within += part.length;
    }
    let mut start = 0;
    while start < rest.len() {
        let mut stop = start + 1;
        while stop < rest.len() && rest[stop].output == rest[stop - 1].output + rest[stop - 1].length {
            stop += 1;
        }
        let length = rest[start..stop].iter().map(|p| p.length).sum();
        ops.push(CopyOp {
            source: Source::Compact,
            src: gathers[start].dst,
            dst: rest[start].output,
            length,
        });
        start = stop;
    }
    (ops, gathers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{Layout, Segment};

    fn config(alignment: u64, max_gap: u64, max_extent_bytes: u64) -> PlanConfig {
        PlanConfig {
            alignment,
            max_gap,
            max_extent_bytes,
        }
    }

    /// A small deterministic generator (no dependency for tests).
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

    fn brute_blocks(plan: &Plan) -> u64 {
        let mut blocks = std::collections::BTreeSet::new();
        for run in &plan.runs {
            for b in run.offset / IO_BLOCK_BYTES..=(run.offset + run.length - 1) / IO_BLOCK_BYTES {
                blocks.insert((run.file, b));
            }
        }
        blocks.len() as u64
    }

    #[test]
    fn plans_cover_exactly_the_requested_rows() {
        let mut rng = Lcg(7);
        for (alignment, max_gap, max_extent) in [
            (4096, 0, 8 << 20),
            (512, 0, 8 << 20),
            (4096, 8192, 8 << 20),
            (4096, 0, 8192),
        ] {
            for case in 0..200u64 {
                let row_bytes = [1, 7, 292, 1152, 9000][(case % 5) as usize];
                let segment = Segment::plain("s", 0, rng.below(10_000), 3000, row_bytes).unwrap();
                let fraction = [0, 2, 50, 500, 1000][(case % 5) as usize];
                let rows: Vec<u64> = (0..3000).filter(|_| rng.below(1000) < fraction).collect();
                let plan = plan_reads(&segment, Some(&rows), None, &config(alignment, max_gap, max_extent)).unwrap();
                let Layout::Plain { offset: base } = segment.layout else {
                    unreachable!()
                };
                // Runs, in order, are exactly the requested rows' bytes, back to back, each inside its extent.
                let mut covered = Vec::new();
                let mut position = 0;
                for run in &plan.runs {
                    assert_eq!(run.output, position);
                    assert_eq!(run.length % row_bytes, 0);
                    let first = (run.offset - base) / row_bytes;
                    covered.extend(first..first + run.length / row_bytes);
                    position += run.length;
                    let e = plan.extents[run.extent as usize];
                    assert!(e.offset <= run.offset && run.offset + run.length <= e.offset + e.length);
                }
                assert_eq!(covered, rows);
                assert_eq!(plan.logical_bytes(), position);
                for pair in plan.extents.windows(2) {
                    assert!(pair[1].offset >= pair[0].offset + pair[0].length);
                }
                for e in &plan.extents {
                    assert!(e.offset % alignment == 0 && e.length % alignment == 0 && e.length > 0);
                }
                assert_eq!(plan.blocks_4k(), brute_blocks(&plan));
                if alignment == IO_BLOCK_BYTES && max_gap == 0 && max_extent >= 8 << 20 {
                    assert_eq!(plan.physical_bytes(), plan.blocks_4k() * IO_BLOCK_BYTES);
                }
            }
        }
    }

    #[test]
    fn pieces_hold_every_part_once_within_their_slot() {
        let mut rng = Lcg(11);
        for slot in [8192u64, 65536, 1 << 20] {
            for case in 0..100u64 {
                let row_bytes = [292, 4096, 9000, 300_000][(case % 4) as usize];
                let segment = Segment::plain("s", 0, rng.below(5000), 200, row_bytes).unwrap();
                let rows: Vec<u64> = (0..200).filter(|_| rng.below(100) < 30).collect();
                let plan = plan_reads(&segment, Some(&rows), None, &config(4096, 0, 8 << 20)).unwrap();
                let pieces = pieces(&plan, slot);
                let mut bytes = 0;
                for piece in &pieces {
                    let used: u64 = piece.extents.iter().map(|e| e.length).sum();
                    assert!(used <= slot.max(piece.extents.iter().map(|e| e.length).max().unwrap_or(0)));
                    for part in &piece.parts {
                        assert!(part.staging + part.length <= used);
                        bytes += part.length;
                    }
                }
                assert_eq!(bytes, plan.logical_bytes());
            }
        }
    }

    #[test]
    fn rows_and_positions_are_checked() {
        let segment = Segment::plain("s", 0, 0, 10, 8).unwrap();
        assert!(matches!(check_rows(&segment, Some(&[3, 2])), Err(Error::Invalid(_))));
        assert!(matches!(check_rows(&segment, Some(&[2, 2])), Err(Error::Invalid(_))));
        assert!(matches!(check_rows(&segment, Some(&[10])), Err(Error::Index(_))));
        assert!(matches!(check_rows(&segment, Some(&[-1])), Err(Error::Index(_))));
        let c = config(4096, 0, 8 << 20);
        assert!(plan_reads(&segment, Some(&[1, 2]), Some(&[0, 0]), &c).is_err());
        assert!(plan_reads(&segment, Some(&[1, 2]), Some(&[0]), &c).is_err());
        assert!(plan_reads(&segment, None, Some(&[0]), &c).is_err());
        let plan = plan_reads(&segment, Some(&[1, 2]), Some(&[5, 0]), &c).unwrap();
        assert_eq!(plan.output_rows, 6);
    }

    #[test]
    fn copy_ops_follow_the_python_streamer() {
        let parts = [
            Part {
                staging: 0,
                length: 300_000,
                output: 0,
            },
            Part {
                staging: 300_000,
                length: 100,
                output: 300_000,
            },
            Part {
                staging: 400_000,
                length: 100,
                output: 300_100,
            },
            Part {
                staging: 500_000,
                length: 100,
                output: 900_000,
            },
        ];
        let (ops, gathers) = copy_ops(&parts, 256 << 10, true);
        assert_eq!(
            ops[0],
            CopyOp {
                source: Source::Staging,
                src: 0,
                dst: 0,
                length: 300_000
            }
        );
        assert_eq!(gathers.len(), 3);
        assert_eq!(
            ops[1],
            CopyOp {
                source: Source::Compact,
                src: 0,
                dst: 300_000,
                length: 200
            }
        );
        assert_eq!(
            ops[2],
            CopyOp {
                source: Source::Compact,
                src: 200,
                dst: 900_000,
                length: 100
            }
        );
        let (ops, gathers) = copy_ops(&parts[1..2], 256 << 10, true);
        assert!(gathers.is_empty() && ops.len() == 1 && ops[0].source == Source::Staging);
    }
}
