//! Where a segment's rows are: fixed-size rows, each the concatenation of byte spans of the engine's files.
//!
//! The description of Python's `awpmi.storage.layout` (decisions 0006 and 0007). A plain segment is `rows` records of
//! `row_bytes` bytes at an offset of one file (a safetensors tensor as it is). A composed segment's row r is the
//! concatenation of its parts, part p being `part_bytes[p]` bytes at `spans[r][p]` = (file, offset). The core receives
//! these descriptions from its caller (the checkpoint index); it parses no checkpoint format and names no tensor.

use crate::error::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub name: String,
    pub rows: u64,
    pub row_bytes: u64,
    /// The segment's files as engine file ids, in the caller's order (Python: the sorted file keys). Runs are sorted
    /// by their index in this list, then by offset, as Python's planner sorts them.
    pub files: Vec<u32>,
    pub layout: Layout,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Row r is `row_bytes` bytes at `offset + r * row_bytes` of the segment's only file.
    Plain { offset: u64 },
    /// Row r, part p is `part_bytes[p]` bytes at `spans[r * parts + p]` = (index into `files`, offset).
    Composed {
        part_bytes: Vec<u64>,
        spans: Vec<(u32, u64)>,
    },
}

/// `length` bytes at `offset` of the segment's file `file` (an index into `Segment::files`), which go to byte `output`
/// of the request's output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub file: u32,
    pub offset: u64,
    pub length: u64,
    pub output: u64,
}

impl Segment {
    pub fn plain(name: &str, file: u32, offset: u64, rows: u64, row_bytes: u64) -> Result<Self> {
        if rows == 0 || row_bytes == 0 {
            return Err(Error::invalid(format!(
                "{name}: a segment needs rows of at least one byte"
            )));
        }
        rows.checked_mul(row_bytes)
            .and_then(|n| n.checked_add(offset))
            .ok_or_else(|| Error::invalid(format!("{name}: segment extent overflows")))?;
        Ok(Segment {
            name: name.to_owned(),
            rows,
            row_bytes,
            files: vec![file],
            layout: Layout::Plain { offset },
        })
    }

    pub fn composed(
        name: &str,
        files: Vec<u32>,
        rows: u64,
        part_bytes: Vec<u64>,
        spans: Vec<(u32, u64)>,
    ) -> Result<Self> {
        if rows == 0 || part_bytes.is_empty() || part_bytes.contains(&0) {
            return Err(Error::invalid(format!(
                "{name}: a composed segment needs rows and non-empty parts"
            )));
        }
        let parts = part_bytes.len() as u64;
        if spans.len() as u64 != rows * parts {
            return Err(Error::invalid(format!(
                "{name}: expected {rows} rows of {parts} spans, got {} spans",
                spans.len()
            )));
        }
        if spans.iter().any(|&(file, _)| file as usize >= files.len()) {
            return Err(Error::invalid(format!(
                "{name}: a span names a file the segment does not list"
            )));
        }
        let row_bytes = part_bytes.iter().sum();
        Ok(Segment {
            name: name.to_owned(),
            rows,
            row_bytes,
            files,
            layout: Layout::Composed { part_bytes, spans },
        })
    }

    pub fn nbytes(&self) -> u64 {
        self.rows * self.row_bytes
    }

    /// Whether every span lies within its file; `sizes` maps engine file ids to file sizes.
    pub fn within(&self, sizes: &[u64]) -> bool {
        let size = |local: u32| sizes.get(self.files[local as usize] as usize).copied();
        match &self.layout {
            Layout::Plain { offset } => size(0).is_some_and(|n| offset + self.nbytes() <= n),
            Layout::Composed { part_bytes, spans } => spans.chunks(part_bytes.len()).all(|row| {
                row.iter()
                    .zip(part_bytes)
                    .all(|(&(file, offset), &n)| size(file).is_some_and(|s| offset + n <= s))
            }),
        }
    }

    /// The runs of `rows` (ascending, unique, in range: checked by the planner), request i going to output row
    /// `positions[i]`. As Python's `byte_runs`: consecutive pieces whose file bytes and output bytes are both
    /// contiguous form one run; runs come in request order.
    pub fn byte_runs(&self, rows: &[u64], positions: &[u64]) -> Vec<Run> {
        debug_assert_eq!(rows.len(), positions.len());
        let mut runs: Vec<Run> = Vec::new();
        let mut push = |run: Run| {
            if let Some(last) = runs.last_mut() {
                if last.file == run.file
                    && last.offset + last.length == run.offset
                    && last.output + last.length == run.output
                {
                    last.length += run.length;
                    return;
                }
            }
            runs.push(run);
        };
        match &self.layout {
            Layout::Plain { offset } => {
                for (&row, &position) in rows.iter().zip(positions) {
                    push(Run {
                        file: 0,
                        offset: offset + row * self.row_bytes,
                        length: self.row_bytes,
                        output: position * self.row_bytes,
                    });
                }
            }
            Layout::Composed { part_bytes, spans } => {
                let parts = part_bytes.len();
                for (&row, &position) in rows.iter().zip(positions) {
                    let mut start = 0;
                    for (p, &length) in part_bytes.iter().enumerate() {
                        let (file, offset) = spans[row as usize * parts + p];
                        push(Run {
                            file,
                            offset,
                            length,
                            output: position * self.row_bytes + start,
                        });
                        start += length;
                    }
                }
            }
        }
        runs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_runs_merge_consecutive_rows_and_positions() {
        let s = Segment::plain("s", 0, 100, 10, 8).unwrap();
        let runs = s.byte_runs(&[1, 2, 3, 7], &[0, 1, 2, 3]);
        assert_eq!(
            runs,
            vec![
                Run {
                    file: 0,
                    offset: 108,
                    length: 24,
                    output: 0
                },
                Run {
                    file: 0,
                    offset: 156,
                    length: 8,
                    output: 24
                }
            ]
        );
        // Scattered positions break a run even when the rows are consecutive.
        let runs = s.byte_runs(&[1, 2], &[3, 0]);
        assert_eq!(runs.len(), 2);
        assert_eq!(
            runs[0],
            Run {
                file: 0,
                offset: 108,
                length: 8,
                output: 24
            }
        );
    }

    #[test]
    fn composed_runs_merge_adjacent_spans() {
        // Row 0: parts at file 0 offsets 0 and 4 (adjacent); row 1: file 1 offset 10, then file 0 offset 8 (adjacent
        // to row 0).
        let s = Segment::composed("c", vec![5, 6], 2, vec![4, 4], vec![(0, 0), (0, 4), (1, 10), (0, 8)]).unwrap();
        assert_eq!(s.row_bytes, 8);
        let runs = s.byte_runs(&[0, 1], &[0, 1]);
        assert_eq!(
            runs,
            vec![
                Run {
                    file: 0,
                    offset: 0,
                    length: 8,
                    output: 0
                },
                Run {
                    file: 1,
                    offset: 10,
                    length: 4,
                    output: 8
                },
                Run {
                    file: 0,
                    offset: 8,
                    length: 4,
                    output: 12
                },
            ]
        );
        assert!(s.within(&[0, 0, 0, 0, 0, 12, 14]));
        assert!(!s.within(&[0, 0, 0, 0, 0, 11, 14]));
    }

    #[test]
    fn segments_validate() {
        assert!(Segment::plain("x", 0, 0, 0, 8).is_err());
        assert!(Segment::composed("x", vec![0], 2, vec![4], vec![(0, 0)]).is_err());
        assert!(Segment::composed("x", vec![0], 1, vec![4], vec![(1, 0)]).is_err());
        assert!(Segment::composed("x", vec![0], 1, vec![0], vec![(0, 0)]).is_err());
    }
}
