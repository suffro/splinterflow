//! Positioned reads of the engine's files, with or without the OS page cache (Python's `awpmi.storage.fileio`).
//!
//! `direct` bypasses the page cache (Windows `FILE_FLAG_NO_BUFFERING`, Linux `O_DIRECT`): every read goes to the
//! device, and offsets, lengths and buffer addresses must be multiples of `DIRECT_ALIGNMENT`. Positioned reads are the
//! standard library's (`seek_read` on Windows, `read_at` elsewhere); no I/O crate is needed. Each worker thread holds
//! its own handle of every file it reads (`Handles`): a Windows handle opened for synchronous I/O serializes its
//! operations, so threads sharing one would read one at a time.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use crate::error::{Error, Result};

pub const DIRECT_ALIGNMENT: u64 = 4096;
/// One read call reads at most this much (as Python's `_MAX_READ`); the engine's calls are far smaller.
const MAX_CALL: usize = 1 << 30;
#[cfg(windows)]
const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
#[cfg(windows)]
const ERROR_HANDLE_EOF: i32 = 38;

#[derive(Clone, Debug)]
pub struct FileSpec {
    pub key: String,
    pub path: PathBuf,
    pub size: u64,
}

/// The files an engine reads, by id (their order in the table).
#[derive(Debug)]
pub struct FileTable {
    pub files: Vec<FileSpec>,
    pub direct: bool,
}

impl FileTable {
    /// Every file must exist and open (directly, if `direct`); sizes are taken now.
    pub fn open(paths: Vec<(String, PathBuf)>, direct: bool) -> Result<Self> {
        let mut files = Vec::with_capacity(paths.len());
        for (key, path) in paths {
            let shown = path.display().to_string();
            let size = std::fs::metadata(&path).map_err(|e| Error::io(&shown, 0, 0, &e))?.len();
            drop(open_file(&path, direct).map_err(|e| Error::io(&shown, 0, 0, &e))?);
            files.push(FileSpec { key, path, size });
        }
        Ok(FileTable { files, direct })
    }

    pub fn sizes(&self) -> Vec<u64> {
        self.files.iter().map(|f| f.size).collect()
    }
}

fn open_file(path: &std::path::Path, direct: bool) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    if direct {
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            options.custom_flags(FILE_FLAG_NO_BUFFERING);
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_DIRECT);
        }
        #[cfg(not(any(windows, target_os = "linux")))]
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "direct I/O is not supported on this platform",
        ));
    }
    options.open(path)
}

/// One thread's handles of a table's files, opened on first use.
pub struct Handles {
    table: Arc<FileTable>,
    open: Vec<Option<File>>,
}

impl Handles {
    pub fn new(table: Arc<FileTable>) -> Self {
        let open = (0..table.files.len()).map(|_| None).collect();
        Handles { table, open }
    }

    fn handle(&mut self, file: u32) -> Result<&File> {
        let index = file as usize;
        let spec = self
            .table
            .files
            .get(index)
            .ok_or_else(|| Error::invalid(format!("no file {file}")))?;
        if self.open[index].is_none() {
            let opened = open_file(&spec.path, self.table.direct)
                .map_err(|e| Error::io(&spec.path.display().to_string(), 0, 0, &e))?;
            self.open[index] = Some(opened);
        }
        Ok(self.open[index].as_ref().unwrap())
    }

    /// Read `buf.len()` bytes at `offset` of file `file`; returns the bytes read (fewer only at the end of the file).
    pub fn read_at(&mut self, file: u32, offset: u64, buf: &mut [u8]) -> Result<usize> {
        let direct = self.table.direct;
        if direct
            && (offset % DIRECT_ALIGNMENT != 0
                || buf.len() as u64 % DIRECT_ALIGNMENT != 0
                || buf.as_ptr() as u64 % DIRECT_ALIGNMENT != 0)
        {
            return Err(Error::invalid("direct reads need aligned offsets, lengths and buffers"));
        }
        let path_of = |table: &FileTable| table.files[file as usize].path.display().to_string();
        let length = buf.len();
        let mut done = 0;
        while done < length {
            let chunk = (length - done).min(MAX_CALL);
            let handle = self.handle(file)?;
            match read_once(handle, offset + done as u64, &mut buf[done..done + chunk]) {
                Ok(count) => {
                    done += count;
                    if count < chunk {
                        break; // end of file
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(Error::io(&path_of(&self.table), offset + done as u64, chunk as u64, &e)),
            }
        }
        Ok(done)
    }
}

#[cfg(windows)]
fn read_once(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    match file.seek_read(buf, offset) {
        Err(e) if e.raw_os_error() == Some(ERROR_HANDLE_EOF) => Ok(0),
        other => other,
    }
}

#[cfg(unix)]
fn read_once(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buf, offset)
}

/// A host buffer whose address is a multiple of `alignment`, owned by Rust (a `Vec` with slack; no unsafe).
pub struct AlignedBuffer {
    raw: Vec<u8>,
    start: usize,
    len: usize,
}

impl AlignedBuffer {
    pub fn new(len: usize, alignment: usize) -> Self {
        let raw = vec![0u8; len + alignment];
        let start = (alignment - raw.as_ptr() as usize % alignment) % alignment;
        AlignedBuffer { raw, start, len }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.raw[self.start..self.start + self.len]
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.raw[self.start..self.start + self.len]
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_file(name: &str, bytes: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("weightsift-io-{}-{name}", std::process::id()));
        std::fs::File::create(&path).unwrap().write_all(bytes).unwrap();
        path
    }

    #[test]
    fn positioned_reads_buffered_and_direct() {
        let data: Vec<u8> = (0..20_000u32).map(|i| (i * 7 % 251) as u8).collect();
        let path = temp_file("file-rs", &data);
        for direct in [false, true] {
            let table = Arc::new(FileTable::open(vec![("f".into(), path.clone())], direct).unwrap());
            assert_eq!(table.files[0].size, 20_000);
            let mut handles = Handles::new(table);
            let mut buffer = AlignedBuffer::new(8192, 4096);
            let n = handles.read_at(0, 4096, buffer.as_mut_slice()).unwrap();
            assert_eq!(n, 8192);
            assert_eq!(buffer.as_slice(), &data[4096..12288]);
            // The last aligned extent runs past the end of the file: a short read, not an error.
            let n = handles.read_at(0, 16384, buffer.as_mut_slice()).unwrap();
            assert_eq!(n, 20_000 - 16384);
            assert_eq!(&buffer.as_slice()[..n], &data[16384..]);
            if direct {
                let mut small = AlignedBuffer::new(4096, 4096);
                assert!(handles.read_at(0, 1, small.as_mut_slice()).is_err());
                assert!(handles.read_at(0, 0, &mut small.as_mut_slice()[..100]).is_err());
            }
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn missing_files_are_reported() {
        let error = FileTable::open(vec![("x".into(), PathBuf::from("does/not/exist.bin"))], false).unwrap_err();
        assert!(matches!(
            error,
            Error::Io {
                kind: io::ErrorKind::NotFound,
                ..
            }
        ));
    }
}
