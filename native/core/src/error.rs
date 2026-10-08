//! Errors of the native core. They are `Clone`: one failure of a read reaches its job and every job waiting on the
//! same cached row.

use std::fmt;
use std::io;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A request or configuration the core refuses (unknown segment, rows not ascending, misaligned buffer, ...).
    Invalid(String),
    /// Rows outside a segment (Python: IndexError, as `check_rows`).
    Index(String),
    /// An operating-system error of a positioned read or of opening a file.
    Io {
        path: String,
        offset: u64,
        length: u64,
        kind: io::ErrorKind,
        os_code: Option<i32>,
        message: String,
    },
    /// A read returned fewer bytes than the file holds at that offset.
    ShortRead {
        path: String,
        offset: u64,
        wanted: u64,
        got: u64,
    },
    /// The job was cancelled before it finished.
    Cancelled,
    /// The engine was closed.
    Closed,
    /// A worker panicked (a bug of the core); the panic's message.
    Internal(String),
}

impl Error {
    pub fn invalid(message: impl Into<String>) -> Self {
        Error::Invalid(message.into())
    }

    pub fn io(path: &str, offset: u64, length: u64, source: &io::Error) -> Self {
        Error::Io {
            path: path.to_owned(),
            offset,
            length,
            kind: source.kind(),
            os_code: source.raw_os_error(),
            message: source.to_string(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Invalid(message) | Error::Index(message) => write!(f, "{message}"),
            Error::Io {
                path,
                offset,
                length,
                message,
                ..
            } => {
                write!(f, "reading {length} bytes at {offset} of {path}: {message}")
            }
            Error::ShortRead {
                path,
                offset,
                wanted,
                got,
            } => {
                write!(f, "short read of {path} at {offset}: {got} of {wanted} bytes")
            }
            Error::Cancelled => write!(f, "the transfer was cancelled"),
            Error::Closed => write!(f, "the engine is closed"),
            Error::Internal(message) => write!(f, "internal error of the native core: {message}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
