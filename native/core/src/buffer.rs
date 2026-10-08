//! Host memory the engine writes into without owning it: a caller's staging slot (pinned memory allocated by
//! PyTorch), whose bytes Python later copies to the device.
//!
//! This is the core's only `unsafe` type, and its demonstrated requirement is zero copy: reads land directly in the
//! pinned memory the device copies from (decision 0007 removed a host copy of every byte for this reason). Its users
//! are the engine's tasks (six `unsafe` blocks: reads, cache copies, fallback reads, gathers and admissions into or out
//! of a slot, and `Engine::read_rows`'s copy out) and the binding's wrapping of a Python buffer (one), each stating the
//! invariant it relies on. The invariants, upheld by the engine and its binding:
//!
//!   * the memory is live, writable and `len` bytes long for as long as any task may touch it (the binding keeps the
//!     owning Python buffer alive until the job that uses it has no task in flight);
//!   * tasks write disjoint ranges (the planner places every extent, part and gather at its own range of a slot), and
//!     nothing reads a range while a task writes it (a slot is handed to Python only after all its writes finished,
//!     and refilled only after Python released it).

#[derive(Clone, Copy, Debug)]
pub struct RawBuffer {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: a RawBuffer is an address and a length; the invariants above make the ranges each thread touches disjoint.
unsafe impl Send for RawBuffer {}
unsafe impl Sync for RawBuffer {}

impl RawBuffer {
    /// # Safety
    /// `ptr` must point to `len` writable bytes that stay valid while the buffer is used (see the module docs).
    pub unsafe fn new(ptr: *mut u8, len: usize) -> Self {
        RawBuffer { ptr, len }
    }

    /// A buffer over a slice the caller keeps alive and does not otherwise touch while the buffer is used.
    pub fn from_slice(slice: &mut [u8]) -> Self {
        RawBuffer {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn addr(&self) -> usize {
        self.ptr as usize
    }

    fn check(&self, offset: usize, len: usize) {
        assert!(
            offset.checked_add(len).is_some_and(|end| end <= self.len),
            "range {offset}+{len} outside a buffer of {}",
            self.len
        );
    }

    /// # Safety
    /// No other live reference may overlap `[offset, offset + len)` (module invariants).
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice_mut(&self, offset: usize, len: usize) -> &mut [u8] {
        self.check(offset, len);
        std::slice::from_raw_parts_mut(self.ptr.add(offset), len)
    }

    /// # Safety
    /// No task may write `[offset, offset + len)` while the slice lives (module invariants).
    pub unsafe fn slice(&self, offset: usize, len: usize) -> &[u8] {
        self.check(offset, len);
        std::slice::from_raw_parts(self.ptr.add(offset), len)
    }
}
