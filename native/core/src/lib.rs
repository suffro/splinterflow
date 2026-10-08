//! weightsift-io: Weightsift's native I/O core (Phase 6A, decision 0012).
//!
//! The performance-critical part of Weightsift's storage path, moved out of Python: read planning (`plan`), positioned
//! direct reads on a pool of threads (`file`, `engine`), a host-RAM cache of rows under a strict byte budget
//! (`cache`), and transfer jobs that fill caller-owned staging slots piece by piece while the caller copies earlier
//! pieces to the device (`engine`). It is model-agnostic: segments are rows of byte spans of files, described by the
//! caller (the checkpoint index); nothing here knows a model, what a tensor means, how many experts exist or how they
//! are chosen. It holds no CUDA: the caller (PyTorch) owns pinned memory, streams and device copies.

pub mod buffer;
pub mod cache;
pub mod engine;
pub mod error;
pub mod file;
pub mod layout;
pub mod plan;

pub use buffer::RawBuffer;
pub use cache::{CacheStats, HostCache};
pub use engine::{Delivered, Engine, EngineConfig, IoStats, Job, Op, Prefetch, Request, SegmentStats, SlotBuffers};
pub use error::{Error, Result};
pub use file::{AlignedBuffer, FileTable, DIRECT_ALIGNMENT};
pub use layout::Segment;
pub use plan::{Plan, PlanConfig, Source};
