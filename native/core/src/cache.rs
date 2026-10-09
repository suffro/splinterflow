//! The host-RAM tier: rows of segments kept in host memory between requests, under a strict byte budget.
//!
//! A deterministic least-recently-used baseline (the `lru` crate keeps the order; the budget in bytes is this
//! module's), with the patterns reviewed in decision 0012 (ds4, colibri, DwarfStar):
//!
//!   * an entry is exactly one row's bytes, and every byte the cache holds counts against the budget, including a row
//!     being loaded (its bytes are reserved when the load starts): resident bytes never exceed the capacity;
//!   * a row is loaded once however many requests want it at the same time: the first lookup gets a `Fill` ticket
//!     and loads it, the others `Wait` for that load (in-flight deduplication); a fill that fails or is dropped
//!     unfinished hands its waiters the error, and they read the row themselves. A request waits only for a prefetch's
//!     load (Phase 6B, decision 0013): a prefetch never waits for anything, whereas two requests waiting for each other's
//!     loads can each hold the staging the other's load needs (the engine's pieces start in order with a few slots: a
//!     cycle, measured as an occasional hang of six concurrent jobs). A row another request is loading is `Busy`: the
//!     caller reads it itself and does not admit it (counted as a miss and a bypass);
//!   * a request looks up all its rows first (`probe`: hits are promoted and leased), and only then reserves room for
//!     its misses (`fill`), so a miss never evicts a row the same request is about to use;
//!   * entries in use (leased: an `Arc` held by a copy into a staging slot) are never evicted, as ds4 protects every
//!     hit of a request before choosing victims; if a new row would need evicting one, it is not admitted (`Bypass`)
//!     and its request reads it without caching it;
//!   * `admit = false` freezes the contents (DwarfStar's fix for long prefills): hits are served, misses bypass;
//!   * a prefetch (`prefetch_probe`, `prefetch_fill`) loads rows a request will ask for soon without counting as a
//!     lookup; every row it loads is counted once used by a request, or as wasted if evicted (or cleared) unused.
//!
//! Memory (Phase 6B, decision 0013). A row's bytes are held in blocks of the cache's block size, the last part (the
//! row's tail) possibly shorter. An evicted row's whole blocks go to a pool, and the rows admitted next take their
//! blocks from it, whatever the two rows' sizes: once the cache has filled, admitting rows whose size is a multiple of
//! the block allocates and frees nothing. (In Phase 6A an entry was one allocation, reused only by a row of exactly its
//! size: with two row sizes (an expert's gate/up and down rows), about half the evictions freed megabytes inline, a third of a millisecond
//! each.) The pool counts against the budget like everything else the cache holds (`held_bytes`: entries, rows being
//! loaded, aborted loads not yet given back, pooled blocks), and it is trimmed when a row's tail would take that above
//! the capacity. Which rows stay, and every hit, miss, eviction and bypass, depend only on the rows' sizes and the
//! requests, never on the block size or on the pool. Blocks are at least `MIN_BLOCK_BYTES` by default: smaller ones
//! come from the allocator's heap, zeroed one by one (2.2 ms per 11.5 MB row in 512 KiB blocks, measured), where larger
//! ones are fresh pages the system zeroes on first touch.
//!
//! The cache decides which rows stay resident, never what their bytes are: a hit's bytes are the bytes a read of the
//! row returned (the engine tests compare them with direct reads).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use lru::LruCache;

use crate::buffer::RawBuffer;
use crate::error::{Error, Result};

/// A row: (segment id, row index).
pub type Key = (u32, u64);

/// The smallest block size chosen by default (`block_for`).
pub const MIN_BLOCK_BYTES: u64 = 1 << 20;

/// The block size for rows of these sizes: the largest size dividing every row of at least `MIN_BLOCK_BYTES` (those rows
/// are then whole blocks, and an evicted row's memory serves any other; smaller rows are tails whatever the block), or
/// `MIN_BLOCK_BYTES` when that is smaller. Rows of 11 and 5.5 MiB get blocks of 5.5 MiB.
pub fn block_for(row_bytes: impl IntoIterator<Item = u64>) -> u64 {
    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    let common = row_bytes.into_iter().filter(|&n| n >= MIN_BLOCK_BYTES).fold(0, gcd);
    if common >= MIN_BLOCK_BYTES {
        common
    } else {
        MIN_BLOCK_BYTES
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub lookups: u64,
    pub hits: u64,
    pub hit_bytes: u64,
    /// Lookups that found the row being loaded by another request (served once that load completes).
    pub waits: u64,
    pub wait_bytes: u64,
    pub misses: u64,
    pub miss_bytes: u64,
    pub inserts: u64,
    pub insert_bytes: u64,
    pub evictions: u64,
    pub evicted_bytes: u64,
    /// Misses not admitted (no room without evicting a leased entry, a row larger than the cache, or admission frozen).
    pub bypassed: u64,
    pub bypassed_bytes: u64,
    /// Fills that did not complete (their request failed or was cancelled).
    pub aborted_fills: u64,
    /// Fills whose memory was all blocks of evicted rows (no allocation).
    pub recycled: u64,
    /// Bytes of fills' memory taken from evicted rows' blocks, and newly allocated.
    pub recycled_bytes: u64,
    pub allocated_bytes: u64,
    /// Memory given back to the system: rows' tails, blocks trimmed from the pool to keep it within the budget, a
    /// cleared cache.
    pub released_bytes: u64,
    /// Rows a prefetch admitted to load, and their bytes.
    pub prefetch_fills: u64,
    pub prefetch_fill_bytes: u64,
    /// Rows a prefetch found cached or being loaded already (nothing to do), or could not admit.
    pub prefetch_skipped: u64,
    pub prefetch_bypassed: u64,
    /// Prefetched rows a request then used (a hit, or a wait on the prefetch's load), and rows evicted or cleared unused.
    pub prefetch_used: u64,
    pub prefetch_wasted: u64,
}

/// The parts of a row of `len` bytes held in blocks of `block` bytes that cover `[offset, offset + length)`: (part,
/// offset in the part, length, offset in the range).
fn spans(block: u64, len: u64, offset: u64, length: u64) -> impl Iterator<Item = (usize, usize, usize, usize)> {
    assert!(
        offset.checked_add(length).is_some_and(|end| end <= len),
        "range {offset}+{length} outside a row of {len} bytes"
    );
    let end = offset + length;
    let mut at = offset;
    std::iter::from_fn(move || {
        if at >= end {
            return None;
        }
        let within = at % block;
        let n = (block - within).min(end - at);
        let span = (
            (at / block) as usize,
            within as usize,
            n as usize,
            (at - offset) as usize,
        );
        at += n;
        Some(span)
    })
}

/// A row's bytes: whole blocks of the cache's block size, the last part (the tail) possibly shorter.
#[derive(Debug)]
pub struct RowMemory {
    parts: Vec<Box<[u8]>>,
    block: u64,
    len: u64,
}

impl RowMemory {
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The bytes held so far (all of them once allocated).
    fn allocated(&self) -> u64 {
        self.parts.iter().map(|p| p.len() as u64).sum()
    }

    /// Allocate the parts the pool did not give (whole blocks, then the tail); the bytes allocated.
    fn allocate_rest(&mut self) -> u64 {
        let whole = (self.len / self.block) as usize;
        let mut allocated = 0;
        while self.parts.len() < whole {
            self.parts.push(vec![0u8; self.block as usize].into_boxed_slice());
            allocated += self.block;
        }
        let tail = self.len % self.block;
        if tail > 0 && self.parts.len() == whole {
            self.parts.push(vec![0u8; tail as usize].into_boxed_slice());
            allocated += tail;
        }
        allocated
    }

    /// Copy bytes `[from, from + dst.len())` of the row into `dst`.
    pub fn copy_to(&self, from: u64, dst: &mut [u8]) {
        for (part, within, n, at) in spans(self.block, self.len, from, dst.len() as u64) {
            dst[at..at + n].copy_from_slice(&self.parts[part][within..within + n]);
        }
    }

    /// Write `data` at byte `offset` of the row.
    pub fn write(&mut self, offset: u64, data: &[u8]) {
        for (part, within, n, at) in spans(self.block, self.len, offset, data.len() as u64) {
            self.parts[part][within..within + n].copy_from_slice(&data[at..at + n]);
        }
    }

    pub fn to_vec(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.len as usize];
        self.copy_to(0, &mut out);
        out
    }

    fn views(&mut self) -> RowViews {
        RowViews {
            parts: self.parts.iter_mut().map(|p| RawBuffer::from_slice(p)).collect(),
            block: self.block,
            len: self.len,
        }
    }
}

/// Views of a row's memory while it loads: several threads write disjoint ranges of it (the engine's admissions).
#[derive(Clone, Debug)]
pub struct RowViews {
    parts: Vec<RawBuffer>,
    block: u64,
    len: u64,
}

impl RowViews {
    /// The parts covering `[offset, offset + length)` of the row: (part, offset in the part, length, offset in the
    /// range). Writing through a part is `RawBuffer`'s contract: the row's memory is alive while its fill is, and each
    /// range is written by one task only.
    pub fn spans(&self, offset: u64, length: u64) -> impl Iterator<Item = (RawBuffer, usize, usize, usize)> + '_ {
        spans(self.block, self.len, offset, length).map(|(part, within, n, at)| (self.parts[part], within, n, at))
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A cached row's bytes.
#[derive(Debug)]
pub struct Entry {
    memory: RowMemory,
    /// Loaded by a prefetch and not used by a request yet.
    prefetched: AtomicBool,
}

impl Entry {
    pub fn len(&self) -> u64 {
        self.memory.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    /// Copy bytes `[from, from + dst.len())` of the row into `dst`.
    pub fn copy_to(&self, from: u64, dst: &mut [u8]) {
        self.memory.copy_to(from, dst);
    }

    pub fn to_vec(&self) -> Vec<u8> {
        self.memory.to_vec()
    }
}

type Waiter = Box<dyn FnOnce(Result<Arc<Entry>>) + Send>;

enum PendingState {
    Loading(Vec<Waiter>),
    Done(Result<Arc<Entry>>),
}

/// A row being loaded by some request; others wait for its outcome.
pub struct Pending {
    state: Mutex<PendingState>,
    /// Loaded by a prefetch; `claimed` once a request waits for it.
    prefetch: bool,
    claimed: AtomicBool,
}

impl Pending {
    fn new(prefetch: bool) -> Arc<Self> {
        Arc::new(Pending {
            state: Mutex::new(PendingState::Loading(Vec::new())),
            prefetch,
            claimed: AtomicBool::new(false),
        })
    }

    /// Run `waiter` with the load's outcome: now if it is known, else when it is (on the completing thread).
    pub fn on_done(&self, waiter: impl FnOnce(Result<Arc<Entry>>) + Send + 'static) {
        let mut state = self.state.lock().unwrap();
        match &mut *state {
            PendingState::Loading(waiters) => waiters.push(Box::new(waiter)),
            PendingState::Done(outcome) => {
                let outcome = outcome.clone();
                drop(state);
                waiter(outcome);
            }
        }
    }

    /// The load failed: its waiters get `error`.
    pub fn fail(&self, error: Error) {
        self.finish(Err(error));
    }

    fn finish(&self, outcome: Result<Arc<Entry>>) {
        let waiters = {
            let mut state = self.state.lock().unwrap();
            match std::mem::replace(&mut *state, PendingState::Done(outcome.clone())) {
                PendingState::Loading(waiters) => waiters,
                PendingState::Done(_) => Vec::new(),
            }
        };
        for waiter in waiters {
            waiter(outcome.clone());
        }
    }
}

enum Slot {
    Ready(Arc<Entry>),
    Loading(Arc<Pending>),
}

/// A request uses a row: if a prefetch loaded it (or is loading it) and no request used it yet, that prefetch paid.
fn claim(stats: &mut CacheStats, slot: &Slot) {
    let first_use = match slot {
        Slot::Ready(entry) => entry.prefetched.swap(false, Ordering::AcqRel),
        Slot::Loading(pending) => pending.prefetch && !pending.claimed.swap(true, Ordering::AcqRel),
    };
    stats.prefetch_used += first_use as u64;
}

/// What the first pass of a request found for one row.
pub enum Probe {
    /// The row's bytes (a lease: the entry is not evicted while this `Arc` lives).
    Hit(Arc<Entry>),
    /// A prefetch is loading the row.
    Wait(Arc<Pending>),
    /// Another request is loading the row: read it, and do not admit it (a miss and a bypass).
    Busy,
    /// Not cached: admit it with `fill` once every row of the request was probed.
    Miss,
}

/// What a lookup found.
pub enum Lookup {
    /// The row's bytes (a lease: the entry is not evicted while this `Arc` lives).
    Hit(Arc<Entry>),
    /// Another request is loading the row.
    Wait(Arc<Pending>),
    /// The caller loads the row; its bytes are reserved. Complete or abort the ticket (dropping it aborts).
    Fill(Fill),
    /// The row is not admitted; the caller reads it without caching it.
    Bypass,
}

struct State {
    lru: LruCache<Key, Slot>,
    /// Bytes of the rows in the LRU order (ready and loading) and of aborted loads not given back yet.
    used: u64,
    peak: u64,
    /// Whole blocks of evicted rows, for the next rows admitted (counted against the budget).
    pool: Vec<Box<[u8]>>,
    peak_held: u64,
    admit: bool,
    stats: CacheStats,
}

impl State {
    fn pool_bytes(&self, block: u64) -> u64 {
        self.pool.len() as u64 * block
    }
}

pub struct HostCache {
    capacity: u64,
    block: u64,
    state: Mutex<State>,
    /// Memory allocated by fills (outside the lock), since the last reset.
    allocated: AtomicU64,
}

impl HostCache {
    /// A cache of `capacity` bytes holding rows in blocks of `MIN_BLOCK_BYTES`.
    pub fn new(capacity: u64) -> Arc<Self> {
        Self::with_block(capacity, MIN_BLOCK_BYTES)
    }

    /// A cache of `capacity` bytes holding rows in blocks of `block` bytes (at least one).
    pub fn with_block(capacity: u64, block: u64) -> Arc<Self> {
        Arc::new(HostCache {
            capacity,
            block: block.max(1),
            state: Mutex::new(State {
                lru: LruCache::unbounded(),
                used: 0,
                peak: 0,
                pool: Vec::new(),
                peak_held: 0,
                admit: true,
                stats: CacheStats::default(),
            }),
            allocated: AtomicU64::new(0),
        })
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn block_bytes(&self) -> u64 {
        self.block
    }

    /// Look up row `key` of `nbytes` for a request (counts one lookup and its outcome): a hit is promoted and leased.
    pub fn probe(&self, key: Key, nbytes: u64) -> Probe {
        let mut guard = self.state.lock().unwrap();
        let state = &mut *guard;
        state.stats.lookups += 1;
        match state.lru.get(&key) {
            Some(slot @ Slot::Ready(entry)) => {
                let entry = Arc::clone(entry);
                claim(&mut state.stats, slot);
                state.stats.hits += 1;
                state.stats.hit_bytes += nbytes;
                Probe::Hit(entry)
            }
            Some(slot @ Slot::Loading(pending)) if pending.prefetch => {
                let pending = Arc::clone(pending);
                claim(&mut state.stats, slot);
                state.stats.waits += 1;
                state.stats.wait_bytes += nbytes;
                Probe::Wait(pending)
            }
            Some(Slot::Loading(_)) => {
                state.stats.misses += 1;
                state.stats.miss_bytes += nbytes;
                state.stats.bypassed += 1;
                state.stats.bypassed_bytes += nbytes;
                Probe::Busy
            }
            None => {
                state.stats.misses += 1;
                state.stats.miss_bytes += nbytes;
                Probe::Miss
            }
        }
    }

    /// One row on its own: `probe`, then `fill` on a miss.
    pub fn lookup(self: &Arc<Self>, key: Key, nbytes: u64) -> Lookup {
        match self.probe(key, nbytes) {
            Probe::Hit(entry) => Lookup::Hit(entry),
            Probe::Wait(pending) => Lookup::Wait(pending),
            Probe::Busy => Lookup::Bypass,
            Probe::Miss => self.fill(key, nbytes),
        }
    }

    /// Admit a row a probe missed: room is reserved (evicting least recently used rows nobody leases) and the caller
    /// gets the duty to load it (`Fill`), or `Bypass` when it is not admitted (no room, too large, admission frozen).
    /// If it was admitted since the probe: its `Hit`, or `Wait` for a prefetch's load (the probe's miss is counted as
    /// such), or `Bypass` for another request's load (the miss stands).
    pub fn fill(self: &Arc<Self>, key: Key, nbytes: u64) -> Lookup {
        let mut guard = self.state.lock().unwrap();
        let state = &mut *guard;
        let found = match state.lru.get(&key) {
            Some(slot @ Slot::Ready(entry)) => {
                let found = Lookup::Hit(Arc::clone(entry));
                claim(&mut state.stats, slot);
                Some(found)
            }
            Some(slot @ Slot::Loading(pending)) if pending.prefetch => {
                let found = Lookup::Wait(Arc::clone(pending));
                claim(&mut state.stats, slot);
                Some(found)
            }
            Some(Slot::Loading(_)) => {
                state.stats.bypassed += 1;
                state.stats.bypassed_bytes += nbytes;
                return Lookup::Bypass;
            }
            None => None,
        };
        if let Some(found) = found {
            let stats = &mut state.stats;
            stats.misses -= 1;
            stats.miss_bytes -= nbytes;
            if matches!(found, Lookup::Hit(_)) {
                stats.hits += 1;
                stats.hit_bytes += nbytes;
            } else {
                stats.waits += 1;
                stats.wait_bytes += nbytes;
            }
            return found;
        }
        match self.admit_row(state, key, nbytes, false) {
            Some(fill) => Lookup::Fill(fill),
            None => {
                state.stats.bypassed += 1;
                state.stats.bypassed_bytes += nbytes;
                Lookup::Bypass
            }
        }
    }

    /// Reserve room for an absent row and hand out the duty to load it (None: not admitted). The row's memory starts
    /// with blocks from the pool (evicted rows'); the fill allocates the rest when it starts writing.
    fn admit_row(self: &Arc<Self>, state: &mut State, key: Key, nbytes: u64, prefetch: bool) -> Option<Fill> {
        if !self.make_room(state, nbytes) {
            return None;
        }
        let whole = (nbytes / self.block) as usize;
        let taken = whole.min(state.pool.len());
        let parts = state.pool.split_off(state.pool.len() - taken);
        let recycled = taken as u64 * self.block;
        state.stats.recycled_bytes += recycled;
        state.stats.recycled += (recycled == nbytes) as u64;
        state.used += nbytes;
        // The fill allocates the rest of the row: what the cache holds must stay within its capacity.
        self.trim(state);
        state.peak = state.peak.max(state.used);
        state.peak_held = state.peak_held.max(state.used + state.pool_bytes(self.block));
        let pending = Pending::new(prefetch);
        state.lru.push(key, Slot::Loading(Arc::clone(&pending)));
        Some(Fill {
            cache: Arc::clone(self),
            key,
            nbytes,
            pending,
            memory: Some(RowMemory {
                parts,
                block: self.block,
                len: nbytes,
            }),
            recycled,
            outcome: Outcome::Open,
        })
    }

    /// A prefetch's first pass (not a lookup): a cached row is promoted, since a request will use it soon, and leased
    /// while the prefetch admits its other rows; `None` for a row being loaded (skipped) or absent (`absent` set).
    pub fn prefetch_probe(&self, key: Key, absent: &mut bool) -> Option<Arc<Entry>> {
        let mut state = self.state.lock().unwrap();
        let found = match state.lru.get(&key) {
            Some(Slot::Ready(entry)) => Some(Arc::clone(entry)),
            Some(Slot::Loading(_)) => None,
            None => {
                *absent = true;
                return None;
            }
        };
        state.stats.prefetch_skipped += 1;
        found
    }

    /// A prefetch's second pass: admit an absent row for the prefetch to load (None: present since, or not admitted).
    pub fn prefetch_fill(self: &Arc<Self>, key: Key, nbytes: u64) -> Option<Fill> {
        let mut guard = self.state.lock().unwrap();
        let state = &mut *guard;
        if state.lru.contains(&key) {
            state.stats.prefetch_skipped += 1;
            return None;
        }
        let fill = self.admit_row(state, key, nbytes, true);
        if fill.is_some() {
            state.stats.prefetch_fills += 1;
            state.stats.prefetch_fill_bytes += nbytes;
        } else {
            state.stats.prefetch_bypassed += 1;
        }
        fill
    }

    /// Make room for `nbytes` by evicting least recently used ready entries that nobody leases (false: no room; evicts
    /// nothing when that would not make room). The victims' whole blocks go to the pool.
    fn make_room(&self, state: &mut State, nbytes: u64) -> bool {
        if !state.admit || nbytes > self.capacity {
            return false;
        }
        if state.used + nbytes <= self.capacity {
            return true;
        }
        let need = state.used + nbytes - self.capacity;
        let mut victims = Vec::new();
        let mut freed = 0;
        for (key, slot) in state.lru.iter().rev() {
            if let Slot::Ready(entry) = slot {
                if Arc::strong_count(entry) == 1 {
                    victims.push(*key);
                    freed += entry.len();
                    if freed >= need {
                        break;
                    }
                }
            }
        }
        if freed < need {
            return false;
        }
        for key in victims {
            if let Some(Slot::Ready(entry)) = state.lru.pop(&key) {
                let len = entry.len();
                state.used -= len;
                state.stats.evictions += 1;
                state.stats.evicted_bytes += len;
                state.stats.prefetch_wasted += entry.prefetched.load(Ordering::Acquire) as u64;
                // Nobody else holds it (checked above, under this lock): its memory is the cache's to reuse.
                if let Ok(entry) = Arc::try_unwrap(entry) {
                    self.recycle(state, entry.memory);
                }
            }
        }
        true
    }

    /// A row's memory back to the cache: whole blocks to the pool, the tail to the system.
    fn recycle(&self, state: &mut State, memory: RowMemory) {
        for part in memory.parts {
            if part.len() as u64 == self.block {
                state.pool.push(part);
            } else {
                state.stats.released_bytes += part.len() as u64;
            }
        }
    }

    /// Give pooled blocks back to the system while the cache would hold more than its capacity.
    fn trim(&self, state: &mut State) {
        while state.used + state.pool_bytes(self.block) > self.capacity && state.pool.pop().is_some() {
            state.stats.released_bytes += self.block;
        }
    }

    fn complete(&self, key: Key, nbytes: u64, memory: RowMemory, pending: &Pending) -> Arc<Entry> {
        let mut state = self.state.lock().unwrap();
        // Still unused when its prefetch's load completes (a request that waited for it has claimed it).
        let prefetched = AtomicBool::new(pending.prefetch && !pending.claimed.load(Ordering::Acquire));
        let entry = Arc::new(Entry { memory, prefetched });
        // The slot is still there: loading slots are never evicted, and only their fill settles them.
        if let Some(slot) = state.lru.peek_mut(&key) {
            *slot = Slot::Ready(Arc::clone(&entry));
        }
        state.stats.inserts += 1;
        state.stats.insert_bytes += nbytes;
        entry
    }

    /// An aborted load leaves the order (lookups miss it again); its bytes stay reserved until its memory comes back.
    fn abort(&self, key: Key) {
        let mut state = self.state.lock().unwrap();
        state.lru.pop(&key);
        state.stats.aborted_fills += 1;
    }

    /// The memory of a fill that did not complete, once nothing can write it any more.
    fn give_back(&self, nbytes: u64, memory: Option<RowMemory>) {
        let mut guard = self.state.lock().unwrap();
        let state = &mut *guard;
        state.used -= nbytes;
        if let Some(memory) = memory {
            self.recycle(state, memory);
        }
        self.trim(state);
    }

    pub fn set_admit(&self, admit: bool) {
        self.state.lock().unwrap().admit = admit;
    }

    pub fn admit(&self) -> bool {
        self.state.lock().unwrap().admit
    }

    pub fn stats(&self) -> CacheStats {
        let mut stats = self.state.lock().unwrap().stats;
        stats.allocated_bytes = self.allocated.load(Ordering::Acquire);
        stats
    }

    pub fn reset_stats(&self) {
        let mut state = self.state.lock().unwrap();
        state.stats = CacheStats::default();
        state.peak = state.used;
        state.peak_held = state.used + state.pool_bytes(self.block);
        self.allocated.store(0, Ordering::Release);
    }

    /// Bytes of the rows held (ready entries and rows being loaded).
    pub fn resident_bytes(&self) -> u64 {
        self.state.lock().unwrap().used
    }

    pub fn peak_resident_bytes(&self) -> u64 {
        self.state.lock().unwrap().peak
    }

    /// Bytes of the pooled blocks of evicted rows.
    pub fn pool_bytes(&self) -> u64 {
        self.state.lock().unwrap().pool_bytes(self.block)
    }

    /// Everything the cache holds: its rows and its pooled blocks (never above the capacity).
    pub fn held_bytes(&self) -> u64 {
        let state = self.state.lock().unwrap();
        state.used + state.pool_bytes(self.block)
    }

    pub fn peak_held_bytes(&self) -> u64 {
        self.state.lock().unwrap().peak_held
    }

    /// Ready entries and rows being loaded.
    pub fn len(&self) -> usize {
        self.state.lock().unwrap().lru.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, key: Key) -> bool {
        matches!(self.state.lock().unwrap().lru.peek(&key), Some(Slot::Ready(_)))
    }

    /// The ready rows, least recently used first (for tests and reports).
    pub fn keys_lru_first(&self) -> Vec<Key> {
        let state = self.state.lock().unwrap();
        state
            .lru
            .iter()
            .rev()
            .filter(|(_, slot)| matches!(slot, Slot::Ready(_)))
            .map(|(key, _)| *key)
            .collect()
    }

    /// Drop every ready entry nobody leases (rows being loaded stay), and give their memory and the pool back to the
    /// system.
    pub fn clear(&self) {
        let mut guard = self.state.lock().unwrap();
        let state = &mut *guard;
        let keys: Vec<Key> = state
            .lru
            .iter()
            .filter(|(_, slot)| matches!(slot, Slot::Ready(entry) if Arc::strong_count(entry) == 1))
            .map(|(key, _)| *key)
            .collect();
        for key in keys {
            if let Some(Slot::Ready(entry)) = state.lru.pop(&key) {
                state.used -= entry.len();
                state.stats.released_bytes += entry.len();
                state.stats.prefetch_wasted += entry.prefetched.load(Ordering::Acquire) as u64;
            }
        }
        state.stats.released_bytes += state.pool_bytes(self.block);
        state.pool = Vec::new();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    Open,
    Completed,
    Aborted,
}

/// The right and duty to load a row into the cache. It owns the row's memory until the row is complete; a fill that
/// does not complete gives its memory and its reserved bytes back when it is dropped (its loader's writes are over).
pub struct Fill {
    cache: Arc<HostCache>,
    key: Key,
    nbytes: u64,
    pending: Arc<Pending>,
    memory: Option<RowMemory>,
    recycled: u64,
    outcome: Outcome,
}

impl Fill {
    pub fn key(&self) -> Key {
        self.key
    }

    pub fn nbytes(&self) -> u64 {
        self.nbytes
    }

    /// Whether a prefetch holds this duty.
    pub fn is_prefetch(&self) -> bool {
        self.pending.prefetch
    }

    /// Bytes of the row's memory that came from evicted rows' blocks (the rest is allocated by the fill).
    pub fn recycled_bytes(&self) -> u64 {
        self.recycled
    }

    pub fn is_aborted(&self) -> bool {
        self.outcome == Outcome::Aborted
    }

    fn memory(&mut self) -> &mut RowMemory {
        let memory = self.memory.as_mut().expect("an open fill holds its row's memory");
        if memory.allocated() < memory.len() {
            let allocated = memory.allocate_rest();
            self.cache.allocated.fetch_add(allocated, Ordering::AcqRel);
        }
        memory
    }

    /// Views of the row's memory for writes of disjoint ranges by several threads (what the pool did not give is
    /// allocated now). They are valid until the fill completes or is dropped.
    pub fn views(&mut self) -> RowViews {
        self.memory().views()
    }

    /// Write `data` at byte `offset` of the row (one thread).
    pub fn write(&mut self, offset: u64, data: &[u8]) {
        self.memory().write(offset, data);
    }

    /// Every byte of the row was written: the entry becomes ready and its waiters are served.
    pub fn complete(&mut self) -> Result<Arc<Entry>> {
        if self.outcome != Outcome::Open {
            return Err(Error::Internal(format!(
                "a fill completed after it was {:?}",
                self.outcome
            )));
        }
        self.memory();
        let memory = self.memory.take().expect("an open fill holds its row's memory");
        self.outcome = Outcome::Completed;
        let entry = self.cache.complete(self.key, self.nbytes, memory, &self.pending);
        self.pending.finish(Ok(Arc::clone(&entry)));
        Ok(entry)
    }

    /// The load failed: the row leaves the cache and its waiters get `error` (they read it themselves). The memory and
    /// the bytes stay reserved until the fill is dropped: a write of its loader may still be running.
    pub fn abort(&mut self, error: Error) {
        if let Some(pending) = self.detach() {
            pending.fail(error);
        }
    }

    /// `abort`, but the waiters are told by the caller (`Pending::fail` on the result), once it holds no lock: they run
    /// their callbacks on the thread that tells them. None if the fill is not open.
    pub fn detach(&mut self) -> Option<Arc<Pending>> {
        if self.outcome != Outcome::Open {
            return None;
        }
        self.outcome = Outcome::Aborted;
        self.cache.abort(self.key);
        Some(Arc::clone(&self.pending))
    }
}

impl Drop for Fill {
    fn drop(&mut self) {
        if self.outcome == Outcome::Completed {
            return;
        }
        self.abort(Error::Cancelled);
        self.cache.give_back(self.nbytes, self.memory.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn fill(cache: &Arc<HostCache>, key: Key, nbytes: u64) -> Arc<Entry> {
        match cache.lookup(key, nbytes) {
            Lookup::Fill(mut ticket) => {
                ticket.write(0, &vec![key.1 as u8; nbytes as usize]);
                ticket.complete().unwrap()
            }
            _ => panic!("expected a fill"),
        }
    }

    #[test]
    fn lru_order_budget_and_hits() {
        let cache = HostCache::new(100);
        drop(fill(&cache, (0, 1), 40));
        drop(fill(&cache, (0, 2), 40));
        assert!(matches!(cache.lookup((0, 1), 40), Lookup::Hit(_))); // 1 becomes most recent
        drop(fill(&cache, (0, 3), 40)); // evicts 2, the least recent
        assert_eq!(cache.keys_lru_first(), vec![(0, 1), (0, 3)]);
        assert_eq!(cache.resident_bytes(), 80);
        let stats = cache.stats();
        assert_eq!(
            (stats.hits, stats.misses, stats.evictions, stats.evicted_bytes),
            (1, 3, 1, 40)
        );
        assert!(matches!(cache.lookup((0, 9), 101), Lookup::Bypass)); // larger than the cache
        assert_eq!(cache.resident_bytes(), 80);
    }

    #[test]
    fn leased_entries_are_never_evicted() {
        let cache = HostCache::new(100);
        let lease = fill(&cache, (0, 1), 60);
        // A second row would need evicting the leased one: not admitted, nothing evicted.
        assert!(matches!(cache.lookup((0, 2), 60), Lookup::Bypass));
        assert!(cache.contains((0, 1)));
        drop(lease);
        drop(fill(&cache, (0, 2), 60));
        assert!(!cache.contains((0, 1)) && cache.contains((0, 2)));
        assert!(cache.resident_bytes() <= cache.capacity());
    }

    #[test]
    fn loading_rows_count_against_the_budget_and_are_loaded_once() {
        let cache = HostCache::new(100);
        let mut ticket = cache.prefetch_fill((1, 1), 70).expect("admitted");
        assert_eq!(cache.resident_bytes(), 70);
        // A request for the row a prefetch is loading waits for that load instead of loading it again.
        let Lookup::Wait(pending) = cache.lookup((1, 1), 70) else {
            panic!()
        };
        let served = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&served);
        pending.on_done(move |outcome| {
            assert_eq!(outcome.unwrap().to_vec(), vec![5u8; 70]);
            counter.fetch_add(1, Ordering::SeqCst);
        });
        // Another row cannot evict a loading one.
        assert!(matches!(cache.lookup((1, 2), 40), Lookup::Bypass));
        ticket.write(0, &[5u8; 70]);
        ticket.complete().unwrap();
        assert_eq!(served.load(Ordering::SeqCst), 1);
        // A waiter registered after completion is served at once.
        let counter = Arc::clone(&served);
        pending.on_done(move |outcome| {
            assert!(outcome.is_ok());
            counter.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(served.load(Ordering::SeqCst), 2);
        let stats = cache.stats();
        assert_eq!((stats.misses, stats.waits, stats.inserts, stats.bypassed), (1, 1, 1, 1));
    }

    #[test]
    fn a_row_another_request_is_loading_is_read_not_waited_for() {
        // Two requests waiting for each other's loads could each hold what the other needs: a request reads a row another
        // request is loading itself (a miss and a bypass), and the row is still admitted once.
        let cache = HostCache::new(100);
        let Lookup::Fill(mut ticket) = cache.lookup((1, 1), 40) else {
            panic!()
        };
        assert!(matches!(cache.probe((1, 1), 40), Probe::Busy));
        assert!(matches!(cache.lookup((1, 1), 40), Lookup::Bypass));
        ticket.write(0, &[1u8; 40]);
        ticket.complete().unwrap();
        assert!(matches!(cache.lookup((1, 1), 40), Lookup::Hit(_)));
        let stats = cache.stats();
        assert_eq!(
            (
                stats.lookups,
                stats.hits,
                stats.waits,
                stats.misses,
                stats.bypassed,
                stats.inserts
            ),
            (4, 1, 0, 3, 2, 1)
        );
    }

    #[test]
    fn a_dropped_fill_releases_its_bytes_and_fails_its_waiters() {
        let cache = HostCache::new(100);
        let ticket = cache.prefetch_fill((2, 1), 50).expect("admitted");
        let Lookup::Wait(pending) = cache.lookup((2, 1), 50) else {
            panic!()
        };
        let failed = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&failed);
        pending.on_done(move |outcome| {
            assert_eq!(outcome.unwrap_err(), Error::Cancelled);
            counter.fetch_add(1, Ordering::SeqCst);
        });
        drop(ticket);
        assert_eq!(failed.load(Ordering::SeqCst), 1);
        assert_eq!(cache.resident_bytes(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.stats().aborted_fills, 1);
        // The row can be loaded again.
        drop(fill(&cache, (2, 1), 50));
        assert!(cache.contains((2, 1)));
    }

    #[test]
    fn an_aborted_fill_keeps_its_bytes_until_its_memory_comes_back() {
        // Its loader's writes may still be running when it is aborted: the bytes stay reserved, so what the cache holds
        // never exceeds its capacity, until the fill is dropped.
        let cache = HostCache::with_block(100, 10);
        let Lookup::Fill(mut ticket) = cache.lookup((2, 1), 60) else {
            panic!()
        };
        let views = ticket.views();
        assert_eq!(views.len(), 60);
        ticket.abort(Error::Cancelled);
        assert!(!cache.contains((2, 1)) && cache.is_empty());
        assert_eq!(cache.resident_bytes(), 60);
        // Room is not handed out twice: a row that needs those bytes is not admitted (nothing to evict).
        assert!(matches!(cache.lookup((2, 2), 50), Lookup::Bypass));
        drop(ticket);
        assert_eq!((cache.resident_bytes(), cache.held_bytes()), (0, 60)); // its six blocks are pooled
        drop(fill(&cache, (2, 2), 50)); // and serve the next row
        assert_eq!(cache.stats().recycled, 1);
    }

    #[test]
    fn evicted_blocks_serve_rows_of_other_sizes() {
        // Rows of two sizes, both multiples of the block, evicting each other: memory is allocated only to grow the
        // cache to its capacity (everything allocated is still held), every other row is served by evicted rows' blocks,
        // nothing is freed, and what the cache holds never exceeds the budget.
        let cache = HostCache::with_block(1000, 50);
        let mut rng = 7u64;
        for _ in 0..400u64 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let (segment, row) = ((rng >> 40) % 2, (rng >> 20) % 40);
            let nbytes = if segment == 0 { 100 } else { 250 };
            if let Lookup::Fill(mut ticket) = cache.lookup((segment as u32, row), nbytes) {
                ticket.write(0, &vec![row as u8; nbytes as usize]);
                assert_eq!(ticket.complete().unwrap().to_vec(), vec![row as u8; nbytes as usize]);
            }
            assert!(cache.held_bytes() <= cache.capacity() && cache.peak_held_bytes() <= cache.capacity());
        }
        let stats = cache.stats();
        assert!(stats.evictions > 100, "{stats:?}");
        assert_eq!(stats.allocated_bytes, cache.held_bytes(), "{stats:?}");
        assert_eq!(stats.released_bytes, 0);
        assert!(
            stats.recycled_bytes > 20 * cache.capacity() && stats.recycled > 100,
            "{stats:?}"
        );
    }

    #[test]
    fn pooled_blocks_count_against_the_budget() {
        // Rows with tails: evicting a row of whole blocks for a row that is mostly tail leaves blocks in the pool; the
        // pool is trimmed so that rows and pooled blocks together stay within the capacity.
        let cache = HostCache::with_block(100, 40);
        drop(fill(&cache, (0, 1), 80)); // two blocks
        drop(fill(&cache, (0, 2), 30)); // a tail: evicts row 1, whose two blocks are pooled, then trimmed to fit
        assert!(cache.held_bytes() <= cache.capacity());
        assert_eq!(cache.resident_bytes(), 30);
        assert!(cache.stats().released_bytes > 0);
        drop(fill(&cache, (0, 3), 70)); // one block (pooled if left) and a tail
        assert!(cache.held_bytes() <= cache.capacity() && cache.peak_held_bytes() <= cache.capacity());
        assert_eq!(fill(&cache, (0, 4), 30).to_vec(), vec![4u8; 30]);
    }

    #[test]
    fn hits_misses_and_evictions_do_not_depend_on_the_block_size() {
        // The same requests through caches that differ only in their block size: the same outcomes, step by step.
        let outcomes = |block: u64| {
            let cache = HostCache::with_block(10_000, block);
            let mut rng = 11u64;
            let mut seen = Vec::new();
            for _ in 0..2_000 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let segment = ((rng >> 50) % 3) as u32;
                let row = (rng >> 20) % 30;
                let nbytes = [700u64, 1_400, 333][segment as usize];
                seen.push(match cache.lookup((segment, row), nbytes) {
                    Lookup::Hit(_) => 'h',
                    Lookup::Wait(_) => 'w',
                    Lookup::Bypass => 'b',
                    Lookup::Fill(mut ticket) => {
                        ticket.write(0, &vec![row as u8; nbytes as usize]);
                        ticket.complete().unwrap();
                        'm'
                    }
                });
                assert!(cache.held_bytes() <= cache.capacity());
            }
            let stats = cache.stats();
            (
                seen,
                cache.keys_lru_first(),
                (stats.hits, stats.misses, stats.evictions, stats.bypassed, stats.inserts),
            )
        };
        let reference = outcomes(1 << 30); // every row one allocation (Phase 6A's memory)
        for block in [1, 7, 100, 350, 700] {
            assert_eq!(outcomes(block), reference, "block {block}");
        }
    }

    #[test]
    fn frozen_admission_serves_hits_and_bypasses_misses() {
        let cache = HostCache::new(100);
        drop(fill(&cache, (0, 1), 10));
        cache.set_admit(false);
        assert!(matches!(cache.lookup((0, 1), 10), Lookup::Hit(_)));
        assert!(matches!(cache.lookup((0, 2), 10), Lookup::Bypass));
        cache.set_admit(true);
        assert!(matches!(cache.lookup((0, 2), 10), Lookup::Fill(_)));
    }

    #[test]
    fn clearing_gives_rows_and_pooled_blocks_back() {
        let cache = HostCache::with_block(100, 10);
        drop(fill(&cache, (0, 1), 60));
        drop(fill(&cache, (0, 2), 60)); // evicts row 1: its six blocks serve row 2
        let lease = fill(&cache, (0, 3), 50); // evicts row 2: five of its blocks serve row 3, one is pooled
        assert_eq!((cache.resident_bytes(), cache.held_bytes()), (50, 60));
        assert_eq!(cache.stats().allocated_bytes, 60);
        cache.clear();
        // The leased row stays; everything else is gone, the pool included.
        assert_eq!((cache.resident_bytes(), cache.held_bytes(), cache.len()), (50, 50, 1));
        drop(lease);
        cache.clear();
        assert_eq!((cache.resident_bytes(), cache.held_bytes(), cache.len()), (0, 0, 0));
    }

    #[test]
    fn concurrent_lookups_load_each_row_once() {
        let cache = HostCache::with_block(1 << 20, 64);
        let fills = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let cache = Arc::clone(&cache);
                let fills = Arc::clone(&fills);
                scope.spawn(move || {
                    for row in 0..200u64 {
                        match cache.lookup((0, row), 100) {
                            Lookup::Fill(mut ticket) => {
                                fills.fetch_add(1, Ordering::SeqCst);
                                ticket.write(0, &[row as u8; 100]);
                                ticket.complete().unwrap();
                            }
                            Lookup::Hit(entry) => assert_eq!(entry.to_vec()[0], row as u8),
                            Lookup::Wait(_) => panic!("only prefetch loads are waited for"),
                            Lookup::Bypass => {} // another thread was loading the row: read, not admitted
                        }
                    }
                });
            }
        });
        assert_eq!(fills.load(Ordering::SeqCst), 200);
        assert_eq!(cache.resident_bytes(), 200 * 100);
    }

    #[test]
    fn the_default_block_divides_every_row() {
        assert_eq!(block_for([11_534_336, 5_767_168]), 5_767_168); // an expert's two rows of 11 and 5.5 MiB
        assert_eq!(block_for([4 << 20, 8 << 20, 12 << 20]), 4 << 20);
        assert_eq!(block_for([300_001, 12_392]), MIN_BLOCK_BYTES); // no row of a megabyte: tails all
        assert_eq!(block_for([2_867_200, 1_433_600, 4_096]), 1_433_600); // a small row does not shrink the block
        assert_eq!(block_for([3 << 20, 2 << 20]), MIN_BLOCK_BYTES); // no common megabyte: the rows get tails
        assert_eq!(block_for([]), MIN_BLOCK_BYTES);
    }

    #[test]
    fn spans_cover_ranges_across_parts() {
        let mut memory = RowMemory {
            parts: Vec::new(),
            block: 4,
            len: 10,
        };
        assert_eq!(memory.allocate_rest(), 10);
        assert_eq!(memory.parts.iter().map(|p| p.len()).collect::<Vec<_>>(), vec![4, 4, 2]);
        let data: Vec<u8> = (0..10).collect();
        memory.write(0, &data[..3]);
        memory.write(3, &data[3..]);
        assert_eq!(memory.to_vec(), data);
        let mut out = [0u8; 5];
        memory.copy_to(3, &mut out);
        assert_eq!(out, [3, 4, 5, 6, 7]);
        assert_eq!(
            spans(4, 10, 3, 6).collect::<Vec<_>>(),
            vec![(0, 3, 1, 0), (1, 0, 4, 1), (2, 0, 1, 5)]
        );
    }
}
