//! The host-RAM tier: rows of segments kept in host memory between requests, under a strict byte budget.
//!
//! A deterministic least-recently-used baseline (the `lru` crate keeps the order; the budget in bytes is this
//! module's), with the patterns reviewed in decision 0012 (ds4, colibri, DwarfStar):
//!
//!   * an entry is exactly one row's bytes, and every byte the cache holds counts against the budget, including a row
//!     being loaded (its bytes are reserved when the load starts): resident bytes never exceed the capacity;
//!   * a row is loaded once however many requests want it at the same time: the first lookup gets a `Fill` ticket
//!     and loads it, the others `Wait` for that load (in-flight deduplication); a fill that fails or is dropped
//!     unfinished hands its waiters the error, and they read the row themselves;
//!   * a request looks up all its rows first (`probe`: hits are promoted and leased), and only then reserves room for
//!     its misses (`fill`), so a miss never evicts a row the same request is about to use;
//!   * entries in use (leased: an `Arc` held by a copy into a staging slot) are never evicted, as ds4 protects every
//!     hit of a request before choosing victims; if a new row would need evicting one, it is not admitted (`Bypass`)
//!     and its request reads it without caching it;
//!   * `admit = false` freezes the contents (DwarfStar's fix for long prefills): hits are served, misses bypass;
//!   * a prefetch (`prefetch_probe`, `prefetch_fill`) loads rows a request will ask for soon without counting as a
//!     lookup; every row it loads is counted once used by a request, or as wasted if evicted (or cleared) unused.
//!
//! The cache decides which rows stay resident, never what their bytes are: a hit's bytes are the bytes a read of the
//! row returned (the engine tests compare them with direct reads).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use lru::LruCache;

use crate::error::{Error, Result};

/// A row: (segment id, row index).
pub type Key = (u32, u64);

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
    /// Fills given the memory of an entry evicted for them (same size): no allocation, no first-touch page faults.
    pub recycled: u64,
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

/// A cached row's bytes.
#[derive(Debug)]
pub struct Entry {
    data: Box<[u8]>,
    /// Loaded by a prefetch and not used by a request yet.
    prefetched: AtomicBool,
}

impl Entry {
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn len(&self) -> u64 {
        self.data.len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
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
    /// Another request is loading the row.
    Wait(Arc<Pending>),
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
    used: u64,
    peak: u64,
    admit: bool,
    stats: CacheStats,
}

pub struct HostCache {
    capacity: u64,
    state: Mutex<State>,
}

impl HostCache {
    pub fn new(capacity: u64) -> Arc<Self> {
        Arc::new(HostCache {
            capacity,
            state: Mutex::new(State {
                lru: LruCache::unbounded(),
                used: 0,
                peak: 0,
                admit: true,
                stats: CacheStats::default(),
            }),
        })
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
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
            Some(slot @ Slot::Loading(pending)) => {
                let pending = Arc::clone(pending);
                claim(&mut state.stats, slot);
                state.stats.waits += 1;
                state.stats.wait_bytes += nbytes;
                Probe::Wait(pending)
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
            Probe::Miss => self.fill(key, nbytes),
        }
    }

    /// Admit a row a probe missed: room is reserved (evicting least recently used rows nobody leases) and the caller
    /// gets the duty to load it (`Fill`), or `Bypass` when it is not admitted (no room, too large, admission frozen).
    /// If another request admitted the row since the probe, its `Hit` or `Wait` (the probe's miss is counted as such).
    pub fn fill(self: &Arc<Self>, key: Key, nbytes: u64) -> Lookup {
        let mut guard = self.state.lock().unwrap();
        let state = &mut *guard;
        let found = match state.lru.get(&key) {
            Some(slot @ Slot::Ready(entry)) => {
                let found = Lookup::Hit(Arc::clone(entry));
                claim(&mut state.stats, slot);
                Some(found)
            }
            Some(slot @ Slot::Loading(pending)) => {
                let found = Lookup::Wait(Arc::clone(pending));
                claim(&mut state.stats, slot);
                Some(found)
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

    /// Reserve room for an absent row and hand out the duty to load it (None: not admitted).
    fn admit_row(self: &Arc<Self>, state: &mut State, key: Key, nbytes: u64, prefetch: bool) -> Option<Fill> {
        let buffer = Self::reserve(state, self.capacity, nbytes)?;
        state.stats.recycled += buffer.is_some() as u64;
        let pending = Pending::new(prefetch);
        state.lru.push(key, Slot::Loading(Arc::clone(&pending)));
        state.used += nbytes;
        state.peak = state.peak.max(state.used);
        Some(Fill {
            cache: Arc::clone(self),
            key,
            nbytes,
            pending,
            buffer,
            settled: false,
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

    /// Make room for `nbytes` by evicting least recently used ready entries that nobody leases (None: no room; evicts
    /// nothing when that would not make room). An evicted entry of exactly `nbytes` gives the new row its memory.
    fn reserve(state: &mut State, capacity: u64, nbytes: u64) -> Option<Option<Box<[u8]>>> {
        if !state.admit || nbytes > capacity {
            return None;
        }
        if state.used + nbytes <= capacity {
            return Some(None);
        }
        let need = state.used + nbytes - capacity;
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
            return None;
        }
        let mut recycled = None;
        for key in victims {
            if let Some(Slot::Ready(entry)) = state.lru.pop(&key) {
                let len = entry.len();
                state.used -= len;
                state.stats.evictions += 1;
                state.stats.evicted_bytes += len;
                state.stats.prefetch_wasted += entry.prefetched.load(Ordering::Acquire) as u64;
                if recycled.is_none() && len == nbytes {
                    // Nobody else holds it (checked above, under this lock): its memory moves to the new row.
                    recycled = Arc::try_unwrap(entry).ok().map(|entry| entry.data);
                }
            }
        }
        Some(recycled)
    }

    fn settle(&self, key: Key, nbytes: u64, data: Option<Box<[u8]>>, pending: &Pending) -> Option<Arc<Entry>> {
        let mut state = self.state.lock().unwrap();
        match data {
            Some(data) => {
                // Still unused when its prefetch's load completes (a request that waited for it has claimed it).
                let prefetched = AtomicBool::new(pending.prefetch && !pending.claimed.load(Ordering::Acquire));
                let entry = Arc::new(Entry { data, prefetched });
                // The slot is still there: loading slots are never evicted, and only their fill settles them.
                if let Some(slot) = state.lru.peek_mut(&key) {
                    *slot = Slot::Ready(Arc::clone(&entry));
                }
                state.stats.inserts += 1;
                state.stats.insert_bytes += nbytes;
                Some(entry)
            }
            None => {
                state.lru.pop(&key);
                state.used -= nbytes;
                state.stats.aborted_fills += 1;
                None
            }
        }
    }

    pub fn set_admit(&self, admit: bool) {
        self.state.lock().unwrap().admit = admit;
    }

    pub fn admit(&self) -> bool {
        self.state.lock().unwrap().admit
    }

    pub fn stats(&self) -> CacheStats {
        self.state.lock().unwrap().stats
    }

    pub fn reset_stats(&self) {
        let mut state = self.state.lock().unwrap();
        state.stats = CacheStats::default();
        state.peak = state.used;
    }

    /// Bytes held (ready entries and rows being loaded).
    pub fn resident_bytes(&self) -> u64 {
        self.state.lock().unwrap().used
    }

    pub fn peak_resident_bytes(&self) -> u64 {
        self.state.lock().unwrap().peak
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

    /// Drop every ready entry nobody leases (rows being loaded stay).
    pub fn clear(&self) {
        let mut state = self.state.lock().unwrap();
        let keys: Vec<Key> = state
            .lru
            .iter()
            .filter(|(_, slot)| matches!(slot, Slot::Ready(entry) if Arc::strong_count(entry) == 1))
            .map(|(key, _)| *key)
            .collect();
        for key in keys {
            if let Some(Slot::Ready(entry)) = state.lru.pop(&key) {
                state.used -= entry.len();
                state.stats.prefetch_wasted += entry.prefetched.load(Ordering::Acquire) as u64;
            }
        }
    }
}

/// The right and duty to load a row into the cache.
pub struct Fill {
    cache: Arc<HostCache>,
    key: Key,
    nbytes: u64,
    pending: Arc<Pending>,
    buffer: Option<Box<[u8]>>,
    settled: bool,
}

impl Fill {
    /// Memory of exactly `nbytes` left by an entry evicted for this row (its old bytes are all overwritten), if any.
    pub fn take_buffer(&mut self) -> Option<Box<[u8]>> {
        self.buffer.take()
    }

    pub fn key(&self) -> Key {
        self.key
    }

    pub fn nbytes(&self) -> u64 {
        self.nbytes
    }

    /// The row's bytes are `data` (exactly `nbytes`): the entry becomes ready and its waiters are served.
    pub fn complete(mut self, data: Box<[u8]>) -> Result<Arc<Entry>> {
        if data.len() as u64 != self.nbytes {
            return Err(Error::Internal(format!(
                "a fill of {} bytes completed with {}",
                self.nbytes,
                data.len()
            )));
        }
        self.settled = true;
        let entry = self
            .cache
            .settle(self.key, self.nbytes, Some(data), &self.pending)
            .expect("a completed fill has an entry");
        self.pending.finish(Ok(Arc::clone(&entry)));
        Ok(entry)
    }

    /// The load failed: the reservation is released and the waiters get `error`.
    pub fn abort(mut self, error: Error) {
        self.settled = true;
        self.cache.settle(self.key, self.nbytes, None, &self.pending);
        self.pending.finish(Err(error));
    }

    /// Whether a prefetch holds this duty.
    pub fn is_prefetch(&self) -> bool {
        self.pending.prefetch
    }
}

impl Drop for Fill {
    fn drop(&mut self) {
        if !self.settled {
            self.settled = true;
            self.cache.settle(self.key, self.nbytes, None, &self.pending);
            self.pending.finish(Err(Error::Cancelled));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn fill(cache: &Arc<HostCache>, key: Key, nbytes: u64) -> Arc<Entry> {
        match cache.lookup(key, nbytes) {
            Lookup::Fill(ticket) => ticket
                .complete(vec![key.1 as u8; nbytes as usize].into_boxed_slice())
                .unwrap(),
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
        let Lookup::Fill(ticket) = cache.lookup((1, 1), 70) else {
            panic!()
        };
        assert_eq!(cache.resident_bytes(), 70);
        // A second request for the same row waits for this load instead of loading it again.
        let Lookup::Wait(pending) = cache.lookup((1, 1), 70) else {
            panic!()
        };
        let served = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&served);
        pending.on_done(move |outcome| {
            assert_eq!(outcome.unwrap().bytes(), &[5u8; 70][..]);
            counter.fetch_add(1, Ordering::SeqCst);
        });
        // Another row cannot evict a loading one.
        assert!(matches!(cache.lookup((1, 2), 40), Lookup::Bypass));
        ticket.complete(vec![5u8; 70].into_boxed_slice()).unwrap();
        assert_eq!(served.load(Ordering::SeqCst), 1);
        // A waiter registered after completion is served at once.
        let counter = Arc::clone(&served);
        pending.on_done(move |outcome| {
            assert!(outcome.is_ok());
            counter.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(served.load(Ordering::SeqCst), 2);
        let stats = cache.stats();
        assert_eq!((stats.misses, stats.waits, stats.inserts, stats.bypassed), (2, 1, 1, 1));
    }

    #[test]
    fn a_dropped_fill_releases_its_bytes_and_fails_its_waiters() {
        let cache = HostCache::new(100);
        let Lookup::Fill(ticket) = cache.lookup((2, 1), 50) else {
            panic!()
        };
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
    fn an_evicted_entry_of_the_same_size_gives_its_memory_to_the_new_row() {
        let cache = HostCache::new(100);
        drop(fill(&cache, (0, 1), 50));
        // Full: the next row evicts.
        drop(fill(&cache, (0, 2), 50));
        // The same size: the new row gets the evicted entry's memory (its old bytes, all to be overwritten).
        let Lookup::Fill(mut same) = cache.lookup((0, 3), 50) else {
            panic!()
        };
        let buffer = same.take_buffer().expect("recycled");
        assert_eq!(&buffer[..], &[1u8; 50][..]);
        let entry = same.complete(vec![3u8; 50].into_boxed_slice()).unwrap();
        assert_eq!(entry.bytes(), &[3u8; 50][..]);
        drop(entry);
        // Another size: the evicted entries' memory is freed, not reused.
        let Lookup::Fill(mut other) = cache.lookup((0, 4), 60) else {
            panic!()
        };
        assert!(other.take_buffer().is_none());
        drop(other.complete(vec![4u8; 60].into_boxed_slice()).unwrap());
        let stats = cache.stats();
        assert_eq!((stats.evictions, stats.recycled), (3, 1));
        assert_eq!(cache.resident_bytes(), 60);
        assert_eq!(cache.keys_lru_first(), vec![(0, 4)]);
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
    fn concurrent_lookups_load_each_row_once() {
        let cache = HostCache::new(1 << 20);
        let fills = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let cache = Arc::clone(&cache);
                let fills = Arc::clone(&fills);
                scope.spawn(move || {
                    for row in 0..200u64 {
                        match cache.lookup((0, row), 100) {
                            Lookup::Fill(ticket) => {
                                fills.fetch_add(1, Ordering::SeqCst);
                                ticket.complete(vec![row as u8; 100].into_boxed_slice()).unwrap();
                            }
                            Lookup::Hit(entry) => assert_eq!(entry.bytes()[0], row as u8),
                            Lookup::Wait(pending) => {
                                pending.on_done(move |outcome| assert_eq!(outcome.unwrap().bytes()[0], row as u8))
                            }
                            Lookup::Bypass => panic!("room for every row"),
                        }
                    }
                });
            }
        });
        assert_eq!(fills.load(Ordering::SeqCst), 200);
        assert_eq!(cache.resident_bytes(), 200 * 100);
    }
}
