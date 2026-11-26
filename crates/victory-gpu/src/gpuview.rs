//! (F3, `VICTORY_LAZY_GPU_VIEW`) A page table of the GPU view of the title's
//! streaming pool, kept from its `CopyPageMappingsBatchX` calls.
//!
//! The title keeps three views of its streaming pool: a CPU view, a GPU view
//! and a linear view of the whole pool. Per streaming batch it maps the CPU
//! view (`XMemMapPhysicalPagesScatter`) and the GPU view (one
//! `CopyPageMappingsBatchX`); its CPU code never reads through the GPU view,
//! only this layer does. A lazy GPU view would map a page there only when the
//! layer first reads it.
//!
//! THIS BUILD IS METER ONLY: the pages are still mapped at call time, as
//! before. The layer reads title memory through the kernel
//! (`ReadProcessMemory`, which reads an unmapped page as a failure, not a
//! fault), asks the kernel whether pages are there (`QueryWorkingSetEx`,
//! `VirtualQuery`), guards pages with `VirtualProtect` and opens heaps on the
//! sections behind views; a page left unmapped would be read as zeros on all
//! those paths, none of which an exception handler sees. Until they all ask
//! this table, the table counts what a lazy view would have saved:
//! `VICTORY_LAZY_GPU_VIEW=1` turns it on (default off). A "read" is what the
//! layer's kernel copy, its in-place copy and its read holds see
//! ([`note_read`]); reads through plain pointers are not seen, so "never
//! materialised" counts are an upper bound.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

/// Bytes per page of the view.
pub const PAGE: u64 = 64 << 10;

/// Whether the table is kept (`VICTORY_LAZY_GPU_VIEW=1`).
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_LAZY_GPU_VIEW").is_some_and(|v| v == "1");
        if on {
            tracing::warn!(
                "gpu view: VICTORY_LAZY_GPU_VIEW=1 is meter only in this build: pages are still mapped at call time; \
                 the table counts what lazy mapping would avoid"
            );
        }
        on
    })
}

/// What a page of the view shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backing {
    /// Nothing (never mapped, or unmapped).
    Unmapped,
    /// A pool page (kind 0).
    Pool(u32),
    /// The constant page every page of a kind-1 run shows.
    Constant(u32),
}

// An entry: bits 0-1 state (0 unmapped, 1 pool, 2 constant), bit 2 read since
// mapped, bits 8-31 pool page, bits 32-63 generation (one more at each change
// of backing).
const STATE: u64 = 3;
const READ: u64 = 4;

fn encode(b: Backing, read: bool, generation: u32) -> u64 {
    let (state, page) = match b {
        Backing::Unmapped => (0, 0),
        Backing::Pool(p) => (1, p),
        Backing::Constant(p) => (2, p),
    };
    let read = if read { READ } else { 0 };
    state | read | (u64::from(page & 0xff_ffff) << 8) | (u64::from(generation) << 32)
}

fn decode(e: u64) -> (Backing, bool, u32) {
    let page = (e >> 8 & 0xff_ffff) as u32;
    let b = match e & STATE {
        1 => Backing::Pool(page),
        2 => Backing::Constant(page),
        _ => Backing::Unmapped,
    };
    (b, e & READ != 0, (e >> 32) as u32)
}

/// What the meter counts (since the last line).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    /// Mapping calls.
    pub calls: u64,
    /// Pages noted (mapped or unmapped), by kind 0, 1, 2.
    pub pages: [u64; 3],
    /// Pages read for the first time since mapped (a lazy view maps them then).
    pub materialised: u64,
    /// Unmaps of pages mapped but never read since: a table update only.
    pub unmaps_never_read: u64,
    /// Unmaps of pages that were read (a lazy view unmaps these, with the hold).
    pub unmaps_read: u64,
    /// Unmaps of pages already unmapped.
    pub unmaps_of_nothing: u64,
    /// Remaps of a page mapped but never read since: collapsed into the last.
    pub collapsed: u64,
    /// Remaps to what the page shows already.
    pub same: u64,
    /// Pages of ranges outside the view.
    pub outside: u64,
    /// Microseconds in the mapping calls.
    pub us: u64,
}

/// The table: one entry per page of the view.
pub struct Table {
    base: u64,
    entries: Box<[AtomicU64]>,
    counts: Mutex<Counts>,
    materialised: AtomicU64,
}

impl Table {
    pub fn new(base: u64, len: u64) -> Table {
        Table {
            base,
            entries: (0..len / PAGE).map(|_| AtomicU64::new(0)).collect(),
            counts: Mutex::new(Counts::default()),
            materialised: AtomicU64::new(0),
        }
    }

    pub fn contains(&self, at: u64) -> bool {
        at.wrapping_sub(self.base) < self.entries.len() as u64 * PAGE
    }

    /// The page at `at` (in the view).
    pub fn get(&self, at: u64) -> Option<(Backing, bool, u32)> {
        let i = at.checked_sub(self.base)? / PAGE;
        Some(decode(self.entries.get(i as usize)?.load(Relaxed)))
    }

    /// One range of a mapping batch, as the layer decodes it: `count` pages
    /// from `at`, kind 0 = pool pages from `start` on, 1 = the pool page
    /// `start` at each, 2 = unmapped, 3 = left as they are.
    pub fn note(&self, at: u64, kind: u32, start: u32, count: u32) {
        let mut c = self.counts.lock().unwrap_or_else(|e| e.into_inner());
        if (kind as usize) < 3 {
            c.pages[kind as usize] += u64::from(count);
        }
        for k in 0..u64::from(count) {
            let backing = match kind {
                0 => Backing::Pool(start.wrapping_add(k as u32)),
                1 => Backing::Constant(start),
                2 => Backing::Unmapped,
                _ => return,
            };
            let page = at + k * PAGE;
            let Some(e) = page.checked_sub(self.base).and_then(|o| self.entries.get((o / PAGE) as usize)) else {
                c.outside += 1;
                continue;
            };
            let (old, read, generation) = decode(e.load(Relaxed));
            let new = match (old, backing) {
                (Backing::Unmapped, Backing::Unmapped) => {
                    c.unmaps_of_nothing += 1;
                    continue;
                }
                (_, Backing::Unmapped) => {
                    if read {
                        c.unmaps_read += 1;
                    } else {
                        c.unmaps_never_read += 1;
                    }
                    encode(Backing::Unmapped, false, generation.wrapping_add(1))
                }
                (old, new) if old == new => {
                    c.same += 1;
                    continue;
                }
                (old, new) => {
                    if old != Backing::Unmapped && !read {
                        c.collapsed += 1;
                    }
                    encode(new, false, generation.wrapping_add(1))
                }
            };
            e.store(new, Relaxed);
        }
    }

    /// The layer reads `[at, at + len)`: pages of the view mapped and not
    /// read since are counted (a lazy view would map them now).
    pub fn note_read(&self, at: u64, len: u64) {
        let Some(from) = at.checked_sub(self.base) else {
            return;
        };
        let (first, end) = (from / PAGE, from.saturating_add(len.max(1)).div_ceil(PAGE));
        let end = end.min(self.entries.len() as u64);
        for i in first..end {
            let e = &self.entries[i as usize];
            let v = e.load(Relaxed);
            if v & STATE != 0 && v & READ == 0 && e.fetch_or(READ, Relaxed) & READ == 0 {
                self.materialised.fetch_add(1, Relaxed);
            }
        }
    }

    /// The counts since the last call, reset.
    pub fn take(&self, us: u64) -> Counts {
        let mut c = self.counts.lock().unwrap_or_else(|e| e.into_inner());
        c.us += us;
        let mut out = std::mem::take(&mut *c);
        out.materialised = self.materialised.swap(0, Relaxed);
        out
    }

    fn add_call(&self, us: u64) {
        let mut c = self.counts.lock().unwrap_or_else(|e| e.into_inner());
        c.calls += 1;
        c.us += us;
    }
}

/// The line the meter writes.
pub fn line(c: &Counts, secs: f64) -> String {
    format!(
        "gpu view: {} calls, {} pages noted, {} materialised on read, {} unmaps of pages never materialised, \
         {} collapsed remaps, {:.1} ms in calls (over {secs:.0} s; meter only, mapping eager) | pages by kind 0/1/2: \
         {}/{}/{}; unmaps of read pages {}, of unmapped pages {}; remaps to the same page {}; outside the view {}",
        c.calls,
        c.pages.iter().sum::<u64>(),
        c.materialised,
        c.unmaps_never_read,
        c.collapsed,
        c.us as f64 / 1e3,
        c.pages[0],
        c.pages[1],
        c.pages[2],
        c.unmaps_read,
        c.unmaps_of_nothing,
        c.same,
        c.outside,
    )
}

static TABLE: OnceLock<Table> = OnceLock::new();
/// The GPU view as `XMemVirtualAlloc` made it: base and length (0 until seen).
static VIEW: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// The title image's base, for the cross-check with the title's globals.
static IMAGE: AtomicUsize = AtomicUsize::new(0);

/// The title's image base (for the cross-check of the view with its globals).
pub fn use_title_image(base: usize) {
    IMAGE.store(base, Relaxed);
}

/// The runtime reserved `[at, at + len)` for the title with the title's own
/// arguments: the GPU view is the large reservation of 64 KiB placeholder
/// pages (kind 0x20402000) with XMem flags 3 and console protection 0x801
/// (the CPU view: flags 0x21, protection 4).
pub fn reserved(at: usize, len: usize, kind: u32, flags: u64, protect: u32) {
    if !on() || at == 0 || len < (1 << 33) || kind != 0x2040_2000 || flags != 3 || protect != 0x801 {
        return;
    }
    if VIEW[0].load(Relaxed) != 0 {
        tracing::warn!(at = format_args!("{at:#x}"), len, "gpu view: a second reservation like the GPU view; kept the first");
        return;
    }
    VIEW[1].store(len as u64, Relaxed);
    VIEW[0].store(at as u64, Relaxed);
    tracing::info!(at = format_args!("{at:#x}"), len = format_args!("{len:#x}"), "gpu view: reservation seen");
}

/// The table, made at the first mapping call once the view is known
/// (cross-checked with the title's globals then).
fn table() -> Option<&'static Table> {
    if let Some(t) = TABLE.get() {
        return Some(t);
    }
    let (mut base, mut len) = (VIEW[0].load(Relaxed), VIEW[1].load(Relaxed));
    // The title's globals: CPU view base, size, GPU view base (image +0x9e1dac0).
    let globals = globals();
    match (base, globals) {
        (0, Some((_, size, gpu))) if gpu != 0 && size != 0 => {
            tracing::warn!(
                base = format_args!("{gpu:#x}"),
                len = format_args!("{size:#x}"),
                "gpu view: no reservation like it seen; using the title's globals"
            );
            (base, len) = (gpu, size);
        }
        (0, _) => return None,
        (_, Some((_, size, gpu))) if gpu != base || size != len => tracing::warn!(
            base = format_args!("{base:#x}"),
            len = format_args!("{len:#x}"),
            globals_base = format_args!("{gpu:#x}"),
            globals_len = format_args!("{size:#x}"),
            "gpu view: the reservation does not match the title's globals"
        ),
        (_, Some(_)) => tracing::info!("gpu view: the reservation matches the title's globals"),
        (_, None) => tracing::warn!("gpu view: the title's globals could not be read"),
    }
    Some(TABLE.get_or_init(|| Table::new(base, len)))
}

#[cfg(windows)]
fn globals() -> Option<(u64, u64, u64)> {
    let image = IMAGE.load(Relaxed);
    if image == 0 {
        return None;
    }
    let mut bytes = [0u8; 24];
    crate::host::read_into((image + 0x9e1_dac0) as *const u8, &mut bytes).then(|| {
        let v = |i: usize| u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
        (v(0), v(1), v(2))
    })
}

#[cfg(not(windows))]
fn globals() -> Option<(u64, u64, u64)> {
    None
}

/// The layer reads title memory `[at, at + len)` (cheap when off or outside).
#[inline]
pub fn note_read(at: usize, len: usize) {
    if let Some(t) = TABLE.get()
        && t.contains(at as u64)
    {
        t.note_read(at as u64, len as u64);
    }
}

/// The table for a mapping call, if the meter is on and the view known.
pub fn begin() -> Option<&'static Table> {
    if on() { table() } else { None }
}

/// A mapping call whose ranges were noted in `t` took `us`: counted, with a
/// line every five seconds.
pub fn end(t: &Table, us: u64) {
    t.add_call(us);
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let Ok(mut last) = LAST.try_lock() else { return };
    let now = std::time::Instant::now();
    let secs = now.duration_since(*last.get_or_insert(now)).as_secs_f64();
    if secs >= 5.0 {
        *last = Some(now);
        let c = t.take(0);
        // (`VICTORY_MAP_FAST`) Where the time in the calls went.
        tracing::info!("{}{}", line(&c, secs), crate::mapfast::take_line(c.us));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = 0x10_0000_0000;

    fn table() -> Table {
        Table::new(BASE, 64 * PAGE)
    }

    #[test]
    fn run_lengths_and_kinds() {
        let t = table();
        t.note(BASE, 0, 100, 3);
        t.note(BASE + 3 * PAGE, 1, 7, 2);
        t.note(BASE + 5 * PAGE, 3, 0, 1); // left as it is
        assert_eq!(t.get(BASE).unwrap().0, Backing::Pool(100));
        assert_eq!(t.get(BASE + 2 * PAGE).unwrap().0, Backing::Pool(102));
        // Kind 1: every page shows the one constant page.
        assert_eq!(t.get(BASE + 3 * PAGE).unwrap().0, Backing::Constant(7));
        assert_eq!(t.get(BASE + 4 * PAGE).unwrap().0, Backing::Constant(7));
        assert_eq!(t.get(BASE + 5 * PAGE).unwrap().0, Backing::Unmapped);
        let c = t.take(0);
        assert_eq!(c.pages, [3, 2, 0]);
        assert_eq!(c.collapsed, 0);
        // Outside the view.
        t.note(BASE + 63 * PAGE, 0, 1, 3);
        assert_eq!(t.take(0).outside, 2);
        assert_eq!(t.get(BASE + 64 * PAGE), None);
    }

    #[test]
    fn remaps_before_a_read_collapse_into_the_last() {
        let t = table();
        t.note(BASE, 0, 10, 2);
        t.note(BASE, 0, 20, 2);
        t.note(BASE, 1, 30, 2);
        let c = t.take(0);
        assert_eq!(c.collapsed, 4);
        assert_eq!(t.get(BASE).unwrap().0, Backing::Constant(30));
        // Read, then remapped: not collapsed.
        t.note_read(BASE, 1);
        t.note(BASE, 0, 40, 2);
        let c = t.take(0);
        assert_eq!((c.materialised, c.collapsed), (1, 1));
        // The same pages again: nothing changes.
        t.note(BASE, 0, 40, 2);
        assert_eq!(t.take(0).same, 2);
    }

    #[test]
    fn unmaps_of_pages_never_read_are_table_updates() {
        let t = table();
        t.note(BASE, 0, 10, 4);
        t.note_read(BASE + PAGE + 5, 10);
        t.note(BASE, 2, 0, 4);
        t.note(BASE, 2, 0, 1);
        let c = t.take(0);
        assert_eq!((c.unmaps_never_read, c.unmaps_read, c.unmaps_of_nothing), (3, 1, 1));
        assert_eq!(c.materialised, 1);
        assert_eq!(t.get(BASE + PAGE).unwrap().0, Backing::Unmapped);
        // A read of unmapped pages materialises nothing.
        t.note_read(BASE, 4 * PAGE);
        assert_eq!(t.take(0).materialised, 0);
    }

    #[test]
    fn a_change_of_backing_is_a_new_generation_read_again() {
        let t = table();
        t.note(BASE, 0, 10, 1);
        let g0 = t.get(BASE).unwrap().2;
        t.note_read(BASE, 8);
        t.note_read(BASE, 8); // once per mapping
        assert_eq!(t.take(0).materialised, 1);
        assert!(t.get(BASE).unwrap().1);
        t.note(BASE, 0, 11, 1);
        let (b, read, g1) = t.get(BASE).unwrap();
        assert_eq!((b, read, g1), (Backing::Pool(11), false, g0 + 1));
        t.note_read(BASE, 8);
        assert_eq!(t.take(0).materialised, 1);
        // The same backing again keeps its generation.
        t.note(BASE, 0, 11, 1);
        assert_eq!(t.get(BASE).unwrap().2, g1);
        // A read spanning pages counts each mapped one once.
        t.note(BASE + PAGE, 0, 50, 2);
        t.note_read(BASE + PAGE - 1, PAGE + 2);
        assert_eq!(t.take(0).materialised, 2);
    }

    #[test]
    fn the_line_names_its_counts() {
        let c = Counts { calls: 2, pages: [3, 1, 4], materialised: 5, unmaps_never_read: 6, collapsed: 7, us: 1500, ..Default::default() };
        let l = line(&c, 5.0);
        assert!(l.starts_with("gpu view: 2 calls, 8 pages noted, 5 materialised on read, 6 unmaps of pages never materialised, 7 collapsed remaps, 1.5 ms in calls"), "{l}");
    }
}
