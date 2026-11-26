//! (F3, `VICTORY_MAP_FAST`) Cheaper `CopyPageMappingsBatchX` calls, and the
//! timing split of the calls for the `VICTORY_LAZY_GPU_VIEW` meter line.
//!
//! What the title sends (its batch executor 0x140f2c470): for every page it
//! maps into the GPU view, one range `{kind 1, pool page, 1}` of its own,
//! each a different pool page taken off its free list (never a run of one
//! page repeated), in batches whose ranges cover consecutive destination
//! pages; for every run it unmaps, one range `{kind 2, 0, n}`. Taken range
//! by range that is one remapping of the runtime's page table per page:
//! the lock, a hold over the bytes, a placeholder split, a view, a commit
//! and a set of notes to the GPU side for each, and one unmap per page view
//! later.
//!
//! With the switch (default on; `VICTORY_MAP_FAST=0`: range by range, as
//! before) ranges that follow one another in destination and in the pool
//! are mapped as one run ([`push`]), unmapped runs that follow one another
//! as one; the runtime maps such a run as one view where the pool pages
//! are one run of one section (joining the placeholders under it), and
//! notes the GPU side once per run of views instead of once per view.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether ranges are mapped as runs (`VICTORY_MAP_FAST`, default on).
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MAP_FAST").is_none_or(|v| v != "0"))
}

/// Whether `XMemMapPhysicalPagesScatter` (the title's CPU view) is grouped
/// into runs too (`VICTORY_MAP_FAST_CPU`, default OFF; needs
/// `VICTORY_MAP_FAST` on). Off since run-45: a run there joins placeholders
/// and cuts views, which for a moment unmaps CPU-view pages the title's own
/// threads read with no lock the layer can take (access violation at scene
/// load). Off: page by page, as before the grouping.
pub fn on_cpu() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| on() && std::env::var_os("VICTORY_MAP_FAST_CPU").is_some_and(|v| v == "1"))
}

/// Whether the split is timed: with the meter (`VICTORY_LAZY_GPU_VIEW=1`).
pub fn timing() -> bool {
    crate::gpuview::on()
}

/// Bytes per page of the view.
const PAGE: u64 = 64 << 10;

/// The most pages one run takes: a run is one hold over its bytes, which
/// readers of them wait out.
pub const MOST_PAGES: u64 = 1024;

/// One step of a mapping call, as the runtime is asked for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// `pages` pages from `from` on (pool addresses) at `to` on.
    Map { to: u64, from: u64, pages: u64 },
    /// The one page at `from` at each of `pages` pages from `to` on (a kind
    /// 1 range of more than one page): page by page.
    Repeat { to: u64, from: u64, pages: u64 },
    /// `pages` pages from `to` on unmapped.
    Unmap { to: u64, pages: u64 },
    /// A range the runtime refuses (unknown kind, pages outside the pool).
    Bad,
}

/// Adds `op` to `ops`, joined with the last one if it carries it on: maps
/// whose destinations and pool pages both follow on, unmaps whose
/// destinations do (up to [`MOST_PAGES`]). Nothing else is joined, so the
/// steps keep their order and every page is changed once, as range by
/// range.
pub fn push(ops: &mut Vec<Op>, op: Op) {
    if let Some(last) = ops.last_mut() {
        match (last, op) {
            (Op::Map { to, from, pages }, Op::Map { to: t, from: f, pages: p })
                if *to + *pages * PAGE == t && *from + *pages * PAGE == f && *pages + p <= MOST_PAGES =>
            {
                *pages += p;
                return;
            }
            (Op::Unmap { to, pages }, Op::Unmap { to: t, pages: p }) if *to + *pages * PAGE == t && *pages + p <= MOST_PAGES => {
                *pages += p;
                return;
            }
            _ => {}
        }
    }
    if !matches!(op, Op::Map { pages: 0, .. } | Op::Unmap { pages: 0, .. } | Op::Repeat { pages: 0, .. }) {
        ops.push(op);
    }
}

/// Where the time of the mapping calls goes (nanoseconds, since the last
/// line). Disjoint: each is timed around its own calls.
#[derive(Debug, Clone, Copy)]
pub enum Split {
    /// Waiting for the runtime's page table (DirectStorage workers and the
    /// title's other mapping calls hold it).
    LockWait,
    /// Taking the hold over the bytes changed: settling read-back writes
    /// and waiting for readers in place of those bytes.
    Hold,
    /// `UnmapViewOfFile2`.
    UnmapOs,
    /// Placeholder split, `MapViewOfFile3` and commit of each view.
    MapOs,
    /// Placeholder queries, splits and joins ahead of a run.
    Placeholder,
    /// Notes to the GPU side: remapped (textures due, views forgotten, guards
    /// and write-watch state dropped), view, really unmapped, will-write.
    Notes,
    /// All of the runtime's `map_alias` calls (the rest of it is its own
    /// bookkeeping and look-ups).
    Alias,
}

/// What is counted alongside the split.
#[derive(Debug, Clone, Copy)]
pub enum Count {
    /// `UnmapViewOfFile2` calls.
    UnmapCalls,
    /// Views mapped.
    Views,
    /// Placeholder calls (queries, splits, joins).
    PlaceholderCalls,
    /// Runs of more than one page asked for as one, and their pages.
    Runs,
    RunPages,
    /// Kind 1 ranges of more than one page (one page repeated).
    Repeated,
    /// Views cut by an unmap and their rest mapped again.
    Cut,
    /// Runs that failed as one and were done page by page.
    PageByPage,
}

/// Which mapping path a part of the split is charged to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    /// `CopyPageMappingsBatchX` (the runtime's `map_alias`): the GPU view.
    GpuView = 0,
    /// `XMemMapPhysicalPagesScatter`: the title's CPU view of the pool.
    CpuView = 1,
}

static NS: [[AtomicU64; 7]; 2] = [const { [const { AtomicU64::new(0) }; 7] }; 2];
static N: [[AtomicU64; 8]; 2] = [const { [const { AtomicU64::new(0) }; 8] }; 2];
/// Pages asked of `map_alias` (mapped or unmapped) since the last line.
static GPU_PAGES: AtomicU64 = AtomicU64::new(0);
/// The CPU view since the last line: calls, pages, runs (as grouped; one
/// per page with `VICTORY_MAP_FAST=0`), nanoseconds in the calls.
static CPU: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Adds `ns` to `split` (the GPU view).
pub fn add(split: Split, ns: u64) {
    add_to(Path::GpuView, split, ns);
}

/// Adds `ns` to `split` of `path`.
pub fn add_to(path: Path, split: Split, ns: u64) {
    NS[path as usize][split as usize].fetch_add(ns, Relaxed);
}

/// Adds `n` to `count` (the GPU view).
pub fn count(c: Count, n: u64) {
    count_to(Path::GpuView, c, n);
}

/// Adds `n` to `count` of `path`.
pub fn count_to(path: Path, c: Count, n: u64) {
    N[path as usize][c as usize].fetch_add(n, Relaxed);
}

/// Pages asked of one `map_alias` call.
pub fn gpu_pages(n: u64) {
    GPU_PAGES.fetch_add(n, Relaxed);
}

/// One `XMemMapPhysicalPagesScatter` call: its pages, the runs it was done
/// in, its time. Counted always (the frame log's "cpu view"), the split
/// only while timed.
pub fn cpu_call(pages: u64, runs: u64, ns: u64) {
    CPU[0].fetch_add(1, Relaxed);
    CPU[1].fetch_add(pages, Relaxed);
    CPU[2].fetch_add(runs, Relaxed);
    CPU[3].fetch_add(ns, Relaxed);
    crate::framestream::mapping_call(ns);
    #[cfg(windows)]
    {
        crate::framelog::add(crate::framelog::T::CpuViewCalls, 1);
        crate::framelog::add(crate::framelog::T::CpuViewPages, pages);
        crate::framelog::add(crate::framelog::T::CpuViewUs, ns / 1000);
    }
}

/// Milliseconds per 1,000 pages of each part of a split (`ns` as
/// [`NS`]'s), over `pages` pages.
fn per_thousand(ns: &[u64], pages: u64) -> String {
    if pages == 0 {
        return "no pages".to_owned();
    }
    let k = |v: u64| v as f64 / 1e6 * 1000.0 / pages as f64;
    let parts: u64 = ns[..6].iter().sum();
    format!(
        "lock wait {:.1}, hold {:.1}, unmap calls {:.1}, map calls {:.1}, placeholders {:.1}, notes {:.1}, bookkeeping {:.1}",
        k(ns[Split::LockWait as usize]),
        k(ns[Split::Hold as usize]),
        k(ns[Split::UnmapOs as usize]),
        k(ns[Split::MapOs as usize]),
        k(ns[Split::Placeholder as usize]),
        k(ns[Split::Notes as usize]),
        k(ns[Split::Alias as usize].saturating_sub(parts)),
    )
}

/// The split since the last call, taken: as the meter line says it, given
/// the microseconds in the calls over the same time; then the CPU view's.
pub fn take_line(call_us: u64) -> String {
    let ns: Vec<u64> = NS[0].iter().map(|a| a.swap(0, Relaxed)).collect();
    let n: Vec<u64> = N[0].iter().map(|a| a.swap(0, Relaxed)).collect();
    let pages = GPU_PAGES.swap(0, Relaxed);
    format!("{}; ms per 1,000 pages ({pages} pages): {}{}", split_line(&ns, &n, call_us), per_thousand(&ns, pages), take_cpu_line())
}

/// The CPU view since the last call, taken: " | cpu view: C calls, P
/// pages, R runs, T ms; split ...".
pub fn take_cpu_line() -> String {
    let c: Vec<u64> = CPU.iter().map(|a| a.swap(0, Relaxed)).collect();
    let mut ns: Vec<u64> = NS[1].iter().map(|a| a.swap(0, Relaxed)).collect();
    let n: Vec<u64> = N[1].iter().map(|a| a.swap(0, Relaxed)).collect();
    // The whole of the calls stands where `map_alias`'s does.
    ns[Split::Alias as usize] = c[3];
    let ms = |v: u64| v as f64 / 1e6;
    let parts: u64 = ns[..6].iter().sum();
    format!(
        " | cpu view (XMemMapPhysicalPagesScatter, VICTORY_MAP_FAST={}): {} calls, {} pages, {} runs, {:.1} ms ({:.1} ms per 1,000 pages); \
         split ms{}: lock wait {:.1}, hold {:.1}, unmap calls {:.1} ({}), map calls {:.1} ({} views), placeholders {:.1} ({} calls), \
         notes {:.1}, bookkeeping {:.1}; ms per 1,000 pages: {}; views cut {}, runs redone page by page {}",
        u8::from(on()),
        c[0],
        c[1],
        c[2],
        ms(c[3]),
        if c[1] == 0 { 0.0 } else { ms(c[3]) * 1000.0 / c[1] as f64 },
        if timing() { "" } else { " (not timed: VICTORY_LAZY_GPU_VIEW=1 times them)" },
        ms(ns[Split::LockWait as usize]),
        ms(ns[Split::Hold as usize]),
        ms(ns[Split::UnmapOs as usize]),
        n[Count::UnmapCalls as usize],
        ms(ns[Split::MapOs as usize]),
        n[Count::Views as usize],
        ms(ns[Split::Placeholder as usize]),
        n[Count::PlaceholderCalls as usize],
        ms(ns[Split::Notes as usize]),
        ms(c[3].saturating_sub(parts)),
        per_thousand(&ns, c[1]),
        n[Count::Cut as usize],
        n[Count::PageByPage as usize],
    )
}

fn split_line(ns: &[u64], n: &[u64], call_us: u64) -> String {
    let ms = |v: u64| v as f64 / 1e6;
    let parts: u64 = ns[..6].iter().sum();
    let alias = ns[Split::Alias as usize];
    format!(
        " | split ms (VICTORY_MAP_FAST={}): page-table lock wait {:.1}, hold {:.1}, unmap calls {:.1} ({}), map calls {:.1} ({} views), \
         placeholders {:.1} ({} calls), notes {:.1}, bookkeeping {:.1}, outside the runtime {:.1}; runs {} of {} pages, \
         kind-1 ranges of more than one page {}, views cut {}, runs redone page by page {}",
        u8::from(on()),
        ms(ns[Split::LockWait as usize]),
        ms(ns[Split::Hold as usize]),
        ms(ns[Split::UnmapOs as usize]),
        n[Count::UnmapCalls as usize],
        ms(ns[Split::MapOs as usize]),
        n[Count::Views as usize],
        ms(ns[Split::Placeholder as usize]),
        n[Count::PlaceholderCalls as usize],
        ms(ns[Split::Notes as usize]),
        ms(alias.saturating_sub(parts)),
        (call_us as f64 / 1e3 - ms(alias)).max(0.0),
        n[Count::Runs as usize],
        n[Count::RunPages as usize],
        n[Count::Repeated as usize],
        n[Count::Cut as usize],
        n[Count::PageByPage as usize],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const AT: u64 = 0x7_0000_0000;
    const POOL: u64 = 0x2_0000_0000;

    fn map(to: u64, from: u64, pages: u64) -> Op {
        Op::Map { to: AT + to * PAGE, from: POOL + from * PAGE, pages }
    }

    #[test]
    fn ranges_that_follow_on_are_one_run() {
        let mut ops = Vec::new();
        // The title's way: one page each, pool pages 10, 11, 12, then 40.
        for (k, from) in [10, 11, 12, 40].into_iter().enumerate() {
            push(&mut ops, map(k as u64, from, 1));
        }
        assert_eq!(ops, [map(0, 10, 3), map(3, 40, 1)]);
        // A gap in the destination (a kind 3 range) starts a run.
        push(&mut ops, map(5, 41, 1));
        assert_eq!(ops.len(), 3);
        // Unmaps that follow on are one; a map after them starts anew.
        push(&mut ops, Op::Unmap { to: AT + 6 * PAGE, pages: 2 });
        push(&mut ops, Op::Unmap { to: AT + 8 * PAGE, pages: 3 });
        push(&mut ops, map(11, 42, 1));
        assert_eq!(ops[3..], [Op::Unmap { to: AT + 6 * PAGE, pages: 5 }, map(11, 42, 1)]);
        // Pool pages that follow on at a destination that does not: apart.
        push(&mut ops, map(13, 43, 1));
        assert_eq!(ops.len(), 6);
    }

    #[test]
    fn order_and_odd_ranges_are_kept() {
        let mut ops = Vec::new();
        // An unmap then a map of the same page: two steps, in order.
        push(&mut ops, Op::Unmap { to: AT, pages: 1 });
        push(&mut ops, map(0, 7, 1));
        // The same pool page again (a page repeated) is not a run.
        push(&mut ops, map(1, 7, 1));
        // A repeated page and a refused range are never joined.
        push(&mut ops, Op::Repeat { to: AT + 2 * PAGE, from: POOL + 8 * PAGE, pages: 4 });
        push(&mut ops, map(6, 9, 1));
        push(&mut ops, Op::Bad);
        push(&mut ops, map(7, 10, 1));
        // Nothing of no pages.
        push(&mut ops, map(8, 11, 0));
        assert_eq!(
            ops,
            [
                Op::Unmap { to: AT, pages: 1 },
                map(0, 7, 1),
                map(1, 7, 1),
                Op::Repeat { to: AT + 2 * PAGE, from: POOL + 8 * PAGE, pages: 4 },
                map(6, 9, 1),
                Op::Bad,
                map(7, 10, 1),
            ]
        );
    }

    #[test]
    fn runs_stop_at_the_most_pages() {
        let mut ops = Vec::new();
        for k in 0..MOST_PAGES + 3 {
            push(&mut ops, map(k, 100 + k, 1));
        }
        assert_eq!(ops, [map(0, 100, MOST_PAGES), map(MOST_PAGES, 100 + MOST_PAGES, 3)]);
    }

    #[test]
    fn the_split_line_names_its_parts() {
        let ns = [1_000_000, 2_000_000, 3_000_000, 4_000_000, 500_000, 6_000_000, 20_000_000];
        let n = [7, 8, 9, 10, 11, 12, 13, 14];
        let l = split_line(&ns, &n, 25_000);
        assert!(l.contains("page-table lock wait 1.0, hold 2.0, unmap calls 3.0 (7), map calls 4.0 (8 views)"), "{l}");
        assert!(l.contains("placeholders 0.5 (9 calls), notes 6.0, bookkeeping 3.5, outside the runtime 5.0"), "{l}");
        assert!(l.contains("runs 10 of 11 pages, kind-1 ranges of more than one page 12, views cut 13, runs redone page by page 14"), "{l}");
    }
}
