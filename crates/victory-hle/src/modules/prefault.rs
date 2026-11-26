//! (`VICTORY_MAP_PREFAULT`, default 0 until it is run again; `=1`: on)
//! Page-table entries of freshly mapped views filled in off the title's
//! threads.
//!
//! Every 64 KiB page the title maps is a new view of its section, in its CPU
//! view and again in its GPU view; a new view has no valid page-table entry,
//! so the first touch of each of its 4 KiB pages is a soft fault (16 a page)
//! on whichever thread touches it first: the title's threads (CPU view) and
//! the layer's readers (GPU view), inside the frame. A streaming burst of
//! 8,000 pages is ~250,000 such faults.
//!
//! One thread ("victory-prefault", below normal priority) reads one byte of
//! each 4 KiB page of what is queued ([`queued`]), with NO lock: the read is
//! the GPU side's guarded copy (`victory_gpu::prefault_touch`), so a view
//! taken away meanwhile ends it harmlessly, and a view replaced meanwhile is
//! only read. Read only: guards (read-only blocks) are never tripped.
//! Queued: (a) at a map, only the 64 KiB pages already resident in the
//! section (written through the layer's lasting view, not discarded since);
//! (b) after a DirectStorage write, the title addresses that show the bytes
//! written (`pages_written`). Never a page that was never written: reading
//! one makes the kernel zero-fill it, RAM the title never used.
//!
//! (Run-59/60, the first version, read every mapped view under the page
//! table's lock, never-written pages and the 6 GiB pool view included.
//! Start-up stalled ~30 s (nothing logged from 02:47:10 to 02:47:40) and the
//! title's own watchdog thread (0x142215a60: 1 s sleeps, a heartbeat table
//! at 0x146844a20) deliberately wrote to address 0 at 0x142215c35: no
//! title-facing call failed, a title thread stopped beating. Likely cause:
//! zero-fill of gigabytes of untouched pages and the worker's lock holds
//! starving the title's mapping calls.)
//!
//! Measured (unit test `prefault_measure`, 4 MiB section, written through a
//! lasting view first), faults for 1,024 pages of 4 KiB: no prefault: a full
//! read 1,119, a write after it 47, a full write first 1,084; after the read
//! touch (1,073 faults on the worker, ~0.48 us a page): read 0, write after
//! 47, write first 0; `PrefetchVirtualMemory` (2 faults) leaves the read at
//! 1,024 and the write first at 1,026 (it brings pages in, it does not fill
//! the view's entries): not used.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Condvar, Mutex};

/// `VICTORY_MAP_PREFAULT` (default off).
pub(crate) fn on() -> bool {
    #[cfg(test)]
    {
        TEST_ON.load(Relaxed)
    }
    #[cfg(not(test))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            let on = std::env::var_os("VICTORY_MAP_PREFAULT").is_some_and(|v| v == "1");
            tracing::info!(on, "VICTORY_MAP_PREFAULT: page-table entries of freshly mapped views over resident pages filled in on the victory-prefault thread (no lock)");
            on
        })
    }
}

#[cfg(test)]
pub(crate) static TEST_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A range of title addresses to read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ask {
    pub at: usize,
    pub len: usize,
}

/// The most asks waiting; past it new ones are dropped (counted skipped).
const MOST: usize = 1 << 16;
/// Bytes read before the worker lets other threads run.
const PIECE: usize = 256 << 10;

static QUEUE: Mutex<Vec<Ask>> = Mutex::new(Vec::new());
static MORE: Condvar = Condvar::new();

/// Meter since the last line: 4 KiB pages read, ranges (asks merged where
/// they touch), ns on the worker, 4 KiB pages skipped (not there any more,
/// or dropped from a full queue).
static M: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
const PAGES: usize = 0;
const RANGES: usize = 1;
const NS: usize = 2;
const SKIPPED: usize = 3;

/// Queues `ask` for the worker (started on first use). Cheap: no lock but
/// the queue's.
pub(crate) fn queued(ask: Ask) {
    if ask.len == 0 {
        return;
    }
    // (Tests read the queue themselves: [`drain_now`].)
    #[cfg(not(test))]
    {
        static STARTED: std::sync::Once = std::sync::Once::new();
        STARTED.call_once(|| {
            let spawned = std::thread::Builder::new().name("victory-prefault".into()).spawn(work);
            if spawned.is_err() {
                tracing::error!("VICTORY_MAP_PREFAULT: the victory-prefault thread did not start; views fault on first touch as before");
            }
        });
    }
    let mut q = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if q.len() >= MOST {
        M[SKIPPED].fetch_add(ask.len.div_ceil(4096) as u64, Relaxed);
        return;
    }
    q.push(ask);
    drop(q);
    MORE.notify_one();
}

#[cfg_attr(test, allow(dead_code))]
fn work() {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL};
    // SAFETY: this thread's own priority.
    unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL) };
    loop {
        let _idle = victory_gpu::hangwatch::at(&"prefault worker: idle, waiting for views to touch");
        let mut asks = {
            let mut q = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
            while q.is_empty() {
                q = MORE.wait(q).unwrap_or_else(|e| e.into_inner());
            }
            std::mem::take(&mut *q)
        };
        drop(_idle);
        let _site = victory_gpu::hangwatch::at(&"prefault worker: touching views");
        let t = std::time::Instant::now();
        process(&mut asks);
        M[NS].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    }
}

/// (Tests) What is queued, read on this thread now.
#[cfg(test)]
pub(crate) fn drain_now() {
    let mut asks = std::mem::take(&mut *QUEUE.lock().unwrap_or_else(|e| e.into_inner()));
    process(&mut asks);
}

/// The asks as ranges of whole 4 KiB pages, in order, merged where they
/// touch or overlap.
pub(crate) fn merged(asks: &mut [Ask]) -> Vec<(usize, usize)> {
    asks.sort_unstable_by_key(|a| a.at);
    let mut out: Vec<(usize, usize)> = Vec::new();
    for a in asks.iter() {
        let (lo, hi) = (a.at & !4095, a.at.saturating_add(a.len).next_multiple_of(4096));
        match out.last_mut() {
            Some((_, end)) if lo <= *end => *end = (*end).max(hi),
            _ => out.push((lo, hi)),
        }
    }
    out
}

/// Reads the asks, merged, in pieces (other threads let run between
/// them); a piece that ends early (a page not there) skips the rest of its
/// range.
pub(crate) fn process(asks: &mut [Ask]) {
    for (lo, hi) in merged(asks) {
        M[RANGES].fetch_add(1, Relaxed);
        let mut from = lo;
        while from < hi {
            let to = (from + PIECE).min(hi);
            let (read, gone) = touch_with(from, to - from);
            M[PAGES].fetch_add(read as u64, Relaxed);
            if gone {
                M[SKIPPED].fetch_add(((hi - from) / 4096 - read) as u64, Relaxed);
                break;
            }
            from = to;
            std::thread::yield_now();
        }
    }
}

fn touch_with(at: usize, len: usize) -> (usize, bool) {
    victory_gpu::prefault_touch(at, len)
}

/// Reads one byte of each 4 KiB page of `[at, at + len)`; the pages read.
#[cfg(test)]
pub(crate) fn touch(at: usize, len: usize) -> usize {
    touch_with(at, len).0
}

/// The fault drivers line's part, the counts since the last call.
pub(crate) fn take_line() -> String {
    let m: Vec<u64> = M.iter().map(|a| a.swap(0, Relaxed)).collect();
    format!(
        "; prefaulted {} pages in {} ranges ({:.1} ms on the map worker), skipped unmapped {} (4 KiB pages, VICTORY_MAP_PREFAULT={})",
        m[PAGES],
        m[RANGES],
        m[NS] as f64 / 1e6,
        m[SKIPPED],
        u8::from(on())
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MEM_COMMIT, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READWRITE,
        PrefetchVirtualMemory, SEC_RESERVE, UnmapViewOfFile, VirtualAlloc, WIN32_MEMORY_RANGE_ENTRY,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    fn faults() -> u64 {
        victory_gpu::faults::process_faults()
    }

    /// Faults of a read of one byte of every 4 KiB page, then of a write.
    fn read_then_write(at: usize, len: usize) -> (u64, u64) {
        let f0 = faults();
        for p in (at..at + len).step_by(4096) {
            // SAFETY: a committed read-write view of ours.
            unsafe { std::ptr::read_volatile(p as *const u8) };
        }
        let f1 = faults();
        for p in (at..at + len).step_by(4096) {
            // SAFETY: as above.
            unsafe { std::ptr::write_volatile((p + 8) as *mut u8, 7) };
        }
        (f1 - f0, faults() - f1)
    }

    /// The measurement of the module docs: a 4 MiB section written whole
    /// through a lasting view (resident), then a second view of it mapped
    /// (the title's new view); faults of a full read and a full write of the
    /// new view with no prefault, after a read touch ([`touch`]), after
    /// `PrefetchVirtualMemory`, and the faults the prefault itself took.
    #[test]
    fn prefault_measure() {
        const LEN: usize = 4 << 20;
        // SAFETY: plain section and views of our own, released at the end.
        unsafe {
            let section = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE | SEC_RESERVE, 0, LEN as u32, std::ptr::null());
            assert!(!section.is_null());
            let lasting = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 0).Value as usize;
            assert_ne!(lasting, 0);
            assert!(!VirtualAlloc(lasting as *const _, LEN, MEM_COMMIT, PAGE_READWRITE).is_null());
            std::ptr::write_bytes(lasting as *mut u8, 0x5a, LEN);
            let mut out = Vec::new();
            for how in ["none", "touch", "prefetch"] {
                let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 0).Value as usize;
                assert_ne!(view, 0);
                let f0 = faults();
                match how {
                    "touch" => assert_eq!(touch(view, LEN), LEN / 4096),
                    "prefetch" => {
                        let range = WIN32_MEMORY_RANGE_ENTRY { VirtualAddress: view as *mut _, NumberOfBytes: LEN };
                        assert_ne!(PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0), 0);
                    }
                    _ => {}
                }
                let cost = faults() - f0;
                let (read, write) = read_then_write(view, LEN);
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ });
                // A write first (the title filling a freshly mapped page).
                let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 0).Value as usize;
                let t = std::time::Instant::now();
                if how == "touch" {
                    touch(view, LEN);
                }
                let ns = t.elapsed().as_nanos() as u64 / (LEN / 4096) as u64;
                let f1 = faults();
                for p in (view..view + LEN).step_by(4096) {
                    std::ptr::write_volatile(p as *mut u8, 9);
                }
                let write_first = faults() - f1;
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ });
                out.push((how, cost, read, write, write_first, ns));
            }
            println!("prefault_measure (1024 pages of 4 KiB, resident): (how, faults of the prefault, of a full read after, of a full write after that, of a full write first, prefault ns a page) {out:?}");
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: lasting as *mut _ });
            CloseHandle(section);
            let none = out[0];
            let touched = out[1];
            assert!(none.2 >= (LEN / 4096) as u64 * 9 / 10, "a fresh view faults on first read: {out:?}");
            assert!(touched.2 < (LEN / 4096) as u64 / 10, "after the touch a read does not fault: {out:?}");
        }
    }

    /// Asks merged into whole pages; a range that is not mapped ends early
    /// with no fault (skipped), the rest is read.
    #[test]
    fn asks_merge_and_a_missing_page_is_skipped() {
        let mut asks = [Ask { at: 0x10_0100, len: 16 }, Ask { at: 0x10_0000, len: 0x2000 }, Ask { at: 0x20_0000, len: 1 }];
        assert_eq!(merged(&mut asks), [(0x10_0000, 0x10_2000), (0x20_0000, 0x20_1000)]);
        let held = vec![1u8; 3 * 4096];
        assert!(!touch_with(held.as_ptr() as usize, 3 * 4096).1);
        // Address 4096 is never mapped: ends at once, no crash.
        assert_eq!(touch_with(4096, 8192), (0, true));
        let _ = take_line();
        process(&mut [Ask { at: 4096, len: 8192 }]);
        let line = take_line();
        assert!(line.contains("prefaulted 0 pages in 1 ranges") && line.contains("skipped unmapped 2"), "{line}");
    }
}
