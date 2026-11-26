//! Finding F10 (meter only, no behaviour change): how much of the title's
//! streaming work runs on its frame thread (the thread that calls Present):
//! time in the layer's two page-mapping calls (`XMemMapPhysicalPagesScatter`
//! for the CPU view, `CopyPageMappingsBatchX` for the GPU view) on that
//! thread and calls from other threads; back-to-back DirectStorage status
//! polls answered "not complete" on it (the batch wait 0x1400e8960, an
//! IsComplete + Sleep(1) loop) and the time from the first such answer to
//! the first "complete" or the next other DirectStorage call; enqueues that
//! blocked on a full queue; batches outstanding (status entries enqueued,
//! not yet written complete) sampled each frame; and each batch's time from
//! enqueue to complete, with how often a newer one finished while an older
//! one was still outstanding (the title retires them oldest first).
//!
//! Per frame in the frame log ("frame thread streaming: ..."), and a line a
//! second: "frame thread streaming: ...".

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::time::Instant;

/// `VICTORY_DSTORAGE_OLDEST_FIRST` as the runtime read it (for the line).
pub static OLDEST_FIRST_MODE: AtomicU32 = AtomicU32::new(0);

/// The thread that called Present last (0 before the first).
static PRESENT_TID: AtomicU32 = AtomicU32::new(0);

fn tid() -> u32 {
    #[cfg(windows)]
    {
        // SAFETY: no arguments.
        unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
    }
    #[cfg(not(windows))]
    0
}

/// (Run-46) By calling thread since the last line: mapping calls, their ns;
/// DirectStorage enqueues/submits; status polls (any answer); not-complete
/// polls. The frame thread is not the one that maps (run-46: "on the frame
/// thread 0"), so the line names the threads that do.
static BY_THREAD: Mutex<Option<std::collections::HashMap<u32, [u64; 5]>>> = Mutex::new(None);
const T_MAP: usize = 0;
const T_MAP_NS: usize = 1;
const T_DS: usize = 2;
const T_POLLS: usize = 3;
const T_PENDING: usize = 4;

fn by_thread(what: usize, n: u64) {
    let mut g = BY_THREAD.lock().unwrap_or_else(|e| e.into_inner());
    g.get_or_insert_with(Default::default).entry(tid()).or_default()[what] += n;
}

/// A thread's name (its description), cached; empty if it has none.
fn thread_name(id: u32) -> String {
    static NAMES: Mutex<Option<std::collections::HashMap<u32, String>>> = Mutex::new(None);
    let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    let names = names.get_or_insert_with(Default::default);
    if let Some(n) = names.get(&id) {
        return n.clone();
    }
    #[cfg(windows)]
    let name = {
        use windows_sys::Win32::Foundation::{CloseHandle, LocalFree};
        use windows_sys::Win32::System::Threading::{GetThreadDescription, OpenThread, THREAD_QUERY_LIMITED_INFORMATION};
        // SAFETY: a handle we open and close; the system allocates the text,
        // freed with LocalFree.
        unsafe {
            let h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, id);
            let mut text: *mut u16 = std::ptr::null_mut();
            let mut name = String::new();
            if !h.is_null() {
                if GetThreadDescription(h, &mut text) >= 0 && !text.is_null() {
                    let len = (0..).take_while(|&i| *text.add(i) != 0).count();
                    name = String::from_utf16_lossy(std::slice::from_raw_parts(text, len));
                    LocalFree(text.cast());
                }
                CloseHandle(h);
            }
            name
        }
    };
    #[cfg(not(windows))]
    let name = String::new();
    names.insert(id, name.clone());
    name
}

/// The top three threads by mapping ms, by DirectStorage calls and by
/// polls, since the last line.
fn by_thread_line() -> String {
    let all = BY_THREAD.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
    let mut rows: Vec<(u32, [u64; 5])> = all.into_iter().collect();
    let top = |rows: &mut Vec<(u32, [u64; 5])>, key: usize, show: &dyn Fn(&[u64; 5]) -> String| -> String {
        rows.sort_by_key(|r| std::cmp::Reverse((r.1[key], r.1[T_MAP])));
        rows.iter()
            .take(3)
            .filter(|r| r.1[key] > 0)
            .map(|(id, c)| format!("{id} '{}' {}", thread_name(*id), show(c)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let map = top(&mut rows, T_MAP_NS, &|c| format!("{} calls {:.1} ms", c[T_MAP], c[T_MAP_NS] as f64 / 1e6));
    let ds = top(&mut rows, T_DS, &|c| format!("{} calls", c[T_DS]));
    let polls = top(&mut rows, T_POLLS, &|c| format!("{} polls ({} not complete)", c[T_POLLS], c[T_PENDING]));
    format!("; by thread: mapping [{map}], dstorage enqueue/submit [{ds}], status polls [{polls}]")
}

fn on_frame_thread() -> bool {
    let t = PRESENT_TID.load(Relaxed);
    t != 0 && t == tid()
}

/// Since the last line: mapping calls on the frame thread, their ns; on
/// other threads; not-complete polls, their streak ns; enqueue-blocked ns;
/// frames sampled; batches completed, their ns, the most ns; newer finished
/// first.
static C: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
const MAP_CALLS: usize = 0;
const MAP_NS: usize = 1;
const MAP_OTHER: usize = 2;
const POLLS: usize = 3;
const POLL_NS: usize = 4;
const BLOCKED_NS: usize = 5;
const DONE: usize = 6;
const DONE_NS: usize = 7;
const DONE_MAX_NS: usize = 8;
const NEWER_FIRST: usize = 9;
const OUT_MAX: usize = 10;
const OUT_FULL_FRAMES: usize = 11;

/// (F10, `VICTORY_DSTORAGE_OLDEST_FIRST`) Batches completed while an older
/// one was still outstanding is [`NEWER_FIRST`]; for the batch that was the
/// oldest outstanding when it completed: how many, their ns from enqueue,
/// the most.
static OLDEST: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// More of the line, from the runtime (`VICTORY_MAP_DEFER`'s counts).
static EXTRA: Mutex<Vec<fn() -> String>> = Mutex::new(Vec::new());

/// Adds `part` (its text said at the end of each line) to the line.
pub fn add_to_line(part: fn() -> String) {
    let mut e = EXTRA.lock().unwrap_or_else(|e| e.into_inner());
    if !e.iter().any(|&p| std::ptr::fn_addr_eq(p, part)) {
        e.push(part);
    }
}

/// The batches outstanding: (status array, index) by enqueue order.
struct Outstanding {
    next: u64,
    by_key: std::collections::HashMap<(usize, u32), (u64, Instant)>,
    by_seq: BTreeMap<u64, (usize, u32)>,
}

static OUT: Mutex<Option<Outstanding>> = Mutex::new(None);

/// Seconds since the last line in which the outstanding count reached 16,
/// and seconds counted.
static SECONDS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

thread_local! {
    /// The start of this thread's streak of not-complete polls.
    static STREAK: std::cell::Cell<Option<Instant>> = const { std::cell::Cell::new(None) };
}

#[cfg(windows)]
fn frame_add(t: crate::framelog::T, n: u64) {
    crate::framelog::add(t, n);
}

/// One page-mapping call of `ns` nanoseconds on this thread.
pub fn mapping_call(ns: u64) {
    {
        let mut g = BY_THREAD.lock().unwrap_or_else(|e| e.into_inner());
        let c = g.get_or_insert_with(Default::default).entry(tid()).or_default();
        c[T_MAP] += 1;
        c[T_MAP_NS] += ns;
    }
    if on_frame_thread() {
        C[MAP_CALLS].fetch_add(1, Relaxed);
        C[MAP_NS].fetch_add(ns, Relaxed);
        #[cfg(windows)]
        {
            frame_add(crate::framelog::T::FrameMapCalls, 1);
            frame_add(crate::framelog::T::FrameMapUs, ns / 1000);
        }
    } else {
        C[MAP_OTHER].fetch_add(1, Relaxed);
        #[cfg(windows)]
        frame_add(crate::framelog::T::OtherMapCalls, 1);
    }
}

/// Ends this thread's streak of not-complete polls, if one is under way.
fn end_streak() {
    if let Some(start) = STREAK.with(|s| s.take()) {
        let ns = start.elapsed().as_nanos() as u64;
        C[POLL_NS].fetch_add(ns, Relaxed);
        #[cfg(windows)]
        frame_add(crate::framelog::T::PollWaitUs, ns / 1000);
    }
}

/// The title polled a DirectStorage status; `complete`: what it was told.
pub fn status_polled(complete: bool) {
    {
        let mut g = BY_THREAD.lock().unwrap_or_else(|e| e.into_inner());
        let c = g.get_or_insert_with(Default::default).entry(tid()).or_default();
        c[T_POLLS] += 1;
        c[T_PENDING] += u64::from(!complete);
    }
    if !on_frame_thread() {
        return;
    }
    if complete {
        end_streak();
        return;
    }
    C[POLLS].fetch_add(1, Relaxed);
    #[cfg(windows)]
    frame_add(crate::framelog::T::Polls, 1);
    STREAK.with(|s| {
        if s.get().is_none() {
            s.set(Some(Instant::now()));
        }
    });
}

/// Any other DirectStorage call of the title on this thread: ends a streak.
pub fn other_call() {
    by_thread(T_DS, 1);
    if on_frame_thread() {
        end_streak();
    }
}

/// An enqueue on this thread waited `ns` for room in its queue.
pub fn enqueue_blocked(ns: u64) {
    if on_frame_thread() {
        C[BLOCKED_NS].fetch_add(ns, Relaxed);
        #[cfg(windows)]
        frame_add(crate::framelog::T::EnqueueBlockedUs, ns / 1000);
    }
}

/// A status entry (`array`, `index`) was enqueued: a batch outstanding.
pub fn status_enqueued(array: usize, index: u32) {
    let mut g = OUT.lock().unwrap_or_else(|e| e.into_inner());
    let o = g.get_or_insert_with(|| Outstanding { next: 0, by_key: Default::default(), by_seq: BTreeMap::new() });
    let seq = o.next;
    o.next += 1;
    if let Some((old, _)) = o.by_key.insert((array, index), (seq, Instant::now())) {
        o.by_seq.remove(&old);
    }
    o.by_seq.insert(seq, (array, index));
}

/// The status entry (`array`, `index`) was written: its batch complete.
pub fn status_written(array: usize, index: u32) {
    let mut g = OUT.lock().unwrap_or_else(|e| e.into_inner());
    let Some(o) = g.as_mut() else { return };
    let Some((seq, at)) = o.by_key.remove(&(array, index)) else { return };
    o.by_seq.remove(&seq);
    let ns = at.elapsed().as_nanos() as u64;
    if o.by_seq.first_key_value().is_some_and(|(&older, _)| older < seq) {
        C[NEWER_FIRST].fetch_add(1, Relaxed);
    } else {
        OLDEST[0].fetch_add(1, Relaxed);
        OLDEST[1].fetch_add(ns, Relaxed);
        OLDEST[2].fetch_max(ns, Relaxed);
    }
    C[DONE].fetch_add(1, Relaxed);
    C[DONE_NS].fetch_add(ns, Relaxed);
    C[DONE_MAX_NS].fetch_max(ns, Relaxed);
}

/// The status entry (`array`, `index`) will never be written (its queue
/// closed): no longer outstanding, not counted complete.
pub fn status_dropped(array: usize, index: u32) {
    let mut g = OUT.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(o) = g.as_mut()
        && let Some((seq, _)) = o.by_key.remove(&(array, index))
    {
        o.by_seq.remove(&seq);
    }
}

/// The frame thread presents: noted, the outstanding count sampled, and
/// the line said once a second.
pub fn presented() {
    PRESENT_TID.store(tid(), Relaxed);
    let out = OUT.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, |o| o.by_seq.len() as u64);
    C[OUT_MAX].fetch_max(out, Relaxed);
    if out >= 16 {
        C[OUT_FULL_FRAMES].fetch_add(1, Relaxed);
    }
    #[cfg(windows)]
    frame_add(crate::framelog::T::BatchesOutstanding, out);
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    let Ok(mut last) = LAST.try_lock() else { return };
    let now = Instant::now();
    let secs = now.duration_since(*last.get_or_insert(now)).as_secs_f64();
    if secs < 1.0 {
        return;
    }
    *last = Some(now);
    #[cfg(windows)]
    crate::faults::drivers_line(secs);
    let c: Vec<u64> = C.iter().map(|a| a.swap(0, Relaxed)).collect();
    SECONDS[1].fetch_add(1, Relaxed);
    if c[OUT_MAX] >= 16 {
        SECONDS[0].fetch_add(1, Relaxed);
    }
    let ms = |ns: u64| ns as f64 / 1e6;
    let oldest: Vec<u64> = OLDEST.iter().map(|a| a.swap(0, Relaxed)).collect();
    let extra: String = EXTRA.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|p| p()).collect::<String>() + &by_thread_line();
    if c[MAP_CALLS] + c[MAP_OTHER] + c[POLLS] + c[DONE] == 0 && c[OUT_MAX] == 0 {
        return;
    }
    tracing::info!(
        "frame thread streaming, over {secs:.1} s: mapping calls on the frame thread {} ({:.1} ms), on other threads {}; \
         not-complete status polls {} ({:.1} ms from the first to complete); enqueues blocked on a full queue {:.1} ms; \
         batches outstanding max {} (frames at 16+: {}; seconds at 16+ so far {} of {}); batches completed {} \
         (enqueue to complete mean {:.1} ms, max {:.1} ms), newer finished first {}; batches completed out of age order {},          oldest batch age when completed: mean {:.1} ms, max {:.1} ms (VICTORY_DSTORAGE_OLDEST_FIRST={}){extra}",
        c[MAP_CALLS],
        ms(c[MAP_NS]),
        c[MAP_OTHER],
        c[POLLS],
        ms(c[POLL_NS]),
        ms(c[BLOCKED_NS]),
        c[OUT_MAX],
        c[OUT_FULL_FRAMES],
        SECONDS[0].load(Relaxed),
        SECONDS[1].load(Relaxed),
        c[DONE],
        if c[DONE] == 0 { 0.0 } else { ms(c[DONE_NS]) / c[DONE] as f64 },
        ms(c[DONE_MAX_NS]),
        c[NEWER_FIRST],
        c[NEWER_FIRST],
        if oldest[0] == 0 { 0.0 } else { ms(oldest[1]) / oldest[0] as f64 },
        ms(oldest[2]),
        OLDEST_FIRST_MODE.load(Relaxed),
    );
}
