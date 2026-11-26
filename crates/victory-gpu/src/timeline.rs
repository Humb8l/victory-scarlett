//! (P3, recording cost) The title's timeline: what bounds a frame on the
//! title's side (`docs/critical-path-analysis.md`, section 5). Meter only;
//! on with the frame log (diagnostics builds; `VICTORY_TITLE_TIMELINE=0` or
//! `=1` says otherwise).
//!
//! Per frame (a frame is what the title submits between two presents):
//! - recording wall: from the first list call of the lists it submits to
//!   the last close of them (each list's first call since its reset and its
//!   close are noted on the list, and taken when the list is submitted);
//! - per recording thread, the time inside our list calls, the time between
//!   them under 0.5 ms (the title's own code between its calls), and the
//!   longer gaps (waiting for work, or the title's long jobs);
//! - the present call, from the title's call to its return;
//! - the fence gate: the title allows two frames in flight and waits, before
//!   it records frame N + 2, for the fence it signalled after Present(N)
//!   (the first Signal after a present is taken as that fence). Estimated as
//!   the time from the end of frame N + 1's recording to our setting of that
//!   fence, for the frames whose recording began after it was set.
//!
//! One line a second: "title timeline per frame over the last S s (N
//! frames): recording wall R ms (...)".

use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

/// Whether the timeline is kept.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var_os("VICTORY_TITLE_TIMELINE") {
        Some(v) => v != "0",
        None => crate::framelog::on(),
    })
}

/// Nanoseconds since the first ask (never 0).
pub(crate) fn now_ns() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64 + 1
}

/// A gap between two of our list calls on one thread under this is the
/// title's own code; longer ones are counted apart.
const OWN_GAP_NS: u64 = 500_000;

/// The recording threads followed, at most.
const THREADS: usize = 32;

struct Slot {
    id: AtomicU32,
    /// Inside our list calls, between them (short gaps), long gaps (ns); calls.
    ours: AtomicU64,
    own: AtomicU64,
    idle: AtomicU64,
    calls: AtomicU64,
}

static SLOTS: [Slot; THREADS] = [const {
    Slot { id: AtomicU32::new(0), ours: AtomicU64::new(0), own: AtomicU64::new(0), idle: AtomicU64::new(0), calls: AtomicU64::new(0) }
}; THREADS];
static CLAIMED: AtomicU32 = AtomicU32::new(0);

thread_local! {
    /// This thread's slot (1-based; 0 none yet, `u8::MAX` none to have), the
    /// depth of list calls it is in, and when its last one returned.
    static MINE: std::cell::Cell<(u8, u32, u64)> = const { std::cell::Cell::new((0, 0, 0)) };
}

fn slot() -> Option<&'static Slot> {
    let (mine, depth, last) = MINE.get();
    let at = match mine {
        0 => {
            let n = CLAIMED.fetch_add(1, Relaxed) as usize;
            let at = if n < THREADS { n as u8 + 1 } else { u8::MAX };
            if n < THREADS {
                // SAFETY: a plain query.
                SLOTS[n].id.store(unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }, Relaxed);
            }
            MINE.set((at, depth, last));
            at
        }
        at => at,
    };
    (at != u8::MAX).then(|| &SLOTS[usize::from(at) - 1])
}

/// One list call of the title's, timed while it lasts ([`Call::begin`]).
pub(crate) struct Call {
    /// When it began (0: not timed: off, or inside another call).
    start: u64,
}

impl Call {
    /// A list call begins; `first` is the list's first-call stamp (set now
    /// if the list has none since its reset).
    pub(crate) fn begin(first: &AtomicU64) -> Call {
        if !on() {
            return Call { start: 0 };
        }
        let (mine, depth, last) = MINE.get();
        if depth > 0 {
            MINE.set((mine, depth + 1, last));
            return Call { start: 0 };
        }
        let now = now_ns();
        if first.load(Relaxed) == 0 {
            let _ = first.compare_exchange(0, now, Relaxed, Relaxed);
        }
        if let Some(s) = slot()
            && last != 0
        {
            let gap = now.saturating_sub(last);
            if gap < OWN_GAP_NS { &s.own } else { &s.idle }.fetch_add(gap, Relaxed);
        }
        let (mine, _, last) = MINE.get();
        MINE.set((mine, 1, last));
        Call { start: now }
    }
}

impl Drop for Call {
    fn drop(&mut self) {
        let (mine, depth, last) = MINE.get();
        if self.start == 0 {
            if depth > 0 {
                MINE.set((mine, depth - 1, last));
            }
            return;
        }
        let end = now_ns();
        if let Some(s) = slot() {
            s.ours.fetch_add(end.saturating_sub(self.start), Relaxed);
            s.calls.fetch_add(1, Relaxed);
        }
        let (mine, _, _) = MINE.get();
        MINE.set((mine, 0, end));
    }
}

/// The recording of the lists submitted since the last present: the first
/// call, the last close, how many lists.
static FIRST: AtomicU64 = AtomicU64::new(u64::MAX);
static LAST: AtomicU64 = AtomicU64::new(0);
static LISTS: AtomicU64 = AtomicU64::new(0);

/// A title list is submitted: its first call and its close (0: not known).
pub(crate) fn submitted(first: u64, closed: u64) {
    if !on() || first == 0 {
        return;
    }
    FIRST.fetch_min(first, Relaxed);
    LAST.fetch_max(closed.max(first), Relaxed);
    LISTS.fetch_add(1, Relaxed);
}

/// One frame as the timeline keeps it.
#[derive(Clone, Copy, Default)]
struct Frame {
    /// Its recording: first call, last close (ns; 0 none).
    first: u64,
    last: u64,
    /// The fence the title signalled after its present, and when we set it.
    fence: Option<(usize, u64)>,
    fence_set: u64,
}

/// Sums over the frames of the current second (ns unless said otherwise).
#[derive(Default)]
struct Second {
    frames: u64,
    wall: u64,
    wall_max: u64,
    lists: u64,
    present: u64,
    gate: u64,
    fence_known: u64,
    after_fence: u64,
    before_fence: u64,
}

struct Timeline {
    presents: u64,
    /// The last four frames, newest last.
    recent: std::collections::VecDeque<Frame>,
    /// A present returned on this queue and its fence is not known yet.
    awaiting: Option<usize>,
    present_began: u64,
    since: Option<std::time::Instant>,
    second: Second,
    /// The slot totals at the second's start.
    slots_then: [(u64, u64, u64, u64); THREADS],
}

static TIMELINE: Mutex<Option<Timeline>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut Timeline) -> R) -> R {
    let mut g = TIMELINE.lock().unwrap_or_else(|e| e.into_inner());
    let t = g.get_or_insert_with(|| Timeline {
        presents: 0,
        recent: std::collections::VecDeque::new(),
        awaiting: None,
        present_began: 0,
        since: None,
        second: Second::default(),
        slots_then: [(0, 0, 0, 0); THREADS],
    });
    f(t)
}

/// The title calls Present.
pub(crate) fn present_begins() {
    if !on() {
        return;
    }
    let now = now_ns();
    with(|t| t.present_began = now);
}

/// The title's Present on `queue` returns.
pub(crate) fn present_ends(queue: usize) {
    if !on() {
        return;
    }
    let now = now_ns();
    let (first, last, lists) = (FIRST.swap(u64::MAX, Relaxed), LAST.swap(0, Relaxed), LISTS.swap(0, Relaxed));
    let line = with(|t| {
        t.presents += 1;
        let frame = Frame {
            first: if first == u64::MAX { 0 } else { first },
            last,
            fence: None,
            fence_set: 0,
        };
        let s = &mut t.second;
        s.frames += 1;
        s.lists += lists;
        s.present += now.saturating_sub(t.present_began);
        if frame.first != 0 {
            let wall = frame.last.saturating_sub(frame.first);
            s.wall += wall;
            s.wall_max = s.wall_max.max(wall);
        }
        // The gate of this frame's recording: the fence of the present two
        // before, against the end of the recording of the frame before.
        let n = t.recent.len();
        if n >= 2 && frame.first != 0 {
            let (gated, before) = (t.recent[n - 2], t.recent[n - 1]);
            if gated.fence.is_some() && gated.fence_set != 0 {
                s.fence_known += 1;
                if frame.first >= gated.fence_set {
                    s.after_fence += 1;
                    if before.last != 0 {
                        s.gate += gated.fence_set.saturating_sub(before.last);
                    }
                } else {
                    s.before_fence += 1;
                }
            }
        }
        if t.recent.len() == 4 {
            t.recent.pop_front();
        }
        t.recent.push_back(frame);
        t.awaiting = Some(queue);
        let since = *t.since.get_or_insert_with(std::time::Instant::now);
        if since.elapsed() < std::time::Duration::from_secs(1) {
            return None;
        }
        Some(line(t, since.elapsed().as_secs_f64()))
    });
    if let Some(line) = line {
        tracing::info!("{line}");
    }
}

/// The title signals `fence` to `value` on `queue`: the first such after a
/// present on that queue is the present's fence.
pub(crate) fn signalled(queue: usize, fence: usize, value: u64) {
    if !on() {
        return;
    }
    with(|t| {
        if t.awaiting == Some(queue) {
            t.awaiting = None;
            if let Some(f) = t.recent.back_mut() {
                f.fence = Some((fence, value));
            }
        }
    });
}

/// We set `fence` to `value`.
pub(crate) fn fence_set(fence: usize, value: u64) {
    if !on() {
        return;
    }
    let now = now_ns();
    with(|t| {
        for f in t.recent.iter_mut() {
            if f.fence_set == 0 && f.fence.is_some_and(|(at, wanted)| at == fence && wanted <= value) {
                f.fence_set = now;
            }
        }
    });
}

/// The second's line, and a new second begun.
fn line(t: &mut Timeline, seconds: f64) -> String {
    let s = std::mem::take(&mut t.second);
    let frames = s.frames.max(1) as f64;
    let ms = |ns: u64| ns as f64 / 1e6 / frames;
    // The recording threads: what each did this second.
    let mut threads: Vec<(u32, u64, u64, u64, u64)> = Vec::new();
    for (i, slot) in SLOTS.iter().enumerate() {
        let now = (slot.ours.load(Relaxed), slot.own.load(Relaxed), slot.idle.load(Relaxed), slot.calls.load(Relaxed));
        let then = std::mem::replace(&mut t.slots_then[i], now);
        let d = (now.0 - then.0, now.1 - then.1, now.2 - then.2, now.3 - then.3);
        if d.3 > 0 {
            threads.push((slot.id.load(Relaxed), d.0, d.1, d.2, d.3));
        }
    }
    threads.sort_by_key(|r| std::cmp::Reverse(r.1));
    let ours: u64 = threads.iter().map(|r| r.1).sum();
    let own: u64 = threads.iter().map(|r| r.2).sum();
    let idle: u64 = threads.iter().map(|r| r.3).sum();
    let by_thread: Vec<String> =
        threads.iter().take(8).map(|r| format!("{}: {:.1}/{:.1}/{:.1}", r.0, ms(r.1), ms(r.2), ms(r.3))).collect();
    t.since = Some(std::time::Instant::now());
    format!(
        "title timeline per frame over the last {seconds:.1} s ({} frames): recording wall R {:.1} ms (longest {:.1}; {:.0} lists) \
         (in our calls C {:.1} ms on {} threads, title's own T {:.1} ms (gaps under 0.5 ms between our calls), longer gaps {:.1} ms), \
         present call P {:.1} ms, fence gate G {:.1} ms (present fence known for {} frames: recording began after it was set {}, before {}) \
         | recording threads (thread: ours/own/longer gaps ms a frame): {} | {}",
        s.frames,
        ms(s.wall),
        s.wall_max as f64 / 1e6,
        s.lists as f64 / frames,
        ms(ours),
        threads.len(),
        ms(own),
        ms(idle),
        ms(s.present),
        if s.after_fence > 0 { s.gate as f64 / 1e6 / s.after_fence as f64 } else { 0.0 },
        s.fence_known,
        s.after_fence,
        s.before_fence,
        by_thread.join(", "),
        crate::draw::filter_text(frames),
    )
}
