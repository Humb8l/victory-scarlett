//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! Writes into physical pages through another view are seen too: the
//! runtime tells us the views of the pages it wrote itself ([`writing`],
//! xmem's `write_physical` and DirectStorage's page destinations), and a
//! write by the title through any view of a guarded page faults like one
//! at the guarded address (the guard makes every view of the page
//! read-only, [`open`] gives them all back; not a page of more than
//! [`MANY_VIEWS`] views). A
//! system call writing into a guarded block fails (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

/// A title-memory range whose changed pages another subsystem needs to read.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    seen: Mutex<BTreeMap<usize, u64>>,
}

/// Tracks a runtime GPU-memory range, including physical-page views when
/// `views` is true. Other process memory is not tracked.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    if len == 0 || writes(start, end, views).is_none() {
        return None;
    }
    Some(Tracked {
        start,
        end,
        seen: Mutex::default(),
    })
}

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Page ranges being asked about right now, without the journal's
    /// lock ([`unlocked`]): an asking or a journal answer of memory that
    /// overlaps one waits for it ([`ASKED`]), others go on.
    in_flight: Vec<(usize, usize)>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Vec<(usize, usize)>,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    in_flight: Vec::new(),
    ahead: Vec::new(),
    ahead_asked: 0,
    kept: false,
});

/// An asking ended ([`unlocked`]): those waiting for memory it held look again.
static ASKED: std::sync::Condvar = std::sync::Condvar::new();

type JournalHeld = std::sync::MutexGuard<'static, Journal>;

fn journal() -> JournalHeld {
    JOURNAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Waits, the journal's lock let go meanwhile, until no asking of memory
/// overlapping pages `[first, last)` is under way ([`unlocked`]): what
/// such an asking took from the kernel (and reset there) is in the journal
/// once this returns.
fn settled(mut j: JournalHeld, first: usize, last: usize) -> JournalHeld {
    while j.in_flight.iter().any(|&(a, b)| a < last && first < b) {
        j = ASKED.wait(j).unwrap_or_else(|e| e.into_inner());
    }
    j
}

/// Runs `ask` (a kernel asking, or the guards' pass, over pages `[first,
/// last)`) without the journal's lock, and takes it again for the caller,
/// who notes what was found before letting go (JoshuuDev watch-less
/// 3b270b1: the submitting thread spent 58% of its waiting on this lock,
/// held through every `GetWriteWatch`).
///
/// An asking resets what it reports, so the pages it holds are not in the
/// kernel and not yet in the journal while it runs: an asking of
/// overlapping memory, and a journal answer of it ([`writes`] from an
/// asking ahead, [`asked`]), wait for it to end ([`settled`]). Askings of
/// other memory go on side by side. Whoever notes what was found takes the
/// asking's number after the call, under the lock again, so it is higher
/// than any number given out meanwhile.
fn unlocked<T>(j: JournalHeld, first: usize, last: usize, ask: impl FnOnce() -> T) -> (JournalHeld, T) {
    let mut j = settled(j, first, last);
    j.in_flight.push((first, last));
    drop(j);
    let out = ask();
    let mut j = journal();
    if let Some(i) = j.in_flight.iter().position(|&r| r == (first, last)) {
        j.in_flight.swap_remove(i);
    }
    // The waiters take the lock after the caller has noted what was found.
    ASKED.notify_all();
    (j, out)
}

/// Whether `[first, last)` (pages) holds physical-page view memory, which
/// is not write-watched: `GetWriteWatch` refuses any range holding some,
/// after walking it (JoshuuDev 3b270b1, 40f9ce3). Views are told to us as
/// they are mapped and unmapped (`crate::inplace`).
fn has_views(first: usize, last: usize) -> bool {
    crate::inplace::any_view_between(first * PAGE, last * PAGE)
}

/// The kernel's written pages (64 KiB page numbers, each once) of pages
/// `[first, last)`, reset as it reports them; `None` if it refuses the
/// range (not write-watched memory of one reservation) or the range holds
/// views ([`has_views`]). Takes no lock.
fn kernel_written(first: usize, last: usize) -> Option<Vec<usize>> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    thread_local! {
        /// Room for the kernel's answer, per thread (askings run side by side).
        static FOUND: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    if has_views(first, last) {
        return None;
    }
    FOUND.with(|found| {
        let mut found = found.borrow_mut();
        let mut count = (last - first) * PAGE / KERNEL_PAGE;
        if found.len() < count {
            found.resize(count, 0);
        }
        let mut granularity = 0;
        let _faults = crate::faults::scope(crate::faults::Kind::Watch);
        // SAFETY: the answer buffer holds `count` entries.
        let failed = unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
        if failed {
            return None;
        }
        crate::faults::count(crate::faults::Driver::WatchResets, 1);
        crate::faults::count(crate::faults::Driver::WatchPages, count as u64);
        let mut pages: Vec<usize> = found[..count].iter().map(|&at| at / PAGE).collect();
        pages.dedup();
        Some(pages)
    })
}

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.0 < r.1)
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    let mut j = journal();
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`.
    if j.kept {
        return;
    }
    // View memory the guards hold is left to `writes`, which answers it
    // from the guards ([`known_view`]): the kernel's write watch can only
    // fail there. Asked here with the others, a run holding such a range
    // failed, was asked again by halves down to single ranges, failing
    // each time, and the range then went to `writes` all the same (after
    // triippywhale d0581a55). `VICTORY_VIEW_ASK=1` asks as before.
    #[cfg(feature = "diagnostics")]
    let all = pieces.len();
    pieces.retain(|&(first, _)| !known_view(first * PAGE));
    #[cfg(feature = "diagnostics")]
    AHEAD_VIEWS.fetch_add((all - pieces.len()) as u64, std::sync::atomic::Ordering::Relaxed);
    j.asked += 1;
    j.ahead_asked = j.asked;
    j.ahead.clear();
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + AHEAD_GAP
            && last.max(pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        j = ask_run(j, &pieces[i..k]);
        i = k;
    }
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// Since the last frame line: ranges of view memory the guards hold that a
/// submission's askings ahead ([`ask_ahead`], [`ask_once`]) left out. Each
/// was a kernel asking that failed, or several, before.
#[cfg(feature = "diagnostics")]
pub(crate) static AHEAD_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    SUBMISSIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if !once() {
        return;
    }
    ask_once_end();
    if quiet_frames() > 0 {
        ask_once_quiet(ranges);
        return;
    }
    ask_ahead(ranges);
    let mut j = journal();
    if j.ahead.is_empty() {
        // `ask_ahead` asked nothing (fewer than two ranges).
        j.asked += 1;
        j.ahead_asked = j.asked;
    }
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its 70 flushes (run 76: 2,300
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    for &(start, end) in ranges {
        if start >= end {
            continue;
        }
        // View memory the guards hold is not asked about: the asking can
        // only fail there, one failed kernel call for each such copy at
        // every submission, and `writes` answers it from the guards
        // (after triippywhale cc368c7c).
        if known_view(start) {
            #[cfg(feature = "diagnostics")]
            AHEAD_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
            continue;
        }
        let asked;
        (j, asked) = ask_kernel(j, first, last);
        if asked {
            j.ahead.push((first, last));
        }
    }
    j.kept = !j.ahead.is_empty();
}

/// (Neptune) `VICTORY_ASK_QUIET=<frames>` (0 or unset: off): memory nothing
/// was found written in for that many frames is asked about once a frame,
/// not once a submission.
///
/// Neptune: the kernel walks every page of a range it is asked about, and
/// each of a frame's thirty submissions asked about the ranges its lists
/// refresh, mostly the same ones: some 3,000 askings over 11 GB, a third
/// of the submitting thread's time. Title memory is taken in blocks of
/// 1 MiB here, and a block is asked about whole. One in which a written
/// page was found within the last `frames` frames is asked about at every
/// submission that refreshes part of it, as before; any other at the first
/// such submission of a frame, and the journal answers for it until the
/// next frame.
///
/// What changes: a write into a quiet block that arrives after the first
/// asking of the frame is seen by the next frame, so a list later in the
/// same frame draws from the bytes of before, once (suspected by zm0kie of
/// darkening objects: off by default, Neptune's launcher sets 8). From
/// then on the block is asked about at every submission.
fn quiet_frames() -> u64 {
    static FRAMES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *FRAMES.get_or_init(|| std::env::var("VICTORY_ASK_QUIET").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// Blocks that may lie between two blocks of one asking
/// (`VICTORY_ASK_QUIET_GAP_MB`, 32: a clean page walked is 4 nanoseconds,
/// a call that resets pages some 50 microseconds).
fn quiet_gap() -> usize {
    static GAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *GAP.get_or_init(|| std::env::var("VICTORY_ASK_QUIET_GAP_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(32))
}

/// Journal pages to a block of [`quiet_frames`].
const QUIET_BLOCK: usize = (1 << 20) / PAGE;

/// Of a block: the frame it was last asked about whole (`u64::MAX`:
/// never), the last frame a written page was found in it (`u64::MAX`:
/// never), and whether asking about it whole fails (it is not all inside
/// one write-watched reservation).
#[derive(Clone, Copy)]
struct Quiet {
    asked: u64,
    written: u64,
    partial: bool,
}

const NEVER: Quiet = Quiet { asked: u64::MAX, written: u64::MAX, partial: false };

struct QuietBlocks {
    blocks: std::collections::HashMap<usize, Quiet>,
    /// Since the last line in the log: frames, blocks asked about because
    /// they were written lately, blocks asked about as the first of a
    /// frame, ranges answered by the journal alone, ranges asked about on
    /// their own.
    told: [u64; 5],
    frame: u64,
}

static BLOCKS: Mutex<Option<QuietBlocks>> = Mutex::new(None);

/// [`ask_once`] with [`quiet_frames`] (Neptune's `ask_once_quiet`, with
/// our asking outside the journal's lock: [`ask_kernel_pages`]).
fn ask_once_quiet(ranges: &[(usize, usize)]) {
    let frame = crate::queue::presented();
    let keep = quiet_frames();
    let mut held = BLOCKS.lock().unwrap_or_else(|e| e.into_inner());
    let state = held.get_or_insert_with(|| QuietBlocks { blocks: Default::default(), told: [0; 5], frame });
    let mut j = journal();
    j.asked += 1;
    j.ahead_asked = j.asked;
    j.ahead.clear();
    if state.frame != frame {
        state.told[0] += frame.wrapping_sub(state.frame).min(1 << 20);
        state.frame = frame;
        if state.told[0] >= 512 {
            let n = state.told[0] as f64;
            tracing::info!(
                "(Neptune) title memory asked about per frame (VICTORY_ASK_QUIET={keep}): {:.0} blocks of 1 MiB written lately, {:.0} quiet ones as the first of a frame, {:.0} ranges answered from the journal, {:.0} ranges on their own; {} blocks known",
                state.told[1] as f64 / n,
                state.told[2] as f64 / n,
                state.told[3] as f64 / n,
                state.told[4] as f64 / n,
                state.blocks.len()
            );
            state.told = [0; 5];
        }
    }
    // The blocks to ask about now. View memory the guards hold is left to
    // `writes`, as in `ask_once`.
    let mut wanted: Vec<usize> = Vec::new();
    for &(start, end) in ranges {
        if start >= end || known_view(start) {
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        for block in first / QUIET_BLOCK..=(last - 1) / QUIET_BLOCK {
            let q = *state.blocks.entry(block).or_insert(NEVER);
            if q.partial {
                continue;
            }
            let lately = q.written != u64::MAX && frame.wrapping_sub(q.written) <= keep;
            if q.asked != frame || lately {
                wanted.push(block);
            }
        }
    }
    wanted.sort_unstable();
    wanted.dedup();
    let mut i = 0;
    while i < wanted.len() {
        let mut k = i + 1;
        // Blocks a few apart are one asking: the kernel walks the pages
        // between them in less time than another call takes (and they
        // count as asked about).
        while k < wanted.len() && wanted[k] <= wanted[k - 1] + 1 + quiet_gap() && wanted[k] - wanted[i] < 4096 {
            k += 1;
        }
        let (first, last) = (wanted[i], wanted[k - 1] + 1);
        let whole;
        (j, whole) = ask_blocks(j, state, first, last, frame, keep);
        if !whole && k - i > 1 {
            // One of them is not all write-watched memory: each on its own.
            for idx in i..k {
                let block = wanted[idx];
                let one;
                (j, one) = ask_blocks(j, state, block, block + 1, frame, keep);
                if !one && let Some(q) = state.blocks.get_mut(&block) {
                    q.partial = true;
                }
            }
        } else if !whole && let Some(q) = state.blocks.get_mut(&first) {
            q.partial = true;
        }
        i = k;
    }
    // What the journal now answers for: a range whose blocks were all
    // asked about this frame. Any other is asked about on its own, as
    // without the switch.
    for &(start, end) in ranges {
        if start >= end || known_view(start) {
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        let covered = (first / QUIET_BLOCK..=(last - 1) / QUIET_BLOCK)
            .all(|block| state.blocks.get(&block).is_some_and(|q| !q.partial && q.asked == frame));
        if covered {
            state.told[3] += 1;
            j.ahead.push((first, last));
        } else if !j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
            let asked;
            (j, asked) = ask_kernel(j, first, last);
            if asked {
                state.told[4] += 1;
                j.ahead.push((first, last));
            }
        }
    }
    j.kept = !j.ahead.is_empty();
}

/// Asks about blocks `[first, last)` whole and notes the answer in them;
/// false if the kernel does not answer for all of it.
fn ask_blocks(j: JournalHeld, state: &mut QuietBlocks, first: usize, last: usize, frame: u64, keep: u64) -> (JournalHeld, bool) {
    let (j, found) = ask_kernel_pages(j, first * QUIET_BLOCK, last * QUIET_BLOCK);
    let Some(pages) = found else {
        return (j, false);
    };
    for block in first..last {
        let q = state.blocks.entry(block).or_insert(NEVER);
        let lately = q.written != u64::MAX && frame.wrapping_sub(q.written) <= keep;
        state.told[if q.asked == frame || lately { 1 } else { 2 }] += 1;
        q.asked = frame;
    }
    for page in pages {
        if let Some(q) = state.blocks.get_mut(&(page / QUIET_BLOCK)) {
            q.written = frame;
        }
    }
    (j, true)
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept || len == 0 {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    // The asking that stands moves with it: `asked` and `writes` give it as
    // "now", and a page noted written after "now" stays stale for every
    // copy that takes it (copied at `now`, written at `now + 1`): copied
    // again at every refresh, and a reason to settle flights
    // (fxceless/victory-fork#410).
    j.ahead_asked = now;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        j.written.insert(page, now);
    }
}

/// We wrote the 64 KiB page numbered `page` of title memory through a
/// view of our own (`crate::inplace::write_own`): no guard was opened and
/// no kernel write-watch bit set, so the journal is told here. The page,
/// and the same physical page at its other title addresses (`others`,
/// page numbers), count as written at a new asking, which every host copy
/// that holds them takes at its next refresh. Returns the asking that had
/// last found `page` written (0: never) and the new one, for the copy
/// that wrote to note itself current.
pub(crate) fn wrote_own(page: usize, others: &[usize]) -> (u64, u64) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    j.asked += 1;
    let now = j.asked;
    // As in [`wrote`]: an asking that stands for a submission, or for a
    // flush, moves with it, or a copy taking the page at the standing
    // asking would find it written "later" at every refresh.
    if j.kept || !j.ahead.is_empty() {
        j.ahead_asked = now;
    }
    let before = j.written.insert(page, now).unwrap_or(0);
    for &other in others {
        if other != page {
            j.written.insert(other, now);
        }
    }
    (before, now)
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`].
fn ask_run(j: JournalHeld, pieces: &[(usize, usize)]) -> JournalHeld {
    if pieces.len() < 2 {
        return j;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    let (mut j, asked) = ask_kernel(j, first, last);
    if asked {
        j.ahead.push((first, last));
        return j;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    let j = ask_run(j, a);
    ask_run(j, b)
}

/// Asks the kernel for the written pages of pages `[first, last)` and notes
/// them in the journal under the ahead asking; false (and nothing changed)
/// if the range is not write-watched memory of one reservation. The
/// journal's lock is let go during the kernel call ([`unlocked`]).
///
/// The ahead asking moves on when it notes what it found (as [`wrote`]
/// moves it): a copy that asked by itself while this call was under way
/// holds a higher number than the ahead asking had, and pages this call
/// took from the kernel (written before the call, after that copy's own
/// asking) must count as written after it, or that copy would take them for
/// copied. Pages noted under the earlier number keep it: lower than the
/// number now standing, as before.
fn ask_kernel(j: JournalHeld, first: usize, last: usize) -> (JournalHeld, bool) {
    let (j, found) = ask_kernel_pages(j, first, last);
    (j, found.is_some())
}

/// [`ask_kernel`], with the pages found written (`None`: not asked).
fn ask_kernel_pages(j: JournalHeld, first: usize, last: usize) -> (JournalHeld, Option<Vec<usize>>) {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let (mut j, found) = unlocked(j, first, last, || kernel_written(first, last));
    let Some(pages) = found else {
        return (j, None);
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    if !pages.is_empty() {
        j.asked += 1;
        j.ahead_asked = j.asked;
    }
    let now = j.ahead_asked;
    for &page in &pages {
        j.written.insert(page, now);
    }
    (j, Some(pages))
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // What an asking of overlapping memory under way took is not in the
    // journal yet ([`unlocked`]).
    let j = settled(journal(), first, last);
    j.ahead
        .iter()
        .any(|&(a, b)| a <= first && last <= b)
        .then(|| (j.ahead_asked, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // An asking of overlapping memory under way holds pages the kernel no
    // longer has and the journal not yet: waited for ([`unlocked`]).
    let j = settled(journal(), first, last);
    // Asked ahead with the submission's other ranges: the journal has it.
    if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
        return Some((
            j.ahead_asked,
            j.written
                .range(first..last)
                .map(|(&p, &w)| (p, w))
                .collect(),
        ));
    }
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // The kernel, else (view memory) the guards; without the journal's
    // lock ([`unlocked`], JoshuuDev watch-less 3b270b1). `None`: not
    // write-watched and not guarded.
    let (mut j, found) = unlocked(j, first, last, || {
        // View memory the guards already hold cannot be write-watched: the
        // kernel asking would only fail. Copies of it are flushed many times
        // a frame, each time paying that failed call before the guards answer.
        let known_view = guarded && known_view(first * PAGE);
        let pages = if known_view { None } else { kernel_written(first, last) };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(
            if known_view {
                crate::copystat::C::WatchKnownView
            } else {
                crate::copystat::C::WatchFailed
            },
            pages.is_none() as u64,
        );
        if pages.is_some() || !guarded {
            return pages;
        }
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard(first * PAGE, last * PAGE);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        written
    });
    let pages = found?;
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    // The number is taken now, under the lock again: higher than any given
    // out while the asking was under way.
    j.asked += 1;
    let now = j.asked;
    for page in pages {
        j.written.insert(page, now);
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

impl Tracked {
    /// Takes changed pages of `[from, to)`, cut to the tracked range. A
    /// partial page stays changed so a later take can read its other bytes.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        let Some((_, pages)) = writes(from, to, true) else {
            return Vec::new();
        };
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (page, write) in pages {
            if seen.get(&page).is_some_and(|&last| last >= write) {
                continue;
            }
            let (start, end) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
            if start >= end {
                continue;
            }
            // Keep partial pages dirty: the caller has not taken the rest.
            if (page * PAGE).max(self.start) >= from
                && ((page + 1) * PAGE).min(self.end) <= to
            {
                seen.insert(page, write);
            }
            match ranges.last_mut() {
                Some((_, last)) if *last == start => *last = end,
                _ => ranges.push((start, end)),
            }
        }
        ranges
    }
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// Whether the block holding `start` is view memory the guards already
/// hold (read-only by our hand, or noted absent): asking the kernel's write
/// watch about it can only fail. `VICTORY_VIEW_ASK=1` (or
/// `VICTORY_GUARD_ASK=1`) asks the kernel anyway, as before.
fn known_view(start: usize) -> bool {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let ask = *ASK.get_or_init(|| {
        ["VICTORY_VIEW_ASK", "VICTORY_GUARD_ASK"]
            .iter()
            .any(|n| std::env::var_os(n).is_some_and(|v| v == "1"))
    });
    if ask {
        return false;
    }
    let g = guards_read();
    let first = start / BLOCK;
    g.protected.contains(&first)
        || (g.absent.contains(&first) && !absent_private())
        || (view_known() && g.viewed.contains(&first))
        || (view_set() && covered(&g.views, first, first + 1))
}

/// Whether a block noted absent is asked about again before it is taken
/// for view memory (myokex c3723f3): the title may commit private memory
/// there later (`MEM_WRITE_WATCH` or ordinary), which must not be made a
/// read-only guard. Such a block is never guarded, stays noted, and counts
/// as written at every asking (nothing watches it).
/// `VICTORY_ABSENT_PRIVATE=0`: an absent block is a view, as before.
fn absent_private() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_ABSENT_PRIVATE").is_none_or(|v| v != "0");
        tracing::info!(
            "write watch: blocks noted absent that turn into private memory {} (VICTORY_ABSENT_PRIVATE={})",
            if on { "are never guarded and count as written" } else { "are guarded as views, as before" },
            if on { "1, the default" } else { "0" }
        );
        on
    })
}

/// With [`absent_private`]: whether `block`, noted absent and committed
/// now, is memory that is no view.
fn turned_private(block: usize) -> bool {
    absent_private() && !is_view(block * BLOCK, (block + 1) * BLOCK)
}

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: Blocks,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: BTreeSet<usize>,
    /// Blocks whose physical page was guarded together with its other views
    /// (those views' blocks are in it too): a write to one opens them all.
    /// A write range can be huge (`fread`), so only these are looked up.
    aliased: BTreeSet<usize>,
    /// With [`view_known`]: blocks made read-only at least once since the
    /// runtime last mapped or unmapped them. They are view memory whether
    /// their guard stands now or a write has opened it.
    viewed: BTreeSet<usize>,
    /// Blocks known to be view memory ([`guard`]), in order and joined:
    /// a range in them needs no question to Windows to tell.
    views: Vec<(usize, usize)>,
    /// The blocks of `views` that are not clean (written since a copy took
    /// them, never taken, or not guardable), but for the parked ones: an
    /// asking whose range holds none of them is answered in one lookup,
    /// and otherwise walks only them ([`guard_unclean`]).
    unclean: BTreeSet<usize>,
    /// View blocks found uncommitted ([`Guards::absent`]), taken out of
    /// `unclean` while they stay so ([`park`]). One goes back when the
    /// runtime maps it ([`title_memory_remapped`]), when the title commits
    /// it through the runtime ([`title_memory_committed`]), or, if it is no
    /// placeholder, when the look at those every [`park_epoch`] submissions
    /// finds it committed (an ordinary commit no one tells of). The walk
    /// over every block of a range asked Windows about each at every
    /// asking: in the first room ~86,000-96,000 such blocks, nearly all
    /// 64 KiB placeholders, each its own region, and ~88,000 questions a
    /// frame (guards 26 ms of the submitting thread's frame on an RTX 5080;
    /// 14 ms parked).
    parked: BTreeSet<usize>,
    /// The parked blocks that are no placeholder: the ones looked at again.
    parked_ordinary: BTreeSet<usize>,
    /// The epoch ([`SUBMISSIONS`] / [`park_epoch`]) of the last look at
    /// `parked_ordinary`.
    parked_at: u64,
    /// (`VICTORY_GUARD_REST`, avatarmanz 7bc7c07) Blocks a write fault
    /// opened, and when: left open by the askings of the next [`rest`].
    opened: BTreeMap<usize, std::time::Instant>,
    /// While a fault opens blocks: every block [`open_here`] opens, the
    /// other views of the page too (they rest with it).
    opening: Option<Vec<usize>>,
}

impl Guards {
    /// Whether `block` is left open for now: the title has just written it
    /// ([`rest`]).
    fn rests(&self, block: usize, rest: std::time::Duration) -> bool {
        !rest.is_zero() && self.opened.get(&block).is_some_and(|at| at.elapsed() < rest)
    }

    /// A write fault opened `blocks`.
    fn opened(&mut self, blocks: &[usize]) {
        const MANY: usize = 1 << 13;
        if self.opened.len() >= MANY {
            self.opened.retain(|_, at| at.elapsed().as_millis() < 50);
            if self.opened.len() >= MANY {
                self.opened.clear();
            }
        }
        let now = std::time::Instant::now();
        for &block in blocks {
            self.opened.insert(block, now);
        }
    }
}

/// How long a block the title has just opened by a write fault is left
/// open by the askings that take it (`VICTORY_GUARD_REST=<microseconds>`,
/// 3,000 unless given; `0`: not at all, as before; avatarmanz 7bc7c07, also
/// Neptune 25fbf69). Their fault book: 586 faults a frame, 245 on a block
/// that had faulted in the same frame, 220 within 3 ms of the refresh that
/// made it read-only; 69 -> ~6 ms a frame of the title's threads.
///
/// No write is missed: a resting block is open and not clean (the fault
/// took it out of `protected` and `clean`, and put it in `unclean`), and
/// the askings neither guard it nor note it clean while it rests, so every
/// asking over it reports its pages written and they are copied at each
/// submission. The other views of its page that the fault opened rest with
/// it (a write through it is a write of their bytes). Once its rest is over
/// the next asking reports it written once more and guards it, as an open
/// block always was: writes after that fault again. A guard put on early
/// (by another view's `cover`, an alias) only costs a fault.
fn rest() -> std::time::Duration {
    // (Tests take a block as guarded again right after a write.)
    if cfg!(test) {
        return std::time::Duration::ZERO;
    }
    static US: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let us = *US.get_or_init(|| {
        std::env::var("VICTORY_GUARD_REST").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(3000)
    });
    std::time::Duration::from_micros(us)
}

/// [`rest`] in microseconds, for the log.
pub(crate) fn guard_rest_us() -> u128 {
    rest().as_micros()
}

/// Since the start: blocks an asking left open because they rest
/// ([`rest`]), and faults that started a rest.
pub(crate) static RESTED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether a block once guarded stays known as view memory while its
/// guard is open (`VICTORY_VIEW_KNOWN=1`; off by default).
fn view_known() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_KNOWN").is_some_and(|v| v == "1"))
}

// (Neptune) A read-write lock: the checks that change nothing (is it a
// view, is all of a range clean) share it; Neptune measured the guards' lock
// at a sixth and an eighth of the submitting thread's time.
static GUARDS: std::sync::RwLock<Guards> = std::sync::RwLock::new(Guards {
    protected: BTreeSet::new(),
    clean: Blocks::new(),
    absent: BTreeSet::new(),
    aliased: BTreeSet::new(),
    viewed: BTreeSet::new(),
    views: Vec::new(),
    unclean: BTreeSet::new(),
    parked: BTreeSet::new(),
    parked_ordinary: BTreeSet::new(),
    parked_at: 0,
    opened: BTreeMap::new(),
    opening: None,
});

fn guards() -> std::sync::RwLockWriteGuard<'static, Guards> {
    GUARDS.write().unwrap_or_else(|e| e.into_inner())
}

/// (Neptune) Since the last view line: ranges of view memory answered as
/// clean under the guards' shared lock (`guard`).
pub(crate) static GUARD_SHARED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// (Neptune) The guards' tables shared, for checks that change nothing.
fn guards_read() -> std::sync::RwLockReadGuard<'static, Guards> {
    GUARDS.read().unwrap_or_else(|e| e.into_inner())
}

/// `[first, last)` added to `ranges` (in order, joined). For ranges seen
/// rarely (the views, once per new range): it sorts the whole list.
fn add_range(ranges: &mut Vec<(usize, usize)>, first: usize, last: usize) {
    if first >= last {
        return;
    }
    ranges.push((first, last));
    ranges.sort_unstable();
    let mut joined: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(a, b) in ranges.iter() {
        match joined.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => joined.push((a, b)),
        }
    }
    *ranges = joined;
}

/// Whether blocks `[first, last)` all lie in `ranges` (in order, joined).
fn covered(ranges: &[(usize, usize)], first: usize, last: usize) -> bool {
    let at = ranges.partition_point(|r| r.1 <= first);
    ranges.get(at).is_some_and(|r| r.0 <= first && last <= r.1)
}

/// Whether to keep views' not-clean blocks in a set and answer from it
/// ([`Guards::unclean`], [`Guards::parked`]; `VICTORY_VIEW_SET=0` walks
/// every block of a range, as before). Only where Windows tells
/// placeholders apart ([`placeholders_told`]).
fn view_set() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_SET").is_none_or(|v| v != "0") && placeholders_told())
}

/// Whether [`in_placeholder`] knows a placeholder, asked once of one of
/// our own. Where it does not (Wine may not fill the flag), every parked
/// block would be looked at each epoch, more questions than the walk
/// asks: the walk stays there.
fn placeholders_told() -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_RELEASE, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, VirtualAlloc2, VirtualFree,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: a fresh placeholder of one block, released at once.
    let p = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            std::ptr::null(),
            BLOCK,
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS,
            std::ptr::null_mut(),
            0,
        )
    } as usize;
    if p == 0 {
        return false;
    }
    let told = in_placeholder(p / BLOCK);
    // SAFETY: our own placeholder, known to nothing else.
    unsafe { VirtualFree(p as _, 0, MEM_RELEASE) };
    if !told {
        tracing::info!("Windows does not tell placeholders apart here: view memory askings walk every block (VICTORY_VIEW_SET)");
    }
    told
}

/// Since the last frame line ([`view_stats`]): view askings answered in one
/// lookup, view askings that walked their not-clean blocks, the blocks they
/// walked, the questions to Windows of whether a range is a view, the
/// questions whether blocks found uncommitted still are, and the parked
/// blocks found committed ([`Guards::parked`]).
static VIEW_STATS: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// [`VIEW_STATS`], taken.
pub(crate) fn view_stats() -> [u64; 6] {
    [0, 1, 2, 3, 4, 5].map(|i| VIEW_STATS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// Submissions' askings begun ([`ask_once`]).
static SUBMISSIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Submissions' askings between looks at the parked blocks that are no
/// placeholder (`VICTORY_PARK_EPOCH`, default 32; 1 looks at every one).
fn park_epoch() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_PARK_EPOCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(32)
    })
}

/// The view blocks parked as uncommitted now ([`Guards::parked`]), and
/// those of them that are no placeholder.
pub(crate) fn parked_blocks() -> (usize, usize) {
    let g = guards_read();
    (g.parked.len(), g.parked_ordinary.len())
}

/// Parks `block`, found uncommitted and not clean ([`Guards::parked`]). A
/// block in a placeholder is not looked at again until the runtime maps
/// it: a plain commit into a placeholder fails, only a call that replaces
/// one fills it, and on title memory the runtime's are the only such calls
/// and tell of themselves ([`title_memory_remapped`]).
fn park(g: &mut Guards, block: usize) {
    g.unclean.remove(&block);
    g.parked.insert(block);
    if !in_placeholder(block) {
        g.parked_ordinary.insert(block);
    }
}

/// Whether `block` lies in a placeholder.
fn in_placeholder(block: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MemoryRegionInfo, QueryVirtualMemoryInformation, WIN32_MEMORY_REGION_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // The region's flag for a placeholder reservation, after Private,
    // MappedDataFile, MappedImage, MappedPageFile, MappedPhysical,
    // DirectMapped, SoftwareEnclave and PageSize64K: the SDK's header calls
    // these bits reserved, the kernel fills them (Windows 10 1803 on). A
    // fresh, split or unmapped-to placeholder shows it, an ordinary
    // reservation (even one with no access) does not; `VirtualQuery` tells
    // the two apart by nothing.
    const PLACEHOLDER: u32 = 1 << 8;
    // SAFETY: plain data filled in by the query.
    let mut info: WIN32_MEMORY_REGION_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size; the returned size is
    // optional. Free memory fails the query: no placeholder.
    let ok = unsafe {
        QueryVirtualMemoryInformation(
            GetCurrentProcess(),
            (block * BLOCK) as *const _,
            MemoryRegionInfo,
            (&raw mut info).cast(),
            size_of_val(&info),
            std::ptr::null_mut(),
        )
    } != 0;
    // SAFETY: the flags word of the union, filled in by the query.
    ok && unsafe { info.Anonymous.Flags } & PLACEHOLDER != 0
}

/// Once per [`park_epoch`] submissions: the parked blocks that are no
/// placeholder and were committed since go back to the unclean set, no
/// longer absent, for the next asking that covers them to find written.
/// One question to Windows per uncommitted region.
fn unpark_committed(g: &mut Guards) {
    use std::sync::atomic::Ordering::Relaxed;
    let epoch = SUBMISSIONS.load(Relaxed) / park_epoch();
    if g.parked_at == epoch {
        return;
    }
    g.parked_at = epoch;
    let parked: Vec<usize> = g.parked_ordinary.iter().copied().collect();
    let mut until = 0;
    for block in parked {
        if block >= until {
            VIEW_STATS[4].fetch_add(1, Relaxed);
            // The block itself if it is (partly) committed now.
            until = uncommitted_end(block);
        }
        if block >= until {
            VIEW_STATS[5].fetch_add(1, Relaxed);
            g.parked_ordinary.remove(&block);
            g.parked.remove(&block);
            // (Left absent if private memory now, for `guard_unclean`.)
            if !turned_private(block) {
                g.absent.remove(&block);
            }
            g.unclean.insert(block);
        }
    }
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // With the view set, a range known to be views is one too, whatever
    // its first block's state: a written first block (open, not absent)
    // asked Windows again, the walk to the region's end.
    let known = !*ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1")) && {
        let g = guards_read();
        let known = g.protected.contains(&first)
            || (g.absent.contains(&first) && !absent_private())
            || (view_known() && g.viewed.contains(&first))
            || (view_set() && covered(&g.views, first, last));
        // (Neptune) A range nothing was written in since it was last taken
        // is answered under the shared lock, as the locked way below would
        // answer it: with the view set, the range among the views, no
        // unclean block in it and the parked blocks looked at in this epoch
        // (`guard_unclean`); else all of it clean.
        if known {
            let clean = if view_set() {
                let epoch = SUBMISSIONS.load(std::sync::atomic::Ordering::Relaxed) / park_epoch();
                g.parked_at == epoch && covered(&g.views, first, last) && g.unclean.range(first..last).next().is_none()
            } else {
                g.clean.holds(first..last)
            };
            if clean {
                if view_set() {
                    VIEW_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                GUARD_SHARED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some(Vec::new());
            }
        }
        known
    };
    if !known {
        VIEW_STATS[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if !is_view(start, end) {
            return None;
        }
    }
    install();
    let rest = rest();
    let mut g = guards();
    if view_set() {
        return Some(guard_unclean(&mut g, first, last));
    }
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    // The range's clean blocks, in order: read off the set in one pass,
    // not looked up one block at a time. A frame asks about some 10 GB of
    // view memory, 150,000 blocks, nearly all clean: the lookups were most
    // of the 60-90 ms a frame of "guards" (run 54). What the loop adds to
    // the set meanwhile is the block it stands at or ones before it.
    // Whether all of it is clean is answered a word of 64 blocks at a time
    // (soldsawn f48ce4b), before any list is made.
    if g.clean.holds(first..last) {
        // Nothing written, mapped or new since the range was last taken.
        return Some(written);
    }
    let clean: Vec<usize> = g.clean.range(first..last).collect();
    let mut next_clean = 0;
    let mut uncommitted_until = first;
    for block in first..=last {
        while next_clean < clean.len() && clean[next_clean] < block {
            next_clean += 1;
        }
        let is_clean = next_clean < clean.len() && clean[next_clean] == block;
        let mut open = block < last && !is_clean;
        if open && g.absent.contains(&block) {
            // Ordinary commits need no runtime mapping notification. Reuse
            // this asking's query for the following blocks in the same region.
            if block >= uncommitted_until {
                uncommitted_until = hole_end(&mut g, block);
            }
            open = block >= uncommitted_until;
            if open && turned_private(block) {
                // Private memory now (`VICTORY_ABSENT_PRIVATE`): not
                // guarded, still noted, written.
                written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
                open = false;
            } else if open {
                g.absent.remove(&block);
            }
        }
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else if g.rests(block, rest) {
                // Just written by the title: left open, not clean
                // (`VICTORY_GUARD_REST`).
                RESTED[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// [`guard`] with [`view_set`]: the same answer, from the view blocks that
/// are not clean ([`Guards::unclean`]) instead of every block of the range.
/// A range seen for the first time is noted as views, its blocks not clean
/// yet put in the set (once). Then: none in the range, nothing to do; else
/// each, in order, as the walk over the whole range treats it (one found
/// uncommitted parked; one written through the runtime, or guarded with
/// another view of its page, but still read-only noted clean; the others
/// guarded again in runs).
fn guard_unclean(g: &mut Guards, first: usize, last: usize) -> Vec<usize> {
    use std::sync::atomic::Ordering::Relaxed;
    unpark_committed(g);
    if !covered(&g.views, first, last) {
        let fresh: Vec<usize> = (first..last)
            .filter(|b| !g.clean.contains(*b) && !covered(&g.views, *b, *b + 1))
            .collect();
        g.unclean.extend(fresh);
        add_range(&mut g.views, first, last);
    }
    let unclean: Vec<usize> = g.unclean.range(first..last).copied().collect();
    let mut written = Vec::new();
    if unclean.is_empty() {
        VIEW_STATS[0].fetch_add(1, Relaxed);
        return written;
    }
    VIEW_STATS[1].fetch_add(1, Relaxed);
    VIEW_STATS[2].fetch_add(unclean.len() as u64, Relaxed);
    let rest = rest();
    let mut run: Option<(usize, usize)> = None;
    let flush = |g: &mut Guards, run: &mut Option<(usize, usize)>| {
        if let Some((a, b)) = run.take() {
            protect(g, a, b);
        }
    };
    for block in unclean {
        if g.absent.contains(&block) {
            // Found uncommitted before and still in the set (`protect`
            // parks the blocks it finds so): parked while it stays so.
            VIEW_STATS[4].fetch_add(1, Relaxed);
            if block < hole_end(g, block) {
                park(g, block);
                flush(g, &mut run);
                continue;
            }
            if turned_private(block) {
                // Private memory now (`VICTORY_ABSENT_PRIVATE`): not
                // guarded, still noted, written.
                written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
                flush(g, &mut run);
                continue;
            }
            g.absent.remove(&block);
        }
        written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
        if g.protected.contains(&block) {
            // Written through the runtime, or guarded with another view of
            // its page: still read-only.
            g.clean.insert(block);
            g.unclean.remove(&block);
            flush(g, &mut run);
            continue;
        }
        if g.rests(block, rest) {
            // Just written by the title (`VICTORY_GUARD_REST`): left open
            // and in `unclean`, so the next asking reports it again.
            RESTED[0].fetch_add(1, Relaxed);
            flush(g, &mut run);
            continue;
        }
        run = match run {
            Some((a, b)) if b == block => Some((a, block + 1)),
            other => {
                if let Some((a, b)) = other {
                    protect(g, a, b);
                }
                Some((block, block + 1))
            }
        };
    }
    flush(g, &mut run);
    written
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent. The same physical pages
/// at every other view of them are made read-only too: a write through
/// one of those is this memory written, and only a fault can say so.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
fn protect(g: &mut Guards, from: usize, to: usize) {
    protect_here(g, from, to);
    if !view_guard() {
        return;
    }
    for (base, bytes) in crate::inplace::views_between(from.saturating_mul(BLOCK), to.saturating_mul(BLOCK)) {
        let first = (base / BLOCK).max(from);
        let last = (base.saturating_add(bytes).div_ceil(BLOCK)).min(to);
        for block in first..last {
            let shown = crate::inplace::pages_of(block * BLOCK);
            // One view (nothing else to guard), or the page the title shows
            // wherever it has nothing (thousands of views): each view of
            // that one is guarded by itself.
            if shown.len() < 2 || shown.len() > MANY_VIEWS {
                continue;
            }
            #[cfg(feature = "diagnostics")]
            ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for alias in shown {
                let alias = alias / BLOCK;
                g.aliased.insert(alias);
                if alias != block {
                    protect_alias(g, alias);
                }
            }
        }
    }
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks
/// opened because another view of theirs was written, and blocks opened
/// because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Makes `block` read-only: another view of pages a copy has just taken
/// through its own view, so that a write through this one faults too. It is
/// not noted clean. The copies of title memory at this view have not taken
/// the pages: noted clean, the block would not be reported written at their
/// next check and they would keep the bytes from before the write (`guard`
/// reports a block that is read-only and not clean as written, then notes it
/// clean). A block that is clean stays so: had the pages been written since
/// its copies took them, `open` would have opened it.
fn protect_alias(g: &mut Guards, block: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    if g.protected.contains(&block) {
        return;
    }
    let mut old = 0;
    // SAFETY: title view memory; only its protection changes.
    if unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0 {
        crate::faults::count(crate::faults::Driver::GuardsRearmed, 1);
        g.protected.insert(block);
        if view_known() {
            g.viewed.insert(block);
        }
    }
}

/// [`protect`] for one range, without the other views of its pages.
fn protect_here(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        crate::faults::count(crate::faults::Driver::GuardsRearmed, (to - from) as u64);
        g.protected.extend(from..to);
        g.clean.extend(from..to);
        if view_known() {
            g.viewed.extend(from..to);
        }
        for block in from..to {
            g.unclean.remove(&block);
        }
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    // Block by block; a block that is not committed answers for the whole
    // uncommitted run it begins (one question, no protection asked for the
    // rest of it; soldsawn cc642f9). The blocks noted absent and parked are
    // the ones the question per block noted.
    let mut block = from;
    while block < to {
        if set(block, block + 1) {
            note(g, block, block + 1);
            block += 1;
            continue;
        }
        let end = uncommitted_end(block).min(to);
        if end <= block {
            block += 1;
            continue;
        }
        for b in block..end {
            g.absent.insert(b);
            // Out of the unclean set while it stays so ([`Guards::parked`]).
            if g.unclean.contains(&b) {
                park(g, b);
            }
        }
        block = end;
    }
}

/// Whether what Windows said of a run of uncommitted blocks holds for the
/// rest of the frame (`VICTORY_GUARD_HOLES_FRAME=1`, soldsawn cc642f9; off
/// by default: every look asks Windows again). Kept until the next frame
/// is presented, or until the runtime maps, unmaps or commits memory
/// ([`title_memory_remapped`], [`title_memory_committed`]). What is given
/// up: a block the title commits by an ordinary commit (no word from the
/// runtime) after the frame's first question about it is copied as zeros
/// until the next frame. soldsawn's level 2 (holes held until the runtime
/// tells of a commit) is what [`Guards::parked`] already does here: `2` is
/// taken as `1`.
fn holes_by_frame() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let v = std::env::var("VICTORY_GUARD_HOLES_FRAME").unwrap_or_default();
        let on = matches!(v.trim(), "1" | "2");
        if on {
            tracing::info!(
                "guards: what Windows says of uncommitted runs is kept for a frame (VICTORY_GUARD_HOLES_FRAME={}; \
                 holes until a commit is told are the parked blocks here)",
                v.trim()
            );
        }
        on
    })
}

/// Runs of uncommitted blocks as Windows described them in frame `.0`
/// (with [`holes_by_frame`]), first block to end.
static HOLES: Mutex<(u64, BTreeMap<usize, usize>)> = Mutex::new((0, BTreeMap::new()));

/// [`uncommitted_end`] of an absent `block`, from this frame's answers
/// with [`holes_by_frame`]. (The guards' lock is held by the caller; the
/// holes' lock is only ever taken inside it or alone.)
fn hole_end(_g: &mut Guards, block: usize) -> usize {
    if !holes_by_frame() {
        return uncommitted_end(block);
    }
    let frame = crate::queue::presented();
    let mut h = HOLES.lock().unwrap_or_else(|e| e.into_inner());
    if h.0 != frame {
        h.1.clear();
        h.0 = frame;
    }
    if let Some((_, &end)) = h.1.range(..=block).next_back()
        && block < end
    {
        return end;
    }
    let end = uncommitted_end(block);
    if end > block {
        h.1.insert(block, end);
    }
    end
}

/// Forgets the holes [`hole_end`] kept: the runtime changed what is mapped
/// or committed.
fn holes_forget() {
    if holes_by_frame() {
        HOLES.lock().unwrap_or_else(|e| e.into_inner()).1.clear();
    }
}

/// A set of blocks, a bit each, 64 to a word, the words by their number
/// (soldsawn f48ce4b). The guards ask whether all of a range is clean some
/// 2,300 times a frame, over 140,000 blocks in all: counted one by one in a
/// tree of blocks, that was most of their 18 ms a frame (soldsawn run int-c).
struct Blocks(BTreeMap<usize, u64>);

impl Blocks {
    const fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// The bits of `range`'s blocks in word `word`.
    fn mask(word: usize, range: &std::ops::Range<usize>) -> u64 {
        let low = word * 64;
        let (from, to) = (range.start.max(low), range.end.min(low + 64));
        if from >= to {
            return 0;
        }
        (u64::MAX >> (64 - (to - from))) << (from - low)
    }

    fn insert(&mut self, block: usize) {
        *self.0.entry(block / 64).or_insert(0) |= 1 << (block % 64);
    }

    fn remove(&mut self, block: usize) {
        if let Some(bits) = self.0.get_mut(&(block / 64)) {
            *bits &= !(1 << (block % 64));
            if *bits == 0 {
                self.0.remove(&(block / 64));
            }
        }
    }

    fn contains(&self, block: usize) -> bool {
        self.0.get(&(block / 64)).is_some_and(|bits| bits >> (block % 64) & 1 != 0)
    }

    fn extend(&mut self, range: std::ops::Range<usize>) {
        if range.is_empty() {
            return;
        }
        for word in range.start / 64..=(range.end - 1) / 64 {
            *self.0.entry(word).or_insert(0) |= Self::mask(word, &range);
        }
    }

    /// Whether every block of `range` is in the set.
    fn holds(&self, range: std::ops::Range<usize>) -> bool {
        if range.is_empty() {
            return true;
        }
        let (first, last) = (range.start / 64, (range.end - 1) / 64);
        let mut words = self.0.range(first..=last);
        (first..=last).all(|word| {
            let mask = Self::mask(word, &range);
            words.next().is_some_and(|(&at, &bits)| at == word && bits & mask == mask)
        })
    }

    /// The set's blocks in `range`, in order.
    fn range(&self, range: std::ops::Range<usize>) -> impl Iterator<Item = usize> + '_ {
        let first = range.start / 64;
        let last = (range.end.max(range.start + 1) - 1) / 64;
        self.0.range(first..=last).flat_map(move |(&word, &bits)| {
            let mut left = bits & Self::mask(word, &range);
            std::iter::from_fn(move || {
                (left != 0).then(|| {
                    let bit = left.trailing_zeros() as usize;
                    left &= left - 1;
                    word * 64 + bit
                })
            })
        })
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded. The same physical pages at
/// every other view of them are opened too: the write that faulted (or is
/// ours, `crate::host::copy_mapped`) goes through one view, and every
/// guard over the pages must copy them again.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    let any = open_here(g, start, end);
    if start >= end || g.aliased.is_empty() {
        return any;
    }
    let aliased: Vec<usize> = g.aliased.range(start / BLOCK..end.div_ceil(BLOCK)).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
        for alias in crate::inplace::pages_of(block * BLOCK) {
            if alias / BLOCK != block && open_here(g, alias, alias.saturating_add(BLOCK)) {
                #[cfg(feature = "diagnostics")]
                ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            g.aliased.remove(&(alias / BLOCK));
        }
    }
    any
}

/// [`open`] for one range, without the other views of its pages.
fn open_here(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
        // A view block: not clean ([`Guards::unclean`]).
        if covered(&g.views, block, block + 1) {
            g.unclean.insert(block);
        }
    }
    if let Some(opening) = g.opening.as_mut() {
        opening.extend_from_slice(&blocks);
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
/// After the layer wrote `[at, at + len)` of title memory through the
/// kernel: the dirty bits the kernel holds for those pages are the layer's
/// own writes, and a refresh that took them for the title's copies the same
/// bytes up again every frame. Asks the watch to clear them and drops the
/// answer (nothing is written into title memory between the write and this
/// asking by design: both are the layer's).
pub(crate) fn forgot(at: usize, len: usize) {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const RESET: u32 = 1;
    if len == 0 {
        return;
    }
    let mut buf = [0usize; 512];
    let first = at / KERNEL_PAGE * KERNEL_PAGE;
    let end = at.saturating_add(len);
    let mut page = first;
    while page < end {
        let pages = (end - page).div_ceil(KERNEL_PAGE).min(buf.len());
        let span = pages * KERNEL_PAGE;
        let mut count = pages;
        let mut granularity = 0u32;
        let _faults = crate::faults::scope(crate::faults::Kind::Watch);
        // SAFETY: `count` entries in the buffer, a whole number of kernel
        // pages of our own process's memory.
        let _ = unsafe {
            GetWriteWatch(
                RESET,
                page as *const _,
                span,
                buf.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        page += span;
    }
}

/// Title memory `[at, at + len)` counts as written now, as a write of the
/// title's found by an asking would (`crate::readback`: a copy-back held,
/// not made): every copy that took its pages takes them again.
pub(crate) fn noted_written(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let mut j = journal();
    j.asked += 1;
    let now = j.asked;
    for page in at / PAGE..=(at + len - 1) / PAGE {
        j.written.insert(page, now);
    }
}

/// Whether every 64 KiB block of `[at, at + len)` is guarded now: a
/// title write there faults before it lands (stage 1 of the copy-back,
/// `crate::readback::consumers`).
pub(crate) fn armed(at: usize, len: usize) -> bool {
    if len == 0 {
        return false;
    }
    let g = guards_read();
    (at / BLOCK..(at + len).div_ceil(BLOCK)).all(|b| g.protected.contains(&b))
}

pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len));
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]), as bits:
/// 1, the guarded blocks under it are opened and count as written (host
/// copies of buffers and textures read through tables take them again);
/// 2, the placed textures picked by index that lie under it are due again;
/// 4, both also for reads to plain addresses when they have finished.
/// Unset or `guards`: bit 1 alone. `0`: nothing, counted only (the default
/// until run 199). `1`: bits 1 and 2. `slots`: bit 2 alone. `all`:
/// everything.
/// (Bit 1 by default: in every run 10,000 to 50,000 guarded blocks of host
/// copies lay under such writes and kept the bytes from before; runs 197
/// and 198 opened them, 9,766 and 10,796 blocks, with nothing worse for
/// it. Bit 2 makes texture slots due and is the one that flooded the
/// queue in an earlier form: still asked for by name.)
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PAGE_WRITES").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("1") => 3,
        Ok("slots") => 2,
        Ok("all") => 7,
        _ if upload_on_write() => 3,
        _ => 1,
    })
}

/// Whether a DirectStorage write that lands in the memory of a placed
/// texture picked by index makes its slot due for an upload at the next
/// frame, an upload made before the bytes were there included
/// (`VICTORY_UPLOAD_ON_WRITE`, on by default: [`page_writes`]' bit 2 unless
/// `VICTORY_PAGE_WRITES` says otherwise; `=0`: only the guards are opened
/// and the fingerprint looks find the change, as before). The uploads go
/// through the texture queue, within its budget.
pub(crate) fn upload_on_write() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UPLOAD_ON_WRITE").is_none_or(|v| v != "0"))
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped (one for each run
/// of addresses); their bytes; guarded blocks that lay under them (host
/// copies that would otherwise keep the bytes from before); reads to
/// plain addresses that finished.
pub(crate) static WRITTEN: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`. `pages`: DirectStorage wrote them into physical pages
/// through a view of the runtime's own, and this is an address where the
/// title has those pages mapped (no title address faulted, nothing was
/// mapped or unmapped; the shortcut this file's header names). Otherwise a
/// read to a plain address has finished, of which [`writing`] was told
/// before it began.
///
/// Upstream's c75fdba does the same for its trackers (its `Writing` marks
/// the pages once more when dropped; page destinations are reported at
/// each address showing them). Here it is counted always and acted on as
/// [`page_writes`] says. Called without the runtime's page-table lock;
/// takes the guards' lock, then, alone, the lock of the ranges kept for
/// the texture pass.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let mode = page_writes();
    let end = at.saturating_add(len);
    // (Stage 1) What is held on the card under these bytes: the title's
    // stand there.
    crate::readback::consumers::title_wrote(at, len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        if mode & 4 == 0 {
            return;
        }
    }
    {
        let mut g = guards();
        let guarded = g.protected.range(at / BLOCK..end.div_ceil(BLOCK)).count() as u64;
        if pages {
            WRITTEN[2].fetch_add(guarded, Relaxed);
        }
        if mode & 1 != 0 && guarded != 0 {
            open(&mut g, at, end);
        }
    }
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len);
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again — and are, the pages noted written, which also has the next check
/// guard the new view (`protect`).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    if ranges.is_empty() {
        return;
    }
    let mut g = guards();
    for &(at, n) in &ranges {
        if open_here(&mut g, at, at.saturating_add(n)) {
            #[cfg(feature = "diagnostics")]
            ALIASES[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
///
/// Only with `VICTORY_MAPPING_RANGES=0` ([`by_range`]); otherwise reads and
/// changes hold the bytes they touch ([`SPANS`]).
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Whether a read in place and a change of mapping keep apart only when
/// their bytes meet ([`SPANS`]), not whatever memory each touches
/// ([`MAPPING`]); `VICTORY_MAPPING_RANGES=0` holds all of title memory for
/// each, as before. On by default (allforks c79c751).
///
/// With one lock over all of title memory each texture refresh waited for
/// every change of mapping anywhere (or, with `VICTORY_TEXTURE_NO_WAIT`,
/// was read through the kernel and often left for later), and each change
/// waited for the refresh under way. While the world streams in the title
/// changes mappings 650-1,100 times a frame, a page at a time. With this a
/// refresh waits for, or with `VICTORY_TEXTURE_NO_WAIT` steps around, only
/// changes of its own bytes.
///
/// Lock order: [`SPANS`] is a leaf, held only to look at and change its
/// lists, never while anything else is asked for (the wait on
/// [`SPANS_LEFT`] lets it go). A read or a change waits here only where the
/// one lock made it wait as well (bytes that meet are some of all bytes,
/// and a change that waits keeps new reads of its bytes out as the lock's
/// waiting writer kept all reads out), so no wait is added to those the
/// locks taken around these already had (the runtime's page table,
/// `JOURNAL`, `GUARDS`, the buffer table's `VICTORY_MIRROR_FAST` lock).
pub(crate) fn by_range() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MAPPING_RANGES").is_none_or(|v| v != "0"))
}

/// Reads of title memory in place under way, and changes of its mapping,
/// by the bytes they touch ([`by_range`]).
struct Spans {
    /// To tell them apart.
    next: u64,
    /// Reads under way: number, start and end.
    reads: Vec<(u64, usize, usize)>,
    /// Changes: number, start, end, and whether under way (else waiting for
    /// the reads over its bytes to end; no read begins over them meanwhile,
    /// so that one read after another cannot keep a change waiting).
    changes: Vec<(u64, usize, usize, bool)>,
    /// Threads waiting on [`SPANS_LEFT`].
    waiting: usize,
}

static SPANS: Mutex<Spans> = Mutex::new(Spans {
    next: 0,
    reads: Vec::new(),
    changes: Vec::new(),
    waiting: 0,
});

/// Told when a read or a change ends while a thread waits.
static SPANS_LEFT: std::sync::Condvar = std::sync::Condvar::new();

fn spans() -> std::sync::MutexGuard<'static, Spans> {
    SPANS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Waits until a read or a change ends.
fn spans_wait(mut s: std::sync::MutexGuard<'static, Spans>) -> std::sync::MutexGuard<'static, Spans> {
    s.waiting += 1;
    let mut s = SPANS_LEFT.wait(s).unwrap_or_else(|e| e.into_inner());
    s.waiting -= 1;
    s
}

/// Whether `[a, b)` and `[c, d)` share a byte.
fn meet(a: usize, b: usize, c: usize, d: usize) -> bool {
    a < d && c < b
}

impl Spans {
    fn read(&mut self, at: usize, end: usize) -> Span {
        self.next += 1;
        self.reads.push((self.next, at, end));
        Span(self.next, false)
    }
}

/// A read's or a change's place in [`SPANS`] (its number, and whether a
/// change), given up when dropped.
struct Span(u64, bool);

impl Drop for Span {
    fn drop(&mut self) {
        let mut s = spans();
        if self.1 {
            s.changes.retain(|c| c.0 != self.0);
        } else if let Some(i) = s.reads.iter().position(|r| r.0 == self.0) {
            s.reads.swap_remove(i);
        }
        if s.waiting > 0 {
            SPANS_LEFT.notify_all();
        }
    }
}

/// What a read or a change of mapping holds: its bytes ([`by_range`]) or
/// the lock over all of title memory.
enum Held<G> {
    All(#[allow(dead_code)] G),
    Bytes(#[allow(dead_code)] Span),
}

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping(#[allow(dead_code)] Held<std::sync::RwLockWriteGuard<'static, ()>>, std::time::Instant);

impl Drop for Remapping {
    fn drop(&mut self) {
        // From asking for the lock until it is let go (the frame log).
        crate::framelog::add(crate::framelog::T::RemapUs, self.1.elapsed().as_micros() as u64);
    }
}

/// The runtime is about to map, unmap or move physical pages of title
/// memory `[at, at + len)`: until what it returns is dropped, no read of
/// those bytes in place is under way and none begins
/// ([`title_memory_reading`]). Taken around the whole change, the
/// unmapping and the mapping again, over all the bytes whose mapping it
/// changes (with `VICTORY_MAPPING_RANGES=0`, all of title memory).
pub fn title_memory_remapping(at: usize, len: usize) -> Remapping {
    crate::readback::settle(at, len);
    crate::readback::consumers::touched(at, len, crate::readback::consumers::Kind::Remap);
    let asked = std::time::Instant::now();
    let held = if by_range() {
        let end = at.saturating_add(len);
        let mut s = spans();
        s.next += 1;
        let id = s.next;
        s.changes.push((id, at, end, false));
        while s.reads.iter().any(|r| meet(r.1, r.2, at, end))
            || s.changes.iter().any(|c| c.3 && meet(c.1, c.2, at, end))
        {
            s = spans_wait(s);
        }
        if let Some(c) = s.changes.iter_mut().find(|c| c.0 == id) {
            c.3 = true;
        }
        Held::Bytes(Span(id, true))
    } else {
        Held::All(MAPPING.write().unwrap_or_else(|e| e.into_inner()))
    };
    crate::framelog::add(crate::framelog::T::Remaps, 1);
    crate::framelog::add(crate::framelog::T::RemapWaitUs, asked.elapsed().as_micros() as u64);
    Remapping(held, asked)
}

/// (Run-45 follow-up) Whether a change of mapping was under way over title
/// memory `[at, at + len)` (one that holds its bytes, not one still waiting
/// for reads): if so, waits until none is (at most `most`) and says true, so
/// that a read which failed there (a view cut and mapped again around a
/// range, its neighbours gone for a moment) is tried again instead of
/// reading as zeros. False at once when none was under way, and with
/// `VICTORY_MAPPING_RANGES=0`. A change still waiting may be waiting for
/// the caller's own read, so it is not waited for.
pub(crate) fn title_memory_change_waited(at: usize, len: usize, most: std::time::Duration) -> bool {
    if !by_range() {
        return false;
    }
    let end = at.saturating_add(len);
    let until = std::time::Instant::now() + most;
    let mut s = spans();
    let busy = |s: &Spans| s.changes.iter().any(|c| c.3 && meet(c.1, c.2, at, end));
    if !busy(&s) {
        return false;
    }
    while busy(&s) {
        let now = std::time::Instant::now();
        if now >= until {
            break;
        }
        s.waiting += 1;
        let (g, _) = SPANS_LEFT.wait_timeout(s, until - now).unwrap_or_else(|e| e.into_inner());
        s = g;
        s.waiting -= 1;
    }
    true
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] Held<std::sync::RwLockReadGuard<'static, ()>>);

/// For reading title memory `[at, at + len)` in place: until what it
/// returns is dropped, the runtime changes no mapping of those bytes (with
/// `VICTORY_MAPPING_RANGES=0`, of any title memory), so memory found mapped
/// stays mapped. Not to be taken twice by one thread, nor held across
/// anything that maps title memory.
pub fn title_memory_reading(at: usize, len: usize) -> Reading {
    // (F3, `VICTORY_LAZY_GPU_VIEW`) A read of the GPU view, for its meter.
    crate::gpuview::note_read(at, len);
    if !by_range() {
        return Reading(Held::All(MAPPING.read().unwrap_or_else(|e| e.into_inner())));
    }
    let end = at.saturating_add(len);
    let mut s = spans();
    while s.changes.iter().any(|c| meet(c.1, c.2, at, end)) {
        s = spans_wait(s);
    }
    Reading(Held::Bytes(s.read(at, end)))
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping of those bytes (or waits to; with
/// `VICTORY_MAPPING_RANGES=0`, of any title memory), for a caller that has
/// another way to read and would rather not wait.
pub fn title_memory_reading_now(at: usize, len: usize) -> Option<Reading> {
    if !by_range() {
        return match MAPPING.try_read() {
            Ok(guard) => Some(Reading(Held::All(guard))),
            Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading(Held::All(e.into_inner()))),
            Err(std::sync::TryLockError::WouldBlock) => None,
        };
    }
    let end = at.saturating_add(len);
    let mut s = spans();
    if s.changes.iter().any(|c| meet(c.1, c.2, at, end)) {
        return None;
    }
    Some(Reading(Held::Bytes(s.read(at, end))))
}

/// The runtime really unmapped title memory `[at, at + len)`: its pages
/// were taken away and nothing was mapped again there (call after
/// [`title_memory_remapped`] of the views that held it, and after what was
/// mapped again around it). Host copies of placed textures and windows of
/// buffers whose memory has no page mapped any more go at the next sweep
/// (`VICTORY_UNMAP_EXACT`, `VICTORY_WINDOW_UNMAP`). Records only: one lock
/// each, no walk of any table.
pub fn title_memory_unmapped(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    crate::resource::note_unmapped_exact(at as u64, len as u64);
    crate::shadow::note_window_unmapped(at as u64, len as u64);
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    // Host copies of placed textures there have nothing left to come from.
    // Record the range; the sweep lets go of them, so that the thread doing
    // the unmapping does not walk the texture table thousands of times a
    // second ([`crate::resource::note_unmapped`]). With `VICTORY_UNMAP_EXACT`
    // (the default) this is only counted: what goes is what
    // [`title_memory_unmapped`] names.
    crate::resource::note_unmapped(at as u64, len as u64);
    holes_forget();
    let mut g = guards();
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    let absent: Vec<usize> = g.absent.range(range.clone()).copied().collect();
    for block in absent {
        g.absent.remove(&block);
    }
    let blocks: Vec<usize> = g.protected.range(range.clone()).copied().collect();
    for block in blocks {
        g.protected.remove(&block);
        g.clean.remove(block);
    }
    let aliased: Vec<usize> = g.aliased.range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
    }
    // What the range shows is new: no longer known to be a view.
    let viewed: Vec<usize> = g.viewed.range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).copied().collect();
    for block in viewed {
        g.viewed.remove(&block);
    }
    // Nor parked: the views below go back to the unclean set.
    let parked: Vec<usize> = g.parked.range(range.clone()).copied().collect();
    for block in parked {
        g.parked.remove(&block);
        g.parked_ordinary.remove(&block);
    }
    // View blocks there are not clean any more.
    let from = g.views.partition_point(|r| r.1 <= range.start);
    let views: Vec<usize> = g.views[from..]
        .iter()
        .take_while(|r| r.0 < range.end)
        .flat_map(|&(a, b)| a.max(range.start)..b.min(range.end))
        .collect();
    g.unclean.extend(views);
}

/// The title committed `[at, at + len)` through the runtime
/// (`XMemVirtualAlloc`): blocks of it found uncommitted before are not
/// known so any more, and the next asking that covers them finds them
/// written. A commit no one tells of is found at the next look at the
/// parked blocks that are no placeholder ([`Guards::parked`]).
pub fn title_memory_committed(at: usize, len: usize) {
    holes_forget();
    let mut g = guards();
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    let parked: Vec<usize> = g.parked.range(range.clone()).copied().collect();
    for block in parked {
        g.parked.remove(&block);
        g.parked_ordinary.remove(&block);
        g.unclean.insert(block);
    }
    let absent: Vec<usize> = g.absent.range(range).copied().collect();
    for block in absent {
        g.absent.remove(&block);
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // (Stage 1) A title write about to land where shaders' bytes are
        // held on the card: they go into title memory first.
        if crate::readback::consumers::tracking() {
            crate::readback::consumers::touched(at, 1, crate::readback::consumers::Kind::TitleWrite);
        }
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        let resting = !rest().is_zero();
        if resting {
            g.opening = Some(Vec::new());
        }
        let opened = open(&mut g, at, at + 1);
        if let Some(blocks) = g.opening.take()
            && !blocks.is_empty()
        {
            // (`VICTORY_GUARD_REST`) They rest, the other views too.
            g.opened(&blocks);
            RESTED[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if opened {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            crate::framelog::add(crate::framelog::T::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        // With the view set: found at the next look at the blocks parked as
        // uncommitted that are no placeholder.
        SUBMISSIONS.fetch_add(park_epoch(), std::sync::atomic::Ordering::Relaxed);
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// Blocks found uncommitted are parked: not asked about again until the
    /// title commits them through the runtime (found at once) or the next
    /// look at the parked blocks that are no placeholder (a commit no one
    /// tells of).
    #[test]
    fn parked_blocks_come_back_when_committed() {
        if !view_set() {
            return;
        }
        // SAFETY: a fresh reservation of two blocks, nothing committed.
        let p = unsafe { VirtualAlloc(std::ptr::null(), 2 * BLOCK, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(p, 0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..2).flat_map(pages).collect();
        // Nothing there: a view, its blocks reported once, then parked.
        assert_eq!(guard(p, p + 2 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 2 * BLOCK), Some(Vec::new()));
        {
            // No placeholder: looked at again each epoch.
            let g = guards();
            assert!(g.parked.contains(&(p / BLOCK)) && g.parked_ordinary.contains(&(p / BLOCK)));
        }
        // SAFETY: the first block of our reservation.
        assert_eq!(unsafe { VirtualAlloc(p as _, BLOCK, MEM_COMMIT, PAGE_READWRITE) } as usize, p);
        title_memory_committed(p, BLOCK);
        assert_eq!(guard(p, p + 2 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 2 * BLOCK), Some(Vec::new()));
        // SAFETY: the second block of our reservation.
        assert_eq!(unsafe { VirtualAlloc((p + BLOCK) as _, BLOCK, MEM_COMMIT, PAGE_READWRITE) } as usize, p + BLOCK);
        SUBMISSIONS.fetch_add(park_epoch(), std::sync::atomic::Ordering::Relaxed);
        assert_eq!(guard(p, p + 2 * BLOCK), Some(pages(1).collect()));
        if absent_private() {
            // Private memory now, committed with no word from the runtime
            // (`VICTORY_ABSENT_PRIVATE`): never guarded, written at every
            // asking.
            assert!(writable(p + BLOCK));
            assert_eq!(guard(p, p + 2 * BLOCK), Some(pages(1).collect()));
            guards().absent.remove(&(p / BLOCK + 1));
        } else {
            assert_eq!(guard(p, p + 2 * BLOCK), Some(Vec::new()));
        }
        title_memory_remapped(p, 2 * BLOCK);
        // SAFETY: our reservation, no longer known to the guards.
        unsafe { VirtualFree(p as _, 0, MEM_RELEASE) };
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        if view_set() {
            // Placeholders: parked until the runtime maps them, never asked
            // about again meanwhile.
            let g = guards();
            for block in [1, 2].map(|k| p / BLOCK + k) {
                assert!(g.parked.contains(&block) && !g.parked_ordinary.contains(&block));
            }
        }
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages
    /// (`writing`): the other views' next checks report the pages written.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        writing(other + BLOCK / 2, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 1)]), "written through this view, checked second");
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0)]), "written through this view, checked second");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }

    /// An uncommitted address noted in `g.absent` that is later allocated as
    /// `MEM_WRITE_WATCH` or ordinary `MEM_PRIVATE` memory must not be hijacked
    /// into `PAGE_READONLY` guarded memory.
    #[test]
    fn absent_block_reallocated_as_private_memory_is_not_guarded() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;

        // 1. Reserve an uncommitted 64 KiB block and observe it via `writes`,
        //    which records the block in `g.absent`.
        let reserved = unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!reserved.is_null());
        let addr = reserved as usize;
        assert!(writes(addr, addr + PAGE, true).is_some());
        assert!(guards().absent.contains(&(addr / BLOCK)));
        assert_ne!(unsafe { VirtualFree(reserved, 0, MEM_RELEASE) }, 0);

        // 2. Reallocate the exact same virtual address as MEM_WRITE_WATCH private memory.
        let watched = unsafe {
            VirtualAlloc(
                reserved,
                PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        };
        assert_eq!(watched, reserved);
        assert!(writes(addr, addr + PAGE, true).is_some());
        assert!(
            writable(addr),
            "MEM_WRITE_WATCH private memory at a formerly absent address must remain writable"
        );
        assert!(!guards().protected.contains(&(addr / BLOCK)));
        assert_ne!(unsafe { VirtualFree(watched, 0, MEM_RELEASE) }, 0);

        // 3. Re-mark `addr` absent while free, then commit ordinary MEM_PRIVATE memory there.
        let reserved2 = unsafe { VirtualAlloc(reserved, PAGE, MEM_RESERVE, PAGE_NOACCESS) };
        assert_eq!(reserved2, reserved);
        assert!(writes(addr, addr + PAGE, true).is_some());
        assert!(guards().absent.contains(&(addr / BLOCK)));
        assert_ne!(unsafe { VirtualFree(reserved2, 0, MEM_RELEASE) }, 0);

        let plain = unsafe { VirtualAlloc(reserved, PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(plain, reserved);
        // Ours: with the view set (`VICTORY_VIEW_SET`) the range stays
        // noted as views until the runtime remaps it, so `writes` answers
        // (myokex: `None`); what matters is that the block is not guarded.
        let answer = writes(addr, addr + PAGE, true);
        assert!(
            answer.is_none() || view_set(),
            "ordinary MEM_PRIVATE memory at a formerly absent address must not be treated as a view"
        );
        SUBMISSIONS.fetch_add(park_epoch(), std::sync::atomic::Ordering::Relaxed);
        let _ = writes(addr, addr + PAGE, true);
        assert!(writable(addr));
        assert!(!guards().protected.contains(&(addr / BLOCK)));
        // (Left allocated: what was noted as written there would be told to
        // the next test that gets the address.)
        // Clean up the test's absent entry.
        guards().absent.remove(&(addr / BLOCK));
        guards().unclean.remove(&(addr / BLOCK));
    }

    /// The same behind a view: a block of a range that starts with a view
    /// turns into private memory. It stays writable however often the range
    /// is asked about, and counts as written each time (nothing watches it).
    #[test]
    fn private_memory_behind_a_view_is_never_guarded() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE,
            MapViewOfFileEx, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        // SAFETY: memory of this test alone: a view of one block and a
        // reservation right behind it, where a larger one was just released.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                BLOCK as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            // Another test's thread may take the released addresses first.
            let mut found = None;
            for _ in 0..64 {
                let both = VirtualAlloc(std::ptr::null(), 2 * BLOCK, MEM_RESERVE, PAGE_NOACCESS);
                assert!(!both.is_null());
                assert_ne!(VirtualFree(both, 0, MEM_RELEASE), 0);
                let view = MapViewOfFileEx(section, FILE_MAP_WRITE, 0, 0, BLOCK, both).Value;
                let behind = both as usize + BLOCK;
                let reserved = VirtualAlloc(behind as *const _, BLOCK, MEM_RESERVE, PAGE_NOACCESS);
                if view == both && reserved as usize == behind {
                    found = Some((both as usize, behind, reserved));
                    break;
                }
            }
            let (start, behind, reserved) = found.expect("a view with a reservation behind it");
            assert!(writes(start, start + 2 * BLOCK, true).is_some());
            assert!(guards().absent.contains(&(behind / BLOCK)));
            let plain = VirtualAlloc(reserved, BLOCK, MEM_COMMIT, PAGE_READWRITE);
            assert_eq!(plain, reserved);
            for round in 0..3 {
                let (_, written) = writes(start, start + 2 * BLOCK, true).unwrap();
                assert!(writable(behind), "round {round}: private memory was made read-only");
                assert!(!guards().protected.contains(&(behind / BLOCK)), "round {round}");
                assert!(
                    written.iter().any(|&(page, _)| page * PAGE == behind),
                    "round {round}: what nothing watches counts as written"
                );
            }
            // The two blocks stay taken: what was noted as written there
            // would be told to the next test that gets these addresses.
            guards().absent.remove(&(behind / BLOCK));
            CloseHandle(section);
        }
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{PAGE, by_range, title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// Addresses only these tests hold (nothing is read there).
    const HERE: usize = 0x7e00_0000_0000;

    /// A change of mapping waits for a read in place of its bytes that is
    /// under way, and such a read waits for the change.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading(HERE + PAGE / 2, PAGE);
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping(HERE + PAGE, PAGE);
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping(HERE, 2 * PAGE);
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading(HERE + PAGE + 16, 16);
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }

    /// Reads and changes whose bytes do not meet do not wait for each other,
    /// and a change waiting for a read keeps new reads of its bytes out (but
    /// not of others).
    #[test]
    fn reads_and_remappings_of_other_bytes_go_on() {
        if !by_range() {
            return; // VICTORY_MAPPING_RANGES=0: one lock over all of it
        }
        let at = HERE + 16 * PAGE;
        let reading = title_memory_reading(at, PAGE);
        // Another page changes while the read is under way, and another
        // read goes on while that change is.
        let (done, finished) = mpsc::channel();
        std::thread::spawn(move || {
            let changing = title_memory_remapping(at + PAGE, PAGE);
            let (read, was_read) = mpsc::channel();
            std::thread::spawn(move || {
                let _reading = title_memory_reading(at + 2 * PAGE, PAGE);
                read.send(()).unwrap();
            });
            let other = was_read.recv_timeout(Duration::from_secs(10)).is_ok();
            drop(changing);
            done.send(other).unwrap();
        });
        assert_eq!(finished.recv_timeout(Duration::from_secs(10)), Ok(true), "kept apart though their bytes do not meet");
        // A change of the read's page waits, and keeps a new read of its
        // bytes out meanwhile; a read elsewhere is not kept out.
        let changed = Arc::new(AtomicBool::new(false));
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                let _changing = title_memory_remapping(at, PAGE);
                changed.store(true, Ordering::SeqCst);
            })
        };
        while !super::spans().changes.iter().any(|c| c.1 == at) {
            std::thread::yield_now();
        }
        assert!(super::title_memory_reading_now(at + 8, 8).is_none(), "a read began over a change that waits");
        assert!(super::title_memory_reading_now(at + 3 * PAGE, 8).is_some());
        std::thread::sleep(Duration::from_millis(50));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
    }
}

#[cfg(test)]
mod blocks_tests {
    use super::Blocks;
    use std::collections::BTreeSet;

    /// The bit set answers as a tree of blocks does, across word ends:
    /// membership of whole ranges, and the blocks of a range in order
    /// (soldsawn f48ce4b's test).
    #[test]
    fn blocks_answer_as_a_tree_of_blocks() {
        let mut bits = Blocks::new();
        let mut tree = BTreeSet::new();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |most: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize % most
        };
        for step in 0..4000 {
            let (at, len) = (1000 + next(400), next(140));
            match step % 4 {
                0 => {
                    bits.extend(at..at + len);
                    tree.extend(at..at + len);
                }
                1 => {
                    bits.insert(at);
                    tree.insert(at);
                }
                _ => {
                    bits.remove(at);
                    tree.remove(&at);
                }
            }
            let probe = 990 + next(420);
            assert_eq!(bits.contains(probe), tree.contains(&probe), "block {probe} at step {step}");
            let (from, n) = (990 + next(420), next(200));
            let range = from..from + n;
            assert_eq!(
                bits.holds(range.clone()),
                tree.range(range.clone()).count() == n,
                "all of {range:?} at step {step}"
            );
            assert_eq!(
                bits.range(range.clone()).collect::<Vec<_>>(),
                tree.range(range.clone()).copied().collect::<Vec<_>>(),
                "the blocks of {range:?} at step {step}"
            );
        }
    }
}

#[cfg(test)]
mod journal_tests {
    use super::{journal, settled, unlocked};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// Askings made without the journal's lock (JoshuuDev 3b270b1's test):
    /// one of other memory goes on while another is under way; one of
    /// overlapping memory, and a journal answer of it, wait for it.
    #[test]
    fn askings_of_overlapping_memory_wait_and_others_go_on() {
        // Pages far from anything else the tests ask about.
        let (first, last) = (0x7f00_0000, 0x7f00_0010);
        let (release, held) = mpsc::channel::<()>();
        let (inside, entered) = mpsc::channel::<()>();
        let a = std::thread::spawn(move || {
            let (_j, ()) = unlocked(journal(), first, last, || {
                inside.send(()).unwrap();
                held.recv().unwrap();
            });
        });
        entered.recv().unwrap();
        // Other memory: at once, while the first asking is under way.
        let (j, ran) = unlocked(journal(), last + 100, last + 110, || true);
        assert!(ran);
        drop(j);
        // Overlapping memory: not until the first asking ends.
        let done = Arc::new(AtomicBool::new(false));
        let b = {
            let done = done.clone();
            std::thread::spawn(move || {
                let (_j, ()) = unlocked(journal(), last - 1, last + 1, || done.store(true, Ordering::SeqCst));
            })
        };
        let answered = Arc::new(AtomicBool::new(false));
        let c = {
            let answered = answered.clone();
            std::thread::spawn(move || {
                let _j = settled(journal(), first + 2, first + 3);
                answered.store(true, Ordering::SeqCst);
            })
        };
        std::thread::sleep(Duration::from_millis(100));
        assert!(!done.load(Ordering::SeqCst), "an asking of overlapping memory ran beside another");
        assert!(!answered.load(Ordering::SeqCst), "a journal answer of memory under asking did not wait");
        release.send(()).unwrap();
        a.join().unwrap();
        b.join().unwrap();
        c.join().unwrap();
        assert!(done.load(Ordering::SeqCst) && answered.load(Ordering::SeqCst));
        assert!(journal().in_flight.is_empty());
    }
}
