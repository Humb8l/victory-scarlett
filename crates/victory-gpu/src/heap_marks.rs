//! Which buffer views of the view heap shaders store through when they pick
//! them at an index they compute (SM 6.6 dynamic resources). Ported from
//! the all-forks tree (WillyboPeep/victory-all-forks a1ce531), adapted.
//!
//! On the console such a view names title memory itself: the store is in
//! the title's memory at once. Here it lands in the host copy the view
//! shows, and only ranges a list declared written
//! ([`crate::shadow::Shadow::gpu_writes`]) are compared and copied back
//! after its submission. Tables and root views say which buffers a draw can
//! write; a computed index says nothing, and declaring every unordered
//! access view of the heap is not affordable (960 MB of them in the world,
//! for some 30 lists a frame, in the all-forks run of 2026-11-24).
//!
//! So the translated shader says it: before every store or atomic operation
//! through a picked buffer view it sets dword `slot` of one buffer of ours
//! (`victory_shader::emit::SLOT_HEAP_MARKS`, bound as a root unordered
//! access view) to its tag (`victory_shader::emit::heap_mark_tag` of its
//! container). The marks are read and cleared at every list that runs such
//! a shader (the heap's unordered views) and once a frame (all its buffer
//! views); a slot marked within the last [`KEEP_FRAMES`] frames is one
//! shaders write, and lists that run a shader storing through picked views
//! declare the ranges of those slots' views written, and nothing else.
//! `VICTORY_HEAP_MARKS=0`: no marks are emitted nor read (as before).
//!
//! shortcut: the marks come back after the shader ran, so the stores of the
//! lists that ran before a slot's first mark was read (at least the list
//! that first stored) are not copied back; a buffer a shader writes once
//! and never again stays unseen by the title. [`FIRST_MARKS`] counts them.
//! The whole answer needs snapshots of every unordered view before each
//! submission (the 960 MB above).
//! shortcut: one buffer of marks for every view heap; slot `n` marked in
//! one heap is taken as marked in the others (more declared, never less).
//! shortcut: every list that runs a marking shader declares every marked
//! view, whichever of them its own shaders store through.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::emit::HEAP_MARKS_DWORDS;
use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};

/// Frames a slot stays one that shaders write after its last mark: a
/// buffer written every few frames is not dropped and found again (losing
/// a list's stores each time).
pub(crate) const KEEP_FRAMES: u64 = 120;

/// Slots found marked for the first time (or again after [`KEEP_FRAMES`]
/// without a mark), since the start: each is at least a list of stores
/// that were not copied back.
pub(crate) static FIRST_MARKS: AtomicU64 = AtomicU64::new(0);
/// Marks found on slots whose view is not an unordered access one, since
/// the start: the host made a read-only view where a shader stores, and
/// the store was dropped.
pub(crate) static READ_ONLY_MARKS: AtomicU64 = AtomicU64::new(0);
/// Since the last frame line: marks found set (slots), views declared
/// written by lists because of marks, and their bytes (what the copy-back
/// compares, and copies where shaders changed it).
pub(crate) static MARKED: AtomicU64 = AtomicU64::new(0);
pub(crate) static DECLARED_VIEWS: AtomicU64 = AtomicU64::new(0);
pub(crate) static DECLARED_BYTES: AtomicU64 = AtomicU64::new(0);
/// Lists that ran a shader storing through picked views, since the last
/// frame line.
pub(crate) static LISTS: AtomicU64 = AtomicU64::new(0);
/// Slots taken as written at the last reading.
pub(crate) static WRITTEN_SLOTS: AtomicU64 = AtomicU64::new(0);

/// Whether translated shaders mark the slots they store through
/// (`victory_shader::emit::heap_marks`, `VICTORY_HEAP_MARKS`).
pub(crate) fn on() -> bool {
    victory_shader::emit::heap_marks()
}

/// The frame line of the marks, per frame over the `frames` since the last
/// (said when a list ran a marking shader in them, or with
/// `VICTORY_HEAP_EVERY_LIST=1` refreshes to count).
pub(crate) fn line(frames: f64) {
    let take = |c: &AtomicU64| c.swap(0, Ordering::Relaxed) as f64 / frames;
    let lists = take(&LISTS);
    let late = take(&crate::draw::HEAP_LATE_REFRESHES);
    if lists == 0.0 && late == 0.0 {
        return;
    }
    let (marked, views, mb, back_mb) = (
        take(&MARKED),
        take(&DECLARED_VIEWS),
        take(&DECLARED_BYTES) / 1048576.0,
        take(&BACK_BYTES) / 1048576.0,
    );
    tracing::info!(
        "stores through views picked by index, per frame (VICTORY_HEAP_MARKS={}): {lists:.1} lists ran a marking shader, \
         {marked:.1} slots found marked, {views:.1} views declared written through marks ({mb:.2} MB compared), \
         {back_mb:.3} MB copied back because of marks; {} slots taken as written at the last reading, \
         {} first marks and {} marks on read-only views so far; {late:.0} later refreshes (VICTORY_HEAP_EVERY_LIST)",
        if on() { "on" } else { "0" },
        WRITTEN_SLOTS.load(Ordering::Relaxed),
        FIRST_MARKS.load(Ordering::Relaxed),
        READ_ONLY_MARKS.load(Ordering::Relaxed),
    );
}

/// Per heap slot: the frame its mark was last found set.
#[derive(Default)]
struct Written(HashMap<u32, u64>);

impl Written {
    /// `slot` was found marked in `frame`. Whether it was not known as
    /// written before.
    fn mark(&mut self, slot: u32, frame: u64) -> bool {
        match self.0.insert(slot, frame) {
            Some(last) => frame.saturating_sub(last) > KEEP_FRAMES,
            None => true,
        }
    }

    /// Forgets the slots not marked for [`KEEP_FRAMES`] frames.
    fn expire(&mut self, frame: u64) {
        self.0
            .retain(|_, last| frame.saturating_sub(*last) <= KEEP_FRAMES);
    }

    fn has(&self, slot: u32) -> bool {
        self.0.contains_key(&slot)
    }
}

/// `ranges` (`[from, to)` each) as the fewest ranges covering the same
/// bytes, ascending.
pub(crate) fn merged(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.retain(|r| r.0 < r.1);
    ranges.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// Bytes that went back into title memory inside ranges declared written
/// because of marks, since the last frame line ([`back`]).
pub(crate) static BACK_BYTES: AtomicU64 = AtomicU64::new(0);

/// Title ranges (`start` to `end`) lists last declared written because of
/// marks ([`declared`]), for [`back`].
static DECLARED: std::sync::RwLock<std::collections::BTreeMap<u64, u64>> =
    std::sync::RwLock::new(std::collections::BTreeMap::new());
static ANY_DECLARED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A list declared the title ranges `ranges` written because of marks
/// (they replace those it had before; a range a shader stops marking is
/// forgotten after [`KEEP_FRAMES`]).
pub(crate) fn declared(ranges: impl Iterator<Item = (u64, u64)>) {
    let mut map = DECLARED.write().unwrap_or_else(|e| e.into_inner());
    map.clear();
    for (from, to) in merged(ranges.collect()) {
        map.insert(from, to);
    }
    ANY_DECLARED.store(!map.is_empty(), Ordering::Relaxed);
}

/// The copy-back put `n` bytes shaders wrote into title memory at `at`:
/// those inside ranges declared because of marks are counted
/// ([`BACK_BYTES`]).
pub(crate) fn back(at: usize, n: usize) {
    if n == 0 || !ANY_DECLARED.load(Ordering::Relaxed) {
        return;
    }
    let (from, to) = (at as u64, (at + n) as u64);
    let map = DECLARED.read().unwrap_or_else(|e| e.into_inner());
    let mut inside = 0;
    for (&a, &e) in map.range(..to).rev() {
        if e <= from {
            break;
        }
        inside += e.min(to) - a.max(from);
    }
    if inside > 0 {
        BACK_BYTES.fetch_add(inside, Ordering::Relaxed);
    }
}

/// The attribution lines (`heap_marks` target) said so far, and their cap.
const SAY_AT_MOST: usize = 512;

struct Marks {
    buffer: ID3D12Resource,
    data: *mut u32,
    gpu: u64,
    written: Written,
    /// Program tag (`victory_shader::emit::heap_mark_tag`) to its hash as
    /// the "first bind" lines give it, and its stage.
    tags: HashMap<u32, (String, &'static str)>,
    /// The (program tag, slot) pairs whose attribution line was said.
    said: HashSet<(u32, u32)>,
    /// The programs (hashes) in `tags`.
    tagged: HashSet<String>,
}

// SAFETY: the mapping is of a buffer the struct owns; all access is under
// the mutex.
unsafe impl Send for Marks {}

static MARKS: Mutex<Option<Marks>> = Mutex::new(None);

/// The GPU address of the marks for a draw of `list` to bind as its root
/// unordered access view; the list holds the buffer. `shader` is the
/// program that marks (for the attribution of its marks). None if the
/// host could not make the buffer.
pub(crate) fn bind(list: &HostList, shader: &crate::pipeline::Shader) -> Option<u64> {
    let mut guard = MARKS.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let bytes = u64::from(HEAP_MARKS_DWORDS) * 4;
        let (buffer, data) = host()?.shared_buffer(bytes)?;
        crate::host::name(&buffer, || {
            "marks of heap slots shaders store through".to_string()
        });
        // SAFETY: a fresh mapping of at least that many bytes.
        unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        tracing::info!(
            "a shader stores through buffer views it picks from the view heap by index: the slots it marks are copied back (VICTORY_HEAP_MARKS=0: not)"
        );
        *guard = Some(Marks {
            buffer,
            data: data.cast(),
            gpu,
            written: Written::default(),
            tags: HashMap::new(),
            said: HashSet::new(),
            tagged: HashSet::new(),
        });
    }
    let m = guard.as_mut()?;
    const KEEP: usize = usize::MAX - 0x60;
    if list.first_use(KEEP) {
        list.keep(&m.buffer);
    }
    // Each program once (its tag is a hash of its whole container).
    if m.tagged.insert(shader.hash.clone()) {
        let tag = victory_shader::emit::heap_mark_tag(shader.container());
        m.tags.insert(tag, (shader.hash.clone(), shader.stage));
    }
    Some(m.gpu)
}

/// A slot of the heap holding a buffer view: whether the view is an
/// unordered access one, and the title range it shows (for the
/// attribution line).
pub(crate) struct Slot {
    pub slot: u32,
    pub unordered: bool,
    pub address: u64,
    pub len: u64,
}

/// Reads and clears the marks of `slots` and returns those of them
/// (unordered views only) shaders write, by the marks of this and earlier
/// frames. Empty until a shader that marks has been bound. Says the
/// attribution line of each (program, slot) found the first time.
pub(crate) fn take<'a>(frame: u64, slots: impl Iterator<Item = &'a Slot>) -> Vec<u32> {
    let mut guard = MARKS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(m) = guard.as_mut() else {
        return Vec::new();
    };
    m.written.expire(frame);
    let mut out = Vec::new();
    for s in slots {
        let slot = s.slot;
        // The last dword is the sink of lanes that do not store.
        if slot >= HEAP_MARKS_DWORDS - 1 {
            continue;
        }
        // SAFETY: inside the mapping; the GPU may be writing, and a mark
        // missed now is set again by the next store (the slot is already
        // taken as written when the value read is not 0).
        let at = unsafe { m.data.add(slot as usize) };
        // SAFETY: as above.
        let tag = unsafe { at.read_volatile() };
        if tag != 0 {
            // SAFETY: as above.
            unsafe { at.write_volatile(0) };
            MARKED.fetch_add(1, Ordering::Relaxed);
            if m.said.len() <= SAY_AT_MOST && m.said.insert((tag, slot)) {
                if m.said.len() <= SAY_AT_MOST {
                    let (hash, stage) = m
                        .tags
                        .get(&tag)
                        .map_or(("unknown", "unknown"), |(h, s)| (h.as_str(), *s));
                    tracing::info!(
                        target: "heap_marks",
                        "a program stores through a view it picks by index: hash={hash} stage={stage} slot={slot} address={:#x} len={:#x} unordered={} frame={frame}",
                        s.address,
                        s.len,
                        s.unordered
                    );
                } else {
                    tracing::info!(
                        target: "heap_marks",
                        "a program stores through a view it picks by index: no more of these lines ({SAY_AT_MOST} said)"
                    );
                }
            }
            if !s.unordered {
                READ_ONLY_MARKS.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if m.written.mark(slot, frame) {
                FIRST_MARKS.fetch_add(1, Ordering::Relaxed);
            }
        }
        if s.unordered && m.written.has(slot) {
            out.push(slot);
        }
    }
    WRITTEN_SLOTS.store(out.len() as u64, Ordering::Relaxed);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_merge_where_they_touch_or_overlap() {
        assert_eq!(merged(vec![]), vec![]);
        assert_eq!(merged(vec![(8, 8), (4, 2)]), vec![]);
        assert_eq!(
            merged(vec![
                (40, 50),
                (0, 10),
                (10, 20),
                (15, 18),
                (19, 30),
                (60, 70),
                (45, 61)
            ]),
            vec![(0, 30), (40, 70)]
        );
    }

    #[test]
    fn a_slot_stays_written_for_the_kept_frames_after_its_last_mark() {
        let mut w = Written::default();
        assert!(w.mark(7, 100));
        assert!(!w.mark(7, 101));
        w.expire(101 + KEEP_FRAMES);
        assert!(w.has(7));
        w.expire(102 + KEEP_FRAMES);
        assert!(!w.has(7));
        // Found again: a first mark once more.
        assert!(w.mark(7, 400));
        assert!(!w.has(8));
    }

    #[test]
    fn bytes_copied_back_count_inside_ranges_declared_through_marks_only() {
        BACK_BYTES.store(0, Ordering::Relaxed);
        declared([(0x2000, 0x3000), (0x1000, 0x1800), (0x1800, 0x1900)].into_iter());
        back(0x0f00, 0x200); // 0x100 inside
        back(0x2f80, 0x100); // 0x80 inside
        back(0x4000, 0x100); // outside
        assert_eq!(BACK_BYTES.load(Ordering::Relaxed), 0x180);
        declared(std::iter::empty());
        back(0x2000, 0x100);
        assert_eq!(BACK_BYTES.swap(0, Ordering::Relaxed), 0x180);
    }

    #[test]
    fn the_tag_a_program_marks_with_is_never_zero_and_is_the_translators() {
        assert_eq!(victory_shader::emit::heap_mark_tag(&[]) & 1, 1);
        assert_ne!(
            victory_shader::emit::heap_mark_tag(b"a"),
            victory_shader::emit::heap_mark_tag(b"b")
        );
    }
}
