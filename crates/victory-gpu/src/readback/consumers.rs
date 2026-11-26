//! What later consumes the bytes the copy-back puts into title memory
//! (stage 0 of docs/copy-back-design.md), and the copy-back that leaves
//! alone what nothing consumes (stage 1).
//!
//! * `VICTORY_BACK_CONSUMERS` (default 1 in diagnostics builds, 0
//!   otherwise): the meter. Title memory is cut into 64 KiB pieces (title
//!   addresses). A copy-back write into a piece ([`produced`], from
//!   `readback::back`) opens a "generation" of it (one per landing,
//!   [`landing`]); the first consumer of the piece's bytes after that
//!   ([`reading`]: every layer read of title memory through
//!   `host::read_into` / `host::copy_mapped`, tagged with a [`Kind`] by the
//!   caller's [`scope`]) takes all of the generation's bytes, so no byte is
//!   counted twice; a piece written again before anything read it, or left
//!   unread for 150 frames, counts its bytes as consumed by nothing. Bytes
//!   inside the read-back set count as "title read-back" when written.
//!   A line every 5 s ([`line`]).
//! * `VICTORY_BACK_SKIP_UNREAD` (default 0): stage 1. A piece of a copy in
//!   the card's memory whose last [`K`] generations nobody consumed, that
//!   lies outside the read-back set at each of its addresses, whose guard
//!   blocks stand (a title write faults) and that our own view can write,
//!   is not fetched from the card nor written ([`try_skip`]): title memory
//!   keeps its bytes (kept here as `t0`) and the piece is "card newer".
//!   Before anything that could need the bytes ([`demand`]): a layer read
//!   or write of it at any of its addresses, a title write (the guard
//!   fault, before the write lands), a title buffer lock (the lock hook,
//!   on with this switch), a remap or unmap, the piece fetched again by a
//!   later landing (merged there with `t0`), the newest bytes are fetched
//!   from the copy that holds them (`host::demand_fetch`, on the direct
//!   queue behind everything sent) and merged as the landing would have
//!   (`Shadow::card_merge_relay_with`), except where title memory no
//!   longer holds `t0` or DirectStorage wrote since ([`title_wrote`]): the
//!   title's bytes are newer there. The piece is then held to the old way
//!   for `VICTORY_BACK_SKIP_HOLD_FRAMES` (300). Needs the relay, our own
//!   views (`VICTORY_BACK_OWN_VIEW=1`) and nothing in place.
//!
//! shortcut: the title's own CPU reads that are neither a ring read nor a
//! buffer lock are not seen (the bet F1 measured: 27 rings); the switch is
//! the way back.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::shadow::Shadow;

/// Bytes of a piece.
pub(crate) const PIECE: usize = 64 << 10;
/// Generations with no consumer before a piece may be left on the card.
pub(crate) const K: u32 = 8;
/// Frames a generation waits for a consumer before it counts as unread.
const STALE_FRAMES: u64 = 150;

/// Who read (or touched) title memory.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub(crate) enum Kind {
    /// Bytes inside the read-back set (the title reads them).
    Title = 0,
    /// The refresh of a window of title memory.
    Window,
    /// The refresh of a copy of a raw view's own bytes (`shadow::exact`).
    OwnView,
    /// The refresh of a title buffer's copy.
    Plain,
    /// A texture's host copy taking its bytes.
    Texture,
    /// Descriptor records read on the CPU.
    Descriptor,
    /// Draw, indirect and constant records read on the CPU.
    Records,
    /// The layer's compares and checks of title memory against its copies.
    Hashing,
    /// Any other layer read.
    Other,
    /// A title buffer lock (the lock hook).
    TitleLock,
    /// A title write (guard fault).
    TitleWrite,
    /// A remap or unmap of the memory.
    Remap,
    /// The layer writing title memory.
    LayerWrite,
    /// Not a consumer: the copy-back's own reads, meters.
    Ignore,
}

const KINDS: usize = Kind::Ignore as usize;
const NAMES: [&str; KINDS] = [
    "title read-back",
    "window",
    "own-view copy",
    "plain copy",
    "texture source",
    "descriptor records",
    "draw/indirect records",
    "layer hashing",
    "other layer read",
    "title lock",
    "title write",
    "remap",
    "layer write",
];

thread_local! {
    static KIND: Cell<Kind> = const { Cell::new(Kind::Other) };
    /// The copy whose landing writes now, and its writer.
    static PRODUCER: Cell<(usize, bool, u64)> = const { Cell::new((0, false, 0)) };
    /// The pipeline whose shader declares writes now (`set_writer`).
    static WRITER: Cell<u64> = const { Cell::new(0) };
    /// Inside a demand: no demand, no metering.
    static IN_DEMAND: Cell<bool> = const { Cell::new(false) };
    /// A copy's reference lock (`Shadow::before`) is held: a demand here
    /// could wait for a thread that waits for it. The holders demand
    /// before they take it.
    static SIDE_HELD: Cell<u32> = const { Cell::new(0) };
}

/// `VICTORY_BACK_CONSUMERS`.
pub(crate) fn meter_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_BACK_CONSUMERS") {
        Ok(v) => v.trim() == "1",
        Err(_) => cfg!(feature = "diagnostics"),
    } && super::set_on())
}

/// `VICTORY_BACK_SKIP_UNREAD=1`, with what it needs.
pub(crate) fn skip_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let asked = std::env::var("VICTORY_BACK_SKIP_UNREAD").is_ok_and(|v| v.trim() == "1");
        let can = super::set_on() && crate::inplace::level() == 0 && crate::inplace::back_own() && crate::shadow::relay();
        if asked && !can {
            tracing::warn!(
                "VICTORY_BACK_SKIP_UNREAD=1 not used: it needs the read-back set, VICTORY_BACK_OWN_VIEW=1, the relay and VICTORY_INPLACE=0"
            );
        }
        asked && can
    })
}

/// Whether pieces are tracked at all.
pub(crate) fn tracking() -> bool {
    meter_on() || skip_on()
}

fn hold_frames() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_BACK_SKIP_HOLD_FRAMES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(300))
}

fn skip_most() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_BACK_SKIP_UNREAD_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(512) << 20)
}

/// Until dropped, reads on this thread are of `kind`.
pub(crate) struct Scope(Kind);

pub(crate) fn scope(kind: Kind) -> Scope {
    Scope(KIND.replace(kind))
}

impl Drop for Scope {
    fn drop(&mut self) {
        KIND.set(self.0);
    }
}

/// Until dropped, this thread's landing writes are `copy`'s (a copy in
/// the card's memory if `card`), and its reads are not consumers.
pub(crate) struct Producing((usize, bool, u64), #[allow(dead_code)] Scope);

pub(crate) fn producing(copy: usize, card: bool) -> Producing {
    let writer = if tracking() { writer_of(copy) } else { 0 };
    Producing(PRODUCER.replace((copy, card, writer)), scope(Kind::Ignore))
}

impl Drop for Producing {
    fn drop(&mut self) {
        PRODUCER.set(self.0);
    }
}

/// Until dropped, this thread holds a copy's reference lock.
pub(crate) struct SideHeld;

pub(crate) fn side_held() -> SideHeld {
    SIDE_HELD.set(SIDE_HELD.get() + 1);
    SideHeld
}

impl Drop for SideHeld {
    fn drop(&mut self) {
        SIDE_HELD.set(SIDE_HELD.get().saturating_sub(1));
    }
}

/// Whether this thread is inside a demand.
pub(crate) fn in_demand() -> bool {
    IN_DEMAND.get()
}

/// The pipeline (its shader's hash, first 16 hex digits) whose stage
/// declares shader writes now; 0: none known.
pub(crate) fn set_writer(hash: &str) {
    if tracking() {
        WRITER.set(u64::from_str_radix(&hash[..hash.len().min(16)], 16).unwrap_or(0));
    }
}

pub(crate) fn clear_writer() {
    WRITER.set(0);
}

static WRITERS: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);

/// Shaders of the current stage declare writes to `copy`.
pub(crate) fn declared(copy: usize) {
    if !tracking() {
        return;
    }
    let w = WRITER.get();
    if w == 0 {
        return;
    }
    let mut m = WRITERS.lock().unwrap_or_else(|e| e.into_inner());
    let m = m.get_or_insert_with(HashMap::new);
    if m.len() < 1 << 16 || m.contains_key(&copy) {
        m.insert(copy, w);
    }
}

fn writer_of(copy: usize) -> u64 {
    WRITERS.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| m.get(&copy).copied()).unwrap_or(0)
}

/// A piece of title memory as the meter knows it.
#[derive(Default, Clone, Debug)]
pub(crate) struct Piece {
    /// Bytes of the open generation not consumed yet.
    pending: u64,
    generation: u64,
    writer: u64,
    /// Frame of the last write.
    frame: u64,
    /// Generations in a row that nothing consumed.
    pub(crate) quiet: u32,
    /// No skip before this frame (demoted).
    hold_until: u64,
}

/// The meter's counts, since the last line.
#[derive(Default)]
pub(crate) struct Counts {
    pub(crate) back: u64,
    pub(crate) ranges: u64,
    pub(crate) card: u64,
    pub(crate) by: [u64; KINDS],
    pub(crate) nothing: u64,
    pub(crate) pairs: HashMap<(u64, u8), u64>,
}

/// The pieces (by piece number) and the counts.
#[derive(Default)]
pub(crate) struct Table {
    pub(crate) pieces: BTreeMap<usize, Piece>,
    pub(crate) counts: Counts,
}

impl Counts {
    fn pair(&mut self, writer: u64, kind: u8, n: u64) {
        if self.pairs.len() < 4096 || self.pairs.contains_key(&(writer, kind)) {
            *self.pairs.entry((writer, kind)).or_insert(0) += n;
        }
    }
}

/// `KINDS` stands for "nothing" in the pairs.
const NOTHING: u8 = KINDS as u8;

impl Table {
    /// A copy-back write of `n` bytes at `at` (`inside` of them in the
    /// read-back set), in generation `generation`, frame `frame`.
    pub(crate) fn produced(&mut self, at: usize, n: usize, inside: u64, card: bool, writer: u64, generation: u64, frame: u64) {
        if n == 0 {
            return;
        }
        let c = &mut self.counts;
        c.back += n as u64;
        c.ranges += 1;
        if card {
            c.card += n as u64;
        }
        c.by[Kind::Title as usize] += inside;
        c.pair(writer, Kind::Title as u8, inside);
        let mut inside = inside;
        let end = at + n;
        for page in at / PIECE..=(end - 1) / PIECE {
            let overlap = (end.min((page + 1) * PIECE) - at.max(page * PIECE)) as u64;
            let left = inside.min(overlap);
            inside -= left;
            let bytes = overlap - left;
            if bytes == 0 {
                continue;
            }
            let p = self.pieces.entry(page).or_default();
            if p.pending > 0 && p.generation != generation {
                self.counts.nothing += p.pending;
                let (w, n) = (p.writer, p.pending);
                p.quiet = p.quiet.saturating_add(1);
                p.pending = 0;
                self.counts.pair(w, NOTHING, n);
            }
            let p = self.pieces.entry(page).or_default();
            p.generation = generation;
            p.pending += bytes;
            p.writer = writer;
            p.frame = frame;
        }
    }

    /// Whether a read of `[at, at + len)` meets a piece with bytes not
    /// consumed yet.
    pub(crate) fn meets(&self, at: usize, len: usize) -> bool {
        len != 0 && self.pieces.range(at / PIECE..=(at + len - 1) / PIECE).any(|(_, p)| p.pending > 0)
    }

    /// `kind` read `[at, at + len)`: the pending bytes of the pieces it
    /// meets are its.
    pub(crate) fn consumed(&mut self, at: usize, len: usize, kind: Kind) {
        if len == 0 {
            return;
        }
        let mut taken = Vec::new();
        for (_, p) in self.pieces.range_mut(at / PIECE..=(at + len - 1) / PIECE) {
            if p.pending > 0 {
                taken.push((p.writer, p.pending));
                p.pending = 0;
                p.quiet = 0;
            }
        }
        for (w, n) in taken {
            self.counts.by[kind as usize] += n;
            self.counts.pair(w, kind as u8, n);
        }
    }

    /// Generations older than [`STALE_FRAMES`] with no consumer count as
    /// unread; pieces idle for long are forgotten.
    pub(crate) fn age(&mut self, frame: u64) {
        let mut gone = Vec::new();
        let mut unread = Vec::new();
        for (&page, p) in &mut self.pieces {
            if p.pending > 0 && p.frame + STALE_FRAMES <= frame {
                unread.push((p.writer, p.pending));
                p.pending = 0;
                p.quiet = p.quiet.saturating_add(1);
            }
            if p.pending == 0 && p.frame + 20 * STALE_FRAMES <= frame && p.hold_until <= frame {
                gone.push(page);
            }
        }
        for (w, n) in unread {
            self.counts.nothing += n;
            self.counts.pair(w, NOTHING, n);
        }
        for page in gone {
            self.pieces.remove(&page);
        }
    }

    /// Whether the piece of `at` may be left on the card in `frame`.
    pub(crate) fn quiet_enough(&self, at: usize, frame: u64) -> bool {
        self.pieces.get(&(at / PIECE)).is_some_and(|p| p.quiet >= K && p.hold_until <= frame)
    }

    /// A demand met the piece of `at`: back to the old way for a while.
    /// Whether it was not held already.
    pub(crate) fn demote(&mut self, at: usize, frame: u64, hold: u64) -> bool {
        let p = self.pieces.entry(at / PIECE).or_default();
        p.quiet = 0;
        let was = p.hold_until > frame;
        p.hold_until = frame + hold;
        !was
    }
}

static TABLE: std::sync::LazyLock<RwLock<Table>> = std::sync::LazyLock::new(|| RwLock::new(Table::default()));
static GEN: AtomicU64 = AtomicU64::new(1);

fn table() -> std::sync::RwLockWriteGuard<'static, Table> {
    TABLE.write().unwrap_or_else(|e| e.into_inner())
}

/// A landing begins: its writes are a generation of their own.
pub(crate) fn landing() {
    GEN.fetch_add(1, Ordering::Relaxed);
}

/// From `readback::back`: the copy-back writes `n` bytes at `at`, `inside`
/// of them in the read-back set.
pub(crate) fn produced(at: usize, n: usize, inside: u64) {
    if !tracking() || IN_DEMAND.get() {
        return;
    }
    let (_, card, writer) = PRODUCER.get();
    table().produced(at, n, inside, card, writer, GEN.load(Ordering::Relaxed), crate::frame::current());
}

/// A layer read of title memory `[at, at + len)` is about to begin (from
/// `host::read_into` and `host::copy_mapped`): metered, and what is held
/// on the card there comes first.
pub(crate) fn reading(at: usize, len: usize) {
    if !tracking() || len == 0 || IN_DEMAND.get() {
        return;
    }
    let kind = KIND.get();
    if kind == Kind::Ignore {
        return;
    }
    touched(at, len, kind);
}

/// `kind` is about to read or change `[at, at + len)`.
pub(crate) fn touched(at: usize, len: usize, kind: Kind) {
    if !tracking() || len == 0 || IN_DEMAND.get() {
        return;
    }
    let meets = TABLE.read().unwrap_or_else(|e| e.into_inner()).meets(at, len);
    if meets {
        table().consumed(at, len, kind);
    }
    demand(at, len, kind);
}

// ---- stage 1 --------------------------------------------------------------

/// A piece left on the card: offsets `[lo, hi)` of `copy`, at title
/// address `title`, which held `t0` then; its 64 KiB pages at every
/// title address that shows them; ranges DirectStorage wrote since
/// (title addresses), where the title's bytes stand.
pub(crate) struct Skip {
    pub(crate) copy: Arc<Shadow>,
    pub(crate) lo: usize,
    pub(crate) hi: usize,
    pub(crate) title: usize,
    pub(crate) t0: Vec<u8>,
    pub(crate) pages: Vec<usize>,
    pub(crate) forced: Vec<(usize, usize)>,
    pub(crate) seq: u64,
}

/// The pieces left on the card.
#[derive(Default)]
pub(crate) struct Skips {
    by_seq: BTreeMap<u64, Skip>,
    by_page: BTreeMap<usize, Vec<u64>>,
    by_copy: HashMap<(usize, usize), u64>,
    pub(crate) bytes: u64,
    next: u64,
}

impl Skips {
    pub(crate) fn len(&self) -> usize {
        self.by_seq.len()
    }

    /// Whether `copy`'s piece at `lo` is held.
    pub(crate) fn holds(&self, copy: usize, lo: usize) -> bool {
        self.by_copy.contains_key(&(copy, lo))
    }

    pub(crate) fn hold(&mut self, mut s: Skip) {
        self.next += 1;
        s.seq = self.next;
        for &p in &s.pages {
            self.by_page.entry(p).or_default().push(s.seq);
        }
        self.by_copy.insert((Arc::as_ptr(&s.copy) as usize, s.lo), s.seq);
        self.bytes += s.t0.len() as u64;
        self.by_seq.insert(s.seq, s);
    }

    fn remove(&mut self, seq: u64) -> Option<Skip> {
        let s = self.by_seq.remove(&seq)?;
        for p in &s.pages {
            if let Some(v) = self.by_page.get_mut(p) {
                v.retain(|&q| q != seq);
                if v.is_empty() {
                    self.by_page.remove(p);
                }
            }
        }
        self.by_copy.remove(&(Arc::as_ptr(&s.copy) as usize, s.lo));
        self.bytes -= s.t0.len() as u64;
        Some(s)
    }

    /// The pieces held at any address of `[at, at + len)`, in the order
    /// they were left.
    fn meeting(&self, at: usize, len: usize) -> Vec<u64> {
        if len == 0 {
            return Vec::new();
        }
        let mut seqs: Vec<u64> = self.by_page.range(at / PIECE..=(at + len - 1) / PIECE).flat_map(|(_, v)| v.iter().copied()).collect();
        seqs.sort_unstable();
        seqs.dedup();
        seqs
    }

    /// Takes out the pieces held at any address of `[at, at + len)`.
    pub(crate) fn take(&mut self, at: usize, len: usize) -> Vec<Skip> {
        self.meeting(at, len).into_iter().filter_map(|q| self.remove(q)).collect()
    }

    /// Takes out `copy`'s pieces held over its offsets `[lo, hi)`.
    pub(crate) fn take_copy(&mut self, copy: usize, lo: usize, hi: usize) -> Vec<Skip> {
        let seqs: Vec<u64> = self
            .by_seq
            .values()
            .filter(|s| Arc::as_ptr(&s.copy) as usize == copy && s.lo < hi && s.hi > lo)
            .map(|s| s.seq)
            .collect();
        seqs.into_iter().filter_map(|q| self.remove(q)).collect()
    }

    /// DirectStorage wrote `[at, at + len)` (a title address): the title's
    /// bytes stand there, in every piece held over it.
    pub(crate) fn forced(&mut self, at: usize, len: usize) {
        for q in self.meeting(at, len) {
            if let Some(s) = self.by_seq.get_mut(&q) {
                s.forced.push((at, at + len));
            }
        }
    }
}

/// The title's bytes as they stand when a held piece goes in: `merged`
/// (at title address `title`) is what the landing's merge made of title
/// memory `now`; where `now` no longer holds `t0` (the title wrote since
/// the piece was left; `t0` from title address `t0_at`), or a forced range
/// covers it, `now` stands. Bytes outside `t0` are left as merged.
pub(crate) fn keep_title(title: usize, now: &[u8], merged: &mut [u8], t0_at: usize, t0: &[u8], forced: &[(usize, usize)]) {
    for (i, (m, &n)) in merged.iter_mut().zip(now).enumerate() {
        let a = title + i;
        let Some(&old) = a.checked_sub(t0_at).and_then(|j| t0.get(j)) else { continue };
        if old != n || forced.iter().any(|&(f, e)| a >= f && a < e) {
            *m = n;
        }
    }
}

static SKIPS: Mutex<Option<Skips>> = Mutex::new(None);
static ANY_SKIP: AtomicBool = AtomicBool::new(false);
/// One demand at a time.
static DEMAND: Mutex<()> = Mutex::new(());

fn skips() -> std::sync::MutexGuard<'static, Option<Skips>> {
    SKIPS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Since the last line: skipped bytes and pieces (new and again), demanded
/// bytes and pieces, demands and their microseconds summed and most,
/// pieces demoted, demands left to the pre-demand (a reference lock held),
/// fetches that failed.
static S: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
const SKIPPED_B: usize = 0;
const SKIPPED_N: usize = 1;
const DEMANDED_B: usize = 2;
const DEMANDED_N: usize = 3;
const DEMANDS: usize = 4;
const DEMAND_US: usize = 5;
const DEMAND_MAX: usize = 6;
const DEMOTED: usize = 7;
const HELD_LEFT: usize = 8;
const FAILED: usize = 9;

/// Notes the merges of demands could not make where they ran (the
/// journal, placed textures): (title address, bytes, other page numbers).
static NOTES: Mutex<Vec<(usize, usize, Vec<usize>)>> = Mutex::new(Vec::new());

/// A demand stored `n` bytes at `at` through our own view; `shown`: the
/// page's addresses in every title view.
pub(crate) fn note_later(at: usize, n: usize, shown: Vec<usize>) {
    NOTES.lock().unwrap_or_else(|e| e.into_inner()).push((at, n, shown));
}

/// Where the landing thread runs (no lock of ours held): the notes of
/// demands go in.
pub(crate) fn drain_notes() {
    let notes = std::mem::take(&mut *NOTES.lock().unwrap_or_else(|e| e.into_inner()));
    for (at, n, shown) in notes {
        let page = at / PIECE;
        let others: Vec<usize> = shown.iter().map(|&a| a / PIECE).filter(|&p| p != page).collect();
        crate::watch::wrote_own(page, &others);
        crate::descriptor::host::remapped(at, n);
    }
}

/// The landing is about to fetch `copy`'s piece `[lo, hi)`: whether it is
/// left on the card instead ([`skip_on`]; see the file's header). On the
/// landing thread.
pub(crate) fn try_skip(copy: &Arc<Shadow>, lo: usize, hi: usize) -> bool {
    if !skip_on() || lo >= hi {
        return false;
    }
    let title = copy.title_start() as usize + lo;
    let n = hi - lo;
    let key = Arc::as_ptr(copy) as usize;
    let frame = crate::frame::current();
    let again = skips().as_ref().is_some_and(|s| s.holds(key, lo));
    if !again {
        if title / PIECE != (title + n - 1) / PIECE || !TABLE.read().unwrap_or_else(|e| e.into_inner()).quiet_enough(title, frame) {
            return false;
        }
        if skips().as_ref().map_or(0, |s| s.bytes) + n as u64 > skip_most() {
            return false;
        }
        let Some(shown) = crate::inplace::shown_at(title, n) else { return false };
        let set = super::SET.read().unwrap_or_else(|e| e.into_inner());
        let off = title % PIECE;
        if set.inside(title as u64, n as u64) != 0
            || shown.iter().any(|&a| set.inside((a - a % PIECE + off) as u64, n as u64) != 0)
        {
            return false;
        }
        drop(set);
        if !crate::watch::armed(title, n) || shown.iter().any(|&a| !crate::watch::armed(a - a % PIECE + off, n)) {
            return false;
        }
        let mut t0 = vec![0u8; n];
        {
            let _s = scope(Kind::Ignore);
            if !crate::host::read_into(title as *const u8, &mut t0) {
                return false;
            }
        }
        let mut pages: Vec<usize> = shown.iter().map(|&a| a / PIECE).collect();
        pages.push(title / PIECE);
        pages.sort_unstable();
        pages.dedup();
        {
            let mut s = skips();
            s.get_or_insert_with(Skips::default).hold(Skip { copy: copy.clone(), lo, hi, title, t0, pages, forced: Vec::new(), seq: 0 });
        }
        ANY_SKIP.store(true, Ordering::Release);
        // As the write would have told them: every other copy of the page
        // takes it again (and so meets the piece), the writer's stays.
        copy.note_skipped(title, n, &shown);
    }
    {
        let _p = producing(key, true);
        produced(title, n, 0);
    }
    S[SKIPPED_B].fetch_add(n as u64, Ordering::Relaxed);
    S[SKIPPED_N].fetch_add(1, Ordering::Relaxed);
    drain_notes();
    true
}

/// The landing merges `copy`'s offsets `[lo, hi)` fetched now: what is
/// held there is the merge's (`t0` and forced ranges), taken out.
pub(crate) fn take_kept(copy: usize, lo: usize, hi: usize) -> Vec<Skip> {
    if !ANY_SKIP.load(Ordering::Acquire) {
        return Vec::new();
    }
    let mut s = skips();
    let Some(s) = s.as_mut() else { return Vec::new() };
    let taken = s.take_copy(copy, lo, hi);
    ANY_SKIP.store(s.len() > 0, Ordering::Release);
    taken
}

/// Holds `skip` as [`try_skip`] would (tests).
#[cfg(test)]
pub(crate) fn hold_for_tests(skip: Skip) {
    skips().get_or_insert_with(Skips::default).hold(skip);
    ANY_SKIP.store(true, Ordering::Release);
}

/// Before `kind` reads or changes title memory `[at, at + len)` (at any
/// of its addresses): the pieces held there are fetched from the card and
/// go into title memory first, and are held to the old way for a while.
pub(crate) fn demand(at: usize, len: usize, kind: Kind) {
    if !ANY_SKIP.load(Ordering::Acquire) || len == 0 || IN_DEMAND.get() {
        return;
    }
    if SIDE_HELD.get() > 0 {
        S[HELD_LEFT].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let _one = DEMAND.lock().unwrap_or_else(|e| e.into_inner());
    let taken = {
        let mut s = skips();
        let Some(s) = s.as_mut() else { return };
        let taken = s.take(at, len);
        ANY_SKIP.store(s.len() > 0, Ordering::Release);
        taken
    };
    if taken.is_empty() {
        return;
    }
    let started = std::time::Instant::now();
    IN_DEMAND.set(true);
    let pieces: Vec<(Arc<Shadow>, usize, usize)> = taken.iter().map(|s| (s.copy.clone(), s.lo, s.hi)).collect();
    match crate::host::demand_fetch(&pieces) {
        Some((fetched, after)) => {
            let mut title = Vec::new();
            for (s, gpu) in taken.iter().zip(&fetched) {
                s.copy.card_merge_relay_with(s.lo, gpu, after, &mut title, std::slice::from_ref(s));
            }
        }
        None => {
            S[FAILED].fetch_add(1, Ordering::Relaxed);
            tracing::error!(pieces = taken.len(), "copy-back: what was left on the card could not be fetched; title memory keeps older bytes there (VICTORY_BACK_SKIP_UNREAD=1)");
        }
    }
    IN_DEMAND.set(false);
    let us = started.elapsed().as_micros() as u64;
    let frame = crate::frame::current();
    let mut demoted = 0;
    {
        let mut t = table();
        for s in &taken {
            demoted += u64::from(t.demote(s.title, frame, hold_frames()));
        }
    }
    let bytes: u64 = taken.iter().map(|s| s.t0.len() as u64).sum();
    S[DEMANDED_B].fetch_add(bytes, Ordering::Relaxed);
    S[DEMANDED_N].fetch_add(taken.len() as u64, Ordering::Relaxed);
    S[DEMANDS].fetch_add(1, Ordering::Relaxed);
    S[DEMAND_US].fetch_add(us, Ordering::Relaxed);
    S[DEMAND_MAX].fetch_max(us, Ordering::Relaxed);
    S[DEMOTED].fetch_add(demoted, Ordering::Relaxed);
    let n = DEMAND_KINDS[(kind as usize).min(KINDS - 1)].fetch_add(1, Ordering::Relaxed);
    let _ = n;
}

/// Demands by the kind that asked, since the last line.
static DEMAND_KINDS: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];

/// DirectStorage's bytes landed at `[at, at + len)` (`watch::title_memory_written`).
pub(crate) fn title_wrote(at: usize, len: usize) {
    if !ANY_SKIP.load(Ordering::Acquire) || len == 0 {
        return;
    }
    if let Some(s) = skips().as_mut() {
        s.forced(at, len);
    }
}

/// The line, every 5 s (from `readback::line`).
pub(crate) fn line() {
    if !tracking() {
        return;
    }
    let frame = crate::frame::current();
    let (c, held) = {
        let mut t = table();
        t.age(frame);
        let held = t.pieces.len();
        (std::mem::take(&mut t.counts), held)
    };
    let mb = |v: u64| v as f64 / (1 << 20) as f64;
    let by = |k: Kind| c.by[k as usize];
    let copies = by(Kind::Window) + by(Kind::OwnView) + by(Kind::Plain) + by(Kind::Texture);
    let cpu = by(Kind::Descriptor) + by(Kind::Records) + by(Kind::Other);
    let title_side = by(Kind::TitleLock) + by(Kind::TitleWrite) + by(Kind::Remap) + by(Kind::LayerWrite);
    let mut pairs: Vec<((u64, u8), u64)> = c.pairs.into_iter().filter(|p| p.1 > 0).collect();
    pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    let pairs: Vec<String> = pairs
        .iter()
        .take(8)
        .map(|((w, k), n)| {
            let to = if *k == NOTHING { "nothing" } else { NAMES[usize::from(*k)] };
            let from = if *w == 0 { "(writer not known)".to_owned() } else { format!("{w:016x}") };
            format!("{from} -> {to} {:.1} MB", mb(*n))
        })
        .collect();
    let all = c.back.max(1);
    let s: Vec<u64> = S.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect();
    let kinds: Vec<String> = DEMAND_KINDS
        .iter()
        .enumerate()
        .filter_map(|(i, a)| {
            let n = a.swap(0, Ordering::Relaxed);
            (n > 0).then(|| format!("{} {n}", NAMES[i]))
        })
        .collect();
    let (holding, held_mb) = skips().as_ref().map_or((0, 0), |s| (s.len(), s.bytes));
    tracing::info!(
        "copy-back consumers (VICTORY_BACK_CONSUMERS; 64 KiB pieces of title memory, one generation per landing, the first consumer of a generation takes all of its bytes, a generation unread for {STALE_FRAMES} frames or written again counts as nothing): copied back {:.1} MB in {} ranges ({:.1} MB of copies in the card's memory); later consumed by: title read-back {:.1} MB, other card copies {:.1} MB (windows {:.1}, own-view {:.1}, plain {:.1}, texture sources {:.1}), layer hashing {:.1} MB, layer CPU reads {:.1} MB (descriptor records {:.1}, draw/indirect records {:.1}, other {:.1}), title locks/writes, remaps and layer writes {:.1} MB, nothing {:.1} MB ({:.1} %); top 8 producer->consumer pairs by MB: [{}]; pieces tracked {held}",
        mb(c.back),
        c.ranges,
        mb(c.card),
        mb(by(Kind::Title)),
        mb(copies),
        mb(by(Kind::Window)),
        mb(by(Kind::OwnView)),
        mb(by(Kind::Plain)),
        mb(by(Kind::Texture)),
        mb(by(Kind::Hashing)),
        mb(cpu),
        mb(by(Kind::Descriptor)),
        mb(by(Kind::Records)),
        mb(by(Kind::Other)),
        mb(title_side),
        mb(c.nothing),
        c.nothing as f64 * 100.0 / all as f64,
        pairs.join(", "),
    );
    if skip_on() {
        let demands = s[DEMANDS].max(1);
        tracing::info!(
            "copy-back skip (VICTORY_BACK_SKIP_UNREAD=1, K={K}): skipped {:.1} MB in {} ranges (fetch and write); demanded later {:.1} MB in {} ranges ({} demands, sync fetch mean {:.2} / max {:.2} ms; asked by [{}]); demoted {} ranges; held on the card now {holding} ranges {:.1} MB; left to the pre-demand {}; failed fetches {}",
            mb(s[SKIPPED_B]),
            s[SKIPPED_N],
            mb(s[DEMANDED_B]),
            s[DEMANDED_N],
            s[DEMANDS],
            s[DEMAND_US] as f64 / demands as f64 / 1000.0,
            s[DEMAND_MAX] as f64 / 1000.0,
            kinds.join(", "),
            s[DEMOTED],
            mb(held_mb),
            s[HELD_LEFT],
            s[FAILED],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: usize = PIECE;

    #[test]
    fn each_write_goes_to_its_first_consumer_or_to_nothing() {
        let mut t = Table::default();
        // Generation 1: two stretches of one piece, and one inside the set.
        t.produced(10 * P + 16, 100, 0, true, 0xa, 1, 5);
        t.produced(10 * P + 4096, 50, 0, true, 0xa, 1, 5);
        t.produced(11 * P, 40, 40, false, 0xb, 1, 5);
        assert!(t.meets(10 * P, 1));
        assert!(!t.meets(11 * P, 40), "set bytes are consumed when written");
        // A window reads part of the piece: all 150 bytes are its, once.
        t.consumed(10 * P + 8000, 4, Kind::Window);
        t.consumed(10 * P, P, Kind::Texture);
        assert_eq!(t.counts.by[Kind::Window as usize], 150);
        assert_eq!(t.counts.by[Kind::Texture as usize], 0);
        assert_eq!(t.counts.by[Kind::Title as usize], 40);
        assert_eq!(t.counts.pairs[&(0xa, Kind::Window as u8)], 150);
        // Generation 2 unread, generation 3 overwrites it: nothing.
        t.produced(10 * P, 64, 0, true, 0xa, 2, 6);
        t.produced(10 * P, 64, 0, true, 0xa, 3, 7);
        assert_eq!(t.counts.nothing, 64);
        assert_eq!(t.pieces[&10].quiet, 1);
        // Aging: generation 3 unread for long.
        t.age(7 + STALE_FRAMES);
        assert_eq!(t.counts.nothing, 128);
        assert_eq!(t.pieces[&10].quiet, 2);
        // Every byte copied back is in exactly one bucket.
        let total: u64 = t.counts.by.iter().sum::<u64>() + t.counts.nothing;
        assert_eq!(total, t.counts.back);
        assert_eq!(t.counts.card, 100 + 50 + 64 + 64);
    }

    #[test]
    fn a_write_across_pieces_is_split_and_set_bytes_come_off_the_first() {
        let mut t = Table::default();
        t.produced(3 * P - 10, 30, 4, false, 1, 1, 1);
        assert_eq!(t.pieces[&2].pending, 6);
        assert_eq!(t.pieces[&3].pending, 20);
        assert_eq!(t.counts.by[Kind::Title as usize], 4);
    }

    #[test]
    fn quiet_pieces_may_skip_until_a_demand_demotes_them() {
        let mut t = Table::default();
        for g in 1..=K as u64 + 1 {
            t.produced(5 * P, 8, 0, true, 1, g, g);
        }
        assert_eq!(t.pieces[&5].quiet, K);
        assert!(t.quiet_enough(5 * P + 100, 20));
        assert!(t.demote(5 * P, 20, 300));
        assert!(!t.demote(5 * P, 21, 300), "held already");
        assert!(!t.quiet_enough(5 * P, 21));
        // Quiet again, but held until frame 321.
        for g in 20..=40u64 {
            t.produced(5 * P, 8, 0, true, 1, g, g);
        }
        assert!(!t.quiet_enough(5 * P, 300));
        assert!(t.quiet_enough(5 * P, 321));
        // A consumer resets it.
        t.consumed(5 * P, 1, Kind::Plain);
        assert!(!t.quiet_enough(5 * P, 400));
    }

    #[test]
    fn the_title_s_newer_bytes_stand_over_a_held_piece() {
        // Left at 0x1000 with title bytes [1, 2, 3, 4]; the title wrote
        // the second byte since (2 -> 9); DirectStorage the fourth (4 -> 4,
        // the same value: only the forced range tells).
        let t0 = [1u8, 2, 3, 4];
        let now = [1u8, 9, 3, 4];
        let mut merged = [7u8, 7, 3, 7];
        keep_title(0x1000, &now, &mut merged, 0x1000, &t0, &[(0x1003, 0x1004)]);
        assert_eq!(merged, [7, 9, 3, 4]);
        // Bytes outside t0 stay as merged.
        let mut merged = [7u8, 7];
        keep_title(0x0fff, &[5, 1], &mut merged, 0x1000, &t0, &[]);
        assert_eq!(merged, [7, 7]);
    }
}
