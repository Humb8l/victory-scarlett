//! Finding F4: where the terrain virtual texture's feedback is lost
//! (`VICTORY_TERRAIN_FEEDBACK=1`, off by default): a line every ~2 s.
//!
//! The title's per-frame update 0x140fbf540 binds a 240x135 R32_UINT
//! feedback target that terrain draws write, compacts it with three
//! dispatches (((w>>3)+1)^2 groups with w = u32 at 0x1487b3358, then
//! 31x17, then again) into the ring's write buffer (0x200 u32, count in
//! element 0x1ff), marks it written (0x1453f1950), and two frames later
//! reads it (0x14524dab0) into 0x1487b2b08; the count lands at 0x1487b3304.
//!
//! Lines:
//! * `terrain feedback: enabled E, count C (max seen M), pages touched P,
//!   build list B, marked written W; ...` (title memory, the falsifier);
//! * `terrain feedback ring: ...`: the stages on the layer's side for the
//!   ring's three buffers: ops whose writable views cover them, the views
//!   without a host copy (null: writes dropped), the ranges left out as
//!   "the shader cannot store", declarations for copy-back, the copy that
//!   holds them, the bytes compared / fetched from the card / landed, and
//!   the non-zero bytes of the copy when compared (what the GPU left);
//! * `terrain feedback dispatches: ...`: dispatches of the compaction's
//!   shapes and what became of them (issued, or dropped and why).
//!
//! note: the 25 pages a frame budget (0x19, cmp near 0x140fbebd0 in
//! 0x140fbeba0) is not patched here; a patch would go in
//! `victory_hle::readback` beside the F5 stream waits.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use super::{Buffer, title_u32};

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_TERRAIN_FEEDBACK").is_ok_and(|v| v.trim() == "1"))
}

/// The ring and its buffers: (index, CPU address, GPU address, bytes).
static RING: AtomicU64 = AtomicU64::new(0);
static BUFFERS: Mutex<Vec<(usize, u64, u64, u64)>> = Mutex::new(Vec::new());
static ANY: AtomicBool = AtomicBool::new(false);
static OP: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// The op (draw or dispatch) being bound on this thread: its number
    /// and pipeline name, and whether it is a dispatch.
    static CURRENT: Cell<u64> = const { Cell::new(0) };
    static NAME: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
    static GROUPS: Cell<[u32; 3]> = const { Cell::new([0; 3]) };
}

#[derive(Default)]
struct Period {
    ops: u64,
    last_op: u64,
    first_ops: Vec<String>,
    uav_views: u64,
    uav_no_copy: u64,
    uav_no_store: u64,
    declared: u64,
    declared_bytes: u64,
    declared_in_place: u64,
    declared_by: [u64; 3],
    heap_listed: u64,
    heap_marked: u64,
    compared: u64,
    compared_nonzero: u64,
    fetched: u64,
    fetched_nonzero: u64,
    landed: u64,
    landed_nonzero: u64,
    landed_by: [u64; 3],
    marked: u64,
    reads: u64,
    read_count_nonzero: u64,
    read_last_count: u32,
    frames: u64,
    frames_count_nonzero: u64,
    count_max: u32,
    /// Per compaction shape (0: ((w>>3)+1)^2, 1: 31x17): seen, issued,
    /// dropped by reason.
    shapes: [[u64; 2]; 2],
    dropped: Vec<(u8, &'static str, u64)>,
    views_refused: Vec<(&'static str, u64)>,
}

static PERIOD: Mutex<Option<Period>> = Mutex::new(None);
static COUNT_MAX: AtomicU64 = AtomicU64::new(0);
static WIDTH: AtomicU64 = AtomicU64::new(0);

fn with<R>(f: impl FnOnce(&mut Period) -> R) -> R {
    let mut p = PERIOD.lock().unwrap_or_else(|e| e.into_inner());
    f(p.get_or_insert_with(Period::default))
}

pub(crate) fn ring_known(ring: u64, made: &[Buffer]) {
    if !on() {
        return;
    }
    RING.store(ring, Ordering::Relaxed);
    let mut b = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
    *b = made.iter().map(|m| (usize::from(m.index.min(2)), m.cpu, m.gpu, m.bytes)).collect();
    ANY.store(!b.is_empty(), Ordering::Relaxed);
}

/// The ring's buffer holding bytes of `[at, at + len)` (by either address),
/// with the bytes it holds.
fn hit(at: u64, len: u64) -> Option<(usize, u64)> {
    if !ANY.load(Ordering::Relaxed) {
        return None;
    }
    let end = at.saturating_add(len.max(1));
    let b = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
    for &(i, cpu, gpu, bytes) in b.iter() {
        for f in [cpu, gpu] {
            let (a, e) = (at.max(f), end.min(f + bytes));
            if a < e {
                return Some((i, e - a));
            }
        }
    }
    None
}

/// The op the current one is: counted once as an op writing the ring.
fn count_op(p: &mut Period) {
    let op = CURRENT.get();
    if op != 0 && p.last_op == op {
        return;
    }
    p.last_op = op;
    p.ops += 1;
    if p.first_ops.len() < 4 {
        let name = NAME.with(|n| n.borrow().clone());
        let g = GROUPS.get();
        p.first_ops.push(if g == [0; 3] { name } else { format!("{name} {}x{}x{}", g[0], g[1], g[2]) });
    }
}

/// `State::bind` starts binding an op of pipeline `name`.
pub(crate) fn op(name: &str, compute: bool) {
    if !on() {
        return;
    }
    CURRENT.set(OP.fetch_add(1, Ordering::Relaxed) + 1);
    NAME.with(|n| {
        let mut n = n.borrow_mut();
        n.clear();
        n.push_str(name);
    });
    if !compute {
        GROUPS.set([0; 3]);
    }
}

/// A writable buffer view of `[at, at + len)` is bound: `copy` whether it
/// has a host copy, `store` whether it is declared written (false: the
/// shader cannot store, `VICTORY_WRITE_SKIP`).
pub(crate) fn uav(at: u64, len: u64, copy: bool, store: bool) {
    if !on() || hit(at, len).is_none() {
        return;
    }
    with(|p| {
        count_op(p);
        p.uav_views += 1;
        p.uav_no_copy += u64::from(!copy);
        p.uav_no_store += u64::from(copy && !store);
    });
}

/// A buffer UAV over the ring was refused by the view maker.
pub(crate) fn view_refused(at: u64, len: u64, why: &'static str) {
    if !on() || hit(at, len).is_none() {
        return;
    }
    with(|p| match p.views_refused.iter_mut().find(|v| v.0 == why) {
        Some(v) => v.1 += 1,
        None => p.views_refused.push((why, 1)),
    });
}

/// `Shadow::gpu_writes`: `[at, at + len)` declared written (`in_place`:
/// the copy is title memory itself, nothing is copied back).
pub(crate) fn declared(at: u64, len: u64, in_place: bool) {
    if !on() {
        return;
    }
    let Some((i, n)) = hit(at, len) else { return };
    with(|p| {
        count_op(p);
        p.declared += 1;
        p.declared_bytes += n;
        p.declared_in_place += u64::from(in_place);
        p.declared_by[i] += 1;
    });
}

/// Index-picked unordered views of the heap over the ring: listed, and
/// marked written by shaders.
pub(crate) fn heap_view(at: u64, len: u64, marked: bool) {
    if !on() || hit(at, len).is_none() {
        return;
    }
    with(|p| {
        p.heap_listed += 1;
        p.heap_marked += u64::from(marked);
    });
}

/// The copy-back compares `copy` (the host copy's bytes at title address
/// `at`, what the GPU left) with its snapshot.
pub(crate) fn compared(at: u64, copy: &[u8]) {
    if !on() || hit(at, copy.len() as u64).is_none() {
        return;
    }
    let (nonzero, n) = ring_bytes(at, copy);
    with(|p| {
        p.compared += n;
        p.compared_nonzero += nonzero;
    });
}

/// The bytes of `data` (at `at`) inside the ring: non-zero, all.
fn ring_bytes(at: u64, data: &[u8]) -> (u64, u64) {
    let b = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
    let (mut nonzero, mut n) = (0, 0);
    let end = at + data.len() as u64;
    for &(_, cpu, gpu, bytes) in b.iter() {
        for f in if cpu == gpu { vec![cpu] } else { vec![cpu, gpu] } {
            let (a, e) = (at.max(f), end.min(f + bytes));
            if a < e {
                let part = &data[(a - at) as usize..(e - at) as usize];
                n += part.len() as u64;
                nonzero += part.iter().filter(|&&x| x != 0).count() as u64;
            }
        }
    }
    (nonzero, n)
}

/// A card copy's bytes at `at` came back (a piece found differing).
pub(crate) fn fetched(at: u64, gpu: &[u8]) {
    if !on() || hit(at, gpu.len() as u64).is_none() {
        return;
    }
    let (nonzero, n) = ring_bytes(at, gpu);
    with(|p| {
        p.fetched += n;
        p.fetched_nonzero += nonzero;
    });
}

/// The copy-back puts `bytes` into title memory at `at`.
pub(crate) fn landed(at: u64, bytes: &[u8]) {
    if !on() {
        return;
    }
    let Some((i, _)) = hit(at, bytes.len() as u64) else { return };
    let (nonzero, n) = ring_bytes(at, bytes);
    with(|p| {
        p.landed += n;
        p.landed_nonzero += nonzero;
        p.landed_by[i] += n;
    });
}

/// "Mark written" for `ring`: once a frame for the terrain ring; the
/// count the consumer last took is sampled then.
pub(crate) fn marked(ring: u64) {
    if !on() || ring != RING.load(Ordering::Relaxed) {
        return;
    }
    let count = title_u32(0x1487b3304).unwrap_or(0);
    COUNT_MAX.fetch_max(u64::from(count), Ordering::Relaxed);
    with(|p| {
        p.marked += 1;
        p.frames += 1;
        p.frames_count_nonzero += u64::from(count != 0);
        p.count_max = p.count_max.max(count);
    });
}

/// A successful ring read of buffer `b` holding `content`.
pub(crate) fn read(ring: u64, _b: &Buffer, content: &[u8]) {
    if !on() || ring != RING.load(Ordering::Relaxed) || content.len() < 0x800 {
        return;
    }
    let count = u32::from_le_bytes(content[0x7fc..0x800].try_into().unwrap_or([0; 4]));
    with(|p| {
        p.reads += 1;
        p.read_count_nonzero += u64::from(count != 0);
        p.read_last_count = count;
    });
}

/// `State::dispatch` of `x`x`y`x`z` groups: which compaction shape it has
/// (0, 1), if any; remembered for the ops line.
pub(crate) fn dispatch_begin(x: u32, y: u32, z: u32) -> Option<u8> {
    if !on() {
        return None;
    }
    GROUPS.set([x, y, z]);
    let w = WIDTH.load(Ordering::Relaxed) as u32;
    let g = (w >> 3) + 1;
    let shape = if z == 1 && w != 0 && ((x == g && y == g) || (x == g * g && y == 1)) {
        Some(0)
    } else if (x, y, z) == (31, 17, 1) {
        Some(1)
    } else {
        None
    };
    if let Some(s) = shape {
        with(|p| p.shapes[usize::from(s)][0] += 1);
    }
    shape
}

/// What became of a dispatch [`dispatch_begin`] matched: issued (`why`
/// empty) or dropped and why.
pub(crate) fn dispatch_end(shape: Option<u8>, why: &'static str) {
    let Some(s) = shape else { return };
    with(|p| {
        if why.is_empty() {
            p.shapes[usize::from(s)][1] += 1;
        } else {
            match p.dropped.iter_mut().find(|d| d.0 == s && d.1 == why) {
                Some(d) => d.2 += 1,
                None => p.dropped.push((s, why, 1)),
            }
        }
    });
}

pub(super) fn line() {
    static LAST: AtomicU64 = AtomicU64::new(0);
    if !on() {
        return;
    }
    let now = crate::host::now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 2000 || LAST.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    WIDTH.store(u64::from(title_u32(0x1487b3358).unwrap_or(0)), Ordering::Relaxed);
    let p = PERIOD.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
    let enabled = title_u32(0x1487ba1a0).map_or(-1, |v| i64::from(v & 0xff));
    let count = title_u32(0x1487b3304).unwrap_or(0);
    COUNT_MAX.fetch_max(u64::from(count), Ordering::Relaxed);
    tracing::info!(
        "terrain feedback: enabled {enabled}, count {count} (max seen {}), pages touched {}, build list {}, marked written {}; frames with a non-zero count {} of {} (max this period {}); successful ring reads {} (count non-zero in {}, last count {}); w {}",
        COUNT_MAX.load(Ordering::Relaxed),
        title_u32(0x1487b2a88).unwrap_or(0),
        title_u32(0x1487b29bc).unwrap_or(0),
        p.marked,
        p.frames_count_nonzero,
        p.frames,
        p.count_max,
        p.reads,
        p.read_count_nonzero,
        p.read_last_count,
        WIDTH.load(Ordering::Relaxed),
    );
    let buffers = BUFFERS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let copies: Vec<String> = buffers
        .iter()
        .map(|&(i, cpu, gpu, bytes)| {
            let copy = crate::shadow::find(cpu, bytes)
                .or_else(|| crate::shadow::find(gpu, bytes))
                .map_or("no host copy".to_owned(), |s| s.describe());
            format!("{i}: {cpu:#x} ({bytes} B) in {copy}")
        })
        .collect();
    tracing::info!(
        "terrain feedback ring: dispatches writing the ring: {} (first: [{}]); writable views over it bound {} times, {} without a host copy (null view: writes dropped), {} left out as the shader cannot store (VICTORY_WRITE_SKIP); views refused by the view maker {:?}; index-picked views over it listed {}, marked written {}; declared written for copy-back {} times ({} bytes, per buffer {:?}, {} in place: no copy-back); copy-back compared {} bytes ({} non-zero in the copy), fetched from the card {} bytes ({} non-zero), landed {} bytes ({} non-zero, per buffer {:?}); copies: [{}]",
        p.ops,
        p.first_ops.join(", "),
        p.uav_views,
        p.uav_no_copy,
        p.uav_no_store,
        p.views_refused,
        p.heap_listed,
        p.heap_marked,
        p.declared,
        p.declared_bytes,
        p.declared_by,
        p.declared_in_place,
        p.compared,
        p.compared_nonzero,
        p.fetched,
        p.fetched_nonzero,
        p.landed,
        p.landed_nonzero,
        p.landed_by,
        copies.join("; ")
    );
    let dropped: Vec<String> = p.dropped.iter().map(|(s, why, n)| format!("shape {s}: {why} {n}")).collect();
    tracing::info!(
        "terrain feedback dispatches: shape 0 (((w>>3)+1)^2 groups) seen {} issued {}; shape 1 (31x17) seen {} issued {}; dropped: [{}]",
        p.shapes[0][0],
        p.shapes[0][1],
        p.shapes[1][0],
        p.shapes[1][1],
        dropped.join(", ")
    );
}
