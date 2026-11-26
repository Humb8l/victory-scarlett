//! What the title does with read-back memory (finding F1, iteration 2):
//! its buffer locks (`VICTORY_READBACK_LOCKS=1`) and the feedback health of
//! each read-back ring.
//!
//! The engine's generic lock (0x145121550) takes a multi-frame handle and
//! a record; it picks the handle's buffer for this frame, sets the record's
//! byte at +0x3c to 2 whatever the caller wants, and maps nothing for a
//! buffer without a resource: the record's CPU pointer (+0) and bytes
//! (u32 at +0x18) name raw title memory. The caller's intent is not an
//! argument, so the mode is measured: the locked bytes are summed at the
//! lock and again at the matching unlock (0x144e8abe0, same record): a
//! lock whose bytes changed in between is one the CPU wrote through.
//!
//! What shaders wrote is known exactly as the copy-back puts it into title
//! memory ([`written`]): byte runs where the GPU's bytes differ from the
//! snapshot (copies in the PC's memory), first to last differing byte of a
//! 64 KiB piece (copies in the card's memory and the relay). It is kept by
//! range with the layer frame it landed in, for the last [`RECENT`] frames.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{ALIAS, Buffer, SET, locks_meter, only_read, read_u64, settle};

/// Frames a landed shader write stays in [`WRITTEN`].
const RECENT: u64 = 8;
/// The first locks logged with their raw arguments.
const SAY_ARGS: u64 = 200;
/// Locked bytes summed at most (a bigger lock is summed over its first ones).
const SUM_MOST: u64 = 1 << 20;

/// A checksum that shows a change of content (FNV-1a, 32 bits).
pub(crate) fn sum(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811c_9dc5u32, |h, &b| (h ^ u32::from(b)).wrapping_mul(0x0100_0193))
}

fn read(at: u64, n: u64) -> Option<Vec<u8>> {
    let mut v = vec![0u8; n.min(SUM_MOST) as usize];
    crate::host::read_into(at as *const u8, &mut v).then_some(v)
}

/// Ranges shaders wrote as the copy-back landed them: start to (end, layer
/// frame), disjoint; a later landing replaces what it covers.
#[derive(Default)]
pub(crate) struct Written(BTreeMap<u64, (u64, u64)>);

impl Written {
    pub(crate) fn mark(&mut self, from: u64, to: u64, frame: u64) {
        if from >= to {
            return;
        }
        let meeting: Vec<u64> = self.0.range(..to).rev().take_while(|(_, v)| v.0 > from).map(|(&a, _)| a).collect();
        for a in meeting {
            let (e, f) = self.0.remove(&a).unwrap_or((a, 0));
            if a < from {
                self.0.insert(a, (from, f));
            }
            if e > to {
                self.0.insert(to, (e, f));
            }
        }
        self.0.insert(from, (to, frame));
    }

    /// Bytes of `[from, to)` landed in frame `since` or later.
    pub(crate) fn recent(&self, from: u64, to: u64, since: u64) -> u64 {
        let mut n = 0;
        for (&a, &(e, f)) in self.0.range(..to).rev() {
            if e <= from {
                break;
            }
            if f >= since {
                n += e.min(to) - a.max(from);
            }
        }
        n
    }

    fn forget_before(&mut self, frame: u64) {
        self.0.retain(|_, v| v.1 >= frame);
    }
}

static WRITTEN: Mutex<Written> = Mutex::new(Written(BTreeMap::new()));

/// The copy-back wrote `[at, at + n)` (some of it outside the set).
pub(super) fn written(at: u64, n: u64) {
    let mut w = WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    w.mark(at, at + n, crate::frame::current());
    if w.0.len() > 1 << 21 {
        w.0.clear();
        tracing::warn!("read-back locks: the record of shader-written ranges grew past 2M ranges and was cleared");
    }
}

/// One read-back ring's feedback: since the last line, and since the start.
#[derive(Default)]
struct Ring {
    name: &'static str,
    site: u64,
    // Since the last line.
    landed: u64,
    last_frame: u64,
    bytes: u64,
    nonzero: u64,
    changed: u64,
    locks: u64,
    late: u64,
    zero: u64,
    latency: u64,
    latencies: u64,
    /// Ring reads (0x14524dab0): calls, succeeded, refused as too young,
    /// refused before any data (stamp never written), all-zero at a
    /// successful read; "mark written" calls (0x1453f1950). Per buffer:
    /// successful reads and marks.
    reads: u64,
    read_ok: u64,
    read_young: u64,
    read_nodata: u64,
    read_zero: u64,
    marked: u64,
    read_by: [u64; 3],
    marked_by: [u64; 3],
    landed_by: [u64; 3],
    // Since the start, per buffer: the sum of the last landing, and the
    // title frame whose GPU write it carried (-1: none yet).
    sums: [u32; 3],
    carried: [i64; 3],
}

static RINGS: Mutex<Option<HashMap<u64, Ring>>> = Mutex::new(None);

pub(super) fn ring(ring: u64, site: u64, name: &'static str) {
    let mut r = RINGS.lock().unwrap_or_else(|e| e.into_inner());
    r.get_or_insert_with(HashMap::new).insert(ring, Ring { name, site, carried: [-1; 3], ..Ring::default() });
}

/// The ring's state: the index the CPU reads (+0x4c) and the packed
/// write state (+0x50: bit 31 valid, bits 4-30 the title's frame, bits
/// 0-3 the index the GPU writes this frame). `None` if not valid.
fn state(ring: u64) -> Option<(u64, u64, i64)> {
    let v = read_u64(ring + 0x4c)?;
    let (read, packed) = (v & 0xffff_ffff, v >> 32);
    (packed >> 31 != 0).then_some((read, packed & 0xf, ((packed >> 4) & 0x7ff_ffff) as i64))
}

/// Frames since the GPU wrote buffer `index` of a ring of `n` that writes
/// buffer `writing` now (each frame the next one, in turn).
pub(crate) fn age(index: u64, writing: u64, n: u64) -> u64 {
    if n == 0 { 0 } else { (writing + n - index % n) % n }
}

/// The copy-back is about to land `bytes` at `at`; `over`: the read-back
/// buffers it meets and the parts of it they hold.
pub(super) fn landed(at: u64, bytes: &[u8], over: &[(Buffer, u64, u64)]) {
    let frame = crate::frame::current();
    let mut r = RINGS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(rings) = r.as_mut() else { return };
    for &(b, a, e) in over {
        let Some(ring) = rings.get_mut(&b.ring) else { continue };
        let part = &bytes[(a - at) as usize..(e - at) as usize];
        if ring.last_frame != frame {
            ring.landed += 1;
            ring.last_frame = frame;
        }
        ring.bytes += part.len() as u64;
        ring.landed_by[usize::from(b.index.min(2))] += part.len() as u64;
        ring.nonzero += part.iter().filter(|&&x| x != 0).count() as u64;
        let i = usize::from(b.index.min(2));
        let s = sum(part) ^ (a - b.cpu.min(a)) as u32;
        if s != ring.sums[i] {
            ring.changed += 1;
            ring.sums[i] = s;
        }
        if let Some((_, writing, title_frame)) = state(b.ring) {
            let late = age(u64::from(b.index), writing, u64::from(b.count));
            ring.latency += late;
            ring.latencies += 1;
            ring.carried[i] = title_frame - late as i64;
        }
    }
}

fn read_u32(at: u64) -> Option<u32> {
    let mut b = [0u8; 4];
    crate::host::read_into(at as *const u8, &mut b).then(|| u32::from_le_bytes(b))
}

/// The ring reader returned (`ok`): on success the record's +0 is the CPU
/// address of the buffer read; on refusal the stamp of the buffer it
/// looked at (+0x40 + 4 * (+0x4c mod n)) says why: bit 0 clear = never
/// marked written (read before data), else too young.
pub fn ring_read(ring: u64, record: u64, ok: bool) {
    let flags = read_u32(ring + 0x54).unwrap_or(0);
    let n = if flags & 8 != 0 { 1 } else { 3 };
    let (mut index, mut zero, mut nodata) = (None, false, false);
    if ok {
        let cpu = read_u64(record).unwrap_or(0);
        let found = SET.read().unwrap_or_else(|e| e.into_inner()).buffers.iter().find(|b| b.ring == ring && cpu >= b.cpu && cpu < b.cpu + b.bytes).copied();
        if let Some(b) = found {
            index = Some(usize::from(b.index.min(2)));
            let content = read(b.cpu, b.bytes).unwrap_or_default();
            zero = content.iter().all(|&x| x == 0);
            super::terrain::read(ring, &b, &content);
        }
    } else {
        let idx = read_u32(ring + 0x4c).unwrap_or(0) % n;
        nodata = read_u32(ring + 0x40 + 4 * u64::from(idx)).unwrap_or(0) & 1 == 0;
    }
    let mut r = RINGS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(ring) = r.as_mut().and_then(|r| r.get_mut(&ring)) else { return };
    ring.reads += 1;
    if ok {
        ring.read_ok += 1;
        ring.read_zero += u64::from(zero);
        if let Some(i) = index {
            ring.read_by[i] += 1;
        }
    } else if nodata {
        ring.read_nodata += 1;
    } else {
        ring.read_young += 1;
    }
}

/// "Mark written" was entered for `ring`: the buffer it marks is the one
/// its packed state (+0x50) names, or the next one if that state is of an
/// earlier frame (u32 at 0x14a9437ac), as the function decides.
pub fn ring_marked(ring: u64) {
    let packed = read_u32(ring + 0x50).unwrap_or(0);
    let flags = read_u32(ring + 0x54).unwrap_or(0);
    let n = if flags & 8 != 0 { 1 } else { 3 };
    let frame = super::title_u32(0x14a9437ac).unwrap_or(0);
    let index = if packed >> 31 != 0 && ((frame << 4) ^ packed) & 0x7fff_fff0 != 0 { ((packed + 1) & 0xf) % n } else { (packed & 0xf) % n };
    super::terrain::marked(ring);
    let mut r = RINGS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(ring) = r.as_mut().and_then(|r| r.get_mut(&ring)) else { return };
    ring.marked += 1;
    ring.marked_by[(index as usize).min(2)] += 1;
}

/// A lock under way: what it returned, and its bytes' sum then.
struct Open {
    handle: u64,
    cpu: u64,
    bytes: u64,
    sum: u32,
}

/// Per lock handle outside the set, since the last line.
#[derive(Default)]
struct Handle {
    caller: u64,
    stride: u64,
    count: u64,
    locks: u64,
    wrote: u64,
    kept: u64,
    overlap: u64,
    nonzero: u64,
    bytes: u64,
    changed: u64,
    last_sum: u32,
}

#[derive(Default)]
struct Locks {
    open: HashMap<u64, Open>,
    handles: HashMap<u64, Handle>,
    /// Since the last line: locks; unlocks seen; of them those whose bytes
    /// changed between lock and unlock; locks outside the set.
    n: [u64; 4],
}

static LOCKS: Mutex<Option<Locks>> = Mutex::new(None);
static SAID: AtomicU64 = AtomicU64::new(0);

/// The lock hook at the lock's entry: its handle (rcx), record (rdx), r8,
/// r9, and the caller's return address (as 0x14...). The first
/// [`SAY_ARGS`] are logged with the record's first 8 words.
pub fn lock_entered(handle: u64, record: u64, r8: u64, r9: u64, caller: u64) {
    if SAID.fetch_add(1, Ordering::Relaxed) >= SAY_ARGS {
        return;
    }
    let words: Vec<String> = (0..8).map(|i| read_u64(record + 8 * i).map_or("?".into(), |v| format!("{v:#x}"))).collect();
    let flags = read_u64(handle + 0x18).map_or(0, |v| v & 0xffff_ffff);
    tracing::info!(
        "lock args: caller={caller:#x} handle={handle:#x} handle+0x18={flags:#x} r8={r8:#x} r9={r9:#x} record={record:#x} [{}]",
        words.join(" ")
    );
}

/// The lock returned (`ok`: its `al`): the record holds the CPU address
/// (+0), bytes (u32 at +0x18) and the buffer object (+0x30).
pub fn lock_returned(handle: u64, record: u64, caller: u64, ok: bool) {
    if !ok || record == 0 {
        return;
    }
    let (Some(cpu), Some(bytes)) = (read_u64(record), read_u64(record + 0x18)) else { return };
    let bytes = bytes & 0xffff_ffff;
    if cpu == 0 || bytes == 0 || bytes > 1 << 30 || cpu.checked_add(bytes).is_none() {
        return;
    }
    if only_read() {
        settle(cpu as usize, bytes as usize);
    }
    // (Stage 1) What is held on the card there goes in before the title
    // touches it.
    super::consumers::touched(cpu as usize, bytes as usize, super::consumers::Kind::TitleLock);
    if !locks_meter() {
        return;
    }
    let content = read(cpu, bytes).unwrap_or_default();
    let s = sum(&content);
    let nonzero = content.iter().filter(|&&x| x != 0).count() as u64;
    let (inside, over) = {
        let set = SET.read().unwrap_or_else(|e| e.into_inner());
        let alias = *ALIAS.lock().unwrap_or_else(|e| e.into_inner());
        (set.inside(cpu, bytes) == bytes, set.over(cpu, bytes, alias))
    };
    ring_locked(&over, nonzero);
    let mut l = LOCKS.lock().unwrap_or_else(|e| e.into_inner());
    let l = l.get_or_insert_with(Locks::default);
    l.n[0] += 1;
    l.open.insert(record, Open { handle, cpu, bytes, sum: s });
    if l.open.len() > 1 << 16 {
        l.open.clear();
    }
    if inside {
        return;
    }
    l.n[3] += 1;
    let overlap = WRITTEN.lock().unwrap_or_else(|e| e.into_inner()).recent(cpu, cpu + bytes, crate::frame::current().saturating_sub(RECENT));
    let object = read_u64(record + 0x30).unwrap_or(0);
    let head = if object != 0 { read_u64(object + 8).unwrap_or(0) } else { 0 };
    let h = l.handles.entry(handle).or_default();
    h.caller = caller;
    (h.count, h.stride) = (head & 0xffff_ffff, (head >> 32) & 0xfff);
    h.locks += 1;
    h.overlap += overlap;
    h.nonzero = nonzero;
    h.bytes = bytes;
    if h.locks > 1 && s != h.last_sum {
        h.changed += 1;
    }
    h.last_sum = s;
}

/// The read-back rings' buffers a lock met: counted for their feedback.
fn ring_locked(over: &[(Buffer, u64, u64)], nonzero: u64) {
    if over.is_empty() {
        return;
    }
    let mut r = RINGS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(rings) = r.as_mut() else { return };
    for &(b, _, _) in over {
        let Some(ring) = rings.get_mut(&b.ring) else { continue };
        ring.locks += 1;
        if nonzero == 0 {
            ring.zero += 1;
        }
        // The title frame whose GPU write this buffer holds now, by the
        // ring's state, against the one its last landing carried.
        if let Some((_, writing, title_frame)) = state(b.ring) {
            let wrote = title_frame - age(u64::from(b.index), writing, u64::from(b.count)) as i64;
            if ring.carried[usize::from(b.index.min(2))] < wrote {
                ring.late += 1;
            }
        }
    }
}

/// The unlock hook (rcx: the record).
pub fn unlocked(record: u64) {
    let open = {
        let mut l = LOCKS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(l) = l.as_mut() else { return };
        let Some(open) = l.open.remove(&record) else { return };
        open
    };
    let now = read(open.cpu, open.bytes).map(|v| sum(&v));
    let wrote = now.is_some_and(|s| s != open.sum);
    let mut l = LOCKS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(l) = l.as_mut() else { return };
    l.n[1] += 1;
    l.n[2] += u64::from(wrote);
    let set_has = SET.read().unwrap_or_else(|e| e.into_inner()).inside(open.cpu, open.bytes) == open.bytes;
    if set_has {
        return;
    }
    if let Some(h) = l.handles.get_mut(&open.handle) {
        if wrote {
            h.wrote += 1;
        } else {
            h.kept += 1;
        }
    }
}

pub(super) fn line() {
    let frame = crate::frame::current();
    WRITTEN.lock().unwrap_or_else(|e| e.into_inner()).forget_before(frame.saturating_sub(RECENT));
    if locks_meter() {
        let (n, mut handles) = {
            let mut l = LOCKS.lock().unwrap_or_else(|e| e.into_inner());
            let l = l.get_or_insert_with(Locks::default);
            (std::mem::take(&mut l.n), std::mem::take(&mut l.handles))
        };
        // Read-capable: no unlock saw a change.
        let reading: Vec<(&u64, &Handle)> = handles.iter().filter(|(_, h)| h.wrote == 0).collect();
        let over: Vec<&(&u64, &Handle)> = reading.iter().filter(|(_, h)| h.overlap > 0).collect();
        tracing::info!(
            "read-back locks this period: {} locks, {} unlocks seen ({} wrote between lock and unlock, {} did not); outside the read-back set: {} locks of {} buffers; read-capable buffers outside the set (never wrote): {}, of them {} over bytes the copy-back wrote in the last {RECENT} frames ({} bytes overlapping exactly; {} with non-zero bytes at the last lock; {} whose bytes changed between locks)",
            n[0],
            n[1],
            n[2],
            n[1] - n[2],
            n[3],
            handles.len(),
            reading.len(),
            over.len(),
            over.iter().map(|(_, h)| h.overlap).sum::<u64>(),
            over.iter().filter(|(_, h)| h.nonzero > 0).count(),
            over.iter().filter(|(_, h)| h.changed > 0).count(),
        );
        let mut top: Vec<(u64, Handle)> = handles.drain().collect();
        top.sort_unstable_by(|a, b| b.1.locks.cmp(&a.1.locks));
        for (handle, h) in top.iter().take(16) {
            tracing::info!(
                "lock outside the read-back set: handle {handle:#x} (stride {}, count {}), caller {:#x}, locks {}, wrote {}, did not write {}, shader-written bytes overlapped {}, non-zero at the last lock {} of {} bytes, changed between locks {}",
                h.stride, h.count, h.caller, h.locks, h.wrote, h.kept, h.overlap, h.nonzero, h.bytes, h.changed
            );
        }
    }
    let mut r = RINGS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(rings) = r.as_mut() else { return };
    let mut list: Vec<(&u64, &mut Ring)> = rings.iter_mut().collect();
    list.sort_unstable_by_key(|(a, _)| **a);
    for (at, ring) in list {
        if ring.name.is_empty() && ring.landed + ring.locks + ring.reads + ring.marked == 0 {
            continue;
        }
        let name = if ring.name.is_empty() { format!("site {:#x} ring {at:#x}", ring.site) } else { ring.name.to_string() };
        tracing::info!(
            "feedback ring {name}: landed {} times, non-zero {} of {} bytes, changed {} times, game locks {}, lock-before-data {} times (data arrived after the game read), all-zero at lock {} times; landing {:.2} frames after the GPU write on average; game reads {} (succeeded {}, refused {} as too young), read-before-data {}, all-zero at read {}, marked written {}; per buffer 0/1/2: marked {:?}, read {:?}, landed bytes {:?}",
            ring.landed,
            ring.nonzero,
            ring.bytes,
            ring.changed,
            ring.locks,
            ring.late,
            ring.zero,
            if ring.latencies == 0 { 0.0 } else { ring.latency as f64 / ring.latencies as f64 },
            ring.reads,
            ring.read_ok,
            ring.read_young,
            ring.read_nodata,
            ring.read_zero,
            ring.marked,
            ring.marked_by,
            ring.read_by,
            ring.landed_by
        );
        (ring.reads, ring.read_ok, ring.read_young, ring.read_nodata, ring.read_zero, ring.marked) = (0, 0, 0, 0, 0, 0);
        (ring.read_by, ring.marked_by, ring.landed_by) = ([0; 3], [0; 3], [0; 3]);
        (ring.landed, ring.bytes, ring.nonzero, ring.changed, ring.locks, ring.late, ring.zero, ring.latency, ring.latencies) = (0, 0, 0, 0, 0, 0, 0, 0, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_later_landing_replaces_what_it_covers() {
        let mut w = Written::default();
        w.mark(0x100, 0x200, 1);
        w.mark(0x180, 0x280, 5);
        assert_eq!(w.recent(0x100, 0x300, 0), 0x180);
        assert_eq!(w.recent(0x100, 0x300, 5), 0x100, "only the newer part");
        assert_eq!(w.recent(0x100, 0x180, 2), 0, "the older part is older");
        w.mark(0x120, 0x140, 9);
        assert_eq!(w.recent(0x100, 0x300, 9), 0x20);
        assert_eq!(w.recent(0x100, 0x300, 0), 0x180, "exact bytes, nothing counted twice");
        assert_eq!(w.recent(0x0, 0x100, 0), 0, "adjacent is not overlapping");
        w.forget_before(5);
        assert_eq!(w.recent(0x0, 0x1000, 0), 0x120);
    }

    #[test]
    fn a_buffer_ages_one_frame_per_step_of_the_ring() {
        assert_eq!(age(2, 2, 3), 0, "being written now");
        assert_eq!(age(1, 2, 3), 1, "written last frame: what the CPU reads");
        assert_eq!(age(0, 2, 3), 2);
        assert_eq!(age(2, 0, 3), 1);
        assert_eq!(age(0, 0, 1), 0);
    }

    #[test]
    fn the_sum_sees_a_change() {
        assert_ne!(sum(&[0, 0, 0, 1]), sum(&[0, 0, 1, 0]));
        assert_eq!(sum(&[]), 0x811c_9dc5);
    }
}
