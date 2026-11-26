//! The title's CPU read-back buffers (finding F1), and what the copy-back
//! puts into title memory outside them.
//!
//! The engine makes every buffer its CPU reads back through one class
//! ("CpuReadBackShaderResourceBuffer"); a hook in the title's code
//! (`victory_hle::readback`) hands each ring of them to [`ring_created`]
//! once its buffers exist. The ranges form the read-back set.
//!
//! * `VICTORY_READBACK_SET` (default 1; 0: no hook): record the set, and
//!   meter what the copy-back writes into title memory ([`back`], at each
//!   place it writes) inside and outside it; a line every 5 s ([`line`]).
//! * `VICTORY_READBACK_LOCKS` (default 0): the engine's generic buffer
//!   lock is hooked too ([`locked`]); a lock whose range holds memory
//!   shaders wrote outside the set is logged (the completeness test).
//! * `VICTORY_BACK_ONLY_READ` (default 0): what the copy-back would write
//!   wholly outside the set is not written. It is kept here instead (the
//!   card's bytes newer than title memory), and every read of title memory
//!   by the layer's copies ([`overlay`]: `read_into`, `copy_mapped`) sees
//!   it as if written, byte by byte where title memory still holds what it
//!   held when the write was skipped (a byte the title wrote since is the
//!   title's). It goes into title memory ([`settle`]) before the layer
//!   writes title memory there, before the runtime remaps it, when the
//!   title locks it (with the lock hook), and when more than
//!   `VICTORY_BACK_ONLY_READ_MB` (256) is held.
//!
//! shortcut: the set only grows (no hook on the buffers' release).
//! shortcut: the title's own CPU reads of skipped memory see the old bytes:
//! that is the bet the switch tests (the lock meter is its evidence).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

pub(crate) mod consumers;
mod feedback;
mod stream;
pub(crate) mod terrain;
pub use stream::{drawable_removed, group_dropped};
pub(crate) use stream::models_frame;
pub use feedback::{lock_entered, lock_returned, ring_marked, ring_read, unlocked};

/// The title image's base (from the hook installer): title globals are
/// read at `base + (va - 0x140000000)`.
static IMAGE: AtomicU64 = AtomicU64::new(0);

pub fn image_base(base: u64) {
    IMAGE.store(base, Ordering::Relaxed);
}

/// The title global at `va` (as the analyst's, 0x14...); `None` before the
/// base is known or if unreadable.
pub(crate) fn title_u32(va: u64) -> Option<u32> {
    let base = IMAGE.load(Ordering::Relaxed);
    if base == 0 {
        return None;
    }
    let mut b = [0u8; 4];
    crate::host::read_into((base + va - 0x1_4000_0000) as *const u8, &mut b).then(|| u32::from_le_bytes(b))
}

pub(crate) fn title_u64(va: u64) -> Option<u64> {
    let base = IMAGE.load(Ordering::Relaxed);
    if base == 0 {
        return None;
    }
    read_u64(base + va - 0x1_4000_0000)
}

/// Whether the ring reader and "mark written" are hooked: with
/// `VICTORY_READBACK_LOCKS=1` or `VICTORY_TERRAIN_FEEDBACK=1`.
pub fn reads_on() -> bool {
    set_on() && (locks_on() || terrain::on())
}

fn switch(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => v.trim() == "1",
        Err(_) => default,
    }
}

/// `VICTORY_READBACK_SET` (on unless `=0`).
pub fn set_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch("VICTORY_READBACK_SET", true))
}

/// Whether the title's buffer lock is hooked: `VICTORY_READBACK_LOCKS=1`,
/// or (unless `=0`) with `VICTORY_BACK_SKIP_UNREAD=1`, whose held pieces a
/// lock fetches first ([`consumers::demand`]).
pub fn locks_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| set_on() && switch("VICTORY_READBACK_LOCKS", consumers::skip_on()))
}

/// `VICTORY_READBACK_LOCKS=1`: the lock meter itself (off by default).
pub(crate) fn locks_meter() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| set_on() && switch("VICTORY_READBACK_LOCKS", false))
}

/// `VICTORY_BACK_ONLY_READ=1` (off by default).
fn only_read() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| set_on() && switch("VICTORY_BACK_ONLY_READ", false))
}

fn held_most() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_BACK_ONLY_READ_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(256) << 20
    })
}

/// The GPU's form of CPU address `cpu`: title memory in `[alias[0],
/// alias[0] + alias[1])` is given to the GPU at `alias[2]` on; other memory
/// at its own address.
pub(crate) fn gpu_form(cpu: u64, alias: [u64; 3]) -> u64 {
    let [from, len, base] = alias;
    if len != 0 && cpu >= from && cpu - from < len { base.wrapping_add(cpu - from) } else { cpu }
}

/// A read-back buffer: its ring, CPU address, GPU address, bytes,
/// creation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Buffer {
    pub(crate) ring: u64,
    /// Its index in the ring, and the ring's number of buffers.
    pub(crate) index: u8,
    pub(crate) count: u8,
    pub(crate) cpu: u64,
    pub(crate) gpu: u64,
    pub(crate) bytes: u64,
    pub(crate) order: u64,
}

/// The read-back set: its buffers, and the address ranges they cover in
/// either form (start to end, disjoint, merged).
#[derive(Default)]
pub(crate) struct Set {
    pub(crate) buffers: Vec<Buffer>,
    spans: BTreeMap<u64, u64>,
    pub(crate) bytes: u64,
    pub(crate) rings: u64,
}

/// Bytes of `[from, to)` that `spans` (disjoint, merged) cover.
fn covered(spans: &BTreeMap<u64, u64>, from: u64, to: u64) -> u64 {
    let mut n = 0;
    for (&a, &e) in spans.range(..to).rev() {
        if e <= from {
            break;
        }
        n += e.min(to) - a.max(from);
    }
    n
}

fn add_span(spans: &mut BTreeMap<u64, u64>, mut from: u64, mut to: u64) {
    if from >= to {
        return;
    }
    let meeting: Vec<(u64, u64)> = spans.range(..=to).rev().take_while(|&(_, &e)| e >= from).map(|(&a, &e)| (a, e)).collect();
    for (a, e) in meeting {
        spans.remove(&a);
        from = from.min(a);
        to = to.max(e);
    }
    spans.insert(from, to);
}

impl Set {
    /// Records a buffer: its CPU range, its GPU range as the object gives
    /// it (0: none), and its CPU range's alias form.
    pub(crate) fn add(&mut self, ring: (u64, u8, u8), cpu: u64, gpu: u64, bytes: u64, alias: [u64; 3]) -> Buffer {
        let gpu = if gpu == 0 { gpu_form(cpu, alias) } else { gpu };
        let order = self.buffers.len() as u64;
        let (ring, index, count) = ring;
        let buffer = Buffer { ring, index, count, cpu, gpu, bytes, order };
        if self.buffers.len() < 1 << 16 {
            self.buffers.push(buffer);
        }
        for at in [cpu, gpu, gpu_form(cpu, alias)] {
            add_span(&mut self.spans, at, at.saturating_add(bytes));
        }
        self.bytes += bytes;
        buffer
    }

    /// The buffers that hold bytes of `[at, at + len)`, each with the part
    /// of the range it holds (`[a, e)`, in the range's addresses; by the
    /// first of its forms that meets it).
    pub(crate) fn over(&self, at: u64, len: u64, alias: [u64; 3]) -> Vec<(Buffer, u64, u64)> {
        let end = at.saturating_add(len);
        let mut out = Vec::new();
        for b in &self.buffers {
            for f in [b.cpu, b.gpu, gpu_form(b.cpu, alias)] {
                let (a, e) = (at.max(f), end.min(f.saturating_add(b.bytes)));
                if a < e {
                    out.push((*b, a, e));
                    break;
                }
            }
        }
        out
    }

    /// The rings whose buffers hold bytes of `[at, at + len)` (in either
    /// form), with those bytes.
    #[cfg(test)]
    pub(crate) fn rings_over(&self, at: u64, len: u64, alias: [u64; 3]) -> Vec<(u64, u64)> {
        let mut out: Vec<(u64, u64)> = Vec::new();
        for (b, a, e) in self.over(at, len, alias) {
            match out.iter_mut().find(|r| r.0 == b.ring) {
                Some(r) => r.1 += e - a,
                None => out.push((b.ring, e - a)),
            }
        }
        out
    }

    /// Bytes of `[at, at + len)` inside the set, in any form.
    pub(crate) fn inside(&self, at: u64, len: u64) -> u64 {
        covered(&self.spans, at, at.saturating_add(len))
    }
}

static SET: RwLock<Set> = RwLock::new(Set { buffers: Vec::new(), spans: BTreeMap::new(), bytes: 0, rings: 0 });
static RINGS: AtomicU64 = AtomicU64::new(0);
/// The alias words as the last ring read them ([`gpu_form`]).
static ALIAS: Mutex<[u64; 3]> = Mutex::new([0; 3]);
/// Since the last [`line`]: bytes the copy-back wrote inside the set, by ring.
static BY_RING: Mutex<Option<HashMap<u64, u64>>> = Mutex::new(None);

/// Since the last [`line`]: bytes the copy-back wrote inside and outside
/// the set, bytes not written (`VICTORY_BACK_ONLY_READ`), bytes written
/// later (settled), title buffer locks seen, and of them those over memory
/// shaders wrote outside the set.
static PERIOD: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
const IN: usize = 0;
const OUT: usize = 1;
const SKIPPED: usize = 2;
const SETTLED: usize = 3;

/// Since the last [`line`]: bytes written outside the set by range start
/// (at most [`TOP_KEYS`] starts; the rest go under 0).
static OUTSIDE: Mutex<Option<HashMap<u64, u64>>> = Mutex::new(None);
const TOP_KEYS: usize = 1 << 14;

fn read_u64(at: u64) -> Option<u64> {
    let mut b = [0u8; 8];
    crate::host::read_into(at as *const u8, &mut b).then(|| u64::from_le_bytes(b))
}

/// From the hook after a read-back ring's constructor made its buffers:
/// `ring` holds up to three buffer objects (one if its flags, at +0x54,
/// have 8), each with its element count (u32 at +8), stride (u16 at +0xc,
/// low 12 bits), CPU address (+0x18) and GPU address (+0x38). `alias_at`:
/// the title's three words of [`gpu_form`] (start, length, GPU base).
/// `site`: where the title asked for the ring (the return address into the
/// wrapper's caller, as 0x14...; 0: not known), and its name if known.
/// Reads go through the kernel: a bad pointer reads as nothing.
pub fn ring_created(ring: usize, alias_at: u64, site: u64, name: &'static str) {
    if !set_on() || ring == 0 {
        return;
    }
    let alias = [0, 8, 16].map(|o| read_u64(alias_at + o).unwrap_or(0));
    *ALIAS.lock().unwrap_or_else(|e| e.into_inner()) = alias;
    let ring = ring as u64;
    let flags = read_u64(ring + 0x50).map_or(0, |v| v >> 32);
    let count = if flags & 8 != 0 { 1 } else { 3 };
    let mut made = Vec::new();
    for i in 0..count {
        let Some(object) = read_u64(ring + 8 * i).filter(|&o| o != 0) else {
            continue;
        };
        let (Some(head), Some(cpu), Some(gpu)) = (read_u64(object + 8), read_u64(object + 0x18), read_u64(object + 0x38)) else {
            continue;
        };
        let elements = head & 0xffff_ffff;
        let stride = (head >> 32) & 0xfff;
        let bytes = elements * stride;
        if cpu == 0 || bytes == 0 || bytes > 1 << 30 {
            tracing::debug!(object = format_args!("{object:#x}"), cpu, elements, stride, "read-back buffer object not taken");
            continue;
        }
        made.push(SET.write().unwrap_or_else(|e| e.into_inner()).add((ring, i as u8, count as u8), cpu, gpu, bytes, alias));
    }
    let n = RINGS.fetch_add(1, Ordering::Relaxed) + 1;
    SET.write().unwrap_or_else(|e| e.into_inner()).rings = n;
    feedback::ring(ring, site, name);
    if name == "terrain_virtual_texture" {
        terrain::ring_known(ring, &made);
    }
    if n <= 64 || n.is_power_of_two() {
        let at: Vec<String> = made
            .iter()
            .map(|b| if b.gpu == b.cpu { format!("{:#x}", b.cpu) } else { format!("{:#x} (gpu {:#x})", b.cpu, b.gpu) })
            .collect();
        tracing::info!(
            ring = n,
            flags,
            "read-back ring: {} buffers of {} bytes at {} (ring {ring:#x}, site {site:#x} {name})",
            made.len(),
            made.first().map_or(0, |b| b.bytes),
            at.join(", ")
        );
    }
}

/// The copy-back is about to put `bytes` (what shaders wrote, merged with
/// the title's) into title memory at `at`: metered, and with
/// `VICTORY_BACK_ONLY_READ` kept here instead if none of it is in the set
/// (true: the caller writes nothing).
pub(crate) fn back(at: usize, bytes: &[u8]) -> bool {
    let n = bytes.len() as u64;
    if !set_on() || n == 0 {
        return false;
    }
    terrain::landed(at as u64, bytes);
    let inside = SET.read().unwrap_or_else(|e| e.into_inner()).inside(at as u64, n);
    consumers::produced(at, bytes.len(), inside);
    PERIOD[IN].fetch_add(inside, Ordering::Relaxed);
    if inside > 0 {
        let alias = *ALIAS.lock().unwrap_or_else(|e| e.into_inner());
        let over = SET.read().unwrap_or_else(|e| e.into_inner()).over(at as u64, n, alias);
        {
            let mut by = BY_RING.lock().unwrap_or_else(|e| e.into_inner());
            let by = by.get_or_insert_with(HashMap::new);
            for &(b, a, e) in &over {
                *by.entry(b.ring).or_insert(0) += e - a;
            }
        }
        feedback::landed(at as u64, bytes, &over);
    }
    if inside < n {
        PERIOD[OUT].fetch_add(n - inside, Ordering::Relaxed);
        let mut top = OUTSIDE.lock().unwrap_or_else(|e| e.into_inner());
        let top = top.get_or_insert_with(HashMap::new);
        let key = if top.len() < TOP_KEYS || top.contains_key(&(at as u64)) { at as u64 } else { 0 };
        *top.entry(key).or_insert(0) += n - inside;
        if locks_meter() {
            feedback::written(at as u64, n);
        }
    }
    if inside == 0 && only_read() && RINGS.load(Ordering::Relaxed) > 0 && crate::inplace::level() == 0 && defer(at, bytes) {
        PERIOD[SKIPPED].fetch_add(n, Ordering::Relaxed);
        return true;
    }
    // Written now: what is held over it goes first (the caller's bytes
    // were made from title memory as `overlay` shows it).
    settle(at, bytes.len());
    false
}

/// Bytes the copy-back did not write: `new` for title memory at its key,
/// where it still holds `old` (what it held when the write was skipped).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Held {
    pub(crate) old: Vec<u8>,
    pub(crate) new: Vec<u8>,
}

/// Held writes by address: disjoint.
#[derive(Default)]
pub(crate) struct Ledger {
    map: BTreeMap<usize, Held>,
    bytes: u64,
}

impl Ledger {
    fn meeting(&self, from: usize, to: usize) -> Vec<usize> {
        let mut keys = Vec::new();
        for (&a, h) in self.map.range(..to).rev() {
            if a + h.new.len() <= from {
                break;
            }
            keys.push(a);
        }
        keys
    }

    /// Takes out what is held over `[from, to)`.
    pub(crate) fn take(&mut self, from: usize, to: usize) -> Vec<(usize, Held)> {
        let keys = self.meeting(from, to);
        keys.into_iter()
            .filter_map(|a| {
                let h = self.map.remove(&a)?;
                self.bytes -= h.new.len() as u64;
                Some((a, h))
            })
            .collect()
    }

    /// Holds `new` at `at` in place of an earlier write of the same range;
    /// false if there is none.
    pub(crate) fn renew(&mut self, at: usize, new: &[u8]) -> bool {
        match self.map.get_mut(&at) {
            Some(h) if h.new.len() == new.len() => {
                h.new.copy_from_slice(new);
                true
            }
            _ => false,
        }
    }

    /// Holds a write; false (nothing held) if something else is held over
    /// it.
    pub(crate) fn hold(&mut self, at: usize, held: Held) -> bool {
        if !self.meeting(at, at + held.new.len()).is_empty() {
            return false;
        }
        self.bytes += held.new.len() as u64;
        self.map.insert(at, held);
        true
    }

    /// Title memory `dst` read at `src`, as it would be had the held
    /// writes been made.
    pub(crate) fn overlay(&self, src: usize, dst: &mut [u8]) {
        for a in self.meeting(src, src + dst.len()) {
            let h = &self.map[&a];
            let (from, to) = (a.max(src), (a + h.new.len()).min(src + dst.len()));
            for at in from..to {
                let (d, j) = (&mut dst[at - src], at - a);
                if *d == h.old[j] {
                    *d = h.new[j];
                }
            }
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }
}

/// Title memory `ram` with `held` written where it still holds the old bytes.
pub(crate) fn resolved(ram: &[u8], held: &Held) -> Vec<u8> {
    ram.iter().zip(&held.old).zip(&held.new).map(|((&r, &o), &n)| if r == o { n } else { r }).collect()
}

static LEDGER: RwLock<Ledger> = RwLock::new(Ledger { map: BTreeMap::new(), bytes: 0 });
/// Whether anything is held: [`overlay`] and [`settle`] do nothing
/// otherwise.
static ANY: AtomicBool = AtomicBool::new(false);

fn ledger() -> std::sync::RwLockWriteGuard<'static, Ledger> {
    LEDGER.write().unwrap_or_else(|e| e.into_inner())
}

fn defer(at: usize, bytes: &[u8]) -> bool {
    let over = {
        let mut l = ledger();
        if l.renew(at, bytes) {
            drop(l);
            noted(at, bytes.len());
            return true;
        }
        let over = l.take(at, at + bytes.len());
        ANY.store(l.len() > 0, Ordering::Relaxed);
        over
    };
    write_back(over);
    if LEDGER.read().unwrap_or_else(|e| e.into_inner()).bytes + bytes.len() as u64 > held_most() {
        return false;
    }
    // What title memory holds now (nothing else is held over it).
    let mut old = vec![0u8; bytes.len()];
    if !crate::host::read_into(at as *const u8, &mut old) {
        return false;
    }
    let held = {
        let mut l = ledger();
        let held = l.hold(at, Held { old, new: bytes.to_vec() });
        ANY.store(l.len() > 0, Ordering::Relaxed);
        held
    };
    if held {
        noted(at, bytes.len());
    }
    held
}

/// A held write counts as written for every copy of its memory, as the
/// write would have: they take it again, through [`overlay`].
fn noted(at: usize, n: usize) {
    crate::watch::writing(at, n);
    crate::watch::noted_written(at, n);
}

/// Writes held bytes into title memory.
fn write_back(held: Vec<(usize, Held)>) {
    for (at, h) in held {
        let n = h.new.len();
        let mut ram = vec![0u8; n];
        if !crate::host::read_into(at as *const u8, &mut ram) {
            continue;
        }
        let out = resolved(&ram, &h);
        if out != ram {
            crate::watch::writing(at, n);
            crate::host::write_mapped(at as *mut u8, out.as_ptr(), n);
        }
        PERIOD[SETTLED].fetch_add(n as u64, Ordering::Relaxed);
    }
}

/// Title memory `[at, at + len)` is about to be written, remapped or read
/// by the title: what is held over it goes in first.
pub(crate) fn settle(at: usize, len: usize) {
    if !ANY.load(Ordering::Relaxed) || len == 0 {
        return;
    }
    let held = {
        let mut l = ledger();
        let held = l.take(at, at.saturating_add(len));
        ANY.store(l.len() > 0, Ordering::Relaxed);
        held
    };
    write_back(held);
}

/// The layer read title memory `[src, src + len)` into `dst`: shown as if
/// the held writes had been made.
pub(crate) fn overlay(src: usize, dst: *mut u8, len: usize) {
    if !ANY.load(Ordering::Relaxed) || len == 0 {
        return;
    }
    // SAFETY: the caller's destination of `len` bytes, just written.
    let dst = unsafe { std::slice::from_raw_parts_mut(dst, len) };
    LEDGER.read().unwrap_or_else(|e| e.into_inner()).overlay(src, dst);
}

/// The lines, every 5 s (called once a second from the frame-rate line).
pub(crate) fn line() {
    static LAST: AtomicU64 = AtomicU64::new(0);
    stream::line();
    terrain::line();
    if !set_on() {
        return;
    }
    let now = crate::host::now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 5000 || LAST.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    let p: Vec<u64> = PERIOD.iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
    let mut top: Vec<(u64, u64)> = OUTSIDE.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default().into_iter().collect();
    top.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    // Key 0: every range past the first TOP_KEYS starts of the period.
    let top: Vec<String> =
        top.iter().take(8).map(|(a, n)| if *a == 0 { format!("(other ranges) {n}") } else { format!("{a:#x} {n}") }).collect();
    let (rings, bytes) = {
        let s = SET.read().unwrap_or_else(|e| e.into_inner());
        (s.rings, s.bytes)
    };
    let mb = |v: u64| v as f64 / (1 << 20) as f64;
    let all = p[IN] + p[OUT];
    tracing::info!(
        "read-back set: {rings} rings, {bytes} bytes; copy-back this period: {:.1} MB inside the set, {:.1} MB outside ({:.1} % outside); ranges outside the set that were copied back: top 8 by bytes with address and bytes: [{}]",
        mb(p[IN]),
        mb(p[OUT]),
        if all == 0 { 0.0 } else { p[OUT] as f64 * 100.0 / all as f64 },
        top.join(", ")
    );
    let mut rings: Vec<(u64, u64)> = BY_RING.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default().into_iter().collect();
    if !rings.is_empty() {
        rings.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        let list: Vec<String> = rings.iter().take(8).map(|(r, n)| format!("ring {r:#x} {n}")).collect();
        tracing::info!("read-back set: copy-back this period inside the set by ring, top 8 of {}: [{}]", rings.len(), list.join(", "));
    }
    if only_read() {
        let (ranges, held) = {
            let l = LEDGER.read().unwrap_or_else(|e| e.into_inner());
            (l.len(), l.bytes)
        };
        tracing::info!(
            "copy-back skipped as never read: {:.1} MB (VICTORY_BACK_ONLY_READ=1; {:.1} MB written later, when needed; {ranges} ranges, {:.1} MB held newer than title memory)",
            mb(p[SKIPPED]),
            mb(p[SETTLED]),
            mb(held)
        );
    }
    feedback::line();
    consumers::line();
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALIAS: [u64; 3] = [0x2_0000_0000, 0x1000_0000, 0x9_0000_0000];

    #[test]
    fn the_alias_moves_only_memory_in_its_window() {
        assert_eq!(gpu_form(0x2_0000_1000, ALIAS), 0x9_0000_1000);
        assert_eq!(gpu_form(0x2_1000_0000, ALIAS), 0x2_1000_0000, "the window's end is outside it");
        assert_eq!(gpu_form(0x1_ffff_ffff, ALIAS), 0x1_ffff_ffff);
        assert_eq!(gpu_form(0x2_0000_1000, [0x2_0000_0000, 0, 0x9_0000_0000]), 0x2_0000_1000);
    }

    #[test]
    fn ranges_are_inside_by_either_address() {
        let mut s = Set::default();
        // GPU address not given: the alias gives it.
        let b = s.add((0x10, 0, 3), 0x2_0000_1000, 0, 0x100, ALIAS);
        assert_eq!(b.gpu, 0x9_0000_1000);
        // Given, and different from both.
        s.add((0x10, 0, 3), 0x5000_0000, 0x7000_0000, 0x40, ALIAS);
        assert_eq!(s.inside(0x2_0000_1000, 0x100), 0x100);
        assert_eq!(s.inside(0x9_0000_1000, 0x100), 0x100, "the GPU's form");
        assert_eq!(s.inside(0x9_0000_10f0, 0x20), 0x10, "half in");
        assert_eq!(s.inside(0x2_0000_0f00, 0x100), 0, "just before");
        assert_eq!(s.inside(0x2_0000_1100, 0x100), 0, "just after");
        assert_eq!(s.inside(0x7000_0020, 0x40), 0x20);
        assert_eq!(s.inside(0x5000_0000, 0x40), 0x40);
        // Overlapping and adjacent buffers merge, and count once.
        s.add((0x10, 0, 3), 0x2_0000_1080, 0, 0x100, ALIAS);
        s.add((0x10, 0, 3), 0x2_0000_1180, 0, 0x80, ALIAS);
        assert_eq!(s.inside(0x2_0000_0000, 0x10000), 0x200);
        assert_eq!(s.buffers.len(), 4);
        assert_eq!(s.buffers[3].order, 3);
        // By ring, in either form: ring 0x10 holds all of these.
        assert_eq!(s.rings_over(0x9_0000_1000, 0x40, ALIAS), [(0x10, 0x40)]);
        assert!(s.rings_over(0x3000_0000, 0x40, ALIAS).is_empty());
        // The part each buffer holds, in the asked addresses.
        let over = s.over(0x9_0000_10f0, 0x20, ALIAS);
        assert_eq!(over.len(), 2);
        assert_eq!((over[0].1, over[0].2), (0x9_0000_10f0, 0x9_0000_1100));
        assert_eq!((over[1].1, over[1].2), (0x9_0000_10f0, 0x9_0000_1110));
    }

    #[test]
    fn a_held_write_shows_where_title_memory_is_unchanged() {
        let mut l = Ledger::default();
        assert!(l.hold(0x1000, Held { old: vec![1, 2, 3, 4], new: vec![9, 9, 3, 9] }));
        assert!(!l.hold(0x1002, Held { old: vec![0; 4], new: vec![0; 4] }), "overlapping");
        // The title wrote 0x1001 since (2 -> 7): its byte stays.
        let mut read = vec![0, 1, 7, 3, 4, 5];
        l.overlay(0x0fff, &mut read);
        assert_eq!(read, [0, 9, 7, 3, 9, 5]);
        // A partial read.
        let mut read = vec![3, 4];
        l.overlay(0x1002, &mut read);
        assert_eq!(read, [3, 9]);
        // A newer write of the same range replaces it, the old bytes stay.
        assert!(l.renew(0x1000, &[8, 8, 8, 8]));
        assert!(!l.renew(0x1000, &[8, 8]));
        let held = l.take(0x1003, 0x1004);
        assert_eq!(held.len(), 1);
        assert_eq!(resolved(&[1, 7, 3, 4], &held[0].1), [8, 7, 8, 8]);
        assert_eq!(l.len(), 0);
        assert_eq!(l.bytes, 0);
        assert!(l.take(0, usize::MAX).is_empty());
    }
}
