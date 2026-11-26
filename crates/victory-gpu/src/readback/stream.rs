//! Finding F5: the title's resource streamer, paced in frames, as its
//! globals show it (`VICTORY_STREAM_METER`, on by default in diagnostics
//! builds): a line a second. The lists are the render thread's and are
//! read without locks: a torn read shows as an odd count, never a fault
//! (every read goes through the kernel), and nothing is written.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{title_u32, title_u64};

/// Nodes walked per list at most.
const CAP: usize = 4096;

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_STREAM_METER") {
        Ok(v) => v.trim() == "1",
        Err(_) => cfg!(feature = "diagnostics"),
    })
}

fn read_u64(at: u64) -> Option<u64> {
    let mut b = [0u8; 8];
    crate::host::read_into(at as *const u8, &mut b).then(|| u64::from_le_bytes(b))
}

fn read_u32(at: u64) -> Option<u32> {
    let mut b = [0u8; 4];
    crate::host::read_into(at as *const u8, &mut b).then(|| u32::from_le_bytes(b))
}

/// A list of groups from the head pointer at `head_va`: groups, bytes
/// (+0x80), requests (+0x8c), groups whose "not before frame" (+0x88) is
/// past `frame`; and whether the walk stopped on something odd or the cap.
fn walk(head_va: u64, frame: u32) -> (u64, u64, u64, u64, &'static str) {
    let Some(mut at) = title_u64(head_va) else { return (0, 0, 0, 0, " (head unreadable)") };
    let (mut groups, mut bytes, mut requests, mut gated) = (0u64, 0u64, 0u64, 0u64);
    let mut seen = std::collections::HashSet::new();
    while at != 0 {
        if at & 7 != 0 || at < 0x1_0000 || at >> 47 != 0 || !seen.insert(at) {
            return (groups, bytes, requests, gated, " (stopped: odd pointer)");
        }
        if groups as usize >= CAP {
            return (groups, bytes, requests, gated, " (cap)");
        }
        let (Some(next), Some(b), Some(stamp), Some(n)) = (read_u64(at), read_u64(at + 0x80), read_u32(at + 0x88), read_u32(at + 0x8c)) else {
            return (groups, bytes, requests, gated, " (stopped: unreadable)");
        };
        groups += 1;
        bytes += b & 0xffff_ffff;
        requests += u64::from(n);
        gated += u64::from(stamp > frame);
        at = next;
    }
    (groups, bytes, requests, gated, "")
}

/// (F9) Groups the enqueue refused and the builder dropped (0x1420cf7b0
/// entered), since the start.
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// (F12) Of the drops since the last line: empty polls (the group's entry
/// count u32 at +0x70 is 0), real ones (not 0), unreadable groups.
static DROP_KINDS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// A title pointer that may be read through: 8-byte aligned, past the
/// first 64 KiB, in user space.
fn sane(p: u64) -> bool {
    p & 7 == 0 && p >= 0x1_0000 && p >> 47 == 0
}

/// (F9) The drop of a streamer group (the runtime's entry hook on
/// 0x1420cf7b0; `group` is its rcx at entry). (F12) Classified by the
/// group's entry count (u32 at +0x70), read safely.
pub fn group_dropped(group: u64) {
    DROPPED.fetch_add(1, Ordering::Relaxed);
    let kind = match sane(group).then(|| read_u32(group + 0x70)).flatten() {
        Some(0) => 0,
        Some(_) => 1,
        None => 2,
    };
    DROP_KINDS[kind].fetch_add(1, Ordering::Relaxed);
}

/// (F13) Drawables created a frame (u32 at 0x149f360f8), sampled once per
/// presented frame: the largest since the last line, and the frames sampled.
static CREATED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// (F13) Once per presented frame: the title's count of drawables created
/// this frame.
pub(crate) fn models_frame() {
    if !on() {
        return;
    }
    REMOVED[1].fetch_add(1, Ordering::Relaxed);
    if let Some(c) = title_u32(0x149f360f8) {
        CREATED[0].fetch_max(u64::from(c), Ordering::Relaxed);
        CREATED[1].fetch_add(1, Ordering::Relaxed);
    }
    list_b_frame();
    // (F15) The global LOD scale's range over the second.
    if let Some(g) = title_f32(0x14a15e100) {
        let bits = u64::from(g.to_bits());
        let mut lo = LOD_RANGE[0].load(Ordering::Relaxed);
        while (lo == u64::MAX || g < f32::from_bits(lo as u32))
            && let Err(v) = LOD_RANGE[0].compare_exchange(lo, bits, Ordering::Relaxed, Ordering::Relaxed)
        {
            lo = v;
        }
        let mut hi = LOD_RANGE[1].load(Ordering::Relaxed);
        while (hi == u64::MAX || g > f32::from_bits(hi as u32))
            && let Err(v) = LOD_RANGE[1].compare_exchange(hi, bits, Ordering::Relaxed, Ordering::Relaxed)
        {
            hi = v;
        }
    }
}

/// (F15) The global LOD scale's least and greatest this second (f32 bits;
/// `u64::MAX`: none yet).
static LOD_RANGE: [AtomicU64; 2] = [AtomicU64::new(u64::MAX), AtomicU64::new(u64::MAX)];

fn title_f32(va: u64) -> Option<f32> {
    title_u32(va).map(f32::from_bits)
}

/// (F15) The per-second "lod scale:" line (reads only), and every 30 lines
/// its summary.
fn lod_line() {
    let f = |va: u64| title_f32(va).map_or("?".to_owned(), |v| format!("{v:.3}"));
    let global = title_f32(0x14a15e100);
    let (lo, hi) = (LOD_RANGE[0].swap(u64::MAX, Ordering::Relaxed), LOD_RANGE[1].swap(u64::MAX, Ordering::Relaxed));
    let r = |b: u64| if b == u64::MAX { "?".to_owned() } else { format!("{:.3}", f32::from_bits(b as u32)) };
    let types: Vec<String> = (0..7).map(|i| f(0x14a15e12c + 4 * i)).collect();
    let flags = title_u32(0x14a15e238).map(|v| (v & 0xff) as u8);
    let bit = |b: u8| flags.map_or("?".to_owned(), |f| u8::from(f & (1 << b) != 0).to_string());
    tracing::info!(
        "lod scale: global {} (min {} / max {} this second), base {}, script factor {}, per type [{}], final/base {}, blend source {}, flags {} (series-x bit6 {}, override bit1 {} value {}, remap bit3 {}, changed bit5 {}), console distance multipliers [{} {} {}] (expect 0.5 0.3 0.8 on Series X), camera-speed term {}",
        global.map_or("?".to_owned(), |v| format!("{v:.3}")),
        r(lo),
        r(hi),
        f(0x14a15e200),
        f(0x14ab4a20c),
        types.join(" "),
        f(0x14a15e20c),
        f(0x14a15e234),
        flags.map_or("?".to_owned(), |v| format!("{v:#04x}")),
        bit(6),
        bit(1),
        f(0x14a15e204),
        bit(3),
        bit(5),
        f(0x14a15e090),
        f(0x14a15e0c8),
        f(0x14a15e0dc),
        f(0x146c69808),
    );
    // Per run: lines, seconds with global < 1.0, the least seen (f32 bits), seconds with the series-x bit.
    static RUN: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(u64::MAX), AtomicU64::new(0)];
    let lines = RUN[0].fetch_add(1, Ordering::Relaxed) + 1;
    let least = if lo != u64::MAX { Some(f32::from_bits(lo as u32)) } else { global };
    if let Some(g) = least {
        RUN[1].fetch_add(u64::from(g < 1.0), Ordering::Relaxed);
        let was = RUN[2].load(Ordering::Relaxed);
        if was == u64::MAX || g < f32::from_bits(was as u32) {
            RUN[2].store(u64::from(g.to_bits()), Ordering::Relaxed);
        }
    }
    RUN[3].fetch_add(u64::from(flags.is_some_and(|f| f & 0x40 != 0)), Ordering::Relaxed);
    if lines % 30 == 0 {
        tracing::info!(
            "lod scale summary: seconds with global < 1.0 {} of {lines} (min seen {}); series-x bit set in {} of {lines} seconds",
            RUN[1].load(Ordering::Relaxed),
            r(RUN[2].load(Ordering::Relaxed)),
            RUN[3].load(Ordering::Relaxed),
        );
    }
}

/// (F14) The game's "list B" (visible entities with no drawable whose
/// archetype is loaded; what F13 read as "drawables created"): its size
/// u32 at 0x149f360f8 (at most 0x2000), entries at 0x149f160f8 of 0x10
/// bytes (+0 entity, +8 f32 score). Classified once per present, summed
/// until the line: entries looked at, created, hidden by flag, other
/// refused, also listed the frame before, the weight of a sampled entry
/// (every 4th past 2,000), presents; seconds the list was full (per run).
static LIST_B: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
/// (F17) Scores (f32 at entry+8) of the list-B entries looked at, scaled
/// like [`LIST_B`]: exactly 0.0, >= 1.0, anything else.
static LIST_B_SCORES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static LIST_B_FULL_S: AtomicU64 = AtomicU64::new(0);
static LIST_B_FULL_NOW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Presents that found the scene pass not run yet.
static LIST_B_NOT_RUN: AtomicU64 = AtomicU64::new(0);
/// Post-creation queue: max count this second, its cap, presents at the cap.
static POST_CREATE: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Other refused entries by (entity type, archetype flags), and last
/// frame's entities.
struct ListB {
    kinds: std::collections::HashMap<(u8, u8), u64>,
    last: std::collections::HashSet<u64>,
}

static LIST_B_STATE: std::sync::Mutex<Option<ListB>> = std::sync::Mutex::new(None);

fn read_u8(at: u64) -> Option<u8> {
    let mut b = [0u8; 1];
    crate::host::read_into(at as *const u8, &mut b).then(|| b[0])
}

fn list_b_frame() {
    let base = super::IMAGE.load(Ordering::Relaxed);
    let Some(n) = title_u32(0x149f360f8) else { return };
    if base == 0 {
        return;
    }
    // (F14) The post-creation queue: its largest count this second, and
    // presents that found it at its cap (the game skips entries then).
    if let (Some(q), Some(cap)) = (title_u32(0x14957f1c4), title_u32(0x14957f1c0)) {
        POST_CREATE[0].fetch_max(u64::from(q), Ordering::Relaxed);
        POST_CREATE[1].store(u64::from(cap), Ordering::Relaxed);
        POST_CREATE[2].fetch_add(u64::from(cap != 0 && q >= cap), Ordering::Relaxed);
    }
    // 0x2000 with 0x149f160f4 still 0 is the value before the scene pass
    // first ran, not a list.
    if n == 0x2000 && title_u32(0x149f160f4) == Some(0) {
        LIST_B_NOT_RUN.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let n = (n as usize).min(0x2000);
    // Full: the count reached the list's cap (u32 at 0x149f160f0).
    if title_u32(0x149f160f0).is_some_and(|cap| cap != 0 && n as u32 >= cap) {
        LIST_B_FULL_NOW.store(true, Ordering::Relaxed);
    }
    let mut bytes = vec![0u8; n * 0x10];
    if n > 0 && !crate::host::read_into((base + 0x149f160f8 - 0x1_4000_0000) as *const u8, &mut bytes) {
        return;
    }
    // Every 4th entry past 2,000 (and at most about 500 entries a present:
    // each is a kernel read on the presenting thread).
    let step = if n > 2000 { 4usize.max(n.div_ceil(500)) } else { 1usize.max(n.div_ceil(500)) };
    let mut state = LIST_B_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let state = state.get_or_insert_with(|| ListB { kinds: Default::default(), last: Default::default() });
    let mut now = std::collections::HashSet::with_capacity(n / step + 1);
    let (mut looked, mut created, mut hidden, mut other, mut again) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for i in (0..n).step_by(step) {
        let e = u64::from_le_bytes(bytes[i * 0x10..i * 0x10 + 8].try_into().unwrap_or_default());
        if !sane(e) {
            continue;
        }
        // (F17) The entry's streaming score.
        let score = f32::from_le_bytes(bytes[i * 0x10 + 8..i * 0x10 + 12].try_into().unwrap_or_default());
        let class = if score == 0.0 {
            0
        } else if score >= 1.0 {
            1
        } else {
            2
        };
        LIST_B_SCORES[class].fetch_add(step as u64, Ordering::Relaxed);
        // One read of the entity's fields (+0x20..+0xb8).
        let mut f = [0u8; 0x98];
        if !crate::host::read_into((e + 0x20) as *const u8, &mut f) {
            continue;
        }
        let u64_at = |o: usize| u64::from_le_bytes(f[o - 0x20..o - 0x18].try_into().unwrap_or_default());
        let u32_at = |o: usize| u32::from_le_bytes(f[o - 0x20..o - 0x1c].try_into().unwrap_or_default());
        let (d, a, f31, f34, fb4, kind) = (u64_at(0x28), u64_at(0x20), f[0x11], u32_at(0x34), u32_at(0xb4), f[0x10]);
        looked += 1;
        again += u64::from(state.last.contains(&e));
        now.insert(e);
        if d != 0 {
            created += 1;
        } else if f31 & 0x80 == 0 && f34 & 0x4001 == 0 && !matches!(fb4 & 0x3e, 0x0e | 0x0a) {
            hidden += 1;
        } else {
            other += 1;
            let flags = if sane(a) { read_u8(a + 0x84).unwrap_or(0xff) } else { 0xff };
            *state.kinds.entry((kind, flags)).or_default() += step as u64;
        }
    }
    state.last = now;
    let w = step as u64;
    for (i, v) in [looked * w, created * w, hidden * w, other * w, again * w, n as u64, 1].into_iter().enumerate() {
        LIST_B[i].fetch_add(v, Ordering::Relaxed);
    }
}

/// (F14) Drawable removals (the runtime's entry hook on 0x140151bb0, the
/// title's single drawable-removal function) that remove one: flag bit 0
/// set and a drawable at entity+0x28. Since the last line: the count, the
/// presents, and histograms by caller (return address, as 0x14...) and by
/// (entity type u8 +0x30, archetype flags u8 [+0x20]+0x84).
static REMOVED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
#[allow(clippy::type_complexity)]
static REMOVED_BY: std::sync::Mutex<Option<(std::collections::HashMap<[u64; 3], u64>, std::collections::HashMap<(u8, u8), u64>)>> =
    std::sync::Mutex::new(None);

/// (F17) Known removal call sites (return addresses, as 0x14...).
const REMOVAL_SITES: [(u64, &str); 3] = [
    (0x14227a0b5, "cleanup list 2 (score 0)"),
    (0x142279fc1, "cleanup list 1 (score < 0)"),
    (0x14227a236, "cleanup list 3 (pressure)"),
];

/// (F17) Whether `ret` (as 0x14...) looks like a return address in the
/// title's code: inside the code part of the image and right behind a call
/// (e8 rel32, or ff /2 in its 2-, 3-, 6- or 7-byte forms), read safely.
fn looks_returned_to(ret: u64) -> bool {
    if !(0x1_4000_1000..0x1_4620_0000).contains(&ret) {
        return false;
    }
    let base = super::IMAGE.load(Ordering::Relaxed);
    if base == 0 {
        return false;
    }
    let mut b = [0u8; 7];
    if !crate::host::read_into((base + ret - 7 - 0x1_4000_0000) as *const u8, &mut b) {
        return false;
    }
    // b[k] is the byte at ret - 7 + k.
    let ff2 = |k: usize| b[k] == 0xff && b[k + 1] & 0x38 == 0x10;
    b[2] == 0xe8 || ff2(0) || ff2(1) || ff2(4) || ff2(5)
}

/// (F17) Two more return addresses above the hook's return slot `slot`
/// (the stack scanned upwards, at most 48 slots, read safely, each value
/// checked by [`looks_returned_to`]); 0 where none was found.
fn callers_above(slot: u64) -> [u64; 2] {
    let base = super::IMAGE.load(Ordering::Relaxed);
    let mut found = [0u64; 2];
    if slot == 0 || slot & 7 != 0 || base == 0 {
        return found;
    }
    let mut words = [0u8; 48 * 8];
    if !crate::host::read_into((slot + 8) as *const u8, &mut words) {
        return found;
    }
    let mut n = 0;
    for w in words.chunks_exact(8) {
        let v = u64::from_le_bytes(w.try_into().unwrap_or_default());
        if v <= base {
            continue;
        }
        let ret = v - base + 0x1_4000_0000;
        if looks_returned_to(ret) {
            found[n] = ret;
            n += 1;
            if n == 2 {
                break;
            }
        }
    }
    found
}

fn site_name(ret: u64) -> Option<&'static str> {
    REMOVAL_SITES.iter().find(|s| s.0 == ret).map(|s| s.1)
}

/// (F14) The entry of 0x140151bb0: `entity` (rcx), `flags` (edx), `caller`
/// (the return address, image-relative as 0x14...); `slot`: where that
/// return address lies on the stack (0: not known), for two more callers.
pub fn drawable_removed(entity: u64, flags: u64, caller: u64, slot: u64) {
    if !on() || flags & 1 == 0 || !sane(entity) || read_u64(entity + 0x28).is_none_or(|d| d == 0) {
        return;
    }
    REMOVED[0].fetch_add(1, Ordering::Relaxed);
    let kind = read_u8(entity + 0x30).unwrap_or(0xff);
    let arch = read_u64(entity + 0x20).filter(|&p| sane(p)).and_then(|p| read_u8(p + 0x84)).unwrap_or(0xff);
    let mut by = REMOVED_BY.lock().unwrap_or_else(|e| e.into_inner());
    let (callers, kinds) = by.get_or_insert_with(Default::default);
    let [up1, up2] = callers_above(slot);
    let chain = [caller, up1, up2];
    if callers.len() < 4096 || callers.contains_key(&chain) {
        *callers.entry(chain).or_insert(0) += 1;
    }
    *kinds.entry((kind, arch)).or_insert(0) += 1;
}

/// (F14) The per-second "drawables removed" line.
fn removed_line() {
    let (n, presents) = (REMOVED[0].swap(0, Ordering::Relaxed), REMOVED[1].swap(0, Ordering::Relaxed));
    let (callers, kinds) = REMOVED_BY.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
    // (F17) By the real call site: the first known one of the chain, else
    // the chain as found.
    let mut by_site: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for (chain, k) in callers {
        let key = match chain.iter().find_map(|&a| site_name(a).map(|n| (a, n))) {
            Some((a, name)) => format!("{a:#x} {name}"),
            None => chain.iter().filter(|&&a| a != 0).map(|a| format!("{a:#x}")).collect::<Vec<_>>().join(" <- "),
        };
        *by_site.entry(key).or_insert(0) += k;
    }
    let mut callers: Vec<(String, u64)> = by_site.into_iter().collect();
    callers.sort_by(|a, b| b.1.cmp(&a.1));
    let mut kinds: Vec<((u8, u8), u64)> = kinds.into_iter().collect();
    kinds.sort_by(|a, b| b.1.cmp(&a.1));
    let c: Vec<String> = callers.iter().take(6).map(|(a, k)| format!("{a} x{k}")).collect();
    let k: Vec<String> = kinds.iter().take(4).map(|((t, f), k)| format!("{t:#04x}/{f:#04x} x{k}")).collect();
    tracing::info!(
        "drawables removed {n} per second (per presented frame {:.1}): callers [{}] (call site <- two above, by the first known site), kinds [{}] (type u8 +0x30 / archetype flags u8 +0x84; hook 0x140151bb0)",
        n as f64 / presents.max(1) as f64,
        c.join(", "),
        k.join(", ")
    );
}

/// (F14) The per-second "models list B:" line.
fn list_b_line() {
    let c: Vec<u64> = LIST_B.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect();
    if LIST_B_FULL_NOW.swap(false, Ordering::Relaxed) {
        LIST_B_FULL_S.fetch_add(1, Ordering::Relaxed);
    }
    let not_run = LIST_B_NOT_RUN.swap(0, Ordering::Relaxed);
    let post = format!(
        "post-create queue max {} of {} (frames at the cap: {})",
        POST_CREATE[0].swap(0, Ordering::Relaxed),
        POST_CREATE[1].load(Ordering::Relaxed),
        POST_CREATE[2].swap(0, Ordering::Relaxed)
    );
    if c[6] == 0 && not_run > 0 {
        tracing::info!("models list B: pass not run ({not_run} presents); {post}");
        return;
    }
    let presents = c[6].max(1);
    let f17 = f17_tail(presents);
    let top = {
        let mut state = LIST_B_STATE.lock().unwrap_or_else(|e| e.into_inner());
        let mut kinds: Vec<((u8, u8), u64)> = state.as_mut().map(|s| s.kinds.drain().collect()).unwrap_or_default();
        kinds.sort_by(|a, b| b.1.cmp(&a.1));
        kinds
            .iter()
            .take(4)
            .map(|((t, f), k)| format!("{t:#04x}/{f:#04x} x{}", k / presents))
            .collect::<Vec<_>>()
            .join(", ")
    };
    tracing::info!(
        "models list B: size {} (full {} s so far, at the cap at 0x149f160f0), created {}, hidden by flag {}, other refused {} [type/flags {}], also listed last frame {:.0} % (per presented frame, mean of {} presents, {} more with the pass not run; sampled to about 500 entries a present, scaled); {}{}",
        c[5] / presents,
        LIST_B_FULL_S.load(Ordering::Relaxed),
        c[1] / presents,
        c[2] / presents,
        c[3] / presents,
        top,
        if c[0] == 0 { 0.0 } else { 100.0 * c[4] as f64 / c[0] as f64 },
        c[6],
        not_run,
        post,
        f17,
    );
}

/// (F17) The game's guard against create/remove churn: the byte at
/// 0x147455e44 (a config bool cached), its init guard u32 at 0x147455e48.
/// `VICTORY_MODEL_SCORE_GUARD=1` sets the byte to 1, once a second, when
/// the init guard is non-zero and the byte is 0 (default 0: unchanged).
fn score_guard() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_MODEL_SCORE_GUARD").is_ok_and(|v| v.trim() == "1"))
}

/// (F17) Times the guard byte was set: the first, then re-writes after the
/// game cleared it.
static GUARD_SET: AtomicU64 = AtomicU64::new(0);

/// (F17) The tail of the "models list B:" line, and the guard switch.
fn f17_tail(presents: u64) -> String {
    let base = super::IMAGE.load(Ordering::Relaxed);
    let word = title_u32(0x14ab44324);
    let scorer = title_u64(0x1476d5420);
    let guard = title_u32(0x147455e44).map(|v| v & 0xff);
    let init = title_u32(0x147455e48);
    let s = match scorer {
        None => "?".to_owned(),
        Some(v) if base != 0 && v == base + 0x2282d20 => "1".to_owned(),
        Some(0) => "0".to_owned(),
        Some(v) => format!("0 (raw {v:#x})"),
    };
    if score_guard() && base != 0 && init.is_some_and(|i| i != 0) && guard == Some(0) {
        let at = (base + 0x147455e44 - 0x1_4000_0000) as usize;
        let one = [1u8];
        crate::watch::writing(at, 1);
        crate::host::write_mapped(at as *mut u8, one.as_ptr(), 1);
        let n = GUARD_SET.fetch_add(1, Ordering::Relaxed);
        if n == 0 {
            tracing::info!("models: score guard byte set (VICTORY_MODEL_SCORE_GUARD=1)");
        } else if n.is_power_of_two() {
            tracing::info!("models: score guard byte set again, the game had cleared it ({n} re-writes so far; VICTORY_MODEL_SCORE_GUARD=1)");
        }
    }
    let sc: Vec<u64> = LIST_B_SCORES.iter().map(|a| a.swap(0, Ordering::Relaxed) / presents).collect();
    let shown = |v: Option<u32>| v.map_or("?".to_owned(), |v| v.to_string());
    let set = GUARD_SET.load(Ordering::Relaxed);
    format!(
        "; scene-load word {} (bits 0x1c0: {}, bit 0x1000: {}), alt scorer installed {s}, guard byte {} (init {}){}, list B scores: zero {}, >= 1 {}, other {}",
        word.map_or("?".to_owned(), |w| format!("{w:#06x}")),
        word.map_or("?".to_owned(), |w| format!("{:#x}", w & 0x1c0)),
        word.map_or("?".to_owned(), |w| u8::from(w & 0x1000 != 0).to_string()),
        shown(guard),
        shown(init),
        if score_guard() { format!(" (VICTORY_MODEL_SCORE_GUARD=1: set {set} times, re-writes {})", set.saturating_sub(1)) } else { String::new() },
        sc[0],
        sc[1],
        sc[2],
    )
}

/// (F13) The per-second "models:" line, and every 30 lines its summary.
fn models_line() {
    let byte = |va: u64| title_u32(va).map(|v| v & 0xff);
    let shown = |v: Option<u32>| v.map_or("?".to_owned(), |v| v.to_string());
    let (wanted, urgent, any) = (title_u32(0x149f16034), title_u32(0x149f16040), byte(0x14875c029));
    let created = title_u32(0x149f360f8);
    let (max, frames) = (CREATED[0].swap(0, Ordering::Relaxed), CREATED[1].swap(0, Ordering::Relaxed));
    let cap = byte(0x149f360fc).is_some_and(|b| b != 0) || title_u32(0x14ab44324).is_some_and(|v| v & 0x1000 != 0);
    let machine = title_u64(0x147bc5d90).filter(|&p| sane(p));
    let (assets, stepped) = machine.map_or((None, None), |p| (read_u32(p + 0x15c), read_u32(p + 0xbe74)));
    tracing::info!(
        "models: assets wanted for visible entities {} (urgent {}, any-wanted flag {}); list B size {} (max this second {max} over {frames} presented frames; cap mode {}); \
         assets in the state machine {} (stepped last loop {})",
        shown(wanted),
        shown(urgent),
        shown(any),
        shown(created),
        u8::from(cap),
        shown(assets),
        shown(stepped),
    );
    list_b_line();
    removed_line();
    lod_line();
    // Per run: lines, seconds with assets wanted, wanted summed, the most, seconds in cap mode.
    static RUN: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
    let lines = RUN[0].fetch_add(1, Ordering::Relaxed) + 1;
    let w = u64::from(wanted.unwrap_or(0));
    RUN[1].fetch_add(u64::from(w > 0), Ordering::Relaxed);
    RUN[2].fetch_add(w, Ordering::Relaxed);
    RUN[3].fetch_max(w, Ordering::Relaxed);
    RUN[4].fetch_add(u64::from(cap), Ordering::Relaxed);
    if lines % 30 == 0 {
        tracing::info!(
            "models summary: seconds with assets wanted {} of {lines} (mean wanted {:.1}, max {}); seconds in cap mode {}",
            RUN[1].load(Ordering::Relaxed),
            RUN[2].load(Ordering::Relaxed) as f64 / lines as f64,
            RUN[3].load(Ordering::Relaxed),
            RUN[4].load(Ordering::Relaxed),
        );
    }
}

/// (F9) `VICTORY_STREAM_SLOT_CEILING=1`: the streamer's slot ceiling
/// [0x149c3e11c] (a high-water mark the enqueue refuses groups at) raised
/// to the slot table's capacity [0x1492a7f4c] - 1, once a second when lower
/// and the capacity reads sane. Default 0: unchanged.
fn slot_ceiling() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_STREAM_SLOT_CEILING").is_ok_and(|v| v.trim() == "1"))
}

/// (F9) The ceiling raised if [`slot_ceiling`] asks and it is due.
fn raise_ceiling(ceiling: Option<u32>, table: Option<u32>) {
    let (Some(ceiling), Some(table)) = (ceiling, table) else { return };
    if !slot_ceiling() || !(4096..=1_000_000).contains(&table) || ceiling >= table - 1 {
        return;
    }
    let base = super::IMAGE.load(Ordering::Relaxed);
    if base == 0 {
        return;
    }
    let at = (base + 0x149c3e11c - 0x1_4000_0000) as usize;
    let new = (table - 1).to_le_bytes();
    crate::watch::writing(at, 4);
    crate::host::write_mapped(at as *mut u8, new.as_ptr(), 4);
    let now = title_u32(0x149c3e11c);
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!("streamer: slot ceiling raised from {ceiling} to {} (VICTORY_STREAM_SLOT_CEILING=1; reads back {now:?})", table - 1)
    });
}

pub(super) fn line() {
    static LAST: AtomicU64 = AtomicU64::new(0);
    static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    if !on() {
        return;
    }
    let now = crate::host::now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 1000 || LAST.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    let Some(frame) = title_u32(0x14a9437ac) else { return };
    let before = FRAME.swap(u64::from(frame), Ordering::Relaxed);
    let frames = if before == u64::MAX { 0 } else { u64::from(frame).wrapping_sub(before) as i64 };
    let mb = |b: u64| b as f64 / (1 << 20) as f64;
    let (g1, b1, r1, gate1, s1) = walk(0x149c3d808, frame);
    let (g2, b2, _, _, s2) = walk(0x149c3d818, frame);
    let (g3, b3, _, _, s3) = walk(0x149c3d828, frame);
    let flight = title_u64(0x149c3e078).unwrap_or(0);
    let batches = title_u32(0x149c3e0c0).unwrap_or(0) & 0xff;
    // (F6) The ten i64 at 0x149c3e120.. are bytes committed per request
    // category 0-9 (no page factor: run-36's figures were 4096 times too big).
    let committed: Vec<String> = (0..10u64)
        .map(|i| format!("{:.1}", mb(title_u64(0x149c3e120 + 8 * i).unwrap_or(0))))
        .collect();
    let byte = |va: u64| title_u32(va).map_or(0, |v| v & 0xff);
    let pressure: Vec<f32> = (0..7u64).map(|i| title_u32(0x149f36104 + 4 * i).map_or(0.0, f32::from_bits)).collect();
    let culled = pressure.iter().any(|&f| f != 0.0);
    let no_fit = byte(0x1498d26d0);
    // Per run: seconds with any category culled, lines read, could-not-fit seen.
    static PRESSURE: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
    PRESSURE[0].fetch_add(u64::from(culled), Ordering::Relaxed);
    let lines = PRESSURE[1].fetch_add(1, Ordering::Relaxed) + 1;
    PRESSURE[2].fetch_add(u64::from(no_fit != 0), Ordering::Relaxed);
    // (F9) The slot ceiling, the table's capacity, groups dropped.
    static DROPPED_SEEN: AtomicU64 = AtomicU64::new(0);
    let dropped_total = DROPPED.load(Ordering::Relaxed);
    let dropped = dropped_total - DROPPED_SEEN.swap(dropped_total, Ordering::Relaxed);
    PRESSURE[3].fetch_add(u64::from(dropped != 0), Ordering::Relaxed);
    let (ceiling, table) = (title_u32(0x149c3e11c), title_u32(0x1492a7f4c));
    let shown = |v: Option<u32>| v.map_or("?".to_owned(), |v| v.to_string());
    let kinds: Vec<u64> = DROP_KINDS.iter().map(|n| n.swap(0, Ordering::Relaxed)).collect();
    let unreadable = if kinds[2] > 0 { format!(", group unreadable {}", kinds[2]) } else { String::new() };
    let f9 = format!(
        "; slot ceiling {} of table {}; groups dropped {dropped} (empty {}, real {}{unreadable}) (total {dropped_total})",
        shown(ceiling),
        shown(table),
        kinds[0],
        kinds[1],
    );
    raise_ceiling(ceiling, table);
    tracing::info!(
        "streamer: game frames +{frames}; to-issue {g1} groups ({:.1} MB){s1}, in-flight {g2}{s2}, to-finish {g3}{s3}; in-flight bytes {:.1} MB; batches in flight {batches}; to-issue requests {r1}, gated by frame {gate1}; in-flight groups {:.1} MB, to-finish groups {:.1} MB;          budget {:.1} of {:.1} MB (peak {:.1}), series-s {}; pressure [{}], could-not-fit {no_fit}, evicting {} (to free {:.1} MB); slots {}/{}; caps {}/{}/{}; committed by category [{}] MB{f9}",
        mb(b1),
        mb(flight),
        mb(b2),
        mb(b3),
        mb(title_u64(0x1498d26d8).unwrap_or(0)),
        mb(title_u64(0x149c3e060).unwrap_or(0)),
        mb(title_u64(0x1475ddb98).unwrap_or(0)),
        byte(0x1471b63b0),
        pressure.iter().map(|f| format!("{f}")).collect::<Vec<_>>().join(" "),
        byte(0x1498cc2a6),
        mb(title_u64(0x1498cc298).unwrap_or(0)),
        title_u32(0x149c3e118).unwrap_or(0),
        title_u32(0x149c3e11c).unwrap_or(0),
        title_u32(0x149c3e1c0).unwrap_or(0),
        title_u32(0x149c3e1c4).unwrap_or(0),
        title_u32(0x149c3e1c8).unwrap_or(0),
        committed.join(" "),
    );
    // (F13) Model detail requests and drawable creation.
    models_line();
    // With the housekeeping cadence: the run so far.
    if lines % 30 == 0 {
        tracing::info!(
            "streamer pressure: seconds with any category culled {} of {lines}; could-not-fit seen {} times; seconds with a dropped group {}",
            PRESSURE[0].load(Ordering::Relaxed),
            PRESSURE[2].load(Ordering::Relaxed),
            PRESSURE[3].load(Ordering::Relaxed)
        );
    }
}
