//! (Prefault job, task 2) What shader writes are declared over title memory
//! for copy-back (`Shadow::gpu_writes`, every way there), and why: once per
//! indirect split line, "shader-write declarations per frame ...": count and
//! MB by size class, and the 8 extents with the most MB, each with its way
//! (a UAV of a descriptor table: the view as the title described it, first
//! element, count, element bytes, stride; a root UAV: no size, so all that is
//! left of the title's buffer from its address; a view the heap marks found
//! written; another way), the shader of the stage binding it, and the buffer
//! it is cut to.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Mutex;

/// The ways a declaration is made.
pub(crate) const TABLE_UAV: u8 = 1;
pub(crate) const ROOT_UAV: u8 = 2;
pub(crate) const HEAP_MARKED: u8 = 3;
/// A shader reading title memory by address (`crate::memory`): the whole
/// copy its root arguments name.
pub(crate) const BY_ADDRESS: u8 = 4;

/// What the declaring code knows, set around its `gpu_writes` call.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Ctx {
    /// 0 another way, else [`TABLE_UAV`], [`ROOT_UAV`], [`HEAP_MARKED`].
    pub way: u8,
    /// The shader of the stage binding it (0: the stage that stores, as
    /// [`set_writer`] last said).
    pub hash: u64,
    /// A table UAV's description: first element, count, element bytes,
    /// structure stride (0 none), raw.
    pub view: Option<(u64, u32, u64, u32, bool)>,
}

thread_local! {
    static CTX: Cell<Ctx> = const { Cell::new(Ctx { way: 0, hash: 0, view: None }) };
    /// The storing stage's shader of the draw being bound (0: none).
    static WRITER: Cell<u64> = const { Cell::new(0) };
}

/// The storing stage's shader (its hash as the "first bind" lines give
/// it; empty: none).
pub(crate) fn set_writer(hash: &str) {
    WRITER.with(|w| w.set(u64::from_str_radix(hash, 16).unwrap_or(0)));
}

/// `f` with `ctx` noted for the declarations it makes.
pub(crate) fn with<R>(ctx: Ctx, f: impl FnOnce() -> R) -> R {
    let was = CTX.with(|c| c.replace(ctx));
    let r = f();
    CTX.with(|c| c.set(was));
    r
}

/// Size classes: under 64 KiB, 1 MiB, 4 MiB, 16 MiB, 64 MiB, and more.
const CLASSES: [(u64, &str); 6] = [
    (64 << 10, "<64 KiB"),
    (1 << 20, "64 KiB-1 MiB"),
    (4 << 20, "1-4 MiB"),
    (16 << 20, "4-16 MiB"),
    (64 << 20, "16-64 MiB"),
    (u64::MAX, ">=64 MiB"),
];

pub(crate) fn class(len: u64) -> usize {
    CLASSES.iter().position(|&(top, _)| len < top).unwrap_or(CLASSES.len() - 1)
}

#[derive(Default)]
struct Stats {
    /// By class: declarations, bytes.
    classes: [(u64, u64); 6],
    in_place: u64,
    /// By (way, shader, address, bytes, view): declarations, buffer (start, bytes).
    top: HashMap<(Ctx, u64, u64), (u64, (u64, u64))>,
}

static STATS: Mutex<Option<Stats>> = Mutex::new(None);

/// A declaration of `[address, address + len)` of the copy of the title's
/// buffer `buffer` (start, bytes).
pub(crate) fn note(address: u64, len: u64, buffer: (u64, u64), in_place: bool) {
    let mut ctx = CTX.with(Cell::get);
    if ctx.hash == 0 {
        ctx.hash = WRITER.with(Cell::get);
    }
    let mut s = STATS.lock().unwrap_or_else(|e| e.into_inner());
    let s = s.get_or_insert_with(Stats::default);
    let c = class(len);
    s.classes[c].0 += 1;
    s.classes[c].1 += len;
    s.in_place += u64::from(in_place);
    if s.top.len() < 1 << 16 || s.top.contains_key(&(ctx, address, len)) {
        s.top.entry((ctx, address, len)).or_insert((0, buffer)).0 += 1;
    }
}

/// The line, the counts since the last one, over `frames` frames.
pub(crate) fn line(frames: f64) {
    let Some(s) = STATS.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    tracing::info!("{}", text(&s, frames));
}

fn text(s: &Stats, frames: f64) -> String {
    let f = frames.max(1.0);
    let mb = |b: u64| b as f64 / f / (1u64 << 20) as f64;
    let classes: Vec<String> =
        CLASSES.iter().zip(&s.classes).map(|(&(_, name), &(n, b))| format!("{name} {:.1} ({:.1} MB)", n as f64 / f, mb(b))).collect();
    let mut top: Vec<(&(Ctx, u64, u64), &(u64, (u64, u64)))> = s.top.iter().collect();
    top.sort_by_key(|(k, v)| std::cmp::Reverse(v.0 * k.2));
    let top: Vec<String> = top
        .iter()
        .take(8)
        .map(|((ctx, at, len), (n, (start, bytes)))| {
            let how = match (ctx.way, ctx.view) {
                (TABLE_UAV, Some((first, count, element, stride, raw))) => {
                    format!("table UAV first {first} count {count} element {element} B stride {stride}{}", if raw { " raw" } else { "" })
                }
                (TABLE_UAV, None) => "table UAV".to_owned(),
                (ROOT_UAV, _) => format!("root UAV: rest of the buffer from +{:#x}", at.saturating_sub(*start)),
                (HEAP_MARKED, _) => "heap-marked view".to_owned(),
                (BY_ADDRESS, _) => "read by address: the whole copy named".to_owned(),
                _ => "other way".to_owned(),
            };
            format!(
                "{:.1} MB/frame: {at:#x}+{len:#x} ({how}; shader {:016x}; {:.1} a frame; of buffer {start:#x}+{bytes:#x})",
                mb(n * len),
                ctx.hash,
                *n as f64 / f
            )
        })
        .collect();
    let (n, b) = s.classes.iter().fold((0, 0), |a, c| (a.0 + c.0, a.1 + c.1));
    format!(
        "shader-write declarations per frame: {:.1} ({:.1} MB; in place {:.1}); by extent: {}; top 8 by MB: [{}]",
        n as f64 / f,
        mb(b),
        s.in_place as f64 / f,
        classes.join(", "),
        top.join("; ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declarations_by_class_and_top_extent() {
        assert_eq!(class(4096), 0);
        assert_eq!(class(64 << 10), 1);
        assert_eq!(class(20 << 20), 4);
        assert_eq!(class(182 << 20), 5);
        let mut s = Stats::default();
        let heap = (0xb10000, 182 << 20);
        let root = Ctx { way: ROOT_UAV, hash: 0xabc, view: None };
        let table = Ctx { way: TABLE_UAV, hash: 0xdef, view: Some((16, 256, 16, 16, false)) };
        for _ in 0..2 {
            *s.top.entry((root, 0xb20000, 180 << 20)).or_insert((0, heap)) = (2, heap);
            s.top.entry((table, 0xb10100, 4096)).or_insert((0, heap)).0 += 1;
        }
        s.classes[5] = (2, 360 << 20);
        s.classes[0] = (2, 8192);
        let t = text(&s, 2.0);
        assert!(t.starts_with("shader-write declarations per frame: 2.0 (180.0 MB"), "{t}");
        let root_at = t.find("root UAV: rest of the buffer from +0x10000").expect(&t);
        let table_at = t.find("table UAV first 16 count 256 element 16 B stride 16").expect(&t);
        assert!(root_at < table_at, "most MB first: {t}");
        assert!(t.contains(">=64 MiB 1.0 (180.0 MB)"), "{t}");
    }
}
