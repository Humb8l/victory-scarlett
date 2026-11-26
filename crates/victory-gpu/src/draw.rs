//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::resource::RootIndexedView;
use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// The data probe (`VICTORY_PROBE`).
#[cfg(feature = "diagnostics")]
pub(crate) mod probe;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

/// (P3, recording cost) Whether what a draw would give the host list again
/// unchanged is left out (`VICTORY_BIND_FILTER`, default 1; `=0` gives
/// everything again for every draw, as before). D3D12 keeps root arguments,
/// vertex and index buffer views and the rasteriser and output-merger state
/// of a command list from one draw to the next: a value set again that equals
/// what the list holds changes nothing. What the list holds is remembered per
/// recording ([`Issued`]) and forgotten wherever it may have changed outside
/// this record: a new recording (reset, a cut, another host list), a root
/// signature set (D3D12 drops the root arguments of that kind; heaps changed
/// set it again), a pass of ours that binds its own (clamps: the compute
/// root signature is set again after them), the parts of a large dispatch,
/// rectangles' own vertex buffers, the mesh test's scissors.
pub(crate) fn bind_filter() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BIND_FILTER").is_none_or(|v| v != "0"))
}

/// (P3) Since the start: values given to host lists (even) and left out as
/// unchanged (odd), for root constants, root views, root tables, vertex
/// buffers, the index buffer, and the rest (topology, viewports, scissors,
/// blend factor, stencil reference, depth bounds).
pub(crate) static FILTERED: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
const F_CONSTANTS: usize = 0;
const F_VIEWS: usize = 2;
const F_TABLES: usize = 4;
const F_VERTEX: usize = 6;
const F_INDEX: usize = 8;
const F_OTHER: usize = 10;

/// (P3) A root argument as last given to the host list.
#[derive(Debug, Clone, PartialEq, Default)]
enum Given {
    #[default]
    None,
    /// The whole parameter's words, from the first.
    Constants(Vec<u32>),
    /// A root view: constant buffer, shader resource, unordered access.
    View(u8, u64),
    /// A descriptor table's first host descriptor.
    Table(u64),
}

/// (P3) What a host list recording was given so far ([`bind_filter`]).
#[derive(Default)]
struct Issued {
    /// The recording ([`HostList::serial`]); 0: none yet.
    serial: u64,
    /// By root parameter, graphics and compute.
    roots: [Vec<Given>; 2],
    /// The vertex buffer views from slot 0 and how many; the index buffer view.
    vertex: Option<([D3D12_VERTEX_BUFFER_VIEW; 32], usize)>,
    index: Option<(u64, u32, i32)>,
    topology: Option<i32>,
    viewports: Option<Vec<[u32; 6]>>,
    scissors: Option<Vec<[i32; 4]>>,
    blend: Option<Option<[u32; 4]>>,
    stencil: Option<u32>,
    depth_bounds: Option<[u32; 2]>,
}

impl Issued {
    /// Brought to `list`'s recording: everything forgotten if it is another.
    fn at(&mut self, list: &HostList) -> &mut Self {
        self.at_serial(list.serial())
    }

    /// [`Issued::at`] by the recording's number.
    fn at_serial(&mut self, serial: u64) -> &mut Self {
        if self.serial != serial {
            *self = Issued { serial, ..Issued::default() };
        }
        self
    }

    /// Whether `value` is what root parameter `param` holds; if not, it is
    /// noted as what it holds from now on (the caller sets it). Counted.
    fn root(&mut self, list: &HostList, compute: bool, param: u32, value: Given, group: usize) -> bool {
        self.root_in(list.serial(), bind_filter(), compute, param, value, group)
    }

    /// [`Issued::root`] by the recording's number, the filter on or off.
    fn root_in(&mut self, serial: u64, on: bool, compute: bool, param: u32, value: Given, group: usize) -> bool {
        let roots = &mut self.at_serial(serial).roots[usize::from(compute)];
        let p = param as usize;
        if roots.len() <= p {
            roots.resize(p + 1, Given::None);
        }
        let same = on && roots[p] == value;
        if !same {
            roots[p] = value;
        }
        FILTERED[group + usize::from(same)].fetch_add(1, Ordering::Relaxed);
        same
    }

    /// [`Issued::root`] for a parameter of root constants given whole: no
    /// copy of the words unless they changed.
    fn words(&mut self, list: &HostList, compute: bool, param: u32, values: &[u32]) -> bool {
        self.words_in(list.serial(), bind_filter(), compute, param, values)
    }

    /// [`Issued::words`] by the recording's number, the filter on or off.
    fn words_in(&mut self, serial: u64, on: bool, compute: bool, param: u32, values: &[u32]) -> bool {
        let roots = &mut self.at_serial(serial).roots[usize::from(compute)];
        let p = param as usize;
        if roots.len() <= p {
            roots.resize(p + 1, Given::None);
        }
        let same = on && matches!(&roots[p], Given::Constants(c) if c.as_slice() == values);
        if !same {
            match &mut roots[p] {
                Given::Constants(c) => {
                    c.clear();
                    c.extend_from_slice(values);
                }
                r => *r = Given::Constants(values.to_vec()),
            }
        }
        FILTERED[F_CONSTANTS + usize::from(same)].fetch_add(1, Ordering::Relaxed);
        same
    }

    /// Whether the host list holds vertex buffer views `views[..count]` from
    /// slot 0; noted if not. Counted.
    fn vertex_in(&mut self, serial: u64, on: bool, views: &[D3D12_VERTEX_BUFFER_VIEW; 32], count: usize) -> bool {
        let issued = self.at_serial(serial);
        let same = on
            && issued.vertex.as_ref().is_some_and(|(given, n)| {
                *n == count
                    && given[..count].iter().zip(&views[..count]).all(|(a, b)| {
                        a.BufferLocation == b.BufferLocation && a.SizeInBytes == b.SizeInBytes && a.StrideInBytes == b.StrideInBytes
                    })
            });
        if !same {
            issued.vertex = Some((*views, count));
        }
        FILTERED[F_VERTEX + usize::from(same)].fetch_add(1, Ordering::Relaxed);
        same
    }

    /// Whether the host list holds index buffer view `given` (address,
    /// bytes, format); noted if not. Counted.
    fn index_in(&mut self, serial: u64, on: bool, given: (u64, u32, i32)) -> bool {
        let issued = self.at_serial(serial);
        let same = on && issued.index == Some(given);
        issued.index = Some(given);
        FILTERED[F_INDEX + usize::from(same)].fetch_add(1, Ordering::Relaxed);
        same
    }

    /// The root arguments of one kind are gone from the host list (its root
    /// signature was set, or a pass of ours set its own).
    fn roots_gone(&mut self, compute: bool) {
        self.roots[usize::from(compute)].clear();
    }

    /// One root parameter was set outside this record.
    fn root_gone(&mut self, compute: bool, param: u32) {
        if let Some(r) = self.roots[usize::from(compute)].get_mut(param as usize) {
            *r = Given::None;
        }
    }

    /// Whether `value` is what `slot` of the rest holds; noted if not. Counted.
    fn other<T: PartialEq>(on: bool, slot: &mut Option<T>, value: T) -> bool {
        let same = on && slot.as_ref() == Some(&value);
        if !same {
            *slot = Some(value);
        }
        FILTERED[F_OTHER + usize::from(same)].fetch_add(1, Ordering::Relaxed);
        same
    }
}

/// (P3, recording cost, `VICTORY_CHEAP_DRAWS`) The host addresses a recording's
/// draws found for their root views, vertex and index buffers, by (title
/// address, bytes, kind), with the copy brought up to date there: the lookup
/// (the tables of title buffers and windows, under their locks) and the
/// refresh are skipped for the same ask while (a) it is the same recording,
/// (b) the list's refreshed ranges were not forgotten since (a refresh asked
/// again is then nothing: `HostList::defer_refresh` keeps one per range until
/// a title copy or work that may write title memory is recorded), and (c)
/// nothing changed what an address finds (`crate::shadow::epoch`).
#[derive(Default)]
struct Looked {
    /// Recording, its refresh count, the copies' epoch, when filled.
    key: (u64, u64, u64),
    found: std::collections::HashMap<(u64, u64, u8), u64, BuildPlain>,
}

/// A hasher for keys of addresses (no hashing against attackers needed).
#[derive(Default, Clone, Copy)]
struct BuildPlain;

#[derive(Default)]
struct Plain(u64);

impl std::hash::Hasher for Plain {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ u64::from(b)).wrapping_mul(0x51_7cc1_b727_220a_95);
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x51_7cc1_b727_220a_95);
    }
    fn write_u8(&mut self, n: u8) {
        self.write_u64(u64::from(n));
    }
}

impl std::hash::BuildHasher for BuildPlain {
    type Hasher = Plain;
    fn build_hasher(&self) -> Plain {
        Plain(0)
    }
}

/// (P3) Since the start: lookups of root views and vertex and index buffers
/// skipped ([`Looked`]), and those made.
pub(crate) static LOOKED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Kinds of [`Looked`] asks.
const LOOK_VERTEX: u8 = 0;
const LOOK_INDEX: u8 = 1;
const LOOK_CBV: u8 = 2;
const LOOK_SRV: u8 = 3;

impl Looked {
    /// The host address for an ask, from the record or from `look` (which
    /// finds the copy and brings it up to date; `None` from it is not kept,
    /// nor is an answer `look` says is not to be kept).
    fn get(&mut self, list: &HostList, ask: (u64, u64, u8), look: impl FnOnce() -> Option<(u64, bool)>) -> Option<u64> {
        let key = || (list.serial(), list.refresh_epoch(), crate::shadow::epoch());
        self.get_in(crate::host::cheap_draws(), key, ask, look)
    }

    /// [`Looked::get`] with what says the record is still good (`key`: the
    /// recording, its refresh count, the copies' epoch), on or off.
    fn get_in(
        &mut self,
        on: bool,
        key: impl Fn() -> (u64, u64, u64),
        ask: (u64, u64, u8),
        look: impl FnOnce() -> Option<(u64, bool)>,
    ) -> Option<u64> {
        if !on {
            return look().map(|(gpu, _)| gpu);
        }
        let now = key();
        if self.key != now {
            self.key = now;
            self.found.clear();
        }
        if let Some(&gpu) = self.found.get(&ask) {
            LOOKED[0].fetch_add(1, Ordering::Relaxed);
            return Some(gpu);
        }
        LOOKED[1].fetch_add(1, Ordering::Relaxed);
        let (gpu, keep) = look()?;
        // The look itself may have moved the epoch (a copy made): kept
        // under the key as it stands now, which that copy is part of.
        let now = key();
        if now != self.key {
            self.key = now;
            self.found.clear();
        }
        if keep {
            self.found.insert(ask, gpu);
        }
        Some(gpu)
    }
}

/// (P3) Root views by kind, for [`Given::View`].
const GIVEN_CBV: u8 = 0;
const GIVEN_SRV: u8 = 1;
const GIVEN_UAV: u8 = 2;

/// (P3) The filter's counts since the last call, as the per-second text:
/// "root constants set S skipped K, root views S/K, ...", per frame over `frames`.
pub(crate) fn filter_text(frames: f64) -> String {
    static LAST: std::sync::Mutex<[u64; 12]> = std::sync::Mutex::new([0; 12]);
    let now: [u64; 12] = std::array::from_fn(|i| FILTERED[i].load(Ordering::Relaxed));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let d: [f64; 12] = std::array::from_fn(|i| now[i].wrapping_sub(last[i]) as f64 / frames.max(1.0));
    *last = now;
    static LAST_LOOKED: std::sync::Mutex<[u64; 2]> = std::sync::Mutex::new([0; 2]);
    let now_looked: [u64; 2] = std::array::from_fn(|i| LOOKED[i].load(Ordering::Relaxed));
    let mut last_looked = LAST_LOOKED.lock().unwrap_or_else(|e| e.into_inner());
    let looked: [f64; 2] = std::array::from_fn(|i| now_looked[i].wrapping_sub(last_looked[i]) as f64 / frames.max(1.0));
    *last_looked = now_looked;
    static LAST_ASKS: std::sync::Mutex<[u64; 2]> = std::sync::Mutex::new([0; 2]);
    let now_asks: [u64; 2] = std::array::from_fn(|i| crate::host::WRITTEN_ASKS[i].load(Ordering::Relaxed));
    let mut last_asks = LAST_ASKS.lock().unwrap_or_else(|e| e.into_inner());
    let asks: [f64; 2] = std::array::from_fn(|i| now_asks[i].wrapping_sub(last_asks[i]) as f64 / frames.max(1.0));
    *last_asks = now_asks;
    format!(
        "bind filter per frame (VICTORY_BIND_FILTER={}): root constants set {:.0} skipped {:.0}, root views {:.0}/{:.0}, tables {:.0}/{:.0}, \
         vertex buffers {:.0}/{:.0}, index buffers {:.0}/{:.0}, other {:.0}/{:.0}; lookups of root views, vertex and index buffers \
         skipped {:.0} of {:.0}, shader-written asks of indirect records answered again {:.0} of {:.0} (VICTORY_CHEAP_DRAWS={})",
        u8::from(bind_filter()),
        d[0],
        d[1],
        d[2],
        d[3],
        d[4],
        d[5],
        d[6],
        d[7],
        d[8],
        d[9],
        d[10],
        d[11],
        looked[0],
        looked[0] + looked[1],
        asks[0],
        asks[0] + asks[1],
        u8::from(crate::host::cheap_draws()),
    )
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// Host states of a depth target, and of a texture shaders read.
const DEPTH_WRITE: u32 = 0x10;
const DEPTH_READ: u32 = 0x20;
const SHADER_RESOURCE: u32 = 0x40 | 0x80;

thread_local! {
    /// While a draw is prepared (`State::bind`, then its targets' states):
    /// the depth target the draw only tests (0: none, or one it writes),
    /// and whether a view its shaders read is of that same texture.
    static TESTED_DEPTH: std::cell::Cell<(usize, bool)> =
        const { std::cell::Cell::new((0, false)) };
    /// While a stage's tables are walked (`State::tables`): whether that
    /// stage's shader can store to buffers (`Translated::stores`).
    static STORES: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    /// While a stage's tables are walked (`State::bind`): its container's
    /// hash as the "first bind" lines give it, as a number, once some
    /// buffer view has been refused or given its own bytes as a raw view
    /// (`crate::shadow::flag_view`); 0 otherwise.
    static STAGE_HASH: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// While `State::early_textures` runs the texture pass: the
    /// milliseconds and the bytes that run may take, in place of the
    /// pass's own per frame.
    static PASS_BUDGET: std::cell::Cell<Option<(u64, u64)>> = const { std::cell::Cell::new(None) };
}

/// The milliseconds of every frame the pass that uploads textures picked
/// by index takes before a shader that picks by index has asked for it
/// (`VICTORY_EARLY_TEXTURES_MS`, 12; 0: never, as before), and the
/// milliseconds it may take just before a draw of the title's decoration
/// bake (`VICTORY_BAKE_TEXTURES_MS`, 1500; 0: nothing special).
///
/// The pass ran only from the first world frame on, where it met some
/// 24,000 slots at once and needed 60-80 frames for them. The title bakes
/// its characters' skin and decorations into textures of its own while it
/// loads, 159-1,165 frames before that (runs 156-174), with a pixel shader
/// that picks its sources by index: whatever was not on the card then was
/// baked as nothing, once, for the run. On the console a texture is there
/// when its bytes are.
fn early_budgets() -> (u64, u64) {
    static MS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        let ms = |name: &str, default: u64| {
            std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default)
        };
        (ms("VICTORY_EARLY_TEXTURES_MS", 12), ms("VICTORY_BAKE_TEXTURES_MS", 1500))
    })
}

/// The pixel shaders of the title's decoration bake (the title:
/// `PS_BakeDecorations_*`), by the first 16 hex digits of their hash as the
/// frame listings print it. They draw into a 256x256 corner of 2048x2048
/// targets and name no texture in a table.
const BAKE_PIXEL_SHADERS: [&str; 4] =
    ["c0a7fe77b1862e10", "0d1575a396dfe513", "b7cfd8dda91ebb6b", "280967ad158cee0b"];

/// (Neptune 25fbf69) The least milliseconds between two passes over textures picked by
/// index within a frame, when the title has written slots since the last one
/// (`VICTORY_TEXTURE_CATCH_UP_MS`, 2; 0: no pass to catch up, as before). 47 % of such
/// textures were there one frame after their slot was written, 37 % in the same frame
/// (Neptune run ve143): the frame's pass had run before the title wrote them.
fn catch_up_ms() -> u64 {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| std::env::var("VICTORY_TEXTURE_CATCH_UP_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2))
}

/// (Neptune) Since the last log line: passes made to catch up ([`catch_up_ms`]) and the
/// microseconds they took.
pub(crate) static EARLY_CATCH_UPS: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY_CATCH_UP_MICROS: AtomicU64 = AtomicU64::new(0);

/// The log line of the passes that caught up (from the 5-second meter in `host.rs`).
pub(crate) fn catch_up_line() {
    let n = EARLY_CATCH_UPS.swap(0, Ordering::Relaxed);
    let us = EARLY_CATCH_UP_MICROS.swap(0, Ordering::Relaxed);
    tracing::info!(
        "textures picked by index: {n} passes made within frames to catch up with slots the title wrote, {:.1} ms in them \
         (VICTORY_TEXTURE_CATCH_UP_MS={}; 0: off)",
        us as f64 / 1000.0,
        catch_up_ms()
    );
}

/// When a pass over textures picked by index last began (ms), and whether one catches up.
static TEXTURE_PASS_MS: AtomicU64 = AtomicU64::new(0);
static CATCHING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

thread_local! {
    /// While a pass catches up ([`catch_up_turn`]): it leaves out the frame's bookkeeping.
    static PASS_LIGHT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The turn of the one thread catching up; given back when dropped.
struct CatchUp;

impl Drop for CatchUp {
    fn drop(&mut self) {
        CATCHING.store(false, Ordering::Release);
    }
}

/// A turn to catch up, if [`catch_up_ms`] passed since the last pass of any kind and the
/// title wrote slots due at once since ([`crate::descriptor::host::FRESH_WRITES`]).
fn catch_up_turn() -> Option<CatchUp> {
    let ms = catch_up_ms();
    if ms == 0 || crate::descriptor::host::FRESH_WRITES.load(Ordering::Relaxed) == 0 {
        return None;
    }
    let now = crate::host::now_ms();
    if now.saturating_sub(TEXTURE_PASS_MS.load(Ordering::Relaxed)) < ms {
        return None;
    }
    CATCHING.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed).ok()?;
    Some(CatchUp)
}

/// Since the start: early runs of the texture pass, and those of them
/// made for a bake draw.
pub(crate) static EARLY_TEXTURE_PASSES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Whether a buffer range a shader names through an unordered access view
/// is given to shaders as written (snapshot before the submission, compare
/// and write back after) only if that shader's code can store to buffers.
/// The title binds buffers its shaders only read through such views, and
/// every range declared costs a snapshot and a compare of its bytes: 3.97
/// GB a frame in 31 submissions, 0.49 s of snapshots and 0.28 s of
/// comparing (run 28). On by default; `VICTORY_WRITE_SKIP=0` declares
/// every such range, as before.
fn write_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WRITE_SKIP").is_none_or(|v| v != "0"))
}

/// A run of indirect records clamped in one pass ([`windows`]): the host
/// copy they came from (its number), their kind and stride, the offset of
/// the first in that copy, how many, and the list's scratch buffer that
/// holds them clamped, record `i` at `i * step`.
#[derive(Clone)]
struct Window {
    copy: u64,
    kind: u32,
    stride: u32,
    /// How far apart the records are in the copy, and in the scratch
    /// buffer: what the title walks them by ([`windows`]).
    step: u32,
    base: u64,
    records: u32,
    scratch: ID3D12Resource,
}

/// The most records clamped in one pass ([`windows`]).
const WINDOW_MOST: u64 = 64;

/// Whether indirect draws that read record after record of one argument
/// buffer are clamped a window at a time (`VICTORY_INDIRECT_WINDOW`; on by
/// default, `=0` turns it off; off by default with
/// `VICTORY_CARD_FETCH_INFLIGHT=1`, the one run with a device loss since the
/// fix below had both on).
///
/// On by default since a ten-and-a-half-minute run on an RTX 4070 with no
/// device loss: the card's time for the title's lists went from about 85
/// to 37-43 ms a frame, and the frame from about 174 ms to 144-154.
///
/// The title issues its mesh draws one record per call, each from the next
/// slot of the same argument buffer: 3,460 a frame in gameplay, and before
/// each our clamp ran as a compute dispatch of its own between two waits,
/// so the card went from drawing to compute and back for every record. In
/// the timing run of the lit scene (run 79) a shadow draw of next to
/// nothing cost 14-16 microseconds, 4,126 records a frame.
///
/// With the switch, when a draw asks for a record a short step after the
/// one drawn before, the clamp takes that record and the ones that step
/// apart after it (16, then twice as many each time, [`WINDOW_MOST`] at
/// most, never past the buffer) into
/// the list's scratch buffer in one pass, and the draws that follow read
/// their record there with no pass of ours and no barrier (the scratch
/// buffer stays readable as arguments until the next clamp), until one of
/// these: a compute
/// dispatch, a direct draw, an indirect call with a count buffer or more
/// than one record, a draw whose shaders can store to buffers, a record
/// outside the window, another host list.
///
/// The step is the one the title is seen to take, not the stride its
/// command signature names: it draws one record a call, for which the
/// stride means nothing, and the title names 12 bytes (the three
/// words of a mesh dispatch) for records it keeps 32 bytes apart (2,173 of
/// a frame's 3,349 mesh records in run 87's listed frame; 283 are 16
/// apart, which ends a window). Walked by the named stride the window
/// never opened: run 87, with the switch, still clamped 5,068 records a
/// frame one at a time.
///
/// What it costs the guard against garbage records: nothing against a
/// hang (every record is bounded as before and charged to the list's
/// budget as before), but the budget is charged when a window is clamped,
/// so also for records of the window's tail that are never drawn. Those
/// hold garbage in this title (slots of all ones: run 93 spent lists'
/// budgets on them and emptied the draws after), so in a window a record
/// that asks for more than can run is left empty and charges nothing
/// (`crate::indirect::bound_ahead`); a tail record that looks real is
/// charged what it asks, a few thousand of 2^27. And arguments the title's CPU side changes are seen as
/// before, at the next submission.
///
/// Off for the rest of the run from the first error the card's driver
/// reports to the system (`crate::drvwatch`): every loss of the card in a
/// run with windows was announced that way seconds ahead.
fn windows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var_os("VICTORY_INDIRECT_WINDOW") {
        Some(v) => v == "1",
        None => std::env::var_os("VICTORY_CARD_FETCH_INFLIGHT").is_none_or(|v| v != "1"),
    }) && !crate::drvwatch::tripped()
}

/// Whether a single record that misses its window is gathered: given a
/// slot that the list's prelude fills in one pass with the list's other
/// such records, in place of a pass of ours in front of its draw
/// (`crate::indirect::gather`; fxceless 465b94f, `VICTORY_INDIRECT_GATHER=1`;
/// `alt`: twenty seconds with, twenty without, in turn, each change said in
/// the log). Off by default. Needs [`windows`].
fn gathers() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_INDIRECT_GATHER").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "single indirect records are gathered for their list's prelude from now (VICTORY_INDIRECT_GATHER=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

/// (Neptune) Whether a single indirect record without a count buffer,
/// which nothing the list has recorded so far was declared to write, is
/// clamped in its list's prelude and drawn from its slot there, instead of
/// by a dispatch of ours between two waits in front of its draw
/// (`VICTORY_CLAMP_AHEAD=0`: in place, as before). As in Neptune it acts
/// only while windows are off ([`windows`]; ours has them on by default and
/// gathers records that miss them with `VICTORY_INDIRECT_GATHER=1`); here
/// the record takes a slot of the list's gathered records
/// (`crate::indirect::gather_ahead`). Neptune: 3,200 to 5,000 records a
/// frame clamped in place in the middle of the title's draws.
fn clamp_ahead() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CLAMP_AHEAD").is_none_or(|v| v != "0"))
}

/// Whether, with [`gathers`] on, a record that misses its window after its
/// list has stopped gathering (a dispatch or a storing shader was recorded)
/// is still gathered when no shader of the list so far was declared to
/// write its bytes (`HostList::gpu_wrote`), as [`clamp_ahead`] does with
/// windows off. On unless `VICTORY_GATHER_LATE=0`. Why: in run-50 every
/// record gathering refused (about 440 a frame) was refused for "the list
/// had stopped gathering", none for the limits; each went to a clamp pass
/// of ours in front of its draw.
fn gather_late() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GATHER_LATE").is_none_or(|v| v != "0"))
}

/// (Card kinds job, task 3) Whether a single dispatch record (no count
/// buffer, not in parts) that no write the list declared so far touches is
/// gathered for the list's prelude like a draw's under [`gather_late`],
/// instead of clamped by a pass of ours in front of it. Dispatch records
/// were never gathered: gathering was made for windowed draws, and a
/// dispatch ends a window and closes gathering (it may write what later
/// commands read); neither concerns the dispatch's OWN record. On unless
/// `VICTORY_GATHER_DISPATCH=0`.
fn gather_dispatch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GATHER_DISPATCH").is_none_or(|v| v != "0"))
}

/// (Neptune) `VICTORY_INDIRECT_RAW=1`, an experiment: a mesh draw of one
/// record without a count buffer reads the title's own argument record,
/// with no clamp of ours before it. A garbage record runs as it stands.
fn raw_indirect() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_RAW").is_some_and(|v| v == "1"))
}

/// (Neptune) `VICTORY_INDIRECT_RAW=2`, an experiment: every indirect call
/// (draws, indexed draws, mesh draws, dispatches, with or without a count)
/// is drawn from the title's own arguments in their host copy, with no
/// pass of ours to bound them; not a dispatch split for its first group's
/// number. Garbage arguments are the card's to deal with (it hung the card
/// on zm0kie's PC).
fn raw_all() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_RAW").is_some_and(|v| v == "2"))
}

/// (Neptune) `VICTORY_INDIRECT_RAW=3`: like 2 for single draws and indexed
/// draws without a count whose record the CPU can read, that no shader was
/// declared to write in this frame or the two before (nor this list so
/// far), and that look sane ([`record_sane`]); every other indirect call
/// goes the clamped way (mesh records too).
fn raw_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_RAW").is_some_and(|v| v == "3"))
}

/// (Neptune) Whether the record of a draw (`kind` 0) or indexed draw (1) at
/// title address `at` asks for something that can run: counts up to 2^24
/// (their product up to 2^26) and starts below 2^28. `None` if it cannot be
/// read.
/// shortcut: the buffers the starts index are not looked at, and the record
/// is read now, not when the list runs.
fn record_sane(kind: u32, at: u64) -> Option<bool> {
    let mut b = [0u8; 20];
    let n = if kind == 1 { 20 } else { 16 };
    let _kind = crate::readback::consumers::scope(crate::readback::consumers::Kind::Records);
    if !crate::host::read_into(at as *const u8, &mut b[..n]) {
        return None;
    }
    let w = |i: usize| u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
    const CAP: u32 = 1 << 24;
    const START: u32 = 1 << 28;
    let (count, instances) = (w(0), w(1));
    let starts = if kind == 1 { w(2).max((w(3) as i32).unsigned_abs()).max(w(4)) } else { w(2).max(w(3)) };
    Some(count <= CAP && instances <= CAP && u64::from(count) * u64::from(instances) <= 1 << 26 && starts < START)
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// The host state the bound depth view needs ([`depth_state`]).
    depth_state: u32,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the title set that goes straight onto the host list, kept for
    /// the host list that continues this one after a cut ([`State::replay`]):
    /// whether a topology was set, the render target and depth handles, the
    /// viewports, the scissors, the blend factor and the stencil reference.
    topology_set: bool,
    target_handles: Vec<u64>,
    depth_handle: u64,
    targets_set: bool,
    /// (fxceless a35c557) The formats of the views [`State::set_targets`] bound on the
    /// host list: a draw's pipeline is one for them (`crate::pipeline::host_pipeline_for`).
    target_formats: crate::pipeline::host::TargetFormats,
    viewports: Vec<D3D12_VIEWPORT>,
    scissors: Vec<RECT>,
    blend: Option<[f32; 4]>,
    stencil: Option<u32>,
    depth_bounds: Option<[f32; 2]>,
    /// The indirect records clamped last, if later draws may still read
    /// them ([`windows`]).
    window: Option<Window>,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, slot count, contents) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u32, u64), u32>,
    /// (Neptune) Of those, the ones whose views were made ready for a draw
    /// (`readable`) with nothing to do again while the host list's count of state
    /// changes stands: the host list, that count, and the textures the table shows.
    /// A draw that only tests its depth target and reads it too is not among them.
    ready_tables: std::collections::HashMap<(u64, u32, u64), (usize, u64, Vec<usize>)>,
    /// (Neptune) The tables (address, slot count, contents) this list has seen, with tables
    /// told apart by their contents alone (`tables_by_content`): one at a new address whose
    /// contents another had takes that one's host slots, and only its own title slots'
    /// notes are made here.
    seen_tables: std::collections::HashSet<(u64, u32, u64)>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list has run, is reset or released.
    dynamic: Vec<(usize, u32, u32)>,
    /// Root SRVs bound as one-view tables in this list: (title heap, title
    /// address; 0 for none) → dynamic slot, and the host copy the slot
    /// shows: later draws of the list bring that one up to date, whatever
    /// copy the address would find by then (`shadow.rs` `tails`).
    root_srv_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// Root UAVs bound as one-view tables in this list, likewise.
    root_uav_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// The list ran and its dynamic slots were given back.
    tables_gone: bool,
    /// The host list's budget charged by arena records issued from the CPU
    /// (`crate::indirect::arena`), as the clamp charges its own.
    arena_budget: u64,
    /// (P3) What the host list recording was given, for [`bind_filter`].
    issued: Issued,
    /// (P3) What its draws' root views and buffers were found at ([`Looked`]).
    looked: Looked,
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.dynamic {
            crate::descriptor::release_dynamic(heap, first, n);
        }
    }
}

impl State {
    /// The list's submission has run (submissions are waited for): the
    /// dynamic slots its tables were made in are given back, not kept
    /// until the title resets the list. A frame of the opening cutscene
    /// asks for 12,500 tables; held until the resets they ran the 131,072
    /// slots out, and up to 4,900 draws a frame went without their tables.
    /// shortcut: a list the title runs a second time without a reset then
    /// reads slots others may have taken (logged once).
    pub(crate) fn tables_done(&mut self) {
        if self.tables_gone && self.dynamic.is_empty() {
            return;
        }
        if self.tables_gone {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("a list ran again after its descriptor tables were given back"));
        }
        for (heap, first, n) in self.dynamic.drain(..) {
            crate::descriptor::release_dynamic(heap, first, n);
        }
        self.fresh_tables.clear();
        self.ready_tables.clear();
        self.seen_tables.clear();
        self.root_srv_slots.clear();
        self.root_uav_slots.clear();
        self.tables_gone = true;
    }
}

/// An experiment on the title's scissor rectangles (`VICTORY_SCISSOR_TEST=full`):
/// they are all widened to the whole target. The title's room at
/// the start of the prologue lost its walls in five runs of seven, and in
/// those runs the player's body and the world outside were cut by one
/// rectangle around the window (run 85, frame 2046): one run with this on
/// says whether the walls are drawn and cut away by such a rectangle, or
/// not drawn at all. Off by default: a rectangle the title sets is part of
/// its picture.
fn scissor_test() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_SCISSOR_TEST").as_deref() == Ok("full"))
}

/// An experiment on mesh draws (`VICTORY_MESH_TEST`): 1 for `skip` (none
/// is issued), 2 for `scissor` (each runs its shaders but rasterises one
/// pixel), 0 otherwise. It tells a card hung by a mesh shader from one hung
/// by what the shader's triangles cover.
fn mesh_test() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_MESH_TEST").as_deref() {
        Ok("skip") => 1,
        Ok("scissor") => 2,
        _ => 0,
    })
}

/// `VICTORY_MESH_TEST=keep:<hex digits>[:<hex digits>]`: only the mesh
/// draws whose mesh shader's hash ends in one of the first digits (and
/// whose digit before it is one of the second, if given) are issued: a
/// search for the shaders that stop the card.
fn mesh_kept(pipeline: usize) -> bool {
    static KEEP: std::sync::OnceLock<Option<(String, String)>> = std::sync::OnceLock::new();
    let Some((last, before)) = KEEP.get_or_init(|| {
        let value = std::env::var("VICTORY_MESH_TEST").ok()?;
        let mut parts = value.strip_prefix("keep:")?.split(':');
        Some((parts.next()?.to_owned(), parts.next().unwrap_or("").to_owned()))
    }) else {
        return true;
    };
    let Some(p) = crate::pipeline::host_pipeline(pipeline as This) else {
        return true;
    };
    let Some(at) = p.name.find("mesh ") else {
        return true;
    };
    let hash = p.name[at + 5..].as_bytes();
    if hash.len() < 16 {
        return true;
    }
    last.as_bytes().contains(&hash[15]) && (before.is_empty() || before.as_bytes().contains(&hash[14]))
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
/// A root SRV's dynamic slot in a list, and the host copy it shows.
type RootSlot = (u32, Option<std::sync::Arc<crate::shadow::Shadow>>);

/// Buffer views of the heap whose slot showed another host buffer than
/// the copy the pass brought up to date for them, since the start
/// (diagnostics; for the memory line). It stays 0 while the pass and the
/// slots agree; with `VICTORY_VIEW_EXTENT=0` it counts the views that were
/// read stale.
#[cfg(feature = "diagnostics")]
pub(crate) static HEAP_VIEWS_ELSEWHERE: AtomicU64 = AtomicU64::new(0);
/// Passes over the view heap's buffer views for shaders that pick them by
/// index ([`State::heap_buffers`]), and the microseconds they took.
pub(crate) static HEAP_BUFFER_PASSES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_BUFFER_MICROS: AtomicU64 = AtomicU64::new(0);
/// A buffer view of the view heap that is an unordered access one, and the
/// host copy it shows: what a shader storing through picked views may
/// write ([`crate::heap_marks`]).
type HeapUnordered = (crate::heap_marks::Slot, std::sync::Arc<crate::shadow::Shadow>);
/// Per host view heap (its address): its unordered access buffer views, as
/// the last pass over the heap found them ([`State::heap_buffers`]), for
/// [`State::heap_buffer_writes`].
#[allow(clippy::type_complexity)]
static HEAP_UNORDERED: std::sync::Mutex<
    Option<std::collections::HashMap<usize, std::sync::Arc<Vec<HeapUnordered>>>>,
> = std::sync::Mutex::new(None);
/// Per host view heap (its address): the spans of host copies its buffer
/// views show, as the last pass found them, for [`heap_every_list`].
#[allow(clippy::type_complexity)]
static HEAP_SPANS: std::sync::Mutex<
    Option<std::collections::HashMap<usize, Vec<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)>>>,
> = std::sync::Mutex::new(None);
/// Refreshes of those spans in lists after the frame's first
/// ([`heap_every_list`]), since the last frame-rate line.
pub(crate) static HEAP_LATE_REFRESHES: AtomicU64 = AtomicU64::new(0);

/// Whether every list that runs a shader picking buffers by index brings
/// the heap's buffer views up to date, not only the frame's first
/// (`VICTORY_HEAP_EVERY_LIST=1`; off by default until a run prices it;
/// ported from the all-forks tree, a498c0c).
///
/// On the console a shader reads the title's memory as it is when the
/// shader runs. The pass over the heap's views runs once a frame, so what
/// the title writes into such buffers after the frame's first list is seen
/// a frame late. With the switch, later lists of the frame refresh the
/// spans the frame's pass found, for memory that is write-watched or
/// guarded (only pages the title wrote are copied); memory that is neither
/// is still copied once a frame.
fn heap_every_list() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_EVERY_LIST").is_some_and(|v| v == "1"))
}
/// Of [`HEAP_BUFFER_MICROS`], the microseconds of the pass over textures
/// picked by index that runs inside it (`heap_textures`), and the buffer
/// slots the pass over buffer views looked at; since the last frame line.
pub(crate) static HEAP_PASS_PARTS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Textures brought up to date for shaders that pick them by index
/// ([`State::heap_textures`]), the bytes of their memory, and the slots
/// still waiting.
pub(crate) static HEAP_TEXTURES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURE_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURES_WAITING: AtomicU64 = AtomicU64::new(0);
/// Passes that made no texture because the card had no room
/// (`crate::resource::room`), for the memory line.
pub(crate) static HEAP_TEXTURES_FULL: AtomicU64 = AtomicU64::new(0);

/// Frames after a whole upload of a texture picked by index at which its
/// memory is looked at again, each counted from the look before.
const RECHECK_FRAMES: [u64; 5] = [4, 12, 40, 120, 400];

/// Looks at uploaded textures picked by index that found the memory as it
/// was, and that found it changed (uploaded again), since the start.
pub(crate) static HEAP_TEXTURES_RECHECKED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Uploads of textures already on the card made while it had no room for
/// new ones, since the start.
pub(crate) static HEAP_TEXTURES_REUPLOADED_FULL: AtomicU64 = AtomicU64::new(0);

/// Whether a pass over the textures picked by index on a full card holds
/// back the slots that need a new host copy (`VICTORY_FULL_CARD_PASS=0`:
/// no, as before: they stay due, each pass looks at the same ones again
/// and stops at the 257th). Held back, such a slot is looked at again
/// after about four seconds or as soon as the card has room, and the pass
/// goes on to the slots behind it: the ones whose host copy exists and
/// only wants bringing up to date, and the ones whose memory went (letting
/// their host copies go is what makes room). What is uploaded while the
/// card has room is the same either way.
fn full_card_pass() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FULL_CARD_PASS").is_none_or(|v| v != "0"))
}

/// Whether a texture that has its host copy is still brought up to date
/// while the card has no room for new ones (`VICTORY_REUPLOAD_WHEN_FULL=0`:
/// no).
fn reupload_when_full() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REUPLOAD_WHEN_FULL").is_none_or(|v| v != "0"))
}

/// Whether textures picked by index are looked at again after their upload
/// (`VICTORY_HEAP_RECHECK=0`: no).
fn heap_recheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_RECHECK").is_none_or(|v| v != "0"))
}
/// Times a placed texture's slot was passed over because its memory is not
/// mapped.
pub(crate) static HEAP_TEXTURES_AWAY: AtomicU64 = AtomicU64::new(0);
/// Slots the pass found up to date already: their texture is uploaded by
/// the draws that read it through a table (it has a staging buffer).
pub(crate) static HEAP_TEXTURES_TABLED: AtomicU64 = AtomicU64::new(0);
/// The queue of textures picked by index at the last pass
/// (`TextureQueue::census`): slots in it, due, waiting for their memory,
/// uploaded in part, and the frames the oldest due one has waited.
pub(crate) static HEAP_TEXTURES_QUEUE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// The same queue's ages in milliseconds at the last pass: since the write
/// of the oldest written due slot, and the longest a due slot has waited
/// unserved; and, since the last frame line, the slots handed out ahead of
/// the others because they had been due for `VICTORY_TEXTURE_MAX_AGE_MS`
/// ([`texture_max_age_ms`]).
pub(crate) static HEAP_TEXTURES_AGE: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Milliseconds of the pass's work over textures picked by index per
/// second of real time (`VICTORY_TEXTURE_BUDGET_MS_PER_S`, 500; 0: the
/// budget per frame alone, as before).
///
/// The pass had its bytes and its time per frame (192 MB, 250 ms), so at
/// the 9 frames a second of a world streaming in textures that had
/// arrived waited thousands of frames, minutes of real time, while the
/// title streams by its frame rate. Taken over a second at 30 frames
/// those numbers are 5.8 GB and 7.5 s, more than a second holds: they
/// were never the pass's share of the time, only a ceiling on one frame.
/// They stay that ceiling, for all the passes of a frame together; under
/// it the pass gets what the clock gives it at this rate (half the time
/// of the thread recording frames: at 30 frames a second 17 ms a frame,
/// at 9 a second 55 ms), however slow the frames, and never a slow
/// frame's credit on top of the ceiling (`VICTORY_TEXTURE_CATCHUP`'s three
/// times the bytes made the world 0.2 frames a second for minutes in
/// allforks: with this budget on the catch-up is not used). Slots due for
/// longer than [`texture_max_age_ms`] are served past this rate, up to
/// the ceiling.
pub(crate) fn texture_budget_ms_per_s() -> u64 {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_BUDGET_MS_PER_S").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(500).min(1000)
    })
}

/// The longest a slot due in the queue of textures picked by index waits
/// before it is served ahead of all others and past
/// [`texture_budget_ms_per_s`] (`VICTORY_TEXTURE_MAX_AGE_MS`, 2000; 0:
/// never ahead for its age). Still within the ceiling per frame, so with
/// more of them than a frame's ceiling takes, the longest due go first and
/// the rest a frame later.
pub(crate) fn texture_max_age_ms() -> u64 {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| std::env::var("VICTORY_TEXTURE_MAX_AGE_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2000))
}
/// Since the start: looks at a slot uploaded in part that found no more of
/// its memory mapped; uploads of textures not all mapped; and of those,
/// the ones whose first page was not there (asked by the first page alone,
/// as before, they counted as not there at all).
pub(crate) static HEAP_TEXTURES_PARTLY: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Counts a draw not run on the host by its reason, and marks the last
/// traced draw (diagnostics).
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    #[cfg(feature = "diagnostics")]
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// Compute dispatches since the last frame line: direct ones of more than
/// 65,535 groups in x run in parts, and the parts they ran in; direct ones of
/// more given to the host whole, past what it runs (their program takes no
/// first group, `group_base`, or `VICTORY_DISPATCH_PARTS=0`); and indirect
/// ones executed in parts (`crate::indirect::bound_split`; whether a second
/// part has groups only the card knows, which counts those itself).
pub(crate) static DISPATCH_PARTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB
/// (cut at their buffer's end), other views as all that is left of their
/// buffer. (They were taken as 1 MiB: 1,452 of the 2,021 mesh vertex
/// streams of a cutscene frame reach further, and what lay past the first
/// megabyte was never brought up to date - meshes without bodies, exploded
/// triangles. `VICTORY_ROOT_VIEW_MB` sets a limit again.)
/// With whether the answer may be used again within the recording
/// ([`Looked`]): only the plain one, a host copy brought up to date (not a
/// fill into the list's upload memory or a copy on the card, made per draw).
fn root_view(list: &HostList, address: u64, srv: bool) -> Option<(u64, bool)> {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("VICTORY_ROOT_VIEW_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(u64::MAX, |mb| mb << 20)
    });
    let shadow = crate::shadow::find(address, 1)?;
    let len = shadow
        .left(address)
        .min(if srv { limit } else { 1 << 16 });
    let gpu = shadow.gpu(address);
    if srv || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        return Some((gpu, true));
    }
    // As table constant buffers: bytes a flight in the air may write are
    // copied from the host copy on the card, behind the flight, instead of
    // filled from title memory at submission (`VICTORY_FILL_ON_CARD=0`
    // fills as before; Trexies 9dd3e8f).
    let range = (address as usize, (address + len) as usize);
    if list.fill_wants_card(range.0, range.1) {
        use crate::host::read_wait::{NC_HOLDS, NC_ROOM, why};
        if list.card_holds(&shadow, range.0, range.1) {
            shadow.refresh(list, address, len);
            if let Some(gpu) = list.card_constants(&shadow, address, len) {
                crate::host::ON_CARD[1].fetch_add(1, Ordering::Relaxed);
                why(0);
                return Some((gpu, false));
            }
            why(NC_ROOM);
        } else {
            why(NC_HOLDS);
        }
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some((gpu, false))
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

/// (P6) Microseconds of the pass over buffer views made aside
/// (`VICTORY_HEAP_PASS_ASIDE`), since the last frame line.
pub(crate) static HEAP_ASIDE_MICROS: AtomicU64 = AtomicU64::new(0);

/// The pass over the buffer views of `heap` that shaders pick by index
/// ([`State::heap_buffers`]): their host slots are made, and the host copies
/// they show are to be refreshed (host copy, address, bytes; cut as
/// `Shadow::refresh` cuts them). `hash`: the recording's shader, for the
/// note of a refused view. (P6) The slots are looked at on several threads
/// (`VICTORY_VIEW_THREADS`; 1: one after another, as before): each slot's
/// host descriptor is its own, made as recording threads make them side by
/// side; what is kept from pass to pass is changed here alone, in slot
/// order, after all are looked at.
fn heap_buffer_pass(
    heap: &crate::descriptor::Heap,
    host: &crate::descriptor::host::HostHeap,
    frame: u64,
    hash: u64,
) -> Vec<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    let started = std::time::Instant::now();
    // Per host heap: the host copy and range each slot's buffer view
    // reads, kept from pass to pass. Only the slots written since the
    // last pass are looked at again (83,000 slots a pass otherwise, 0.1 s
    // at best); all of them every 16 passes, because a view of a
    // released resource goes stale without a write, and a slot written
    // before its memory had a host copy is only made then.
    type Ranges = std::collections::HashMap<u32, (std::sync::Arc<crate::shadow::Shadow>, u64, u64, bool)>;
    static RANGES: std::sync::Mutex<Option<std::collections::HashMap<usize, (u64, Ranges)>>> =
        std::sync::Mutex::new(None);
    let mut kept = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let (passes, ranges) = kept
        .get_or_insert_with(Default::default)
        .entry(std::ptr::from_ref(host) as usize)
        .or_insert_with(|| (0, Ranges::new()));
    let changed = host.buffer_changes();
    let slots = if crate::heap_pass::marked() {
        // (fxceless 895a23b, `VICTORY_HEAP_PASS_MARKED`) Only the marked.
        let had_copy: Vec<u32> = ranges.keys().copied().collect();
        let all = host.buffer_slots();
        let records = heap_records(heap, &all);
        let (slots, full) =
            crate::heap_pass::slots(std::ptr::from_ref(host) as usize, *passes, changed, &had_copy, || all, |i| records.get(&i).copied());
        if full {
            ranges.clear();
        }
        slots
    } else if *passes % 16 == 0 {
        ranges.clear();
        host.buffer_slots()
    } else {
        changed
    };
    *passes += 1;
    // Whether the device was there before the pass: a view that removes
    // it is named below.
    static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let alive = std::sync::atomic::AtomicBool::new(!GONE.load(Ordering::Relaxed) && crate::descriptor::host::device_removed().is_none());
    HEAP_PASS_PARTS[1].fetch_add(slots.len() as u64, Ordering::Relaxed);
    for index in &slots {
        ranges.remove(index);
    }
    const AT_A_TIME: usize = 256;
    type Found = (u32, std::sync::Arc<crate::shadow::Shadow>, u64, u64, bool);
    let parts: Vec<std::sync::Mutex<Vec<Found>>> = (0..slots.len().div_ceil(AT_A_TIME)).map(|_| Default::default()).collect();
    /// The title heap, for the threads of `crate::views`. SAFETY: what is
    /// read of it (`handle`, its range) is set when it is made; the
    /// recording threads read it side by side through the heap list.
    struct Shared<'a>(&'a crate::descriptor::Heap);
    unsafe impl Sync for Shared<'_> {}
    impl<'a> Shared<'a> {
        fn get(&self) -> &'a crate::descriptor::Heap {
            self.0
        }
    }
    let shared = Shared(heap);
    crate::views::each(parts.len(), 2, &|part| {
        let heap = shared.get();
        let mut found = Vec::new();
        let first = part * AT_A_TIME;
        for (k, &index) in slots[first..(first + AT_A_TIME).min(slots.len())].iter().enumerate() {
            let n = first + k;
            let Some(handle) = heap.handle(index) else {
                continue;
            };
            let Some(view) = crate::descriptor::host::not_changed_here(|| host.mirror(index, handle, true)) else {
                continue;
            };
            // Every 64 views while the device lives (a cheap query).
            if n % 64 == 63
                && alive.load(Ordering::Relaxed)
                && let Some(why) = crate::descriptor::host::device_removed()
                && alive.swap(false, Ordering::Relaxed)
            {
                GONE.store(true, Ordering::Relaxed);
                let last: Vec<String> = slots[n - 63..=n]
                    .iter()
                    .filter_map(|&i| crate::descriptor::view(heap.handle(i)?).map(|v| format!("{i}: {v:?}")))
                    .map(|s| s.chars().take(200).collect())
                    .collect();
                tracing::error!(why, slot = index, "the host device went while buffer views of the heap were made; one of: {last:#?}");
                crate::host::log_removal_context();
            }
            // The copy the slot shows (`buffer_copy`), not the one the
            // view's range alone would find.
            let Some((shadow, address, len)) = crate::descriptor::host::buffer_copy(&view) else {
                continue;
            };
            let unordered = matches!(view, View::Unordered(..) | View::Placed(true, ..));
            if let Some(own) = crate::shadow::flagged(address, len) {
                crate::shadow::flagged_use(hash, handle, unordered, own, address, len, true);
            }
            #[cfg(feature = "diagnostics")]
            if host.shows(index, &shadow.buffer) == Some(false) {
                HEAP_VIEWS_ELSEWHERE.fetch_add(1, Ordering::Relaxed);
            }
            found.push((index, shadow, address, address + len, unordered));
        }
        *parts[part].lock().unwrap_or_else(|e| e.into_inner()) = found;
    });
    for part in parts {
        for (index, shadow, from, to, unordered) in part.into_inner().unwrap_or_else(|e| e.into_inner()) {
            ranges.insert(index, (shadow, from, to, unordered));
        }
    }
    let views = ranges.len() as u32;
    // Per host copy, the span its picked views cover.
    // (soldsawn 5acc7c5) Sized for the slots it is filled from.
    let mut spans: std::collections::HashMap<usize, (std::sync::Arc<crate::shadow::Shadow>, u64, u64)> =
        std::collections::HashMap::with_capacity(ranges.len().min(8192));
    for (shadow, from, to, _) in ranges.values() {
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        let span = spans.entry(key).or_insert_with(|| (shadow.clone(), *from, *to));
        span.1 = span.1.min(*from);
        span.2 = span.2.max(*to);
    }
    if heap_every_list() {
        HEAP_SPANS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(Default::default)
            .insert(std::ptr::from_ref(host) as usize, spans.values().cloned().collect());
    }
    // The heap's unordered views, for the lists that run shaders
    // storing through picked views ([`State::heap_buffer_writes`]);
    // and the marks of every buffer view read and cleared once a frame
    // (a mark on a read-only view is counted: the store was dropped).
    let unordered: Vec<HeapUnordered> = ranges
        .iter()
        .filter(|(_, r)| r.3)
        .map(|(&slot, r)| {
            let s = crate::heap_marks::Slot {
                slot,
                unordered: true,
                address: r.1,
                len: r.2 - r.1,
            };
            (s, r.0.clone())
        })
        .collect();
    let uav_views = unordered.len();
    let uav_bytes: u64 = unordered.iter().map(|u| u.0.len).sum();
    if crate::heap_marks::on() {
        let all: Vec<crate::heap_marks::Slot> = ranges
            .iter()
            .map(|(&slot, r)| crate::heap_marks::Slot {
                slot,
                unordered: r.3,
                address: r.1,
                len: r.2 - r.1,
            })
            .collect();
        let _ = crate::heap_marks::take(frame, all.iter());
        HEAP_UNORDERED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(Default::default)
            .insert(std::ptr::from_ref(host) as usize, std::sync::Arc::new(unordered));
    }
    drop(kept);
    let bytes: u64 = spans.values().map(|s| s.2 - s.1).sum();
    // Memory that is not write-watched is copied whole by every refresh.
    // It was once refreshed only every 120 frames (unmapped pages read
    // one by one cost 14 GB a frame in run 11, before they were
    // remembered): streamed geometry then drew from stale buffers, and
    // the garbage ended runs (the card stopped answering 14-90 frames
    // into the opening cutscene; with every frame it ran on). The pass
    // costs some 30 ms a frame this way.
    static WHOLE: std::sync::Mutex<Option<std::collections::HashMap<usize, u64>>> = std::sync::Mutex::new(None);
    let mut whole = WHOLE.lock().unwrap_or_else(|e| e.into_inner());
    let whole = whole.get_or_insert_with(Default::default);
    let mut refreshes = Vec::with_capacity(spans.len());
    for (shadow, from, to) in spans.values() {
        if shadow.unwatched() {
            // By the copy's number: a copy made where one let go was
            // (`shadow::let_go_idle`) has never been refreshed.
            let key = shadow.id() as usize;
            // `VICTORY_UNWATCHED_FRAMES` sets the interval (1: every frame).
            static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            let every = *EVERY.get_or_init(|| {
                std::env::var("VICTORY_UNWATCHED_FRAMES")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1)
            });
            match whole.get(&key) {
                Some(&last) if frame < last + every => continue,
                _ => {
                    whole.insert(key, frame);
                }
            }
        }
        // As `Shadow::refresh` cuts it.
        refreshes.push((shadow.clone(), *from, (to - from).min(shadow.left(*from))));
    }
    if crate::host::aside::on() {
        HEAP_ASIDE_MICROS.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    // The first pass and every growth of the set by half.
    let seen = LOGGED.load(Ordering::Relaxed);
    if u64::from(views) > seen + seen / 2 {
        LOGGED.store(u64::from(views), Ordering::Relaxed);
        tracing::info!(
            slots = slots.len(),
            views,
            copies = spans.len(),
            mb = bytes >> 20,
            uav_views,
            uav_mb = uav_bytes >> 20,
            ms = started.elapsed().as_millis(),
            "buffer views of the heap made ready for shaders that pick them by index"
        );
    }
    refreshes
}

/// (P6, `VICTORY_HEAP_PASS_MARKED`) The first 12 bytes (tag, entry,
/// generation) of the records of `slots` in title memory, read a 64 KiB
/// page at a time through the kernel (a page the title unmapped is left
/// out, as `descriptor::sync_heaps` reads).
fn heap_records(heap: &crate::descriptor::Heap, slots: &[u32]) -> std::collections::HashMap<u32, [u32; 3]> {
    let mut sorted: Vec<(u64, u32)> = slots.iter().filter_map(|&s| Some((heap.handle(s)?, s))).collect();
    sorted.sort_unstable();
    let mut out = std::collections::HashMap::with_capacity(sorted.len());
    let mut page = vec![0u8; crate::watch::PAGE + 16];
    let mut at = 0;
    while at < sorted.len() {
        let base = sorted[at].0 & !(crate::watch::PAGE as u64 - 1);
        let mut end = at;
        while end < sorted.len() && sorted[end].0 < base + crate::watch::PAGE as u64 {
            end += 1;
        }
        let last = sorted[end - 1].0 + 12;
        let bytes = &mut page[..(last - base) as usize];
        if crate::host::read_into(base as *const u8, bytes) {
            for &(handle, slot) in &sorted[at..end] {
                let o = (handle - base) as usize;
                let w = |k: usize| u32::from_le_bytes(bytes[o + 4 * k..o + 4 * k + 4].try_into().unwrap_or_default());
                out.insert(slot, [w(0), w(1), w(2)]);
            }
        }
        at = end;
    }
    out
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        self.topology_set = true;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // (P3) The host list has this topology already.
        if Issued::other(bind_filter(), &mut self.issued.at(list).topology, host as i32) {
            return;
        }
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.target_handles.clear();
        self.target_handles.extend_from_slice(handles);
        self.depth_handle = depth;
        self.targets_set = true;
        self.targets.clear();
        // (fxceless a35c557) The formats of the host views, for the draws' pipelines.
        let mut formats = [0; 9];
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                // A typed null view (or a target without a host copy) still
                // takes its slot: left out, the next live target would move
                // into it and the pixel shader's outputs reach the wrong
                // targets.
                // shortcut: a view without a description or a format has no
                // null view to stand for it and is left out, as before.
                let null = record
                    .desc()
                    .filter(|d| d.Format.0 != 0 && d.ViewDimension.0 != 0);
                if let Some(desc) = null {
                    if rtvs.len() < 8 {
                        formats[rtvs.len()] = desc.Format.0 as u32;
                    }
                    rtvs.push(host.rtv(None, Some(&desc)));
                }
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            if rtvs.len() < 8 {
                formats[rtvs.len()] = bound_format(texture, record.desc().map(|d| d.Format));
            }
            rtvs.push(host.rtv(Some(texture), record.desc().as_ref()));
        }
        self.depth = 0;
        self.depth_state = DEPTH_WRITE;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            self.depth_state = depth_state(texture, record.depth_desc().as_ref());
            formats[8] = bound_format(texture, record.depth_desc().map(|d| d.Format));
            if crate::resource::depth_and_unordered(record.resource) {
                crate::resource::DEPTH_UNORDERED[1].fetch_add(1, Ordering::Relaxed);
            }
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some(host.dsv(texture, record.depth_desc().as_ref()))
        });
        self.target_formats = formats;
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// Viewports, scissors, the blend factor and the stencil reference: set
    /// on the host list at once, and kept for [`State::replay`].
    pub(crate) fn set_viewports(&mut self, list: &HostList, viewports: &[D3D12_VIEWPORT]) {
        self.viewports.clear();
        self.viewports.extend_from_slice(viewports);
        // (P3) By their bits: the host list has these already.
        let bits: Vec<[u32; 6]> = viewports
            .iter()
            .map(|v| [v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth].map(f32::to_bits))
            .collect();
        if Issued::other(bind_filter(), &mut self.issued.at(list).viewports, bits) {
            return;
        }
        crate::draw::viewports(list, viewports);
    }

    pub(crate) fn set_scissors(&mut self, list: &HostList, rects: &[RECT]) {
        self.scissors.clear();
        self.scissors.extend_from_slice(rects);
        // (P3) As the title gave them (the experiment below widens them the
        // same way each time): the host list has these already.
        let given: Vec<[i32; 4]> = rects.iter().map(|r| [r.left, r.top, r.right, r.bottom]).collect();
        if Issued::other(bind_filter(), &mut self.issued.at(list).scissors, given) {
            return;
        }
        if scissor_test() {
            // The experiment: every rectangle the title sets is widened to
            // the largest target there is, so nothing is cut by one. What
            // the listing shows (`op_text`) stays the title's own.
            let full = RECT {
                left: 0,
                top: 0,
                right: 16384,
                bottom: 16384,
            };
            crate::draw::scissors(list, &vec![full; rects.len()]);
            return;
        }
        crate::draw::scissors(list, rects);
    }

    pub(crate) fn set_blend_factor(&mut self, list: &HostList, factor: Option<&[f32; 4]>) {
        self.blend = factor.copied();
        let bits = factor.map(|f| f.map(f32::to_bits));
        if Issued::other(bind_filter(), &mut self.issued.at(list).blend, bits) {
            return;
        }
        crate::draw::blend_factor(list, factor);
    }

    pub(crate) fn set_stencil_ref(&mut self, list: &HostList, reference: u32) {
        self.stencil = Some(reference);
        if Issued::other(bind_filter(), &mut self.issued.at(list).stencil, reference) {
            return;
        }
        crate::draw::stencil_ref(list, reference);
    }

    pub(crate) fn set_depth_bounds(&mut self, list: &HostList, min: f32, max: f32) {
        self.depth_bounds = Some([min, max]);
        if Issued::other(bind_filter(), &mut self.issued.at(list).depth_bounds, [min.to_bits(), max.to_bits()]) {
            return;
        }
        crate::draw::depth_bounds(list, min, max);
    }

    /// `list` is a new host list that continues the title's list (it was
    /// cut at a wait for a memory value, `crate::list::cut`): a new host
    /// list has nothing set, so what the title set so far is set again.
    /// The pipeline, the root signatures and the heaps are set by the next
    /// draw or dispatch (`bind`, which is told here that the host list has
    /// none); root arguments, vertex and index buffers are set by every
    /// draw anyway. The dynamic descriptor slots and the tables made in
    /// them stay: the parts before the cut have not run yet, and they are
    /// given back when the title's list is reset.
    /// shortcut: the render target views are made again from the handles'
    /// descriptors as they are now, not as they were when the title set
    /// the targets (a descriptor rewritten in between names another view).
    pub(crate) fn replay(&mut self, list: &HostList) {
        self.host_pipeline = 0;
        self.host_roots = [0; 2];
        self.host_heaps = [0; 2];
        // (P3) A new host list holds nothing (its own recording number says so too).
        self.issued = Issued::default();
        if self.topology_set {
            self.set_topology(list, self.topology);
        }
        if self.targets_set {
            let handles = std::mem::take(&mut self.target_handles);
            self.set_targets(list, &handles, self.depth_handle);
        }
        if !self.viewports.is_empty() {
            crate::draw::viewports(list, &self.viewports);
        }
        if !self.scissors.is_empty() {
            let rects = std::mem::take(&mut self.scissors);
            self.set_scissors(list, &rects);
        }
        if let Some(factor) = self.blend {
            crate::draw::blend_factor(list, Some(&factor));
        }
        if let Some(reference) = self.stencil {
            crate::draw::stencil_ref(list, reference);
        }
        if let Some([min, max]) = self.depth_bounds {
            crate::draw::depth_bounds(list, min, max);
        }
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            // The host's own ([`Self::host_word`]): no root argument.
            (_, RootType::Other(_)) => 0,
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// A user-data dword the host computes for a translated shader
    /// (`victory_shader::emit::USER_MEMORY_BASE`, `USER_VIEW_ADDRESS`).
    fn host_word(
        &self,
        compute: bool,
        kind: u32,
        slot: u8,
        dword: u32,
        memory: Option<&crate::memory::Ready>,
    ) -> u32 {
        use victory_shader::emit::{USER_GROUP_BASE, USER_MEMORY_BASE, USER_VIEW_ADDRESS, USER_VIEW_BIAS};
        // Set between the parts of a large dispatch (`group_base`).
        if kind == USER_GROUP_BASE {
            return 0;
        }
        if kind == USER_MEMORY_BASE {
            return memory.map_or(0, |m| m.base);
        }
        if kind == USER_VIEW_BIAS {
            // The view begins on 16 bytes of the copy the address finds
            // (`root_uav`, `root_srv_table`).
            // shortcut: found again here, not taken from the list's slot.
            return match self.roots_ref(compute).get(usize::from(slot)) {
                Some(&Root::View(address)) => crate::shadow::find(address, 1).map_or(0, |s| {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    ((s.gpu(address) - base) & 15) as u32
                }),
                _ => 0,
            };
        }
        if kind != USER_VIEW_ADDRESS && kind != USER_VIEW_ADDRESS + 1 {
            once!(kind, "a shader reads a host value this build does not know");
            return 0;
        }
        // The title address of the buffer view at that dword of the table.
        let address = match self.roots_ref(compute).get(usize::from(slot)) {
            Some(&Root::Table(table)) => crate::descriptor::view(table + u64::from(dword) * 4)
                .and_then(|view| crate::descriptor::host::buffer_range(&view))
                .map_or(0, |range| range.0),
            _ => 0,
        };
        if address == 0 {
            once!(slot, dword, "a shader takes the address of a buffer view that is not there: it reads 0");
        }
        if kind == USER_VIEW_ADDRESS {
            address as u32
        } else {
            (address >> 32) as u32
        }
    }

    /// For a shader that reads title memory by address
    /// ([`crate::memory`]): its directory, fault notes and view slots,
    /// with the pages of the root views the list has set made ready.
    fn memory_ready(&self, list: &HostList, compute: bool, writes: bool) -> Option<crate::memory::Ready> {
        let heap = match self.heaps[0] {
            0 => None,
            h => crate::descriptor::heap(h as This).host(),
        }?;
        let (mut addresses, mut constants) = (Vec::new(), Vec::new());
        for root in self.roots_ref(compute) {
            match root {
                Root::View(address) => addresses.push(*address),
                Root::Constants(values) => constants.extend_from_slice(values),
                _ => {}
            }
        }
        crate::memory::ready(list, heap, &addresses, &constants, writes)
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline_for(self.pipeline as This, &self.target_formats) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            return false;
        }
        if p.rects && self.topology != RECTANGLES {
            // Its geometry stage would turn each triangle into a rectangle.
            once!("a rectangle pipeline drawn with another topology");
        }
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let found = self.looked.get(list, (v.address, u64::from(v.size), LOOK_VERTEX), || {
                let shadow = crate::shadow::find_any(v.address, u64::from(v.size))?;
                shadow.refresh(list, v.address, u64::from(v.size));
                Some((shadow.gpu(v.address), true))
            });
            let Some(gpu) = found else {
                continue;
            };
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // (P3) The host list holds these views already ([`bind_filter`]).
        let same = count > 0 && self.issued.vertex_in(list.serial(), bind_filter(), &views, count);
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 && !same {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let found = self.looked.get(list, (v.address, u64::from(v.size), LOOK_INDEX), || {
                let shadow = crate::shadow::find_any(v.address, u64::from(v.size))?;
                shadow.refresh(list, v.address, u64::from(v.size));
                Some((shadow.gpu(v.address), true))
            });
            let Some(gpu) = found else {
                return false;
            };
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            let given = (view.BufferLocation, view.SizeInBytes, view.Format.0);
            let same = self.issued.index_in(list.serial(), bind_filter(), given);
            // SAFETY: an open list; the view names a live host buffer.
            if !same {
                unsafe { l.IASetIndexBuffer(Some(&view)) };
            }
        }
        const RENDER_TARGET: u32 = 0x4;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(self.depth as This, false, true);
            // A view that writes neither plane keeps the texture readable:
            // the depth read state, together with the shader read states
            // if this draw's shaders read the same texture (the write state
            // put it out of the state those views need, and a move to it
            // and back for every such draw).
            let state = if self.depth_state == DEPTH_READ {
                let (tested, read) = TESTED_DEPTH.get();
                if read && tested == self.depth {
                    DEPTH_READ | SHADER_RESOURCE
                } else {
                    DEPTH_READ
                }
            } else {
                DEPTH_WRITE
            };
            crate::list::ensure_state(list, self.depth as This, state);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a root SRV it reads
    /// could not be given a view.
    pub(crate) fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        crate::readback::terrain::op(&p.name, compute);
        let l = &list.list;
        // Textures picked by index, before anything asks for them.
        self.early_textures(list, p, compute);
        if crate::host::walls_trace() {
            self.walls_census(p, compute);
        }
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        // The depth target a draw only tests, for the views of it that the
        // draw's shaders read (`readable`).
        let tested = if !compute && self.depth_state == DEPTH_READ {
            self.depth
        } else {
            0
        };
        TESTED_DEPTH.set((tested, false));
        STAGE_HASH.set(0);
        crate::declstat::set_writer("");
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
            // (P3) D3D12 drops the root arguments of this kind.
            self.issued.at(list).roots_gone(compute);
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        crate::readback::consumers::clear_writer();
        // (fxceless 5d659e0) `VICTORY_TABLE_STAGES`: what this draw's stages read of each table.
        STAGE_SPANS.with_borrow_mut(|(spans, staged)| {
            *spans = if table_stages() { stage_spans(p) } else { Vec::new() };
            staged.clear();
        });
        for stage in &p.stages {
            let t = stage.translated();
            STORES.set(t.stores);
            if t.stores {
                crate::readback::consumers::set_writer(&stage.shader.hash);
                // (Prefault job) The declarations meter's writer.
                crate::declstat::set_writer(&stage.shader.hash);
            }
            if crate::shadow::any_flagged() {
                STAGE_HASH.set(u64::from_str_radix(&stage.shader.hash, 16).unwrap_or(0));
            }
            // (fxceless #927) The table places this stage moves 16-bit halves through.
            HALF_PLACES.with_borrow_mut(|h| {
                h.clear();
                h.extend_from_slice(&t.host.half_views);
            });
            if !t.host.half_views.is_empty() {
                note_half_views(&p.name, &stage.shader.hash, &t.host.half_views);
            }
            let made = self.tables(list, &t.used, &t.host.root_indexed_views, compute);
            STORES.set(true);
            if made.iter().any(|m| m.1 == u32::MAX) {
                once!("a draw whose descriptor table found no free slots is dropped");
                // (fxceless 5d659e0) Which draws those are, in the frame line's list of drops.
                #[cfg(feature = "diagnostics")]
                crate::pipeline::note_why(format!("no free dynamic slots for a table of {}", p.name));
                return false;
            }
            // What the shader stores through root views and raw addresses is
            // copied back into title memory afterwards.
            let writes = (t.stores || !write_skip())
                && !crate::pipeline::builder_quarantined(self.pipeline as This);
            // Title memory by address: what the shader finds it by.
            let memory = if t.host.root_srvs.contains(&victory_shader::emit::SLOT_MEMORY) {
                let Some(ready) = self.memory_ready(list, compute, writes) else {
                    once!("a shader that reads title memory by address without a view heap to hold its views: the draw is dropped");
                    return false;
                };
                Some(ready)
            } else {
                None
            };
            if let Some((param, dwords)) = stage.user_data {
                // (P3) On the stack for the usual sizes: one allocation less a stage.
                let mut stack = [0u32; 64];
                let mut heap = Vec::new();
                let values: &mut [u32] = if dwords as usize <= stack.len() {
                    &mut stack[..dwords as usize]
                } else {
                    heap.resize(dwords as usize, 0);
                    &mut heap
                };
                for u in &t.host.user_data {
                    if let RootType::Other(kind) = u.root {
                        values[u.index as usize] =
                            self.host_word(compute, kind, u.slot, u.dword, memory.as_ref());
                        continue;
                    }
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                // (P3) The host list holds these words in this parameter already.
                if !self.issued.words(list, compute, param, values) {
                    // SAFETY: an open list; `values` covers the parameter.
                    unsafe {
                        if compute {
                            l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                        } else {
                            l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                        }
                    }
                }
            }
            for (views, srv) in [(&stage.root_cbvs, false), (&stage.root_srvs, true)] {
                for &(slot, param) in views {
                    if srv && slot == victory_shader::emit::SLOT_MEMORY {
                        // Ours: the directory of title memory by address.
                        if let Some(m) = &memory
                            && !self.issued.root(list, compute, param, Given::View(GIVEN_SRV, m.directory), F_VIEWS)
                        {
                            // SAFETY: an open list; the directory lives as
                            // long as the process and the list holds it.
                            unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, m.directory);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, m.directory);
                                }
                            }
                        }
                        continue;
                    }
                    if srv && slot == victory_shader::emit::SLOT_STRIDES {
                        // Ours, not the title's: the shader picks buffers
                        // from the view heap by an index it computes.
                        match self.heap_buffers(list) {
                            Some(gpu) if self.issued.root(list, compute, param, Given::View(GIVEN_SRV, gpu), F_VIEWS) => {}
                            // SAFETY: an open list; the table lives as long
                            // as the heap the list holds.
                            Some(gpu) => unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, gpu);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, gpu);
                                }
                            },
                            None => once!("a shader picks buffers from a view heap that is not set"),
                        }
                        continue;
                    }
                    let tabled = srv && crate::pipeline::host::bounded_root_srvs();
                    let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        // A table left unset is undefined: a null view.
                        if tabled && !self.root_srv_table(list, None, param, compute, false) {
                            return false;
                        }
                        continue;
                    };
                    // The shader adds the bytes a raw view begins early by.
                    let biased = srv
                        && t.host.user_data.iter().any(|u| {
                            u.slot == slot
                                && u.root == RootType::Other(victory_shader::emit::USER_VIEW_BIAS)
                        });
                    if tabled {
                        if !self.root_srv_table(list, Some(address), param, compute, biased) {
                            return false;
                        }
                        continue;
                    }
                    let found = self.looked.get(list, (address, 0, if srv { LOOK_SRV } else { LOOK_CBV }), || root_view(list, address, srv));
                    let Some(mut gpu) = found else {
                        continue;
                    };
                    if biased {
                        gpu -= u64::from(self.host_word(
                            compute,
                            victory_shader::emit::USER_VIEW_BIAS,
                            slot,
                            0,
                            None,
                        ));
                    }
                    // (P3) The host list holds this root view in this parameter already.
                    if self.issued.root(list, compute, param, Given::View(if srv { GIVEN_SRV } else { GIVEN_CBV }, gpu), F_VIEWS) {
                        continue;
                    }
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, srv) {
                            (false, false) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, true) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (true, false) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, true) => l.SetComputeRootShaderResourceView(param, gpu),
                        }
                    }
                }
            }
            for &(slot, param) in &stage.root_uavs {
                if slot == victory_shader::emit::SLOT_HEAP_MARKS {
                    // Ours: where shaders mark the heap slots of the buffer
                    // views they pick by index and store through.
                    let Some(marks) = crate::heap_marks::bind(list, &stage.shader) else {
                        once!("a shader that stores through buffers it picks by index, and no buffer for its marks: the draw is dropped");
                        return false;
                    };
                    if !self.issued.root(list, compute, param, Given::View(GIVEN_UAV, marks), F_VIEWS) {
                        // SAFETY: an open list; the buffer lives as long as the
                        // process and the list holds it.
                        unsafe {
                            if compute {
                                l.SetComputeRootUnorderedAccessView(param, marks);
                            } else {
                                l.SetGraphicsRootUnorderedAccessView(param, marks);
                            }
                        }
                    }
                    // After the stage's root views: the pass that brings the
                    // heap's buffer views up to date has run for this list
                    // (here if the stage has no table of strides).
                    if writes {
                        let _ = self.heap_buffers(list);
                        self.heap_buffer_writes(list);
                    }
                    continue;
                }
                if slot == victory_shader::emit::SLOT_FAULTS {
                    // Ours: where shaders note the pages they found empty.
                    if let Some(m) = &memory
                        && !self.issued.root(list, compute, param, Given::View(GIVEN_UAV, m.faults), F_VIEWS)
                    {
                        // SAFETY: an open list; the buffer lives as long as
                        // the process and the list holds it.
                        unsafe {
                            if compute {
                                l.SetComputeRootUnorderedAccessView(param, m.faults);
                            } else {
                                l.SetGraphicsRootUnorderedAccessView(param, m.faults);
                            }
                        }
                    }
                    continue;
                }
                let address = match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::View(address)) => Some(address),
                    _ => {
                        once!(slot, "a shader writes through a root view the title did not set");
                        None
                    }
                };
                if !self.root_uav(list, address, param, compute, writes) {
                    return false;
                }
            }
        }
        // Depth textures it writes through unordered access: their colour
        // twins hold what they lack from here on.
        list.twins_written();
        true
    }

    /// Binds a root UAV: a raw unordered access view of the title buffer's
    /// host copy from `address` on, as a one-view table (a null view
    /// without an address, without a host copy shaders can write, or at an
    /// address not on 16 bytes), or as a root descriptor with
    /// `VICTORY_ROOT_SRV_DIRECT=1` (the dispatch is dropped then if there
    /// is nothing to bind). With `writes`, what the shader stored is copied
    /// back into title memory when the list has run. False if the draw
    /// must be dropped.
    /// shortcut: a root view has no size: the view, the refresh and the
    /// copy back cover all that is left of the title's buffer
    /// (`VICTORY_ROOT_UAV_MB` sets a limit for all three).
    fn root_uav(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        writes: bool,
    ) -> bool {
        const UNORDERED: i32 = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
        static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let limit = *LIMIT.get_or_init(|| {
            std::env::var("VICTORY_ROOT_UAV_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map_or(u64::MAX, |mb| mb << 20)
        });
        let find = |a: u64| {
            let s = crate::shadow::find(a, 1)?;
            // SAFETY: a live buffer.
            let flags = unsafe { s.buffer.GetDesc() }.Flags;
            if flags.0 & UNORDERED == 0 {
                once!("a root UAV of a buffer whose host copy shaders cannot write");
                return None;
            }
            Some(s)
        };
        if let Some(a) = address
            && crate::readback::terrain::on()
        {
            crate::readback::terrain::uav(a, 4, find(a).is_some(), writes);
        }
        let ready = |a: u64, s: &std::sync::Arc<crate::shadow::Shadow>| {
            let len = s.left(a).min(limit);
            s.refresh(list, a, len);
            if writes {
                let ctx = crate::declstat::Ctx { way: crate::declstat::ROOT_UAV, hash: 0, view: None };
                crate::declstat::with(ctx, || s.gpu_writes(list, a, len));
            }
        };
        if !crate::pipeline::host::bounded_root_srvs() {
            let Some((a, s)) = address.and_then(|a| Some((a, find(a)?))) else {
                once!("a root UAV with nothing to bind: the draw is dropped");
                return false;
            };
            ready(a, &s);
            // From the 16 bytes below: the shader adds the rest.
            // SAFETY: a live buffer.
            let base = unsafe { s.buffer.GetGPUVirtualAddress() };
            let gpu = base + ((s.gpu(a) - base) & !15);
            if self.issued.root(list, compute, param, Given::View(GIVEN_UAV, gpu), F_VIEWS) {
                return true;
            }
            // SAFETY: an open list; the address is in a live buffer.
            unsafe {
                if compute {
                    list.list.SetComputeRootUnorderedAccessView(param, gpu);
                } else {
                    list.list.SetGraphicsRootUnorderedAccessView(param, gpu);
                }
            }
            return true;
        }
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root UAV without a bound view heap: the draw is dropped");
            return false;
        };
        let key = (std::ptr::from_ref(heap) as usize, address.unwrap_or(0));
        let (index, shadow) = match self.root_uav_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| find(a).map(|s| (a, s)));
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                self.dynamic.push((key.0, index, 1));
                if let Some((a, s)) = &shadow {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    let offset = s.gpu(*a) - base;
                    // Raw views start on 16 bytes: from the 16 bytes
                    // below, and the shader adds the rest to its offsets
                    // (`victory_shader::emit::USER_VIEW_BIAS`). (An address
                    // not on 16 bytes got a null view at first: the
                    // builder's counters are at such addresses.)
                    let bias = offset & 15;
                    host.raw_uav_at(index, &s.buffer, offset - bias, s.left(*a).min(limit) + bias);
                }
                if shadow.is_none() {
                    host.null_raw_uav_at(index);
                }
                self.root_uav_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                (index, shadow)
            }
        };
        if let Some((a, s)) = &shadow {
            ready(*a, s);
        }
        let handle = host.gpu_handle(index);
        if self.issued.root(list, compute, param, Given::Table(handle.ptr), F_TABLES) {
            return true;
        }
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// Binds a root SRV as a one-view table
    /// ([`crate::pipeline::host::bounded_root_srvs`]): a raw view of the
    /// title buffer's host copy from `address` to the copy's end, or a null
    /// view (no address, no host copy, or an address not on 16 bytes). One
    /// dynamic slot per address per list, given back with the list's
    /// others. False if no slot could be had or no view heap is bound.
    fn root_srv_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        biased: bool,
    ) -> bool {
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root SRV without a bound view heap: the draw is dropped");
            return false;
        };
        // A view that begins early is another view than one that does not
        // (title addresses stay below bit 63).
        let key = (
            std::ptr::from_ref(heap) as usize,
            address.unwrap_or(0) | u64::from(biased) << 63,
        );
        let (index, shadow) = match self.root_srv_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| crate::shadow::find(a, 1).map(|s| (a, s)));
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                self.dynamic.push((key.0, index, 1));
                self.root_srv_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                match &shadow {
                    Some((a, s)) => {
                        // SAFETY: a live buffer.
                        let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                        let offset = s.gpu(*a) - base;
                        if biased {
                            // From the 16 bytes below; the shader adds the
                            // rest (`victory_shader::emit::USER_VIEW_BIAS`).
                            let bias = offset & 15;
                            host.raw_at(index, &s.buffer, offset - bias, s.left(*a) + bias);
                        } else if offset.is_multiple_of(16) {
                            host.raw_at(index, &s.buffer, offset, s.left(*a));
                        } else {
                            // Raw views start on 16 bytes; one from the
                            // aligned start would shift every read.
                            once!(
                                address = format_args!("{a:#x}"),
                                "a root SRV not on 16 bytes reads zeros"
                            );
                            host.null_raw_at(index);
                        }
                    }
                    None => host.null_raw_at(index),
                }
                (index, shadow)
            }
        };
        // As for root descriptors (`root_view`): all that is left of the
        // buffer is brought up to date. The first 1 MiB only left what lay
        // past it stale (most mesh vertex streams reach further).
        if let Some((a, s)) = &shadow {
            s.refresh(list, *a, s.left(*a));
        }
        let handle = host.gpu_handle(index);
        if self.issued.root(list, compute, param, Given::Table(handle.ptr), F_TABLES) {
            return true;
        }
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// For a shader that picks buffers from the view heap by an index it
    /// computes (SM 6.6 dynamic resources): the GPU address of the heap's
    /// table of element strides, with every buffer view of the heap given
    /// its host slot and its memory brought up to date when `list` is
    /// submitted. Tables tell us which views a draw reads; nothing tells
    /// us which of the heap's a computed index names, so all of them are
    /// made ready, once per list.
    /// What such a shader stores through a picked unordered access view
    /// goes back to title memory by [`State::heap_buffer_writes`], for the
    /// slots its stores marked ([`crate::heap_marks`]; with
    /// `VICTORY_HEAP_MARKS=0` it stays in the host copy).
    /// A list refreshes them where it first runs such a shader;
    /// title writes recorded later in the same list are not seen.
    fn heap_buffers(&self, list: &HostList) -> Option<u64> {
        let heap = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())?;
        let host = heap.host()?;
        let gpu = host.strides_gpu()?;
        // One key per list, as textures have: no title resource is there.
        const ONCE: usize = usize::MAX - 0x5e;
        if !list.first_use(ONCE) {
            return Some(gpu);
        }
        // Raw unordered access views given their own bytes: written back
        // after every list that may write them (`crate::shadow::raw_own_writes`).
        crate::shadow::raw_own_writes(list);
        // The list may outlive the title's heap object, which owns the table.
        if let Some(buffer) = host.strides_buffer() {
            list.keep(buffer);
        }
        // The pass itself once per frame: per list it ran up to 51 times a
        // frame over 81,000 views and 4,100 host copies, 18 s of every
        // frame (2026-11-24, run 11).
        // shortcut: what the title writes into such buffers between lists
        // of one frame is seen a frame late, unless [`heap_every_list`]
        // brings the spans the frame's pass found up to date here too.
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) == frame {
            if heap_every_list() {
                crate::host::aside::wait_pass(std::ptr::from_ref(host) as usize);
                let spans = HEAP_SPANS
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    .and_then(|m| m.get(&(std::ptr::from_ref(host) as usize)).cloned())
                    .unwrap_or_default();
                for (shadow, from, to) in spans {
                    // Write-watched or guarded memory only: the rest is
                    // copied whole by a refresh (the pass does it once a
                    // frame).
                    if !shadow.unwatched() {
                        shadow.refresh(list, from, to - from);
                        HEAP_LATE_REFRESHES.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            return Some(gpu);
        }
        let started = std::time::Instant::now();
        let hash = STAGE_HASH.get();
        let key = std::ptr::from_ref(host) as usize;
        // (P6, `VICTORY_HEAP_PASS_ASIDE`) On a thread of ours while the
        // title records on: the refreshes are put among the list's steps
        // here when it closes; the draws that read what the pass leaves wait
        // for it (`crate::host::aside::wait_pass`).
        if crate::host::aside::on() {
            let at = std::ptr::from_ref(heap) as usize;
            list.refreshes_coming(key, move || {
                crate::descriptor::with_heap(at, |heap| heap.host().map(|host| heap_buffer_pass(heap, host, frame, hash)))
                    .flatten()
                    .unwrap_or_default()
            });
        } else {
            for (shadow, from, len) in heap_buffer_pass(heap, host, frame, hash) {
                shadow.refresh(list, from, len);
            }
        }
        let textures_began = std::time::Instant::now();
        self.heap_textures(list, heap, host);
        HEAP_PASS_PARTS[0].fetch_add(textures_began.elapsed().as_micros() as u64, Ordering::Relaxed);
        HEAP_BUFFER_PASSES.fetch_add(1, Ordering::Relaxed);
        HEAP_BUFFER_MICROS.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        Some(gpu)
    }

    /// A shader that stores through buffer views it picks by index ran in
    /// `list` (it has the root view of marks, [`crate::heap_marks`]): the
    /// marks of the heap's unordered views are read (those set by lists
    /// that have run), and the ranges of the views shaders marked within
    /// the last [`crate::heap_marks::KEEP_FRAMES`] frames are brought up
    /// to date (write-watched or guarded memory: the pages the title wrote)
    /// and declared written by the list, once per list, merged per host
    /// copy, so that what its shaders stored goes back into title memory
    /// after it ran. It is the declaration a table's unordered views get
    /// ([`crate::shadow::Shadow::gpu_writes`]), with its snapshots, card
    /// fetches (inside flights with `VICTORY_BACK_ASYNC`) and masked merges
    /// (`VICTORY_MERGE_MASK`).
    /// shortcut: every such list declares every marked view, whichever of
    /// them its own shaders store through.
    fn heap_buffer_writes(&self, list: &HostList) {
        const ONCE: usize = usize::MAX - 0x5f;
        if !crate::heap_marks::on() || !list.first_use(ONCE) {
            return;
        }
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return;
        };
        let Some(host) = heap.host() else {
            return;
        };
        // (P6) What the aside pass leaves, once it is done.
        crate::host::aside::wait_pass(std::ptr::from_ref(host) as usize);
        let views = HEAP_UNORDERED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .and_then(|m| m.get(&(std::ptr::from_ref(host) as usize)).cloned())
            .unwrap_or_default();
        crate::heap_marks::LISTS.fetch_add(1, Ordering::Relaxed);
        let written: std::collections::HashSet<u32> =
            crate::heap_marks::take(crate::frame::current(), views.iter().map(|v| &v.0)).into_iter().collect();
        if crate::readback::terrain::on() {
            for (v, _) in views.iter() {
                crate::readback::terrain::heap_view(v.address, v.len, written.contains(&v.slot));
            }
        }
        if written.is_empty() {
            return;
        }
        #[allow(clippy::type_complexity)]
        let mut by_copy: std::collections::HashMap<usize, (std::sync::Arc<crate::shadow::Shadow>, Vec<(u64, u64)>)> =
            std::collections::HashMap::new();
        for (s, shadow) in views.iter().filter(|v| written.contains(&v.0.slot)) {
            by_copy
                .entry(std::sync::Arc::as_ptr(shadow) as usize)
                .or_insert_with(|| (shadow.clone(), Vec::new()))
                .1
                .push((s.address, s.address + s.len));
        }
        crate::heap_marks::DECLARED_VIEWS.fetch_add(written.len() as u64, Ordering::Relaxed);
        let mut all = Vec::new();
        for (shadow, ranges) in by_copy.into_values() {
            for (from, to) in crate::heap_marks::merged(ranges) {
                if !shadow.unwatched() {
                    shadow.refresh(list, from, to - from);
                }
                let ctx = crate::declstat::Ctx { way: crate::declstat::HEAP_MARKED, hash: 0, view: None };
                crate::declstat::with(ctx, || shadow.gpu_writes(list, from, to - from));
                crate::heap_marks::DECLARED_BYTES.fetch_add(to - from, Ordering::Relaxed);
                all.push((from, to));
            }
        }
        crate::heap_marks::declared(all.into_iter());
    }

    /// Runs the pass that uploads textures picked by index
    /// ([`State::heap_textures`]) where nothing has asked for it yet
    /// ([`early_budgets`]): once a frame for a few milliseconds, at the
    /// frame's first draw with a view heap set, and with a larger budget
    /// of its own before every draw whose pixel shader is one of the
    /// title's bake programs. The uploads are recorded into `list` ahead
    /// of the draw being bound, as the world's pass records them.
    fn early_textures(&self, list: &HostList, p: &HostPipeline, compute: bool) {
        let (early_ms, bake_ms) = early_budgets();
        if compute || (early_ms == 0 && bake_ms == 0) {
            return;
        }
        let bake = bake_ms > 0
            && p.stages.iter().any(|s| BAKE_PIXEL_SHADERS.contains(&s.shader.hash.as_str()));
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        // The milliseconds this frame's bake draws have taken so far: a
        // frame has several, and together they get `bake_ms`.
        static BAKED_MS: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        let first = FRAME.swap(frame, Ordering::Relaxed) != frame;
        if first {
            BAKED_MS.store(0, Ordering::Relaxed);
        }
        let left = bake_ms.saturating_sub(BAKED_MS.load(Ordering::Relaxed));
        let budget = if bake && left > 0 {
            // What a bake may read must be there now: up to 2 GiB.
            (left, 2 << 30)
        } else if first && early_ms > 0 {
            (early_ms, 64 << 20)
        } else if let Some(turn) = catch_up_turn() {
            return self.catch_up(list, turn);
        } else {
            return;
        };
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return;
        };
        let Some(host) = heap.host() else {
            return;
        };
        let bake = bake && left > 0;
        let n = EARLY_TEXTURE_PASSES[usize::from(bake)].fetch_add(1, Ordering::Relaxed) + 1;
        if bake && n.is_power_of_two() {
            let [slots, due, ..] = HEAP_TEXTURES_QUEUE.each_ref().map(|c| c.load(Ordering::Relaxed));
            tracing::info!(
                frame,
                bakes = n,
                queued_at_the_last_pass = slots,
                due_at_the_last_pass = due,
                uploaded_so_far = HEAP_TEXTURES.load(Ordering::Relaxed),
                "a draw of the title's decoration bake: textures picked by index are uploaded first (VICTORY_BAKE_TEXTURES_MS=0: not)"
            );
        }
        let began = std::time::Instant::now();
        PASS_BUDGET.set(Some(budget));
        self.heap_textures(list, heap, host);
        PASS_BUDGET.set(None);
        if bake {
            BAKED_MS.fetch_add(began.elapsed().as_millis() as u64, Ordering::Relaxed);
        }
    }

    /// (Neptune 25fbf69) A pass over textures picked by index within a frame, for slots the
    /// title wrote since the last pass ([`catch_up_turn`]): only the uploads that are due, at
    /// most 512 slots, none of the frame's own pass's bookkeeping. Charged to the clock's
    /// credit (`VICTORY_TEXTURE_BUDGET_MS_PER_S`) like any pass; with that off, 6 ms and
    /// 48 MB on top of the frame's spending, as Neptune.
    fn catch_up(&self, list: &HostList, _turn: CatchUp) {
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return;
        };
        let Some(host) = heap.host() else {
            return;
        };
        EARLY_CATCH_UPS.fetch_add(1, Ordering::Relaxed);
        let began = std::time::Instant::now();
        let budget = (texture_budget_ms_per_s() == 0).then_some((6, 48 << 20));
        PASS_BUDGET.set(budget);
        PASS_LIGHT.set(true);
        self.heap_textures(list, heap, host);
        PASS_LIGHT.set(false);
        PASS_BUDGET.set(None);
        EARLY_CATCH_UP_MICROS.fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
    }

    /// The queue's side of the card accounting line
    /// (`crate::resource::card_accounting`, which says what the line
    /// holds): the slots that are due, by what their texture has on the
    /// card. Off unless `VICTORY_CARD_ACCOUNTING=1`; then at the first pass
    /// that finds the card without room, and every five seconds after it.
    /// Nothing is made, uploaded or counted as used here. For 18,000 due
    /// slots: 18,000 looks at a title slot and at the texture it views, a
    /// lock each, some milliseconds of the thread that records the frame.
    fn card_accounting(
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
        frame: u64,
        room: bool,
    ) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ACCOUNTING").is_some_and(|v| v == "1")) {
            return;
        }
        // When the next line is due (0: none yet, the card has had room).
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let now = crate::host::now_ms().max(1);
        let next = NEXT.load(Ordering::Relaxed);
        // (With levels on demand the card may never be without room: the
        // lines then start with the first texture made that way.)
        let by_level = crate::tiles::STATS[4].load(Ordering::Relaxed) > 0;
        let due_now = if next == 0 { !room || by_level } else { now >= next };
        if !due_now || NEXT.compare_exchange(next, now + 5000, Ordering::Relaxed, Ordering::Relaxed).is_err() {
            return;
        }
        // Due slots; of them uploaded before; of them viewing nothing the
        // title fills; and the textures (each once) whose host copy is
        // alive, with their bytes on the card.
        let mut due = [0u64; 5];
        // Textures (each once) that never had a host copy, and ones whose
        // host copy was let go: where their memory is, and its bytes.
        let (mut never, mut gone) = (Vec::new(), Vec::new());
        let mut seen = std::collections::BTreeSet::new();
        for (index, uploaded) in host.texture_slots_all_due(frame) {
            due[0] += 1;
            due[1] += u64::from(uploaded);
            let state = match heap.handle(index).and_then(crate::descriptor::view) {
                Some(View::Placed(false, resource, desc)) if resource[0] != 1 => {
                    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                    match crate::resource::placed_known(address, &resource) {
                        Some(texture) => crate::resource::copy_state(texture),
                        // No title texture was made for it: the pass never
                        // got as far as making one.
                        None => Some((0, crate::resource::placed_bytes(&resource).unwrap_or(0), address)),
                    }
                }
                Some(View::Shader(object, _)) if crate::resource::title_fills(object as This) => {
                    crate::resource::copy_state(object as This)
                }
                _ => None,
            };
            let Some((state, bytes, memory)) = state else {
                due[2] += 1;
                continue;
            };
            if !seen.insert(memory) {
                continue;
            }
            match state {
                2 => {
                    due[3] += 1;
                    due[4] += bytes;
                }
                1 => gone.push((memory, bytes)),
                _ => never.push((memory, bytes)),
            }
        }
        crate::resource::card_accounting(room, due, never, gone);
    }

    /// The texture side of [`State::heap_buffers`]: textures the title fills
    /// itself (streamed pages, layout-library copies) whose views sit in the
    /// view heap are created (placed textures have no host texture until
    /// something reads them) and uploaded, as a table read would do, for
    /// shaders that pick them by a computed index. Within a budget of bytes
    /// per frame and in all; the rest waits for later lists and frames.
    /// The most recently written slots first (`TextureQueue::due`); slots
    /// draws read through tables are not in the queue.
    /// shortcut: uploaded once, and again only when the title rewrites the
    /// slot. A placed texture not all mapped yet is uploaded with what is
    /// there and looked at again (1, 2, 4, ... frames later, at most about
    /// a second): uploaded again when more of it is mapped. Pages mapped
    /// in place of others, and contents changed with no descriptor write,
    /// are missed (hook the page-mapping events for those).
    /// `VICTORY_HEAP_TEXTURES=0` turns it off.
    fn heap_textures(
        &self,
        list: &HostList,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
    ) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_TEXTURES").is_none_or(|v| v != "0")) {
            return;
        }
        // (Neptune) A pass that catches up within a frame (`State::catch_up`) leaves out the
        // frame's own bookkeeping; every pass restarts the wait for the next one.
        let light = PASS_LIGHT.get();
        TEXTURE_PASS_MS.store(crate::host::now_ms(), Ordering::Relaxed);
        crate::descriptor::host::FRESH_WRITES.store(0, Ordering::Relaxed);
        /// Bytes of texture memory staged per frame.
        // `VICTORY_TEXTURE_STAGE_MB` raises it: a backlog of tens of thousands of
        // textures (a scene's first minutes) then drains in fewer frames.
        static PER_FRAME_MB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let per_frame: u64 = *PER_FRAME_MB.get_or_init(|| {
            std::env::var("VICTORY_TEXTURE_STAGE_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(192)
        }) << 20;
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        static SPENT: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) != frame {
            SPENT.store(0, Ordering::Relaxed);
        }
        // No more of them are made while the card has no room
        // (`resource::room`): by what is alive now, not by what was ever
        // staged (4 GB in all, after which later textures stayed empty for
        // the run). The ones nothing names any more are let go, which
        // makes room again: so the slots are still looked at (the pass
        // returned here, and a slot whose texture had been streamed out
        // kept its host copy for as long as the card was full).
        let mut room = if light { crate::resource::room_lately() } else { crate::resource::room() };
        // Bytes uploaded by this pass since the card was last asked
        // whether it has room: asked again every 32 MiB (a pass may upload
        // 192 MB and more, and one asking a pass let the queue run some
        // 400 MB past its line while the world loaded: run 193).
        let mut since_asked = 0u64;
        if !room {
            HEAP_TEXTURES_FULL.fetch_add(1, Ordering::Relaxed);
            once!("textures picked by index: the card has no room for more; they wait");
        }
        let mut refused = 0;
        // Room again: the slots held back while the card was full are due
        // now (`full_card_pass`).
        if room && !light {
            host.texture_slots_room(frame);
        }
        // Slots whose texture's memory the runtime mapped, unmapped or
        // wrote since the last pass are due again.
        if !light {
            host.texture_slots_remapped(frame);
        }
        // Levels on demand: the slots of textures whose first level with
        // memory changed since the last pass are made again from it.
        if !light {
            host.texture_views_remade();
        }
        // And with room, the textures whose upload left a held level
        // without the card's memory are uploaded again.
        if room && !light {
            host.texture_slots_waiting(frame);
        }
        let (due, census, aged, ages) = host.texture_slots_due_aged(
            frame,
            if light { 512 } else { 4096 },
            crate::host::now_ms(),
            texture_max_age_ms(),
        );
        if !light {
        HEAP_TEXTURES_WAITING.store(census[0], Ordering::Relaxed);
        for (count, now) in HEAP_TEXTURES_QUEUE.iter().zip(census) {
            count.store(now, Ordering::Relaxed);
        }
        for (count, now) in HEAP_TEXTURES_AGE.iter().zip(ages) {
            count.store(now, Ordering::Relaxed);
        }
        // Where the card's memory is, and what the due slots would need
        // (`VICTORY_CARD_ACCOUNTING=1`; a line when the card first has no
        // room, and every five seconds after it).
        Self::card_accounting(heap, host, frame, room);
        }
        // The frames in about four seconds, between 16 and 256: how long a
        // slot with nothing to upload waits at most for its next look. In
        // frames alone it was 120: four minutes at the opening cutscene's
        // half a frame per second, while the title streams by the clock.
        // (About a second before the runtime told of the memory it maps:
        // the looks are now only for what that does not tell.)
        static PASSED: AtomicU64 = AtomicU64::new(0);
        // (Neptune) A pass that catches up keeps the frame's: by the time since it, four
        // seconds came to 256 frames.
        static MOST: AtomicU64 = AtomicU64::new(40);
        let most = if light {
            MOST.load(Ordering::Relaxed)
        } else {
            let now = crate::host::now_ms();
            let took = now.saturating_sub(PASSED.swap(now, Ordering::Relaxed)).max(1);
            let most = (4000 / took).clamp(16, 256);
            MOST.store(most, Ordering::Relaxed);
            most
        };
        // What a table entry of a texture a shader samples says.
        let sampled = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        // Also within a time per pass: the pass runs on a thread that
        // records the frame, and making and staging 192 MB of textures took
        // 1.6-2.2 s of each of the world's first twenty frames (run 20).
        // The rest waits for the next frames, as it does past the bytes.
        static SPAN_MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let span = std::time::Duration::from_millis(*SPAN_MS.get_or_init(|| {
            std::env::var("VICTORY_TEXTURE_STAGE_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(250)
        }));
        // A backlog (a scene's first minutes: 22,000 slots due, 2-3 minutes of black
        // surfaces at 192 MB a frame) gets three times the bytes and the time until it
        // drains; `VICTORY_TEXTURE_CATCHUP=0` keeps the plain budget.
        static CATCHUP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let catchup = *CATCHUP.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_CATCHUP").is_none_or(|v| v != "0"));
        let rate = texture_budget_ms_per_s();
        let (per_frame, span) =
            if rate == 0 && catchup && due.len() >= 1024 { (per_frame * 3, span * 3) } else { (per_frame, span) };
        // With `VICTORY_TEXTURE_BUDGET_MS_PER_S` (`texture_budget_ms_per_s`):
        // the time the passes of this frame have taken so far, and the
        // credit the clock gave since the last pass (microseconds; at most
        // the frame's ceiling).
        static FRAME_US: AtomicU64 = AtomicU64::new(0);
        static CREDIT_US: AtomicU64 = AtomicU64::new(0);
        static CREDITED_US: AtomicU64 = AtomicU64::new(u64::MAX);
        static FRAME_OF_US: AtomicU64 = AtomicU64::new(u64::MAX);
        if FRAME_OF_US.swap(frame, Ordering::Relaxed) != frame {
            FRAME_US.store(0, Ordering::Relaxed);
        }
        let span_us = span.as_micros() as u64;
        // The ceiling left to this frame, and the clock's credit within it.
        let (ceiling, credit) = if rate > 0 && PASS_BUDGET.get().is_none() {
            let clock = crate::host::now_us();
            let last = CREDITED_US.swap(clock, Ordering::Relaxed);
            let gained = if last == u64::MAX { span_us } else { clock.saturating_sub(last).saturating_mul(rate) / 1000 };
            let credit = CREDIT_US.load(Ordering::Relaxed).saturating_add(gained).min(span_us);
            CREDIT_US.store(credit, Ordering::Relaxed);
            let left = span_us.saturating_sub(FRAME_US.load(Ordering::Relaxed));
            (
                std::time::Duration::from_micros(left),
                std::time::Duration::from_micros(credit.min(left)),
            )
        } else {
            (span, span)
        };
        // A run made early (`State::early_textures`) has a budget of its
        // own: its bytes on top of what the frame has spent so far.
        // (Not `most`: that is the longest wait between looks, used below.)
        let (span, bytes_at_most) = match PASS_BUDGET.get() {
            Some((ms, bytes)) => {
                (std::time::Duration::from_millis(ms), SPENT.load(Ordering::Relaxed).saturating_add(bytes))
            }
            None => (credit, per_frame),
        };
        let began = std::time::Instant::now();
        // What the pass took, from the credit and the frame's ceiling.
        struct Took(std::time::Instant, bool);
        impl Drop for Took {
            fn drop(&mut self) {
                if self.1 {
                    let us = self.0.elapsed().as_micros() as u64;
                    FRAME_US.fetch_add(us, Ordering::Relaxed);
                    let _ = CREDIT_US
                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |c| Some(c.saturating_sub(us)));
                }
            }
        }
        let _took = Took(began, rate > 0 && PASS_BUDGET.get().is_none());
        for (k, (index, write)) in due.into_iter().enumerate() {
            // Slots due too long (`texture_max_age_ms`, at the head of
            // `due`) go past the clock's credit, up to the frame's ceiling.
            let span = if k < aged && PASS_BUDGET.get().is_none() { ceiling.max(span) } else { span };
            if SPENT.load(Ordering::Relaxed) >= bytes_at_most || began.elapsed() > span {
                break;
            }
            if k < aged {
                HEAP_TEXTURES_AGE[2].fetch_add(1, Ordering::Relaxed);
            }
            // A placed texture whose memory is not mapped (not streamed in
            // yet, or streamed out again: the heap keeps tens of thousands
            // of such views) gets no host texture; looked at again soon (a
            // title writes a texture's view before its pages are mapped,
            // and the pass runs early in a frame). One of which some is
            // mapped (a texture streaming in: its smaller mips lie at the
            // start of its memory and arrive first) is uploaded with what
            // is there, and again when more is. (It was uploaded again
            // every 120 frames, changed or not; and asked by its first
            // page alone, one whose first page is the one missing counted
            // as not there.)
            // A slot uploaded whole is looked at again a few times
            // (`RECHECK_FRAMES`): the title writes a texture's view, and
            // may fill the texture's memory itself only afterwards, with
            // no descriptor write and nothing from the runtime to tell of
            // it. The host copy then kept what the memory held at the
            // first look for the rest of the run. The look reads 48 KiB;
            // only a changed texture is uploaded again.
            // `VICTORY_HEAP_RECHECK=0`: uploaded once, as before.
            if let Some((at, len, print, looks, uploaded)) = host.texture_slot_checked(index) {
                if crate::texture::fingerprint(at, len) == print {
                    HEAP_TEXTURES_RECHECKED[0].fetch_add(1, Ordering::Relaxed);
                    match RECHECK_FRAMES.get(usize::from(looks) + 1) {
                        Some(wait) => {
                            host.texture_slot_recheck(index, write, frame + wait, (at, len), print, looks + 1, uploaded)
                        }
                        None => host.texture_slot_done(index, write, None),
                    }
                    continue;
                }
                let n = HEAP_TEXTURES_RECHECKED[1].fetch_add(1, Ordering::Relaxed);
                if n < 200 || n.is_multiple_of(1000) {
                    tracing::warn!(
                        slot = index,
                        address = format_args!("{at:#x}"),
                        bytes = len,
                        frames_after_upload = frame.saturating_sub(uploaded),
                        looks,
                        changed = n + 1,
                        "a texture picked by index changed after its upload with nothing to tell of it: uploaded again"
                    );
                }
            }
            let mut partly = None;
            if let Some(View::Placed(false, resource, desc)) = heap.handle(index).and_then(crate::descriptor::view) {
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                // Of the mips the view shows: what lies behind them is
                // another texture's (`VICTORY_TEXTURE_FIRST_MIP`).
                let len = crate::resource::placed_bytes_from(&resource, crate::resource::first_mip(&desc)).unwrap_or(1);
                let mapped = crate::texture::mapped_bytes(address as *const u8, len as usize);
                if mapped == 0 {
                    HEAP_TEXTURES_AWAY.fetch_add(1, Ordering::Relaxed);
                    // Streamed out: what the slot showed is let go.
                    host.texture_slot_gone(index);
                    host.texture_slot_later(index, write, frame, most);
                    continue;
                }
                if mapped < len {
                    if mapped <= host.texture_slot_mapped(index) {
                        HEAP_TEXTURES_PARTLY[0].fetch_add(1, Ordering::Relaxed);
                        host.texture_slot_later(index, write, frame, most);
                        continue;
                    }
                    if !crate::texture::committed(address as *const u8, 1) {
                        HEAP_TEXTURES_PARTLY[2].fetch_add(1, Ordering::Relaxed);
                    }
                    partly = Some(mapped);
                }
            }
            // A texture object the host renders to or writes holds the
            // host's own contents: nothing to bring up to date, and this
            // pass is no use of it (its host copy is made when something
            // uses it, and its slots then: `descriptor::make_waiting`).
            if let Some(View::Shader(object, _)) = heap.handle(index).and_then(crate::descriptor::view)
                && crate::resource::buffer_bytes(object as This).is_none()
                && !crate::resource::title_fills(object as This)
            {
                host.texture_slot_done(index, write, None);
                continue;
            }
            // From here on a host texture is made: not without room. The
            // slot stays due.
            // Not so a slot already made from this very view: its host
            // texture exists, and uploading it again takes none of the
            // card's memory. It was refused all the same, so a texture the
            // title filled after its first upload kept its empty host copy
            // from the moment the card reached the line, which is the first
            // world frame in eight launches of nine: the same slots were
            // found changed again at every pass and never uploaded (run
            // 179: 33,000 such findings), and the main character's face
            // had its colour only in the launches that entered the world
            // under the line. `VICTORY_REUPLOAD_WHEN_FULL=0`: refused, as
            // before.
            if !room {
                let has_its_texture = reupload_when_full()
                    && heap.handle(index).is_some_and(|handle| host.texture_slot_made_from(index, handle));
                if !has_its_texture {
                    refused += 1;
                    if full_card_pass() {
                        // Held back for the frames of about four seconds
                        // (`most`), or until the card has room; the pass
                        // goes on, within a count that bounds one pass.
                        host.texture_slot_held(index, write, frame + most);
                        if refused > 1024 {
                            break;
                        }
                    } else if refused > 256 {
                        break;
                    }
                    continue;
                }
                let n = HEAP_TEXTURES_REUPLOADED_FULL.fetch_add(1, Ordering::Relaxed) + 1;
                if n == 1 || n.is_multiple_of(500) {
                    tracing::info!(
                        so_far = n,
                        slot = index,
                        "a texture already on the card is brought up to date although the card has no room for new ones"
                    );
                }
            }
            let view = heap.handle(index).and_then(|handle| host.mirror(index, handle, true));
            let placed = view.as_ref().and_then(crate::descriptor::host::placed_texture);
            let resource = match placed.as_ref().or(view.as_ref()) {
                Some(&View::Shader(resource, _)) => resource as This,
                _ => {
                    host.texture_slot_done(index, write, None);
                    continue;
                }
            };
            let title_fills = crate::resource::host_resource(resource).is_some_and(crate::texture::title_written);
            if !title_fills {
                // a render target, a depth or unordered access texture: the host's own
                host.texture_slot_done(index, write, None);
                continue;
            }
            // A texture that has a staging buffer now is one draws read
            // through a table: they upload it, and check it at every
            // submission that samples it.
            let tabled = crate::texture::is_staged(resource);
            if let Some(view) = &view {
                readable(list, sampled.kind, view, 0);
            }
            // A texture view the host refuses removes the device: name it.
            static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !GONE.load(Ordering::Relaxed)
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                GONE.store(true, Ordering::Relaxed);
                let text: String = format!("{view:?}").chars().take(300).collect();
                tracing::error!(why, slot = index, "the host device went while textures picked by index were made ready; last: {text}");
                crate::host::log_removal_context();
                return;
            }
            let (bytes, whole) = crate::texture::staged(resource).unwrap_or((0, true));
            // Not all of it there: looked at again next frame, and from
            // then on as a slot with nothing new (a placed texture: until
            // more of it is mapped; a texture object at an address the
            // title mapped: again in 120 frames, as before).
            let again = match partly {
                Some(mapped) => Some((frame + 1, mapped)),
                None => (!whole).then_some((frame + 120, 0)),
            };
            if again.is_some() {
                HEAP_TEXTURES_PARTLY[1].fetch_add(1, Ordering::Relaxed);
            }
            if tabled {
                // Nothing was staged here, so nothing is counted or spent,
                // and the staging buffer stays: taking it away made the
                // next draw that reads the texture through a table stage,
                // detile and upload all of it once more.
                HEAP_TEXTURES_TABLED.fetch_add(1, Ordering::Relaxed);
                host.texture_slot_done(index, write, again);
                continue;
            }
            // Uploaded: the list holds the staging buffer until it has run;
            // a table read, or the next look at a texture not all mapped
            // yet, makes it anew. (Kept for those until now: every partly
            // streamed texture of the heap held a staging buffer of its full
            // size for good.)
            let memory = if again.is_none() && heap_recheck() {
                crate::texture::staged_memory(resource)
            } else {
                None
            };
            crate::texture::unstage(resource);
            SPENT.fetch_add(bytes, Ordering::Relaxed);
            since_asked += bytes;
            if room && since_asked >= 32 << 20 {
                since_asked = 0;
                room = crate::resource::room();
            }
            HEAP_TEXTURE_BYTES.fetch_add(bytes, Ordering::Relaxed);
            HEAP_TEXTURES.fetch_add(1, Ordering::Relaxed);
            match memory {
                // Whole and uploaded: looked at again (see above).
                Some((at, len)) => {
                    let print = crate::texture::fingerprint(at, len);
                    host.texture_slot_recheck(index, write, frame + RECHECK_FRAMES[0], (at, len), print, 0, frame);
                }
                None => host.texture_slot_done(index, write, again),
            }
        }
    }

    /// For the frame listing, after a dropped dispatch's line: its group
    /// counts and the compute root arguments it was recorded with (root
    /// constants as hex dwords, root views with the title buffer they
    /// point into). A program that does not translate has no bindings to
    /// list; these say which memory it would have read and written (the
    /// ray-tracing structure builder passes its buffers this way).
    #[cfg(feature = "diagnostics")]
    fn list_dropped_roots(&self, groups: [u32; 3]) {
        crate::capture::op(0, || {
            use std::fmt::Write;
            let [x, y, z] = groups;
            let mut text = format!("  its roots ({x}x{y}x{z}):");
            for (slot, root) in self.compute.iter().enumerate() {
                match root {
                    Root::None => {}
                    Root::Table(a) => {
                        let _ = write!(text, " {slot}=table {a:#x}");
                    }
                    Root::Constants(v) => {
                        let _ = write!(text, " {slot}=constants {:x?}", &v[..v.len().min(48)]);
                    }
                    Root::View(a) => {
                        let _ = write!(text, " {slot}=view {a:#x}");
                        match crate::shadow::extent(*a) {
                            Some((start, len)) => {
                                let _ = write!(text, " in buffer {start:#x}+{len:#x}");
                            }
                            None => text.push_str(" in no buffer"),
                        }
                    }
                }
            }
            text
        });
    }

    /// Whether the shaders of the pipeline bound now can store to buffers
    /// (or the pipeline is not known): a draw with them may write the
    /// arguments of draws after it, which ends a list's gathering
    /// (`crate::indirect::gather`). The same test keeps such a draw out of
    /// a window ([`windows`]).
    fn may_store(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_none_or(|p| p.stages.iter().any(|stage| stage.translated().stores))
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        let shape = crate::readback::terrain::dispatch_begin(x, y, z);
        // It may write the arguments a window holds clamped.
        self.window = None;
        list.gather_close();
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            crate::readback::terrain::dispatch_end(shape, "skipped pipeline");
            return;
        }
        // A program that walks ray-tracing structures nothing has built.
        if crate::pipeline::walk_held(self.pipeline as This) {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_why("a walk of ray-tracing structures that are not built (VICTORY_BVH_HOLD=1)".to_owned());
            crate::readback::terrain::dispatch_end(shape, "walk held");
            return;
        }
        // The ray-tracing structure builder's stages run all or none.
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_why("a stage of the ray-tracing structure builder, which is off".to_owned());
            #[cfg(feature = "diagnostics")]
            self.list_dropped_roots([x, y, z]);
            crate::readback::terrain::dispatch_end(shape, "builder off");
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            #[cfg(feature = "diagnostics")]
            self.list_dropped_roots([x, y, z]);
            crate::readback::terrain::dispatch_end(shape, "no host pipeline");
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            crate::readback::terrain::dispatch_end(shape, "graphics pipeline");
            return;
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            // A stage of the structure builder that cannot be bound is a
            // dropped stage: the others stop too.
            if crate::pipeline::is_builder(self.pipeline as This) {
                crate::pipeline::builder_off("one of its dispatches could not be bound");
            }
            crate::readback::terrain::dispatch_end(shape, "bind failed");
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        #[cfg(feature = "diagnostics")]
        let probed = probe::direct(self, list, p, [x, y, z]);
        let base = group_base(p);
        list.inside_end();
        match base {
            Some((param, dword)) if x > crate::indirect::PART => {
                // More than the host runs in one dimension: in parts, each
                // told the number of its first group.
                let mut parts = 0;
                // SAFETY: an open list with everything set; the parameter
                // and dword are the pipeline's.
                unsafe {
                    let mut first = 0;
                    while first < x {
                        list.list.SetComputeRoot32BitConstant(param, first, dword);
                        list.list.Dispatch((x - first).min(crate::indirect::PART), y, z);
                        first += crate::indirect::PART;
                        parts += 1;
                    }
                    list.list.SetComputeRoot32BitConstant(param, 0, dword);
                }
                // (P3) Its words set past the record.
                self.issued.root_gone(true, param);
                DISPATCH_PARTS[0].fetch_add(1, Ordering::Relaxed);
                DISPATCH_PARTS[1].fetch_add(parts, Ordering::Relaxed);
            }
            _ => {
                if x > crate::indirect::PART {
                    DISPATCH_PARTS[2].fetch_add(1, Ordering::Relaxed);
                }
                // SAFETY: an open list with everything set.
                unsafe { list.list.Dispatch(x, y, z) };
            }
        }
        #[cfg(feature = "diagnostics")]
        probe::after(probed, list);
        crate::readback::terrain::dispatch_end(shape, "");
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        // What it draws into (a dispatch writes through its views below).
        if !compute {
            let described = |t: usize| {
                crate::resource::host_made(t as This).map(|r| {
                    // SAFETY: a live resource.
                    let d = unsafe { r.GetDesc() };
                    format!("{t:#x} {}x{} f{}", d.Width, d.Height, d.Format.0)
                })
            };
            let targets: Vec<String> = self.targets.iter().filter_map(|&t| described(t)).collect();
            let _ = write!(text, " targets {targets:?} depth {:?} state [{}]", described(self.depth), p.state);
            // The title's scissor and viewport at this draw (the first of
            // each): the title draws what it sees through a
            // portal (a window, from inside a room) under the portal's
            // rectangle, and the listing did not say which draws those are.
            if let Some(r) = self.scissors.first() {
                let _ = write!(text, " scissor [{} {} {} {}]", r.left, r.top, r.right, r.bottom);
            }
            if let Some(v) = self.viewports.first() {
                let _ = write!(
                    text,
                    " viewport [{} {} {} {} depth {}..{}]",
                    v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth
                );
            }
        }
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = match u.root {
                    RootType::Other(kind) => self.host_word(compute, kind, u.slot, u.dword, None),
                    _ => self.user_value(compute, u.slot, u.root, u.dword),
                };
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage.root_cbvs.iter().chain(&stage.root_srvs).chain(&stage.root_uavs) {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        if !compute && !list_slots().is_empty() {
            text.push_str(&self.slots_text());
        }
        text
    }

    /// For the frame listing, after a draw whose shader `VICTORY_LIST_SLOTS`
    /// names: the heap slots it will pick textures from by index, and what
    /// the title and the host hold there (one `~slot` line each), with the
    /// words they were read from (`~raw` lines, to decode offline what is
    /// not decoded here).
    ///
    /// Where the slots are, for the title's character shaders
    /// (`PS_DeferredPedTexturedNC`, from their native code): in the small
    /// buffers and constants of the draw's own table (every word of those
    /// that names a texture view gets a line); in the material record, at
    /// the byte offset the draw's constants (table 1, dword 56) hold at
    /// 0x1c, of the buffer at table 0, dword 160; and in the record at the
    /// byte offset the second root constant holds of the buffer at table 0,
    /// dword 688. A word that is a small number by chance and names a
    /// texture gets a line too.
    #[cfg(feature = "diagnostics")]
    fn slots_text(&self) -> String {
        use std::fmt::Write;
        const MATERIALS: u32 = 160;
        const GLOBAL: u32 = 688;
        const DRAW_CONSTANTS: u32 = 56;
        let wanted = list_slots();
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return String::new();
        };
        let Some(stage) = p.stages.iter().find(|s| wanted.iter().any(|w| s.shader.hash.starts_with(w.as_str())))
        else {
            return String::new();
        };
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return String::new();
        };
        let t = stage.translated();
        let mut text = String::new();
        let entry = |slot: u8, dwords: u32| -> Option<View> {
            let Some(Root::Table(table)) = self.roots_ref(false).get(usize::from(slot)) else {
                return None;
            };
            crate::descriptor::view(table + u64::from(dwords) * 4)
        };
        let mut material = None;
        for u in t.used.iter().filter(|u| u.root == RootType::Table && u.kind != ResourceType::Sampler) {
            let label = format!("s{}+{}", u.slot, u.table_offset_dwords);
            let Some(view) = entry(u.slot, u.table_offset_dwords) else {
                let _ = write!(text, "\n    ~table {label} no view");
                continue;
            };
            let range = match &view {
                View::Constant(address, size) => Some((*address, u64::from(*size))),
                other => crate::descriptor::host::buffer_range(other),
            };
            match range {
                // A small buffer or constants: its words.
                Some((address, len)) if len <= 2048 => {
                    Self::slot_words(&mut text, heap, &label, address, len as usize);
                    if u.slot == 1 && u.table_offset_dwords == DRAW_CONSTANTS && len >= 0x20 {
                        let mut word = [0u8; 4];
                        if crate::host::read_into((address + 0x1c) as *const u8, &mut word) {
                            material = Some(u32::from_le_bytes(word));
                        }
                    }
                }
                Some(_) => {}
                // A texture read through the table: the title slot it sits in.
                None => {
                    let Some(Root::Table(table)) = self.roots_ref(false).get(usize::from(u.slot)) else {
                        continue;
                    };
                    match heap.index(table + u64::from(u.table_offset_dwords) * 4) {
                        Some(index) => text.push_str(&Self::slot_line(heap, &format!("table:{label}"), 0, index)),
                        None => {
                            let _ = write!(text, "\n    ~table {label} title=[{}]", view_state(Some(&view)));
                        }
                    }
                }
            }
        }
        let uses = |slot: u8, dwords: u32| {
            t.used.iter().any(|u| u.root == RootType::Table && u.slot == slot && u.table_offset_dwords == dwords)
        };
        if let Some(at) = material
            && uses(0, MATERIALS)
            && let Some((base, len)) = entry(0, MATERIALS).as_ref().and_then(crate::descriptor::host::buffer_range)
        {
            let at = u64::from(at);
            let _ = write!(text, "\n    ~material record at {at:#x} of the buffer at {base:#x} bytes {len}");
            if at + 64 <= len {
                Self::slot_words(&mut text, heap, "M", base + at, 64);
            }
            if at + 1936 + 512 <= len {
                Self::slot_words(&mut text, heap, "M+1936", base + at + 1936, 512);
            }
        }
        if uses(0, GLOBAL)
            && let Some(slot) = t.used.iter().find(|u| u.root == RootType::Constants).map(|u| u.slot)
            && let Some(Root::Constants(words)) = self.roots_ref(false).get(usize::from(slot))
            && let Some(&at) = words.get(1)
            && let Some((base, len)) = entry(0, GLOBAL).as_ref().and_then(crate::descriptor::host::buffer_range)
        {
            let at = u64::from(at);
            let _ = write!(text, "\n    ~root constants {words:08x?}: record at {at:#x} of the buffer at {base:#x} bytes {len}");
            if at + 64 <= len {
                Self::slot_words(&mut text, heap, "G", base + at, 64);
            }
        }
        text
    }

    /// Up to 512 bytes of title memory at `address` as hex words (a `~raw`
    /// line), and a `~slot` line for every word of them that names a
    /// texture view in `heap`.
    #[cfg(feature = "diagnostics")]
    fn slot_words(text: &mut String, heap: &crate::descriptor::Heap, label: &str, address: u64, len: usize) {
        use std::fmt::Write;
        let mut bytes = vec![0u8; len.min(512) & !3];
        if bytes.is_empty() || !crate::host::read_into(address as *const u8, &mut bytes) {
            let _ = write!(text, "\n    ~raw {label} at {address:#x} bytes {len}: unreadable");
            return;
        }
        let words: Vec<u32> =
            bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
        let _ = write!(text, "\n    ~raw {label} at {address:#x} bytes {len}:");
        for w in &words {
            let _ = write!(text, " {w:08x}");
        }
        for (i, &index) in words.iter().enumerate() {
            let names = index != 0
                && heap
                    .handle(index)
                    .and_then(crate::descriptor::view)
                    .as_ref()
                    .is_some_and(crate::descriptor::host::is_texture_view);
            if names {
                text.push_str(&Self::slot_line(heap, label, i * 4, index));
            }
        }
    }

    /// One `~slot` line: where the slot number was read (`from`, byte
    /// `at`), what the title's heap holds in slot `index` with the state
    /// of that texture ([`view_state`]), what the host slot was made from,
    /// and what the queue of textures picked by index knows of it.
    #[cfg(feature = "diagnostics")]
    fn slot_line(heap: &crate::descriptor::Heap, from: &str, at: usize, index: u32) -> String {
        let title = heap.handle(index).and_then(crate::descriptor::view);
        let host: String = heap
            .host()
            .and_then(|h| h.describe_slot(index))
            .unwrap_or_else(|| "none".into())
            .chars()
            .take(90)
            .collect();
        let queue = match heap.host() {
            Some(h) => h.texture_slot_state(index, crate::frame::current()),
            None => "no host heap".into(),
        };
        format!(
            "\n    ~slot from={from}+{at:#x} slot={index} title=[{}] host=[{host}] queue=[{queue}]",
            view_state(title.as_ref())
        )
    }

    /// A mesh draw under `VICTORY_MESH_TEST=scissor` rasterises one pixel:
    /// the scissors before it (`on`), the title's after it.
    fn mesh_scissor(&mut self, list: &HostList, on: bool) {
        if mesh_test() != 2 {
            return;
        }
        // (P3) Set here past the record of what the host list holds.
        self.issued.scissors = None;
        if on {
            let one = RECT { left: 0, top: 0, right: 1, bottom: 1 };
            crate::draw::scissors(list, &vec![one; self.scissors.len().max(1)]);
        } else if !self.scissors.is_empty() {
            crate::draw::scissors(list, &self.scissors);
        }
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        self.draw_ends_window();
        if self.may_store() {
            list.gather_close();
        }
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = mesh_test() != 1
            && mesh_kept(self.pipeline)
            && self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    self.mesh_scissor(list, true);
                    list.inside_end();
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    self.mesh_scissor(list, false);
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            dropped(self.pipeline);
        }
    }

    /// The walls trace (`crate::host::walls_trace`): one log line per
    /// frame with the draws of its main pass (four or more colour targets)
    /// by scissor rectangle, and how often each of the title's culling
    /// programs was dispatched. The title clips what it draws
    /// through a window or door to that opening's rectangle while it holds
    /// the camera to be in a room (run 182, frame 1529: 1,386 draws under
    /// `531 0 1789 1337`); in a launch without the room's walls the same
    /// shot has no such rectangle (run 187, frame 1414). The line shows
    /// the frame in which that changes.
    fn walls_census(&self, p: &HostPipeline, compute: bool) {
        const CULL: [(&str, &str); 9] = [
            ("55679a0547649d0b", "tree"),
            ("89d08e561897d4e5", "tree_imposter"),
            ("9209777e8face9dd", "grass"),
            ("17af15c4015cb4b6", "prop"),
            ("fdd56d97a806bcca", "prop_nocull"),
            ("42f717f1b4d18e94", "simple"),
            ("1c4b3c0547e3e6fa", "slod"),
            ("9146291b38ea9ce1", "meshlet_args"),
            ("42683eebec081520", "depth_pyramid_cpu_occlusion"),
        ];
        type Census = (u64, std::collections::BTreeMap<[i32; 4], u32>, [u32; 9]);
        static CENSUS: std::sync::Mutex<Census> =
            std::sync::Mutex::new((u64::MAX, std::collections::BTreeMap::new(), [0; 9]));
        let frame = crate::frame::current();
        let mut census = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
        if census.0 != frame {
            let draws: u32 = census.1.values().sum();
            if draws > 0 {
                let mut rects: Vec<([i32; 4], u32)> = census.1.iter().map(|(r, n)| (*r, *n)).collect();
                rects.sort_unstable_by_key(|r| std::cmp::Reverse(r.1));
                let scissors: Vec<String> = rects
                    .iter()
                    .take(5)
                    .map(|(r, n)| format!("{} {} {} {} x{n}", r[0], r[1], r[2], r[3]))
                    .collect();
                let counts = census.2;
                let culling: Vec<String> = CULL
                    .iter()
                    .zip(counts)
                    .filter(|(_, n)| *n > 0)
                    .map(|(c, n)| format!("{}={n}", c.1))
                    .collect();
                tracing::info!(
                    frame = census.0,
                    main_pass_draws = draws,
                    rectangles = rects.len(),
                    scissors = ?scissors,
                    culling = ?culling,
                    "walls trace: the frame's main pass by scissor rectangle, and its culling dispatches"
                );
            }
            census.0 = frame;
            census.1.clear();
            census.2 = [0; 9];
        }
        if compute {
            for (i, c) in CULL.iter().enumerate() {
                if p.stages.iter().any(|s| s.shader.hash.as_str() == c.0) {
                    census.2[i] += 1;
                }
            }
        } else if self.targets.iter().filter(|&&t| t != 0).count() >= 4 {
            let r = self.scissors.first().map_or([0; 4], |r| [r.left, r.top, r.right, r.bottom]);
            *census.1.entry(r).or_insert(0) += 1;
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            readable(list, u.kind, &view, handle);
            // Up to date by this draw: not the texture pass's to make.
            if crate::descriptor::host::is_texture_view(&view) {
                host.texture_slot_tabled(index, true);
            }
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constant data copied to upload memory when the list is
    /// submitted: the host wants constant buffers 256-byte aligned, the
    /// console only 16. Returns those tables' new byte offsets by root
    /// parameter; the other tables' slots are made from the title's.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    pub(crate) fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        indexed: &[RootIndexedView],
        compute: bool,
    ) -> Vec<(u8, u32)> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| u.kind == ResourceType::Cbv)
            .map(|u| u.slot)
            .collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        // (fxceless #927) A table with a 16-bit float or signed buffer view at a place the
        // shader moves halves through is made afresh too, with that view in the unsigned
        // format of the same layout (`descriptor::host::half_view`); the title's own slot
        // keeps the title's format for the shaders that read it as floats.
        let half: Vec<(u8, u32)> = HALF_PLACES.with_borrow(|h| h.clone());
        for &(slot, at) in &half {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let Some(view) = crate::descriptor::view(table + u64::from(at) * 4) else {
                continue;
            };
            if crate::descriptor::host::half_view(&view).is_some() {
                fresh.push(slot);
            } else if let Some(format) = crate::descriptor::host::half_inexact(&view) {
                once!(
                    format,
                    "a shader moves a buffer's 16-bit channels as integers through a view whose values the console converts: they are read and written unconverted"
                );
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            if heap.samplers() || (table - heap.start) % 32 != 0 {
                continue;
            }
            let Some(host) = heap.host() else {
                continue;
            };
            let mut selected: Vec<u32> = indexed
                .iter()
                .filter(|index| index.table_slot == slot)
                .map(|index| {
                    let constant = self.user_value(
                        compute,
                        index.constant_slot,
                        RootType::Constants,
                        index.constant_dword,
                    );
                    // The shader adds i32 byte offsets and shifts by 5.
                    index.byte_offset.wrapping_add(constant << index.shift) / 32
                })
                .collect();
            // (fxceless #927) The places the shader moves halves through are made too, and
            // the table reaches them (they may lie past the range starts the entries give).
            selected.extend(half.iter().filter(|&&(s, _)| s == slot).map(|&(_, at)| at / 8));
            selected.sort_unstable();
            selected.dedup();
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8)
                .chain(selected.last().copied())
                .max()
                .unwrap_or(0)
                + 1;
            // (fxceless 5d659e0, `VICTORY_TABLE_STAGES`) As many slots as the stage of this
            // draw that reads furthest into the table needs ([`stage_spans`]; empty with the
            // switch off), so the stages share one table.
            let (n, staged) = STAGE_SPANS.with_borrow(|(spans, staged)| match spans.iter().find(|s| s.0 == slot) {
                Some(&(_, with_constants, all)) => {
                    let past = heap.index(table).is_some_and(|at| u64::from(at) + u64::from(all) > u64::from(host.count()));
                    let n = n.max(if past { all } else { with_constants });
                    (n, staged.contains(&(table, n)))
                }
                None => (n, false),
            });
            let last = table.checked_add(u64::from(n - 1) * 32);
            if n > crate::descriptor::host::DYNAMIC
                || last.and_then(|handle| heap.index(handle)).is_none()
            {
                once!(slot, n, "fresh descriptor table exceeds the available span");
                continue;
            }
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                    // (Another table than the same slots made for a shader that reads them
                    // in the title's formats.)
                    if half.contains(&(slot, k * 8)) {
                        [u32::MAX, u32::from(slot), k].hash(&mut h);
                    }
                }
                // (Neptune) By contents alone (`tables_by_content`): the title writes every
                // draw's table at a place of its own, and two thirds of a frame's tables
                // were made afresh although others with the same views in the same order
                // had been made in the same list. Record words name one interned view each
                // (index and generation, `descriptor::record`), so equal words are equal
                // views. Ours: the title heap stands in for the address, not 0: the host
                // slots are another heap's after a change of heaps.
                let place = if tables_by_content() { std::ptr::from_ref(heap) as u64 } else { table };
                (place, n, h.finish())
            };
            let here = (table, n, key.2);
            if let Some(&first) = self.fresh_tables.get(&key) {
                // (Or found in the table an earlier stage of this draw made, `staged`: this
                // stage's title slots are noted as its own making would have noted them.)
                if (tables_by_content() && self.seen_tables.insert(here)) || staged {
                    // Another table's host slots; this one's own title slots are noted as
                    // its making would have noted them.
                    TABLES[4].fetch_add(1, Ordering::Relaxed);
                    if let Some(title) = heap.index(table).filter(|first| first + n <= host.count()) {
                        for k in 0..n {
                            let handle = table + u64::from(k) * 32;
                            if let Some(view) = crate::descriptor::view(handle)
                                && crate::descriptor::host::is_texture_view(&view)
                            {
                                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                                let kind = used
                                    .map(|u| u.kind)
                                    .or_else(|| selected.binary_search(&k).ok().and_then(|_| view_kind(&view)));
                                host.texture_slot_tabled(title + k, kind.is_some());
                                if kind.is_some() {
                                    host.mirror(title + k, handle, true);
                                }
                            }
                        }
                    }
                }
                // (Neptune) The same table again in the same host list, with no texture's
                // state or buffer's bytes changed in it since its views were made ready
                // (`HostList::changes`): nothing to do. A fifth of the recording threads'
                // working time went into doing it again for every draw.
                let at_list = std::ptr::from_ref(list) as usize;
                let changes = list.changes.load(Ordering::Relaxed);
                let (tested, _) = TESTED_DEPTH.get();
                let ready = ready_tables_on()
                    && self.ready_tables.get(&key).is_some_and(|(at, then, textures)| {
                        *at == at_list && *then == changes && (tested == 0 || !textures.contains(&tested))
                    });
                if ready {
                    TABLES[0].fetch_add(1, Ordering::Relaxed);
                    made.push((slot, first * 32));
                    continue;
                }
                TABLES[1].fetch_add(1, Ordering::Relaxed);
                // Why not: no note of it, another host list, the list's count moved, or
                // the draw's tested depth target among its textures.
                match self.ready_tables.get(&key) {
                    None => TABLES[5].fetch_add(1, Ordering::Relaxed),
                    Some((at, _, _)) if *at != at_list => TABLES[6].fetch_add(1, Ordering::Relaxed),
                    Some((_, then, _)) if *then != changes => TABLES[7].fetch_add(1, Ordering::Relaxed),
                    Some(_) => TABLES[8].fetch_add(1, Ordering::Relaxed),
                };
                // Whether the table is one to remember: views that are only read, of
                // textures without another texture's memory behind them.
                let mut plain = true;
                let mut textures: Vec<usize> = Vec::new();
                let mut note = |kind: ResourceType, view: &View| {
                    // (Only what shaders write is not remembered.)
                    if kind == ResourceType::Uav {
                        plain = false;
                    }
                    let placed = crate::descriptor::host::placed_texture(view);
                    if let View::Shader(resource, _) = placed.as_ref().unwrap_or(view) {
                        if crate::resource::alias_of(*resource as This).is_some() {
                            plain = false;
                        }
                        textures.push(*resource);
                    }
                };
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        note(u.kind, &view);
                        readable(list, u.kind, &view, handle);
                    }
                }
                for k in selected {
                    if entries.iter().any(|u| u.table_offset_dwords / 8 == k) {
                        continue;
                    }
                    let handle = table + u64::from(k) * 32;
                    if let Some(view) = crate::descriptor::view(handle)
                        && let Some(kind) = view_kind(&view)
                    {
                        note(kind, &view);
                        readable(list, kind, &view, handle);
                    }
                }
                // The count after these reads: what they did themselves (a transition,
                // a copy) is part of the state the next draw finds.
                if plain && (tested == 0 || !textures.contains(&tested)) {
                    self.ready_tables.insert(key, (at_list, list.changes.load(Ordering::Relaxed), textures));
                } else {
                    self.ready_tables.remove(&key);
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = host.dynamic(n) else {
                // No room: the draw must not run with the title's own
                // offsets (slots never made for it); `bind` drops it.
                made.push((slot, u32::MAX));
                continue;
            };
            self.fresh_tables.insert(key, first);
            self.seen_tables.insert(here);
            STAGE_SPANS.with_borrow_mut(|(spans, staged)| {
                if !spans.is_empty() {
                    staged.push((table, n));
                }
            });
            TABLES[2].fetch_add(1, Ordering::Relaxed);
            TABLES[3].fetch_add(u64::from(n), Ordering::Relaxed);
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, first, n));
            // The title's own slots of this table: what the draw reads of
            // them it brings up to date here, in the dynamic slots.
            let title_first = heap.index(table).filter(|first| first + n <= host.count());
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    // Bytes a flight in the air may write: filled on the
                    // card behind it (`VICTORY_FILL_ON_CARD`, Trexies 25fbcce).
                    Some(View::Constant(address, size))
                        if list.fill_wants_card(address as usize, address as usize + size as usize)
                            && let Some(shadow) = crate::shadow::find(address, u64::from(size))
                            && list.card_holds(&shadow, address as usize, address as usize + size as usize)
                            && let Some(gpu) = {
                                shadow.refresh(list, address, u64::from(size));
                                list.card_constants(&shadow, address, u64::from(size))
                            } =>
                    {
                        crate::host::ON_CARD[0].fetch_add(1, Ordering::Relaxed);
                        host.constant_at(index, gpu, size);
                    }
                    Some(View::Constant(address, size)) => {
                        // (P1 meter) Wanted the card and did not get it: why.
                        {
                            use crate::host::read_wait::{NC_HOLDS, NC_LATER, NC_NO_COPY, NC_ROOM, pending, why};
                            if pending() == NC_LATER {
                                let (a, e) = (address as usize, address as usize + size as usize);
                                why(match crate::shadow::find(address, u64::from(size)) {
                                    None => NC_NO_COPY,
                                    Some(s) if !list.card_holds(&s, a, e) => NC_HOLDS,
                                    Some(_) => NC_ROOM,
                                });
                            }
                        }
                        let Some((gpu, to)) = list.scratch(u64::from(size)) else {
                            host.make_null(index);
                            continue;
                        };
                        list.defer_fill(to, address as *const u8, size as usize);
                        host.constant_at(index, gpu, size);
                    }
                    Some(view) => {
                        // (fxceless #927) The unsigned twin where the shader moves halves.
                        let twin = if half.contains(&(slot, k * 8)) {
                            crate::descriptor::host::half_view(&view)
                        } else {
                            None
                        };
                        match twin {
                            Some(twin) => {
                                HALF_VIEWS_SHOWN.fetch_add(1, Ordering::Relaxed);
                                host.make_at(index, &twin);
                            }
                            None => host.make_at(index, &view),
                        }
                        let kind = used.map(|u| u.kind).or_else(|| {
                            selected.binary_search(&k).ok().and_then(|_| view_kind(&view))
                        });
                        if let Some(kind) = kind {
                            readable(list, kind, &view, table + u64::from(k) * 32);
                        }
                        // The title's slot is a table's: out of the queue
                        // of textures picked by index if the draw read it
                        // (its texture is up to date now), and made in
                        // place as the texture pass would have made it,
                        // for a shader that picks it by index all the same.
                        if let Some(title) = title_first
                            && crate::descriptor::host::is_texture_view(&view)
                        {
                            host.texture_slot_tabled(title + k, kind.is_some());
                            if kind.is_some() {
                                host.mirror(title + k, table + u64::from(k) * 32, true);
                            }
                        }
                    }
                    None => {
                        host.make_null(index);
                        if used.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_made(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        self.draw_ends_window();
        if self.may_store() {
            list.gather_close();
        }
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let ready = self.prepare(list, false);
        // A pipeline with the rectangle geometry stage draws the title's
        // vertices as they are; any other gets its rectangles expanded.
        let rects = self.topology == RECTANGLES && !self.completes_rects();
        let ran = ready && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            list.inside_end();
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            dropped(self.pipeline);
        }
    }

    /// Whether the bound pipeline completes rectangles itself, in the
    /// geometry stage rectangle pipelines get (`HostPipeline::rects`).
    fn completes_rects(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rects)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&mut self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        // (P3) Its own vertex buffers go onto the host list past the record.
        self.issued.vertex = None;
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // The vertices are read in place, at submission: no change
                // of mapping under the read (see
                // `crate::watch::title_memory_reading`).
                let _mapped = crate::watch::title_memory_reading(from, n * 3 * stride);
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        self.draw_ends_window();
        if self.may_store() {
            list.gather_close();
        }
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        // Indexed rectangles are drawn by the rectangle geometry stage
        // alone (nothing expands them on the CPU).
        let lost = self.topology == RECTANGLES && !self.completes_rects();
        if lost {
            once!("indexed rectangles not drawn");
        }
        let ran = !lost && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            list.inside_end();
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// A direct draw ends the window ([`windows`]). With
    /// `VICTORY_INDIRECT_WINDOW_DRAWS=1` (off by default) only a draw whose
    /// shaders can store to buffers does: such a draw may write the
    /// arguments the window holds clamped, and no other draw touches them
    /// or the list's scratch buffer.
    fn draw_ends_window(&mut self) {
        static OVER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if self.window.is_none() {
            return;
        }
        let over = *OVER.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_WINDOW_DRAWS").is_some_and(|v| v == "1"));
        let stores = || {
            crate::pipeline::host_pipeline(self.pipeline as This)
                .is_none_or(|p| p.stages.iter().any(|stage| stage.translated().stores))
        };
        if !over || stores() {
            self.window = None;
        }
    }

    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
        arena_call: bool,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            return false;
        }
        // A program that walks ray-tracing structures nothing has built:
        // dropped before the argument clamp, like a skipped one.
        if crate::pipeline::walk_held(self.pipeline as This) {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_why("a walk of ray-tracing structures that are not built (VICTORY_BVH_HOLD=1)".to_owned());
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        };
        // Indirect rectangles (the title's tiled lighting, 256
        // indirect draws a frame, makes their corners from the vertex
        // number) are completed by the pipeline's geometry stage: the
        // arguments stay the title's, three vertices a rectangle. They
        // cannot be expanded on the CPU, which does not know their count.
        // shortcut: in a pipeline without that stage, a rectangle with no
        // per-vertex inputs is drawn as its first triangle (three vertices,
        // half of it), and one with vertex buffers is not drawn.
        let made_rects = !p.rects
            && kind == DRAW
            && self.topology == RECTANGLES
            && !p.inputs.iter().any(|e| !e.per_instance);
        if made_rects {
            once!("indirect rectangles without vertex inputs drawn as their first triangle");
        }
        let matches_kind = match kind {
            DRAW | INDEXED => {
                (self.topology != RECTANGLES || made_rects || p.rects) && !p.compute && !p.mesh
            }
            MESH => !p.compute && p.mesh,
            DISPATCH => p.compute,
            _ => false,
        };
        if !matches_kind {
            why(&format!(
                "pipeline mismatch (compute {}, mesh {}, topology {})",
                p.compute, p.mesh, self.topology
            ));
            return false;
        }
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find_any(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find_any(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        if kind == MESH && (mesh_test() == 1 || !mesh_kept(self.pipeline)) {
            why("mesh draws are switched off (VICTORY_MESH_TEST=skip)");
            return false;
        }
        // `VICTORY_INDIRECT_ARENA_CPU`: a record in the CPU-written arena
        // (finding F8), read on the CPU now and issued as a direct call
        // when nothing but the title's CPU writes it and the clamp would
        // leave it as it is (`crate::indirect::arena`); else on below.
        if arena_call && max == 1 && count.is_none() && crate::indirect::arena::on() {
            use crate::indirect::arena;
            let n = arena::record_bytes(kind);
            if n > 0 && crate::indirect::split::in_arena(arguments, n as u64) {
                // The list's budget charged by records issued this way: anew
                // for each host list, as the clamp's is.
                const ARENA_BUDGET: usize = usize::MAX - 0x79;
                if list.first_use(ARENA_BUDGET) {
                    self.arena_budget = 0;
                }
                let from = arguments as usize;
                // Per record (its own bytes): off for the run only on an F8
                // anomaly; then this list's writes before the call, then
                // writes declared in this frame or the two before (page
                // stamps first, exact ranges on a recent page), then
                // flights in the air.
                let off = arena::seen_off();
                let list_wrote = !off && list.gpu_wrote_any(from, n);
                // (`VICTORY_ARENA_BY_CHANGED`) Let through by what the
                // copy-back found changed: the flights' big declarations are
                // judged the same way.
                let (shader_written, by_changed) = if !off && !list_wrote { arena::record_test(from, n) } else { (false, false) };
                let blocked = off || list_wrote || shader_written;
                let producer_unsent = !blocked
                    && if by_changed {
                        crate::host::flights_write_except_big(from, from + n, arena::BIG)
                    } else {
                        crate::host::flights_write(from, from + n)
                    };
                let mut b = [0u8; 20];
                let _kind = crate::readback::consumers::scope(crate::readback::consumers::Kind::Records);
                let read = !blocked && !producer_unsent && crate::host::read_into(arguments as *const u8, &mut b[..n]);
                let facts = arena::Facts { on: true, off, list_wrote, shader_written, producer_unsent, budget_used: self.arena_budget };
                // A mesh call needs the newer list interface (as the clamped way does through the driver).
                let l6 = if kind == MESH { list.list.cast::<ID3D12GraphicsCommandList6>().ok() } else { None };
                let way = match arena::decide(kind, read.then_some(&b[..n]), facts) {
                    arena::Way::Direct(..) if kind == MESH && l6.is_none() => arena::Way::Fallback(arena::CLAMPED),
                    way => way,
                };
                match way {
                    arena::Way::Direct(w, cost) => {
                        self.window = None;
                        // What is recorded from here may write arguments, as on the raw way.
                        if kind == DISPATCH || p.stages.iter().any(|stage| stage.translated().stores) {
                            list.gather_close();
                        }
                        let ran = match kind {
                            DRAW | INDEXED => self.prepare(list, kind == INDEXED),
                            MESH => self.prepare_for(list, false, true),
                            _ => self.bind(list, p, true),
                        };
                        if !ran {
                            why("prepare failed");
                            return false;
                        }
                        self.arena_budget = (self.arena_budget + cost).min(arena::BUDGET);
                        #[cfg(feature = "diagnostics")]
                        list.note_op(|| self.op_text(format!("indirect kind {kind} at {arguments:#x} (arena, read on the CPU: {w:?})"), kind == DISPATCH));
                        list.inside_end();
                        // SAFETY: an open list with everything set; the values
                        // are a fixed point of the clamp (`arena::decide`).
                        unsafe {
                            match kind {
                                DRAW => list.list.DrawInstanced(w[0], w[1], w[2], w[3]),
                                INDEXED => list.list.DrawIndexedInstanced(w[0], w[1], w[2], w[3] as i32, w[4]),
                                DISPATCH => list.list.Dispatch(w[0], w[1], w[2]),
                                _ => {
                                    self.mesh_scissor(list, true);
                                    if let Some(l6) = &l6 {
                                        l6.DispatchMesh(w[0], w[1], w[2]);
                                    }
                                    self.mesh_scissor(list, false);
                                }
                            }
                        }
                        // Diagnostics: the falsifier of "the CPU bytes are
                        // final": at submission, where the clamped way's
                        // refresh would read the record, it is read again
                        // and compared with what was issued.
                        #[cfg(feature = "diagnostics")]
                        {
                            let issued = b;
                            list.defer_host(move || {
                                let mut now = [0u8; 20];
                                if crate::host::read_into(arguments as *const u8, &mut now[..n]) && now[..n] != issued[..n] {
                                    let all = arena::CHANGED.fetch_add(1, Ordering::Relaxed) + 1;
                                    if all == 1 {
                                        tracing::warn!(
                                            "arena indirect record changed after it was read: {all} (first: kind {kind} at {arguments:#x}, issued {:02x?}, at submission {:02x?})",
                                            &issued[..n],
                                            &now[..n]
                                        );
                                    }
                                }
                            });
                        }
                        crate::indirect::split::note_path(crate::indirect::split::ARENA_CPU);
                        if kind == DISPATCH {
                            DISPATCHED.fetch_add(1, Ordering::Relaxed);
                        } else {
                            DRAWN.fetch_add(1, Ordering::Relaxed);
                        }
                        return true;
                    }
                    arena::Way::Fallback(place) => {
                        arena::FELL[place].fetch_add(1, Ordering::Relaxed);
                    }
                    arena::Way::Off => {}
                }
            }
        }
        // (Neptune) `VICTORY_INDIRECT_RAW`: drawn from the title's own
        // arguments in their host copy, with no pass of ours ([`raw_all`],
        // [`raw_guard`], [`raw_indirect`]).
        let split = kind == DISPATCH && max == 1 && count.is_none() && group_base(p).is_some();
        let mesh_raw = raw_indirect() && kind == MESH && max == 1 && count.is_none();
        let trusted = raw_guard() && matches!(kind, DRAW | INDEXED) && max == 1 && count.is_none() && {
            let n = if kind == INDEXED { 20 } else { 16 };
            // (F7) The card reads the record as title memory holds it when the
            // list is prepared, and as lists ahead of it on the card leave it:
            // a CPU read now is only good if no list not yet run writes it.
            // A flight in the air that declared a write there, or, with parts
            // joined (more producers ahead of it unsent, prepared together),
            // a record outside the arena the title's CPU writes: left to the
            // clamp.
            let unsent = crate::host::flights_write(arguments as usize, arguments as usize + n)
                || (crate::queue::join_parts() && !crate::indirect::split::in_arena(arguments, n as u64));
            let slot = if list.gpu_wrote(arguments as usize, n) || crate::host::shader_written(arguments as usize, n) {
                1
            } else if unsent {
                4
            } else {
                match record_sane(kind, arguments) {
                    Some(true) => 0,
                    Some(false) => 2,
                    None => 3,
                }
            };
            crate::indirect::RAW3[slot].fetch_add(1, Ordering::Relaxed);
            slot == 0
        };
        if mesh_raw || ((raw_all() || trusted) && !split) {
            self.window = None;
            // What is recorded from here may write arguments: no more
            // records gathered for the prelude, as on the clamped way.
            if kind == DISPATCH || p.stages.iter().any(|stage| stage.translated().stores) {
                list.gather_close();
            }
            args.refresh(list, arguments, len);
            if let Some((s, at)) = &counter {
                s.refresh(list, *at, 4);
            }
            let ran = match kind {
                DRAW | INDEXED => self.prepare(list, kind == INDEXED),
                MESH => self.prepare_for(list, false, true),
                _ => self.bind(list, p, true),
            };
            if !ran {
                why("prepare failed");
                return false;
            }
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x} (raw)"), kind == DISPATCH));
            if kind == MESH {
                self.mesh_scissor(list, true);
            }
            list.inside_end();
            // SAFETY: an open list with everything set; the list keeps the
            // host copies alive, and the signature is cached.
            unsafe {
                list.list.ExecuteIndirect(
                    &signature,
                    max.max(1),
                    &args.buffer,
                    args.offset(arguments),
                    counter.as_ref().map(|(s, _)| &s.buffer),
                    counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
                )
            };
            if kind == MESH {
                self.mesh_scissor(list, false);
            }
            crate::indirect::CLAMPS[2].fetch_add(1, Ordering::Relaxed);
            crate::indirect::split::note_path(if trusted { crate::indirect::split::CPU_READ } else { crate::indirect::split::RAW });
            if kind == DISPATCH {
                DISPATCHED.fetch_add(1, Ordering::Relaxed);
            } else {
                DRAWN.fetch_add(1, Ordering::Relaxed);
            }
            return true;
        }
        // (Neptune) A single record without a count buffer, which nothing
        // the list has recorded so far was declared to write, while windows
        // are off: clamped in the list's prelude and drawn from its slot
        // there ([`clamp_ahead`]).
        let ahead = if clamp_ahead()
            && !windows()
            && kind != DISPATCH
            && max == 1
            && count.is_none()
            && !list.gpu_wrote(arguments as usize, len as usize)
        {
            crate::indirect::gather_ahead(list, (&args.buffer, args.id()), args.offset(arguments), (kind, stride))
        } else {
            None
        };
        crate::indirect::CLAMPS[usize::from(ahead.is_none())].fetch_add(1, Ordering::Relaxed);
        // A window of records clamped in one pass (`windows`): for single
        // records without a count buffer, drawn by shaders that store to no
        // buffer. Anything else ends the window: it uses the scratch buffer
        // or may write the arguments.
        const WINDOW_USE: usize = usize::MAX - 0x77;
        let offset = args.offset(arguments);
        // A dispatch that may have more groups than the host runs in one
        // dimension, by a program that takes the number of its first group.
        let parts = if kind == DISPATCH && max == 1 && count.is_none() {
            group_base(p).zip(host_signature(DISPATCH, crate::indirect::PART_BYTES as u32))
        } else {
            None
        };
        let windowed = windows()
            && kind != DISPATCH
            && max == 1
            && count.is_none()
            && !p.stages.iter().any(|stage| stage.translated().stores);
        // Another host list (a new one after a cut, or this one after its
        // reset) has another scratch buffer, or this one's with old bytes.
        if !windowed || list.first_use(WINDOW_USE) {
            self.window = None;
        }
        if !windowed {
            // A dispatch, or shaders that store to buffers: arguments may
            // be written from here on (no more records gathered, [`gathers`]).
            list.gather_close();
        }
        let same = |w: &Window| w.copy == args.id() && w.kind == kind && w.stride == stride;
        let hit = self
            .window
            .as_ref()
            .filter(|w| same(w) && offset >= w.base)
            .filter(|w| (offset - w.base) % u64::from(w.step) == 0 && (offset - w.base) / u64::from(w.step) < u64::from(w.records))
            .map(|w| (w.scratch.clone(), offset - w.base, w.step));
        // A record that misses its window: a slot the list's prelude fills,
        // if the list still gathers ([`gathers`]).
        // (Neptune) A record clamped ahead ([`clamp_ahead`]; windows are
        // off, so there is no hit) is drawn as a gathered one.
        let gathered = if ahead.is_some() {
            ahead
        } else if hit.is_none() && windowed && gathers() {
            // `VICTORY_GATHER_LATE`: once the list has stopped gathering, a
            // record no shader of the list so far was declared to write is
            // still gathered (what the prelude reads is what the draw would).
            let closed = list.gather_closed();
            let late = closed && gather_late() && !list.gpu_wrote_any(arguments as usize, len as usize);
            let slot = if late {
                crate::indirect::gather_ahead(list, (&args.buffer, args.id()), offset, (kind, stride))
            } else {
                crate::indirect::gather(list, (&args.buffer, args.id()), offset, (kind, stride))
            };
            if slot.is_some() {
                crate::indirect::GATHER_TAKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if late {
                    crate::indirect::GATHER_LATE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
            slot
        } else if kind == DISPATCH
            && gather_dispatch()
            && gathers()
            && max == 1
            && count.is_none()
            && parts.is_none()
            && !list.gpu_wrote_any(arguments as usize, len as usize)
        {
            // (Card kinds job, task 3, `VICTORY_GATHER_DISPATCH`) A single
            // dispatch record no write the list declared so far touches:
            // gathered as a draw's under `VICTORY_GATHER_LATE` (the clamp
            // treats kind 2 the same in the prelude as in the list; what
            // the prelude reads is what the dispatch would). Dispatches in
            // parts keep their own pass (`bound_split`).
            let slot = crate::indirect::gather_ahead(list, (&args.buffer, args.id()), offset, (kind, stride));
            if slot.is_some() {
                crate::indirect::GATHER_TAKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                crate::indirect::GATHER_DISPATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            slot
        } else {
            None
        };
        let way = if hit.is_some() {
            crate::indirect::split::WINDOW
        } else if gathered.is_some() {
            crate::indirect::split::GATHERED
        } else {
            crate::indirect::split::CLAMPED
        };
        let (bounded, at) = match (hit, gathered) {
            (Some((scratch, at, step)), _) => {
                // Diagnostics of windows: the record clamped again and
                // compared (`crate::indirect::window_check`).
                if crate::indirect::window_check()
                    && crate::indirect::check_hit(
                        list,
                        (&args.buffer, offset),
                        (kind, step),
                        &scratch,
                        at,
                    )
                {
                    self.host_pipeline = 0;
                    self.host_roots[1] = 0;
                    self.issued.roots_gone(true);
                }
                (scratch, at)
            }
            (None, Some((scratch, at))) => {
                // The record's bytes are up to date when the list is
                // submitted, as for a record clamped in the list.
                args.refresh(list, arguments, len);
                // Diagnostics of windows, for gathered records too: the
                // record clamped again here and compared.
                if crate::indirect::window_check()
                    && crate::indirect::check_hit(list, (&args.buffer, offset), (kind, stride), &scratch, at)
                {
                    self.host_pipeline = 0;
                    self.host_roots[1] = 0;
                    self.issued.roots_gone(true);
                }
                (scratch, at)
            }
            (None, None) => {
                // A record a short step after the last window's last one
                // (the same step as that window's, if it had one): the
                // title walks the buffer, so the ones that step apart
                // after it are clamped with it.
                let (step, records) = match &self.window {
                    Some(w) if windowed && same(w) => {
                        let last = w.base + u64::from(w.records - 1) * u64::from(w.step);
                        let step = offset.saturating_sub(last);
                        let walks = step >= u64::from(stride)
                            && step <= 256
                            && step % 4 == 0
                            && (w.records == 1 || step == u64::from(w.step));
                        if walks {
                            let fit = (args.left(arguments) / step).min(WINDOW_MOST) as u32;
                            (step as u32, (if w.records == 1 { 16 } else { w.records * 2 }).min(fit).max(1))
                        } else {
                            (stride, max)
                        }
                    }
                    _ => (stride, max),
                };
                args.refresh(list, arguments, len.max(u64::from(step) * u64::from(records)));
                if let Some((s, at)) = &counter {
                    s.refresh(list, *at, 4);
                }
                // Bounded first: the clamp binds a compute pipeline of its
                // own and copies clamped active records into a scratch
                // buffer at offset 0.
                let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
                // (Card kinds job) A clamp pass of ours inside the title's
                // list: two waits, two transitions of the scratch buffer,
                // and the title's pipeline and roots bound again after it.
                list.inside(crate::cardmeter::Inside::Clamp);
                crate::cardmeter::inside_barriers(crate::cardmeter::Inside::Clamp, 2, 2);
                crate::cardmeter::INSIDE_REBINDS.fetch_add(1, Ordering::Relaxed);
                crate::indirect::in_list_reason(kind, max, count.is_some(), parts.is_some(), windowed, list.gather_closed(), list.gpu_wrote_any(arguments as usize, len as usize));
                // More than the record asked for: a window clamped ahead,
                // whose unfilled slots must not spend the list's budget.
                let bounded = if parts.is_some() {
                    crate::indirect::bound_split(list, (&args.buffer, offset))
                } else if windowed && records > 1 {
                    crate::indirect::bound_ahead(list, (&args.buffer, offset), (kind, step, records))
                } else {
                    crate::indirect::bound(list, (&args.buffer, offset), count_arg, (kind, step, records))
                };
                let Some(bounded) = bounded else {
                    self.window = None;
                    why("not bounded (no clamp, or arguments not 4-byte aligned)");
                    return false;
                };
                self.host_pipeline = 0;
                self.host_roots[1] = 0;
                self.issued.roots_gone(true);
                self.window = windowed.then(|| Window {
                    copy: args.id(),
                    kind,
                    stride,
                    step,
                    base: offset,
                    records,
                    scratch: bounded.clone(),
                });
                (bounded, 0)
            }
        };
        let ran = match kind {
            DRAW | INDEXED => self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => self.bind(list, p, true),
            _ => false,
        };
        if !ran {
            why("prepare failed");
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        if kind == MESH {
            self.mesh_scissor(list, true);
        }
        #[cfg(feature = "diagnostics")]
        let probed = (kind == DISPATCH)
            .then(|| probe::indirect(self, list, p, &args, (arguments, max), parts.is_some()))
            .flatten();
        list.inside_end();
        // SAFETY: an open list with everything set; the scratch buffer and
        // host copies are kept alive by the list, and the signature is cached.
        unsafe {
            if let Some(((param, dword), signature)) = &parts {
                // [`crate::indirect::bound_split`]'s records, one after the
                // other, each told the number of its first group (the
                // second is empty unless the title asked for more than one
                // part: what it asked is known only on the card).
                for part in 0..crate::indirect::SPLIT {
                    list.list.SetComputeRoot32BitConstant(*param, part * crate::indirect::PART, *dword);
                    let at = u64::from(part) * crate::indirect::PART_BYTES;
                    list.list.ExecuteIndirect(signature, 1, &bounded, at, None, 0);
                }
                list.list.SetComputeRoot32BitConstant(*param, 0, *dword);
                // (P3) Its words set past the record.
                self.issued.root_gone(true, *param);
            } else {
                list.list.ExecuteIndirect(
                    &signature,
                    max,
                    &bounded,
                    at,
                    counter.as_ref().map(|(s, _)| &s.buffer),
                    counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
                )
            }
        };
        #[cfg(feature = "diagnostics")]
        probe::after(probed, list);
        if parts.is_some() {
            DISPATCH_PARTS[3].fetch_add(1, Ordering::Relaxed);
        }
        if kind == MESH {
            self.mesh_scissor(list, false);
        }
        // The scratch buffer stays readable as arguments: the next draw of
        // the window reads it with no barrier, and the next clamp moves it
        // back (`crate::indirect`).
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        crate::indirect::split::note_path(way);
        true
    }
}

/// Where a compute pipeline's program takes the number of its first thread
/// group (`victory_shader::emit::USER_GROUP_BASE`): the root parameter of
/// its user data and the dword in it. None for every pipeline with
/// `VICTORY_DISPATCH_PARTS=0` (its programs are translated without it).
fn group_base(p: &crate::pipeline::host::HostPipeline) -> Option<(u32, u32)> {
    if !victory_shader::emit::dispatch_parts() {
        return None;
    }
    let stage = p.stages.first().filter(|_| p.compute)?;
    let (param, _) = stage.user_data?;
    let base = RootType::Other(victory_shader::emit::USER_GROUP_BASE);
    let word = stage.translated().host.user_data.iter().find(|u| u.root == base)?;
    Some((param, word.index))
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn view_kind(view: &View) -> Option<ResourceType> {
    match view {
        View::Shader(..) | View::Placed(false, ..) => Some(ResourceType::Srv),
        View::Unordered(..) | View::Placed(true, ..) => Some(ResourceType::Uav),
        _ => None,
    }
}

/// (fxceless a35c557) The format of a host view of `texture` whose description names
/// `described` (none without a description): that one, or the texture's own, which is what
/// the host gives a view without a format.
fn bound_format(texture: &ID3D12Resource, described: Option<DXGI_FORMAT>) -> u32 {
    match described {
        Some(format) if format.0 != 0 => format.0 as u32,
        // SAFETY: a live texture.
        _ => unsafe { texture.GetDesc() }.Format.0 as u32,
    }
}

/// The host state a depth view needs of its texture: the read state if the
/// view writes no plane the format has (the title's read-only flags), else
/// the write state.
/// shortcut: whole textures have one state, so a view that writes one of
/// two planes counts as writing.
fn depth_state(texture: &ID3D12Resource, desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>) -> u32 {
    // D16_UNORM and D32_FLOAT: depth formats without a stencil plane.
    const ONE_PLANE: [i32; 2] = [55, 40];
    let Some(desc) = desc else {
        return DEPTH_WRITE;
    };
    let format = if desc.Format.0 == 0 {
        // SAFETY: a live texture; a view without a format has its own.
        unsafe { texture.GetDesc() }.Format.0
    } else {
        desc.Format.0
    };
    let mut needed = D3D12_DSV_FLAG_READ_ONLY_DEPTH.0;
    if !ONE_PLANE.contains(&format) {
        needed |= D3D12_DSV_FLAG_READ_ONLY_STENCIL.0;
    }
    if desc.Flags.0 & needed == needed {
        DEPTH_READ
    } else {
        DEPTH_WRITE
    }
}

thread_local! {
    /// (fxceless 5d659e0) While a draw's stages make their tables ([`table_stages`]): what
    /// the stages read of each table ([`stage_spans`]), and the tables its earlier stages
    /// made afresh (address, slots).
    static STAGE_SPANS: std::cell::RefCell<(Vec<(u8, u32, u32)>, Vec<(u64, u32)>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// (fxceless 5d659e0) Whether the stages of one draw share a table they both need made
/// afresh (`VICTORY_TABLE_STAGES=1`; off by default as its author's: each stage makes its
/// own). A mesh stage and a pixel stage that read one root table to different depths made
/// it twice in dynamic slots of their own (the count is in the key of `fresh_tables`); with
/// this both ask for the larger count and the second finds the first one's table: fewer
/// slots of the dynamic ring, fewer tables refused for a full ring (fxceless run 300:
/// 1.79 -> 0.48 % refused; frame time +7.4 ms there, so off by default).
fn table_stages() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TABLE_STAGES").is_some_and(|v| v == "1"))
}

/// For [`table_stages`]: per root table the stages of `p` read, the slots it has up to the
/// last entry read by a stage that also reads a constant view in it (0: no stage does), and
/// up to the last entry any stage reads: (root parameter, with constants, all).
fn stage_spans(p: &HostPipeline) -> Vec<(u8, u32, u32)> {
    let mut spans: Vec<(u8, u32, u32)> = Vec::new();
    for stage in &p.stages {
        let t = stage.translated();
        let mut constant = [false; 256];
        for u in t.used.iter().filter(|u| u.root == RootType::Table && u.kind == ResourceType::Cbv) {
            constant[usize::from(u.slot)] = true;
        }
        for u in t.used.iter().filter(|u| u.root == RootType::Table) {
            let n = u.table_offset_dwords / 8 + 1;
            let with = if constant[usize::from(u.slot)] { n } else { 0 };
            match spans.iter_mut().find(|s| s.0 == u.slot) {
                Some(s) => {
                    s.1 = s.1.max(with);
                    s.2 = s.2.max(n);
                }
                None => spans.push((u.slot, with, n)),
            }
        }
    }
    spans
}

thread_local! {
    /// (fxceless #927) While a stage's tables are made: the table places it moves 16-bit
    /// halves through (`HostShader::half_views`).
    static HALF_PLACES: std::cell::RefCell<Vec<(u8, u32)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// (fxceless #927) Since the start: host slots of tables made with the unsigned twin of a
/// 16-bit float or signed buffer view (`descriptor::host::half_view`).
pub(crate) static HALF_VIEWS_SHOWN: AtomicU64 = AtomicU64::new(0);

/// (fxceless #927) Logs, once per shader (at most 256), that a stage moves 16-bit halves as
/// unsigned integers through table places: the face programs (`compute_graph_ops`,
/// `ped_standard_head_cs`, `ped_skinning_cs`) should show here.
fn note_half_views(pipeline: &str, shader: &str, places: &[(u8, u32)]) {
    static SEEN: std::sync::Mutex<Option<std::collections::HashSet<String>>> = std::sync::Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let seen = seen.get_or_insert_with(Default::default);
    if seen.len() >= 256 || !seen.insert(shader.to_owned()) {
        return;
    }
    tracing::info!(
        pipeline,
        shader,
        places = ?places,
        "a shader moves 16-bit halves as unsigned integers through these table places (root parameter, dword): \
         16-bit float views there are shown as their unsigned twin (VICTORY_HALF_VIEWS=0: off, needs a re-warmed shader cache)"
    );
}

/// (Neptune) Whether a table made afresh is remembered as ready within its host list
/// (`VICTORY_READY_TABLES=0`: its views are made ready again for every draw, as before).
fn ready_tables_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_READY_TABLES").is_none_or(|v| v != "0"))
}

/// (Neptune) Whether a list's tables made afresh are told apart by their contents alone
/// (`VICTORY_TABLES_BY_CONTENT=0`: by their address too, as before).
fn tables_by_content() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TABLES_BY_CONTENT").is_none_or(|v| v != "0"))
}

/// (Neptune) Since they were last taken: tables made afresh that a draw found ready,
/// found made but made ready again, made, the slots made for them, at a new address
/// that took another's slots; and why the second kind were not ready (no note,
/// another host list, a change in the list, the tested depth target).
pub(crate) static TABLES: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

/// `record`: the title descriptor the view is in (0: not known), for the
/// diagnostics of views refused or given their own bytes.
fn readable(list: &HostList, kind: ResourceType, view: &View, record: u64) {
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    // (F11) The texture views draws bind, meter only.
    #[cfg(feature = "diagnostics")]
    if kind == ResourceType::Srv {
        crate::texdetail::bound(view, placed.as_ref(), record);
    }
    let view = placed.as_ref().unwrap_or(view);
    if kind == ResourceType::Uav
        && crate::readback::terrain::on()
        && let Some((address, len)) = crate::descriptor::host::buffer_range(view)
    {
        let copy = crate::descriptor::host::buffer_copy(view).is_some();
        crate::readback::terrain::uav(address, len, copy, STORES.get() || !write_skip());
    }
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((shadow, address, len)) = crate::descriptor::host::buffer_copy(view) {
        if let Some(own) = crate::shadow::flagged(address, len) {
            crate::shadow::flagged_use(STAGE_HASH.get(), record, kind == ResourceType::Uav, own, address, len, false);
        }
        shadow.refresh(list, address, len);
        if kind == ResourceType::Uav {
            // A shader that cannot store to buffers leaves the range as
            // the title has it: nothing to find afterwards.
            if STORES.get() || !write_skip() {
                let ctx = crate::declstat::Ctx {
                    way: crate::declstat::TABLE_UAV,
                    hash: 0,
                    view: crate::descriptor::host::view_fields(view),
                };
                crate::declstat::with(ctx, || shadow.gpu_writes(list, address, len));
            } else {
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOut, 1);
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOutBytes, len);
                }
            }
        }
    }
    match (view, kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            // A view the title placed, in another format, in the memory of
            // a texture the host writes: that texture's texels, copied on
            // the host (title memory does not have them).
            let aliased = crate::resource::alias_of(*resource as This)
                .is_some_and(|source| crate::texture::alias(list, *resource as This, source));
            if !aliased
                && let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                // Of a texture the title placed, the mips the view shows
                // (`VICTORY_TEXTURE_FIRST_MIP`).
                let first = match (&placed, view) {
                    (Some(_), View::Shader(_, Some(desc))) => crate::resource::first_mip(desc),
                    _ => 0,
                };
                crate::texture::upload_from(list, *resource as This, t, first);
            }
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(*resource as This, false, false);
            // The depth target the draw only tests is read in both ways at
            // once: both read states, which its preparation asks for too.
            let (tested, _) = TESTED_DEPTH.get();
            let state = if tested != 0 && tested == *resource {
                TESTED_DEPTH.set((tested, true));
                DEPTH_READ | SHADER_RESOURCE
            } else {
                SHADER_RESOURCE
            };
            crate::list::ensure_state(list, *resource as This, state);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            // Unordered access to a depth texture is its colour twin's.
            if !list.twin_use(*resource as This, true, true) {
                crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            }
        }
        _ => {}
    }
}

/// The pixel shaders whose draws get their slots listed
/// ([`State::slots_text`]): `VICTORY_LIST_SLOTS`, hash prefixes separated
/// by commas. None by default.
#[cfg(feature = "diagnostics")]
fn list_slots() -> &'static [String] {
    static LIST: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        std::env::var("VICTORY_LIST_SLOTS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    })
}

/// A fingerprint of the 4 KiB of title memory at `at` (FNV-1a, 64 bits; of
/// 4 KiB of zeros it is always the same value, which `slots.py` knows), or
/// "unreadable".
#[cfg(feature = "diagnostics")]
fn print_4k(at: u64) -> String {
    let mut bytes = [0u8; 4096];
    if at == 0 || !crate::host::read_into(at as *const u8, &mut bytes) {
        return "unreadable".into();
    }
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for b in bytes {
        hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// The `bytes` of title memory at `at`: how many are mapped, and the
/// fingerprints of the first and the last 4 KiB (a texture's smallest
/// levels lie first, its largest last).
#[cfg(feature = "diagnostics")]
fn memory_state(at: u64, bytes: u64) -> String {
    let mapped = crate::texture::mapped_bytes(at as *const u8, bytes as usize);
    let last = at + bytes.saturating_sub(4096);
    format!("memory={at:#x} mapped={mapped}/{bytes} first4k={} last4k={}", print_4k(at), print_4k(last))
}

/// Title texture `this` for the frame listing: its size, whether it has a
/// host copy (not made here), its staging buffer's state, whether the host
/// copied into it, and its memory ([`memory_state`]).
#[cfg(feature = "diagnostics")]
fn texture_state(this: This) -> String {
    let Some((at, desc)) = crate::resource::texture_memory(this) else {
        return "not a texture".into();
    };
    // SAFETY: 56 bytes in the description's layout.
    let words = unsafe { std::ptr::from_ref(&desc).cast::<[u32; 14]>().read_unaligned() };
    let bytes = crate::resource::placed_bytes(&words).unwrap_or(0);
    let staged = match crate::texture::staged_state(this) {
        Some((uploaded, dirty)) => format!("staged=yes uploaded={uploaded} dirty={dirty}"),
        None => "staged=no".into(),
    };
    format!(
        "{}x{} f{} levels={} host_copy={} {staged} host_written={} title_fills={} {}",
        u64::from(words[4]) | u64::from(words[5]) << 32,
        words[6],
        words[8],
        words[7] >> 16,
        crate::resource::host_made(this).is_some(),
        crate::texture::host_wrote(this as usize),
        crate::resource::title_fills(this),
        memory_state(at as u64, bytes)
    )
}

/// A view of the title's heap for the frame listing: a texture with its
/// state ([`texture_state`]; a placed one that has no resource yet with
/// its memory alone), anything else in a few words.
#[cfg(feature = "diagnostics")]
fn view_state(view: Option<&View>) -> String {
    match view {
        None => "no view".into(),
        Some(View::Placed(false, r, d)) if r[0] != 1 => {
            let address = u64::from(d[10]) | u64::from(d[11]) << 32;
            let state = match crate::resource::texture_at_made(address, r) {
                Some(this) => texture_state(this),
                None => format!(
                    "{}x{} f{} levels={} resource=none {}",
                    u64::from(r[4]) | u64::from(r[5]) << 32,
                    r[6],
                    r[8],
                    r[7] >> 16,
                    memory_state(address, crate::resource::placed_bytes(r).unwrap_or(0))
                ),
            };
            // (Last, so that readers of the older lines still find
            // theirs: the level the view starts at, and how many it shows.)
            format!("placed texture view_format={} {state} view_levels={}+{}", d[0], d[4], d[5] as i32)
        }
        Some(View::Shader(r, d)) if crate::resource::texture_memory(*r as This).is_some() => {
            format!(
                "texture object {r:#x} view_format={} {} view_levels={}",
                d.as_ref().map_or(0, |d| d[0]),
                texture_state(*r as This),
                d.as_ref().map_or("all".to_owned(), |d| format!("{}+{}", d[4], d[5] as i32))
            )
        }
        Some(v) => format!("{v:?}").chars().take(80).collect(),
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_made(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!("{}x{} f{} flags {:#x} of {r:?}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        Some(v @ View::Unordered(r, ..)) => match texture(*r as This) {
            Some(t) => format!("uav texture {t}"),
            None => format!(
                "uav resource {r:#x} buffer {:?}",
                crate::descriptor::host::buffer_range(v)
            ),
        },
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || format!("STENCIL REF {reference:#x}"));
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

pub(crate) fn depth_bounds(list: &HostList, min: f32, max: f32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || {
        format!("DEPTH BOUNDS {min}..{max}")
    });
    if crate::host::host().is_some_and(|h| h.depth_bounds())
        && let Ok(l1) = list.list.cast::<ID3D12GraphicsCommandList1>()
    {
        // SAFETY: an open list on a device that supports depth bounds.
        unsafe { l1.OMSetDepthBounds(min, max) };
    }
}
/// (P3, recording cost) The bind filter's rules against a model of what a
/// D3D12 list holds, and the per-draw cost with the filter on and off.
#[cfg(test)]
mod filter_tests {
    use super::*;

    /// What a D3D12 command list holds of root arguments, as D3D12 defines
    /// it: set values stay until the root signature of their kind is set
    /// (then nothing is defined), or something sets them again.
    #[derive(Default)]
    struct Model {
        roots: [Vec<Option<Given>>; 2],
        vertex: Option<Vec<u64>>,
    }

    impl Model {
        fn set(&mut self, compute: bool, param: u32, value: Given) {
            let r = &mut self.roots[usize::from(compute)];
            if r.len() <= param as usize {
                r.resize(param as usize + 1, None);
            }
            r[param as usize] = Some(value);
        }
        fn signature(&mut self, compute: bool) {
            self.roots[usize::from(compute)].clear();
        }
        fn get(&self, compute: bool, param: u32) -> Option<&Given> {
            self.roots[usize::from(compute)].get(param as usize).and_then(Option::as_ref)
        }
    }

    /// What the title list's state and `bind` do, reduced to the filter: the
    /// root signature bound per kind, the record, and the host list (the model).
    struct Rig {
        on: bool,
        serial: u64,
        host_roots: [usize; 2],
        issued: Issued,
        host: Model,
        sets: u64,
        skips: u64,
    }

    impl Rig {
        fn new(on: bool) -> Rig {
            Rig { on, serial: 1, host_roots: [0; 2], issued: Issued::default(), host: Model::default(), sets: 0, skips: 0 }
        }
        /// `bind`'s root signature step.
        fn signature(&mut self, compute: bool, root: usize) {
            if self.host_roots[usize::from(compute)] != root {
                self.host.signature(compute);
                self.host_roots[usize::from(compute)] = root;
                self.issued.at_serial(self.serial).roots_gone(compute);
            }
        }
        /// `bind`'s root argument step.
        fn root(&mut self, compute: bool, param: u32, value: Given) {
            let same = match &value {
                Given::Constants(words) => self.issued.words_in(self.serial, self.on, compute, param, words),
                v => self.issued.root_in(self.serial, self.on, compute, param, v.clone(), F_VIEWS),
            };
            if same {
                self.skips += 1;
            } else {
                self.sets += 1;
                self.host.set(compute, param, value);
            }
        }
        fn vertex(&mut self, addresses: &[u64]) {
            let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
            for (v, &a) in views.iter_mut().zip(addresses) {
                *v = D3D12_VERTEX_BUFFER_VIEW { BufferLocation: a, SizeInBytes: 64, StrideInBytes: 16 };
            }
            if !self.issued.vertex_in(self.serial, self.on, &views, addresses.len()) {
                self.host.vertex = Some(addresses.to_vec());
            }
        }
        /// Heaps changed: `bind` sets them and forgets the root signatures.
        fn heaps(&mut self) {
            self.host_roots = [0; 2];
        }
        /// A clamp pass of ours: its own compute root signature and arguments
        /// on the host list, and `execute_indirect`'s forgetting.
        fn clamp(&mut self) {
            self.host.signature(true);
            self.host.set(true, 0, Given::Constants(vec![0xdead; 7]));
            self.host_roots[1] = 0;
            self.issued.roots_gone(true);
        }
        /// A dispatch in parts: its first parameter's group base set apart.
        fn parts(&mut self, param: u32) {
            self.host.set(true, param, Given::Constants(vec![0x7777]));
            self.issued.root_gone(true, param);
        }
        /// Rectangles' own vertex buffers.
        fn rectangles(&mut self) {
            self.host.vertex = Some(vec![0xbad]);
            self.issued.vertex = None;
        }
        /// The title's list reset (a new recording: `State::reset`, a new
        /// host list recording number), or a cut (`replay`).
        fn reset(&mut self) {
            self.serial += 1;
            self.host = Model::default();
            self.host_roots = [0; 2];
        }
    }

    /// A small deterministic random sequence.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, n: u64) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) % n
        }
    }

    /// Every draw of a filtered sequence finds on the host list exactly what
    /// the unfiltered sequence gives it, through root signature changes,
    /// heap changes, clamp passes, dispatch parts, rectangles and resets.
    #[test]
    fn a_filtered_sequence_leaves_the_host_list_as_the_unfiltered_one() {
        let (mut on, mut off) = (Rig::new(true), Rig::new(false));
        let mut r = Lcg(77);
        for _ in 0..20_000 {
            // Something between draws, now and then.
            match r.next(40) {
                0 => [&mut on, &mut off].iter_mut().for_each(|g| g.heaps()),
                1 => [&mut on, &mut off].iter_mut().for_each(|g| g.clamp()),
                2 => [&mut on, &mut off].iter_mut().for_each(|g| g.parts(1)),
                3 => [&mut on, &mut off].iter_mut().for_each(|g| g.rectangles()),
                4 => [&mut on, &mut off].iter_mut().for_each(|g| g.reset()),
                _ => {}
            }
            let compute = r.next(4) == 0;
            // Another root signature now and then, as pipelines change.
            let root = if r.next(10) == 0 { 2 } else { 1 };
            // Few values, so that most repeat.
            let words = vec![r.next(3) as u32, 5, r.next(2) as u32];
            let view = Given::View(GIVEN_CBV, 0x1000 * (1 + r.next(3)));
            let table = Given::Table(0x40 * (1 + r.next(2)));
            let vertex: Vec<u64> = (0..1 + r.next(2)).map(|i| 0x10_000 * (i + r.next(2))).collect();
            for g in [&mut on, &mut off] {
                g.signature(compute, root);
                g.root(compute, 0, Given::Constants(words.clone()));
                g.root(compute, 1, view.clone());
                g.root(compute, 2, table.clone());
                if !compute {
                    g.vertex(&vertex);
                }
                // The draw: what it reads is what the title gave.
                assert_eq!(g.host.get(compute, 0), Some(&Given::Constants(words.clone())));
                assert_eq!(g.host.get(compute, 1), Some(&view));
                assert_eq!(g.host.get(compute, 2), Some(&table));
                if !compute {
                    assert_eq!(g.host.vertex.as_deref(), Some(vertex.as_slice()));
                }
            }
        }
        assert_eq!(off.skips, 0);
        // The values are random among a few: a fair share repeats (the title repeats far more).
        assert!(on.skips * 10 > on.skips + on.sets, "the filter left out {} of {}", on.skips, on.skips + on.sets);
        eprintln!("bind filter: {} root arguments set, {} left out (unfiltered: {} set)", on.sets, on.skips, off.sets);
    }

    /// The rules one at a time: a root signature set (even the same one
    /// again after a heap change), a clamp, parts and a reset each make the
    /// next draw set its arguments again.
    #[test]
    fn each_invalidation_makes_the_next_draw_set_again() {
        let mut g = Rig::new(true);
        let words = || Given::Constants(vec![1, 2, 3]);
        let draw = |g: &mut Rig, compute: bool| {
            g.signature(compute, 1);
            g.root(compute, 0, words());
        };
        draw(&mut g, false);
        draw(&mut g, false);
        assert_eq!((g.sets, g.skips), (1, 1), "the same draw again is left out");
        g.signature(false, 2);
        g.root(false, 0, words());
        assert_eq!(g.sets, 2, "another root signature: set again");
        g.heaps();
        g.signature(false, 2);
        g.root(false, 0, words());
        assert_eq!(g.sets, 3, "heaps changed: the root signature and its arguments set again");
        draw(&mut g, true);
        draw(&mut g, true);
        let before = g.sets;
        g.clamp();
        draw(&mut g, true);
        assert_eq!(g.sets, before + 1, "after a clamp pass of ours");
        g.parts(0);
        draw(&mut g, true);
        assert_eq!(g.sets, before + 2, "after a dispatch in parts");
        g.reset();
        draw(&mut g, false);
        assert_eq!(g.sets, before + 3, "after a reset");
        // Graphics arguments survive a compute pass of ours (separate in D3D12).
        let before = g.sets;
        g.clamp();
        draw(&mut g, false);
        assert_eq!(g.sets, before, "graphics roots are not the clamp's");
        assert_eq!(g.host.get(false, 0), Some(&words()));
    }

    /// The record of lookups holds while its key stands, and is dropped when
    /// any part moves (recording, refresh count, the copies' epoch).
    #[test]
    fn lookups_are_kept_only_while_nothing_moved() {
        let mut looked = Looked::default();
        let key = std::cell::Cell::new((1, 0, 1));
        let looks = std::cell::Cell::new(0);
        let ask = |looked: &mut Looked, keep: bool| {
            looked.get_in(true, || key.get(), (0x1000, 64, LOOK_VERTEX), || {
                looks.set(looks.get() + 1);
                Some((0xabc, keep))
            })
        };
        assert_eq!(ask(&mut looked, true), Some(0xabc));
        assert_eq!(ask(&mut looked, true), Some(0xabc));
        assert_eq!(looks.get(), 1, "the second ask is answered from the record");
        for moved in [(2, 0, 1), (2, 1, 1), (2, 1, 2)] {
            key.set(moved);
            ask(&mut looked, true);
            ask(&mut looked, true);
        }
        assert_eq!(looks.get(), 4, "each move looks once more");
        // An answer not to be kept (a fill, a copy on the card) is looked for every time.
        let mut other = Looked::default();
        ask(&mut other, false);
        ask(&mut other, false);
        assert_eq!(looks.get(), 6);
        // Off: always looked for.
        let mut off = Looked::default();
        for _ in 0..3 {
            off.get_in(false, || key.get(), (1, 1, LOOK_INDEX), || {
                looks.set(looks.get() + 1);
                Some((1, true))
            });
        }
        assert_eq!(looks.get(), 9);
    }

    /// The per-draw cost on a WARP list: N draws with the same state (and
    /// slowly changing constants), each looking up two vertex buffers, an
    /// index buffer and two root views of a registered title buffer and
    /// giving them to the list, with the filter and the lookup record on and
    /// off. Prints microseconds per draw; asserts only that the list's state
    /// calls were left out. Run with `VICTORY_D3D12_WARP=1` (diagnostics) or
    /// on any host device.
    #[test]
    fn per_draw_cost_with_the_filter_on_and_off() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let Some(host) = crate::host::host() else {
            eprintln!("per_draw_cost: skipped, no host device");
            return;
        };
        const LEN: usize = 1 << 20;
        // SAFETY: a plain allocation of our own ("title memory").
        let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as u64;
        assert_ne!(title, 0);
        crate::shadow::register(0x5033_0001, title, LEN as u64);
        let params: Vec<D3D12_ROOT_PARAMETER1> = std::iter::once(D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 16 } },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        })
        .chain((1..3).map(|r| D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 { ShaderRegister: r, RegisterSpace: 0, Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        }))
        .collect();
        let Ok((root, _)) = crate::pipeline::host::root_signature(&host.device, &params, &[], false) else {
            eprintln!("per_draw_cost: skipped, no root signature");
            return;
        };
        const N: u32 = 20_000;
        let mut results = Vec::new();
        for (on, changing) in [(false, false), (true, false), (false, true), (true, true)] {
            let list = host.new_list().expect("a host list");
            let mut issued = Issued::default();
            let mut looked = Looked::default();
            let skipped_before = FILTERED[F_CONSTANTS + 1].load(Ordering::Relaxed);
            // SAFETY: an open list and a live root signature.
            unsafe { list.list.SetGraphicsRootSignature(&root) };
            let started = std::time::Instant::now();
            for draw in 0..N {
                let key = || (list.serial(), list.refresh_epoch(), crate::shadow::epoch());
                let mut find = |at: u64, len: u64, kind: u8| {
                    looked.get_in(on, key, (at, len, kind), || {
                        let s = crate::shadow::find_any(at, len)?;
                        s.refresh(&list, at, len);
                        Some((s.gpu(at), true))
                    })
                };
                let v0 = find(title, 4096, LOOK_VERTEX).unwrap();
                let v1 = find(title + 4096, 4096, LOOK_VERTEX).unwrap();
                let ib = find(title + 8192, 4096, LOOK_INDEX).unwrap();
                let c0 = find(title + 0x10000, 0, LOOK_CBV).unwrap();
                let c1 = find(title + 0x20000, 0, LOOK_CBV).unwrap();
                let mut words = [7u32; 16];
                if changing {
                    words[0] = draw / 8;
                }
                let serial = list.serial();
                let l = &list.list;
                // SAFETY: an open list; addresses of live host buffers.
                unsafe {
                    if !issued.words_in(serial, on, false, 0, &words) {
                        l.SetGraphicsRoot32BitConstants(0, 16, words.as_ptr().cast(), 0);
                    }
                    for (param, gpu) in [(1, c0), (2, c1)] {
                        if !issued.root_in(serial, on, false, param, Given::View(GIVEN_CBV, gpu), F_VIEWS) {
                            l.SetGraphicsRootConstantBufferView(param, gpu);
                        }
                    }
                    let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
                    views[0] = D3D12_VERTEX_BUFFER_VIEW { BufferLocation: v0, SizeInBytes: 4096, StrideInBytes: 32 };
                    views[1] = D3D12_VERTEX_BUFFER_VIEW { BufferLocation: v1, SizeInBytes: 4096, StrideInBytes: 16 };
                    if !issued.vertex_in(serial, on, &views, 2) {
                        l.IASetVertexBuffers(0, Some(&views[..2]));
                    }
                    if !issued.index_in(serial, on, (ib, 4096, 57)) {
                        l.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW { BufferLocation: ib, SizeInBytes: 4096, Format: DXGI_FORMAT(57) }));
                    }
                }
            }
            let us = started.elapsed().as_secs_f64() * 1e6 / f64::from(N);
            let skipped = FILTERED[F_CONSTANTS + 1].load(Ordering::Relaxed) - skipped_before;
            if on {
                assert!(skipped > 0, "the filter left nothing out");
            }
            // SAFETY: our open list.
            let _ = unsafe { list.list.Close() };
            results.push(format!(
                "filter and lookup record {}, constants {}: {us:.2} us a draw",
                if on { "on" } else { "off" },
                if changing { "changing every 8th draw" } else { "the same" }
            ));
        }
        crate::shadow::unregister(0x5033_0001, title);
        // SAFETY: our allocation; the copy no longer reads it (nothing submitted).
        unsafe { VirtualFree(title as *mut _, 0, MEM_RELEASE) };
        eprintln!("per_draw_cost (5 lookups and 5 list calls a draw, {} device): {}", if std::env::var_os("VICTORY_D3D12_WARP").is_some() { "WARP" } else { "host" }, results.join("; "));
    }
}
