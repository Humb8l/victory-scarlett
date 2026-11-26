//! What the card's time goes to, by kind of list, in every run.
//!
//! The card's clock (`crate::host`) says how long the card had work of
//! ours, not which work. The timing run (`VICTORY_GPU_TIMING`) names every
//! title command but sends each list on its own, which changes the frame,
//! and it never timed the lists the layer adds for itself: the one that
//! brings copies in the card's memory up to date and hashes their pieces
//! again, and the one that compares pieces after a submission's lists.
//! Run 187's table left 64 ms a frame of card time unexplained beside
//! them (40 ms by the count outside timing runs); with draw windows off
//! the card is the frame's limit (run 225: 85 to 98 % busy), so that sum
//! decides what is worth changing.
//!
//! Here a list gets one timestamp where it begins and one where it ends,
//! both resolved into one read-back buffer by the list itself. The card
//! writes them when it gets there; a pair is read a second and a half
//! after it was recorded, when every flight of that time has long landed,
//! and its length goes to the frame log's totals by kind. So the frame
//! log's card columns are of the frames of a moment ago: right for the
//! usual frame, not for placing one long frame.
//!
//! On by default in diagnostics builds; `VICTORY_CARD_METER=0` or `1` says
//! otherwise. Two queries and a resolve a list.

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

/// What a timed list is.
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    /// Ours: changed bytes go up to copies in the card's memory and their
    /// pieces are hashed again (`Host::card_before`).
    Sync,
    /// Ours: the compare of the pieces a submission's shaders may have
    /// written, and the fetch of the flagged ones inside the flight.
    Check,
    /// Every other host list: the title's own, our present copies.
    Lists,
    /// Inside one of those: from where a pipeline is bound to where the
    /// next one is, or the list ends ([`begin_named`]).
    Pipeline,
    /// A title list's prelude (`HostList::prelude`), with [`kinds`].
    Prelude,
}

/// Pairs of timestamps in the ring.
const PAIRS: u32 = 32768;
/// A pair is read this long after it was recorded.
const AFTER: std::time::Duration = std::time::Duration::from_millis(1000);

/// A pair being recorded: its place in the ring, or none.
#[derive(Clone, Copy)]
pub(crate) struct Span(u32);

impl Span {
    pub(crate) const NONE: Span = Span(u32::MAX);
    /// For keeping in an atomic.
    pub(crate) fn raw(self) -> u32 {
        self.0
    }
    pub(crate) fn from_raw(raw: u32) -> Span {
        Span(raw)
    }
}

struct Meter {
    heap: ID3D12QueryHeap,
    back: ID3D12Resource,
    /// The read-back buffer's bytes, mapped for good.
    mapped: usize,
    /// Ticks a second of the queue's clock.
    frequency: u64,
    next: u32,
    /// Pairs recorded and not read yet: place, kind, when recorded.
    pending: std::collections::VecDeque<(u32, Kind, std::time::Instant)>,
    /// What the list of a pair mostly is ([`end_named`]), by place, until
    /// the pair is read.
    names: std::collections::HashMap<u32, String>,
    /// Since the last table: by name, the lists and their microseconds.
    by_name: std::collections::HashMap<String, (u64, u64)>,
    /// The same by pipeline ([`begin_named`]): the times it was bound and
    /// the microseconds until the next one was.
    by_pipeline: std::collections::HashMap<String, (u64, u64)>,
    /// Frames since the last table.
    frames: u64,
    /// What each place held when it was last read: a place the card has
    /// not written since (its list never ran) still holds that, and is
    /// not counted again.
    seen: Vec<(u64, u64)>,
}

// SAFETY: host objects (free-threaded) and a mapping only read, behind a lock.
unsafe impl Send for Meter {}

static METER: Mutex<Option<Option<Meter>>> = Mutex::new(None);

/// Whether lists are timed.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var_os("VICTORY_CARD_METER") {
        Some(v) => v != "0",
        None => cfg!(feature = "diagnostics"),
    })
}

fn make() -> Option<Meter> {
    let host = crate::host::host()?;
    let device = host.device();
    let mut heap: Option<ID3D12QueryHeap> = None;
    let mut back: Option<ID3D12Resource> = None;
    let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: u64::from(PAIRS) * 16,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    // SAFETY: plain object creation; the buffer is mapped once and stays so.
    unsafe {
        device
            .CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: PAIRS * 2, NodeMask: 0 },
                &mut heap,
            )
            .ok()?;
        device
            .CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back)
            .ok()?;
        let (heap, back) = (heap?, back?);
        let mut mapped = std::ptr::null_mut();
        back.Map(0, None, Some(&mut mapped)).ok()?;
        let frequency = host.timestamp_frequency();
        if mapped.is_null() || frequency == 0 {
            return None;
        }
        tracing::info!(
            "the card's time is measured by kind of list, a timestamp where each list begins and ends (VICTORY_CARD_METER=0: not)"
        );
        Some(Meter {
            heap,
            back,
            mapped: mapped as usize,
            frequency,
            next: 0,
            pending: Default::default(),
            names: Default::default(),
            by_name: Default::default(),
            by_pipeline: Default::default(),
            frames: 0,
            seen: vec![(0, 0); PAIRS as usize],
        })
    }
}

/// A list of `kind` begins here: the first timestamp of a pair.
pub(crate) fn begin(list: &ID3D12GraphicsCommandList, kind: Kind) -> Span {
    if !on() {
        return Span::NONE;
    }
    let mut meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(m) = meter.get_or_insert_with(make) else {
        return Span::NONE;
    };
    // The ring must not come round to pairs that were not read yet.
    if m.pending.len() as u32 >= PAIRS - 64 {
        return Span::NONE;
    }
    let pair = m.next;
    m.next = (m.next + 1) % PAIRS;
    m.pending.push_back((pair, kind, std::time::Instant::now()));
    // SAFETY: an open list and our query heap.
    unsafe { list.EndQuery(&m.heap, D3D12_QUERY_TYPE_TIMESTAMP, pair * 2) };
    Span(pair)
}

/// The list ends here: the pair's second timestamp, and both go to the
/// read-back buffer when the card has run the list.
pub(crate) fn end(list: &ID3D12GraphicsCommandList, span: Span) {
    if span.0 == u32::MAX {
        return;
    }
    let meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(m)) = meter.as_ref() else {
        return;
    };
    // SAFETY: an open list, our heap and our read-back buffer, the pair's
    // own sixteen bytes of it.
    unsafe {
        list.EndQuery(&m.heap, D3D12_QUERY_TYPE_TIMESTAMP, span.0 * 2 + 1);
        list.ResolveQueryData(&m.heap, D3D12_QUERY_TYPE_TIMESTAMP, span.0 * 2, 2, &m.back, u64::from(span.0) * 16);
    }
}

/// Whether the card's time is also measured pipeline by pipeline inside the
/// title's lists (`VICTORY_CARD_METER_PIPELINES=1`; off by default: two
/// more queries and a resolve at every change of pipeline).
pub(crate) fn pipelines() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| on() && std::env::var_os("VICTORY_CARD_METER_PIPELINES").is_some_and(|v| v == "1"))
}

/// The pipeline `name` is bound on `list` here: the first timestamp of a
/// pair that [`end`] closes where the next pipeline is bound or the list
/// ends. What the card does between the two (the draws or dispatches of
/// that pipeline, and what the layer records around them) is that
/// pipeline's time in the table.
pub(crate) fn begin_named(list: &ID3D12GraphicsCommandList, name: &str) -> Span {
    let span = begin(list, Kind::Pipeline);
    if span.0 != u32::MAX {
        let mut meter = METER.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(Some(m)) = meter.as_mut() {
            m.names.insert(span.0, name.to_owned());
        }
    }
    span
}

/// [`end`] for one of the title's lists, with what it mostly is: the
/// pipelines it binds (`names`, in order, as the list keeps them). Every
/// 256 frames the card's time is logged by that name: which of the title's
/// passes the card's time goes to, in a run like any other (the timing run
/// that names every command sends each list alone and needs the
/// device-removed notes).
pub(crate) fn end_named(list: &ID3D12GraphicsCommandList, span: Span, names: &std::collections::BTreeSet<String>) {
    if span.0 == u32::MAX {
        return;
    }
    end(list, span);
    let name = name_of(names);
    let mut meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(Some(m)) = meter.as_mut() {
        m.names.insert(span.0, name);
    }
}

/// A list's name in the table: how many pipelines it binds and the first
/// two of them (the set is in the order of the names, so a pass has the
/// same name every frame).
fn name_of(names: &std::collections::BTreeSet<String>) -> String {
    let short = |n: &String| n.chars().take(60).collect::<String>();
    let mut first = names.iter();
    match (first.next(), first.next()) {
        (None, _) => "no pipeline (copies, clears, barriers)".to_string(),
        (Some(a), None) => format!("1 pipeline: {}", short(a)),
        (Some(a), Some(b)) => format!("{} pipelines: {} / {}", names.len(), short(a), short(b)),
    }
}

/// (Card kinds job) What the layer records INSIDE one of the title's lists,
/// between its draws, timed by kind (`VICTORY_CARD_METER_KINDS`, on with the
/// card meter unless 0): a run of consecutive layer commands of one kind
/// (all the constant fills a draw's binding makes, say) gets one pair of
/// timestamps, opened by the first ([`HostList::inside`]) and closed where
/// the title's next command is recorded ([`HostList::inside_end`]: the
/// list's entry for every title call, and the host draw / dispatch calls).
///
/// Cost: two timestamp writes a run on the card (the [`Inside::Floor`]
/// kind measures an empty pair once per block of [`BLOCK`] runs), and one
/// resolve of a list's block when the list closes, instead of one a pair.
/// Kept apart from the list ring: a frame has thousands of runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(crate) enum Inside {
    /// Our clamp of indirect records in the list (a compute pass between
    /// two waits; the title's pipeline and roots bound again after it).
    Clamp = 0,
    /// Constant fills on the card (`HostList::card_constants`).
    Fill = 1,
    /// Title buffer copies run on the card (`crate::list` `gpu_buffer_copy`).
    Copy = 2,
    /// Anything else of ours there (copies of shader-written sources, the
    /// window check).
    Other = 3,
    /// An empty pair: what a pair costs by itself.
    Floor = 4,
}

pub(crate) const INSIDE: usize = 5;

/// Pairs a list takes at a time from the kinds ring.
pub(crate) const BLOCK: u32 = 64;
/// Pairs in the kinds ring (2^19: 8 MiB of timestamps and as much read back).
const KIND_PAIRS: u32 = 1 << 19;
const KIND_BLOCKS: u32 = KIND_PAIRS / BLOCK;

/// Whether layer work inside the title's lists is timed by kind.
pub(crate) fn kinds() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| on() && std::env::var_os("VICTORY_CARD_METER_KINDS").is_none_or(|v| v != "0"))
}

/// Since the start, by [`Inside`] kind: commands of ours recorded (counted
/// whether or not timed), runs timed, and their microseconds as read.
pub(crate) static INSIDE_COMMANDS: [std::sync::atomic::AtomicU64; INSIDE] = [const { std::sync::atomic::AtomicU64::new(0) }; INSIDE];
pub(crate) static INSIDE_RUNS: [std::sync::atomic::AtomicU64; INSIDE] = [const { std::sync::atomic::AtomicU64::new(0) }; INSIDE];
pub(crate) static INSIDE_US: [std::sync::atomic::AtomicU64; INSIDE] = [const { std::sync::atomic::AtomicU64::new(0) }; INSIDE];
/// Since the start, inside the title's lists: barriers of ours (UAV waits),
/// resource-state transitions of ours, and the title's pipeline / root
/// signature bound again because a pass of ours bound its own (each a
/// pipeline and a signature).
pub(crate) static INSIDE_BARRIERS: [std::sync::atomic::AtomicU64; INSIDE] = [const { std::sync::atomic::AtomicU64::new(0) }; INSIDE];
pub(crate) static INSIDE_TRANSITIONS: [std::sync::atomic::AtomicU64; INSIDE] = [const { std::sync::atomic::AtomicU64::new(0) }; INSIDE];
pub(crate) static INSIDE_REBINDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Since the start: preludes recorded and their microseconds as read.
pub(crate) static PRELUDES: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Counts `n` barriers (UAV waits) and `t` transitions of ours of `kind`
/// inside a title list.
pub(crate) fn inside_barriers(kind: Inside, n: u64, t: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    INSIDE_BARRIERS[kind as usize].fetch_add(n, Relaxed);
    INSIDE_TRANSITIONS[kind as usize].fetch_add(t, Relaxed);
}

struct KindMeter {
    heap: ID3D12QueryHeap,
    back: ID3D12Resource,
    mapped: usize,
    frequency: u64,
    next: u32,
    /// Blocks handed out and not read yet.
    out: u32,
    /// Blocks closed: block, the kinds of its pairs in order, when.
    pending: std::collections::VecDeque<(u32, Vec<u8>, std::time::Instant)>,
}

// SAFETY: host objects (free-threaded) and a mapping only read, behind a lock.
unsafe impl Send for KindMeter {}

static KIND_METER: Mutex<Option<Option<KindMeter>>> = Mutex::new(None);

fn make_kinds() -> Option<KindMeter> {
    let host = crate::host::host()?;
    let device = host.device();
    let mut heap: Option<ID3D12QueryHeap> = None;
    let mut back: Option<ID3D12Resource> = None;
    let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: u64::from(KIND_PAIRS) * 16,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    // SAFETY: plain object creation; the buffer is mapped once and stays so.
    unsafe {
        device
            .CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: KIND_PAIRS * 2, NodeMask: 0 },
                &mut heap,
            )
            .ok()?;
        device
            .CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back)
            .ok()?;
        let (heap, back) = (heap?, back?);
        let mut mapped = std::ptr::null_mut();
        back.Map(0, None, Some(&mut mapped)).ok()?;
        let frequency = host.timestamp_frequency();
        if mapped.is_null() || frequency == 0 {
            return None;
        }
        // Nothing written yet reads as zeros.
        std::ptr::write_bytes(mapped.cast::<u8>(), 0, KIND_PAIRS as usize * 16);
        tracing::info!(
            "layer work inside the title's lists is timed by kind: a timestamp pair a run of our commands (VICTORY_CARD_METER_KINDS=0: not)"
        );
        Some(KindMeter { heap, back, mapped: mapped as usize, frequency, next: 0, out: 0, pending: Default::default() })
    }
}

/// A list's pairs of the kinds ring ([`kinds`]): the block it writes, its
/// pairs' kinds so far, blocks it filled; the run open now.
#[derive(Default)]
pub(crate) struct ListKinds {
    block: Option<(u32, Vec<u8>)>,
    full: Vec<(u32, Vec<u8>)>,
    open: Option<(Inside, u32)>,
}

impl ListKinds {
    /// Whether a run is open.
    pub(crate) fn is_open(&self) -> bool {
        self.open.is_some()
    }

    /// A pair of `kind` for `list`: its first timestamp is written here.
    fn begin(&mut self, list: &ID3D12GraphicsCommandList, kind: Inside) -> Option<u32> {
        let mut meter = KIND_METER.lock().unwrap_or_else(|e| e.into_inner());
        let m = meter.get_or_insert_with(make_kinds).as_mut()?;
        if self.block.as_ref().is_none_or(|(_, k)| k.len() as u32 >= BLOCK) {
            if m.out + 2 >= KIND_BLOCKS {
                return None;
            }
            if let Some(full) = self.block.take() {
                self.full.push(full);
            }
            let block = m.next;
            m.next = (m.next + 1) % KIND_BLOCKS;
            m.out += 1;
            // The block's first pair is empty: what a pair costs alone.
            let floor = block * BLOCK;
            // SAFETY: an open list and our heap.
            unsafe {
                list.EndQuery(&m.heap, D3D12_QUERY_TYPE_TIMESTAMP, floor * 2);
                list.EndQuery(&m.heap, D3D12_QUERY_TYPE_TIMESTAMP, floor * 2 + 1);
            }
            self.block = Some((block, vec![Inside::Floor as u8]));
        }
        let (block, kinds) = self.block.as_mut()?;
        let pair = *block * BLOCK + kinds.len() as u32;
        kinds.push(kind as u8);
        // SAFETY: an open list and our heap.
        unsafe { list.EndQuery(&m.heap, D3D12_QUERY_TYPE_TIMESTAMP, pair * 2) };
        Some(pair)
    }

    /// Our command of `kind` is about to be recorded in `list`: a run of that
    /// kind is open from here (the open one, if it is of that kind).
    pub(crate) fn inside(&mut self, list: &ID3D12GraphicsCommandList, kind: Inside) {
        INSIDE_COMMANDS[kind as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if !kinds() {
            return;
        }
        if self.open.is_some_and(|(k, _)| k == kind) {
            return;
        }
        self.end(list);
        if let Some(pair) = self.begin(list, kind) {
            self.open = Some((kind, pair));
        }
    }

    /// The title's next command is about to be recorded: the open run ends.
    pub(crate) fn end(&mut self, list: &ID3D12GraphicsCommandList) {
        let Some((_, pair)) = self.open.take() else {
            return;
        };
        let meter = KIND_METER.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(Some(m)) = meter.as_ref() {
            // SAFETY: an open list and our heap.
            unsafe { list.EndQuery(&m.heap, D3D12_QUERY_TYPE_TIMESTAMP, pair * 2 + 1) };
        }
    }

    /// The list closes: the open run ends, and the blocks' pairs go to the
    /// read-back buffer when the card has run the list (one resolve a block).
    pub(crate) fn close(&mut self, list: &ID3D12GraphicsCommandList) {
        self.end(list);
        let mut blocks = std::mem::take(&mut self.full);
        blocks.extend(self.block.take());
        if blocks.is_empty() {
            return;
        }
        let mut meter = KIND_METER.lock().unwrap_or_else(|e| e.into_inner());
        let Some(Some(m)) = meter.as_mut() else {
            return;
        };
        let now = std::time::Instant::now();
        for (block, kinds) in blocks {
            let first = block * BLOCK;
            // SAFETY: an open list, our heap and read-back buffer, the
            // block's own bytes of it.
            unsafe {
                list.ResolveQueryData(&m.heap, D3D12_QUERY_TYPE_TIMESTAMP, first * 2, kinds.len() as u32 * 2, &m.back, u64::from(first) * 16);
            }
            m.pending.push_back((block, kinds, now));
        }
    }

    /// The list is reset without having closed (blocks not resolved): they
    /// are read as never run.
    pub(crate) fn forget(&mut self) {
        self.open = None;
        let mut blocks = std::mem::take(&mut self.full);
        blocks.extend(self.block.take());
        if blocks.is_empty() {
            return;
        }
        let mut meter = KIND_METER.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(Some(m)) = meter.as_mut() {
            let now = std::time::Instant::now();
            m.pending.extend(blocks.into_iter().map(|(b, k)| (b, k, now)));
        }
    }
}

/// The kinds ring's blocks recorded long enough ago are read (and zeroed:
/// a pair whose list never ran reads as nothing).
fn collect_kinds() {
    use std::sync::atomic::Ordering::Relaxed;
    let mut meter = KIND_METER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(m)) = meter.as_mut() else {
        return;
    };
    let now = std::time::Instant::now();
    while let Some((block, _, at)) = m.pending.front() {
        if now.duration_since(*at) < AFTER {
            break;
        }
        let block = *block;
        let (_, kinds, _) = m.pending.pop_front().expect("a block");
        m.out = m.out.saturating_sub(1);
        for (i, &kind) in kinds.iter().enumerate() {
            let pair = (block * BLOCK) as usize + i;
            // SAFETY: the pair's sixteen bytes of the mapped read-back buffer;
            // the card wrote them long ago or never.
            let (begin, end) = unsafe {
                let p = (m.mapped as *mut u64).add(pair * 2);
                let v = (p.read_volatile(), p.add(1).read_volatile());
                p.write_volatile(0);
                p.add(1).write_volatile(0);
                v
            };
            if begin == 0 || end == 0 {
                continue;
            }
            let k = usize::from(kind).min(INSIDE - 1);
            INSIDE_RUNS[k].fetch_add(1, Relaxed);
            INSIDE_US[k].fetch_add(micros(begin, end, m.frequency), Relaxed);
        }
    }
}

/// A prelude list is timed (its own pair of the list ring).
pub(crate) fn begin_prelude(list: &ID3D12GraphicsCommandList) -> Span {
    if !kinds() {
        return Span::NONE;
    }
    PRELUDES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    begin(list, Kind::Prelude)
}

/// Microseconds between two timestamps of a clock of `frequency` ticks a
/// second; 0 if the second is not after the first (a pair the card has not
/// written, or a list that never ran).
fn micros(begin: u64, end: u64, frequency: u64) -> u64 {
    if end <= begin || frequency == 0 {
        return 0;
    }
    ((end - begin) as u128 * 1_000_000 / frequency as u128).min(10_000_000) as u64
}

/// The pairs recorded long enough ago are read and added to the frame
/// log's totals. Called once a frame.
pub(crate) fn collect() {
    if !on() {
        return;
    }
    let mut meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(m)) = meter.as_mut() else {
        return;
    };
    let now = std::time::Instant::now();
    while let Some(&(pair, kind, at)) = m.pending.front() {
        if now.duration_since(at) < AFTER {
            break;
        }
        m.pending.pop_front();
        // SAFETY: the pair's sixteen bytes of the mapped read-back buffer;
        // the card wrote them long ago or never.
        let (begin, end) = unsafe {
            let p = (m.mapped as *const u64).add(pair as usize * 2);
            (p.read_volatile(), p.add(1).read_volatile())
        };
        if std::mem::replace(&mut m.seen[pair as usize], (begin, end)) == (begin, end) {
            continue;
        }
        let us = micros(begin, end, m.frequency);
        use crate::framelog::{T, add};
        match kind {
            Kind::Sync => {
                add(T::CardSyncUs, us);
                add(T::CardSyncLists, 1);
            }
            Kind::Check => {
                add(T::CardCheckUs, us);
                add(T::CardCheckLists, 1);
            }
            Kind::Lists => {
                add(T::CardListsUs, us);
                add(T::CardLists, 1);
                if let Some(name) = m.names.remove(&pair) {
                    let e = m.by_name.entry(name).or_insert((0, 0));
                    e.0 += 1;
                    e.1 += us;
                }
            }
            Kind::Pipeline => {
                if let Some(name) = m.names.remove(&pair) {
                    let e = m.by_pipeline.entry(name).or_insert((0, 0));
                    e.0 += 1;
                    e.1 += us;
                }
            }
            Kind::Prelude => {
                PRELUDES[1].fetch_add(us, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    drop(meter);
    collect_kinds();
    let mut meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(m)) = meter.as_mut() else {
        return;
    };
    m.frames += 1;
    if m.frames >= 256 {
        let frames = m.frames as f64;
        let mut table: Vec<(String, (u64, u64))> = m.by_name.drain().collect();
        table.sort_by_key(|&(_, (_, us))| std::cmp::Reverse(us));
        let all: u64 = table.iter().map(|&(_, (_, us))| us).sum();
        tracing::info!(
            "card time of the title's lists by what they bind, per frame over {} frames: {:.1} ms in {:.0} lists of {} kinds; the largest follow",
            m.frames,
            all as f64 / 1e3 / frames,
            table.iter().map(|&(_, (n, _))| n).sum::<u64>() as f64 / frames,
            table.len()
        );
        for (name, (n, us)) in table.into_iter().take(24) {
            tracing::info!("  card {:6.2} ms a frame in {:5.1} lists: {name}", us as f64 / 1e3 / frames, n as f64 / frames);
        }
        if !m.by_pipeline.is_empty() {
            let mut table: Vec<(String, (u64, u64))> = m.by_pipeline.drain().collect();
            table.sort_by_key(|&(_, (_, us))| std::cmp::Reverse(us));
            let all: u64 = table.iter().map(|&(_, (_, us))| us).sum();
            tracing::info!(
                "card time by pipeline (from where it is bound to where the next is), per frame over {} frames: {:.1} ms in {:.0} bindings of {} pipelines; the largest follow",
                m.frames,
                all as f64 / 1e3 / frames,
                table.iter().map(|&(_, (n, _))| n).sum::<u64>() as f64 / frames,
                table.len()
            );
            for (name, (n, us)) in table.into_iter().take(48) {
                tracing::info!(
                    "  pipeline {:6.2} ms a frame in {:6.1} bindings ({:7.1} us each): {name}",
                    us as f64 / 1e3 / frames,
                    n as f64 / frames,
                    us as f64 / n.max(1) as f64
                );
            }
        }
        m.frames = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_list_is_named_by_the_pipelines_it_binds() {
        let mut names = std::collections::BTreeSet::new();
        assert_eq!(name_of(&names), "no pipeline (copies, clears, barriers)");
        names.insert("compute CS_Blur".to_string());
        assert_eq!(name_of(&names), "1 pipeline: compute CS_Blur");
        names.insert("mesh VS_A, pixel PS_B".to_string());
        names.insert("a".repeat(100));
        assert_eq!(name_of(&names), format!("3 pipelines: {} / compute CS_Blur", "a".repeat(60)));
    }

    #[test]
    fn a_pairs_length_is_in_microseconds_of_the_queues_clock() {
        assert_eq!(micros(1_000, 11_000, 10_000_000), 1_000);
        assert_eq!(micros(0, 25_000_000, 25_000_000), 1_000_000);
        // Not written, or the wrong way round: nothing.
        assert_eq!(micros(5, 5, 1_000), 0);
        assert_eq!(micros(9, 5, 1_000), 0);
        assert_eq!(micros(1, 2, 0), 0);
        // A stale second stamp from the ring's last round cannot add hours.
        assert_eq!(micros(1, u64::MAX, 1), 10_000_000);
    }
}
