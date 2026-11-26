//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    #[cfg(feature = "diagnostics")]
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    /// First mip to read from `layout` for host mip zero.
    first_mip: u32,
    /// Full title extent for the indexed upload budget and residency checks.
    title_bytes: usize,
    /// The host subresource of the staging buffer's first part: the first
    /// mip a view shows, when only those from it on are uploaded
    /// ([`first_mip_switch`]); else 0.
    view_first: u32,
    /// The staging buffer's bytes (counted in `STAGED` while it lives).
    bytes: u64,
    memory: *const u8,
    last: Mutex<Last>,
    /// Whether the staging buffer was detiled since a submitted list last
    /// uploaded it, and whether one has uploaded it at all: a list records
    /// the upload only then. (Every list that sampled a title texture
    /// uploaded all of it again, changed or not: 5,411 subresource copies
    /// of 1,120 textures in one frame of run 18.)
    dirty: std::sync::atomic::AtomicBool,
    uploaded: std::sync::atomic::AtomicBool,
    /// When a list last recorded a use of it (`crate::host::moment`).
    used: std::sync::atomic::AtomicU64,
    /// Levels on demand (`crate::tiles`): the texture's host copy was made
    /// without memory of its own when the staging buffer was made.
    on_demand: bool,
    /// For such a texture: the levels that were the title's at the last
    /// detiling (a bit a level; the staging buffer holds zeros for the
    /// others), and what that was judged by ([`Staging::judgement`]).
    detiled: std::sync::atomic::AtomicU32,
    judged: [std::sync::atomic::AtomicU64; 2],
    /// For such a texture: when an upload last left a held level without
    /// tiles (`crate::host::moment`, never 0; 0: none did). The upload is
    /// recorded again about a second later if the card has room by then
    /// ([`upload`]).
    waiting: std::sync::atomic::AtomicU64,
    /// The size of the buffer if it came from the pool of spare staging
    /// buffers and goes back to it ([`staging_buffer`]); 0: a buffer of its
    /// own, released when this is dropped.
    pooled: u64,
    /// Its range of a block of upload memory ([`upload_memory`]); the
    /// range is given back when the submissions started by the time this
    /// is dropped have run. `None`: `buffer` is the staging's own.
    piece: Option<Piece>,
}

/// Whether a texture the title placed is staged, read and uploaded from the
/// first mip its view shows (`VICTORY_TEXTURE_FIRST_MIP=0`: from mip 0,
/// all of its description, as before). On by default (myokex c3723f3).
///
/// A title that streams textures describes every mip of them and has
/// memory for the ones it streamed in, the smaller ones, which lie at the
/// texture's start; right behind them it places the next texture. Read for
/// its whole description such a texture lay over its neighbours (in
/// myokex's runs 109,638 views were taken for stale in 8 minutes and showed
/// nothing from then on, and 700 KB were read for the 12 KB there). Only
/// a single 2D slice, not with levels on demand, nor for a host copy made
/// without its largest levels (which has its own prefix,
/// [`staging_layout`]).
pub(crate) fn first_mip_switch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_FIRST_MIP").is_none_or(|v| v != "0"))
}

/// A staging buffer nothing holds: the buffer, where it is mapped, and
/// when it was given back (`crate::host::moment`), for [`trim_pools`].
struct Spare(ID3D12Resource, usize, u64);

// SAFETY: a D3D12 buffer and its mapping, touched by nobody while it is
// spare.
unsafe impl Send for Spare {}

/// Spare staging buffers by size (a power of two), and their bytes in all.
static SPARE: Mutex<(std::collections::BTreeMap<u64, Vec<Spare>>, u64)> =
    Mutex::new((std::collections::BTreeMap::new(), 0));
/// The largest buffer that is pooled (the second); the first is no longer
/// read: [`spare_most`].
const SPARE_MOST: (u64, u64) = (256 << 20, 8 << 20);

/// (neptune-ve 39f5da3) The most bytes of staging buffers kept spare
/// (`VICTORY_STAGE_POOL_MB`; 256 before). Default 2,048: the buffers are
/// in the upload heap (`Host::upload_buffer`, `D3D12_HEAP_TYPE_UPLOAD`),
/// which is processor RAM on this card, not VRAM; spares idle past the
/// trim go back as before (`trim_pools`). The author saw 29,000 buffers
/// made and freed (14 GB) in the first 80 s of the world fall to 258.
fn spare_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        let mb = std::env::var("VICTORY_STAGE_POOL_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(2048);
        tracing::info!(mb, "staging buffers kept spare at most (VICTORY_STAGE_POOL_MB; upload heap = processor RAM; 256 before)");
        mb << 20
    })
}
/// Staging buffers taken from the spare ones, and made, since the start.
pub(crate) static SPARE_STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// A mapped upload buffer of `total` bytes for a texture's staging, all
/// zeros, and its pool size (0: not pooled).
///
/// Every texture the title streams in got a buffer of its own from the
/// host and gave it back when its staging went: about a quarter of a
/// millisecond to make (run 213's profile: 25 ms a frame on the recording
/// threads for some 100 uploads, most of it the graphics kernel waiting)
/// and more to free on the thread that submits. A streaming burst is 200
/// to 500 uploads in a frame (run 222), and those are the frames that
/// dip. Buffers of up to 8 MiB are now kept by size and used again,
/// zeroed as a new one is (levels not there yet must read as zeros).
/// `VICTORY_STAGING_POOL=0`: made and freed each time, as before.
fn staging_buffer(host: &crate::host::Host, total: u64) -> Option<(ID3D12Resource, *mut u8, u64)> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let size = total.max(64 << 10).next_power_of_two();
    if size > SPARE_MOST.1 || !*ON.get_or_init(|| std::env::var_os("VICTORY_STAGING_POOL").is_none_or(|v| v != "0")) {
        return host.upload_buffer(total).map(|(buffer, data)| (buffer, data, 0));
    }
    let spare = {
        let mut all = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        let taken = all.0.get_mut(&size).and_then(Vec::pop);
        if taken.is_some() {
            all.1 -= size;
        }
        taken
    };
    if let Some(Spare(buffer, data, _)) = spare {
        SPARE_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: the buffer's own mapping, `size` bytes, which nothing
        // else holds.
        unsafe { std::ptr::write_bytes(data as *mut u8, 0, total as usize) };
        return Some((buffer, data as *mut u8, size));
    }
    SPARE_STATS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    host.upload_buffer(size).map(|(buffer, data)| (buffer, data, size))
}

/// Whether staging memory is cut from blocks of upload memory
/// ([`upload_memory`]; on by default, `VICTORY_STAGING_BLOCKS=0`: a buffer
/// per texture from [`staging_buffer`], as before).
fn blocks_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_STAGING_BLOCKS").is_none_or(|v| v != "0"))
}

/// The size of a block of upload memory staging buffers are cut from, and
/// the unit they are cut on (a page; copies want 512).
const BLOCK: u64 = 64 << 20;
const GRAIN: u64 = 4096;

/// The most bytes of blocks alive at once (`VICTORY_STAGING_BLOCKS_MB`,
/// 1,024: sixteen blocks; 64 to 8,192). Past it a staging buffer is made
/// on its own ([`staging_buffer`]), as without blocks. Run dev-3-rawoff
/// had 50 to 1,400 staging buffers alive, 36 to 433 MB (1,192 MB once,
/// at the world's first frame), so the cap is rarely reached.
fn blocks_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_STAGING_BLOCKS_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(1024, |mb| mb.clamp(64, 8192))
            << 20
    })
}

/// A block of upload memory, mapped for its life, what is free in it, and
/// since when it holds nothing (`crate::host::moment`; 0: it holds some).
struct Block {
    buffer: ID3D12Resource,
    data: usize,
    ranges: crate::ranges::Ranges,
    empty_since: u64,
}
// SAFETY: a free-threaded D3D12 object and the address of its mapping.
unsafe impl Send for Block {}

/// The blocks by index (a released one leaves `None`, so that the pieces
/// of the others keep their index).
static BLOCKS: Mutex<Vec<Option<Block>>> = Mutex::new(Vec::new());
/// Since the start: staging buffers cut from blocks and their bytes,
/// blocks made, blocks released as idle.
pub(crate) static POOLED: [std::sync::atomic::AtomicU64; 4] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 4];
/// Bytes of the blocks' ranges cut now (inside the blocks' bytes).
static IN_PIECES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A staging buffer's range of a block; dropping it gives the range back.
struct Piece {
    block: usize,
    offset: u64,
    len: u64,
}

impl Drop for Piece {
    fn drop(&mut self) {
        let mut blocks = BLOCKS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(Some(block)) = blocks.get_mut(self.block) {
            block.ranges.give(self.offset, self.len);
            IN_PIECES.fetch_sub(self.len, std::sync::atomic::Ordering::Relaxed);
            if block.ranges.whole() {
                block.empty_since = crate::host::moment().max(1);
            }
        }
    }
}

/// Upload memory for a staging buffer of `total` bytes, all zeros: the
/// buffer and its mapping, the offset of the staging buffer's first byte in
/// both, its pool size ([`staging_buffer`]; 0: not pooled) and its range of
/// a block if it was cut from one.
///
/// A committed upload buffer per texture was most of what the pass over
/// textures picked by index did in rorodev6's opening cutscene (1,000 to
/// 2,200 a frame at about 0.5 ms each, run `dl_diag1`, its commit 0781f61),
/// and here the layer made 160,000 upload buffers (120 GB) in six minutes
/// of run dev-3-rawoff while the process's write-combined memory stood at
/// 9 to 15 GB, though what the layer held of it alive was under 1 GB.
/// Staging buffers of up to a quarter block are now ranges of 64 MB blocks
/// used again and again.
fn upload_memory(
    host: &crate::host::Host,
    total: u64,
) -> Option<(ID3D12Resource, *mut u8, u64, u64, Option<Piece>)> {
    use std::sync::atomic::Ordering::Relaxed;
    let len = total.max(1).next_multiple_of(GRAIN);
    let own = || {
        staging_buffer(host, total).map(|(buffer, data, pooled)| (buffer, data, 0, pooled, None))
    };
    if !blocks_on() || len > BLOCK / 4 {
        return own();
    }
    let mut blocks = BLOCKS.lock().unwrap_or_else(|e| e.into_inner());
    // The newest block first: the older ones are full but for holes.
    let found = blocks
        .iter_mut()
        .enumerate()
        .rev()
        .filter_map(|(i, b)| b.as_mut().map(|b| (i, b)))
        .find_map(|(i, b)| b.ranges.take(len, GRAIN).map(|offset| (i, offset)));
    let (index, offset, fresh) = match found {
        Some((i, offset)) => (i, offset, false),
        None => {
            let alive = blocks.iter().flatten().count() as u64;
            if (alive + 1) * BLOCK > blocks_most() {
                drop(blocks);
                return own();
            }
            let (buffer, data) = host.upload_buffer(BLOCK)?;
            let index = blocks
                .iter()
                .position(Option::is_none)
                .unwrap_or(blocks.len());
            crate::host::name(&buffer, || format!("staging block {index}"));
            let mut ranges = crate::ranges::Ranges::new(BLOCK);
            let offset = ranges.take(len, GRAIN)?;
            let block = Block {
                buffer,
                data: data as usize,
                ranges,
                empty_since: 0,
            };
            if index == blocks.len() {
                blocks.push(Some(block));
            } else {
                blocks[index] = Some(block);
            }
            POOLED[2].fetch_add(1, Relaxed);
            (index, offset, true)
        }
    };
    let b = blocks[index].as_mut()?;
    b.empty_since = 0;
    // A range used before holds what its last staging buffer did: zeros,
    // as a new buffer's are (levels not there yet must read as zeros).
    if !fresh {
        // SAFETY: the block's own mapping; the range is this staging
        // buffer's alone from now on, and no list copies from it any more
        // (a range comes back only after the submissions that might have).
        unsafe {
            std::ptr::write_bytes((b.data as *mut u8).add(offset as usize), 0, total as usize)
        };
    }
    POOLED[0].fetch_add(1, Relaxed);
    POOLED[1].fetch_add(len, Relaxed);
    IN_PIECES.fetch_add(len, Relaxed);
    Some((
        b.buffer.clone(),
        b.data as *mut u8,
        offset,
        0,
        Some(Piece {
            block: index,
            offset,
            len,
        }),
    ))
}

/// For the memory line: what the staging pools keep. Spare staging buffers
/// and their bytes; blocks alive and their bytes free; staging buffers cut
/// from blocks since the start; blocks made and released since the start.
#[cfg(feature = "diagnostics")]
pub(crate) fn pools() -> [u64; 7] {
    use std::sync::atomic::Ordering::Relaxed;
    let (spare, spare_bytes) = {
        let all = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        (all.0.values().map(|v| v.len() as u64).sum::<u64>(), all.1)
    };
    let (alive, free) = {
        let blocks = BLOCKS.lock().unwrap_or_else(|e| e.into_inner());
        blocks
            .iter()
            .flatten()
            .fold((0u64, 0u64), |(n, f), b| (n + 1, f + b.ranges.room()))
    };
    [
        spare,
        spare_bytes,
        alive,
        free,
        POOLED[0].load(Relaxed),
        POOLED[2].load(Relaxed),
        POOLED[3].load(Relaxed),
    ]
}

/// For the memory map line, the upload memory of textures' staging the
/// layer holds now: staging buffers of their own (not cut from blocks,
/// also those only a list still holds), spare ones, and whole blocks.
#[cfg(feature = "diagnostics")]
pub(crate) fn upload_bytes() -> [u64; 3] {
    use std::sync::atomic::Ordering::Relaxed;
    let spare = SPARE.lock().unwrap_or_else(|e| e.into_inner()).1;
    let blocks = BLOCKS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .flatten()
        .count() as u64;
    // (Rounded to pages in the blocks: never more than all staging.)
    let own = STAGED[1]
        .load(Relaxed)
        .saturating_sub(IN_PIECES.load(Relaxed));
    [own, spare, blocks * BLOCK]
}

/// Lets go of spare staging buffers given back `idle_ms` ago and more, and
/// of blocks that held nothing for as long. Both pools are capped
/// ([`SPARE_MOST`], [`blocks_most`]) but were kept whole for the run once
/// filled: a streaming burst's worth stayed for good.
fn trim_pools(idle_ms: u64, idle_frames: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if !crate::host::pool_trim() {
        return;
    }
    let mut went: Vec<ID3D12Resource> = Vec::new();
    {
        let mut all = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        let (map, bytes) = &mut *all;
        for (&size, spares) in map.iter_mut() {
            spares.retain(|Spare(buffer, _, given)| {
                let keep = !crate::host::long_ago(*given, idle_ms, idle_frames);
                if !keep {
                    *bytes -= size;
                    went.push(buffer.clone());
                }
                keep
            });
        }
        map.retain(|_, spares| !spares.is_empty());
    }
    let mut blocks_went = Vec::new();
    {
        let mut blocks = BLOCKS.lock().unwrap_or_else(|e| e.into_inner());
        for slot in blocks.iter_mut() {
            let idle = slot.as_ref().is_some_and(|b| {
                b.ranges.whole()
                    && b.empty_since != 0
                    && crate::host::long_ago(b.empty_since, idle_ms, idle_frames)
            });
            if idle && let Some(block) = slot.take() {
                POOLED[3].fetch_add(1, Relaxed);
                blocks_went.push(block.buffer);
            }
        }
        while blocks.last().is_some_and(Option::is_none) {
            blocks.pop();
        }
    }
    if !went.is_empty() || !blocks_went.is_empty() {
        let (n, m) = (went.len(), blocks_went.len());
        crate::host::trail(|| {
            format!("sweep: {n} spare staging buffers and {m} empty staging blocks let go")
        });
    }
    // Nothing in the air names them (spare buffers came back when their
    // staging was dropped, blocks once all their ranges had), but the
    // retire discipline costs nothing here.
    for buffer in went.into_iter().chain(blocks_went) {
        crate::host::retire(buffer);
    }
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    asked: u64,
    /// The frame a partly mapped texture was last read in.
    partial: Option<u64>,
    /// The texture was found changed without a write the guard saw
    /// ([`Staging::unseen_write`]): it is hashed at every check until this
    /// frame.
    unguarded: u64,
}

/// Whether a guarded texture found changed with no write the guard saw is
/// hashed at every check for a while, and a linear one-level picture at
/// every check (`VICTORY_TEXTURE_UNSEEN=0`: every eighth frame, as before).
/// On by default (myokex c3723f3: the intro's movie, three pictures in turn
/// that only changed every eighth frame, one old picture after another).
fn unseen_switch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_UNSEEN").is_none_or(|v| v != "0"))
}

/// Whether, with `VICTORY_TEXTURE_NO_WAIT`, a texture nothing was read for
/// yet whose pages are not all there when the runtime remaps is read
/// around the missing ones (zeros, as a partly mapped texture is)
/// (`VICTORY_TEXTURE_READ_AROUND=0`: left for its next refresh, as
/// before). On by default (myokex c3723f3): its upload is recorded behind
/// the refresh, and copied a staging buffer of zeros (black) over what the
/// host texture showed, for good until more memory was mapped.
fn read_around_switch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_READ_AROUND").is_none_or(|v| v != "0"))
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// The buffer a texture the host writes is copied through into the texture
/// of a view the title placed in its memory in another format
/// ([`alias`]): every level of every layer as each of the two formats lays
/// it out for a copy (the same bytes: the texels are as wide), the written
/// texture's layout first.
pub(crate) struct Through {
    buffer: ID3D12Resource,
    parts: Vec<(
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    )>,
}

/// An alias copy left to the list's prelude ([`alias_once`]): the copy runs
/// at submission, before the list, unless no list submitted since the last
/// such copy of the same pair wrote the source.
pub(crate) struct AliasCopy {
    /// The source's host copy (its address names it in the write marks).
    pub(crate) source: Arc<crate::resource::HostTexture>,
    pub(crate) to: ID3D12Resource,
    pub(crate) through: Arc<Through>,
}

/// Whether an alias copy a list needs before it touches the source or the
/// placed texture is made by its prelude at submission, and left out where
/// no list submitted since the same copy last ran wrote the source
/// (`VICTORY_ALIAS_ONCE=1`; off by default).
///
/// The title writes R9G9B9E5 targets as R32_UINT and reads them through
/// views placed in the same memory; each list that reads one copied it
/// again: 78 copies a frame, 26-31 ms of the card's time on an RTX 4050
/// Laptop, for a few textures written once a frame. Preludes are recorded
/// at submission, in the queue's order, so whether a list submitted since
/// wrote the source is known there. What it assumes: only these copies
/// write the placed texture's host copy (a view's reads are never uploaded
/// from title memory while it is aliased, see `crate::draw::readable`).
pub(crate) fn alias_once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ALIAS_ONCE").is_some_and(|v| v == "1"))
}

/// The copy of an [`AliasCopy`], recorded with the source in the copy
/// source state and `to` in the copy destination state.
pub(crate) fn record_alias(list: &ID3D12GraphicsCommandList, from: &ID3D12Resource, to: &ID3D12Resource, through: &Through) {
    let part = |texture: &ID3D12Resource, index: usize| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index as u32,
        },
    };
    let rows = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(&through.buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, read, written) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: an open list and live resources; the buffer is as big as
    // the footprints and rests in the common state between copies.
    unsafe {
        list.ResourceBarrier(&[crate::host::transition(&through.buffer, common, written)]);
        for (index, (of_written, _)) in through.parts.iter().enumerate() {
            list.CopyTextureRegion(&rows(of_written), 0, 0, 0, &part(from, index), None);
        }
        list.ResourceBarrier(&[crate::host::transition(&through.buffer, written, read)]);
        for (index, (_, of_read)) in through.parts.iter().enumerate() {
            list.CopyTextureRegion(&part(to, index), 0, 0, 0, &rows(of_read), None);
        }
        list.ResourceBarrier(&[crate::host::transition(&through.buffer, read, common)]);
    }
}

/// The most levels times layers [`alias`] copies (a cube of nine levels).
const ALIAS_PARTS: u32 = 54;

// SAFETY: a device-local buffer only lists record copies with.
unsafe impl Send for Through {}
// SAFETY: as above.
unsafe impl Sync for Through {}

/// (fxceless 19a45d1) Since the start: copies [`alias`] made into a view the list had
/// used before (an upload from title memory), which it skipped before.
pub(crate) static ALIAS_AFTER_UPLOAD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The copy buffers of [`alias`], by the placed view's texture.
static THROUGH: Mutex<Option<HashMap<usize, Arc<Through>>>> = Mutex::new(None);

/// Brings `placed`, the texture of a view the title placed in the memory
/// of `source` in another format (`crate::resource::alias_of`), up to date
/// with what the host wrote to `source`: the texels of every level and
/// layer are copied as they are, through a buffer (the host copies between
/// two formats of one texel width in no other way). Recorded the first time
/// the list reads the view
/// and again whenever the list used `source` in another way since (wrote
/// it, or read it itself). Leaves `placed` in the copy destination state,
/// as [`upload`] does. False if the copy cannot be made: the caller fills
/// the texture from title memory, as before.
pub(crate) fn alias(list: &HostList, placed: This, source: This) -> bool {
    const TARGET_OR_UNORDERED: u32 = 0x4 | 0x8;
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    // Host copies with other levels than each other's: filled from title
    // memory instead (`crate::resource::drop_levels`).
    if crate::resource::dropped(placed) != crate::resource::dropped(source) {
        return false;
    }
    let (Some(to), Some(from)) = (
        crate::resource::host_resource(placed),
        crate::resource::host_resource(source),
    ) else {
        return false;
    };
    let Some(host) = host() else { return false };
    let through = {
        let mut map = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
        let map = map.get_or_insert_with(HashMap::new);
        match map.get(&(placed as usize)) {
            Some(through) => through.clone(),
            None => {
                // SAFETY: live textures.
                let (of_from, of_to) = unsafe { (from.GetDesc(), to.GetDesc()) };
                // Levels times layers (a volume's slices are one part).
                let count = |d: &D3D12_RESOURCE_DESC| {
                    let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                        1
                    } else {
                        u32::from(d.DepthOrArraySize)
                    };
                    u32::from(d.MipLevels) * layers
                };
                let parts = count(&of_from);
                if parts == 0 || parts > ALIAS_PARTS || parts != count(&of_to) {
                    return false;
                }
                let mut rows = [(); 2]
                    .map(|()| vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); parts as usize]);
                let mut bytes = [0u64; 2];
                // SAFETY: plain queries; the out pointers are ours and
                // hold `parts` layouts each.
                unsafe {
                    host.device.GetCopyableFootprints(
                        &of_from,
                        0,
                        parts,
                        0,
                        Some(rows[0].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[0]),
                    );
                    host.device.GetCopyableFootprints(
                        &of_to,
                        0,
                        parts,
                        0,
                        Some(rows[1].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[1]),
                    );
                }
                let alike = rows[0].iter().zip(&rows[1]).all(|(a, b)| {
                    let (fa, fb) = (a.Footprint, b.Footprint);
                    a.Offset == b.Offset
                        && (fa.Width, fa.Height, fa.Depth, fa.RowPitch)
                            == (fb.Width, fb.Height, fb.Depth, fb.RowPitch)
                });
                if bytes[0] == 0 || bytes[0] != bytes[1] || !alike {
                    return false;
                }
                let Some(buffer) = host.uav_buffer(bytes[0]) else {
                    return false;
                };
                crate::host::name(&buffer, || {
                    format!(
                        "alias copy {}x{} f{} to f{}",
                        of_from.Width, of_from.Height, of_from.Format.0, of_to.Format.0
                    )
                });
                tracing::info!(
                    view = ?placed,
                    texture = ?source,
                    width = of_from.Width,
                    height = of_from.Height,
                    parts,
                    written_as = of_from.Format.0,
                    read_as = of_to.Format.0,
                    "a view placed in the memory of a texture the host writes, in another format: read from that texture by a copy on the host"
                );
                let [of_written, of_read] = rows;
                let through = Arc::new(Through {
                    buffer,
                    parts: of_written.into_iter().zip(of_read).collect(),
                });
                map.insert(placed as usize, through.clone());
                through
            }
        }
    };
    let first = list.first_use(placed as usize);
    // (fxceless 19a45d1) An upload of this view from title memory earlier in the list has
    // used its first use already: only a copy from the same card-written source satisfies
    // this read (before: the copy was skipped and the shader read the uploaded bytes, zero
    // where the card wrote packed light). Counted in [`ALIAS_AFTER_UPLOAD`].
    let new_source = list.alias_source_changed(to, from);
    if new_source && !first {
        ALIAS_AFTER_UPLOAD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    // Before the list touches either texture: the copy goes to the
    // prelude, which leaves it out if the source was not written since.
    if alias_once()
        && first
        && !list.touches(source)
        && !list.touches(placed)
        && let Some(copy) = crate::resource::host_copy(source)
    {
        let _ = list.transition(source, COPY_SOURCE);
        card_writes(list, placed, to, None);
        crate::list::ensure_state(list, placed, COPY_DEST);
        list.keep(&through.buffer);
        list.alias_at_submission(AliasCopy {
            source: copy,
            to: to.clone(),
            through: through.clone(),
        });
        return true;
    }
    // The source's first use in the list records nothing (its prelude
    // brings it to the state); a use in another way since the last copy
    // shows as a change of state.
    let before = list.transition(source, COPY_SOURCE).map(|(texture, before)| {
        let state = crate::host::legal_state(texture, COPY_SOURCE);
        let barrier = crate::host::transition(
            texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(state as i32),
        );
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
        before
    });
    if !new_source && before.is_none() {
        return true;
    }
    card_writes(list, placed, to, None);
    crate::list::ensure_state(list, placed, COPY_DEST);
    let part = |texture: &ID3D12Resource, index: usize| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index as u32,
        },
    };
    let rows = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(&through.buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, read, written) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("alias copy of {source:?} into {placed:?}"));
    // SAFETY: an open list and live resources; the buffer is as big as
    // the footprints and rests in the common state between copies.
    unsafe {
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, common, written)]);
        for (index, (of_written, _)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&rows(of_written), 0, 0, 0, &part(from, index), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, written, read)]);
        for (index, (_, of_read)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&part(to, index), 0, 0, 0, &rows(of_read), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, read, common)]);
    }
    list.keep(&through.buffer);
    // A draw that also writes the source (a view of it handled before
    // this one) needs it in that state again; the next read of the alias
    // then copies once more, as it must after a write.
    if let Some(before) = before
        && before & TARGET_OR_UNORDERED != 0
    {
        crate::list::ensure_state(list, source, before);
    }
    true
}

/// Whether a texture refresh that finds the runtime remapping title memory
/// goes on without waiting for it (`VICTORY_TEXTURE_NO_WAIT`; on by
/// default, `=0` waits as before).
///
/// On by default since an RTX 4070 run: 327 one-second frame lines with
/// refreshes waiting for the lock (nearly all while the world loads), none
/// with the switch.
///
/// A texture whose memory is all mapped is read in place, under a lock
/// the runtime takes alone while it changes a mapping
/// (`crate::watch::title_memory_reading`), so that no page goes away under
/// the read. While the world streams in the runtime remaps all the time,
/// and each of the frame's 150-360 refreshes waited for it: runs 99-101
/// have 4, 15 and 16 frame lines with over 300 ms of "textures", up to
/// 3,708 ms, and on every one of them the wait for this lock is the whole
/// of it (1,345 of 1,381 ms; 3,630 of 3,708; the write-watch askings 3-12
/// ms, the texture's own lock 0).
///
/// With the switch the refresh asks for the lock without waiting. If the
/// runtime has it, the texture is read through the kernel instead, which
/// needs no lock: it fails if a page is not there, and then the texture is
/// left as it is and read at its next refresh, a frame later. A read
/// through the kernel costs about 0.6 ms a megabyte against 0.3 in place.
///
/// What changes for the picture: a texture that arrives while the runtime
/// is remapping may show a frame later. It never shows partly read.
fn no_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_NO_WAIT").is_none_or(|v| v != "0"))
}

/// Since the last frame-rate line, with [`no_wait`]: refreshes that read
/// through the kernel because title memory was being remapped, and those
/// left for the next refresh because a page was not there.
/// Then those of the latter that had nothing uploaded yet and were read
/// around the pages that were not there ([`read_around_switch`]).
pub(crate) static UNLOCKED: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Since the start: textures found changed with no write the guard saw
/// ([`unseen_switch`]).
pub(crate) static UNSEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Since the last frame-rate line, of texture refreshes on every thread, in
/// microseconds: waiting for the texture's own lock (another thread is
/// refreshing the same texture), asking the write watch and the guards,
/// waiting for title memory to hold still (the lock the runtime takes
/// alone while it remaps, `crate::watch::title_memory_reading`), and the
/// refreshes in all.
///
/// While the world streams in, the frame line's "textures" is seconds a
/// frame in some runs and tenths in others for the same megabytes (2,566
/// ms for 110 MB in run 93, 314 ms for 132 MB in run 91; before these
/// counters, runs 81, 83, 85 and 94 each had lines of 1.5-2.9 s), and the
/// submitting thread is not running while it lasts (run 93: 240-310 ms of
/// its own time a frame beside 1,100-1,800 ms of "textures"): it waits.
/// These say for what.
pub(crate) static REFRESH_WAITS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// `f`'s result, its time added to [`REFRESH_WAITS`]' place `slot`.
fn waited<T>(slot: usize, f: impl FnOnce() -> T) -> T {
    let started = std::time::Instant::now();
    let out = f();
    let us = started.elapsed().as_micros() as u64;
    REFRESH_WAITS[slot].fetch_add(us, std::sync::atomic::Ordering::Relaxed);
    if slot == 2 {
        crate::framelog::add(crate::framelog::T::RefreshStillUs, us);
    }
    out
}

thread_local! {
    /// Room to read a texture's memory into: one buffer per submitting
    /// thread. Each texture kept its own, a second copy in memory of every
    /// write-watched or partly mapped texture ever sampled, for the run.
    static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Calls `f` with `size` bytes of this thread's read buffer (their
/// contents are whatever the last texture left).
fn with_read<R>(size: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    // (Copy-back consumers) A texture's host copy reads title memory.
    let _kind = crate::readback::consumers::scope(crate::readback::consumers::Kind::Texture);
    READ.with(|read| {
        let mut read = read.borrow_mut();
        if read.len() < size {
            read.resize(size, 0);
        }
        f(&mut read[..size])
    })
}

/// Whether a texture all of whose memory is mapped is read through the
/// kernel's copy instead of in place (`VICTORY_TEXTURE_COPY=1`; off by
/// default). Round 3 of the memory work read every such texture that way
/// to be safe from pages taken away under the read; it cost 0.6 ms per MB
/// against 0.3 in place (90-105 ms a frame while the world streams in, runs
/// 72 and 73, against 20-40), and both runs with it lost the graphics
/// device, for a reason not found. The read in place is safe now for
/// another reason (`crate::watch::title_memory_reading`); this switch is
/// left to tell whether the copy was what lost the device.
fn through_copy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_COPY").is_some_and(|v| v == "1"))
}

/// Calls `f` with the `size` bytes of title memory at `memory`, read
/// through the kernel into this thread's read buffer; pages that are not
/// there read as zeros. The bytes are those a read in place would see.
fn with_title<R>(memory: *const u8, size: usize, f: impl FnOnce(&[u8]) -> R) -> R {
    with_read(size, |read| {
        crate::host::copy_mapped(read.as_mut_ptr(), memory, size, false);
        f(read)
    })
}

/// Staging buffers alive (also those only a list still holds) and their
/// bytes.
static STAGED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Staging {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        STAGED[0].fetch_sub(1, Relaxed);
        STAGED[1].fetch_sub(self.bytes, Relaxed);
        // Nothing holds the staging any more (lists keep it until they are
        // reset, after their submission has run): its buffer is spare.
        if self.pooled != 0 {
            let mut all = SPARE.lock().unwrap_or_else(|e| e.into_inner());
            if all.1 + self.pooled <= spare_most() {
                all.1 += self.pooled;
                all.0.entry(self.pooled).or_default().push(Spare(
                    self.buffer.clone(),
                    self.data as usize,
                    crate::host::moment(),
                ));
            }
        }
        // A list the card has not run yet may still copy from the range
        // (lists hold the staging until they are reset, so none should).
        if let Some(piece) = self.piece.take() {
            crate::host::retire(piece);
        }
    }
}

/// For the memory line: staging buffers alive, their bytes, and how many
/// of them their texture still has (the others wait for a list's reset).
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 3] {
    use std::sync::atomic::Ordering::Relaxed;
    let kept = STAGING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or(0, |map| map.len() as u64);
    [STAGED[0].load(Relaxed), STAGED[1].load(Relaxed), kept]
}

/// Staging buffers let go as idle since the start, and their bytes.
static LET_GO: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// For the memory line: staging buffers let go as idle, and their bytes.
#[cfg(feature = "diagnostics")]
pub(crate) fn let_go() -> [u64; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    [LET_GO[0].load(Relaxed), LET_GO[1].load(Relaxed)]
}

/// Lets go of the staging buffers no list recorded a use of for `IDLE_MS`
/// and `IDLE_FRAMES` frames and no list holds: a staging buffer is system memory the size of its
/// texture, kept so that an unchanged texture is not uploaded again, and
/// those of textures no longer sampled were kept for the run (2,246 of
/// them, 2.1 GB, at the end of run 24). The next list that samples such a
/// texture makes a staging buffer anew, detiles and uploads once more.
/// Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    use std::sync::atomic::Ordering::Relaxed;
    const IDLE_FRAMES: u64 = 4;
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let Some(map) = map.as_mut() else {
        return;
    };
    if map.is_empty() {
        return;
    }
    let idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
    trim_pools(idle_ms, IDLE_FRAMES);
    // A list that recorded an upload or a check holds the staging until it
    // is reset: one only the table holds is named by no list, recorded or
    // running.
    let (mut went, mut bytes) = (0u64, 0u64);
    map.retain(|_, s| {
        let keep = Arc::strong_count(s) > 1
            || !crate::host::long_ago(s.used.load(Relaxed), idle_ms, IDLE_FRAMES);
        if !keep {
            LET_GO[0].fetch_add(1, Relaxed);
            LET_GO[1].fetch_add(s.bytes, Relaxed);
            went += 1;
            bytes += s.bytes;
        }
        keep
    });
    if went > 0 {
        crate::host::trail(|| format!("sweep: {went} staging buffers of textures let go ({} KiB), held by no list", bytes >> 10));
    }
}

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let new = {
        let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        set.get_or_insert_with(Default::default).insert(resource as usize)
    };
    // Diagnostics (run 177): where its title memory is, for placed views
    // the title reads that memory through ([`placed_in_host_written`]).
    if new && let Some((at, desc)) = crate::resource::texture_memory(resource) {
        // SAFETY: 56 bytes in the description's layout.
        let words = unsafe { std::ptr::from_ref(&desc).cast::<[u32; 14]>().read_unaligned() };
        let bytes = crate::resource::placed_bytes(&words).unwrap_or(0);
        if !at.is_null() && bytes > 0 {
            HOST_WRITTEN_AT
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(at as u64, (bytes, words[4], words[6], words[7], words[8], resource as usize));
        }
    }
}

/// The title memory of textures the host copied into, by address: bytes
/// and description words 4, 6, 7 and 8 (width, height, layers and levels,
/// format).
static HOST_WRITTEN_AT: Mutex<std::collections::BTreeMap<u64, (u64, u32, u32, u32, u32, usize)>> =
    Mutex::new(std::collections::BTreeMap::new());

/// A texture the title placed at `address` (description words `desc`) was
/// made: says so once if it lies in the memory of a texture the host
/// copied into ([`host_written`]). Its own host copy is filled from title
/// memory, which the copy never reached: the copied texels are on the
/// card in the other texture, where no shader that reads this view finds
/// them.
pub(crate) fn placed_in_host_written(address: u64, desc: &[u32; 14]) {
    let hit = {
        let map = HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner());
        map.range(..=address)
            .next_back()
            .filter(|(at, d)| address < **at + d.0)
            .map(|(at, d)| (*at, *d))
    };
    if let Some((at, (bytes, width, height, layers_levels, format, _))) = hit {
        static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::warn!(
                view_address = format_args!("{address:#x}"),
                view_offset = address - at,
                view_width = desc[4],
                view_height = desc[6],
                view_layers_levels = format_args!("{:#x}", desc[7]),
                view_format = desc[8],
                copied_address = format_args!("{at:#x}"),
                copied_bytes = bytes,
                copied_width = width,
                copied_height = height,
                copied_layers_levels = format_args!("{layers_levels:#x}"),
                copied_format = format,
                "HERO LEAD: a placed texture view lies in the memory of a texture the host copied into: it is filled from title memory, which the copy never reached"
            );
        }
    }
}

/// Whether the host GPU wrote `resource` by a copy or resolve
/// ([`host_written`]): its host copy cannot be made again from title
/// memory.
pub(crate) fn host_wrote(resource: usize) -> bool {
    HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&resource))
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
    // Its memory may be given to other textures now (run 177: most of the
    // "HERO LEAD" lines were views in memory a released texture had held).
    HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner()).retain(|_, d| d.5 != resource);
    let mut through = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(through) = through.as_mut() {
        through.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    upload_from(list, resource, texture, 0);
}

/// [`upload`] of the mips from `first` on, the ones a view shows: of a
/// placed texture the title may have no memory for the larger ones, and
/// what lies there is another texture's ([`first_mip_switch`]).
pub(crate) fn upload_from(list: &HostList, resource: This, texture: &ID3D12Resource, first: u32) {
    let host_written = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if host_written || !list.first_use(resource as usize) && staged_from(resource) <= first {
        return;
    }
    let Some(staging) = staging(resource, texture, first) else {
        return;
    };
    use std::sync::atomic::Ordering::Relaxed;
    staging.used.store(crate::host::moment(), Relaxed);
    // Levels on demand: a view of the texture now starts at another
    // level, or another texture now starts inside its span, with no byte
    // of its memory changed. The levels are judged anew: detiled again at
    // this list's submission, and uploaded by this list.
    if staging.on_demand && staging.judgement() != staging.judged() {
        staging.forget();
        staging.dirty.store(true, Relaxed);
    }
    // Levels on demand: the last upload left a held level without tiles
    // (the card had no room, or no heap could be had), and nothing else
    // would record an upload again until the title writes the texture: the
    // level stayed empty, the texture blurred or black. Asked again about
    // once a second, and only when the card has room; the staging buffer
    // still holds the level's bytes.
    if staging.on_demand {
        let since = staging.waiting.load(Relaxed);
        if since != 0 && crate::host::long_ago(since, 1000, 2) {
            if crate::resource::room_lately() {
                staging.waiting.store(0, Relaxed);
                staging.dirty.store(true, Relaxed);
                crate::tiles::STATS[10].fetch_add(1, Relaxed);
            } else {
                staging.waiting.store(crate::host::moment().max(1), Relaxed);
            }
        }
    }
    // The host texture has the staging buffer's contents: only the check
    // at submission is left, and what it finds changed the next list that
    // samples the texture uploads (a frame late where every list of the
    // frame was recorded before that check).
    if staging.uploaded.load(Relaxed) && !staging.dirty.load(Relaxed) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexSkipped, 1);
        list.defer_texture(staging);
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::TexUploads, 1);
    crate::framelog::add(crate::framelog::T::Uploads, 1);
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: staging.view_first + i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    // When the list is submitted, its copy takes what the staging buffer
    // holds by then: noted before the check below, which sets `dirty`
    // again if it detiles (the next list then uploads once more).
    {
        let staging = staging.clone();
        list.defer_host(move || {
            staging.uploaded.store(true, Relaxed);
            staging.dirty.store(false, Relaxed);
        });
    }
    // Levels on demand (`crate::tiles`): after the check below, in the
    // order of this list's steps, and before the list is executed, the
    // levels the staging buffer holds as the title's are given the card's
    // memory if they have none. The list copies every level, so each one
    // given here is written whole by it (the rest are copies into no
    // memory, which the card drops).
    let on_demand = if staging.on_demand { crate::resource::host_tiled(resource) } else { None };
    let Some(on_demand) = on_demand else {
        // Once per submission however many of its lists sample the texture.
        list.defer_texture(staging);
        return;
    };
    list.defer_texture(staging.clone());
    list.defer_host(move || {
        let waiting = on_demand.uploaded(staging.detiled.load(Relaxed), staging.memory as u64);
        staging.waiting.store(if waiting { crate::host::moment().max(1) } else { 0 }, Relaxed);
        // A texture picked by index has no staging buffer after this
        // upload: its slots are made due again by the texture pass.
        if waiting {
            crate::descriptor::host::levels_waiting(staging.memory as u64);
        }
    });
}

/// Levels on demand: the levels a list claimed for its zeros
/// (`crate::tiles::Tiled::claim`), held by the list's submission-time
/// step. A list that is reset without having been submitted gives the
/// claim up, or the next list that writes those levels would send no zeros
/// and what its write does not cover would be what the tiles' last owner
/// left.
struct Claimed {
    on_demand: crate::resource::OnDemand,
    fresh: u32,
    ran: std::sync::atomic::AtomicBool,
}

impl Drop for Claimed {
    fn drop(&mut self) {
        if self.fresh != 0
            && !*self.ran.get_mut()
            && let Some(tiled) = self.on_demand.tiled()
        {
            tiled.unclaim(self.fresh);
        }
    }
}

/// Levels on demand (`crate::tiles`): the card itself is about to write
/// title texture `resource` in `list` (a copy or a resolve into level
/// `level`, or into every level). If its host copy `texture` was made
/// without memory, the levels written get theirs when the list is
/// submitted, ahead of it on the queue; the ones that have none yet are
/// first written with zeros, here in the same list, so that what the
/// card's write does not cover is not what the tiles' last owner left.
/// To be called before the list is brought to the state of that write.
/// shortcut: two lists that write the same new level, submitted in the
/// other order than recorded, lose the first one's write under the zeros
/// of the other; so does an upload recorded and not yet submitted.
pub(crate) fn card_writes(list: &HostList, resource: This, texture: &ID3D12Resource, level: Option<u32>) {
    let Some(on_demand) = crate::resource::host_tiled(resource) else {
        return;
    };
    let Some(tiled) = on_demand.tiled() else {
        return;
    };
    let levels = match level {
        Some(level) => tiled.with_level(level),
        None => tiled.every_level(),
    };
    if levels == 0 {
        return;
    }
    let fresh = tiled.claim(levels);
    // Of them, the ones whose zeros could not be recorded.
    let mut unzeroed = if host().is_none() { fresh } else { 0 };
    if fresh != 0
        && let Some(host) = host()
    {
        const COPY_DEST: u32 = 0x400;
        crate::list::ensure_state(list, resource, COPY_DEST);
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        for level in (0..u32::from(d.MipLevels).min(32)).filter(|&level| fresh & (1u32 << level) != 0) {
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut bytes = 0u64;
            // SAFETY: outputs for one subresource.
            unsafe {
                host.device
                    .GetCopyableFootprints(&d, level, 1, 0, Some(&mut footprint), None, None, Some(&mut bytes));
            }
            let Some((buffer, data)) = list.upload(host, bytes) else {
                // (With the levels that share its tiles, if it is one of
                // the small ones.)
                unzeroed |= tiled.with_level(level);
                continue;
            };
            // SAFETY: a mapped buffer of `bytes` bytes, just made.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: level },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above; the list holds the buffer until it is reset.
                pResource: unsafe { std::mem::transmute_copy(&buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }
    // A new level without its zeros gets no tiles from this list: the
    // card's write into it is lost (the level reads zeros, as before the
    // write), which is less wrong than a part write over what the tiles'
    // last owner left. The next list that writes it claims it again.
    unzeroed &= fresh;
    if unzeroed != 0 {
        tiled.unclaim(unzeroed);
    }
    let levels = levels & !unzeroed;
    if levels == 0 {
        return;
    }
    let memory = crate::resource::texture_memory(resource).map_or(0, |(memory, _)| memory as u64);
    let step = Claimed {
        on_demand,
        fresh: fresh & !unzeroed,
        ran: std::sync::atomic::AtomicBool::new(false),
    };
    list.defer_host(move || {
        step.ran.store(true, std::sync::atomic::Ordering::Relaxed);
        step.on_demand.card_wrote(levels, memory);
    });
}

impl Staging {
    /// Levels on demand: what says, beside the mapping, which levels of
    /// the texture are the title's now. First the lowest level a live view
    /// of the texture shows (0 without a view that says): the title's own
    /// word, a texture it holds from level 3 on is viewed from level 3.
    /// Then where the next texture placed in this one's span starts, in
    /// bytes from this one's start (`u64::MAX`: none does): for a texture
    /// viewed from level 0 although a neighbour sits where its large
    /// levels would be.
    fn judgement(&self) -> [u64; 2] {
        let start = self.memory as u64;
        let first = crate::descriptor::host::first_viewed(start).map_or(0, u64::from);
        let limit = crate::descriptor::host::next_start(start, start.saturating_add(self.layout.size))
            .map_or(u64::MAX, |at| at - start);
        [first, limit]
    }

    /// (Neptune) Levels on demand: the bytes at the start of the texture's
    /// memory that hold the levels that are the title's now (the smallest
    /// levels first, level 0 last). What lies behind them is not mapped, or
    /// is another texture's.
    fn held_bytes(&self) -> usize {
        let held = self.held_levels(self.judgement());
        if held == u32::MAX {
            return self.title_bytes;
        }
        if held == 0 {
            return (self.layout.alignment.min(self.layout.slice_size) as usize).min(self.title_bytes);
        }
        // The lowest bit is the largest level held.
        (self.layout.bytes_from(held.trailing_zeros()) as usize).min(self.title_bytes)
    }

    /// What [`Staging::judgement`] said at the last detiling.
    fn judged(&self) -> [u64; 2] {
        use std::sync::atomic::Ordering::Relaxed;
        [self.judged[0].load(Relaxed), self.judged[1].load(Relaxed)]
    }

    /// Levels on demand: the levels that are the title's now, a bit a
    /// level. A level is held if no view starts above it, its bytes end
    /// before the next texture starts, and all of them are mapped. The
    /// smallest levels share the texture's first block and are held
    /// together. (The console layout: that block first, then the other
    /// levels from the smallest up, level 0 last:
    /// `victory_formats::texture::Layout`.) A texture this cannot tell
    /// the levels of holds them all.
    fn held_levels(&self, [first, limit]: [u64; 2]) -> u32 {
        let mips = &self.layout.mips;
        if self.layout.slices != 1 || mips.is_empty() || mips.len() > 32 {
            return u32::MAX;
        }
        let tail = (self.layout.first_tail_mip as usize).min(mips.len());
        let there = |from: u64, to: u64| {
            to > from && to <= limit && committed(self.memory.wrapping_add(from as usize), (to - from) as usize)
        };
        let mut held = 0u32;
        for (level, m) in mips.iter().enumerate().take(tail) {
            let end = if level == 0 { self.layout.slice_size } else { mips[level - 1].offset };
            if level as u64 >= first && there(m.offset, end) {
                held |= 1 << level;
            }
        }
        if tail < mips.len() && there(0, self.layout.alignment.min(self.layout.slice_size)) {
            for level in tail..mips.len() {
                if level as u64 >= first {
                    held |= 1 << level;
                }
            }
        }
        held
    }

    /// Forgets what the staging buffer was last filled from: the next
    /// refresh detiles anew, whether the memory changed or not.
    fn forget(&self) {
        *self.last.lock().unwrap_or_else(|e| e.into_inner()) = Last::default();
    }

    /// The title memory the texture is read from: address and bytes.
    pub(crate) fn memory(&self) -> (usize, usize) {
        (self.memory as usize, self.layout.size as usize)
    }

    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let read = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let whole = std::time::Instant::now();
        let read = {
            let _faults = crate::faults::scope(crate::faults::Kind::Upload);
            self.refresh_timed()
        };
        crate::faults::count(crate::faults::Driver::UploadBytes, read as u64);
        let us = whole.elapsed().as_micros() as u64;
        REFRESH_WAITS[3].fetch_add(us, std::sync::atomic::Ordering::Relaxed);
        crate::framelog::add(crate::framelog::T::RefreshUs, us);
        read
    }

    fn refresh_timed(&self) -> usize {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = waited(0, || self.last.lock().unwrap_or_else(|e| e.into_inner()));
        let last = &mut *last;
        if let Some((now, written)) = waited(1, || crate::watch::writes(start, start + size, false)) {
            if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                return 0;
            }
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            with_read(size, |read| {
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                self.detile(read);
            });
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return size;
        }
        // View memory has no write-watch, and a texture there was hashed
        // whole at every check: 229 MB and 51 ms a frame in a gameplay view
        // where two textures change (run 58). It is guarded now, as buffers
        // in view memory are: a write of the title's to it faults once and
        // is noted, a mapping is announced by the runtime. While neither
        // happened the texture is as it was. What the guard cannot see (a
        // write to the same pages through another view) is still found by
        // the hash, which is taken every eighth frame only, by the
        // texture's address, so the textures take turns.
        // `VICTORY_TEXTURE_GUARD=0` hashes at every check, as before.
        static GUARD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        // A texture that changed without the guard seeing it is hashed at
        // every check for a while (`unseen_switch`).
        let mut unseen = false;
        if *GUARD.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_GUARD").is_none_or(|v| v != "0"))
            && let Some((now, written)) = waited(1, || crate::watch::writes(start, start + size, true))
        {
            // (Neptune) `VICTORY_TEXTURE_HASH_EVERY=<frames>` (8): how often
            // a guarded texture is hashed all the same. Neptune: the hashing
            // was 5 ms of every frame of the first room (its launcher: 32).
            static HASH_EVERY_SET: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            let hash_every: u64 = *HASH_EVERY_SET.get_or_init(|| {
                std::env::var("VICTORY_TEXTURE_HASH_EVERY").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(8).max(1)
            });
            let unwritten = last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked);
            last.asked = now;
            let frame = crate::frame::current();
            let turn = frame.wrapping_add((start >> 16) as u64).is_multiple_of(hash_every);
            // Linear pictures of one level are what the processor draws (the
            // planes of a movie), through a view of its own.
            let desc = &self.layout.desc;
            let watched = unseen_switch() && (desc.swizzle.0 == 0 && desc.mips == 1 || frame < last.unguarded);
            if unwritten && !turn && !watched {
                return 0;
            }
            unseen = unwritten && unseen_switch();
        }
        // From here on the texture's memory may be read in place. Held until
        // that read is done: the runtime changes what title memory shows
        // only with this taken alone, so what `committed` finds mapped
        // stays mapped (see `crate::watch::title_memory_reading`; without
        // it a page went between the asking and the read in runs 71 and
        // 74).
        let _mapped = if no_wait() {
            // Not waited for: while the runtime remaps title memory the
            // texture is read through the kernel instead, whole or not at
            // all (`no_wait`).
            let Some(mapped) = crate::watch::title_memory_reading_now(start, size) else {
                let never = last.hash.is_none();
                let read = self.refresh_unlocked(last, size);
                if read != 0 || !never || !read_around_switch() {
                    return read;
                }
                UNLOCKED[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return self.refresh_around(last, size);
            };
            mapped
        } else {
            waited(2, || crate::watch::title_memory_reading(start, size))
        };
        if !committed(self.memory, size) {
            // Partly mapped: a streamed texture with only some mips' pages
            // there, the usual state while it streams in. Read through the
            // copy, which stands unmapped pages (zeros), and upload what
            // is there; nothing at all mapped changes nothing.
            // shortcut: read again at most every 30 frames (reading around
            // unmapped pages is slow, and nothing tells us when pages
            // arrive): a mip that streams in shows up to 30 frames late.
            let frame = crate::frame::current();
            if last.partial.is_some_and(|at| frame < at + 30) {
                return 0;
            }
            last.partial = Some(frame);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
            let changed = with_read(size, |read| {
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                let hash = blake3::hash(read);
                if last.hash == Some(hash) {
                    return false;
                }
                last.hash = Some(hash);
                if read.iter().all(|&b| b == 0) {
                    return false;
                }
                self.detile(read);
                true
            });
            #[cfg(feature = "diagnostics")]
            if changed {
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
            }
            let _ = changed;
            return size;
        }
        if through_copy() {
            return with_title(self.memory, size, |tiled| {
                let hash = {
                    let _faults = crate::faults::scope(crate::faults::Kind::Hash);
                    blake3::hash(tiled)
                };
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
                if last.hash == Some(hash) {
                    return size;
                }
                if unseen && last.hash.is_some() {
                    self.unseen_write(last);
                }
                last.hash = Some(hash);
                self.detile(tiled);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
                size
            });
        }
        // SAFETY: committed memory of the texture's layout size, and it
        // stays committed while `_mapped` is held.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = {
            let _faults = crate::faults::scope(crate::faults::Kind::Hash);
            blake3::hash(tiled)
        };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return size;
        }
        if unseen && last.hash.is_some() {
            self.unseen_write(last);
        }
        last.hash = Some(hash);
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// The texture's memory changed and the guard says nothing wrote it: it
    /// is hashed at every check for the next frames, the longer the more
    /// often this happens (a texture streamed in once is soon guarded
    /// again, a movie's is not). (myokex c3723f3.)
    fn unseen_write(&self, last: &mut Last) {
        let frame = crate::frame::current();
        let more = if frame < last.unguarded { 120 } else { 16 };
        let n = UNSEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 24 || n.is_multiple_of(1000) {
            tracing::info!(
                frame,
                so_far = n + 1,
                memory = ?self.memory,
                size = self.layout.size,
                "a guarded texture changed and no write to it was seen: hashed at every check for a while (VICTORY_TEXTURE_UNSEEN=0: not)"
            );
        }
        last.unguarded = frame + more;
    }

    /// A refresh of a texture that has been given nothing yet, while the
    /// runtime is remapping its memory ([`no_wait`], [`read_around_switch`]):
    /// read around the pages that are not there, through the kernel (the
    /// fault-guarded copy), as a partly mapped texture is. Nothing but zeros
    /// is taken for "not there yet". Returns the bytes read. (myokex
    /// c3723f3.)
    fn refresh_around(&self, last: &mut Last, size: usize) -> usize {
        with_read(size, |read| {
            crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
            if read.iter().all(|&b| b == 0) {
                last.asked = 0;
                return 0;
            }
            last.hash = Some(blake3::hash(read));
            last.partial = Some(crate::frame::current());
            self.detile(read);
            size
        })
    }

    /// A refresh while the runtime is remapping title memory ([`no_wait`]):
    /// the texture is read through the kernel, which fails if a page of it
    /// is not there at this moment. Then nothing is taken and the texture
    /// is looked at again at its next refresh (what was learnt about writes
    /// to it is forgotten, so that it is not passed over as unwritten).
    /// Returns the bytes read.
    fn refresh_unlocked(&self, last: &mut Last, size: usize) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let read = with_read(size, |read| {
            if !crate::host::read_into(self.memory, read) {
                return false;
            }
            let hash = blake3::hash(read);
            if last.hash != Some(hash) {
                last.hash = Some(hash);
                self.detile(read);
            }
            true
        });
        if read {
            UNLOCKED[0].fetch_add(1, Relaxed);
            size
        } else {
            UNLOCKED[1].fetch_add(1, Relaxed);
            last.asked = 0;
            0
        }
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        // Levels on demand: which levels are the title's is judged here,
        // once, for what the staging buffer holds from now on. The step
        // that gives levels their memory goes by this answer.
        let held = if self.on_demand {
            use std::sync::atomic::Ordering::Relaxed;
            let judged = self.judgement();
            self.judged[0].store(judged[0], Relaxed);
            self.judged[1].store(judged[1], Relaxed);
            let held = self.held_levels(judged);
            self.detiled.store(held, Relaxed);
            held
        } else {
            u32::MAX
        };
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if mip < 32 && held & (1 << mip) == 0 {
                // Not the title's: zeros, not what lies in that memory.
                linear.fill(0);
                continue;
            }
            if let Err(e) = detile_wide(&self.layout, tiled, mip + self.first_mip, slice, linear, pitch, rows) {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

/// A subresource of at least twice this many bytes is detiled in bands of
/// about this many on the kept threads ([`detile_wide`]): one thread
/// detiles some 4.5 GB/s, and a 33-67 MB texture went 1.7-2 times as fast
/// on four (Trexies 58c7bb0: 33 MB 7.3 -> 4.2 ms, 67 MB 16.6-17.1 ->
/// 7.4-8.5 ms).
const BAND_BYTES: usize = 4 << 20;

/// Whether big subresources are detiled in bands on several threads
/// (`VICTORY_TEXTURE_BANDS=0`: whole, on the refreshing thread, as
/// before). On by default. The bytes are the same either way.
fn banded() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_BANDS").is_none_or(|v| v != "0"))
}

/// One subresource's bands ([`detile_wide`]): what each needs, as
/// addresses (the caller's borrows, alive until every band taken is done),
/// the next band to take, and the bands done.
struct Bands {
    layout: usize,
    tiled: (usize, usize),
    linear: usize,
    pitch: usize,
    height: u32,
    band: u32,
    count: u32,
    mip: u32,
    slice: u32,
    next: std::sync::atomic::AtomicU32,
    done: Mutex<u32>,
    all_done: std::sync::Condvar,
    failed: Mutex<Option<victory_formats::texture::Error>>,
}

impl Bands {
    /// Takes bands until none is left. A helper that starts after the
    /// last band was taken touches nothing of the caller's.
    fn work(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        loop {
            let k = self.next.fetch_add(1, Relaxed);
            if k >= self.count {
                return;
            }
            let y = k * self.band;
            let end = (y + self.band).min(self.height);
            // SAFETY: band `k` was taken, so the caller of `detile_wide`
            // waits for it to be counted done before its borrows of the
            // layout and both buffers end; the bands are disjoint rows of
            // `linear`.
            let (layout, tiled, part) = unsafe {
                (
                    &*(self.layout as *const victory_formats::texture::Layout),
                    std::slice::from_raw_parts(self.tiled.0 as *const u8, self.tiled.1),
                    std::slice::from_raw_parts_mut(
                        (self.linear + y as usize * self.pitch) as *mut u8,
                        (end - y) as usize * self.pitch,
                    ),
                )
            };
            if let Err(e) = layout.detile_rows(tiled, self.mip, self.slice, part, self.pitch, y..end) {
                *self.failed.lock().unwrap_or_else(|e| e.into_inner()) = Some(e);
            }
            let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
            *done += 1;
            if *done == self.count {
                self.all_done.notify_all();
            }
        }
    }
}

/// [`Layout::detile`](victory_formats::texture::Layout::detile) of one
/// subresource into `linear` (`rows` rows `pitch` bytes apart, depth slices
/// `pitch * rows` apart); a big one of one slice deep is cut into bands of
/// rows that the kept threads ([`crate::shadow::wide_offer`]) and this one
/// detile at once (Trexies 58c7bb0). This thread takes bands of its own
/// subresource only, never another job of the queue: it may hold a read of
/// title memory in place and the texture's lock.
fn detile_wide(
    layout: &victory_formats::texture::Layout,
    tiled: &[u8],
    mip: u32,
    slice: u32,
    linear: &mut [u8],
    pitch: usize,
    rows: u32,
) -> Result<(), victory_formats::texture::Error> {
    let height = layout.mips.get(mip as usize).map_or(0, |m| m.height);
    let deep = layout.subresource_depth(mip) != 1;
    if !banded() || deep || linear.len() < 2 * BAND_BYTES || pitch == 0 || height < 2 || height > rows {
        return layout.detile(tiled, mip, slice, linear, pitch, pitch * rows as usize);
    }
    let band = (BAND_BYTES / pitch).clamp(1, height as usize) as u32;
    let count = height.div_ceil(band);
    let bands = Arc::new(Bands {
        layout: std::ptr::from_ref(layout) as usize,
        tiled: (tiled.as_ptr() as usize, tiled.len()),
        linear: linear.as_mut_ptr() as usize,
        pitch,
        height,
        band,
        count,
        mip,
        slice,
        next: std::sync::atomic::AtomicU32::new(0),
        done: Mutex::new(0),
        all_done: std::sync::Condvar::new(),
        failed: Mutex::new(None),
    });
    let helpers = crate::shadow::wide_threads().min(count as usize - 1);
    crate::shadow::wide_offer(
        (0..helpers)
            .map(|_| {
                let bands = bands.clone();
                Box::new(move || bands.work()) as Box<dyn FnOnce() + Send>
            })
            .collect(),
    );
    bands.work();
    let mut done = bands.done.lock().unwrap_or_else(|e| e.into_inner());
    while *done < count {
        done = bands.all_done.wait(done).unwrap_or_else(|e| e.into_inner());
    }
    drop(done);
    match bands.failed.lock().unwrap_or_else(|e| e.into_inner()).take() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Lets go of a title texture's staging buffer (lists that recorded an
/// upload from it hold it until they have run); the next upload makes one
/// anew.
pub(crate) fn unstage(resource: This) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&(resource as usize));
    }
}

/// Where a staged title texture's memory is: address and bytes.
pub(crate) fn staged_memory(resource: This) -> Option<(u64, u64)> {
    let s = {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        map.as_ref()?.get(&(resource as usize))?.clone()
    };
    // (Neptune) Of a texture made level by level: the levels the title
    // holds. The memory behind them is another texture's, and its changes
    // are not this one's.
    let (at, len) = s.memory();
    let len = if s.on_demand { s.held_bytes().min(len) } else { len };
    Some((at as u64, len as u64))
}

/// A fingerprint of the title memory `[at, at + len)`, never 0: of all of
/// it up to 48 KiB, else of 16 KiB at its start, its middle and its end
/// (a texture streams in whole mips; the small ones lie first). Read
/// through the kernel, so pages not there count as zeros.
pub(crate) fn fingerprint(at: u64, len: u64) -> u64 {
    const WINDOW: usize = 16 << 10;
    thread_local! {
        static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let len = len as usize;
    let mut hasher = blake3::Hasher::new();
    READ.with(|read| {
        let mut read = read.borrow_mut();
        read.resize(3 * WINDOW, 0);
        let mut take = |from: usize, bytes: usize| {
            read[..bytes].fill(0);
            let _kind = crate::readback::consumers::scope(crate::readback::consumers::Kind::Texture);
            crate::host::copy_mapped(read.as_mut_ptr(), (at as usize + from) as *const u8, bytes, false);
            hasher.update(&read[..bytes]);
        };
        if len <= 3 * WINDOW {
            take(0, len);
        } else {
            take(0, WINDOW);
            take((len / 2) & !0xfff, WINDOW);
            take(len - WINDOW, WINDOW);
        }
    });
    let hash = hasher.finalize();
    let mut first = [0u8; 8];
    first.copy_from_slice(&hash.as_bytes()[..8]);
    u64::from_le_bytes(first) | 1
}

/// Whether a title texture has a staging buffer: a list recorded its
/// upload, and the lists that sample it check it at submission.
/// For the frame listing: whether a list has uploaded `resource`'s staging
/// buffer, and whether it was detiled since. None: it has none.
#[cfg(feature = "diagnostics")]
pub(crate) fn staged_state(resource: This) -> Option<(bool, bool)> {
    use std::sync::atomic::Ordering::Relaxed;
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let staging = map.as_ref()?.get(&(resource as usize))?;
    Some((staging.uploaded.load(Relaxed), staging.dirty.load(Relaxed)))
}

pub(crate) fn is_staged(resource: This) -> bool {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref().is_some_and(|map| map.contains_key(&(resource as usize)))
}

/// The bytes of a title texture's memory, and whether all of it is mapped,
/// if the texture has been staged for upload.
pub(crate) fn staged(resource: This) -> Option<(u64, bool)> {
    let s = {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        map.as_ref()?.get(&(resource as usize))?.clone()
    };
    if s.on_demand {
        // (Neptune) A texture made level by level is all there when the
        // levels the title holds are: that the others' memory is not mapped
        // is its lasting state, not a texture streaming in. Told "not
        // whole", the pass over the heap's views uploaded each of them again
        // every 120 frames for good (Neptune: 1,300 slots "uploaded in
        // part" in the first room, some 200 uploads a second, each of the
        // texture's whole span read, hashed and detiled). When the title
        // maps more of it or writes its view the slot is due again.
        return Some((s.held_bytes() as u64, true));
    }
    let size = s.title_bytes;
    Some((size as u64, committed(s.memory, size)))
}

/// The first mip a title texture's staging buffer holds of its host copy
/// ([`first_mip_switch`]); `u32::MAX` without one.
fn staged_from(resource: This) -> u32 {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref().and_then(|map| map.get(&(resource as usize))).map_or(u32::MAX, |s| s.view_first)
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    if by_query() {
        return committed_by_query(at, len);
    }
    if len == 0 {
        return true;
    }
    let (there, of) = readable(at as usize, len, true);
    there == of
}

/// Whether the two questions above and below are put to `VirtualQuery`, as
/// they were (`VICTORY_COMMITTED_QUERY=1`). `VirtualQuery` answers with the
/// length of the run of pages in the same state, which it finds by walking
/// the page tables to the run's end: 8.8 ms for one question in the title's
/// 1.1 GB block (round 6). The pass over the heap's views asks three or
/// more of them for every texture slot that is due, 3,000 slots a frame
/// when the card has no room (run 75): that pass then ran 208-264 ms, and
/// while it ran the submitting thread's write-watch askings waited behind
/// the walks (run 76: 8-22 askings a frame of 6-21 ms each, 46-172 ms).
fn by_query() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COMMITTED_QUERY").is_some_and(|v| v == "1"))
}

/// Of the kernel's pages that hold `[at, at + len)` (`len > 0`): how many
/// can be read, and how many there are; with `all`, it stops at the first
/// that cannot. One question to the kernel for up to 256 pages at once
/// (`QueryWorkingSetEx`: the state of each page asked about, no walk); a
/// page not in the working set may still be there (never touched, or
/// trimmed), and a one-byte read through the kernel says.
/// A page that is committed and may not be read counts as not there, which
/// is what the callers ask for: they read or write the memory next.
fn readable(at: usize, len: usize, all: bool) -> (usize, usize) {
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const KERNEL_PAGE: usize = 4096;
    let (first, last) = (at / KERNEL_PAGE, at.saturating_add(len - 1) / KERNEL_PAGE);
    let of = last - first + 1;
    let mut there = 0;
    // SAFETY: plain data.
    let mut info: [PSAPI_WORKING_SET_EX_INFORMATION; 256] = unsafe { std::mem::zeroed() };
    let mut page = first;
    while page <= last {
        let n = (last - page + 1).min(info.len());
        for (i, entry) in info[..n].iter_mut().enumerate() {
            entry.VirtualAddress = ((page + i) * KERNEL_PAGE) as *mut _;
        }
        // SAFETY: `n` entries of the size given; our own process.
        let asked = unsafe {
            K32QueryWorkingSetEx(
                GetCurrentProcess(),
                info.as_mut_ptr().cast(),
                (n * size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32,
            )
        } != 0;
        for (i, entry) in info[..n].iter().enumerate() {
            // Bit 0 of the answer: the page is in the working set.
            // SAFETY: plain data the query filled in.
            let valid = asked && unsafe { entry.VirtualAttributes.Flags } & 1 != 0;
            let mut byte = [0u8];
            if valid || crate::host::read_into(((page + i) * KERNEL_PAGE) as *const u8, &mut byte) {
                there += 1;
            } else if all {
                return (there, of);
            }
        }
        page += n;
    }
    (there, of)
}

fn committed_by_query(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// How many bytes of `[at, at + len)` are committed memory: all of a
/// texture that is there, some of one streaming in (its smaller mips lie
/// at the start of its memory, and their pages are mapped first), none of
/// one not there. With neither the first nor the last page committed the
/// answer is none without a look at the pages between (two queries for the
/// slots whose texture is away, which are asked again and again).
pub(crate) fn mapped_bytes(at: *const u8, len: usize) -> u64 {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    if len == 0 {
        return 0;
    }
    let start = at as usize;
    let end = start.saturating_add(len);
    if !committed(at, 1) && !committed((end - 1) as *const u8, 1) {
        return 0;
    }
    if !by_query() {
        static BY_PAGE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *BY_PAGE.get_or_init(|| std::env::var_os("VICTORY_MAPPED_BY_PAGE").is_some_and(|v| v == "1")) {
            // By page, without the walk: whole pages, cut to the range.
            let (there, _) = readable(start, len, false);
            return (there as u64 * 4096).min(len as u64);
        }
        // By 64 KiB block, one page asked of each: the title's memory is
        // mapped in blocks of that size (the console's page), and the
        // answer is a measure of how much of a texture has arrived, not a
        // promise that every byte can be read (the upload reads through the
        // kernel). By page, a texture half there cost a failed kernel read
        // for each of its missing pages, every frame it stayed due: 1,663
        // textures "uploaded in part" kept the pass over the heap's views at
        // 90-160 ms a frame (run 81). `VICTORY_MAPPED_BY_PAGE=1`: by page.
        const BLOCK: usize = 64 << 10;
        let mut there = 0u64;
        let mut block = start & !(BLOCK - 1);
        while block < end {
            let (lo, hi) = (block.max(start), block.saturating_add(BLOCK).min(end));
            if readable(lo, 1, true).0 == 1 {
                there += (hi - lo) as u64;
            }
            block = block.saturating_add(BLOCK);
        }
        return there;
    }
    let (mut p, mut there) = (start, 0u64);
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize).min(end);
        if n == 0 || next <= p {
            break;
        }
        if info.State == MEM_COMMIT {
            there += (next - p) as u64;
        }
        p = next;
    }
    there
}

/// For a single 2D slice the smaller mips form a prefix of the title
/// allocation. Keep their original tiled offsets and tail placement while
/// shortening the source that refreshes watch, copy and hash. Array slices
/// have the original stride between their prefixes.
fn staging_layout(
    mut layout: victory_formats::texture::Layout,
    drop: u32,
) -> (victory_formats::texture::Layout, u32) {
    use victory_formats::texture::Dimension;
    if drop == 0 || layout.desc.dimension != Dimension::Tex2d || layout.slices != 1 {
        return (layout, drop);
    }
    // drop_levels always leaves a mip. A cutoff within the packed tail
    // keeps its entire block, including any bytes shared with a dropped mip.
    let last_dropped = &layout.mips[drop as usize - 1];
    let bytes = if last_dropped.tail.is_some() {
        layout.alignment
    } else {
        last_dropped.offset
    };
    layout.mips.drain(..drop as usize);
    layout.first_tail_mip = layout.first_tail_mip.saturating_sub(drop);
    layout.desc.width = (layout.desc.width >> drop).max(1);
    layout.desc.height = (layout.desc.height >> drop).max(1);
    layout.desc.mips -= drop;
    layout.size = bytes;
    layout.slice_size = bytes;
    (layout, 0)
}

/// The staging state of a title texture from the mip `first` on
/// ([`first_mip_switch`]), made on first use (and made anew for a view of
/// larger mips than the one there holds).
fn staging(resource: This, texture: &ID3D12Resource, first: u32) -> Option<Arc<Staging>> {
    // Asked before the table is locked: the sweep lets go of a staging
    // buffer with its texture's host side locked, in that order.
    let on_demand = crate::resource::host_tiled(resource).is_some();
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize))
        && s.view_first <= first
    {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    // From the view's first mip only where the title's layout can end
    // behind the smaller mips: one 2D slice, all its mips in the host copy.
    let drop = crate::resource::drop_levels(&desc);
    let first = if first_mip_switch()
        && first > 0
        && !on_demand
        && drop == 0
        && d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && d.DepthOrArraySize == 1
        && first < u32::from(d.MipLevels)
        && first < layout.desc.mips
        && layout.slices == 1
        && layout.desc.dimension == victory_formats::texture::Dimension::Tex2d
        && layout.bytes_from(first) < layout.size
    {
        first
    } else {
        0
    };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize) - first
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            first,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data, base, pooled, piece) = upload_memory(host, total)?;
    for footprint in &mut footprints {
        footprint.Offset += base;
    }
    if piece.is_none() {
        crate::host::name(&buffer, || {
            format!(
                "staging of {resource:?} {}x{}x{} f{}",
                d.Width, d.Height, d.DepthOrArraySize, d.Format.0
            )
        });
    }
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    STAGED[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    STAGED[1].fetch_add(total, std::sync::atomic::Ordering::Relaxed);
    let (title_bytes, (layout, first_mip)) = if first > 0 {
        // The smaller mips' prefix, read, hashed and watched as the whole.
        let (layout, _) = staging_layout(layout, first);
        (layout.size as usize, (layout, 0))
    } else {
        (layout.size as usize, staging_layout(layout, drop))
    };
    let s = Arc::new(Staging {
        buffer,
        #[cfg(feature = "diagnostics")]
        size: total,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        first_mip,
        title_bytes,
        view_first: first,
        bytes: total,
        memory,
        last: Mutex::default(),
        dirty: std::sync::atomic::AtomicBool::new(true),
        uploaded: std::sync::atomic::AtomicBool::new(false),
        used: std::sync::atomic::AtomicU64::new(crate::host::moment()),
        on_demand,
        detiled: std::sync::atomic::AtomicU32::new(0),
        judged: [const { std::sync::atomic::AtomicU64::new(0) }; 2],
        waiting: std::sync::atomic::AtomicU64::new(0),
        pooled,
        piece,
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}

#[cfg(test)]
mod title_memory_tests {
    use super::with_title;

    /// Title memory whose tail is reserved and not committed reads through
    /// the copy as its bytes and then zeros, without a fault; so does
    /// memory a page of which is taken away between two reads, and the
    /// zeros are zeros whatever the read buffer held before.
    #[test]
    fn an_uncommitted_tail_reads_as_zeros() {
        use windows_sys::Win32::System::Memory::*;
        const HEAD: usize = 0xb_0000;
        const SIZE: usize = 0x20_0000;
        // SAFETY: a private reservation of our own; only committed pages
        // are written in place, and nothing else reads them in place.
        unsafe {
            // The read buffer holds something else first.
            let other = vec![0x77u8; SIZE];
            with_title(other.as_ptr(), SIZE, |bytes| assert!(bytes.iter().all(|&b| b == 0x77)));
            let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base.cast(), HEAD, MEM_COMMIT, PAGE_READWRITE), base.cast());
            base.write_bytes(0x5a, HEAD);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..HEAD].iter().all(|&b| b == 0x5a), "what is mapped");
                assert!(bytes[HEAD..].iter().all(|&b| b == 0), "what is not: zeros");
            });
            // The title takes a page in the middle away: zeros from then
            // on, the rest as before.
            assert_ne!(VirtualFree(base.add(0x4_0000).cast(), 0x1_0000, MEM_DECOMMIT), 0);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..0x4_0000].iter().all(|&b| b == 0x5a));
                assert!(bytes[0x4_0000..0x5_0000].iter().all(|&b| b == 0));
                assert!(bytes[0x5_0000..HEAD].iter().all(|&b| b == 0x5a));
                assert!(bytes[HEAD..].iter().all(|&b| b == 0));
            });
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn retained_source_detiles_every_mip_from_its_original_offset() {
        use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
        for swizzle in [Swizzle::LINEAR, Swizzle::SW_4KB_S, Swizzle::SW_64KB_S] {
            let title = Layout::new(&Desc {
                dimension: Dimension::Tex2d,
                width: 512,
                height: 512,
                depth: 1,
                mips: 10,
                element_bytes: 16,
                block: 4,
                samples: 1,
                swizzle,
            })
            .unwrap();
            let bytes: Vec<_> = (0..title.size)
                .map(|i| (i ^ (i >> 8) ^ (i >> 16)) as u8)
                .collect();
            for drop in [1, 2, 4] {
                let (source, first) = super::staging_layout(title.clone(), drop);
                assert_eq!(first, 0);
                assert!(source.size < title.size);
                let read = &bytes[..source.size as usize];
                for mip in drop..title.desc.mips {
                    let m = &title.mips[mip as usize];
                    let pitch = (m.width * title.desc.element_bytes) as usize;
                    let len = pitch * m.height as usize;
                    let mut expected = vec![0; len];
                    let mut actual = vec![0; len];
                    title
                        .detile(&bytes, mip, 0, &mut expected, pitch, len)
                        .unwrap();
                    source
                        .detile(read, mip - drop, 0, &mut actual, pitch, len)
                        .unwrap();
                    assert!(
                        actual == expected,
                        "{swizzle:?}, dropped {drop}, title mip {mip}"
                    );
                }
            }
        }
    }

    #[test]
    fn texture_refresh_reads_retained_mips_and_keeps_array_slice_stride() {
        use super::{Last, fingerprint, staged, staged_memory, staging};
        use crate::{object, resource};
        use std::sync::atomic::Ordering::Relaxed;
        use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
        use windows::core::Interface;

        let Some(host) = crate::host::host() else {
            eprintln!("no D3D12 device; skipping texture refresh");
            return;
        };
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
        };
        // SAFETY: the LUID comes from the device that creates the staging buffer.
        let adapter = unsafe {
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid()))
                .and_then(|a| a.GetDesc1())
        };
        if let Ok(adapter) = adapter {
            let n = adapter
                .Description
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(adapter.Description.len());
            eprintln!(
                "texture refresh D3D12 adapter: {}",
                String::from_utf16_lossy(&adapter.Description[..n])
            );
        }
        for slices in [1u32, 2] {
            // Four RGBA8 mips, 512x512 down to 64x64. The last row fills
            // its entire 256-byte host pitch. The backing is ours.
            let words = [
                3u32,
                0,
                0,
                0,
                512,
                0,
                512,
                (4 << 16) | slices,
                28,
                1,
                0,
                0,
                0,
                0,
            ];
            // SAFETY: the 56-byte title resource description is plain numbers.
            let desc: resource::ResourceDesc = unsafe { std::mem::transmute(words) };
            let title = &resource::texture_layout(&desc).unwrap().planes[0].layout;
            let drop = resource::drop_levels(&desc);
            if std::env::var("VICTORY_TEXTURE_DROP_MIPS").as_deref() == Ok("1") {
                assert_eq!(drop, 1);
            }
            let mut bytes = vec![1u8; title.size as usize];
            let address = bytes.as_ptr() as u64;
            let mut image = std::ptr::null();
            assert_eq!(
                resource::create_at_address(
                    std::ptr::null(),
                    address,
                    &desc,
                    0,
                    std::ptr::null(),
                    std::ptr::from_ref(&ID3D12Resource::IID).cast(),
                    &mut image,
                ),
                object::S_OK
            );
            let upload = staging(image, resource::host_resource(image).unwrap(), 0).unwrap();
            let (at, size) = staged_memory(image).unwrap();
            assert_eq!(at, address);
            let cropped = drop > 0 && slices == 1;
            let expected = if cropped {
                title.mips[drop as usize - 1].offset
            } else {
                title.size
            };
            eprintln!(
                "texture refresh: drop={drop} slices={slices} title_bytes={} source_bytes={size} expected_bytes={expected}",
                title.size
            );
            assert_eq!(size, expected, "refresh source bytes with {slices} slices");
            assert_eq!(
                staged(image).unwrap().0,
                title.size,
                "original title budget"
            );
            let mut last = Last::default();
            let refresh = |last: &mut Last| {
                assert_eq!(upload.refresh_unlocked(last, size as usize), size as usize);
            };
            refresh(&mut last);
            let original = last.hash;
            let print = fingerprint(at, size);
            upload.dirty.store(false, Relaxed);
            // The final byte lies in the largest mip in the final slice.
            *bytes.last_mut().unwrap() ^= 0x55;
            refresh(&mut last);
            assert_eq!(last.hash == original, cropped);
            assert_eq!(fingerprint(at, size) == print, cropped);
            assert_eq!(upload.dirty.load(Relaxed), !cropped);

            upload.dirty.store(false, Relaxed);
            let slice = slices - 1;
            let source = title.offset(drop, 0, 0, slice).unwrap() as usize;
            bytes[0] ^= 0x11; // the first window sampled by the recheck
            bytes[source] = 0xab;
            refresh(&mut last);
            assert_ne!(last.hash, original);
            assert_ne!(fingerprint(at, size), print);
            assert!(upload.dirty.load(Relaxed));
            let host_mip = (slice * (4 - drop)) as usize;
            let offset = upload.footprints[host_mip].Offset as usize;
            // SAFETY: first byte of this live, mapped staging subresource.
            assert_eq!(unsafe { upload.data.add(offset).read() }, 0xab);
            // The same texel's other channels retain their original bytes.
            // SAFETY: the next three bytes are in the same RGBA8 texel.
            assert_eq!(
                unsafe { std::slice::from_raw_parts(upload.data.add(offset + 1), 3) },
                &[1, 1, 1]
            );
            std::mem::drop(upload);
            assert_eq!(
                object::slot::<extern "system" fn(object::This) -> u32>(image, 0x10)(image),
                0
            );
        }
    }

    /// The texture both device-loss runs named last (16 texels wide, 4 or 8
    /// high, format 10, layout 0x100, console flags 0x300000; its memory
    /// began 0x300 into a page in run 73 and 0x700 in run 72, and here also
    /// 0xf00, so that a page line runs through it): what the detiling is
    /// given and what it writes are the same read in place and read through
    /// the copy, and the rows it writes end where the staging buffer the
    /// host sizes for it ends (896 bytes for 4 rows of 128 bytes 256 apart,
    /// 1,920 for 8).
    #[test]
    fn the_small_texture_of_runs_72_and_73_detiles_the_same_both_ways() {
        use windows_sys::Win32::System::Memory::*;
        for (height, within) in [(4u32, 0x300usize), (8, 0x700), (4, 0xf00), (8, 0xf00)] {
            // The description as the log printed it, word for word.
            let words: [u32; 14] = [3, 0, 0, 0, 16, 0, height, 65537, 10, 1, 0, 256, 3_145_728, 0];
            // SAFETY: fourteen words are a resource description (56 bytes,
            // plain numbers).
            let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
            let layout = crate::resource::texture_layout(&desc)
                .expect("a console layout")
                .planes[0]
                .layout
                .clone();
            let size = layout.size as usize;
            let (row, pitch) = (16 * 8, 256usize);
            assert!(size >= row * height as usize, "the layout holds every texel");
            // Committed pages and a reserved one after them; the texture
            // begins `within` bytes into the first, as the title placed it.
            // SAFETY: our own reservation; the committed part is written
            // and read in place, by this thread alone.
            unsafe {
                let pages = size.div_ceil(4096) + 1;
                let base = VirtualAlloc(std::ptr::null(), (pages + 1) * 4096, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
                assert!(!base.is_null());
                assert_eq!(VirtualAlloc(base.cast(), pages * 4096, MEM_COMMIT, PAGE_READWRITE), base.cast());
                for i in 0..pages * 4096 {
                    base.add(i).write((i * 7 + i / 251) as u8);
                }
                let memory = base.add(within);
                let in_place = std::slice::from_raw_parts(memory, size);
                // The bytes: the same in place and through the copy.
                with_title(memory, size, |copied| assert_eq!(copied, in_place));
                // What the detiling writes: the same, and inside the bytes
                // the host's staging buffer for this texture has.
                let staged = (height as usize - 1) * pitch + row;
                assert_eq!(staged, if height == 4 { 896 } else { 1920 });
                let detile = |tiled: &[u8]| {
                    // The slice `Staging::detile` hands over: whole rows.
                    let mut linear = vec![0xeeu8; pitch * height as usize];
                    layout
                        .detile(tiled, 0, 0, &mut linear, pitch, pitch * height as usize)
                        .expect("detiled");
                    linear
                };
                let direct = detile(in_place);
                let through = with_title(memory, size, |copied| detile(copied));
                assert_eq!(direct, through);
                for y in 0..height as usize {
                    assert!(direct[y * pitch..][..row].iter().any(|&b| b != 0xee), "row {y} written");
                    assert!(direct[y * pitch + row..(y + 1) * pitch].iter().all(|&b| b == 0xee), "nothing past row {y}");
                }
                // A buffer of exactly the staged bytes is enough.
                let mut exact = vec![0u8; staged];
                layout
                    .detile(in_place, 0, 0, &mut exact, pitch, pitch * height as usize)
                    .expect("the staging buffer's bytes are enough");
                assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
            }
        }
    }
}

#[cfg(test)]
mod block_tests {
    use super::{BLOCK, BLOCKS, GRAIN, trim_pools, upload_memory};

    /// Staging buffers are ranges of one block, page aligned, zeroed when a
    /// range is used again; a block that holds nothing goes when its idle
    /// time is over.
    #[test]
    fn staging_ranges_come_from_blocks_zeroed_and_go_back() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let (a_buffer, a_data, a_at, a_pooled, a) = upload_memory(host, 100_000).expect("staging");
        let (b_buffer, _, b_at, _, b) = upload_memory(host, 3 * GRAIN).expect("staging");
        let (Some(a), Some(b)) = (a, b) else {
            eprintln!("staging blocks are off; skipping");
            return;
        };
        assert_eq!(a_pooled, 0);
        assert_eq!(a_buffer, b_buffer, "one block");
        assert_eq!(a_at % GRAIN, 0);
        assert_eq!(b_at % GRAIN, 0);
        assert!(
            b_at >= a_at + 100_000 || b_at + 3 * GRAIN <= a_at,
            "the ranges do not overlap"
        );
        // SAFETY: the range of `a` in the block's mapping.
        unsafe { std::ptr::write_bytes(a_data.add(a_at as usize), 0x5a, 100_000) };
        let at = a.offset;
        drop(a);
        // The same range again, all zeros.
        let (_, data, again, _, c) = upload_memory(host, 100_000).expect("staging");
        assert_eq!(again, at, "the hole is used again");
        // SAFETY: the range of `c` in the block's mapping.
        let bytes = unsafe { std::slice::from_raw_parts(data.add(again as usize), 100_000) };
        assert!(
            bytes.iter().all(|&x| x == 0),
            "a range used again reads as zeros"
        );
        // Too big for a block: a buffer of its own.
        let (_, _, base, _, own) = upload_memory(host, BLOCK / 2).expect("staging");
        assert!(own.is_none() && base == 0);
        drop((b, c));
        trim_pools(0, 0);
        assert!(
            BLOCKS
                .lock()
                .unwrap()
                .iter()
                .flatten()
                .all(|b| !b.ranges.whole()),
            "empty blocks went"
        );
    }
}

#[cfg(test)]
mod detile_tests {
    use super::{BAND_BYTES, detile_wide};
    use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};

    /// Big subresources detiled in bands on several threads, and small ones
    /// whole, give the bytes of one whole detile, rows and padding alike,
    /// in every tiling mode the layout can address, with 16-byte block
    /// compressed elements, odd sizes and the mips of a big texture.
    #[test]
    fn a_texture_detiled_in_bands_is_the_one_detiled_whole() {
        let desc = |w, h, mips, bpe, block, swizzle| Desc {
            dimension: Dimension::Tex2d,
            width: w,
            height: h,
            depth: 1,
            mips,
            element_bytes: bpe,
            block,
            samples: 1,
            swizzle,
        };
        let mut modes = 0;
        for swizzle in (0..32).map(Swizzle).filter(|s| s.block_log2().is_some()) {
            let cases = [
                desc(2048, 1500, 3, 4, 1, swizzle),
                desc(1024, 1024, 2, 16, 4, swizzle),
                desc(1030, 2051, 1, 8, 1, swizzle),
                desc(256, 256, 1, 4, 1, swizzle),
            ];
            let mut banded_here = false;
            for d in cases {
                let Ok(l) = Layout::new(&d) else {
                    continue;
                };
                if !l.addressable() {
                    continue;
                }
                let tiled: Vec<u8> = (0..l.size as usize).map(|i| (i * 7 + i / 4093) as u8).collect();
                for mip in 0..d.mips {
                    let m = &l.mips[mip as usize];
                    let pitch = (m.width as usize * d.element_bytes as usize).next_multiple_of(256);
                    let rows = m.height;
                    let size = pitch * rows as usize;
                    banded_here |= size >= 2 * BAND_BYTES;
                    let mut whole = vec![0xA5u8; size];
                    l.detile(&tiled, mip, 0, &mut whole, pitch, size).unwrap();
                    let mut banded = vec![0xA5u8; size];
                    detile_wide(&l, &tiled, mip, 0, &mut banded, pitch, rows).unwrap();
                    assert!(banded == whole, "{d:?} mip {mip} ({} bands)", size.div_ceil(BAND_BYTES));
                }
            }
            if banded_here {
                modes += 1;
            }
        }
        // linear, 256 B, 4 KiB and 64 KiB, standard and display
        assert_eq!(modes, 7, "modes detiled in bands");
    }

    /// Many big detiles at once from several threads (as the kept threads
    /// refresh textures together, `VICTORY_WIDE_TEXTURES=1`): each comes
    /// out whole, and none waits for another's bands.
    #[test]
    fn bands_of_several_textures_at_once_do_not_mix() {
        let d = Desc {
            dimension: Dimension::Tex2d,
            width: 2048,
            height: 2048,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S,
        };
        let l = Layout::new(&d).unwrap();
        let pitch = 2048 * 4;
        let size = pitch * 2048;
        std::thread::scope(|s| {
            for t in 0..6u8 {
                let l = &l;
                s.spawn(move || {
                    let tiled: Vec<u8> = (0..l.size as usize).map(|i| (i as u8) ^ t).collect();
                    let mut whole = vec![0u8; size];
                    l.detile(&tiled, 0, 0, &mut whole, pitch, size).unwrap();
                    for _ in 0..4 {
                        let mut banded = vec![0u8; size];
                        detile_wide(l, &tiled, 0, 0, &mut banded, pitch, 2048).unwrap();
                        assert!(banded == whole, "thread {t}");
                    }
                });
            }
        });
    }
}
