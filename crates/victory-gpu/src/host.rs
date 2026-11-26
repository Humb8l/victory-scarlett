//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap
//! (unless `VICTORY_LATE_WAIT=1`, see [`late`]);
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
    /// The adapter the device is on, for its memory budget
    /// ([`Host::video_memory`]); `None` if the system does not name it.
    adapter: Option<IDXGIAdapter3>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Host objects the title let go of while a submission may still use them
/// (a texture its descriptors name, a descriptor slot's old resource), by
/// the submission count they wait for. The title's descriptor writes and
/// releases reach the host at once, from its own threads, while the host
/// runs a submission: freeing there is a GPU page fault on the freed
/// texture (a device hang).
static RETIRED: Mutex<Vec<(u64, Retired)>> = Mutex::new(Vec::new());
/// Submissions started and finished (waited for).
static STARTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FINISHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A retired object, dropped on whatever thread finishes the submission.
struct Retired(#[allow(dead_code)] Box<dyn std::any::Any>);
// SAFETY: what is retired is D3D12 objects (free-threaded) and the
// reference-counted wrappers holding them, dropped once.
unsafe impl Send for Retired {}

/// (P3) A number no host list recording had before ([`HostList::serial`]).
fn next_serial() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Milliseconds since the first call.
pub(crate) fn now_ms() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// This moment, for telling later how long ago it was ([`long_ago`]): the
/// frames begun so far (high half, `crate::frame::current`) and tenths of a
/// second since the start. Host copies of title memory, staging buffers and
/// host textures note when they were last used with it.
pub(crate) fn moment() -> u64 {
    (crate::frame::current() << 32) | ((moment_ms() / 100) & 0xffff_ffff)
}

/// (P3, recording cost) Whether the per-draw bookkeeping takes its cheap
/// forms (`VICTORY_CHEAP_DRAWS`, default 1; `=0` as before): [`moment`] read
/// from the system's tick count (the precise clock it read cost about 5 % of
/// the recording threads' time in our calls: every texture a draw names
/// notes its use with it; tenths of a second need no more than the tick's
/// 16 ms), and a draw's root views, vertex and index buffers not looked up
/// and brought up to date again while nothing that could change the answer
/// happened (`crate::draw`).
pub(crate) fn cheap_draws() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CHEAP_DRAWS").is_none_or(|v| v != "0"))
}

/// Milliseconds since the start for [`moment`] ([`cheap_draws`]).
fn moment_ms() -> u64 {
    if !cheap_draws() {
        return now_ms();
    }
    static EPOCH: OnceLock<u64> = OnceLock::new();
    // SAFETY: a plain query.
    let now = unsafe { windows_sys::Win32::System::SystemInformation::GetTickCount64() };
    now.saturating_sub(*EPOCH.get_or_init(|| now))
}

/// Whether `moment` lies at least `ms` milliseconds and `frames` frames
/// back. Both: at seconds per frame, what every frame uses is not touched
/// for seconds, and at many frames per second not for many frames. (A title
/// that never waits for frames has no frame count: nothing is then idle.)
pub(crate) fn long_ago(moment: u64, ms: u64, frames: u64) -> bool {
    let now = self::moment();
    let tenths = (now & 0xffff_ffff).saturating_sub(moment & 0xffff_ffff);
    (now >> 32).saturating_sub(moment >> 32) >= frames && tenths * 100 >= ms
}

/// Available physical RAM on the host in bytes, from `GlobalMemoryStatusEx`.
pub(crate) fn available_physical_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    let ok = unsafe { GlobalMemoryStatusEx(&mut m) } != 0;
    ok.then_some(m.ullAvailPhys)
}

/// Effective idle timeout (in milliseconds) for VRAM-backed host textures,
/// scaled down under VRAM pressure. Never returns less than 1,000 ms so
/// handles not yet recorded into a list or slot remain protected.
pub(crate) fn vram_idle_ms(video_memory: Option<[u64; 4]>) -> u64 {
    const NORMAL_MS: u64 = 5_000;
    const MODERATE_MS: u64 = 2_500;
    const HIGH_MS: u64 = 1_000;
    let Some([budget, used, ..]) = video_memory else {
        return NORMAL_MS;
    };
    if budget == 0 {
        return NORMAL_MS;
    }
    let used_pct = (u128::from(used) * 100) / u128::from(budget);
    if used_pct >= 85 {
        HIGH_MS
    } else if used_pct >= 75 {
        MODERATE_MS
    } else {
        NORMAL_MS
    }
}

/// Effective idle timeout (in milliseconds) for system-RAM caches (staging
/// buffers, loose shadows, and the mapped pipeline library), scaled down
/// when free physical RAM is low. Never returns less than 2,000 ms.
pub(crate) fn ram_idle_ms(avail_phys: Option<u64>) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    const NORMAL_MS: u64 = 10_000;
    const MODERATE_MS: u64 = 5_000;
    const HIGH_MS: u64 = 2_000;
    let Some(avail) = avail_phys else {
        return NORMAL_MS;
    };
    if avail < 2 * GIB {
        HIGH_MS
    } else if avail < 4 * GIB {
        MODERATE_MS
    } else {
        NORMAL_MS
    }
}

/// How many references are held on a host object (its COM count): ours
/// all, since the host's views and commands hold none. A buffer or texture
/// with one is held by its owner alone: no descriptor slot keeps it
/// (`HostHeap::kept`) and no list (`HostList::keep`).
pub(crate) fn references(object: &impl Interface) -> u32 {
    type Method = unsafe extern "system" fn(*mut std::ffi::c_void) -> u32;
    let raw = object.as_raw();
    // SAFETY: a live COM object, whose first three methods are those of
    // `IUnknown` (query, add a reference, release one); the pair leaves the
    // count as it was, and the release returns it. The first is not called.
    unsafe {
        let methods = *raw.cast::<*const [Method; 3]>();
        ((*methods)[1])(raw);
        ((*methods)[2])(raw)
    }
}

/// Once a second, before a submission starts: lets go of what nothing
/// names any more (copies of windows of title memory, staging buffers,
/// host copies of textures the title fills).
/// `VICTORY_SWEEP=0` keeps everything, as before.
/// When [`sweep`] last did its work (ms).
static SWEEP_LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// (P6) Whether [`sweep`] would do its work now.
fn sweep_due() -> bool {
    now_ms() >= SWEEP_LAST.load(Ordering::Relaxed) + 1000
}

fn sweep() {
    // The in-place probe and census, if asked for (`VICTORY_INPLACE_PROBE`).
    crate::inplace::tick();
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_SWEEP").is_none_or(|v| v != "0")) {
        return;
    }
    let now = now_ms();
    let last = SWEEP_LAST.load(Ordering::Relaxed);
    if now < last + 1000
        || SWEEP_LAST
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    // (P6) Each part timed: the sweep is a spike on the submitting thread
    // once a second (the prepare line's "sweep").
    let mut t = std::time::Instant::now();
    let mut part = |i: usize| {
        SWEEP_PARTS[i].fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        t = std::time::Instant::now();
    };
    crate::pipeline::trim_library();
    part(0);
    crate::shadow::let_go_idle();
    part(1);
    crate::texture::let_go_idle();
    part(2);
    crate::resource::let_go_idle();
    part(3);
    trim_scratch();
    part(4);
    crate::list::let_go_idle_lists();
    part(5);
    SWEEP_PARTS[6].fetch_add(1, Ordering::Relaxed);
}

/// (P6) Since the last frame line: microseconds of the sweep's parts
/// (pipeline library trim, host copies, texture stagings, resources and
/// texture heaps, scratch, idle lists) and sweeps made.
static SWEEP_PARTS: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];

/// Drops `object` once every submission running now has finished: at once
/// if none is.
pub(crate) fn retire<T: 'static>(object: T) {
    let started = STARTED.load(Ordering::Acquire);
    if started <= FINISHED.load(Ordering::Acquire) {
        RETIREMENTS[0].fetch_add(1, Ordering::Relaxed);
        drop(object);
        return;
    }
    RETIREMENTS[1].fetch_add(1, Ordering::Relaxed);
    RETIRED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((started, Retired(Box::new(object))));
}

/// Drops what waited for submissions up to `finished`.
fn bury(finished: u64) {
    let done: Vec<(u64, Retired)> = {
        let mut retired = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
        if retired.is_empty() {
            return;
        }
        let (done, keep) = std::mem::take(&mut *retired)
            .into_iter()
            .partition(|(at, _)| *at <= finished);
        *retired = keep;
        done
    };
    drop(done);
}

/// Objects [`retire`] dropped at once (no submission was running) and
/// kept until a running submission finished, since the start.
static RETIREMENTS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// What the layer did to the host's objects lately, newest last (sweeps
/// that let host copies go, pipeline library calls, saves and trims): said
/// once when the host device is removed ([`log_removal_context`]), with
/// what the submitting thread and the pipeline workers were doing at that
/// moment. A device the driver removes "internally" names no command, and
/// the pass that happens to notice first names only the view it was at:
/// the three such losses of runs 72, 73 and 82 each came within 5 ms of a
/// pipeline library call that no line of the log tied to them.
/// `VICTORY_TRAIL=0`: nothing is kept and nothing is said.
static TRAIL: Mutex<std::collections::VecDeque<(u64, u32, String)>> = Mutex::new(std::collections::VecDeque::new());
/// The entries kept.
const TRAIL_MOST: usize = 96;

fn trail_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TRAIL").is_none_or(|v| v != "0"))
}

/// Microseconds since the first call.
pub(crate) fn now_us() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64
}

/// The calling thread, as the log's other lines name threads.
pub(crate) fn thread_id() -> u32 {
    // SAFETY: a plain query.
    unsafe { windows::Win32::System::Threading::GetCurrentThreadId() }
}

/// Notes something done to the host's objects ([`TRAIL`]); `what` is only
/// run when the trail is kept.
pub(crate) fn trail(what: impl FnOnce() -> String) {
    if !trail_on() {
        return;
    }
    let entry = (now_us(), thread_id(), what());
    // Sweeps also go to the log as they happen: the trail is only said at
    // a loss, and the driver's errors in the system's event log (run 224's
    // first fell inside a sweep of 4,711 host textures) need the sweeps of
    // clean runs beside them.
    if entry.2.starts_with("sweep:") {
        tracing::info!("{}", entry.2);
    }
    let mut trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    if trail.len() >= TRAIL_MOST {
        trail.pop_front();
    }
    trail.push_back(entry);
}

/// What the submitting thread is doing (0 nothing: between submissions,
/// 1 preparing one, 2 waiting for the card to run it, 3 copying back what
/// it wrote), the lists of that submission, and since when ([`now_us`]).
static SUBMITTING: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn submitting(phase: u64, lists: usize) {
    SUBMITTING[1].store(lists as u64, Ordering::Relaxed);
    SUBMITTING[2].store(now_us(), Ordering::Relaxed);
    SUBMITTING[0].store(phase, Ordering::Relaxed);
}

/// What was going on when the host device went: whether a submission was
/// on the card, the pipeline library's state and the calls running in it,
/// and the trail.
pub(crate) fn removal_context() -> String {
    use std::fmt::Write;
    let now = now_us();
    let ms = |at: u64| now.saturating_sub(at) as f64 / 1000.0;
    let phase = match SUBMITTING[0].load(Ordering::Relaxed) {
        0 => "between submissions (nothing of ours was running on the card)",
        1 => "preparing a submission (nothing of ours was running on the card)",
        2 => "waiting for the card to run a submission",
        _ => "copying back what a submission wrote (nothing of ours was running on the card)",
    };
    let mut text = format!(
        "the submitting thread was {phase}, of {} lists, for {:.1} ms; {}; objects let go at once {} and after a running submission {}; submissions started {} finished {}",
        SUBMITTING[1].load(Ordering::Relaxed),
        ms(SUBMITTING[2].load(Ordering::Relaxed)),
        crate::pipeline::library_removal_note(),
        RETIREMENTS[0].load(Ordering::Relaxed),
        RETIREMENTS[1].load(Ordering::Relaxed),
        STARTED.load(Ordering::Relaxed),
        FINISHED.load(Ordering::Relaxed),
    );
    let trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    let _ = write!(text, "; the last {} things done, newest last:", trail.len());
    for (at, thread, what) in trail.iter() {
        let _ = write!(text, "\n  {:.1} ms before, thread {thread}: {what}", ms(*at));
    }
    text
}

/// Says [`removal_context`] in the log, once: called by whoever finds the
/// host device removed.
pub(crate) fn log_removal_context() {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !trail_on() || SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    tracing::error!("when the host device went: {}", removal_context());
}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Adds to [`SUBMISSION_US`] and to the frame log's totals.
fn submission_us(i: usize, us: u64) {
    SUBMISSION_US[i].fetch_add(us, Ordering::Relaxed);
    crate::framelog::add([crate::framelog::T::PrepareUs, crate::framelog::T::CardWaitUs][i], us);
    if i == 0 {
        crate::framelog::add(crate::framelog::T::Submissions, 1);
    }
}

/// Adds to [`PREPARING`]'s copies (time, bytes) and to the frame log's totals.
/// (P6) Whether a submission's independent preparing pieces run on the wide
/// workers (`VICTORY_PREPARE_WIDE`, default 1; 0: as before):
/// - CPU fills of a list (title memory into slices of our upload memory) are
///   gathered and done together, several at a time, before the next step
///   that writes title memory or runs work (a title copy, a step that runs
///   work, a shader-write snapshot) and at the list's end. What lies between
///   a fill and that point writes neither title memory nor upload memory
///   (refreshes of host copies, texture checks), so each fill reads the
///   bytes it read before; their destinations are disjoint slices (checked:
///   one that overlaps another keeps them in order on this thread).
/// - refreshes of host copies go wide from two copies on (four before): each
///   writes its own copy and notes and reads title memory
///   (`wide_refresh`).
/// - texture checks go wide as with `VICTORY_WIDE_TEXTURES=1`: due at the
///   next flush, which comes before every step that writes title memory and
///   before the lists are sent; each writes its own staging buffer.
/// Order of completion does not matter: everything is done before the lists
/// go to the queue.
fn prepare_wide() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PREPARE_WIDE").is_none_or(|v| v != "0"))
}

/// (P6) Since the last frame line, microseconds of a submission's preparing:
/// 0 wall (as "preparing"), 1 of it in pieces run wide, 2 items run wide,
/// then the serial pieces: 3 write-watch asking once, 4 stale under flights,
/// 5 title copies, 6 steps that run work, 7 shader-write snapshots (with
/// their copy's refresh), 8 copies in card memory (`card_before`), 9 sweep,
/// 10 slots waiting for textures, 11 snapshots taken, 12 fills done here,
/// 13 refreshes done here, 14 texture checks done here, 15 the heap sync
/// waited for.
pub(crate) static PREP: [std::sync::atomic::AtomicU64; 16] = [const { std::sync::atomic::AtomicU64::new(0) }; 16];

fn prep(i: usize, since: std::time::Instant) {
    PREP[i].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// (P6) The per-second line of [`PREP`].
fn prep_line(frames: f64) {
    let v: Vec<f64> = PREP.iter().map(|c| c.swap(0, Ordering::Relaxed) as f64).collect();
    let ms = |i: usize| v[i] / 1000.0 / frames;
    let names = [
        (3, "write-watch asking"),
        (4, "stale under flights"),
        (5, "title copies"),
        (6, "steps that run work"),
        (7, "shader-write snapshots"),
        (8, "copies in card memory"),
        (9, "sweep"),
        (10, "slots waiting for textures"),
        (11, "snapshots"),
        (12, "fills here"),
        (13, "refreshes here"),
        (14, "texture checks here"),
        (15, "heap sync waited for"),
    ];
    let mut serial: Vec<(f64, &str)> = names.iter().map(|&(i, n)| (ms(i), n)).collect();
    serial.sort_by(|a, b| b.0.total_cmp(&a.0));
    let biggest: Vec<String> = serial.iter().take(6).map(|(m, n)| format!("{n} {m:.2}")).collect();
    let wall = ms(0);
    let sweeps = SWEEP_PARTS[6].swap(0, Ordering::Relaxed) as f64;
    let sweep_ms: Vec<f64> = SWEEP_PARTS[..6].iter().map(|c| c.swap(0, Ordering::Relaxed) as f64 / 1000.0 / sweeps.max(1.0)).collect();
    tracing::info!(
        "prepare: {:.1} ms wall (serial part {:.1}, parallel part {:.1} on {} workers, items {:.0}; biggest serial pieces: {}) \
         (VICTORY_PREPARE_WIDE={}); sweeps {:.0} in the line, ms each: library {:.1}, host copies {:.1}, stagings {:.1}, resources {:.1}, scratch {:.1},          lists {:.1}; clock reads spared {:.0} (VICTORY_STEP_CLOCK_LEAN={}); pages held in part left as taken whole {:.0}          (VICTORY_RANGED={})",
        wall,
        (wall - ms(1)).max(0.0),
        ms(1),
        crate::shadow::wide_threads() + 1,
        v[2] / frames,
        biggest.join(", "),
        u8::from(prepare_wide()),
        sweeps,
        sweep_ms[0],
        sweep_ms[1],
        sweep_ms[2],
        sweep_ms[3],
        sweep_ms[4],
        sweep_ms[5],
        STEP_CLOCKS_SAVED.swap(0, Ordering::Relaxed) as f64 / frames,
        u8::from(step_clock_lean()),
        crate::shadow::ranged_left() as f64 / frames,
        u8::from(crate::shadow::ranged_on())
    );
}

/// (neptune-ve 25m) Whether a refresh step of a submission is handed to its
/// batch without reading the clock (`VICTORY_STEP_CLOCK_LEAN`, default 1:
/// the time was never counted for it, the step `continue`s before; 0: read
/// for every step, as before).
fn step_clock_lean() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_STEP_CLOCK_LEAN").is_none_or(|v| v != "0"))
}

/// Since the last frame line: steps handed on without a clock read.
static STEP_CLOCKS_SAVED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn preparing(i: usize, n: u64) {
    PREPARING[i].fetch_add(n, Ordering::Relaxed);
    crate::framelog::add([crate::framelog::T::CopyUs, crate::framelog::T::CopyBytes][i], n);
}

/// Adds to [`COPY_BACK`] and to the frame log's totals.
fn copy_back(i: usize, n: u64) {
    COPY_BACK[i].fetch_add(n, Ordering::Relaxed);
    crate::framelog::add([crate::framelog::T::CopyBackUs, crate::framelog::T::CopyBackBytes][i], n);
}

/// Adds to [`CARD_CLOCK_US`] and, for the card's idle times, to the frame
/// log's totals.
fn card_clock_us(i: usize, us: u64) {
    CARD_CLOCK_US[i].fetch_add(us, Ordering::Relaxed);
    match i {
        0 => crate::framelog::add(crate::framelog::T::CardIdleUs, us),
        1 => crate::framelog::add(crate::framelog::T::CardIdleBehindUs, us),
        _ => {}
    }
}

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// The card's clock, for the frame line of every run, in microseconds since
/// the last one: how long the card had nothing of ours queued; how much of
/// that while the thread that holds the submission lock was working (the
/// card idle behind us: preparing and copying back are done with the card
/// standing); and how long that thread stood waiting for the card.
///
/// Every send of lists to the queue is followed by a signal on a fence of
/// the clock's own ([`Host::card_sent`]). A thread of the clock waits for
/// the newest signal and notes the moment the card reached it with nothing
/// newer sent; the next send ends that idle stretch.
static CARD_CLOCK_US: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Of the thread that holds the submission lock: the microseconds it has
/// worked so far (preparing, copying back; not its waits for the card), and
/// since when it works now ([`now_us`] plus one; 0: it does not).
static WORKING: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The thread that holds the submission lock starts (`on`) or stops
/// working; whether it worked until now.
fn working(on: bool) -> bool {
    let now = now_us() + 1;
    let since = WORKING[1].swap(if on { now } else { 0 }, Ordering::Relaxed);
    if since != 0 {
        WORKING[0].fetch_add(now.saturating_sub(since), Ordering::Relaxed);
    }
    since != 0
}

/// (Card kinds job) Of the thread that holds the submission lock: the
/// microseconds it has stood waiting for flights to land so far
/// ([`relay_wait`]), and since when it waits now (0: it does not); and what
/// that said when the card's current idle stretch began.
static LANDING: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn landing(on: bool) {
    let now = now_us() + 1;
    let since = LANDING[1].swap(if on { now } else { 0 }, Ordering::Relaxed);
    if since != 0 {
        LANDING[0].fetch_add(now.saturating_sub(since), Ordering::Relaxed);
    }
}

fn landing_waited() -> u64 {
    let since = LANDING[1].load(Ordering::Relaxed);
    let now = if since != 0 { (now_us() + 1).saturating_sub(since) } else { 0 };
    LANDING[0].load(Ordering::Relaxed) + now
}

/// (Card kinds job) At a present: how many frames have submissions that
/// have not all landed, this one included.
pub(crate) fn frames_in_air() -> u64 {
    static PRESENTS: Mutex<std::collections::VecDeque<u64>> = Mutex::new(std::collections::VecDeque::new());
    let started = STARTED.load(Ordering::Acquire);
    let finished = FINISHED.load(Ordering::Acquire);
    let mut p = PRESENTS.lock().unwrap_or_else(|e| e.into_inner());
    if p.back() != Some(&started) {
        p.push_back(started);
    }
    while p.front().is_some_and(|&s| s <= finished) || p.len() > 64 {
        p.pop_front();
    }
    p.len() as u64
}

/// [`WORKING`] as of now: microseconds worked so far.
fn worked() -> u64 {
    let since = WORKING[1].load(Ordering::Relaxed);
    let now = if since != 0 { (now_us() + 1).saturating_sub(since) } else { 0 };
    WORKING[0].load(Ordering::Relaxed) + now
}

/// The thread that holds the submission lock stood waiting for the card
/// from `since` until now ([`CARD_CLOCK_US`]).
fn stood(since: std::time::Instant) {
    card_clock_us(2, since.elapsed().as_micros() as u64);
}

/// What [`CARD_CLOCK_US`] is measured with.
struct CardClock {
    fence: ID3D12Fence,
    /// Signals sent so far; and, while the card has reached the last of
    /// them, since when and what [`worked`] said then.
    state: Mutex<(u64, Option<(std::time::Instant, u64)>)>,
    sent: std::sync::Condvar,
}

// SAFETY: a fence (free-threaded) behind a lock.
unsafe impl Send for CardClock {}
// SAFETY: as above.
unsafe impl Sync for CardClock {}

/// The clock's thread: waits for the newest signal sent and notes when the
/// card got there with nothing newer behind it.
fn card_clock_watch(clock: &'static CardClock) {
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        return;
    };
    let mut seen = 0;
    loop {
        let target = {
            let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
            while state.0 == seen {
                state = clock.sent.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            state.0
        };
        // SAFETY: the clock's fence and this thread's event.
        unsafe {
            if clock.fence.SetEventOnCompletion(target, event).is_ok() {
                WaitForSingleObject(event, INFINITE);
            }
        }
        seen = target;
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.0 == target && state.1.is_none() {
            state.1 = Some((std::time::Instant::now(), worked()));
            LANDING[2].store(landing_waited(), Ordering::Relaxed);
        }
    }
}

/// For the frame-rate line: [`CARD_CLOCK_US`] per frame, `ms` being the
/// frame's time.
/// (trustno1 5f86fa3) Keeping the card's clocks up between submissions.
pub(crate) mod awake;
pub(crate) mod aside;

fn card_clock_line(frames: f64, ms: f64) {
    if awake::on() {
        let [spins, waited] = [0, 1].map(|i| awake::SPINS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        tracing::info!(
            "card kept awake (VICTORY_CARD_AWAKE): {spins:.1} spins per frame (about 2 ms of one or two wave slots each at full clocks),              its thread waited {:.1} ms a frame for them to run",
            waited / 1000.0
        );
    }
    let [idle, behind, stood] = [0, 1, 2].map(|i| CARD_CLOCK_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
    tracing::info!(
        "the card's clock per frame: nothing of ours queued {idle:.1} ms (the card about {:.0}% busy by this count), \
         {behind:.1} ms of that while the submitting thread was working (the card idle behind us), the rest while the title \
         had not submitted; the submitting thread stood waiting for the card {stood:.1} ms",
        if ms > 0.0 { (100.0 * (1.0 - idle / ms)).clamp(0.0, 100.0) } else { 0.0 }
    );
}

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Host buffers made since the start, as (count, bytes): in the upload heap
/// (staging, list scratch), shared with the processor (host copies of title
/// memory) and the graphics card's own (indirect arguments, twins' copies).
/// For the memory line: against what is alive, they show what was let go.
pub(crate) static MADE: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// What of that is alive now, for the memory line (`MADE` counts what was
/// ever made: 42,607 upload buffers, 77.9 GB, in five minutes of run 60,
/// and nothing said how many were alive): host lists; the lists' scratch
/// chunks and their bytes; of those, the chunks no list holds (kept for the
/// next list, [`Spare`]) and their bytes; chunks taken from those kept
/// instead of made, since the start; and the lists' other upload buffers
/// (texture data copied from buffers, scratch too big for a chunk) and
/// their bytes.
pub(crate) static ALIVE: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// About host lists, for the memory line: lists made by the host since the
/// start; preludes made (a list's second allocator and list, made when it
/// first needs one, [`lazy_prelude`]); and, in diagnostics builds, the
/// bytes the process's private memory grew by while lists were being made
/// (what a list costs: host objects with the driver's command memory; 631
/// lists were alive in run 65, each with a prelude it may never use).
/// Then the lists alive that were never reset since they were made: the
/// host sides of title lists the title made and has not used (a host list
/// cost 0.68 MB of private memory in run 69: 431 MB for the 632 the title
/// makes at its start).
pub(crate) static LISTS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The process's private bytes (diagnostics: around the making of lists).
#[cfg(feature = "diagnostics")]
pub(crate) fn private_bytes() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.PagefileUsage as u64
}

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
            log_removal_context();
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    // Tests have no log: the reason goes to their output.
    #[cfg(test)]
    eprintln!("host GPU failure: {text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    crate::drvwatch::start();
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug =
            cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DEBUG").is_some();
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if std::env::var_os("VICTORY_D3D12_GBV").is_some()
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_WARP").is_some() {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        // `VICTORY_D3D12_BACKGROUND=0`: the runtime's background work (the
        // driver compiling shaders again on the "D3D Background Thread"s,
        // for minutes after pipelines were made or loaded) is switched off.
        // An experiment: while those threads ran, the same shot took
        // 1,000-1,200 ms a frame, and 830-950 ms from the moment they
        // stopped (run 54, 08:32:01 to 08:32:06). Left as it is by default:
        // whether the card is slower with the shaders it then keeps is not
        // known.
        if std::env::var_os("VICTORY_D3D12_BACKGROUND").is_some_and(|v| v == "0")
            && let Ok(device6) = device.cast::<ID3D12Device6>()
        {
            let set = device6.SetBackgroundProcessingMode(
                D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK,
                D3D12_MEASUREMENTS_ACTION_KEEP_ALL,
                None,
                None,
            );
            tracing::info!(ok = set.is_ok(), "the host's background shader work is switched off (VICTORY_D3D12_BACKGROUND=0)");
        }
        // the widest wave the GPU runs: 32 (NVIDIA) gets console wave64 compute programs whose
        // cross-lane code stays in 32-lane halves translated for waves of 32
        let mut o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        if device
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, (&raw mut o1).cast(), size_of_val(&o1) as u32)
            .is_ok()
        {
            tracing::info!(min = o1.WaveLaneCountMin, max = o1.WaveLaneCountMax, "host wave lanes");
            victory_shader::emit::set_host_max_wave(o1.WaveLaneCountMax);
        }
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let adapter = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
            .ok();
        tracing::info!("host D3D12 device ready");
        Ok(Host {
            device,
            queue,
            fence,
            submitted: Mutex::new(0),
            event,
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(None),
            adapter,
        })
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
pub(crate) fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sparse_copies_preserve_islands_and_observe_remaps() {
        use windows_sys::Win32::System::Memory::*;
        const N: usize = 5 << 16;
        // SAFETY: we reserve a private region, commit selected pages inside it,
        // and only directly access the pages while they are writable.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let mut expected = vec![0; N];
            for (offset, value) in [(0, 9), (2 << 16, 37), (N - 4096, 85)] {
                let at = base.add(offset);
                assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                at.write_bytes(value, 4096);
                expected[offset..offset + 4096].fill(value);
            }
            for remap in [false, true] {
                if remap {
                    let mut old = 0;
                    assert_ne!(VirtualProtect(base.cast(), 4096, PAGE_NOACCESS, &mut old), 0);
                    expected[..4096].fill(0);
                    let at = base.add(1 << 16);
                    assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                    at.write_bytes(0xc7, 4096);
                    expected[1 << 16..(1 << 16) + 4096].fill(0xc7);
                }
                let mut out = vec![0xee; N];
                super::copy_mapped(out.as_mut_ptr().add(27), base.add(27), N - 54, false);
                assert_eq!(out[27..N - 27], expected[27..N - 27]);
                assert!(out[..27].iter().chain(&out[N - 27..]).all(|&v| v == 0xee));
            }
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    use super::*;

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// (Neptune) Counts what changes a texture's state or a buffer's bytes in this
    /// list, and its resets: a draw's table that was made ready when the count stood
    /// where it stands now needs nothing again (`crate::draw`, `VICTORY_READY_TABLES`).
    pub(crate) changes: std::sync::atomic::AtomicU64,
    /// (P3, recording cost) This recording's own number: new for every host list and
    /// at every reset, so that what a title list remembers it gave its host list
    /// (`crate::draw` `Issued`, `VICTORY_BIND_FILTER`) is never taken for another's
    /// (a title list's host list lives at one address through all its recordings).
    serial: std::sync::atomic::AtomicU64,
    /// (P3) Moves whenever the list's set of ranges refreshed so far is emptied (a title
    /// copy or work that may write title memory recorded, a reset): while it stands, a
    /// refresh asked again of a range asked before adds nothing ([`HostList::defer_refresh`]).
    refreshes: std::sync::atomic::AtomicU64,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// What the list does with depth textures that have a colour twin, by
    /// twin ([`TwinUse`]).
    twins: Mutex<std::collections::HashMap<usize, TwinUse>>,
    /// Alias copies left to the prelude (`crate::texture::alias_once`).
    aliases: Mutex<Vec<crate::texture::AliasCopy>>,
    /// The host copies the list puts in a state that writes them (keys of
    /// `states`), for [`ALIAS_WRITES`].
    written: Mutex<std::collections::HashSet<usize>>,
    /// The list that runs those transitions just before this one, made
    /// when the list first has such transitions ([`lazy_prelude`]).
    prelude: OnceLock<Option<Prelude>>,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted, and after
    /// it what the clamp notes about mesh records. In memory the CPU maps.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// The same words in the card's own memory ([`card_budget`]), which the
    /// clamp uses instead when there is one: zeroed in the list before its
    /// first clamp, its notes copied to `budget` when the list closes.
    card_budget: OnceLock<Option<ID3D12Resource>>,
    /// Device-local scratch buffer for clamped indirect arguments (`indirect`),
    /// grown to the largest indirect execution recorded on the list.
    indirect_args: Mutex<Option<(ID3D12Resource, u64)>>,
    /// With [`fill_on_card`]: a device-local buffer constant buffer views of
    /// the list read, filled by copies on the card ([`HostList::card_constants`]):
    /// the buffer, the bytes used, and whether it is in the constant buffer
    /// state (it decays to the common state when a submission has run).
    card_constants: Mutex<Option<(ID3D12Resource, u64, bool)>>,
    /// Title memory (address, end) this list copied on the card
    /// (`crate::list` `gpu_buffer_copy`), with the host copy written there
    /// (its pointer): later copies and constant fills of the list that
    /// touch it go on the card too, in its order, through that copy.
    card_copied: Mutex<Vec<(usize, usize, usize)>>,
    /// The buffers of `crate::indirect::window_check`, made at its first use.
    window_check: OnceLock<Option<crate::indirect::WindowCheck>>,
    /// Counts the list in [`ALIVE`] while it lives.
    _alive: Counted,
    /// (P6) Refreshes of host copies that a thread of ours is still
    /// finding ([`aside`]): where among the list's steps they belong, the
    /// set of refreshed ranges as it was there, and the work.
    #[allow(clippy::type_complexity)]
    coming: Mutex<Option<(usize, std::collections::HashSet<(usize, usize, usize)>, std::sync::Arc<aside::Coming>)>>,
    /// The pair of timestamps its run is timed with since its last reset
    /// (`crate::cardmeter`), or none.
    meter: AtomicU32,
    /// With `crate::cardmeter::pipelines`: the pair of the pipeline bound
    /// last, open until the next one is bound or the list closes.
    pipeline_meter: AtomicU32,
    /// The records its draws read from slots that its prelude fills
    /// (`crate::indirect::gather`).
    gather: Mutex<Gather>,
    /// (Card kinds job) Its runs of layer commands timed by kind
    /// (`crate::cardmeter::kinds`), and whether one is open now.
    kinds: Mutex<crate::cardmeter::ListKinds>,
    kinds_open: std::sync::atomic::AtomicBool,
    /// (Card kinds job, `VICTORY_CARD_FILLS_BATCH`) Constant fills moved to
    /// the list's prelude ([`HostList::card_constants`]).
    hoisted: Mutex<Hoisted>,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

/// A list's prelude: an allocator and a closed list.
type Prelude = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// What a list gathers for its prelude to clamp (`crate::indirect::gather`).
#[derive(Default)]
struct Gather {
    /// Something recorded in the list may write arguments: no record is
    /// gathered from here on (the ones gathered before are read before
    /// the list runs, which is before that).
    closed: bool,
    /// One set per argument buffer, kind and stride.
    sets: Vec<crate::indirect::GatherSet>,
    /// The budget words the gathered records are charged to.
    budget: Option<ID3D12Resource>,
    /// Scratch buffers of earlier recordings, kept for the next sets.
    spare: Vec<ID3D12Resource>,
}

/// The most argument buffers a list gathers from.
const GATHER_SETS: usize = 8;

/// (Card kinds job) A list's constant fills moved to its prelude
/// ([`fills_batch`], [`HostList::card_constants`]).
#[derive(Default)]
struct Hoisted {
    /// The device-local buffer the prelude fills (kept over resets), and
    /// the bytes of it used.
    buffer: Option<ID3D12Resource>,
    used: u64,
    /// The prelude's copies: source buffer, offset there, offset in
    /// `buffer`, bytes; in the order the fills were recorded.
    copies: Vec<(ID3D12Resource, u64, u64, u64)>,
    /// The same source bytes asked again (source buffer, offset, bytes):
    /// the place they already have.
    seen: std::collections::HashMap<(usize, u64, u64), u64>,
}

/// Whether frames are shown (`VICTORY_PRESENT_SKIP`, default 0: every frame
/// presented as before). 1: the frame is copied to the swap chain's back
/// buffer as always but `Present` is never called (with the displays off
/// Windows throttles a present to ~150 ms; for unattended measurement).
/// N > 1: `Present` every Nth frame with `DXGI_PRESENT_DO_NOT_WAIT`, its
/// result ignored, the other frames not presented. Nothing else of ours
/// waits on the swap chain (no waitable object, no latency handle).
pub(crate) fn present_skip() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| {
        let n = std::env::var("VICTORY_PRESENT_SKIP").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0);
        match n {
            0 => {}
            1 => tracing::info!("presentation: frames are not shown (VICTORY_PRESENT_SKIP=1): for unattended measurement with the display off"),
            n => tracing::info!(
                "presentation: one frame in {n} is shown, without waiting (VICTORY_PRESENT_SKIP={n}, DXGI_PRESENT_DO_NOT_WAIT): for unattended measurement with the display off"
            ),
        }
        n
    })
}

/// (Card kinds job) Whether constant fills on the card are moved to their
/// list's prelude where that provably changes nothing a draw reads
/// (`VICTORY_CARD_FILLS_BATCH`): 0 never; 1 (default) only while nothing
/// recorded in the list so far may write buffer memory (no dispatch, no
/// storing draw: the list still gathers indirect records, the rule the
/// gathered records have had from the start) and no write of the list was
/// declared over the source bytes; 2 also after that, when no write the
/// list declared so far touches the source bytes (the rule of
/// `VICTORY_GATHER_LATE`).
pub(crate) fn fills_batch() -> u8 {
    static ON: OnceLock<u8> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_CARD_FILLS_BATCH").ok().as_deref() {
        Some("0") => 0,
        Some("2") => 2,
        _ => 1,
    })
}

/// Where a constant fill on the card goes ([`fills_batch`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FillPlace {
    /// Into the prelude: nothing before it in the list may change its source.
    Prelude,
    /// In place: a write of the list was declared over its source bytes.
    Written,
    /// In place: a dispatch or a storing draw was recorded before it, and
    /// none declared a write over its source (`=2` would move it).
    Storing,
    /// In place: the switch is off.
    Off,
}

/// The hoisting decision for one fill: `mode` ([`fills_batch`]), whether the
/// list has stopped gathering (something recorded may write buffer memory),
/// and whether a write the list declared so far (shader writes, title
/// copies run on the card) touches the fill's source bytes.
pub(crate) fn fill_place(mode: u8, closed: bool, declared_over_source: bool) -> FillPlace {
    if mode == 0 {
        FillPlace::Off
    } else if declared_over_source {
        FillPlace::Written
    } else if closed && mode < 2 {
        FillPlace::Storing
    } else {
        FillPlace::Prelude
    }
}

/// (Card kinds job) Since the start: constant fills on the card moved to the
/// prelude, of them served by the bytes of an earlier one (same source,
/// nothing written between: one copy), prelude copies joined with the one
/// before (contiguous source and place); fills left in place because a
/// declared write touched the source, because a dispatch or storing draw
/// came first, for other reasons (switch off, no room, no buffer).
pub(crate) static FILLS: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];
pub(crate) const FILL_HOISTED: usize = 0;
pub(crate) const FILL_SHARED: usize = 1;
pub(crate) const FILL_JOINED: usize = 2;
pub(crate) const FILL_WRITTEN: usize = 3;
pub(crate) const FILL_STORING: usize = 4;
pub(crate) const FILL_OTHER: usize = 5;
/// Lists whose prelude took fills.
pub(crate) const FILL_LISTS: usize = 6;

/// Whether a host list's prelude is made when the list first needs one. A
/// prelude runs the texture transitions and twin copies a list needs before
/// it; lists that name no texture never have one, and every host list was
/// made with one all the same: twice the host objects per list, for the
/// title's 630 lists and for every list made anew.
/// `VICTORY_LAZY_PRELUDE=0`: made with the list, as before.
/// The write marks of [`crate::texture::alias_once`], kept at submission:
/// the submissions' numbers, the last one that wrote each host copy (by
/// the copy's address), and the last one that made each alias copy (by
/// source copy, source texture, placed texture).
#[derive(Default)]
struct AliasWrites {
    seq: u64,
    written: std::collections::HashMap<usize, u64>,
    copied: std::collections::HashMap<(usize, usize, usize), u64>,
}

static ALIAS_WRITES: Mutex<Option<AliasWrites>> = Mutex::new(None);

/// Alias copies left to preludes: made, and left out (source unwritten).
pub(crate) static ALIAS_COUNTS: [std::sync::atomic::AtomicU64; 2] = [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

fn lazy_prelude() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAZY_PRELUDE").is_none_or(|v| v != "0"))
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

/// What a list does with a depth texture that has a colour twin
/// ([`crate::resource::Twin`]). Lists are recorded on many threads and run
/// in submission order, so a list cannot know which side an earlier list
/// leaves behind: it notes which sides it uses as it finds them, and its
/// prelude copies one side into the other at submission if that side is
/// behind then. From its first use of a side on, the list knows.
struct TwinUse {
    /// The depth texture (the title resource's host copy) and its twin.
    depth: std::sync::Arc<crate::resource::HostTexture>,
    twin: std::sync::Arc<crate::resource::Twin>,
    /// The sides (depth texture, twin) the list uses as it finds them.
    needs: [bool; 2],
    /// Whether each side lacks what the other was written last, where the
    /// list is now; `None`: as the list found it.
    behind: [Option<bool>; 2],
    /// The draw or dispatch being prepared writes the twin.
    writing: bool,
    /// The depth plane's twin holds what the plane lacks although the list
    /// copied the twins since: the plane's pass waits for the list's close.
    late: bool,
}

/// The pass that draws the twin of a depth plane into the plane: one
/// triangle over each mip of each slice, its pixel shader writing the
/// twin's texel as the depth, with the depth test "always" and no stencil
/// (the stencil plane is copied through a buffer and not touched here).
///
/// A 16-bit twin holds whole steps of 1/65535, and the depth written is
/// the step times the float nearest 1/65535 (0x37800080). That is how
/// The title's tiled lighting computes the depth of the
/// rectangles it then draws with the test "equal", one class of materials
/// a draw (all six of its vertex shaders multiply the class number by that
/// constant), so the pass and the draw give the host the same float, and
/// the host turns the same float into the same 24 bits. The step divided
/// by 65535 is another float for 512 of the 65,536 steps (none under 257),
/// and lands in another 24-bit step for 127 of them (none under 385).
const PLANE_CORNERS: &str = r#"
float4 corner(uint i : SV_VertexID) : SV_Position {
    return float4(float(i & 1) * 4.0 - 1.0, float(i >> 1) * 4.0 - 1.0, 0.0, 1.0);
}
"#;
const PLANE_STEPS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<uint> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return float(twin.Load(int4(int2(p.xy), 0, 0))) * asfloat(0x37800080);
}
"#;
const PLANE_FLOATS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<float> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return twin.Load(int4(int2(p.xy), 0, 0));
}
"#;

/// Views of twins the pass reads, in a heap of ours used as a ring.
/// shortcut: a view must last until its list has run; with one or two
/// passes a frame the ring is far from coming round.
const PLANE_VIEWS: u32 = 256;

/// What the pass draws with: the heap of its views of twins, and per kind
/// of twin (16-bit steps or floats) and depth format the root signature
/// and pipeline, made when first needed (`None`: the host refused).
struct PlanePass {
    views: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
    step: u32,
    next: AtomicU32,
    made: Mutex<Vec<(bool, i32, Option<PlanePipeline>)>>,
}

/// The pass's root signature and pipeline for one kind of twin and one
/// depth format.
type PlanePipeline = (ID3D12RootSignature, ID3D12PipelineState);

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for PlanePass {}
// SAFETY: as above.
unsafe impl Sync for PlanePass {}

fn plane_pass() -> Option<&'static PlanePass> {
    static PASS: OnceLock<Option<PlanePass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: plain heap creation and queries.
        unsafe {
            let views: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: PLANE_VIEWS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .ok()?;
            Some(PlanePass {
                cpu: views.GetCPUDescriptorHandleForHeapStart().ptr,
                gpu: views.GetGPUDescriptorHandleForHeapStart().ptr,
                step: host.device.GetDescriptorHandleIncrementSize(kind),
                views,
                next: AtomicU32::new(0),
                made: Mutex::default(),
            })
        }
    })
    .as_ref()
}

/// Compiles function `entry` of `source` for `target` with the system's
/// compiler, as the indirect-argument clamp is (`crate::indirect`).
pub(crate) fn compile(
    source: &str,
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> windows::core::Result<Vec<u8>> {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            windows::core::s!("victory"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("a shader of ours: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    };
    Ok(bytes.to_vec())
}

impl PlanePass {
    /// The root signature and pipeline for a twin of 16-bit steps (`steps`)
    /// or of floats and the depth format `format`.
    fn pipeline(&self, steps: bool, format: DXGI_FORMAT) -> Option<PlanePipeline> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, pipeline)) = made.iter().find(|m| (m.0, m.1) == (steps, format.0)) {
            return pipeline.clone();
        }
        let pipeline = match Self::make(steps, format) {
            Ok(pipeline) => Some(pipeline),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    steps,
                    format = format.0,
                    "no pipeline to draw a depth plane from its twin: such planes stay as they are"
                );
                None
            }
        };
        made.push((steps, format.0, pipeline.clone()));
        pipeline
    }

    fn make(steps: bool, format: DXGI_FORMAT) -> windows::core::Result<PlanePipeline> {
        use windows::core::s;
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        let vs = compile(PLANE_CORNERS, s!("corner"), s!("vs_5_1"))?;
        let source = if steps { PLANE_STEPS } else { PLANE_FLOATS };
        let ps = compile(source, s!("plane"), s!("ps_5_1"))?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        // SAFETY: the root signature is inside the pixel shader; the
        // description points at live data and never releases its pointer.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &ps[..])?;
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs[..]),
                PS: code(&ps[..]),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Every texel is written; stencil stays off.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                DSVFormat: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
            Ok((root, pipeline))
        }
    }
}

/// Records into `list` the pass that draws `plane`, the twin of the depth
/// plane of `depth`, into that plane: `depth` in the depth-write state,
/// `plane` readable by pixel shaders. The pass sets the list's descriptor
/// heap, graphics root signature, pipeline, depth target, viewport,
/// scissor and topology and puts none of them back: it is for a list that
/// records nothing of the title's after it.
fn record_plane_pass(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    plane: &ID3D12Resource,
) {
    let (Some(host), Some(pass)) = (host(), plane_pass()) else {
        return;
    };
    // SAFETY: live resources.
    let (of, twin) = unsafe { (depth.GetDesc(), plane.GetDesc()) };
    // An R16 typeless twin (53) holds steps, read as R16_UINT (57); an R32
    // typeless one floats, read as R32_FLOAT (41). The depth view names
    // the depth-stencil format of the texture's family (44: 45, 19: 20).
    let steps = twin.Format.0 == 53;
    let format = DXGI_FORMAT(match of.Format.0 {
        44 => 45,
        19 => 20,
        typed => typed,
    });
    let Some((root, pipeline)) = pass.pipeline(steps, format) else {
        return;
    };
    for slice in 0..u32::from(of.DepthOrArraySize) {
        for mip in 0..u32::from(of.MipLevels) {
            let slot = pass.next.fetch_add(1, Ordering::Relaxed) % PLANE_VIEWS;
            let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT(if steps { 57 } else { 41 }),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                        MostDetailedMip: mip,
                        MipLevels: 1,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                        ..Default::default()
                    },
                },
            };
            let target = D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                        MipSlice: mip,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                    },
                },
                ..Default::default()
            };
            let target = host.dsv(depth, Some(&target));
            let (width, height) = ((of.Width >> mip).max(1) as u32, (of.Height >> mip).max(1));
            let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: pass.cpu + (slot * pass.step) as usize,
            };
            let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: pass.gpu + u64::from(slot * pass.step),
            };
            // SAFETY: an open list; a slot of our heap; live resources and
            // objects (the pass's own live as long as the process).
            unsafe {
                host.device
                    .CreateShaderResourceView(plane, Some(&view), cpu);
                list.SetDescriptorHeaps(&[Some(pass.views.clone())]);
                list.SetGraphicsRootSignature(&root);
                list.SetPipelineState(&pipeline);
                list.SetGraphicsRootDescriptorTable(0, gpu);
                list.OMSetRenderTargets(0, None, false, Some(&target));
                list.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: width as f32,
                    Height: height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                list.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                }]);
                list.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY(4),
                );
                list.DrawInstanced(3, 1, 0, 0);
            }
            crate::resource::PLANES[0].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Records into `list` the copy of a depth texture into its colour twin
/// (`colour`) or of the twin into the depth texture, both already in their
/// copy states: whole textures of one size in one format family, which the
/// host copies as they are; or, for the twin of a stencil plane, that plane
/// alone, through the twin's buffer.
fn record_twin_copy(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    twin: &crate::resource::Twin,
    colour: bool,
) {
    let (from, to) = if colour {
        (depth, &twin.colour.texture)
    } else {
        (&twin.colour.texture, depth)
    };
    crate::resource::TWINS[2 + usize::from(colour)].fetch_add(1, Ordering::Relaxed);
    // shortcut: the depth plane's twin is not given the plane (that is a
    // pass the other way: the plane read by a shader, the twin as its
    // target). The twin then lacks what draws and clears wrote to the
    // plane since shaders last wrote the twin. In the title
    // nothing writes that plane but the twin's pass (its 192 draws against
    // it only test, run 29), so the twin lacks nothing. Counted.
    if colour && twin.plane.is_some() {
        crate::resource::PLANES[1].fetch_add(1, Ordering::Relaxed);
    }
    let Some(through) = &twin.through else {
        // SAFETY: an open list and two live textures.
        unsafe { list.CopyResource(to, from) };
        return;
    };
    // A stencil plane and its twin: each mip and slice of the plane goes
    // into the buffer, laid out as the host lays the plane out for copies,
    // and from the buffer into the other texture, whole subresources both
    // times. In the depth texture the stencil plane's subresources follow
    // the depth plane's; the twin has the one plane.
    let Some(host) = host() else { return };
    // SAFETY: a live texture.
    let of = unsafe { depth.GetDesc() };
    let planes = u32::from(of.MipLevels) * u32::from(of.DepthOrArraySize);
    let mut rows = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); planes as usize];
    // SAFETY: a plain query writing `planes` footprints.
    unsafe {
        host.device.GetCopyableFootprints(
            &of,
            planes,
            planes,
            0,
            Some(rows.as_mut_ptr()),
            None,
            None,
            None,
        );
    }
    let (first_from, first_to) = if colour { (planes, 0) } else { (0, planes) };
    let subresource = |texture: &ID3D12Resource, index: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index,
        },
    };
    let placed = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(through) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, source, dest) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: an open list and live resources; the buffer was made as big
    // as these footprints, and rests in the common state between copies.
    unsafe {
        list.ResourceBarrier(&[transition(through, common, dest)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_from + i as u32;
            list.CopyTextureRegion(&placed(row), 0, 0, 0, &subresource(from, index), None);
        }
        list.ResourceBarrier(&[transition(through, dest, source)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_to + i as u32;
            list.CopyTextureRegion(&subresource(to, index), 0, 0, 0, &placed(row), None);
        }
        list.ResourceBarrier(&[transition(through, source, common)]);
    }
}

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DRED").is_some()
    })
}

/// Indirect mesh dispatches since the last frame line: the largest group
/// count a record asked for, and the groups asked in all (from the lists'
/// budget buffers, a run late).
static MESH_GROUPS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];
/// Indirect compute records the clamp still cut to what one host dispatch
/// runs (more than 65,535 groups in a dimension, not in parts), since the
/// last frame line (from the lists' budget buffers, a run late).
static COMPUTE_CLAMPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Indirect compute records run in more than one part (`crate::indirect::bound_split`),
/// counted the same way.
static COMPUTE_SPLIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether a list's indirect-work budget lives in the card's own memory
/// (`VICTORY_INDIRECT_CARD_BUDGET=0` keeps it in memory the CPU maps, as
/// before). The clamp charges it with atomics, at least two in a row for
/// every clamp; in mapped memory on a card of its own each of them
/// crosses the bus.
fn card_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_CARD_BUDGET").is_none_or(|v| v != "0"))
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_TIMING").is_some())
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// One in [`ALIVE`]`[0]` while it lives (a host list), and one in
/// [`LISTS`]`[3]` until it is first reset.
struct Counted {
    /// Not reset since it was made.
    fresh: std::sync::atomic::AtomicBool,
}

impl Counted {
    fn new() -> Counted {
        ALIVE[0].fetch_add(1, Ordering::Relaxed);
        LISTS[3].fetch_add(1, Ordering::Relaxed);
        Counted {
            fresh: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// The list is reset: it is in use.
    fn used(&self) {
        if self.fresh.swap(false, Ordering::Relaxed) {
            LISTS[3].fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        ALIVE[0].fetch_sub(1, Ordering::Relaxed);
        self.used();
    }
}

/// A chunk of upload memory for lists' scratch data: the buffer, its
/// mapping, its host GPU address and its bytes.
struct Chunk {
    _buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
    size: u64,
}

// SAFETY: a D3D12 resource (free-threaded) and its persistent mapping,
// written only by the list that holds the chunk.
unsafe impl Send for Chunk {}

/// Upload chunks handed out front to back: the chunks (each with its
/// bytes), the one in use and the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(Chunk, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

impl Drop for Scratch {
    fn drop(&mut self) {
        give_chunks(std::mem::take(&mut self.chunks));
    }
}

const SCRATCH_CHUNK: u64 = 4 << 20;
/// A list's first chunk when the chunks are shared ([`scratch_pool`]): most
/// lists put a few constants or a rectangle's corners there, and every
/// list that did took 4 MiB of the PC's memory for it.
const SCRATCH_FIRST: u64 = 256 << 10;
/// The most bytes of chunks kept while no list holds them.
const SPARE_MOST: u64 = 128 << 20;

/// Whether lists share their scratch chunks: a list takes a chunk when it
/// first needs one, a small one first, and gives all of them back when it
/// is reset or dropped, for the next list. Before, a list kept every chunk
/// it ever took (4 MiB each) for as long as it lived, used or not, and the
/// host lists of held submissions, which live for one submission, each
/// made theirs anew: some 57 upload buffers of 4 MiB made, zeroed and
/// freed per frame of gameplay (run 60).
/// `VICTORY_SCRATCH_POOL=0`: chunks of 4 MiB kept by their list, as before.
fn scratch_pool() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SCRATCH_POOL").is_none_or(|v| v != "0"))
}

/// Things of some sizes that nothing holds now, kept for whoever next asks
/// for one of the same size: each with its bytes and when it was given
/// (`moment`).
struct Spare<T> {
    kept: Vec<(T, u64, u64)>,
    bytes: u64,
}

impl<T> Spare<T> {
    const fn new() -> Self {
        Spare { kept: Vec::new(), bytes: 0 }
    }

    /// A kept thing of `size` bytes, the one given last.
    fn take(&mut self, size: u64) -> Option<T> {
        let at = self.kept.iter().rposition(|(_, s, _)| *s == size)?;
        self.bytes -= size;
        Some(self.kept.remove(at).0)
    }

    /// Takes out what was given at a moment `old` says is long ago.
    fn trim(&mut self, old: impl Fn(u64) -> bool) -> Vec<T> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.kept.len() {
            if old(self.kept[i].2) {
                let (thing, size, _) = self.kept.remove(i);
                self.bytes -= size;
                out.push(thing);
            } else {
                i += 1;
            }
        }
        out
    }

    /// Keeps `thing` (`size` bytes) unless that takes the kept bytes past
    /// `most`: then it comes back.
    fn give(&mut self, thing: T, size: u64, most: u64) -> Option<T> {
        if self.bytes + size > most {
            return Some(thing);
        }
        self.bytes += size;
        self.kept.push((thing, size, moment()));
        None
    }
}

/// Whether the pools of upload memory let go of what stayed unused for
/// their idle time (spare scratch chunks, spare staging buffers and empty
/// staging blocks; on by default, `VICTORY_POOL_TRIM=0`: kept up to their
/// caps for the run, as before).
pub(crate) fn pool_trim() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_POOL_TRIM").is_none_or(|v| v != "0"))
}

/// Frees the scratch chunks no list took for the idle time of the PC's
/// memory (`ram_idle_ms`, 2 to 10 s, and 4 frames): the pool is capped at
/// [`SPARE_MOST`] but a burst's worth stayed for the run once kept.
fn trim_scratch() {
    if !pool_trim() {
        return;
    }
    let idle_ms = ram_idle_ms(available_physical_ram());
    let freed = SPARE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .trim(|given| long_ago(given, idle_ms, 4));
    if freed.is_empty() {
        return;
    }
    let n = freed.len() as u64;
    ALIVE[3].fetch_sub(n, Ordering::Relaxed);
    ALIVE[1].fetch_sub(n, Ordering::Relaxed);
    let sizes: u64 = freed.iter().map(|c| c.size).sum();
    ALIVE[4].fetch_sub(sizes, Ordering::Relaxed);
    ALIVE[2].fetch_sub(sizes, Ordering::Relaxed);
    trail(|| {
        format!(
            "sweep: {n} spare scratch chunks let go ({} KiB)",
            sizes >> 10
        )
    });
    // Spare chunks are named by no list, recorded or in the air.
    for chunk in freed {
        retire(chunk);
    }
}

/// Scratch chunks no list holds.
static SPARE: Mutex<Spare<Chunk>> = Mutex::new(Spare::new());

/// A scratch chunk of `size` bytes: one kept ([`SPARE`]) or a new one.
fn take_chunk(host: &Host, size: u64) -> Option<Chunk> {
    if scratch_pool()
        && let Some(chunk) = SPARE.lock().unwrap_or_else(|e| e.into_inner()).take(size)
    {
        ALIVE[3].fetch_sub(1, Ordering::Relaxed);
        ALIVE[4].fetch_sub(size, Ordering::Relaxed);
        ALIVE[5].fetch_add(1, Ordering::Relaxed);
        return Some(chunk);
    }
    let (buffer, data) = host.upload_buffer(size)?;
    name(&buffer, || "list scratch".into());
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    ALIVE[1].fetch_add(1, Ordering::Relaxed);
    ALIVE[2].fetch_add(size, Ordering::Relaxed);
    Some(Chunk {
        _buffer: buffer,
        data: data as usize,
        gpu,
        size,
    })
}

/// A list is done with `chunks` (it was reset after its submissions ran,
/// or dropped): kept for the next list, or freed past [`SPARE_MOST`] and
/// without the pool.
fn give_chunks(chunks: Vec<(Chunk, u64)>) {
    if chunks.is_empty() {
        return;
    }
    let mut freed = Vec::new();
    {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        for (chunk, size) in chunks {
            let back = if scratch_pool() {
                spare.give(chunk, size, SPARE_MOST)
            } else {
                Some(chunk)
            };
            match back {
                None => {
                    ALIVE[3].fetch_add(1, Ordering::Relaxed);
                    ALIVE[4].fetch_add(size, Ordering::Relaxed);
                }
                Some(chunk) => {
                    ALIVE[1].fetch_sub(1, Ordering::Relaxed);
                    ALIVE[2].fetch_sub(size, Ordering::Relaxed);
                    freed.push(chunk);
                }
            }
        }
    }
    // Freed outside the lock.
    drop(freed);
}

/// What the host keeps for copies in the card's own memory
/// ([`crate::shadow::cards`]): three lists of ours and the compare.
struct CardWork {
    /// Before a submission's lists: what changed on the copies' CPU side
    /// goes to their card side. Those not in use: a plan takes one and
    /// gives it back when its submission has run (`Host::card_done`), so
    /// that submissions sent ahead ([`late`]) each have their own.
    sync: Vec<Prelude>,
    /// After them, in the same submission: the compare of the declared
    /// ranges with their bytes from before. As `sync`.
    check: Vec<Prelude>,
    /// After it ran: the pieces that differ come down.
    fetch: Prelude,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// One dword per piece compared, set where the piece differs: PC
    /// memory, mapped (buffer, mapping, bytes).
    differs: (ID3D12Resource, usize, u64),
    /// Where the pieces that differ are fetched to: PC memory, mapped.
    fetched: Option<(ID3D12Resource, usize, u64)>,
    /// The dwords of `differs` taken by plans whose results were not read
    /// yet.
    taken: u32,
    /// Plans made and not done with (`Host::card_done`).
    out: u32,
    /// The fetch inside flights ([`inflight_fetch`]), if it is on and was
    /// made.
    inflight: Option<CardInflight>,
    /// The masked merge ([`CARD_MASK`]), if it was made.
    mask: Option<(ID3D12RootSignature, ID3D12PipelineState)>,
    /// Buffers of PC memory for masks, free (buffer, mapping, bytes): a
    /// plan takes one and `Host::card_done` gives it back.
    mask_free: Vec<(ID3D12Resource, usize, u64)>,
}

/// What the fetch inside flights needs ([`inflight_fetch`], [`CARD_FETCH`]).
struct CardInflight {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// PC memory, mapped (buffer, mapping, bytes): dword 0 the counter of
    /// tickets, then one dword per dword of `CardWork::differs`, the entry
    /// of that piece.
    table: (ID3D12Resource, usize, u64),
    /// PC memory, mapped (buffer, mapping, slots of [`INFLIGHT_SLOT`]
    /// bytes): where the pieces come down.
    fetched: (ID3D12Resource, usize, u32),
    /// Every ticket below it belongs to a landed flight: its slot may be
    /// written over. Raised by the thread that lands flights; read when a
    /// compare's list is recorded (an older value only gives fewer slots).
    landed: u32,
    /// What fetches the old way brought (copy, offsets, and the highest
    /// send whose lists were surely on the queue ahead of the fetch: its
    /// order of the queue less one). A flight sent at or before that ran
    /// ahead of the fetch, so the CPU's side holds those bytes as the card
    /// had them after the flight, or newer: the flight's own slot is older
    /// there and is not merged there. Dropped once every flight sent up to
    /// it has landed.
    brought: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)>,
    /// The frame ([`FRAMES_SEEN`]) of the last overflow or resize.
    quiet_from: u64,
}

// SAFETY: D3D12 objects (free-threaded) and persistent mappings, used only
// under the submission lock.
unsafe impl Send for CardWork {}

static CARD_WORK: Mutex<Option<CardWork>> = Mutex::new(None);

/// The compare of a declared range of a copy in the card's memory with
/// its bytes from before, one group of 64 threads per 64 KiB piece (cut at
/// multiples of 64 KiB of title memory, as `Shadow::card_piece` cuts them):
/// `differs` gets a dword per piece, 1 where a byte differs. The offsets are
/// widened to 16 bytes; buffers are multiples of 64 KiB, and a piece flagged
/// for a byte just outside the range costs a fetch that finds nothing.
///
/// Each thread hashes a kilobyte of the piece (the last one what is left),
/// 64 steps one after the other, and the first adds the 64 hashes up. One
/// thread per piece ([`CARD_COMPARE_ONE_THREAD`], `VICTORY_CARD_HASH_WIDE=0`)
/// walked its 4,096 steps alone, so a pass took the time of a whole piece
/// however few its pieces were, with the card all but idle (upstream: some
/// 350 such passes a frame in the room, 30 ms of the card's 136 busy ms).
/// A thread's hash starts from a value of its own place in the piece, so the
/// same bytes elsewhere in the piece hash otherwise; a change of one 16-byte
/// word always changes its thread's hash (each step maps the state one to
/// one), and so the sum. The hashes it stores are only ever compared by the
/// same program (the choice is made once a run), never by the CPU; the tests
/// check both programs against a CPU model of them, bit for bit.
///
/// The host dispatches one group per piece ([`card_dispatch`]).
pub(crate) const CARD_COMPARE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;   // as many groups are dispatched
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint compare;  // 0: the hashes are stored; else compared with those stored (2: and then stored)
};
ByteAddressBuffer now : register(t0);
// 16 bytes per piece of the copy: the hash of its bytes when they were
// last sent up (or hashed whole, the first time).
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
groupshared uint4 part[64];
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint piece = first + group.x;
    uint lo = piece * 65536u;
    lo = lo > skew ? lo - skew : 0u;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    // This thread's kilobyte; the last thread takes the rest (the widening
    // above makes a piece 16 bytes longer than 64 KiB where the copy does
    // not start at a multiple of 16).
    uint from = min(lo + lane * 1024u, hi);
    uint to = lane == 63u ? hi : min(from + 1024u, hi);
    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu)
        ^ ((lane + 1u) * uint4(0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du, 0x27D4EB2Fu));
    [loop]
    for (uint at = from; at < to; at += 16u) {
        h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    part[lane] = h;
    GroupMemoryBarrierWithGroupSync();
    if (lane != 0u) return;
    uint4 sum = uint4(0u, 0u, 0u, 0u);
    for (uint i = 0u; i < 64u; i++) {
        sum += part[i];
    }
    if (compare != 0u) {
        differs.Store(out_at + group.x * 4u, any(sum != hashes.Load4(piece * 16u)) ? 1u : 0u);
        // With the copy-back on its own thread: the next compare of the
        // piece finds what is written from here on.
        if (compare == 2u) hashes.Store4(piece * 16u, sum);
    } else {
        hashes.Store4(piece * 16u, sum);
    }
}
"#;

/// The compare as it was before [`CARD_COMPARE`]: one thread walks each
/// 64 KiB piece, 64 threads to a group (`VICTORY_CARD_HASH_WIDE=0`).
pub(crate) const CARD_COMPARE_ONE_THREAD: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint compare;  // 0: the hashes are stored; else compared with those stored (2: and then stored)
};
ByteAddressBuffer now : register(t0);
// 16 bytes per piece of the copy: the hash of its bytes when they were
// last sent up (or hashed whole, the first time).
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= pieces) return;
    uint piece = first + id.x;
    uint lo = piece * 65536u;
    lo = lo > skew ? lo - skew : 0u;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu);
    [loop]
    for (uint at = lo; at < hi; at += 16u) {
        h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    if (compare != 0u) {
        differs.Store(out_at + id.x * 4u, any(h != hashes.Load4(piece * 16u)) ? 1u : 0u);
        // With the copy-back on its own thread: the next compare of the
        // piece finds what is written from here on.
        if (compare == 2u) hashes.Store4(piece * 16u, h);
    } else {
        hashes.Store4(piece * 16u, h);
    }
}
"#;

/// The masked merge of a page of title bytes into a copy in the card's
/// memory (`crate::shadow::merge_mask`), recorded in the send list before a
/// submission's lists (`Host::card_before`), so it runs in the queue's
/// order behind the flights in the air: per word of the page,
/// `card = (card & !mask) | (side & mask)`, where `side` is the copy's
/// CPU side (holding the title's bytes there) and `mask` 0xff per byte the
/// title changed. Words with an empty mask are not written. One thread a
/// word, one dispatch a page. After the rexglue SDK (BSD-3-Clause,
/// `src/gpu/xenos/d3d12/shared_memory.cpp`): uploads are commands in the
/// queue, so work in flight keeps the old bytes and the CPU does not wait.
pub(crate) const CARD_MASK: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), SRV(t1), UAV(u0)"
cbuffer Page : register(b0) {
    uint at;       // the page's first word in the copy, in bytes
    uint words;    // its words
    uint mask_at;  // where its mask starts in `mask`, in bytes
    uint unused;
};
ByteAddressBuffer side : register(t0);
ByteAddressBuffer mask : register(t1);
RWByteAddressBuffer card : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= words) return;
    uint m = mask.Load(mask_at + id.x * 4u);
    if (m == 0u) return;
    uint off = at + id.x * 4u;
    card.Store(off, (card.Load(off) & ~m) | (side.Load(off) & m));
}
"#;

/// Whether the masked merge ([`CARD_MASK`]) was made with the compare, so
/// that pages may go up masked (`crate::shadow::merge_mask`).
static CARD_MASK_READY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// [`CARD_MASK_READY`].
pub(crate) fn card_mask_ready() -> bool {
    CARD_MASK_READY.load(Ordering::Acquire)
}

/// Since the last frame line: masked merges recorded ([`CARD_MASK`]) and
/// the bytes of upload memory their masks took; then, alive now, the
/// mask buffers and their bytes.
static MASK_STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Whether a piece of a card copy is hashed by a group of 64 threads, a
/// kilobyte each ([`CARD_COMPARE`]); `VICTORY_CARD_HASH_WIDE=0` goes back to
/// one thread a piece ([`CARD_COMPARE_ONE_THREAD`]). Read once: the hashes a
/// run keeps are only compared by the program that stored them.
pub(crate) fn card_hash_wide() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_HASH_WIDE").is_none_or(|v| v != "0"))
}

/// The most groups one dispatch takes in a dimension.
const CARD_GROUPS: u32 = 65_535;

/// The parts a pass of [`CARD_COMPARE`] over `pieces` pieces from `first`
/// is dispatched in, as (first piece, pieces, the first's dword in the
/// results in bytes): one group per piece, and no more groups in a
/// dispatch than the host runs. (A copy in the card's memory is under
/// 1 GiB, 16,384 pieces, so one part; the cut is for whatever changes
/// that.)
fn card_parts(first: u32, pieces: u32, out_at: u32) -> impl Iterator<Item = (u32, u32, u32)> {
    (0..pieces.div_ceil(CARD_GROUPS)).map(move |part| {
        let done = part * CARD_GROUPS;
        (first + done, (pieces - done).min(CARD_GROUPS), out_at + done * 4)
    })
}

/// Records a pass of the compare into `list`, whose compute root signature,
/// pipeline and views are the compare's: `constants` as the shader names
/// them (first piece, skew, size, pieces, results offset, mode). A group
/// per piece ([`card_parts`]), or with [`card_hash_wide`] off a thread per
/// piece as before.
///
/// # Safety
///
/// `list` is an open list with the compare's root signature and pipeline
/// set, and live buffers at its three views.
unsafe fn card_dispatch(list: &ID3D12GraphicsCommandList, constants: [u32; 6]) {
    if !card_hash_wide() {
        // SAFETY: the caller's.
        unsafe {
            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            list.Dispatch(constants[3].div_ceil(64), 1, 1);
        }
        return;
    }
    let [first, skew, size, pieces, out_at, mode] = constants;
    for (first, pieces, out_at) in card_parts(first, pieces, out_at) {
        let constants = [first, skew, size, pieces, out_at, mode];
        // SAFETY: the caller's.
        unsafe {
            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            list.Dispatch(pieces, 1, 1);
        }
    }
}

#[cfg(test)]
mod card_dispatch_tests {
    use super::{CARD_GROUPS, card_parts};

    /// A pass over a copy's pieces is one dispatch of a group per piece;
    /// past the most groups a dispatch takes it is cut, each part with its
    /// own first piece and place in the results.
    #[test]
    fn a_pass_is_cut_at_the_most_groups_a_dispatch_takes() {
        assert_eq!(card_parts(5, 0, 8).count(), 0);
        assert_eq!(card_parts(5, 3, 8).collect::<Vec<_>>(), [(5, 3, 8)]);
        assert_eq!(card_parts(0, CARD_GROUPS, 0).collect::<Vec<_>>(), [(0, CARD_GROUPS, 0)]);
        assert_eq!(
            card_parts(10, 2 * CARD_GROUPS + 7, 40).collect::<Vec<_>>(),
            [
                (10, CARD_GROUPS, 40),
                (10 + CARD_GROUPS, CARD_GROUPS, 40 + 4 * CARD_GROUPS),
                (10 + 2 * CARD_GROUPS, 7, 40 + 8 * CARD_GROUPS)
            ]
        );
    }
}

/// Whether the pieces a flight's compare flags come down to the PC inside
/// the flight itself (`VICTORY_CARD_FETCH_INFLIGHT=1`; off by default, and
/// only with [`relay`]).
///
/// With the relay the thread that lands flights fetched the flagged pieces
/// with a list of its own on the host's queue, the one the submitting
/// thread keeps filling: the fetch ran behind every flight sent meanwhile.
/// Measured on an RTX 3070 Laptop: 140.9 ms a frame of that thread waiting
/// for fetches, for 63 pieces (3.2 MB) in about five fetches, some 28 ms a
/// fetch spent queued behind newer flights; and the submitting thread
/// waited 81.7 ms a frame in "title memory read" settles for that thread.
///
/// With the switch the compare's list does the fetch: right behind the
/// compare, [`CARD_FETCH`] copies each piece it flagged into a slot of a
/// buffer in the PC's memory ([`CardInflight`]), so when the flight's fence
/// is reached its bytes are there, and the landing merges them
/// (`Shadow::card_merge_relay`) with no list, signal or wait of its own.
/// The merge's "fetched after" is the flight's own send number
/// (`CardPlan::sent_as`): see [`Host::relay_card_back`] for why that is
/// exact and when a landing falls back to the old fetch.
fn inflight_fetch() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        relay()
            && match std::env::var("VICTORY_CARD_FETCH_INFLIGHT").as_deref() {
                Ok("1") => true,
                Ok("0") => false,
                // On with the copy-back that does not wait (`back_async`).
                _ => back_async(),
            }
    })
}

/// The copy-back that does not make the thread that lands flights wait
/// for a fetch of its own (`VICTORY_BACK_ASYNC=1`; off by default: a game
/// run measured it much slower, GPU compare lists 8 -> 175 ms a frame). It turns [`inflight_fetch`] on unless
/// `VICTORY_CARD_FETCH_INFLIGHT=0`, and lets its slots grow when a landing
/// overflowed them, up to [`back_async_most`], and shrink back after
/// [`back_async_idle`] frames without one (`Host::card_inflight_resize`).
///
/// It is the rexglue SDK's fence-checked double buffer for memexport
/// read-backs (BSD-3-Clause; `src/gpu/xenos/d3d12/command_processor.cpp`,
/// `IssueDrawMemexportReadbackFastPath`): each flight records its own
/// copy of what its shaders wrote into a slot of read-back memory, right
/// behind its lists in the queue's order; a slot is read only once its
/// flight's fence has passed (the relay lands flights in the line's order,
/// after the card has run them), and a flight whose slot is missing, too
/// small or overwritten falls back to the full fetch and wait. What it
/// takes out is the wait: the fetch the relay sends of its own stands
/// behind the flights it lands (on the copy queue, `VICTORY_QUEUE2` bit 1)
/// or behind everything sent meanwhile (on the direct queue), and the
/// submitting thread's settles wait for that landing.
///
/// What it does not do: land a flight's results after the title was told
/// the flight ran. Every bottom-of-pipe write and fence signal behind a
/// flight is still done after its bytes are in title memory, as on the
/// console, where memory the card wrote is there when its fence says so;
/// title memory sees shader writes exactly when it did before.
pub(crate) fn back_async() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // Opt-in: GPU compare lists 8 -> 175 ms a frame with it on (measured on an RTX 4080 SUPER).
    *ON.get_or_init(|| std::env::var_os("VICTORY_BACK_ASYNC").is_some_and(|v| v == "1"))
}

/// The most bytes of slots [`back_async`] grows the fetch inside flights to
/// (`VICTORY_BACK_ASYNC_MB`, 512; never below [`inflight_fetch_bytes`]).
fn back_async_most() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        let mb = std::env::var("VICTORY_BACK_ASYNC_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(512);
        (mb << 20).max(inflight_fetch_bytes())
    })
}

/// Frames without an overflow after which grown slots go back to
/// [`inflight_fetch_bytes`] (`VICTORY_BACK_ASYNC_IDLE_FRAMES`, 600).
fn back_async_idle() -> u64 {
    static FRAMES: OnceLock<u64> = OnceLock::new();
    *FRAMES.get_or_init(|| std::env::var("VICTORY_BACK_ASYNC_IDLE_FRAMES").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(600).max(1))
}

/// The slots the fetch inside flights is to have ([`back_async`]): `now`
/// slots, `base` at the start, `most` at most; twice as many after an
/// overflow, `base` again after `quiet` frames without one of at least
/// `idle`. Powers of two stay powers of two ([`inflight_slots`]).
pub(crate) fn inflight_wanted(now: u32, base: u32, most: u32, overflowed: bool, quiet: u64, idle: u64) -> u32 {
    if overflowed {
        return now.saturating_mul(2).min(most).max(now);
    }
    if now > base && quiet >= idle {
        return base;
    }
    now
}

/// A landing overflowed the slots ([`back_async`]): raised by the thread
/// that lands flights, taken when the slots may be resized.
static INFLIGHT_OVERFLOWED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Frames seen by the frame lines so far, for [`back_async_idle`].
static FRAMES_SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The bytes of the slots alive now, and the times they were resized.
static INFLIGHT_BYTES: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// [`INFLIGHT_BYTES`]' bytes and [`back_async_most`], for the memory line;
/// 0 and 0 without the fetch inside flights.
pub(crate) fn back_slots_bytes() -> [u64; 2] {
    if !inflight_fetch() {
        return [0, 0];
    }
    [INFLIGHT_BYTES[0].load(Ordering::Relaxed), if back_async() { back_async_most() } else { inflight_fetch_bytes() }]
}

/// Why landings fell back from the fetch inside flights to a fetch and a
/// wait, since the last frame line: the slots overflowed; a range was not
/// compared or had no hashes; a flagged piece had no ticket; a piece was
/// fetched beside the flight by an earlier fetch on the copy queue
/// ([`BESIDE`]); anything else. Then, by the same reasons, the bytes those
/// fetches brought.
static BACK_FALLBACK: [std::sync::atomic::AtomicU64; 10] = [const { std::sync::atomic::AtomicU64::new(0) }; 10];

/// The reason ([`BACK_FALLBACK`]) the last landing fell back, set by the
/// thread that lands flights for its own fetch to count.
static BACK_REASON: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(4);

/// The bytes of the PC's memory pieces fetched inside flights go to
/// ([`inflight_fetch`]; `VICTORY_CARD_FETCH_INFLIGHT_MB`, 64): 1,024 slots
/// ([`inflight_slots`]). A flight whose flagged pieces find no free slot
/// lands the old way.
fn inflight_fetch_bytes() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_FETCH_INFLIGHT_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(64).max(1) << 20
    })
}

/// A slot of [`CARD_FETCH`]: a 64 KiB piece as the compare cuts it, widened
/// to 16 bytes at both ends (one more 16-byte step where the copy does not
/// start at a multiple of 16 in its piece).
pub(crate) const INFLIGHT_SLOT: u64 = (64 << 10) + 16;

/// The slots for `bytes` asked of [`inflight_fetch_bytes`]: one per 64 KiB,
/// rounded down to a power of two (at least 1, at most 2^15: the shader's
/// byte offsets into the slots are 32-bit), so that
/// `ticket % slots` runs on without a jump when the 32-bit ticket wraps
/// (2^32 is a multiple of it; with 1,023 slots it would jump by 4 and two
/// tickets in the air could share a slot). The buffer is `slots` times
/// [`INFLIGHT_SLOT`], a little over `bytes`.
pub(crate) fn inflight_slots(bytes: u64) -> u32 {
    let wanted = (bytes >> 16).clamp(1, 1 << 15);
    1u32 << (63 - wanted.leading_zeros())
}

/// [`CARD_FETCH`]'s table entry for a piece the compare did not flag.
pub(crate) const INFLIGHT_NONE: u32 = 0xffff_ffff;

/// Where piece `i` of `span` starts in its [`CARD_FETCH`] slot, as an
/// offset of the copy: the compare's lower bound of the piece, widened down
/// to 16 bytes (`CARD_COMPARE`, before clamping to the declared range).
/// What `Shadow::card_piece` gives for the piece lies at or above it, and
/// within [`INFLIGHT_SLOT`] of it.
pub(crate) fn inflight_piece_from(span: &crate::shadow::CardSpan, i: u32) -> usize {
    let lo = (span.first + i) as usize * 65536;
    let lo = lo.saturating_sub(span.skew as usize);
    lo & !15
}

/// Whether the ticket `k` of [`CARD_FETCH`]'s counter gets a slot, with
/// every ticket below `floor` landed and `slots` slots: the slot is
/// `k % slots`. The tickets of flights still in the air are at or above
/// `floor` (they are taken in the queue's order and flights land in that
/// order), so two that get a slot are fewer than `slots` apart and never
/// share one. The arithmetic of the shader, for the tests.
pub(crate) fn inflight_gets_slot(k: u32, floor: u32, slots: u32) -> bool {
    k.wrapping_sub(floor) < slots
}

/// The fetch inside a flight ([`inflight_fetch`]), run in the compare's list
/// right behind the compare of a declared range (a UAV barrier between):
/// one group per piece (looping where a range has more than 65,535); where
/// `differs` flags the piece, its first thread takes a ticket of the
/// counter (`table`'s dword 0) and, if [`inflight_gets_slot`], the group
/// copies the piece, cut and widened as the compare hashes it, into slot
/// `ticket % slots` of `fetched`. Every piece's entry in `table` (dword 1 +
/// its dword in `differs`) is written: the ticket, also one that got no
/// slot (the landing repeats the test with the plan's `landed`), or
/// [`INFLIGHT_NONE`].
pub(crate) const CARD_FETCH: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), SRV(t0), UAV(u0), UAV(u1), UAV(u2)"
#define SLOT 65552u
#define NONE 0xffffffffu
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint landed;   // every ticket below it is landed
    uint slots;    // slots of `fetched`
    uint groups;   // groups dispatched
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer differs : register(u0);
RWByteAddressBuffer table : register(u1);
RWByteAddressBuffer fetched : register(u2);
groupshared uint ticket;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint t : SV_GroupIndex) {
    [loop]
    for (uint i = group.x; i < pieces; i += groups) {
        if (t == 0u) {
            uint entry = NONE;
            if (differs.Load(out_at + i * 4u) != 0u) {
                uint k;
                table.InterlockedAdd(0, 1u, k);
                entry = k;
            }
            table.Store(4u + out_at + i * 4u, entry);
            ticket = entry;
        }
        GroupMemoryBarrierWithGroupSync();
        uint k = ticket;
        GroupMemoryBarrierWithGroupSync();
        if (k != NONE && (k - landed) < slots) {
            uint piece = first + i;
            uint lo = piece * 65536u;
            lo = lo > skew ? lo - skew : 0u;
            uint hi = min((piece + 1u) * 65536u - skew, size);
            lo = lo & ~15u;
            hi = (hi + 15u) & ~15u;
            uint into = (k % slots) * SLOT;
            [loop]
            for (uint at = lo + t * 16u; at < hi; at += 1024u) {
                fetched.Store4(into + (at - lo), now.Load4(at));
            }
        }
    }
}
"#;

/// Whether the pieces hashed again after ranges were sent up to a copy in
/// the card's memory are hashed in runs of pieces that touch, each piece
/// once (`VICTORY_CARD_REHASH_RUNS=1`; off by default), and not in a pass
/// per range sent up.
///
/// These passes are ours, they run on the card before every submission's
/// lists, and no timing so far names them: each walks its 64 KiB pieces
/// 16 bytes at a time, 4,096 steps one after the other, so a pass takes
/// the time of one piece however few its pieces are. Run 102 sent up 514
/// ranges a frame: 514 such passes. The same pieces are hashed either way;
/// a piece hashed when it need not be, or not hashed when it should, costs
/// a fetch that finds nothing and loses no shader write.
///
/// `VICTORY_CARD_REHASH_RUNS=alt`: twenty seconds with, twenty without, in
/// turn, each change said in the log: one run then measures both on the
/// same scene (the card meter's "send-and-hash lists" in the frame log).
/// Runs 105-110 found no change in the frame, but the card was not the
/// frame's limit then.
fn rehash_runs() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: OnceLock<u8> = OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_CARD_REHASH_RUNS").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            let joined = (now_us() / 20_000_000) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(joined), Ordering::Relaxed) != u8::from(joined) {
                tracing::info!(
                    "card rehash passes from now (VICTORY_CARD_REHASH_RUNS=alt): {}",
                    if joined { "JOINED, one pass a run of touching pieces" } else { "one pass a range sent up, as usual" }
                );
            }
            joined
        }
    }
}

/// Since the last frame-rate line, of our own passes over copies in the
/// card's memory: hash passes after ranges were sent up, the pieces they
/// hashed, and compare passes after the lists.
static CARD_PASSES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Whether what shaders wrote to copies in the card's memory is merged
/// into title memory on several threads (`VICTORY_WIDE_MERGE=1`; off by
/// default). The pieces the card's compare found differing come down in
/// one fetch and were then merged one after the other on the thread that
/// settles: about 630 pieces and 40 MB a frame (run 104), each compared
/// with the copy's CPU side byte by byte and put into title memory through
/// the kernel. A piece is its own bytes of its copy and of title memory,
/// so eight threads take an eighth of them each.
fn wide_merge() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_MERGE").is_some_and(|v| v == "1"))
}

/// One declared range of a copy in the card's memory, in a submission.
struct CardRange {
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    /// As declared (title addresses), for the statistics.
    declared: (usize, usize),
    span: crate::shadow::CardSpan,
    /// Its first piece's dword in the compare's results.
    at: u32,
    /// Whether the card compares it; else every piece is fetched.
    compared: bool,
}

/// What a submission does for copies in the card's memory.
struct CardPlan {
    ranges: Vec<CardRange>,
    /// The compare, run after the submission's lists.
    check: Option<ID3D12CommandList>,
    /// The copies whose card side was brought up to date, alive until the
    /// submission has run.
    _synced: Vec<std::sync::Arc<crate::shadow::Shadow>>,
    /// The lists of ours the plan recorded (whether the compare, the
    /// list), given back by `Host::card_done`.
    lists: Vec<(bool, Prelude)>,
    /// With [`relay`], the number of this plan's send of changed ranges
    /// (`QUEUE_ORDER`); else 0.
    sent_as: u64,
    /// Whether the compare's list fetches the flagged pieces itself
    /// ([`inflight_fetch`]): every compared range has its [`CARD_FETCH`].
    inflight: bool,
    /// The `CardInflight::landed` its [`CARD_FETCH`] was told: which of its
    /// tickets got a slot ([`inflight_gets_slot`]).
    inflight_landed: u32,
    /// The masks of its masked merges ([`CARD_MASK`]): held until its send
    /// list has run, then free for the next plan.
    masks: Option<(ID3D12Resource, usize, u64)>,
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>),
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// The bytes of `uploads` (counted in [`ALIVE`] while they are held).
    upload_bytes: u64,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// (fxceless 19a45d1) The last card-written source copied into each aliased host
    /// texture in this list (`crate::texture::alias`).
    texture_aliases: std::collections::HashMap<usize, usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Textures read back (texture to buffer copies), done once the list
    /// has run: the host buffer the list copies the texels into, the title
    /// memory they go to, the bytes of a row, the rows, the title's row
    /// pitch.
    backs: Vec<(ID3D12Resource, usize, [usize; 3])>,
    /// The bytes of `backs`' buffers (counted in [`READ_BACKS`] while
    /// they are held).
    back_bytes: u64,
    /// Whether the indirect-work budget is zeroed for this recording (a
    /// step on the CPU, or in the list for the card's copy).
    budget_zeroed: bool,
    /// The indirect scratch buffer (`HostList::indirect_args`, by its
    /// interface pointer) that a clamp in this recording left readable as
    /// indirect arguments; 0 if none. It stays readable until the next
    /// clamp into it ([`HostList::indirect_state`]).
    indirect_readable: usize,
    /// (Neptune) The title memory ranges `[from, to)` the list's shaders
    /// were declared to write so far ([`HostList::defer_gpu_write`]): what
    /// [`HostList::gpu_wrote`] looks through, instead of every step.
    gpu_ranges: Vec<(usize, usize)>,
    /// The title memory ranges `[from, to)` the list's shaders write in a
    /// copy placed in title memory itself (nothing deferred for them):
    /// what [`HostList::gpu_wrote_any`] adds to `gpu_ranges`.
    in_place_ranges: Vec<(usize, usize)>,
    /// (P1 meter) Why each fill / copy step of the list was not done on the
    /// card when it was recorded: source, bytes, reason ([`read_wait`]).
    not_card: Vec<(usize, usize, u8)>,
}

impl Deferred {
    /// Holds `buffer` (`size` bytes of upload memory) until the list is
    /// reset.
    fn hold_upload(&mut self, buffer: ID3D12Resource, size: u64) {
        ALIVE[6].fetch_add(1, Ordering::Relaxed);
        ALIVE[7].fetch_add(size, Ordering::Relaxed);
        self.upload_bytes += size;
        self.uploads.push(buffer);
    }
}

impl Drop for Deferred {
    fn drop(&mut self) {
        ALIVE[6].fetch_sub(self.uploads.len() as u64, Ordering::Relaxed);
        ALIVE[7].fetch_sub(self.upload_bytes, Ordering::Relaxed);
        READ_BACKS[0].fetch_sub(self.backs.len() as u64, Ordering::Relaxed);
        READ_BACKS[1].fetch_sub(self.back_bytes, Ordering::Relaxed);
    }
}

/// Read-back buffers lists hold now (texture to buffer copies whose texels
/// go to title memory once the list has run) and their bytes, for the
/// memory line: the layer's read-back memory that is not a pool.
pub(crate) static READ_BACKS: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether a queue's submissions go to the card without a wait for each
/// (`VICTORY_LATE_WAIT=1`; off by default, and off in the timing run, which
/// waits for every list on its own).
///
/// Every submission was prepared, sent, waited for and copied back before
/// the next one was looked at, so the card stood idle while we prepared
/// and we stood idle while it ran: in run 86 a frame of 446 ms is 134 ms of
/// preparing, 194 ms of waiting and 78 ms of copying back, one after the
/// other, on the one thread that submits.
///
/// With the switch a queue's submission is sent and joins a line of
/// "flights" ([`Flight`]); the thread goes on to the next one while the
/// card runs. The flights are settled together ([`Host::settle`]: one wait
/// for the card, what their shaders wrote copied back once per byte, then
/// their bottom-of-pipe writes and the fence signals behind them, in
/// order) at the next point where the title, or our own preparing, can
/// observe what they do. Those points, exactly:
///
/// 1. a synchronous submission of ours ([`Host::execute`]): a present, a
///    read-back or map of a texture, a clear or upload run at once;
/// 2. the next submission reads title memory that a flight's shaders may
///    write: a title buffer copy, a fill of upload memory, a texture's
///    memory, any step that runs code of ours on title memory
///    ([`late_blocked`]);
/// 3. the next submission refreshes a range of a host copy that a flight's
///    shaders may write, and the title wrote a page of it since it was last
///    copied ([`stale_under_flights`]): the refresh leaves such a range
///    alone until the flight's writes are back, as it always did within a
///    submission;
/// 4. a queue stands at a wait for a memory value that no flight's
///    bottom-of-pipe write is known to satisfy, at an address a flight may
///    write ([`late_met`]);
/// 5. the title blocks on a fence without an event;
/// 6. the card has run every flight (a thread of ours waits for that:
///    [`late_helper`]), or the flights hold [`late_most`] host lists.
///
/// What the title is told (bottom-of-pipe writes, fence values) is told
/// when the card has really run the work, later than before and never
/// earlier; a fence signalled on a queue while flights are in the air is
/// set behind them. A wait for a memory value that a flight's
/// bottom-of-pipe write satisfies counts as met: the waiting part goes to
/// the card behind that flight, which is the order the wait asks for.
///
/// What it assumes beyond today: memory the title rewrites between two of
/// its submissions, without waiting for the first, may be seen by the
/// first's shaders (as on the console, where the card reads title memory
/// when it runs, not when the list is submitted).
pub(crate) fn late() -> bool {
    late_stage() > 0
}

/// `VICTORY_LATE_WAIT`: 0 (off), 1, or 2 ([`late_exact`]).
fn late_stage() -> u8 {
    static STAGE: OnceLock<u8> = OnceLock::new();
    *STAGE.get_or_init(|| {
        if std::env::var_os("VICTORY_GPU_TIMING").is_some() {
            return 0;
        }
        match std::env::var("VICTORY_LATE_WAIT").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            Ok("3") => 3,
            // The relay alone: on top of stage 3.
            Err(_) if crate::shadow::relay() => 3,
            _ => 0,
        }
    })
}

/// Stage 2 of [`late`] (`VICTORY_LATE_WAIT=2`): point 3 of its list is no
/// longer a point. Where the title wrote a page that lies in a range a
/// flight's shaders may write, the bytes the title changed are found (the
/// page as it is now against what the copy held when the flight was
/// prepared) and put into the copy, and only those: what the flight's
/// shaders write beside them stays (`Shadow::merge_under_writes`). The
/// flights are settled for it only where that cannot be done: memory that
/// is not write-watched, or a copy in the card's memory with more changed
/// runs than a submission should send up one by one.
///
/// In run 92 (stage 1) this point settled the flights five times a frame
/// of twelve, each time with the card's work still to wait for: 172 ms a
/// frame of waiting were left of 200.
///
/// What differs from settling first: a byte both the title and a flight's
/// shader wrote is the title's if the shader wrote it first, the shader's
/// if it wrote it afterwards; settled first it was always the shader's.
/// The title writing a byte a shader of its own is writing is a race on
/// the console too.
fn late_exact() -> bool {
    late_stage() > 1
}

/// Stage 3 of [`late`] (`VICTORY_LATE_WAIT=3`: stage 2, and the present is
/// no longer a point where flights are settled). The copy to the back
/// buffer is sent as a flight of its own (`Host::execute_own`, a list of
/// its own for each present in the air) and the swap chain is told to
/// present behind it; the queue's call returns, and the card finishes the
/// frame while the title records the next one. What the title is told
/// about the frame (bottom-of-pipe writes, fences) is told when the card
/// has run it, by the settling thread.
///
/// What is left out with it: the look whether the presented frame is black
/// (diagnostics; it reads the frame back at once). A frame that is saved
/// or listed is presented the old way. Once every second and a half a
/// submission settles the flights all the same, for the sweep that lets
/// idle host objects go (it needs nothing in the air).
pub(crate) fn late_free_present() -> bool {
    late_stage() > 2
}

/// The relay's first stage (`VICTORY_RELAY=1`; off by default; it turns
/// [`late`] on at stage 3 unless `VICTORY_LATE_WAIT` says another stage,
/// and is off with it, in the timing run and with kept snapshots).
///
/// With [`late`] alone every settle does three things on the thread that
/// holds the submission lock: it waits for everything in the air, copies
/// back with the card standing, and only then goes on preparing. The frame
/// stays a relay: the card works about 242 ms of it and one thread
/// prepares, waits and copies back for about 220.
///
/// With the switch a thread of its own ([`relay_worker`]) lands flights as
/// soon as the card has run them, in the order they were sent and without
/// the submission lock: it fetches and merges what their shaders wrote to
/// copies in the card's memory, compares and merges the other copies,
/// frees the flights' ranges, and then does their read-backs, their
/// bottom-of-pipe writes and the fence signals behind them, in the line's
/// order. The submitting thread meanwhile prepares and sends the next
/// submissions. [`Host::settle`] no longer settles anything: it waits until
/// that thread has landed everything in the air, at the same points as
/// before (the list in [`late`]), so whoever can observe title memory, a
/// fence or a present still does so only after the flights before it are
/// merged.
///
/// What makes that safe is in `crate::shadow::relay` (ranges counted per
/// flight; a merge that never stores into memory the card may be writing;
/// sends to card copies numbered) and here: flights land in the order of
/// [`LINE`] by one thread, so a later flight's results never reach title
/// memory before an earlier flight's; the compare on the card stores the
/// hash it made, so each flight's compare flags exactly what changed since
/// the compare before it, however late the pieces are fetched; the fetch
/// goes to the queue under [`QUEUE_ORDER`], which says which sends of
/// title bytes the card had received when it ran.
pub(crate) fn relay() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| late_stage() > 0 && crate::shadow::relay())
}

/// The relay's second stage (`VICTORY_RELAY=2`): a submission that reads
/// title memory a flight's shaders may write (a title buffer copy, a fill
/// of upload memory, a texture) waits only until the flights that declared
/// that memory have landed, not for everything in the air; and at
/// [`late_most`] lists in the air it waits until there are fewer. At `1`
/// both wait for everything.
fn relay_exact() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // Stage 2 unless `VICTORY_RELAY=1` asks for the first one (the default
    // since the walls of the first room were found drawn in every run with
    // this set and in fewer than half without it).
    *ON.get_or_init(|| relay() && std::env::var_os("VICTORY_RELAY").is_none_or(|v| v == "2"))
}

/// Told when something joins [`LINE`], for the thread that lands flights
/// ([`relay`]); [`LINE_MOVED`] stays the word for the thread that looks at
/// what queues hold.
static RELAY_MOVED: std::sync::Condvar = std::sync::Condvar::new();

/// Told when flights have landed ([`relay`]): who waits for that looks
/// again.
static LINE_LANDED: std::sync::Condvar = std::sync::Condvar::new();

/// With [`relay`], the order of what we put on the queue: the number of
/// sends of changed ranges to copies in the card's memory so far
/// (`Host::card_before`). Held from the taking of the ranges until their
/// list is on the queue, and while a fetch of pieces goes to the queue: a
/// fetch sent with the count at N ran behind sends 1..N and ahead of every
/// later one.
static QUEUE_ORDER: Mutex<u64> = Mutex::new(0);

/// [`QUEUE_ORDER`]'s value as last raised, read without its lock
/// ([`demand_fetch`]).
static QUEUE_ORDER_SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// (Stage 1, `crate::readback::consumers`) Fetches `pieces` (copy, offsets
/// `[lo, hi)`) of copies in the card's memory, now, from any thread: one
/// list of copies on the host's queue, behind everything sent so far
/// (every flight that wrote them among it), waited for. Gives their bytes
/// and the merge's "fetched after" (`Shadow::card_merge_relay`): the last
/// numbered send of title bytes known to be on the queue ahead of the
/// fetch, taken one lower than last raised (the submission raising it may
/// not have its sends on the queue yet; a send taken as later only leaves
/// the title's own bytes to stand, which they are). One at a time (its
/// own list, fence and event).
pub(crate) fn demand_fetch(pieces: &[(std::sync::Arc<crate::shadow::Shadow>, usize, usize)]) -> Option<(Vec<Vec<u8>>, u64)> {
    struct Held(Option<RelayFetch>);
    // SAFETY: the list, fence and event are used under the mutex only.
    unsafe impl Send for Held {}
    static FETCH: Mutex<Held> = Mutex::new(Held(None));
    let host = host()?;
    let total: u64 = pieces.iter().map(|(_, lo, hi)| (hi - lo) as u64).sum();
    if total == 0 {
        return Some((pieces.iter().map(|_| Vec::new()).collect(), 0));
    }
    let mut held = FETCH.lock().unwrap_or_else(|e| e.into_inner());
    if held.0.is_none() {
        held.0 = host.make_relay_fetch();
    }
    let f = held.0.as_mut()?;
    if f.buffer.as_ref().is_none_or(|b| b.2 < total) {
        let size = total.next_multiple_of(4 << 20);
        f.buffer = host.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
    }
    let (buffer, data, _) = f.buffer.clone()?;
    f.sent += 1;
    let ((allocator, list), fence, event, sent) = (f.list.clone(), f.fence.clone(), f.event, f.sent);
    let after = QUEUE_ORDER_SEEN.load(Ordering::Acquire).saturating_sub(1);
    // SAFETY: this list's last run was waited for (under the mutex); live
    // buffers (the copies are held by the caller); our fence and event.
    let ran = unsafe {
        (|| -> windows::core::Result<()> {
            allocator.Reset()?;
            list.Reset(&allocator, None)?;
            let mut at = 0u64;
            for (shadow, lo, hi) in pieces {
                list.CopyBufferRegion(&buffer, at, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
                at += (*hi - *lo) as u64;
            }
            list.Close()?;
            host.queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>()?)]);
            host.card_sent();
            host.queue.Signal(&fence, sent)?;
            fence.SetEventOnCompletion(sent, event)?;
            WaitForSingleObject(event, INFINITE);
            Ok(())
        })()
    };
    if let Err(e) = ran {
        tracing::error!(error = %e, "copy-back: a demanded fetch from the card failed (VICTORY_BACK_SKIP_UNREAD=1)");
        return None;
    }
    let mut out = Vec::with_capacity(pieces.len());
    let mut at = 0usize;
    for (_, lo, hi) in pieces {
        // SAFETY: the mapping of the fetched bytes, `total` of them; the
        // copy into it has run.
        out.push(unsafe { std::slice::from_raw_parts((data + at) as *const u8, hi - lo) }.to_vec());
        at += hi - lo;
    }
    Some((out, after))
}

/// Since the last frame-rate line, of the thread that lands flights
/// ([`relay`]): landings, flights landed, microseconds it waited for the
/// card with nothing to land, microseconds it waited for fetches of card
/// copies' pieces, microseconds of landing in all.
static RELAY_STATS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Since the last frame-rate line, with [`inflight_fetch`]: pieces merged
/// from the slots their flights fetched them to, their bytes, landings that
/// went the old way because a plan's pieces overflowed the slots, and
/// landings that fetched the old way (for any reason, overflow too).
static INFLIGHT_STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// After a fetch the old way taken with the queue's count at `after`, for
/// plans whose highest send is `landed`: up to which send the notes of
/// sends to card copies may go (`crate::shadow::relay_fetched`).
///
/// Without [`inflight_fetch`] every later merge is of a later fetch, whose
/// count is `after` or more: notes up to `after` are of no use, as before.
/// With it, a flight still in the air, sent before the fetch, merges its
/// own slot with "fetched after" its own send, which may be below `after`:
/// it needs the notes of the sends between, or it takes the title's bytes
/// that went up in them for its shaders' writes and puts its older card
/// bytes over them. (Flights 5, 6, 7 in the air; 5 lands the old way with
/// the count at 7; a title byte went up in send 7 into a piece 6's shaders
/// write; 6 lands from its slot, which holds that byte from before send 7.)
/// So the mark goes only to the landed plans' highest send: every flight
/// still to land was sent after it, and every later fetch the old way
/// takes a count at least that high. (`CardInflight::brought` is another
/// matter: the card's bytes the fetch put in the CPU's side, not the
/// title's.)
pub(crate) fn fallback_fetched_mark(inflight: bool, after: u64, landed: Option<u64>) -> u64 {
    match (inflight, landed) {
        (false, _) => after,
        (true, Some(landed)) => landed.min(after),
        (true, None) => 0,
    }
}

/// The landed mark of [`CARD_FETCH`]'s tickets after ticket `k` landed: one
/// past `k` if that is ahead of `landed`, in wrapping order (tickets in the
/// air are always within 2^31 of the mark), else `landed`.
pub(crate) fn inflight_raise(landed: u32, k: u32) -> u32 {
    let next = k.wrapping_add(1);
    if (next.wrapping_sub(landed) as i32) > 0 { next } else { landed }
}

/// With [`inflight_fetch`], after a landing merged (either way): the tickets
/// these plans' fetches took are landed, and their slots may be written
/// over. Plans land in the line's order and tickets are taken in the
/// queue's (a UAV barrier opens every compare's list), so every ticket of a
/// plan in the air is ahead of these.
///
/// The counter is never reset and wraps after 2^32 tickets: the mark is
/// raised in wrapping order ([`inflight_raise`]), and the slots are a power
/// of two, so `k % slots` runs on across the wrap. shortcut: ticket
/// 0xffff_ffff reads as [`INFLIGHT_NONE`]; a flagged piece that draws it
/// copies nothing and its landing goes the old way, once every 2^32.
fn inflight_tickets_landed(plans: &[&CardPlan]) {
    let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(work) = work.as_mut() else {
        return;
    };
    let dwords = (work.differs.2 / 4) as usize;
    let Some(inflight) = work.inflight.as_mut() else {
        return;
    };
    // SAFETY: the mapping of the tickets; these plans' fetches have run.
    let table = unsafe { std::slice::from_raw_parts((inflight.table.1 as *const u32).add(1), dwords) };
    for plan in plans.iter().filter(|p| p.inflight) {
        for r in plan.ranges.iter().filter(|r| r.compared) {
            for p in 0..r.span.pieces {
                if let Some(&k) = table.get((r.at + p) as usize)
                    && k != INFLIGHT_NONE
                {
                    inflight.landed = inflight_raise(inflight.landed, k);
                }
            }
        }
    }
    // Every flight still to land was sent after these: what fetches the old
    // way brought up to them concerns none of them.
    if let Some(last) = plans.iter().map(|p| p.sent_as).max() {
        inflight.brought.retain(|b| b.3 > last);
    }
}

/// Of the line's things in order (a flight: the value signalled on the
/// host's fence behind it; a write or a signal behind a flight: `None`):
/// how many, from the front, can be landed when the card has reached
/// `completed`: everything ahead of the first flight it has not run. A
/// later flight never lands before an earlier one, and what follows a
/// flight (its bottom-of-pipe writes, signals) never before the flight.
fn landable(line: &[Option<u64>], completed: u64) -> usize {
    line.iter().position(|f| f.is_some_and(|f| f > completed)).unwrap_or(line.len())
}

/// For one plan of copies in the card's memory, with [`relay`]: `flags`
/// holds, for every piece of every range of the plan, the copy, the
/// piece's number in the copy and whether the compare flagged it there (or
/// did not run). The pieces to fetch, under whichever range: all of them
/// flagged under any. Two ranges of a plan may hold parts of the same
/// 64 KiB piece; the compare hashes the whole piece for each and stores
/// the hash, so the second of them may find it stored already and not
/// flag it.
fn flagged_pieces(flags: &[(usize, u32, bool)]) -> std::collections::HashSet<(usize, u32)> {
    flags.iter().filter(|f| f.2).map(|f| (f.0, f.1)).collect()
}

/// Whether pieces fetched on a copy queue beside later flights are fetched
/// again when those flights land (`VICTORY_QUEUE2_REFETCH=0`: not, as
/// before; on by default with `VICTORY_QUEUE2` bit 1).
fn refetch() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_QUEUE2_REFETCH").is_none_or(|v| v != "0"))
}

/// With fetches on a copy queue (`crate::queue2`, bit 1): the pieces (copy,
/// piece number) a fetch brought down while later submissions may have been
/// running, each with the number of the last submission started when the
/// fetch had run ([`STARTED`]).
///
/// Such a fetch stands behind the flights being landed and nothing else,
/// so it runs beside whatever was sent after them. A later flight that
/// writes the piece (a buffer cleared and filled again) may be half way
/// through it: the bytes that come down are neither the earlier flight's
/// nor the later one's, and they go into title memory. If the later flight
/// leaves the piece as the earlier one left it (a still scene does, every
/// frame), its own compare finds the stored hash unchanged and flags
/// nothing: title memory kept the half-written bytes until the piece next
/// changed. On the direct queue a fetch only ever sees whole flights.
///
/// So a flight that declared such a piece and had started before the fetch
/// had run fetches it at its own landing whatever its compare says
/// ([`fetched_beside`]); that fetch stands behind the flight, and is noted
/// here in turn if still later flights were in the air. Between the two
/// landings title memory may hold the half-written bytes, as it does on
/// the console, where the title reads the memory the card is writing: what
/// the title is promised is the piece as the flight left it once that
/// flight's fence is signalled, and the landing comes before the signal.
static BESIDE: Mutex<Option<std::collections::HashMap<(usize, u32), u64>>> = Mutex::new(None);

/// Since the last frame-rate line: pieces fetched because an earlier fetch
/// of them ran beside their flight ([`BESIDE`]), and pieces noted there.
static BESIDE_STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// [`BESIDE`]: whether the flight numbered `number` (among the submissions
/// started) must fetch a piece that an earlier landing's fetch brought
/// down when submissions up to `started` had been started: it had started
/// by then, so it may have been writing the piece under that fetch.
fn fetched_beside(started: Option<u64>, number: u64) -> bool {
    started.is_some_and(|started| started >= number)
}

/// [`BESIDE`], after a landing of flights up to number `landed` whose
/// fetch brought `fetched` down while submissions up to `started` had
/// been started: what is noted of the pieces afterwards, and how many were
/// noted now. A note is done with once every submission it names has
/// landed (each fetched the piece again if it declared it); the pieces
/// just fetched are noted if anything later than the flights landed had
/// started.
fn note_beside(noted: &mut std::collections::HashMap<(usize, u32), u64>, fetched: &[(usize, u32)], landed: u64, started: u64) -> u64 {
    noted.retain(|_, upto| *upto > landed);
    if started <= landed {
        return 0;
    }
    for piece in fetched {
        noted.insert(*piece, started);
    }
    fetched.len() as u64
}

#[cfg(test)]
mod relay_tests {
    use super::{fetched_beside, flagged_pieces, landable, note_beside};

    /// A piece fetched on the copy queue while a later flight was in the
    /// air is fetched again by that flight, and by no flight sent after
    /// the fetch had run; the note goes when its flights have landed.
    #[test]
    fn a_piece_fetched_beside_a_later_flight_is_fetched_again_by_it() {
        let mut noted = std::collections::HashMap::new();
        let (a, p) = (0x1000usize, 7u32);
        // Flight 5 lands; its fetch ran while 6 and 7 had started.
        assert_eq!(note_beside(&mut noted, &[(a, p)], 5, 7), 1);
        assert!(fetched_beside(noted.get(&(a, p)).copied(), 6));
        assert!(fetched_beside(noted.get(&(a, p)).copied(), 7));
        // Flight 8 was sent after the fetch had run: its compare decides.
        assert!(!fetched_beside(noted.get(&(a, p)).copied(), 8));
        // Another piece was never noted.
        assert!(!fetched_beside(noted.get(&(a, p + 1)).copied(), 6));
        // Flight 6 lands and fetches it again, with 7 still in the air and
        // nothing newer: the note stays for 7.
        assert_eq!(note_beside(&mut noted, &[(a, p)], 6, 7), 1);
        assert!(fetched_beside(noted.get(&(a, p)).copied(), 7));
        // Flight 7 lands with nothing behind it: nothing is noted any more.
        assert_eq!(note_beside(&mut noted, &[(a, p)], 7, 7), 0);
        assert!(noted.is_empty());
        // A fetch with nothing later in the air notes nothing.
        assert_eq!(note_beside(&mut noted, &[(a, p)], 9, 9), 0);
        assert!(noted.is_empty());
    }

    /// Two flights and what follows each: nothing of the second lands
    /// before the card has run it, and nothing behind a flight lands
    /// before the flight.
    #[test]
    fn flights_land_in_the_order_they_were_sent() {
        // Flight (fence 5), its write, a signal, flight (fence 6), its write.
        let line = [Some(5), None, None, Some(6), None];
        assert_eq!(landable(&line, 4), 0);
        assert_eq!(landable(&line, 5), 3);
        assert_eq!(landable(&line, 6), 5);
        assert_eq!(landable(&line, u64::MAX), 5);
        // Signals with no flight ahead land at once; behind one they wait.
        assert_eq!(landable(&[None, None], 0), 2);
        assert_eq!(landable(&[None, Some(9), None], 8), 1);
        assert_eq!(landable(&[], 3), 0);
        // After the first three landed, the rest when its turn comes.
        assert_eq!(landable(&line[3..], 5), 0);
        assert_eq!(landable(&line[3..], 6), 2);
    }

    /// The slots of the copy-back inside flights (`back_async`) double
    /// after an overflow up to the most, go back to the start after a quiet
    /// while, and stay powers of two; a ticket in the air never shares a
    /// slot with another under the new count.
    #[test]
    fn back_slots_grow_on_overflow_and_shrink_when_quiet() {
        use super::{inflight_gets_slot, inflight_slots, inflight_wanted};
        let (base, most) = (inflight_slots(64 << 20), inflight_slots(512 << 20));
        assert_eq!((base, most), (1024, 8192));
        let mut now = base;
        for want in [2048, 4096, 8192, 8192] {
            now = inflight_wanted(now, base, most, true, 0, 600);
            assert_eq!(now, want);
            assert!(now.is_power_of_two());
        }
        // Not quiet long enough: kept; then back to the start.
        assert_eq!(inflight_wanted(now, base, most, false, 599, 600), now);
        assert_eq!(inflight_wanted(now, base, most, false, 600, 600), base);
        // At the start nothing shrinks; a most below now never shrinks it.
        assert_eq!(inflight_wanted(base, base, most, false, 10_000, 600), base);
        assert_eq!(inflight_wanted(base, base, base, true, 0, 600), base);
        // Resized with nothing in the air (every ticket below `landed`):
        // the tickets taken from there get distinct slots up to the count.
        for slots in [1024u32, 2048, 8192] {
            for landed in [0u32, 5000, u32::MAX - 100] {
                let mut seen = std::collections::HashSet::new();
                for k in 0..slots {
                    let ticket = landed.wrapping_add(k);
                    assert!(inflight_gets_slot(ticket, landed, slots));
                    assert!(seen.insert(ticket % slots), "two tickets in the air share a slot");
                }
                assert!(!inflight_gets_slot(landed.wrapping_add(slots), landed, slots));
            }
        }
    }

    /// A piece flagged under one range of a plan is fetched under every
    /// range that holds a part of it; pieces of other copies are not.
    #[test]
    fn a_piece_flagged_under_one_range_is_fetched_under_all() {
        let (a, b) = (0x1000usize, 0x2000usize);
        // Copy a: piece 3 under two ranges, flagged under the first only;
        // piece 4 not flagged; copy b: piece 3 not flagged, piece 9 flagged.
        let flags = [(a, 3, true), (a, 4, false), (a, 3, false), (b, 3, false), (b, 9, true)];
        let fetch = flagged_pieces(&flags);
        assert!(fetch.contains(&(a, 3)));
        assert!(!fetch.contains(&(a, 4)));
        assert!(!fetch.contains(&(b, 3)));
        assert!(fetch.contains(&(b, 9)));
        assert_eq!(fetch.len(), 2);
    }
}

/// The most host lists in the air before the flights are settled
/// (`VICTORY_LATE_WAIT_LISTS`, 256; `=128` is the value before: with the
/// relay it stopped the submitting thread twice a frame for 92-103 ms in
/// all while the card stood idle 64-113 ms a frame, runs 158, 159 and 162;
/// 32 settled the flights three times a
/// frame in run 92 and 64 still 1.3-1.7 times in run 99, with 12,919 tables
/// asked and none refused; watch "tables asked, N refused"): their dynamic
/// descriptor slots, upload
/// memory and scratch chunks are held until then, and a frame's lists held
/// all at once ran the slots out before (run 45).
fn late_most() -> usize {
    static MOST: OnceLock<usize> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_LATE_WAIT_LISTS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(256).max(1)
    })
}

/// Whether a submission with bottom-of-pipe writes may fly
/// (`VICTORY_LATE_WAIT_WRITES=0`: it is settled as soon as it is sent, so
/// the title is told at once, as before; for telling a title that polls
/// such a value from one that does not).
fn late_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LATE_WAIT_WRITES").is_none_or(|v| v != "0"))
}

/// Per host copy, the ranges a submission's shaders may write, as declared
/// (address, end).
type Declared = (std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize)>);

/// What [`Host::submit`] hands back for a submission it sent and did not
/// wait for.
struct Sent {
    /// Its number among the submissions started.
    number: u64,
    card: Option<CardPlan>,
    back: Vec<Declared>,
    /// Its bottom-of-pipe writes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
}

/// A submission on the card that has not been waited for ([`late`]): its
/// host lists, taken from the title's (which it may reset meanwhile), and
/// what is to be done once it has run.
struct Flight {
    number: u64,
    /// The value signalled on the host's fence behind it.
    fence: u64,
    lists: Vec<crate::list::Detached>,
    order: Vec<usize>,
    card: Option<CardPlan>,
    back: Vec<Declared>,
}

/// One thing in the line of what waits for the card, in order.
enum Late {
    Run(Flight),
    /// A bottom-of-pipe write of the flight before it: address, what, bytes.
    Write(usize, crate::queue::Written, usize),
    /// A fence (its address) signalled behind the flights before it.
    Signal(usize, u64),
}

struct Line {
    line: Vec<Late>,
    /// The title memory the flights' shaders may write (address, end), in
    /// order and joined.
    written: Vec<(usize, usize)>,
    /// Host lists in the flights.
    lists: usize,
    /// The value signalled on the host's fence behind the last flight.
    target: u64,
    /// A settle has taken the line and is not done: a signal still goes
    /// behind it.
    settling: bool,
    /// Something was settled away from a queue's own calls: the helper
    /// thread looks at what queues hold.
    poke: bool,
    /// Host lists of settled flights, reset and given back outside the
    /// submission lock ([`land`]: a settle may run under a caller's own
    /// locks).
    landed: Vec<(Vec<crate::list::Detached>, Vec<usize>)>,
    /// With [`relay`]: the bottom-of-pipe writes of the flights being
    /// landed (taken out of `line`, not done yet), for [`late_met`].
    in_hand: Vec<(usize, crate::queue::Written, usize)>,
}

// SAFETY: D3D12 objects (free-threaded) and host lists nothing records
// into; touched under the lock.
unsafe impl Send for Line {}

static LINE: Mutex<Line> = Mutex::new(Line {
    line: Vec::new(),
    written: Vec::new(),
    lists: 0,
    target: 0,
    settling: false,
    poke: false,
    landed: Vec::new(),
    in_hand: Vec::new(),
});
/// Told when a flight joins [`LINE`] or it is poked.
static LINE_MOVED: std::sync::Condvar = std::sync::Condvar::new();

fn line() -> std::sync::MutexGuard<'static, Line> {
    LINE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Why flights were settled ([`LATE_STATS`]' places from 2 on).
const WHY_SYNC: usize = 2;
const WHY_CARD_DONE: usize = 3;
const WHY_VALUE: usize = 4;
const WHY_STALE: usize = 5;
const WHY_READ: usize = 6;
const WHY_RUN: usize = 7;
const WHY_MOST: usize = 8;
const WHY_WRITES: usize = 9;
const WHY_FENCE: usize = 10;

/// Since the last frame-rate line: submissions sent without a wait;
/// settles; settles by reason ([`WHY_SYNC`]..); flights settled;
/// microseconds settles waited for the card; value waits met by a flight's
/// bottom-of-pipe write; fence signals set behind flights.
/// Then: the settles for title memory read, by what read it (a buffer
/// copy, a fill of upload memory, a texture).
static LATE_STATS: [std::sync::atomic::AtomicU64; 18] = [const { std::sync::atomic::AtomicU64::new(0) }; 18];

/// Since the last frame-rate line: the microseconds settles waited for the
/// card, by why they settled (the `WHY_` places).
static LATE_WAITED: [std::sync::atomic::AtomicU64; 11] = [const { std::sync::atomic::AtomicU64::new(0) }; 11];

/// (Neptune) Title memory declared as written by shaders in this frame or
/// the two before it (start and end, with the frame of the last
/// declaration): what `VICTORY_INDIRECT_RAW=3` must not trust a CPU read of.
/// (Prefault job) With each range: whether it is of a copy placed in title
/// memory (`in_place`: no copy-back compares it, so what its shaders really
/// changed is never known).
#[allow(clippy::type_complexity)]
static SHADER_WRITTEN: Mutex<Option<(u64, std::collections::HashMap<(usize, usize), (u64, bool)>)>> = Mutex::new(None);

/// Notes `[address, address + len)` as written by shaders now.
fn shader_writes(address: usize, len: usize, in_place: bool) {
    let frame = crate::frame::current();
    // A shader write over the CPU-written indirect arena stamps its pages
    // (`crate::indirect::arena`: records there are then checked exactly).
    crate::indirect::arena::note_shader_write(address, len, frame);
    let mut written = SHADER_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    let (swept, ranges) = written.get_or_insert_with(Default::default);
    let before = ranges.len();
    let mut moved = false;
    if *swept != frame {
        *swept = frame;
        ranges.retain(|_, at| at.0 + 2 >= frame);
        moved = ranges.len() != before;
    }
    let n = ranges.len();
    let entry = ranges.entry((address, address.saturating_add(len))).or_insert((frame, in_place));
    moved |= !entry.1 && in_place;
    *entry = (frame, entry.1 || in_place);
    // (P3) The set of ranges or a range's kind changed: answers kept for
    // the old set are void ([`shader_written_split`]). A range declared
    // again (dispatches declare the same ones thousands of times) changes
    // neither.
    if moved || ranges.len() != n {
        SHADER_WRITTEN_GEN.fetch_add(1, Ordering::Release);
    }
}

/// (P3) Moves when [`SHADER_WRITTEN`]'s ranges or their kinds change.
static SHADER_WRITTEN_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// (P3) Since the start: [`shader_written_split`] answered from the thread's
/// record of earlier answers, and looked up.
pub(crate) static WRITTEN_ASKS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// (Neptune) Whether shaders were declared to write title memory in
/// `[address, address + len)` in this frame or the two before it.
pub(crate) fn shader_written(address: usize, len: usize) -> bool {
    let end = address.saturating_add(len);
    let written = SHADER_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    written.as_ref().is_some_and(|(_, ranges)| ranges.keys().any(|&(from, to)| from < end && address < to))
}

/// (Prefault job, `VICTORY_ARENA_BY_CHANGED`) [`shader_written`] split by
/// the declared extent: (a range of at most `big` bytes, or of a copy in
/// place, touches it; a range of more than `big` bytes of a compared copy
/// touches it).
pub(crate) fn shader_written_split(address: usize, len: usize, big: usize) -> (bool, bool) {
    // (P3, `VICTORY_CHEAP_DRAWS`) The answer depends on the set of ranges
    // alone, not on their frames (the sweep takes the old ones out, and
    // moves the count): an ask answered while the count stood is answered
    // the same way, without the lock and the walk over every range (the
    // arena's records are asked once per indirect call, from every
    // recording thread).
    type Answers = (u64, std::collections::HashMap<(usize, usize, usize), (bool, bool)>);
    thread_local! {
        static ANSWERS: std::cell::RefCell<Answers> = std::cell::RefCell::new((0, std::collections::HashMap::new()));
    }
    let cheap = cheap_draws();
    if cheap {
        let generation = SHADER_WRITTEN_GEN.load(Ordering::Acquire);
        let kept = ANSWERS.with(|a| {
            let a = a.borrow();
            if a.0 == generation { a.1.get(&(address, len, big)).copied() } else { None }
        });
        if let Some(answer) = kept {
            WRITTEN_ASKS[0].fetch_add(1, Ordering::Relaxed);
            return answer;
        }
    }
    let (answer, generation) = {
        let written = SHADER_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        // Read under the lock: the count of the set this answer is of.
        let generation = SHADER_WRITTEN_GEN.load(Ordering::Acquire);
        (shader_written_split_in(written.as_ref(), address, len, big), generation)
    };
    if cheap {
        WRITTEN_ASKS[1].fetch_add(1, Ordering::Relaxed);
        ANSWERS.with(|a| {
            let mut a = a.borrow_mut();
            if a.0 != generation || a.1.len() >= 4096 {
                a.0 = generation;
                a.1.clear();
            }
            a.1.insert((address, len, big), answer);
        });
    }
    answer
}

/// [`shader_written_split`] over the ranges as they are.
#[allow(clippy::type_complexity)]
fn shader_written_split_in(
    written: Option<&(u64, std::collections::HashMap<(usize, usize), (u64, bool)>)>,
    address: usize,
    len: usize,
    big: usize,
) -> (bool, bool) {
    let end = address.saturating_add(len);
    let Some((_, ranges)) = written else {
        return (false, false);
    };
    let (mut small, mut large) = (false, false);
    for (&(from, to), &(_, in_place)) in ranges {
        if from < end && address < to {
            if to - from > big && !in_place {
                large = true;
            } else {
                small = true;
            }
        }
    }
    (small, large)
}

/// (Prefault job, `VICTORY_ARENA_BY_CHANGED`) [`flights_write`] with the
/// ranges of more than `big` bytes left out (as the flights declared them
/// for copy-back, before they are joined).
pub(crate) fn flights_write_except_big(from: usize, to: usize, big: usize) -> bool {
    let line = line();
    line.line.iter().any(|item| match item {
        Late::Run(flight) => flight.back.iter().any(|(_, ranges)| ranges.iter().any(|&(a, b)| b - a <= big && a < to && from < b)),
        _ => false,
    })
}

/// Each range shaders were declared to write in this frame or the two
/// before it that touches `[from, to)`: its start, end and frame.
pub(crate) fn shader_written_each(from: usize, to: usize, mut each: impl FnMut(usize, usize, u64)) {
    let frame = crate::frame::current();
    let written = SHADER_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, ranges)) = written.as_ref() {
        for (&(a, b), &(at, _)) in ranges {
            if a < to && from < b && at + 2 >= frame {
                each(a, b, at);
            }
        }
    }
}

/// Whether title memory `[from, to)` lies in what flights in the air may
/// write (with [`late`]).
pub(crate) fn flights_write(from: usize, to: usize) -> bool {
    touches(&line().written, from, to)
}

/// Whether a constant buffer view (of a table, or a root one not on 256
/// bytes in its copy) whose title memory flights may write is filled on
/// the card, behind them ([`HostList::card_constants`]), instead of in
/// upload memory at submission, which waits for them to land or reads
/// ahead of them (`fill_ahead`). On unless `VICTORY_FILL_ON_CARD=0`.
/// Ported from Trexies 25fbcce/9dd3e8f: one such 480-512-byte buffer a
/// frame cost a wait of 52-80 ms there.
pub(crate) fn fill_on_card() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FILL_ON_CARD").is_none_or(|v| v != "0"))
}

/// Since the last frame line, done on the card behind the flights: table
/// constant fills, root constant fills ([`fill_on_card`]), title buffer
/// copies (`crate::list` `gpu_buffer_copy`).
pub(crate) static ON_CARD: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// `VICTORY_SETTLE_DIAG=1`: the first 80 settles a fill or buffer copy
/// causes, with what it reads and which copy holds that memory.
fn settle_diag(what: &str, src: usize, dst: usize, len: usize) {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_SETTLE_DIAG").is_some_and(|v| v == "1")) {
        return;
    }
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) >= 80 {
        return;
    }
    let copy = crate::shadow::find(src as u64, len as u64).map(|s| {
        let (a, l) = s.span();
        format!("{a:#x}+{l:#x}")
    });
    tracing::info!(
        src = format_args!("{src:#x}"),
        dst = format_args!("{dst:#x}"),
        len,
        copy = ?copy,
        "settle diag: a {what} reads what a flight writes"
    );
}

/// The parts of `ranges` (in order, joined) inside `[from, to)`.
fn clipped(ranges: &[(usize, usize)], from: usize, to: usize) -> Vec<(usize, usize)> {
    let at = ranges.partition_point(|r| r.1 <= from);
    ranges[at..].iter().take_while(|r| r.0 < to).map(|r| (r.0.max(from), r.1.min(to))).collect()
}

/// Whether `[from, to)` touches one of `ranges` (in order, joined).
fn touches(ranges: &[(usize, usize)], from: usize, to: usize) -> bool {
    if from >= to {
        return false;
    }
    let at = ranges.partition_point(|r| r.1 <= from);
    ranges.get(at).is_some_and(|r| r.0 < to)
}

/// `ranges` with `more`, in order, joined where they overlap or touch.
fn join_ranges(ranges: &mut Vec<(usize, usize)>, more: impl IntoIterator<Item = (usize, usize)>) {
    ranges.extend(more.into_iter().filter(|r| r.0 < r.1));
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(from, to) in ranges.iter() {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    *ranges = out;
}

/// Whether anything is in the air.
fn flying() -> bool {
    if !late() {
        return false;
    }
    let line = line();
    // (With the relay: also while flights are being landed.)
    line.settling || line.line.iter().any(|l| matches!(l, Late::Run(_)))
}

/// With flights in the air ([`late`]): why `runs` cannot be prepared
/// before they are settled, if so. Preparing reads title memory on the
/// CPU (the title's buffer copies, fills of upload memory, textures) and
/// runs steps of ours that may read or write it anywhere; what the
/// flights' shaders write is not there yet.
fn late_blocked(runs: &[&HostList]) -> Option<usize> {
    let written = {
        let line = line();
        if line.written.is_empty() {
            return None;
        }
        line.written.clone()
    };
    blocked_by(&written, runs, true)
}

/// [`late_blocked`] against `written`, the title memory flights' shaders
/// may write (in order, joined); `count` whether the kind of read is
/// counted for the frame line.
fn blocked_by(written: &[(usize, usize)], runs: &[&HostList], count: bool) -> Option<usize> {
    if written.is_empty() {
        return None;
    }
    for list in runs {
        // One hold of the list's steps for the whole look: the meter below
        // reads `not_card` through this guard (taking `deferred()` again on
        // this thread deadlocked the submit thread at the first fill that
        // waited after a fill-ahead trip: runs 63-70).
        let deferred = list.deferred();
        for step in &deferred.copies {
            match step {
                &Step::Copy(dst, src, len) => {
                    if touches(written, src, src + len) || touches(written, dst, dst + len) {
                        // A copy that only READS where flights may write is
                        // not waited for either, and checked like a fill
                        // (`fill_ahead`): run 210 looked at 197 such waits
                        // (20,480 bytes out of the 174 MB buffer into a small
                        // buffer of its own, four a frame, 43 to 60 ms a
                        // frame) and neither side had other bytes after
                        // one. A copy INTO such a range still waits.
                        if !touches(written, dst, dst + len) && fill_ahead::on(src, len) {
                            if count {
                                fill_ahead::read(src, len);
                            }
                            continue;
                        }
                        if count {
                            LATE_STATS[15].fetch_add(1, Ordering::Relaxed);
                            let into = touches(written, dst, dst + len);
                            read_wait::blocked(&deferred.not_card, read_wait::COPY, src, len, if into { 3 } else { fill_ahead::why_not(src, len) });
                            settle_diag("buffer copy", src, dst, len);
                            #[cfg(feature = "diagnostics")]
                            fill_check::blocked_copy(written, dst, src, len);
                        }
                        return Some(WHY_READ);
                    }
                }
                &Step::Fill(dst, src, len) => {
                    if touches(written, src, src + len) {
                        // Not waited for, and checked when the flights land
                        // (`fill_ahead`).
                        if fill_ahead::on(src, len) {
                            if count {
                                fill_ahead::read(src, len);
                            }
                            continue;
                        }
                        if count {
                            LATE_STATS[16].fetch_add(1, Ordering::Relaxed);
                            read_wait::blocked(&deferred.not_card, read_wait::FILL, src, len, fill_ahead::why_not(src, len));
                            settle_diag("fill", src, dst, len);
                            #[cfg(feature = "diagnostics")]
                            fill_check::blocked(written, src, len);
                        }
                        return Some(WHY_READ);
                    }
                }
                Step::Texture(staging) => {
                    let (at, len) = staging.memory();
                    if touches(written, at, at + len) {
                        if count {
                            LATE_STATS[17].fetch_add(1, Ordering::Relaxed);
                            read_wait::blocked(&deferred.not_card, read_wait::OTHER, at, len, 3);
                        }
                        return Some(WHY_READ);
                    }
                }
                Step::Run(_) => return Some(WHY_RUN),
                Step::Refresh(..) | Step::GpuWrite(..) | Step::Host(_) => {}
            }
        }
    }
    None
}

/// Fills of upload memory (a constant buffer a table names, copied into
/// the list's own memory when the list is submitted) that read title memory
/// flights "may write" are no longer waited for; what the flights then
/// write is checked against them, and the first fill that read bytes a
/// flight changed afterwards turns the waiting back on for good.
///
/// Why: a flight declares every range its shaders can store to, and the
/// title has an unordered view over the whole of its 174 MB buffer, whose
/// copy is in the card's memory. Its per-draw constants lie in the same
/// buffer, at new addresses every frame. So the first such fill of a frame
/// waited until those flights had landed: one wait a frame, 56 to 95 ms of
/// the submitting thread standing still (runs 202 and 206; 145 ms while
/// the world loaded in 208), the largest single wait of the frame. Run 208
/// looked at every one of them: 44 fills of 480 bytes, and after the wait
/// the bytes were the same 44 times.
///
/// What it risks, and how it is bounded: a fill that reads bytes a flight
/// in the air is about to change takes the old bytes, and the lists of
/// that one submission draw with them. `wrote` sees it when the flight
/// lands (the bytes going into title memory are compared with the ones
/// the fill read, over every fill that read ahead of them) and from then
/// on fills that read the 64 KiB pieces where that happened wait as before;
/// after `HELD_MAX` such pieces, every fill waits again.
/// `VICTORY_FILL_WAIT=1`: they wait from the start.
///
/// A fill is only taken for missed when title memory still holds what the
/// fill read (its fingerprint at the fill): bytes the title rewrote after
/// the fill say nothing about what the fill saw, and comparing with them
/// used to turn the waiting back on for good on the first such rewrite (one
/// settle of about 100 ms a frame for the rest of the run).
mod fill_ahead {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
    use std::sync::{Mutex, OnceLock};

    /// Fills are looked up by the 64 KiB of title memory they read.
    const PIECE: usize = 1 << 16;
    /// A fill is checked against flights that land this many frames after
    /// it at most (flights are settled at 256 lists in the air, and by
    /// every present that waits).
    const KEEP: u64 = 16;
    /// Pieces held back (fills reading them wait) before all fills wait
    /// (`VICTORY_FILL_AHEAD_PER_PIECE=0`, as before).
    const HELD_MAX: usize = 8;

    /// (P1, `VICTORY_FILL_AHEAD_PER_PIECE`, default 1; =0: the global
    /// switch above, as before) No global switch: a piece a flight changed
    /// after a fill read it ahead is held (its fills wait) for
    /// [`HOLD_FRAMES`] frames, then retried; at most [`HELD_MOST`] pieces
    /// are held, the expired and then the least recently tripped making room.
    /// Run-57/55: the ninth miss, 11 s in, made every fill wait for the rest
    /// of the run (7-12 ms of waiting a frame became 34 ms) although 84 % of
    /// the waits since found the bytes unchanged.
    pub(crate) fn per_piece() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            let on = std::env::var_os("VICTORY_FILL_AHEAD_PER_PIECE").is_none_or(|v| v != "0");
            tracing::info!(on, "VICTORY_FILL_AHEAD_PER_PIECE: fills read ahead of flights held back per 64 KiB piece (300 frames, then retried), never all at once");
            on
        })
    }

    /// Frames a tripped piece is held before it is retried.
    pub(super) const HOLD_FRAMES: u64 = 300;
    /// Pieces held at most.
    pub(super) const HELD_MOST: usize = 4096;
    /// WARN lines about misses at most (each piece once per hold).
    const WARN_MOST: u64 = 64;

    /// More than `HELD_MAX` pieces had a miss: every fill waits (only with
    /// `VICTORY_FILL_AHEAD_PER_PIECE=0`).
    static WAIT: AtomicBool = AtomicBool::new(false);
    /// Whether `HELD` has any piece (a cheap look before the lock).
    static ANY_HELD: AtomicBool = AtomicBool::new(false);

    /// The pieces where a flight changed bytes a fill had read ahead of it,
    /// with the frame each last tripped in: fills that read them wait (for
    /// good with the old switch, for [`HOLD_FRAMES`] per piece with P1).
    #[derive(Default)]
    pub(super) struct Held {
        pub(super) tripped: HashMap<usize, u64>,
        /// Pieces retried after their hold, since the start.
        pub(super) retried: u64,
    }

    impl Held {
        /// Whether `piece` is held in `frame` (`hold`: frames a trip holds,
        /// `u64::MAX` for good). A piece whose hold is over counts once as
        /// retried and is held again only by a new trip.
        pub(super) fn held(&mut self, piece: usize, frame: u64, hold: u64) -> bool {
            match self.tripped.get(&piece) {
                Some(&at) if frame < at.saturating_add(hold) => true,
                Some(_) => {
                    self.tripped.remove(&piece);
                    self.retried += 1;
                    false
                }
                None => false,
            }
        }

        /// `piece` tripped in `frame`; past `most` pieces, the expired and
        /// then the least recently tripped go. Whether it was not held then
        /// (a new trip, worth saying).
        pub(super) fn trip(&mut self, piece: usize, frame: u64, hold: u64, most: usize) -> bool {
            let new = self.tripped.get(&piece).is_none_or(|&at| frame >= at.saturating_add(hold));
            self.tripped.insert(piece, frame);
            if self.tripped.len() > most {
                self.tripped.retain(|_, at| frame < at.saturating_add(hold));
                while self.tripped.len() > most {
                    let Some((&oldest, _)) = self.tripped.iter().filter(|e| *e.0 != piece).min_by_key(|(_, at)| **at) else {
                        break;
                    };
                    self.tripped.remove(&oldest);
                }
            }
            new
        }
    }

    static HELD: Mutex<Option<Held>> = Mutex::new(None);
    /// Since the last frame line: fills that read ahead; landings' writes
    /// that touched one's piece. Since the start: fills that read bytes a
    /// flight then changed; fills not compared because the title had
    /// rewritten their bytes since; WARN lines said.
    static COUNT: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

    fn hold_frames() -> u64 {
        if per_piece() { HOLD_FRAMES } else { u64::MAX }
    }

    /// Why a fill of `len` bytes at `src` may not read ahead now: 0 it may,
    /// 1 a held piece, 2 waiting is on for every fill (`VICTORY_FILL_WAIT=1`,
    /// or the old global switch).
    pub(super) fn why_not(src: usize, len: usize) -> u8 {
        static ON: OnceLock<bool> = OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_FILL_WAIT").is_none_or(|v| v != "1")) || WAIT.load(Relaxed) {
            return 2;
        }
        if len == 0 || !ANY_HELD.load(Relaxed) {
            return 0;
        }
        let frame = crate::frame::current();
        let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
        let Some(held) = held.as_mut() else { return 0 };
        let hold = hold_frames();
        u8::from((src / PIECE..=(src + len - 1) / PIECE).any(|piece| held.held(piece, frame, hold)))
    }

    /// (Hang watch) Whether `HELD` and `READ` are held now (never waits).
    pub(super) fn locks() -> (&'static str, &'static str) {
        let held = |busy: bool| if busy { "HELD" } else { "free" };
        (
            held(matches!(HELD.try_lock(), Err(std::sync::TryLockError::WouldBlock))),
            held(matches!(READ.try_lock(), Err(std::sync::TryLockError::WouldBlock))),
        )
    }

    /// Pieces held now.
    pub(super) fn held_count() -> usize {
        HELD.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, |h| h.tripped.len())
    }

    /// Whether `[src, src + len)` touches a held piece now (P1's prediction
    /// for `VICTORY_FILL_PREDICT_CARD`).
    pub(crate) fn held_now(src: usize, len: usize) -> bool {
        why_not(src, len) == 1
    }

    /// The fingerprint of the title memory a fill reads.
    fn print(src: usize, len: usize) -> u64 {
        crate::texture::fingerprint(src as u64, len as u64)
    }

    #[derive(Default)]
    struct Piece {
        /// The frame its old fills were last dropped in.
        pruned: u64,
        /// From, to, frame, and the number of the last submission started
        /// when it read: the flights in the air then have that number or
        /// a lower one, and only their writes can be ones it missed; and
        /// the fingerprint of the bytes it read.
        fills: Vec<(usize, usize, u64, u64, u64)>,
    }

    /// The lowest submission number among the flights the landing thread
    /// is landing now; 0: not known (a landing by another way), and then
    /// every write counts. (Run 211: a read of frame 1381 was taken for
    /// missed because shaders wrote the same 80 bytes in frame 1397, in a
    /// flight sent sixteen frames after it.)
    static LANDING: AtomicU64 = AtomicU64::new(0);

    pub(super) fn landing(first: u64) {
        LANDING.store(first, Relaxed);
    }

    static READ: Mutex<Option<HashMap<usize, Piece>>> = Mutex::new(None);

    /// Whether a fill of `len` bytes at `src` may read ahead of flights.
    pub(super) fn on(src: usize, len: usize) -> bool {
        why_not(src, len) == 0
    }

    /// Fills that read `[from, to)` wait from now on (P1: for
    /// [`HOLD_FRAMES`]; old switch: for good, and every fill past
    /// `HELD_MAX` pieces). Whether to say so: the first time anything is
    /// held (old switch), or a piece newly held (P1, up to [`WARN_MOST`]).
    fn hold(from: usize, to: usize) -> bool {
        let frame = crate::frame::current();
        let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
        let held = held.get_or_insert_with(Held::default);
        let first = held.tripped.is_empty();
        let mut new = false;
        for piece in from / PIECE..=(to - 1) / PIECE {
            new |= held.trip(piece, frame, hold_frames(), if per_piece() { HELD_MOST } else { usize::MAX });
        }
        ANY_HELD.store(true, Relaxed);
        if per_piece() {
            return new && COUNT[4].fetch_add(1, Relaxed) < WARN_MOST;
        }
        if held.tripped.len() > HELD_MAX && !WAIT.swap(true, Relaxed) {
            tracing::warn!(
                "fills of upload memory read ahead of flights that changed their bytes in {} pieces of 64 KiB: every fill \
                 waits for flights again from here on, as with VICTORY_FILL_WAIT=1",
                held.tripped.len()
            );
        }
        first
    }

    /// A fill reads `len` bytes at `src` ahead of flights that may write
    /// them.
    pub(super) fn read(src: usize, len: usize) {
        if len == 0 {
            return;
        }
        COUNT[0].fetch_add(1, Relaxed);
        crate::framelog::add(crate::framelog::T::FillAhead, 1);
        let frame = crate::frame::current();
        let print = print(src, len);
        let mut read = READ.lock().unwrap_or_else(|e| e.into_inner());
        let map = read.get_or_insert_with(HashMap::new);
        for piece in src / PIECE..=(src + len - 1) / PIECE {
            let p = map.entry(piece).or_default();
            if p.pruned != frame {
                p.pruned = frame;
                p.fills.retain(|f| f.2 + KEEP >= frame);
            }
            p.fills.push((src, src + len, frame, super::STARTED.load(Relaxed), print));
        }
    }

    /// A landing is about to put `new` into title memory at `at`: bytes
    /// shaders wrote, merged into the title's.
    pub(crate) fn wrote(at: usize, new: &[u8]) {
        if new.is_empty() || WAIT.load(Relaxed) {
            return;
        }
        let frame = crate::frame::current();
        let landing = LANDING.load(Relaxed);
        let mut hits: Vec<(usize, usize, u64, u64)> = {
            let read = READ.lock().unwrap_or_else(|e| e.into_inner());
            let Some(map) = read.as_ref() else { return };
            let end = at + new.len();
            (at / PIECE..=(end - 1) / PIECE)
                .filter_map(|piece| map.get(&piece))
                .flat_map(|p| p.fills.iter().copied())
                .filter(|f| f.0 < end && f.1 > at && f.2 + KEEP >= frame && (landing == 0 || landing <= f.3))
                .map(|f| (f.0, f.1, f.2, f.4))
                .collect()
        };
        if hits.is_empty() {
            return;
        }
        // A fill that spans pieces is kept in each of them.
        hits.sort_unstable();
        hits.dedup();
        COUNT[1].fetch_add(1, Relaxed);
        let mut now = Vec::new();
        for (from, to, read_in, print_then) in hits {
            // The title rewrote the bytes since the fill (or a landing before
            // this one did, and was compared then): what is there now is not
            // what the fill read, so it cannot tell a miss.
            if print(from, to - from) != print_then {
                COUNT[3].fetch_add(1, Relaxed);
                continue;
            }
            let (a, b) = (from.max(at), to.min(at + new.len()));
            now.clear();
            now.resize(b - a, 0);
            // Title memory as it is: what the fill read. Not there: nothing
            // to say.
            if !super::read_into(a as *const u8, &mut now) || now[..] == new[a - at..b - at] {
                continue;
            }
            let n = COUNT[2].fetch_add(1, Relaxed) + 1;
            crate::framelog::add(crate::framelog::T::FillAheadChanged, 1);
            let first = now.iter().zip(&new[a - at..b - at]).position(|(x, y)| x != y).unwrap_or(0);
            if hold(from, to) {
                if per_piece() {
                    tracing::warn!(
                        "a fill of upload memory read {} bytes at {from:#x} in frame {read_in} ahead of a flight whose shaders changed \
                         them (first at {:#x}, landed in frame {frame}): the lists of that submission drew with the bytes from before. \
                         Fills that read its 64 KiB wait for flights for the next {HOLD_FRAMES} frames, then it is retried \
                         (VICTORY_FILL_AHEAD_PER_PIECE=1; {n} such fills; said once per piece and hold, at most {WARN_MOST} times)",
                        to - from,
                        a + first
                    );
                } else {
                    tracing::warn!(
                        "a fill of upload memory read {} bytes at {from:#x} in frame {read_in} ahead of a flight whose shaders changed \
                         them (first at {:#x}, landed in frame {frame}): the lists of that submission drew with the bytes from before. \
                         Fills that read its 64 KiB wait for flights from here on; past {HELD_MAX} such pieces, every fill does \
                         ({n} such fills)",
                        to - from,
                        a + first
                    );
                }
            }
        }
    }

    /// With the frame line.
    pub(super) fn line(frames: f64) {
        let [ahead, looked] = [0, 1].map(|i| COUNT[i].swap(0, Relaxed));
        if ahead == 0 && looked == 0 {
            return;
        }
        let kept = {
            let frame = crate::frame::current();
            let mut read = READ.lock().unwrap_or_else(|e| e.into_inner());
            read.as_mut().map_or(0, |map| {
                map.retain(|_, p| {
                    p.fills.retain(|f| f.2 + KEEP >= frame);
                    !p.fills.is_empty()
                });
                map.values().map(|p| p.fills.len()).sum::<usize>()
            })
        };
        // Read once, before the line: a guard taken inside the macro's
        // arguments lives to the end of the statement, and a second lock
        // there (the retried count) deadlocked this thread with
        // `VICTORY_FILL_AHEAD_PER_PIECE=1`, the landing behind it (runs 68-70).
        let (held_now, retried) = HELD.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or((0, 0), |h| (h.tripped.len(), h.retried));
        tracing::info!(
            "fills of upload memory that read title memory flights may write, not waited for (VICTORY_FILL_WAIT=1 waits): {:.1} per \
             frame; {:.1} landings' writes per frame fell in the same 64 KiB as one and were compared with it; fills that read \
             bytes a flight then changed, since the start: {} ({} pieces of 64 KiB held back{}{}); fills not compared because the \
             title had rewritten their bytes since: {}; {kept} fills kept to check",
            ahead as f64 / frames,
            looked as f64 / frames,
            COUNT[2].load(Relaxed),
            held_now,
            if per_piece() {
                format!(" now, {retried} retried after {HOLD_FRAMES} frames, VICTORY_FILL_AHEAD_PER_PIECE=1")
            } else {
                String::new()
            },
            if WAIT.load(Relaxed) { ", every fill waits again" } else { "" },
            COUNT[3].load(Relaxed)
        );
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn held(at: usize) -> bool {
            HELD.lock().unwrap().as_ref().is_some_and(|h| h.tripped.contains_key(&(at / PIECE)))
        }

        /// (P1) Trip -> held -> retried after the hold; the set is bounded,
        /// the expired and then the least recently tripped making room, and
        /// a piece tripped again while held is not news.
        #[test]
        fn pieces_are_held_then_retried_and_bounded() {
            let mut h = Held::default();
            assert!(!h.held(7, 100, 300));
            assert!(h.trip(7, 100, 300, 4), "a new trip");
            assert!(h.held(7, 101, 300) && h.held(7, 399, 300));
            assert!(!h.trip(7, 150, 300, 4), "tripped again while held: not news");
            assert!(h.held(7, 449, 300), "the hold runs from the last trip");
            assert!(!h.held(7, 450, 300), "retried after the hold");
            assert_eq!(h.retried, 1);
            assert!(!h.tripped.contains_key(&7));
            assert!(h.trip(7, 460, 300, 4), "tripped again after a retry: news");
            // Bound: 4 pieces; the oldest trip makes room.
            for (p, f) in [(1, 461), (2, 462), (3, 463)] {
                h.trip(p, f, 300, 4);
            }
            assert_eq!(h.tripped.len(), 4);
            h.trip(9, 470, 300, 4);
            assert_eq!(h.tripped.len(), 4);
            assert!(!h.tripped.contains_key(&7), "the least recently tripped went");
            assert!(h.held(9, 471, 300) && h.held(1, 471, 300));
            // Expired pieces go first.
            let mut h = Held::default();
            h.trip(1, 0, 300, 2);
            h.trip(2, 400, 300, 2);
            h.trip(3, 401, 300, 2);
            assert!(!h.tripped.contains_key(&1) && h.tripped.contains_key(&2) && h.tripped.contains_key(&3));
            // For good (the old switch): never retried.
            let mut h = Held::default();
            h.trip(5, 0, u64::MAX, usize::MAX);
            assert!(h.held(5, 1_000_000, u64::MAX));
        }

        /// One test: the module's state is shared by the whole process.
        #[test]
        fn only_real_misses_hold_fills_back() {
            const LEN: usize = 480;
            // Three pieces of 64 KiB of their own, one per case.
            let mut memory = vec![0u8; 4 * PIECE];
            let start = memory.as_ptr() as usize;
            let base = start.next_multiple_of(PIECE) - start;
            let at = |i: usize| start + base + i * PIECE;
            let (rewritten, missed, same) = (at(0), at(1), at(2));
            for i in 0..3 {
                memory[base + i * PIECE..][..LEN].fill(1);
            }

            // The title rewrites the bytes after the fill, then a flight's
            // shaders land other bytes there: the fill read the 1s, nothing
            // can be said, and nothing is held back (this used to turn
            // waiting on for good).
            read(rewritten, LEN);
            memory[base..][..LEN].fill(2);
            wrote(rewritten, &[3; LEN]);
            assert!(!held(rewritten), "a rewrite by the title is not a miss");
            assert!(on(rewritten, LEN));

            // Nothing rewrote them and the shaders changed them: a miss;
            // fills of that piece wait, fills elsewhere do not.
            read(missed, LEN);
            wrote(missed, &[3; LEN]);
            assert!(held(missed), "a real miss holds its piece back");
            assert!(!on(missed, LEN));
            assert!(on(same, LEN));
            assert!(!WAIT.load(Relaxed), "one piece does not make every fill wait");
            assert_eq!(why_not(missed, LEN), 1);

            // The shaders wrote what was there already: no miss.
            read(same, LEN);
            wrote(same, &[1; LEN]);
            assert!(!held(same));
            assert!(on(same, LEN));
        }
    }
}

pub(crate) use fill_ahead::wrote as shaders_write;

/// (P1, meter 1 of docs/frame-pipeline-analysis.md) The submit thread's
/// waits for a CPU read of title memory behind flights (`WHY_READ`): by
/// kind, their time split into "the card still running the flights" and
/// "the landing after the card had finished", whether the bytes read had
/// changed after the wait, flights in the air at it, why the read was a
/// waiting fill or copy and not done on the card. Totals go to the frame
/// log (usual and long frame lines); [`line`] says them per second.
pub(crate) mod read_wait {
    use crate::framelog::{T, add, total};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    /// Kinds of waiting reads.
    pub(super) const FILL: u8 = 1;
    pub(super) const COPY: u8 = 2;
    pub(super) const OTHER: u8 = 3;

    /// Why a step was not done on the card (recorded with it): 0 not known
    /// (a way with no card path), switch off (`VICTORY_FILL_ON_CARD=0`,
    /// `VICTORY_GPU_BUFFER_COPY=0`, or not sent ahead), no flight in the
    /// air declared it at record time, it went for the card and then: no
    /// copy holds it, `card_holds` refused, no room.
    pub(crate) const NC_SWITCH: u8 = 1;
    pub(crate) const NC_UNDECLARED: u8 = 2;
    /// Wanted the card; a later failure says which.
    pub(crate) const NC_LATER: u8 = 3;
    pub(crate) const NC_NO_COPY: u8 = 4;
    pub(crate) const NC_HOLDS: u8 = 5;
    pub(crate) const NC_ROOM: u8 = 6;

    thread_local! {
        static WHY: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    }

    /// The card path's decision for the step about to be recorded.
    pub(crate) fn why(why: u8) {
        WHY.with(|w| w.set(why));
    }

    /// The decision as it stands.
    pub(crate) fn pending() -> u8 {
        WHY.with(std::cell::Cell::get)
    }

    /// The decision, taken by the step recorded now (0 after).
    pub(super) fn take_why() -> u8 {
        WHY.with(|w| w.replace(0))
    }

    /// `VICTORY_FILL_PREDICT_CARD` (default 0; 1 on): fills and buffer
    /// copies go on the card also where no flight in the air declared their
    /// bytes, if a shader is known to rewrite them (a held piece, or a
    /// declaration in this frame or the two before).
    pub(crate) fn predict_card() -> bool {
        predict_mode() != 0
    }

    /// `VICTORY_FILL_PREDICT_CARD`: 0 (default) off; 1 a held piece or a
    /// declaration in this frame or the two before; (P6) 2 only a piece
    /// held now (tripped) or a source that made a submission wait in this
    /// frame or the two before ([`waited_lately`]): the fills of a shader-
    /// written arena that never made anyone wait keep reading ahead on the
    /// CPU (run-69 with 1 put 1,800-8,500 fills a frame on the card for 5-7
    /// held pieces).
    pub(crate) fn predict_mode() -> u8 {
        static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        *MODE.get_or_init(|| {
            let mode = match std::env::var("VICTORY_FILL_PREDICT_CARD").ok().as_deref().map(str::trim) {
                Some("1") => 1,
                Some("2") => 2,
                _ => 0,
            };
            tracing::info!(mode, "VICTORY_FILL_PREDICT_CARD: fills and copies of bytes shaders rewrite go on the card by prediction (1: held or declared lately; 2: held now or waited for lately), not only when a flight in the air declared them");
            mode
        })
    }

    /// (P6) Since the start, fills and copies put on the card by mode 2:
    /// for a held piece, for a source waited for lately.
    pub(crate) static PREDICTED2: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

    /// (P6) Per 64 KiB piece of title memory: the frame a fill or copy of
    /// it last made a submission wait ([`begin`]).
    static WAITED: Mutex<Option<std::collections::HashMap<usize, u64>>> = Mutex::new(None);
    const WAITED_PIECE: usize = 1 << 16;

    fn waited_note(src: usize, len: usize) {
        if predict_mode() != 2 || len == 0 {
            return;
        }
        let frame = crate::frame::current();
        let mut w = WAITED.lock().unwrap_or_else(|e| e.into_inner());
        let w = w.get_or_insert_with(Default::default);
        if w.len() > 4096 {
            w.retain(|_, f| *f + 2 >= frame);
        }
        for piece in src / WAITED_PIECE..=(src + len - 1) / WAITED_PIECE {
            w.insert(piece, frame);
        }
    }

    /// (P6) Whether `[src, src + len)` touches a piece that made a
    /// submission wait in this frame or the two before.
    pub(crate) fn waited_lately(src: usize, len: usize) -> bool {
        if len == 0 {
            return false;
        }
        let frame = crate::frame::current();
        let w = WAITED.lock().unwrap_or_else(|e| e.into_inner());
        let Some(w) = w.as_ref() else {
            return false;
        };
        (src / WAITED_PIECE..=(src + len - 1) / WAITED_PIECE).any(|p| w.get(&p).is_some_and(|&f| f + 2 >= frame))
    }

    /// Whether `[src, src + len)` is predicted rewritten by shaders: mode 1
    /// a held piece, or declared in this frame or the two before; mode 2 a
    /// piece held now, or a source waited for lately (counted).
    pub(crate) fn predicted(src: usize, len: usize) -> bool {
        if predict_mode() == 2 {
            if super::fill_ahead::held_now(src, len) {
                PREDICTED2[0].fetch_add(1, Relaxed);
                return true;
            }
            if waited_lately(src, len) {
                PREDICTED2[1].fetch_add(1, Relaxed);
                return true;
            }
            return false;
        }
        super::fill_ahead::held_now(src, len) || super::shader_written(src, len)
    }

    /// The step that made this submission wait: kind, source, bytes, why
    /// it could not read ahead (1 held piece, 2 every fill waits, 3 no
    /// read-ahead rule: a copy into a declared range, a texture), why it
    /// was not on the card.
    static BLOCKED: Mutex<(u8, usize, usize, u8, u8)> = Mutex::new((0, 0, 0, 0, 0));
    /// When the relay last found flights the card had run (ns since START).
    static CARD_DONE_NS: AtomicU64 = AtomicU64::new(0);
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

    fn now_ns() -> u64 {
        START.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64
    }

    pub(super) fn card_done() {
        CARD_DONE_NS.store(now_ns(), Relaxed);
    }

    /// (Hang watch) Whether `BLOCKED` is held now (never waits).
    pub(super) fn lock_state() -> &'static str {
        if matches!(BLOCKED.try_lock(), Err(std::sync::TryLockError::WouldBlock)) { "HELD" } else { "free" }
    }

    /// `not_card`: the list's [`super::Deferred::not_card`], read through
    /// the guard the caller already holds (never `deferred()` again here).
    pub(super) fn blocked(not_card: &[(usize, usize, u8)], kind: u8, src: usize, len: usize, ahead: u8) {
        let nc = not_card.iter().rev().find(|&&(s, n, _)| s == src && n == len).map_or(0, |e| e.2);
        *BLOCKED.lock().unwrap_or_else(|e| e.into_inner()) = (kind, src, len, ahead, nc);
    }

    /// A wait about to start.
    pub(super) struct Wait {
        at: u64,
        kind: u8,
        src: usize,
        bytes: Option<Vec<u8>>,
    }

    pub(super) fn begin() -> Wait {
        let (kind, src, len, ahead, nc) = *BLOCKED.lock().unwrap_or_else(|e| e.into_inner());
        if kind == FILL || kind == COPY {
            waited_note(src, len);
        }
        add(T::ReadWaits, 1);
        add(
            match kind {
                FILL => T::ReadWaitFill,
                COPY => T::ReadWaitCopy,
                _ => T::ReadWaitOther,
            },
            1,
        );
        if kind == FILL || kind == COPY {
            add(
                match ahead {
                    1 => T::FillWaitHeld,
                    3 => T::FillWaitNoPrediction,
                    _ => T::FillWaitOther,
                },
                1,
            );
            add(
                match nc {
                    NC_SWITCH => T::NotCardSwitch,
                    NC_UNDECLARED => T::NotCardUndeclared,
                    NC_NO_COPY => T::NotCardNoCopy,
                    NC_HOLDS => T::NotCardHolds,
                    NC_ROOM | NC_LATER => T::NotCardRoom,
                    _ => T::NotCardOther,
                },
                1,
            );
            if nc == NC_UNDECLARED && super::shader_written(src, len) {
                add(T::NotCardShaderWritten, 1);
            }
        }
        {
            let line = super::line();
            let flights = line.line.iter().filter(|l| matches!(l, super::Late::Run(_))).count();
            add(T::ReadWaitFlights, flights as u64);
            add(T::ReadWaitLists, line.lists as u64);
        }
        let bytes = (len > 0 && len <= 1 << 16 && kind != OTHER).then(|| {
            let mut b = vec![0u8; len];
            super::read_into(src as *const u8, &mut b).then_some(b)
        });
        Wait { at: now_ns(), kind, src, bytes: bytes.flatten() }
    }

    pub(super) fn end(w: Wait) {
        let now = now_ns();
        let all = now.saturating_sub(w.at);
        let done = CARD_DONE_NS.load(Relaxed);
        // Until the card had run the last flights landed in the wait: the
        // card's; after: the landing's.
        let card = if done > w.at { (done - w.at).min(all) } else { 0 };
        add(T::ReadWaitUs, all / 1000);
        add(T::ReadWaitCardUs, card / 1000);
        add(T::ReadWaitLandUs, (all - card) / 1000);
        if let Some(before) = w.bytes {
            let mut after = vec![0u8; before.len()];
            if super::read_into(w.src as *const u8, &mut after) {
                add(if after != before { T::ReadChangedYes } else { T::ReadChangedNo }, 1);
            }
        }
        let _ = w.kind;
    }

    /// The text of the read waits from the frame log's values `t` (one
    /// frame, a mean of frames, or a sum).
    pub(crate) fn text(t: impl Fn(T) -> f64, held: usize) -> String {
        format!(
            "read waits {:.1} (fill {:.1}, copy {:.1}, other {:.1}), {:.1} ms (card still running the flights {:.1}, landing after the card had finished {:.1}); \
             bytes had changed after the wait: yes {:.1}, no {:.1}; flights in the air at the waits {:.1} (lists {:.0}); fills ahead {:.0}, of them later found \
             changed {:.1} (pieces held {held}); fills/copies that waited {:.1} (held piece {:.1}, no prediction {:.1}, other {:.1}); not on the card because: \
             switch off {:.1}, no flight declared it at record time {:.1} (shader-written in the 3 frames {:.1}), no copy holds it {:.1}, card_holds refused \
             {:.1}, no room {:.1}, no card path {:.1}; put on the card by prediction {:.0} (VICTORY_FILL_AHEAD_PER_PIECE={}, VICTORY_FILL_PREDICT_CARD={}; \
             mode 2 since the start: held piece {}, waited for lately {})",
            t(T::ReadWaits),
            t(T::ReadWaitFill),
            t(T::ReadWaitCopy),
            t(T::ReadWaitOther),
            t(T::ReadWaitUs) / 1e3,
            t(T::ReadWaitCardUs) / 1e3,
            t(T::ReadWaitLandUs) / 1e3,
            t(T::ReadChangedYes),
            t(T::ReadChangedNo),
            t(T::ReadWaitFlights),
            t(T::ReadWaitLists),
            t(T::FillAhead),
            t(T::FillAheadChanged),
            t(T::FillWaitHeld) + t(T::FillWaitNoPrediction) + t(T::FillWaitOther),
            t(T::FillWaitHeld),
            t(T::FillWaitNoPrediction),
            t(T::FillWaitOther),
            t(T::NotCardSwitch),
            t(T::NotCardUndeclared),
            t(T::NotCardShaderWritten),
            t(T::NotCardNoCopy),
            t(T::NotCardHolds),
            t(T::NotCardRoom),
            t(T::NotCardOther),
            t(T::PredictedCard),
            u8::from(super::fill_ahead::per_piece()),
            predict_mode(),
            PREDICTED2[0].load(Relaxed),
            PREDICTED2[1].load(Relaxed),
        )
    }

    /// Pieces held back now.
    pub(crate) fn held_pieces() -> usize {
        super::fill_ahead::held_count()
    }

    /// Per second (with the frame-rate line): per frame over `frames`.
    pub(super) fn line(frames: f64) {
        const KEYS: [T; 25] = [
            T::ReadWaits,
            T::ReadWaitFill,
            T::ReadWaitCopy,
            T::ReadWaitOther,
            T::ReadWaitUs,
            T::ReadWaitCardUs,
            T::ReadWaitLandUs,
            T::ReadChangedYes,
            T::ReadChangedNo,
            T::ReadWaitFlights,
            T::ReadWaitLists,
            T::FillAhead,
            T::FillAheadChanged,
            T::FillWaitHeld,
            T::FillWaitNoPrediction,
            T::FillWaitOther,
            T::NotCardSwitch,
            T::NotCardUndeclared,
            T::NotCardShaderWritten,
            T::NotCardNoCopy,
            T::NotCardHolds,
            T::NotCardRoom,
            T::NotCardOther,
            T::PredictedCard,
            T::Count,
        ];
        static LAST: [AtomicU64; 24] = [const { AtomicU64::new(0) }; 24];
        let d: Vec<f64> =
            KEYS[..24].iter().zip(&LAST).map(|(&k, l)| { let n = total(k); n.saturating_sub(l.swap(n, Relaxed)) as f64 / frames.max(1.0) }).collect();
        if d.iter().all(|&x| x == 0.0) {
            return;
        }
        let at = |k: T| KEYS.iter().position(|&x| x as usize == k as usize).map_or(0.0, |i| d[i]);
        tracing::info!("read waits per frame over {frames:.0} frames: {}", text(at, held_pieces()));
    }
}

/// Diagnostics: the fills of upload memory that made a submission wait for
/// flights (`blocked_by`), which one they are, and whether the wait was
/// needed: the bytes the fill reads are kept when it blocks and compared
/// with what it reads once the flights have landed.
///
/// Runs 202 and 206: one such wait a frame, 56 to 95 ms of the submitting
/// thread standing still, the largest single wait of the frame. The ranges
/// flights "may write" are whole views shaders can store to; if the bytes
/// a fill reads never change under the wait, the wait is for nothing.
#[cfg(feature = "diagnostics")]
mod fill_check {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    /// The fill waited for now: source, length, its bytes when it blocked.
    static PENDING: Mutex<Option<(usize, usize, Vec<u8>)>> = Mutex::new(None);
    /// Fills that blocked by (source page, length): how often.
    static SEEN: Mutex<Vec<(usize, usize, u64)>> = Mutex::new(Vec::new());
    /// Fills checked after their wait; those whose bytes had changed; the
    /// bytes that had.
    static CHECKED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

    /// A fill of `len` bytes at `src` blocks on `written`.
    pub(super) fn blocked(written: &[(usize, usize)], src: usize, len: usize) {
        let at = written.partition_point(|r| r.1 <= src);
        let touched = written.get(at).copied().unwrap_or((0, 0));
        let key = (src & !0xfff, len);
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let known = seen.iter().position(|s| (s.0, s.1) == key);
        match known {
            Some(k) => seen[k].2 += 1,
            None if seen.len() < 256 => {
                seen.push((key.0, key.1, 1));
                if seen.len() <= 12 {
                    tracing::info!(
                        "a fill of upload memory waits for flights: {len} bytes at {src:#x}; the range shaders may write that it                          touches: {:#x}+{:#x} ({} ranges in the air); frame {}",
                        touched.0,
                        touched.1 - touched.0,
                        written.len(),
                        crate::frame::current()
                    );
                }
            }
            None => {}
        }
        drop(seen);
        let mut bytes = vec![0u8; len.min(1 << 20)];
        if super::read_into(src as *const u8, &mut bytes) {
            *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((src, len, bytes));
        }
    }

    /// The fill of `len` bytes at `src` runs now.
    pub(super) fn runs(src: usize, len: usize) {
        let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        if !pending.as_ref().is_some_and(|p| (p.0, p.1) == (src, len)) {
            return;
        }
        let Some((_, _, before)) = pending.take() else { return };
        drop(pending);
        let mut now = vec![0u8; before.len()];
        if !super::read_into(src as *const u8, &mut now) {
            return;
        }
        let differ = before.iter().zip(&now).filter(|(a, b)| a != b).count() as u64;
        let n = CHECKED[0].fetch_add(1, Relaxed) + 1;
        if differ > 0 {
            let bad = CHECKED[1].fetch_add(1, Relaxed) + 1;
            CHECKED[2].fetch_add(differ, Relaxed);
            if bad <= 12 {
                let first = before.iter().zip(&now).position(|(a, b)| a != b).unwrap_or(0);
                tracing::info!(
                    "a fill of upload memory that waited for flights read other bytes after the wait: {differ} of {len} bytes at                      {src:#x} changed, the first at +{first:#x} (check {n}, frame {})",
                    crate::frame::current()
                );
            }
        }
    }

    /// The title buffer copy waited for now: destination, source, length,
    /// the bytes of both when it blocked.
    static PENDING_COPY: Mutex<Option<(usize, usize, usize, Vec<u8>, Vec<u8>)>> = Mutex::new(None);
    /// Copies checked after their wait; those whose source had changed;
    /// those whose destination had.
    static COPIES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
    /// Copies that blocked by (source page, destination page, length).
    static SEEN_COPIES: Mutex<Vec<(usize, usize, usize, u64)>> = Mutex::new(Vec::new());

    /// A title buffer copy of `len` bytes from `src` to `dst` blocks on
    /// `written`.
    pub(super) fn blocked_copy(written: &[(usize, usize)], dst: usize, src: usize, len: usize) {
        let by_src = super::touches(written, src, src + len);
        let by_dst = super::touches(written, dst, dst + len);
        let key = (src & !0xfff, dst & !0xfff, len);
        let mut seen = SEEN_COPIES.lock().unwrap_or_else(|e| e.into_inner());
        let known = seen.iter().position(|s| (s.0, s.1, s.2) == key);
        match known {
            Some(k) => seen[k].3 += 1,
            None if seen.len() < 256 => {
                seen.push((key.0, key.1, key.2, 1));
                if seen.len() <= 16 {
                    let range = |at: usize| {
                        let k = written.partition_point(|r| r.1 <= at);
                        written.get(k).copied().unwrap_or((0, 0))
                    };
                    let (s, d) = (range(src), range(dst));
                    tracing::info!(
                        "a title buffer copy waits for flights: {len} bytes from {src:#x} to {dst:#x}; source in a range shaders may                          write: {by_src} ({:#x}+{:#x}); destination: {by_dst} ({:#x}+{:#x}); {} ranges in the air; frame {}",
                        s.0,
                        s.1 - s.0,
                        d.0,
                        d.1 - d.0,
                        written.len(),
                        crate::frame::current()
                    );
                }
            }
            None => {}
        }
        drop(seen);
        let n = len.min(1 << 20);
        let (mut a, mut b) = (vec![0u8; n], vec![0u8; n]);
        if super::read_into(src as *const u8, &mut a) && super::read_into(dst as *const u8, &mut b) {
            *PENDING_COPY.lock().unwrap_or_else(|e| e.into_inner()) = Some((dst, src, len, a, b));
        }
    }

    /// The title buffer copy runs now.
    pub(super) fn copy_runs(dst: usize, src: usize, len: usize) {
        let mut pending = PENDING_COPY.lock().unwrap_or_else(|e| e.into_inner());
        if !pending.as_ref().is_some_and(|p| (p.0, p.1, p.2) == (dst, src, len)) {
            return;
        }
        let Some((_, _, _, src_before, dst_before)) = pending.take() else { return };
        drop(pending);
        let (mut a, mut b) = (vec![0u8; src_before.len()], vec![0u8; dst_before.len()]);
        if !super::read_into(src as *const u8, &mut a) || !super::read_into(dst as *const u8, &mut b) {
            return;
        }
        let n = COPIES[0].fetch_add(1, Relaxed) + 1;
        let (src_changed, dst_changed) = (a != src_before, b != dst_before);
        if src_changed {
            COPIES[1].fetch_add(1, Relaxed);
        }
        if dst_changed {
            COPIES[2].fetch_add(1, Relaxed);
        }
        if (src_changed || dst_changed) && COPIES[1].load(Relaxed) + COPIES[2].load(Relaxed) <= 16 {
            let differ = |x: &[u8], y: &[u8]| x.iter().zip(y).filter(|(p, q)| p != q).count();
            tracing::info!(
                "a title buffer copy that waited for flights found other bytes after the wait: {len} bytes from {src:#x} to {dst:#x};                  {} source bytes changed, {} destination bytes changed (check {n}, frame {})",
                differ(&a, &src_before),
                differ(&b, &dst_before),
                crate::frame::current()
            );
        }
    }

    /// With the frame line.
    pub(super) fn line() {
        let copies = COPIES[0].load(Relaxed);
        if copies > 0 {
            let seen = SEEN_COPIES.lock().unwrap_or_else(|e| e.into_inner());
            let mut most: Vec<_> = seen.clone();
            most.sort_unstable_by_key(|s| std::cmp::Reverse(s.3));
            let most: Vec<String> =
                most.iter().take(6).map(|s| format!("{:#x}->{:#x}+{:#x} {}x", s.0, s.1, s.2, s.3)).collect();
            tracing::info!(
                "title buffer copies that waited for flights, since the start: {copies} checked after their wait; the source had                  other bytes after it {} times, the destination {} times; {} distinct copies, the most frequent: {}",
                COPIES[1].load(Relaxed),
                COPIES[2].load(Relaxed),
                seen.len(),
                most.join(", ")
            );
        }
        let n = CHECKED[0].load(Relaxed);
        if n == 0 {
            return;
        }
        let seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let mut most: Vec<_> = seen.clone();
        most.sort_unstable_by_key(|s| std::cmp::Reverse(s.2));
        let most: Vec<String> = most.iter().take(6).map(|s| format!("{:#x}+{:#x} {}x", s.0, s.1, s.2)).collect();
        tracing::info!(
            "fills of upload memory that waited for flights, since the start: {n} checked after their wait, {} of them read other              bytes than before it ({} bytes in all); {} distinct fills (page, length), the most frequent: {}",
            CHECKED[1].load(Relaxed),
            CHECKED[2].load(Relaxed),
            seen.len(),
            most.join(", ")
        );
    }
}

/// With flights in the air, after the submission's write-watch asking:
/// whether `runs` refresh a range of a host copy that a flight's shaders
/// may write and that the title wrote since it was last copied. The
/// refresh leaves ranges shaders are writing alone (`Shadow::update`), so
/// the title's bytes there would reach the card a submission late.
fn stale_under_flights(runs: &[&HostList]) -> bool {
    use std::sync::Arc;
    // Per copy, the span of what is refreshed (a flush brings that much
    // up to date at most).
    let mut spans: Vec<(Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::Refresh(shadow, address, len) = step
                && shadow.writes_pending()
            {
                match spans.iter_mut().find(|s| Arc::ptr_eq(&s.0, shadow)) {
                    Some(s) => (s.1, s.2) = (s.1.min(*address), s.2.max(*address + *len)),
                    None => spans.push((shadow.clone(), *address, *address + *len)),
                }
            }
        }
    }
    if late_exact() {
        // The title's bytes are put in beside the flights' writes; a
        // settle is still due where that could not be done. At most this
        // many changed runs go up to copies in the card's memory, each a
        // copy of its own there.
        // (`VICTORY_MERGE_RUNS`: the cap. Run 218 had 500 to 5,400 changed
        // runs a frame of a few bytes each, and the cap of 512 settled the
        // flights twice a frame, 50 ms of the submitting thread.)
        static RUNS: OnceLock<usize> = OnceLock::new();
        let mut runs_left = *RUNS.get_or_init(|| {
            // (Neptune) `VICTORY_UNDER_WRITES_RUNS` is the same cap there
            // (512 by default; its launcher sets 100,000): read when
            // `VICTORY_MERGE_RUNS` is not set.
            let num = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<usize>().ok());
            num("VICTORY_MERGE_RUNS").or_else(|| num("VICTORY_UNDER_WRITES_RUNS")).map_or(512, |n| n.clamp(1, 1 << 17))
        });
        // Pages over the cap go up masked (`crate::shadow::merge_mask`).
        let mut avoided = false;
        let settle = spans.iter().any(|(shadow, from, to)| shadow.merge_under_writes(*from, *to - *from, &mut runs_left, &mut avoided));
        if avoided && !settle {
            crate::shadow::mask_settle_avoided();
        }
        return settle;
    }
    spans.iter().any(|(shadow, from, to)| shadow.stale_under_writes(*from, *to - *from))
}

/// A fence signal on a queue ([`late`]): goes behind the flights in the
/// air, if there are any, and is set when they are settled. Whether it
/// did; if not the caller sets the fence now.
pub(crate) fn late_signal(fence: usize, value: u64) -> bool {
    if !late() {
        return false;
    }
    let mut line = line();
    if line.line.is_empty() && !line.settling {
        return false;
    }
    line.line.push(Late::Signal(fence, value));
    LATE_STATS[14].fetch_add(1, Ordering::Relaxed);
    drop(line);
    // A signal alone (behind a settle under way) has no flight to wake
    // the helper thread for it.
    LINE_MOVED.notify_one();
    RELAY_MOVED.notify_one();
    true
}

/// A queue stands at `wait`, which memory does not meet now ([`late`]):
/// whether it is met once the flights in the air have run. A flight's
/// bottom-of-pipe write of a value that meets it says yes without a wait
/// for the card: the part behind the wait goes to the card behind that
/// flight, which is the order the wait asks for. If the flights may write
/// the value in a way not known here (shaders, a timestamp, another
/// width), they are settled and memory is asked again.
pub(crate) fn late_met(wait: &crate::list::ValueWait) -> bool {
    let (address, bytes) = wait.place();
    {
        let line = line();
        if line.line.is_empty() && line.in_hand.is_empty() {
            return false;
        }
        let mut unsure = touches(&line.written, address, address + bytes);
        // (With the relay: first the writes of flights being landed.)
        let writes = line.in_hand.iter().map(|(at, what, n)| (at, what, n)).chain(line.line.iter().filter_map(|item| match item {
            Late::Write(at, what, n) => Some((at, what, n)),
            _ => None,
        }));
        for (at, what, n) in writes {
            if *at >= address + bytes || address >= *at + *n {
                continue;
            }
            match what {
                crate::queue::Written::Value(v) if *at == address && *n >= bytes => {
                    if wait.holds_for(*v) {
                        LATE_STATS[13].fetch_add(1, Ordering::Relaxed);
                        return true;
                    }
                }
                _ => unsure = true,
            }
        }
        if !unsure {
            return false;
        }
    }
    settle_now(WHY_VALUE);
    wait.met()
}

/// [`late_met`] without its settle: whether a bottom-of-pipe write of a
/// flight in the air, or of one being landed, is known to meet `wait`.
/// False says nothing about what else the flights may write there.
/// (`VICTORY_JOIN_PARTS`, fxceless 3973179.)
pub(crate) fn late_met_known(wait: &crate::list::ValueWait) -> bool {
    let (address, bytes) = wait.place();
    let line = line();
    let writes = line.in_hand.iter().map(|(at, what, n)| (at, what, n)).chain(line.line.iter().filter_map(|item| match item {
        Late::Write(at, what, n) => Some((at, what, n)),
        _ => None,
    }));
    for (at, what, n) in writes {
        if let crate::queue::Written::Value(v) = what
            && *at == address
            && *n >= bytes
            && wait.holds_for(*v)
        {
            LATE_STATS[13].fetch_add(1, Ordering::Relaxed);
            return true;
        }
    }
    false
}

/// Settles the flights in the air, if any ([`Host::settle`]), for reason
/// `why` (one of the `WHY_` places). Not from inside a submission.
pub(crate) fn settle_now(why: usize) {
    if !late() {
        return;
    }
    let Some(host) = host() else {
        return;
    };
    let mut submitted = {
        let _site = crate::hangwatch::at(&"settle_now (present, fence or value wait): waiting for the submission lock");
        host.submitted.lock().unwrap_or_else(|e| e.into_inner())
    };
    host.settle(&mut submitted, why);
}

/// [`settle_now`] at a present, and for a fence the title blocks on.
pub(crate) fn settle_for_present() {
    settle_now(WHY_SYNC);
}
pub(crate) fn settle_for_fence() {
    settle_now(WHY_FENCE);
}

/// Resets the host lists of settled flights and keeps them for reuse, as
/// a queue does for the lists it ran (`crate::list::recycle`).
fn land() {
    let landed = std::mem::take(&mut line().landed);
    for (lists, order) in landed {
        crate::list::recycle(lists, &order);
    }
}

/// The thread that settles the flights once the card has run them all
/// ([`late`]): without it a bottom-of-pipe write would reach the title only
/// at the next call that settles, and a title thread that polls such a
/// value while the submitting thread waits for that thread would never
/// see it.
fn late_helper() {
    let Some(host) = host() else {
        return;
    };
    crate::framelog::keep_on_fast_cores("victory-late-wait");
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        tracing::warn!("no event for the thread that settles submissions sent ahead");
        return;
    };
    loop {
        let target = {
            let _site = crate::hangwatch::at(&"late helper: idle, waiting for a poke or flights (LINE_MOVED)");
            let mut line = line();
            loop {
                if line.poke {
                    line.poke = false;
                    break None;
                }
                // (With the relay the flights are another thread's.)
                if !relay() && !line.line.is_empty() && !line.settling {
                    break Some(line.target);
                }
                line = LINE_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
            }
        };
        let Some(target) = target else {
            let _site = crate::hangwatch::at(&"late helper: resetting landed lists, looking at what queues hold");
            land();
            crate::queue::release_held(false);
            continue;
        };
        // The card first, without the submission lock: the submitting
        // thread prepares and sends meanwhile.
        // SAFETY: our fence and event.
        unsafe {
            let _site = crate::hangwatch::at(&"late helper: waiting for the card (fence)");
            if host.fence.SetEventOnCompletion(target, event).is_ok() {
                WaitForSingleObject(event, INFINITE);
            }
        }
        let _site = crate::hangwatch::at(&"late helper: settling (submission lock, then settle, land, queues)");
        {
            let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            // Something newer went up meanwhile: the card is waited for
            // again, the submitting thread is not held up for it.
            if line().target != target {
                continue;
            }
            host.settle(&mut submitted, WHY_CARD_DONE);
        }
        land();
        crate::queue::release_held(false);
    }
}

/// Starts the threads of [`late`], once: the one that settles flights when
/// the card has run them and looks at what queues hold, and with [`relay`]
/// the one that lands flights.
fn late_threads() {
    static HELPER: std::sync::Once = std::sync::Once::new();
    HELPER.call_once(|| {
        if let Err(e) = std::thread::Builder::new().name("victory-late-wait".into()).spawn(late_helper) {
            tracing::warn!(error = %e, "no thread to settle submissions sent ahead: they are settled by the calls that follow");
        }
        tracing::info!(
            most = late_most(),
            writes = late_writes(),
            "submissions are sent to the card without a wait for each (VICTORY_LATE_WAIT=1)"
        );
        if relay() {
            // Without it nothing would ever land.
            if let Err(e) = std::thread::Builder::new().name("victory-relay".into()).spawn(relay_worker) {
                fail(format_args!("no thread to land submissions sent ahead (VICTORY_RELAY=1): {e}"));
            }
            tracing::info!(
                exact = relay_exact(),
                "what shaders wrote is copied back by a thread of its own, beside the submitting thread (VICTORY_RELAY=1)"
            );
        }
    });
}

/// (Hang watch) Presents so far.
static PRESENTS_SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn presents_seen() -> u64 {
    PRESENTS_SEEN.load(Ordering::Relaxed)
}

/// (Hang watch) The layer's locks held now (looked at with `try_lock`, so
/// never waited for) and where the flights stand.
pub(crate) fn hang_state() -> String {
    fn held<T>(m: &Mutex<T>) -> &'static str {
        match m.try_lock() {
            Ok(_) => "free",
            Err(std::sync::TryLockError::Poisoned(_)) => "free (poisoned)",
            Err(std::sync::TryLockError::WouldBlock) => "HELD",
        }
    }
    let line = match LINE.try_lock() {
        Ok(line) => {
            let flights = line.line.iter().filter(|l| matches!(l, Late::Run(_))).count();
            let first = line.line.iter().find_map(|l| match l {
                Late::Run(f) => Some(f.fence),
                _ => None,
            });
            format!(
                "line free: flights in the air {flights} (lists {}, first waits for fence value {}), settling {}, being landed {}, \
                 written ranges {}, landed lists not yet reset {}",
                line.lists,
                first.map_or_else(|| "-".to_string(), |v| v.to_string()),
                line.settling,
                line.in_hand.len(),
                line.written.len(),
                line.landed.len()
            )
        }
        Err(std::sync::TryLockError::WouldBlock) => "line HELD".to_string(),
        Err(std::sync::TryLockError::Poisoned(_)) => "line poisoned".to_string(),
    };
    let (submission, completed) = host().map_or(("no host", 0), |h| {
        // SAFETY: a plain query.
        (held(&h.submitted), unsafe { h.fence.GetCompletedValue() })
    });
    format!(
        "{line}; submission lock {submission}; fence completed {completed}; submissions started {}, finished {}; fill-ahead \
         held-pieces lock {}, reads lock {}; read-wait lock {}; shader-written lock {}",
        STARTED.load(Ordering::Relaxed),
        FINISHED.load(Ordering::Relaxed),
        fill_ahead::locks().0,
        fill_ahead::locks().1,
        read_wait::lock_state(),
        held(&SHADER_WRITTEN)
    )
}

/// With [`relay`]: waits until `done` says so of the line, looked at each
/// time flights have landed. Counted as a settle for `why`, and its time
/// as waiting for the card. The caller may hold the submission lock: the
/// thread that lands flights never takes it.
fn relay_wait(why: usize, done: impl Fn(&Line) -> bool) {
    let mut line = line();
    if done(&line) {
        return;
    }
    LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
    LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
    let was_working = working(false);
    landing(true);
    let waiting = std::time::Instant::now();
    while !done(&line) {
        line = LINE_LANDED.wait(line).unwrap_or_else(|e| e.into_inner());
    }
    drop(line);
    landing(false);
    stood(waiting);
    working(was_working);
    let us = waiting.elapsed().as_micros() as u64;
    LATE_STATS[12].fetch_add(us, Ordering::Relaxed);
    LATE_WAITED[why.min(10)].fetch_add(us, Ordering::Relaxed);
    submission_us(1, us);
}

/// What the thread that lands flights fetches card copies' pieces with
/// ([`relay`]): a list, a fence and an event of its own (the host's are
/// behind the submission lock), and the memory the pieces come down to.
struct RelayFetch {
    list: Prelude,
    fence: ID3D12Fence,
    event: HANDLE,
    /// The last value signalled on `fence`.
    sent: u64,
    /// Buffer, mapping, bytes.
    buffer: Option<(ID3D12Resource, usize, u64)>,
}

/// The thread that lands flights ([`relay`]): as soon as the card has run
/// the first flights of the line it takes them, with what follows them,
/// and does what they left to do ([`Host::relay_land`]); else it waits for
/// the card, or for a flight. It takes neither the submission lock nor
/// anything a caller of [`Host::settle`] may hold: resetting the landed
/// lists and looking at what queues hold are left to [`late_helper`].
fn relay_worker() {
    // (Neptune) This thread's jobs have kept threads of their own.
    crate::shadow::wide_for_back();
    let Some(host) = host() else {
        return;
    };
    crate::framelog::keep_on_fast_cores("victory-relay");
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        fail(format_args!("no event for the thread that lands submissions sent ahead (VICTORY_RELAY=1)"));
    };
    let mut fetch: Option<RelayFetch> = None;
    loop {
        let next = {
            let _site = crate::hangwatch::at(&"relay: waiting for a flight (RELAY_MOVED) or the line lock");
            let mut line = line();
            loop {
                if line.line.is_empty() {
                    line = RELAY_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
                    continue;
                }
                let fences: Vec<Option<u64>> = line
                    .line
                    .iter()
                    .map(|l| match l {
                        Late::Run(flight) => Some(flight.fence),
                        _ => None,
                    })
                    .collect();
                // SAFETY: a plain query.
                let completed = unsafe { host.fence.GetCompletedValue() };
                let n = landable(&fences, completed);
                if n == 0 {
                    // The first thing is a flight the card has not run.
                    break Err(fences[0].unwrap_or(0));
                }
                line.settling = true;
                // (P1 meter) The card has run these: what a read wait waits
                // for from here on is the landing.
                read_wait::card_done();
                let items: Vec<Late> = line.line.drain(..n).collect();
                line.in_hand = items
                    .iter()
                    .filter_map(|l| match l {
                        Late::Write(at, what, bytes) => Some((*at, *what, *bytes)),
                        _ => None,
                    })
                    .collect();
                break Ok(items);
            }
        };
        match next {
            Ok(items) => {
                let _site = crate::hangwatch::at(&"relay: landing flights (relay_land: copy-back, fill-ahead checks, read-backs, signals)");
                host.relay_land(items, &mut fetch);
            }
            Err(target) => {
                let _site = crate::hangwatch::at(&"relay: waiting for the card to run the first flight (fence)");
                let waiting = std::time::Instant::now();
                // SAFETY: the host's fence and this thread's event.
                let waited = unsafe {
                    host.fence.SetEventOnCompletion(target, event).map(|()| {
                        WaitForSingleObject(event, INFINITE);
                    })
                };
                if waited.is_err() {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                RELAY_STATS[2].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
        }
    }
}

/// [`relay`]: what shaders wrote to the declared ranges `back` of copies
/// in the PC's memory goes into title memory. As [`hand_back`], each byte
/// once, the pieces that differ found on several threads; but the merge is
/// `Shadow::relay_merge`, which does not write the copy, and the ranges
/// are not done with here (each flight's are, afterwards).
fn relay_hand_back(back: &[Declared]) {
    let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    for (shadow, declared) in back {
        if shadow.carded() {
            continue;
        }
        let mut spans: Vec<(usize, usize)> = Vec::new();
        join_ranges(&mut spans, declared.iter().copied());
        for &(from, to) in &spans {
            shadow.relay_compared(from, to - from);
        }
        work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
    }
    if work.is_empty() {
        return;
    }
    #[cfg(feature = "diagnostics")]
    let scanning = std::time::Instant::now();
    // Looked through without a lock: a piece that changes meanwhile (a
    // later flight writing, the title's bytes going in) is found now or by
    // that flight's landing, and the merge looks again under the lock.
    let found = crate::shadow::find_all(&work);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::BackScanUs, scanning.elapsed().as_micros() as u64);
    let (mut now, mut title) = (Vec::new(), Vec::new());
    for ((shadow, from, to), found) in work.iter().zip(found) {
        let mut written = 0;
        for (at, n) in found {
            written += shadow.relay_merge(at, n, &mut now, &mut title);
        }
        copy_back(1, written as u64);
        #[cfg(feature = "diagnostics")]
        crate::copystat::compared(*from, to - from, written);
        let _ = (from, to);
    }
}

/// For the frame-rate line, with [`late`]: what the line of flights did
/// per frame since the last one.
fn late_line(frames: f64) {
    FRAMES_SEEN.fetch_add(frames as u64, Ordering::Relaxed);
    if crate::shadow::lookup_cache() {
        let found = [0, 1].map(|i| crate::shadow::LOOKUPS[i].swap(0, Ordering::Relaxed) as f64 / frames.max(1.0));
        tracing::info!(
            "asks for a buffer's host copy per frame: {:.0} answered from the asking thread's memory of them, {:.0} from the tables (VICTORY_LOOKUP_CACHE)",
            found[0],
            found[1]
        );
    }
    if crate::shadow::refresh_compare() {
        let [read, taken, runs] = crate::shadow::compared_stats().map(|n| n as f64 / frames.max(1.0));
        tracing::info!(
            "refreshes that compare per frame (VICTORY_REFRESH_COMPARE=1): {:.1} MB of pages written since their last \
             copy read, {:.1} MB of them differed from the copy and were taken, in {runs:.0} runs",
            read / (1 << 20) as f64,
            taken / (1 << 20) as f64
        );
    }
    if !late() {
        return;
    }
    #[cfg(feature = "diagnostics")]
    fill_check::line();
    fill_ahead::line(frames);
    read_wait::line(frames);
    let s: Vec<f64> = LATE_STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "submissions sent ahead per frame: {:.1} in {:.1} settles ({:.1} flights each time), {:.1} ms waiting for the card in them; \
         settled for: a synchronous submission or present {:.1}, the card had run them {:.1}, a wait for a memory value {:.1}, \
         title memory changed under shader writes {:.1}, title memory read {:.1} (by a buffer copy {:.1}, a fill of upload memory {:.1}, \
         a texture {:.1}), a step of ours {:.1}, {} lists in the air {:.1}, \
         bottom-of-pipe writes (VICTORY_LATE_WAIT_WRITES=0) {:.1}, a blocking fence wait {:.1}; \
         {:.1} waits for memory values met by a flight's bottom-of-pipe write, {:.1} fence signals set behind flights",
        s[0],
        s[1],
        if s[1] > 0.0 { s[11] / s[1] } else { 0.0 },
        s[12] / 1000.0,
        s[WHY_SYNC],
        s[WHY_CARD_DONE],
        s[WHY_VALUE],
        s[WHY_STALE],
        s[WHY_READ],
        s[15],
        s[16],
        s[17],
        s[WHY_RUN],
        late_most(),
        s[WHY_MOST],
        s[WHY_WRITES],
        s[WHY_FENCE],
        s[13],
        s[14],
    );
    {
        let c: Vec<f64> = ON_CARD.iter().map(|c| c.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        tracing::info!(
            "done on the card behind the flights per frame instead of settling: {:.1} table constant fills, \
             {:.1} root constant fills (VICTORY_FILL_ON_CARD), {:.1} title buffer copies (VICTORY_GPU_BUFFER_COPY)",
            c[0],
            c[1],
            c[2],
        );
    }
    {
        let w: Vec<f64> = LATE_WAITED.iter().map(|w| w.swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames).collect();
        tracing::info!(
            "of the waiting for the card in settles, per frame, by what settled: a synchronous submission or present {:.1} ms, \
             the card had run them {:.1}, a wait for a memory value {:.1}, title memory changed under shader writes {:.1}, \
             title memory read {:.1}, a step of ours {:.1}, lists in the air {:.1}, bottom-of-pipe writes {:.1}, a blocking fence wait {:.1}",
            w[WHY_SYNC],
            w[WHY_CARD_DONE],
            w[WHY_VALUE],
            w[WHY_STALE],
            w[WHY_READ],
            w[WHY_RUN],
            w[WHY_MOST],
            w[WHY_WRITES],
            w[WHY_FENCE],
        );
    }
    if relay() {
        let r: Vec<f64> = RELAY_STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        let [again, noted] = [0, 1].map(|i| BESIDE_STATS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        tracing::info!(
            "the relay per frame (VICTORY_RELAY={}): {:.1} flights landed in {:.1} landings by the thread that copies back, \
             beside the submitting thread; that thread waited {:.1} ms for the card with nothing to land and {:.1} ms for \
             fetches of card copies' pieces, and was {:.1} ms landing in all (the frame line's \"copying back\" is its time; \
             \"settles\" above are waits of other threads for it); pieces fetched on the copy queue beside later \
             submissions: {noted:.1}, pieces those fetched again at their own landing whatever the compare said: {again:.1} \
             (VICTORY_QUEUE2_REFETCH)",
            if relay_exact() { 2 } else { 1 },
            r[1],
            r[0],
            r[2] / 1000.0,
            r[3] / 1000.0,
            r[4] / 1000.0,
        );
        if inflight_fetch() {
            let i: Vec<f64> = INFLIGHT_STATS.iter().map(|i| i.swap(0, Ordering::Relaxed) as f64 / frames).collect();
            tracing::info!(
                "card copies' pieces fetched inside their flights per frame (VICTORY_CARD_FETCH_INFLIGHT=1): {:.1} pieces \
                 ({:.2} MB) merged with no fetch of their own; {:.1} landings fetched the old way (the wait above), {:.1} of \
                 them because a flight's pieces overflowed the slots",
                i[0],
                i[1] / (1 << 20) as f64,
                i[3],
                i[2],
            );
            let b: Vec<f64> = BACK_FALLBACK.iter().map(|b| b.swap(0, Ordering::Relaxed) as f64 / frames).collect();
            let mb = |v: f64| v / (1 << 20) as f64;
            tracing::info!(
                "copy-back without a wait per frame (VICTORY_BACK_ASYNC={}): fetches landed without a wait {:.1} pieces \
                 {:.2} MB; fell back to a fetch and a wait: slots overflowed {:.1} ({:.2} MB), not compared {:.1} ({:.2} MB), \
                 no ticket {:.1} ({:.2} MB), fetched beside on the copy queue {:.1} ({:.2} MB), other {:.1} ({:.2} MB); \
                 slots {} MB (at most {} MB), resized {} times since the start",
                u8::from(back_async()),
                i[0],
                mb(i[1]),
                b[0],
                mb(b[5]),
                b[1],
                mb(b[6]),
                b[2],
                mb(b[7]),
                b[3],
                mb(b[8]),
                b[4],
                mb(b[9]),
                INFLIGHT_BYTES[0].load(Ordering::Relaxed) >> 20,
                back_slots_bytes()[1] >> 20,
                INFLIGHT_BYTES[1].load(Ordering::Relaxed),
            );
        }
    }
    if late_exact() {
        let [pages, runs, bytes] = crate::shadow::merged_stats().map(|n| n as f64 / frames);
        let why = crate::shadow::merge_why_stats().map(|n| n as f64 / frames);
        let [masked, mask_bytes, avoided] = crate::shadow::masked_stats().map(|n| n as f64 / frames);
        let [merges, upload] = [0, 1].map(|i| MASK_STATS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        tracing::info!(
            "title bytes put in under flights' shader writes per frame (VICTORY_LATE_WAIT=2): {pages:.0} pages looked at, \
             {runs:.0} changed runs, {:.2} MB; could not, and settled, per frame: copy not write-watched {:.1}, nothing \
             could say what the title wrote {:.1}, over VICTORY_MERGE_RUNS {:.1}, no snapshot {:.1}; masked pages \
             (VICTORY_MERGE_MASK={}): {masked:.1} pages, {:.2} MB of title bytes in their masks, {merges:.1} masked merges \
             on the card ({:.2} MB of masks), settles avoided {avoided:.2}; mask buffers alive {} ({} MB)",
            bytes / (1 << 20) as f64,
            why[0],
            why[1],
            why[2],
            why[3],
            u8::from(crate::shadow::merge_mask()),
            mask_bytes / (1 << 20) as f64,
            upload / (1 << 20) as f64,
            MASK_STATS[2].load(Ordering::Relaxed),
            MASK_STATS[3].load(Ordering::Relaxed) >> 20,
        );
    }
}

#[cfg(test)]
mod late_tests {
    use super::{continues, join_ranges, touches};

    /// Copies join only when both ends continue and the whole does not
    /// overlap itself.
    #[test]
    fn copies_join_only_end_to_end_and_apart() {
        // (destination, source, bytes)
        assert!(continues((1000, 5000, 16), (1016, 5016, 32)));
        assert!(continues((5000, 1000, 16), (5016, 1016, 32)));
        // A gap, or only one side continues.
        assert!(!continues((1000, 5000, 16), (1020, 5016, 32)));
        assert!(!continues((1000, 5000, 16), (1016, 5020, 32)));
        assert!(!continues((1000, 5000, 16), (1000, 5000, 16)));
        // Together they would overlap: the second reads what the first wrote.
        assert!(!continues((1000, 1016, 16), (1016, 1032, 16)));
        assert!(!continues((1016, 1000, 16), (1032, 1016, 16)));
        // They touch without overlapping.
        assert!(continues((1000, 1032, 16), (1016, 1048, 16)));
        assert!(!continues((usize::MAX - 8, 0, 8), (0, 8, 8)));
    }

    /// Ranges are found by what they hold, not by where they start, and a
    /// range that only touches another's end does not touch it.
    #[test]
    fn ranges_touch_where_they_overlap() {
        let mut ranges = Vec::new();
        join_ranges(&mut ranges, [(40, 50), (10, 20), (20, 25), (45, 60), (7, 7)]);
        assert_eq!(ranges, [(10, 25), (40, 60)]);
        join_ranges(&mut ranges, [(25, 40)]);
        assert_eq!(ranges, [(10, 60)]);
        let ranges = [(10usize, 25usize), (40, 60), (100, 101)];
        for (from, to, hit) in [
            (0, 10, false),
            (0, 11, true),
            (24, 25, true),
            (25, 40, false),
            (25, 41, true),
            (59, 200, true),
            (60, 100, false),
            (100, 101, true),
            (101, 500, false),
            (30, 30, false),
            (50, 50, false),
        ] {
            assert_eq!(touches(&ranges, from, to), hit, "{from}..{to}");
        }
        assert!(!touches(&[], 0, usize::MAX));
    }
}

/// Whether a title buffer copy that continues the one recorded before it
/// in its list becomes one step with it (`VICTORY_JOIN_COPIES=1`; off by
/// default).
///
/// Each copy is done at submission through the kernel, which reports a
/// page that is not there instead of faulting (`copy_mapped`), tells the
/// write watch and the guards, and ends the refreshes gathered so far: a
/// call into the kernel for some 480 bytes, 14,042 times a frame in run 86
/// (6.8 MB; "moves 19 ms" with the refreshes' own reads). How many of them
/// continue the one before is counted either way and said on its own line
/// with the frame rate; the switch is worth that many kernel calls.
///
/// Joined, the same bytes move: [`continues`] joins only copies whose
/// ranges, taken together, do not overlap, and nothing is recorded between
/// two steps that follow each other. What differs: a page that is not
/// mapped now fails the joined copy as a whole (it then goes page by page,
/// zeros for what cannot be read, as a single copy does), where before
/// only the copy that held it did.
fn join_copies() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOIN_COPIES").is_some_and(|v| v == "1"))
}

/// Whether a buffer-to-texture copy whose source this list's shaders write
/// takes those bytes from the memory's host copy, on the card, in the
/// list's order ([`HostList::staged_written_source`];
/// `VICTORY_COPY_WRITTEN_SOURCE=0`: from title memory at submission, as
/// before). On by default (allforks a9b474f, PalmVIDev 4d1ce258).
pub(crate) fn copy_written_source() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COPY_WRITTEN_SOURCE").is_none_or(|v| v != "0"))
}

/// Buffer-to-texture copies whose source this list's shaders write, staged
/// on the card ([`HostList::staged_written_source`]), since the start.
pub(crate) static STAGED_WRITTEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Since the last frame-rate line: title buffer copies recorded, and those
/// that continue the one before ([`join_copies`]).
static COPIES_JOINED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether the copy `next` (destination, source, bytes) goes on exactly
/// where `last` ended, source and destination both, and the two as one do
/// not overlap (source over destination): then one copy of both moves the
/// bytes the two would move in turn.
fn continues(last: (usize, usize, usize), next: (usize, usize, usize)) -> bool {
    let ((d, s, n), (dst, src, len)) = (last, next);
    let Some(all) = n.checked_add(len) else {
        return false;
    };
    d.checked_add(n) == Some(dst)
        && s.checked_add(n) == Some(src)
        && (d.checked_add(all).is_some_and(|end| end <= s) || s.checked_add(all).is_some_and(|end| end <= d))
}

/// The ranges the shaders of `runs` may write, per host copy, as declared.
/// One list's lock at a time: a submission may name a list twice.
fn declared_writes(runs: &[&HostList]) -> Vec<Declared> {
    let mut back: Vec<Declared> = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::GpuWrite(shadow, address, len) = step {
                let range = (*address, *address + *len);
                match back.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                    Some((_, ranges)) => {
                        if !ranges.contains(&range) {
                            ranges.push(range);
                        }
                    }
                    None => back.push((shadow.clone(), vec![range])),
                }
            }
        }
    }
    back
}

/// The walls trace (`VICTORY_WALLS_TRACE=1` or `2`; off by default): log
/// lines that say what the title's CPU was given while it decided whether
/// the camera is in a room. The values of the small read-backs
/// ([`read_backs`]), the frame's main pass by scissor rectangle with its
/// culling dispatches (`draw.rs`, `walls_census`), the shader-written
/// buffers merged into title memory (`shadow.rs`, `walls_merged`), host
/// copies found stale after their refresh (`shadow.rs`; only with
/// `VICTORY_REFRESH_CHECK=1` as well: it costs about 100 ms a frame), and, in the
/// runtime, the read requests that fail (with 2: every request).
pub(crate) fn walls_trace() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WALLS_TRACE").is_some_and(|v| v == "1" || v == "2"))
}

/// After `list` has run: the texels its read-backs copied into host
/// buffers of ours go to the title's memory (`HostList::defer_read_back`).
///
/// A submission that is waited for did this itself. One sent ahead
/// ([`late`]) did not, at any stage: the flights' lists were settled
/// without it, so the title read a buffer nobody wrote, as before read-backs
/// were made (the exposure metering's two small float targets, every frame).
fn read_backs(list: &HostList) {
    for (buffer, at, [row, rows, pitch]) in &list.deferred().backs {
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer of ours, written by the list that has
        // run; the title's destination holds every row (checked when the
        // copy was recorded).
        unsafe {
            if buffer.Map(0, None, Some(&mut data)).is_ok() && !data.is_null() {
                let from = data.cast::<u8>().cast_const();
                for line in 0..*rows {
                    std::ptr::copy_nonoverlapping(from.add(line * row.next_multiple_of(256)), (*at as *mut u8).add(line * pitch), *row);
                }
                // The walls trace: what the title is handed (one row of at
                // most 24 values: its 5x1 and 1x1 float targets).
                if walls_trace() && *rows == 1 && *row <= 96 {
                    let mut values = [0f32; 24];
                    std::ptr::copy_nonoverlapping(from, values.as_mut_ptr().cast::<u8>(), *row & !3);
                    tracing::info!(
                        frame = crate::frame::current(),
                        to = format_args!("{:#x}", *at),
                        bytes = *row,
                        values = ?&values[..*row / 4],
                        "walls trace: a read-back lands in title memory"
                    );
                }
                buffer.Unmap(0, None);
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    if crate::draw::probe::on() {
        let backs: Vec<usize> = list.deferred().backs.iter().map(|b| b.0.as_raw() as usize).collect();
        crate::draw::probe::landed(&backs);
    }
}

/// After the card ran what declared `back`: what shaders wrote there goes
/// into title memory, and the ranges are done with.
fn hand_back(back: &[Declared]) {
    if crate::shadow::wide_back() {
        // Every range first (the pieces that differ, found on several
        // threads), then the merges, in order, as below.
        let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
        for (shadow, declared) in back {
            if shadow.carded() {
                continue;
            }
            let mut sorted = declared.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
        }
        #[cfg(feature = "diagnostics")]
        let scanning = std::time::Instant::now();
        let found = crate::shadow::find_all(&work);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackScanUs, scanning.elapsed().as_micros() as u64);
        for ((shadow, from, to), found) in work.iter().zip(found) {
            let written = shadow.copy_back_with(*from, to - from, Some(found));
            copy_back(1, written as u64);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(*from, to - from, written);
        }
        for (shadow, declared) in back {
            shadow.copy_back_done(declared);
        }
        return;
    }
    for (shadow, declared) in back {
        if shadow.carded() {
            // Done before (`Host::card_back`).
            shadow.copy_back_done(declared);
            continue;
        }
        // Each byte compared once: the world declares dozens of views
        // of its 174 MB heap that overlap almost entirely, and comparing
        // every one whole was gigabytes a frame (0.3-0.5 s) for the few
        // MB shaders wrote. Only ranges that overlap or touch are
        // joined: what lies between two was not given to shaders (the
        // snapshot does not hold it).
        let mut sorted = declared.clone();
        sorted.sort_unstable();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (from, to) in sorted {
            match spans.last_mut() {
                Some(last) if from <= last.1 => last.1 = last.1.max(to),
                _ => spans.push((from, to)),
            }
        }
        for (from, to) in spans {
            let written = shadow.copy_back(from, to - from);
            copy_back(1, written as u64);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
        }
        shadow.copy_back_done(declared);
    }
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// Whether the host GPU supports the depth bounds test; logged once.
    pub(crate) fn depth_bounds(&self) -> bool {
        static BOUNDS: OnceLock<bool> = OnceLock::new();
        *BOUNDS.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
            // SAFETY: writes `options` of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS2,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let supported = ok && options.DepthBoundsTestSupported.as_bool();
            tracing::info!(supported, "host depth bounds support");
            supported
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        crate::census::made(8, 0);
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let before = private_bytes();
        let list = self.make_list();
        LISTS[0].fetch_add(1, Ordering::Relaxed);
        // (May shrink while a list is made, on another thread's account:
        // added as it is, the sum is what counts.)
        #[cfg(feature = "diagnostics")]
        LISTS[2].fetch_add(private_bytes().wrapping_sub(before), Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::HostLists, 1);
            add(C::HostListNs, started.elapsed().as_nanos() as u64);
        }
        list
    }

    /// An allocator and a closed list for a prelude.
    fn make_prelude(&self) -> Option<Prelude> {
        // SAFETY: plain object creation.
        let made: windows::core::Result<Prelude> = unsafe {
            (|| {
                let allocator: ID3D12CommandAllocator =
                    self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList =
                    self.device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                list.Close()?;
                Ok((allocator, list))
            })()
        };
        LISTS[1].fetch_add(1, Ordering::Relaxed);
        match made {
            Ok(prelude) => Some(prelude),
            Err(e) => {
                tracing::warn!(error = %e, "no host list for a prelude");
                None
            }
        }
    }

    fn make_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude = OnceLock::new();
            if !lazy_prelude() {
                let _ = prelude.set(self.make_prelude());
            }
            Ok(HostList {
                allocator,
                list,
                changes: std::sync::atomic::AtomicU64::new(1),
                serial: std::sync::atomic::AtomicU64::new(next_serial()),
                refreshes: std::sync::atomic::AtomicU64::new(0),
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                aliases: Mutex::default(),
                written: Mutex::default(),
                prelude,
                pipelines: Mutex::default(),
                budget: OnceLock::new(),
                card_budget: OnceLock::new(),
                indirect_args: Mutex::default(),
                card_constants: Mutex::default(),
                card_copied: Mutex::default(),
                window_check: OnceLock::new(),
                _alive: Counted::new(),
                coming: Mutex::default(),
                meter: AtomicU32::new(u32::MAX),
                pipeline_meter: AtomicU32::new(u32::MAX),
                gather: Mutex::default(),
                kinds: Mutex::default(),
                kinds_open: std::sync::atomic::AtomicBool::new(false),
                hoisted: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// The card's clock ([`CARD_CLOCK_US`]), made at the first send.
    fn card_clock(&self) -> Option<&'static CardClock> {
        static CLOCK: OnceLock<Option<CardClock>> = OnceLock::new();
        static WATCH: std::sync::Once = std::sync::Once::new();
        let clock = CLOCK
            .get_or_init(|| {
                // SAFETY: a plain fence on our device.
                let fence = unsafe { self.device.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) }.ok()?;
                Some(CardClock {
                    fence,
                    state: Mutex::new((0, None)),
                    sent: std::sync::Condvar::new(),
                })
            })
            .as_ref()?;
        WATCH.call_once(|| {
            if let Err(e) = std::thread::Builder::new().name("victory-card-clock".into()).spawn(move || card_clock_watch(clock)) {
                tracing::warn!(error = %e, "no thread for the card's clock: the frame line's idle times stay 0");
            }
        });
        Some(clock)
    }

    /// Lists of ours just went to the queue: the clock's signal goes
    /// behind them, and an idle stretch of the card, if one was under way,
    /// ends here ([`CARD_CLOCK_US`]).
    fn card_sent(&self) {
        // (`VICTORY_CARD_AWAKE`, trustno1 5f86fa3) The card's clocks kept up
        // between submissions while they go out.
        awake::sent(&self.device);
        let Some(clock) = self.card_clock() else {
            return;
        };
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        state.0 += 1;
        // SAFETY: our queue and the clock's fence.
        let _ = unsafe { self.queue.Signal(&clock.fence, state.0) };
        crate::framelog::add(crate::framelog::T::CardSends, 1);
        if let Some((since, worked_then)) = state.1.take() {
            card_clock_us(0, since.elapsed().as_micros() as u64);
            card_clock_us(1, worked().saturating_sub(worked_then));
            // (Card kinds job) The send found the card empty; of the idle
            // stretch, the submitting thread stood waiting for landings.
            crate::framelog::add(crate::framelog::T::CardSendsEmpty, 1);
            crate::framelog::add(
                crate::framelog::T::CardIdleLandUs,
                landing_waited().saturating_sub(LANDING[2].load(Ordering::Relaxed)),
            );
        }
        drop(state);
        clock.sent.notify_one();
    }

    /// Whether the host has what copies in the card's memory need (three
    /// lists and the compare); made at the first ask. A copy goes to the
    /// card's memory only if it has.
    pub(crate) fn card_ready(&self) -> bool {
        static READY: OnceLock<bool> = OnceLock::new();
        *READY.get_or_init(|| match self.make_card_work() {
            Ok(work) => {
                *CARD_WORK.lock().unwrap_or_else(|e| e.into_inner()) = Some(work);
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "no host copies in the card's memory: the compare could not be made");
                false
            }
        })
    }

    fn make_card_work(&self) -> windows::core::Result<CardWork> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; the out
        // pointer is ours.
        unsafe {
            let source = if card_hash_wide() { CARD_COMPARE } else { CARD_COMPARE_ONE_THREAD };
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                windows::core::s!("card compare"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        let list = || self.make_prelude().ok_or_else(windows::core::Error::empty);
        // A megabyte of results: 262,144 pieces, 16 GB of declared ranges
        // in one submission.
        const RESULTS: u64 = 1 << 20;
        let (differs, data) = self.shared_buffer(RESULTS).ok_or_else(windows::core::Error::empty)?;
        name(&differs, || "card compare results".into());
        // SAFETY: the blob's own bytes; the root signature is inside the
        // shader.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
            let pso: ID3D12PipelineState = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(CardWork {
                sync: vec![list()?],
                check: vec![list()?],
                fetch: list()?,
                root,
                pso,
                differs: (differs, data as usize, RESULTS),
                fetched: None,
                taken: 0,
                out: 0,
                inflight: if inflight_fetch() { self.make_card_inflight(RESULTS) } else { None },
                mask: self.make_card_mask(),
                mask_free: Vec::new(),
            })
        }
    }

    /// [`back_async`]: the slots of the fetch inside flights grow after a
    /// landing overflowed them and shrink back after a quiet while
    /// ([`inflight_wanted`]). Only with no plan out (`CardWork::out` 0):
    /// every ticket taken has landed, so no slot is in the air, and a
    /// ticket's slot (`k % slots`) is found again with the new count from
    /// here on. The old slots are let go: the card has run every list that
    /// named them.
    fn card_inflight_resize(&self, work: &mut CardWork) {
        if !back_async() {
            return;
        }
        let Some(inflight) = work.inflight.as_mut() else {
            return;
        };
        let overflowed = INFLIGHT_OVERFLOWED.swap(false, Ordering::AcqRel);
        let frames = FRAMES_SEEN.load(Ordering::Relaxed);
        if overflowed {
            inflight.quiet_from = frames;
        }
        let now = inflight.fetched.2;
        let wanted = inflight_wanted(
            now,
            inflight_slots(inflight_fetch_bytes()),
            inflight_slots(back_async_most()),
            overflowed,
            frames.saturating_sub(inflight.quiet_from),
            back_async_idle(),
        );
        if wanted == now {
            return;
        }
        let bytes = u64::from(wanted) * INFLIGHT_SLOT;
        let Some((fetched, data)) = self.shared_buffer(bytes) else {
            return;
        };
        name(&fetched, || "card fetch in flight: pieces".into());
        tracing::info!(
            from = now,
            to = wanted,
            mb = bytes >> 20,
            "slots of the copy-back inside flights resized (VICTORY_BACK_ASYNC, at most VICTORY_BACK_ASYNC_MB)"
        );
        inflight.fetched = (fetched, data as usize, wanted);
        inflight.quiet_from = frames;
        INFLIGHT_BYTES[0].store(bytes, Ordering::Relaxed);
        INFLIGHT_BYTES[1].fetch_add(1, Ordering::Relaxed);
    }

    /// The masked merge ([`CARD_MASK`]): its root signature and pipeline,
    /// made beside the compare; `None` (and pages over the cap settle the
    /// flights, as before) if it cannot be made or `VICTORY_MERGE_MASK=0`.
    fn make_card_mask(&self) -> Option<(ID3D12RootSignature, ID3D12PipelineState)> {
        if std::env::var_os("VICTORY_MERGE_MASK").is_some_and(|v| v == "0") {
            return None;
        }
        let made = (|| -> windows::core::Result<(ID3D12RootSignature, ID3D12PipelineState)> {
            let code = compile(CARD_MASK, windows::core::s!("main"), windows::core::s!("cs_5_1"))?;
            // SAFETY: the shader's own bytes; the root signature is inside it.
            unsafe {
                let root: ID3D12RootSignature = self.device.CreateRootSignature(0, &code)?;
                let pso: ID3D12PipelineState = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })?;
                Ok((root, pso))
            }
        })();
        match made {
            Ok(mask) => {
                CARD_MASK_READY.store(true, Ordering::Release);
                Some(mask)
            }
            Err(e) => {
                tracing::warn!(error = %e, "no masked merge of title bytes into card copies (VICTORY_MERGE_MASK): pages over VICTORY_MERGE_RUNS settle the flights");
                None
            }
        }
    }

    /// [`CardInflight`] for `results` bytes of the compare's results; `None`
    /// (and the old fetch) if the shader or the memory cannot be had.
    fn make_card_inflight(&self, results: u64) -> Option<CardInflight> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let made = (|| -> windows::core::Result<CardInflight> {
            let mut code: Option<ID3DBlob> = None;
            // SAFETY: the source and names live through the call; the out
            // pointer is ours.
            unsafe {
                D3DCompile(
                    CARD_FETCH.as_ptr().cast(),
                    CARD_FETCH.len(),
                    windows::core::s!("card fetch in flight"),
                    None,
                    None,
                    windows::core::s!("main"),
                    windows::core::s!("cs_5_1"),
                    0,
                    0,
                    &mut code,
                    None,
                )?;
            }
            let code = code.ok_or_else(windows::core::Error::empty)?;
            let slots = inflight_slots(inflight_fetch_bytes());
            INFLIGHT_BYTES[0].store(u64::from(slots) * INFLIGHT_SLOT, Ordering::Relaxed);
            let (table, table_data) = self.shared_buffer(4 + results).ok_or_else(windows::core::Error::empty)?;
            name(&table, || "card fetch in flight: tickets".into());
            let fetched_bytes = u64::from(slots) * INFLIGHT_SLOT;
            let (fetched, fetched_data) = self.shared_buffer(fetched_bytes).ok_or_else(windows::core::Error::empty)?;
            name(&fetched, || "card fetch in flight: pieces".into());
            // SAFETY: the mapping holds the counter; nothing of ours ran
            // on the card with it yet.
            unsafe { table_data.cast::<u32>().write_volatile(0) };
            // SAFETY: the blob's own bytes; the root signature is inside
            // the shader.
            unsafe {
                let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
                let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
                let pso: ID3D12PipelineState = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })?;
                Ok(CardInflight {
                    root,
                    pso,
                    table: (table, table_data as usize, 4 + results),
                    fetched: (fetched, fetched_data as usize, slots),
                    landed: 0,
                    brought: Vec::new(),
                    quiet_from: 0,
                })
            }
        })();
        match made {
            Ok(inflight) => {
                tracing::info!(
                    slots = inflight.fetched.2,
                    "card copies' flagged pieces come down inside their flights (VICTORY_CARD_FETCH_INFLIGHT=1)"
                );
                Some(inflight)
            }
            Err(e) => {
                tracing::warn!(error = %e, "no fetch inside flights (VICTORY_CARD_FETCH_INFLIGHT=1): pieces are fetched the old way");
                None
            }
        }
    }

    /// Before a submission's lists run, for copies in the card's memory
    /// ([`crate::shadow::cards`]): what changed on their CPU side since
    /// the last submission goes to their card side (a list of ours, run
    /// now; the submission's lists follow it on the queue), and the ranges
    /// this submission's shaders may write get a compare to run after its
    /// lists. `None` if there is nothing to do.
    ///
    /// `flown`: the submission is sent ahead with [`relay`]. Its compare
    /// then stores the hashes it makes (see [`flagged_pieces`] and
    /// `relay`).
    ///
    /// `sent`: the last value signalled on the host's fence; the caller
    /// signals the next one behind this submission's lists.
    ///
    /// With `VICTORY_QUEUE2` bit 2 (`crate::queue2`; off by default) the
    /// changed bytes cross from the PC's memory to the card's on a copy
    /// queue, on the card's copy engine, while the direct queue still runs
    /// what was sent before: they go into a landing buffer in the card's
    /// memory that nothing else uses. The direct queue is told to wait for
    /// that copy and then moves the bytes from the landing buffer into the
    /// copies, inside the card's memory, in the same place of its order as
    /// before. So the copies change exactly when they did on one queue; what
    /// leaves the direct queue is the crossing from the PC's memory.
    fn card_before(&self, runs: &[&HostList], flown: bool, sent: u64) -> Option<CardPlan> {
        use std::sync::Arc;
        if !crate::shadow::cards() {
            return None;
        }
        // The ranges this submission's shaders may write, per copy.
        let mut declared: Vec<(Arc<crate::shadow::Shadow>, Vec<(usize, usize)>)> = Vec::new();
        for list in runs {
            for step in &list.deferred().copies {
                if let Step::GpuWrite(shadow, address, len) = step
                    && shadow.carded()
                {
                    let range = (*address, *address + *len);
                    match declared.iter_mut().find(|(s, _)| Arc::ptr_eq(s, shadow)) {
                        Some((_, ranges)) => {
                            if !ranges.contains(&range) {
                                ranges.push(range);
                            }
                        }
                        None => declared.push((shadow.clone(), vec![range])),
                    }
                }
            }
        }
        // Experiment (`VICTORY_SKIP_WC_COMPARE=1`): copies of write-combining
        // title memory are left out of the hashing and the compare; nothing
        // of them is fetched back. Pending ranges are still closed by the
        // flights' own declared writes (`declared_writes`).
        if crate::census::skip_wc_compare() {
            let before = declared.len();
            declared.retain(|(shadow, _)| !crate::census::is_wc(shadow.title_start()));
            crate::census::UNCOMPARED.fetch_add((before - declared.len()) as u64, Ordering::Relaxed);
        }
        // With the relay: the order of the queue is held from here until
        // what is taken is on the queue, and the taking is numbered.
        let mut order = relay().then(|| QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        let sent_as = match order.as_mut() {
            Some(order) => {
                **order += 1;
                QUEUE_ORDER_SEEN.store(**order, Ordering::Release);
                **order
            }
            None => 0,
        };
        let mut dirty = crate::shadow::card_dirty(sent_as);
        // Pages of title bytes that go up as masks (`crate::shadow::merge_mask`),
        // taken and noted in the same order of the queue as the ranges.
        let masked = crate::shadow::card_masked(sent_as);
        for (shadow, _) in &masked {
            if !dirty.iter().any(|(s, _)| Arc::ptr_eq(s, shadow)) {
                dirty.push((shadow.clone(), Vec::new()));
            }
        }
        // With fetches on a copy queue (`crate::queue2`, bit 1 of
        // `VICTORY_QUEUE2`): the send's number is signalled on the direct
        // queue behind the send, so that a fetch knows which sends the card
        // has finished (`Second::order_done`).
        let numbered = || {
            if sent_as != 0
                && let Some(second) = crate::queue2::second().filter(|s| s.fetches())
            {
                second.sent_up(&self.queue, sent_as);
            }
        };
        if declared.is_empty() && dirty.is_empty() {
            numbered();
            return None;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let work = work.as_mut()?;
        // With no plan out, no ticket is in the air: the slots of the fetch
        // inside flights may be resized (`back_async`).
        if work.out == 0 {
            self.card_inflight_resize(work);
        }
        // A copy whose shader writes are declared for the first time gets
        // its bytes from before, whole.
        let mut whole: Vec<Arc<crate::shadow::Shadow>> = Vec::new();
        for (shadow, _) in &declared {
            if shadow.card_make_before(|size| {
                let before = self.uav_buffer(crate::shadow::card_hash_bytes(size))?;
                name(&before, || "title buffer on the card, hashes of its pieces".into());
                Some(before)
            }) {
                whole.push(shadow.clone());
                if !dirty.iter().any(|(s, _)| Arc::ptr_eq(s, shadow)) {
                    dirty.push((shadow.clone(), Vec::new()));
                }
            }
        }
        let mut used: Vec<(bool, Prelude)> = Vec::new();
        // The masks, in a buffer of PC memory the masked merges read
        // (`CARD_MASK`): per copy, each page's (offset, words, place of
        // its mask). Held by the plan until its send list has run.
        let mut mask_buffer: Option<(ID3D12Resource, usize, u64)> = None;
        let mut mask_pages: Vec<(Arc<crate::shadow::Shadow>, Vec<(u32, u32, u32)>)> = Vec::new();
        if !masked.is_empty() {
            let total: u64 = masked.iter().flat_map(|(_, m)| m.iter().map(|m| m.mask.len() as u64)).sum();
            let found = work.mask_free.iter().position(|b| b.2 >= total);
            let buffer = match found {
                Some(i) => Some(work.mask_free.swap_remove(i)),
                None => {
                    let size = total.next_power_of_two().max(1 << 20);
                    let made = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
                    if made.is_some() {
                        MASK_STATS[2].fetch_add(1, Ordering::Relaxed);
                        MASK_STATS[3].fetch_add(size, Ordering::Relaxed);
                    }
                    made
                }
            };
            let Some(buffer) = buffer else {
                fail(format_args!("no memory for the masks of title bytes put into host copies in the card's memory ({total} bytes)"));
            };
            let mut at = 0usize;
            for (shadow, masks) in &masked {
                let mut pages = Vec::with_capacity(masks.len());
                for m in masks {
                    // SAFETY: inside the buffer's mapping (`total` bytes or
                    // more); nothing of ours reads it before this list runs.
                    unsafe { std::ptr::copy_nonoverlapping(m.mask.as_ptr(), (buffer.1 + at) as *mut u8, m.mask.len()) };
                    pages.push((m.at as u32, (m.mask.len() / 4) as u32, at as u32));
                    at += m.mask.len();
                }
                mask_pages.push((shadow.clone(), pages));
            }
            MASK_STATS[0].fetch_add(masked.iter().map(|(_, m)| m.len() as u64).sum(), Ordering::Relaxed);
            crate::framelog::add(crate::framelog::T::Merges, masked.iter().map(|(_, m)| m.len() as u64).sum());
            MASK_STATS[1].fetch_add(total, Ordering::Relaxed);
            mask_buffer = Some(buffer);
        }
        // The changed bytes through the copy queue (`VICTORY_QUEUE2` bit
        // 2): the landing buffer they are in, in the order of `dirty`'s
        // ranges, once the direct queue has waited. `None`: straight from
        // the PC's memory on the direct queue, as before.
        let mut landed: Option<ID3D12Resource> = None;
        if let Some(second) = crate::queue2::second().filter(|s| s.uploads()) {
            let bytes: u64 = dirty
                .iter()
                .filter(|(shadow, _)| shadow.carded())
                .flat_map(|(_, ranges)| ranges.iter().map(|&(a, b)| (b - a) as u64))
                .sum();
            // SAFETY: a plain query.
            let completed = unsafe { self.fence.GetCompletedValue() };
            // Held until the card has passed this submission's signal: the
            // direct queue's copy out of it runs ahead of that.
            if bytes > 0
                && let Some(landing) = second.landing(bytes, completed, sent + 1)
            {
                // SAFETY: an open copy list of the lane's; live buffers,
                // and the ranges lie in them; the host's queue.
                let sent_up = unsafe {
                    (|| -> windows::core::Result<()> {
                        let copy = second.up.list()?;
                        let mut at = 0u64;
                        for (shadow, ranges) in &dirty {
                            let Some((staging, _, _)) = shadow.card_sides() else {
                                continue;
                            };
                            for &(a, b) in ranges {
                                copy.1.CopyBufferRegion(&landing, at, staging, a as u64, (b - a) as u64);
                                at += (b - a) as u64;
                            }
                        }
                        let job = second.up.send(copy, &[], bytes)?;
                        // Nothing sent to the direct queue from here on
                        // runs before the bytes have landed.
                        second.hold(&self.queue, job)
                    })()
                };
                if let Err(e) = sent_up {
                    fail(format_args!(
                        "host copies in the card's memory could not be brought up to date through the copy queue ({e}, VICTORY_QUEUE2)"
                    ));
                }
                landed = Some(landing);
            }
        }
        if !dirty.is_empty() {
            let Some(sync) = work.sync.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list to bring host copies in the card's memory up to date"));
            };
            let (allocator, list) = &sync;
            // SAFETY: the list's last run finished (it ran ahead of a
            // submission that was waited for, and came back after that);
            // live buffers, and the ranges lie in them.
            let recorded = unsafe {
                (|| -> windows::core::Result<()> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    let timed = crate::cardmeter::begin(list, crate::cardmeter::Kind::Sync);
                    // Where the next range lies in the landing buffer.
                    let mut at = 0u64;
                    for (shadow, ranges) in &dirty {
                        let Some((staging, before, size)) = shadow.card_sides() else {
                            continue;
                        };
                        for &(a, b) in ranges {
                            match &landed {
                                Some(landing) => list.CopyBufferRegion(&shadow.buffer, a as u64, landing, at, (b - a) as u64),
                                None => list.CopyBufferRegion(&shadow.buffer, a as u64, staging, a as u64, (b - a) as u64),
                            }
                            at += (b - a) as u64;
                            crate::shadow::CARD_STATS[0].fetch_add((b - a) as u64, Ordering::Relaxed);
                            crate::faults::count(crate::faults::Driver::CopyUpBytes, (b - a) as u64);
                        }
                        crate::shadow::CARD_STATS[1].fetch_add(ranges.len() as u64, Ordering::Relaxed);
                        crate::census::sent(
                            shadow.title_start(),
                            ranges.iter().map(|&(a, b)| (b - a) as u64).sum(),
                            ranges.len() as u64,
                        );
                        // Where the copy's state is now: promoted to a copy's
                        // destination by the copies above, else common.
                        let mut state = if ranges.is_empty() { D3D12_RESOURCE_STATE_COMMON } else { D3D12_RESOURCE_STATE_COPY_DEST };
                        // Masked pages (`crate::shadow::merge_mask`): one
                        // pass each, behind the copies, then a barrier
                        // before the hash reads what they wrote.
                        let pages = mask_pages.iter().find(|(s, _)| Arc::ptr_eq(s, shadow)).map(|(_, p)| p.as_slice()).unwrap_or(&[]);
                        if let (false, Some((root, pso)), Some(masks)) = (pages.is_empty(), &work.mask, &mask_buffer) {
                            list.ResourceBarrier(&[transition(&shadow.buffer, state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
                            state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                            list.SetComputeRootSignature(root);
                            list.SetPipelineState(pso);
                            list.SetComputeRootShaderResourceView(1, staging.GetGPUVirtualAddress());
                            list.SetComputeRootShaderResourceView(2, masks.0.GetGPUVirtualAddress());
                            list.SetComputeRootUnorderedAccessView(3, shadow.buffer.GetGPUVirtualAddress());
                            for &(at, words, mask_at) in pages {
                                let constants = [at, words, mask_at, 0u32];
                                list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                                list.Dispatch(words.div_ceil(64), 1, 1);
                                crate::shadow::CARD_STATS[0].fetch_add(u64::from(words) * 4, Ordering::Relaxed);
                            }
                            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                        pResource: std::mem::ManuallyDrop::new(None),
                                    }),
                                },
                            }]);
                        }
                        let Some(before) = before else {
                            continue;
                        };
                        // No twin (it cost the card the copy's size again,
                        // 399 MB in run 75): `before` holds a 128-bit hash
                        // of each 64 KiB piece. The pieces just sent up are
                        // hashed again, all of them the first time; the
                        // compare after the lists hashes the declared
                        // pieces and flags those whose hash is another.
                        if state != D3D12_RESOURCE_STATE_COMMON {
                            list.ResourceBarrier(&[transition(&shadow.buffer, state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
                        }
                        list.SetComputeRootSignature(&work.root);
                        list.SetPipelineState(&work.pso);
                        list.SetComputeRootShaderResourceView(1, shadow.buffer.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                        let runs: Vec<crate::shadow::CardSpan> = if whole.iter().any(|s| Arc::ptr_eq(s, shadow)) {
                            vec![shadow.card_run(0, usize::MAX)]
                        } else {
                            // The masked pages' pieces with the ranges'.
                            ranges
                                .iter()
                                .map(|&(a, b)| shadow.card_run(a, b))
                                .chain(pages.iter().map(|&(at, words, _)| shadow.card_run(at as usize, at as usize + words as usize * 4)))
                                .collect()
                        };
                        // The same pieces in fewer passes (`rehash_runs`).
                        let runs = if rehash_runs() { crate::shadow::joined_piece_runs(runs) } else { runs };
                        CARD_PASSES[0].fetch_add(runs.len() as u64, Ordering::Relaxed);
                        CARD_PASSES[1].fetch_add(runs.iter().map(|r| u64::from(r.pieces)).sum::<u64>(), Ordering::Relaxed);
                        for run in runs {
                            card_dispatch(list, [run.first, run.skew, size as u32, run.pieces, 0, 0]);
                        }
                    }
                    crate::cardmeter::end(list, timed);
                    list.Close()?;
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    Ok(())
                })()
            };
            if let Err(e) = recorded {
                fail(format_args!("host copies in the card's memory could not be brought up to date ({e})"));
            }
            used.push((false, sync));
        }
        numbered();
        drop(order);
        let mut plan = CardPlan {
            ranges: Vec::new(),
            check: None,
            _synced: dirty.into_iter().map(|(shadow, _)| shadow).collect(),
            lists: used,
            sent_as,
            inflight: false,
            inflight_landed: 0,
            masks: mask_buffer,
        };
        // Each byte once, as the copy-back joins the declared ranges. The
        // results go behind those of plans still in the air.
        let mut at = work.taken;
        for (shadow, ranges) in &declared {
            let mut sorted = ranges.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            let twin = shadow.card_sides().is_some_and(|(_, before, _)| before.is_some());
            for (from, to) in spans {
                let Some(span) = shadow.card_span(from, to) else {
                    continue;
                };
                let compared = twin && u64::from(at + span.pieces) * 4 <= work.differs.2;
                plan.ranges.push(CardRange {
                    shadow: shadow.clone(),
                    declared: (from, to),
                    span,
                    at,
                    compared,
                });
                if compared {
                    at += span.pieces;
                }
            }
        }
        work.taken = at;
        if plan.ranges.iter().any(|r| r.compared) {
            let Some(check) = work.check.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list for the compare of host copies in the card's memory"));
            };
            let (allocator, list) = &check;
            // SAFETY: the list's last run finished (its submission was
            // waited for, and it came back after that); live buffers; the
            // compare's own root signature.
            let recorded = unsafe {
                (|| -> windows::core::Result<ID3D12CommandList> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    let timed = crate::cardmeter::begin(list, crate::cardmeter::Kind::Check);
                    // What the submission's shaders wrote, first.
                    list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                pResource: std::mem::ManuallyDrop::new(None),
                            }),
                        },
                    }]);
                    list.SetComputeRootSignature(&work.root);
                    list.SetPipelineState(&work.pso);
                    list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                    for r in plan.ranges.iter().filter(|r| r.compared) {
                        let Some((_, Some(before), size)) = r.shadow.card_sides() else {
                            continue;
                        };
                        let constants = [r.span.first, r.span.skew, size as u32, r.span.pieces, r.at * 4, if flown { 2 } else { 1 }];
                        list.SetComputeRootShaderResourceView(1, r.shadow.buffer.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                        card_dispatch(list, constants);
                        CARD_PASSES[2].fetch_add(1, Ordering::Relaxed);
                    }
                    // The fetch inside the flight (`inflight_fetch`): behind
                    // every compare of the list, the pieces they flagged go
                    // to slots in the PC's memory.
                    if flown && let Some(inflight) = &work.inflight {
                        list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                    pResource: std::mem::ManuallyDrop::new(None),
                                }),
                            },
                        }]);
                        list.SetComputeRootSignature(&inflight.root);
                        list.SetPipelineState(&inflight.pso);
                        list.SetComputeRootUnorderedAccessView(2, work.differs.0.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(3, inflight.table.0.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(4, inflight.fetched.0.GetGPUVirtualAddress());
                        let mut every = true;
                        for r in plan.ranges.iter().filter(|r| r.compared) {
                            let Some((_, Some(_), size)) = r.shadow.card_sides() else {
                                every = false;
                                continue;
                            };
                            let groups = r.span.pieces.clamp(1, 65535);
                            let constants =
                                [r.span.first, r.span.skew, size as u32, r.span.pieces, r.at * 4, inflight.landed, inflight.fetched.2, groups];
                            list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                            list.SetComputeRootShaderResourceView(1, r.shadow.buffer.GetGPUVirtualAddress());
                            list.Dispatch(groups, 1, 1);
                        }
                        plan.inflight = every;
                        plan.inflight_landed = inflight.landed;
                    }
                    crate::cardmeter::end(list, timed);
                    list.Close()?;
                    list.cast()
                })()
            };
            match recorded {
                Ok(list) => plan.check = Some(list),
                Err(e) => fail(format_args!("the compare of host copies in the card's memory could not be recorded ({e})")),
            }
            plan.lists.push((true, check));
        }
        work.out += 1;
        Some(plan)
    }

    /// With [`relay`]: whether the compare's result area is half taken by
    /// plans in the air. It starts from the beginning only when no plan is
    /// out, so the caller lets the flights land first.
    fn card_results_low(&self) -> bool {
        CARD_WORK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|work| u64::from(work.taken) * 8 > work.differs.2)
    }

    /// The plan's submission has run and its results were read
    /// ([`Host::card_back`]): its lists are free for the next plan. Called
    /// when nothing of ours is on the card, so the compare's results start
    /// from the beginning again.
    fn card_done(&self, plan: CardPlan) {
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        for (check, list) in plan.lists {
            if check { work.check.push(list) } else { work.sync.push(list) }
        }
        // Its masks' buffer: kept for the next plans, a few at most.
        if let Some(masks) = plan.masks {
            if work.mask_free.len() < 8 {
                work.mask_free.push(masks);
            } else {
                MASK_STATS[2].fetch_sub(1, Ordering::Relaxed);
                MASK_STATS[3].fetch_sub(masks.2, Ordering::Relaxed);
            }
        }
        work.out = work.out.saturating_sub(1);
        // With the relay other plans may be on the card, their results
        // not read yet: the area starts again only when none is out.
        if !relay() || work.out == 0 {
            work.taken = 0;
        }
    }

    /// The relay's own list, fence, event for [`Host::relay_card_back`].
    fn make_relay_fetch(&self) -> Option<RelayFetch> {
        let list = self.make_prelude()?;
        // SAFETY: plain objects of our own.
        unsafe {
            let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            Some(RelayFetch {
                list,
                fence,
                event,
                sent: 0,
                buffer: None,
            })
        }
    }

    /// [`relay`], on the thread that lands flights: for the plans of the
    /// flights being landed (the card has run them), the pieces their
    /// compares flagged are fetched, once each, with a list and a wait of
    /// this thread's own, and what shaders wrote in them goes into title
    /// memory (`Shadow::card_merge_relay`). As [`Host::card_back`], but:
    /// no submission lock; a piece flagged under one range of a plan is
    /// fetched under all ([`flagged_pieces`]); the fetch runs behind
    /// whatever was sent meanwhile, so it may bring a later flight's
    /// bytes, which that flight's landing then finds merged already; and
    /// nothing is sent back up.
    ///
    /// With [`inflight_fetch`] the pieces are first looked for where the
    /// flights' own compare lists put them ([`Host::relay_card_inflight`]);
    /// only a landing that cannot take them from there fetches, as here.
    /// `behind`: the value signalled on the host's fence behind the last of
    /// the flights. With `VICTORY_QUEUE2` bit 1 (`crate::queue2`) the fetch
    /// goes to a copy queue, behind that value and nothing else: it does
    /// not stand behind what the submitting thread sent meanwhile (run
    /// 118: 114-157 ms a frame of this thread's time). It then reads the
    /// card's side while later flights may be writing it: a byte is the
    /// old one or the new one, and the flight that writes it has a compare
    /// and a landing of its own behind it.
    ///
    /// `plans` come with their flights' numbers and `landed` is the number
    /// of the last flight being landed: a piece an earlier fetch on the
    /// copy queue brought down beside one of these flights is fetched
    /// whatever the compare says ([`BESIDE`]).
    fn relay_card_back(&self, plans: &[(u64, &CardPlan)], fetch: &mut Option<RelayFetch>, behind: u64, landed: u64) {
        if inflight_fetch() {
            let only: Vec<&CardPlan> = plans.iter().map(|p| p.1).collect();
            // A piece an earlier fetch on the copy queue brought down while
            // one of these flights may have been writing it ([`BESIDE`]):
            // its slot may not hold it (its compare need not flag it), so
            // the landing goes the old way, which fetches it again.
            let beside = crate::queue2::second().is_some_and(|s| s.fetches())
                && refetch()
                && BESIDE.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|noted| {
                    !noted.is_empty()
                        && plans.iter().any(|&(number, plan)| {
                            plan.ranges.iter().any(|r| {
                                let copy = std::sync::Arc::as_ptr(&r.shadow) as usize;
                                (0..r.span.pieces).any(|p| fetched_beside(noted.get(&(copy, r.span.first + p)).copied(), number))
                            })
                        })
                });
            let taken = if beside {
                BACK_REASON.store(3, Ordering::Relaxed);
                false
            } else {
                self.relay_card_inflight(&only)
            };
            inflight_tickets_landed(&only);
            if taken {
                return;
            }
        }
        let lane = crate::queue2::second().filter(|s| s.fetches());
        // The pieces fetched beside flights that had started, as noted by
        // earlier landings: looked at for these plans, renewed below.
        let beside = lane.is_some() && refetch();
        let noted: std::collections::HashMap<(usize, u32), u64> = if beside {
            BESIDE.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_default()
        } else {
            Default::default()
        };
        let mut again = 0u64;
        // Every piece fetched now: (copy, piece number).
        let mut numbers: Vec<(usize, u32)> = Vec::new();
        let differs = {
            let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
            work.as_ref().map(|work| (work.differs.1, (work.differs.2 / 4) as usize))
        };
        let mut pieces: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        if let Some((differs, dwords)) = differs {
            // SAFETY: the mapping of the results. The compares of these
            // plans have run, and their dwords are given to no other plan
            // before these are done with (`card_done`).
            let differs = unsafe { std::slice::from_raw_parts(differs as *const u32, dwords) };
            let mut seen = std::collections::HashSet::new();
            for &(number, plan) in plans {
                let mut flags: Vec<(usize, u32, bool)> = plan
                    .ranges
                    .iter()
                    .flat_map(|r| {
                        (0..r.span.pieces).map(move |p| {
                            let flagged = !r.compared || differs.get((r.at + p) as usize).is_none_or(|&d| d != 0);
                            (std::sync::Arc::as_ptr(&r.shadow) as usize, r.span.first + p, flagged)
                        })
                    })
                    .collect();
                // A piece an earlier fetch brought down while this flight
                // may have been writing it: fetched, flagged or not.
                for flag in flags.iter_mut().filter(|f| !f.2) {
                    if fetched_beside(noted.get(&(flag.0, flag.1)).copied(), number) {
                        flag.2 = true;
                        again += 1;
                    }
                }
                let fetched = flagged_pieces(&flags);
                numbers.extend(fetched.iter().copied());
                for r in &plan.ranges {
                    let copy = std::sync::Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        if !fetched.contains(&(copy, r.span.first + p)) {
                            continue;
                        }
                        let (lo, hi) = r.shadow.card_piece(&r.span, p);
                        // (`VICTORY_ARENA_BY_CHANGED`) What the compare found
                        // changed, fetched or left on the card.
                        crate::indirect::arena::note_changed(r.shadow.title_start() as usize + lo, hi.saturating_sub(lo));
                        if lo < hi
                            && !crate::census::skip_back(r.shadow.title_start(), (hi - lo) as u64)
                            && seen.insert((copy, lo, hi))
                            // (Stage 1, `VICTORY_BACK_SKIP_UNREAD`) Left on
                            // the card: nothing consumed its last writes.
                            && !crate::readback::consumers::try_skip(&r.shadow, lo, hi)
                        {
                            pieces.push((r.shadow.clone(), lo, hi, total));
                            total += (hi - lo) as u64;
                        }
                    }
                    crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
                    crate::census::compared(r.shadow.title_start(), u64::from(r.span.pieces));
                }
            }
        }
        BESIDE_STATS[0].fetch_add(again, Ordering::Relaxed);
        if pieces.is_empty() {
            if inflight_fetch() {
                // The next merge may be of a flight in the air now, with
                // bytes from when it ran: only sends up to these plans' are
                // older than it.
                if let Some(last) = plans.iter().map(|p| p.1.sent_as).max() {
                    crate::shadow::relay_fetched(last);
                }
                return;
            }
            if beside {
                // Notes whose flights have all landed are done with.
                let mut all = BESIDE.lock().unwrap_or_else(|e| e.into_inner());
                note_beside(all.get_or_insert_with(Default::default), &[], landed, landed);
            }
            // Nothing to fetch: every send so far is older than the next
            // fetch (on a copy queue: every send the card has finished).
            crate::shadow::relay_fetched(match lane {
                Some(second) => second.order_done(),
                None => *QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()),
            });
            return;
        }
        if inflight_fetch() {
            INFLIGHT_STATS[3].fetch_add(1, Ordering::Relaxed);
            let why = BACK_REASON.load(Ordering::Relaxed).min(4);
            BACK_FALLBACK[why].fetch_add(1, Ordering::Relaxed);
            BACK_FALLBACK[5 + why].fetch_add(total, Ordering::Relaxed);
        }
        if fetch.is_none() {
            *fetch = self.make_relay_fetch();
        }
        let Some(f) = fetch.as_mut() else {
            fail(format_args!("no list, fence or event to fetch what shaders wrote to the card's memory (VICTORY_RELAY=1)"));
        };
        if f.buffer.as_ref().is_none_or(|b| b.2 < total) {
            let size = total.next_multiple_of(16 << 20);
            f.buffer = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
        }
        let Some((buffer, data, _)) = f.buffer.clone() else {
            fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes, VICTORY_RELAY=1)"));
        };
        f.sent += 1;
        let ((allocator, list), fence, event, sent) = (f.list.clone(), f.fence.clone(), f.event, f.sent);
        let waiting = std::time::Instant::now();
        // SAFETY: the list's last run was waited for by this thread; live
        // buffers (the copies are held by the plans); the fence and the
        // event are this thread's.
        let ran = unsafe {
            (|| -> windows::core::Result<u64> {
                if let Some(second) = lane {
                    // The sends of title bytes the card has finished are in
                    // what comes down. A later one may be too, if the card
                    // runs it before this fetch: the merge leaves its bytes
                    // out as a send that came after, and they are the
                    // title's either way.
                    let after = second.order_done();
                    let copy = second.down.list()?;
                    for (shadow, lo, hi, at) in &pieces {
                        copy.1.CopyBufferRegion(&buffer, *at, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
                    }
                    let job = second.down.send(copy, &[(&self.fence, behind)], total)?;
                    second.down.wait(job, event)?;
                    if beside {
                        // Whatever had started by now may have been
                        // writing these pieces while they came down
                        // (counted before a submission's lists go to the
                        // queue, `Host::submit`: nothing later ran yet).
                        let started = STARTED.load(Ordering::Acquire);
                        let mut all = BESIDE.lock().unwrap_or_else(|e| e.into_inner());
                        let noted = note_beside(all.get_or_insert_with(Default::default), &numbers, landed, started);
                        BESIDE_STATS[1].fetch_add(noted, Ordering::Relaxed);
                    }
                    second.down.stats.waited_us.fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
                    return Ok(after);
                }
                allocator.Reset()?;
                list.Reset(&allocator, None)?;
                for (shadow, lo, hi, at) in &pieces {
                    list.CopyBufferRegion(&buffer, *at, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
                }
                list.Close()?;
                let lists = [Some(list.cast::<ID3D12CommandList>()?)];
                // In the order of the queue: the sends of title bytes so
                // far are ahead of this fetch, every later one behind it.
                let order = QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner());
                self.queue.ExecuteCommandLists(&lists);
                let after = *order;
                drop(order);
                self.card_sent();
                self.queue.Signal(&fence, sent)?;
                fence.SetEventOnCompletion(sent, event)?;
                WaitForSingleObject(event, INFINITE);
                Ok(after)
            })()
        };
        let after = match ran {
            Ok(after) => after,
            Err(e) => fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e}, VICTORY_RELAY=1)")),
        };
        // This fetch may bring bytes of flights in the air now: their own
        // slots are older than what it puts in the CPU's side there
        // (`relay_card_inflight`). Sends up to `after` are numbered, and the
        // submission numbered `after` may not have its lists on the queue
        // yet; every one before it has (the submission lock).
        if inflight_fetch()
            && after > 1
            && let Some(inflight) = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|w| w.inflight.as_mut())
        {
            inflight.brought.extend(pieces.iter().map(|(shadow, lo, hi, _)| (shadow.clone(), *lo, *hi, after - 1)));
        }
        let us = waiting.elapsed().as_micros() as u64;
        RELAY_STATS[3].fetch_add(us, Ordering::Relaxed);
        {
            use crate::shadow::CARD_STATS;
            CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
            CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
            CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
            CARD_STATS[6].fetch_add(us, Ordering::Relaxed);
            if crate::census::on() {
                for (shadow, lo, hi, _) in &pieces {
                    crate::census::fetched(shadow.title_start(), (hi - lo) as u64);
                }
            }
        }
        let mut title = Vec::new();
        let mut written = 0;
        for (shadow, lo, hi, at) in &pieces {
            // SAFETY: the mapping of the fetched bytes, `total` of them;
            // the copy into it has run.
            let gpu = unsafe { std::slice::from_raw_parts((data + *at as usize) as *const u8, hi - lo) };
            let merged = shadow.card_merge_relay(*lo, gpu, after, &mut title);
            shadow.walls_merged(gpu.len(), merged);
            written += merged;
        }
        copy_back(1, written as u64);
        // Sends up to `after` are of no use to any later merge; with the
        // fetch inside flights only those up to these plans' are
        // (`fallback_fetched_mark`).
        let landed = plans.iter().map(|p| p.1.sent_as).max();
        crate::shadow::relay_fetched(fallback_fetched_mark(inflight_fetch(), after, landed));
    }

    /// [`Host::relay_card_back`] with [`inflight_fetch`]: the pieces these
    /// plans' compares flagged are taken from the slots their own compare
    /// lists copied them to ([`CARD_FETCH`]) and merged, plan by plan in the
    /// line's order, with no list or wait of ours. `false` (nothing merged)
    /// if any plan cannot land so; the caller then lands them all the old
    /// way, as without the switch.
    ///
    /// The merge's "fetched after" (`Shadow::card_merge_relay`) is the
    /// plan's own `sent_as`. `QUEUE_ORDER` is raised only in
    /// `Host::card_before`, under the submission lock, and that lock is held
    /// until the submission's lists and its compare's list are on the queue
    /// behind its sync list: so when the compare's list ran, the card had
    /// received exactly the sends 1..=`sent_as` of title bytes and no later
    /// one, which is what the merge's skip of later sends asks. (The old
    /// fetch, by contrast, ran whenever it reached the queue, and took the
    /// count at its own ExecuteCommandLists.)
    ///
    /// When a landing goes the old way instead, all of it does: a fetch the
    /// old way brings the card's bytes as they are now, newer than the slots
    /// of plans in the air then, and merging an older slot after it would
    /// put a flight's bytes back over a later flight's. For the same reason
    /// the bytes such a fetch brought (`CardInflight::brought`) are left
    /// out of the slots of flights whose lists were on the queue ahead of
    /// it: the CPU's side holds what the card held after them there. The
    /// rest of their slots is merged, newer than anything merged there
    /// before. (Left out whole, a flight sent before such a fetch would have
    /// had to land the old way, its fetch taking a later count still: one
    /// overflow kept every landing on the old way while flights were in
    /// the air.) Plans land in the line's order, each plan's slot as it was
    /// right behind its own lists, so every merge here is of bytes at least
    /// as new as the CPU's side holds.
    ///
    /// shortcut: all or nothing per landing, and a plan whose pieces overflow
    /// the slots lands the old way for all its pieces, not only the ones
    /// left out.
    fn relay_card_inflight(&self, plans: &[&CardPlan]) -> bool {
        use std::sync::Arc;
        type Piece = (Arc<crate::shadow::Shadow>, usize, usize, usize, u64);
        let mut pieces: Vec<Piece> = Vec::new();
        {
            let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
            let Some(work) = work.as_ref() else {
                return false;
            };
            let Some(inflight) = &work.inflight else {
                return false;
            };
            BACK_REASON.store(4, Ordering::Relaxed);
            let dwords = (work.differs.2 / 4) as usize;
            // SAFETY: the mappings of the results and the tickets. The
            // compares and fetches of these plans have run, and their dwords
            // are given to no other plan before these are done with
            // (`card_done`).
            let (differs, table) = unsafe {
                (
                    std::slice::from_raw_parts(work.differs.1 as *const u32, dwords),
                    std::slice::from_raw_parts((inflight.table.1 as *const u32).add(1), dwords),
                )
            };
            let slots = inflight.fetched.2;
            for plan in plans {
                if plan.ranges.is_empty() {
                    continue;
                }
                if !plan.inflight || plan.ranges.iter().any(|r| !r.compared) {
                    BACK_REASON.store(1, Ordering::Relaxed);
                    return false;
                }
                // Pieces (copy, number) the compare flagged, with their ticket.
                let mut tickets: std::collections::HashMap<(usize, u32), u32> = std::collections::HashMap::new();
                let mut flags: Vec<(usize, u32, bool)> = Vec::new();
                for r in &plan.ranges {
                    let copy = Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        let at = (r.at + p) as usize;
                        let (Some(&flag), Some(&entry)) = (differs.get(at), table.get(at)) else {
                            return false;
                        };
                        if flag != 0 {
                            if entry == INFLIGHT_NONE {
                                BACK_REASON.store(2, Ordering::Relaxed);
                                return false;
                            }
                            if !inflight_gets_slot(entry, plan.inflight_landed, slots) {
                                INFLIGHT_STATS[2].fetch_add(1, Ordering::Relaxed);
                                BACK_REASON.store(0, Ordering::Relaxed);
                                INFLIGHT_OVERFLOWED.store(true, Ordering::Release);
                                return false;
                            }
                            tickets.insert((copy, r.span.first + p), entry);
                        }
                        flags.push((copy, r.span.first + p, flag != 0));
                    }
                }
                let fetched = flagged_pieces(&flags);
                let mut seen = std::collections::HashSet::new();
                for r in &plan.ranges {
                    let copy = Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        let piece = r.span.first + p;
                        if !fetched.contains(&(copy, piece)) {
                            continue;
                        }
                        let (lo, hi) = r.shadow.card_piece(&r.span, p);
                        // (`VICTORY_ARENA_BY_CHANGED`) What the compare found changed.
                        crate::indirect::arena::note_changed(r.shadow.title_start() as usize + lo, hi.saturating_sub(lo));
                        if lo >= hi || !seen.insert((copy, lo, hi)) {
                            continue;
                        }
                        let from = inflight_piece_from(&r.span, p);
                        let Some(&k) = tickets.get(&(copy, piece)) else {
                            return false;
                        };
                        if lo < from || (hi - from) as u64 > INFLIGHT_SLOT {
                            return false;
                        }
                        let slot = inflight.fetched.1 + (k % slots) as usize * INFLIGHT_SLOT as usize;
                        // Bytes a fetch the old way brought after this flight ran.
                        let newer: Vec<(usize, usize)> = inflight
                            .brought
                            .iter()
                            .filter(|b| b.3 >= plan.sent_as && Arc::ptr_eq(&b.0, &r.shadow) && b.1 < hi && b.2 > lo)
                            .map(|b| (b.1, b.2))
                            .collect();
                        for (a, b) in crate::shadow::outside(lo, hi, newer) {
                            pieces.push((r.shadow.clone(), a, b, slot + (a - from), plan.sent_as));
                        }
                    }
                }
            }
        }
        // Outside the lock: no slot of these plans is written over before
        // their tickets are landed (`inflight_tickets_landed`, after this).
        let mut title = Vec::new();
        let (mut written, mut bytes) = (0, 0u64);
        for (shadow, lo, hi, at, after) in &pieces {
            // SAFETY: inside the mapping of the slots (checked above); the
            // copy into it ran in the plan's flight, which the card has run.
            let gpu = unsafe { std::slice::from_raw_parts(*at as *const u8, hi - lo) };
            written += shadow.card_merge_relay(*lo, gpu, *after, &mut title);
            bytes += (hi - lo) as u64;
        }
        for plan in plans {
            for r in &plan.ranges {
                crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
                crate::census::compared(r.shadow.title_start(), u64::from(r.span.pieces));
            }
        }
        copy_back(1, written as u64);
        INFLIGHT_STATS[0].fetch_add(pieces.len() as u64, Ordering::Relaxed);
        INFLIGHT_STATS[1].fetch_add(bytes, Ordering::Relaxed);
        // The next merge is of a later plan, or a fetch the old way sent now.
        if let Some(last) = plans.iter().map(|p| p.sent_as).max() {
            crate::shadow::relay_fetched(last);
        }
        true
    }

    /// [`relay`], on the thread that lands flights: `items` are the first
    /// things of the line, flights the card has run and what follows them.
    /// What their shaders wrote goes into title memory (copies in the
    /// card's memory first, then the others, each byte once whichever
    /// flights declared it); then every flight's ranges are done with;
    /// then, in the line's order, each flight's read-backs, its
    /// bottom-of-pipe writes and the fence signals behind it. Title memory
    /// is complete before the title is told anything.
    fn relay_land(&self, items: Vec<Late>, fetch: &mut Option<RelayFetch>) {
        let landing = std::time::Instant::now();
        // (Copy-back consumers) A generation of its own; demands' notes in.
        crate::readback::consumers::landing();
        crate::readback::consumers::drain_notes();
        {
            let flights: Vec<&Flight> = items
                .iter()
                .filter_map(|l| match l {
                    Late::Run(flight) => Some(flight),
                    _ => None,
                })
                .collect();
            if !flights.is_empty() {
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = flights.iter().flat_map(|f| f.order.iter().map(|&i| f.lists[i].host())).collect();
                    fail(format_args!(
                        "host GPU device removed after {} submissions sent ahead (VICTORY_RELAY=1) of {} lists{}{}",
                        flights.len(),
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                fill_ahead::landing(flights.iter().map(|f| f.number).min().unwrap_or(0));
                let plans: Vec<(u64, &CardPlan)> = flights.iter().filter_map(|f| f.card.as_ref().map(|card| (f.number, card))).collect();
                let behind = flights.iter().map(|f| f.fence).max().unwrap_or(0);
                let landed = flights.iter().map(|f| f.number).max().unwrap_or(0);
                self.relay_card_back(&plans, fetch, behind, landed);
                let mut back: Vec<Declared> = Vec::new();
                for flight in &flights {
                    for (shadow, ranges) in &flight.back {
                        match back.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                            Some((_, all)) => {
                                for range in ranges {
                                    if !all.contains(range) {
                                        all.push(*range);
                                    }
                                }
                            }
                            None => back.push((shadow.clone(), ranges.clone())),
                        }
                    }
                }
                relay_hand_back(&back);
                // Only now, and flight by flight: a range another flight
                // in the air declared too stays closed to refreshes.
                for flight in &flights {
                    for (shadow, ranges) in &flight.back {
                        shadow.copy_back_done(ranges);
                    }
                }
                copy_back(0, landing.elapsed().as_micros() as u64);
                RELAY_STATS[0].fetch_add(1, Ordering::Relaxed);
                RELAY_STATS[1].fetch_add(flights.len() as u64, Ordering::Relaxed);
            }
        }
        // In the order given: what tells the title the card is done.
        let mut last = 0;
        let mut landed = Vec::new();
        for item in items {
            match item {
                Late::Run(mut flight) => {
                    last = flight.number;
                    for &i in &flight.order {
                        read_backs(flight.lists[i].host());
                    }
                    if let Some(card) = flight.card {
                        self.card_done(card);
                    }
                    crate::list::release_landed(&mut flight.lists);
                    landed.push((flight.lists, flight.order));
                }
                Late::Write(at, what, bytes) => what.write(at as *mut u8, bytes),
                Late::Signal(fence, value) => crate::fence::set(fence as crate::object::This, value),
            }
        }
        if last > 0 {
            FINISHED.store(last, Ordering::Release);
            bury(last);
        }
        self.log_messages();
        fill_ahead::landing(0);
        RELAY_STATS[4].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
        {
            let mut line = line();
            let Line {
                line: items,
                written,
                lists,
                settling,
                poke,
                landed: kept,
                in_hand,
                ..
            } = &mut *line;
            kept.extend(landed);
            in_hand.clear();
            // What is still in the air, from the flights that are.
            written.clear();
            *lists = 0;
            let mut ranges = Vec::new();
            for item in items.iter() {
                if let Late::Run(flight) = item {
                    *lists += flight.order.len().max(1);
                    ranges.extend(flight.back.iter().flat_map(|(_, r)| r.iter().copied()));
                }
            }
            join_ranges(written, ranges);
            *settling = false;
            // The landed lists are reset, and what queues hold looked at,
            // by the helper thread (they take locks a waiter may hold).
            *poke = true;
        }
        LINE_MOVED.notify_all();
        LINE_LANDED.notify_all();
    }

    /// After a submission ran, for copies in the card's memory: the pieces
    /// of the declared ranges that the compare found differing (all of a
    /// range it did not compare) are fetched, with one more wait for the
    /// card, and what shaders wrote in them goes into title memory.
    /// `submitted` is the last value signalled on the fence. `seen` holds
    /// the pieces fetched already since the card was last waited for
    /// (several submissions settled together name the same ones).
    fn card_back(
        &self,
        plan: &CardPlan,
        submitted: &mut u64,
        seen: &mut std::collections::HashSet<(usize, usize, usize)>,
    ) {
        if plan.ranges.is_empty() {
            return;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        // SAFETY: the mapping of the results, which the compare (it has
        // run) wrote.
        let differs = unsafe { std::slice::from_raw_parts(work.differs.1 as *const u32, (work.differs.2 / 4) as usize) };
        // The pieces to fetch: the range, the offsets in its copy, and
        // where the bytes come down.
        let mut pieces: Vec<(usize, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        for (i, r) in plan.ranges.iter().enumerate() {
            for p in 0..r.span.pieces {
                if r.compared && differs[(r.at + p) as usize] == 0 {
                    continue;
                }
                let (lo, hi) = r.shadow.card_piece(&r.span, p);
                // (`VICTORY_ARENA_BY_CHANGED`) What the compare found changed.
                crate::indirect::arena::note_changed(r.shadow.title_start() as usize + lo, hi.saturating_sub(lo));
                if lo < hi
                    && !crate::census::skip_back(r.shadow.title_start(), (hi - lo) as u64)
                    && seen.insert((std::sync::Arc::as_ptr(&r.shadow) as usize, lo, hi))
                {
                    pieces.push((i, lo, hi, total));
                    total += (hi - lo) as u64;
                }
            }
            crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
            crate::census::compared(r.shadow.title_start(), u64::from(r.span.pieces));
        }
        let mut written = vec![0usize; plan.ranges.len()];
        if !pieces.is_empty() {
            if work.fetched.as_ref().is_none_or(|f| f.2 < total) {
                let size = total.next_multiple_of(16 << 20);
                work.fetched = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
            }
            let Some((fetched, data, _)) = &work.fetched else {
                fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes)"));
            };
            let waiting = std::time::Instant::now();
            let (allocator, list) = &work.fetch;
            // SAFETY: the list's last run was waited for; live buffers; the
            // fence and event are ours, under the submission lock.
            let ran = unsafe {
                (|| -> windows::core::Result<()> {
                    // With `VICTORY_QUEUE2` bit 1 the fetch goes to the copy
                    // queue here too, behind everything sent to the direct
                    // queue. Nothing is gained (the card has run it all
                    // when this is reached); the path is the same one the
                    // relay uses, in the plainest run there is.
                    if let Some(second) = crate::queue2::second().filter(|s| s.fetches()) {
                        let copy = second.down.list()?;
                        for &(i, lo, hi, at) in &pieces {
                            copy.1.CopyBufferRegion(fetched, at, &plan.ranges[i].shadow.buffer, lo as u64, (hi - lo) as u64);
                        }
                        let job = second.down.send(copy, &[(&self.fence, *submitted)], total)?;
                        let was_working = working(false);
                        second.down.wait(job, self.event)?;
                        stood(waiting);
                        working(was_working);
                        second.down.stats.waited_us.fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
                        return Ok(());
                    }
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    for &(i, lo, hi, at) in &pieces {
                        list.CopyBufferRegion(fetched, at, &plan.ranges[i].shadow.buffer, lo as u64, (hi - lo) as u64);
                    }
                    list.Close()?;
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    self.card_sent();
                    *submitted += 1;
                    self.queue.Signal(&self.fence, *submitted)?;
                    self.fence.SetEventOnCompletion(*submitted, self.event)?;
                    let was_working = working(false);
                    WaitForSingleObject(self.event, INFINITE);
                    stood(waiting);
                    working(was_working);
                    Ok(())
                })()
            };
            if let Err(e) = ran {
                fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e})"));
            }
            {
                use crate::shadow::CARD_STATS;
                CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
                CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
                CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
                CARD_STATS[6].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
                if crate::census::on() {
                    for &(i, lo, hi, _) in &pieces {
                        crate::census::fetched(plan.ranges[i].shadow.title_start(), (hi - lo) as u64);
                    }
                }
            }
            if wide_merge() && pieces.len() >= 8 {
                // The pieces are apart from one another (each its own bytes
                // of its copy and of title memory): several at a time.
                let fetched = *data;
                let counts: std::sync::Arc<Vec<std::sync::atomic::AtomicUsize>> =
                    std::sync::Arc::new((0..plan.ranges.len()).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect());
                let each = pieces.len().div_ceil(8);
                let jobs: Vec<Box<dyn FnOnce() + Send>> = pieces
                    .chunks(each)
                    .map(|chunk| {
                        let chunk: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, usize, u64)> =
                            chunk.iter().map(|&(i, lo, hi, at)| (plan.ranges[i].shadow.clone(), i, lo, hi, at)).collect();
                        let counts = counts.clone();
                        Box::new(move || {
                            let mut title = Vec::new();
                            for (shadow, i, lo, hi, at) in chunk {
                                // SAFETY: the mapping of the fetched bytes;
                                // the copy into it has run, and the caller
                                // waits for this job.
                                let gpu = unsafe { std::slice::from_raw_parts((fetched + at as usize) as *const u8, hi - lo) };
                                let merged = shadow.card_merge(lo, gpu, &mut title);
                                shadow.walls_merged(gpu.len(), merged);
                                counts[i].fetch_add(merged, Ordering::Relaxed);
                            }
                        }) as Box<dyn FnOnce() + Send>
                    })
                    .collect();
                crate::shadow::wide_run(jobs);
                for (written, count) in written.iter_mut().zip(counts.iter()) {
                    *written += count.load(Ordering::Relaxed);
                }
            } else {
                let mut title = Vec::new();
                for &(i, lo, hi, at) in &pieces {
                    // SAFETY: the mapping of the fetched bytes, `total` of
                    // them; the copy into it has run.
                    let gpu = unsafe { std::slice::from_raw_parts((*data + at as usize) as *const u8, hi - lo) };
                    let merged = plan.ranges[i].shadow.card_merge(lo, gpu, &mut title);
                    plan.ranges[i].shadow.walls_merged(gpu.len(), merged);
                    written[i] += merged;
                }
            }
        }
        for (r, written) in plan.ranges.iter().zip(written) {
            copy_back(1, written as u64);
            let (from, to) = r.declared;
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
            let _ = (from, to);
        }
    }

    /// Runs closed lists and waits for them, and with [`late`] for
    /// everything sent to the card before them, which is settled first.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        // Behind what the queues were given before it and have not run
        // yet (`crate::queue::behind`).
        crate::queue::before_sync();
        let _ = self.submit(runs, false);
    }

    /// A queue's submission with [`late`]: the host lists `lists`, taken
    /// from the title's, go to the card in `order` and are not waited for.
    /// They join the line of flights and are settled, with the others in
    /// the air, at the next point where what they do can be observed
    /// ([`late`] lists those points); their host lists are reset and kept
    /// for reuse then, as a queue does for the lists it ran.
    pub(crate) fn execute_late(&self, lists: Vec<crate::list::Detached>, order: Vec<usize>) {
        if order.is_empty() {
            crate::list::recycle(lists, &order);
            return;
        }
        late_threads();
        let sent = {
            let runs: Vec<&HostList> = order.iter().map(|&i| lists[i].host()).collect();
            self.submit(&runs, true)
        };
        let Some((mut submitted, sent)) = sent else {
            // (Not reached: a submission sent ahead always comes back.)
            crate::list::recycle(lists, &order);
            return;
        };
        let told = !sent.writes.is_empty();
        let most = {
            let mut line = line();
            let Line { line: items, written, lists: flying, target, .. } = &mut *line;
            join_ranges(written, sent.back.iter().flat_map(|(_, ranges)| ranges.iter().copied()));
            *flying += order.len();
            *target = *submitted;
            items.push(Late::Run(Flight {
                number: sent.number,
                fence: *target,
                lists,
                order,
                card: sent.card,
                back: sent.back,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
            *flying >= late_most()
        };
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        RELAY_MOVED.notify_one();
        if told && !late_writes() {
            self.settle(&mut submitted, WHY_WRITES);
        } else if most && relay_exact() {
            // Until there are fewer, not until there are none.
            relay_wait(WHY_MOST, |line| line.lists < late_most());
        } else if most {
            self.settle(&mut submitted, WHY_MOST);
        } else {
            LINE_MOVED.notify_one();
        }
        drop(submitted);
        land();
    }

    /// Sends `list`, a closed list of ours (the present's copy), as a
    /// flight ([`late_free_present`]): not waited for, settled with the
    /// others. The caller keeps the list and does not reset it before the
    /// submission whose number comes back has finished ([`FINISHED`]); 0 if
    /// it ran and was waited for after all.
    fn execute_own(&self, list: &HostList) -> u64 {
        late_threads();
        let Some((submitted, sent)) = self.submit(&[list], true) else {
            return 0;
        };
        let number = sent.number;
        {
            let mut line = line();
            let Line { line: items, written, lists: flying, target, .. } = &mut *line;
            join_ranges(written, sent.back.iter().flat_map(|(_, ranges)| ranges.iter().copied()));
            *flying += 1;
            *target = *submitted;
            items.push(Late::Run(Flight {
                number,
                fence: *target,
                lists: Vec::new(),
                order: Vec::new(),
                card: sent.card,
                back: sent.back,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
        }
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        LINE_MOVED.notify_one();
        RELAY_MOVED.notify_one();
        drop(submitted);
        number
    }

    /// Waits for the card to run every flight in the air ([`late`]) and
    /// does what they left to do, as [`Host::execute`] does for one
    /// submission after its wait: the pieces their shaders wrote in copies
    /// in the card's memory are fetched and merged (each once), the ranges
    /// they declared in the other copies are compared and copied back
    /// (each byte once, whichever flights declared it), and then, in the
    /// order they were given: their bottom-of-pipe writes and the fence
    /// signals behind them. `submitted` is the submission lock's value;
    /// `why` one of the `WHY_` places. Nothing happens if nothing flies.
    ///
    /// Everything in the air is settled, never the first flights alone:
    /// the merge writes the copies' CPU side, which a flight still on the
    /// card may be writing too.
    fn settle(&self, submitted: &mut u64, why: usize) {
        if !late() {
            return;
        }
        let _site = crate::hangwatch::at(&"settle: waiting until every flight has landed (holds the submission lock)");
        if relay() {
            // Nothing is settled here: the thread that lands flights does
            // it, in order, and this waits until nothing is in the air and
            // nothing is being landed. Every point that settled before
            // comes here, so each still sees title memory, fences and
            // bottom-of-pipe values as after a settle.
            relay_wait(why, |line| line.line.is_empty() && !line.settling);
            return;
        }
        let was_working = working(true);
        let mut counted = false;
        loop {
            let items = {
                let mut line = line();
                if line.line.is_empty() {
                    line.settling = false;
                    break;
                }
                line.settling = true;
                line.written.clear();
                line.lists = 0;
                std::mem::take(&mut line.line)
            };
            let flights = items.iter().filter(|i| matches!(i, Late::Run(_))).count();
            if !counted {
                counted = true;
                LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
                LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
            }
            LATE_STATS[11].fetch_add(flights as u64, Ordering::Relaxed);
            if flights > 0 {
                submitting(2, flights);
                let waiting = std::time::Instant::now();
                // SAFETY: the fence and event are ours, under the
                // submission lock; `submitted` is the last value signalled.
                unsafe {
                    match self.fence.SetEventOnCompletion(*submitted, self.event) {
                        Ok(()) => {
                            working(false);
                            WaitForSingleObject(self.event, INFINITE);
                            stood(waiting);
                            working(true);
                        }
                        Err(e) => tracing::warn!(error = %e, "submissions sent ahead not waited for"),
                    }
                }
                let gpu = waiting.elapsed();
                LATE_STATS[12].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                LATE_WAITED[why.min(10)].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                submission_us(1, gpu.as_micros() as u64);
                #[cfg(feature = "diagnostics")]
                crate::copystat::card_waited(gpu.as_micros() as u64, || {
                    format!("{flights} submissions sent ahead, settled together")
                });
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = items
                        .iter()
                        .filter_map(|i| match i {
                            Late::Run(f) => Some(f),
                            _ => None,
                        })
                        .flat_map(|f| f.order.iter().map(|&i| f.lists[i].host()))
                        .collect();
                    fail(format_args!(
                        "host GPU device removed after {flights} submissions sent ahead (VICTORY_LATE_WAIT=1) of {} lists{}{}",
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                submitting(3, flights);
                let copying = std::time::Instant::now();
                crate::readback::consumers::landing();
                let mut seen = std::collections::HashSet::new();
                let mut back: Vec<Declared> = Vec::new();
                for item in &items {
                    let Late::Run(flight) = item else {
                        continue;
                    };
                    if let Some(card) = &flight.card {
                        self.card_back(card, submitted, &mut seen);
                    }
                    for (shadow, ranges) in &flight.back {
                        match back.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                            Some((_, all)) => {
                                for range in ranges {
                                    if !all.contains(range) {
                                        all.push(*range);
                                    }
                                }
                            }
                            None => back.push((shadow.clone(), ranges.clone())),
                        }
                    }
                }
                hand_back(&back);
                copy_back(0, copying.elapsed().as_micros() as u64);
            }
            // In the order given: what tells the title the card is done.
            let mut last = 0;
            let mut landed = Vec::new();
            for item in items {
                match item {
                    Late::Run(mut flight) => {
                        last = flight.number;
                        for &i in &flight.order {
                            read_backs(flight.lists[i].host());
                        }
                        if let Some(card) = flight.card {
                            self.card_done(card);
                        }
                        crate::list::release_landed(&mut flight.lists);
                        landed.push((flight.lists, flight.order));
                    }
                    Late::Write(at, what, bytes) => what.write(at as *mut u8, bytes),
                    Late::Signal(fence, value) => crate::fence::set(fence as crate::object::This, value),
                }
            }
            if last > 0 {
                FINISHED.store(last, Ordering::Release);
                bury(last);
            }
            line().landed.extend(landed);
        }
        if counted {
            self.log_messages();
            submitting(0, 0);
            // What was settled may be what a queue stands at, and the
            // caller may not be one that looks (a read-back, a present).
            if why != WHY_CARD_DONE {
                line().poke = true;
                LINE_MOVED.notify_one();
            }
        }
        working(was_working);
    }

    /// [`Host::execute`], or with `late` the first half of
    /// [`Host::execute_late`]: the lists are prepared and sent, and what is
    /// to be done once they have run comes back with the submission lock
    /// still held.
    fn submit<'a>(&'a self, runs: &[&HostList], late: bool) -> Option<(std::sync::MutexGuard<'a, u64>, Sent)> {
        #[cfg(feature = "diagnostics")]
        let called = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_begins();
        #[cfg(feature = "diagnostics")]
        crate::profile::seen(crate::profile::Seen::Submit);
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = {
            let _site = crate::hangwatch::at(&"submit: waiting for the submission lock");
            self.submitted.lock().unwrap_or_else(|e| e.into_inner())
        };
        let _site = crate::hangwatch::at(&"submit: preparing and sending lists (holds the submission lock)");
        let was_working = working(true);
        submitting(1, runs.len());
        // Flights in the air ([`late`]): a submission that is waited for
        // settles them first, one sent ahead only if preparing it reads
        // what they write.
        if !late {
            self.settle(&mut submitted, WHY_SYNC);
        } else if let Some(why) = {
            let _site = crate::hangwatch::at(&"submit: looking at the lists' steps for reads behind flights (late_blocked, holds a list's step lock)");
            late_blocked(runs)
        } {
            // (P1 meter) What the read waits for, and how it went.
            let meter = (why == WHY_READ).then(read_wait::begin);
            if relay_exact() && why == WHY_READ {
                // Only until the flights that may write what is read have
                // landed: their copy-back's end, not the card's other work.
                let _site = crate::hangwatch::at(&"submit: relay_wait for flights to land before a read (WHY_READ, holds the submission lock)");
                relay_wait(why, |line| blocked_by(&line.written, runs, false).is_none());
            } else {
                self.settle(&mut submitted, why);
            }
            if let Some(meter) = meter {
                read_wait::end(meter);
            }
        }
        // Descriptors the title copied into placed heaps directly, before
        // command lists that index those slots are submitted.
        // (P6, `VICTORY_HEAP_SYNC_ASIDE`) Beside the preparing, waited for
        // before the sweep (when it is due) and before the lists are sent.
        let mut heaps_synced = crate::descriptor::begin_sync_heaps();
        let behind = late && flying();
        // Before this submission counts as started: what the sweep lets go
        // of is named by no list, and nothing runs now (behind flights the
        // sweep waits for a submission that finds none in the air: there is
        // one at every present).
        static SWEPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if !behind {
            if sweep_due() {
                let waiting = std::time::Instant::now();
                heaps_synced.wait();
                prep(15, waiting);
            }
            let sweep_began = std::time::Instant::now();
            sweep();
            prep(9, sweep_began);
            SWEPT.store(now_ms(), Ordering::Relaxed);
        } else if late_free_present() && now_ms() > SWEPT.load(Ordering::Relaxed) + 1500 {
            // With the present sent ahead nothing else brings a moment
            // with nothing in the air: one is made for the sweep.
            self.settle(&mut submitted, WHY_SYNC);
            heaps_synced.wait();
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        }
        let behind = late && flying();
        // The slots that waited for a texture's host copy, before the
        // lists that may read them run.
        let waiting_began = std::time::Instant::now();
        crate::descriptor::make_waiting();
        prep(10, waiting_began);
        let this_submission = STARTED.fetch_add(1, Ordering::AcqRel) + 1;
        // The ranges shaders write are counted for it (with the relay).
        crate::shadow::preparing(this_submission);
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let lock = (started - called).as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch::default();
        // The time a settle of flights takes inside this preparing: the
        // wait for the card and the copy-back have their own places on the
        // frame line (run 92 showed them twice, "preparing" 250 ms for 135).
        let mut settled = std::time::Duration::ZERO;
        // One write-watch asking for everything the submission refreshes
        // (`VICTORY_ASK_ONCE`): per copy, the span of its refreshed ranges.
        if crate::watch::once() {
            let mut spans: std::collections::HashMap<usize, (usize, usize)> = std::collections::HashMap::new();
            for list in runs {
                for step in &list.deferred().copies {
                    if let Step::Refresh(shadow, address, len) = step
                        && !shadow.unwatched()
                    {
                        let span = spans
                            .entry(std::sync::Arc::as_ptr(shadow) as usize)
                            .or_insert((*address, *address + *len));
                        *span = (span.0.min(*address), span.1.max(*address + *len));
                    }
                }
            }
            let ranges: Vec<(usize, usize)> = spans.into_values().collect();
            let asking = std::time::Instant::now();
            crate::watch::ask_once(&ranges);
            prep(3, asking);
            // Behind flights: the title wrote memory that a flight's shaders
            // may write too, and this submission refreshes it. The flights
            // are settled, and the asking made again: the copy-back wrote
            // title memory, which other copies of it must see.
            let stale_began = std::time::Instant::now();
            let stale = behind && stale_under_flights(runs);
            prep(4, stale_began);
            if stale {
                let settling = std::time::Instant::now();
                self.settle(&mut submitted, WHY_STALE);
                settled += settling.elapsed();
                crate::watch::ask_once(&ranges);
            }
        } else if behind && {
            let stale_began = std::time::Instant::now();
            let stale = stale_under_flights(runs);
            prep(4, stale_began);
            stale
        } {
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_STALE);
            settled += settling.elapsed();
        }
        for list in runs {
            list.run_deferred(&mut batch);
            firsts.push(lists.len());
            if let Some(prelude) = list.prelude() {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        batch.flush();
        crate::watch::ask_once_end();
        // The snapshots of the ranges shaders may write, if they were left
        // for now (`crate::shadow::wide_back`): the refreshes are done, the
        // lists have not gone to the card.
        {
            #[cfg(feature = "diagnostics")]
            let snapshots = std::time::Instant::now();
            let snapshots_began = std::time::Instant::now();
            crate::shadow::take_snapshots();
            prep(11, snapshots_began);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::SnapshotUs, snapshots.elapsed().as_micros() as u64);
        }
        // Copies in the card's memory: what changed on their CPU side goes
        // up ahead of the lists, and a compare of what the shaders may
        // write follows them in the same submission.
        let flown = late && relay();
        if flown && self.card_results_low() {
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_MOST);
            settled += settling.elapsed();
        }
        let card_began = std::time::Instant::now();
        let card = self.card_before(runs, flown, *submitted);
        prep(8, card_began);
        if let Some(check) = card.as_ref().and_then(|c| c.check.clone()) {
            lists.push(Some(check));
        }
        // (P6) The host heaps' slots the lists read are made.
        {
            let _site = crate::hangwatch::at(&"submit: waiting for the aside heap sync (VICTORY_HEAP_SYNC_ASIDE)");
            let waiting = std::time::Instant::now();
            heaps_synced.wait();
            prep(15, waiting);
        }
        let prepared = std::time::Instant::now();
        PREP[0].fetch_add((prepared - started).saturating_sub(settled).as_micros() as u64, Ordering::Relaxed);
        if late {
            *submitted += 1;
            // SAFETY: closed lists; the fence is ours.
            let signalled = unsafe {
                self.queue.ExecuteCommandLists(&lists);
                self.queue.Signal(&self.fence, *submitted)
            };
            self.card_sent();
            if let Err(e) = signalled {
                fail(format_args!("a submission sent ahead could not be followed by a signal ({e})"));
            }
            let back = declared_writes(runs);
            let writes = runs.iter().flat_map(|l| l.deferred().writes.clone()).collect();
            submission_us(0, (prepared - started).saturating_sub(settled).as_micros() as u64);
            submitting(0, 0);
            #[cfg(feature = "diagnostics")]
            crate::copystat::submission_ends(lock);
            let sent = Sent {
                number: this_submission,
                card,
                back,
                writes,
            };
            working(was_working);
            return Some((submitted, sent));
        }
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if timing() {
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| &lists[a..b])
                .collect()
        } else {
            vec![&lists[..]]
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [&lists[..]];
        submitting(2, runs.len());
        for (i, group) in groups.iter().enumerate() {
            *submitted += 1;
            let started = std::time::Instant::now();
            // SAFETY: closed lists; the fence and event are ours.
            unsafe {
                self.queue.ExecuteCommandLists(group);
                self.card_sent();
                let waited = self
                    .queue
                    .Signal(&self.fence, *submitted)
                    .and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event));
                match waited {
                    Ok(()) => {
                        working(false);
                        WaitForSingleObject(self.event, INFINITE);
                        stood(started);
                        working(true);
                    }
                    Err(e) => tracing::warn!(error = %e, "host submission not waited for"),
                }
            }
            // With timing on each group is one of `runs` (and the compare
            // of copies in the card's memory, with the last): every
            // command's time on the card goes to the sums by pipeline.
            #[cfg(feature = "diagnostics")]
            if timing() {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                for (ms, op) in runs[i].slowest(frequency, usize::MAX) {
                    crate::copystat::command_took(&op, ms);
                }
            }
            // A submission of one list was never named here (`groups.len()
            // > 1` was the test): the two 50 ms lists of the gameplay
            // frame, each a submission of its own, printed nothing.
            #[cfg(feature = "diagnostics")]
            if timing() && started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
                tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                for (ms, op) in runs[i].slowest(frequency, 6) {
                    tracing::warn!("slow list command, {ms:.1} ms: {op}");
                }
            }
            let _ = (i, started);
        }
        let _ = &firsts;
        FINISHED.store(this_submission, Ordering::Release);
        bury(this_submission);
        let gpu = prepared.elapsed();
        submission_us(0, (prepared - started).as_micros() as u64);
        submission_us(1, gpu.as_micros() as u64);
        #[cfg(feature = "diagnostics")]
        crate::copystat::card_waited(gpu.as_micros() as u64, || {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            let text = format!("{} lists, pipelines {pipelines:?}", runs.len());
            text.chars().take(1200).collect()
        });
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        submitting(3, runs.len());
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        crate::readback::consumers::landing();
        // For copies in the card's memory the compare has run there: the
        // pieces that differ are fetched and merged.
        if let Some(card) = &card {
            self.card_back(card, &mut submitted, &mut std::collections::HashSet::new());
        }
        let back = declared_writes(runs);
        hand_back(&back);
        if let Some(card) = card {
            self.card_done(card);
        }
        #[cfg(feature = "diagnostics")]
        if !back.is_empty() {
            crate::copystat::add(crate::copystat::C::BackSubmissions, 1);
        }
        // Nothing of this submission goes back to the title: no range of
        // shader writes, no value written when it is done. The wait for
        // such a one is only for order and for what the card still holds.
        #[cfg(feature = "diagnostics")]
        if back.is_empty() && runs.iter().all(|l| l.deferred().writes.is_empty()) {
            use crate::copystat::{C, add};
            add(C::NothingBack, 1);
            add(C::NothingBackUs, gpu.as_micros() as u64);
        }
        copy_back(0, copying.elapsed().as_micros() as u64);
        for list in runs {
            // Read backs first: a value written at the bottom of the pipe
            // may be what tells the title that they are there.
            read_backs(list);
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        self.log_messages();
        submitting(0, 0);
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_ends(lock);
        // With the relay: nothing is in the air now, so no fetch still to
        // be merged is older than the sends to card copies so far, and
        // their notes can go.
        if relay() {
            crate::shadow::relay_fetched(*QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        }
        working(was_working);
        None
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = runs.iter().position(|r| Some(r.list.as_raw()) == ours);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = runs.iter().find(|r| Some(r.list.as_raw()) == list);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// With the frame line (`VICTORY_D3D12_DEBUG=1`, diagnostics builds):
    /// the debug layer's messages since the last call go to our log, the
    /// first five of each kind in full, the rest counted, with one line of
    /// the counts by kind.
    ///
    /// They were only read when the device was lost ([`fail`]): a run that
    /// survived (run 205) said nothing of what the layer had found, and
    /// what it finds is what a driver's fault would have to be traced back
    /// to (an object let go while the card uses it, a resource in the
    /// wrong state, a view of something gone).
    pub(crate) fn drain_messages(&self) {
        let Some(queue) = &self.messages else {
            return;
        };
        static SEEN: Mutex<std::collections::BTreeMap<i32, (u64, i32)>> = Mutex::new(std::collections::BTreeMap::new());
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        let stored = unsafe {
            let stored = queue.GetNumStoredMessages();
            for i in 0..stored {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let kind = seen.entry(m.ID.0).or_insert((0, m.Severity.0));
                    kind.0 += 1;
                    if kind.0 <= 5 {
                        let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                        let text = String::from_utf8_lossy(text);
                        let text = text.trim_end_matches('\0');
                        tracing::warn!(id = m.ID.0, severity = m.Severity.0, nth = kind.0, "D3D12 debug layer: {text}");
                    }
                }
            }
            queue.ClearStoredMessages();
            stored
        };
        if stored > 0 {
            let counts: Vec<String> = seen.iter().map(|(id, (n, severity))| format!("id {id} (severity {severity}) x{n}")).collect();
            tracing::info!(
                "D3D12 debug layer messages so far by kind (severity 0 corruption, 1 error, 2 warning, 3 info, 4 message): {}",
                counts.join(", ")
            );
        }
    }

    /// The host device (`crate::cardmeter`).
    pub(crate) fn device(&self) -> &ID3D12Device {
        &self.device
    }

    /// Ticks a second of the queue's timestamps; 0 if it has none.
    pub(crate) fn timestamp_frequency(&self) -> u64 {
        // SAFETY: a plain query.
        unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(0)
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: Option<&ID3D12Resource>,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource, or none with a
        // description (a typed null view).
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    pub(crate) fn close(&self) {
        self.take_coming();
        self.twins_to_depth();
        let scratch = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if let Some((scratch, _)) = scratch {
            self.indirect_common(&scratch);
        }
        self.card_budget_out();
        // The scratch buffers its prelude fills and leaves readable as
        // arguments: common again behind the list, as every scratch buffer.
        for set in self.gather.lock().unwrap_or_else(|e| e.into_inner()).sets.iter().filter(|s| s.records() > 0) {
            // SAFETY: an open list and a live buffer.
            unsafe {
                self.list.ResourceBarrier(&[transition(
                    &set.scratch,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
        }
        // (Card kinds job) The prelude leaves its fills' buffer readable as
        // constants: common again behind the list, as the gathered records'.
        {
            let h = self.hoisted.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(buffer) = h.buffer.as_ref().filter(|_| !h.copies.is_empty()) {
                // SAFETY: an open list and a live buffer.
                unsafe {
                    self.list.ResourceBarrier(&[transition(
                        buffer,
                        D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                }
            }
        }
        self.kinds.lock().unwrap_or_else(|e| e.into_inner()).close(&self.list);
        self.kinds_open.store(false, Ordering::Relaxed);
        crate::cardmeter::end(&self.list, crate::cardmeter::Span::from_raw(self.pipeline_meter.swap(u32::MAX, Ordering::Relaxed)));
        crate::cardmeter::end_named(
            &self.list,
            crate::cardmeter::Span::from_raw(self.meter.swap(u32::MAX, Ordering::Relaxed)),
            &self.pipelines.lock().unwrap_or_else(|e| e.into_inner()),
        );
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self._alive.used();
        // (P6) A pass still running for the recording that ends: waited for
        // (it holds host copies), its refreshes dropped with the steps.
        if let Some((_, _, coming)) = self.coming.lock().unwrap_or_else(|e| e.into_inner()).take() {
            coming.wait();
        }
        self.changes.fetch_add(1, Ordering::Relaxed);
        self.serial.store(next_serial(), Ordering::Relaxed);
        self.refreshes.fetch_add(1, Ordering::Relaxed);
        *self.deferred() = Deferred::default();
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.aliases.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.written.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
            let Gather { closed, sets, spare, .. } = &mut *gather;
            *closed = false;
            for set in sets.drain(..) {
                if spare.len() < GATHER_SETS {
                    spare.push(set.into_scratch());
                }
            }
        }
        if let Some(c) = self.card_constants.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            // Every submission of the list has run: the buffer is free, and
            // in the common state again.
            (c.1, c.2) = (0, false);
        }
        self.card_copied.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.kinds.lock().unwrap_or_else(|e| e.into_inner()).forget();
        self.kinds_open.store(false, Ordering::Relaxed);
        {
            let mut h = self.hoisted.lock().unwrap_or_else(|e| e.into_inner());
            h.used = 0;
            h.copies.clear();
            h.seen.clear();
        }
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
            // Every submission of the list has run: nothing reads its
            // scratch memory any more, and the next list to ask takes it.
            if scratch_pool() {
                give_chunks(std::mem::take(&mut scratch.chunks));
            }
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        self.meter
            .store(crate::cardmeter::begin(&self.list, crate::cardmeter::Kind::Lists).raw(), Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ResetNs, started.elapsed().as_nanos() as u64);
    }

    /// With [`crate::texture::alias_once`]: notes that the list writes host
    /// copy `key` if `state` is one that writes.
    fn note_write(&self, key: usize, state: u32) {
        // Render target, unordered access, depth write, stream out, copy
        // destination, resolve destination.
        const WRITES: u32 = 0x4 | 0x8 | 0x10 | 0x100 | 0x400 | 0x1000;
        if crate::texture::alias_once() && state & WRITES != 0 {
            self.written.lock().unwrap_or_else(|e| e.into_inner()).insert(key);
        }
    }

    /// Whether the list has used title texture `resource`'s host copy yet.
    pub(crate) fn touches(&self, resource: crate::object::This) -> bool {
        crate::resource::host_copy(resource).is_some_and(|copy| {
            self.states
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains_key(&(std::sync::Arc::as_ptr(&copy) as usize))
        })
    }

    /// Leaves an alias copy to the list's prelude.
    pub(crate) fn alias_at_submission(&self, copy: crate::texture::AliasCopy) {
        self.aliases.lock().unwrap_or_else(|e| e.into_inner()).push(copy);
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let texture = crate::resource::host_resource(resource)?;
        let copy = crate::resource::host_copy(resource)?;
        let wanted = legal_state(texture, wanted);
        self.note_write(std::sync::Arc::as_ptr(&copy) as usize, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(&copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                if before != wanted {
                    self.changes.fetch_add(1, Ordering::Relaxed);
                }
                (before != wanted).then_some((texture, before))
            }
        }
    }

    /// Moves host texture `copy` to `wanted` (a host state) in this list,
    /// as [`HostList::transition`] does a title resource's host copy, and
    /// records the barrier: for textures that are not a title resource's
    /// host copy (colour twins) and for the copies between the two.
    fn settle(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) {
        let wanted = legal_state(&copy.texture, wanted);
        self.note_write(std::sync::Arc::as_ptr(copy) as usize, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let before = match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                if before != wanted {
                    self.changes.fetch_add(1, Ordering::Relaxed);
                }
                (before != wanted).then_some(before)
            }
        };
        drop(states);
        if let Some(before) = before {
            let barrier = transition(
                &copy.texture,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(wanted as i32),
            );
            // SAFETY: an open list and a live texture.
            unsafe { self.list.ResourceBarrier(&[barrier]) };
        }
    }

    /// A use of title texture `resource` about to be recorded, if it is a
    /// depth texture with a colour twin (false if it is not): of the twin
    /// (`colour`: unordered access) or of the depth texture (depth target,
    /// clear, shader read, copy, resolve), writing it or only reading. The
    /// side used is first brought up to date: by a copy recorded here if
    /// this list wrote the other side since, by the list's prelude if the
    /// list uses the side as it finds it. The twin is left in the
    /// unordered access state; the caller moves the depth texture to the
    /// state its use needs, as for any texture. A twin still ahead when the
    /// list closes is copied then (`twins_to_depth`).
    /// shortcut: every depth target bound for a draw counts as written
    /// (also when the draw only tests), and every unordered access view a
    /// shader names as written; views a shader picks from the heap by a
    /// computed index are not seen here at all.
    pub(crate) fn twin_use(
        &self,
        resource: crate::object::This,
        colour: bool,
        write: bool,
    ) -> bool {
        const UNORDERED_ACCESS: u32 = 0x8;
        // Asked for every texture a draw reads: most have no twin.
        let Some(twin) = crate::resource::twin(resource) else {
            return false;
        };
        let Some(depth) = crate::resource::host_copy(resource) else {
            return false;
        };
        let side = usize::from(colour);
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let u = twins
            .entry(std::sync::Arc::as_ptr(&twin) as usize)
            .or_insert_with(|| TwinUse {
                depth: depth.clone(),
                twin: twin.clone(),
                needs: [false; 2],
                behind: [None; 2],
                writing: false,
                late: false,
            });
        let copy = match u.behind[side] {
            Some(behind) => behind,
            None => {
                u.needs[side] = true;
                false
            }
        };
        u.behind[side] = Some(false);
        match (write, colour) {
            // Shaders write the twin when the draw or dispatch runs: what
            // the same one reads of the depth texture is prepared before
            // (`twins_written`).
            (true, true) => u.writing = true,
            (true, false) => u.behind[1] = Some(true),
            (false, _) => {}
        }
        // shortcut: the depth plane's twin is not drawn into the plane
        // here: the pass sets a pipeline and a target of ours, which a list
        // still recording the title's commands cannot take. This use sees
        // the plane as it was; the pass runs when the list closes
        // (`twins_to_depth`). Counted.
        if copy && !colour && twin.plane.is_some() {
            u.late = true;
            crate::resource::PLANES[2].fetch_add(1, Ordering::Relaxed);
        }
        drop(twins);
        // (Neptune) Anything but a read with nothing to bring over changes what a later
        // read finds (`HostList::changes`).
        if write || colour || copy {
            self.changes.fetch_add(1, Ordering::Relaxed);
        }
        if copy {
            self.twin_copy(&depth, &twin, colour);
        } else if colour {
            self.settle(&twin.colour, UNORDERED_ACCESS);
        }
        // The depth plane's twin is written through unordered access too.
        if colour && let Some(plane) = &twin.plane {
            self.settle(plane, UNORDERED_ACCESS);
        }
        true
    }

    /// Records the pass that draws the twin of the depth plane of `depth`
    /// into the plane, if `twin` has one, with the moves to the states the
    /// pass needs; the twin is left in the unordered access state. Only
    /// for a list that records nothing of the title's afterwards
    /// (`record_plane_pass`).
    fn pass_plane(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const DEPTH_WRITE: u32 = 0x10;
        const PIXEL_SHADER_RESOURCE: u32 = 0x80;
        let Some(plane) = &twin.plane else { return };
        self.settle(depth, DEPTH_WRITE);
        self.settle(plane, PIXEL_SHADER_RESOURCE);
        record_plane_pass(&self.list, &depth.texture, &plane.texture);
        self.settle(plane, UNORDERED_ACCESS);
    }

    /// Records the copy of a depth texture into its colour twin (`colour`)
    /// or of the twin into the depth texture, with the moves to the copy
    /// states. The twin is left in the unordered access state.
    fn twin_copy(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
        colour: bool,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let (from, to) = if colour {
            (depth, &twin.colour)
        } else {
            (&twin.colour, depth)
        };
        self.settle(from, COPY_SOURCE);
        self.settle(to, COPY_DEST);
        record_twin_copy(&self.list, &depth.texture, twin, colour);
        self.settle(&twin.colour, UNORDERED_ACCESS);
    }

    /// Before the list closes: the depth textures whose twins it wrote last
    /// are given what was written, and left readable by shaders. Shaders
    /// that pick a view from the heap by a computed index are not seen by
    /// `twin_use` and read the depth texture in lists that never name it
    /// (in a captured frame of the title no table names a shadow
    /// map for reading but the pass that fills the next one).
    fn twins_to_depth(&self) {
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let written: Vec<_> = twins
            .values_mut()
            .filter(|u| u.behind[0] == Some(true) || u.late)
            .map(|u| {
                // Not copied again if the list copied since the last write
                // and only the depth plane's pass is left.
                let copy = u.behind[0] == Some(true);
                u.behind[0] = Some(false);
                u.late = false;
                (u.depth.clone(), u.twin.clone(), copy)
            })
            .collect();
        drop(twins);
        for (depth, twin, copy) in &written {
            if *copy {
                self.twin_copy(depth, twin, false);
            }
            // Nothing of the title's is recorded after this: the pass may
            // leave its own pipeline and target set.
            self.pass_plane(depth, twin);
            self.settle(depth, SHADER_RESOURCE);
        }
    }

    /// The draw or dispatch just prepared is recorded: the twins it writes
    /// now hold what their depth textures lack.
    pub(crate) fn twins_written(&self) {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values_mut().filter(|u| u.writing) {
            u.writing = false;
            u.behind[0] = Some(true);
            self.changes.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with. For depth textures with a
    /// colour twin it first copies the side the list uses as it finds it
    /// from the other, if a list submitted earlier left it behind.
    fn prelude(&self) -> Option<ID3D12CommandList> {
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let host_state = |state: u32| D3D12_RESOURCE_STATES(state as i32);
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        // The copies (depth texture, twin, whether into the twin); the
        // sides are then as this list leaves them.
        let mut syncs = Vec::new();
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values() {
            let real = u.twin.behind.load(Ordering::Relaxed);
            let mut behind = [real & 1 != 0, real & 2 != 0];
            for side in [0, 1] {
                if u.needs[side] && behind[side] {
                    syncs.push((u.depth.clone(), u.twin.clone(), side == 1));
                    behind = [false; 2];
                }
            }
            let left = [0, 1].map(|side| u.behind[side].unwrap_or(behind[side]));
            let left = u32::from(left[0]) | (u32::from(left[1]) << 1);
            u.twin.behind.store(left, Ordering::Relaxed);
        }
        drop(twins);
        // A copy runs with its textures in the copy states: each moves
        // there from its real state, and from there to the state the list
        // starts with (back to its real state if the list does not use it).
        let mut to_copy = Vec::new();
        let mut barriers = Vec::new();
        let mut copying = std::collections::HashMap::new();
        for (depth, twin, colour) in &syncs {
            let (from, to) = if *colour {
                (depth, &twin.colour)
            } else {
                (&twin.colour, depth)
            };
            for (texture, state) in [(from, COPY_SOURCE), (to, COPY_DEST)] {
                let key = std::sync::Arc::as_ptr(texture) as usize;
                let real = texture.state.load(Ordering::Relaxed);
                copying.insert(key, state);
                if real != state {
                    to_copy.push(transition(
                        &texture.texture,
                        host_state(real),
                        host_state(state),
                    ));
                    if !states.contains_key(&key) {
                        barriers.push(transition(
                            &texture.texture,
                            host_state(state),
                            host_state(real),
                        ));
                    }
                }
            }
        }
        barriers.extend(states.iter().filter_map(|(key, (first, last, copy))| {
            let real = copy.state.swap(*last, Ordering::Relaxed);
            let real = copying.get(key).copied().unwrap_or(real);
            (real != *first)
                .then(|| transition(&copy.texture, host_state(real), host_state(*first)))
        }));
        // The records the list's draws read from slots (`crate::indirect::gather`).
        let gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
        let gathered = gather.sets.iter().any(|s| s.records() > 0);
        // Alias copies (`crate::texture::alias_once`, soldsawn eb0d648):
        // this submission gets a number; a copy runs unless it ran in a
        // submission after the source's last write; then this list's writes
        // are marked.
        type Run = (ID3D12Resource, ID3D12Resource, std::sync::Arc<crate::texture::Through>);
        let aliases: Vec<Run> = {
            let wanted = self.aliases.lock().unwrap_or_else(|e| e.into_inner());
            let written = self.written.lock().unwrap_or_else(|e| e.into_inner());
            let mut marks = ALIAS_WRITES.lock().unwrap_or_else(|e| e.into_inner());
            let marks = marks.get_or_insert_with(Default::default);
            let mut run = Vec::new();
            if !wanted.is_empty() || !written.is_empty() {
                marks.seq += 1;
                let seq = marks.seq;
                for a in wanted.iter() {
                    let source = std::sync::Arc::as_ptr(&a.source) as usize;
                    let key = (source, a.source.texture.as_raw() as usize, a.to.as_raw() as usize);
                    let last_write = marks.written.get(&source).copied().unwrap_or(0);
                    let fresh = marks.copied.get(&key).is_some_and(|&at| last_write < at);
                    ALIAS_COUNTS[usize::from(fresh)].fetch_add(1, Ordering::Relaxed);
                    if !fresh {
                        marks.copied.insert(key, seq);
                        run.push((a.source.texture.clone(), a.to.clone(), a.through.clone()));
                    }
                }
                for &key in written.iter() {
                    marks.written.insert(key, seq);
                }
            }
            run
        };
        // (Card kinds job) Constant fills moved here ([`HostList::hoist_fill`]).
        let hoisted = self.hoisted.lock().unwrap_or_else(|e| e.into_inner());
        let fills = !hoisted.copies.is_empty() && hoisted.buffer.is_some();
        if barriers.is_empty() && syncs.is_empty() && !gathered && aliases.is_empty() && !fills {
            return None;
        }
        let Some((allocator, list)) = self
            .prelude
            .get_or_init(|| host().and_then(|h| h.make_prelude()))
            .as_ref()
        else {
            fail(format_args!("no host list for a state prelude"));
        };
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                // (Card kinds job) The prelude's time on the card.
                let timed = crate::cardmeter::begin_prelude(list);
                if !to_copy.is_empty() {
                    list.ResourceBarrier(&to_copy);
                }
                for (depth, twin, colour) in &syncs {
                    record_twin_copy(list, &depth.texture, twin, *colour);
                }
                if !barriers.is_empty() {
                    list.ResourceBarrier(&barriers);
                }
                if gathered {
                    // The slots must hold clamped records before a draw
                    // reads them: a prelude that cannot record the passes
                    // is no prelude.
                    let filled = match &gather.budget {
                        Some(budget) => crate::indirect::record_gathers(list, &gather.sets, budget),
                        None => false,
                    };
                    if !filled {
                        fail(format_args!("a list's gathered indirect records could not be clamped in its prelude"));
                    }
                }
                for (from, to, through) in &aliases {
                    crate::texture::record_alias(list, from, to, through);
                }
                if fills && let Some(buffer) = hoisted.buffer.as_ref() {
                    // What earlier lists' shaders wrote to the sources
                    // first (an in-place fill waits for them too); the
                    // buffer is common here (it decays behind every
                    // submission) and is left readable as constants for the
                    // list, which makes it common again when it closes.
                    list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        ..Default::default()
                    }]);
                    for (source, from, at, len) in &hoisted.copies {
                        list.CopyBufferRegion(buffer, *at, source, *from, *len);
                    }
                    list.ResourceBarrier(&[transition(
                        buffer,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                    )]);
                }
                crate::cardmeter::end(list, timed);
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// From here on the list gathers no records (`crate::indirect::gather`):
    /// what is about to be recorded may write arguments.
    pub(crate) fn gather_close(&self) {
        self.gather.lock().unwrap_or_else(|e| e.into_inner()).closed = true;
    }

    /// Whether [`HostList::gather_close`] was called on the list.
    pub(crate) fn gather_closed(&self) -> bool {
        self.gather.lock().unwrap_or_else(|e| e.into_inner()).closed
    }

    /// (Card kinds job) A command of ours of `kind` is about to be recorded
    /// inside the title's list (`crate::cardmeter::Inside`): counted, and
    /// timed with the run of such commands it opens or continues.
    pub(crate) fn inside(&self, kind: crate::cardmeter::Inside) {
        let mut k = self.kinds.lock().unwrap_or_else(|e| e.into_inner());
        k.inside(&self.list, kind);
        self.kinds_open.store(k.is_open(), Ordering::Relaxed);
    }

    /// (Card kinds job) The title's next command is about to be recorded:
    /// the run of our commands open, if any, ends here.
    pub(crate) fn inside_end(&self) {
        if !self.kinds_open.load(Ordering::Relaxed) {
            return;
        }
        self.kinds.lock().unwrap_or_else(|e| e.into_inner()).end(&self.list);
        self.kinds_open.store(false, Ordering::Relaxed);
    }

    /// A slot for the record at `offset` of the host copy `src` in the
    /// scratch buffer the list's prelude fills (`crate::indirect::gather`,
    /// which checked the record): the buffer and the slot's place in it.
    ///
    /// (Neptune `VICTORY_CLAMP_AHEAD`) With `ahead`, also after
    /// [`HostList::gather_close`]: the caller has checked that nothing the
    /// list recorded so far was declared to write the record.
    pub(crate) fn gather_slot(&self, src: &ID3D12Resource, src_id: u64, offset: u32, (kind, stride): (u32, u32), ahead: bool) -> Option<(ID3D12Resource, u64)> {
        let none = |why: usize| {
            crate::indirect::GATHER_NONE[why].fetch_add(1, Ordering::Relaxed);
            None
        };
        let mut gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
        if gather.closed && !ahead {
            return none(1);
        }
        let found = gather
            .sets
            .iter()
            .position(|s| s.src_id == src_id && s.kind == kind && s.stride == stride && s.src.as_raw() == src.as_raw());
        let at = match found {
            Some(at) => at,
            None => {
                if gather.sets.len() >= GATHER_SETS {
                    return none(2);
                }
                // The prelude zeroes the budget as the list zeroes its own.
                if self.list.cast::<ID3D12GraphicsCommandList2>().is_err() {
                    return none(4);
                }
                let Some(host) = host() else {
                    return none(4);
                };
                if gather.budget.is_none() {
                    let Some(budget) = host.uav_buffer(256) else {
                        return none(4);
                    };
                    name(&budget, || "indirect budget of gathered records".to_owned());
                    gather.budget = Some(budget);
                }
                let scratch = match gather.spare.pop() {
                    Some(scratch) => scratch,
                    None => {
                        let Some(scratch) = host.uav_buffer(u64::from(crate::indirect::GATHER_MOST) * u64::from(crate::indirect::GATHER_STRIDE)) else {
                            return none(4);
                        };
                        name(&scratch, || "indirect arguments, gathered".to_owned());
                        scratch
                    }
                };
                let Some(table) = self.scratch(u64::from(crate::indirect::GATHER_MOST) * 4) else {
                    gather.spare.push(scratch);
                    return none(4);
                };
                gather.sets.push(crate::indirect::GatherSet::new(src.clone(), src_id, (kind, stride), scratch, table));
                gather.sets.len() - 1
            }
        };
        let set = &mut gather.sets[at];
        let Some(slot) = set.slot(offset) else {
            return none(3);
        };
        Some((set.scratch.clone(), u64::from(slot) * u64::from(stride)))
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if !dred_on() {
            crate::capture::op(std::ptr::from_ref(self) as usize, what);
            return;
        }
        {
            let started = std::time::Instant::now();
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            crate::capture::op(std::ptr::from_ref(self) as usize, || what.clone());
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
            crate::copystat::add(crate::copystat::C::NoteNs, started.elapsed().as_nanos() as u64);
        }
    }

    /// Whether nothing recorded since the list was opened does any work:
    /// no draw or dispatch (the first keeps its pipeline), no copy, clear,
    /// upload or resolve (they defer a step or move a texture's state), no
    /// bottom-of-pipe write. State set on the list does not count.
    pub(crate) fn untouched(&self) -> bool {
        {
            let deferred = self.deferred();
            if !(deferred.copies.is_empty()
                && deferred.uploads.is_empty()
                && deferred.kept.is_empty()
                && deferred.textures.is_empty()
                && deferred.writes.is_empty()
                && deferred.backs.is_empty())
            {
                return false;
            }
        }
        self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.twins.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
        drop(pipelines);
        // `VICTORY_CARD_METER_PIPELINES=1` (fxceless 3973179): the pipeline
        // before it ends here on the card's clock.
        if crate::cardmeter::pipelines() {
            crate::cardmeter::end(&self.list, crate::cardmeter::Span::from_raw(self.pipeline_meter.swap(u32::MAX, Ordering::Relaxed)));
            self.pipeline_meter.store(crate::cardmeter::begin_named(&self.list, name).raw(), Ordering::Relaxed);
        }
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        // (Neptune) Title memory or a buffer changes at this place in the list: a table
        // made ready before it is made ready again after it (`crate::draw`).
        self.changes.fetch_add(1, Ordering::Relaxed);
        let mut deferred = self.deferred();
        COPIES_JOINED[0].fetch_add(1, Ordering::Relaxed);
        // A copy that goes on where the one recorded just before it ended,
        // source and destination both: with `join_copies` the two are one
        // step.
        if let Some(Step::Copy(d, s, n)) = deferred.copies.last_mut()
            && continues((*d, *s, *n), (dst as usize, src as usize, len))
        {
            COPIES_JOINED[1].fetch_add(1, Ordering::Relaxed);
            if join_copies() {
                *n += len;
                return;
            }
        }
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
        self.refreshes.fetch_add(1, Ordering::Relaxed);
        deferred.not_card.push((src as usize, len, read_wait::take_why()));
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        let why = read_wait::take_why();
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Fill(dst as usize, src as usize, len));
        deferred.not_card.push((src as usize, len, why));
    }

    /// Whether a constant fill of title memory `[from, to)` goes on the
    /// card ([`HostList::card_constants`]): with [`fill_on_card`] and
    /// [`late`], where flights in the air may write it, or where this list
    /// copied on the card before ([`HostList::card_copied`]). The copy it
    /// reads must then pass [`HostList::card_holds`].
    pub(crate) fn fill_wants_card(&self, from: usize, to: usize) -> bool {
        if !(fill_on_card() && late()) {
            read_wait::why(read_wait::NC_SWITCH);
            return false;
        }
        if flights_write(from, to) || self.card_copied(from, to) {
            read_wait::why(read_wait::NC_LATER);
            return true;
        }
        // (P1 step 2, `VICTORY_FILL_PREDICT_CARD`) The producer is not in
        // the air yet, but the bytes are known to be rewritten by shaders:
        // a held piece, or declared shader-written in this frame or the two
        // before. On the card, behind whatever writes them.
        if read_wait::predict_card() && read_wait::predicted(from, to - from) {
            crate::framelog::add(crate::framelog::T::PredictedCard, 1);
            read_wait::why(read_wait::NC_LATER);
            return true;
        }
        read_wait::why(read_wait::NC_UNDECLARED);
        false
    }

    /// Whether this list copied title memory touching `[from, to)` on the
    /// card (`crate::list` `gpu_buffer_copy`): what the list does with it
    /// afterwards has to run on the card too, after that copy, since steps
    /// at submission run before every command of the list.
    pub(crate) fn card_copied(&self, from: usize, to: usize) -> bool {
        from < to && self.card_copied.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|&(a, e, _)| a < to && e > from)
    }

    /// Notes title memory `[from, to)` copied on the card by this list,
    /// into `shadow`.
    pub(crate) fn note_card_copy(&self, shadow: &crate::shadow::Shadow, from: usize, to: usize) {
        let copy = std::ptr::from_ref(shadow) as usize;
        self.card_copied.lock().unwrap_or_else(|e| e.into_inner()).push((from, to, copy));
    }

    /// Whether the card reading or writing `[from, to)` through `shadow`
    /// sees what the title memory there will hold once the flights have
    /// landed: every part of it flights in the air may write is a range
    /// `shadow`'s own shaders may be writing (another copy of the same
    /// bytes, a title buffer inside a bigger one with a copy of its own,
    /// holds the flights' writes elsewhere until they land), and what this
    /// list copied there on the card went into `shadow`. If not, the old
    /// way (a settle, then the CPU) is taken.
    pub(crate) fn card_holds(&self, shadow: &crate::shadow::Shadow, from: usize, to: usize) -> bool {
        let copy = std::ptr::from_ref(shadow) as usize;
        if self
            .card_copied
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|&(a, e, c)| a < to && e > from && c != copy)
        {
            return false;
        }
        let parts = clipped(&line().written, from, to);
        parts.is_empty() || shadow.pending_holds(&parts)
    }

    /// With [`fill_on_card`]: `len` bytes of the host copy `shadow` from title
    /// address `address`, copied on the card where the list stands into its
    /// device-local constants buffer: the GPU address of the bytes there (on
    /// 256 bytes, for a constant buffer view), or `None` (no buffer, no room).
    /// Call after the copy's [`crate::shadow::Shadow::refresh`] of the range.
    ///
    /// A fill of upload memory reads title memory on the CPU at submission,
    /// and a submission whose fill reads what a flight's shaders write waits
    /// for the flight to land (52-80 ms a frame in Trexies' world, one table
    /// constant buffer of 512 bytes). Read on the card, behind the flight on
    /// the same queue, those bytes are what the shaders wrote without the
    /// wait; title writes to them reach the copy first, through its refresh.
    /// The copy reads the host copy exactly where and when a shader of the
    /// list's next draw would: whatever keeps that read right (refreshes
    /// before the lists, a masked merge ahead of them on the card, the
    /// queue's order behind earlier submissions) keeps this one right.
    pub(crate) fn card_constants(&self, shadow: &crate::shadow::Shadow, address: u64, len: u64) -> Option<u64> {
        const SIZE: u64 = 1 << 20;
        let host = host()?;
        let offset = shadow.offset_in(address)?;
        if len == 0 || shadow.left(address) < len {
            return None;
        }
        // (Card kinds job, `VICTORY_CARD_FILLS_BATCH`) Into the prelude when
        // nothing recorded before it in the list may change its source: the
        // prelude runs just before the list, after everything that ran
        // ahead of the list on the queue, so it copies the same bytes, into
        // a place of its own no draw read before.
        let (a, e) = (address as usize, (address + len) as usize);
        let declared = self.gpu_wrote_any(a, len as usize) || self.card_copied(a, e);
        let place = fill_place(fills_batch(), self.gather_closed(), declared);
        let counted = match place {
            FillPlace::Prelude => match self.hoist_fill(host, shadow, offset, len) {
                Some(gpu) => return Some(gpu),
                None => FILL_OTHER,
            },
            FillPlace::Written => FILL_WRITTEN,
            FillPlace::Storing => FILL_STORING,
            FillPlace::Off => FILL_OTHER,
        };
        FILLS[counted].fetch_add(1, Ordering::Relaxed);
        let room = len.next_multiple_of(256);
        let mut c = self.card_constants.lock().unwrap_or_else(|e| e.into_inner());
        if c.is_none() {
            let buffer = host.uav_buffer(SIZE)?;
            name(&buffer, || "list constants on the card".into());
            *c = Some((buffer, 0, false));
        }
        let (buffer, used, readable) = c.as_mut()?;
        let buffer: &ID3D12Resource = buffer;
        if *used + room > SIZE {
            return None;
        }
        let at = *used;
        *used += room;
        let constants = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        #[cfg(feature = "diagnostics")]
        self.note_op(|| format!("constants filled on the card from {address:#x} ({len} bytes)"));
        // (Card kinds job) Timed with the run of fills it is part of.
        self.inside(crate::cardmeter::Inside::Fill);
        crate::cardmeter::inside_barriers(crate::cardmeter::Inside::Fill, 1, 1 + u64::from(*readable));
        // SAFETY: an open list; both buffers are live and the ranges inside
        // them (checked above).
        unsafe {
            if *readable {
                self.list.ResourceBarrier(&[transition(buffer, constants, D3D12_RESOURCE_STATE_COPY_DEST)]);
            }
            // What shaders of this list wrote to the copy before, first.
            self.list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                ..Default::default()
            }]);
            self.list.CopyBufferRegion(buffer, at, &shadow.buffer, offset, len);
            self.list.ResourceBarrier(&[transition(buffer, D3D12_RESOURCE_STATE_COPY_DEST, constants)]);
            *readable = true;
            Some(buffer.GetGPUVirtualAddress() + at)
        }
    }

    /// (Card kinds job) [`HostList::card_constants`] moved to the list's
    /// prelude: `len` bytes at `offset` of `shadow`'s buffer get a place in
    /// the list's prelude buffer (the same bytes asked again share one),
    /// copied there by the prelude; the GPU address of the place, or `None`
    /// (no buffer, no room: the caller fills in place).
    fn hoist_fill(&self, host: &Host, shadow: &crate::shadow::Shadow, offset: u64, len: u64) -> Option<u64> {
        const SIZE: u64 = 1 << 20;
        let mut h = self.hoisted.lock().unwrap_or_else(|e| e.into_inner());
        let key = (shadow.buffer.as_raw() as usize, offset, len);
        if h.buffer.is_none() {
            let buffer = host.uav_buffer(SIZE)?;
            name(&buffer, || "list constants on the card, filled in the prelude".into());
            h.buffer = Some(buffer);
        }
        // SAFETY: a live buffer.
        let base = unsafe { h.buffer.as_ref()?.GetGPUVirtualAddress() };
        if let Some(&at) = h.seen.get(&key) {
            FILLS[FILL_HOISTED].fetch_add(1, Ordering::Relaxed);
            FILLS[FILL_SHARED].fetch_add(1, Ordering::Relaxed);
            return Some(base + at);
        }
        let room = len.next_multiple_of(256);
        if h.used + room > SIZE {
            return None;
        }
        let at = h.used;
        h.used += room;
        if h.copies.is_empty() {
            FILLS[FILL_LISTS].fetch_add(1, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        self.note_op(|| format!("constants filled on the card in the prelude from offset {offset:#x} ({len} bytes)"));
        match h.copies.last_mut() {
            // Right behind the last copy's place, and as far behind its
            // source: one copy (the bytes between are the padding of the
            // last place, read from the source buffer and never used).
            Some(last) if last.0.as_raw() == shadow.buffer.as_raw() && last.2 + last.3 <= at && last.1 + (at - last.2) == offset => {
                last.3 = at - last.2 + len;
                FILLS[FILL_JOINED].fetch_add(1, Ordering::Relaxed);
            }
            _ => h.copies.push((shadow.buffer.clone(), offset, at, len)),
        }
        h.seen.insert(key, at);
        FILLS[FILL_HOISTED].fetch_add(1, Ordering::Relaxed);
        Some(base + at)
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        self.changes.fetch_add(1, Ordering::Relaxed);
        shader_writes(address, len, false);
        let mut deferred = self.deferred();
        deferred.gpu_ranges.push((address, address.saturating_add(len)));
        deferred.copies.push(Step::GpuWrite(shadow.clone(), address, len));
    }

    /// A read back: once the list has run, `rows` rows of `row` bytes go
    /// from `buffer` (rows 256-byte pitched, see `PlacedFootprint::host_rows`)
    /// to title memory at `dst`, pitched `pitch`. The buffer lives until
    /// the list is reset.
    pub(crate) fn defer_read_back(&self, buffer: ID3D12Resource, dst: *mut u8, [row, rows, pitch]: [usize; 3]) {
        // SAFETY: a live buffer.
        let bytes = unsafe { buffer.GetDesc() }.Width;
        self.changes.fetch_add(1, Ordering::Relaxed);
        READ_BACKS[0].fetch_add(1, Ordering::Relaxed);
        READ_BACKS[1].fetch_add(bytes, Ordering::Relaxed);
        let mut deferred = self.deferred();
        deferred.back_bytes += bytes;
        deferred
            .backs
            .push((buffer, dst as usize, [row, rows, pitch]));
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.changes.fetch_add(1, Ordering::Relaxed);
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// Whether the list, once it has run, stores at `address` a value of
    /// `bytes` bytes or more for which `holds` says yes (one of its
    /// bottom-of-pipe writes, as [`late_met`] reads them of a flight;
    /// `VICTORY_JOIN_PARTS`, fxceless 3973179).
    pub(crate) fn will_write(&self, address: usize, bytes: usize, holds: impl Fn(u64) -> bool) -> bool {
        self.deferred()
            .writes
            .iter()
            .any(|(at, what, n)| *at == address && *n >= bytes && matches!(what, crate::queue::Written::Value(v) if holds(*v)))
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().hold_upload(buffer.clone(), size);
        Some((buffer, data))
    }

    /// Whether shaders of this list, as recorded so far, may write title
    /// memory in `[address, address + len)` (a range declared with
    /// [`HostList::defer_gpu_write`] touches it).
    pub(crate) fn gpu_wrote(&self, address: usize, len: usize) -> bool {
        let end = address.saturating_add(len);
        self.deferred().gpu_ranges.iter().any(|&(from, to)| from < end && address < to)
    }

    /// Shaders of the list write `[address, address + len)` of a copy
    /// placed in title memory (`Shadow::gpu_writes`): noted for
    /// [`HostList::gpu_wrote_any`] and [`shader_written`].
    pub(crate) fn note_in_place_write(&self, address: usize, len: usize) {
        shader_writes(address, len, true);
        self.deferred().in_place_ranges.push((address, address.saturating_add(len)));
    }

    /// [`HostList::gpu_wrote`], and writes of the list's shaders to copies
    /// placed in title memory: what reads a record before the list's own
    /// shaders could have written it must ask.
    pub(crate) fn gpu_wrote_any(&self, address: usize, len: usize) -> bool {
        let end = address.saturating_add(len);
        let d = self.deferred();
        d.gpu_ranges.iter().chain(&d.in_place_ranges).any(|&(from, to)| from < end && address < to)
    }

    /// For a copy out of title buffer memory that this list's shaders
    /// write (a dispatch fills a buffer, the list then copies the buffer
    /// into a texture): a host buffer holding `[address, address + len)`
    /// as it is at this point of the list, in the copy source state. The
    /// bytes are copied on the card from the memory's host copy, after the
    /// dispatches recorded before; the title's memory only gets them once
    /// the list has run, so a copy filled from it at submission took what
    /// was there before the list (zeros the first time). `None` if no
    /// shader of the list writes the range, it has no host copy, or
    /// [`copy_written_source`] is off: the caller then copies from title
    /// memory as before. (allforks a9b474f, PalmVIDev 4d1ce258.)
    pub(crate) fn staged_written_source(&self, host: &Host, address: u64, len: u64) -> Option<ID3D12Resource> {
        if len == 0 || !copy_written_source() || !self.gpu_wrote(address as usize, usize::try_from(len).ok()?) {
            return None;
        }
        let shadow = crate::shadow::find(address, len)?;
        let offset = shadow.offset_in(address)?;
        if shadow.left(address) < len {
            return None;
        }
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: len,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut staged: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a buffer.
        let made = unsafe {
            host.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut staged,
            )
        };
        if let Err(e) = made {
            tracing::warn!(error = %e, len, "no buffer to stage a shader-written copy source");
            return None;
        }
        let staged = staged?;
        name(&staged, || format!("shader-written copy source {address:#x}+{len:#x}"));
        // What the title wrote there itself reaches the host copy at
        // submission, as for a shader that reads the range.
        shadow.refresh(self, address, len);
        // (Card kinds job) Ours, inside the title's list.
        self.inside(crate::cardmeter::Inside::Other);
        crate::cardmeter::inside_barriers(crate::cardmeter::Inside::Other, 1, 1);
        // SAFETY: an open list; the range lies in the host copy (checked
        // above) and the staged buffer holds `len` bytes.
        unsafe {
            // What shaders of this list wrote to the copy before, first.
            self.list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                ..Default::default()
            }]);
            self.list.CopyBufferRegion(&staged, 0, &shadow.buffer, offset, len);
            self.list.ResourceBarrier(&[transition(
                &staged,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
        }
        self.keep(&staged);
        STAGED_WRITTEN.fetch_add(1, Ordering::Relaxed);
        let n = STAGED_WRITTEN.load(Ordering::Relaxed);
        if n == 1 || n.is_multiple_of(1000) {
            tracing::info!(
                so_far = n,
                bytes = len,
                "a buffer-to-texture copy takes what the list's own shaders wrote (VICTORY_COPY_WRITTEN_SOURCE=0: from title memory)"
            );
        }
        Some(staged)
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().hold_upload(buffer, len);
            return Some((gpu, data));
        }
        // The chunk in use while `len` fits in what is left of it, else the
        // next one (a list's first chunk may be a small one).
        while s.current < s.chunks.len() && s.used + len > s.chunks[s.current].1 {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let size = if scratch_pool() && s.chunks.is_empty() && len <= SCRATCH_FIRST {
                SCRATCH_FIRST
            } else {
                SCRATCH_CHUNK
            };
            let chunk = take_chunk(host, size)?;
            s.chunks.push((chunk, size));
            s.used = 0;
        }
        let (data, gpu) = (s.chunks[s.current].0.data, s.chunks[s.current].0.gpu);
        let at = s.used;
        s.used += len;
        Some((gpu + at, (data + at as usize) as *mut u8))
    }

    /// The bytes of scratch chunks the list holds now (after a reset:
    /// none while lists share their chunks).
    pub(crate) fn scratch_kept(&self) -> u64 {
        let scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        scratch.chunks.iter().map(|c| c.1).sum()
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let card = self.card_budget.get_or_init(|| self.make_card_budget()).as_ref();
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped words, written before it runs.
            // The words after the budget hold what the clamp noted about the
            // list's last run (mesh dispatches: the largest group count
            // asked for, the groups asked in all; compute records still
            // clamped; with the card's copy, as the list's end copied them
            // here): taken, then zeroed.
            deferred.copies.push(Step::Host(Box::new(move || unsafe {
                let words = at as *mut u32;
                let (largest, all) = (words.add(1).read_volatile(), words.add(2).read_volatile());
                MESH_GROUPS[0].fetch_max(u64::from(largest), Ordering::Relaxed);
                MESH_GROUPS[1].fetch_add(u64::from(all), Ordering::Relaxed);
                COMPUTE_CLAMPED.fetch_add(u64::from(words.add(3).read_volatile()), Ordering::Relaxed);
                COMPUTE_SPLIT.fetch_add(u64::from(words.add(4).read_volatile()), Ordering::Relaxed);
                for k in 0..5 {
                    words.add(k).write_volatile(0);
                }
            })));
            drop(deferred);
            if let Some(card) = card {
                self.zero_card_budget(card);
            }
        }
        // SAFETY: a live buffer.
        Some(unsafe { card.unwrap_or(buffer).GetGPUVirtualAddress() })
    }

    /// The budget's words in the card's own memory ([`card_budget`]), if
    /// the list can zero them itself (`WriteBufferImmediate`).
    fn make_card_budget(&self) -> Option<ID3D12Resource> {
        if !card_budget() {
            return None;
        }
        self.list.cast::<ID3D12GraphicsCommandList2>().ok()?;
        let buffer = host()?.uav_buffer(256)?;
        name(&buffer, || "indirect budget".to_owned());
        Some(buffer)
    }

    /// Records, before the list's first clamp, the zeroing of the card's
    /// copy of the budget: common, written as a copy, then left for
    /// unordered access until the list closes ([`HostList::card_budget_out`]).
    fn zero_card_budget(&self, card: &ID3D12Resource) {
        let Ok(list2) = self.list.cast::<ID3D12GraphicsCommandList2>() else {
            return;
        };
        // SAFETY: a live buffer.
        let at = unsafe { card.GetGPUVirtualAddress() };
        let zeros = [0u64, 4, 8, 12, 16].map(|offset| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
            Dest: at + offset,
            Value: 0,
        });
        // SAFETY: an open list; the buffer is the list's own and in the
        // common state at the start of every recording.
        unsafe {
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            list2.WriteBufferImmediate(zeros.len() as u32, zeros.as_ptr(), None);
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
    }

    /// Before the list closes, if this recording charged the card's copy
    /// of the budget: its notes on mesh records go to the mapped words
    /// (read at the next submission), and it goes back to common.
    fn card_budget_out(&self) {
        if !self.deferred().budget_zeroed {
            return;
        }
        let (Some(Some(card)), Some(Some((mapped, _)))) =
            (self.card_budget.get(), self.budget.get())
        else {
            return;
        };
        // SAFETY: an open list; both buffers are the list's own. The
        // mapped one is promoted from common to the copy destination.
        unsafe {
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            self.list.CopyBufferRegion(mapped, 4, card, 4, 16);
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
    }

    /// A device-local unordered-access buffer of at least `size` bytes for
    /// clamped indirect arguments, reused across executions and resets of the
    /// list. Smaller buffers replaced mid-list stay alive until
    /// [`HostList::reset`].
    pub(crate) fn indirect_scratch(&self, size: u64) -> Option<ID3D12Resource> {
        let needed = size.max(64 << 10);
        let mut slot = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, cap)) = &*slot
            && *cap >= needed
        {
            return Some(buffer.clone());
        }
        let prev = slot.as_ref().map_or(0, |(_, cap)| *cap);
        let alloc = prev
            .saturating_mul(2)
            .max(needed)
            .next_multiple_of(64 << 10);
        let buffer = host()?.uav_buffer(alloc)?;
        name(&buffer, || format!("indirect arguments {alloc:#x}"));
        let old = slot.replace((buffer.clone(), alloc));
        drop(slot);
        if let Some((old_buf, _)) = old {
            // It ends the list common, as every scratch buffer does.
            self.indirect_common(&old_buf);
            self.keep(&old_buf);
        }
        Some(buffer)
    }

    /// Moves `scratch` back to the common state if a clamp left it readable
    /// as indirect arguments ([`HostList::indirect_readable`]): where the
    /// list's first clamp expects it. Done before the list closes and before
    /// a bigger scratch buffer replaces it: one barrier per list (where
    /// there was one per indirect execution), and the list ends as it
    /// begins however it is submitted.
    fn indirect_common(&self, scratch: &ID3D12Resource) {
        {
            let mut deferred = self.deferred();
            if deferred.indirect_readable != scratch.as_raw() as usize {
                return;
            }
            deferred.indirect_readable = 0;
        }
        let barrier = transition(
            scratch,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        );
        // SAFETY: an open list and a live buffer this recording left in
        // `INDIRECT_ARGUMENT`.
        unsafe { self.list.ResourceBarrier(&[barrier]) };
    }

    /// The state the list's indirect scratch buffer `scratch` is in at this
    /// point of the recording: indirect arguments if a clamp in this
    /// recording left it so ([`HostList::indirect_readable`]), otherwise
    /// common: the state a buffer is made in, and the one a recording
    /// leaves it in when it closes or replaces it ([`HostList::indirect_common`]).
    pub(crate) fn indirect_state(&self, scratch: &ID3D12Resource) -> D3D12_RESOURCE_STATES {
        if self.deferred().indirect_readable == scratch.as_raw() as usize {
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
        } else {
            D3D12_RESOURCE_STATE_COMMON
        }
    }

    /// A clamp recorded in this list left `scratch` readable as indirect
    /// arguments. It stays so for every execution that reads it, until the
    /// next clamp moves it back to unordered access.
    pub(crate) fn indirect_readable(&self, scratch: &ID3D12Resource) {
        self.deferred().indirect_readable = scratch.as_raw() as usize;
    }

    /// The list's buffers for `crate::indirect::window_check`.
    pub(crate) fn window_check(&self) -> Option<&crate::indirect::WindowCheck> {
        self.window_check.get_or_init(crate::indirect::WindowCheck::new).as_ref()
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// (fxceless 19a45d1) Whether `source` is not the last card-written texture copied
    /// into the aliased host texture `view` in this list (and notes it as that now).
    pub(crate) fn alias_source_changed(&self, view: &ID3D12Resource, source: &ID3D12Resource) -> bool {
        let view = view.as_raw() as usize;
        let source = source.as_raw() as usize;
        self.deferred().texture_aliases.insert(view, source) != Some(source)
    }

    /// (P6, `VICTORY_HEAP_PASS_ASIDE`) `work` finds, on a thread of ours,
    /// refreshes that belong at this place among the list's steps; they are
    /// there when the list is closed. `host`: the host heap it is for.
    pub(crate) fn refreshes_coming(&self, host: usize, work: impl FnOnce() -> Vec<aside::Refresh> + Send + 'static) {
        self.take_coming();
        let (at, set) = {
            let deferred = self.deferred();
            (deferred.copies.len(), deferred.refreshed.clone())
        };
        let coming = aside::begin(host, work);
        *self.coming.lock().unwrap_or_else(|e| e.into_inner()) = Some((at, set, coming));
    }

    /// (P6) The refreshes that were still to come are put among the steps.
    fn take_coming(&self) {
        let Some((at, set, coming)) = self.coming.lock().unwrap_or_else(|e| e.into_inner()).take() else {
            return;
        };
        let found = coming.take();
        let mut deferred = self.deferred();
        let Deferred { copies, refreshed, .. } = &mut *deferred;
        aside::splice(copies, refreshed, at, set, found);
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        self.changes.fetch_add(1, Ordering::Relaxed);
        self.deferred().copies.push(Step::Run(Box::new(work)));
        self.deferred().refreshed.clear();
        self.refreshes.fetch_add(1, Ordering::Relaxed);
    }

    /// (P3) This recording's number ([`HostList::serial`]).
    pub(crate) fn serial(&self) -> u64 {
        self.serial.load(Ordering::Relaxed)
    }

    /// (P3) The count of times the list's refreshed ranges were forgotten
    /// ([`HostList::refreshes`]).
    pub(crate) fn refresh_epoch(&self) -> u64 {
        self.refreshes.load(Ordering::Relaxed)
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.changes.fetch_add(1, Ordering::Relaxed);
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset.
    pub(crate) fn defer_texture(&self, staging: std::sync::Arc<crate::texture::Staging>) {
        self.changes.fetch_add(1, Ordering::Relaxed);
        self.deferred().copies.push(Step::Texture(staging));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    fn run_deferred(&self, batch: &mut Batch) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        // (P6) Closed lists have none; a list submitted unclosed would.
        self.take_coming();
        for step in &self.deferred().copies {
            // (P6) Fills gathered so far are done before a step that may
            // write title memory or our memory.
            if !matches!(step, Step::Fill(..) | Step::Refresh(..) | Step::Texture(..)) {
                batch.run_fills();
            }
            // (neptune-ve 25m, `VICTORY_STEP_CLOCK_LEAN`) A refresh only goes
            // into the batch: no clock read for it (tens of thousands of
            // steps a frame; 14 of 400 samples of the author's submit thread).
            if step_clock_lean()
                && let Step::Refresh(shadow, address, len) = step
            {
                #[cfg(feature = "diagnostics")]
                {
                    add(C::Refreshes, 1);
                    add(C::RefreshBytes, *len as u64);
                }
                batch.add(shadow, *address, *len);
                STEP_CLOCKS_SAVED.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let started = std::time::Instant::now();
            let copied = match step {
                &Step::Copy(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.title_wrote(dst, len);
                    let copy_began = std::time::Instant::now();
                    #[cfg(feature = "diagnostics")]
                    fill_check::copy_runs(dst, src, len);
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    // With the submission's one asking: the journal learns it.
                    crate::watch::wrote(dst, len);
                    prep(5, copy_began);
                    len
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                        fill_check::runs(src, len);
                    }
                    if prepare_wide() {
                        batch.fills.push((dst, src, len));
                        continue;
                    }
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    prep(12, started);
                    len
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work) | Step::Host(work) => {
                    if matches!(step, Step::Run(_)) {
                        batch.title_written();
                    }
                    work();
                    // It wrote title memory, where is not known here: the
                    // submission's one asking ends, later refreshes ask anew.
                    if matches!(step, Step::Run(_)) {
                        crate::watch::ask_once_end();
                    }
                    prep(6, started);
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len) => {
                    // The shaders start from the title's data: this copy's
                    // pending refresh first. The other copies' wait for
                    // the next flush of all (flushing all of them at every
                    // range a dispatch writes made thousands of write-watch
                    // askings a frame).
                    batch.flush_one(shadow);
                    #[cfg(feature = "diagnostics")]
                    let snapshot = std::time::Instant::now();
                    shadow.writing(*address, *len);
                    prep(7, started);
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::GpuWrites, 1);
                        add(C::SnapshotUs, snapshot.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::Texture(staging) => {
                    if batch.texture_due(staging) {
                        if wide_textures() || prepare_wide() {
                            batch.textures_due.push(staging.clone());
                        } else {
                            staging.refresh();
                            prep(14, started);
                        }
                    }
                    continue;
                }
            };
            preparing(0, started.elapsed().as_micros() as u64);
            preparing(1, copied as u64);
        }
        batch.run_fills();
    }
}

/// Whether a submission's refreshes of host copies run on several threads,
/// a copy to a thread (`VICTORY_WIDE_REFRESH`; on by default, `=0` turns
/// it off).
///
/// A submission brings some sixty copies up to date, one after the other
/// on the submitting thread: for each, the pages the title wrote are asked
/// for and read through the kernel into the copy. That is the frame line's
/// "copies", 45-50 ms a frame, and the copies have nothing to do with one
/// another: each refresh writes its own copy and its own notes, and reads
/// title memory. With the switch the copies of a flush (four or more) are
/// spread over the threads kept for the snapshots (`crate::shadow::wide`).
/// The write-watch journal and the guards are behind their own locks, as
/// they are for the title's threads.
///
/// On by default since an RTX 4070 12 GB / Ryzen 5 7500F run: "copies" went from
/// 54-56 to 45-47 ms a frame and the frame from 144-154 to 131-139 ms.
fn wide_refresh() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_REFRESH").is_none_or(|v| v != "0"))
}

/// Whether a range of a host copy brought up to date in a submission is
/// left alone for the rest of that submission, until a step of ours writes
/// title memory there (`VICTORY_REFRESH_ONCE=1`; off by default). After
/// upstream VIctory 6d021fe ("a copy's range brought up to date is not
/// copied again in the submission until a step writes title memory").
///
/// A list of dispatches records "refresh this range, shaders write it" for
/// every dispatch, and each of those refreshes was done on its own
/// (`Batch::flush_one`): a walk of the guards or a failed kernel asking
/// for a copy of view memory, some 1,800 to 2,700 a frame and 23 to 29 ms
/// (runs 222 and 225, "watches"), nearly all finding nothing new.
///
/// What changes: what the title's other threads write into a range while
/// its submission is being prepared is taken by the next submission, not
/// by a later step of this one, as `crate::watch::once` already has it for
/// write-watched memory. What this submission's lists read was written
/// before they were submitted. A title buffer copy of ours takes its
/// bytes out of what counts as up to date ([`Batch::title_wrote`]); any
/// other step that writes title memory takes everything out.
fn refresh_once() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_ONCE").is_some_and(|v| v == "1"))
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them: done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
#[derive(Default)]
struct Batch {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, RefreshRanges)>,
    /// Where each copy (by address) is in `refreshes`: the list that makes
    /// the heap's buffers ready adds some 4,100 copies, and searching them
    /// for every refresh of a submission was tens of millions of
    /// comparisons a frame.
    index: std::collections::HashMap<usize, usize>,
    /// With [`refresh_once`], by copy (its address; the copy is held, so
    /// that the address names no other): the range of title addresses
    /// brought up to date in this submission that no step of ours has
    /// written since.
    current: std::collections::HashMap<usize, (std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Staging buffers (by address) refreshed since the last title write.
    textures: std::collections::HashSet<usize>,
    /// With [`wide_textures`]: textures due, refreshed together at the next
    /// flush (before a step that writes title memory, and before the lists
    /// run), several at a time.
    textures_due: Vec<std::sync::Arc<crate::texture::Staging>>,
    /// (P6, [`prepare_wide`]) CPU fills gathered (destination, source,
    /// bytes), done at [`Batch::run_fills`].
    fills: Vec<(usize, usize, usize)>,
}

/// Whether a copy's refreshes in a batch keep their disjoint ranges
/// ([`RefreshRanges`]) instead of the one range covering them all, which
/// also copied every page the title wrote between them that nothing asked
/// for (on unless `VICTORY_DISJOINT_REFRESH=0`; Trexies 3cd984d, after
/// Marvisak/VIctory#1818).
fn disjoint_refresh() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DISJOINT_REFRESH").is_none_or(|v| v != "0"))
}

/// The ranges (address, end) of one copy a batch is to refresh: one range
/// inline, more once a range that does not touch it comes
/// ([`disjoint_refresh`]); overlapping and touching ones are joined before
/// the refresh ([`RefreshRanges::merge`]).
enum RefreshRanges {
    One((usize, usize)),
    Many(Vec<(usize, usize)>),
}

impl RefreshRanges {
    fn add(&mut self, (from, to): (usize, usize)) {
        match self {
            Self::One((start, end)) if !disjoint_refresh() || (from <= *end && *start <= to) => {
                *start = (*start).min(from);
                *end = (*end).max(to);
            }
            Self::One(previous) => *self = Self::Many(vec![*previous, (from, to)]),
            Self::Many(ranges) => ranges.push((from, to)),
        }
    }

    fn merge(&mut self) {
        if let Self::Many(ranges) = self {
            ranges.sort_unstable();
            ranges.dedup_by(|next, previous| {
                if next.0 > previous.1 {
                    return false;
                }
                previous.1 = previous.1.max(next.1);
                true
            });
        }
    }

    fn as_slice(&self) -> &[(usize, usize)] {
        match self {
            Self::One(range) => std::slice::from_ref(range),
            Self::Many(ranges) => ranges,
        }
    }
}

impl Batch {
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let end = address + len;
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        // Brought up to date in this submission already ([`refresh_once`];
        // nothing is in `current` without the switch).
        if let Some(&(_, from, to)) = self.current.get(&key)
            && from <= address
            && end <= to
        {
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::add(crate::copystat::C::Skipped, 1);
                crate::copystat::add(crate::copystat::C::SkippedBytes, len as u64);
            }
            return;
        }
        match self.index.get(&key) {
            Some(&at) => self.refreshes[at].1.add((address, end)),
            None => {
                self.index.insert(key, self.refreshes.len());
                self.refreshes.push((shadow.clone(), RefreshRanges::One((address, end))));
            }
        }
    }

    /// Before shaders are given a range of `shadow`'s copy: its pending
    /// refresh is done, the others stay pending.
    /// (P6, [`prepare_wide`]) The fills gathered: several at a time on the
    /// wide workers when there are enough of them and their destinations
    /// are disjoint, else in order here.
    fn run_fills(&mut self) {
        if self.fills.is_empty() {
            return;
        }
        let fills = std::mem::take(&mut self.fills);
        let started = std::time::Instant::now();
        let bytes: usize = fills.iter().map(|f| f.2).sum();
        let disjoint = {
            let mut by_dst: Vec<(usize, usize)> = fills.iter().map(|f| (f.0, f.0 + f.2)).collect();
            by_dst.sort_unstable();
            by_dst.windows(2).all(|w| w[0].1 <= w[1].0)
        };
        // Below this the handing out costs more than the copies.
        const WIDE_FROM: (usize, usize) = (32, 256 << 10);
        if !disjoint || fills.len() < WIDE_FROM.0 || bytes < WIDE_FROM.1 {
            for &(dst, src, len) in &fills {
                copy_mapped(dst as *mut u8, src as *const u8, len, false);
            }
            prep(12, started);
        } else {
            let parts = (crate::shadow::wide_threads() + 1).min(fills.len());
            let per = bytes.div_ceil(parts);
            let mut jobs: Vec<Box<dyn FnOnce() + Send>> = Vec::with_capacity(parts);
            let mut part: Vec<(usize, usize, usize)> = Vec::new();
            let mut held = 0;
            let n = fills.len() as u64;
            for f in fills {
                held += f.2;
                part.push(f);
                if held >= per {
                    let these = std::mem::take(&mut part);
                    jobs.push(Box::new(move || {
                        for (dst, src, len) in these {
                            copy_mapped(dst as *mut u8, src as *const u8, len, false);
                        }
                    }));
                    held = 0;
                }
            }
            if !part.is_empty() {
                jobs.push(Box::new(move || {
                    for (dst, src, len) in part {
                        copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    }
                }));
            }
            crate::shadow::wide_run(jobs);
            prep(1, started);
            PREP[2].fetch_add(n, Ordering::Relaxed);
        }
        preparing(0, started.elapsed().as_micros() as u64);
        preparing(1, bytes as u64);
    }

    fn flush_one(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>) {
        let Some(at) = self.index.remove(&(std::sync::Arc::as_ptr(shadow) as usize)) else {
            return;
        };
        let (shadow, mut ranges) = self.refreshes.swap_remove(at);
        // The last entry took its place.
        if let Some(moved) = self.refreshes.get(at) {
            self.index.insert(std::sync::Arc::as_ptr(&moved.0) as usize, at);
        }
        ranges.merge();
        for &(from, to) in ranges.as_slice() {
            Self::update(&shadow, from, to);
            Self::brought(&mut self.current, &shadow, from, to);
        }
    }

    /// With [`refresh_once`]: `[from, to)` of `shadow`'s copy was brought
    /// up to date. Kept: its union with what was up to date if they touch,
    /// else the larger of the two.
    fn brought(
        current: &mut std::collections::HashMap<usize, (std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        from: usize,
        to: usize,
    ) {
        if !refresh_once() || from >= to {
            return;
        }
        let entry = current
            .entry(std::sync::Arc::as_ptr(shadow) as usize)
            .or_insert_with(|| (shadow.clone(), from, to));
        if from <= entry.2 && entry.1 <= to {
            (entry.1, entry.2) = (entry.1.min(from), entry.2.max(to));
        } else if to - from > entry.2 - entry.1 {
            (entry.1, entry.2) = (from, to);
        }
    }

    /// Brings `[from, to)` of a copy up to date, timed.
    fn update(shadow: &crate::shadow::Shadow, from: usize, to: usize) {
        let started = std::time::Instant::now();
        let copied = shadow.update(from, to - from);
        let us = started.elapsed().as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::range(from, to - from, us, 0);
        preparing(0, us);
        preparing(1, copied as u64);
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.textures.insert(std::sync::Arc::as_ptr(staging) as *const () as usize)
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.textures.clear();
        // Where it writes is not known: no copy counts as up to date.
        self.current.clear();
    }

    /// [`Batch::title_written`] for a write of ours whose place is known
    /// (a title buffer copy into `[at, at + len)`): of a copy up to date
    /// in this submission that holds some of those bytes, the larger part
    /// beside them stays so.
    fn title_wrote(&mut self, at: usize, len: usize) {
        self.flush();
        self.textures.clear();
        let end = at + len;
        self.current.retain(|_, entry| {
            let (from, to) = (entry.1, entry.2);
            if end <= from || to <= at {
                return true;
            }
            let (left, right) = (at.saturating_sub(from), to.saturating_sub(end));
            if left >= right {
                entry.2 = from + left;
            } else {
                entry.1 = end;
            }
            entry.1 < entry.2
        });
    }

    fn flush(&mut self) {
        self.index.clear();
        // One write-watch asking for the copies that lie close together,
        // not one per copy.
        for (_, ranges) in &mut self.refreshes {
            ranges.merge();
        }
        let ahead = self.refreshes.len() > 1 || self.refreshes.first().is_some_and(|r| r.1.as_slice().len() > 1);
        if ahead {
            let started = std::time::Instant::now();
            let ranges: Vec<(usize, usize)> = self
                .refreshes
                .iter()
                .filter(|r| !r.0.unwatched())
                .flat_map(|r| r.1.as_slice().iter().copied())
                .collect();
            crate::watch::ask_ahead(&ranges);
            preparing(0, started.elapsed().as_micros() as u64);
        }
        let refreshes_began = std::time::Instant::now();
        let wide_items = self.refreshes.len() as u64;
        if wide_refresh() && self.refreshes.len() >= if prepare_wide() { 2 } else { 4 } {
            // Each copy on its own, several at a time: a copy's refresh
            // writes that copy and reads title memory, nothing of another's.
            // The time counted is the time it took, not the threads' sum.
            let started = std::time::Instant::now();
            let before = PREPARING[0].load(Ordering::Relaxed);
            // Noted now: nothing reads the notes before the jobs are done.
            for (shadow, ranges) in &self.refreshes {
                for &(from, to) in ranges.as_slice() {
                    Self::brought(&mut self.current, shadow, from, to);
                }
            }
            let jobs: Vec<Box<dyn FnOnce() + Send>> = self
                .refreshes
                .drain(..)
                .map(|(shadow, ranges)| {
                    Box::new(move || {
                        for &(from, to) in ranges.as_slice() {
                            Self::update(&shadow, from, to);
                        }
                    }) as Box<dyn FnOnce() + Send>
                })
                .collect();
            crate::shadow::wide_run(jobs);
            let summed = PREPARING[0].load(Ordering::Relaxed).saturating_sub(before);
            let over = summed.saturating_sub(started.elapsed().as_micros() as u64);
            PREPARING[0].fetch_sub(over, Ordering::Relaxed);
            crate::framelog::sub(crate::framelog::T::CopyUs, over);
            prep(1, refreshes_began);
            PREP[2].fetch_add(wide_items, Ordering::Relaxed);
        } else {
            for (shadow, ranges) in self.refreshes.drain(..) {
                for &(from, to) in ranges.as_slice() {
                    Self::update(&shadow, from, to);
                    Self::brought(&mut self.current, &shadow, from, to);
                }
            }
            prep(13, refreshes_began);
        }
        if ahead {
            crate::watch::ask_ahead_done();
        }
        self.refresh_textures();
    }

    /// The textures queued by [`wide_textures`], several at a time. Each
    /// refresh reads its texture's title memory and writes its own staging
    /// buffer and notes (under its own lock); a texture is queued once per
    /// batch ([`Batch::texture_due`]). The time counted is the time it
    /// took, not the threads' sum.
    fn refresh_textures(&mut self) {
        match self.textures_due.len() {
            0 => {}
            1 => {
                let began = std::time::Instant::now();
                if let Some(staging) = self.textures_due.pop() {
                    staging.refresh();
                }
                prep(14, began);
            }
            _ => {
                let started = std::time::Instant::now();
                let before = PREPARING[2].load(Ordering::Relaxed);
                let jobs: Vec<Box<dyn FnOnce() + Send>> = self
                    .textures_due
                    .drain(..)
                    .map(|staging| Box::new(move || staging.refresh()) as Box<dyn FnOnce() + Send>)
                    .collect();
                let n = jobs.len() as u64;
                crate::shadow::wide_run(jobs);
                let summed = PREPARING[2].load(Ordering::Relaxed).saturating_sub(before);
                PREPARING[2].fetch_sub(summed.saturating_sub(started.elapsed().as_micros() as u64), Ordering::Relaxed);
                prep(1, started);
                PREP[2].fetch_add(n, Ordering::Relaxed);
            }
        }
    }
}

/// Whether a submission's texture checks run several at a time
/// (`VICTORY_WIDE_TEXTURES=1`; off by default).
///
/// While the world streams in, a submission checks hundreds of textures,
/// one after another on the submitting thread: each is read, hashed whole
/// and, when it changed, detiled into its staging buffer. On an RTX 3090 Ti
/// machine that was the frame line's "textures", 100-1,400 ms and 250-930
/// MB a frame for the first two minutes of the world (the opening
/// cutscene at 0.3-2 fps, some of its frames white), against about 5 ms
/// later. The textures have nothing to do with one another, so with the
/// switch the checks due in a batch are queued and run at its next flush
/// (before a step that writes title memory, and before the lists run: the
/// order against those steps is as before) on the threads kept for the
/// snapshots (`crate::shadow::wide`).
fn wide_textures() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_TEXTURES").is_some_and(|v| v == "1"))
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped is left alone.
///
/// The source is read through the kernel (`ReadProcessMemory`), which
/// reports an unmapped page instead of faulting. Asking `VirtualQuery` first
/// cost about 2 ms per copy in the title's big pools (it walks the page
/// tables to the region's end), which made copies run at 100-200 MB/s.
/// (Run-45 follow-up) Copies read again after a mapping change under way.
static REREAD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    // A title destination that is not all mapped is left alone. Asking
    // Windows first (`VirtualQuery`) walks the page tables to the end of
    // the destination's region: in the title's big blocks that is
    // milliseconds a copy, and it was a seventh of the submitting thread's
    // time in gameplay (run 60: 71 of 500 looks). The kernel's copy below
    // says the same when it fails, so the question is only asked then.
    // `VICTORY_COPY_ASK=1` asks first, as before.
    static ASK: OnceLock<bool> = OnceLock::new();
    let ask_first = title_dst
        && *ASK.get_or_init(|| std::env::var_os("VICTORY_COPY_ASK").is_some_and(|v| v == "1"));
    if ask_first && !crate::texture::committed(dst, len) {
        return;
    }
    if title_dst {
        // Copy-back held over the destination goes in first (F1).
        crate::readback::settle(dst as usize, len);
        crate::readback::consumers::touched(dst as usize, len, crate::readback::consumers::Kind::LayerWrite);
        // The kernel writes the destination: it must not be guarded.
        crate::watch::writing(dst as usize, len);
    }
    if !title_dst {
        crate::readback::consumers::reading(src as usize, len);
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    // Into host memory: in place first, if asked for (`crate::plaincopy`);
    // through the kernel if that is off or met a page that was not there.
    let whole = (!title_dst && crate::plaincopy::copy(dst, src, len)) || read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        if !title_dst {
            crate::readback::overlay(src as usize, dst, len);
        }
        return;
    }
    // (Run-45 follow-up) A change of mapping under way over the source (a
    // run's view cut and mapped again around an unmapped range takes its
    // still-mapped neighbours away for a moment): waited out and read again,
    // not read as zeros where pages were only gone for that moment.
    if crate::watch::title_memory_change_waited(src as usize, len, std::time::Duration::from_millis(50))
        || (title_dst && crate::watch::title_memory_change_waited(dst as usize, len, std::time::Duration::from_millis(50)))
    {
        let n = REREAD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n.is_power_of_two() {
            tracing::info!(
                "a copy of title memory met a mapping change under way: waited and read again ({} so far)",
                n + 1
            );
        }
        if read(src, dst, len) {
            if !title_dst {
                crate::readback::overlay(src as usize, dst, len);
            }
            return;
        }
    }
    // The copy failed: a source page is not there, or the destination is
    // not all mapped. Then the destination stays as it is (the kernel may
    // have written its first pages).
    if title_dst && !ask_first && !crate::texture::committed(dst, len) {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    #[cfg(feature = "diagnostics")]
    let sparse = std::time::Instant::now();
    if !title_dst && len >= 64 << 10 {
        read_sparse(src, dst, len);
    } else {
        read_pieces(src, dst, len);
    }
    if !title_dst {
        crate::readback::overlay(src as usize, dst, len);
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::SparseUs, sparse.elapsed().as_micros() as u64);
}

/// After a failed bulk read into host memory, skip contiguous holes. Never
/// cache the query across copies: the title may remap the same address.
fn read_sparse(src: *const u8, dst: *mut u8, len: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualQuery,
    };
    let mut offset = 0;
    while offset < len {
        let at = src.wrapping_add(offset);
        let to = dst.wrapping_add(offset);
        let remaining = len - offset;
        // SAFETY: the query inspects an address and fills a local structure.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let size = unsafe { VirtualQuery(at.cast(), &mut info, size_of_val(&info)) };
        let n = (info.BaseAddress as usize).saturating_add(info.RegionSize)
            .saturating_sub(at as usize).min(remaining);
        if size != size_of_val(&info) || n == 0 {
            read_pieces(at, to, remaining);
            break;
        }
        if info.State != MEM_COMMIT || info.Protect == PAGE_NOACCESS {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
            // SAFETY: this path is only used with a writable host destination.
            unsafe { to.write_bytes(0, n) };
        } else if !read(at, to, n) {
            // A mapping may change between query and read.
            read_pieces(at, to, n);
        }
        offset += n;
    }
}

fn read_pieces(src: *const u8, dst: *mut u8, len: usize) {
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    // (Copy-back consumers) Metered; what is held on the card there first.
    crate::readback::consumers::reading(src as usize, dst.len());
    // In place first, if asked for (`crate::plaincopy`): the destination is
    // memory of ours, as in `copy_mapped`.
    let whole = crate::plaincopy::copy(dst.as_mut_ptr(), src, dst.len())
        || read(src, dst.as_mut_ptr(), dst.len())
        // (Run-45 follow-up) Not a page gone only while a mapping changed.
        || (crate::watch::title_memory_change_waited(src as usize, dst.len(), std::time::Duration::from_millis(50))
            && read(src, dst.as_mut_ptr(), dst.len()));
    if whole {
        crate::readback::overlay(src as usize, dst.as_mut_ptr(), dst.len());
    }
    whole
}

/// Copies `len` bytes from `src` to `dst` unless a source page is not
/// readable.
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // (F3, `VICTORY_LAZY_GPU_VIEW`) A read of the GPU view, for its meter.
    crate::gpuview::note_read(src as usize, len);
    let mut done = 0;
    // SAFETY: the destination is mapped writable memory of `len` bytes; the
    // kernel checks the source.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    /// The kernel's write into a process's memory, `(process, address,
    /// bytes, length, written out)`: an NTSTATUS, negative when a page of
    /// the range is not there or not writable. It changes no protection.
    fn NtWriteVirtualMemory(
        process: *mut std::ffi::c_void,
        base: *mut std::ffi::c_void,
        buffer: *const std::ffi::c_void,
        size: usize,
        written: *mut usize,
    ) -> i32;
}

/// Whether title memory is written with the kernel's own call
/// (`NtWriteVirtualMemory`) and not `WriteProcessMemory`, which first asks
/// what the pages are (as `VirtualQuery` does: about 2 ms a call in the
/// title's big pools, see [`copy_mapped`]): putting back what shaders wrote
/// cost 1.2-1.9 s a frame for 430-800 pieces, 2 ms each whatever their size
/// (run 24). On by default; `VICTORY_DIRECT_WRITES=0` writes as before.
fn direct_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DIRECT_WRITES").is_none_or(|v| v != "0"))
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let direct = direct_writes();
    let write = |at: usize, n: usize| {
        let from = src.wrapping_add(at - dst as usize);
        let mut done = 0;
        if direct {
            // SAFETY: as below; the kernel checks the destination and
            // writes nothing it cannot (the caller opened its guards:
            // `crate::watch::writing`).
            return unsafe {
                NtWriteVirtualMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) >= 0
            };
        }
        // SAFETY: the source is our mapped host copy; the kernel checks the
        // destination.
        unsafe { WriteProcessMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) != 0 }
    };
    // In place first, if asked for (`crate::plaincopy::write`,
    // `VICTORY_WRITE_PLAIN`); a write that met a page that is not there or
    // not writable is done whole below, through the kernel, as before.
    if len == 0 || crate::plaincopy::write(dst, src, len) || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

impl Host {
    /// The queue everything of ours runs on, for a call that must go to it
    /// in order with the lists (`crate::tiles`: a texture's levels are
    /// given memory ahead of the list that uploads them).
    pub(crate) fn queue(&self) -> &ID3D12CommandQueue {
        &self.queue
    }

    /// A buffer in the upload heap, mapped for writing.
    /// A buffer of `size` bytes the card writes and we read (a read back),
    /// in the state copies into it need.
    pub(crate) fn read_back_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        crate::census::made(0, size);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a read-back buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no read-back buffer");
                None
            }
        }
    }

    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        crate::census::made(1, size);
        MADE[0].fetch_add(1, Ordering::Relaxed);
        MADE[1].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    /// With presents sent ahead ([`late_free_present`]): a list for each
    /// present in the air, and the submission it last flew as (it is used
    /// again once that one has finished).
    ahead: Vec<(HostList, u64)>,
    hwnd: HWND,
    meter: Meter,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        #[cfg(feature = "diagnostics")]
        crate::copystat::presented();
        // (Hang watch) Progress, and the watch from the first present on.
        PRESENTS_SEEN.fetch_add(1, Ordering::Relaxed);
        crate::hangwatch::start();
        self.frames += 1;
        #[cfg(feature = "diagnostics")]
        {
            self.copy_stats_frames += 1;
        }
        // This frame's draws, for the scripted input's waits on the screen.
        {
            static PREVIOUS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let drawn = crate::draw::drawn();
            let before = PREVIOUS.swap(drawn, Ordering::Relaxed);
            let n = drawn.saturating_sub(before).min(u64::from(u32::MAX)) as u32;
            crate::frame::PRESENTED_DRAWS.store(n, Ordering::Relaxed);
        }
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        let mesh_largest = MESH_GROUPS[0].swap(0, Ordering::Relaxed);
        let mesh_groups = MESH_GROUPS[1].swap(0, Ordering::Relaxed) as f64 / frames;
        tracing::info!(
            "indirect mesh dispatches: the largest group count asked for {mesh_largest} (run: at most 65535), {mesh_groups:.0} groups asked per frame"
        );
        // (Run 200 counted indirect compute records asking for 77,000 to
        // 98,000 groups in x, 0.2 to 0.8 a frame, cut to the 65,535 one
        // dispatch runs; the counter went when the clamp's budget moved to
        // the card's memory, and is back as the clamp's fourth word.)
        {
            let [split, parts, direct, executed] =
                [0, 1, 2, 3].map(|i| crate::draw::DISPATCH_PARTS[i].swap(0, Ordering::Relaxed) as f64 / frames);
            let [clamped, indirect_split] = [&COMPUTE_CLAMPED, &COMPUTE_SPLIT].map(|c| c.swap(0, Ordering::Relaxed) as f64 / frames);
            tracing::info!(
                "compute dispatches of more than 65535 groups per frame (VICTORY_DISPATCH_PARTS={}): direct {split:.1} split into                  {parts:.1} parts, {direct:.1} run whole past the host's limit; indirect {indirect_split:.1} records split                  (of {executed:.1} executions in parts), {clamped:.1} records still clamped",
                if victory_shader::emit::dispatch_parts() { "on" } else { "0" }
            );
        }
        {
            let [texture_us, slots] = [0, 1].map(|i| crate::draw::HEAP_PASS_PARTS[i].swap(0, Ordering::Relaxed));
            tracing::info!(
                "the pass over the heap per frame (the frame line's \"heap buffers ... ms\" is both parts): {:.1} ms in its                  pass over textures picked by index; {:.0} buffer slots looked at by its pass over buffer views;                  (P6) {}: {:.1} passes aside, {:.1} ms on the aside thread, {:.2} ms waited for at list close and by draws;                  VICTORY_VIEW_THREADS={}, VICTORY_HEAP_PASS_MARKED={}",
                texture_us as f64 / 1e3 / frames,
                slots as f64 / frames,
                if aside::on() { "VICTORY_HEAP_PASS_ASIDE=1" } else { "VICTORY_HEAP_PASS_ASIDE=0" },
                aside::WAITED[1].swap(0, Ordering::Relaxed) as f64 / frames,
                crate::draw::HEAP_ASIDE_MICROS.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames,
                aside::WAITED[0].swap(0, Ordering::Relaxed) as f64 / 1e3 / frames,
                crate::views::threads(),
                u8::from(crate::heap_pass::marked())
            );
        }
        crate::shadow::exact_line(frames);
        crate::descriptor::sync_line(frames);
        prep_line(frames);
        {
            // (Neptune) `VICTORY_TABLES_BY_CONTENT`, `VICTORY_READY_TABLES`.
            let t = [0, 1, 2, 3, 4, 5, 6, 7, 8].map(|i| crate::draw::TABLES[i].swap(0, Ordering::Relaxed) as f64 / frames);
            tracing::info!(
                "descriptor tables per frame: {:.0} found ready, {:.0} found made and made ready again ({:.0} not noted ready, {:.0} in another host list, {:.0} after a change in the list, {:.0} for the tested depth), {:.0} made ({:.0} slots), {:.0} at a new address that took another's slots",
                t[0], t[1], t[5], t[6], t[7], t[8], t[2], t[3], t[4]
            );
        }
        {
            let [waits, micros, given_up] =
                [0, 1, 2].map(|i| crate::descriptor::host::DYNAMIC_WAITS[i].swap(0, Ordering::Relaxed) as f64 / frames);
            let early = crate::list::RELEASED_AT_LANDING.swap(0, Ordering::Relaxed) as f64 / frames;
            if waits + given_up > 0.0 || early > 0.0 {
                tracing::info!(
                    "descriptor tables that found no free slots, per frame: {waits:.1} waited ({:.2} ms in all), {given_up:.2} given up (their draws not drawn); \
                     landed lists that gave their dynamic slots back at landing {early:.1} (VICTORY_RELEASE_AT_LANDING={})",
                    micros / 1000.0,
                    u8::from(crate::list::release_at_landing())
                );
            }
        }
        late_line(frames);
        card_clock_line(frames, ms);
        crate::queue2::line(frames);
        crate::plaincopy::line(frames);
        crate::indirect::gather_line(frames);
        if let Some(host) = host() {
            host.drain_messages();
        }
        {
            let held = crate::pipeline::WALKS_HELD.swap(0, Ordering::Relaxed);
            if held > 0 {
                tracing::info!(
                    "dispatches of programs that walk ray-tracing structures held back per frame (VICTORY_BVH_HOLD=1; the                      builder has not built them): {:.1}",
                    held as f64 / frames
                );
            }
        }
        crate::shadow::back_line(frames);
        crate::readback::line();
        {
            let [passes, pieces, compares] = [0, 1, 2].map(|i| CARD_PASSES[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if passes + compares > 0.0 {
                tracing::info!(
                    "our passes over copies in the card's memory per frame: {passes:.0} hash passes over {pieces:.0} pieces after \
                     ranges were sent up (VICTORY_CARD_REHASH_RUNS=1 joins them: now {}), {compares:.0} compare passes",
                    if rehash_runs() { "on" } else { "off" }
                );
            }
        }
        {
            let [own, asking, still, all] =
                [0, 1, 2, 3].map(|i| crate::texture::REFRESH_WAITS[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
            if all >= 1.0 {
                tracing::info!(
                    "texture refreshes per frame, all threads: {all:.1} ms, of them {own:.1} ms waiting for a texture another thread \
                     is refreshing, {asking:.1} ms asking the write watch and the guards, {still:.1} ms waiting for title memory to \
                     hold still (the runtime remapping it); the rest is reading, hashing and detiling"
                );
            }
            let [through, left, around] = [0, 1, 2].map(|i| crate::texture::UNLOCKED[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if through + left > 0.0 {
                tracing::info!(
                    "texture refreshes that did not wait for title memory to hold still, per frame (VICTORY_TEXTURE_NO_WAIT=1): \
                     {through:.1} read through the kernel, {left:.1} left for the next refresh (a page was not there), \
                     {around:.1} of those read around the missing pages instead (nothing uploaded yet, VICTORY_TEXTURE_READ_AROUND); \
                     a refresh steps around {}",
                    if crate::watch::by_range() {
                        "changes of its own bytes only (VICTORY_MAPPING_RANGES)"
                    } else {
                        "any change of title memory's mapping (VICTORY_MAPPING_RANGES=0)"
                    }
                );
            }
        }
        if crate::texture::alias_once() {
            let [made, left] = [0, 1].map(|i| ALIAS_COUNTS[i].swap(0, Ordering::Relaxed) as f64 / frames);
            tracing::info!("alias copies in preludes per frame (VICTORY_ALIAS_ONCE=1): {made:.1} made, {left:.1} left out (source not written since)");
        }
        {
            let [copies, joined] = [0, 1].map(|i| COPIES_JOINED[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if copies > 0.0 {
                tracing::info!(
                    "title buffer copies recorded per frame: {copies:.0}, of them {joined:.0} continue the one before \
                     (VICTORY_JOIN_COPIES=1 makes those one copy with it: now {})",
                    if join_copies() { "on" } else { "off" }
                );
            }
        }
        crate::heap_marks::line(frames);
        // Passes over the heap's buffer views for shaders that pick them by index.
        let picked = crate::draw::HEAP_BUFFER_PASSES.swap(0, Ordering::Relaxed) as f64 / frames;
        let picked_ms =
            crate::draw::HEAP_BUFFER_MICROS.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames;
        // Copies between depth textures and their colour twins, each way.
        let [twin_to_depth, twin_to_colour] =
            [2, 3].map(|i| crate::resource::TWINS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        // Depth planes drawn from their twins.
        let planes = crate::resource::PLANES[0].swap(0, Ordering::Relaxed) as f64 / frames;
        // Copies in the card's memory (`crate::shadow::cards`).
        let card = crate::shadow::card_stats();
        #[cfg(feature = "diagnostics")]
        let slow = crate::watch::slow_asks();
        // Over the same frames as the counts above.
        let drops = crate::pipeline::take_drops();
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            tracing::info!(
                "placed texture views: {} writes of another view of the texture a slot showed (it kept the old view until the new was made); \
                 {} writes of a view of another texture (it kept showing the old texture until then, VICTORY_KEEP_OTHER_VIEWS=0: null at once)",
                crate::descriptor::host::KEPT_VIEWS.load(Ordering::Relaxed),
                crate::descriptor::host::KEPT_OTHER_VIEWS.load(Ordering::Relaxed)
            );
            crate::descriptor::host::stale_kept_line();
            crate::draw::catch_up_line();
            tracing::info!(
                "since the start: {} copies of a card-written texture into its aliased view after the list had uploaded that view \
                 from title memory (fxceless 19a45d1; skipped before); {} pipelines made again for the render target formats bound \
                 at their draw (VICTORY_TARGET_VARIANTS=0: off)",
                crate::texture::ALIAS_AFTER_UPLOAD.load(Ordering::Relaxed),
                crate::pipeline::variants_made()
            );
            tracing::info!(
                "since the start: {} table slots made with the unsigned twin of a 16-bit float buffer view, for shaders that move halves \
                 as integers (the face programs; VICTORY_HALF_VIEWS=0: off)",
                crate::draw::HALF_VIEWS_SHOWN.load(Ordering::Relaxed)
            );
            tracing::info!(
                "copies left out since the start because both sides were one host texture: {}",
                crate::list::SELF_COPIES.load(Ordering::Relaxed)
            );
            if crate::drvwatch::errors() > 0 {
                tracing::warn!(
                    "errors the graphics driver reported to the system since the start: {} (the paths under suspicion are off)",
                    crate::drvwatch::errors()
                );
            }
            crate::tiles::status();
            {
                let [queued, due, away, partly, oldest] =
                    [0, 1, 2, 3, 4].map(|i| crate::draw::HEAP_TEXTURES_QUEUE[i].load(Ordering::Relaxed));
                let [oldest_ms, longest_ms] = [0, 1].map(|i| crate::draw::HEAP_TEXTURES_AGE[i].load(Ordering::Relaxed));
                let promoted = crate::draw::HEAP_TEXTURES_AGE[2].swap(0, Ordering::Relaxed);
                let [same, parts, tails] =
                    [0, 1, 2].map(|i| crate::draw::HEAP_TEXTURES_PARTLY[i].load(Ordering::Relaxed));
                tracing::info!(
                    "textures picked by index, the queue: {queued} slots, {due} of them due (the oldest written {oldest} frames ago, \
                     {oldest_ms} ms; the longest due {longest_ms} ms; {promoted} served ahead for age since the last line, \
                     VICTORY_TEXTURE_MAX_AGE_MS={}, VICTORY_TEXTURE_BUDGET_MS_PER_S={}), \
                     {away} waiting for their memory, {partly} uploaded in part; since the start: {} slots found uploaded already by \
                     the draws that read them through tables, {} writes of tables' slots (due some frames later), {} slots taken out \
                     as read through a table, {parts} uploads of textures not all mapped ({tails} of them without their first page), \
                     {same} looks that found no more mapped; {} slots made stale by a newer view of other memory there, \
                     {} ranges of memory mapped, unmapped or written under texture views ({} gaps joined at capacity), {} slots made due by them, \
                     {} made slots found unmapped and made null",
                    crate::draw::texture_max_age_ms(),
                    crate::draw::texture_budget_ms_per_s(),
                    crate::draw::HEAP_TEXTURES_TABLED.load(Ordering::Relaxed),
                    crate::descriptor::host::TABLED_WRITES[0].load(Ordering::Relaxed),
                    crate::descriptor::host::TABLED_WRITES[1].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[0].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[1].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[2].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[3].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[4].load(Ordering::Relaxed)
                );
            }
            {
                let mode = crate::watch::page_writes();
                let [writes, bytes, guarded, plain] = [0, 1, 2, 3].map(|i| crate::watch::WRITTEN[i].load(Ordering::Relaxed));
                let [kept, gaps, slots, uploaded] =
                    [0, 1, 2, 3].map(|i| crate::descriptor::host::PAGE_WRITES[i].load(Ordering::Relaxed));
                tracing::info!(
                    "writes behind the guards, since the start: {writes} DirectStorage writes into physical pages at addresses the                      title had mapped ({} MB); {guarded} guarded 64 KiB blocks of host copies lay under them ({}); {plain} reads to                      plain addresses finished ({}); {kept} ranges under texture views kept for the texture pass ({gaps} gaps joined at capacity);                      {slots} slots of placed textures held their bytes, {uploaded} of them uploaded before the write ({})                      (VICTORY_UPLOAD_ON_WRITE, on unless 0: made due again at the next frame; VICTORY_PAGE_WRITES: unset or guards opens the guarded blocks, 0 counts only, 1, slots, all)",
                    bytes >> 20,
                    if mode & 1 != 0 { "opened, taken again" } else { "left as they were" },
                    if mode & 4 != 0 { "told as well" } else { "counted only" },
                    if mode & 2 != 0 { "made due again" } else { "left as they were" }
                );
            }
            #[cfg(feature = "diagnostics")]
            tracing::info!(
                "write-watch askings of a millisecond or more: {:.1} a frame, {:.1} ms a frame; the longest asking {:.1} ms",
                slow[0] as f64 / frames,
                slow[1] as f64 / 1e3 / frames,
                slow[2] as f64 / 1e3
            );
            {
                let [fast, slow, walked, asked, absent_asked, unparked] = crate::watch::view_stats().map(|n| n as f64 / frames);
                let (parked, ordinary) = crate::watch::parked_blocks();
                tracing::info!(
                    "view memory askings per frame (VICTORY_VIEW_SET): {fast:.0} answered in one lookup, {slow:.0} walked \
                     {walked:.0} blocks that were not clean; {parked} blocks parked as uncommitted ({ordinary} of them \
                     no placeholder, looked at again), {absent_asked:.0} questions to Windows whether they still are, \
                     {unparked:.1} found committed; {asked:.1} questions to Windows whether a range is a view; \
                     {whole:.1} refresh pieces copied whole, unwatched (VICTORY_SPLIT_UNWATCHED)",
                    whole = crate::shadow::WHOLE_FALLBACKS.swap(0, Ordering::Relaxed) as f64 / frames
                );
                let shared = crate::watch::GUARD_SHARED.swap(0, Ordering::Relaxed) as f64 / frames;
                tracing::info!("(Neptune) view memory askings answered clean under the guards' shared lock per frame: {shared:.0}");
                {
                    static LAST: std::sync::Mutex<[u64; 2]> = std::sync::Mutex::new([0; 2]);
                    let now: [u64; 2] = std::array::from_fn(|i| crate::watch::RESTED[i].load(Ordering::Relaxed));
                    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
                    let d: [f64; 2] = std::array::from_fn(|i| now[i].wrapping_sub(last[i]) as f64 / frames);
                    *last = now;
                    tracing::info!(
                        "write faults that started a rest per frame: {:.0}; blocks left open by askings because they rest: {:.0} \
                         (VICTORY_GUARD_REST={} us)",
                        d[1],
                        d[0],
                        crate::watch::guard_rest_us()
                    );
                }
            }
            if crate::shadow::cards() {
                let per_frame = |i: usize| card[i] as f64 / frames;
                tracing::info!(
                    "host copies in the card's memory: {} copies {} MB, their bytes from before {} MB; per frame: \
                     {:.1} MB sent up in {:.0} ranges, {:.0} pieces compared on the card, {:.0} of them fetched \
                     ({:.1} MB) with {:.1} more waits for the card ({:.1} ms)",
                    card[7],
                    card[8] >> 20,
                    card[9] >> 20,
                    per_frame(0) / 1e6,
                    per_frame(1),
                    per_frame(2),
                    per_frame(3),
                    per_frame(4) / 1e6,
                    per_frame(5),
                    per_frame(6) / 1e3
                );
            }
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect; heap buffers {picked:.1} passes {picked_ms:.1} ms;                  texture slots written so far: {placed} placed, {objects} of objects; \
                 textures picked by index: {picked_textures} uploaded so far ({picked_mb} MB), \
                 {picked_waiting} waiting, {picked_away} passed over unmapped; depth textures with                  unordered access: {du0} made, bound as depth {du1}x, cleared as depth {du2}x,                  unordered views refused {du3}x; buffer views left null as the host would refuse them: {refused}",
                picked_textures = crate::draw::HEAP_TEXTURES.load(Ordering::Relaxed),
                picked_mb = crate::draw::HEAP_TEXTURE_BYTES.load(Ordering::Relaxed) >> 20,
                picked_waiting = crate::draw::HEAP_TEXTURES_WAITING.load(Ordering::Relaxed),
                picked_away = crate::draw::HEAP_TEXTURES_AWAY.load(Ordering::Relaxed),
                du0 = crate::resource::DEPTH_UNORDERED[0].load(Ordering::Relaxed),
                du1 = crate::resource::DEPTH_UNORDERED[1].load(Ordering::Relaxed),
                du2 = crate::resource::DEPTH_UNORDERED[2].load(Ordering::Relaxed),
                du3 = crate::resource::DEPTH_UNORDERED[3].load(Ordering::Relaxed),
                refused = format!(
                    "{}; depth twins: {} alive, {} depth textures left without one; twin copies per frame: {twin_to_depth:.1} into depth textures, {twin_to_colour:.1} into twins; depth planes: {planes:.1} drawn from their twins per frame, {} times not given back to a twin, {} times used before the pass",
                    crate::descriptor::host::REFUSED_VIEWS.load(Ordering::Relaxed),
                    crate::resource::TWINS[0].load(Ordering::Relaxed),
                    crate::resource::TWINS[1].load(Ordering::Relaxed),
                    crate::resource::PLANES[1].load(Ordering::Relaxed),
                    crate::resource::PLANES[2].load(Ordering::Relaxed)
                ),
                making = crate::pipeline::making(),
                placed = crate::descriptor::host::PLACED_TEXTURE_SLOTS.load(Ordering::Relaxed),
                objects = crate::descriptor::host::OBJECT_TEXTURE_SLOTS.load(Ordering::Relaxed)
            );
            #[cfg(feature = "diagnostics")]
            crate::copystat::log(std::mem::take(&mut self.copy_stats_frames) as f64);
            // Which shaders and pipelines the dropped draws are for, most
            // first: release builds show the ten that cost the most.
            let shown = if cfg!(feature = "diagnostics") { 40 } else { 10 };
            for (why, n) in drops.iter().take(shown) {
                tracing::info!("dropped {:.1}/frame without a host pipeline: {why}", *n as f64 / frames);
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        // The game's display name from its manifest; else what the window
        // said when it was first read (the title's caption).
        let name = match crate::shell::display_name() {
            Some(name) => name.to_owned(),
            None => crate::shell::game_in_caption(&String::from_utf16_lossy(caption)).to_owned(),
        };
        let text = crate::shell::caption(&name, fps, draws);
        let mut text: Vec<u16> = text.encode_utf16().collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown; else, in diagnostics builds, whether the frame was black.
    pub(crate) fn present(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        let showing = std::time::Instant::now();
        let shown = self.present_now(texture, state, frame);
        crate::framelog::add(crate::framelog::T::PresentShowUs, showing.elapsed().as_micros() as u64);
        shown
    }

    fn present_now(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(width, height, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // Presents sent ahead hold back buffers until they have run.
            settle_now(WHY_SYNC);
            // SAFETY: no back buffer is held; every submission was waited for.
            let resized = unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            };
            match resized {
                Ok(()) => {
                    (p.width, p.height, p.format) = (width, height, format);
                    tracing::info!(
                        width,
                        height,
                        format = format.0,
                        frame,
                        "the swap chain's buffers were made again for another size or format"
                    );
                    self.log_video_memory("right after the swap chain's buffers were made again");
                }
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    return None;
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            // Sent ahead ([`late_free_present`]) unless the frame is to be
            // read back at once (a saved or listed frame): then a list of
            // its own, one whose last flight has finished or a new one.
            #[cfg(feature = "diagnostics")]
            let read_now = crate::capture::wanted(frame) || crate::capture::listed(frame);
            #[cfg(not(feature = "diagnostics"))]
            let read_now = false;
            let slot = if late_free_present() && !read_now {
                let finished = FINISHED.load(Ordering::Acquire);
                p.ahead.iter().position(|(_, flown)| *flown <= finished).or_else(|| {
                    let list = self.new_list().ok()?;
                    list.close();
                    p.ahead.push((list, 0));
                    Some(p.ahead.len() - 1)
                })
            } else {
                None
            };
            let copy = match slot {
                Some(i) => &p.ahead[i].0,
                None => &p.copy,
            };
            copy.reset();
            let list = &copy.list;
            list.ResourceBarrier(&[
                transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
            ]);
            list.CopyResource(&back, texture);
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| crate::capture::read_back(&self.device, list, texture))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = if slot.is_some() { None } else { crate::capture::probe(&self.device, list, texture) };
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            list.ResourceBarrier(&[
                transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ),
            ]);
            copy.close();
            match slot {
                Some(i) => {
                    let flown = self.execute_own(copy);
                    p.ahead[i].1 = flown;
                }
                None => self.execute(&[copy]),
            }
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            // `VICTORY_PRESENT_SKIP` ([`present_skip`]): everything above as
            // always (the card does the same work, into the current back
            // buffer, which does not rotate without a present), and the
            // present itself left out, or made every Nth frame without
            // waiting.
            static PRESENTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            match present_skip() {
                0 => {
                    let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
                }
                1 => {}
                n => {
                    if PRESENTS.fetch_add(1, Ordering::Relaxed) % n == 0 {
                        // DXGI_ERROR_WAS_STILL_DRAWING and the like ignored.
                        let _ = p.swap_chain.Present(0, DXGI_PRESENT_DO_NOT_WAIT);
                    }
                }
            }
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Presenter> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height)?,
        };
        // The game's logos as the window's icons, on a thread of their own
        // (nothing here waits for them; a failure leaves the window's own).
        crate::shell::set_icons(hwnd);
        self.log_video_memory("right before the swap chain is made");
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?
                .cast::<IDXGISwapChain3>()?;
            let copy = self.new_list()?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            self.log_video_memory("right after the swap chain is made");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                ahead: Vec::new(),
                hwnd,
                meter: Meter::new(),
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// The card's memory as the system accounts it to this process, in
    /// bytes: the budget and the use of the card's own memory, then of the
    /// PC's memory shared with the card (where the system moves what does
    /// not fit on the card).
    pub(crate) fn video_memory(&self) -> Option<[u64; 4]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        // The card's budget moved by 256 MB or more since the last look:
        // said at once, with what was in use at the last look, so the log
        // has the moment. (It fell from 11,242 MB to some 10,300 MB with
        // the first world frame in runs 179 to 185, and the memory line,
        // seconds later, was the first to show it. We reserve nothing
        // and make the swap chain once, at the start.)
        static LAST: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
        let now = [own.Budget, own.CurrentUsage, shared.CurrentUsage];
        let last = [0, 1, 2].map(|i| LAST[i].swap(now[i], Ordering::Relaxed));
        if last[0] != 0 && last[0].abs_diff(own.Budget) >= (256u64 << 20) {
            tracing::info!(
                "the system changed the budget of the card's memory from {} MB to {} MB (frame {}); at the last look {} MB of the card's and {} MB of the PC's shared memory were in use",
                last[0] >> 20,
                own.Budget >> 20,
                crate::frame::current(),
                last[1] >> 20,
                last[2] >> 20
            );
            self.log_video_memory("right after that change");
        }
        Some([own.Budget, own.CurrentUsage, shared.Budget, shared.CurrentUsage])
    }

    /// Both of the system's memory groups for this process, in bytes: for
    /// the card's own memory and then for the PC's memory shared with the
    /// card, the budget, the use, what could still be reserved and what
    /// is reserved (the layer reserves none).
    pub(crate) fn video_memory_groups(&self) -> Option<[[u64; 4]; 2]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        let all = |m: &DXGI_QUERY_VIDEO_MEMORY_INFO| {
            [m.Budget, m.CurrentUsage, m.AvailableForReservation, m.CurrentReservation]
        };
        Some([all(&own), all(&shared)])
    }

    /// Logs [`Self::video_memory_groups`], with `when` it is said.
    pub(crate) fn log_video_memory(&self, when: &str) {
        let Some([own, shared]) = self.video_memory_groups() else {
            return;
        };
        tracing::info!(
            "video memory as the system counts it, {}: the card's own: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved; the PC's shared with the card: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved (the layer reserves none)",
            when,
            own[0] >> 20,
            own[1] >> 20,
            own[2] >> 20,
            own[3] >> 20,
            shared[0] >> 20,
            shared[1] >> 20,
            shared[2] >> 20,
            shared[3] >> 20
        );
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        crate::census::made(2, size);
        MADE[2].fetch_add(1, Ordering::Relaxed);
        MADE[3].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }

    /// A device-local buffer shaders may write (unordered access) and indirect
    /// execution may read, starting in the common state.
    pub(crate) fn uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        crate::census::made(3, size);
        MADE[4].fetch_add(1, Ordering::Relaxed);
        MADE[5].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a device-local buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no unordered-access buffer");
                None
            }
        }
    }
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    // From just before the command the GPU had not finished: the first
    // twelve after the barrier named draws long done when a list of 79
    // mesh draws stopped at its 45th (run 41).
    for i in from.max(done.saturating_sub(4))..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    let result = unsafe { DefWindowProcW(hwnd, message, wparam, lparam) };
    if message == WM_NCDESTROY {
        crate::shell::window_destroyed(hwnd.0 as usize);
    }
    result
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("Scarlett"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod trail {
    /// The trail keeps the newest things, and the report made when the
    /// host device is removed says them and the pipeline library's state
    /// (no card is asked for any of it).
    #[test]
    fn the_removal_report_says_the_newest_things_done() {
        for n in 0..super::TRAIL_MOST + 5 {
            super::trail(|| format!("thing {n} done"));
        }
        let text = super::removal_context();
        assert!(text.contains("the submitting thread was"));
        assert!(text.contains("pipeline library: no call running"));
        assert!(text.contains(&format!("thing {} done", super::TRAIL_MOST + 4)));
        assert!(!text.contains("thing 4 done"), "the oldest entries are dropped");
    }
}

#[cfg(test)]
mod spare {
    use super::Spare;

    /// Kept things come back by size, the last given first, and nothing is
    /// kept past the limit.
    /// What was given long ago comes out; the rest stays, its bytes too.
    #[test]
    fn things_given_long_ago_are_trimmed() {
        let mut spare: Spare<u32> = Spare::new();
        assert_eq!(spare.give(1, 4, 100), None);
        assert_eq!(spare.give(2, 8, 100), None);
        spare.kept[0].2 = 1;
        let out = spare.trim(|given| given == 1);
        assert_eq!(out, vec![1]);
        assert_eq!(spare.bytes, 8);
        assert_eq!(spare.take(8), Some(2));
        assert_eq!(spare.bytes, 0);
    }

    #[test]
    fn kept_things_come_back_by_size_within_the_limit() {
        let mut spare: Spare<u32> = Spare::new();
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.give(1, 4, 10), None);
        assert_eq!(spare.give(2, 1, 10), None);
        assert_eq!(spare.give(3, 4, 10), None);
        assert_eq!(spare.bytes, 9);
        // Past the limit: handed back, nothing changes.
        assert_eq!(spare.give(4, 4, 10), Some(4));
        assert_eq!(spare.bytes, 9);
        assert_eq!(spare.take(2), None, "no thing of that size");
        assert_eq!(spare.take(4), Some(3));
        assert_eq!(spare.take(4), Some(1));
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.take(1), Some(2));
        assert_eq!(spare.bytes, 0);
        assert!(spare.kept.is_empty());
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::Batch;
    use std::sync::Arc;

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// A copy's ranges stay apart in a batch unless they overlap or touch
    /// (`VICTORY_DISJOINT_REFRESH`), and are joined before the refresh.
    #[test]
    fn refresh_ranges_keep_disjoint_ranges() {
        use super::RefreshRanges;
        let mut ranges = RefreshRanges::One((0x1000, 0x2000));
        ranges.add((0x1800, 0x2400));
        assert_eq!(ranges.as_slice(), &[(0x1000, 0x2400)], "an overlapping range widens the one");
        ranges.add((0x8000, 0x9000));
        ranges.add((0x2400, 0x3000));
        ranges.add((0x8800, 0x8900));
        ranges.merge();
        assert_eq!(ranges.as_slice(), &[(0x1000, 0x3000), (0x8000, 0x9000)], "the gap between them is left out");
    }

    /// Trexies 3cd984d's test: a refresh of two ranges of a copy leaves the
    /// page between them, which the title wrote, as it was; the next refresh
    /// that asks for it takes it.
    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "requires an isolated process with VICTORY_D3D12_WARP=1"]
    fn disjoint_refresh_leaves_the_dirty_gap_pending() {
        use super::{IDXGIAdapter1, Interface, Ordering, PREPARING, host, warp_adapter};
        use crate::{shadow, watch::PAGE};
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};

        assert!(std::env::var_os("VICTORY_D3D12_WARP").is_some());
        let host = host().expect("WARP host device");
        // SAFETY: the WARP adapter and the device are both live.
        let (adapter, device_luid) = unsafe {
            (
                warp_adapter().unwrap().cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap(),
                host.device.GetAdapterLuid(),
            )
        };
        assert_eq!(device_luid.HighPart, adapter.AdapterLuid.HighPart);
        assert_eq!(device_luid.LowPart, adapter.AdapterLuid.LowPart);

        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const SIZE: usize = 3 * PAGE;
        // SAFETY: a new committed, write-watched allocation owned by the test.
        let title = unsafe { VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as usize;
        assert_ne!(title, 0);
        let fill_page = move |page: usize, value: u8| {
            assert!(page < 3);
            // SAFETY: the test retains the committed allocation until all lists finish.
            unsafe { ((title + page * PAGE) as *mut u8).write_bytes(value, PAGE) };
        };
        shadow::register(title, title as u64, SIZE as u64);
        let shadow = shadow::find(title as u64, SIZE as u64).expect("shared shadow buffer");

        let submit = |ranges: &[(usize, usize)], write: bool| {
            let list = host.new_list().unwrap();
            let readback = host.read_back_buffer(SIZE as u64).unwrap();
            let copied = || PREPARING[1].load(Ordering::Relaxed);
            let before = copied();
            for &(offset, len) in ranges {
                shadow.refresh(&list, (title + offset) as u64, len as u64);
            }
            if write {
                shadow.gpu_writes(&list, title as u64, 4);
                list.defer_host(move || {
                    assert_eq!(copied() - before, (2 * PAGE) as u64, "GPU write drains both ranges");
                });
                // A title write after the declaration makes the third page stale again.
                list.defer(move || fill_page(2, 66));
                shadow.refresh(&list, (title + 2 * PAGE) as u64, PAGE as u64);
            }
            // (The buffer may be folded into its window: `VICTORY_FOLD_BUFFERS`.)
            let at = shadow.offset_in(title as u64).expect("the copy holds the buffer");
            // SAFETY: both buffers cover SIZE bytes and live through synchronous execution.
            unsafe { list.list.CopyBufferRegion(&readback, 0, &shadow.buffer, at, SIZE as u64) };
            let mut data = vec![0u8; SIZE];
            list.defer_read_back(readback, data.as_mut_ptr(), [SIZE, 1, SIZE]);
            list.close();
            host.execute(&[&list]);
            let markers = std::array::from_fn(|page| {
                let bytes = &data[page * PAGE..(page + 1) * PAGE];
                assert!(bytes.iter().all(|&byte| byte == bytes[0]), "page {page} is uniform");
                bytes[0]
            });
            (copied() - before, markers)
        };

        for (page, value) in [1, 2, 3].into_iter().enumerate() {
            fill_page(page, value);
        }
        assert_eq!(submit(&[(0, SIZE)], false), (SIZE as u64, [1, 2, 3]));
        for (page, value) in [11, 22, 33].into_iter().enumerate() {
            fill_page(page, value);
        }
        let first = submit(&[(2 * PAGE, PAGE), (0, PAGE)], false);
        let gap = submit(&[(PAGE, PAGE)], false);

        fill_page(0, 44);
        let overlap = submit(&[(PAGE / 2, PAGE / 2), (0, 3 * PAGE / 4)], false);
        fill_page(0, 55);
        fill_page(2, 60);
        let write = submit(&[(0, PAGE), (2 * PAGE, PAGE)], true);

        drop(shadow);
        shadow::unregister(title, title as u64);
        // SAFETY: the allocation is no longer used by any list or shadow.
        assert_ne!(unsafe { VirtualFree(title as *mut _, 0, MEM_RELEASE) }, 0);

        println!("copied bytes and page contents: first {first:?}, gap {gap:?}, overlap {overlap:?}, write {write:?}");
        assert_eq!(overlap, (PAGE as u64, [44, 22, 33]));
        assert_eq!(write, (SIZE as u64, [55, 22, 66]));
        assert_eq!(first, ((2 * PAGE) as u64, [11, 2, 33]));
        assert_eq!(gap, (PAGE as u64, [11, 22, 33]));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }

    #[test]
    fn vram_idle_ms_scales_with_pressure() {
        use super::vram_idle_ms;
        let gib = 1024 * 1024 * 1024;
        // Normal (< 75%)
        assert_eq!(vram_idle_ms(Some([10 * gib, 0, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 7 * gib, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([100, 74, 0, 0])), 5_000);
        // Moderate (75%..85%)
        assert_eq!(vram_idle_ms(Some([100, 75, 0, 0])), 2_500);
        assert_eq!(vram_idle_ms(Some([100, 84, 0, 0])), 2_500);
        // High (>= 85%, including over-budget)
        assert_eq!(vram_idle_ms(Some([100, 85, 0, 0])), 1_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 12 * gib, 0, 0])), 1_000);
        // Missing or zero budget falls back to normal
        assert_eq!(vram_idle_ms(None), 5_000);
        assert_eq!(vram_idle_ms(Some([0, 100, 0, 0])), 5_000);
    }

    #[test]
    fn ram_idle_ms_scales_with_pressure() {
        use super::ram_idle_ms;
        let gib = 1024 * 1024 * 1024;
        // Normal (>= 4 GiB)
        assert_eq!(ram_idle_ms(Some(8 * gib)), 10_000);
        assert_eq!(ram_idle_ms(Some(4 * gib)), 10_000);
        // Moderate (2 GiB..4 GiB)
        assert_eq!(ram_idle_ms(Some(4 * gib - 1)), 5_000);
        assert_eq!(ram_idle_ms(Some(2 * gib)), 5_000);
        // High (< 2 GiB)
        assert_eq!(ram_idle_ms(Some(2 * gib - 1)), 2_000);
        assert_eq!(ram_idle_ms(Some(0)), 2_000);
        // Missing falls back to normal
        assert_eq!(ram_idle_ms(None), 10_000);
    }
}

/// The host's own paths for a copy in the card's memory, end to end, on
/// the software adapter only: the title's bytes go up, a shader reads them
/// and writes, and what it wrote comes back into title memory. Run it with
/// the switches under test in the environment (they are read once per
/// process); it does nothing unless the host device is the software
/// adapter (`VICTORY_D3D12_WARP=1` in a diagnostics build) and the copy is
/// in the card's memory (`VICTORY_CARD_BUFFERS_MIN_MB=1`):
///
/// `VICTORY_D3D12_WARP=1 VICTORY_CARD_BUFFERS_MIN_MB=1 [VICTORY_QUEUE2=n]
///  [VICTORY_RELAY=2 | VICTORY_LATE_WAIT=3] cargo test -p victory-gpu
///  --features diagnostics queue2_tests`
///
/// The same title memory must come out whatever the switches: through the
/// direct queue alone, and through the copy queues.
#[cfg(test)]
mod queue2_tests {
    use super::*;

    /// `memory[to] = memory[from] + add`, after `spin` steps.
    const ADD: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0)"
cbuffer What : register(b0) { uint to; uint from; uint add; uint spin; };
RWByteAddressBuffer memory : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint h = 0u;
    [loop]
    for (uint i = 0u; i < spin; i++) {
        h = (h ^ memory.Load(from)) * 0x01000193u;
    }
    memory.Store(to, memory.Load(from) + add + (h & 0u));
}
"#;

    #[test]
    fn warp_title_bytes_go_up_and_shader_writes_come_back_whatever_the_queues() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        let Some(host) = host() else {
            return;
        };
        // Never on a real card.
        // SAFETY: a plain query.
        let software = host
            .adapter
            .as_ref()
            .and_then(|a| unsafe { a.GetDesc1() }.ok())
            .is_some_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0);
        if !software {
            eprintln!("queue2_tests: skipped, the host device is not the software adapter (VICTORY_D3D12_WARP=1, diagnostics)");
            return;
        }
        const PIECE: usize = 64 << 10;
        const LEN: usize = 4 << 20;
        const ROUNDS: usize = 12;
        // "Title memory": write-watched, as the title's is.
        // SAFETY: a plain allocation of our own.
        let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as *mut u8;
        assert!(!title.is_null());
        let dword = |at: usize| unsafe { title.add(at).cast::<u32>() };
        for i in 0..LEN / 4 {
            // SAFETY: inside the allocation.
            unsafe { dword(i * 4).write_volatile(0x5A00_0000 | i as u32) };
        }
        crate::shadow::register(0x51_5151, title as u64, LEN as u64);
        let shadow = crate::shadow::find(title as u64, LEN as u64).expect("a host copy");
        if !shadow.carded() {
            eprintln!("queue2_tests: skipped, the copy is not in the card's memory (VICTORY_CARD_BUFFERS_MIN_MB=1)");
            return;
        }
        let code = compile(ADD, windows::core::s!("main"), windows::core::s!("cs_5_1")).expect("the test's shader");
        // SAFETY: the shader's own root signature; plain creation.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).expect("a root signature");
            let pso = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .expect("a pipeline");
            (root, pso)
        };
        // Round k: the title writes a dword in piece 1+k; a submission's
        // shader adds k to it and stores the sum in piece 30+k, a range
        // declared as shader-written. Nothing is waited for between the
        // rounds where submissions are sent ahead.
        let from = |k: usize| (1 + k) * PIECE + 4 * k + 64;
        let to = |k: usize| (30 + k) * PIECE + 8 * k + 128;
        let titles = |k: usize| 0x0100_0000 + 0x1000 * k as u32 + 7;
        let mut lists = Vec::new();
        for k in 0..ROUNDS {
            // SAFETY: inside the allocation.
            unsafe { dword(from(k)).write_volatile(titles(k)) };
            let list = host.new_list().expect("a host list");
            shadow.refresh(&list, title as u64, LEN as u64);
            shadow.gpu_writes(&list, (title as usize + (30 + k) * PIECE) as u64, PIECE as u64);
            // SAFETY: an open list; the copy's buffer, with unordered
            // access; the shader's own root signature.
            unsafe {
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                let constants = [to(k) as u32, from(k) as u32, k as u32, 30_000];
                list.list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                list.list.SetComputeRootUnorderedAccessView(1, shadow.buffer.GetGPUVirtualAddress());
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            if late() {
                host.execute_own(&list);
            } else {
                host.execute(&[&list]);
            }
            lists.push(list);
        }
        settle_now(WHY_SYNC);
        for k in 0..ROUNDS {
            // SAFETY: inside the allocation.
            let (sum, kept) = unsafe { (dword(to(k)).read_volatile(), dword(from(k)).read_volatile()) };
            assert_eq!(kept, titles(k), "round {k}: the title's own dword changed");
            assert_eq!(
                sum,
                titles(k) + k as u32,
                "round {k}: the shader read {:#x} where the title had written {:#x}, or its write did not come back",
                sum.wrapping_sub(k as u32),
                titles(k)
            );
        }
        // Nothing else moved.
        let touched: std::collections::HashSet<usize> = (0..ROUNDS).flat_map(|k| [from(k), to(k)]).collect();
        for i in (0..LEN / 4).filter(|i| !touched.contains(&(i * 4))) {
            // SAFETY: inside the allocation.
            let got = unsafe { dword(i * 4).read_volatile() };
            assert_eq!(got, 0x5A00_0000 | i as u32, "dword {i} of title memory changed");
        }
        // The queues the switches name did the work.
        if let Some(second) = crate::queue2::second() {
            let (down, up) = (second.down.stats.jobs.load(Ordering::Relaxed), second.up.stats.jobs.load(Ordering::Relaxed));
            // (Sends that found no landing buffer went the old way.)
            let no_room = second.landing[1].load(Ordering::Relaxed);
            eprintln!(
                "queue2_tests: VICTORY_QUEUE2={} late {} relay {}: {down} fetches and {up} sends through the copy queues ({no_room} more found no landing buffer), the direct queue told to wait {} times",
                crate::queue2::bits(),
                late(),
                relay(),
                second.held[0].load(Ordering::Relaxed)
            );
            // (With the fetch inside flights, `VICTORY_BACK_ASYNC`, flights
            // land from their own slots and fetch nothing of their own.)
            if !inflight_fetch() {
                assert_eq!(down > 0, second.fetches(), "fetches through the copy queue");
            }
            assert_eq!(up + no_room > 0, second.uploads(), "sends through the copy queue");
            assert_eq!(up, second.held[0].load(Ordering::Relaxed), "the direct queue waits for every send");
        } else {
            eprintln!("queue2_tests: the direct queue alone (late {}, relay {})", late(), relay());
        }
        drop(lists);
    }

    /// `memory[to] = first` at once; then `spin` steps, each a store to
    /// `memory[to + 4]`; then `memory[to + 4] = 0` and `memory[to] = last`.
    /// Whatever `spin` is, the bytes it leaves are the same.
    const TWICE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0)"
cbuffer What : register(b0) { uint to; uint first; uint last; uint spin; };
RWByteAddressBuffer memory : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    memory.Store(to, first);
    [loop]
    for (uint i = 0u; i < spin; i++) {
        memory.Store(to + 4u, i);
    }
    memory.Store(to + 4u, 0u);
    memory.Store(to, last);
}
"#;

    /// The hole a fetch on the copy queue had (`VICTORY_QUEUE2` bit 1 with
    /// the relay): flight A writes a piece; flight B, sent behind it,
    /// writes the same piece through a half-way value and leaves it as A
    /// left it. A is landed while B is half way: the fetch, beside B on
    /// the copy queue, brings B's half-way bytes into title memory. B's
    /// compare finds the piece as A's compare left it and flags nothing,
    /// so B's landing fetched nothing and title memory kept the half-way
    /// bytes. With the fix B's landing fetches the piece all the same
    /// (`BESIDE`).
    ///
    /// `VICTORY_D3D12_WARP=1 VICTORY_CARD_BUFFERS_MIN_MB=1 VICTORY_QUEUE2=1
    ///  VICTORY_RELAY=1 cargo test -p victory-gpu --features diagnostics
    ///  fetched_again -- --nocapture` (alone: it lands its two flights
    /// itself; `VICTORY_QUEUE2_REFETCH=0` shows the fault).
    #[test]
    fn warp_a_piece_fetched_while_a_later_flight_writes_it_is_fetched_again() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        let Some(host) = host() else {
            return;
        };
        // Never on a real card.
        // SAFETY: a plain query.
        let software = host
            .adapter
            .as_ref()
            .and_then(|a| unsafe { a.GetDesc1() }.ok())
            .is_some_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0);
        if !software {
            eprintln!("queue2_tests: skipped, the host device is not the software adapter (VICTORY_D3D12_WARP=1, diagnostics)");
            return;
        }
        if !relay() || !crate::queue2::second().is_some_and(|s| s.fetches()) {
            eprintln!("queue2_tests: skipped, fetches do not go to a copy queue beside the relay (VICTORY_RELAY=1 VICTORY_QUEUE2=1)");
            return;
        }
        // With the fetch inside flights (`VICTORY_BACK_ASYNC=1`, off by
        // default) A lands from its own slot, which holds A's bytes and no
        // later flight's: there is no fetch beside B to show.
        if inflight_fetch() {
            eprintln!("queue2_tests: skipped, flights land from their own slots (unset VICTORY_BACK_ASYNC shows the fetch beside them)");
            return;
        }
        const PIECE: usize = 64 << 10;
        const LEN: usize = 4 << 20;
        // SAFETY: a plain allocation of our own.
        let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as *mut u8;
        assert!(!title.is_null());
        let dword = |at: usize| unsafe { title.add(at).cast::<u32>() };
        for i in 0..LEN / 4 {
            // SAFETY: inside the allocation.
            unsafe { dword(i * 4).write_volatile(0x5A00_0000 | i as u32) };
        }
        crate::shadow::register(0x52_5252, title as u64, LEN as u64);
        let shadow = crate::shadow::find(title as u64, LEN as u64).expect("a host copy");
        if !shadow.carded() {
            eprintln!("queue2_tests: skipped, the copy is not in the card's memory (VICTORY_CARD_BUFFERS_MIN_MB=1)");
            return;
        }
        let code = compile(TWICE, windows::core::s!("main"), windows::core::s!("cs_5_1")).expect("the test's shader");
        // SAFETY: the shader's own root signature; plain creation.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).expect("a root signature");
            let pso = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .expect("a pipeline");
            (root, pso)
        };
        // A list whose shader writes the dword at 256 of `piece`, declared.
        let record = |piece: usize, first: u32, last: u32, spin: u32| {
            let list = host.new_list().expect("a host list");
            shadow.refresh(&list, title as u64, LEN as u64);
            shadow.gpu_writes(&list, (title as usize + piece * PIECE) as u64, PIECE as u64);
            // SAFETY: an open list; the copy's buffer, with unordered
            // access; the shader's own root signature.
            unsafe {
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                let constants = [(piece * PIECE + 256) as u32, first, last, spin];
                list.list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                list.list.SetComputeRootUnorderedAccessView(1, shadow.buffer.GetGPUVirtualAddress());
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            list
        };
        // SAFETY: a plain query.
        let completed = || unsafe { host.fence.GetCompletedValue() };
        let wait_for = |value: u64| {
            while completed() < value {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        // How many steps keep the card busy for some seconds: timed on
        // another piece, waited for.
        const TRIAL: u32 = 1 << 20;
        let trial = record(40, 1, 2, TRIAL);
        let timing = std::time::Instant::now();
        host.execute(&[&trial]);
        let took = timing.elapsed().as_secs_f64().max(1e-4);
        let spin = (f64::from(TRIAL) * 3.0 / took).clamp(f64::from(TRIAL), 2.0e9) as u32;
        // SAFETY: inside the allocation.
        assert_eq!(unsafe { dword(40 * PIECE + 256).read_volatile() }, 2, "the trial's write did not come back");
        const AT: usize = 20 * PIECE + 256;
        const HALF_WAY: u32 = 0xDEAD_0001;
        const LEFT: u32 = 0x600D_0002;
        // Flight A, and the card has run it.
        let a = record(20, HALF_WAY, LEFT, 16);
        let (submitted, mut sent_a) = host.submit(&[&a], true).expect("flight A sent ahead");
        let fence_a = *submitted;
        drop(submitted);
        wait_for(fence_a);
        // Flight B: the same piece, through the half-way value, for seconds.
        let b = record(20, HALF_WAY, LEFT, spin);
        let (submitted, mut sent_b) = host.submit(&[&b], true).expect("flight B sent ahead");
        let fence_b = *submitted;
        drop(submitted);
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(
            completed() < fence_b,
            "flight B had run after 0.3 s ({spin} steps; the trial's {TRIAL} took {took:.3} s): nothing to show"
        );
        // A lands, as `Host::relay_land` lands it, while B is half way.
        let mut fetch = None;
        {
            let plan = sent_a.card.as_ref().expect("flight A has a plan for the card's copies");
            host.relay_card_back(&[(sent_a.number, plan)], &mut fetch, fence_a, sent_a.number);
        }
        for (copy, ranges) in &sent_a.back {
            copy.copy_back_done(ranges);
        }
        // SAFETY: inside the allocation.
        let seen = unsafe { dword(AT).read_volatile() };
        let b_ran = completed() >= fence_b;
        eprintln!(
            "queue2_tests: after A's landing title memory holds {seen:#x} (half way {HALF_WAY:#x}, left {LEFT:#x}); B had run: {b_ran}; {spin} steps"
        );
        assert_eq!(
            seen, HALF_WAY,
            "the fetch for A did not come down beside B (it waited for B, or B had not begun): this machine cannot show the fault"
        );
        // B has run; it left the piece as A left it, so its compare flags
        // nothing. Its landing must bring the piece down all the same.
        wait_for(fence_b);
        {
            let plan = sent_b.card.as_ref().expect("flight B has a plan for the card's copies");
            host.relay_card_back(&[(sent_b.number, plan)], &mut fetch, fence_b, sent_b.number);
        }
        for (copy, ranges) in &sent_b.back {
            copy.copy_back_done(ranges);
        }
        // SAFETY: inside the allocation.
        let (left, beside) = unsafe { (dword(AT).read_volatile(), dword(AT + 4).read_volatile()) };
        assert_eq!(
            left, LEFT,
            "title memory kept flight B's half-way bytes: B's compare found the piece as A left it and B's landing fetched nothing"
        );
        assert_eq!(beside, 0, "the dword beside it holds a half-way count");
        for sent in [&mut sent_a, &mut sent_b] {
            if let Some(plan) = sent.card.take() {
                host.card_done(plan);
            }
        }
        drop((a, b, trial));
    }
}

#[cfg(test)]
mod mask_tests {
    use super::*;

    /// A page put in masked (`crate::shadow::merge_mask`) on a real copy in
    /// the card's memory, on the software adapter: the send list's masked
    /// merge (`CARD_MASK`) puts the title's bytes in under the mask and
    /// leaves every other byte of the card's side as the card held it, also
    /// in the words the mask cuts and where the CPU side holds other bytes.
    ///
    /// `VICTORY_D3D12_WARP=1 VICTORY_CARD_BUFFERS_MIN_MB=1 cargo test
    ///  --release -p victory-gpu --features diagnostics --lib masked_page --
    ///  --test-threads=1`
    #[test]
    fn warp_a_masked_page_keeps_the_cards_bytes_outside_the_mask() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        let Some(host) = host() else {
            return;
        };
        // SAFETY: a plain query.
        let software = host
            .adapter
            .as_ref()
            .and_then(|a| unsafe { a.GetDesc1() }.ok())
            .is_some_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0);
        if !software {
            eprintln!("mask_tests: skipped, the host device is not the software adapter (VICTORY_D3D12_WARP=1)");
            return;
        }
        const PIECE: usize = 64 << 10;
        const LEN: usize = 4 << 20;
        // SAFETY: a plain allocation of our own.
        let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as *mut u8;
        assert!(!title.is_null());
        let pattern = |i: usize| (i * 13 + i / 4099) as u8;
        for i in 0..LEN {
            // SAFETY: inside the allocation.
            unsafe { title.add(i).write(pattern(i)) };
        }
        crate::shadow::register(0x53_5353, title as u64, LEN as u64);
        let shadow = crate::shadow::find(title as u64, LEN as u64).expect("a host copy");
        if !shadow.carded() || !card_mask_ready() {
            eprintln!("mask_tests: skipped, the copy is not in the card's memory (VICTORY_CARD_BUFFERS_MIN_MB=1) or no masked merge");
            return;
        }
        // SAFETY: a plain query and a fence of our own.
        let fence: ID3D12Fence = unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.expect("a fence");
        let mut signalled = 0u64;
        let mut drain = || {
            signalled += 1;
            // SAFETY: our fence on the host's queue.
            unsafe { host.queue.Signal(&fence, signalled) }.expect("a signal");
            while unsafe { fence.GetCompletedValue() } < signalled {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        // The card's side takes title memory as it is.
        let list = host.new_list().expect("a host list");
        shadow.refresh(&list, title as u64, LEN as u64);
        list.close();
        host.execute(&[&list]);
        drain();
        // Page 3: the CPU side holds other bytes than the card's side, and
        // the title changed some of them (odd offsets: masks that cut words).
        let page = 3 * PIECE;
        let side = shadow.side_for_tests();
        let unmasked = [page + 100, page + 202, page + 4096, page + PIECE - 1];
        for &i in &unmasked {
            // SAFETY: inside the CPU side's mapping.
            unsafe { side.add(i).write(pattern(i) ^ 0xff) };
        }
        let masked: Vec<usize> = (page + 1..page + PIECE).step_by(37).filter(|i| !unmasked.contains(i)).collect();
        assert!(masked.contains(&(page + 1 + 37 * 5)));
        // SAFETY: the page of the CPU side; nothing else writes it now.
        let held = unsafe { std::slice::from_raw_parts_mut(side.add(page), PIECE) };
        let mut now = held.to_vec();
        for &i in &masked {
            now[i - page] = pattern(i) ^ 0x5a;
        }
        let bytes = shadow.put_masked(page, &now, held);
        assert_eq!(bytes, masked.len());
        // The send list, behind what is on the queue.
        let plan = host.card_before(&[], false, 0).expect("a plan for the masked page");
        drain();
        // The card's side, read back.
        let (back, data) = host.shared_buffer(LEN as u64).expect("read-back memory");
        let (allocator, copy) = host.make_prelude().expect("a list");
        // SAFETY: a fresh list; live buffers.
        unsafe {
            allocator.Reset().unwrap();
            copy.Reset(&allocator, None).unwrap();
            copy.CopyBufferRegion(&back, 0, &shadow.buffer, 0, LEN as u64);
            copy.Close().unwrap();
            host.queue.ExecuteCommandLists(&[Some(copy.cast().unwrap())]);
        }
        drain();
        // SAFETY: the read-back's mapping; its copy has run.
        let card = unsafe { std::slice::from_raw_parts(data, LEN) };
        for &i in &masked {
            assert_eq!(card[i], pattern(i) ^ 0x5a, "byte {i:#x} under the mask is not the title's");
        }
        for i in (0..LEN).filter(|i| !masked.contains(i)) {
            assert_eq!(card[i], pattern(i), "byte {i:#x} outside the mask changed on the card");
        }
        host.card_done(plan);
        eprintln!("mask_tests: {} title bytes put in by one masked merge; the other {} bytes kept", masked.len(), LEN - masked.len());
    }
}

#[cfg(test)]
mod written_source_tests {
    use super::*;

    /// A copy out of a range this list's shaders write is staged from the
    /// host copy on the card, after what the list wrote there before (a
    /// copy into the host copy stands in for the shader), not from title
    /// memory as it was before the list (allforks a9b474f, PalmVIDev
    /// 4d1ce258); a range no shader of the list writes is not staged.
    #[test]
    fn a_copy_source_the_list_wrote_is_staged_from_the_host_copy() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        let Some(host) = host() else {
            return; // no host GPU here
        };
        if !copy_written_source() {
            return; // VICTORY_COPY_WRITTEN_SOURCE=0
        }
        const LEN: usize = 1 << 16;
        // "Title memory": write-watched, as the title's is.
        // SAFETY: a plain allocation of our own.
        let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as *mut u8;
        assert!(!title.is_null());
        // SAFETY: inside the allocation.
        unsafe { std::ptr::write_bytes(title, 0x11, LEN) };
        let start = title as u64;
        let resource = 0x7e57_0a9busize;
        crate::shadow::register(resource, start, LEN as u64);
        let shadow = crate::shadow::find(start, LEN as u64).expect("a host copy");
        let offset = shadow.offset_in(start).expect("the copy holds its start");
        let list = host.new_list().unwrap();
        // Nothing declared written: no staging.
        assert!(list.staged_written_source(host, start + 256, 256).is_none());
        // The "shader" declares and writes 256 bytes of the copy.
        list.defer_gpu_write(&shadow, start as usize + 256, 256);
        let pattern: Vec<u8> = (0..256u32).map(|i| (i * 5 + 1) as u8).collect();
        let (upload, data) = host.upload_buffer(256).unwrap();
        // SAFETY: the upload buffer holds 256 mapped bytes; an open list,
        // live buffers, ranges inside them.
        unsafe {
            std::ptr::copy_nonoverlapping(pattern.as_ptr(), data, 256);
            list.list.CopyBufferRegion(&shadow.buffer, offset + 256, &upload, 0, 256);
        }
        let staged = list.staged_written_source(host, start + 256, 256).expect("staged");
        let (back, mapped) = host.shared_buffer(256).expect("read-back memory");
        // SAFETY: an open list; the staged buffer is in the copy source state.
        unsafe { list.list.CopyBufferRegion(&back, 0, &staged, 0, 256) };
        list.close();
        host.execute(&[&list]);
        // SAFETY: the copies finished (execute waits); 256 mapped bytes.
        let read = unsafe { std::slice::from_raw_parts(mapped, 256).to_vec() };
        assert_eq!(read, pattern, "what the list wrote, not the title's 0x11");
        crate::shadow::unregister(resource, start);
    }
}

#[cfg(test)]
mod fill_on_card_tests {
    use super::*;

    /// A readback buffer of `len` bytes.
    fn readback(host: &Host, len: u64) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: len,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buffer)
                .unwrap();
        }
        buffer.unwrap()
    }

    /// The parts of joined ranges inside an ask (`HostList::card_holds`).
    #[test]
    fn clipped_keeps_the_parts_inside() {
        let ranges = [(0x100, 0x200), (0x300, 0x400), (0x500, 0x600)];
        assert_eq!(clipped(&ranges, 0x180, 0x380), vec![(0x180, 0x200), (0x300, 0x380)]);
        assert_eq!(clipped(&ranges, 0x200, 0x300), vec![]);
        assert_eq!(clipped(&ranges, 0, 0x1000), ranges.to_vec());
        assert_eq!(clipped(&[], 0, 10), vec![]);
    }

    /// What a flight's shaders wrote to a host copy (bytes the title's memory
    /// does not have) reaches constants filled on the card, byte for byte,
    /// for two fills in one list (Trexies 25fbcce's test).
    #[test]
    fn constants_filled_on_the_card_read_what_the_card_wrote() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        const LEN: usize = 1 << 16;
        // Title memory: a buffer the title fills with 0xAA.
        let title: &'static mut [u8] = Box::leak(vec![0xAAu8; LEN].into_boxed_slice());
        let start = title.as_ptr() as u64;
        let resource = 0x7e57_0001usize;
        crate::shadow::register(resource, start, LEN as u64);
        let shadow = crate::shadow::find(start + 64, 512).expect("a host copy");
        let offset = shadow.offset_in(start).expect("the copy holds the buffer");
        // A "flight" writes a pattern into the copy on the card.
        let pattern: Vec<u8> = (0..1024u32).map(|i| (i * 7 + 3) as u8).collect();
        let (upload, data) = host.upload_buffer(1024).unwrap();
        // SAFETY: the upload buffer holds 1024 mapped bytes.
        unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), data, 1024) };
        let flight = host.new_list().unwrap();
        // SAFETY: an open list; live buffers, ranges inside them.
        unsafe { flight.list.CopyBufferRegion(&shadow.buffer, offset + 64, &upload, 0, 1024) };
        flight.close();
        host.execute(&[&flight]);
        // The next list fills two constant buffers from it on the card, in
        // place (the list has stopped gathering: `VICTORY_CARD_FILLS_BATCH=1`
        // keeps such fills where they are).
        let list = host.new_list().unwrap();
        list.gather_close();
        let first = list.card_constants(&shadow, start + 64, 512).expect("room");
        let second = list.card_constants(&shadow, start + 64 + 512, 300).expect("room");
        assert_eq!(first % 256, 0);
        assert_eq!(second % 256, 0);
        let constants = list.card_constants.lock().unwrap().as_ref().unwrap().0.clone();
        // SAFETY: a live buffer.
        let base = unsafe { constants.GetGPUVirtualAddress() };
        let back = readback(host, 1024);
        // SAFETY: an open list; the constants buffer is in the constant
        // buffer state after the fills; ranges inside both buffers.
        unsafe {
            list.list.ResourceBarrier(&[transition(
                &constants,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list.CopyBufferRegion(&back, 0, &constants, first - base, 512);
            list.list.CopyBufferRegion(&back, 512, &constants, second - base, 300);
        }
        list.close();
        host.execute(&[&list]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copies finished (execute waits); 1024 bytes.
        let read = unsafe {
            back.Map(0, None, Some(&mut mapped)).unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 1024).to_vec()
        };
        assert_eq!(read[..512], pattern[..512], "the first fill");
        assert_eq!(read[512..812], pattern[512..812], "the second fill");
        // The title's memory was never read: it still holds 0xAA.
        assert!(title[64..1088].iter().all(|&b| b == 0xAA));
        crate::shadow::unregister(resource, start);
    }

    /// (Card kinds job) The hoisting decision: a fill goes to the prelude
    /// only when nothing recorded before it may change its source.
    #[test]
    fn a_fill_is_hoisted_only_when_nothing_before_it_may_write_its_source() {
        // Nothing written, nothing storing recorded: hoisted in both modes.
        assert_eq!(fill_place(1, false, false), FillPlace::Prelude);
        assert_eq!(fill_place(2, false, false), FillPlace::Prelude);
        // A write of the list declared over the source: in place, always.
        assert_eq!(fill_place(1, false, true), FillPlace::Written);
        assert_eq!(fill_place(2, true, true), FillPlace::Written);
        assert_eq!(fill_place(1, true, true), FillPlace::Written);
        // A dispatch or storing draw came first, none declared over the
        // source: mode 1 keeps it, mode 2 (the GATHER_LATE rule) moves it.
        assert_eq!(fill_place(1, true, false), FillPlace::Storing);
        assert_eq!(fill_place(2, true, false), FillPlace::Prelude);
        // Off.
        assert_eq!(fill_place(0, false, false), FillPlace::Off);
    }

    /// (Card kinds job) On the card: fills hoisted to the prelude read what
    /// a flight before wrote, as fills in place do; the same bytes asked
    /// twice share one place, adjacent ones one copy; a fill after a
    /// declared write of the list over its source stays in place and reads
    /// the bytes as they are there.
    #[test]
    fn hoisted_fills_read_what_fills_in_place_read() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        if fills_batch() == 0 {
            return;
        }
        const LEN: usize = 1 << 16;
        let title: &'static mut [u8] = Box::leak(vec![0xAAu8; LEN].into_boxed_slice());
        let start = title.as_ptr() as u64;
        let resource = 0x7e57_0002usize;
        crate::shadow::register(resource, start, LEN as u64);
        let shadow = crate::shadow::find(start + 64, 512).expect("a host copy");
        let offset = shadow.offset_in(start).expect("the copy holds the buffer");
        let pattern: Vec<u8> = (0..2048u32).map(|i| (i * 13 + 5) as u8).collect();
        let (upload, data) = host.upload_buffer(2048).unwrap();
        // SAFETY: the upload buffer holds 2048 mapped bytes.
        unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), data, 2048) };
        let flight = host.new_list().unwrap();
        // SAFETY: an open list; live buffers, ranges inside them.
        unsafe { flight.list.CopyBufferRegion(&shadow.buffer, offset + 64, &upload, 0, 2048) };
        flight.close();
        host.execute(&[&flight]);
        let hoisted0 = FILLS.each_ref().map(|n| n.load(Ordering::Relaxed));
        let list = host.new_list().unwrap();
        let a = list.card_constants(&shadow, start + 64, 512).expect("room");
        let again = list.card_constants(&shadow, start + 64, 512).expect("room");
        let b = list.card_constants(&shadow, start + 64 + 512, 256).expect("room");
        assert_eq!(a, again, "the same bytes share a place");
        assert_eq!(b, a + 512, "the next place");
        {
            let h = list.hoisted.lock().unwrap();
            assert_eq!(h.copies.len(), 1, "adjacent sources and places: one copy");
            assert_eq!(h.copies[0].3, 768);
        }
        // A declared write of the list over the next range: in place.
        let arc = crate::shadow::find(start + 1024, 256).expect("a host copy");
        list.defer_gpu_write(&arc, start as usize + 1024, 256);
        let c = list.card_constants(&shadow, start + 1024, 256).expect("room");
        let in_place = list.card_constants.lock().unwrap().as_ref().unwrap().0.clone();
        let prelude_buffer = list.hoisted.lock().unwrap().buffer.clone().unwrap();
        // SAFETY: live buffers.
        let (pb, ib) = unsafe { (prelude_buffer.GetGPUVirtualAddress(), in_place.GetGPUVirtualAddress()) };
        assert!(c >= ib && c < ib + (1 << 20), "the fill after the write is in place");
        let after = FILLS.each_ref().map(|n| n.load(Ordering::Relaxed));
        assert!(after[FILL_HOISTED] - hoisted0[FILL_HOISTED] >= 3);
        assert!(after[FILL_WRITTEN] > hoisted0[FILL_WRITTEN]);
        let back = readback(host, 1024);
        // SAFETY: an open list; the prelude leaves its buffer readable as
        // constants, the in-place fill left its own so; ranges inside.
        unsafe {
            let constants = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
            list.list.ResourceBarrier(&[
                transition(&prelude_buffer, constants, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(&in_place, constants, D3D12_RESOURCE_STATE_COPY_SOURCE),
            ]);
            list.list.CopyBufferRegion(&back, 0, &prelude_buffer, a - pb, 512);
            list.list.CopyBufferRegion(&back, 512, &prelude_buffer, b - pb, 256);
            list.list.CopyBufferRegion(&back, 768, &in_place, c - ib, 256);
            list.list.ResourceBarrier(&[
                transition(&prelude_buffer, D3D12_RESOURCE_STATE_COPY_SOURCE, constants),
                transition(&in_place, D3D12_RESOURCE_STATE_COPY_SOURCE, constants),
            ]);
        }
        list.close();
        host.execute(&[&list]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copies finished (execute waits); 1024 bytes.
        let read = unsafe {
            back.Map(0, None, Some(&mut mapped)).unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 1024).to_vec()
        };
        assert_eq!(read[..512], pattern[..512], "the hoisted fill");
        assert_eq!(read[512..768], pattern[512..768], "the joined hoisted fill");
        assert_eq!(read[768..1024], pattern[960..1216], "the fill in place");
        eprintln!("hoisted_fills: 3 fills hoisted (one shared, one joined) and 1 in place read the flight's bytes");
        crate::shadow::unregister(resource, start);
    }
}

/// (Hang job, runs 63-70) The fill-ahead trip path must never block the
/// submit thread: a fill reads ahead of a flight, the flight's shaders
/// change its bytes (a trip: the piece is held), then another fill of the
/// same piece is looked at with the read-waits meter on (it used to take
/// the list's step lock a second time on the same thread and hang for
/// good), then the flights land. Run under both settings:
/// `VICTORY_FILL_AHEAD_PER_PIECE=0` and default (the setting is read once
/// per process).
#[cfg(test)]
mod hang_tests {
    use super::*;

    #[test]
    fn a_trip_then_a_fill_of_the_held_piece_then_a_landing_does_not_block() {
        let Some(host) = host() else {
            eprintln!("hang_tests: skipped, no host device");
            return;
        };
        let list = host.new_list().expect("a host list");
        // The time-out: a thread of its own ends the process if the path
        // below blocks (the list cannot go to another thread).
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let watch = std::thread::spawn(move || {
            if rx.recv_timeout(std::time::Duration::from_secs(20)).is_err() {
                eprintln!("hang_tests: the trip path blocked for 20 s");
                std::process::exit(101);
            }
        });
        {
            let list = &list;
            {
                const LEN: usize = 480;
                const PIECE: usize = 1 << 16;
                // Title memory: a 64 KiB piece of its own.
                let mut memory = vec![1u8; 3 * PIECE];
                let src = (memory.as_ptr() as usize).next_multiple_of(PIECE) + PIECE / 2;
                let mut upload = vec![0u8; 2 * LEN];
                // What flights in the air may write: the whole piece.
                let written = [(src - PIECE / 2, src + PIECE / 2)];

                // 1. The fill reads ahead of the flights (not blocked).
                list.defer_fill(upload.as_mut_ptr(), src as *const u8, LEN);
                assert_eq!(blocked_by(&written, &[list], true), None, "a fill reads ahead");

                // 2. A flight lands other bytes there: a trip, the piece held.
                fill_ahead::wrote(src, &[3u8; LEN]);
                assert!(!fill_ahead::on(src, LEN), "the tripped piece is held (either setting)");

                // 2b. The per-second line at a present, with a piece held
                // (runs 68-71: it took the held-pieces lock twice in one
                // statement with PER_PIECE=1 and the relay's next landing
                // waited behind it for good), while a landing checks fills
                // on another thread, as the relay does.
                fill_ahead::read(src + 128, LEN);
                let landing = std::thread::spawn(move || {
                    for _ in 0..100 {
                        fill_ahead::wrote(src + 128, &[5u8; LEN]);
                    }
                });
                for _ in 0..100 {
                    fill_ahead::read(src + 256, LEN);
                    fill_ahead::line(1.0);
                }
                landing.join().unwrap();

                // 3. Another fill of the same piece, looked at with the
                // meter (count) and then as relay_wait's condition does
                // (holding the line lock): it waits, and says so.
                list.defer_fill(upload[LEN..].as_mut_ptr(), (src + 64) as *const u8, LEN);
                assert_eq!(blocked_by(&written, &[list], true), Some(WHY_READ), "a fill of a held piece waits");
                {
                    let line = line();
                    let _ = &line;
                    assert_eq!(blocked_by(&written, &[list], false), Some(WHY_READ));
                }

                // 4. The flights land: nothing is in the air, nothing waits.
                fill_ahead::wrote(src + 64, &[3u8; LEN]);
                assert_eq!(blocked_by(&[], &[list], true), None, "nothing in the air: no wait");
                drop(memory);
                tx.send(()).unwrap();
            }
        }
        watch.join().unwrap();
        eprintln!(
            "hang_tests: trip, fill of the held piece, landing: no block (VICTORY_FILL_AHEAD_PER_PIECE={})",
            u8::from(fill_ahead::per_piece())
        );
    }
}
