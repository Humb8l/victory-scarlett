//! The console's DirectStorage (`XCRDAPI.dll`): a factory that opens files
//! and creates request queues and status arrays the title polls. Its
//! interfaces have grown since 2020 (new interface ids, new slots); the
//! slots here are the ones the title calls, read from its call
//! sites. Slots nobody has needed yet log their offset.
//!
//! Submitted requests of every queue run on one pool of workers, as the
//! console's hardware runs them: in parallel, started in queue order,
//! queues served by weighted round robin over their priorities (about 10x
//! per level, counted in bytes), with at most the staging buffer's worth of
//! compressed source in flight. Each request reads its source (a file or
//! memory), inflates zlib data and/or unpacks BCPack data, and writes the
//! destination (an address, or physical pages). Completion is what stays
//! in order: a status entry is written once every request enqueued before
//! it on its queue has finished, with the first failure (in queue order)
//! since the previous status entry, or success.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ffi::{CStr, c_char, c_void};
use std::os::windows::fs::FileExt;
use std::os::windows::io::FromRawHandle;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};

use victory_formats::bcpack::{self, Tables};

use super::xasync::{E_INVALIDARG, E_NOTIMPL, E_PENDING};
use super::xgameruntime::{Guid, HRESULT, Methods, Object, S_OK, add_ref, missing, put};
use crate::{HleModule, hle_export};

/// The factory interface the title asks for.
const FACTORY: Guid = Guid::new(
    0x6295_9d11,
    0xd45f,
    0x4884,
    [0xa0, 0xa3, 0xf7, 0x86, 0x2e, 0xa2, 0x43, 0xae],
);
const E_NOINTERFACE: HRESULT = 0x8000_4002_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;
/// `HRESULT_FROM_WIN32(ERROR_HANDLE_EOF)`: the source ends early.
const END_OF_FILE: HRESULT = 0x8007_0026_u32 as i32;
/// `HRESULT_FROM_WIN32(ERROR_INVALID_DATA)`: a bad compressed stream.
const INVALID_DATA: HRESULT = 0x8007_000D_u32 as i32;

type Out = *mut *mut c_void;

static FACTORY_OBJECT: Object = Object::new("dstorage factory", &FACTORY_VTABLE);

pub extern "system" fn DStorageGetFactory(iid: *const Guid, out: Out) -> HRESULT {
    // SAFETY: the title passes a GUID (null tolerated).
    match unsafe { iid.as_ref() } {
        Some(&iid) if iid == FACTORY => {
            put(out, std::ptr::from_ref(&FACTORY_OBJECT).cast_mut().cast())
        }
        iid => {
            tracing::warn!(iid = ?iid.map(Guid::to_string), "DStorageGetFactory: unknown interface");
            E_NOINTERFACE
        }
    }
}

/// One of our objects with state: the runtime's object header (method
/// table first), a reference count, then `T`.
#[repr(C)]
struct Instance<T> {
    base: Object,
    refs: AtomicU32,
    state: T,
}

impl<T> Instance<T> {
    fn create(base: Object, state: T) -> *mut c_void {
        Box::into_raw(Box::new(Instance {
            base,
            refs: AtomicU32::new(1),
            state,
        }))
        .cast()
    }
}

extern "system" fn add_ref_instance<T>(this: *const Instance<T>) -> u32 {
    // SAFETY: an instance we created and the title still holds.
    unsafe { &*this }.refs.fetch_add(1, Ordering::Relaxed) + 1
}

extern "system" fn release<T>(this: *mut Instance<T>) -> u32 {
    // SAFETY: as above.
    let left = unsafe { &*this }.refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if left == 0 {
        // SAFETY: the last reference; created by `Instance::create`.
        drop(unsafe { Box::from_raw(this) });
    }
    left
}

fn iid_text(iid: *const Guid) -> String {
    // SAFETY: the title passes a GUID (null tolerated).
    unsafe { iid.as_ref() }.map_or_else(|| "null".into(), Guid::to_string)
}

/// `HRESULT_FROM_WIN32` of a host error.
fn from_win32(error: &std::io::Error) -> HRESULT {
    error
        .raw_os_error()
        .map_or(E_INVALIDARG, |e| (0x8007_0000 | (e as u32 & 0xffff)) as i32)
}

// The factory.

type Factory = *const Object;

/// Opens `path` for reading; the file object owns the handle.
extern "system" fn open_file(_: Factory, path: *const u16, iid: *const Guid, out: Out) -> HRESULT {
    const GENERIC_READ: u32 = 0x8000_0000;
    const FILE_SHARE_READ: u32 = 1;
    const OPEN_EXISTING: u32 = 3;
    if path.is_null() {
        return E_INVALIDARG;
    }
    // Goes through `files`, so paths on our virtual drives work too.
    let handle = super::files::CreateFileW(
        path,
        GENERIC_READ,
        FILE_SHARE_READ,
        std::ptr::null(),
        OPEN_EXISTING,
        0,
        std::ptr::null_mut(),
    );
    // SAFETY: the title passes a NUL-terminated path.
    let name = unsafe {
        let len = (0..).take_while(|&i| *path.add(i) != 0).count();
        String::from_utf16_lossy(std::slice::from_raw_parts(path, len))
    };
    if handle as isize == -1 {
        let error = std::io::Error::last_os_error();
        tracing::warn!(path = name, %error, "dstorage: opening a file failed");
        return from_win32(&error);
    }
    let kept = name.clone();
    tracing::info!(path = name, iid = iid_text(iid), "dstorage: file opened");
    // SAFETY: a handle we just opened and nobody else owns.
    let file = unsafe { std::fs::File::from_raw_handle(handle) };
    let file = Instance::create(Object::new("dstorage file", &FILE_VTABLE), File { file, more: Mutex::default() });
    WALLS_FILES.lock().unwrap_or_else(|e| e.into_inner()).push((file as usize, kept));
    put(out, file)
}

/// `(capacity, name, iid, out)`.
extern "system" fn create_status_array(
    _: Factory,
    capacity: u32,
    name: *const c_char,
    iid: *const Guid,
    out: Out,
) -> HRESULT {
    tracing::info!(
        capacity,
        name = text(name),
        iid = iid_text(iid),
        "dstorage: status array"
    );
    let entries = (0..capacity).map(|_| AtomicI32::new(S_OK)).collect();
    let array = Instance::create(
        Object::new("dstorage status array", &STATUS_VTABLE),
        StatusArray { entries },
    );
    put(out, array)
}

/// The queue's description: as in 2020 (source type, capacity, priority,
/// name), plus a field the title sets to 1.
#[repr(C)]
struct QueueDesc {
    source_type: u64,
    capacity: u16,
    priority: i8,
    name: *const c_char,
    unknown: u64,
}

extern "system" fn create_queue(
    _: Factory,
    desc: *const QueueDesc,
    iid: *const Guid,
    out: Out,
) -> HRESULT {
    // SAFETY: the title passes its description (null tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let name = text(desc.name);
    tracing::info!(
        source = desc.source_type,
        capacity = desc.capacity,
        priority = desc.priority,
        name,
        unknown = desc.unknown,
        iid = iid_text(iid),
        "dstorage: queue"
    );
    let Some(scheduler) = scheduler() else {
        return E_OUTOFMEMORY;
    };
    let mut lanes = scheduler.lock();
    let id = lanes.next_id;
    lanes.next_id += 1;
    let lane = Lane::new(desc.priority, desc.capacity);
    lanes.lanes.insert(id, lane);
    drop(lanes);
    put(
        out,
        Instance::create(Object::new("dstorage queue", &QUEUE_VTABLE), Queue { id }),
    )
}

/// The title asks for 32 MiB and stops if this fails.
extern "system" fn set_staging_buffer_size(_: Factory, size: u32) -> HRESULT {
    tracing::info!(size, "dstorage: staging buffer size");
    if let Some(scheduler) = scheduler() {
        scheduler.lock().staging = u64::from(size);
    }
    S_OK
}

fn text(name: *const c_char) -> String {
    if name.is_null() {
        return String::new();
    }
    // SAFETY: a NUL-terminated name from the title.
    unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

static FACTORY_VTABLE: Methods<16> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    missing::<3> as *const (),            // 0x18 (2020: queue creation)
    open_file as *const (),               // 0x20
    create_status_array as *const (),     // 0x28
    missing::<6> as *const (),            // 0x30 (2020: CPU affinity)
    missing::<7> as *const (),            // 0x38 (2020: debug flags)
    set_staging_buffer_size as *const (), // 0x40
    missing::<9> as *const (),
    create_queue as *const (), // 0x50
    missing::<11> as *const (),
    missing::<12> as *const (),
    missing::<13> as *const (),
    missing::<14> as *const (),
    missing::<15> as *const (),
]);

// Files.

struct File {
    file: std::fs::File,
    /// (Neptune) More handles of the same file, each with a file object of its own
    /// ([`reopened`]), for the workers to read through at the same time.
    more: Mutex<Vec<std::fs::File>>,
}

/// (Neptune) Whether each worker reads a file through a handle of its own
/// (`VICTORY_DSTORAGE_OWN_HANDLES=0`: all through the one the file was opened with, as
/// before).
///
/// Windows runs the reads of one file object one after the other when it was not opened for
/// overlapped reads: in a load eight or nine workers stood in line behind one read at a
/// time, on NVMe drives that serve many at once.
fn own_handles() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DSTORAGE_OWN_HANDLES").is_none_or(|v| v != "0"))
}

/// Another handle of `file`, with a file object of its own (read only, as the file was
/// opened).
fn reopened(file: &std::fs::File) -> Option<std::fs::File> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::ReOpenFile;
    const GENERIC_READ: u32 = 0x8000_0000;
    const FILE_SHARE_READ: u32 = 1;
    // SAFETY: a live handle of ours; the new one is owned by the `File` made of it.
    let handle = unsafe { ReOpenFile(file.as_raw_handle(), GENERIC_READ, FILE_SHARE_READ, 0) };
    if handle as isize == -1 || handle.is_null() {
        static SAID: AtomicBool = AtomicBool::new(false);
        if !SAID.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                error = %std::io::Error::last_os_error(),
                "dstorage: a file could not be opened once more; its reads go one after the other"
            );
        }
        return None;
    }
    // SAFETY: a handle just opened that nobody else owns.
    Some(unsafe { std::fs::File::from_raw_handle(handle) })
}

static FILE_VTABLE: Methods<8> = Methods([
    missing::<0> as *const (),
    add_ref_instance::<File> as *const (),
    release::<File> as *const (),
    missing::<3> as *const (),
    missing::<4> as *const (),
    missing::<5> as *const (),
    missing::<6> as *const (),
    missing::<7> as *const (),
]);

// Status arrays.

struct StatusArray {
    entries: Box<[AtomicI32]>,
}

type Array = *mut Instance<StatusArray>;

/// The result of the requests before entry `index`: pending, the first
/// failure, or success.
extern "system" fn status_result(this: Array, index: u32) -> HRESULT {
    // SAFETY: an array we created and the title still holds.
    let entries = &unsafe { &*this }.state.entries;
    entries
        .get(index as usize)
        .map_or(E_INVALIDARG, |e| e.load(Ordering::Acquire))
}

/// The title polls this (entry 0, a `Sleep(1)` loop) and never
/// reads the result itself.
extern "system" fn is_complete(this: Array, index: u32) -> bool {
    let complete = status_result(this, index) != E_PENDING;
    // (F10) The frame thread's polls, meter only.
    victory_gpu::framestream::status_polled(complete);
    complete
}

static STATUS_VTABLE: Methods<5> = Methods([
    missing::<0> as *const (),
    add_ref_instance::<StatusArray> as *const (),
    release::<StatusArray> as *const (),
    is_complete as *const (),   // 0x18
    status_result as *const (), // 0x20
]);

// Queues.

/// A queue object: its lane in the scheduler.
struct Queue {
    id: u64,
}

impl Drop for Queue {
    fn drop(&mut self) {
        close_lane(self.id);
    }
}

type QueueRef = *const Instance<Queue>;

/// (F10) `VICTORY_DSTORAGE_OLDEST_FIRST` (default 1; 0: as before; 2: over
/// priority classes too): workers take the reads of the oldest batch first.
/// The title retires its batches strictly oldest first and blocks its frame
/// thread when 16 are outstanding, so the oldest one decides when it goes on.
fn oldest_first() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = std::env::var("VICTORY_DSTORAGE_OLDEST_FIRST").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1u8).min(2);
        victory_gpu::framestream::OLDEST_FIRST_MODE.store(u32::from(mode), Ordering::Relaxed);
        mode
    })
}

/// Every queue's requests, run by a pool of workers.
struct Scheduler {
    lanes: Mutex<Lanes>,
    /// Requests to start, or staging space freed.
    work: Condvar,
    /// A request finished (for full queues and `close`).
    finished: Condvar,
}

/// Priority weights per class (low, normal, high and real-time), as the
/// console's: about 10x per level, counted against request sizes.
const COST: [u64; 3] = [100, 10, 1];
/// The console's default staging buffer.
const STAGING: u64 = 32 << 20;

struct Lanes {
    lanes: HashMap<u64, Lane>,
    next_id: u64,
    /// Submission order over all queues.
    next_seq: u64,
    /// Bytes of staging held by running requests, and the limit.
    staged: u64,
    staging: u64,
    /// A start waited for staging space since the last finish.
    starved: bool,
    /// Weighted round robin: each class's virtual time (bytes × cost), and
    /// the time of the last start; within a class, the lane clock of the
    /// last start.
    clock: [u64; 3],
    now: u64,
    lane_now: [u64; 3],
    /// Since the last debug summary: requests, bytes written, time spent
    /// running them.
    done: (u64, u64, std::time::Duration),
    refused: u64,
    /// BCPack bytes written and time spent on them.
    bcpack: (u64, std::time::Duration),
    since: Option<std::time::Instant>,
}

/// A queue's requests in the scheduler.
struct Lane {
    class: usize,
    capacity: usize,
    /// Bytes started, for round robin among the lanes of a class.
    clock: u64,
    /// Enqueued, not yet submitted.
    unsubmitted: Vec<Command>,
    /// Submitted, by sequence number: reads not started, status entries
    /// not written, reads running, and failures not yet reported.
    /// (Storage wait job) With each read: whether its page array is known
    /// filled (it was submitted by the title's own Submit, see
    /// [`pages_at_submit`]).
    reads: VecDeque<(u64, Request, bool)>,
    statuses: VecDeque<(u64, Status)>,
    running: BTreeSet<u64>,
    failures: BTreeMap<u64, HRESULT>,
    closed: bool,
}

// SAFETY: requests point into title memory and our own objects, which stay
// valid until the title has seen them complete; status arrays count their
// references atomically.
unsafe impl Send for Lane {}

/// A read a worker runs.
struct Job {
    lane: u64,
    seq: u64,
    request: Request,
    staging: u64,
    /// Its page array is filled ([`pages_at_submit`]).
    ready: bool,
}

// SAFETY: as for `Lane`.
unsafe impl Send for Job {}

/// The scheduler, with its workers started on first use; `None` if not
/// even one worker could start.
/// How many storage workers run at once (soldsawn 0bba762,
/// `VICTORY_DSTORAGE_WORKERS=<n>` sets it outright, 1..=64): the PC's
/// PHYSICAL cores, not its logical processors (8 here, not 16). The pool
/// reads, inflates and unpacks in parallel; SMT twins add little and
/// crowd the submitting thread and the copies. The author on an 8c/16t
/// PC: copying back 2.4-3.5 ms/MB with 8 against 5.7 ms/MB with 16.
fn worker_count() -> usize {
    static COUNT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *COUNT.get_or_init(|| {
        if let Some(n) = std::env::var("VICTORY_DSTORAGE_WORKERS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|n| (1..=64).contains(n))
        {
            return n;
        }
        let logical = std::thread::available_parallelism().map_or(4, |n| n.get());
        physical_cores().unwrap_or_else(|| (logical / 2).max(2)).clamp(1, 32)
    })
}

/// The PC's physical cores, if the system says (one `RelationProcessorCore`
/// record per core, whatever its SMT twins).
fn physical_cores() -> Option<usize> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetLogicalProcessorInformationEx(relationship: i32, buffer: *mut u8, length: *mut u32) -> i32;
    }
    const RELATION_PROCESSOR_CORE: i32 = 0;
    let mut len: u32 = 0;
    // SAFETY: a null buffer asks for the size (the call fails, `len` filled).
    unsafe { GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut len) };
    if len == 0 {
        return None;
    }
    let mut buf = vec![0u8; len as usize];
    // SAFETY: the buffer is `len` bytes; the call writes at most that.
    if unsafe { GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf.as_mut_ptr(), &mut len) } == 0 {
        return None;
    }
    // Records of { Relationship: u32, Size: u32, ... }. (The author read the
    // size at offset 0, the relationship, and so always fell back to half
    // the logical processors.)
    let (mut at, mut cores) = (0usize, 0usize);
    while at + 8 <= (len as usize).min(buf.len()) {
        let size = u32::from_le_bytes(buf[at + 4..at + 8].try_into().ok()?) as usize;
        if size < 8 || at + size > buf.len() {
            break;
        }
        cores += 1;
        at += size;
    }
    (cores > 0).then_some(cores)
}

fn scheduler() -> Option<&'static Scheduler> {
    static SCHEDULER: OnceLock<Option<Scheduler>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let scheduler = SCHEDULER
        .get_or_init(|| {
            Some(Scheduler {
                lanes: Mutex::new(Lanes {
                    lanes: HashMap::new(),
                    next_id: 0,
                    next_seq: 0,
                    staged: 0,
                    staging: STAGING,
                    starved: false,
                    clock: [0; 3],
                    now: 0,
                    lane_now: [0; 3],
                    done: Default::default(),
                    refused: 0,
                    bcpack: Default::default(),
                    since: None,
                }),
                work: Condvar::new(),
                finished: Condvar::new(),
            })
        })
        .as_ref()?;
    STARTED.call_once(|| {
        let workers = worker_count();
        let started = (0..workers)
            .filter(|i| {
                std::thread::Builder::new()
                    .name(format!("dstorage {i}"))
                    .spawn(|| work(scheduler))
                    .is_ok()
            })
            .count();
        tracing::info!(
            workers = started,
            logical = std::thread::available_parallelism().map_or(0, |n| n.get()),
            "dstorage: workers (VICTORY_DSTORAGE_WORKERS; physical cores unless set)"
        );
    });
    Some(scheduler)
}

impl Scheduler {
    fn lock(&self) -> MutexGuard<'_, Lanes> {
        self.lanes.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The walls trace (`VICTORY_WALLS_TRACE`): 0 or unset: off; 1: a line for
/// every read request that fails, with its file; 2: for every request (the
/// list of a launch's reads, by file and offset, is what a launch with the
/// first room's walls and one without are compared by).
fn walls_trace() -> u8 {
    static ON: OnceLock<u8> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("VICTORY_WALLS_TRACE").ok().and_then(|v| v.trim().parse::<u8>().ok()).unwrap_or(0)
    })
}

/// The files opened, by the address of our file object, with their paths
/// (for the walls trace: a request names its file by that address).
static WALLS_FILES: Mutex<Vec<(usize, String)>> = Mutex::new(Vec::new());

/// The walls trace's line for a request that is done.
fn walls_request(request: &Request, result: &Result<(), HRESULT>, started: std::time::Instant) {
    let level = walls_trace();
    if level == 0 || (level == 1 && result.is_ok()) {
        return;
    }
    let file = if request.options & MEMORY_SOURCE != 0 {
        "memory".to_string()
    } else {
        let files = WALLS_FILES.lock().unwrap_or_else(|e| e.into_inner());
        let source = request.source as usize;
        files.iter().find(|f| f.0 == source).map_or_else(|| "?".to_string(), |f| f.1.clone())
    };
    let hr = match result {
        Ok(()) => 0,
        Err(hr) => *hr as u32,
    };
    tracing::info!(
        file,
        offset = request.offset,
        source_size = request.source_size,
        destination_size = request.destination_size,
        to_pages = request.options & PAGE_DESTINATION != 0,
        options = format_args!("{:#x}", request.options),
        ms = started.elapsed().as_millis() as u64,
        ok = result.is_ok(),
        hr = format_args!("{hr:#010x}"),
        "walls trace: a read request is done"
    );
}

/// A worker: runs the next read of the most deserving queue.
fn work(scheduler: &'static Scheduler) {
    // `VICTORY_DISK_CORES=slow`: the workers leave the fast cores to the
    // title's threads (`victory_gpu::framelog::keep_on_slow_cores`; off by
    // default until it is measured).
    #[cfg(windows)]
    {
        static SLOW: OnceLock<bool> = OnceLock::new();
        if *SLOW.get_or_init(|| std::env::var_os("VICTORY_DISK_CORES").is_some_and(|v| v == "slow")) {
            victory_gpu::framelog::keep_on_slow_cores("dstorage worker");
        }
    }
    let mut lanes = scheduler.lock();
    loop {
        let Some(job) = lanes.start() else {
            lanes = scheduler.work.wait(lanes).unwrap_or_else(|e| e.into_inner());
            continue;
        };
        drop(lanes);
        let started = std::time::Instant::now();
        PAGES_READY.set(job.ready);
        #[cfg(windows)]
        let faults = victory_gpu::faults::scope(victory_gpu::faults::Kind::Disk);
        let result = execute(&job.request);
        #[cfg(windows)]
        drop(faults);
        PAGES_READY.set(false);
        split_add(S_ALL, started);
        SPLIT[S_REQUESTS].fetch_add(1, Ordering::Relaxed);
        walls_request(&job.request, &result, started);
        lanes = scheduler.lock();
        lanes.finish(&job, result);
        lanes.count(&job, started);
        if std::mem::take(&mut lanes.starved) {
            scheduler.work.notify_all();
        }
        scheduler.finished.notify_all();
    }
}

impl Lanes {
    /// Counts a finished request; every 5 s a debug line sums them up.
    fn count(&mut self, job: &Job, started: std::time::Instant) {
        let (requests, bytes, busy) = &mut self.done;
        *requests += 1;
        *bytes += u64::from(job.request.destination_size);
        *busy += started.elapsed();
        #[cfg(windows)]
        {
            use victory_gpu::framelog::{T, add};
            add(T::DiskReads, 1);
            add(T::DiskBytes, u64::from(job.request.destination_size));
            add(T::DiskUs, started.elapsed().as_micros() as u64);
        }
        if job.request.options & BCPACK != 0 {
            self.bcpack.0 += u64::from(job.request.destination_size);
            self.bcpack.1 += started.elapsed();
        }
        let since = *self.since.get_or_insert(started);
        let elapsed = since.elapsed();
        if elapsed.as_secs() >= 5 {
            let waiting: usize = self.lanes.values().map(|l| l.reads.len()).sum();
            let reading = READING.swap(0, Ordering::Relaxed) as f64 / 1e9;
            tracing::debug!(
                requests = *requests,
                mb_per_s = format_args!("{:.1}", *bytes as f64 / elapsed.as_secs_f64() / 1e6),
                busy_workers = format_args!("{:.2}", busy.as_secs_f64() / elapsed.as_secs_f64()),
                reading_workers = format_args!("{:.2}", reading / elapsed.as_secs_f64()),
                arrays_as_enqueued = STALE[0].load(Ordering::Relaxed),
                arrays_filled_late = STALE[1].load(Ordering::Relaxed),
                arrays_waited_ms = STALE[2].load(Ordering::Relaxed) / 1000,
                arrays_trusted = STALE[3].load(Ordering::Relaxed),
                staging_waits = std::mem::take(&mut self.refused),
                bcpack_mb = self.bcpack.0 / 1_000_000,
                bcpack_workers = format_args!("{:.2}", self.bcpack.1.as_secs_f64() / elapsed.as_secs_f64()),
                waiting,
                "dstorage: last {:.1} s",
                elapsed.as_secs_f64()
            );
            self.done = Default::default();
            self.bcpack = Default::default();
            self.since = None;
        }
    }

    /// Starts the next read of the class with the least virtual time, from
    /// its lane that has started the fewest bytes, if the staging buffer
    /// has room for it (or nothing else holds any).
    fn start(&mut self) -> Option<Job> {
        self.start_as(oldest_first())
    }

    /// [`Lanes::start`] with `VICTORY_DSTORAGE_OLDEST_FIRST` = `mode`: 0 the
    /// lane of the class that has started the fewest bytes; 1 (default) the
    /// class as before, then the lane whose next read was submitted first
    /// (submission numbers are over all queues, a lane's reads and status
    /// entries in order, so that read belongs to the oldest batch not yet
    /// all started; within a batch, today's order); 2 the oldest read of
    /// any class (priorities ignored).
    fn start_as(&mut self, mode: u8) -> Option<Job> {
        let busy = || self.lanes.iter().filter(|(_, l)| !l.reads.is_empty());
        let class = if mode >= 2 {
            busy().filter_map(|(_, l)| Some((l.reads.front()?.0, l.class))).min()?.1
        } else {
            busy().map(|(_, l)| (self.clock[l.class], l.class)).min()?.1
        };
        let (_, _, id) = busy()
            .filter(|(_, l)| l.class == class)
            .filter_map(|(&id, l)| {
                let seq = l.reads.front()?.0;
                Some(if mode >= 1 { (seq, l.clock, id) } else { (l.clock, seq, id) })
            })
            .min()?;
        let lane = self.lanes.get_mut(&id)?;
        let &(seq, request, ready) = lane.reads.front()?;
        let staging = staging_bytes(&request);
        if self.staged > 0 && self.staged + staging > self.staging {
            self.starved = true;
            self.refused += 1;
            return None;
        }
        lane.reads.pop_front();
        lane.running.insert(seq);
        self.staged += staging;
        let size = u64::from(request.destination_size.max(request.source_size)).max(1);
        self.now = self.clock[class];
        self.clock[class] += size * COST[class];
        self.lane_now[class] = lane.clock;
        lane.clock += size;
        Some(Job {
            lane: id,
            seq,
            request,
            staging,
            ready,
        })
    }

    fn finish(&mut self, job: &Job, result: Result<(), HRESULT>) {
        self.staged -= job.staging;
        if let Some(lane) = self.lanes.get_mut(&job.lane) {
            lane.running.remove(&job.seq);
            if let Err(hr) = result {
                lane.failures.insert(job.seq, hr);
            }
            lane.flush();
        }
    }

    /// Moves a queue's enqueued commands to the workers: the title's own
    /// Submit, after which every page array of them is filled.
    fn submit(&mut self, id: u64) {
        self.submit_as(id, usize::MAX, true);
    }

    /// (Storage wait job) The first `upto` enqueued commands to the workers,
    /// their page arrays known filled (`ready`) or not.
    fn submit_as(&mut self, id: u64, upto: usize, ready: bool) {
        let Some(class) = self.lanes.get(&id).map(|l| l.class) else {
            return;
        };
        // A class that had nothing waiting starts from now, not from the
        // credit it left behind.
        if !self
            .lanes
            .values()
            .any(|l| l.class == class && !l.reads.is_empty())
        {
            self.clock[class] = self.clock[class].max(self.now);
        }
        let Some(lane) = self.lanes.get_mut(&id) else {
            return;
        };
        if lane.reads.is_empty() {
            lane.clock = lane.clock.max(self.lane_now[class]);
        }
        let upto = upto.min(lane.unsubmitted.len());
        let rest = lane.unsubmitted.split_off(upto);
        for command in std::mem::replace(&mut lane.unsubmitted, rest) {
            let seq = self.next_seq;
            self.next_seq += 1;
            match command {
                Command::Read(request) => {
                    if !ready && pages_at_submit() && request.options & PAGE_DESTINATION != 0 {
                        // Noted now, as at enqueue before: the title has not
                        // reached its mapping step for them yet.
                        as_enqueued(&request);
                    }
                    lane.reads.push_back((seq, request, ready));
                }
                Command::Status(status) => lane.statuses.push_back((seq, status)),
            }
        }
        lane.flush();
    }
}

impl Lane {
    fn new(priority: i8, capacity: u16) -> Lane {
        Lane {
            class: priority.clamp(-1, 1).wrapping_add(1) as usize,
            capacity: usize::from(capacity),
            clock: 0,
            unsubmitted: Vec::new(),
            reads: VecDeque::new(),
            statuses: VecDeque::new(),
            running: BTreeSet::new(),
            failures: BTreeMap::new(),
            closed: false,
        }
    }

    /// Requests and status entries held, submitted or not.
    fn entries(&self) -> usize {
        self.unsubmitted.len() + self.reads.len() + self.statuses.len() + self.running.len()
    }

    /// Writes the status entries whose earlier reads have all finished.
    fn flush(&mut self) {
        while let Some(&(seq, _)) = self.statuses.front() {
            let waiting = self.reads.front().map(|r| r.0);
            if waiting.into_iter().chain(self.running.first().copied()).any(|s| s < seq) {
                return;
            }
            let Some((_, status)) = self.statuses.pop_front() else {
                return;
            };
            let later = self.failures.split_off(&seq);
            let first = std::mem::replace(&mut self.failures, later)
                .into_values()
                .next();
            status
                .entry()
                .store(first.unwrap_or(S_OK), Ordering::Release);
            // (F10) Its batch complete, meter only.
            victory_gpu::framestream::status_written(status.array as usize, status.index);
        }
    }
}

/// Staging bytes a request holds while it runs: what it buffers between
/// the file and the destination. Raw reads go straight to their
/// destination, memory sources need none (as on the console).
fn staging_bytes(request: &Request) -> u64 {
    let buffered = request.options & (ZLIB | BCPACK | PAGE_DESTINATION) != 0;
    if buffered && request.options & MEMORY_SOURCE == 0 {
        u64::from(request.source_size)
    } else {
        0
    }
}

/// Ends a queue: what was not started is dropped, and this waits for the
/// reads already running (they write title memory).
fn close_lane(id: u64) {
    let Some(scheduler) = scheduler() else {
        return;
    };
    let mut lanes = scheduler.lock();
    let Some(lane) = lanes.lanes.get_mut(&id) else {
        return;
    };
    lane.closed = true;
    lane.unsubmitted.clear();
    lane.reads.clear();
    for (_, s) in &lane.statuses {
        victory_gpu::framestream::status_dropped(s.array as usize, s.index);
    }
    lane.statuses.clear(); // releases their arrays
    while lanes.lanes.get(&id).is_some_and(|l| !l.running.is_empty()) {
        lanes = scheduler
            .finished
            .wait(lanes)
            .unwrap_or_else(|e| e.into_inner());
    }
    lanes.lanes.remove(&id);
    scheduler.finished.notify_all();
}

/// A read request as the title builds it, 0x48 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct Request {
    options: u64,
    /// An address, or with `PAGE_DESTINATION` an array of 64 KiB page
    /// numbers.
    destination: *mut c_void,
    /// With `PAGE_DESTINATION`, the low 16 bits are where in the first page
    /// the destination starts.
    page_offset: u64,
    destination_size: u32,
    /// Our file object, or with `MEMORY_SOURCE` an address.
    source: *const c_void,
    offset: u64,
    source_size: u32,
    /// Compressed size between two decompression stages; unused here.
    intermediate_size: u32,
    /// Matched by cancellation; the title leaves it 0.
    tag: u64,
    name: *const c_char,
}

// Request options.
/// Inflate zlib (RFC 1950) data.
const ZLIB: u64 = 1;
/// Bits 1-4: decode BCPack data of this BC format number (0: none).
const BCPACK: u64 = 0xf << 1;
const PAGE_DESTINATION: u64 = 1 << 21;
const MEMORY_SOURCE: u64 = 1 << 22;
// Bits 5-20 select a further transform, bit 23 a source in physical pages;
// The title's packages need neither, so requests with them fail.

const PAGE: usize = 64 * 1024;

enum Command {
    Read(Request),
    Status(Status),
}


/// A status entry to write, holding a reference on its array.
struct Status {
    array: Array,
    index: u32,
}

impl Drop for Status {
    fn drop(&mut self) {
        release(self.array);
    }
}

impl Status {
    fn entry(&self) -> &AtomicI32 {
        // SAFETY: we hold a reference; the index was checked when enqueued.
        &unsafe { &*self.array }.state.entries[self.index as usize]
    }
}

extern "system" fn enqueue_request(this: QueueRef, request: *const Request) {
    // SAFETY: the title passes its request (null tolerated).
    let Some(&request) = (unsafe { request.as_ref() }) else {
        return;
    };
    victory_gpu::framestream::other_call();
    if !pages_at_submit() {
        as_enqueued(&request);
    }
    queue(this, Command::Read(request));
}

extern "system" fn enqueue_status(this: QueueRef, array: Array, index: u32) {
    // SAFETY: the title passes an array we created (null tolerated).
    let Some(entries) = (unsafe { array.as_ref() }).map(|a| &a.state.entries) else {
        return;
    };
    let Some(entry) = entries.get(index as usize) else {
        tracing::warn!(index, "dstorage: status entry out of range");
        return;
    };
    entry.store(E_PENDING, Ordering::Release);
    add_ref_instance(array);
    victory_gpu::framestream::other_call();
    victory_gpu::framestream::status_enqueued(array as usize, index);
    queue(this, Command::Status(Status { array, index }));
    // The title's batch submit enqueues its status after its mapping step.
    title_moved();
}

/// Adds a command to the queue; blocks while the queue is full, and
/// submits on its own once half of it is enqueued, as the console does.
fn queue(this: QueueRef, command: Command) {
    // SAFETY: a queue we created and the title still holds.
    let id = unsafe { &*this }.state.id;
    let Some(scheduler) = scheduler() else {
        return;
    };
    let mut lanes = scheduler.lock();
    let mut blocked: Option<std::time::Instant> = None;
    loop {
        let Some(lane) = lanes.lanes.get(&id).filter(|l| !l.closed) else {
            return; // closed: dropped (a status releases its array)
        };
        // One slot stays free, as the console's count of empty slots says.
        if lane.entries() + 1 < lane.capacity.max(2) {
            break;
        }
        // (Storage wait job) Full with commands held back for the title's
        // Submit: they go now (their page arrays checked as before), or the
        // title would wait here for reads that never start.
        if !lane.unsubmitted.is_empty() {
            ARRAYS[A_FULL].fetch_add(1, Ordering::Relaxed);
            lanes.submit_as(id, usize::MAX, false);
            scheduler.work.notify_all();
            continue;
        }
        // (F10) How long the frame thread waits here, meter only.
        blocked.get_or_insert_with(std::time::Instant::now);
        lanes = scheduler
            .finished
            .wait(lanes)
            .unwrap_or_else(|e| e.into_inner());
    }
    if let Some(since) = blocked {
        victory_gpu::framestream::enqueue_blocked(since.elapsed().as_nanos() as u64);
    }
    let Some(lane) = lanes.lanes.get_mut(&id) else {
        return;
    };
    lane.unsubmitted.push(command);
    if lane.unsubmitted.len() >= (lane.capacity / 2).max(1) {
        if pages_at_submit() {
            // (Storage wait job) Half full: what comes before the first
            // read into physical pages goes; that read and what follows wait
            // for the title's Submit, which comes after its mapping step has
            // filled their page arrays (0x1400e88f0: executor, EnqueueStatus,
            // Submit).
            let upto = lane
                .unsubmitted
                .iter()
                .position(|c| matches!(c, Command::Read(r) if r.options & PAGE_DESTINATION != 0))
                .unwrap_or(lane.unsubmitted.len());
            if upto < lane.unsubmitted.len() {
                ARRAYS[A_HELD].fetch_add(1, Ordering::Relaxed);
            }
            if upto > 0 {
                lanes.submit_as(id, upto, true);
                scheduler.work.notify_all();
            }
        } else {
            lanes.submit_as(id, usize::MAX, false);
            scheduler.work.notify_all();
        }
    }
}

extern "system" fn submit(this: QueueRef) {
    victory_gpu::framestream::other_call();
    // SAFETY: as above.
    let id = unsafe { &*this }.state.id;
    let Some(scheduler) = scheduler() else {
        return;
    };
    let mut lanes = scheduler.lock();
    let pending = lanes.lanes.get(&id).map_or(0, |l| l.unsubmitted.len());
    tracing::debug!(commands = pending, "dstorage: submit");
    if pages_at_submit() {
        lanes.submit(id);
    } else {
        lanes.submit_as(id, usize::MAX, false);
    }
    scheduler.work.notify_all();
    drop(lanes);
    title_moved();
}

/// Ends the queue whatever its reference count: what was not started never
/// completes, and this returns once the reads already running have
/// finished. The title calls this, not `Release`.
extern "system" fn close(this: QueueRef) {
    // SAFETY: as above.
    close_lane(unsafe { &*this }.state.id);
}

/// Reads, inflates and/or unpacks, and writes one request.
fn execute(request: &Request) -> Result<(), HRESULT> {
    let options = request.options;
    let fail = |hr: HRESULT, why: &str| {
        tracing::warn!(
            options = format_args!("{options:#x}"),
            offset = request.offset,
            source_size = request.source_size,
            destination_size = request.destination_size,
            intermediate_size = request.intermediate_size,
            tag = request.tag,
            name = text(request.name),
            hr = format_args!("{:#010x}", hr as u32),
            "dstorage: request failed: {why}"
        );
        hr
    };
    if options & !(ZLIB | BCPACK | PAGE_DESTINATION | MEMORY_SOURCE) != 0 {
        return Err(fail(E_NOTIMPL, "unsupported options"));
    }
    if request.source.is_null() || request.destination.is_null() {
        return Err(fail(E_INVALIDARG, "no source or destination"));
    }
    let bc = ((options & BCPACK) >> 1) as u8;
    let size = request.destination_size as usize;
    if options & (ZLIB | BCPACK) == 0 {
        if request.source_size > request.destination_size {
            return Err(fail(E_INVALIDARG, "source larger than destination"));
        }
        let size = request.source_size as usize;
        // A file read into a title address is the kernel's write, and the
        // kernel does not fault into the GPU side's guard: a block guarded
        // again between `kernel_will_write` and the read (the guard over all
        // the views of a page protects other views from other threads) made
        // the read fail with ERROR_INVALID_USER_BUFFER, and the title stops
        // at the failed request (runs amd-C and amd-C2, seconds after the
        // world's first frame). It is read beside and copied in: our copy
        // faults into the guard like the title's own writes.
        let bounce = options & (PAGE_DESTINATION | MEMORY_SOURCE) == 0;
        return into_destination(request, size, |out| {
            if !bounce {
                return read_into(request, out).map(|()| size);
            }
            let mut data = vec![0; size];
            read_into(request, &mut data)?;
            out.copy_from_slice(&data);
            Ok(size)
        })
        .map_err(|hr| fail(hr, "copying"));
    }
    let tables = if bc != 0 { Some(bcpack_tables()?) } else { None };
    let t = std::time::Instant::now();
    let source = read_source(request).map_err(|hr| fail(hr, "reading the source"));
    split_add(S_SRC, t);
    let source = source?;
    // The console's order: zlib first, then BCPack.
    let source = if bc != 0 && options & ZLIB != 0 {
        let t = std::time::Instant::now();
        let mut packed = vec![0; request.intermediate_size as usize];
        let n = inflate(&source, &mut packed).map_err(|hr| fail(hr, "zlib"));
        split_add(S_INFLATE, t);
        let n = n?;
        packed.truncate(n);
        Cow::Owned(packed)
    } else {
        source
    };
    into_destination(request, size, |out| match tables {
        Some(tables) => bcpack::decode(bc, &source, out, tables)
            .map(|()| out.len())
            .map_err(|e| fail(INVALID_DATA, &e.to_string())),
        None => inflate(&source, out).map_err(|hr| fail(hr, "zlib")),
    })?;
    tracing::trace!(
        options = format_args!("{options:#x}"),
        offset = request.offset,
        size,
        "dstorage: request done"
    );
    Ok(())
}

/// Inflates a zlib stream into `out`, returning the bytes written. A stream
/// that ends short of `out` passes: the title's chunks fill it exactly, and
/// nothing says that is required.
fn inflate(source: &[u8], out: &mut [u8]) -> Result<usize, HRESULT> {
    thread_local! {
        static INFLATE: std::cell::RefCell<zlib_rs::Inflate> =
            std::cell::RefCell::new(zlib_rs::Inflate::new(true, 15));
    }
    INFLATE.with_borrow_mut(|state| {
        state.reset(true);
        // `Inflate::decompress` cannot distinguish a valid empty stream from
        // an insufficient buffer when `avail_out == 0`, so use a 1-byte scratch
        // destination and verify `written <= out_len` afterward.
        let out_len = out.len();
        let mut scratch = [0u8; 1];
        let dest = if out_len == 0 { &mut scratch[..] } else { out };
        let (mut in_pos, mut out_pos) = (0usize, 0usize);
        let status = loop {
            match state.decompress(
                &source[in_pos..],
                &mut dest[out_pos..],
                zlib_rs::InflateFlush::NoFlush,
            ) {
                Ok(zlib_rs::Status::StreamEnd) => break Ok(zlib_rs::Status::StreamEnd),
                Ok(zlib_rs::Status::Ok) => {
                    let next_in = state.total_in() as usize;
                    let next_out = state.total_out() as usize;
                    if next_in == in_pos && next_out == out_pos {
                        break Ok(zlib_rs::Status::BufError);
                    }
                    in_pos = next_in;
                    out_pos = next_out;
                }
                other => break other,
            }
        };
        let written = state.total_out() as usize;
        match status {
            Ok(zlib_rs::Status::StreamEnd) if written <= out_len => Ok(written),
            other => {
                tracing::debug!(?other, "dstorage: zlib stream");
                Err(INVALID_DATA)
            }
        }
    })
}

/// The BCPack format's constant tables, parsed once for the process. If
/// they do not parse every BCPack request fails: one warning, then debug.
fn bcpack_tables() -> Result<&'static Tables, HRESULT> {
    static TABLES: OnceLock<Result<Tables, String>> = OnceLock::new();
    static WARNED: AtomicBool = AtomicBool::new(false);
    match TABLES.get_or_init(load_bcpack_tables) {
        Ok(tables) => Ok(tables),
        Err(why) if !WARNED.swap(true, Ordering::Relaxed) => {
            tracing::warn!("dstorage: BCPack requests fail: {why} (more at debug)");
            Err(E_NOTIMPL)
        }
        Err(_) => {
            tracing::debug!("dstorage: BCPack request failed: no tables");
            Err(E_NOTIMPL)
        }
    }
}

/// The tables VIctory ships with.
#[cfg(not(test))]
fn load_bcpack_tables() -> Result<Tables, String> {
    Tables::builtin().map_err(|e| format!("the built-in BCPack tables: {e}"))
}

/// Tests decode streams made against tables of their own, from this folder.
#[cfg(test)]
static BCPACK_DIR: OnceLock<std::path::PathBuf> = OnceLock::new();

#[cfg(test)]
fn use_bcpack_tables(dir: std::path::PathBuf) {
    let _ = BCPACK_DIR.set(dir);
}

#[cfg(test)]
fn load_bcpack_tables() -> Result<Tables, String> {
    let dir = BCPACK_DIR.get().ok_or("no folder set for the test tables")?;
    let bytes = std::fs::read(dir.join("tables.bin")).map_err(|e| e.to_string())?;
    Tables::parse(&bytes).map_err(|e| e.to_string())
}

/// A compressed request's source: borrowed from memory, or read from the
/// file.
fn read_source(request: &Request) -> Result<Cow<'static, [u8]>, HRESULT> {
    let size = request.source_size as usize;
    if request.options & MEMORY_SOURCE != 0 {
        // SAFETY: the title's source holds `size` bytes until completion.
        let bytes = unsafe { std::slice::from_raw_parts(request.source.cast(), size) };
        return Ok(Cow::Borrowed(bytes));
    }
    let mut bytes = vec![0; size];
    read_into(request, &mut bytes)?;
    Ok(Cow::Owned(bytes))
}

/// Nanoseconds workers spent reading files, for the debug summary.
static READING: AtomicU64 = AtomicU64::new(0);

/// Fills `out` (the source size) from the request's file or memory.
fn read_into(request: &Request, out: &mut [u8]) -> Result<(), HRESULT> {
    if request.options & MEMORY_SOURCE != 0 {
        // SAFETY: the title's source holds `out.len()` bytes until completion.
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(request.source.cast(), out.len()) });
        return Ok(());
    }
    // SAFETY: a file object we created; the title keeps it open.
    let state = &unsafe { &*request.source.cast::<Instance<File>>() }.state;
    // (Neptune) Through a handle of this read's own, so that it does not wait for the
    // other workers' reads of the same file (`VICTORY_DSTORAGE_OWN_HANDLES`).
    let own = if own_handles() {
        let spare = state.more.lock().unwrap_or_else(|e| e.into_inner()).pop();
        spare.or_else(|| reopened(&state.file))
    } else {
        None
    };
    let file = own.as_ref().unwrap_or(&state.file);
    let started = std::time::Instant::now();
    let mut done = 0;
    let result = loop {
        if done == out.len() {
            break Ok(());
        }
        match file.seek_read(&mut out[done..], request.offset + done as u64) {
            Ok(0) => break Err(END_OF_FILE),
            Ok(n) => done += n,
            Err(error) => break Err(from_win32(&error)),
        }
    };
    READING.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    split_add(S_READ, started);
    if let Some(own) = own {
        state.more.lock().unwrap_or_else(|e| e.into_inner()).push(own);
    }
    result
}

/// Runs `fill` on the destination's first `len` bytes; `fill` returns how
/// many it produced. GPU-visible destinations are host memory like the
/// rest (GPU address = CPU address), so an address is written in place;
/// physical pages are too when they are consecutive pages of one
/// allocation, and otherwise go through a buffer.
fn into_destination(
    request: &Request,
    len: usize,
    fill: impl FnOnce(&mut [u8]) -> Result<usize, HRESULT>,
) -> Result<(), HRESULT> {
    if request.options & PAGE_DESTINATION == 0 {
        // Opens the guarded blocks and tells the GPU side a placed texture
        // there changes. `fill` writes from user mode (a file read is read
        // beside first, see `execute`), so a block guarded again meanwhile
        // faults into the guard instead of failing the write.
        let t = std::time::Instant::now();
        victory_gpu::kernel_will_write(request.destination as usize, len);
        split_add(S_NOTES, t);
        // SAFETY: the title's destination holds `destination_size` bytes, at
        // least `len`, and nothing else writes it until completion.
        let out = unsafe { std::slice::from_raw_parts_mut(request.destination.cast::<u8>(), len) };
        let t = std::time::Instant::now();
        let filled = fill(out).map(drop);
        split_add(S_FILL, t);
        let t = std::time::Instant::now();
        // Said again now that the bytes are there (upstream's c75fdba holds
        // the blocks open over the read and marks them once more at its
        // end): a copy or a texture pass that took the memory while the
        // read was under way took what was there before. Counted; acted on
        // only with `VICTORY_PAGE_WRITES=all`.
        victory_gpu::title_memory_written(request.destination as usize, len, false);
        split_add(S_NOTES, t);
        return filled;
    }
    let offset = (request.page_offset & 0xffff) as usize;
    let (array, count) = (request.destination.cast::<usize>().cast_const(), (offset + len).div_ceil(PAGE));
    let t = std::time::Instant::now();
    let mut pages = page_array(array, count);
    split_add(S_PAGES, t);
    // Consecutive pages of one allocation (the usual case): written in
    // place through one view.
    if let Some(mut window) = super::xmem::page_window(&pages, offset, len) {
        touch(window.bytes());
        let t = std::time::Instant::now();
        let filled = fill(window.bytes()).map(drop);
        split_add(S_FILL, t);
        drop(window);
        let t = std::time::Instant::now();
        // The title maps pages where the card reads them and has them
        // filled afterwards: the GPU side's copies of those addresses
        // learn of the bytes only from this (upstream c75fdba, fork PR
        // #571). After a decode that failed part way too: what it wrote
        // is there.
        super::xmem::pages_written(&pages, offset, len);
        split_add(S_NOTES, t);
        return filled;
    }
    let t = std::time::Instant::now();
    let mut data = vec![0; len];
    split_add(S_BUF, t);
    let t = std::time::Instant::now();
    let n = fill(&mut data);
    split_add(S_FILL, t);
    let n = n?;
    // A page the title has just made, or a view the host refuses for a
    // moment, comes right a little later: the write is tried again before
    // the request fails (`VICTORY_DSTORAGE_RETRIES`, 12 tries, about 0.9 s
    // in all; 0: not tried again, as before). Runs 175-178 lost 12 to 446
    // reads a launch here, all in one second of the loading; the one build
    // that tried again (run 165) lost none of 13.
    static RETRIES: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let retries = *RETRIES.get_or_init(|| {
        std::env::var("VICTORY_DSTORAGE_RETRIES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(12)
    });
    // Since the start: writes that succeeded on a later try, and writes
    // given up.
    static RETRIED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
    let mut written = super::xmem::write_physical(&pages, offset, &data[..n]);
    let mut tries = 0u32;
    while !written && tries < retries {
        let t = std::time::Instant::now();
        std::thread::sleep(std::time::Duration::from_millis((2u64 << tries.min(6)).min(128)));
        split_add(S_RETRY, t);
        tries += 1;
        // The title may have filled the array only now.
        pages = page_numbers(array, count);
        written = super::xmem::write_physical(&pages, offset, &data[..n]);
    }
    if written {
        if tries > 0 {
            let on_a_retry = RETRIED[0].fetch_add(1, Ordering::Relaxed) + 1;
            tracing::info!(
                offset,
                len,
                tries,
                on_a_retry,
                given_up = RETRIED[1].load(Ordering::Relaxed),
                "dstorage: a destination's physical pages written after a retry"
            );
        }
        // As above, after the write that succeeded.
        let t = std::time::Instant::now();
        super::xmem::pages_written(&pages, offset, n);
        split_add(S_NOTES, t);
        Ok(())
    } else {
        let given_up = RETRIED[1].fetch_add(1, Ordering::Relaxed) + 1;
        tracing::warn!(
            known = super::xmem::pages_known(&pages, offset, len),
            tries,
            given_up,
            on_a_retry = RETRIED[0].load(Ordering::Relaxed),
            "dstorage: retries used up (VICTORY_DSTORAGE_RETRIES)"
        );
        // Said aloud: the title never reads a request's result
        // (it only polls for completion), so a destination left unwritten
        // is a model or texture that silently holds whatever its pages
        // held before.
        tracing::warn!(
            offset,
            len,
            pages = pages.len(),
            first = format_args!("{:#x}", pages.first().copied().unwrap_or(0)),
            options = format_args!("{:#x}", request.options),
            "dstorage: a destination's physical pages could not be written (unknown page or a host failure); its request fails"
        );
        Err(E_INVALIDARG)
    }
}

/// The page numbers the title's array holds now.
fn page_numbers(array: *const usize, n: usize) -> Vec<usize> {
    // SAFETY: the title passes a page number per 64 KiB of destination and
    // keeps the array until the request completes; it may write it from
    // another thread meanwhile.
    (0..n).map(|i| unsafe { array.add(i).read_volatile() }).collect()
}

/// (Storage wait job) `VICTORY_DSTORAGE_PAGES_AT_SUBMIT` (default 1; 0: as
/// before). The title's page array for a request into physical pages is its
/// own page table (`[0x1471bfb10]`, one physical page number per 64 KiB of
/// the GPU view, handed out by 0x144ffd8c0 at enqueue), which its mapping
/// executor (0x140f2c470, via 0x1455aa6a0) writes inside the batch submit
/// 0x1400e88f0, BEFORE that submit calls EnqueueStatus and Submit. So at
/// enqueue the array holds zeros or the page's last mapping, and from the
/// title's Submit on it holds this batch's pages. Reads started by that
/// Submit take the array as it is, with no wait and no comparison; reads
/// held back at a half-full queue go at that Submit too; only reads that go
/// earlier (a full queue) or find a zero still get the old checks
/// ([`filled`]: late and stale pages), counted.
fn pages_at_submit() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_DSTORAGE_PAGES_AT_SUBMIT").is_none_or(|v| v != "0");
        tracing::info!(
            "dstorage: page arrays {} (VICTORY_DSTORAGE_PAGES_AT_SUBMIT={})",
            if on { "are taken as they are by reads the title's Submit starts (its mapping step fills them before); reads at a half-full queue wait for that Submit; the late/stale checks only for reads that go earlier" } else { "are checked for late and stale pages on every read, as before" },
            u8::from(on)
        );
        on
    })
}

thread_local! {
    /// The read this worker runs was started by the title's Submit.
    static PAGES_READY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// (Storage wait job) Per second: reads whose page array was taken at once;
/// reads that waited for theirs (and the ns waited, the most); reads that went
/// through the old checks as a fallback; of the ready ones, those that still
/// found a zero; half-full queues that held reads back; full queues that let
/// them go early.
static ARRAYS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
const A_READY: usize = 0;
const A_WAITED: usize = 1;
const A_WAIT_NS: usize = 2;
const A_WAIT_MAX: usize = 3;
const A_FALLBACK: usize = 4;
const A_ZERO: usize = 5;
const A_HELD: usize = 6;
const A_FULL: usize = 7;

/// A page array's numbers as the read runs: at once when it is known filled.
fn page_array(array: *const usize, n: usize) -> Vec<usize> {
    if pages_at_submit() {
        if PAGES_READY.get() {
            let pages = page_numbers(array, n);
            if !pages.contains(&0) {
                ARRAYS[A_READY].fetch_add(1, Ordering::Relaxed);
                return pages;
            }
            ARRAYS[A_ZERO].fetch_add(1, Ordering::Relaxed);
        } else {
            ARRAYS[A_FALLBACK].fetch_add(1, Ordering::Relaxed);
        }
    }
    filled(array, n)
}

/// Counts a wait for a page array that began at `began`.
fn note_wait(began: std::time::Instant) {
    let ns = began.elapsed().as_nanos() as u64;
    ARRAYS[A_WAITED].fetch_add(1, Ordering::Relaxed);
    ARRAYS[A_WAIT_NS].fetch_add(ns, Ordering::Relaxed);
    ARRAYS[A_WAIT_MAX].fetch_max(ns, Ordering::Relaxed);
}

/// Bumped by the title's Submit and EnqueueStatus, which follow its mapping
/// step: a read waiting for its page array is woken by them.
static TITLE_MOVED: Mutex<u64> = Mutex::new(0);
static TITLE_MOVED_CV: Condvar = Condvar::new();

fn title_moved() {
    *TITLE_MOVED.lock().unwrap_or_else(|e| e.into_inner()) += 1;
    TITLE_MOVED_CV.notify_all();
}

/// Waits for the next [`title_moved`], `most` at most.
fn wait_for_title(most: std::time::Duration) {
    if most.is_zero() {
        return;
    }
    let moved = TITLE_MOVED.lock().unwrap_or_else(|e| e.into_inner());
    let now = *moved;
    let _ = TITLE_MOVED_CV.wait_timeout_while(moved, most, |m| *m == now);
}

/// The page numbers of a destination. The title fills the array after it
/// enqueued the request (a copy taken then holds zeros, or the numbers of
/// the array's last use), so it is read when the request runs; a number
/// that is still zero then is waited for, a tenth of a second at most
/// (myokex c3723f3). `VICTORY_DSTORAGE_LATE_PAGES=0`: not waited for, as
/// before.
///
/// (Neptune, `VICTORY_DSTORAGE_STALE_MS`) And an array that still holds, when the request
/// runs, the numbers it held at enqueue, which are those of its last run ([`stale_pages`]).
fn filled(array: *const usize, n: usize) -> Vec<usize> {
    static LATE: AtomicU64 = AtomicU64::new(0);
    let mut pages = page_numbers(array, n);
    stale_pages(array, n, &mut pages);
    if !pages.contains(&0) || !late_pages() {
        last_run(array, &pages);
        return pages;
    }
    let began = std::time::Instant::now();
    let most = std::time::Duration::from_millis(100);
    while pages.contains(&0) && began.elapsed() < most {
        // Woken by the title's next Submit / EnqueueStatus (its mapping
        // step fills the arrays just before), 100 ms at most.
        wait_for_title(most.saturating_sub(began.elapsed()));
        pages = page_numbers(array, n);
    }
    note_wait(began);
    let late = LATE.fetch_add(1, Ordering::Relaxed) + 1;
    if late <= 8 || late.is_power_of_two() {
        tracing::warn!(
            requests = late,
            pages = n,
            waited_ms = began.elapsed().as_millis() as u64,
            filled = !pages.contains(&0),
            "dstorage: a request ran before its page array was filled"
        );
    }
    last_run(array, &pages);
    pages
}

/// (Neptune) What the page arrays of the requests not yet run held when they were enqueued,
/// by the array's address, in the order of the requests ([`as_enqueued`]).
static ENQUEUED: Mutex<Option<HashMap<usize, VecDeque<Vec<usize>>>>> = Mutex::new(None);

/// (Neptune) What each page array held when a request last ran with it, by the array's
/// address: an array that holds that still at the next request's enqueue has not been
/// touched since, and only such a one is waited for ([`stale_pages`]). An array the title
/// filled before it enqueued holds other numbers than its last use's and is taken at once.
static LAST_RUN: Mutex<Option<HashMap<usize, Vec<usize>>>> = Mutex::new(None);

/// (Neptune) By array: how many requests in a row ran with the pages of the one before after
/// the whole wait had passed and the array stood unchanged. From [`SAME_TRUSTED`] on the
/// array is one the title reads into the same pages with every time (its buffers for what it
/// streams: half of all requests in a Neptune run, a third of the workers' time spent
/// waiting on them), and is not waited for any more. A request with other pages starts the
/// count again.
static SAME_RUNS: Mutex<Option<HashMap<usize, u32>>> = Mutex::new(None);
const SAME_TRUSTED: u32 = 3;

/// (Neptune) Since the start: requests whose page array still held, when they ran, the
/// numbers it had at enqueue (its last use's); of them, those whose array changed while the
/// request waited for it; the microseconds waited in all; and requests not waited for
/// because their array is trusted ([`SAME_RUNS`]).
static STALE: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn same_runs(array: usize, set: Option<u32>) -> u32 {
    let mut all = SAME_RUNS.lock().unwrap_or_else(|e| e.into_inner());
    let runs = all.get_or_insert_with(HashMap::new).entry(array).or_insert(0);
    match set {
        Some(0) => *runs = 0,
        Some(_) => *runs = runs.saturating_add(1),
        None => {}
    }
    *runs
}

/// (Neptune) Notes what the page array of `request` holds as it is enqueued.
///
/// The title fills the array after it has enqueued the request. Until it has, the array
/// holds zeros, which [`filled`] waits for, or the numbers of its last use, which nothing
/// told apart from this request's own: the request then wrote its bytes into the pages of
/// the texture or model read there before, and the pages it was meant for kept what they
/// held (a texture of coloured noise on one character's arms, another's face).
/// `VICTORY_DSTORAGE_STALE_MS=0`: not noted, as before.
fn as_enqueued(request: &Request) {
    if request.options & PAGE_DESTINATION == 0 || request.destination.is_null() || stale_wait().is_zero() {
        return;
    }
    let offset = (request.page_offset & 0xffff) as usize;
    let n = (offset + request.destination_size as usize).div_ceil(PAGE);
    let now = page_numbers(request.destination.cast::<usize>().cast_const(), n);
    let mut all = ENQUEUED.lock().unwrap_or_else(|e| e.into_inner());
    all.get_or_insert_with(HashMap::new).entry(request.destination as usize).or_default().push_back(now);
}

/// How long a request waits for a page array that still holds what it held at enqueue
/// (`VICTORY_DSTORAGE_STALE_MS`, 10; 0: not waited for, as before).
fn stale_wait() -> std::time::Duration {
    static MS: OnceLock<u64> = OnceLock::new();
    std::time::Duration::from_millis(*MS.get_or_init(|| {
        // 10 (Neptune): of the 127 arrays found stale in their runs, 109 were filled within
        // 2 ms, 7 more within 4 and the last 11 within 6.
        let ms = std::env::var("VICTORY_DSTORAGE_STALE_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(10);
        tracing::info!(
            "dstorage: a page array that still holds its last request's pages when the next runs is waited for {ms} ms at most (VICTORY_DSTORAGE_STALE_MS={ms}; trusted after {SAME_TRUSTED} runs in a row with the same pages)"
        );
        ms
    }))
}

/// (Neptune) For [`filled`]: `pages`, read from `array` as the request runs, are what the
/// array held at enqueue and at its last run, and not zeros: the title has not filled it
/// for this request yet (or asks for the same pages again, which a short wait does not
/// harm). Waited for [`stale_wait`] at most, unless the array is trusted
/// ([`SAME_TRUSTED`]); `pages` is read again meanwhile.
fn stale_pages(array: *const usize, n: usize, pages: &mut Vec<usize>) {
    if stale_wait().is_zero() {
        return;
    }
    let before = {
        let mut all = ENQUEUED.lock().unwrap_or_else(|e| e.into_inner());
        all.as_mut().and_then(|all| {
            let line = all.get_mut(&(array as usize))?;
            let before = line.pop_front();
            if line.is_empty() {
                all.remove(&(array as usize));
            }
            before
        })
    };
    let untouched = {
        let last = LAST_RUN.lock().unwrap_or_else(|e| e.into_inner());
        last.as_ref()
            .and_then(|last| last.get(&(array as usize)))
            .is_some_and(|last| last.len() >= n && last[..n] == pages[..])
    };
    if !untouched {
        same_runs(array as usize, Some(0));
    }
    let Some(before) = before else {
        return;
    };
    if !untouched || !before.iter().any(|&p| p != 0) || pages.contains(&0) || before.len() < n || before[..n] != pages[..] {
        return;
    }
    if same_runs(array as usize, None) >= SAME_TRUSTED {
        STALE[3].fetch_add(1, Ordering::Relaxed);
        return;
    }
    STALE[0].fetch_add(1, Ordering::Relaxed);
    let began = std::time::Instant::now();
    let most = stale_wait();
    while before[..n] == pages[..] && began.elapsed() < most {
        // Event-driven as in [`filled`].
        wait_for_title(most.saturating_sub(began.elapsed()));
        *pages = page_numbers(array, n);
    }
    note_wait(began);
    STALE[2].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
    if before[..n] != pages[..] {
        STALE[1].fetch_add(1, Ordering::Relaxed);
        same_runs(array as usize, Some(0));
    } else {
        same_runs(array as usize, Some(1));
    }
}

/// (Neptune) What `array` held as its request ran, for the next request's [`stale_pages`].
fn last_run(array: *const usize, pages: &[usize]) {
    if stale_wait().is_zero() {
        return;
    }
    let mut last = LAST_RUN.lock().unwrap_or_else(|e| e.into_inner());
    last.get_or_insert_with(HashMap::new).insert(array as usize, pages.to_vec());
}

/// Whether [`filled`] waits for page numbers the title writes after the
/// request (`VICTORY_DSTORAGE_LATE_PAGES`, on unless `0`).
fn late_pages() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_DSTORAGE_LATE_PAGES").is_none_or(|v| v != "0");
        tracing::info!(
            "dstorage: page arrays filled after the request {} (VICTORY_DSTORAGE_LATE_PAGES={})",
            if on { "are waited for, 100 ms at most" } else { "are not waited for" },
            if on { "1, the default" } else { "0" }
        );
        on
    })
}

/// Slots by the 2020 order, which the argument shapes at
/// the title's call sites match (0x18, 0x20, 0x30 and 0x40 are called).
static QUEUE_VTABLE: Methods<12> = Methods([
    missing::<0> as *const (),
    add_ref_instance::<Queue> as *const (),
    release::<Queue> as *const (),
    enqueue_request as *const (), // 0x18
    enqueue_status as *const (),  // 0x20
    missing::<5> as *const (),    // 0x28 (2020: fence signal)
    submit as *const (),          // 0x30
    missing::<7> as *const (),    // 0x38 (2020: cancel by tag)
    close as *const (),           // 0x40
    missing::<9> as *const (),    // 0x48 (2020: error event)
    missing::<10> as *const (),   // 0x50 (2020: error record)
    missing::<11> as *const (),   // 0x58 (2020: queue info)
]);

/// (Run-46) The workers' time since the last line, ns: requests run in all,
/// reading files, inflating to the intermediate (zlib before BCPack),
/// waiting for the page array (late or stale), first touch of the
/// destination's pages (their soft faults), filling the destination
/// (decode or copy), telling the GPU side (`kernel_will_write`,
/// `title_memory_written`, `pages_written`); and requests.
static SPLIT: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
const S_ALL: usize = 0;
const S_READ: usize = 1;
const S_INFLATE: usize = 2;
const S_PAGES: usize = 3;
const S_TOUCH: usize = 4;
const S_FILL: usize = 5;
const S_NOTES: usize = 6;
const S_REQUESTS: usize = 7;
/// (Storage wait job) Compressed requests' source reads (outside the fill),
/// buffers made for scattered destinations, and retry sleeps: what "other"
/// held before.
const S_SRC: usize = 8;
const S_BUF: usize = 9;
const S_RETRY: usize = 10;

fn split_add(what: usize, since: std::time::Instant) {
    SPLIT[what].fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// The process's page faults (all kinds) so far.
#[cfg(windows)]
fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS { cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32, ..Default::default() };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// The workers' split for the "frame thread streaming" line.
fn split_line() -> String {
    let c: Vec<u64> = SPLIT.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect();
    #[cfg(windows)]
    let (lock, view, unmap, lookup, copy, note_lookup) = {
        use super::xmem::{W_COPY, W_LOCK, W_LOOKUP, W_NOTE_LOOKUP, W_UNMAP, W_VIEW, WORKER_NS};
        let t = |i: usize| WORKER_NS[i].swap(0, Ordering::Relaxed);
        (t(W_LOCK), t(W_VIEW), t(W_UNMAP), t(W_LOOKUP), t(W_COPY), t(W_NOTE_LOOKUP))
    };
    #[cfg(not(windows))]
    let (lock, view, unmap, lookup, copy, note_lookup) = (0, 0, 0, 0, 0, 0);
    #[cfg(windows)]
    let faults = {
        static LAST: AtomicU64 = AtomicU64::new(0);
        let now = page_faults();
        now.saturating_sub(LAST.swap(now, Ordering::Relaxed))
    };
    #[cfg(not(windows))]
    let faults = 0;
    let ms = |ns: u64| ns as f64 / 1e6;
    let named = c[S_INFLATE]
        + c[S_PAGES]
        + c[S_TOUCH]
        + c[S_FILL]
        + c[S_NOTES]
        + c[S_SRC]
        + c[S_BUF]
        + c[S_RETRY]
        + lock
        + view
        + unmap
        + lookup
        + copy;
    let a: Vec<u64> = ARRAYS.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect();
    let mean = if a[A_WAITED] == 0 { 0.0 } else { ms(a[A_WAIT_NS]) / a[A_WAITED] as f64 };
    format!(
        "; dstorage workers {} requests {:.1} ms: file read {:.1}, inflate to intermediate {:.1}, page-array wait {:.1}, page-table lock wait {:.1}, own view map+commit {:.1}, first touch (soft faults) {:.1}, fill (decode/copy; raw reads: their file read too) {:.1}, own view unmap {:.1}, notes to the GPU side {:.1} (of which page-table lookup {:.1}), compressed source read {:.1}, page lookup {:.1}, scattered-page buffer {:.1} + copy {:.1}, write retries {:.1}, other {:.1} ms (VICTORY_DSTORAGE_LINEAR_VIEW); page arrays (VICTORY_DSTORAGE_PAGES_AT_SUBMIT={}): ready at start {}, waited {} (mean {mean:.2} ms, max {:.2} ms), fallback stale checks {}, zero after submit {}, held at a half-full queue {}, let go by a full queue {}; process page faults {faults}",
        c[S_REQUESTS],
        ms(c[S_ALL]),
        ms(c[S_READ]),
        ms(c[S_INFLATE]),
        ms(c[S_PAGES]),
        ms(lock),
        ms(view),
        ms(c[S_TOUCH]),
        ms(c[S_FILL]),
        ms(unmap),
        ms(c[S_NOTES]),
        ms(note_lookup),
        ms(c[S_SRC]),
        ms(lookup),
        ms(c[S_BUF]),
        ms(copy),
        ms(c[S_RETRY]),
        ms(c[S_ALL].saturating_sub(named)),
        u8::from(pages_at_submit()),
        a[A_READY],
        a[A_WAITED],
        ms(a[A_WAIT_MAX]),
        a[A_FALLBACK],
        a[A_ZERO],
        a[A_HELD],
        a[A_FULL],
    )
}

/// Reads one byte of each 4 KiB page of `bytes`: their first-touch faults
/// taken (and timed) apart from the decode.
fn touch(bytes: &mut [u8]) {
    let t = std::time::Instant::now();
    for k in (0..bytes.len()).step_by(4096) {
        // SAFETY: inside the slice.
        unsafe { std::ptr::read_volatile(bytes.as_ptr().add(k)) };
    }
    split_add(S_TOUCH, t);
}

pub(crate) fn module() -> HleModule {
    victory_gpu::framestream::add_to_line(split_line);
    // (Storage wait job) Page faults by the layer's kinds of work, per second.
    #[cfg(windows)]
    victory_gpu::framestream::add_to_line(victory_gpu::faults::second_line);
    HleModule {
        name: "dstorage",
        dlls: &["xcrdapi.dll"],
        exports: vec![hle_export!(DStorageGetFactory)],
    }
}

#[cfg(test)]
mod tests {
    use std::os::windows::ffi::OsStrExt;
    use std::time::{Duration, Instant};

    use super::*;
    use crate::modules::xgameruntime::method;

    fn call<F: Copy>(object: *mut c_void, slot: usize) -> F {
        // SAFETY: every object here starts with the runtime header.
        method(unsafe { &*object.cast::<Object>() }, slot)
    }

    fn factory() -> *mut c_void {
        let mut factory = std::ptr::null_mut();
        assert_eq!(DStorageGetFactory(&FACTORY, &mut factory), S_OK);
        factory
    }

    fn open(path: &std::path::Path) -> *mut c_void {
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
        let open: extern "system" fn(*mut c_void, *const u16, *const Guid, Out) -> HRESULT =
            call(factory(), 0x20);
        let mut file = std::ptr::null_mut();
        assert_eq!(
            open(factory(), wide.as_ptr(), std::ptr::null(), &mut file),
            S_OK
        );
        file
    }

    fn status_array(capacity: u32) -> *mut c_void {
        let create: extern "system" fn(
            *mut c_void,
            u32,
            *const c_char,
            *const Guid,
            Out,
        ) -> HRESULT = call(factory(), 0x28);
        let mut array = std::ptr::null_mut();
        let name = c"test".as_ptr();
        assert_eq!(
            create(factory(), capacity, name, std::ptr::null(), &mut array),
            S_OK
        );
        array
    }

    fn new_queue(source_type: u64) -> *mut c_void {
        let create: extern "system" fn(*mut c_void, *const QueueDesc, *const Guid, Out) -> HRESULT =
            call(factory(), 0x50);
        let desc = QueueDesc {
            source_type,
            capacity: 0x80,
            priority: 0,
            name: c"test".as_ptr(),
            unknown: 1,
        };
        let mut queue = std::ptr::null_mut();
        assert_eq!(create(factory(), &desc, std::ptr::null(), &mut queue), S_OK);
        queue
    }

    fn request(
        options: u64,
        destination: *mut c_void,
        destination_size: usize,
        source: *const c_void,
        offset: u64,
        source_size: usize,
    ) -> Request {
        Request {
            options,
            destination,
            page_offset: 0,
            destination_size: destination_size as u32,
            source,
            offset,
            source_size: source_size as u32,
            intermediate_size: 0,
            tag: 0,
            name: std::ptr::null(),
        }
    }

    fn enqueue(queue: *mut c_void, request: &Request) {
        let f: extern "system" fn(*mut c_void, *const Request) = call(queue, 0x18);
        f(queue, request);
    }

    fn enqueue_status(queue: *mut c_void, array: *mut c_void, index: u32) {
        let f: extern "system" fn(*mut c_void, *mut c_void, u32) = call(queue, 0x20);
        f(queue, array, index);
    }

    fn void(object: *mut c_void, slot: usize) {
        let f: extern "system" fn(*mut c_void) = call(object, slot);
        f(object);
    }

    fn result(array: *mut c_void, index: u32) -> HRESULT {
        let f: extern "system" fn(*mut c_void, u32) -> HRESULT = call(array, 0x20);
        f(array, index)
    }

    /// Polls as the title does, then returns the entry's result.
    fn wait(array: *mut c_void, index: u32) -> HRESULT {
        let done: extern "system" fn(*mut c_void, u32) -> bool = call(array, 0x18);
        let start = Instant::now();
        while !done(array, index) {
            assert!(start.elapsed() < Duration::from_secs(10), "never completed");
            std::thread::sleep(Duration::from_millis(1));
        }
        result(array, index)
    }

    fn file_with(bytes: &[u8]) -> (tempfile::TempDir, *mut c_void) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        std::fs::write(&path, bytes).unwrap();
        let file = open(&path);
        (dir, file)
    }

    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 7 + i / 251) as u8).collect()
    }

    #[test]
    fn a_page_array_filled_late_is_waited_for() {
        let array: &'static [std::sync::atomic::AtomicUsize] = Box::leak(Box::new([std::sync::atomic::AtomicUsize::new(7), std::sync::atomic::AtomicUsize::new(0)]));
        let title = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            array[1].store(8, Ordering::Release);
        });
        assert_eq!(filled(array.as_ptr().cast(), 2), [7, 8]);
        title.join().unwrap();
        // One never filled is given up on.
        array[0].store(0, Ordering::Release);
        assert_eq!(filled(array.as_ptr().cast(), 2), [0, 8]);
    }

    /// (Neptune, `VICTORY_DSTORAGE_STALE_MS`) An array that holds its last run's pages at
    /// enqueue and still when the request runs is waited for until the title fills it; one
    /// that ran with the same pages after the whole wait three times in a row is trusted.
    #[test]
    fn a_page_array_still_holding_its_last_pages_is_waited_for() {
        let array: &'static [std::sync::atomic::AtomicUsize] =
            Box::leak(Box::new([std::sync::atomic::AtomicUsize::new(5), std::sync::atomic::AtomicUsize::new(6)]));
        let at = array.as_ptr().cast::<usize>();
        let enqueue = || {
            let mut all = ENQUEUED.lock().unwrap();
            all.get_or_insert_with(HashMap::new).entry(at as usize).or_default().push_back(page_numbers(at, 2));
        };
        // A first run: nothing to compare with.
        enqueue();
        assert_eq!(filled(at, 2), [5, 6]);
        // Enqueued with the old pages, filled by the title a moment later.
        enqueue();
        let title = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(3));
            array[0].store(9, Ordering::Release);
            array[1].store(10, Ordering::Release);
        });
        assert_eq!(filled(at, 2), [9, 10]);
        title.join().unwrap();
        // The same pages every time: waited for in full three times, then trusted.
        let trusted = STALE[3].load(Ordering::Relaxed);
        for _ in 0..SAME_TRUSTED {
            enqueue();
            assert_eq!(filled(at, 2), [9, 10]);
        }
        let began = std::time::Instant::now();
        enqueue();
        assert_eq!(filled(at, 2), [9, 10]);
        assert!(began.elapsed() < stale_wait());
        assert_eq!(STALE[3].load(Ordering::Relaxed), trusted + 1);
    }

    #[test]
    fn request_layout_is_the_titles() {
        assert_eq!(size_of::<Request>(), 0x48);
        assert_eq!(std::mem::offset_of!(Request, destination_size), 0x18);
        assert_eq!(std::mem::offset_of!(Request, source), 0x20);
        assert_eq!(std::mem::offset_of!(Request, source_size), 0x30);
        assert_eq!(std::mem::offset_of!(Request, name), 0x40);
        assert_eq!(std::mem::offset_of!(QueueDesc, unknown), 0x18);
    }

    fn compress_zlib(plain: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; zlib_rs::compress_bound(plain.len())];
        let (packed, rc) =
            zlib_rs::compress_slice(&mut buf, plain, zlib_rs::DeflateConfig::new(6));
        assert_eq!(rc, zlib_rs::ReturnCode::Ok);
        packed.to_vec()
    }

    #[test]
    fn raw_and_zlib_reads_from_a_file() {
        let plain = pattern(300_000);
        let packed = compress_zlib(&plain);
        let mut contents = plain.clone();
        contents.extend_from_slice(&packed);
        let (_dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(1));

        let mut raw = vec![0u8; 1000];
        let mut inflated = vec![0u8; plain.len()];
        let at = |b: &mut Vec<u8>| b.as_mut_ptr().cast();
        enqueue(queue, &request(0, at(&mut raw), 1000, file, 5000, 1000));
        let offset = plain.len() as u64;
        let zlib = request(
            ZLIB,
            at(&mut inflated),
            plain.len(),
            file,
            offset,
            packed.len(),
        );
        enqueue(queue, &zlib);
        enqueue_status(queue, array, 0);
        assert_eq!(result(array, 0), E_PENDING, "pending until done");
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(raw, plain[5000..6000]);
        assert_eq!(inflated, plain);
        void(queue, 0x40);
    }

    #[test]
    fn memory_sources_inflate_too() {
        let plain = pattern(70_000);
        let packed = compress_zlib(&plain);
        let (queue, array) = (new_queue(1), status_array(1));
        let mut out = vec![0u8; plain.len()];
        let options = ZLIB | MEMORY_SOURCE;
        let r = request(
            options,
            out.as_mut_ptr().cast(),
            out.len(),
            packed.as_ptr().cast(),
            0,
            packed.len(),
        );
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, plain);
    }

    #[test]
    fn statuses_complete_in_order_with_the_first_failure() {
        let contents = pattern(4096);
        let (_dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(3));
        // Requests in flight together run in any order, so none of them
        // shares a destination (failed ones leave theirs undefined).
        let mut buffers = [[0u8; 16]; 5];
        let [failed, failed_too, out, other, bcpack] = buffers.each_mut().map(|b| b.as_mut_ptr().cast());

        // Segment 0: a read past the end, then a bad stream; the first wins.
        enqueue(queue, &request(0, failed, 16, file, 4090, 16));
        enqueue(queue, &request(ZLIB, failed_too, 16, file, 0, 16));
        enqueue_status(queue, array, 0);
        // Segment 1: two reads.
        enqueue(queue, &request(0, other, 16, file, 0, 16));
        enqueue(queue, &request(0, out, 16, file, 100, 16));
        enqueue_status(queue, array, 1);
        void(queue, 0x30);
        // Segment 2: a BCPack request (BC7) whose stream is empty fails.
        with_test_tables();
        enqueue(queue, &request(7 << 1, bcpack, 16, file, 0, 0));
        enqueue_status(queue, array, 2);
        void(queue, 0x30);

        assert_eq!(wait(array, 2), INVALID_DATA);
        assert_eq!(result(array, 0), END_OF_FILE);
        assert_eq!(result(array, 1), S_OK);
        assert_eq!(buffers[2], contents[100..116]);
        assert_eq!(buffers[3], contents[..16]);
    }

    #[test]
    fn half_a_queue_submits_itself() {
        let (_dir, file) = file_with(&pattern(64));
        let create: extern "system" fn(*mut c_void, *const QueueDesc, *const Guid, Out) -> HRESULT =
            call(factory(), 0x50);
        let desc = QueueDesc {
            source_type: 0,
            capacity: 4,
            priority: 1,
            name: c"small".as_ptr(),
            unknown: 1,
        };
        let mut queue = std::ptr::null_mut();
        assert_eq!(create(factory(), &desc, std::ptr::null(), &mut queue), S_OK);
        let array = status_array(1);
        let mut out = [0u8; 16];
        enqueue(queue, &request(0, out.as_mut_ptr().cast(), 16, file, 8, 16));
        enqueue_status(queue, array, 0); // the second of four: no submit needed
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, pattern(64)[8..24]);
    }

    /// Lanes without workers: what `start` picks, and when statuses land.
    fn lanes(staging: u64) -> Lanes {
        Lanes {
            lanes: HashMap::new(),
            next_id: 0,
            next_seq: 0,
            staged: 0,
            staging,
            starved: false,
            clock: [0; 3],
            now: 0,
            lane_now: [0; 3],
            done: Default::default(),
            refused: 0,
            bcpack: Default::default(),
            since: None,
        }
    }

    fn read(lanes: &mut Lanes, id: u64, options: u64, size: u32) {
        let mut r = request(options, std::ptr::null_mut(), 0, std::ptr::null(), 0, 0);
        (r.destination_size, r.source_size) = (size, size);
        let lane = lanes.lanes.get_mut(&id).unwrap();
        lane.unsubmitted.push(Command::Read(r));
    }

    #[test]
    fn priorities_share_by_bytes_about_tenfold() {
        let mut l = lanes(STAGING);
        l.lanes.insert(0, Lane::new(0, 4096));
        l.lanes.insert(1, Lane::new(1, 4096));
        l.lanes.insert(2, Lane::new(2, 4096)); // real-time: as high
        for _ in 0..400 {
            for id in 0..3 {
                read(&mut l, id, 0, 4096);
            }
        }
        (0..3).for_each(|id| l.submit(id));
        let mut started = [0u32; 3];
        for _ in 0..330 {
            // (As before `VICTORY_DSTORAGE_OLDEST_FIRST`: round robin by bytes
            // among the lanes of a class.)
            let job = l.start_as(0).unwrap();
            started[job.lane as usize] += 1;
            l.finish(&job, Ok(()));
        }
        // Normal gets about a tenth of what high and real-time get together.
        assert_eq!(started[1] + started[2], 300, "{started:?}");
        assert_eq!(started[0], 30, "{started:?}");
        assert!(started[1].abs_diff(started[2]) <= 1, "{started:?}");
    }

    /// (F10, `VICTORY_DSTORAGE_OLDEST_FIRST`) The reads of the oldest batch
    /// go first, whichever lane holds it and however many bytes each lane
    /// has started; within a batch in its order; classes still share by
    /// bytes (mode 1), or not (mode 2).
    #[test]
    fn the_oldest_batch_is_read_first() {
        let mut l = lanes(STAGING);
        l.lanes.insert(0, Lane::new(0, 4096));
        l.lanes.insert(1, Lane::new(0, 4096));
        // Batch 1 on lane 1 (big reads), batch 2 on lane 0, batch 3 on lane 1.
        for _ in 0..3 {
            read(&mut l, 1, 0, 1 << 20);
        }
        l.submit(1);
        for _ in 0..3 {
            read(&mut l, 0, 0, 4096);
        }
        l.submit(0);
        for _ in 0..3 {
            read(&mut l, 1, 0, 4096);
        }
        l.submit(1);
        let mut order = Vec::new();
        while let Some(job) = l.start_as(1) {
            order.push((job.lane, job.seq));
            l.finish(&job, Ok(()));
        }
        assert_eq!(order.iter().map(|o| o.1).collect::<Vec<_>>(), (0..9).collect::<Vec<_>>(), "{order:?}");
        assert_eq!(order.iter().map(|o| o.0).collect::<Vec<_>>(), [1, 1, 1, 0, 0, 0, 1, 1, 1]);
        // Before: lane 0 had started fewer bytes and went ahead of batch 1.
        let mut l2 = lanes(STAGING);
        l2.lanes.insert(0, Lane::new(0, 4096));
        l2.lanes.insert(1, Lane::new(0, 4096));
        read(&mut l2, 1, 0, 1 << 20);
        read(&mut l2, 1, 0, 1 << 20);
        l2.submit(1);
        read(&mut l2, 0, 0, 4096);
        l2.submit(0);
        let first = l2.start_as(0).unwrap();
        l2.finish(&first, Ok(()));
        assert_eq!(l2.start_as(0).unwrap().lane, 0, "round robin by bytes");
        // Mode 2: over classes too.
        let mut l3 = lanes(STAGING);
        l3.lanes.insert(0, Lane::new(-1, 4096));
        l3.lanes.insert(1, Lane::new(1, 4096));
        read(&mut l3, 0, 0, 4096);
        l3.submit(0);
        read(&mut l3, 1, 0, 4096);
        l3.submit(1);
        assert_eq!(l3.start_as(2).unwrap().lane, 0, "the low-priority batch is older");
    }

    #[test]
    fn staging_bounds_compressed_bytes_in_flight() {
        let mut l = lanes(100);
        l.lanes.insert(0, Lane::new(0, 64));
        read(&mut l, 0, ZLIB, 60);
        read(&mut l, 0, ZLIB, 60);
        read(&mut l, 0, 0, 1000); // raw: straight to its destination
        l.submit(0);
        let first = l.start().unwrap();
        assert!(l.start().is_none(), "60 + 60 > 100");
        l.finish(&first, Ok(()));
        let second = l.start().unwrap();
        let raw = l.start().unwrap();
        assert_eq!((second.staging, raw.staging), (60, 0));
        // Alone, a request larger than the buffer still runs.
        read(&mut l, 0, ZLIB, 500);
        l.submit(0);
        assert!(l.start().is_none());
        l.finish(&second, Ok(()));
        assert!(l.start().is_some());
    }

    #[test]
    fn a_status_waits_for_every_earlier_read() {
        let mut l = lanes(STAGING);
        l.lanes.insert(0, Lane::new(0, 64));
        let array = status_array(2).cast::<Instance<StatusArray>>();
        let status = |index| {
            add_ref_instance(array);
            Command::Status(Status { array, index })
        };
        read(&mut l, 0, 0, 16);
        read(&mut l, 0, 0, 16);
        l.lanes.get_mut(&0).unwrap().unsubmitted.push(status(0));
        read(&mut l, 0, 0, 16);
        l.lanes.get_mut(&0).unwrap().unsubmitted.push(status(1));
        // SAFETY: an array we created.
        for entry in unsafe { &(*array).state.entries }.iter() {
            entry.store(E_PENDING, Ordering::Release);
        }
        l.submit(0);
        let jobs: Vec<Job> = (0..3).map(|_| l.start().unwrap()).collect();
        // The last read and the first finish; the second then fails.
        l.finish(&jobs[2], Ok(()));
        l.finish(&jobs[0], Ok(()));
        let results = || (result(array.cast(), 0), result(array.cast(), 1));
        assert_eq!(results(), (E_PENDING, E_PENDING));
        l.finish(&jobs[1], Err(INVALID_DATA));
        assert_eq!(results(), (INVALID_DATA, S_OK));
    }

    #[test]
    fn zlib_then_bcpack() {
        with_test_tables();
        let packed = compress_zlib(&BC4_STREAM);
        let (_dir, file) = file_with(&packed);
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = [0u8; 16];
        let options = ZLIB | 4 << 1;
        let mut r = request(options, out.as_mut_ptr().cast(), 16, file, 0, packed.len());
        r.intermediate_size = BC4_STREAM.len() as u32;
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, bc4_blocks());
    }

    /// BCPack tables we made (uniform models; BC4 index pairs: low 3 bits,
    /// next 3 bits), in a folder kept for the test process.
    fn with_test_tables() {
        static DIR: OnceLock<()> = OnceLock::new();
        DIR.get_or_init(|| {
            let mut t = Vec::new();
            for m in 0..128 {
                let g = m / 16;
                for b in 0..9 {
                    let f: u16 = if b < g + 2 { 1 << (15 - g) } else { 0 };
                    t.extend_from_slice(&f.to_le_bytes());
                }
            }
            t.extend([0; 256 + 64 + 256]);
            t.extend((0..64u8).map(|v| (v & 7) | (v >> 3) << 4));
            t.resize(bcpack::TABLES_SIZE, 0);
            let dir = tempfile::tempdir().unwrap().keep();
            std::fs::write(dir.join("tables.bin"), t).unwrap();
            use_bcpack_tables(dir);
        });
    }

    /// Two BC4 blocks from the format's test encoder: endpoints 10, -3 with
    /// indices p -> p & 7, then the same block plus 1 everywhere.
    const BC4_STREAM: [u8; 88] = [
        0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 4, 4, 4, 4, 1, 1, 1, 1, 0x10, 0x10, 0x10, 0x10, 2, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0x14, 5, 0x20, 0x68, 0xb0, 0xf8, 3, 3, 2, 2, 0x24, 0x24, 0x24,
        0x24, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0x10, 0x10, 0x10, 0x10, 0, 0, 0, 0,
        0x20, 0x68, 0xb0, 0xf8, 0x24, 0x24, 0x24, 0x24, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    /// What `BC4_STREAM` unpacks to.
    fn bc4_blocks() -> [u8; 16] {
        let code = [0u64, 2, 3, 4, 5, 6, 7, 1];
        let block = |e0: u64, e1: u64, add: usize| {
            let bits = (0..16).fold(0, |a, p| a | code[(p + add) & 7] << (16 + 3 * p));
            (e0 | e1 << 8 | bits).to_le_bytes()
        };
        let mut out = [0; 16];
        out[..8].copy_from_slice(&block(138, 125, 0));
        out[8..].copy_from_slice(&block(139, 126, 1));
        out
    }

    #[test]
    fn bcpack_requests_unpack_into_the_destination() {
        with_test_tables();
        let mut contents = pattern(100);
        contents.extend_from_slice(&BC4_STREAM);
        let (_dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = [0u8; 16];
        enqueue(queue, &request(4 << 1, out.as_mut_ptr().cast(), 16, file, 100, 88));
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, bc4_blocks());
    }

    #[test]
    fn a_closed_queue_completes_nothing() {
        let (_dir, file) = file_with(&pattern(64));
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = [0u8; 16];
        enqueue(queue, &request(0, out.as_mut_ptr().cast(), 16, file, 0, 16));
        enqueue_status(queue, array, 0);
        void(queue, 0x40);
        void(queue, 0x30);
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(result(array, 0), E_PENDING);
        assert_eq!(out, [0; 16]);
    }

    #[test]
    fn physical_page_destinations() {
        use crate::modules::xmem;
        let mut pages = [0usize; 2];
        let mut n = 2;
        assert_eq!(
            xmem::XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()),
            1
        );
        // Where the title would see them: a reservation of 64 KiB pages.
        let mapped = xmem::XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            0x2000 | 0x2040_0000, // reserve, 64 KiB pages
            0,
            0x04,
        );
        assert!(!mapped.is_null());
        assert_eq!(
            xmem::XMemMapPhysicalPages(mapped, 2, pages.as_ptr()),
            mapped
        );

        let plain = pattern(100);
        let (queue, array) = (new_queue(1), status_array(1));
        let mut r = request(
            PAGE_DESTINATION | MEMORY_SOURCE,
            pages.as_mut_ptr().cast(),
            plain.len(),
            plain.as_ptr().cast(),
            0,
            plain.len(),
        );
        r.page_offset = (PAGE - 40) as u64; // across the page boundary
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        // SAFETY: both pages are mapped read-write.
        let seen = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>().add(PAGE - 40), 100) };
        assert_eq!(seen, plain);
    }
    #[test]
    fn zlib_into_pages_in_place_and_across_allocations() {
        use crate::modules::xmem;
        let allocate = |n: usize| {
            let (mut pages, mut given) = (vec![0usize; n], n);
            assert_eq!(
                xmem::XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
                1
            );
            pages
        };
        let (a, b) = (allocate(3), allocate(1));
        let plain = pattern(2 * PAGE + 100);
        let packed = compress_zlib(&plain);
        // One allocation's pages in order (one window), then pages of two
        // (through a buffer), both from 50 bytes into the first.
        let mut destinations = [a.clone(), vec![a[0], a[1], b[0]]];
        let (queue, array) = (new_queue(1), status_array(1));
        for pages in &mut destinations {
            let mut r = request(
                ZLIB | PAGE_DESTINATION | MEMORY_SOURCE,
                pages.as_mut_ptr().cast(),
                plain.len(),
                packed.as_ptr().cast(),
                0,
                packed.len(),
            );
            r.page_offset = 50;
            enqueue(queue, &r);
        }
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        for pages in &destinations {
            let mapped = xmem::XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                0x2000 | 0x2040_0000, // reserve, 64 KiB pages
                0,
                0x04,
            );
            assert!(!mapped.is_null());
            assert_eq!(
                xmem::XMemMapPhysicalPages(mapped, 3, pages.as_ptr()),
                mapped
            );
            // SAFETY: all three pages are mapped read-write.
            let seen = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), 3 * PAGE) };
            assert!(seen[..50].iter().all(|&b| b == 0));
            assert_eq!(seen[50..50 + plain.len()], plain);
            assert_eq!(
                xmem::XMemMapPhysicalPages(mapped, 3, std::ptr::null()),
                mapped
            );
        }
        assert_eq!(xmem::XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(xmem::XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn zlib_inflate_handles_short_exact_empty_and_bad_streams() {
        let plain = pattern(65_537);
        let packed = compress_zlib(&plain);

        // Exact fit.
        let mut exact = vec![0u8; plain.len()];
        assert_eq!(inflate(&packed, &mut exact), Ok(plain.len()));
        assert_eq!(exact, plain);

        // Short stream into a larger buffer passes and returns bytes written;
        // guard bytes outside the slice passed to `inflate` stay untouched.
        let mut larger = vec![0xAA; plain.len() + 64];
        assert_eq!(
            inflate(&packed, &mut larger[..plain.len() + 32]),
            Ok(plain.len())
        );
        assert_eq!(&larger[..plain.len()], &plain[..]);
        assert!(larger[plain.len() + 32..].iter().all(|&b| b == 0xAA));

        // Empty valid stream passes for empty or non-empty destinations.
        let empty = compress_zlib(&[]);
        assert_eq!(inflate(&empty, &mut []), Ok(0));
        let mut scratch = [0x55u8; 4];
        assert_eq!(inflate(&empty, &mut scratch), Ok(0));

        // Oversized stream (destination too small, including empty) fails.
        let mut small = vec![0u8; plain.len() - 1];
        assert_eq!(inflate(&packed, &mut small), Err(INVALID_DATA));
        assert_eq!(inflate(&packed, &mut []), Err(INVALID_DATA));

        // Truncated stream (missing trailer or payload bytes) fails.
        let mut out = vec![0u8; plain.len()];
        assert_eq!(
            inflate(&packed[..packed.len() - 1], &mut out),
            Err(INVALID_DATA)
        );
        assert_eq!(
            inflate(&packed[..packed.len() / 2], &mut out),
            Err(INVALID_DATA)
        );
        assert_eq!(inflate(&[], &mut out), Err(INVALID_DATA));

        // Corrupt Adler-32 trailer fails.
        let mut bad_adler = packed.clone();
        *bad_adler.last_mut().unwrap() ^= 1;
        assert_eq!(inflate(&bad_adler, &mut out), Err(INVALID_DATA));

        // Corrupt header/payload fails, and subsequent valid inflate on the
        // same thread still succeeds (verifies thread-local state reset).
        let mut bad_body = packed.clone();
        bad_body[2] ^= 0xFF;
        assert_eq!(inflate(&bad_body, &mut out), Err(INVALID_DATA));
        assert_eq!(inflate(&packed, &mut out), Ok(plain.len()));
        assert_eq!(out, plain);
    }
}
