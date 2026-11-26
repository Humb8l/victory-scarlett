//! (Storage wait job, task 2) Where the process's page faults come from.
//!
//! The process counts its page faults (soft and hard together) only as a
//! whole, never by thread, so each of the layer's own kinds of work reads
//! that count before and after ([`scope`]) and is charged the difference:
//! mapping calls (both views), copies of title memory brought up to date,
//! hashing of title memory, texture refreshes (the source of uploads),
//! DirectStorage requests, write-watch askings that reset. Scopes on one
//! thread nest: an inner one's faults are not charged to the outer one too.
//! Faults other threads take while a scope runs are charged to it as well,
//! so on a busy frame the kinds can sum to more than the process took;
//! "elsewhere" (title code and the rest) is what the process took past the
//! sum, 0 when they overlap. Hard faults come from the system's process
//! list (`NtQuerySystemInformation`, `HardFaultCount`, readable without
//! admin), read by a thread of its own every 250 ms (a list of every
//! process is too slow to read per frame).
//! `VICTORY_FAULT_METER=0`: off (default on in diagnostics builds, off
//! otherwise; 1 forces it on).

use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::framelog::T;

/// The kinds a fault is charged to.
#[derive(Clone, Copy)]
pub enum Kind {
    Map,
    Copy,
    Hash,
    Upload,
    Disk,
    Watch,
}

impl Kind {
    fn total(self) -> T {
        match self {
            Kind::Map => T::FaultsMap,
            Kind::Copy => T::FaultsCopy,
            Kind::Hash => T::FaultsHash,
            Kind::Upload => T::FaultsUpload,
            Kind::Disk => T::FaultsDisk,
            Kind::Watch => T::FaultsWatch,
        }
    }
}

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = match std::env::var("VICTORY_FAULT_METER") {
            Ok(v) => v.trim() != "0",
            Err(_) => cfg!(feature = "diagnostics"),
        };
        if on {
            let _ = std::thread::Builder::new().name("victory-hard-faults".into()).spawn(hard_fault_loop);
        }
        tracing::info!(on, "VICTORY_FAULT_METER: page faults charged to the layer's own work (process count read around each piece)");
        on
    })
}

/// The process's page faults so far.
pub fn process_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: plain data the call fills in.
    let mut c: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    c.cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

thread_local! {
    /// Open scopes of this thread: the count at their start, and faults of
    /// inner scopes to leave out.
    static OPEN: RefCell<Vec<(u64, u64)>> = const { RefCell::new(Vec::new()) };
}

/// Charges the faults until it is dropped to its kind.
pub struct Scope(Option<Kind>);

/// A scope of `kind` (nothing when the meter is off).
pub fn scope(kind: Kind) -> Scope {
    if !on() {
        return Scope(None);
    }
    let now = process_faults();
    OPEN.with(|o| o.borrow_mut().push((now, 0)));
    Scope(Some(kind))
}

impl Drop for Scope {
    fn drop(&mut self) {
        let Some(kind) = self.0 else { return };
        let now = process_faults();
        OPEN.with(|o| {
            let mut o = o.borrow_mut();
            let Some((start, inner)) = o.pop() else { return };
            let delta = now.saturating_sub(start);
            crate::framelog::add(kind.total(), delta.saturating_sub(inner));
            if let Some(outer) = o.last_mut() {
                outer.1 += delta;
            }
        });
    }
}

/// Hard faults of this process so far, as last read.
static HARD: AtomicU64 = AtomicU64::new(0);
/// 1: read; 2: the query is not available.
static HARD_STATE: AtomicU64 = AtomicU64::new(0);

pub fn hard_available() -> bool {
    HARD_STATE.load(Relaxed) != 2
}

fn hard_fault_loop() {
    type Query = unsafe extern "system" fn(u32, *mut core::ffi::c_void, u32, *mut u32) -> i32;
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    // SAFETY: ntdll is always loaded; the name is NUL-terminated.
    let query: Option<Query> = unsafe {
        let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
        GetProcAddress(ntdll, c"NtQuerySystemInformation".as_ptr().cast()).map(|f| std::mem::transmute(f))
    };
    let Some(query) = query else {
        HARD_STATE.store(2, Relaxed);
        return;
    };
    let pid = std::process::id() as usize;
    let mut buf: Vec<u64> = vec![0; 1 << 16];
    loop {
        let mut needed = 0u32;
        // SAFETY: SystemProcessInformation (5) into a buffer of its length.
        let status = unsafe { query(5, buf.as_mut_ptr().cast(), (buf.len() * 8) as u32, &mut needed) };
        if status < 0 {
            if needed as usize > buf.len() * 8 && needed < (64 << 20) {
                buf.resize((needed as usize).div_ceil(8) + 4096, 0);
                continue;
            }
            HARD_STATE.store(2, Relaxed);
            return;
        }
        // SYSTEM_PROCESS_INFORMATION: +0 next entry offset (u32), +0x10
        // HardFaultCount (u32), +0x50 UniqueProcessId (pointer).
        let bytes = buf.as_ptr().cast::<u8>();
        let len = buf.len() * 8;
        let mut at = 0usize;
        while at + 0x58 <= len {
            // SAFETY: inside the buffer the call filled.
            let (next, hard, id) = unsafe {
                (
                    bytes.add(at).cast::<u32>().read_unaligned(),
                    bytes.add(at + 0x10).cast::<u32>().read_unaligned(),
                    bytes.add(at + 0x50).cast::<usize>().read_unaligned(),
                )
            };
            if id == pid {
                HARD.store(u64::from(hard), Relaxed);
                HARD_STATE.store(1, Relaxed);
                break;
            }
            if next == 0 {
                break;
            }
            at += next as usize;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
}

/// Hard faults so far (as of the last read, up to 250 ms old).
pub fn hard_faults() -> u64 {
    HARD.load(Relaxed)
}

/// Candidate drivers of the process's page faults, counted wherever they
/// happen (cheap atomics, always on), said once a second beside the
/// process's fault count ([`drivers_line`]) to be regressed offline.
#[derive(Clone, Copy)]
pub enum Driver {
    /// Write-watch askings that reset (`GetWriteWatch` with reset).
    WatchResets,
    /// 4 KiB pages the write watch reported written.
    WatchPages,
    /// Guard pages armed again (64 KiB blocks given `PAGE_GUARD` / read-only again).
    GuardsRearmed,
    /// Bytes DirectStorage wrote into title memory.
    DiskBytes,
    /// 4 KiB pages those writes touched.
    DiskPages,
    /// Of them, pages written first after a pool discard (`MEM_RESET`) of their 64 KiB page.
    DiskAfterDiscard,
    /// 4 KiB pages discarded (`VICTORY_POOL_DISCARD`, `MEM_RESET`).
    PoolDiscarded,
    /// 4 KiB pages mapped into the CPU view / the GPU view by the title's mapping calls.
    CpuViewPages,
    GpuViewPages,
    /// Bytes sent up to card copies' card sides from their CPU sides.
    CopyUpBytes,
    /// Bytes of texture uploads.
    UploadBytes,
    /// Bytes copied from title memory into windows (host copies of memory outside buffers).
    WindowFillBytes,
    /// 4 KiB pages of title memory newly committed.
    Committed,
    /// (Prefault job) 4 KiB pages discarded and written again within 5 s.
    DiscardRewritten5s,
}

static DRIVERS: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];

/// (Prefault job) The pool discard's wait after an unmapping now, ms (set by
/// the hle side; 0 not yet known).
pub static DISCARD_DELAY_MS: AtomicU64 = AtomicU64::new(0);

/// (Prefault job) A part of the drivers line from the hle side (the
/// prefault meter).
static EXTRA: OnceLock<fn() -> String> = OnceLock::new();

/// Adds `f`'s text to the end of the drivers line (once; later calls ignored).
pub fn add_to_drivers_line(f: fn() -> String) {
    let _ = EXTRA.set(f);
}

/// Counts `n` of `driver`.
pub fn count(driver: Driver, n: u64) {
    DRIVERS[driver as usize].fetch_add(n, Relaxed);
}

/// Once a second (from the frame thread's present): "fault drivers: ..."
/// with the process's faults over the same second (meter on only).
pub fn drivers_line(secs: f64) {
    if !on() {
        return;
    }
    static LAST: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    let f = process_faults();
    let h = hard_faults();
    let df = f.saturating_sub(LAST[0].swap(f, Relaxed));
    let dh = h.saturating_sub(LAST[1].swap(h, Relaxed));
    let d: Vec<u64> = DRIVERS.iter().map(|a| a.swap(0, Relaxed)).collect();
    tracing::info!(
        "fault drivers over {secs:.1} s: faults {df} (hard {}); write-watch: ranges reset {}, pages reported written {}; guard pages re-armed {}; \
         dstorage: bytes written {}, 4 KiB pages touched {}, of them first write after a discard {}; pool pages discarded {} (MEM_RESET, 4 KiB); \
         views mapped: cpu view pages {}, gpu view pages {}; copy-up bytes {}; texture upload bytes {}; card-window fills bytes {}; \
         title memory newly committed {} pages; discarded then rewritten within 5 s: {} pages (discard delay {}){}",
        if hard_available() { dh.to_string() } else { "n/a".to_owned() },
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13],
        match DISCARD_DELAY_MS.load(Relaxed) {
            0 => "2.0 s".to_owned(),
            ms => format!("{:.1} s", ms as f64 / 1e3),
        },
        EXTRA.get().map(|f| f()).unwrap_or_default(),
    );
}

/// The per-second line (added to the "frame thread streaming" line).
pub fn second_line() -> String {
    if !on() {
        return String::new();
    }
    static LAST: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
    let now = [
        crate::framelog::total(T::FaultsMap),
        crate::framelog::total(T::FaultsCopy),
        crate::framelog::total(T::FaultsHash),
        crate::framelog::total(T::FaultsUpload),
        crate::framelog::total(T::FaultsDisk),
        crate::framelog::total(T::FaultsWatch),
        process_faults(),
        hard_faults(),
    ];
    let d: Vec<u64> = now.iter().zip(&LAST).map(|(&n, l)| n.saturating_sub(l.swap(n, Relaxed))).collect();
    text(&d)
}

/// "page faults: ..." from the deltas map, copy, hash, upload, disk, watch,
/// all, hard.
pub fn text(d: &[u64]) -> String {
    let ours: u64 = d[..6].iter().sum();
    format!(
        "; page faults: in mapping calls {}, in copies {}, in hashing {}, in uploads {}, in dstorage {}, in write-watch resets {}, elsewhere (title code and other threads) {}; hard faults {} (VICTORY_FAULT_METER)",
        d[0],
        d[1],
        d[2],
        d[3],
        d[4],
        d[5],
        d[6].saturating_sub(ours),
        if hard_available() { d[7].to_string() } else { "not available".to_owned() }
    )
}
